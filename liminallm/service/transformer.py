"""A real decoder-only transformer forward pass, in plain JAX.

What this replaces: both serving and training used to run LoRA over a
*synthetic* embedding table — ``sin(arange(...))`` — with no attention and no
layers. That made the local lane shape-correct and semantically empty: it
could not answer anything, and there was no KV state, which is why local
`cached_tokens` had nothing to report. This module is the base model those
paths were standing in for.

Plain JAX rather than Flax: the parameters are dicts of arrays, which is all
a frozen base model needs, and it keeps a heavy dependency out of the serving
image. The architecture is the Llama/Qwen family shape (RMSNorm, RoPE, grouped
-query attention, SwiGLU MLP), because that is what the SPEC's own examples
name (qwen2.5-32b-instruct) and what an HF-layout checkout on disk contains.

Two properties the tests pin, because both are the kind of thing that looks
right and is wrong:

* **cached decode == full recompute.** Incremental decoding with a KV cache
  must produce the logits a full forward over the same tokens produces. A
  cache that drifts is a model that quietly degrades the longer it talks.
* **LoRA with B=0 is the identity.** That is how LoRA initializes, so a
  freshly created adapter must not change a single logit.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from liminallm.logging import get_logger

logger = get_logger(__name__)

#: LoRA target names, matching SPEC §5.2's `matrices` field. An adapter names
#: its matrices `layers.{i}.{target}.A` / `.B`; anything else is ignored, so a
#: checkpoint trained for another architecture cannot silently half-apply.
LORA_TARGETS = ("attn_q", "attn_k", "attn_v", "attn_o")


@dataclass(frozen=True)
class ModelConfig:
    """The subset of an HF config.json this forward pass needs."""

    vocab_size: int
    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_size: int
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = False

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "ModelConfig":
        hidden = int(config.get("hidden_size") or config.get("n_embd") or 0)
        heads = int(config.get("num_attention_heads") or config.get("n_head") or 1)
        kv_heads = int(config.get("num_key_value_heads") or heads)
        head_dim = int(config.get("head_dim") or (hidden // heads if heads else 0))
        return cls(
            vocab_size=int(config.get("vocab_size") or 0),
            hidden_size=hidden,
            num_layers=int(
                config.get("num_hidden_layers") or config.get("n_layer") or 0
            ),
            num_heads=heads,
            num_kv_heads=kv_heads,
            head_dim=head_dim,
            intermediate_size=int(
                config.get("intermediate_size") or (4 * hidden)
            ),
            rms_norm_eps=float(
                config.get("rms_norm_eps") or config.get("layer_norm_eps") or 1e-6
            ),
            rope_theta=float(config.get("rope_theta") or 10000.0),
            tie_word_embeddings=bool(config.get("tie_word_embeddings", False)),
        )

    def is_usable(self) -> bool:
        return all(
            value > 0
            for value in (
                self.vocab_size,
                self.hidden_size,
                self.num_layers,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                self.intermediate_size,
            )
        ) and self.num_heads % self.num_kv_heads == 0


def _hf_weight_names(layer: int) -> Dict[str, str]:
    prefix = f"model.layers.{layer}"
    return {
        "attn_norm": f"{prefix}.input_layernorm.weight",
        "attn_q": f"{prefix}.self_attn.q_proj.weight",
        "attn_k": f"{prefix}.self_attn.k_proj.weight",
        "attn_v": f"{prefix}.self_attn.v_proj.weight",
        "attn_o": f"{prefix}.self_attn.o_proj.weight",
        "ffn_norm": f"{prefix}.post_attention_layernorm.weight",
        "gate": f"{prefix}.mlp.gate_proj.weight",
        "up": f"{prefix}.mlp.up_proj.weight",
        "down": f"{prefix}.mlp.down_proj.weight",
    }


def base_identity(value: Any) -> str:
    """The comparable identity of a base model, per SPEC §5.1.

    One definition, because both sides of the ladder ask the same question:
    training refuses to fit an adapter to a base other than the runtime's,
    and serving refuses to apply the result to a base other than the one it
    was fitted to. Two spellings of that rule drift, and the looser one
    decides.

    Identity is the final path component, lowercased, so a checkout at
    ``/models/qwen3-4b`` and the name ``qwen3-4b`` are one model. Nothing
    fuzzier: ``-chat``, ``-base`` and version suffixes mark *different*
    frozen weights, so treating them as one model is what the rule exists to
    prevent. An empty value has no identity and matches nothing, including
    another empty value.
    """
    text = str(value or "").strip().rstrip("/")
    if not text:
        return ""
    return text.split("/")[-1].lower()


def same_base_model(left: Any, right: Any) -> bool:
    """Whether two base-model references name the same checkpoint."""
    identity = base_identity(left)
    return bool(identity) and identity == base_identity(right)


def checkpoint_available(model_dir: str | Path) -> bool:
    """Whether ``model_dir`` looks like a loadable HF-layout checkout."""
    directory = Path(model_dir)
    if not (directory / "config.json").is_file():
        return False
    return any(directory.glob("*.safetensors"))


def load_config(model_dir: str | Path) -> Optional[ModelConfig]:
    """Just the config, for callers that need shapes but not gigabytes.

    Training sizes its LoRA matrices from this (SPEC §5.2 needs `d_in` and
    `d_out` per projection) long before it needs the weights themselves.
    """
    try:
        config = ModelConfig.from_dict(
            json.loads((Path(model_dir) / "config.json").read_text())
        )
    except Exception:  # noqa: BLE001 - missing/unparseable config
        return None
    return config if config.is_usable() else None


def load_checkpoint(model_dir: str | Path) -> Tuple[ModelConfig, Dict[str, Any]]:
    """Read config.json and every shard into a parameter tree.

    Weights arrive as numpy (safetensors' framework-neutral read) and are cast
    to float32 once here; JAX moves them to the device on first use. Missing
    tensors raise rather than defaulting — a half-loaded model answers
    confidently and wrongly, which is worse than not starting.
    """
    import numpy as np
    from safetensors.numpy import load_file

    directory = Path(model_dir)
    config = ModelConfig.from_dict(
        json.loads((directory / "config.json").read_text())
    )
    if not config.is_usable():
        raise ValueError(f"config.json at {directory} does not describe a usable model")

    raw: Dict[str, Any] = {}
    for shard in sorted(directory.glob("*.safetensors")):
        raw.update(load_file(str(shard)))

    def take(name: str) -> Any:
        if name not in raw:
            raise KeyError(f"checkpoint is missing {name!r}")
        return np.asarray(raw[name], dtype=np.float32)

    embed = take("model.embed_tokens.weight")
    layers = []
    for index in range(config.num_layers):
        names = _hf_weight_names(index)
        layers.append({key: take(name) for key, name in names.items()})
    params: Dict[str, Any] = {
        "embed": embed,
        "layers": layers,
        "final_norm": take("model.norm.weight"),
    }
    if config.tie_word_embeddings or "lm_head.weight" not in raw:
        params["lm_head"] = embed
    else:
        params["lm_head"] = take("lm_head.weight")
    logger.info(
        "local_checkpoint_loaded",
        model_dir=str(directory),
        layers=config.num_layers,
        hidden=config.hidden_size,
        vocab=config.vocab_size,
    )
    return config, params


def _rms_norm(jnp, x, weight, eps: float):
    variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    return x * jnp.reciprocal(jnp.sqrt(variance + eps)) * weight


def _rope_tables(jnp, positions, head_dim: int, theta: float):
    """cos/sin tables for the given absolute positions, shaped [1, S, 1, D]."""
    half = head_dim // 2
    inv_freq = 1.0 / (
        theta ** (jnp.arange(0, half, dtype=jnp.float32) * 2.0 / head_dim)
    )
    angles = positions.astype(jnp.float32)[:, None] * inv_freq[None, :]
    # Duplicated to full head_dim so the rotate-half form below lines up, the
    # same layout HF uses.
    angles = jnp.concatenate([angles, angles], axis=-1)
    return jnp.cos(angles)[None, :, None, :], jnp.sin(angles)[None, :, None, :]


def _rotate_half(jnp, x):
    half = x.shape[-1] // 2
    return jnp.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def _apply_rope(jnp, x, cos, sin):
    return x * cos + _rotate_half(jnp, x) * sin


def _lora_delta(jnp, x, lora: Optional[Dict[str, Any]], key: str):
    """The LoRA update for one projection, or nothing.

    ``B @ A @ x`` with B zero-initialized is exactly zero, so a fresh adapter
    is the identity — the property the tests pin.
    """
    if not lora:
        return None
    a = lora.get(f"{key}.A")
    b = lora.get(f"{key}.B")
    if a is None or b is None:
        return None
    scale = float(lora.get(f"{key}.scale", 1.0))
    return (x @ a.T) @ b.T * scale


def _project(jnp, x, weight, lora, key):
    out = x @ weight.T
    delta = _lora_delta(jnp, x, lora, key)
    return out if delta is None else out + delta


def forward(
    jnp,
    config: ModelConfig,
    params: Dict[str, Any],
    tokens,
    *,
    cache: Optional[List[Tuple[Any, Any]]] = None,
    lora: Optional[List[Dict[str, Any]]] = None,
):
    """Logits for ``tokens``, plus the updated KV cache.

    ``tokens`` is [batch, seq]. When ``cache`` is given, those tokens continue
    the cached prefix: positions start at the cached length and each new query
    attends to the whole cache plus the causal part of the new block. Only the
    final position's logits are needed for sampling, but all are returned so
    a caller can score a whole sequence.
    """
    batch, seq = tokens.shape
    cached_len = 0 if not cache else int(cache[0][0].shape[1])
    positions = jnp.arange(cached_len, cached_len + seq, dtype=jnp.int32)
    cos, sin = _rope_tables(jnp, positions, config.head_dim, config.rope_theta)

    # A query at absolute position p may see keys at 0..p. Keys are the cached
    # prefix followed by this block, so key index k maps to absolute k.
    key_positions = jnp.arange(cached_len + seq, dtype=jnp.int32)
    allowed = key_positions[None, :] <= positions[:, None]
    mask = jnp.where(allowed, 0.0, -1e30)[None, None, :, :]

    repeats = config.num_heads // config.num_kv_heads
    scale = 1.0 / math.sqrt(config.head_dim)
    x = params["embed"][tokens]
    new_cache: List[Tuple[Any, Any]] = []

    for index, layer in enumerate(params["layers"]):
        layer_lora = lora[index] if lora and index < len(lora) else None
        normed = _rms_norm(jnp, x, layer["attn_norm"], config.rms_norm_eps)
        q = _project(jnp, normed, layer["attn_q"], layer_lora, "attn_q")
        k = _project(jnp, normed, layer["attn_k"], layer_lora, "attn_k")
        v = _project(jnp, normed, layer["attn_v"], layer_lora, "attn_v")
        q = q.reshape(batch, seq, config.num_heads, config.head_dim)
        k = k.reshape(batch, seq, config.num_kv_heads, config.head_dim)
        v = v.reshape(batch, seq, config.num_kv_heads, config.head_dim)
        q = _apply_rope(jnp, q, cos, sin)
        k = _apply_rope(jnp, k, cos, sin)

        if cache:
            past_k, past_v = cache[index]
            k = jnp.concatenate([past_k, k], axis=1)
            v = jnp.concatenate([past_v, v], axis=1)
        new_cache.append((k, v))

        if repeats > 1:
            k_full = jnp.repeat(k, repeats, axis=2)
            v_full = jnp.repeat(v, repeats, axis=2)
        else:
            k_full, v_full = k, v

        scores = jnp.einsum("bqhd,bkhd->bhqk", q, k_full) * scale + mask
        weights = jax_softmax(jnp, scores)
        attended = jnp.einsum("bhqk,bkhd->bqhd", weights, v_full)
        attended = attended.reshape(batch, seq, config.num_heads * config.head_dim)
        x = x + _project(jnp, attended, layer["attn_o"], layer_lora, "attn_o")

        normed = _rms_norm(jnp, x, layer["ffn_norm"], config.rms_norm_eps)
        gate = normed @ layer["gate"].T
        up = normed @ layer["up"].T
        activated = gate * jax_sigmoid(jnp, gate) * up  # SwiGLU
        x = x + activated @ layer["down"].T

    x = _rms_norm(jnp, x, params["final_norm"], config.rms_norm_eps)
    return x @ params["lm_head"].T, new_cache


def jax_softmax(jnp, scores):
    shifted = scores - jnp.max(scores, axis=-1, keepdims=True)
    exp = jnp.exp(shifted)
    return exp / jnp.sum(exp, axis=-1, keepdims=True)


def jax_sigmoid(jnp, x):
    return 1.0 / (1.0 + jnp.exp(-x))


def sample_token(
    jax,
    jnp,
    logits,
    key,
    *,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> int:
    """One token from the final-position logits.

    Greedy when temperature is 0 — the deterministic default, so a test can
    assert on output at all.
    """
    if temperature <= 0.0:
        return int(jnp.argmax(logits))
    scaled = logits / temperature
    probs = jax_softmax(jnp, scaled)
    if 0.0 < top_p < 1.0:
        order = jnp.argsort(probs)[::-1]
        ordered = probs[order]
        cumulative = jnp.cumsum(ordered)
        # Keep the smallest set whose mass reaches top_p (always ≥ 1 token).
        keep = cumulative - ordered < top_p
        ordered = jnp.where(keep, ordered, 0.0)
        probs = jnp.zeros_like(probs).at[order].set(ordered)
        probs = probs / jnp.sum(probs)
    return int(jax.random.categorical(key, jnp.log(probs + 1e-20)))


def _shape_of(value) -> Tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is not None:
        return tuple(shape)
    # Lists, as they arrive from params.json before conversion.
    if isinstance(value, list):
        if value and isinstance(value[0], list):
            return (len(value), len(value[0]))
        return (len(value),)
    return ()


def validate_lora_weights(
    config: Optional[ModelConfig], weights: Dict[str, Any]
) -> None:
    """Raise unless **every** name is well-formed and correctly shaped.

    SPEC §5.2 says names outside the declared shape are never partially
    applied. Collecting the invalid ones, logging them and returning the
    valid subset — which is what the assembly functions used to do — is
    partial application with a log line attached: an adapter carrying one
    foreign matrix still changed the model through its recognized ones.

    One invalid member invalidates the adapter.

    ``config`` is optional so this stays the *only* validator: name grammar,
    target vocabulary, A/B/scale pairing and rank self-consistency are all
    knowable without a model, and a second partial checker for the no-model
    case is how the two drift apart. Layer bounds and real projection widths
    are checked additionally when a config is available.
    """
    if not weights:
        return
    keys: set = set()
    for name, value in weights.items():
        parts = name.split(".")
        if len(parts) != 4 or parts[0] != "layers" or parts[3] not in ("A", "B", "scale"):
            raise ValueError(
                f"LoRA name {name!r} is outside the declared shape "
                "layers.{i}.{target}.{A|B|scale}; refusing the adapter"
            )
        try:
            index = int(parts[1])
        except ValueError:
            raise ValueError(f"LoRA name {name!r} has a non-numeric layer index") from None
        if parts[2] not in LORA_TARGETS:
            raise ValueError(
                f"LoRA name {name!r} targets {parts[2]!r}, which is not one of "
                f"{sorted(LORA_TARGETS)}; refusing the adapter"
            )
        if config is not None and not 0 <= index < config.num_layers:
            raise ValueError(
                f"LoRA name {name!r} targets layer {index}, outside the model's "
                f"{config.num_layers}; refusing the adapter"
            )
        if parts[3] == "scale" and _shape_of(value) != ():
            # α is a scalar attached to a hooked weight (SPEC §5.2), not a
            # free-standing tensor.
            raise ValueError(
                f"LoRA {name!r} must be a scalar; refusing the adapter"
            )
        # Every prefix, scale included: a `.scale` whose projection has no
        # matrices is an adapter that contributes nothing while looking
        # non-empty — which slipped past the "a selected adapter never
        # silently leaves the stack" rule, because the raw dict was truthy.
        keys.add(".".join(parts[:3]))

    for key in sorted(keys):
        a_value = weights.get(f"{key}.A")
        b_value = weights.get(f"{key}.B")
        if a_value is None or b_value is None:
            missing = "B" if a_value is not None else "A"
            raise ValueError(
                f"LoRA {key} is missing its {missing} matrix; refusing the adapter"
            )
        a_shape, b_shape = _shape_of(a_value), _shape_of(b_value)
        if len(a_shape) != 2 or len(b_shape) != 2:
            raise ValueError(f"LoRA {key} matrices must be 2-D; refusing the adapter")
        if a_shape[0] != b_shape[1]:
            raise ValueError(
                f"LoRA {key} disagrees with itself on rank: A{a_shape} B{b_shape}"
            )
        if config is None:
            continue
        d_out, d_in = projection_shape(config, key.split(".")[2])
        if a_shape[1] != d_in or b_shape[0] != d_out:
            raise ValueError(
                f"LoRA {key} is shaped A{a_shape} B{b_shape}, but this model's "
                f"projection is ({d_out}, {d_in}); refusing the adapter"
            )


def lora_by_layer(
    jnp, weights: Dict[str, Any], num_layers: int
) -> Optional[List[Dict[str, Any]]]:
    """Flat adapter weights into per-layer dicts the forward pass can use.

    Names follow ``layers.{i}.{target}.{A|B}``. Anything unrecognized is
    counted and logged rather than silently dropped: an adapter whose matrices
    never matched is a training/serving mismatch someone needs to hear about.
    """
    if not weights:
        return None
    # Assembly assumes validity: validate_lora_weights (called by both the
    # serving and training paths before this point) has already refused any
    # adapter carrying a name this loop would have had to skip.
    layers: List[Dict[str, Any]] = [{} for _ in range(num_layers)]
    matched = 0
    for name, value in weights.items():
        parts = name.split(".")
        if len(parts) != 4 or parts[0] != "layers" or parts[2] not in LORA_TARGETS:
            continue
        try:
            index = int(parts[1])
        except ValueError:
            continue
        # `scale` is the α of SPEC §5.2 — part of the serialization contract.
        if 0 <= index < num_layers and parts[3] in ("A", "B", "scale"):
            layers[index][f"{parts[2]}.{parts[3]}"] = jnp.asarray(
                value, dtype=jnp.float32
            )
            matched += 1
    return layers if matched else None


def projection_shape(config: ModelConfig, target: str) -> Optional[Tuple[int, int]]:
    """``(d_out, d_in)`` of one attention projection, per SPEC §5.2.

    LoRA is `A ∈ ℝ^{r × d_in}` and `B ∈ ℝ^{d_out × r}`, so a trainer cannot
    size its matrices without these. Grouped-query attention makes k and v
    narrower than q, which is exactly the kind of detail a guessed shape gets
    wrong.
    """
    hidden = config.hidden_size
    heads_width = config.num_heads * config.head_dim
    kv_width = config.num_kv_heads * config.head_dim
    return {
        "attn_q": (heads_width, hidden),
        "attn_k": (kv_width, hidden),
        "attn_v": (kv_width, hidden),
        "attn_o": (hidden, heads_width),
    }.get(target)


def lora_layer_index(
    names: Sequence[str], *, slots: Sequence[str] = ("A", "B")
) -> Dict[str, Tuple[int, str]]:
    """Map flat LoRA names to ``(layer index, "target.slot")``.

    Computed once, outside any traced function: the assembly inside a loss
    function must be pure array plumbing, with the string parsing already
    done. ``slots`` defaults to the trainable matrices — α is a fixed
    hyperparameter, so a trainer asks for it separately and carries it as a
    constant rather than handing it to the optimizer.
    """
    index: Dict[str, Tuple[int, str]] = {}
    for name in names:
        parts = name.split(".")
        if (
            len(parts) == 4
            and parts[0] == "layers"
            and parts[2] in LORA_TARGETS
            and parts[3] in slots
        ):
            try:
                index[name] = (int(parts[1]), f"{parts[2]}.{parts[3]}")
            except ValueError:
                continue
    return index


def prefix_length(a: Sequence[int], b: Sequence[int]) -> int:
    """How many leading tokens two sequences share."""
    count = 0
    for left, right in zip(a, b):
        if left != right:
            break
        count += 1
    return count
