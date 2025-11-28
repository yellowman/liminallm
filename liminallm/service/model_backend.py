from __future__ import annotations

import hashlib
import importlib.util
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

from liminallm.config import (
    AdapterMode,
    ProviderCapabilities,
    RemoteStyle,
    get_provider_capabilities,
)
from liminallm.logging import get_logger
from liminallm.service.fs import safe_join
from liminallm.service.tokenizer_utils import DEFAULT_VOCAB_SIZE, vocab_size_from_tokenizer

# providers that expose fine-tuned models as first-class endpoints (model IDs)
_MODEL_ID_PROVIDERS = {
    "openai",
    "azure",
    "azure_openai",
    "azure-openai",
    "vertex",
    "gemini",
    "google",
    "bedrock",
}

# providers that expose adapters via `adapter_id` / multi-LoRA parameters
_ADAPTER_ID_PROVIDERS = {
    "together",
    "together.ai",
    "lorax",
    "adapter_server",
    "sagemaker",
    "aws_sagemaker",
}

logger = get_logger(__name__)


def get_adapter_mode(adapter: dict) -> str:
    """Extract adapter mode from schema, inferring from legacy fields if needed."""
    if not adapter:
        return AdapterMode.PROMPT

    # Check explicit mode field first
    mode = adapter.get("mode") or adapter.get("schema", {}).get("mode")
    if mode:
        return mode

    # Infer from legacy backend/provider fields
    backend = (adapter.get("backend") or "").lower()
    provider = (adapter.get("provider") or "").lower()

    if backend in {"prompt", "prompt_distill"}:
        return AdapterMode.PROMPT
    if backend in {"local", "local_lora"} or provider == "local":
        if adapter.get("prompt_instructions") or adapter.get("behavior_prompt"):
            return AdapterMode.HYBRID
        return AdapterMode.LOCAL
    if backend in {"api", "remote"} or adapter.get("remote_model_id"):
        return AdapterMode.REMOTE
    if backend == "hybrid":
        return AdapterMode.HYBRID

    return AdapterMode.HYBRID  # Default for backwards compatibility


def filter_adapters_by_mode(adapters: List[dict], compatible_modes: set) -> List[dict]:
    """Filter adapters to only those compatible with the current backend mode."""
    result = []
    for adapter in adapters:
        mode = get_adapter_mode(adapter)
        if mode in compatible_modes:
            result.append(adapter)
        else:
            logger.debug(
                "adapter_mode_incompatible",
                adapter_id=adapter.get("id"),
                mode=mode,
                compatible_modes=list(compatible_modes),
            )
    return result

_OPENAI_SPEC = importlib.util.find_spec("openai")
if _OPENAI_SPEC:
    from openai import OpenAI as _OpenAIClient  # pragma: no cover
else:  # pragma: no cover - optional dependency absent
    _OpenAIClient = None  # type: ignore


class ModelBackend(Protocol):
    """Interface for pluggable generation backends."""

    mode: str

    def generate(self, messages: List[dict], adapters: List[dict], *, user_id: Optional[str] = None) -> dict:
        ...


@dataclass(frozen=True)
class AdapterPlug:
    """Describes a pluggable adapter backend."""

    key: str
    label: str
    description: str
    backend_mode: str

    def build_backend(
        self,
        *,
        base_model: str,
        api_key: Optional[str],
        base_url: Optional[str],
        adapter_server_model: Optional[str],
        fs_root: Optional[str],
    ) -> ModelBackend:
        if self.backend_mode == "local_lora":
            return LocalJaxLoRABackend(base_model, fs_root or "/srv/liminallm")
        adapter_mode = "adapter_server" if self.backend_mode == "adapter_server" else "api_adapters"
        if adapter_mode not in {"api_adapters", "adapter_server"}:
            adapter_mode = "api_adapters"
        return ApiAdapterBackend(
            base_model,
            adapter_mode=adapter_mode,
            api_key=api_key,
            base_url=base_url,
            adapter_server_model=adapter_server_model,
        )


BUILTIN_ADAPTER_PLUGS: dict[str, AdapterPlug] = {
    "openai": AdapterPlug(
        key="openai",
        label="OpenAI / compatible API",
        description="Calls OpenAI-style chat completions with optional adapter passthrough.",
        backend_mode="api_adapters",
    ),
    "local_gpu_lora": AdapterPlug(
        key="local_gpu_lora",
        label="Local GPU LoRA",
        description="Runs a local JAX backend that applies filesystem-backed LoRA adapters.",
        backend_mode="local_lora",
    ),
}


def list_adapter_plugs() -> List[AdapterPlug]:
    """Return the built-in adapter plugs."""

    return list(BUILTIN_ADAPTER_PLUGS.values())


class ApiAdapterBackend:
    """Backend that targets external APIs with capability-aware adapter handling.

    Supports SPEC §5.0.2 provider-specific adapter handling:
    - MODEL_ID style (OpenAI, Azure, Vertex): One fine-tuned model per request
    - ADAPTER_PARAM style (Together, LoRAX): adapter_id parameter with multi-adapter
    - PROMPT style: Inject behavior via system prompt (universal fallback)

    The backend inspects provider capabilities to determine how to format
    adapter requests, respecting multi-adapter limits and gate weight support.
    """

    # Modes compatible with this backend
    COMPATIBLE_MODES = {AdapterMode.REMOTE, AdapterMode.PROMPT, AdapterMode.HYBRID}

    def __init__(
        self,
        base_model: str,
        *,
        adapter_mode: str = "api_adapters",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        adapter_server_model: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> None:
        self.base_model = base_model
        self.adapter_server_model = adapter_server_model
        self.adapter_mode = adapter_mode
        self.mode = adapter_mode
        self.client = _OpenAIClient(api_key=api_key, base_url=base_url) if api_key and _OpenAIClient else None
        # Infer provider from adapter_mode if not specified
        self.provider = provider or self._infer_provider(adapter_mode)
        self.capabilities = get_provider_capabilities(self.provider)

    def _infer_provider(self, adapter_mode: str) -> str:
        """Infer provider from adapter_mode string."""
        mode_lower = (adapter_mode or "").lower()
        if mode_lower in {"openai", "azure", "azure_openai", "vertex", "gemini", "bedrock"}:
            return mode_lower
        if mode_lower in {"together", "together.ai"}:
            return "together"
        if mode_lower in {"lorax", "adapter_server"}:
            return mode_lower
        if mode_lower in {"sagemaker", "aws_sagemaker"}:
            return "sagemaker"
        # Default to openai-style for unknown modes
        return "openai"

    def generate(self, messages: List[dict], adapters: List[dict], *, user_id: Optional[str] = None) -> dict:
        adapter_list = adapters or []
        # Process adapters based on provider capabilities
        processed = self._process_adapters_for_provider(adapter_list)
        target_model = processed["model"]
        extra_body = processed["extra_body"]
        prompt_injections = processed["prompt_injections"]

        # Inject adapter prompts if any hybrid/prompt adapters
        augmented_messages = self._inject_adapter_prompts(messages, prompt_injections)

        if self.client:
            completion = self.client.chat.completions.create(
                model=target_model,
                messages=augmented_messages,
                temperature=0.2,
                extra_body=extra_body,
            )
            choices = getattr(completion, "choices", None) or []
            first_choice = next(iter(choices), None)
            if not first_choice:
                logger.warning("API completion returned no choices; returning empty content")
                content = ""
            else:
                content = first_choice.message.content or ""
            usage = {
                "prompt_tokens": getattr(completion.usage, "prompt_tokens", 0),
                "completion_tokens": getattr(completion.usage, "completion_tokens", 0),
                "total_tokens": getattr(completion.usage, "total_tokens", 0),
            }
            return {"content": content, "usage": usage, "adapters_applied": processed["applied"]}
        fallback = augmented_messages[-1]["content"] if augmented_messages else ""
        return {
            "content": f"[api-backend model={target_model} provider={self.provider} adapters={processed['applied']}] {fallback}",
            "usage": {
                "prompt_tokens": len(fallback.split()),
                "completion_tokens": max(5, min(20, len(fallback.split()))),
            },
            "adapters_applied": processed["applied"],
        }

    def _process_adapters_for_provider(self, adapters: List[dict]) -> dict:
        """Process adapters based on provider capabilities.

        Returns dict with:
        - model: Target model ID
        - extra_body: Additional request body parameters
        - prompt_injections: List of prompt strings to inject
        - applied: List of adapter IDs that were applied
        - dropped: List of adapter IDs that were dropped
        """
        prompt_injections: List[str] = []
        remote_adapters: List[dict] = []
        applied: List[str] = []
        dropped: List[str] = []

        for adapter in adapters:
            mode = get_adapter_mode(adapter)
            adapter_id = adapter.get("id") or adapter.get("name") or "unknown"

            if mode == AdapterMode.LOCAL:
                # Local-only adapter - can't use in API mode
                logger.warning(
                    "adapter_mode_incompatible_api",
                    adapter_id=adapter_id,
                    mode=mode,
                    provider=self.provider,
                )
                dropped.append(adapter_id)
                continue

            if mode == AdapterMode.PROMPT:
                # Pure prompt adapter
                prompt = self._extract_prompt_instructions(adapter)
                if prompt:
                    prompt_injections.append(prompt)
                    applied.append(f"{adapter_id}:prompt")
                continue

            if mode == AdapterMode.HYBRID:
                # Hybrid: always extract prompt, optionally add to remote
                prompt = self._extract_prompt_instructions(adapter)
                if prompt:
                    prompt_injections.append(prompt)
                # Check if has remote component
                if adapter.get("remote_model_id") or adapter.get("remote_adapter_id"):
                    remote_adapters.append(adapter)
                    applied.append(f"{adapter_id}:hybrid")
                else:
                    applied.append(f"{adapter_id}:prompt")
                continue

            if mode == AdapterMode.REMOTE:
                remote_adapters.append(adapter)

        # Process remote adapters based on provider remote_style
        model, extra_body, remote_applied, remote_dropped = self._format_remote_adapters(remote_adapters)

        applied.extend(remote_applied)
        dropped.extend(remote_dropped)

        return {
            "model": model,
            "extra_body": extra_body,
            "prompt_injections": prompt_injections,
            "applied": applied,
            "dropped": dropped,
        }

    def _format_remote_adapters(
        self, adapters: List[dict]
    ) -> Tuple[str, Optional[dict], List[str], List[str]]:
        """Format remote adapters based on provider capabilities.

        Returns:
            Tuple of (model_id, extra_body, applied_ids, dropped_ids)
        """
        if not adapters:
            return self.base_model, None, [], []

        applied: List[str] = []
        dropped: List[str] = []
        caps = self.capabilities

        if caps.remote_style == RemoteStyle.MODEL_ID:
            # Provider uses fine-tuned model as endpoint (OpenAI style)
            # Can only use ONE adapter - pick highest weight or first
            selected = self._select_best_adapter(adapters, max_count=1)[0] if adapters else None
            if selected:
                model_id = selected.get("remote_model_id") or selected.get("model_id")
                if model_id:
                    applied.append(f"{selected.get('id', 'unknown')}:model_id")
                    # Drop other adapters
                    for a in adapters:
                        if a is not selected:
                            dropped.append(a.get("id") or "unknown")
                            logger.debug(
                                "adapter_dropped_single_model",
                                adapter_id=a.get("id"),
                                reason="provider only supports one model_id",
                                provider=self.provider,
                            )
                    return model_id, None, applied, dropped
            # No valid remote_model_id found, fall back to base
            return self.base_model, None, [], [a.get("id", "unknown") for a in adapters]

        elif caps.remote_style == RemoteStyle.ADAPTER_PARAM:
            # Provider uses adapter_id parameter (Together, LoRAX style)
            # Can use multiple adapters up to max_adapters
            selected = self._select_best_adapter(adapters, max_count=caps.max_adapters)
            adapter_ids: List[str] = []
            gate_weights: List[float] = []

            for adapter in selected:
                aid = adapter.get("remote_adapter_id") or adapter.get("adapter_id") or adapter.get("id")
                if aid:
                    adapter_ids.append(aid)
                    applied.append(f"{adapter.get('id', 'unknown')}:adapter_param")
                    if caps.gate_weights:
                        weight = adapter.get("weight") or adapter.get("gate_weight") or 1.0
                        gate_weights.append(float(weight))

            # Mark dropped adapters
            for a in adapters:
                if a not in selected:
                    dropped.append(a.get("id") or "unknown")
                    logger.debug(
                        "adapter_dropped_max_exceeded",
                        adapter_id=a.get("id"),
                        max_adapters=caps.max_adapters,
                        provider=self.provider,
                    )

            if not adapter_ids:
                return self.adapter_server_model or self.base_model, None, [], []

            # Build extra_body based on provider
            extra_body: dict = {caps.adapter_param_name: adapter_ids if len(adapter_ids) > 1 else adapter_ids[0]}
            if caps.gate_weights and gate_weights:
                extra_body["adapter_weights"] = gate_weights if len(gate_weights) > 1 else gate_weights[0]

            return self.adapter_server_model or self.base_model, extra_body, applied, dropped

        else:
            # RemoteStyle.NONE - shouldn't have remote adapters
            for a in adapters:
                dropped.append(a.get("id") or "unknown")
            return self.base_model, None, [], dropped

    def _select_best_adapter(self, adapters: List[dict], max_count: int) -> List[dict]:
        """Select best adapters up to max_count, sorted by weight descending."""
        if not adapters:
            return []

        # Sort by weight/gate_weight descending
        def get_weight(a: dict) -> float:
            return float(a.get("weight") or a.get("gate_weight") or 1.0)

        sorted_adapters = sorted(adapters, key=get_weight, reverse=True)
        return sorted_adapters[:max_count]

    def _process_adapters(self, adapters: List[dict]) -> Tuple[List[dict], List[str]]:
        """Legacy method - delegates to _process_adapters_for_provider."""
        result = self._process_adapters_for_provider(adapters)
        # Return in legacy format for backwards compatibility
        return [], result["prompt_injections"]

    def _extract_prompt_instructions(self, adapter: dict) -> Optional[str]:
        """Extract prompt instructions from adapter for injection."""
        for key in ("prompt_instructions", "behavior_prompt", "instructions", "prompt_template"):
            value = adapter.get(key) or adapter.get("schema", {}).get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        # Try applicability.natural_language
        applicability = adapter.get("applicability") or adapter.get("schema", {}).get("applicability")
        if isinstance(applicability, dict):
            natural = applicability.get("natural_language")
            if isinstance(natural, str) and natural.strip():
                return natural.strip()
        # Fall back to description
        description = adapter.get("description") or adapter.get("schema", {}).get("description")
        if isinstance(description, str) and description.strip():
            return description.strip()
        return None

    def _inject_adapter_prompts(self, messages: List[dict], prompts: List[str]) -> List[dict]:
        """Inject adapter prompt instructions into message list."""
        if not prompts:
            return messages

        prompt_text = "\n".join(f"- {p}" for p in prompts)
        system_addition = f"\n\nAdapter guidance:\n{prompt_text}"

        # Find and augment system message, or prepend new one
        augmented = [dict(m) for m in messages]
        for i, msg in enumerate(augmented):
            if msg.get("role") == "system":
                augmented[i] = {**msg, "content": msg.get("content", "") + system_addition}
                return augmented

        # No system message found, prepend one
        augmented.insert(0, {"role": "system", "content": f"Adapter guidance:\n{prompt_text}"})
        return augmented


class LocalJaxLoRABackend:
    """Backend for local JAX generation with filesystem-backed LoRA adapters.

    Supports SPEC §5 dual-mode operation:
    - LOCAL adapters: Load weights from filesystem, apply LoRA math
    - HYBRID adapters: Load local weights, with prompt fallback
    - PROMPT adapters: Inject behavior via system prompt (no weights)

    The backend keeps a tokenizer and (optional) Flax model resident, reads
    LoRA matrices from ``fs_root`` paths, and runs a lightweight JAX forward
    pass that mirrors the training sketch in ``TrainingService``. It performs
    fixed-shape padding, enforces conservative limits, and emits usage stats
    so callers can track prompt/completion token counts.
    """

    # Modes compatible with this backend
    COMPATIBLE_MODES = {AdapterMode.LOCAL, AdapterMode.HYBRID, AdapterMode.PROMPT}

    def __init__(
        self,
        base_model: str,
        fs_root: str,
        *,
        max_seq_len: int = 512,
        max_batch_size: int = 4,
    ) -> None:
        self.base_model = base_model
        self.fs_root = Path(fs_root)
        self.mode = "local_lora"
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.default_vocab_size = DEFAULT_VOCAB_SIZE
        self._base_vocab_size = DEFAULT_VOCAB_SIZE
        self._adapter_vocab_size: Optional[int] = None
        self._adapter_cache: Dict[str, Tuple[float, dict]] = {}
        self._tokenizer = None
        self._tokenizer_error: Optional[str] = None
        self._jax = None
        self._jnp = None
        self._rng = None
        self._device = None

    def _ensure_jax(self):
        if self._jax is not None and self._jnp is not None and self._device is not None:
            return
        import jax
        import jax.numpy as jnp

        devices = jax.devices()
        self._device = devices[0] if devices else jax.devices("cpu")[0]
        self._jax = jax
        self._jnp = jnp
        self._rng = jax.random.PRNGKey(0)

    def _ensure_tokenizer(self):
        if self._tokenizer is not None or self._tokenizer_error is not None:
            return
        try:  # pragma: no cover - optional dependency
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.base_model)
            self._base_vocab_size = vocab_size_from_tokenizer(
                self._tokenizer, fallback=self.default_vocab_size
            )
        except Exception as exc:  # pragma: no cover - optional dependency
            self._tokenizer = None
            self._base_vocab_size = self.default_vocab_size
            self._tokenizer_error = str(exc)
            logger.warning("tokenizer_load_failed", base_model=self.base_model, error=str(exc))

    def _vocab_size(self) -> int:
        if isinstance(self._adapter_vocab_size, int) and self._adapter_vocab_size > 0:
            return self._adapter_vocab_size
        self._ensure_tokenizer()
        return self._base_vocab_size

    def _normalize_messages(self, messages: List[dict]) -> str:
        if not messages:
            return ""
        return "\n".join([f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages])

    def _apply_adapter_vocab_size(self, adapter: dict) -> None:
        self._adapter_vocab_size = None
        if not isinstance(adapter, dict):
            return
        schema = adapter.get("schema") or {}
        vocab_size = schema.get("vocab_size")
        if isinstance(vocab_size, int) and vocab_size > 0:
            self._adapter_vocab_size = vocab_size

    def _tokenize(self, text: str) -> Tuple[List[int], List[int]]:
        self._ensure_tokenizer()
        if self._tokenizer:
            encoded = self._tokenizer(
                text,
                truncation=True,
                max_length=self.max_seq_len,
                return_tensors="np",
            )
            ids = encoded["input_ids"][0].tolist()
            attention = encoded["attention_mask"][0].tolist()
            return ids, attention
        tokens = text.split()
        ids = [hash(tok) % self._vocab_size() for tok in tokens[: self.max_seq_len]]
        attention = [1] * len(ids)
        return ids, attention

    def _pad_batch(self, ids: List[int], attention: List[int]) -> Tuple[List[int], List[int]]:
        length = min(len(ids), self.max_seq_len)
        ids = ids[:length]
        attention = attention[:length]
        if not ids:
            ids = [0]
            attention = [0]
        return ids, attention

    def _load_adapter_weights(self, adapter: dict, *, user_id: Optional[str] = None) -> dict:
        """Load adapter weights from filesystem with checksum verification.

        Per SPEC §18, checksum of params is verified against schema.checksum before activation.
        Missing checksums are logged as security warnings but allowed for backwards compatibility.
        """
        if not adapter:
            return {}
        adapter_id = adapter.get("id", "unknown")
        path = Path(self._adapter_path(adapter, requested_user_id=user_id))
        params_path = self._resolve_params_path(path)
        if not params_path:
            return {}
        mtime = params_path.stat().st_mtime
        cached = self._adapter_cache.get(adapter_id)
        if cached and cached[0] == mtime:
            return cached[1]
        payload = params_path.read_bytes()
        checksum = adapter.get("checksum") or adapter.get("schema", {}).get("checksum")
        if checksum:
            # SPEC §18: checksum verified against schema.checksum before activation
            digest = hashlib.sha256(payload).hexdigest()
            if digest != checksum:
                logger.error(
                    "adapter_checksum_mismatch",
                    adapter_id=adapter_id,
                    path=str(params_path),
                    expected=checksum,
                    actual=digest,
                )
                raise ValueError("adapter checksum mismatch - refusing to load potentially tampered weights")
        else:
            # SPEC §18 requires checksum verification; missing checksums are a security concern
            logger.warning(
                "adapter_checksum_missing",
                adapter_id=adapter_id,
                path=str(params_path),
                message="Adapter loaded without checksum verification - add schema.checksum for production use",
            )
        weights_raw = json.loads(payload.decode())
        self._ensure_jax()
        weights = {k: self._jnp.array(v, dtype=self._jnp.float32) for k, v in weights_raw.items()}
        self._adapter_cache[adapter_id] = (mtime, weights)
        return weights

    def _resolve_params_path(self, path: Path) -> Optional[Path]:
        if path.is_file() and path.name == "params.json":
            return path
        candidates: list[Path] = []
        direct = path / "params.json"
        if direct.exists():
            candidates.append(direct)
        latest = path / "latest" / "params.json"
        if latest.exists():
            candidates.append(latest)
        versioned = [p for p in path.glob("v*/params.json") if p.parent.is_dir()]
        versioned.sort(key=lambda p: self._version_sort_key(p.parent.name))
        candidates.extend(versioned)
        wildcard = [p for p in path.glob("*/params.json") if p.parent.is_dir()]
        wildcard.sort(key=lambda p: p.stat().st_mtime)
        candidates.extend(wildcard)
        for candidate in reversed(candidates):
            if candidate.exists():
                return candidate
        return None

    def _version_sort_key(self, name: str) -> Tuple[int, str]:
        try:
            if name.startswith("v"):
                return int(name[1:]), name
            return int(name), name
        except ValueError:
            return 0, name


    def _align_width(self, arr, width: int):
        if arr.shape[1] > width:
            return arr[:, :width]
        if arr.shape[1] < width:
            pad = ((0, 0), (0, width - arr.shape[1]))
            return self._jnp.pad(arr, pad)
        return arr

    def _align_last_dim(self, arr, width: int):
        current = arr.shape[-1]
        if current > width:
            slices = (slice(None),) * (arr.ndim - 1) + (slice(0, width),)
            return arr[slices]
        if current < width:
            pad = [(0, 0)] * (arr.ndim - 1) + [(0, width - current)]
            return self._jnp.pad(arr, pad)
        return arr

    def _lora_forward(self, params: dict, inputs):
        hidden_dim = max((mat.shape[1] for name, mat in params.items() if name.endswith(".A")), default=16)
        vocab_size = max(self._vocab_size(), 1)

        if inputs.ndim == 2:
            if inputs.size:
                max_token = int(self._jnp.max(inputs))
                vocab_size = max(vocab_size, max_token + 1)
            emb_table = self._jnp.sin(
                self._jnp.arange(vocab_size * hidden_dim, dtype=self._jnp.float32).reshape(vocab_size, hidden_dim)
                / float(hidden_dim)
            )
            clipped = self._jnp.clip(inputs, 0, vocab_size - 1)
            embeds = emb_table[clipped]
        elif inputs.ndim == 3:
            embeds = self._jnp.asarray(inputs, dtype=self._jnp.float32)
        else:
            raise ValueError("inputs must be token IDs (2D) or embeddings (3D)")

        embeds = self._align_last_dim(embeds, hidden_dim)
        acc = self._jnp.zeros_like(embeds, dtype=self._jnp.float32)
        for name, mat in params.items():
            if not name.endswith(".A"):
                continue
            b_key = name.replace(".A", ".B")
            if b_key not in params:
                continue
            inputs_aligned = self._align_last_dim(embeds, mat.shape[1])
            base = inputs_aligned @ mat.T
            update = base @ params[b_key].T
            update = self._align_last_dim(update, acc.shape[-1])
            acc = acc + update
        return embeds + acc

    def _decode(self, token_ids: List[int]) -> str:
        self._ensure_tokenizer()
        if self._tokenizer:
            try:  # pragma: no cover - optional dependency
                return self._tokenizer.decode(token_ids, skip_special_tokens=True)
            except Exception as exc:
                logger.warning("tokenizer_decode_failed", base_model=self.base_model, error=str(exc))
        return " ".join([f"tok-{tid}" for tid in token_ids])

    def _sample_tokens(self, lora_scores, seed_token: int) -> List[int]:
        vocab = self._vocab_size()
        score = float(self._jnp.mean(lora_scores)) if lora_scores.size else 0.0
        token = int(seed_token)
        generated: List[int] = []
        for _ in range(32):
            token = int(abs(token + score)) % vocab
            generated.append(token)
            score = score * 0.9 + 0.1 * token
        return generated

    def generate(self, messages: List[dict], adapters: List[dict], *, user_id: Optional[str] = None) -> dict:
        prompt = self._normalize_messages(messages)
        adapter = adapters[0] if adapters else {}
        self._apply_adapter_vocab_size(adapter)
        ids, attention = self._tokenize(prompt)
        ids, attention = self._pad_batch(ids, attention)
        if len(ids) > self.max_seq_len:
            raise ValueError(f"prompt exceeds max length ({self.max_seq_len})")
        if len(attention) > self.max_seq_len:
            raise ValueError(f"attention mask exceeds max length ({self.max_seq_len})")
        if self.max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")

        self._ensure_jax()
        token_array = self._jnp.array([ids], dtype=self._jnp.int32)
        attn_array = self._jnp.array([attention], dtype=self._jnp.int32)
        token_array = self._jax.device_put(token_array, self._device)
        attn_array = self._jax.device_put(attn_array, self._device)

        weights = self._blend_adapter_weights(adapters, user_id=user_id) if adapters else {}
        start = time.perf_counter()
        lora_scores = self._lora_forward(weights, token_array) if weights else self._jnp.zeros_like(token_array)
        lora_scores = lora_scores * attn_array
        generated_ids = self._sample_tokens(lora_scores, seed_token=token_array[0][-1])
        completion = self._decode(generated_ids)
        duration = time.perf_counter() - start

        return {
            "content": completion,
            "usage": {
                "prompt_tokens": len(ids),
                "completion_tokens": len(generated_ids),
                "model": self.base_model,
                "adapter_id": ",".join(str(a.get("id")) for a in adapters if a.get("id")) if adapters else None,
                "latency_ms": round(duration * 1000, 2),
            },
        }

    def _blend_adapter_weights(self, adapters: List[dict], user_id: Optional[str]) -> dict:
        combined: dict[str, Any] = {}
        counts: dict[str, int] = {}
        for adapter in adapters:
            weights = self._load_adapter_weights(adapter, user_id=user_id)
            if not weights:
                continue
            for name, tensor in weights.items():
                if name in combined:
                    if combined[name].shape != tensor.shape:
                        logger.warning(
                            "adapter_shape_mismatch", adapter_id=adapter.get("id"), name=name
                        )
                        continue
                    combined[name] = combined[name] + tensor
                    counts[name] += 1
                else:
                    combined[name] = tensor
                    counts[name] = 1
        for name, tensor in combined.items():
            combined[name] = tensor / max(counts.get(name, 1), 1)
        return combined

    def _adapter_path(self, adapter: dict, *, requested_user_id: Optional[str]) -> str:
        if not adapter:
            return str(self.fs_root / "adapters")
        explicit = adapter.get("cephfs_dir") or adapter.get("fs_dir")
        if explicit:
            if not requested_user_id:
                raise ValueError("adapter path resolution requires requesting user context")
            owner = adapter.get("owner_user_id") or adapter.get("schema", {}).get("owner_user_id")
            visibility = adapter.get("visibility") or adapter.get("schema", {}).get("visibility")
            if owner and owner != requested_user_id and visibility not in {"shared", "global"}:
                raise ValueError("adapter owner mismatch")
            base = self.fs_root.resolve()
            candidate = (Path(str(explicit)) if isinstance(explicit, (str, Path)) else Path(""))
            resolved = (candidate if candidate.is_absolute() else base / candidate).resolve()
            # Path must be within fs_root: base must be a parent of resolved, or they must be equal
            if not (base in resolved.parents or resolved == base):
                raise ValueError("adapter path must reside within fs_root")
            return str(resolved)
        adapter_id = adapter.get("id", "unknown")
        candidate = safe_join(self.fs_root, f"adapters/{adapter_id}")
        latest = candidate / "latest"
        if latest.exists():
            return str(latest)
        return str(candidate)
