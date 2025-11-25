from __future__ import annotations

import importlib.util
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Tuple

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

_OPENAI_SPEC = importlib.util.find_spec("openai")
if _OPENAI_SPEC:
    from openai import OpenAI as _OpenAIClient  # pragma: no cover
else:  # pragma: no cover - optional dependency absent
    _OpenAIClient = None  # type: ignore


class ModelBackend(Protocol):
    """Interface for pluggable generation backends."""

    mode: str

    def generate(self, messages: List[dict], adapters: List[dict]) -> dict:
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
    """Backend that targets external APIs, optionally using fine-tuned model IDs as adapters."""

    def __init__(
        self,
        base_model: str,
        *,
        adapter_mode: str = "api_adapters",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        adapter_server_model: Optional[str] = None,
    ) -> None:
        self.base_model = base_model
        self.adapter_server_model = adapter_server_model
        self.adapter_mode = adapter_mode
        self.mode = adapter_mode
        self.client = _OpenAIClient(api_key=api_key, base_url=base_url) if api_key and _OpenAIClient else None

    def generate(self, messages: List[dict], adapters: List[dict]) -> dict:
        adapter_list = adapters or []
        target_model = self._resolve_model(adapter_list)
        extra_body = self._resolve_extra_body(adapter_list)
        if self.client:
            completion = self.client.chat.completions.create(
                model=target_model,
                messages=messages,
                temperature=0.2,
                extra_body=extra_body,
            )
            choice = completion.choices[0]
            content = choice.message.content or ""
            usage = {
                "prompt_tokens": getattr(completion.usage, "prompt_tokens", 0),
                "completion_tokens": getattr(completion.usage, "completion_tokens", 0),
                "total_tokens": getattr(completion.usage, "total_tokens", 0),
            }
            return {"content": content, "usage": usage}
        fallback = messages[-1]["content"] if messages else ""
        return {
            "content": f"[api-backend model={target_model} adapters={self._adapter_summary(adapter_list)}] {fallback}",
            "usage": {"prompt_tokens": len(fallback.split()), "completion_tokens": random.randint(5, 20)},
        }

    def _resolve_model(self, adapters: List[dict]) -> str:
        if self._uses_adapter_id(adapters):
            return self.adapter_server_model or self.base_model
        if self.adapter_mode == "api_adapters" and adapters:
            for adapter in adapters:
                if self._is_prompt_style(adapter):
                    continue
                backend = (adapter.get("backend") or "").lower()
                provider = (adapter.get("provider") or "").lower()
                remote_model = adapter.get("remote_model_id") or adapter.get("model_id")
                if remote_model and (backend in {"api", "remote", ""} or provider in _MODEL_ID_PROVIDERS):
                    return remote_model
            first = adapters[0]
            if not self._is_prompt_style(first) and (first.get("backend") or "").lower() not in {"local"}:
                return first.get("remote_model_id") or first.get("model_id") or self.base_model
        return self.adapter_server_model or self.base_model

    def _resolve_extra_body(self, adapters: List[dict]) -> Optional[dict]:
        if self._uses_adapter_id(adapters):
            adapter_ids = [a.get("id") for a in adapters if a.get("id") and not self._is_prompt_style(a)]
            if adapter_ids:
                return {"adapter_id": adapter_ids}
        return None

    def _adapter_summary(self, adapters: List[dict]) -> str:
        if not adapters:
            return "none"
        return ",".join([a.get("id") or a.get("remote_model_id") or "anon" for a in adapters])

    def _uses_adapter_id(self, adapters: List[dict]) -> bool:
        if not adapters:
            return False
        if self.adapter_mode == "adapter_server":
            return True
        for adapter in adapters:
            if self._is_prompt_style(adapter):
                continue
            backend = (adapter.get("backend") or "").lower()
            provider = (adapter.get("provider") or "").lower()
            if backend in {"adapter_server", "adapter_id", "multi_lora"}:
                return True
            if provider in _ADAPTER_ID_PROVIDERS:
                return True
        return False

    def _is_prompt_style(self, adapter: dict) -> bool:
        backend = (adapter.get("backend") or "").lower()
        return backend in {"prompt", "prompt_distill", "hybrid"}


class LocalJaxLoRABackend:
    """Backend for local JAX generation with filesystem-backed LoRA adapters.

    The backend keeps a tokenizer and (optional) Flax model resident, reads
    LoRA matrices from ``fs_root`` paths, and runs a lightweight JAX forward
    pass that mirrors the training sketch in ``TrainingService``. It performs
    fixed-shape padding, enforces conservative limits, and emits usage stats
    so callers can track prompt/completion token counts.
    """

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

    def _load_adapter_weights(self, adapter: dict) -> dict:
        if not adapter:
            return {}
        adapter_id = adapter.get("id", "unknown")
        path = Path(self._adapter_path(adapter))
        if path.name != "latest" and not (path / "params.json").exists():
            candidates = sorted([p for p in path.glob("v*/params.json") if p.parent.is_dir()])
            if candidates:
                path = candidates[-1].parent
        params_path = path / "params.json"
        if not params_path.exists():
            return {}
        mtime = params_path.stat().st_mtime
        cached = self._adapter_cache.get(adapter_id)
        if cached and cached[0] == mtime:
            return cached[1]
        weights_raw = json.loads(params_path.read_text())
        self._ensure_jax()
        weights = {k: self._jnp.array(v, dtype=self._jnp.float32) for k, v in weights_raw.items()}
        self._adapter_cache[adapter_id] = (mtime, weights)
        return weights


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

    def generate(self, messages: List[dict], adapters: List[dict]) -> dict:
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

        weights = self._load_adapter_weights(adapter)
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
                "adapter_id": adapter.get("id") if isinstance(adapter, dict) else None,
                "latency_ms": round(duration * 1000, 2),
            },
        }

    def _adapter_path(self, adapter: dict) -> str:
        if not adapter:
            return str(self.fs_root / "adapters")
        explicit = adapter.get("cephfs_dir") or adapter.get("fs_dir")
        if explicit:
            return str(explicit)
        adapter_id = adapter.get("id", "unknown")
        candidate = safe_join(self.fs_root, f"adapters/{adapter_id}")
        latest = candidate / "latest"
        if latest.exists():
            return str(latest)
        return str(candidate)
