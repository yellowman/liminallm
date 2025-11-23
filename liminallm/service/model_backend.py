from __future__ import annotations

import importlib.util
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Protocol

from liminallm.service.fs import safe_join

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
    """Backend placeholder for local JAX generation with LoRA adapter application.

    This stub only echoes routing metadata; real implementations should:
    - Load a frozen base model (e.g., Flax/Transformers) once per process and
      keep it resident on the accelerator.
    - Materialize LoRA weights from the filesystem (``fs_root``) and either
      merge them into the base weights or apply them at run time.
    - Provide a tokenizer that matches the base model and stream decoded tokens
      back through the API layer for parity with remote backends.
    - Enforce deterministic batching and shape checks so adapters cannot cause
      OOMs; see ``docs/jax_backend.md`` for a concrete build-out checklist.
    """

    def __init__(self, base_model: str, fs_root: str) -> None:
        self.base_model = base_model
        self.fs_root = Path(fs_root)
        self.mode = "local_lora"

    def generate(self, messages: List[dict], adapters: List[dict]) -> dict:
        latest = messages[-1]["content"] if messages else ""
        adapter = adapters[0] if adapters else {}
        adapter_id = adapter.get("id") if isinstance(adapter, dict) else None
        adapter_path = self._adapter_path(adapter) if adapter else None
        adapter_base = adapter.get("base_model") if isinstance(adapter, dict) else None
        content = (
            f"[local-jax base={adapter_base or self.base_model} adapter={adapter_id or 'none'}"
            f" path={adapter_path or 'n/a'}] {latest}"
        )
        return {
            "content": content,
            "usage": {"prompt_tokens": len(latest.split()), "completion_tokens": random.randint(5, 20)},
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
