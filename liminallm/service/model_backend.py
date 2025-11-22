from __future__ import annotations

import importlib.util
import random
from pathlib import Path
from typing import List, Optional, Protocol

from liminallm.service.fs import safe_join

_OPENAI_SPEC = importlib.util.find_spec("openai")
if _OPENAI_SPEC:
    from openai import OpenAI as _OpenAIClient  # pragma: no cover
else:  # pragma: no cover - optional dependency absent
    _OpenAIClient = None  # type: ignore


class ModelBackend(Protocol):
    """Interface for pluggable generation backends."""

    mode: str

    def generate(self, messages: List[dict], adapters: List[str]) -> dict:
        ...


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

    def generate(self, messages: List[dict], adapters: List[str]) -> dict:
        target_model = self._resolve_model(adapters)
        extra_body = self._resolve_extra_body(adapters)
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
            "content": f"[api-backend model={target_model} adapters={','.join(adapters)}] {fallback}",
            "usage": {"prompt_tokens": len(fallback.split()), "completion_tokens": random.randint(5, 20)},
        }

    def _resolve_model(self, adapters: List[str]) -> str:
        if self.adapter_mode == "api_adapters" and adapters:
            return adapters[0]
        return self.adapter_server_model or self.base_model

    def _resolve_extra_body(self, adapters: List[str]) -> Optional[dict]:
        if self.adapter_mode == "adapter_server" and adapters:
            return {"adapter_id": adapters}
        return None


class LocalJaxLoRABackend:
    """Backend placeholder for local JAX generation with LoRA adapter application."""

    def __init__(self, base_model: str, fs_root: str) -> None:
        self.base_model = base_model
        self.fs_root = Path(fs_root)
        self.mode = "local_lora"

    def generate(self, messages: List[dict], adapters: List[str]) -> dict:
        latest = messages[-1]["content"] if messages else ""
        adapter = adapters[0] if adapters else None
        adapter_path = self._adapter_path(adapter) if adapter else None
        content = (
            f"[local-jax base={self.base_model} adapter={adapter or 'none'}"
            f" path={adapter_path or 'n/a'}] {latest}"
        )
        return {
            "content": content,
            "usage": {"prompt_tokens": len(latest.split()), "completion_tokens": random.randint(5, 20)},
        }

    def _adapter_path(self, adapter_id: str) -> str:
        candidate = safe_join(self.fs_root, f"adapters/{adapter_id}")
        latest = candidate / "latest"
        if latest.exists():
            return str(latest)
        return str(candidate)
