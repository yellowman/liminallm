from __future__ import annotations

from typing import List, Optional

from liminallm.service.model_backend import ApiAdapterBackend, LocalJaxLoRABackend, ModelBackend
from liminallm.storage.models import Message


class LLMService:
    """LLM executor that delegates to a pluggable model backend."""

    def __init__(
        self,
        base_model: str,
        *,
        backend_mode: str = "api_adapters",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        adapter_server_model: Optional[str] = None,
        fs_root: Optional[str] = None,
        backend: Optional[ModelBackend] = None,
    ) -> None:
        self.base_model = base_model
        self.backend = backend or self._build_backend(
            backend_mode,
            api_key=api_key,
            base_url=base_url,
            adapter_server_model=adapter_server_model,
            fs_root=fs_root,
        )

    def generate(self, prompt: str, adapters: List[str], context_snippets: List[str], history: Optional[List[Message]] = None) -> dict:
        messages = [{"role": "system", "content": "You are a concise assistant."}]
        if history:
            for msg in history:
                messages.append({"role": msg.role, "content": msg.content})
            messages = self._inject_context(messages, context_snippets)
        else:
            messages.append({"role": "user", "content": self._format_user(prompt, context_snippets)})
        return self.backend.generate(messages, adapters)

    def _format_user(self, prompt: str, context_snippets: List[str]) -> str:
        if context_snippets:
            return f"{prompt}\nContext: {' | '.join(context_snippets)}"
        return prompt

    def _inject_context(self, messages: List[dict], context_snippets: List[str]) -> List[dict]:
        if not context_snippets:
            return messages
        for msg in reversed(messages):
            if msg.get("role") == "user":
                msg["content"] = f"{msg.get('content', '')}\nContext: {' | '.join(context_snippets)}"
                return messages
        messages.append({"role": "system", "content": f"Context: {' | '.join(context_snippets)}"})
        return messages

    def _build_backend(
        self,
        backend_mode: str,
        *,
        api_key: Optional[str],
        base_url: Optional[str],
        adapter_server_model: Optional[str],
        fs_root: Optional[str],
    ) -> ModelBackend:
        mode = backend_mode or "api_adapters"
        if mode in {"local_lora", "local", "jax_lora"}:
            return LocalJaxLoRABackend(self.base_model, fs_root or "/srv/liminallm")
        adapter_mode = mode if mode in {"api_adapters", "adapter_server"} else "api_adapters"
        return ApiAdapterBackend(
            self.base_model,
            adapter_mode=adapter_mode,
            api_key=api_key,
            base_url=base_url,
            adapter_server_model=adapter_server_model,
        )
