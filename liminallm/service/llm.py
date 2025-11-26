from __future__ import annotations

from typing import List, Optional

from liminallm.service.model_backend import (
    ApiAdapterBackend,
    AdapterPlug,
    BUILTIN_ADAPTER_PLUGS,
    LocalJaxLoRABackend,
    ModelBackend,
)
from liminallm.storage.models import Message


class LLMService:
    """LLM executor that delegates to a pluggable model backend."""

    def __init__(
        self,
        base_model: str,
        *,
        backend_mode: str = "api_adapters",
        adapter_configs: Optional[dict[str, dict[str, Optional[str]]]] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        adapter_server_model: Optional[str] = None,
        fs_root: Optional[str] = None,
        backend: Optional[ModelBackend] = None,
    ) -> None:
        self.base_model = base_model
        self.adapter_configs = adapter_configs or {}
        self.backend = backend or self._build_backend(
            backend_mode,
            api_key=api_key,
            base_url=base_url,
            adapter_server_model=adapter_server_model,
            fs_root=fs_root,
        )

    def generate(
        self,
        prompt: str,
        adapters: List[dict],
        context_snippets: List[str],
        history: Optional[List[Message]] = None,
        *,
        user_id: Optional[str] = None,
    ) -> dict:
        normalized_adapters = self._normalize_adapters(adapters)
        messages = [{"role": "system", "content": "You are a concise assistant."}]
        messages.extend(self._build_adapter_prompts(normalized_adapters))
        if history:
            for msg in history:
                messages.append({"role": msg.role, "content": msg.content})
            messages = self._inject_context(messages, context_snippets)
        else:
            messages.append({"role": "user", "content": self._format_user(prompt, context_snippets)})
        return self.backend.generate(messages, normalized_adapters, user_id=user_id)

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

    def _normalize_adapters(self, adapters: List[dict]) -> List[dict]:
        normalized = []
        for adapter in adapters or []:
            if isinstance(adapter, str):
                normalized.append({"id": adapter})
            elif isinstance(adapter, dict):
                normalized.append(adapter)
        return normalized

    def _build_adapter_prompts(self, adapters: List[dict]) -> List[dict]:
        prompt_backends = {"prompt", "prompt_distill", "hybrid"}
        lines: List[str] = []
        for adapter in adapters:
            backend = (adapter.get("backend") or "").lower()
            if backend not in prompt_backends:
                continue
            name = adapter.get("name") or adapter.get("id") or adapter.get("base_model") or "adapter"
            instructions = self._extract_prompt_instructions(adapter)
            if instructions:
                lines.append(f"{name}: {instructions}")
        if not lines:
            return []
        joined = "\n".join(f"- {line}" for line in lines)
        return [{"role": "system", "content": f"Adapter guidance:\n{joined}"}]

    def _extract_prompt_instructions(self, adapter: dict) -> str:
        for key in ("prompt_instructions", "behavior_prompt", "instructions", "prompt_template"):
            value = adapter.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        applicability = adapter.get("applicability")
        if isinstance(applicability, dict):
            natural = applicability.get("natural_language")
            if isinstance(natural, str) and natural.strip():
                return natural.strip()
        description = adapter.get("description")
        if isinstance(description, str) and description.strip():
            return description.strip()
        return ""

    def _build_backend(
        self,
        backend_mode: str,
        *,
        api_key: Optional[str],
        base_url: Optional[str],
        adapter_server_model: Optional[str],
        fs_root: Optional[str],
    ) -> ModelBackend:
        mode = (backend_mode or "openai").lower()
        plug = self._resolve_plug(mode)
        if plug:
            plug_config = self.adapter_configs.get(plug.key, {})
            plug_api_key = plug_config.get("api_key", api_key)
            plug_base_url = plug_config.get("base_url", base_url)
            plug_adapter_server_model = plug_config.get("adapter_server_model", adapter_server_model)
            return plug.build_backend(
                base_model=self.base_model,
                api_key=plug_api_key,
                base_url=plug_base_url,
                adapter_server_model=plug_adapter_server_model,
                fs_root=fs_root,
            )
        if mode == "local_lora":
            return LocalJaxLoRABackend(self.base_model, fs_root or "/srv/liminallm")
        if mode not in {"api_adapters", "adapter_server"}:
            mode = "api_adapters"
        return ApiAdapterBackend(
            self.base_model,
            adapter_mode=mode,
            api_key=api_key,
            base_url=base_url,
            adapter_server_model=adapter_server_model,
        )

    def _resolve_plug(self, mode: str) -> AdapterPlug | None:
        if not mode:
            return None
        normalized = mode.lower()
        if normalized in BUILTIN_ADAPTER_PLUGS:
            return BUILTIN_ADAPTER_PLUGS[normalized]
        return None
