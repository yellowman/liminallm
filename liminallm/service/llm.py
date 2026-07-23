from __future__ import annotations

from typing import Iterator, List, Optional

import os

from liminallm.config import resolve_provider_endpoint
from liminallm.service.model_backend import (
    ApiAdapterBackend,
    LocalJaxLoRABackend,
    ModelBackend,
    StubBackend,
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

    def _prepare_generation(
        self,
        prompt: str,
        adapters: List[dict],
        context_snippets: List[str],
        history: Optional[List[Message]] = None,
    ) -> tuple[List[dict], List[dict]]:
        """Prepare messages and adapters for generation.

        Returns:
            Tuple of (messages, normalized_adapters) ready for the backend.
        """
        normalized_adapters = self._normalize_adapters(adapters)
        messages = [{"role": "system", "content": "You are a concise assistant."}]
        messages.extend(self._build_adapter_prompts(normalized_adapters))
        if history:
            for msg in history:
                messages.append({"role": msg.role, "content": msg.content})
        messages.append({"role": "user", "content": self._format_user(prompt)})
        messages = self._inject_context(messages, context_snippets)
        return messages, normalized_adapters

    def generate(
        self,
        prompt: str,
        adapters: List[dict],
        context_snippets: List[str],
        history: Optional[List[Message]] = None,
        *,
        user_id: Optional[str] = None,
    ) -> dict:
        messages, normalized_adapters = self._prepare_generation(
            prompt, adapters, context_snippets, history
        )
        return self.backend.generate(messages, normalized_adapters, user_id=user_id)

    def generate_stream(
        self,
        prompt: str,
        adapters: List[dict],
        context_snippets: List[str],
        history: Optional[List[Message]] = None,
        *,
        user_id: Optional[str] = None,
    ) -> Iterator[dict]:
        """Stream tokens from the LLM per SPEC §18.

        Yields events:
        - {"event": "token", "data": "token_text"}
        - {"event": "message_done", "data": {"content": "full_text", "usage": {...}}}
        - {"event": "error", "data": {"code": "...", "message": "..."}}
        """
        messages, normalized_adapters = self._prepare_generation(
            prompt, adapters, context_snippets, history
        )
        yield from self.backend.generate_stream(messages, normalized_adapters, user_id=user_id)

    def _format_user(self, prompt: str) -> str:
        return prompt

    def _inject_context(
        self, messages: List[dict], context_snippets: List[str]
    ) -> List[dict]:
        if not context_snippets:
            return list(messages)
        updated: List[dict] = [dict(msg) for msg in messages]
        for idx in range(len(updated) - 1, -1, -1):
            msg = updated[idx]
            if msg.get("role") == "user":
                context_text = f"Context: {' | '.join(context_snippets)}"
                content = msg.get("content", "")
                if context_text not in content:
                    content = f"{content}\n{context_text}"
                msg["content"] = content
                updated[idx] = msg
                return updated
        updated.append(
            {"role": "system", "content": f"Context: {' | '.join(context_snippets)}"}
        )
        return updated

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
            name = (
                adapter.get("name")
                or adapter.get("id")
                or adapter.get("base_model")
                or "adapter"
            )
            instructions = self._extract_prompt_instructions(adapter)
            if instructions:
                lines.append(f"{name}: {instructions}")
        if not lines:
            return []
        joined = "\n".join(f"- {line}" for line in lines)
        return [{"role": "system", "content": f"Adapter guidance:\n{joined}"}]

    def _extract_prompt_instructions(self, adapter: dict) -> str:
        """Extract prompt instructions using shared utility for consistency.

        See liminallm.service.prompt_utils.extract_prompt_instructions for
        the canonical implementation and priority order per SPEC §5.0.1.
        """
        from liminallm.service.prompt_utils import extract_prompt_instructions

        adapter_id = adapter.get("id") or adapter.get("name") or "unknown"
        result = extract_prompt_instructions(adapter, log_source=adapter_id)
        return result or ""

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
        # Stub backend for testing - returns canned responses
        if mode == "stub":
            return StubBackend()
        if mode in {"local_lora", "local_gpu_lora"}:
            return LocalJaxLoRABackend(self.base_model, fs_root or "/srv/liminallm")

        # OpenAI-compatible API providers (openai, anthropic, zhipu/glm, together,
        # gemini). Each resolves credentials as: explicit adapter_configs override,
        # then the caller-supplied key (openai only), then the provider's env var.
        endpoint = resolve_provider_endpoint(mode)
        if endpoint:
            provider = endpoint["provider"]
            override = (
                self.adapter_configs.get(mode)
                or self.adapter_configs.get(provider)
                or {}
            )
            fallback_key = api_key if provider == "openai" else None
            fallback_base = base_url if provider == "openai" else None
            api_key_env = endpoint["api_key_env"]
            resolved_key = (
                override.get("api_key")
                or fallback_key
                or (os.getenv(api_key_env) if api_key_env else None)
            )
            resolved_base = (
                override.get("base_url") or fallback_base or endpoint["base_url"]
            )
            return ApiAdapterBackend(
                self.base_model,
                adapter_mode="api_adapters",
                api_key=resolved_key,
                base_url=resolved_base,
                adapter_server_model=adapter_server_model,
                provider=provider,
                api_key_env=api_key_env,
            )

        # adapter_server and other adapter-id providers (azure, vertex, bedrock,
        # lorax, sagemaker). Preserve the provider so capabilities resolve
        # correctly rather than defaulting everything to OpenAI.
        adapter_mode = mode if mode in {"api_adapters", "adapter_server"} else "api_adapters"
        provider = None if mode in {"api_adapters", "adapter_server"} else mode
        return ApiAdapterBackend(
            self.base_model,
            adapter_mode=adapter_mode,
            api_key=api_key,
            base_url=base_url,
            adapter_server_model=adapter_server_model,
            provider=provider,
        )
