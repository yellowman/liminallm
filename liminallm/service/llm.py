from __future__ import annotations

import os
from typing import Any, Iterator, List, Optional

from liminallm.config import resolve_provider_endpoint
from liminallm.logging import get_logger
from liminallm.service.model_backend import (
    ApiAdapterBackend,
    LocalJaxLoRABackend,
    ModelBackend,
    StubBackend,
)
from liminallm.storage.models import Message

logger = get_logger(__name__)


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

    @property
    def supports_tools(self) -> bool:
        """Whether the active backend can do model-initiated tool calls.

        A backend may implement the method but still be unusable (no API key,
        so no client); it reports that through its own ``supports_tools``.
        """
        if not callable(getattr(self.backend, "generate_with_tools", None)):
            return False
        backend_flag = getattr(self.backend, "supports_tools", None)
        return bool(backend_flag) if backend_flag is not None else True

    def generate_with_tools(
        self,
        messages: List[dict],
        tools: List[dict],
        adapters: Optional[List[dict]] = None,
        *,
        user_id: Optional[str] = None,
    ) -> dict:
        """One tool-calling turn over a caller-built message list.

        Unlike generate(), the caller owns the messages so it can append tool
        results and iterate.
        """
        if not self.supports_tools:
            raise RuntimeError("active backend does not support tool calling")
        return self.backend.generate_with_tools(
            messages,
            tools,
            self._normalize_adapters(adapters or []),
            user_id=user_id,
        )

    def stream_messages(
        self,
        messages: List[dict],
        adapters: Optional[List[dict]] = None,
        *,
        user_id: Optional[str] = None,
    ) -> Iterator[dict]:
        """Stream a reply for a caller-built message list.

        Used by the attachment agent to stream its final answer after the
        tool-calling rounds have assembled the message history.
        """
        return self.backend.generate_stream(
            messages, self._normalize_adapters(adapters or []), user_id=user_id
        )

    def token_counter(self):
        """Counter for the serving model: exact when we own its tokenizer."""
        from liminallm.service.token_counting import counter_for

        model = (
            getattr(self.backend, "adapter_server_model", None)
            or getattr(self.backend, "base_model", None)
            or ""
        )
        # Local backends load their tokenizer lazily; force it, or the first
        # turn would resolve to the heuristic and cache that decision.
        tokenizer = None
        getter = getattr(self.backend, "get_tokenizer", None)
        if callable(getter):
            try:
                tokenizer = getter()
            except Exception as exc:  # noqa: BLE001 - fall back to estimate
                logger.debug("tokenizer_unavailable", error=str(exc))
        return counter_for(model, tokenizer=tokenizer)

    def observe_usage(self, estimated_prompt_tokens: int, usage: Any) -> None:
        """Feed a provider's reported prompt_tokens back into calibration."""
        try:
            actual = int((usage or {}).get("prompt_tokens") or 0)
        except (AttributeError, TypeError, ValueError):
            return
        if actual > 0:
            self.token_counter().observe(estimated_prompt_tokens, actual)

    def context_window(self) -> int:
        """The serving model's input window (probed/table/config, see backend)."""
        from liminallm.service.model_backend import DEFAULT_CONTEXT_WINDOW

        try:
            window = getattr(self.backend, "context_window", None)
            if isinstance(window, int) and window > 0:
                return window
        except Exception as exc:  # noqa: BLE001 - discovery must not break chat
            logger.warning("context_window_resolution_failed", error=str(exc))
        return DEFAULT_CONTEXT_WINDOW

    def transcribe_image(self, image_bytes: bytes, mime: str, *, prompt: str) -> str:
        """One vision call: read an image with the configured model.

        Capability is probed, never assumed from backend type: a backend that
        implements its own ``transcribe_image`` (a local multimodal model —
        PaliGemma/LLaVA class — would) is used directly; otherwise any
        OpenAI-compatible client gets the standard content-parts message.
        Backends with neither raise NotImplementedError so callers can refuse
        cleanly instead of hallucinating a transcription. (Today's local JAX
        stack is a text LM + tokenizer, so it lands in that last bucket until
        someone loads a vision tower and implements the hook.)
        """
        import base64

        backend_hook = getattr(self.backend, "transcribe_image", None)
        if callable(backend_hook):
            return backend_hook(image_bytes, mime, prompt=prompt) or ""
        if getattr(self.backend, "client", None) is None:
            raise NotImplementedError("backend cannot read images")
        data_url = (
            f"data:{mime};base64,{base64.b64encode(image_bytes).decode('ascii')}"
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ]
        result = self.backend.generate(messages, [], user_id=None)
        return (result or {}).get("content") or ""

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
