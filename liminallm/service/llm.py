from __future__ import annotations

import os
from typing import Any, Iterator, List, Optional

from liminallm.config import AdapterMode, resolve_provider_endpoint
from liminallm.logging import get_logger
from liminallm.service import local_format
from liminallm.service.model_backend import (
    ApiAdapterBackend,
    LocalJaxLoRABackend,
    ModelBackend,
    StubBackend,
    active_adapters,
    get_adapter_mode,
    mode_value,
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
        reasoning_effort: Optional[str] = None,
        temperature: Optional[float] = None,
        backend: Optional[ModelBackend] = None,
    ) -> None:
        self.base_model = base_model
        self.reasoning_effort = reasoning_effort
        self.temperature = temperature
        self.adapter_configs = adapter_configs or {}
        self.backend = backend or self._build_backend(
            backend_mode,
            api_key=api_key,
            base_url=base_url,
            adapter_server_model=adapter_server_model,
            fs_root=fs_root,
        )

    def _prepare_backend_messages(
        self, messages: List[dict], adapters: Optional[List[dict]]
    ) -> tuple[List[dict], List[dict]]:
        """Messages and adapters ready for any backend (SPEC §5.0.1).

        The single materialization point. It canonicalizes the adapter set
        (gate first, §5.0.1) and places `prompt_instructions` exactly once,
        into a copy of the caller's list, choosing the representation from
        (mode, backend) - weights on a local backend, prompt on an API one.

        Every path into a backend goes through here, which is the whole
        point. When only `generate`/`generate_stream` materialized, the API
        backends materialized too "to be safe", so those two paths sent the
        instructions twice while `generate_with_tools` and `stream_messages`
        - which never passed through the service's message builder - sent
        them once. Removing the backend copy alone would have taken the
        latter pair to zero; giving the service one primitive that every
        entry point uses is what makes one copy true everywhere.

        Guidance goes after any leading system messages, so it sits with the
        rest of the system content rather than ahead of the caller's own
        framing.
        """
        normalized_adapters = self._normalize_adapters(adapters or [])
        guidance = self._build_adapter_prompts(normalized_adapters)
        prepared = list(messages or [])
        if guidance:
            index = 0
            while index < len(prepared) and prepared[index].get("role") == "system":
                index += 1
            prepared[index:index] = guidance
        return prepared, normalized_adapters

    def _prepare_generation(
        self,
        prompt: str,
        adapters: List[dict],
        context_snippets: List[str],
        history: Optional[List[Message]] = None,
        *,
        instruction: Optional[str] = None,
    ) -> tuple[List[dict], List[dict]]:
        """Prepare messages and adapters for generation.

        `instruction` is caller-owned system text - how to write the answer
        this prompt is asking for. It goes on the end of the service's own
        system block, not into the user's message: the user prompt is what
        the person asked, and a rule appended to it reads to the model as
        part of the question and comes back quoted in the answer.

        Returns:
            Tuple of (messages, normalized_adapters) ready for the backend.
        """
        system = "You are a concise assistant."
        if instruction:
            system = f"{system}\n\n{instruction}"
        messages = [{"role": "system", "content": system}]
        if history:
            for msg in history:
                messages.append({"role": msg.role, "content": msg.content})
        messages.append({"role": "user", "content": self._format_user(prompt)})
        messages, normalized_adapters = self._prepare_backend_messages(
            messages, adapters
        )
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
        instruction: Optional[str] = None,
    ) -> dict:
        messages, normalized_adapters = self._prepare_generation(
            prompt, adapters, context_snippets, history, instruction=instruction
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

    @property
    def stream_is_cancellable(self) -> bool:
        """Whether a streamed node can be held to its timeout.

        Fail closed: a backend that says nothing cannot stream. The first
        version defaulted to True on the theory that any generator stops
        between events - but the shipped network backends block *inside* an
        event, in a synchronous read bounded only by the provider client's
        own 30–60s timeout, and a stop flag is not read until that read
        returns. A `timeout_ms: 200` was honoured for the waiter while the
        provider request ran on. SPEC §9.2 makes the timeout part of the
        node contract, and a capability that cannot be proven is not claimed.

        `supports_stream_cancel = True` therefore asserts the full contract:
        the backend's stream carries an `abort()` that interrupts a read in
        flight (`CancellableStream`), or it never blocks at all (the stub).
        A backend without the declaration does not stream - the node runs on
        the ordinary executor, whose deadline the driver enforces, and its
        answer reaches the client in the final `message_done`.
        """
        return bool(getattr(self.backend, "supports_stream_cancel", False))

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
        prepared, normalized_adapters = self._prepare_backend_messages(
            messages, adapters
        )
        return self.backend.generate_with_tools(
            prepared,
            tools,
            normalized_adapters,
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
        prepared, normalized_adapters = self._prepare_backend_messages(
            messages, adapters
        )
        return self.backend.generate_stream(
            prepared, normalized_adapters, user_id=user_id
        )

    @property
    def serving_model(self) -> str:
        """The model that will actually answer, not the one configured.

        An adapter server overrides the base model, and both live on the
        backend rather than here. Anything deciding by model identity - the
        tokenizer, the context window, whether a listwise rerank is a
        reasonable ask - has to resolve the pair the same way, so it resolves
        it here once. Read off ``self`` it silently returned nothing, and the
        caller fell back to the configured base model without noticing.
        """
        return str(
            getattr(self.backend, "adapter_server_model", None)
            or getattr(self.backend, "base_model", None)
            # Last resort: not every backend carries the pair (the stub does
            # not), and the configured base is a better answer than nothing.
            or self.base_model
            or ""
        )

    def token_counter(self):
        """Counter for the serving model: exact when we own its tokenizer."""
        from liminallm.service.token_counting import counter_for

        model = self.serving_model
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
        implements its own ``transcribe_image`` (a local multimodal model -
        PaliGemma/LLaVA class - would) is used directly; otherwise any
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
        instruction: Optional[str] = None,
    ) -> Iterator[dict]:
        """Stream tokens from the LLM per SPEC §13.7.

        Yields events:
        - {"event": "token", "data": "token_text"}
        - {"event": "message_done", "data": {"content": "full_text", "usage": {...}}}
        - {"event": "error", "data": {"code": "...", "message": "..."}}

        Returns the backend's own iterator rather than `yield from`-ing it:
        wrapping in a generator here hid the `abort()` a cancellable stream
        carries, so the pump could not interrupt a blocked read.
        """
        messages, normalized_adapters = self._prepare_generation(
            prompt, adapters, context_snippets, history, instruction=instruction
        )
        return self.backend.generate_stream(
            messages, normalized_adapters, user_id=user_id
        )

    def _format_user(self, prompt: str) -> str:
        return prompt

    def _inject_context(
        self, messages: List[dict], context_snippets: List[str]
    ) -> List[dict]:
        if not context_snippets:
            return list(messages)
        if self._backend_applies_lora_weights:
            # SPEC §5.1: the local decoder gets ONE representation, the same
            # one training wrote - marker AND placement, since token order is
            # part of the input for a raw decoder.
            return local_format.place_context(
                [dict(msg) for msg in messages], context_snippets
            )
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
        """The effective adapter set, built once for every path (§5.0.1).

        Gate first, mechanism second: `g == 0` means the adapter is absent
        from the request, so it is dropped here - before prompt injection,
        weight loading, remote passthrough, KV hashing or accounting can see
        it. This is the only funnel into a backend (generate, generate_stream,
        chat and complete all pass through it), which is what keeps those
        five surfaces from disagreeing about which adapters are active.
        """
        normalized = []
        for adapter in adapters or []:
            if isinstance(adapter, str):
                normalized.append({"id": adapter})
            elif isinstance(adapter, dict):
                normalized.append(adapter)
        return active_adapters(normalized)

    @property
    def _backend_applies_lora_weights(self) -> bool:
        """Whether the active backend loads LoRA weights itself."""
        return bool(getattr(self.backend, "applies_lora_weights", False))

    def _build_adapter_prompts(self, adapters: List[dict]) -> List[dict]:
        lines: List[str] = []
        for adapter in adapters:
            # `mode` is authoritative (SPEC §5.0.1); the legacy backend field
            # is only an inference source when mode is absent. Deciding from
            # `backend` directly let `mode: hybrid, backend: prompt` receive
            # both the weights and the prompt after promotion, and
            # `mode: prompt, backend: local` receive neither.
            # Compared as the value, not as `str()` of it. `get_adapter_mode`
            # returns the raw string when the artifact states one and an
            # `AdapterMode` member when it infers, and `str(AdapterMode.HYBRID)`
            # is "AdapterMode.HYBRID" - which matches nothing, so every adapter
            # that did not state a mode was silently skipped here. That is the
            # documented default for legacy adapters, and it went unnoticed
            # because the API backends were injecting the prompt themselves;
            # with materialization now solely the service's (§5.0.1), the same
            # bug would drop those adapters' instructions entirely.
            mode = mode_value(get_adapter_mode(adapter))
            if mode not in {AdapterMode.PROMPT, AdapterMode.HYBRID}:
                continue
            if mode == AdapterMode.HYBRID and self._backend_applies_lora_weights:
                # SPEC §5.0.1: for a hybrid adapter the prompt is the
                # *portable fallback* - it carries the behaviour on API
                # backends, while a local backend applies the trained
                # weights. Injecting both meant a graduated skill served its
                # weights AND the instructions they were distilled from, so
                # the model saw an input the eval gate never scored.
                schema = adapter.get("schema") if isinstance(adapter.get("schema"), dict) else {}
                version = adapter.get("current_version")
                if version is None:
                    version = (schema or {}).get("current_version")
                try:
                    promoted = int(version or 0) > 0
                except (TypeError, ValueError):
                    promoted = False
                if promoted:
                    continue
                # Hybrid but nothing promoted yet: no weights will load, so
                # the fallback is all there is.
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
        if mode == "gemini_native":
            from liminallm.service.gemini_backend import GeminiBackend

            override = (
                self.adapter_configs.get("gemini_native")
                or self.adapter_configs.get("gemini")
                or self.adapter_configs.get("openai")
                or {}
            )
            # Same env resolution as the compat providers: the variable name
            # comes from config's provider table, not a literal here.
            api_key_env = (resolve_provider_endpoint("gemini") or {}).get("api_key_env")
            return GeminiBackend(
                self.base_model,
                api_key=override.get("api_key") or api_key
                or (os.getenv(api_key_env) if api_key_env else None),
                base_url=override.get("base_url") or base_url,
                reasoning_effort=self.reasoning_effort,
                temperature=self.temperature,
            )

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
                reasoning_effort=self.reasoning_effort,
                temperature=self.temperature,
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
            reasoning_effort=self.reasoning_effort,
        )
