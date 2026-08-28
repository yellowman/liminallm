"""Native Gemini backend: generativelanguage.googleapis.com, no compat shim.

The OpenAI-compat shim (`/v1beta/openai`) stays available as
model_backend=gemini; this backend (model_backend=gemini_native) speaks the
native API — generateContent / streamGenerateContent?alt=sse — for the
capabilities the shim flattens: thoughtsTokenCount and cachedContentTokenCount
in usageMetadata (the same rich keys the Responses path keeps), and native
function calling.

The internal message shape stays chat-completions format, exactly as with the
Responses path: a structured history — system prompt, turns, assistant
tool_calls, role:"tool" results — converts to native `contents` here, so a
conversation resumes mid-history on this provider the same as on any other.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterator, List, Optional, Tuple

import httpx

from liminallm.logging import get_logger
from liminallm.service.model_backend import CancellableStream, StreamAbortHandle
from liminallm.service.prompt_utils import extract_prompt_instructions
from liminallm.service.tokenizer_utils import estimate_token_count

logger = get_logger(__name__)

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com"

# JSON-Schema keys Gemini's OpenAPI-subset validator rejects outright.
_UNSUPPORTED_SCHEMA_KEYS = {"$schema", "additionalProperties"}

# Gemini attaches a thoughtSignature to functionCall parts and rejects a
# resumed history whose functionCall lacks one (INVALID_ARGUMENT, live).
# The signature rides the chat-shaped assistant_message as a vendor extra so
# the provider-agnostic loop round-trips it untouched. For a history built
# elsewhere (another provider, a hand-written test), Google documents this
# placeholder as the accepted stand-in:
# https://ai.google.dev/gemini-api/docs/thought-signatures
THOUGHT_SIGNATURE_PLACEHOLDER = "context_engineering_is_the_way_to_go"

# model_reasoning_effort -> generationConfig.thinkingConfig.thinkingLevel.
# The native vocabulary is the setting's own, so low/medium/high pass through.
# "none" is not an accepted level — sending it is a 400 — and "minimal" is the
# floor the API does accept: it answers with zero thought tokens. Verified
# live against gemini-flash-latest (minimal: no thoughtsTokenCount at all;
# low: 87; medium: 164; high: 145 against an unconfigured 158).
_THINKING_LEVELS = {
    "none": "minimal", "minimal": "minimal",
    "low": "low", "medium": "medium", "high": "high",
}


def thinking_config(effort: Optional[str]) -> Optional[dict]:
    """The configured reasoning effort as a thinkingConfig, or None to omit."""
    level = _THINKING_LEVELS.get((effort or "").strip().lower())
    return {"thinkingLevel": level} if level else None


# ---------------------------------------------------------------------------
# Chat-shape -> native conversion (pure, tested directly)
# ---------------------------------------------------------------------------


def _scrub_schema(schema: Any) -> Any:
    if isinstance(schema, dict):
        return {
            k: _scrub_schema(v)
            for k, v in schema.items()
            if k not in _UNSUPPORTED_SCHEMA_KEYS
        }
    if isinstance(schema, list):
        return [_scrub_schema(v) for v in schema]
    return schema


def to_function_declarations(tools: List[dict]) -> List[dict]:
    """Chat tool declarations -> Gemini functionDeclarations."""
    decls = []
    for tool in tools or []:
        fn = tool.get("function") or {}
        if tool.get("type") == "function" and fn:
            decls.append({
                "name": fn.get("name") or "",
                "description": fn.get("description") or "",
                "parameters": _scrub_schema(
                    fn.get("parameters") or {"type": "object", "properties": {}}
                ),
            })
    return decls


def _text_of(content: Any) -> str:
    if isinstance(content, str) or content is None:
        return content or ""
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
            elif isinstance(part, dict):
                parts.append(json.dumps(part))
            else:
                parts.append(str(part))
        return "\n".join(p for p in parts if p)
    return str(content)


def _parse_args(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    except Exception:
        return {"raw": str(raw)}


def to_contents(messages: List[dict]) -> Tuple[Optional[dict], List[dict]]:
    """A chat-completions history as (systemInstruction, contents).

    Structured resume is the whole point: system prompts hoist into
    systemInstruction, assistant tool_calls become model functionCall parts,
    role:"tool" results become user functionResponse parts, and consecutive
    same-role entries merge — Gemini expects alternating roles, and the
    injected adapter-guidance system blocks plus tool results otherwise
    produce runs the API rejects.
    """
    system_texts: List[str] = []
    contents: List[dict] = []

    def emit(role: str, parts: List[dict]) -> None:
        if not parts:
            return
        if contents and contents[-1]["role"] == role:
            contents[-1]["parts"].extend(parts)
        else:
            contents.append({"role": role, "parts": parts})

    for msg in messages or []:
        role = msg.get("role") or "user"
        if role == "system":
            text = _text_of(msg.get("content"))
            if text:
                system_texts.append(text)
            continue
        if role == "tool":
            emit("user", [{
                "functionResponse": {
                    "name": msg.get("name") or msg.get("tool_call_id") or "tool",
                    "response": {"output": _text_of(msg.get("content"))},
                }
            }])
            continue
        if role == "assistant":
            parts: List[dict] = []
            text = _text_of(msg.get("content"))
            if text:
                parts.append({"text": text})
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                parts.append({
                    "functionCall": {
                        "name": fn.get("name") or tc.get("name") or "",
                        "args": _parse_args(fn.get("arguments") or tc.get("arguments")),
                    },
                    "thoughtSignature": tc.get("thought_signature")
                    or THOUGHT_SIGNATURE_PLACEHOLDER,
                })
            emit("model", parts)
            continue
        text = _text_of(msg.get("content"))
        emit("user", [{"text": text}] if text else [])

    system = {"parts": [{"text": "\n\n".join(system_texts)}]} if system_texts else None
    return system, contents


# ---------------------------------------------------------------------------
# Native -> internal conversion
# ---------------------------------------------------------------------------


def usage_dict(payload: dict) -> Dict[str, int]:
    """usageMetadata mapped to the internal keys plus the rich ones —
    thoughtsTokenCount and cachedContentTokenCount ride as reasoning_tokens
    and cached_tokens, summing across agent rounds like every provider's."""
    meta = payload.get("usageMetadata") or {}
    prompt = int(meta.get("promptTokenCount") or 0)
    completion = int(meta.get("candidatesTokenCount") or 0)
    usage = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": int(meta.get("totalTokenCount") or (prompt + completion)),
    }
    if meta.get("thoughtsTokenCount"):
        usage["reasoning_tokens"] = int(meta["thoughtsTokenCount"])
    if meta.get("cachedContentTokenCount"):
        usage["cached_tokens"] = int(meta["cachedContentTokenCount"])
    return usage


def _candidate_parts(payload: dict) -> List[dict]:
    candidates = payload.get("candidates") or []
    if not candidates:
        return []
    return (candidates[0].get("content") or {}).get("parts") or []


def candidate_text(payload: dict) -> str:
    return "".join(
        p.get("text") or "" for p in _candidate_parts(payload) if "text" in p
    )


def function_calls_of(payload: dict) -> List[Dict[str, str]]:
    """functionCall parts in the internal {id, name, arguments} shape.

    Gemini carries no call id; a synthetic one keyed by position keeps the
    loop's bookkeeping working, and the resume path keys functionResponse by
    name, so nothing downstream depends on the id surviving a round trip.
    """
    calls = []
    for i, part in enumerate(_candidate_parts(payload)):
        fc = part.get("functionCall")
        if fc:
            call = {
                "id": f"gemini-call-{i}-{fc.get('name') or 'fn'}",
                "name": fc.get("name") or "",
                "arguments": json.dumps(fc.get("args") or {}),
            }
            if part.get("thoughtSignature"):
                call["thought_signature"] = part["thoughtSignature"]
            calls.append(call)
    return calls


def _assistant_message(content: str, calls: List[Dict[str, str]]) -> Dict[str, Any]:
    """Chat-shaped assistant message, carrying each call's thoughtSignature as
    a vendor extra so the loop's verbatim round trip preserves it."""
    msg: Dict[str, Any] = {"role": "assistant", "content": content or None}
    if calls:
        msg["tool_calls"] = [
            {
                "id": c["id"],
                "type": "function",
                "function": {"name": c["name"], "arguments": c["arguments"]},
                **({"thought_signature": c["thought_signature"]}
                   if c.get("thought_signature") else {}),
            }
            for c in calls
        ]
    return msg


# ---------------------------------------------------------------------------
# The backend
# ---------------------------------------------------------------------------


class GeminiBackend:
    """ModelBackend speaking the native Gemini API over httpx."""

    mode = "gemini_native"
    provider = "gemini"

    def __init__(
        self,
        base_model: str,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        temperature: Optional[float] = None,
        transport: Optional[httpx.BaseTransport] = None,
    ) -> None:
        self.base_model = base_model
        self._api_key = api_key or ""
        self._base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self._reasoning_effort = (reasoning_effort or "").strip().lower() or None
        self._temperature = temperature
        self._transport = transport
        self._client: Optional[httpx.Client] = None
        self._context_window: Optional[int] = None
        # None = not yet contradicted, False = this model predates
        # thinkingConfig and 400s on it (sticky for the process).
        self._thinking_ok: Optional[bool] = None

    # -- plumbing ----------------------------------------------------------

    def _http(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                timeout=httpx.Timeout(60.0, connect=10.0),
                transport=self._transport,
            )
        return self._client

    def _url(self, model: str, verb: str) -> str:
        return f"{self._base_url}/v1beta/models/{model}:{verb}"

    def _headers(self) -> dict:
        return {"x-goog-api-key": self._api_key, "Content-Type": "application/json"}

    def _applied_prompt_adapters(self, adapters: List[dict]) -> List[str]:
        """Which prompt-rung adapters this turn carries, for accounting.

        It reports; it does not materialize. SPEC §5.0.1 gives prompt
        materialization to LLMService, which every path into a backend passes
        through — this method used to extract the text and `_request_body`
        prepended a second guidance block on top of the one already there.
        Weight-bearing modes cannot reach a hosted Gemini model and are
        dropped with their ids logged.
        """
        from liminallm.service.model_backend import (
            AdapterMode,
            active_adapters,
            get_adapter_mode,
        )

        applied: List[str] = []
        # §5.0.1: `g == 0` is absent, not dropped — it was never requested.
        for adapter in active_adapters(adapters):
            adapter_id = adapter.get("id") or adapter.get("name") or "unknown"
            mode = get_adapter_mode(adapter)
            if mode in (AdapterMode.PROMPT, AdapterMode.HYBRID):
                if extract_prompt_instructions(adapter, log_source=adapter_id):
                    applied.append(f"{adapter_id}:prompt")
            else:
                logger.info(
                    "adapter_dropped_gemini_native", adapter_id=adapter_id, mode=str(mode)
                )
        return applied

    def _request_body(
        self,
        messages: List[dict],
        adapters: List[dict],
        tools: Optional[List[dict]] = None,
    ) -> Tuple[dict, List[str]]:
        applied = self._applied_prompt_adapters(adapters)
        # Messages arrive materialized (SPEC §5.0.1); prepending guidance
        # here put the adapter's instructions in the request twice.
        system, contents = to_contents(list(messages or []))
        body: Dict[str, Any] = {"contents": contents}
        if system:
            body["systemInstruction"] = system
        if tools:
            decls = to_function_declarations(tools)
            if decls:
                body["tools"] = [{"functionDeclarations": decls}]
        generation: Dict[str, Any] = {}
        thinking = thinking_config(self._reasoning_effort)
        if thinking and self._thinking_ok is not False:
            generation["thinkingConfig"] = thinking
        # Same policy as the compat backends: Gemini 3 deprecated sampling
        # parameters and warns that lowering temperature can drive it into
        # loops, so only a 2.5-class model takes a configured value.
        from liminallm.service.model_backend import temperature_param

        generation.update(temperature_param(
            self.base_model,
            configured=self._temperature,
            reasoning_effort=self._reasoning_effort,
        ))
        if generation:
            body["generationConfig"] = generation
        return body, applied

    def _drop_rejected_thinking(self, body: dict, status: int, text: str) -> bool:
        """Did this 400 come from thinkingConfig, and can we retry without it?

        A model older than thinking rejects the field outright. Since the
        admin set one effort for whichever backend is serving, dropping it for
        this process — loudly — beats failing every request.
        """
        if status != 400 or "thinking" not in (text or "").lower():
            return False
        config = body.get("generationConfig") or {}
        if not config.pop("thinkingConfig", None):
            return False
        if not config:
            body.pop("generationConfig", None)
        self._thinking_ok = False
        logger.warning(
            "gemini_thinking_config_unsupported",
            model=self.base_model, effort=self._reasoning_effort,
        )
        return True

    def _post(self, verb: str, body: dict) -> httpx.Response:
        url = self._url(self.base_model, verb)
        resp = self._http().post(url, headers=self._headers(), json=body)
        if self._drop_rejected_thinking(body, resp.status_code, resp.text):
            resp = self._http().post(url, headers=self._headers(), json=body)
        resp.raise_for_status()
        return resp

    # -- ModelBackend ------------------------------------------------------

    @property
    def supports_tools(self) -> bool:
        return bool(self._api_key)

    @property
    def context_window(self) -> int:
        """models/{id} states inputTokenLimit; table, then default, backstop."""
        from liminallm.service.model_backend import (
            DEFAULT_CONTEXT_WINDOW,
            context_window_from_table,
        )

        if self._context_window is None:
            window = 0
            try:
                resp = self._http().get(
                    f"{self._base_url}/v1beta/models/{self.base_model}",
                    headers=self._headers(),
                )
                if resp.status_code == 200:
                    window = int(resp.json().get("inputTokenLimit") or 0)
            except Exception as exc:  # noqa: BLE001 - probe is best effort
                logger.debug("gemini_window_probe_failed", error=str(exc))
            self._context_window = (
                window
                or context_window_from_table(self.base_model)
                or DEFAULT_CONTEXT_WINDOW
            )
            logger.info(
                "model_context_window_resolved",
                model=self.base_model, window=self._context_window,
                source="probe" if window else "table/default",
            )
        return self._context_window

    def generate(
        self,
        messages: List[dict],
        adapters: List[dict],
        *,
        user_id: Optional[str] = None,
    ) -> dict:
        body, applied = self._request_body(messages, adapters)
        payload = self._post("generateContent", body).json()
        return {
            "content": candidate_text(payload),
            "usage": usage_dict(payload),
            "adapters_applied": applied,
        }

    def generate_with_tools(
        self,
        messages: List[dict],
        tools: List[dict],
        adapters: List[dict],
        *,
        user_id: Optional[str] = None,
    ) -> dict:
        body, _ = self._request_body(messages, adapters, tools=tools)
        payload = self._post("generateContent", body).json()
        content = candidate_text(payload)
        calls = function_calls_of(payload)
        return {
            "content": content,
            "tool_calls": calls,
            "assistant_message": _assistant_message(content, calls),
            "usage": usage_dict(payload),
        }

    #: The stream below attaches its response's socket to the abort handle,
    #: so a read blocked mid-stream can be interrupted from another thread.
    #: Declared only because that handle exists — see
    #: `ModelBackend.generate_stream`.
    supports_stream_cancel = True

    def generate_stream(
        self,
        messages: List[dict],
        adapters: List[dict],
        *,
        user_id: Optional[str] = None,
    ) -> Iterator[dict]:
        """SSE over streamGenerateContent?alt=sse: each `data:` line is a
        chunk whose candidate parts carry text deltas; the last one carries
        usageMetadata."""
        handle = StreamAbortHandle()
        return CancellableStream(
            self._generate_stream_impl(messages, adapters, handle), handle
        )

    def _generate_stream_impl(
        self,
        messages: List[dict],
        adapters: List[dict],
        abort_handle: StreamAbortHandle,
    ) -> Iterator[dict]:
        body, applied = self._request_body(messages, adapters)
        url = self._url(self.base_model, "streamGenerateContent") + "?alt=sse"
        full_content = ""
        usage: Dict[str, int] = {}
        try:
            # Two attempts at most: the second only happens when the first was
            # a 400 blaming thinkingConfig, which _drop_rejected_thinking has
            # then removed from the body for good.
            for attempt in (1, 2):
                with self._http().stream(
                    "POST", url, headers=self._headers(), json=body,
                ) as resp:
                    abort_handle.attach_response(resp)
                    if attempt == 1 and resp.status_code == 400:
                        resp.read()
                        if self._drop_rejected_thinking(body, 400, resp.text):
                            continue
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        data = line[len("data:"):].strip()
                        if not data or data == "[DONE]":
                            continue
                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        delta = candidate_text(chunk)
                        if delta:
                            full_content += delta
                            yield {"event": "token", "data": delta}
                        if chunk.get("usageMetadata"):
                            usage = usage_dict(chunk)
                break
        except Exception as exc:  # noqa: BLE001 - a stream failure is an event
            logger.error("streaming_error", error=str(exc))
            yield {"event": "error", "data": {"code": "server_error", "message": str(exc)}}
            return
        if not usage:
            # No chunk carried usageMetadata. The shared estimator, not a word
            # count: word counts undercount CJK roughly fourfold.
            estimated = estimate_token_count(full_content)
            usage = {
                "prompt_tokens": 0,
                "completion_tokens": estimated,
                "total_tokens": estimated,
            }
        yield {
            "event": "message_done",
            "data": {"content": full_content, "usage": usage, "adapters_applied": applied},
        }
