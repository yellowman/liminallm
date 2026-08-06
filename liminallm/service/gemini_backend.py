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
from liminallm.service.prompt_utils import extract_prompt_instructions

logger = get_logger(__name__)

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com"

# JSON-Schema keys Gemini's OpenAPI-subset validator rejects outright.
_UNSUPPORTED_SCHEMA_KEYS = {"$schema", "additionalProperties"}


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
                    }
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
            calls.append({
                "id": f"gemini-call-{i}-{fc.get('name') or 'fn'}",
                "name": fc.get("name") or "",
                "arguments": json.dumps(fc.get("args") or {}),
            })
    return calls


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
        transport: Optional[httpx.BaseTransport] = None,
    ) -> None:
        self.base_model = base_model
        self._api_key = api_key or ""
        self._base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self._transport = transport
        self._client: Optional[httpx.Client] = None
        self._context_window: Optional[int] = None

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

    def _prompt_injections(self, adapters: List[dict]) -> Tuple[List[str], List[str]]:
        """Prompt-rung adapters apply; weight-bearing modes cannot reach a
        hosted Gemini model and are dropped with their ids logged."""
        from liminallm.service.model_backend import AdapterMode, get_adapter_mode

        injections: List[str] = []
        applied: List[str] = []
        for adapter in adapters or []:
            adapter_id = adapter.get("id") or adapter.get("name") or "unknown"
            mode = get_adapter_mode(adapter)
            if mode in (AdapterMode.PROMPT, AdapterMode.HYBRID):
                prompt = extract_prompt_instructions(adapter, log_source=adapter_id)
                if prompt:
                    injections.append(prompt)
                    applied.append(f"{adapter_id}:prompt")
            else:
                logger.info(
                    "adapter_dropped_gemini_native", adapter_id=adapter_id, mode=str(mode)
                )
        return injections, applied

    def _request_body(
        self,
        messages: List[dict],
        adapters: List[dict],
        tools: Optional[List[dict]] = None,
    ) -> Tuple[dict, List[str]]:
        injections, applied = self._prompt_injections(adapters)
        augmented = list(messages or [])
        if injections:
            guidance = "Adapter guidance:\n" + "\n\n".join(injections)
            augmented = [{"role": "system", "content": guidance}] + augmented
        system, contents = to_contents(augmented)
        body: Dict[str, Any] = {"contents": contents}
        if system:
            body["systemInstruction"] = system
        if tools:
            decls = to_function_declarations(tools)
            if decls:
                body["tools"] = [{"functionDeclarations": decls}]
        return body, applied

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
        resp = self._http().post(
            self._url(self.base_model, "generateContent"),
            headers=self._headers(), json=body,
        )
        resp.raise_for_status()
        payload = resp.json()
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
        resp = self._http().post(
            self._url(self.base_model, "generateContent"),
            headers=self._headers(), json=body,
        )
        resp.raise_for_status()
        payload = resp.json()
        content = candidate_text(payload)
        calls = function_calls_of(payload)
        from liminallm.service.responses_compat import assistant_message

        return {
            "content": content,
            "tool_calls": calls,
            "assistant_message": assistant_message(content, calls),
            "usage": usage_dict(payload),
        }

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
        body, applied = self._request_body(messages, adapters)
        full_content = ""
        usage: Dict[str, int] = {}
        try:
            with self._http().stream(
                "POST",
                self._url(self.base_model, "streamGenerateContent") + "?alt=sse",
                headers=self._headers(), json=body,
            ) as resp:
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
        except Exception as exc:  # noqa: BLE001 - a stream failure is an event
            logger.error("streaming_error", error=str(exc))
            yield {"event": "error", "data": {"code": "server_error", "message": str(exc)}}
            return
        if not usage:
            usage = {
                "prompt_tokens": 0,
                "completion_tokens": len(full_content.split()),
                "total_tokens": len(full_content.split()),
            }
        yield {
            "event": "message_done",
            "data": {"content": full_content, "usage": usage, "adapters_applied": applied},
        }
