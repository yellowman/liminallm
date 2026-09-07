"""Chat-shape <-> Responses-shape conversion for OpenAI-compatible backends.

The Responses API is the primary endpoint for OpenAI and compatible
providers: richer usage (reasoning and cached-token counts), typed output
items, and first-class reasoning control. The rest of this codebase - the
agent loop, adapters, history assembly - speaks chat-completions shape, and
providers that only ship /chat/completions still exist, so the chat shape
stays the internal lingua franca and this module translates at the wire.

APIBackend probes once per process: first call tries /responses; a 404/405
(or an SDK without the surface) marks the provider chat-only for good.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


def is_unsupported(exc: Exception) -> bool:
    """Does this failure mean "the provider has no /responses endpoint"?

    404/405 are the honest signals. Some compat gateways answer 400/501 with
    prose naming the path; accept those only when the message points at the
    endpoint, so a genuine bad request is never misread as "unsupported" and
    silently retried against chat/completions.
    """
    if isinstance(exc, (AttributeError, TypeError)):
        return True  # SDK predates client.responses / its kwargs
    status = getattr(exc, "status_code", None)
    if status in (404, 405):
        return True
    if status in (400, 501):
        text = str(exc).lower()
        return "responses" in text and any(
            marker in text
            for marker in ("unknown", "not found", "unsupported", "invalid url", "no route")
        )
    return False


def _content_parts(content: Any, *, assistant: bool) -> Any:
    """Chat content (str or parts list) -> Responses message content."""
    if isinstance(content, str) or content is None:
        return content or ""
    if not isinstance(content, list):
        return str(content)
    out = []
    text_type = "output_text" if assistant else "input_text"
    for part in content:
        if not isinstance(part, dict):
            out.append({"type": text_type, "text": str(part)})
        elif part.get("type") in ("text", "input_text", "output_text"):
            out.append({"type": text_type, "text": part.get("text") or ""})
        elif part.get("type") == "image_url":
            url = part.get("image_url")
            url = url.get("url") if isinstance(url, dict) else url
            out.append({"type": "input_image", "image_url": url})
        else:
            out.append({"type": text_type, "text": json.dumps(part)})
    return out


def to_input_items(messages: List[dict]) -> List[dict]:
    """A chat-completions history - including the agent loop's assistant
    tool_calls and role:"tool" results - as Responses input items."""
    items: List[dict] = []
    for msg in messages or []:
        role = msg.get("role") or "user"
        if role == "tool":
            items.append({
                "type": "function_call_output",
                "call_id": msg.get("tool_call_id") or msg.get("name") or "",
                "output": str(msg.get("content") or ""),
            })
            continue
        if role == "assistant" and msg.get("tool_calls"):
            content = msg.get("content")
            if content:
                items.append({
                    "role": "assistant",
                    "content": _content_parts(content, assistant=True),
                })
            for tc in msg["tool_calls"]:
                fn = tc.get("function") or {}
                items.append({
                    "type": "function_call",
                    "call_id": tc.get("id") or "",
                    "name": fn.get("name") or tc.get("name") or "",
                    "arguments": fn.get("arguments") or tc.get("arguments") or "{}",
                })
            continue
        items.append({
            "role": role,
            "content": _content_parts(msg.get("content"), assistant=(role == "assistant")),
        })
    return items


def to_tools(tools: List[dict]) -> List[dict]:
    """Chat tool declarations (nested under "function") -> Responses' flat form."""
    out = []
    for tool in tools or []:
        fn = tool.get("function") or {}
        if tool.get("type") == "function" and fn:
            out.append({
                "type": "function",
                "name": fn.get("name") or "",
                "description": fn.get("description") or "",
                "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
            })
        else:
            out.append(tool)
    return out


#: The fields the wire documents as output-only, by item type. A field
#: listed here is removed before an item is replayed; every other field on
#: every item goes back as it came. This is the whole of what the serializer
#: knows about the items it carries, and it stays this short on purpose.
OUTPUT_ONLY_FIELDS: Dict[str, tuple] = {
    "reasoning": ("status",),
    "compaction": ("created_by",),
}


def replayable_output(response: Any) -> List[Dict[str, Any]]:
    """`response.output` as the items the next request replays, exactly.

    The SDK's own serialization rather than a field list of ours. `model_dump`
    keeps what the provider sent - including fields this SDK version has no
    name for on the item types it does model - and the only edits made here
    are the ones `OUTPUT_ONLY_FIELDS` names. Everything else goes back as it
    came, nulls included: a value the provider set to nothing is a value it
    set. Order is kept because order is meaning, and an item type nobody has
    seen passes through whole for the same reason.

    Refuses rather than degrades. A reply with no `model_dump` is one the SDK
    did not model, and an output that is not a list of mappings is one nothing
    could replay. Either would put a broken turn in front of the model next
    time, so neither becomes a candidate.
    """
    dump = getattr(response, "model_dump", None)
    if not callable(dump):
        raise ValueError("a Responses reply without model_dump cannot be replayed")
    output = dump(mode="json").get("output")
    if not isinstance(output, list):
        raise ValueError("a Responses reply without an output list cannot be replayed")
    items: List[Dict[str, Any]] = []
    for item in output:
        if not isinstance(item, dict):
            raise ValueError("a Responses output item that is not a mapping cannot be replayed")
        kept = dict(item)
        for name in OUTPUT_ONLY_FIELDS.get(str(kept.get("type") or ""), ()):
            kept.pop(name, None)
        items.append(kept)
    return items


def usage_dict(response: Any) -> Dict[str, int]:
    """Responses usage, mapped to the internal shape plus the richer fields.

    reasoning_tokens and cached_tokens ride along as extra int keys - the
    agent loop sums every int in usage, so they aggregate across rounds and
    surface in the turn's usage without any consumer changing.
    """
    u = getattr(response, "usage", None)
    prompt = int(getattr(u, "input_tokens", 0) or 0)
    completion = int(getattr(u, "output_tokens", 0) or 0)
    usage = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": int(getattr(u, "total_tokens", 0) or (prompt + completion)),
    }
    reasoning = getattr(getattr(u, "output_tokens_details", None), "reasoning_tokens", 0)
    if reasoning:
        usage["reasoning_tokens"] = int(reasoning)
    cached = getattr(getattr(u, "input_tokens_details", None), "cached_tokens", 0)
    if cached:
        usage["cached_tokens"] = int(cached)
    return usage


def output_text(response: Any) -> str:
    """The response's text: the SDK convenience field, else walk the items.

    Refusal parts count as text: a reply that is entirely a refusal used to
    flatten to "" here, and the turn then fabricated "No response generated."
    over the model's actual words. The refusal IS the answer.
    """
    parts = []
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) == "message":
            for part in getattr(item, "content", None) or []:
                kind = getattr(part, "type", None)
                if kind == "output_text":
                    parts.append(getattr(part, "text", "") or "")
                elif kind == "refusal":
                    parts.append(getattr(part, "refusal", "") or "")
    if parts:
        return "".join(parts)
    return getattr(response, "output_text", None) or ""


def tool_calls_of(response: Any) -> List[Dict[str, str]]:
    """function_call output items in the internal {id, name, arguments} shape."""
    calls = []
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) == "function_call":
            calls.append({
                "id": getattr(item, "call_id", "") or "",
                "name": getattr(item, "name", "") or "",
                "arguments": getattr(item, "arguments", "") or "{}",
            })
    return calls


def assistant_message(content: str, calls: List[Dict[str, str]]) -> Dict[str, Any]:
    """A chat-shaped assistant message for the loop to append, so the next
    round's history converts back through to_input_items. Reasoning items
    are not on it: the transcript is provider-agnostic on purpose. A backend
    declared OpenAI-native carries them in its own tape instead (see
    `replayable_output`); any other one has the model reason again per
    round."""
    msg: Dict[str, Any] = {"role": "assistant", "content": content or None}
    if calls:
        msg["tool_calls"] = [
            {
                "id": c["id"],
                "type": "function",
                "function": {"name": c["name"], "arguments": c["arguments"]},
            }
            for c in calls
        ]
    return msg


def reasoning_param(effort: Optional[str]) -> Optional[dict]:
    """The configured effort as the Responses reasoning parameter.

    Chat sent it as extra_body reasoning_effort. "none" (disable thinking,
    honored by some compat providers) has no Responses equivalent - omit the
    parameter and let the model default."""
    if not effort or effort == "none":
        return None
    return {"effort": effort}
