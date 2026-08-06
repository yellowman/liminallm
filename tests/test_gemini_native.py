"""The native Gemini backend, and structured history resume on every provider.

Driven through httpx.MockTransport with real SSE bytes, so the wire format —
x-goog-api-key, :generateContent / :streamGenerateContent?alt=sse, contents /
systemInstruction / functionDeclarations — is what's asserted, not internals.
"""

from __future__ import annotations

import json

import httpx
import pytest

from liminallm.service import gemini_backend as gb
from liminallm.service import responses_compat as rc
from liminallm.service.gemini_backend import GeminiBackend

# The canonical structured history: system prompt, a completed turn, an agent
# round (assistant tool_calls + tool result), and the resuming user message.
RESUME_HISTORY = [
    {"role": "system", "content": "You are terse."},
    {"role": "user", "content": "Fix median()"},
    {"role": "assistant", "content": "Looking at it."},
    {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_1", "type": "function",
         "function": {"name": "run_tests", "arguments": '{"fn": "median"}'}}]},
    {"role": "tool", "tool_call_id": "call_1", "name": "run_tests",
     "content": "2 failed: even-length input"},
    {"role": "user", "content": "So what is the bug?"},
]


# ---------------------------------------------------------------------------
# Chat-shape -> native contents
# ---------------------------------------------------------------------------


def test_the_structured_history_resumes_as_native_contents():
    system, contents = gb.to_contents(RESUME_HISTORY)

    assert system == {"parts": [{"text": "You are terse."}]}
    assert contents == [
        {"role": "user", "parts": [{"text": "Fix median()"}]},
        {"role": "model", "parts": [
            {"text": "Looking at it."},
            {"functionCall": {"name": "run_tests", "args": {"fn": "median"}},
             "thoughtSignature": gb.THOUGHT_SIGNATURE_PLACEHOLDER},
        ]},
        {"role": "user", "parts": [
            {"functionResponse": {"name": "run_tests",
                                  "response": {"output": "2 failed: even-length input"}}},
            {"text": "So what is the bug?"},
        ]},
    ]


def test_consecutive_same_role_messages_merge():
    """Gemini expects alternating roles; adapter guidance plus a user message
    (or a tool result plus the next question) otherwise produce runs the API
    rejects."""
    _, contents = gb.to_contents([
        {"role": "user", "content": "one"},
        {"role": "user", "content": "two"},
    ])
    assert len(contents) == 1
    assert [p["text"] for p in contents[0]["parts"]] == ["one", "two"]


def test_multiple_system_messages_hoist_into_one_instruction():
    system, contents = gb.to_contents([
        {"role": "system", "content": "rule one"},
        {"role": "user", "content": "hi"},
        {"role": "system", "content": "rule two"},
    ])
    assert system == {"parts": [{"text": "rule one\n\nrule two"}]}
    assert contents == [{"role": "user", "parts": [{"text": "hi"}]}]


def test_unparseable_tool_arguments_do_not_crash_the_conversion():
    _, contents = gb.to_contents([{
        "role": "assistant", "content": None,
        "tool_calls": [{"id": "c", "type": "function",
                        "function": {"name": "f", "arguments": "not json"}}],
    }])
    assert contents[0]["parts"][0]["functionCall"]["args"] == {"raw": "not json"}


def test_function_declarations_scrub_what_gemini_rejects():
    decls = gb.to_function_declarations([{
        "type": "function",
        "function": {"name": "f", "description": "d", "parameters": {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object", "additionalProperties": False,
            "properties": {"q": {"type": "string"}},
            "required": ["q"],
        }},
    }])
    assert decls == [{"name": "f", "description": "d", "parameters": {
        "type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"],
    }}]


def test_usage_maps_thoughts_and_cache_to_the_rich_keys():
    usage = gb.usage_dict({"usageMetadata": {
        "promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 22,
        "thoughtsTokenCount": 7, "cachedContentTokenCount": 3,
    }})
    assert usage == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 22,
                     "reasoning_tokens": 7, "cached_tokens": 3}
    lean = gb.usage_dict({"usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 2}})
    assert lean == {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}


# ---------------------------------------------------------------------------
# The backend over a mocked wire
# ---------------------------------------------------------------------------


def _payload(text="answer", calls=(), thoughts=0):
    parts = [{"text": text}] if text else []
    parts += [{"functionCall": {"name": c["name"], "args": c["args"]}} for c in calls]
    meta = {"promptTokenCount": 8, "candidatesTokenCount": 4, "totalTokenCount": 12}
    if thoughts:
        meta["thoughtsTokenCount"] = thoughts
    return {"candidates": [{"content": {"role": "model", "parts": parts}}],
            "usageMetadata": meta}


def _backend(handler) -> GeminiBackend:
    return GeminiBackend(
        "gemini-2.5-flash", api_key="g-key",
        transport=httpx.MockTransport(handler),
    )


def test_generate_speaks_the_native_wire():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["key"] = request.headers.get("x-goog-api-key")
        seen["body"] = json.loads(request.read())
        return httpx.Response(200, json=_payload("native answer", thoughts=6))

    out = _backend(handler).generate(RESUME_HISTORY, [])

    assert "models/gemini-2.5-flash:generateContent" in seen["url"]
    assert seen["key"] == "g-key"
    assert seen["body"]["systemInstruction"]["parts"][0]["text"] == "You are terse."
    assert seen["body"]["contents"][0]["role"] == "user"
    assert out["content"] == "native answer"
    assert out["usage"]["reasoning_tokens"] == 6


def test_prompt_adapters_inject_into_the_system_instruction():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.read())
        return httpx.Response(200, json=_payload())

    adapter = {"id": "tone", "schema": {"adapter_mode": "prompt",
                                        "prompt_instructions": "Answer in haiku."}}
    out = _backend(handler).generate([{"role": "user", "content": "hi"}], [adapter])

    assert "Answer in haiku." in seen["body"]["systemInstruction"]["parts"][0]["text"]
    assert out["adapters_applied"] == ["tone:prompt"]


def test_a_weight_bearing_adapter_is_dropped_not_faked():
    def handler(request):
        return httpx.Response(200, json=_payload())

    adapter = {"id": "lora", "schema": {"adapter_mode": "local"}, "fs_dir": "/x"}
    out = _backend(handler).generate([{"role": "user", "content": "hi"}], [adapter])
    assert out["adapters_applied"] == []


def test_tool_round_trip_and_structured_resume_on_the_wire():
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.read()))
        if len(bodies) == 1:
            return httpx.Response(200, json=_payload(
                "", calls=[{"name": "run_tests", "args": {"fn": "median"}}]))
        return httpx.Response(200, json=_payload("the bug is sort()"))

    backend = _backend(handler)
    tools = [{"type": "function", "function": {"name": "run_tests",
                                               "description": "", "parameters": {"type": "object"}}}]
    messages = [{"role": "user", "content": "Fix median()"}]

    r1 = backend.generate_with_tools(messages, tools, [])
    assert r1["tool_calls"][0]["name"] == "run_tests"
    assert json.loads(r1["tool_calls"][0]["arguments"]) == {"fn": "median"}
    assert bodies[0]["tools"][0]["functionDeclarations"][0]["name"] == "run_tests"

    messages.append(r1["assistant_message"])
    messages.append({"role": "tool", "tool_call_id": r1["tool_calls"][0]["id"],
                     "name": "run_tests", "content": "2 failed"})
    r2 = backend.generate_with_tools(messages, tools, [])

    assert r2["content"] == "the bug is sort()"
    resumed = bodies[1]["contents"]
    assert resumed[1]["parts"][0]["functionCall"]["name"] == "run_tests"
    assert resumed[2]["parts"][0]["functionResponse"]["response"]["output"] == "2 failed"


def test_sse_streaming_yields_tokens_then_usage():
    chunks = [
        _payload("Hel"), _payload("lo "), _payload("world"),
    ]
    chunks[-1]["usageMetadata"]["thoughtsTokenCount"] = 5
    sse = "".join(f"data: {json.dumps(c)}\r\n\r\n" for c in chunks)

    def handler(request):
        assert "streamGenerateContent" in str(request.url)
        assert "alt=sse" in str(request.url)
        return httpx.Response(200, content=sse.encode(),
                              headers={"Content-Type": "text/event-stream"})

    events = list(_backend(handler).generate_stream([{"role": "user", "content": "hi"}], []))

    tokens = [e["data"] for e in events if e["event"] == "token"]
    assert tokens == ["Hel", "lo ", "world"]
    done = events[-1]
    assert done["event"] == "message_done"
    assert done["data"]["content"] == "Hello world"
    assert done["data"]["usage"]["reasoning_tokens"] == 5


def test_a_stream_error_is_an_event_not_an_exception():
    def handler(request):
        return httpx.Response(500, text="upstream burst")

    events = list(_backend(handler).generate_stream([{"role": "user", "content": "hi"}], []))
    assert events[-1]["event"] == "error"


def test_the_context_window_comes_from_the_models_probe():
    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json={"name": "models/gemini-2.5-flash",
                                             "inputTokenLimit": 1048576})
        return httpx.Response(200, json=_payload())

    assert _backend(handler).context_window == 1048576


# ---------------------------------------------------------------------------
# Structured resume, all providers: one history, every wire
# ---------------------------------------------------------------------------


def test_the_same_history_resumes_on_every_provider_wire():
    """The chat shape is the lingua franca; each provider's converter must
    carry the whole structure — system prompt, the completed tool round, the
    resuming question — without loss."""
    # chat/completions: the internal shape IS the wire shape.
    assert RESUME_HISTORY[4]["role"] == "tool"

    # OpenAI Responses: input items.
    items = rc.to_input_items(RESUME_HISTORY)
    kinds = [i.get("type") or i.get("role") for i in items]
    assert kinds == ["system", "user", "assistant", "function_call",
                     "function_call_output", "user"]
    call = next(i for i in items if i.get("type") == "function_call")
    assert call["name"] == "run_tests" and call["call_id"] == "call_1"
    out = next(i for i in items if i.get("type") == "function_call_output")
    assert out["call_id"] == "call_1"

    # Gemini native: contents.
    system, contents = gb.to_contents(RESUME_HISTORY)
    assert system is not None
    flat = [(c["role"], list(p.keys())[0]) for c in contents for p in c["parts"]]
    assert flat == [("user", "text"), ("model", "text"), ("model", "functionCall"),
                    ("user", "functionResponse"), ("user", "text")]


def test_every_provider_reports_the_rich_usage_keys():
    """reasoning/cached tokens must not be a Responses-only privilege."""
    from types import SimpleNamespace as NS

    responses_usage = rc.usage_dict(NS(usage=NS(
        input_tokens=1, output_tokens=2, total_tokens=3,
        output_tokens_details=NS(reasoning_tokens=4),
        input_tokens_details=NS(cached_tokens=5))))
    gemini_usage = gb.usage_dict({"usageMetadata": {
        "promptTokenCount": 1, "candidatesTokenCount": 2, "totalTokenCount": 3,
        "thoughtsTokenCount": 4, "cachedContentTokenCount": 5}})
    assert responses_usage == gemini_usage


# ---------------------------------------------------------------------------
# Thought signatures — found live: Gemini 400s a resumed functionCall that
# lacks one ("Function call is missing a thought_signature")
# ---------------------------------------------------------------------------


def test_the_thought_signature_survives_the_chat_shaped_round_trip():
    payload = _payload("", calls=[{"name": "run_tests", "args": {"fn": "median"}}])
    payload["candidates"][0]["content"]["parts"][0]["thoughtSignature"] = "sig-abc"

    calls = gb.function_calls_of(payload)
    assert calls[0]["thought_signature"] == "sig-abc"

    am = gb._assistant_message("", calls)
    assert am["tool_calls"][0]["thought_signature"] == "sig-abc"

    _, contents = gb.to_contents([am])
    assert contents[0]["parts"][0]["thoughtSignature"] == "sig-abc"


def test_a_foreign_history_gets_the_documented_placeholder():
    """A history that came from another provider carries no signature; the
    placeholder Google documents keeps cross-provider resume working
    (verified live against gemini-flash-latest)."""
    _, contents = gb.to_contents([{
        "role": "assistant", "content": None,
        "tool_calls": [{"id": "call_x", "type": "function",
                        "function": {"name": "f", "arguments": "{}"}}],
    }])
    assert contents[0]["parts"][0]["thoughtSignature"] == gb.THOUGHT_SIGNATURE_PLACEHOLDER
