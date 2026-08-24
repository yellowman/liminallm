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


def test_prompt_adapters_reach_the_system_instruction_exactly_once():
    """SPEC §5.0.1: LLMService materializes, the backend transports.

    This used to call the backend directly and assert it placed the text
    itself — which is the second-materializer contract. On the product path
    the service had already placed it, so the same instruction went out
    twice. Driven through the service now, and counted rather than merely
    found present, because "present" was true throughout the defect.
    """
    from liminallm.service.llm import LLMService

    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.read())
        return httpx.Response(200, json=_payload())

    adapter = {"id": "tone", "schema": {"mode": "prompt",
                                        "prompt_instructions": "Answer in haiku."}}
    backend = _backend(handler)
    out = LLMService(base_model="gemini-2.5-flash", backend=backend).generate(
        "hi", [adapter], []
    )

    system = seen["body"]["systemInstruction"]["parts"][0]["text"]
    assert "Answer in haiku." in system
    assert json.dumps(seen["body"]).count("Answer in haiku.") == 1
    assert out["adapters_applied"] == ["tone:prompt"]


def test_a_weight_bearing_adapter_is_dropped_not_faked():
    def handler(request):
        return httpx.Response(200, json=_payload())

    adapter = {"id": "lora", "schema": {"mode": "local"}, "fs_dir": "/x"}
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


# ---------------------------------------------------------------------------
# Admin key resolution — the gemini_api_key setting flows to both Gemini
# backends, with the generic provider key and GEMINI_API_KEY env as fallbacks
# ---------------------------------------------------------------------------


class TestAdminKeyResolution:
    def _service(self, monkeypatch, *, gemini_key=None, env_key=None):
        from liminallm.service.llm import LLMService

        if env_key is None:
            monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        else:
            monkeypatch.setenv("GEMINI_API_KEY", env_key)
        configs = {
            "openai": {"api_key": "sk-openai-field", "base_url": None},
            "gemini": {"api_key": gemini_key},
        }
        return LLMService(
            "gemini-flash-latest", backend_mode="gemini_native",
            adapter_configs=configs, api_key="sk-openai-field",
        )

    def test_the_admin_setting_wins(self, monkeypatch):
        svc = self._service(monkeypatch, gemini_key="g-admin", env_key="g-env")
        assert isinstance(svc.backend, GeminiBackend)
        assert svc.backend._api_key == "g-admin"

    def test_it_falls_back_to_the_provider_key_setting(self, monkeypatch):
        svc = self._service(monkeypatch, env_key="g-env")
        assert svc.backend._api_key == "sk-openai-field"

    def test_the_env_var_is_the_last_fallback(self, monkeypatch):
        from liminallm.service.llm import LLMService

        monkeypatch.setenv("GEMINI_API_KEY", "g-env")
        svc = LLMService(
            "gemini-flash-latest", backend_mode="gemini_native",
            adapter_configs={"gemini": {"api_key": None}},
        )
        assert svc.backend._api_key == "g-env"

    def test_the_compat_shim_reads_the_same_setting(self, monkeypatch):
        from liminallm.service.llm import LLMService

        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        svc = LLMService(
            "gemini-2.5-flash", backend_mode="gemini",
            adapter_configs={"gemini": {"api_key": "g-admin"},
                             "openai": {"api_key": "sk-openai-field"}},
        )
        assert svc.backend._api_key == "g-admin"
        assert "generativelanguage" in svc.backend._base_url

    def test_runtime_wires_the_setting_into_the_gemini_config(self):
        """The adapter_configs dict runtime builds must carry the setting —
        this is the seam _build_backend reads."""
        import inspect

        from liminallm.service import runtime

        src = inspect.getsource(runtime.Runtime._build_model_services)
        assert "gemini_api_key" in src


# ---------------------------------------------------------------------------
# Reasoning effort — the admin setting reached every backend but this one
# ---------------------------------------------------------------------------


class TestThinkingConfig:
    """thinkingLevel values verified live against gemini-flash-latest:
    low/medium/high are accepted verbatim, "none" is a 400, "minimal" is the
    accepted floor and spends zero thought tokens."""

    def test_the_setting_vocabulary_maps_onto_the_native_one(self):
        assert gb.thinking_config("low") == {"thinkingLevel": "low"}
        assert gb.thinking_config("medium") == {"thinkingLevel": "medium"}
        assert gb.thinking_config("HIGH ") == {"thinkingLevel": "high"}

    def test_none_becomes_minimal_because_none_is_a_400(self):
        """The setting advertises "none to disable"; the API rejects that
        literal, so it maps to the level that does disable thinking."""
        assert gb.thinking_config("none") == {"thinkingLevel": "minimal"}

    def test_unset_sends_no_thinking_config_at_all(self):
        assert gb.thinking_config("") is None
        assert gb.thinking_config(None) is None
        assert gb.thinking_config("bogus") is None

    def test_the_effort_reaches_the_wire(self):
        seen = {}

        def handler(request):
            seen.update(json.loads(request.content))
            return httpx.Response(200, json=_payload("hi"))

        GeminiBackend(
            "gemini-flash-latest", api_key="k", reasoning_effort="low",
            transport=httpx.MockTransport(handler),
        ).generate([{"role": "user", "content": "hi"}], [])
        assert seen["generationConfig"] == {"thinkingConfig": {"thinkingLevel": "low"}}

    def test_no_effort_means_no_generation_config(self):
        seen = {}

        def handler(request):
            seen.update(json.loads(request.content))
            return httpx.Response(200, json=_payload("hi"))

        _backend(handler).generate([{"role": "user", "content": "hi"}], [])
        assert "generationConfig" not in seen

    def test_a_model_that_rejects_thinking_degrades_once_and_stays_degraded(self):
        """An older model 400s on thinkingConfig. Failing every request
        because an admin set an effort is the wrong answer: drop the field,
        log it, and keep serving."""
        bodies = []

        def handler(request):
            body = json.loads(request.content)
            bodies.append(body)
            if "generationConfig" in body:
                return httpx.Response(400, json={"error": {
                    "message": "Unknown name \"thinking_level\" at 'generation_config.thinking_config'"}})
            return httpx.Response(200, json=_payload("served"))

        backend = GeminiBackend(
            "gemini-1.0-pro", api_key="k", reasoning_effort="high",
            transport=httpx.MockTransport(handler),
        )
        assert backend.generate([{"role": "user", "content": "hi"}], [])["content"] == "served"
        assert len(bodies) == 2, "should retry once without the rejected field"

        backend.generate([{"role": "user", "content": "again"}], [])
        assert len(bodies) == 3, "the second turn must not re-probe"
        assert all("generationConfig" not in b for b in bodies[2:])

    def test_a_real_400_is_not_mistaken_for_a_thinking_rejection(self):
        def handler(request):
            return httpx.Response(400, json={"error": {"message": "API key not valid"}})

        backend = GeminiBackend(
            "gemini-flash-latest", api_key="bad", reasoning_effort="high",
            transport=httpx.MockTransport(handler),
        )
        with pytest.raises(httpx.HTTPStatusError):
            backend.generate([{"role": "user", "content": "hi"}], [])
        assert backend._thinking_ok is None

    def test_the_service_hands_the_setting_to_the_backend(self):
        from liminallm.service.llm import LLMService

        svc = LLMService(
            "gemini-flash-latest", backend_mode="gemini_native",
            adapter_configs={"gemini": {"api_key": "k"}},
            reasoning_effort="medium",
        )
        assert svc.backend._reasoning_effort == "medium"


def test_the_streaming_usage_fallback_uses_the_shared_estimator():
    """A word count undercounts CJK roughly fourfold — that is why there is
    one estimator."""
    from liminallm.service.tokenizer_utils import estimate_token_count

    cjk = "日本語のテキストです" * 5

    def handler(request):
        body = f'data: {json.dumps({"candidates": [{"content": {"parts": [{"text": cjk}]}}]})}\n\n'
        return httpx.Response(200, content=body.encode(),
                              headers={"content-type": "text/event-stream"})

    events = list(_backend(handler).generate_stream([{"role": "user", "content": "hi"}], []))
    usage = events[-1]["data"]["usage"]
    assert usage["completion_tokens"] == estimate_token_count(cjk)
    assert usage["completion_tokens"] > len(cjk.split()) * 4


def test_the_stream_also_retries_once_without_the_rejected_thinking_config():
    """The two-attempt loop in generate_stream is the fiddliest part of the
    degrade: it has to read a 400 body from a response opened for streaming."""
    bodies = []
    chunk = json.dumps({"candidates": [{"content": {"parts": [{"text": "hello"}]}}],
                        "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 1}})

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        if "generationConfig" in body:
            return httpx.Response(400, json={"error": {
                "message": "Unknown name \"thinking_config\" at 'generation_config'"}})
        return httpx.Response(200, content=f"data: {chunk}\n\n".encode(),
                              headers={"content-type": "text/event-stream"})

    backend = GeminiBackend(
        "gemini-1.0-pro", api_key="k", reasoning_effort="high",
        transport=httpx.MockTransport(handler),
    )
    events = list(backend.generate_stream([{"role": "user", "content": "hi"}], []))
    assert [e["event"] for e in events] == ["token", "message_done"]
    assert events[0]["data"] == "hello"
    assert len(bodies) == 2
    assert events[-1]["data"]["usage"]["prompt_tokens"] == 3


class TestNativeTemperature:
    """The native backend must honour the same policy as the compat ones — a
    setting that works on one backend and is silently ignored on another is
    the bug this branch already fixed once, for reasoning effort."""

    def _generation_config(self, model, temperature):
        seen = {}

        def handler(request):
            seen.update(json.loads(request.content))
            return httpx.Response(200, json=_payload("hi"))

        GeminiBackend(model, api_key="k", temperature=temperature,
                      transport=httpx.MockTransport(handler)).generate(
            [{"role": "user", "content": "hi"}], [])
        return seen.get("generationConfig")

    def test_gemini_3_never_receives_one(self):
        """Google deprecated sampling parameters there and warns that lowering
        temperature can drive the model into loops."""
        assert self._generation_config("gemini-3.6-flash", 0.7) is None
        assert self._generation_config("gemini-flash-latest", 0.7) is None

    def test_a_2_5_model_receives_a_configured_one(self):
        assert self._generation_config("gemini-2.5-flash", 0.7) == {"temperature": 0.7}

    def test_nothing_is_sent_when_unconfigured(self):
        assert self._generation_config("gemini-2.5-flash", None) is None

    def test_it_rides_alongside_the_thinking_config(self):
        seen = {}

        def handler(request):
            seen.update(json.loads(request.content))
            return httpx.Response(200, json=_payload("hi"))

        GeminiBackend("gemini-2.5-flash", api_key="k", temperature=0.4,
                      reasoning_effort="low",
                      transport=httpx.MockTransport(handler)).generate(
            [{"role": "user", "content": "hi"}], [])
        assert seen["generationConfig"] == {
            "thinkingConfig": {"thinkingLevel": "low"}, "temperature": 0.4,
        }

    def test_the_service_hands_the_setting_to_the_native_backend(self):
        from liminallm.service.llm import LLMService

        svc = LLMService(
            "gemini-2.5-flash", backend_mode="gemini_native",
            adapter_configs={"gemini": {"api_key": "k"}}, temperature=0.4,
        )
        assert svc.backend._temperature == 0.4
