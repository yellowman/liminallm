"""The Responses API is the primary endpoint; chat/completions is the fallback.

Everything internal stays chat-shaped — the agent loop, adapters, history
assembly — so responses_compat translates at the wire, and APIBackend probes
once: a provider answering 404/405 for /responses is chat-only for the rest
of the process.
"""

from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from liminallm.service import responses_compat as rc
from liminallm.service.model_backend import ApiAdapterBackend as APIBackend


# ---------------------------------------------------------------------------
# Fakes: the provider ends of both endpoints
# ---------------------------------------------------------------------------


def _response(text="hello", calls=(), reasoning=0, cached=0):
    output = [NS(type="function_call", call_id=c["id"], name=c["name"],
                 arguments=c["arguments"]) for c in calls]
    output.append(NS(type="message", role="assistant",
                     content=[NS(type="output_text", text=text)]))
    return NS(
        output_text=text,
        output=output,
        usage=NS(
            input_tokens=10, output_tokens=5, total_tokens=15,
            output_tokens_details=NS(reasoning_tokens=reasoning),
            input_tokens_details=NS(cached_tokens=cached),
        ),
    )


class _Unsupported(Exception):
    def __init__(self, status=404, message="Not Found"):
        super().__init__(message)
        self.status_code = status


def _backend(client) -> APIBackend:
    b = APIBackend.__new__(APIBackend)
    b.base_model = "gpt-4o-mini"
    b.adapter_server_model = None
    b.adapter_mode = "api_adapters"
    b._api_key = "k"
    b._base_url = None
    b._reasoning_effort = None
    b._api_key_env = "OPENAI_API_KEY"
    b._client_timeout = 30.0
    b._active_api_key = "k"
    b._responses_ok = None
    b.client = client
    b.provider = "openai"
    from liminallm.config import get_provider_capabilities

    b.capabilities = get_provider_capabilities("openai")
    b._context_window = 128000
    return b


def _client(responses_create=None, chat_create=None):
    client = NS()
    if responses_create is not None:
        client.responses = NS(create=responses_create)
    client.chat = NS(completions=NS(create=chat_create or _no_chat))
    return client


def _no_chat(*a, **kw):
    raise AssertionError("chat/completions was called; responses should have won")


def _chat_completion(text="from chat"):
    return NS(
        choices=[NS(message=NS(content=text, tool_calls=None),
                    model_dump=lambda **kw: {"role": "assistant", "content": text})],
        usage=NS(prompt_tokens=3, completion_tokens=2, total_tokens=5),
    )


# ---------------------------------------------------------------------------
# Endpoint selection
# ---------------------------------------------------------------------------


def test_responses_wins_when_the_provider_has_it():
    seen = {}

    def create(**kwargs):
        seen.update(kwargs)
        return _response("via responses", reasoning=7, cached=3)

    backend = _backend(_client(responses_create=create))
    out = backend.generate([{"role": "user", "content": "hi"}], [])

    assert out["content"] == "via responses"
    assert seen["input"] == [{"role": "user", "content": "hi"}]
    assert seen["model"] == "gpt-4o-mini"


def test_the_richer_usage_reaches_the_caller():
    backend = _backend(_client(lambda **kw: _response(reasoning=7, cached=3)))
    usage = backend.generate([{"role": "user", "content": "hi"}], [])["usage"]

    assert usage == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
                     "reasoning_tokens": 7, "cached_tokens": 3}


def test_a_chat_only_provider_falls_back_and_the_verdict_sticks():
    attempts = {"responses": 0, "chat": 0}

    def responses_create(**kw):
        attempts["responses"] += 1
        raise _Unsupported(404)

    def chat_create(**kw):
        attempts["chat"] += 1
        return _chat_completion()

    backend = _backend(_client(responses_create, chat_create))
    first = backend.generate([{"role": "user", "content": "hi"}], [])
    second = backend.generate([{"role": "user", "content": "again"}], [])

    assert first["content"] == second["content"] == "from chat"
    assert attempts["responses"] == 1, "the probe must not repeat per call"
    assert attempts["chat"] == 2
    assert backend._responses_ok is False


def test_an_sdk_without_the_surface_is_chat_only():
    backend = _backend(_client(responses_create=None, chat_create=lambda **kw: _chat_completion()))
    out = backend.generate([{"role": "user", "content": "hi"}], [])
    assert out["content"] == "from chat"
    assert backend._responses_ok is False


def test_a_real_error_after_adoption_raises_instead_of_flapping():
    """Once responses has succeeded, a later 404 (deploy skew, LB flake) is an
    error to surface, not a signal to silently downgrade endpoints."""
    state = {"calls": 0}

    def create(**kw):
        state["calls"] += 1
        if state["calls"] == 1:
            return _response()
        raise _Unsupported(404)

    backend = _backend(_client(create))
    backend.generate([{"role": "user", "content": "hi"}], [])
    with pytest.raises(_Unsupported):
        backend.generate([{"role": "user", "content": "hi"}], [])


def test_a_genuine_400_is_raised_not_misread_as_unsupported():
    def create(**kw):
        raise _Unsupported(400, "invalid value for temperature")

    backend = _backend(_client(create))
    with pytest.raises(_Unsupported):
        backend.generate([{"role": "user", "content": "hi"}], [])
    assert backend._responses_ok is None, "a bad request is not an endpoint verdict"


# ---------------------------------------------------------------------------
# Tool calling through /responses
# ---------------------------------------------------------------------------


def test_tool_calls_round_trip_the_internal_shape():
    seen = {}

    def create(**kw):
        seen.update(kw)
        return _response("thinking...", calls=[
            {"id": "call_1", "name": "web_search", "arguments": '{"q": "x"}'},
        ])

    backend = _backend(_client(create))
    chat_tools = [{"type": "function", "function": {
        "name": "web_search", "description": "search", "parameters": {"type": "object"}}}]
    out = backend.generate_with_tools(
        [{"role": "user", "content": "find x"}], chat_tools, [])

    assert seen["tools"] == [{"type": "function", "name": "web_search",
                              "description": "search", "parameters": {"type": "object"}}]
    assert out["tool_calls"] == [{"id": "call_1", "name": "web_search",
                                  "arguments": '{"q": "x"}'}]
    am = out["assistant_message"]
    assert am["tool_calls"][0]["function"]["name"] == "web_search"


def test_the_loops_chat_shaped_history_converts_to_input_items():
    """The agent loop appends a chat assistant message and role:"tool"
    results; the next round must arrive as function_call / output items."""
    history = [
        {"role": "user", "content": "find x"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "web_search", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "name": "web_search",
         "content": "result text"},
    ]
    items = rc.to_input_items(history)
    assert items == [
        {"role": "user", "content": "find x"},
        {"type": "function_call", "call_id": "call_1", "name": "web_search",
         "arguments": "{}"},
        {"type": "function_call_output", "call_id": "call_1", "output": "result text"},
    ]


def test_the_final_round_offers_no_tools_and_sends_none():
    seen = {}

    def create(**kw):
        seen.update(kw)
        return _response("final answer")

    backend = _backend(_client(create))
    backend.generate_with_tools([{"role": "user", "content": "x"}], [], [])
    assert "tools" not in seen and "tool_choice" not in seen


def test_multimodal_parts_map_to_responses_types():
    items = rc.to_input_items([{
        "role": "user",
        "content": [{"type": "text", "text": "look"},
                    {"type": "image_url", "image_url": {"url": "https://x/i.png"}}],
    }])
    assert items[0]["content"] == [
        {"type": "input_text", "text": "look"},
        {"type": "input_image", "image_url": "https://x/i.png"},
    ]


# ---------------------------------------------------------------------------
# Streaming through /responses
# ---------------------------------------------------------------------------


def _events(text="streamed reply", usage=True):
    for i in range(0, len(text), 5):
        yield NS(type="response.output_text.delta", delta=text[i:i + 5])
    if usage:
        yield NS(type="response.completed", response=_response(text, reasoning=4))


def test_streaming_yields_tokens_then_rich_usage():
    backend = _backend(_client(lambda **kw: _events()))
    events = list(backend.generate_stream([{"role": "user", "content": "hi"}], []))

    tokens = [e["data"] for e in events if e["event"] == "token"]
    assert "".join(tokens) == "streamed reply"
    done = events[-1]
    assert done["event"] == "message_done"
    assert done["data"]["content"] == "streamed reply"
    assert done["data"]["usage"]["reasoning_tokens"] == 4


def test_streaming_falls_back_to_chat_when_responses_404s():
    def responses_create(**kw):
        raise _Unsupported(404)

    def chat_create(**kw):
        return iter([
            NS(choices=[NS(delta=NS(content="chat "))], usage=None),
            NS(choices=[NS(delta=NS(content="stream"))],
               usage=NS(prompt_tokens=2, completion_tokens=2)),
        ])

    backend = _backend(_client(responses_create, chat_create))
    events = list(backend.generate_stream([{"role": "user", "content": "hi"}], []))

    tokens = [e["data"] for e in events if e["event"] == "token"]
    assert "".join(tokens) == "chat stream"
    assert backend._responses_ok is False


def test_a_mid_stream_failure_is_an_error_never_a_silent_retry():
    def create(**kw):
        def gen():
            yield NS(type="response.output_text.delta", delta="part")
            raise _Unsupported(500, "upstream reset")
        return gen()

    backend = _backend(_client(create))
    events = list(backend.generate_stream([{"role": "user", "content": "hi"}], []))
    assert events[-1]["event"] == "error"


# ---------------------------------------------------------------------------
# The classifier and the reasoning parameter
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("exc, verdict", [
    (_Unsupported(404), True),
    (_Unsupported(405), True),
    (AttributeError("no responses"), True),
    (TypeError("unexpected keyword"), True),
    (_Unsupported(400, "Unknown request URL: POST /v1/responses"), True),
    (_Unsupported(400, "temperature must be a number"), False),
    (_Unsupported(500, "boom"), False),
], ids=["404", "405", "attr", "type", "400-url", "400-real", "500"])
def test_is_unsupported_reads_the_right_signals(exc, verdict):
    assert rc.is_unsupported(exc) is verdict


def test_reasoning_effort_travels_as_the_first_class_parameter():
    seen = {}

    def create(**kw):
        seen.update(kw)
        return _response()

    backend = _backend(_client(create))
    backend._reasoning_effort = "high"
    backend.generate([{"role": "user", "content": "hi"}], [])
    assert seen["reasoning"] == {"effort": "high"}
    assert "extra_body" not in seen


def test_effort_none_is_omitted_rather_than_invented():
    assert rc.reasoning_param("none") is None
    assert rc.reasoning_param(None) is None
    assert rc.reasoning_param("low") == {"effort": "low"}


# ---------------------------------------------------------------------------
# The richer usage is kept, not thrown away
# ---------------------------------------------------------------------------


def test_usage_merging_keeps_the_rich_keys():
    """One summing implementation (_merge_usage) everywhere: the parallel-node
    merge used a fixed three-key list that silently discarded reasoning_tokens
    and cached_tokens. finish() stores the dict wholesale into message meta,
    so whatever survives the merges is persisted and surfaced."""
    from liminallm.service.workflow import WorkflowEngine

    eng = WorkflowEngine.__new__(WorkflowEngine)
    merged = eng._merge_usage(
        {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28,
         "reasoning_tokens": 7, "cached_tokens": 12},
    )
    assert merged == {"prompt_tokens": 30, "completion_tokens": 13,
                      "total_tokens": 43, "reasoning_tokens": 7,
                      "cached_tokens": 12}


def test_no_fixed_usage_key_list_survives_in_the_merge_paths():
    """Guard against the pattern coming back: the only place allowed to name
    the three classic keys as a summing allowlist is nowhere."""
    import pathlib

    src = pathlib.Path("liminallm/service/workflow.py").read_text()
    assert 'for key in ["prompt_tokens", "completion_tokens", "total_tokens"]' not in src
