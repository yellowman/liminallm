"""The final streamed answer runs on the state the rounds accepted.

Under a native continuation the provider holds the conversation through the
last turn the parent accepted. The answer the user reads must be produced from
that state, not from a rebuild of the public transcript that drops the
reasoning the state exists to keep. Two shapes: the worker's own last model
call already answered without tools, and that accepted answer is delivered as
it stands; or the rounds ended on a tool result, and the stream is sent the
accepted state and the record past it. Either way the stream produces no
continuation - nothing follows the final answer - and a stream that fails
leaves the accepted state where the rounds put it.

Driven through the real streamed agent path: a real worker, real tool
rounds against a real knowledge context, the model scripted.
"""

from __future__ import annotations

import asyncio
import json
import re

import pytest

from liminallm.service import responses_compat as rc
from liminallm.service.continuation import OPENAI_RESPONSES_NATIVE_V1
from liminallm.service.runtime import get_runtime
from liminallm.service.transcript import ModelTurn, ToolRound
from tests.mcpfixture import allow_local
from tests.test_agent_grounding_is_additive import QUESTION, grounded_context, web_on
from tests.test_continuation_lifecycle import MARK, _declare

ANSWER = "The Kestrel-9 relay module operates at 4417 kilohertz."


class NativeModel:
    """A native adapter's replies, scripted per call: each step either
    answers or asks for the context's search tool, and every reply carries
    the tape the real adapter would - the accepted items, the input sent,
    this turn's output."""

    def __init__(self, script):
        self.script = list(script)
        self.calls: list = []

    def __call__(self, messages, tools, adapters=None, *, user_id=None, continuation=None,
                 context_window=None):
        self.calls.append({"messages": [dict(m) for m in messages],
                           "continuation": continuation})
        n = len(self.calls)
        step = self.script.pop(0) if self.script else "answer"
        accepted = []
        if continuation is not None and continuation.strategy == OPENAI_RESPONSES_NATIVE_V1:
            accepted = [dict(i) for i in continuation.payload["items"]]
        output = [{"type": "reasoning", "id": f"rs_{n}", "encrypted_content": MARK}]
        if step == "search":
            arguments = json.dumps({"query": "Kestrel-9 resonance frequency"})
            calls = [{"id": f"c{n}", "name": "file_search", "arguments": arguments}]
            content = ""
            output.append({"type": "function_call", "id": f"fc_{n}", "call_id": f"c{n}",
                           "name": "file_search", "arguments": arguments})
        else:
            calls = []
            # Answer with the first citation marker the prompt offered, as a
            # model told to copy markers would.
            prompt = "\n".join(str(m.get("content") or "") for m in messages)
            marker = re.search(r"\[cite:[^\]]+\]", prompt)
            content = f"{ANSWER} {marker.group(0)}" if marker else ANSWER
            output.append({"type": "message", "id": f"msg_{n}", "role": "assistant",
                           "content": [{"type": "output_text", "text": content}]})
        return {
            "content": content,
            "tool_calls": calls,
            "assistant_message": rc.assistant_message(content, calls),
            "usage": {"total_tokens": 1},
            "continuation": {
                "strategy": OPENAI_RESPONSES_NATIVE_V1,
                "provider": "openai",
                "transport": "responses",
                "model": "gpt-6-astra",
                "payload": {"items": accepted + rc.to_input_items(messages) + output},
            },
        }


class StreamRecorder:
    """`stream_messages`, recording what it is handed and scripted to reply."""

    def __init__(self, events=None):
        self.calls: list = []
        self.events = events

    def __call__(self, messages, adapters=None, user_id=None, continuation=None):
        self.calls.append({"messages": [dict(m) for m in messages],
                           "continuation": continuation})
        return iter(self.events or [
            {"event": "token", "data": "streamed"},
            {"event": "message_done", "data": {"content": "streamed", "usage": {}}},
        ])


@pytest.fixture
def engine(monkeypatch):
    engine = get_runtime().workflow
    monkeypatch.setattr(engine, "tool_network_policy", allow_local())
    monkeypatch.setattr(
        type(engine.llm.backend), "supports_tools", property(lambda _self: True)
    )
    _declare(engine, monkeypatch, "openai")
    return engine


def _wire(engine, monkeypatch, script, stream_events=None):
    model = NativeModel(script)
    stream = StreamRecorder(stream_events)
    monkeypatch.setattr(engine.llm, "generate_with_tools", model)
    monkeypatch.setattr(engine.llm, "stream_messages", stream, raising=False)
    seen = {}
    real = engine._serve_invocation

    def serving(invocation, worker_tool, plan, context, limits, **kw):
        seen["context"] = context
        return real(invocation, worker_tool, plan, context, limits, **kw)

    monkeypatch.setattr(engine, "_serve_invocation", serving)
    return model, stream, seen


def _run(engine, store, monkeypatch, script, stream_events=None):
    user_id, ctx_id = grounded_context(store)
    web_on(monkeypatch, engine)
    model, stream, seen = _wire(engine, monkeypatch, script, stream_events)

    async def collect():
        return [e async for e in engine.run_streaming(None, None, QUESTION, ctx_id, user_id)]

    events = asyncio.run(collect())
    return events, model, stream, seen["context"]


class TestTheFinalAnswerRunsOnTheAcceptedState:
    def test_an_accepted_answer_is_delivered_and_not_regenerated(
        self, engine, store, monkeypatch
    ):
        """The worker's last model call answered without tools and the parent
        accepted it into the provider's state. It is the answer: no second
        generation, the accepted text goes to the client as it stands."""
        events, model, stream, context = _run(engine, store, monkeypatch, ["answer"])

        tokens = [e["data"] for e in events if e.get("event") == "token"]
        done = [e for e in events if e.get("event") == "message_done"]
        assert len(tokens) == 1 and tokens[0].startswith(ANSWER)
        assert done and done[-1]["data"]["content"] == tokens[0]
        assert stream.calls == [], "the answer was generated a second time"
        # What the client sees carries no handle of this turn's namespace,
        # and what the model cited is read out of the canonical copy of the
        # delivered turn, the way the blocking path reads it.
        assert "[cite:" not in tokens[0]
        cited = "[cite:" in json.dumps(model.calls[0]["messages"])
        assert bool(done[-1]["data"].get("validated_citations")) == cited
        assert len(model.calls) == 1
        assert context.continuation.strategy == OPENAI_RESPONSES_NATIVE_V1
        assert context.continuation.through_operation_seq == 1
        assert isinstance(context.transcript.entries[-1], ModelTurn)

    def test_the_stream_consumes_the_accepted_state_and_only_the_tail(
        self, engine, store, monkeypatch
    ):
        """The rounds ended on a tool result. The stream is handed the state
        the last turn was accepted at and the record past it, and produces
        nothing to accept: the state after the answer is the state before."""
        events, model, stream, context = _run(
            engine, store, monkeypatch, ["search", "search"]
        )

        assert [type(e).__name__ for e in context.transcript.entries] == [
            "ModelTurn", "ToolRound", "ModelTurn", "ToolRound",
        ]
        accepted = context.continuation
        assert accepted.through_operation_seq == 3
        assert len(stream.calls) == 1, stream.calls
        handed = stream.calls[0]
        assert handed["continuation"] == accepted
        assert [m["role"] for m in handed["messages"]] == ["tool"]
        assert handed["messages"][0]["name"] == "file_search"
        done = [e for e in events if e.get("event") == "message_done"]
        assert done and done[-1]["data"]["content"] == "streamed"
        # Consumed, not advanced.
        assert context.continuation == accepted
        assert isinstance(context.transcript.entries[-1], ToolRound)
        assert MARK not in json.dumps(events)

    def test_a_failed_stream_leaves_the_accepted_state_where_it_was(
        self, engine, store, monkeypatch
    ):
        events, _model, stream, context = _run(
            engine, store, monkeypatch, ["search", "search"],
            stream_events=[{"event": "error", "data": {"code": "server_error",
                                                       "message": "boom"}}],
        )

        assert stream.calls and stream.calls[0]["continuation"] is not None
        assert context.continuation.through_operation_seq == 3
        assert len(context.transcript.entries) == 4
        assert any(e.get("event") == "error" for e in events)
