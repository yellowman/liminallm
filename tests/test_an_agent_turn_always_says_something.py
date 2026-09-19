"""The agent's streamed turn never completes with nothing to show.

The attachment agent runs its tool rounds in the worker and streams the
final answer from the parent. The worker's own return has a fallback for an
assembly that produced no prose - `content or "I could not derive an answer
from the available sources."` - but `stream_final` overwrites it with the
raw content, because on that path the parent writes the answer.

The parent then had no fallback of its own. When its final turn produced no
tokens, `completed["content"]` was the empty string, and the workflow layer
substituted the placeholder it keeps for a turn that returned nothing. What
reached the reader was "No response generated." after a visible pause.

Measured against a live provider: asking "Why did the 2023 calls to pause
frontier AI training fail to stop any lab?" produced two `web_fetch` rounds
and then that placeholder, on a machine where fetching reaches nothing. The
same question answered normally on a later run, so it is the empty tool
result the loop cannot recover from, not the question.
"""

from __future__ import annotations

import uuid

import pytest

from liminallm.service.provenance import SourceRegistry

NODE = {"id": "files", "type": "tool_call", "tool": "agent.files_v1"}
QUESTION = "Why did the 2023 calls to pause frontier AI training fail?"

#: What the worker hands back when its rounds ended on a tool result rather
#: than on prose: a conversation to finish, and no answer in it.
ROUNDS_WITHOUT_PROSE = {
    "content": "",
    "usage": {},
    "context_snippets": [],
    "tool_calls": [{"tool": "web.fetch_v1", "arguments": {}}],
    "artifacts": [],
    "injection_findings": [],
    "messages": [{"role": "user", "content": QUESTION}],
}


@pytest.fixture
def engine(store):
    from liminallm.service.runtime import get_runtime

    return get_runtime().workflow


async def _run(engine, monkeypatch, *, final_stream):
    """Drive the streamed agent node and collect what it emitted."""
    from liminallm.service.invocation import InvocationRegistry

    monkeypatch.setattr(type(engine.llm), "supports_tools", True, raising=False)
    monkeypatch.setattr(
        engine,
        "_serve_invocation",
        lambda *a, **k: dict(ROUNDS_WITHOUT_PROSE),
    )
    engine.llm.stream_messages = lambda *a, **k: iter(final_stream)

    invocation = InvocationRegistry().open(
        uuid.uuid4().hex, tool="agent.files_v1", user_id="u", tenant_id=None
    )
    events = []
    async for event in engine._stream_agent_files_node(
        NODE,
        user_message=QUESTION,
        context_id=None,
        conversation_id=None,
        adapters=[],
        history=[],
        vars_scope={},
        source_registry=SourceRegistry(),
        bindings_sink=[],
        user_id="u",
        tenant_id=None,
        invocation=invocation,
    ):
        events.append(event)
    return events


def _done(events):
    for event in events:
        if event.get("event") == "message_done":
            return event.get("data") or {}
    raise AssertionError(f"the turn never completed: {events}")


class TestAFinalTurnThatProducedNothing:
    @pytest.mark.asyncio
    async def test_the_turn_still_carries_an_answer(self, engine, monkeypatch):
        """The defect, at the seam it happens at.

        Empty content here is what the workflow layer turns into "No
        response generated." - a string the reader sees as a malfunction.
        """
        events = await _run(
            engine,
            monkeypatch,
            final_stream=[{"event": "message_done", "data": {"content": "", "usage": {}}}],
        )

        assert _done(events).get("content"), (
            "the agent completed its turn with no content, which reaches the "
            "reader as the server's own placeholder"
        )

    @pytest.mark.asyncio
    async def test_a_stream_of_no_tokens_at_all_still_carries_an_answer(
        self, engine, monkeypatch
    ):
        """The provider may simply end the stream. No `message_done`, no
        tokens - the loop falls out of the pump with empty parts."""
        events = await _run(engine, monkeypatch, final_stream=[])

        assert _done(events).get("content"), (
            "an empty stream completed the turn with nothing to show"
        )

    @pytest.mark.asyncio
    async def test_the_reader_is_told_rather_than_shown_a_blank(
        self, engine, monkeypatch
    ):
        """It has to reach the client as text, not only sit in the result.

        Nothing was streamed, so a client that renders the token stream and
        keeps `message_done` for bookkeeping would show an empty bubble.
        """
        events = await _run(
            engine,
            monkeypatch,
            final_stream=[{"event": "message_done", "data": {"content": "", "usage": {}}}],
        )

        streamed = "".join(
            str(event.get("data") or "")
            for event in events
            if event.get("event") == "token"
        )
        assert streamed, "nothing reached the client as a token"
        assert streamed == _done(events).get("content"), (
            "what was streamed and what the turn recorded disagree"
        )


class TestAnAnswerIsNeverOverwritten:
    """The control. A fix that always substituted its own sentence would
    pass the class above and replace every real answer."""

    @pytest.mark.asyncio
    async def test_a_streamed_answer_survives(self, engine, monkeypatch):
        events = await _run(
            engine,
            monkeypatch,
            final_stream=[
                {"event": "token", "data": "Because a one-sided slowdown "},
                {"event": "token", "data": "moves capability rather than removing it."},
                {"event": "message_done", "data": {"usage": {}}},
            ],
        )

        content = _done(events).get("content")
        assert content == (
            "Because a one-sided slowdown moves capability rather than removing it."
        ), content

    @pytest.mark.asyncio
    async def test_an_answer_delivered_only_in_message_done_survives(
        self, engine, monkeypatch
    ):
        """Some backends report the whole answer on completion rather than
        as tokens; the node prefers that over the parts it accumulated."""
        events = await _run(
            engine,
            monkeypatch,
            final_stream=[
                {"event": "message_done", "data": {"content": "Verification.", "usage": {}}},
            ],
        )

        assert _done(events).get("content") == "Verification."
