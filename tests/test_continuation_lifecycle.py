"""The parent owns the provider's continuation, and only acceptance moves it.

A backend hands back a candidate with every model turn: what it would want
replayed next time, in its own terms. The parent decides. A turn the ledger
commits advances the record to that candidate; a turn refused for its shape,
or answered by a backend that is not on the record's strategy, leaves the
record where the last accepted turn put it - and a replacement attempt
replaying the ledger is handed exactly that, without the provider being
asked again.

Driven through the broker's own entry point, as the worker drives it, with the
model scripted per call and every request it was handed kept for inspection.
"""

from __future__ import annotations

import json
import re
import uuid

from liminallm.service import responses_compat as rc
from liminallm.service.broker import (
    UNOFFERED_TOOL_RESULT,
    CapabilityBroker,
    InvocationContext,
)
from liminallm.service.citation_offers import CITATION_INSTRUCTION
from liminallm.service.continuation import (
    CHAT_STRUCTURED_V1,
    OPENAI_RESPONSES_NATIVE_V1,
    ProviderContinuation,
)
from liminallm.service.invocation import COMMITTED, FAILED, InvocationRegistry
from liminallm.service.provenance import SourceRegistry
from liminallm.service.runtime import get_runtime
from tests.test_trusted_transcript import SEARCH, SUBMITTED, TOOLS, _ask, _web

#: A provider's opaque value, with the characters a careless codec changes.
SENTINEL = "gAAAAB+/x9Q==étape"
#: Its ASCII head, for looking through serializations that escape the rest:
#: an absence check against the whole value would pass on escaping alone.
MARK = "gAAAAB+/x9Q=="

OPENING = [{"role": "system", "content": "the parent's own prompt"}]


def _turn(engine, monkeypatch, *, mode="openai"):
    """An agent turn on a backend declaring `mode`, as `_serve_invocation`
    would build it: the parent's base prompt remembered, a registry so offers
    are live, and a broker told which body it serves."""
    _web(engine, monkeypatch)
    monkeypatch.setattr(engine.llm.backend, "backend_mode", mode, raising=False)
    registry = SourceRegistry()
    invocation = InvocationRegistry().open(
        uuid.uuid4().hex, tool="agent.files_v1", user_id="u", tenant_id=None
    )
    context = InvocationContext(user_id="u", source_registry=registry)
    context.remember_base_prompt(OPENING, TOOLS)
    return registry, invocation, context, CapabilityBroker(
        engine, context, worker_tool="agent.files_v1"
    )


def _replacement(engine, registry):
    context = InvocationContext(user_id="u", source_registry=registry)
    return context, CapabilityBroker(engine, context, worker_tool="agent.files_v1")


def _reasoning(ident, encrypted=SENTINEL):
    return {"type": "reasoning", "id": ident, "encrypted_content": encrypted}


def _message(ident, text):
    return {"type": "message", "id": ident, "role": "assistant",
            "content": [{"type": "output_text", "text": text}]}


def _function_call(ident, call):
    return {"type": "function_call", "id": ident, "call_id": call["id"],
            "name": call["name"], "arguments": call["arguments"]}


class _Provider:
    """The service's `generate_with_tools`, scripted one reply per call.

    Each reply is built the way the OpenAI adapter builds one: the accepted
    tape it was handed, then the input it was sent, then this turn's output,
    all in the candidate. So what the parent accepts here has the shape the
    real adapter returns, and what a retry is handed can be read for what a
    rejected turn left in it.
    """

    def __init__(self, engine, monkeypatch, replies):
        self.calls: list = []
        self.replies = list(replies)
        monkeypatch.setattr(engine.llm, "generate_with_tools", self, raising=False)

    def __call__(self, messages, tools, adapters, *, user_id=None, continuation=None):
        self.calls.append({
            "messages": [dict(m) for m in messages],
            "tools": list(tools or []),
            "continuation": continuation,
        })
        reply = self.replies.pop(0)
        return reply(messages, continuation) if callable(reply) else reply


def native(content="", calls=(), *, tag="1", model="gpt-6-astra", transport="responses"):
    """A native reply: the tape grows by what went and what came back."""
    calls = [dict(c) for c in calls]

    def build(messages, continuation):
        accepted = []
        if continuation is not None and continuation.strategy == OPENAI_RESPONSES_NATIVE_V1:
            accepted = [dict(i) for i in continuation.payload.get("items") or []]
        output = [_reasoning(f"rs_{tag}")]
        output += [_function_call(f"fc_{tag}_{i}", c) for i, c in enumerate(calls)]
        if content:
            output.append(_message(f"msg_{tag}", content))
        return {
            "content": content,
            "tool_calls": calls,
            "assistant_message": rc.assistant_message(content, calls),
            "usage": {},
            "continuation": {
                "strategy": OPENAI_RESPONSES_NATIVE_V1,
                "provider": "openai",
                "transport": transport,
                "model": model,
                "payload": {"items": accepted + rc.to_input_items(messages) + output},
            },
        }

    return build


def chat_structured(content="", calls=(), *, transport="responses", model="grok-4.5"):
    calls = [dict(c) for c in calls]
    return {
        "content": content,
        "tool_calls": calls,
        "assistant_message": rc.assistant_message(content, calls),
        "usage": {},
        "continuation": {
            "strategy": CHAT_STRUCTURED_V1,
            "provider": "openai",
            "transport": transport,
            "model": model,
            "payload": {},
        },
    }


def _model(broker, invocation, seq, *, tools=TOOLS):
    return broker._answer(invocation, {
        "capability": "llm.generate_with_tools",
        "operation_seq": seq,
        "payload": {"messages": [{"role": "user", "content": "hours?"}], "tools": tools},
    })


def _round(broker, invocation, seq, calls=(SUBMITTED,)):
    return broker._answer(invocation, {
        "capability": "tools.round",
        "operation_seq": seq,
        "payload": {"calls": [dict(c) for c in calls], "fallback_query": "hours"},
    })


def _watch_rounds(engine, monkeypatch):
    ran: list = []
    real = engine._run_round_tools

    def watched(parsed, **kwargs):
        ran.append([name for _c, name, _a in parsed])
        return real(parsed, **kwargs)

    monkeypatch.setattr(engine, "_run_round_tools", watched)
    return ran


def _tool_messages(context):
    """The tool messages the record past the first turn rebuilds to."""
    return [
        {"role": "tool", "tool_call_id": r.tool_message_id, "name": r.tool_name,
         "content": r.text}
        for entry in context.transcript.rounds() for r in entry.results
    ]


def _unmarked(messages):
    """The messages with the citation markers the parent placed taken out,
    so a tail can be compared to the record it was rendered from."""
    return [
        {**m, "content": re.sub(r" \[cite:[^\]]+\]", "", str(m.get("content") or ""))}
        for m in messages
    ]


class TestTheOpeningGoesWholeAndOnlyTheTailFollows:
    def test_the_next_call_is_handed_the_accepted_state_and_the_record_since(
        self, store, monkeypatch
    ):
        engine = get_runtime().workflow
        _registry, invocation, context, broker = _turn(engine, monkeypatch)
        provider = _Provider(engine, monkeypatch, [
            native("looking", [SEARCH], tag="1"),
            native("400 hours", tag="2"),
        ])

        assert _model(broker, invocation, 1)["ok"]
        first = context.continuation
        assert first.strategy == OPENAI_RESPONSES_NATIVE_V1
        assert first.through_operation_seq == 1
        # The opening went whole: the parent's prompt, nothing accepted yet.
        assert provider.calls[0]["continuation"] is None
        assert provider.calls[0]["messages"][0]["role"] == "system"

        assert _round(broker, invocation, 2)["ok"]
        assert context.continuation == first, "a round moves nothing"

        assert _model(broker, invocation, 3)["ok"]
        # The second call: the accepted state, and past it only the round.
        assert provider.calls[1]["continuation"] == first
        assert _unmarked(provider.calls[1]["messages"]) == _tool_messages(context)
        assert context.continuation.through_operation_seq == 3
        # And the accepted tape grew by exactly that round and that turn.
        items = context.continuation.payload["items"]
        assert items[: len(first.payload["items"])] == first.payload["items"]
        assert [i.get("type") for i in items[len(first.payload["items"]):]] == [
            "function_call_output", "reasoning", "message",
        ]

    def test_the_native_opening_is_instructed_before_any_marker_exists(
        self, store, monkeypatch
    ):
        """The opening is frozen in the provider's hands. A marker a later
        round places can only be answered by an instruction already there."""
        engine = get_runtime().workflow
        _registry, invocation, context, broker = _turn(engine, monkeypatch)
        provider = _Provider(engine, monkeypatch, [
            native("looking", [SEARCH], tag="1"), native("done", tag="2"),
        ])

        assert _model(broker, invocation, 1)["ok"]
        assert _round(broker, invocation, 2)["ok"]
        assert _model(broker, invocation, 3)["ok"]

        opening = provider.calls[0]["messages"]
        assert CITATION_INSTRUCTION in opening[0]["content"]
        assert all(m["role"] != "system" for m in provider.calls[1]["messages"])

    def test_with_offers_off_the_tail_is_still_the_parents(self, store, monkeypatch):
        """The legacy prompt shape forwards the worker's list for the
        opening. What the provider is told since is the parent's record all
        the same: a list the worker composed is not what the provider holds."""
        engine = get_runtime().workflow
        monkeypatch.setattr(type(engine), "CITATION_OFFERS_ENABLED", False, raising=False)
        _registry, invocation, context, broker = _turn(engine, monkeypatch)
        provider = _Provider(engine, monkeypatch, [
            native("looking", [SEARCH], tag="1"), native("done", tag="2"),
        ])

        assert _model(broker, invocation, 1)["ok"]
        assert _round(broker, invocation, 2)["ok"]
        assert broker._answer(invocation, {
            "capability": "llm.generate_with_tools", "operation_seq": 3,
            "payload": {"messages": [{"role": "user", "content": "the worker's own"}],
                        "tools": TOOLS},
        })["ok"]

        assert provider.calls[0]["messages"] == [{"role": "user", "content": "hours?"}]
        assert provider.calls[1]["messages"] == _tool_messages(context)

    def test_a_compatible_provider_advances_the_cursor_and_is_sent_everything(
        self, store, monkeypatch
    ):
        """No opaque state, so the record is the whole conversation and the
        whole conversation is what goes, every round, as before."""
        engine = get_runtime().workflow
        _registry, invocation, context, broker = _turn(engine, monkeypatch, mode="xai")
        provider = _Provider(engine, monkeypatch, [
            chat_structured("looking", [SEARCH]), chat_structured("400 hours"),
        ])

        assert _model(broker, invocation, 1)["ok"]
        assert context.continuation.strategy == CHAT_STRUCTURED_V1
        assert context.continuation.through_operation_seq == 1
        assert _round(broker, invocation, 2)["ok"]
        assert _model(broker, invocation, 3)["ok"]

        again = provider.calls[1]
        assert again["continuation"].strategy == CHAT_STRUCTURED_V1
        assert again["messages"][0]["role"] == "system"
        assert again["messages"][-1]["role"] == "tool"
        assert context.continuation.through_operation_seq == 3
        # Sent whole every round, so the opening is instructed the way it
        # always was: only once a marker is placed in it.
        assert CITATION_INSTRUCTION not in provider.calls[0]["messages"][0]["content"]


class TestTheStateIsPerInvocation:
    def test_two_contexts_on_one_backend_see_nothing_of_each_other(
        self, store, monkeypatch
    ):
        engine = get_runtime().workflow
        registry, invocation_a, context_a, broker_a = _turn(engine, monkeypatch)
        context_b = InvocationContext(user_id="u", source_registry=registry)
        context_b.remember_base_prompt(OPENING, TOOLS)
        invocation_b = InvocationRegistry().open(
            uuid.uuid4().hex, tool="agent.files_v1", user_id="u", tenant_id=None
        )
        broker_b = CapabilityBroker(engine, context_b, worker_tool="agent.files_v1")
        provider = _Provider(engine, monkeypatch, [
            native("a", tag="a"), native("b", tag="b"),
        ])

        assert _model(broker_a, invocation_a, 1)["ok"]
        assert _model(broker_b, invocation_b, 1)["ok"]

        assert provider.calls[1]["continuation"] is None
        assert context_a.continuation != context_b.continuation
        assert "rs_a" not in json.dumps(context_b.continuation.as_dict())
        # Nothing of it on what the two share.
        for shared in (engine.llm, engine.llm.backend):
            assert MARK not in repr(vars(shared))
            assert not any("continuation" in name for name in vars(shared))

    def test_a_replacement_attempt_is_restored_the_accepted_state_without_a_call(
        self, store, monkeypatch
    ):
        engine = get_runtime().workflow
        registry, invocation, context, broker = _turn(engine, monkeypatch)
        provider = _Provider(engine, monkeypatch, [
            native("looking", [SEARCH], tag="1"), native("400 hours", tag="2"),
        ])
        assert _model(broker, invocation, 1)["ok"]
        assert _round(broker, invocation, 2)["ok"]
        assert _model(broker, invocation, 3)["ok"]
        accepted = context.continuation
        calls_before = len(provider.calls)

        second, replacement = _replacement(engine, registry)
        for seq, ask in ((1, _model), (2, _round), (3, _model)):
            reply = ask(replacement, invocation, seq)
            assert reply["ok"] and reply.get("replayed"), reply

        assert len(provider.calls) == calls_before
        assert second.continuation == accepted
        assert second.continuation.payload["items"][-2]["encrypted_content"] == SENTINEL


class TestARejectedTurnLeavesNothingBehind:
    def _two_calls_one_broken(self, tag="3"):
        broken = {"id": "c2", "name": "web_fetch", "arguments": '{"url": '}
        return native("both", [SEARCH, broken], tag=tag)

    def test_one_malformed_call_rejects_the_turn_runs_nothing_and_keeps_the_state(
        self, store, monkeypatch
    ):
        engine = get_runtime().workflow
        _registry, invocation, context, broker = _turn(engine, monkeypatch)
        _Provider(engine, monkeypatch, [
            native("looking", [SEARCH], tag="1"), self._two_calls_one_broken(),
        ])
        ran = _watch_rounds(engine, monkeypatch)
        assert _model(broker, invocation, 1)["ok"]
        assert _round(broker, invocation, 2)["ok"]
        accepted = context.continuation

        reply = _model(broker, invocation, 3)

        assert reply["ok"] is False and reply["code"] == "model_turn_rejected"
        assert invocation.ledger.get(3).state == FAILED
        assert context.continuation == accepted
        assert len(context.transcript.entries) == 2
        # The worker relaying the calls anyway finds no turn that asked.
        broken = {"id": "c2", "name": "web_fetch", "arguments": {}}
        again = _round(broker, invocation, 4, calls=[SUBMITTED, broken])
        assert again["ok"] is False and again["code"] == "round_not_asked"
        assert ran == [["web_search"]], "only the accepted turn's round ran"

    def test_arguments_that_parse_but_are_not_an_object_reject_the_turn(
        self, store, monkeypatch
    ):
        engine = get_runtime().workflow
        _registry, invocation, context, broker = _turn(engine, monkeypatch)
        listed = {"id": "c1", "name": "web_search", "arguments": "[1, 2]"}
        _Provider(engine, monkeypatch, [native("looking", [listed], tag="1")])

        reply = _model(broker, invocation, 1)

        assert reply["ok"] is False and reply["code"] == "model_turn_rejected"
        assert context.continuation is None
        assert context.transcript.entries == []

    def test_the_retry_is_handed_the_old_state_and_none_of_the_rejected_reply(
        self, store, monkeypatch
    ):
        engine = get_runtime().workflow
        registry, invocation, context, broker = _turn(engine, monkeypatch)
        provider = _Provider(engine, monkeypatch, [
            native("looking", [SEARCH], tag="1"),
            self._two_calls_one_broken(tag="bad"),
            native("400 hours", tag="4"),
        ])
        assert _model(broker, invocation, 1)["ok"]
        assert _round(broker, invocation, 2)["ok"]
        assert _model(broker, invocation, 3)["ok"] is False
        accepted = context.continuation

        second, replacement = _replacement(engine, registry)
        assert _model(replacement, invocation, 1).get("replayed")
        assert _round(replacement, invocation, 2).get("replayed")
        retry = _model(replacement, invocation, 3)

        assert retry["ok"] and not retry.get("replayed")
        handed = provider.calls[-1]
        assert handed["continuation"] == accepted
        assert "bad" not in json.dumps(handed["continuation"].as_dict())
        assert _unmarked(handed["messages"]) == _tool_messages(second)
        assert second.continuation.through_operation_seq == 3

    def test_an_accepted_turn_whose_call_is_refused_stays_accepted(
        self, store, monkeypatch
    ):
        """A valid turn that asked for a tool the parent never offered is
        history: the refusal is the round's result, and it is exactly what
        the provider is told next."""
        engine = get_runtime().workflow
        _registry, invocation, context, broker = _turn(engine, monkeypatch)
        provider = _Provider(engine, monkeypatch, [
            native("looking", [SEARCH], tag="1"), native("then no", tag="2"),
        ])

        assert _model(broker, invocation, 1, tools=[])["ok"]
        accepted = context.continuation
        assert _round(broker, invocation, 2)["ok"]
        assert context.continuation == accepted
        assert _model(broker, invocation, 3)["ok"]

        assert provider.calls[1]["continuation"] == accepted
        assert provider.calls[1]["messages"] == [{
            "role": "tool", "tool_call_id": "c1", "name": "web_search",
            "content": UNOFFERED_TOOL_RESULT,
        }]


class TestTheStrategyIsStickyForTheInvocation:
    def test_a_backend_declaring_otherwise_is_refused_before_it_is_asked(
        self, store, monkeypatch
    ):
        """A replacement attempt in a process configured for a compatible
        provider cannot continue a native record. Refused ahead of the call:
        the tail alone would continue nothing."""
        engine = get_runtime().workflow
        registry, invocation, _context, broker = _turn(engine, monkeypatch)
        provider = _Provider(engine, monkeypatch, [
            native("looking", [SEARCH], tag="1"), chat_structured("nope"),
        ])
        assert _model(broker, invocation, 1)["ok"]
        assert _round(broker, invocation, 2)["ok"]

        monkeypatch.setattr(engine.llm.backend, "backend_mode", "xai", raising=False)
        second, replacement = _replacement(engine, registry)
        assert _model(replacement, invocation, 1).get("replayed")
        assert _round(replacement, invocation, 2).get("replayed")
        reply = _model(replacement, invocation, 3)

        assert reply["ok"] is False and reply["code"] == "continuation_mismatch"
        assert len(provider.calls) == 1, "the provider was not asked"
        assert second.continuation.through_operation_seq == 1
        assert invocation.ledger.get(3).state == FAILED

    def test_a_fallback_to_chat_after_native_state_is_refused(
        self, store, monkeypatch
    ):
        """Responses, then a chat-shaped reply later: the invocation is held
        to the wire it started on, whatever the negotiation now says."""
        engine = get_runtime().workflow
        _registry, invocation, context, broker = _turn(engine, monkeypatch)
        _Provider(engine, monkeypatch, [
            native("looking", [SEARCH], tag="1"),
            chat_structured("400 hours", transport="chat", model="gpt-6-astra"),
        ])
        assert _model(broker, invocation, 1)["ok"]
        assert _round(broker, invocation, 2)["ok"]
        accepted = context.continuation

        reply = _model(broker, invocation, 3)

        assert reply["ok"] is False and reply["code"] == "continuation_mismatch"
        assert context.continuation == accepted
        assert len(context.transcript.entries) == 2

    def test_a_native_backend_that_returns_no_candidate_cannot_continue(
        self, store, monkeypatch
    ):
        engine = get_runtime().workflow
        _registry, invocation, context, broker = _turn(engine, monkeypatch)
        silent = {"content": "hm", "tool_calls": [], "assistant_message": None, "usage": {}}
        _Provider(engine, monkeypatch, [native("looking", [SEARCH], tag="1"), silent])
        assert _model(broker, invocation, 1)["ok"]
        assert _round(broker, invocation, 2)["ok"]
        accepted = context.continuation

        reply = _model(broker, invocation, 3)

        assert reply["ok"] is False and reply["code"] == "continuation_mismatch"
        assert context.continuation == accepted

    def test_a_strategy_change_on_the_same_wire_is_refused(self, store, monkeypatch):
        """Same provider, wire and model, a candidate on another strategy:
        another representation of the conversation, refused like the rest."""
        engine = get_runtime().workflow
        _registry, invocation, context, broker = _turn(engine, monkeypatch)
        _Provider(engine, monkeypatch, [
            native("looking", [SEARCH], tag="1"),
            chat_structured("400 hours", transport="responses", model="gpt-6-astra"),
        ])
        assert _model(broker, invocation, 1)["ok"]
        assert _round(broker, invocation, 2)["ok"]

        reply = _model(broker, invocation, 3)

        assert reply["ok"] is False and reply["code"] == "continuation_mismatch"
        assert context.continuation.through_operation_seq == 1

    def test_a_candidate_on_a_strategy_this_release_does_not_know_is_rejected(
        self, store, monkeypatch
    ):
        """A later adapter's state is foreign here, whatever it resembles."""
        engine = get_runtime().workflow
        _registry, invocation, context, broker = _turn(engine, monkeypatch)
        v2 = native("looking", [SEARCH], tag="1")

        def newer(messages, continuation):
            reply = v2(messages, continuation)
            reply["continuation"]["strategy"] = "openai.responses.native.v2"
            return reply

        _Provider(engine, monkeypatch, [newer])

        reply = _model(broker, invocation, 1)

        assert reply["ok"] is False and reply["code"] == "model_turn_rejected"
        assert context.continuation is None
        assert context.transcript.entries == []

    def test_a_wire_change_mid_invocation_is_refused(self, store, monkeypatch):
        """Same strategy, same model, a different transport: still a
        different representation of the conversation, still refused."""
        engine = get_runtime().workflow
        _registry, invocation, context, broker = _turn(engine, monkeypatch)
        _Provider(engine, monkeypatch, [
            native("looking", [SEARCH], tag="1"),
            native("400 hours", tag="2", transport="responses_v2"),
        ])
        assert _model(broker, invocation, 1)["ok"]
        assert _round(broker, invocation, 2)["ok"]

        reply = _model(broker, invocation, 3)

        assert reply["ok"] is False and reply["code"] == "continuation_mismatch"
        assert context.continuation.through_operation_seq == 1

    def test_a_model_change_mid_invocation_is_refused(self, store, monkeypatch):
        engine = get_runtime().workflow
        _registry, invocation, context, broker = _turn(engine, monkeypatch)
        _Provider(engine, monkeypatch, [
            native("looking", [SEARCH], tag="1"),
            native("400 hours", tag="2", model="gpt-6-other"),
        ])
        assert _model(broker, invocation, 1)["ok"]
        assert _round(broker, invocation, 2)["ok"]

        reply = _model(broker, invocation, 3)

        assert reply["ok"] is False and reply["code"] == "continuation_mismatch"
        assert context.continuation.through_operation_seq == 1


class TestNothingOfItCrossesTheWireOrTheLogs:
    def test_the_reply_and_the_logs_carry_no_accepted_state(
        self, store, monkeypatch, capfd
    ):
        engine = get_runtime().workflow
        _registry, invocation, context, broker = _turn(engine, monkeypatch)
        _Provider(engine, monkeypatch, [
            native("looking", [SEARCH], tag="1"), native("400 hours", tag="2"),
        ])

        first = _model(broker, invocation, 1)
        assert _round(broker, invocation, 2)["ok"]
        second = _model(broker, invocation, 3)

        for reply in (first, second):
            assert reply["ok"]
            crossed = json.dumps(reply)
            assert "continuation" not in reply["result"]
            assert MARK not in crossed and "encrypted_content" not in crossed
        assert SENTINEL in json.dumps(context.continuation.as_dict(), ensure_ascii=False)
        out, err = capfd.readouterr()
        assert MARK not in out and MARK not in err
        assert "continuation_candidate" in out + err
        assert invocation.ledger.get(3).state == COMMITTED
