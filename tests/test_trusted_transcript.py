"""The parent's own record of the conversation the worker is driving.

The worker decides which tools to call and assembles a message list, but every
message in it describes something the parent did. What comes back is a claim
about that conversation, and a later stage that labelled the worker's copy
would be attaching authority to bytes the untrusted side chose.

So the parent keeps its own. These witnesses are about that record being
complete, correctly identified, and restored exactly once - the properties an
offer stage needs before it can build model input from the record rather than
from what the worker sent back.
"""

from __future__ import annotations

import uuid

from liminallm.service.broker import CapabilityBroker, InvocationContext
from liminallm.service.invocation import InvocationRegistry
from liminallm.service.provenance import SourceRegistry
from liminallm.service.runtime import get_runtime
from liminallm.service.transcript import (
    ModelTurn,
    ToolRound,
    TrustedToolResult,
    TrustedTranscript,
    calls_match,
)

RESULTS = [{"title": "A", "url": "https://a.example", "snippet": "four hundred"}]


def _web(engine, monkeypatch):
    monkeypatch.setattr(
        engine,
        "_web_settings",
        lambda: {
            "enabled": True, "provider": "x", "api_key": "k", "engine_id": "",
            "timeout": 5, "proxy": None, "max_bytes": 1000,
            "allow_private": False,
        },
    )
    from liminallm.service import web

    monkeypatch.setattr(web, "search_web", lambda *a, **k: [dict(r) for r in RESULTS])
    monkeypatch.setattr(engine, "tool_network_policy", None)


def _turn(engine, monkeypatch):
    _web(engine, monkeypatch)
    registry = SourceRegistry()
    invocation = InvocationRegistry().open(
        uuid.uuid4().hex, tool="agent.files_v1", user_id="u", tenant_id=None
    )
    context = InvocationContext(user_id="u", source_registry=registry)
    return registry, invocation, context, CapabilityBroker(engine, context)


def _model(engine, monkeypatch, content="", calls=()):
    monkeypatch.setattr(
        engine.llm,
        "generate_with_tools",
        lambda *a, **k: {
            "content": content,
            "tool_calls": [dict(call) for call in calls],
            "assistant_message": None,
            "usage": {},
        },
        raising=False,
    )


def _ask(broker, invocation, capability, payload, seq):
    reply = broker._answer(
        invocation,
        {"capability": capability, "operation_seq": seq, "payload": payload},
    )
    assert reply["ok"], reply
    return reply


SEARCH = {"id": "c1", "name": "web_search", "arguments": '{"query": "hours"}'}
SUBMITTED = {"id": "c1", "name": "web_search", "arguments": {"query": "hours"}}


class TestTheRecordSpansTheWholeTurn:
    def test_an_earlier_round_survives_a_later_model_turn(
        self, store, monkeypatch
    ):
        """The third model call's prompt holds the first round's results.
        `canonical_model_response` is replacement state and cannot say so,
        which is why this record exists beside it rather than instead."""
        engine = get_runtime().workflow
        _registry, invocation, context, broker = _turn(engine, monkeypatch)

        _model(engine, monkeypatch, content="looking", calls=[SEARCH])
        _ask(broker, invocation, "llm.generate_with_tools",
             {"messages": [], "tools": []}, 1)
        _ask(broker, invocation, "tools.round",
             {"calls": [SUBMITTED], "fallback_query": "hours"}, 2)
        _model(engine, monkeypatch, content="400 hours")
        _ask(broker, invocation, "llm.generate_with_tools",
             {"messages": [], "tools": []}, 3)

        assert [
            (type(entry).__name__, entry.operation_seq)
            for entry in context.transcript.entries
        ] == [("ModelTurn", 1), ("ToolRound", 2), ("ModelTurn", 3)]
        # The one the citation layer reads is still only the last.
        assert context.canonical_model_response["content"] == "400 hours"
        # And the round that grounded it is still here.
        assert context.transcript.rounds()[0].results[0].spans

    def test_every_call_of_a_round_is_recorded_even_ungrounded_ones(
        self, store, monkeypatch
    ):
        """A tool message has to be rebuildable, and a gap in the middle of a
        round is as much a gap as a wrong entry."""
        engine = get_runtime().workflow
        _registry, invocation, context, broker = _turn(engine, monkeypatch)
        calls = [
            SEARCH,
            {"id": "c2", "name": "run_python", "arguments": '{"code": "1"}'},
        ]
        _model(engine, monkeypatch, calls=calls)
        _ask(broker, invocation, "llm.generate_with_tools",
             {"messages": [], "tools": []}, 1)
        _ask(broker, invocation, "tools.round", {
            "calls": [
                SUBMITTED,
                {"id": "c2", "name": "run_python", "arguments": {"code": "1"}},
            ],
            "fallback_query": "hours",
        }, 2)

        results = context.transcript.rounds()[0].results
        assert [r.call_index for r in results] == [0, 1]
        assert [r.tool_name for r in results] == ["web_search", "run_python"]
        # The second grounded nothing and is recorded all the same.
        assert results[0].spans and not results[1].spans

    def test_the_parent_keeps_the_text_each_call_produced(
        self, store, monkeypatch
    ):
        """The record exists so a tool message can be rebuilt. Without the
        text there is nothing to rebuild it from, and the only other copy is
        the worker's."""
        engine = get_runtime().workflow
        _registry, invocation, context, broker = _turn(engine, monkeypatch)
        _model(engine, monkeypatch, calls=[SEARCH])
        _ask(broker, invocation, "llm.generate_with_tools",
             {"messages": [], "tools": []}, 1)
        reply = _ask(broker, invocation, "tools.round",
                     {"calls": [SUBMITTED], "fallback_query": "hours"}, 2)

        results = context.transcript.rounds()[0].results
        assert [r.text for r in results] == reply["result"]["results"]
        # And the spans index that text, not some other copy of it.
        span = results[0].spans[0]
        assert "https://a.example" in results[0].text[span.start : span.end]


class TestTheCallIsNamedByWhatTheParentDispatched:
    def test_a_duplicate_call_id_cannot_become_an_identity(
        self, store, monkeypatch
    ):
        """A provider may repeat an id or send none. The parent's own
        enumeration is what separates the calls."""
        engine = get_runtime().workflow
        _registry, invocation, context, broker = _turn(engine, monkeypatch)
        same = {"id": "same", "name": "web_search", "arguments": '{"query": "hours"}'}
        _model(engine, monkeypatch, calls=[same, same])
        _ask(broker, invocation, "llm.generate_with_tools",
             {"messages": [], "tools": []}, 1)
        _ask(broker, invocation, "tools.round", {
            "calls": [
                {"id": "same", "name": "web_search", "arguments": {"query": "hours"}},
                {"id": "same", "name": "web_search", "arguments": {"query": "hours"}},
            ],
            "fallback_query": "hours",
        }, 2)

        results = context.transcript.rounds()[0].results
        assert len({r.submitted_call_id for r in results}) == 1
        assert [r.call_index for r in results] == [0, 1]

    def test_a_call_with_no_id_still_gets_a_tool_message_id(
        self, store, monkeypatch
    ):
        """The same rule the worker's own assembly uses - the id or the name -
        computed here so the parent need not read it back to rebuild the
        message."""
        engine = get_runtime().workflow
        _registry, invocation, context, broker = _turn(engine, monkeypatch)
        anonymous = {"name": "web_search", "arguments": '{"query": "hours"}'}
        _model(engine, monkeypatch, calls=[anonymous])
        _ask(broker, invocation, "llm.generate_with_tools",
             {"messages": [], "tools": []}, 1)
        _ask(broker, invocation, "tools.round", {
            "calls": [{"name": "web_search", "arguments": {"query": "hours"}}],
            "fallback_query": "hours",
        }, 2)

        result = context.transcript.rounds()[0].results[0]
        assert result.submitted_call_id == ""
        assert result.tool_message_id == "web_search"


class TestARoundThatIsNotTheOneAskedForCarriesNoAuthority:
    """It still runs: what a worker may request is the capability layer's
    question, and that layer answers it unchanged. But the parent can no
    longer reconstruct the exchange, so nothing in the round may be cited."""

    def test_a_different_tool_makes_the_round_unofferable(
        self, store, monkeypatch
    ):
        engine = get_runtime().workflow
        _registry, invocation, context, broker = _turn(engine, monkeypatch)
        _model(engine, monkeypatch, calls=[
            {"id": "c1", "name": "file_search", "arguments": '{"query": "manual"}'}
        ])
        _ask(broker, invocation, "llm.generate_with_tools",
             {"messages": [], "tools": []}, 1)
        reply = _ask(broker, invocation, "tools.round", {
            "calls": [
                {"id": "c1", "name": "web_search", "arguments": {"query": "else"}}
            ],
            "fallback_query": "x",
        }, 2)

        # Executed, as before.
        assert reply["result"]["results"]
        # And not eligible to carry a citation.
        assert context.transcript.rounds()[0].offerable is False

    def test_different_arguments_make_the_round_unofferable(
        self, store, monkeypatch
    ):
        engine = get_runtime().workflow
        _registry, invocation, context, broker = _turn(engine, monkeypatch)
        _model(engine, monkeypatch, calls=[SEARCH])
        _ask(broker, invocation, "llm.generate_with_tools",
             {"messages": [], "tools": []}, 1)
        _ask(broker, invocation, "tools.round", {
            "calls": [
                {"id": "c1", "name": "web_search", "arguments": {"query": "other"}}
            ],
            "fallback_query": "x",
        }, 2)
        assert context.transcript.rounds()[0].offerable is False

    def test_the_round_the_model_asked_for_stays_offerable(
        self, store, monkeypatch
    ):
        engine = get_runtime().workflow
        _registry, invocation, context, broker = _turn(engine, monkeypatch)
        _model(engine, monkeypatch, calls=[SEARCH])
        _ask(broker, invocation, "llm.generate_with_tools",
             {"messages": [], "tools": []}, 1)
        _ask(broker, invocation, "tools.round",
             {"calls": [SUBMITTED], "fallback_query": "hours"}, 2)
        assert context.transcript.rounds()[0].offerable is True

    def test_a_different_tool_with_the_same_arguments_is_still_divergence(
        self, store, monkeypatch
    ):
        """Only the name differs, so a check that compared arguments alone
        would call this the round the model asked for."""
        engine = get_runtime().workflow
        _registry, invocation, context, broker = _turn(engine, monkeypatch)
        _model(engine, monkeypatch, calls=[
            {"id": "c1", "name": "note_search", "arguments": '{"query": "hours"}'}
        ])
        _ask(broker, invocation, "llm.generate_with_tools",
             {"messages": [], "tools": []}, 1)
        _ask(broker, invocation, "tools.round",
             {"calls": [SUBMITTED], "fallback_query": "hours"}, 2)
        assert context.transcript.rounds()[0].offerable is False

    def test_an_extra_call_the_model_did_not_ask_for_is_divergence(
        self, store, monkeypatch
    ):
        """Every call the model asked for is present and one more besides.
        Comparing pairwise without comparing the count would miss it."""
        engine = get_runtime().workflow
        _registry, invocation, context, broker = _turn(engine, monkeypatch)
        _model(engine, monkeypatch, calls=[SEARCH])
        _ask(broker, invocation, "llm.generate_with_tools",
             {"messages": [], "tools": []}, 1)
        _ask(broker, invocation, "tools.round", {
            "calls": [
                SUBMITTED,
                {"id": "c2", "name": "web_search", "arguments": {"query": "extra"}},
            ],
            "fallback_query": "hours",
        }, 2)
        assert context.transcript.rounds()[0].offerable is False

    def test_a_round_the_model_never_asked_for_at_all_is_divergence(
        self, store, monkeypatch
    ):
        """No preceding model turn, so nothing offered these calls."""
        engine = get_runtime().workflow
        _registry, invocation, context, broker = _turn(engine, monkeypatch)
        _ask(broker, invocation, "tools.round",
             {"calls": [SUBMITTED], "fallback_query": "hours"}, 1)
        assert context.transcript.rounds()[0].offerable is False

    def test_the_ids_are_not_what_is_compared(self):
        """They are the provider's, they arrive through the worker, and a
        round that renamed them is still the same two calls."""
        assert calls_match(
            [{"id": "a", "name": "t", "arguments": '{"q": 1}'}],
            [{"id": "zzz", "name": "t", "arguments": {"q": 1}}],
        )

    def test_an_unreadable_argument_string_is_not_a_match(self):
        """A round the parent cannot read is a round it cannot say was
        faithfully carried out."""
        assert not calls_match(
            [{"name": "t", "arguments": "{not json"}],
            [{"name": "t", "arguments": {}}],
        )


class TestOneOperationLeavesOneEntry:
    def test_a_replayed_round_is_restored_rather_than_duplicated(
        self, store, monkeypatch
    ):
        engine = get_runtime().workflow
        registry, invocation, first, broker = _turn(engine, monkeypatch)
        _model(engine, monkeypatch, calls=[SEARCH])
        _ask(broker, invocation, "llm.generate_with_tools",
             {"messages": [], "tools": []}, 1)
        payload = {"calls": [SUBMITTED], "fallback_query": "hours"}
        _ask(broker, invocation, "tools.round", payload, 2)

        second = InvocationContext(user_id="u", source_registry=registry)
        replay_broker = CapabilityBroker(engine, second)
        # The replacement attempt replays both operations in order.
        _ask(replay_broker, invocation, "llm.generate_with_tools",
             {"messages": [], "tools": []}, 1)
        reply = replay_broker._answer(
            invocation,
            {"capability": "tools.round", "operation_seq": 2, "payload": payload},
        )
        assert reply.get("replayed"), "the fixture did not exercise a replay"

        assert [
            (type(entry).__name__, entry.operation_seq)
            for entry in second.transcript.entries
        ] == [("ModelTurn", 1), ("ToolRound", 2)]
        assert second.transcript.as_list() == first.transcript.as_list()

    def test_restoring_the_same_delta_twice_leaves_one_entry(self):
        """Two attempts of one node each replay the ledger. One operation has
        one outcome however many times it is read back."""
        transcript = TrustedTranscript()
        entry = ToolRound(
            operation_seq=2,
            results=(TrustedToolResult(operation_seq=2, call_index=0,
                                       tool_name="web_search", text="x"),),
        )
        transcript.restore([entry.as_dict()])
        transcript.restore([entry.as_dict()])
        assert len(transcript.entries) == 1

    def test_the_entries_stay_in_operation_order(self):
        transcript = TrustedTranscript()
        transcript.record(ModelTurn(operation_seq=3, content="third"))
        transcript.record(ToolRound(operation_seq=2))
        transcript.record(ModelTurn(operation_seq=1, content="first"))
        assert [e.operation_seq for e in transcript.entries] == [1, 2, 3]

    def test_a_round_and_a_model_turn_can_share_no_sequence(self):
        """Different kinds are different entries even at the same number:
        an operation is one or the other, and collapsing them would drop one."""
        transcript = TrustedTranscript()
        transcript.record(ModelTurn(operation_seq=1, content="a"))
        transcript.record(ToolRound(operation_seq=1))
        assert len(transcript.entries) == 2


class TestTheRecordSurvivesTheLedger:
    def test_a_round_is_the_same_after_a_round_trip(self):
        entry = ToolRound(
            operation_seq=4,
            offerable=False,
            results=(
                TrustedToolResult(
                    operation_seq=4, call_index=1, tool_name="web_search",
                    submitted_call_id="c2", tool_message_id="c2", text="body",
                ),
            ),
        )
        assert ToolRound.from_dict(entry.as_dict()) == entry

    def test_a_model_turn_is_the_same_after_a_round_trip(self):
        entry = ModelTurn(
            operation_seq=1,
            content="looking",
            tool_calls=({"id": "c1", "name": "web_search", "arguments": "{}"},),
            assistant_message={"role": "assistant", "content": "looking"},
        )
        assert ModelTurn.from_dict(entry.as_dict()) == entry


class TestNoneOfItCrossesThePipe:
    def test_the_round_reply_carries_no_transcript(self, store, monkeypatch):
        engine = get_runtime().workflow
        _registry, invocation, context, broker = _turn(engine, monkeypatch)
        _model(engine, monkeypatch, calls=[SEARCH])
        _ask(broker, invocation, "llm.generate_with_tools",
             {"messages": [], "tools": []}, 1)
        reply = _ask(broker, invocation, "tools.round",
                     {"calls": [SUBMITTED], "fallback_query": "hours"}, 2)
        assert context.transcript.rounds(), "the fixture recorded nothing"
        serialized = str(reply["result"])
        for leak in ("transcript", "call_index", "operation_seq", "offerable",
                     "src_", "tool_message_id"):
            assert leak not in serialized, leak
