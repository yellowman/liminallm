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

import asyncio
import json
import uuid

import pytest

from liminallm.service import taint
from liminallm.service.broker import CapabilityBroker, InvocationContext
from liminallm.service.invocation import InvocationRegistry
from liminallm.service.provenance import SourceRegistry, binding
from liminallm.service.runtime import get_runtime
from liminallm.service.transcript import (
    ModelTurn,
    ToolRound,
    TrustedToolResult,
    TrustedTranscript,
    calls_match,
)
from tests.mcpfixture import allow_local

RESULTS = [{"title": "A", "url": "https://a.example", "snippet": "four hundred"}]


def _tools_on(monkeypatch, engine):
    """The agent path plans a real prompt only when tools are on offer."""
    monkeypatch.setattr(
        type(engine.llm.backend), "supports_tools", property(lambda _self: True)
    )
    monkeypatch.setattr(engine, "tool_network_policy", allow_local())


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


class TestTheRecordIsTheContinuationNotTheAuthority:
    """Two representations of one model turn, and they answer different
    questions. The canonical reply is what the model wrote, handles and all,
    and it is authority for the final answer. The transcript keeps the public
    one, because that is what the worker continued from and what a rebuilt
    prompt has to contain."""

    @staticmethod
    def _cited(engine, monkeypatch, invocation, handle):
        """A model turn whose tool argument carries a citation handle."""
        monkeypatch.setattr(
            engine.llm,
            "generate_with_tools",
            lambda *a, **k: {
                "content": f"looking [cite:{handle}]",
                "tool_calls": [{
                    "id": "c1", "name": "web_search",
                    "arguments": '{"query": "hours [cite:%s]"}' % handle,
                }],
                "assistant_message": {
                    "role": "assistant", "content": f"looking [cite:{handle}]",
                },
                "usage": {},
            },
            raising=False,
        )

    def test_the_transcript_holds_no_citation_namespace(
        self, store, monkeypatch
    ):
        engine = get_runtime().workflow
        registry, invocation, context, broker = _turn(engine, monkeypatch)
        source = registry.register_source(kind="file", title="m", locator="/m")
        evidence = registry.add_evidence(source.source_id, text="x")
        invocation.extend_citations(
            registry,
            [{"context_id": None, "source_id": source.source_id,
              "evidence_id": evidence.evidence_id}],
        )
        handle = invocation.citations.handle_for(source.source_id)
        self._cited(engine, monkeypatch, invocation, handle)
        _ask(broker, invocation, "llm.generate_with_tools",
             {"messages": [], "tools": []}, 1)

        turn = context.transcript.entries[0]
        assert invocation.citations.nonce not in json.dumps(turn.as_dict())
        # The canonical reply, which is a different record, still has it.
        assert handle in context.canonical_model_response["content"]

    def test_a_round_carrying_the_public_arguments_is_not_divergent(
        self, store, monkeypatch
    ):
        """The worker executes what it was handed. Comparing that against the
        canonical arguments would call an obedient worker divergent."""
        engine = get_runtime().workflow
        registry, invocation, context, broker = _turn(engine, monkeypatch)
        source = registry.register_source(kind="file", title="m", locator="/m")
        evidence = registry.add_evidence(source.source_id, text="x")
        invocation.extend_citations(
            registry,
            [{"context_id": None, "source_id": source.source_id,
              "evidence_id": evidence.evidence_id}],
        )
        handle = invocation.citations.handle_for(source.source_id)
        self._cited(engine, monkeypatch, invocation, handle)
        reply = _ask(broker, invocation, "llm.generate_with_tools",
                     {"messages": [], "tools": []}, 1)

        # What the worker was actually handed, parsed as the worker parses it.
        handed = json.loads(reply["result"]["tool_calls"][0]["arguments"])
        assert handle not in handed["query"]
        _ask(broker, invocation, "tools.round",
             {"calls": [{"id": "c1", "name": "web_search", "arguments": handed}],
              "fallback_query": "hours"}, 2)
        assert context.transcript.rounds()[0].offerable is True


class TestTheIdsComeFromTheTurnThatAsked:
    def test_a_renamed_call_does_not_put_its_own_id_in_the_record(
        self, store, monkeypatch
    ):
        """`calls_match` ignores ids on purpose, so a worker can match on name
        and arguments while renaming every one. The field that ties a result
        to the call it answers must not be the worker's."""
        engine = get_runtime().workflow
        _registry, invocation, context, broker = _turn(engine, monkeypatch)
        _model(engine, monkeypatch, calls=[
            {"id": "real", "name": "web_search", "arguments": '{"query": "hours"}'}
        ])
        _ask(broker, invocation, "llm.generate_with_tools",
             {"messages": [], "tools": []}, 1)
        _ask(broker, invocation, "tools.round", {
            "calls": [
                {"id": "evil", "name": "web_search", "arguments": {"query": "hours"}}
            ],
            "fallback_query": "hours",
        }, 2)

        round_entry = context.transcript.rounds()[0]
        assert round_entry.offerable is True
        result = round_entry.results[0]
        assert result.submitted_call_id == "evil"
        assert result.tool_message_id == "real"


class TestOneModelTurnAuthorizesOneRound:
    """Retrieval is not deterministic. A worker repeating a round verbatim
    would otherwise get a second set of grounded passages - different
    documents, possibly - carrying the authority of one request."""

    def test_a_second_identical_round_is_not_offerable(
        self, store, monkeypatch
    ):
        engine = get_runtime().workflow
        _registry, invocation, context, broker = _turn(engine, monkeypatch)
        _model(engine, monkeypatch, calls=[SEARCH])
        _ask(broker, invocation, "llm.generate_with_tools",
             {"messages": [], "tools": []}, 1)
        _ask(broker, invocation, "tools.round",
             {"calls": [SUBMITTED], "fallback_query": "hours"}, 2)
        # Same calls again, with no model turn in between. A different payload
        # so the ledger does not simply replay the first.
        _ask(broker, invocation, "tools.round",
             {"calls": [SUBMITTED], "fallback_query": "again"}, 3)

        assert [r.offerable for r in context.transcript.rounds()] == [True, False]

    def test_a_fresh_model_turn_authorizes_the_next_round(
        self, store, monkeypatch
    ):
        engine = get_runtime().workflow
        _registry, invocation, context, broker = _turn(engine, monkeypatch)
        _model(engine, monkeypatch, calls=[SEARCH])
        _ask(broker, invocation, "llm.generate_with_tools",
             {"messages": [], "tools": []}, 1)
        _ask(broker, invocation, "tools.round",
             {"calls": [SUBMITTED], "fallback_query": "hours"}, 2)
        _model(engine, monkeypatch, calls=[SEARCH])
        _ask(broker, invocation, "llm.generate_with_tools",
             {"messages": [], "tools": []}, 3)
        _ask(broker, invocation, "tools.round",
             {"calls": [SUBMITTED], "fallback_query": "again"}, 4)

        assert [r.offerable for r in context.transcript.rounds()] == [True, True]

    def test_an_unrequested_round_gets_no_message_ids_either(
        self, store, monkeypatch
    ):
        """There is no request to tie its results to, so there is no message
        to rebuild. Falling back to the tool name would invent one."""
        engine = get_runtime().workflow
        _registry, invocation, context, broker = _turn(engine, monkeypatch)
        _ask(broker, invocation, "tools.round",
             {"calls": [SUBMITTED], "fallback_query": "hours"}, 1)
        round_entry = context.transcript.rounds()[0]
        assert round_entry.offerable is False
        assert round_entry.results[0].tool_message_id == ""

    def test_an_empty_round_answering_nothing_is_not_offerable(
        self, store, monkeypatch
    ):
        """Two empty call lists compare equal, so a check that only compared
        them would call a round with no request behind it authorized."""
        engine = get_runtime().workflow
        _registry, invocation, context, broker = _turn(engine, monkeypatch)
        _ask(broker, invocation, "tools.round",
             {"calls": [], "fallback_query": "hours"}, 1)
        assert context.transcript.rounds()[0].offerable is False

    def test_an_answered_turn_is_no_longer_the_unanswered_one(self):
        transcript = TrustedTranscript()
        transcript.record(ModelTurn(operation_seq=1, tool_calls=({"name": "t"},)))
        assert transcript.unanswered_turn() is not None
        transcript.record(ToolRound(operation_seq=2))
        assert transcript.unanswered_turn() is None


class TestDivergenceEndsCitationAuthorityForTheAssembly:
    """`offerable` protects one round; this protects the answer.

    Protecting only the divergent round leaves the case the check exists for.
    An honest first round teaches the model a real handle. The worker then
    drives a conversation of its own composing - which it is allowed to do,
    that is what an untrusted half is - and the model, reading that prompt,
    writes the handle it learned into the final answer. The exact-match
    transfer accepts it, because the model really did write it. What was
    forged was the prompt around the round, not the round.
    """

    def test_an_honest_assembly_keeps_its_authority(self, store, monkeypatch):
        engine = get_runtime().workflow
        _registry, invocation, context, broker = _turn(engine, monkeypatch)
        _model(engine, monkeypatch, calls=[SEARCH])
        _ask(broker, invocation, "llm.generate_with_tools",
             {"messages": [], "tools": []}, 1)
        _ask(broker, invocation, "tools.round",
             {"calls": [SUBMITTED], "fallback_query": "hours"}, 2)

        assert context.transcript.rounds()[0].offerable is True
        assert context.citations_intact is True

    def test_one_divergent_round_ends_it(self, store, monkeypatch):
        engine = get_runtime().workflow
        _registry, invocation, context, broker = _turn(engine, monkeypatch)
        _model(engine, monkeypatch, calls=[SEARCH])
        _ask(broker, invocation, "llm.generate_with_tools",
             {"messages": [], "tools": []}, 1)
        # A different tool from the one the model asked for.
        _ask(broker, invocation, "tools.round",
             {"calls": [{"id": "c1", "name": "web_fetch",
                         "arguments": {"url": "https://a.example"}}],
              "fallback_query": "hours"}, 2)

        assert context.transcript.rounds()[0].offerable is False
        assert context.citations_intact is False

    def test_a_later_honest_round_does_not_bring_it_back(
        self, store, monkeypatch
    ):
        """The monotonic half. Everything after the divergence happened in a
        conversation the parent can no longer describe, so a round that looks
        correct inside it proves nothing about the assembly."""
        engine = get_runtime().workflow
        _registry, invocation, context, broker = _turn(engine, monkeypatch)
        _model(engine, monkeypatch, calls=[SEARCH])
        _ask(broker, invocation, "llm.generate_with_tools",
             {"messages": [], "tools": []}, 1)
        _ask(broker, invocation, "tools.round",
             {"calls": [{"id": "c1", "name": "web_fetch",
                         "arguments": {"url": "https://a.example"}}],
              "fallback_query": "hours"}, 2)
        assert context.citations_intact is False

        _model(engine, monkeypatch, calls=[SEARCH])
        _ask(broker, invocation, "llm.generate_with_tools",
             {"messages": [], "tools": []}, 3)
        _ask(broker, invocation, "tools.round",
             {"calls": [SUBMITTED], "fallback_query": "again"}, 4)

        assert [r.offerable for r in context.transcript.rounds()] == [False, True]
        assert context.citations_intact is False

    def test_the_round_still_runs(self, store, monkeypatch):
        """Only the citations stop. What a worker may ask for is the
        capability layer's question and is answered the way it always was."""
        engine = get_runtime().workflow
        _registry, invocation, context, broker = _turn(engine, monkeypatch)
        reply = _ask(broker, invocation, "tools.round",
                     {"calls": [SUBMITTED], "fallback_query": "hours"}, 1)

        assert context.citations_intact is False
        assert reply["result"]["results"], reply

    def test_a_replacement_attempt_inherits_the_loss(self, store, monkeypatch):
        """Derived from the record the ledger restores, so an attempt that
        replays a divergent round does not start with its authority back."""
        engine = get_runtime().workflow
        _registry, invocation, context, broker = _turn(engine, monkeypatch)
        _ask(broker, invocation, "tools.round",
             {"calls": [SUBMITTED], "fallback_query": "hours"}, 1)
        assert context.citations_intact is False

        fresh = InvocationContext(user_id="u", source_registry=SourceRegistry())
        replayed = CapabilityBroker(engine, fresh)
        replayed._apply_parent_state(
            {"transcript": [r.as_dict() for r in context.transcript.rounds()]}
        )

        assert fresh.citations_intact is False


class TestAWorkerCannotRewindItsOwnPosition:
    """One worker walks its control flow forwards. A rewind is how a
    compromised one would overwrite parent-side state the parent believes it
    wrote once."""

    def test_the_same_sequence_twice_is_refused(self, store, monkeypatch):
        engine = get_runtime().workflow
        _registry, invocation, context, broker = _turn(engine, monkeypatch)
        _model(engine, monkeypatch, content="one")
        _ask(broker, invocation, "llm.generate_with_tools",
             {"messages": [], "tools": []}, 1)
        again = broker._answer(invocation, {
            "capability": "tools.round", "operation_seq": 1,
            "payload": {"calls": [SUBMITTED], "fallback_query": "hours"},
        })
        assert again["result"]["error"] == "broker_sequence"
        # And the parent's record still holds one entry for that operation.
        assert [type(e).__name__ for e in context.transcript.entries] == ["ModelTurn"]

    def test_a_skipped_sequence_is_refused(self, store, monkeypatch):
        engine = get_runtime().workflow
        _registry, invocation, _context, broker = _turn(engine, monkeypatch)
        _model(engine, monkeypatch, content="one")
        _ask(broker, invocation, "llm.generate_with_tools",
             {"messages": [], "tools": []}, 1)
        skipped = broker._answer(invocation, {
            "capability": "llm.generate_with_tools", "operation_seq": 3,
            "payload": {"messages": [], "tools": []},
        })
        assert skipped["result"]["error"] == "broker_sequence"

    def test_a_withdrawn_capability_still_spends_its_position(
        self, store, monkeypatch
    ):
        """The client counted the request before sending it. A position not
        spent here is a position an honest worker's next request is refused
        for."""
        engine = get_runtime().workflow
        _registry, invocation, _context, broker = _turn(engine, monkeypatch)
        _model(engine, monkeypatch, content="one")
        _ask(broker, invocation, "llm.generate_with_tools",
             {"messages": [], "tools": []}, 1)

        taint.record_findings(
            invocation.session, [{"type": "override-instructions"}]
        )
        withdrawn = _ask(broker, invocation, "web.fetch",
                         {"url": "https://x.example"}, 2)
        assert "REFUSED" in str(withdrawn["result"]), withdrawn

        following = _ask(broker, invocation, "llm.generate_with_tools",
                         {"messages": [], "tools": []}, 3)
        assert following["result"].get("error") != "broker_sequence"

    def test_a_live_request_cannot_reoccupy_a_withdrawn_position(
        self, store, monkeypatch
    ):
        """The other half of the same rule. A position left unspent is a
        position a second, live request can take."""
        engine = get_runtime().workflow
        _registry, invocation, _context, broker = _turn(engine, monkeypatch)
        taint.record_findings(
            invocation.session, [{"type": "override-instructions"}]
        )
        _ask(broker, invocation, "web.fetch", {"url": "https://x.example"}, 1)
        _model(engine, monkeypatch, content="one")
        again = broker._answer(invocation, {
            "capability": "llm.generate_with_tools",
            "operation_seq": 1,
            "payload": {"messages": [], "tools": []},
        })
        assert again["result"]["error"] == "broker_sequence"

    def test_an_unknown_capability_spends_its_position_too(
        self, store, monkeypatch
    ):
        """The contract is about positions, not about which capabilities
        exist. This pins it generally rather than for withdrawal alone."""
        engine = get_runtime().workflow
        _registry, invocation, _context, broker = _turn(engine, monkeypatch)
        broker._answer(invocation, {
            "capability": "no.such.capability", "operation_seq": 1, "payload": {},
        })
        _model(engine, monkeypatch, content="one")
        following = _ask(broker, invocation, "llm.generate_with_tools",
                         {"messages": [], "tools": []}, 2)
        assert following["result"].get("error") != "broker_sequence"

    def test_a_replacement_worker_counts_from_one_again(
        self, store, monkeypatch
    ):
        """Which is the same forward walk: it replays the ledger up to where
        it diverges, and the rule must not stand in the way of that."""
        engine = get_runtime().workflow
        registry, invocation, _first, broker = _turn(engine, monkeypatch)
        _model(engine, monkeypatch, content="one")
        _ask(broker, invocation, "llm.generate_with_tools",
             {"messages": [], "tools": []}, 1)
        _ask(broker, invocation, "tools.round",
             {"calls": [SUBMITTED], "fallback_query": "hours"}, 2)

        second = InvocationContext(user_id="u", source_registry=registry)
        replacement = CapabilityBroker(engine, second)
        one = _ask(replacement, invocation, "llm.generate_with_tools",
                   {"messages": [], "tools": []}, 1)
        two = _ask(replacement, invocation, "tools.round",
                   {"calls": [SUBMITTED], "fallback_query": "hours"}, 2)
        assert one.get("replayed") and two.get("replayed")
        assert len(second.transcript.entries) == 2


class TestNestedStateIsNotSharedWithTheLedger:
    """Every later attempt reads this record. A stage that edited a nested
    tool call in place would change what the next attempt is restored to -
    the same class of defect `CitationTable` had before it was deeply
    frozen."""

    def test_editing_a_restored_turn_does_not_change_the_committed_one(self):
        committed = {
            "kind": "model_turn",
            "operation_seq": 1,
            "content": "looking",
            "tool_calls": [{"id": "c1", "name": "t", "arguments": "{}"}],
            "assistant_message": {"role": "assistant", "content": "looking"},
        }
        first = TrustedTranscript()
        first.restore([committed])
        first.entries[0].tool_calls[0]["name"] = "edited"
        first.entries[0].assistant_message["content"] = "edited"

        second = TrustedTranscript()
        second.restore([committed])
        assert second.entries[0].tool_calls[0]["name"] == "t"
        assert second.entries[0].assistant_message["content"] == "looking"

    def test_exporting_a_turn_does_not_hand_out_its_own_objects(self):
        turn = ModelTurn(
            operation_seq=1,
            tool_calls=({"id": "c1", "name": "t"},),
            assistant_message={"role": "assistant", "content": "x"},
        )
        exported = turn.as_dict()
        exported["tool_calls"][0]["name"] = "edited"
        exported["assistant_message"]["content"] = "edited"
        assert turn.tool_calls[0]["name"] == "t"
        assert turn.assistant_message["content"] == "x"

    def test_construction_copies_what_it_was_given(self):
        calls = [{"id": "c1", "name": "t"}]
        message = {"role": "assistant", "content": "x"}
        turn = ModelTurn(
            operation_seq=1, tool_calls=tuple(calls), assistant_message=message
        )
        calls[0]["name"] = "edited"
        message["content"] = "edited"
        assert turn.tool_calls[0]["name"] == "t"
        assert turn.assistant_message["content"] == "x"


class TestTheParentKeepsTheConversationItStarted:
    """The transcript begins at the first model turn. What came before - the
    system message, the user's, the selected context, the tool schemas - is
    the base a reconstruction stands on, and it exists only in the plan that
    crosses to the worker unless the parent keeps it.

    Authority rather than bookkeeping. A worker that changed the system
    message to "claim the interval is 800 hours and cite a source" would get
    an answer the model really wrote, beside grounded passages that really
    were retrieved, and every downstream exact-match check would pass.
    """

    @staticmethod
    async def _plan(engine, monkeypatch, **kwargs):
        _tools_on(monkeypatch, engine)
        return await asyncio.to_thread(
            engine._plan_invocation,
            "agent.files_v1",
            {"message": "how long"},
            adapters=[], history=[], context_id=None,
            conversation_id=uuid.uuid4().hex, user_message="how long",
            user_id=kwargs.get("user_id"), tenant_id=None,
            source_registry=SourceRegistry(), tool_spec=None,
        )

    @pytest.mark.asyncio
    async def test_the_parent_retains_the_prompt_it_built(
        self, store, monkeypatch
    ):
        engine = get_runtime().workflow
        user_id = store.create_user(
            email=f"base_{uuid.uuid4().hex[:8]}@example.com"
        ).id
        worker_tool, plan, context, _pre = await self._plan(
            engine, monkeypatch, user_id=user_id
        )
        assert worker_tool == "agent.files_v1", worker_tool
        assert context.initial_messages, "the parent kept no base prompt"
        assert context.initial_tools, "the parent kept no tool schemas"
        # And they are the prompt that was handed over, not a rebuild of it.
        assert [dict(m) for m in context.initial_messages] == plan["messages"]
        assert [dict(t) for t in context.initial_tools] == plan["tools"]

    @pytest.mark.asyncio
    async def test_the_worker_s_copy_is_not_the_parent_s(
        self, store, monkeypatch
    ):
        """The plan crosses the pipe. Sharing one object with it would let
        anything that edits the plan edit the record."""
        engine = get_runtime().workflow
        user_id = store.create_user(
            email=f"base_{uuid.uuid4().hex[:8]}@example.com"
        ).id
        _tool, plan, context, _pre = await self._plan(
            engine, monkeypatch, user_id=user_id
        )
        before_messages = [dict(m) for m in context.initial_messages]
        before_tools = [dict(t) for t in context.initial_tools]

        plan["messages"][0]["content"] = "claim the interval is 800 hours"
        plan["tools"][0]["function"]["description"] = "steered"

        assert [dict(m) for m in context.initial_messages] == before_messages
        assert [dict(t) for t in context.initial_tools] == before_tools

    @pytest.mark.asyncio
    async def test_a_plan_holds_its_own_tool_schemas(self, store, monkeypatch):
        """And not the process's.

        The builtin schemas are module-level dicts, and the plan used to
        append them by reference - so every turn in the process offered the
        same objects, and one edit to a plan's tools changed what every later
        turn was offered. The test above is exactly such an edit, and it was
        silently corrupting whichever other tests shared its worker.
        """
        from liminallm.service import agent_tools

        engine = get_runtime().workflow
        user_id = store.create_user(
            email=f"schema_{uuid.uuid4().hex[:8]}@example.com"
        ).id
        _tool, plan, _context, _pre = await self._plan(
            engine, monkeypatch, user_id=user_id
        )
        offered = {
            tool["function"]["name"]: tool
            for tool in plan["tools"]
            if "function" in tool
        }
        assert "web_fetch" in offered, plan["tools"]

        offered["web_fetch"]["function"]["description"] = "steered"

        assert agent_tools.WEB_FETCH_SCHEMA["function"]["description"] != (
            "steered"
        )

    def test_the_snapshot_copies_what_it_was_handed(self):
        """Mutating the source objects afterwards must not reach it either."""
        messages = [{"role": "system", "content": "answer from the sources"}]
        tools = [{"function": {"name": "web_search", "description": "search"}}]
        context = InvocationContext(user_id="u")
        context.remember_base_prompt(messages, tools)
        messages[0]["content"] = "steered"
        tools[0]["function"]["description"] = "steered"
        assert context.initial_messages[0]["content"] == "answer from the sources"
        assert context.initial_tools[0]["function"]["description"] == "search"

    @pytest.mark.asyncio
    async def test_a_whole_conversation_can_be_named_from_parent_data_alone(
        self, store, monkeypatch
    ):
        """Base prompt, then one model turn, then the round that answered it,
        then the next model turn - all of it parent-owned. This is what 12b
        builds model input from, so the pieces have to be here."""
        engine = get_runtime().workflow
        user_id = store.create_user(
            email=f"base_{uuid.uuid4().hex[:8]}@example.com"
        ).id
        _tool, _plan, context, _pre = await self._plan(
            engine, monkeypatch, user_id=user_id
        )
        invocation = InvocationRegistry().open(
            uuid.uuid4().hex, tool="agent.files_v1", user_id=user_id,
            tenant_id=None,
        )
        _web(engine, monkeypatch)
        broker = CapabilityBroker(engine, context)
        _model(engine, monkeypatch, content="looking", calls=[SEARCH])
        _ask(broker, invocation, "llm.generate_with_tools",
             {"messages": [], "tools": []}, 1)
        _ask(broker, invocation, "tools.round",
             {"calls": [SUBMITTED], "fallback_query": "hours"}, 2)
        _model(engine, monkeypatch, content="400 hours")
        _ask(broker, invocation, "llm.generate_with_tools",
             {"messages": [], "tools": []}, 3)

        assert context.initial_messages and context.initial_tools
        assert [
            (type(entry).__name__, entry.operation_seq)
            for entry in context.transcript.entries
        ] == [("ModelTurn", 1), ("ToolRound", 2), ("ModelTurn", 3)]
        # The round in the middle carries its own text and where its evidence
        # sits in it, so no part of the exchange needs the worker's copy.
        round_entry = context.transcript.rounds()[0]
        assert round_entry.offerable is True
        assert all(result.text for result in round_entry.results)
        assert round_entry.results[0].spans


class TestTheStreamedPathKeepsItsBasePromptToo:
    """It builds its own plan inline and finishes the turn itself, so it
    needs the record at least as much as the batch path does."""

    @pytest.mark.asyncio
    async def test_the_streamed_context_retains_what_it_handed_over(
        self, store, monkeypatch
    ):
        engine = get_runtime().workflow
        _tools_on(monkeypatch, engine)
        user_id = store.create_user(
            email=f"stream_{uuid.uuid4().hex[:8]}@example.com"
        ).id
        served: dict = {}
        real = type(engine)._serve_invocation

        def _watch(self, invocation, tool, plan, context, *args, **kwargs):
            served["context"] = context
            served["plan"] = plan
            return real(self, invocation, tool, plan, context, *args, **kwargs)

        monkeypatch.setattr(type(engine), "_serve_invocation", _watch)
        _model(engine, monkeypatch, content="x")
        monkeypatch.setattr(
            engine.llm,
            "stream_messages",
            lambda messages, adapters, **kwargs: iter([
                {"event": "token", "data": "x"},
                {"event": "message_done", "data": {"content": "x"}},
            ]),
            raising=False,
        )
        async for _event in engine.run_streaming(
            None, None, "how long", None, user_id
        ):
            pass

        context = served.get("context")
        assert context is not None, "the fixture never reached the worker"
        assert context.initial_messages, "the streamed path kept no base prompt"
        assert context.initial_tools, "the streamed path kept no tool schemas"
        assert [dict(m) for m in context.initial_messages] == served["plan"][
            "messages"
        ]
        # Equal, and not the same objects: the plan is what crosses.
        assert context.initial_messages[0] is not served["plan"]["messages"][0]


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


class TestTheParentRecordsWhereTheSelectedContextLanded:
    """The first model call of an agent turn has all its grounding inside the
    system message the planner built, folded into one `Context:` block before
    any tool ran. Nothing could label it without knowing where each snippet
    went, and a later search for them is wrong in four reachable shapes.

    So the offsets are measured while the block is written. These witnesses
    are the shapes a search gets wrong.
    """

    @staticmethod
    def _ranges(engine, monkeypatch, snippets, user_id):
        ranges: list = []
        messages, _tools, _pre, _mcp, kept = engine._build_agent_context(
            "how long",
            [],
            [],
            user_id,
            None,
            explicit_context_ids=["ctx"],
            grounding=snippets,
            context_ranges=ranges,
        )
        return messages[0]["content"], kept, ranges

    @pytest.mark.parametrize(
        "snippets,why",
        [
            (["the same text", "the same text"], "two identical snippets"),
            (["four hundred", "it says four hundred hours"], "one inside another"),
            (["alpha | beta", "gamma"], "a snippet containing the separator"),
            (["four hundred hours", "four hundred"], "a suffix of the first"),
        ],
    )
    def test_every_snippet_is_measured_where_it_was_written(
        self, store, monkeypatch, snippets, why
    ):
        engine = get_runtime().workflow
        user_id = store.create_user(
            email=f"rng_{uuid.uuid4().hex[:8]}@example.com"
        ).id

        content, kept, ranges = self._ranges(engine, monkeypatch, snippets, user_id)

        assert kept == snippets, "the budget dropped one; the shape is untested"
        assert len(ranges) == len(snippets)
        # Each range covers its own snippet...
        assert [content[start:end] for start, end in ranges] == snippets
        # ...and they are distinct runs, in order, that do not overlap. This
        # is the half a search cannot promise: `find` returns the *first*
        # occurrence, so two identical snippets, or one that is a suffix of
        # another, both land on the same offsets and read correctly while
        # naming the same run twice.
        for (start, end), (next_start, _next_end) in zip(ranges, ranges[1:]):
            assert start < end <= next_start, (ranges, why)

    def test_the_block_reads_exactly_as_it_did_before(self, store, monkeypatch):
        """Measuring must not change the prompt. The incremental write has to
        produce the same string the join produced."""
        engine = get_runtime().workflow
        user_id = store.create_user(
            email=f"blk_{uuid.uuid4().hex[:8]}@example.com"
        ).id
        snippets = ["alpha said four hundred", "beta said eight hundred"]

        content, kept, _ranges = self._ranges(
            engine, monkeypatch, snippets, user_id
        )

        assert content.endswith("\n\nContext: " + " | ".join(kept))

    def test_the_positions_are_married_to_what_each_snippet_is(self):
        """The two halves meet once: the builder measured where, the
        registration said what."""
        registry = SourceRegistry()
        source = registry.register_source(
            kind="file", title="a.md", locator="/files/a.md"
        )
        evidence = registry.add_evidence(source.source_id, text="alpha")
        ground = binding(source.source_id, evidence.evidence_id)
        messages = [{"role": "system", "content": "prelude\n\nContext: alpha"}]
        engine = get_runtime().workflow

        grounded = engine._initial_grounding(
            messages, [None, ground], [(0, 7), (20, 25)]
        )

        assert len(grounded) == 1
        record = grounded[0]
        assert record.message_index == 0
        assert record.text == messages[0]["content"]
        assert [(s.start, s.end, s.source_id) for s in record.spans] == [
            (20, 25, source.source_id)
        ]

    def test_a_disagreement_loses_the_citations_and_not_the_turn(self):
        """The two lists describe the same snippets and are built two
        statements apart, so a mismatch is a programming error - but refusing
        here would cost the user an answer, and under-offering costs a
        marker."""
        engine = get_runtime().workflow
        messages = [{"role": "system", "content": "prelude"}]

        registry = SourceRegistry()
        source = registry.register_source(
            kind="file", title="a.md", locator="/files/a.md"
        )
        evidence = registry.add_evidence(source.source_id, text="alpha")
        ground = binding(source.source_id, evidence.evidence_id)

        # Two relations and one position: zipping would silently describe the
        # first and drop the second, so the whole record is refused.
        assert engine._initial_grounding(
            messages, [ground, ground], [(0, 3)]
        ) == ()
        # One relation and two positions: the same disagreement the other way.
        assert engine._initial_grounding(
            messages, [ground], [(0, 3), (4, 7)]
        ) == ()
        assert engine._initial_grounding(messages, [], []) == ()
        assert engine._initial_grounding([], [ground], [(0, 3)]) == ()


class TestCuttingBeforeAnAnswerThatIsBeingReplaced:
    """`without_trailing_answer` is the streamed final turn's cutoff.

    The worker asks the model for a final answer, throws the reply away and
    hands the conversation back without it, because the parent is about to
    write that answer itself. The parent records the reply anyway - it
    happened - so a reconstruction meant to replace it needs a different
    cutoff from the record.

    Structural, not textual: what is dropped is the shape the worker breaks
    on, never anything matching the answer's words.
    """

    @staticmethod
    def _answer(seq, content="drafted"):
        return ModelTurn(operation_seq=seq, content=content)

    @staticmethod
    def _asked_for_tools(seq):
        return ModelTurn(
            operation_seq=seq,
            content="",
            tool_calls=({"id": "c1", "name": "web_search", "arguments": "{}"},),
        )

    @staticmethod
    def _round(seq):
        return ToolRound(
            operation_seq=seq,
            results=(TrustedToolResult(
                operation_seq=seq, call_index=0, tool_name="web_search",
                tool_message_id="c1", text="found",
            ),),
        )

    def test_a_trailing_answer_is_dropped(self):
        transcript = TrustedTranscript()
        transcript.record(self._asked_for_tools(1))
        transcript.record(self._round(2))
        transcript.record(self._answer(3))

        cut = transcript.without_trailing_answer()

        assert [type(e).__name__ for e in cut.entries] == ["ModelTurn", "ToolRound"]

    def test_a_trailing_round_is_kept(self):
        """The worker ran out of rounds with an exchange outstanding. Cutting
        by position rather than by shape would drop the tool result the
        answer is supposed to be written from."""
        transcript = TrustedTranscript()
        transcript.record(self._asked_for_tools(1))
        transcript.record(self._round(2))

        cut = transcript.without_trailing_answer()

        assert [type(e).__name__ for e in cut.entries] == ["ModelTurn", "ToolRound"]

    def test_a_trailing_turn_that_asked_for_tools_is_kept(self):
        """Not every trailing `ModelTurn` is a discarded answer. One carrying
        tool calls is the assistant message a tool message answers, and
        dropping it leaves that reply attached to nothing."""
        transcript = TrustedTranscript()
        transcript.record(self._round(1))
        transcript.record(self._asked_for_tools(2))

        cut = transcript.without_trailing_answer()

        assert [type(e).__name__ for e in cut.entries] == ["ToolRound", "ModelTurn"]
        assert cut.entries[-1].tool_calls

    def test_an_empty_transcript_survives_the_cut(self):
        assert TrustedTranscript().without_trailing_answer().entries == []

    def test_only_the_last_answer_goes(self):
        """Two answers in a row is not a shape the loop produces, and the
        rule is still 'the last one': anything else would be searching the
        record for drafts rather than cutting at a known point."""
        transcript = TrustedTranscript()
        transcript.record(self._answer(1, "first"))
        transcript.record(self._answer(2, "second"))

        cut = transcript.without_trailing_answer()

        assert [e.content for e in cut.entries] == ["first"]

    def test_the_record_itself_is_untouched(self):
        """A view, not an edit. That the model produced a draft is a fact
        about the turn, and a reader taking a different cutoff must not
        narrow what everyone else sees."""
        transcript = TrustedTranscript()
        transcript.record(self._asked_for_tools(1))
        transcript.record(self._answer(2))

        cut = transcript.without_trailing_answer()
        cut.record(self._answer(3, "added to the view"))

        assert [type(e).__name__ for e in transcript.entries] == [
            "ModelTurn", "ModelTurn"
        ]
        assert transcript.entries[-1].content == "drafted"
