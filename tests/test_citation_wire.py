"""No citation authority crosses the pipe to the worker.

Once the model has been offered a handle it can put that handle anywhere in
its reply, and every part of that reply goes to a worker running model-chosen
control flow over attacker-controlled bytes. A worker holding one valid
handle can attach a real citation to text it wrote itself; one holding the
bare nonce can derive every handle the turn will ever issue.

So the wire rule is stronger than removing well-formed markers: the namespace
itself must not occur anywhere in the serialized public reply. What the model
actually said is kept parent-side, where a replay can restore it.
"""

from __future__ import annotations

import json
import uuid

import pytest

from liminallm.service.broker import CapabilityBroker, InvocationContext
from liminallm.service.citations import (
    assert_scrubbed,
    build_citation_table,
    scrub_namespace,
)
from liminallm.service.invocation import InvocationRegistry
from liminallm.service.provenance import ProvenanceError, SourceRegistry, binding
from liminallm.service.runtime import get_runtime
from liminallm.service.wire import WireError, send_frame

NONCE = "K7Q2ABCD"


class _Sneaky:
    """Not a JSON shape, and its `repr` names the namespace."""

    def __repr__(self) -> str:
        return f"holds {NONCE}-1"


class TestTheScrubberReachesEveryModelControlledString:
    """`content` is not the only one. An assistant message carries text, and
    a tool call's arguments carry whatever the model decided to search for."""

    def test_a_marker_in_the_prose_is_removed(self):
        assert scrub_namespace(f"400 hours [cite:{NONCE}-1].", NONCE) == "400 hours."

    def test_a_marker_nested_in_an_assistant_message_is_removed(self):
        reply = {"assistant_message": {"role": "a", "content": f"see [cite:{NONCE}-2]"}}
        assert scrub_namespace(reply, NONCE)["assistant_message"]["content"] == "see"

    def test_a_marker_inside_tool_call_arguments_is_removed(self):
        """The worker parses these and sends them back to be executed, so a
        handle here is a handle it holds."""
        reply = {
            "tool_calls": [
                {"name": "web_search", "arguments": f'{{"q":"foo [cite:{NONCE}-1]"}}'}
            ]
        }
        scrubbed = scrub_namespace(reply, NONCE)
        assert NONCE not in scrubbed["tool_calls"][0]["arguments"]

    def test_a_marker_in_a_mapping_key_is_removed(self):
        """A key is a string too. No adapter puts model-chosen text in one
        today - they all re-serialize a tool call's arguments - but that is
        a property of the adapters, not of the reply being scrubbed."""
        scrubbed = scrub_namespace({f"q [cite:{NONCE}-1]": "foo"}, NONCE)
        assert list(scrubbed) == ["q"]

    def test_a_bare_handle_without_the_marker_does_not_survive(self):
        """The brackets are how a citation is written, not what makes the
        handle valuable to the untrusted side."""
        assert NONCE not in scrub_namespace(f"the handle is {NONCE}-1", NONCE)

    def test_the_bare_namespace_does_not_survive(self):
        """Given the nonce, every handle this turn will issue is derivable."""
        assert NONCE not in scrub_namespace(f"the namespace is {NONCE}", NONCE)

    def test_nothing_is_shared_with_the_original(self):
        """The canonical record must not change when something edits what
        crossed."""
        reply = {"assistant_message": {"content": "plain"}, "tool_calls": [{"a": 1}]}
        scrubbed = scrub_namespace(reply, NONCE)
        scrubbed["assistant_message"]["content"] = "edited"
        scrubbed["tool_calls"][0]["a"] = 2
        assert reply["assistant_message"]["content"] == "plain"
        assert reply["tool_calls"][0]["a"] == 1

    def test_a_reply_with_no_handles_is_unchanged(self):
        reply = {"content": "400 hours.", "usage": {"tokens": 12}}
        assert scrub_namespace(reply, NONCE) == reply


class TestTheAssertionReadsTheWholeReply:
    def test_a_namespace_anywhere_is_refused(self):
        with pytest.raises(ProvenanceError):
            assert_scrubbed({"a": {"b": [f"{NONCE}-1"]}}, NONCE)

    def test_a_field_the_scrubber_never_heard_of_is_still_checked(self):
        """The point of checking the serialized whole: a model-controlled
        field added later is model-controlled the moment it exists."""
        with pytest.raises(ProvenanceError):
            assert_scrubbed({"some_new_field": f"{NONCE}-3"}, NONCE)

    def test_a_clean_reply_passes(self):
        assert_scrubbed({"content": "400 hours.", "usage": {}}, NONCE)


class TestTheCapabilityReplyCarriesNoNamespace:
    """The whole serialized reply, not the keys this test happens to know."""

    @staticmethod
    def _turn(monkeypatch, response):
        engine = get_runtime().workflow
        monkeypatch.setattr(
            engine.llm, "generate_with_tools",
            lambda *a, **k: response, raising=False,
        )
        registry = SourceRegistry()
        source = registry.register_source(
            kind="file", title="manual.md", locator="/files/manual.md"
        )
        evidence = registry.add_evidence(source.source_id, text="400 hours")
        invocation = InvocationRegistry().open(
            uuid.uuid4().hex, tool="agent.files_v1", user_id="u", tenant_id=None
        )
        invocation.extend_citations(
            registry, [binding(source.source_id, evidence.evidence_id)]
        )
        context = InvocationContext(user_id="u", source_registry=registry)
        return engine, invocation, context, CapabilityBroker(engine, context)

    def _ask(self, broker, invocation, seq=1):
        return broker._answer(
            invocation,
            {
                "capability": "llm.generate_with_tools",
                "operation_seq": seq,
                "payload": {"messages": [], "tools": []},
            },
        )

    def test_no_handle_and_no_nonce_reach_the_worker(self, monkeypatch):
        engine, invocation, context, broker = self._turn(
            monkeypatch, {"content": "", "tool_calls": [], "assistant_message": None}
        )
        handle = invocation.citations.handle_for("src_1")
        # Now that the handle is known, answer with it in every model-controlled
        # place at once.
        monkeypatch.setattr(
            engine.llm, "generate_with_tools",
            lambda *a, **k: {
                "content": f"400 hours [cite:{handle}].",
                "tool_calls": [
                    {"name": "web_search", "arguments": f'{{"q":"[cite:{handle}]"}}'}
                ],
                "assistant_message": {"role": "assistant", "content": f"[cite:{handle}]"},
                "usage": {},
            },
            raising=False,
        )
        reply = self._ask(broker, invocation)

        assert reply["ok"], reply
        serialized = json.dumps(reply, default=repr)
        assert invocation.citations.nonce not in serialized, serialized[:300]
        for issued in invocation.citations.by_handle:
            assert issued not in serialized

    def test_the_parent_keeps_what_the_model_actually_said(self, monkeypatch):
        engine, invocation, context, broker = self._turn(
            monkeypatch, {"content": "", "tool_calls": [], "assistant_message": None}
        )
        handle = invocation.citations.handle_for("src_1")
        monkeypatch.setattr(
            engine.llm, "generate_with_tools",
            lambda *a, **k: {
                "content": f"400 hours [cite:{handle}].",
                "tool_calls": [],
                "assistant_message": None,
                "usage": {},
            },
            raising=False,
        )
        self._ask(broker, invocation)

        canonical = context.canonical_model_response
        assert canonical is not None, "the parent kept no canonical response"
        assert handle in canonical["content"], canonical

    def test_a_replay_restores_the_canonical_response(self, monkeypatch):
        """The handler does not run on a replay, so this is the only place a
        replacement attempt can recover the citations in an answer it is
        otherwise handed intact."""
        engine, invocation, first_context, broker_a = self._turn(
            monkeypatch, {"content": "", "tool_calls": [], "assistant_message": None}
        )
        handle = invocation.citations.handle_for("src_1")
        monkeypatch.setattr(
            engine.llm, "generate_with_tools",
            lambda *a, **k: {
                "content": f"400 hours [cite:{handle}].",
                "tool_calls": [],
                "assistant_message": None,
                "usage": {},
            },
            raising=False,
        )
        reply_a = self._ask(broker_a, invocation)

        second_context = InvocationContext(
            user_id="u", source_registry=first_context.source_registry
        )
        broker_b = CapabilityBroker(engine, second_context)
        ran = {"model": False}

        def _tripwire(*args, **kwargs):
            ran["model"] = True
            return {"content": "different", "tool_calls": [], "usage": {}}

        monkeypatch.setattr(
            engine.llm, "generate_with_tools", _tripwire, raising=False
        )
        reply_b = self._ask(broker_b, invocation)

        assert reply_b.get("replayed"), "the fixture did not exercise a replay"
        assert not ran["model"], "the model ran again on replay"
        assert reply_b["result"] == reply_a["result"]
        assert second_context.canonical_model_response == (
            first_context.canonical_model_response
        )
        assert handle in second_context.canonical_model_response["content"]
        # And the replayed public result is still scrubbed.
        assert invocation.citations.nonce not in json.dumps(
            reply_b["result"], default=repr
        )

    def test_the_canonical_record_is_not_shared_with_the_ledger(self, monkeypatch):
        """Every later attempt reads this record. One that edited it in place
        would change what the next attempt is told the model said."""
        engine, invocation, context, broker = self._turn(
            monkeypatch,
            {
                "content": "400 hours.",
                "tool_calls": [],
                "assistant_message": None,
                "usage": {},
            },
        )
        self._ask(broker, invocation)
        context.canonical_model_response["content"] = "edited"

        second = InvocationContext(user_id="u", source_registry=SourceRegistry())
        CapabilityBroker(engine, second)._answer(
            invocation,
            {
                "capability": "llm.generate_with_tools",
                "operation_seq": 1,
                "payload": {"messages": [], "tools": []},
            },
        )
        assert second.canonical_model_response["content"] == "400 hours."


class TestWhatTheScrubberCannotReachDoesNotCross:
    """The scrubber walks JSON shapes and returns anything else untouched, so
    an object whose `repr` names the namespace passes through it. That is
    safe only because `wire.send_frame` refuses such an object outright
    instead of sending its repr - the behaviour lives in `wire`, and the
    citation boundary is one of the things relying on it."""

    def test_an_object_the_scrubber_cannot_reach_survives_the_scrub(self):
        assert NONCE in repr(scrub_namespace(_Sneaky(), NONCE))

    def test_but_the_wire_refuses_to_send_it(self):
        class _Conn:
            def sendall(self, payload):  # pragma: no cover - never reached
                raise AssertionError("the frame was sent")

        with pytest.raises(WireError):
            send_frame(_Conn(), {"result": {"assistant_message": _Sneaky()}})


class TestTheScrubDoesNotDependOnHavingATable:
    def test_a_turn_that_offered_nothing_still_scrubs_its_namespace(self):
        """Every invocation mints a namespace whether or not it issues a
        handle, so the wire rule holds before any offer exists."""
        registry = SourceRegistry()
        table = build_citation_table(registry, [], nonce=NONCE)
        assert not table
        assert NONCE not in scrub_namespace(f"stray {NONCE}", table.nonce)
