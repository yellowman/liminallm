"""No plain form of the citation namespace crosses the pipe to the worker.

Once the model has been offered a handle it can put that handle anywhere in
its reply, and every part of that reply goes to a worker running model-chosen
control flow over attacker-controlled bytes. So the namespace does not occur
anywhere in the serialized public reply, in the marker or on its own, in
whatever case it was written. What the model actually said is kept
parent-side, where a replay can restore it.

Stated at its real width, because the width matters for what comes next: this
is not a secrecy proof. A model that has seen the nonce can encode it into
text no scrubber recognises. What makes such a disclosure worthless is the
canonical-response transfer rule - citations come from what the model said,
and only when the worker returns it unchanged - not this scrub. The scrub
keeps the namespace out of ordinary worker state, and it is narrow: text
carrying no authority of this turn crosses exactly as the model wrote it.
"""

from __future__ import annotations

import json
import random
import string
import uuid

import pytest

from liminallm.service.broker import CapabilityBroker, InvocationContext
from liminallm.service.citations import (
    assert_scrubbed,
    build_citation_table,
    mint_nonce,
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


class TestOnlyThisTurnsNamespaceIsTakenOut:
    """`strip_citations` is the reader's cleanup and removes every closed
    marker. Using it on the wire would rewrite text carrying no authority."""

    OTHER = "ABCDEFGH"

    def test_another_turns_marker_is_left_exactly_as_written(self):
        text = f"quote [cite:{self.OTHER}-1] exactly"
        assert scrub_namespace(text, NONCE) == text

    def test_another_turns_marker_inside_tool_arguments_survives(self):
        """These arguments are parsed by the worker and sent back to be
        executed. A model asked to search for a literal citation-shaped
        token gets that search run, not a rewritten one."""
        reply = {
            "tool_calls": [
                {
                    "name": "web_search",
                    "arguments": '{"q":"[cite:%s-1]"}' % self.OTHER,
                }
            ]
        }
        assert scrub_namespace(reply, NONCE) == reply

    def test_a_turn_that_was_offered_no_handles_crosses_unchanged(self):
        """Until S6 shows the model a handle, nothing in a reply belongs to
        this namespace, so the scrub must be a no-op on every real reply."""
        reply = {
            "content": "See [cite:OLDTURN-1] and src_1.",
            "tool_calls": [],
            "usage": {},
        }
        assert scrub_namespace(reply, NONCE) == reply


class TestTheNamespaceGoesInWhateverCaseItWasWritten:
    """The alphabet is uppercase, so the lowercase nonce is not an encoding
    of the namespace. It is the namespace, normalized losslessly."""

    def test_a_lowercase_handle_does_not_survive(self):
        scrubbed = scrub_namespace(f"x {NONCE.lower()}-1 y", NONCE)
        assert NONCE.lower() not in scrubbed.lower()

    def test_a_mixed_case_marker_does_not_survive(self):
        scrubbed = scrub_namespace(f"a [CiTe:{NONCE.title()}-2] b", NONCE)
        assert NONCE.lower() not in scrubbed.lower()

    def test_the_assertion_case_folds_too(self):
        with pytest.raises(ProvenanceError):
            assert_scrubbed({"a": f"{NONCE.lower()}-1"}, NONCE)


class TestRemovalDoesNotSpliceTheNamespaceBackTogether:
    def test_a_neighbour_splice_is_removed_as_well(self):
        """Cutting a substring joins what was on either side of it, and for
        any nonce there is a string whose halves rejoin into one: here the
        first pass leaves the nonce behind, so a single pass is not enough."""
        spliced = NONCE[:4] + NONCE + NONCE[4:]
        assert NONCE not in scrub_namespace(spliced, NONCE)


class TestAKeyCollisionIsRefusedRatherThanResolved:
    def test_two_keys_that_scrub_to_one_name_are_refused(self):
        """Dropping one value quietly is the wrong half of the choice for a
        primitive whose job is to protect a field nobody has added yet."""
        with pytest.raises(ProvenanceError):
            scrub_namespace({f"q{NONCE}": "first", "q": "second"}, NONCE)

    def test_keys_that_stay_distinct_are_kept(self):
        scrubbed = scrub_namespace({f"a{NONCE}": 1, "b": 2}, NONCE)
        assert scrubbed == {"a": 1, "b": 2}


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

    def test_json_escaping_does_not_manufacture_a_namespace(self):
        """`\\uXXXX` escapes are hex. A nonce drawn only from hex characters
        can be spliced out of one and the text either side of it, so the
        assertion would refuse a payload the scrubber had correctly cleaned."""
        hexish = "23456789"
        text = chr(0x2345) + "6789"
        assert hexish not in text
        assert_scrubbed(scrub_namespace(text, hexish), hexish)


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


class TestTheScrubberAndTheAssertionAgree:
    """What the scrubber cleans, the assertion passes. This is what makes the
    assertion at the broker boundary a guard on the next field rather than a
    second opinion on this one - and both halves have been rewritten, so it
    is measured over generated input rather than argued."""

    def test_no_generated_reply_is_cleaned_and_then_refused(self):
        random.seed(20260903)
        # Splices, case variants, hex-only nonces and non-ASCII neighbours -
        # the four shapes that separate the two implementations.
        pool = string.printable + chr(0x2345) + chr(0x1234)
        for index in range(2000):
            nonce = (
                mint_nonce()
                if index % 2
                else "".join(random.choice("23456789ABCDEF") for _ in range(8))
            )
            filler = [
                "".join(random.choice(pool) for _ in range(random.randint(0, 6)))
                for _ in range(3)
            ]
            if index % 5 == 0:
                text = nonce[:4] + nonce + nonce[4:]
            else:
                text = filler[0] + nonce + filler[1] + nonce.lower() + filler[2]
            try:
                assert_scrubbed(scrub_namespace(text, nonce), nonce)
            except ProvenanceError:  # pragma: no cover - the failure report
                raise AssertionError(f"cleaned then refused: {nonce=} {text=}")


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
