"""A provider's own continuation, kept beside the readable transcript.

The trusted transcript is what the worker actually continued from, and it is
chat-shaped on purpose - every provider can be reconstructed from it. What it
cannot hold is the state a provider keeps for itself between turns: the
encrypted reasoning a Responses model wants back, the signed parts a Gemini
candidate came with. Rebuilding a turn from its visible text throws that away,
and the model does its thinking over.

So there is a second record. It is opaque to everything but the adapter that
wrote it, it advances only when the parent accepts a complete model turn, and
every backend says which kind it keeps - including the ones that keep none.
"""

from __future__ import annotations

import json

import pytest

from liminallm.config import ModelBackend
from liminallm.service.broker import CapabilityBroker, InvocationContext
from liminallm.service.continuation import (
    CHAT_STRUCTURED_V1,
    GEMINI_NATIVE_V1,
    NATIVE_STRATEGIES,
    OPENAI_RESPONSES_NATIVE_V1,
    STRATEGIES,
    TRANSCRIPT_V1,
    ProviderContinuation,
    declared_strategy,
)

#: What a provider hands back and expects returned unread. The characters are
#: the point: base64 punctuation, a padding tail, and one code point outside
#: ASCII, so a codec that re-encoded, stripped, or went through `str(bytes)`
#: changes it.
SENTINEL = "gAAAAB+/x9Q==étape"


def _state(**over):
    base = dict(
        strategy=OPENAI_RESPONSES_NATIVE_V1,
        provider="openai",
        transport="responses",
        model="gpt-6-astra",
        through_operation_seq=3,
        payload={
            "items": [
                {"type": "reasoning", "id": "rs_1", "encrypted_content": SENTINEL,
                 "phase": None},
                {"type": "message", "id": "msg_1", "role": "assistant",
                 "content": [{"type": "output_text", "text": "found it"}]},
            ]
        },
    )
    base.update(over)
    return ProviderContinuation(**base)


class TestEveryBackendDeclaresItsContinuation:
    """No path is allowed to be silent about what it keeps."""

    def test_every_backend_mode_resolves_to_one_of_the_four(self):
        """Derived from the enum, so a new mode without a declaration fails
        here rather than inheriting whatever the fallback happens to be."""
        for mode in ModelBackend:
            assert declared_strategy(mode.value) in STRATEGIES, mode.value

    def test_only_openai_and_native_gemini_declare_native_state(self):
        """A native strategy is a claim that the adapter preserves the
        provider's own continuation intact. Two adapters can make it."""
        native = {
            mode.value for mode in ModelBackend
            if declared_strategy(mode.value) in NATIVE_STRATEGIES
        }
        assert native == {"openai", "gemini_native"}, native
        assert declared_strategy("openai") == OPENAI_RESPONSES_NATIVE_V1
        assert declared_strategy("gemini_native") == GEMINI_NATIVE_V1

    def test_local_and_stub_keep_no_opaque_state(self):
        for mode in ("stub", "local_lora", "local_gpu_lora"):
            assert declared_strategy(mode) == TRANSCRIPT_V1, mode

    def test_a_compatible_provider_is_never_promoted(self):
        """Answering `/responses` is a transport fact, not an entitlement. A
        gateway that speaks OpenAI's wire does not get OpenAI's encrypted
        reasoning rules by resembling it."""
        for mode in ("together", "xai", "deepseek", "anthropic", "azure",
                     "azure_openai", "zhipu", "gemini", "fireworks", "meta"):
            assert declared_strategy(mode) == CHAT_STRUCTURED_V1, mode

    def test_an_unregistered_mode_is_refused_rather_than_defaulted(self):
        """`_infer_provider` answers "openai" for any mode it does not know.
        That fallback must not reach here: an unknown mode routed through it
        would declare native continuation for a provider nobody looked at."""
        with pytest.raises(ValueError):
            declared_strategy("acme_custom")
        with pytest.raises(ValueError):
            declared_strategy("")


class TestTheStateIsOpaqueAndExact:
    """The adapter that wrote the payload is the only thing that reads it.
    Everything between - the ledger, a replay, a restore - hands it back
    unchanged."""

    def test_a_provider_value_round_trips_string_for_string(self):
        state = _state()

        restored = ProviderContinuation.from_dict(state.as_dict())

        assert restored == state
        assert restored.payload["items"][0]["encrypted_content"] == SENTINEL
        # Nulls are values a provider set, not absences to tidy away.
        assert restored.payload["items"][0]["phase"] is None
        assert "phase" in restored.payload["items"][0]

    def test_the_codec_adds_no_wrapper_and_no_encoding(self):
        """Zero encryption, zero encoding of our own. The exported payload is
        the payload, JSON-native, with the provider's value sitting in it as
        the string it arrived as."""
        exported = _state().as_dict()

        assert exported["payload"]["items"][0]["encrypted_content"] == SENTINEL
        assert exported["strategy"] == OPENAI_RESPONSES_NATIVE_V1
        assert exported["through_operation_seq"] == 3
        # Serializable as it stands - no bytes, no objects, nothing to encode.
        assert json.loads(json.dumps(exported)) == exported

    def test_a_strategy_outside_the_four_is_refused(self):
        with pytest.raises(ValueError):
            _state(strategy="provider_native")
        with pytest.raises(ValueError):
            ProviderContinuation.from_dict({**_state().as_dict(), "strategy": ""})

    def test_the_payload_is_not_shared_with_whoever_supplied_or_read_it(self):
        """The ledger keeps this record for every later attempt. A caller
        editing its own dict afterwards, or a reader editing the export, must
        not be editing what the next attempt is restored to."""
        supplied = {"items": [{"type": "reasoning", "encrypted_content": SENTINEL}]}
        state = _state(payload=supplied)
        supplied["items"].append({"type": "message"})
        assert len(state.payload["items"]) == 1

        exported = state.as_dict()
        exported["payload"]["items"].clear()
        assert len(state.payload["items"]) == 1


class TestTheContinuationLivesOnTheInvocationContext:
    """Parent-owned, per invocation. Not on the backend instance, which is
    shared by every conversation the process serves, and not in the plan,
    which crosses to the worker."""

    def test_a_fresh_context_holds_none(self):
        assert InvocationContext(user_id="u").continuation is None

    def test_applying_parent_state_restores_it_as_its_own_copy(self):
        state = _state()
        context = InvocationContext(user_id="u")
        broker = CapabilityBroker(None, context, worker_tool="agent.files_v1")
        record = {"continuation": state.as_dict()}

        broker._apply_parent_state(record)

        assert context.continuation == state
        # The ledger's copy stays what it was, whatever the context does next.
        record["continuation"]["payload"]["items"].clear()
        assert len(context.continuation.payload["items"]) == 2

    def test_parent_state_without_one_leaves_the_current_one_alone(self):
        """A read's parent state carries bindings, not continuation. Folding
        it in must not reset the state a model turn already advanced."""
        state = _state()
        context = InvocationContext(user_id="u", continuation=state)
        broker = CapabilityBroker(None, context, worker_tool="agent.files_v1")

        broker._apply_parent_state({"provenance_bindings": []})

        assert context.continuation == state
