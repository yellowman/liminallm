"""The native Gemini adapter keeps the provider's own conversation.

`generateContent` is stateless: everything the model is to know goes in every
request. What the provider wants back is what it produced - the selected
candidate's parts, in order, with the thought signatures riding on them,
including the ones on a text part or on a part whose text is empty. The
adapter keeps that whole and replays it whole. It reads none of it, and on a
turn the parent continues natively the placeholder it supplies for a history
built elsewhere never enters the conversation. Measured on the live wire with
3.x models: a candidate replayed whole with our functionResponse is accepted
with or without the provider's call id on the response, thought parts
included; an unsigned call is a 400; the placeholder is accepted, as
documented, at the cost of the reasoning it stands in for.
"""

from __future__ import annotations

import json

import httpx
import pytest

from liminallm.service import gemini_backend as gb
from liminallm.service.continuation import (
    CHAT_STRUCTURED_V1,
    GEMINI_NATIVE_V1,
    OPENAI_RESPONSES_NATIVE_V1,
    ContinuationMismatch,
    ProviderContinuation,
)
from liminallm.service.gemini_backend import GeminiBackend
from tests.test_gemini_native import _backend

TOOLS = [{"type": "function", "function": {
    "name": "web_search", "description": "search", "parameters": {"type": "object"}}}]

OPENING = [
    {"role": "system", "content": "You are terse."},
    {"role": "user", "content": "find x"},
]

#: A candidate as a thinking model returns one: a signed empty text part, a
#: thought, a signed text, a signed call. The order is the provider's.
PARTS = [
    {"text": "", "thoughtSignature": "sig-empty"},
    {"text": "considering", "thought": True, "thoughtSignature": "sig-thought"},
    {"text": "looking", "thoughtSignature": "sig-text"},
    {"functionCall": {"name": "web_search", "args": {"q": "x"}},
     "thoughtSignature": "sig-call"},
]


def _reply(parts):
    return {
        "candidates": [{"content": {"role": "model", "parts": parts},
                        "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 8, "candidatesTokenCount": 4,
                          "totalTokenCount": 12},
    }


def _scripted(*replies):
    bodies = []
    queue = list(replies)

    def handler(request):
        bodies.append(json.loads(request.read()))
        return httpx.Response(200, json=queue.pop(0))

    return _backend(handler), bodies


def _accepted(out, *, through=1):
    c = out["continuation"]
    return ProviderContinuation(
        strategy=c["strategy"], provider=c["provider"], transport=c["transport"],
        model=c["model"], through_operation_seq=through, payload=c["payload"],
    )


class TestTheCandidateIsTheWholeSelectedContent:
    def test_the_opening_and_the_complete_candidate_are_kept_in_order(self):
        backend, bodies = _scripted(_reply(PARTS))

        out = backend.generate_with_tools(OPENING, TOOLS, [])

        candidate = out["continuation"]
        assert candidate["strategy"] == GEMINI_NATIVE_V1
        assert candidate["provider"] == "gemini"
        assert candidate["transport"] == "generateContent"
        assert candidate["model"] == "gemini-2.5-flash"
        payload = candidate["payload"]
        # The request as it went, then the candidate as it came.
        assert payload["systemInstruction"] == bodies[0]["systemInstruction"]
        assert payload["contents"][:-1] == bodies[0]["contents"]
        assert payload["contents"][-1] == {"role": "model", "parts": PARTS}
        # And the turn the loop reads is what it was before any of this: the
        # text parts joined, the calls with their signatures riding along.
        assert out["content"] == "consideringlooking"
        assert out["tool_calls"][0]["name"] == "web_search"
        assert "thought_signature" not in out["tool_calls"][0]

    def test_the_public_reply_carries_no_signature(self):
        """What crosses to the worker is the turn: text and calls. The
        signatures are in the candidate and nowhere else."""
        backend, _bodies = _scripted(_reply(PARTS))

        out = backend.generate_with_tools(OPENING, TOOLS, [])

        public = json.dumps({k: v for k, v in out.items() if k != "continuation"})
        assert "sig-" not in public
        assert "sig-call" in json.dumps(out["continuation"])

    def test_a_reply_without_candidate_content_adds_nothing_and_invents_nothing(self):
        """Both shapes an empty reply takes: no candidate at all, and a
        candidate that stopped before it produced any content."""
        for empty in (
            {"promptFeedback": {"blockReason": "SAFETY"}, "usageMetadata": {}},
            {"candidates": [{"finishReason": "SAFETY"}], "usageMetadata": {}},
            {"candidates": [{"content": {"role": "model"}, "finishReason": "STOP"}],
             "usageMetadata": {}},
        ):
            backend, bodies = _scripted(empty)

            out = backend.generate_with_tools(OPENING, TOOLS, [])

            assert out["content"] == "" and out["tool_calls"] == []
            assert out["continuation"]["payload"]["contents"] == bodies[0]["contents"]

    def test_the_keeper_reads_nothing_and_drops_nothing(self):
        future = {"inlineData": {"mimeType": "x/y", "data": "AAAA"},
                  "thoughtSignature": "sig-future", "someNewField": None}
        kept = gb.selected_content(_reply([future] + PARTS))

        assert kept["parts"] == [future] + PARTS
        assert kept["role"] == "model"


class TestTheNextCallReplaysItWhole:
    TAIL = [{"role": "tool", "tool_call_id": "gemini-call-3-web_search",
             "name": "web_search", "content": "result text"}]

    def test_the_accepted_conversation_goes_first_signatures_and_all(self):
        backend, bodies = _scripted(_reply(PARTS), _reply([{"text": "done"}]))
        first = backend.generate_with_tools(OPENING, TOOLS, [])
        accepted = _accepted(first)

        second = backend.generate_with_tools(self.TAIL, TOOLS, [], continuation=accepted)

        sent = bodies[1]
        assert sent["contents"][:-1] == accepted.payload["contents"]
        assert sent["contents"][-1] == {"role": "user", "parts": [{"functionResponse": {
            "name": "web_search", "response": {"output": "result text"}}}]}
        # Every signature the provider sent, byte for byte, where it sent it.
        model_turn = sent["contents"][-2]
        assert [p.get("thoughtSignature") for p in model_turn["parts"]] == [
            "sig-empty", "sig-thought", "sig-text", "sig-call",
        ]
        assert gb.THOUGHT_SIGNATURE_PLACEHOLDER not in json.dumps(sent)
        # The tail carries no system message; the request still has the one
        # the conversation opened with.
        assert sent["systemInstruction"] == bodies[0]["systemInstruction"]
        # And the new candidate is the whole of that plus this reply.
        items = second["continuation"]["payload"]["contents"]
        assert items[:-1] == sent["contents"]
        assert items[-1] == {"role": "model", "parts": [{"text": "done"}]}

    def test_a_continuation_written_by_another_strategy_is_not_replayed(self):
        for strategy, transport, provider in (
            (OPENAI_RESPONSES_NATIVE_V1, "responses", "openai"),
            (CHAT_STRUCTURED_V1, "chat", "openai"),
        ):
            foreign = ProviderContinuation(
                strategy=strategy, provider=provider, transport=transport,
                model="m", through_operation_seq=1,
                payload={"contents": [{"role": "user", "parts": [{"text": "elsewhere"}]}],
                         "systemInstruction": {"parts": [{"text": "not ours"}]}},
            )
            backend, bodies = _scripted(_reply([{"text": "fresh"}]))

            out = backend.generate_with_tools(OPENING, TOOLS, [], continuation=foreign)

            assert "elsewhere" not in json.dumps(bodies[0])
            assert bodies[0]["systemInstruction"]["parts"][0]["text"] == "You are terse."
            assert out["continuation"]["strategy"] == GEMINI_NATIVE_V1

    def test_an_accepted_conversation_for_another_model_is_refused_before_the_call(self):
        """Signatures are one model's. A replacement process serving another
        model refuses the record before anything is sent."""
        backend, bodies = _scripted(_reply(PARTS))
        accepted = _accepted(backend.generate_with_tools(OPENING, TOOLS, []))
        other_bodies = []

        def handler(request):
            other_bodies.append(json.loads(request.read()))
            return httpx.Response(200, json=_reply([{"text": "done"}]))

        other = GeminiBackend("gemini-3-flash-preview", api_key="g-key",
                              transport=httpx.MockTransport(handler))
        with pytest.raises(ContinuationMismatch, match="gemini-3-flash-preview"):
            other.generate_with_tools(self.TAIL, TOOLS, [], continuation=accepted)

        assert other_bodies == []

    def test_a_signature_is_never_supplied_by_the_adapter_in_its_own_conversation(self):
        """The placeholder exists for a history built elsewhere. A
        conversation this provider produced carries its own signatures, and
        a part it sent unsigned goes back unsigned."""
        unsigned = [{"text": "plain"}, {"functionCall": {"name": "web_search", "args": {}}}]
        backend, bodies = _scripted(_reply(unsigned), _reply([{"text": "done"}]))
        first = backend.generate_with_tools(OPENING, TOOLS, [])

        backend.generate_with_tools(self.TAIL, TOOLS, [], continuation=_accepted(first))

        assert bodies[1]["contents"][-2] == {"role": "model", "parts": unsigned}
        assert "thoughtSignature" not in json.dumps(bodies[1])
