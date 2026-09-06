"""What a cited answer leaves behind after the turn that produced it ends.

Everything the citation layer works in is turn-scoped. The nonce is minted per
turn, the handle is only how the model named a source, and `src_3` restarts at
`src_1` on the next turn - so the transient pair the workflow produces,
`validated_citations` beside `provenance_snapshot`, cannot be written down as
it stands. This is the boundary that decides what can: where in the stored
answer each citation sits, and what the source was.

The asymmetry is the whole of it. The validated list is authority and the
snapshot is a lookup table, because the registry holds everything the turn
consulted and the answer rests only on what it cited.
"""

from __future__ import annotations

import json
import uuid

import pytest

from liminallm.api import chat_turn
from liminallm.service.auth import AuthContext
from liminallm.service.citations import durable_citations
from liminallm.service.runtime import get_runtime

#: One source, as `SourceRegistry.snapshot()` exports it.
SNAPSHOT = {
    "sources": {
        "src_1": {
            "source_id": "src_1",
            "kind": "file",
            "title": "manual.md",
            "origin_id": None,
            "locator": "/files/manual.md",
            "metadata": {},
        },
        # Consulted and never cited. Present in every case below, because the
        # registry is a superset by construction and the rule is that being in
        # it is not eligibility.
        "src_9": {
            "source_id": "src_9",
            "kind": "file",
            "title": "unrelated.md",
            "origin_id": None,
            "locator": "/files/unrelated.md",
            "metadata": {},
        },
    },
    "evidence": [],
}

ANSWER = "Beta"

#: What the answer's own citation looks like coming out of the turn: the
#: marker sat directly after `Beta`, so the anchor is its length.
CITED = [{
    "source_id": "src_1",
    "canonical_start": 5,
    "canonical_end": 22,
    "public_offset": 4,
}]


def _turn(store, content=ANSWER):
    """A real user, a real conversation, and a turn ready to be finished."""
    user = store.create_user(email=f"proj_{uuid.uuid4().hex[:8]}@example.com")
    conversation = store.create_conversation(
        title="citation projection", user_id=user.id
    )
    principal = AuthContext(user_id=user.id, role="user", tenant_id=None)
    return chat_turn.Turn(
        principal=principal,
        conversation_id=conversation.id,
        context_id=None,
        workflow_id=None,
        user_content="how long",
        user_message=None,
        needs_title=False,
    )


async def _finish(store, orchestration, content=ANSWER):
    """Finish a turn and read the row back, not the object that was returned.

    Everything here is a claim about what was stored, so it is read from
    storage.
    """
    turn = _turn(store, content)
    message = await chat_turn.finish(get_runtime(), turn, orchestration)
    stored = store.get_message(message.id)
    assert stored is not None, "the assistant message was never persisted"
    return stored


def _segments(message):
    return list((message.content_struct or {}).get("segments") or [])


def _citations(message):
    return [s for s in _segments(message) if s.get("type") == "citation"]


class TestACitationOutlivesItsTurnAsAnAnchor:
    @pytest.mark.asyncio
    async def test_the_stored_offset_indexes_the_stored_answer(self, store):
        """The positive case, stated as the property a renderer needs.

        The citation is a zero-width anchor into `content` exactly as stored -
        not into the canonical text the model wrote, which contained a marker
        and is never persisted anywhere.
        """
        message = await _finish(store, {
            "content": ANSWER,
            "validated_citations": CITED,
            "provenance_snapshot": SNAPSHOT,
        })

        assert message.content == ANSWER
        cited = _citations(message)
        assert len(cited) == 1, _segments(message)
        assert cited[0]["start"] == cited[0]["end"] == 4
        assert message.content[: cited[0]["start"]] == "Beta"
        # And the source is named in a form that still means something after
        # the registry that minted `src_1` is gone.
        assert cited[0]["locator"] == "/files/manual.md"
        assert cited[0]["source_id"] == "/files/manual.md"
        assert cited[0]["meta"]["title"] == "manual.md"
        # The answer the anchors point into is in the struct too, and it is
        # the answer: a struct of anchors over absent or partial text would
        # place them in a string the renderer does not have.
        assert [
            s.get("text") for s in _segments(message) if s.get("type") == "text"
        ] == [ANSWER]

    @pytest.mark.asyncio
    async def test_nothing_turn_scoped_reaches_the_row(self, store):
        """A nonce, a handle and a `src_#` are all names for this turn only.

        Written down, the first two would put a namespace the model was shown
        into storage, and the third would read later as a stable document id
        when the next turn's `src_1` is a different document.
        """
        nonce = "K7Q2ABCD"
        message = await _finish(store, {
            "content": ANSWER,
            "validated_citations": CITED,
            "provenance_snapshot": SNAPSHOT,
        })

        row = json.dumps({
            "content": message.content,
            "content_struct": message.content_struct,
            "meta": message.meta,
        })
        assert nonce not in row
        assert "[cite:" not in row
        assert "src_1" not in row
        # Coordinates into the marker-bearing text are not coordinates into
        # anything that was stored, so they do not travel either.
        assert set(_citations(message)[0]) == {
            "type", "start", "end", "locator", "source_id", "meta"
        }

    @pytest.mark.asyncio
    async def test_a_turn_that_cited_nothing_stores_what_it_always_did(
        self, store
    ):
        """The gate is off in production, so this is every turn today."""
        message = await _finish(store, {"content": ANSWER, "usage": {}})

        assert message.content == ANSWER
        assert message.content_struct is None


class TestTheSnapshotResolvesNamesAndConfersNothing:
    @pytest.mark.asyncio
    async def test_a_consulted_source_nobody_cited_stays_out(self, store):
        """`src_9` is in the registry because the turn read it. The answer
        does not rest on it, and a durable citation is a claim that it does."""
        message = await _finish(store, {
            "content": ANSWER,
            "validated_citations": CITED,
            "provenance_snapshot": SNAPSHOT,
        })

        located = {item["locator"] for item in _citations(message)}
        assert located == {"/files/manual.md"}
        assert "/files/unrelated.md" not in json.dumps(message.content_struct)

    @pytest.mark.asyncio
    async def test_a_citation_the_snapshot_cannot_resolve_is_refused(
        self, store
    ):
        """Refuse the citation, not the message.

        Nothing durable could say what was cited, and an answer that cites
        nothing is an ordinary answer.
        """
        message = await _finish(store, {
            "content": ANSWER,
            "validated_citations": [{**CITED[0], "source_id": "src_4"}],
            "provenance_snapshot": SNAPSHOT,
        })

        assert message.content == ANSWER
        assert _citations(message) == []

    @pytest.mark.asyncio
    async def test_a_snapshot_that_never_arrived_resolves_nothing(self, store):
        """The names travel with the citations or the citations do not go."""
        message = await _finish(store, {
            "content": ANSWER,
            "validated_citations": CITED,
        })

        assert message.content == ANSWER
        assert _citations(message) == []


class TestAnOffsetIsCheckedAgainstTheAnswerBeingStored:
    @pytest.mark.asyncio
    async def test_an_offset_past_the_end_is_refused(self, store):
        """An offset outside the answer was measured against a different
        string, which is the failure the whole replacement rule exists to
        prevent - checked again here, against the row itself."""
        message = await _finish(store, {
            "content": ANSWER,
            "validated_citations": [{**CITED[0], "public_offset": len(ANSWER) + 1}],
            "provenance_snapshot": SNAPSHOT,
        })

        assert message.content == ANSWER
        assert _citations(message) == []

    @pytest.mark.asyncio
    async def test_the_end_of_the_answer_is_a_position_in_it(self, store):
        """The boundary the check must not exclude: a citation at the very end
        of the answer anchors after the last character."""
        message = await _finish(store, {
            "content": ANSWER,
            "validated_citations": [{**CITED[0], "public_offset": len(ANSWER)}],
            "provenance_snapshot": SNAPSHOT,
        })

        assert [item["start"] for item in _citations(message)] == [len(ANSWER)]

    @pytest.mark.asyncio
    async def test_a_negative_offset_is_refused(self, store):
        message = await _finish(store, {
            "content": ANSWER,
            "validated_citations": [{**CITED[0], "public_offset": -1}],
            "provenance_snapshot": SNAPSHOT,
        })

        assert _citations(message) == []

    def test_an_offset_that_is_not_a_position_is_refused(self):
        """Directly, because a `True` reaching here would index position 1.

        `bool` is an `int` in Python, so a check that asked only for an
        integer would accept it and anchor a citation one character in.
        """
        assert durable_citations(
            [{**CITED[0], "public_offset": True}], SNAPSHOT, ANSWER
        ) == []
        assert durable_citations(
            [{**CITED[0], "public_offset": "4"}], SNAPSHOT, ANSWER
        ) == []
        assert durable_citations(
            [{**CITED[0], "public_offset": None}], SNAPSHOT, ANSWER
        ) == []


class TestTheTurnAndTheRowAgreeEndToEnd:
    """The two halves joined by the real transient pair.

    Everything above hands `finish()` a hand-written orchestration, which
    tests the projection and assumes the shape. This runs a real turn with
    offers on - a real offer, a real handle, a real transfer and a real
    registry snapshot - and stores its result, so a change to what the
    workflow emits shows up here rather than in a fixture nobody updated.
    """

    @pytest.mark.asyncio
    async def test_a_real_cited_turn_lands_as_an_anchor_in_its_own_answer(
        self, store, monkeypatch
    ):
        from types import SimpleNamespace

        from liminallm.storage.models import KnowledgeChunk

        engine = get_runtime().workflow
        monkeypatch.setattr(
            type(engine), "CITATION_OFFERS_ENABLED", True, raising=False
        )
        opened: list = []
        real_open = engine.invocations.open

        def _open(*a, **k):
            invocation = real_open(*a, **k)
            opened.append(invocation)
            return invocation

        monkeypatch.setattr(engine.invocations, "open", _open)
        chunk = KnowledgeChunk(
            context_id="ctx", fs_path="/files/manual.md",
            content="SOURCE-SAYS-400-HOURS", embedding=[], chunk_index=0,
        )
        monkeypatch.setattr(
            engine, "rag", SimpleNamespace(retrieve=lambda *a, **k: [chunk])
        )
        monkeypatch.setattr(engine, "_validate_context_scope", lambda ids, **k: ["ctx"])
        monkeypatch.setattr(engine, "_resolve_context_ids", lambda a, b: ["ctx"])

        def _generate(*a, **k):
            cited = [inv for inv in opened if inv.citations]
            handle = next(iter(cited[-1].citations.by_handle), "") if cited else ""
            return {"content": f"Four hundred hours [cite:{handle}]", "usage": {}}

        monkeypatch.setattr(engine.llm, "generate", _generate, raising=False)

        turn = _turn(store)
        orchestration = await engine.run(
            None, turn.conversation_id, "how long", "ctx",
            user_id=turn.user_id, tenant_id=None,
        )
        assert orchestration.get("validated_citations"), orchestration.get("content")

        message = await chat_turn.finish(get_runtime(), turn, orchestration)
        stored = store.get_message(message.id)

        assert stored.content == "Four hundred hours"
        cited = _citations(stored)
        assert len(cited) == 1, _segments(stored)
        assert cited[0]["start"] == len("Four hundred hours")
        assert cited[0]["locator"] == "/files/manual.md"
        nonce = [inv for inv in opened if inv.citations][-1].citations.nonce
        row = json.dumps({
            "content": stored.content,
            "content_struct": stored.content_struct,
            "meta": stored.meta,
        })
        assert nonce not in row and "[cite:" not in row
