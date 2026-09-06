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

import hashlib
import json
import uuid

import pytest

from liminallm.api import chat_turn
from liminallm.service.auth import AuthContext
from liminallm.service.citations import durable_citations, public_source_id
from liminallm.service.runtime import get_runtime


@pytest.fixture
def client():
    """An unauthenticated client, for the surface that has no principal."""
    from fastapi.testclient import TestClient

    from liminallm import app as app_module

    return TestClient(app_module.app)


def _digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


#: The passages the fixture's evidence records describe, hashed the way the
#: registry hashes them - the projection recomputes the digest, so a
#: hand-written one would only ever prove that the check rejects fixtures.
PASSAGES = {
    "ev_1": "SOURCE SAYS 400 HOURS",
    "ev_2": "the handbook says otherwise",
}


#: A turn's registry as `SourceRegistry.snapshot()` exports it.
#:
#: The file's locator is the shape plain retrieval really records - the
#: server's own path to the bytes, under the shared root and the owner's id -
#: because that is what must not reach a stored row.
SNAPSHOT = {
    "sources": {
        "src_1": {
            "source_id": "src_1",
            "kind": "file",
            "title": "manual.md",
            "origin_id": None,
            "locator": "/srv/liminal/users/8f14e45f/files/manual.md",
            "metadata": {},
        },
        "src_2": {
            "source_id": "src_2",
            "kind": "web",
            "title": "Turbine handbook",
            "origin_id": None,
            "locator": "https://example.test/handbook",
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
            "locator": "/srv/liminal/users/8f14e45f/files/unrelated.md",
            "metadata": {},
        },
    },
    "evidence": [
        {
            "evidence_id": "ev_1",
            "source_id": "src_1",
            "text": PASSAGES["ev_1"],
            "locator": {"chunk_index": 3},
            "content_hash": _digest(PASSAGES["ev_1"]),
        },
        {
            "evidence_id": "ev_2",
            "source_id": "src_2",
            "text": PASSAGES["ev_2"],
            "locator": {},
            "content_hash": _digest(PASSAGES["ev_2"]),
        },
    ],
}

ANSWER = "Beta"

#: What the answer's own citation looks like coming out of the turn: the
#: marker sat directly after `Beta`, so the anchor is its length.
CITED = [{
    "source_id": "src_1",
    "canonical_start": 5,
    "canonical_end": 22,
    "public_offset": 4,
    "evidence_ids": ["ev_1"],
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
        # The source is named by what it is, not by where it was. Plain
        # retrieval gives files no object identity, so the record claims
        # none - and the path it happened to live at is not one.
        assert cited[0]["source_id"] == ""
        assert cited[0]["locator"] == ""
        assert cited[0]["meta"]["title"] == "manual.md"
        assert cited[0]["meta"]["kind"] == "file"
        # What pins the reading: the fingerprint of the passage the answer
        # rested on, and no passage text.
        assert cited[0]["meta"]["evidence"] == [
            {"content_hash": _digest(PASSAGES["ev_1"]),
             "locator": {"chunk_index": 3}}
        ]
        assert "SOURCE SAYS" not in json.dumps(message.content_struct)
        # The answer the anchors point into is in the struct too, and it is
        # the answer: a struct of anchors over absent or partial text would
        # place them in a string the renderer does not have.
        assert [
            s.get("text") for s in _segments(message) if s.get("type") == "text"
        ] == [ANSWER]

    @pytest.mark.asyncio
    async def test_the_anchor_is_counted_in_code_points(self, store):
        """The unit is normative (SPEC §2.2) because the two sides of this
        record count differently by default.

        Python indexes code points and JavaScript indexes UTF-16 code units,
        so an anchor after one emoji is 1 here and 2 in a naive `slice`. The
        stored number is the Python one, and a renderer converts.
        """
        content = "\U0001F600 Alpha"
        message = await _finish(store, {
            "content": content,
            "validated_citations": [{**CITED[0], "public_offset": 1}],
            "provenance_snapshot": SNAPSHOT,
        })

        anchor = _citations(message)[0]["start"]
        assert anchor == 1
        assert message.content[:anchor] == "\U0001F600"
        # The same position measured the other way, which is what a renderer
        # must not use unconverted.
        assert len(message.content[:anchor].encode("utf-16-le")) // 2 == 2

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

        titled = {item["meta"]["title"] for item in _citations(message)}
        assert titled == {"manual.md"}
        assert "unrelated" not in json.dumps(message.content_struct)

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
        assert cited[0]["meta"]["title"] == "manual.md"
        assert cited[0]["locator"] == ""
        assert len(cited[0]["meta"]["evidence"]) == 1
        assert len(cited[0]["meta"]["evidence"][0]["content_hash"]) == 64
        nonce = [inv for inv in opened if inv.citations][-1].citations.nonce
        row = json.dumps({
            "content": stored.content,
            "content_struct": stored.content_struct,
            "meta": stored.meta,
        })
        assert nonce not in row and "[cite:" not in row


class TestALocatorIsPublishedOnlyWhereItIsAReference:
    """A locator says where a source is, and only some of them say it in a
    form a reader may have. A URL is the reference and the identity at once; a
    file's locator is the server's own path to the bytes, which is deployment
    layout rather than provenance."""

    WEB = [{
        "source_id": "src_2",
        "canonical_start": 5,
        "canonical_end": 22,
        "public_offset": 4,
        "evidence_ids": ["ev_2"],
    }]

    @pytest.mark.asyncio
    async def test_a_web_citation_keeps_the_url(self, store):
        message = await _finish(store, {
            "content": ANSWER,
            "validated_citations": self.WEB,
            "provenance_snapshot": SNAPSHOT,
        })

        cited = _citations(message)
        assert cited[0]["locator"] == "https://example.test/handbook"
        assert cited[0]["meta"]["kind"] == "web"

    @pytest.mark.asyncio
    async def test_a_file_citation_carries_no_server_path(self, store):
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
        assert "/srv/liminal" not in row
        assert "8f14e45f" not in row
        assert "/users/" not in row

    def test_evidence_belonging_to_another_source_drops_the_citation(self):
        """The relation the authority gate already enforces, enforced again
        where it becomes durable.

        `build_citation_table` refuses a binding whose evidence names a
        different source. The registry is a consulted superset, so `ev_2`
        legitimately exists - it just belongs to the handbook, not the manual,
        and storing its fingerprint under the manual's citation would say the
        answer rested on a passage it never read.
        """
        assert durable_citations(
            [{**CITED[0], "evidence_ids": ["ev_2"]}], SNAPSHOT, ANSWER
        ) == []

    def test_a_fingerprint_that_does_not_fingerprint_its_passage_drops_it(self):
        """The snapshot carries the passage and the hash; the row will carry
        only the hash. This is the last place the claim can be checked at all,
        so it is - redundantly, since the registry computed that digest."""
        broken = {
            **SNAPSHOT,
            "evidence": [
                {**SNAPSHOT["evidence"][0], "content_hash": "c" * 64},
                SNAPSHOT["evidence"][1],
            ],
        }
        assert durable_citations(CITED, broken, ANSWER) == []

    def test_a_record_that_does_not_carry_its_passage_drops_the_citation(self):
        """The hash is checked against the passage, so a record with no
        passage cannot be checked at all - and an unchecked fingerprint is
        what this boundary exists to refuse, not to pass through."""
        broken = {
            **SNAPSHOT,
            "evidence": [
                {**SNAPSHOT["evidence"][0], "text": None},
                SNAPSHOT["evidence"][1],
            ],
        }
        assert durable_citations(CITED, broken, ANSWER) == []

    def test_a_citation_naming_no_evidence_at_all_is_dropped(self):
        """Upstream a handle is issued only once a binding exists, so an
        empty list is already a disagreement rather than a citation that
        happens to rest on nothing."""
        assert durable_citations(
            [{**CITED[0], "evidence_ids": []}], SNAPSHOT, ANSWER
        ) == []
        entry = {key: value for key, value in CITED[0].items()
                 if key != "evidence_ids"}
        assert durable_citations([entry], SNAPSHOT, ANSWER) == []

    def test_resolving_some_of_the_evidence_is_not_resolving_it(self):
        """No silent subset: a citation that rested on two passages and can
        only account for one is not a citation about one of them."""
        assert durable_citations(
            [{**CITED[0], "evidence_ids": ["ev_1", "ev_missing"]}],
            SNAPSHOT, ANSWER,
        ) == []

    def test_an_evidence_id_the_snapshot_does_not_hold_drops_the_citation(self):
        """Not a citation with an empty fingerprint list.

        Upstream, a handle exists only once a valid binding does, so a
        validated citation naming evidence the snapshot cannot resolve is two
        representations of one turn disagreeing. What would be stored is a
        title and two empty strings, which identifies nothing.
        """
        assert durable_citations(
            [{**CITED[0], "evidence_ids": ["ev_missing"]}], SNAPSHOT, ANSWER
        ) == []


class TestOnlyTheValidatedListMakesACitationSegment:
    """The projection obeys the rule on its own. This is the seam around it:
    an assistant row's `content_struct` arrives from the orchestration, the
    segment schema accepts `type="citation"`, and nothing else may write one.

    No producer emits an assistant `content_struct` today, which is why the
    rule is written rather than left resting on that.
    """

    FORGED = {
        "type": "citation",
        "source_id": "forged-source",
        "locator": "https://attacker.test/page",
        "start": 0,
        "end": 0,
    }

    @pytest.mark.asyncio
    async def test_a_citation_nothing_validated_is_stripped(self, store):
        message = await _finish(store, {
            "content": ANSWER,
            "validated_citations": [],
            "content_struct": {"segments": [dict(self.FORGED)]},
        })

        assert message.content == ANSWER
        assert _citations(message) == []
        assert "attacker.test" not in json.dumps(message.content_struct)
        # The answer is still stored as text: stripping the only segment left
        # a struct that would otherwise hold nothing.
        assert [
            s.get("text") for s in _segments(message) if s.get("type") == "text"
        ] == [ANSWER]

    @pytest.mark.asyncio
    async def test_only_the_validated_one_survives_beside_a_forged_one(
        self, store
    ):
        message = await _finish(store, {
            "content": ANSWER,
            "validated_citations": CITED,
            "provenance_snapshot": SNAPSHOT,
            "content_struct": {
                "segments": [
                    {"type": "text", "text": ANSWER},
                    dict(self.FORGED),
                ]
            },
        })

        cited = _citations(message)
        assert len(cited) == 1, cited
        assert cited[0]["meta"]["title"] == "manual.md"
        assert "forged-source" not in json.dumps(message.content_struct)

    @pytest.mark.asyncio
    async def test_the_other_segment_types_are_kept(self, store):
        """Stripping is about citations. A code block the producer sent is
        not a claim about a source."""
        message = await _finish(store, {
            "content": ANSWER,
            "validated_citations": [],
            "content_struct": {
                "segments": [
                    {"type": "text", "text": ANSWER},
                    {"type": "code", "text": "print('hi')", "language": "python"},
                ]
            },
        })

        assert [s["type"] for s in _segments(message)] == ["text", "code"]

    @pytest.mark.asyncio
    async def test_a_malformed_segment_does_not_hide_the_answer(self, store):
        """The normalizer drops a segment of unknown type. Deciding whether
        the answer's text is needed before that ran left a struct of anchors
        over text nobody stored."""
        message = await _finish(store, {
            "content": ANSWER,
            "validated_citations": CITED,
            "provenance_snapshot": SNAPSHOT,
            "content_struct": {"segments": [{"type": "not-a-segment"}]},
        })

        assert [
            s.get("text") for s in _segments(message) if s.get("type") == "text"
        ] == [ANSWER]
        assert len(_citations(message)) == 1


class TestACitationNamesTheReadingItRestedOn:
    """The two properties a durable citation has to have over time, against a
    file that really lives where uploads put one.

    Plain retrieval is explicit that a chunk under a path claims to be the
    contents of that path *now* - the schema records no generation. So a
    record that identified the source by its path would follow the name to
    whatever the file holds next, and present an old answer as resting on
    bytes it never read.
    """

    #: Long enough to retrieve on, and different enough between the two
    #: generations that their passages cannot hash alike.
    OLD = "Turbine blade inspection interval detail. " * 60
    NEW = "Turbine blade inspection revised schedule detail. " * 60
    QUESTION = "turbine blade inspection"

    @staticmethod
    def _uploaded(store, text):
        """A real user, a real context, and a real file under the user's own
        files directory - the same shape the upload route ingests."""
        import pathlib

        runtime = get_runtime()
        user = store.create_user(email=f"up_{uuid.uuid4().hex[:8]}@example.com")
        context = store.upsert_context(
            name=f"up-{uuid.uuid4().hex[:6]}", description="uploaded",
            owner_user_id=user.id,
        )
        path = (
            pathlib.Path(runtime.settings.shared_fs_root)
            / "users" / user.id / "files" / "manual.md"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        written = runtime.rag.ingest_text(context.id, text, source_path=str(path))
        assert written > 0, "the fixture indexed nothing"
        return user, context, path

    @staticmethod
    async def _cited_turn(store, monkeypatch, user, context):
        """One real turn over that context, whose answer cites what it read."""
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

        def _generate(*a, **k):
            cited = [inv for inv in opened if inv.citations]
            handle = next(iter(cited[-1].citations.by_handle), "") if cited else ""
            return {"content": f"Four hundred hours [cite:{handle}]", "usage": {}}

        monkeypatch.setattr(engine.llm, "generate", _generate, raising=False)
        conversation = store.create_conversation(title="cited", user_id=user.id)
        turn = chat_turn.Turn(
            principal=AuthContext(user_id=user.id, role="user", tenant_id=None),
            conversation_id=conversation.id,
            context_id=context.id,
            workflow_id=None,
            user_content=TestACitationNamesTheReadingItRestedOn.QUESTION,
            user_message=None,
            needs_title=False,
        )
        orchestration = await engine.run(
            None, conversation.id,
            TestACitationNamesTheReadingItRestedOn.QUESTION, context.id,
            user_id=user.id, tenant_id=None,
        )
        assert orchestration.get("validated_citations"), (
            f"the turn cited nothing: {orchestration.get('content')!r}"
        )
        message = await chat_turn.finish(get_runtime(), turn, orchestration)
        stored = store.get_message(message.id)
        assert stored is not None
        return stored

    @pytest.mark.asyncio
    async def test_the_row_carries_no_part_of_the_server_path(
        self, store, monkeypatch
    ):
        """The path is `<shared_fs_root>/users/<user_id>/files/<name>`, and
        every part of it is deployment layout or an account identifier. The
        durable citation is public data - the client renders `source_id` and
        `locator` - so none of it may be in there."""
        user, context, path = self._uploaded(store, self.OLD)

        stored = await self._cited_turn(store, monkeypatch, user, context)

        row = json.dumps({
            "content": stored.content,
            "content_struct": stored.content_struct,
            "meta": stored.meta,
        })
        assert str(get_runtime().settings.shared_fs_root) not in row
        assert user.id not in row
        assert str(path) not in row
        # What is there instead: the file's own name, which is what the file
        # routes address and what a reader can act on.
        assert _citations(stored)[0]["meta"]["title"] == "manual.md"

    @pytest.mark.asyncio
    async def test_replacing_the_file_does_not_retarget_the_old_citation(
        self, store, monkeypatch
    ):
        """Cite one reading of a path, then commit another at the same path.

        The stored citation must still be about the first. It holds the
        fingerprint of the passage it rested on, and that fingerprint matches
        nothing the path holds now - so the old answer reads as resting on
        something that is gone, rather than on the replacement.
        """
        import hashlib

        user, context, path = self._uploaded(store, self.OLD)
        stored = await self._cited_turn(store, monkeypatch, user, context)
        fingerprints = {
            item["content_hash"]
            for item in _citations(stored)[0]["meta"]["evidence"]
        }
        assert fingerprints, "the citation recorded no evidence to be pinned by"

        # A second generation of the same path, exactly as re-uploading does.
        path.write_text(self.NEW, encoding="utf-8")
        assert get_runtime().rag.ingest_text(
            context.id, self.NEW, source_path=str(path)
        ) > 0

        now = {
            hashlib.sha256(chunk.content.encode("utf-8")).hexdigest()
            for chunk in get_runtime().rag.retrieve(
                [context.id], self.QUESTION, user_id=user.id, tenant_id=None,
            )
        }
        assert now, "the replacement indexed nothing"
        assert fingerprints.isdisjoint(now), (
            "the stored citation now matches the file's new contents"
        )
        # And nothing else in the row names the replacement either.
        assert "revised schedule" not in json.dumps({
            "content_struct": stored.content_struct, "meta": stored.meta,
        })


class TestAnInternalIdentityIsNotAutomaticallyAPublicOne:
    """`content_struct` is an API field, so a stored `source_id` is a public
    one whether or not today's client renders it.

    Being the right internal identity does not make an identity the right
    public one. The registry deliberately identifies an MCP source by the
    admin-owned artifact row of its server, because two admins can both
    configure a server called `inventory`; that is a fact about the
    deployment's configuration, seen from a user's chat.
    """

    @staticmethod
    def _snapshot(kind, origin_id, *, passage="a passage"):
        return {
            "sources": {
                "src_1": {
                    "source_id": "src_1", "kind": kind, "title": "the source",
                    "origin_id": origin_id, "locator": None, "metadata": {},
                },
            },
            "evidence": [{
                "evidence_id": "ev_1", "source_id": "src_1", "text": passage,
                "locator": {"chunk_index": 0}, "content_hash": _digest(passage),
            }],
        }

    def _stored(self, kind, origin_id):
        segments = durable_citations(CITED, self._snapshot(kind, origin_id), ANSWER)
        assert len(segments) == 1, segments
        return segments[0]

    def test_an_mcp_server_artifact_id_is_not_published(self):
        artifact = "3f2a1e10-0b7c-4d5e-9a11-77c0de5b1234"
        origin = f'mcp:["{artifact}","lookup_stock"]'

        stored = self._stored("mcp", origin)

        assert artifact not in json.dumps(stored)
        assert stored["source_id"].startswith("opaque:")
        # Still an identity: the same tool cited in another turn groups with
        # this one.
        assert stored["source_id"] == public_source_id("mcp", origin)
        assert public_source_id("mcp", origin) != public_source_id(
            "mcp", f'mcp:["{artifact}","other_tool"]'
        )

    def test_an_inline_chunk_row_id_is_not_published_even_hashed(self):
        """A digest of an enumerable identity is not opaque: anyone who wants
        to know whether a citation is row 42 can hash 42. Inline text names no
        object a reader could open, so it publishes no identity at all."""
        stored = self._stored("unknown", "knowledge_chunk:42")

        assert stored["source_id"] == ""
        assert "42" not in json.dumps(stored["source_id"])
        assert public_source_id("unknown", "knowledge_chunk:42") == ""

    def test_the_reader_s_own_objects_keep_their_names(self):
        """A note and a conversation are the reader's, and the id is how a
        client would navigate to one."""
        assert self._stored("note", "note:7")["source_id"] == "note:7"
        assert self._stored(
            "conversation", "conversation:c-1"
        )["source_id"] == "conversation:c-1"

    def test_an_attachment_keeps_its_digest_identity(self):
        """Built from the real key rather than a plausible one.

        The allowlist writes the prefix out rather than importing it, so this
        is what stops the two from drifting - and a hand-written `gen:...`
        would have proved only that the fixture matched the fixture.
        """
        from liminallm.service.attachments import generation_key

        generation = generation_key("d" * 64, "report.pdf")
        assert generation and generation.startswith("attachment-generation:")

        assert self._stored("file", generation)["source_id"] == generation

    def test_a_file_identity_of_some_other_shape_is_not_published(self):
        """The kind is not the attestation. `file` covers both an attachment
        generation and a plain context file, and a third scheme added later
        publishes nothing until someone reviews its shape."""
        assert public_source_id("file", "file_row:42") == ""

    def test_a_kind_nobody_classified_publishes_nothing(self):
        """Fail-closed, like the locator: a kind added later carries no
        identity until someone decides its identity is safe to publish."""
        assert public_source_id("newly-invented", "internal:1") == ""

    def test_the_title_is_what_a_reader_is_shown(self):
        """Every kind has one, and it is the presentation label - the client
        must not derive a name from an identity or a path."""
        assert self._stored("mcp", 'mcp:["x","y"]')["meta"]["title"] == "the source"


class TestAnEvidenceLocatorSaysWhereNotWhich:
    """The locator answers "where in this source", and a citation needs that
    answer. `chunk_id` and `block_id` answer "which row", which is a different
    claim and gets the same treatment as `origin_id`."""

    @staticmethod
    def _snapshot(locator, *, passage="a passage"):
        return {
            "sources": {
                "src_1": {
                    "source_id": "src_1", "kind": "file", "title": "manual.md",
                    "origin_id": None, "locator": None, "metadata": {},
                },
            },
            "evidence": [{
                "evidence_id": "ev_1", "source_id": "src_1", "text": passage,
                "locator": locator, "content_hash": _digest(passage),
            }],
        }

    def _locator(self, locator):
        segments = durable_citations(CITED, self._snapshot(locator), ANSWER)
        assert len(segments) == 1, segments
        return segments[0]["meta"]["evidence"][0]["locator"]

    def test_a_position_in_the_source_is_kept(self):
        assert self._locator({"chunk_index": 3, "page": 2, "section": "Intro"}) == {
            "chunk_index": 3, "page": 2, "section": "Intro",
        }

    def test_a_row_id_is_not(self):
        kept = self._locator({
            "chunk_index": 3, "chunk_id": "9182", "block_id": "msg-uuid",
        })
        assert kept == {"chunk_index": 3}

    def test_an_empty_locator_stays_empty(self):
        assert self._locator({}) == {}
        assert self._locator({"chunk_index": None}) == {}


class TestAShareViewerIsADifferentAudience:
    """"Public to the owner" and "public to whoever has the link" are two
    boundaries, and the stored citation is the first one's.

    The owner holds the note, the attachment and the passage a citation names.
    A share viewer authenticated as nobody holds none of them, so an identity
    they cannot open - or a hash of a passage they cannot read - is a
    checkable claim rather than provenance.
    """

    #: Everything a stored citation can carry that the owner may see and a
    #: stranger may not. Each is what one production producer really emits.
    PRIVATE = {
        "note": "note:6f1b8c2e-77aa-4f0d-9d2e-2b6f0f6b1234",
        "conversation": "conversation:1d6a5f70-2c3b-4c1a-9c62-3e5c9d70abcd",
        "attachment": "attachment-generation:" + ("e" * 64) + ":.pdf",
        "mcp": 'mcp:["9c1e77aa-4f0d-4d2e-8b6f-0f6b12345678","lookup_stock"]',
        "chunk": "knowledge_chunk:42",
    }

    @staticmethod
    def _snapshot(kind, origin_id, passage):
        return {
            "sources": {
                "src_1": {
                    "source_id": "src_1", "kind": kind, "title": "private notes",
                    "origin_id": origin_id, "locator": None, "metadata": {},
                },
            },
            "evidence": [{
                "evidence_id": "ev_1", "source_id": "src_1", "text": passage,
                "locator": {"chunk_index": 7}, "content_hash": _digest(passage),
            }],
        }

    @pytest.mark.asyncio
    async def test_no_private_identity_or_fingerprint_crosses_the_share(
        self, store, client
    ):
        """The whole boundary, driven through the anonymous route.

        A note is cited, the conversation is shared, and the share is fetched
        with no credentials at all - a fresh client, so not even a cookie
        from another test's login could be carrying the request.
        """
        passage = "the private passage the answer rested on"
        turn = _turn(store)
        message = await chat_turn.finish(get_runtime(), turn, {
            "content": ANSWER,
            "validated_citations": CITED,
            "provenance_snapshot": self._snapshot(
                "note", self.PRIVATE["note"], passage
            ),
        })
        stored = store.get_message(message.id)
        # The owner's own copy carries all of it, which is what makes the
        # share's version a projection rather than a coincidence.
        owner = json.dumps(stored.content_struct)
        assert self.PRIVATE["note"] in owner
        assert _digest(passage) in owner

        store.set_conversation_public(
            turn.conversation_id, user_id=turn.user_id, public=True
        )
        resp = client.get(f"/v1/public/conversations/{turn.conversation_id}")

        assert resp.status_code == 200, resp.text
        body = resp.text
        assert ANSWER in body, "the share showed no answer"
        for name, identity in self.PRIVATE.items():
            assert identity not in body, name
        assert _digest(passage) not in body
        assert passage not in body
        assert "content_hash" not in body
        assert "evidence" not in body

    @pytest.mark.asyncio
    async def test_the_share_still_shows_what_a_citation_is_for(
        self, store, client
    ):
        """Stripping is not deleting: the reader still sees that the answer
        cited something, where, and what it was called."""
        passage = "a page of the handbook"
        turn = _turn(store)
        await chat_turn.finish(get_runtime(), turn, {
            "content": ANSWER,
            "validated_citations": CITED,
            "provenance_snapshot": {
                "sources": {
                    "src_1": {
                        "source_id": "src_1", "kind": "web",
                        "title": "Turbine handbook", "origin_id": None,
                        "locator": "https://example.test/handbook",
                        "metadata": {},
                    },
                },
                "evidence": [{
                    "evidence_id": "ev_1", "source_id": "src_1",
                    "text": passage, "locator": {"chunk_index": 7},
                    "content_hash": _digest(passage),
                }],
            },
        })
        store.set_conversation_public(
            turn.conversation_id, user_id=turn.user_id, public=True
        )

        resp = client.get(f"/v1/public/conversations/{turn.conversation_id}")

        shown = [
            segment
            for message in resp.json()["data"]["messages"]
            for segment in (message.get("content_struct") or {}).get("segments") or []
            if segment.get("type") == "citation"
        ]
        assert len(shown) == 1, resp.text
        assert shown[0]["meta"]["title"] == "Turbine handbook"
        assert shown[0]["locator"] == "https://example.test/handbook"
        assert shown[0]["start"] == 4
        assert set(shown[0]) == {"type", "start", "end", "locator", "meta"}
        assert set(shown[0]["meta"]) == {"kind", "title"}

    @pytest.mark.asyncio
    async def test_a_trace_segment_is_not_part_of_a_share(self, store, client):
        """An anonymous reader is being shown an answer, not a trace. Nothing
        writes a `tool_call` segment onto an assistant row today, which is
        why the allowlist is written rather than left to that."""
        turn = _turn(store)
        await chat_turn.finish(get_runtime(), turn, {
            "content": ANSWER,
            "validated_citations": [],
            "content_struct": {"segments": [
                {"type": "text", "text": ANSWER},
                {"type": "tool_call", "name": "lookup",
                 "arguments": {"token": "SECRET-ARGUMENT"}},
            ]},
        })
        store.set_conversation_public(
            turn.conversation_id, user_id=turn.user_id, public=True
        )

        resp = client.get(f"/v1/public/conversations/{turn.conversation_id}")

        assert "SECRET-ARGUMENT" not in resp.text
        assert "tool_call" not in resp.text
        assert ANSWER in resp.text


class TestTheResponsesSurfaceCarriesWhatTheTurnCited:
    """The OpenAI-compatible surface answers from the same kernel, so a turn
    that cited something must not arrive there having cited nothing.

    Not in `annotations`: that field is the provider's own file-citation
    shape, and filling it would claim a file identity this surface does not
    have. The extension is the honest slot, and what goes in it is the record
    the turn stored rather than the transient coordinates beside it.
    """

    @staticmethod
    def _cited_backend(monkeypatch):
        """Offers on, one retrievable passage, and an answer that cites it."""
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
            content="SOURCE SAYS 400 HOURS", embedding=[], chunk_index=0,
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
        return opened

    def test_a_cited_answer_arrives_with_its_citation(
        self, client, auth_headers, monkeypatch
    ):
        opened = self._cited_backend(monkeypatch)

        resp = client.post(
            "/v1/responses", headers=auth_headers,
            json={"input": "how long between inspections"},
        )

        assert resp.status_code == 200, resp.text
        body = resp.json()
        cited = body["liminallm"]["citations"]
        assert len(cited) == 1, body["liminallm"]
        assert cited[0]["meta"]["title"] == "manual.md"
        # The anchor indexes the text this response carries.
        text = "".join(
            part.get("text", "")
            for item in body["output"] if item.get("type") == "message"
            for part in item.get("content") or []
        )
        assert text == "Four hundred hours"
        assert cited[0]["start"] == len(text)
        # The dialect's own field stays empty, and nothing of the namespace
        # reached the wire.
        for item in body["output"]:
            for part in item.get("content") or []:
                assert part.get("annotations") == []
        nonce = [inv for inv in opened if inv.citations][-1].citations.nonce
        assert nonce not in resp.text and "[cite:" not in resp.text

    def test_an_uncited_answer_carries_an_empty_list(
        self, client, auth_headers
    ):
        """The key is always present, so a client reads one shape."""
        resp = client.post(
            "/v1/responses", headers=auth_headers, json={"input": "hello"},
        )

        assert resp.status_code == 200, resp.text
        assert resp.json()["liminallm"]["citations"] == []
