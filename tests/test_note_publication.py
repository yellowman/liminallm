"""What may become a note, and what a note then says about where it came from.

SPEC §19.5 makes the vault the user's permanent, cross-conversation corpus and
promotion into it a deliberate act by its owner. Everything here is a property
of that sentence: who may promote, what a failed reading leaves behind, and
what the stored note records about the file it is a reading *of* — because a
note is treated as something the user wrote, and an extraction is not.

The route already had the right shape before these tests: resolve beneath the
authenticated user's own attachment root, extract first, create the note only
after. So most of this is proof rather than repair — which is the point, since
the ordering is the whole defence and nothing was asserting it.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from liminallm.service.runtime import get_runtime


def _unique(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _account(client):
    email = f"{_unique('pub')}@example.com"
    resp = client.post(
        "/v1/auth/signup", json={"email": email, "password": "TestPassword123!"}
    )
    assert resp.status_code == 201, resp.text
    data = resp.json()["data"]
    return data["user_id"], {"Authorization": f"Bearer {data['access_token']}"}


def _upload(client, headers, name: str, data: bytes) -> None:
    resp = client.post(
        "/v1/files/upload",
        headers=headers,
        files={"file": (name, data, "text/markdown")},
    )
    assert resp.status_code == 200, resp.text


def _drop(runtime, user_id: str, name: str, data: bytes) -> Path:
    """Place a file directly, for content the upload endpoint would refuse."""
    from liminallm.service.attachments import attachment_path

    path = attachment_path(runtime.settings.shared_fs_root, user_id, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _notes(client, headers) -> list:
    resp = client.get("/v1/notes?limit=200", headers=headers)
    assert resp.status_code == 200, resp.text
    data = resp.json()["data"]
    return data if isinstance(data, list) else data.get("items", [])


def _promote(client, headers, name: str):
    return client.post("/v1/notes/from-file", headers=headers, json={"name": name})


class TestOnlyTheOwnerPromotes:
    """§19.5: the vault is the user's own corpus."""

    def test_a_stranger_cannot_promote_someone_elses_upload(self, client):
        runtime = get_runtime()
        victim, victim_headers = _account(client)
        secret = "the victim's private planning notes about protein"
        _upload(client, victim_headers, "private.md", secret.encode())

        victim_path = (
            Path(runtime.settings.shared_fs_root)
            / "users" / victim / "files" / "private.md"
        )
        assert victim_path.is_file(), "the fixture file is not where it should be"

        _stranger, headers = _account(client)
        # Both spellings, and the relative one is arithmetic rather than a
        # guess: from `<root>/users/<stranger>/files`, two levels up is
        # `<root>/users`, which is where the victim's directory sits. An
        # off-by-one here lands on a path that exists for nobody, and the
        # refusal it earns says nothing about traversal.
        for name in ("private.md", str(victim_path),
                     f"../../{victim}/files/private.md"):
            resp = _promote(client, headers, name)
            assert resp.status_code in (400, 403, 404), (name, resp.status_code)

        # The status code is not the property — the absence of the note is.
        for note in _notes(client, headers):
            full = runtime.store.get_note(note["id"])
            assert secret not in (full.content or ""), (
                "another user's file reached this vault"
            )


class TestAFailedReadingLeavesNothing:
    """A note is what the user has; a refusal must not leave one behind.

    The route extracts before it creates, so a failure returns before the
    store is touched. That ordering is load-bearing and was unasserted.
    """

    def test_binary_content_creates_no_note(self, client):
        runtime = get_runtime()
        user_id, headers = _account(client)
        _drop(runtime, user_id, "blob.bin", b"\x00\x01\x02binary\xff\xfe")

        before = len(_notes(client, headers))
        resp = _promote(client, headers, "blob.bin")
        assert resp.status_code == 400, resp.text
        assert len(_notes(client, headers)) == before, "a refusal left a note"

    def test_an_image_nothing_can_read_creates_no_note(self, client):
        """No OCR, no vision: the honest answer is a refusal, not a note whose
        content is decoded image bytes."""
        runtime = get_runtime()
        user_id, headers = _account(client)
        _drop(runtime, user_id, "chart.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)

        before = len(_notes(client, headers))
        resp = _promote(client, headers, "chart.png")
        assert resp.status_code == 400, resp.text
        assert len(_notes(client, headers)) == before, "a refusal left a note"

    def test_a_file_that_cannot_be_read_creates_no_note(self, client, monkeypatch):
        """The route's other refusal: the file is there and the read fails.

        Triggered by injection rather than by `chmod`, because the suite may
        run as root — measured, it does here — and root reads a 000 file
        happily, so the permission version of this passes for no reason.
        """
        from liminallm.api import routes

        runtime = get_runtime()
        user_id, headers = _account(client)
        _drop(runtime, user_id, "locked.md", b"readable, but not today")

        def unreadable(*_args, **_kwargs):
            raise OSError(5, "input/output error")

        monkeypatch.setattr(routes.extract_service, "extract_text", unreadable)
        before = len(_notes(client, headers))
        resp = _promote(client, headers, "locked.md")
        assert resp.status_code == 400, resp.text
        assert len(_notes(client, headers)) == before, "a failed read left a note"


class TestTheNoteRecordsWhatItIsAReadingOf:
    """§19.5: an OCR or vision result is a *reading* of the file, not a copy.

    So the note has to say which file and by what method, or the vault stops
    being able to tell a transcription from something the user wrote.
    """

    def test_provenance_names_the_file_and_the_method(self, client):
        runtime = get_runtime()
        user_id, headers = _account(client)
        _drop(runtime, user_id, "ideas.md", b"Uploaded thinking about protein.")

        resp = _promote(client, headers, "ideas.md")
        assert resp.status_code == 201, resp.text
        note = runtime.store.get_note(resp.json()["data"]["id"])
        assert note.meta["source"] == "upload"
        assert note.meta["filename"] == "ideas.md"
        assert note.meta["method"] == "text"
        assert note.meta["truncated"] is False

    def test_a_vision_reading_says_so(self, client, monkeypatch):
        runtime = get_runtime()
        user_id, headers = _account(client)
        _drop(runtime, user_id, "board.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
        monkeypatch.setattr(
            runtime.llm,
            "transcribe_image",
            lambda data, mime, *, prompt: "Whiteboard: protein targets.",
            raising=False,
        )

        resp = _promote(client, headers, "board.png")
        assert resp.status_code == 201, resp.text
        note = runtime.store.get_note(resp.json()["data"]["id"])
        assert note.meta["method"] == "vision", note.meta
        assert note.meta["filename"] == "board.png"


class TestTheSizeCapAndItsFlagAgree:
    """`truncated` is what tells the reader the note is not the whole file."""

    def _promote_text(self, client, runtime, user_id, headers, name, text):
        _drop(runtime, user_id, name, text.encode())
        resp = _promote(client, headers, name)
        assert resp.status_code == 201, resp.text
        data = resp.json()["data"]
        return data, runtime.store.get_note(data["id"])

    def test_content_past_the_cap_is_cut_and_flagged(self, client):
        from liminallm.service.notes import NOTE_FROM_FILE_MAX_BYTES

        runtime = get_runtime()
        user_id, headers = _account(client)
        data, note = self._promote_text(
            client, runtime, user_id, headers, "big.md",
            "a" * (NOTE_FROM_FILE_MAX_BYTES + 5000),
        )
        assert data["truncated"] is True
        assert note.meta["truncated"] is True
        assert len(note.content.encode("utf-8")) <= NOTE_FROM_FILE_MAX_BYTES

    def test_content_at_the_cap_is_whole_and_unflagged(self, client):
        """Exactly at the limit is not over it — an off-by-one here would
        mark a complete note as a partial one for the rest of its life."""
        from liminallm.service.notes import NOTE_FROM_FILE_MAX_BYTES

        runtime = get_runtime()
        user_id, headers = _account(client)
        text = "b" * NOTE_FROM_FILE_MAX_BYTES
        data, note = self._promote_text(
            client, runtime, user_id, headers, "exact.md", text
        )
        assert data["truncated"] is False
        assert note.content == text

    def test_a_multibyte_boundary_does_not_leave_a_broken_character(self, client):
        """The cut is on bytes, so it can land inside a character."""
        from liminallm.service.notes import NOTE_FROM_FILE_MAX_BYTES

        runtime = get_runtime()
        user_id, headers = _account(client)
        # 3 bytes each, so the cap never falls on a character boundary.
        text = "☃" * (NOTE_FROM_FILE_MAX_BYTES // 3 + 100)
        data, note = self._promote_text(
            client, runtime, user_id, headers, "snow.md", text
        )
        assert data["truncated"] is True
        assert set(note.content) == {"☃"}, "the cut left a partial character"
        assert len(note.content.encode("utf-8")) <= NOTE_FROM_FILE_MAX_BYTES


class TestFileContentCannotForgeAVisionSlot:
    """The pending-image slots are private-use characters in the extracted
    text, and the parent substitutes into them. Anything that could put those
    characters into the text could name a slot the parent then fills — so
    every source of text is stripped of them before a slot is ever made.
    """

    def test_source_text_carrying_the_markers_keeps_its_characters(self, client):
        """The markers are removed, not the run of text around them.

        Stripping and deleting look the same until the file's own characters
        matter: `_PH_RE` erases a whole `<open>N<close>` group, so a file
        whose text survived to that point would have content silently eaten.
        """
        from liminallm.service.extract import _PH_CLOSE, _PH_OPEN

        runtime = get_runtime()
        user_id, headers = _account(client)
        body = f"before {_PH_OPEN}0{_PH_CLOSE} after"
        _drop(runtime, user_id, "forge.md", body.encode())

        resp = _promote(client, headers, "forge.md")
        assert resp.status_code == 201, resp.text
        note = runtime.store.get_note(resp.json()["data"]["id"])
        assert _PH_OPEN not in note.content and _PH_CLOSE not in note.content
        assert note.content == "before 0 after", note.content

    def test_a_vision_reading_carrying_the_markers_cannot_name_a_slot(
        self, client, monkeypatch
    ):
        """The model's own output is untrusted text too: it is a reading of an
        image somebody else supplied."""
        from liminallm.service.extract import _PH_CLOSE, _PH_OPEN

        runtime = get_runtime()
        user_id, headers = _account(client)
        _drop(runtime, user_id, "evil.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
        monkeypatch.setattr(
            runtime.llm,
            "transcribe_image",
            lambda data, mime, *, prompt: f"seen{_PH_OPEN}9{_PH_CLOSE}text",
            raising=False,
        )

        resp = _promote(client, headers, "evil.png")
        assert resp.status_code == 201, resp.text
        note = runtime.store.get_note(resp.json()["data"]["id"])
        assert _PH_OPEN not in note.content and _PH_CLOSE not in note.content
        assert note.content == "seen9text", note.content

    def test_a_reader_carrying_the_markers_cannot_name_a_slot(self, tmp_path):
        """Third source, same rule. A registered reader runs in the child, so
        this one is checked where it happens rather than through the route."""
        from liminallm.service import extract as extract_service
        from liminallm.service.extract import _PH_CLOSE, _PH_OPEN

        marked = f"ocr{_PH_OPEN}3{_PH_CLOSE}output"
        extract_service.register_reader(
            "test_marker_reader", lambda data, mime, llm: marked, kind="vision"
        )
        try:
            image = tmp_path / "scan.png"
            image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
            result = extract_service.extract_text(
                image, readers=("test_marker_reader",), sandbox=False
            )
        finally:
            extract_service._READERS.pop("test_marker_reader", None)
        assert _PH_OPEN not in result["text"] and _PH_CLOSE not in result["text"]
        assert result["text"] == "ocr3output", result["text"]


class TestRagIngestionRefusesRatherThanIndexingGarbage:
    """§19.5 routes ingestion through the same extractor for the same reason:
    `read_text()` on a PDF "succeeds" and fills the index with stripped binary
    that then wins similarity searches. A refusal is zero chunks, not garbage.
    """

    def test_an_unextractable_file_contributes_no_chunks(self, client, tmp_path):
        runtime = get_runtime()
        _user_id, headers = _account(client)
        created = client.post(
            "/v1/contexts",
            headers=headers,
            json={"name": _unique("ctx"), "description": "ingest"},
        )
        assert created.status_code in (200, 201), created.text
        context_id = created.json()["data"]["id"]

        junk = tmp_path / "photo.jpg"
        junk.write_bytes(b"\xff\xd8\xff\xe0" + bytes(range(256)) * 8)
        assert runtime.rag.ingest_file(context_id, str(junk)) == 0

        hits = runtime.rag.retrieve([context_id], "photo", limit=5)
        assert hits == [], f"decoded binary reached the index: {hits}"

    def test_a_readable_file_still_contributes_chunks(self, client, tmp_path):
        """The refusal above must be about the content, not about ingestion."""
        runtime = get_runtime()
        _user_id, headers = _account(client)
        created = client.post(
            "/v1/contexts",
            headers=headers,
            json={"name": _unique("ctx"), "description": "ingest"},
        )
        assert created.status_code in (200, 201), created.text
        context_id = created.json()["data"]["id"]

        good = tmp_path / "readable.md"
        good.write_text("Protein needs are easily met on a varied diet.\n" * 20)
        assert runtime.rag.ingest_file(context_id, str(good)) > 0
