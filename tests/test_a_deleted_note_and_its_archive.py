"""Deleting a note ends it in the vault, and leaves the archive alone.

Two rules that pull in opposite directions, so they are pinned together.

A note deleted while a `PATCH` is in flight must answer 404, not 500. The
route checks ownership, then writes; a delete landing between those two makes
`update_note` match no row and return None, and the route used to hand that
None to `_save_note_graph`, which dereferenced `note.id`. A deliberate
deletion answered as an internal fault.

Only that ordering is a lie. A delete landing *after* the write returns a real
note, and "the update committed, then the note was deleted" is a truthful
account of that history - measured, and left alone.

A saved sweep report is the other direction. SPEC 19.6 makes it a
self-contained historical snapshot that `GET /v1/notes/sweeps` replays, so the
note's id, its title at the time, and the judgment the witness made from its
excerpt stay in the report after the note is gone. Redacting them would make
the archive describe a sweep that never happened. Erasing the account takes
the archive with everything else.
"""

from __future__ import annotations

import json
import uuid

import pytest

from liminallm.service.runtime import get_runtime

MARK = f"NQA{uuid.uuid4().hex[:10].upper()}"


@pytest.fixture
def auth(client):
    email = f"note_{uuid.uuid4().hex[:8]}@example.com"
    resp = client.post("/v1/auth/signup",
                       json={"email": email, "password": "TestPassword123!"})
    assert resp.status_code == 201, resp.text
    data = resp.json()["data"]
    headers = {"Authorization": f"Bearer {data['access_token']}"}
    user_id = client.get("/v1/me", headers=headers).json()["data"]["id"]
    return {"headers": headers, "user_id": user_id}


def _note(client, headers, title, content):
    resp = client.post("/v1/notes", headers=headers,
                       json={"title": title, "content": content})
    assert resp.status_code in (200, 201), resp.text
    return resp.json()["data"]["id"]


def _count(sql, params):
    with get_runtime().store._connect() as conn:
        return conn.execute(sql, params).fetchone()["n"]


def _reports(user_id):
    with get_runtime().store._connect() as conn:
        return [
            json.dumps(r["report"], sort_keys=True)
            for r in conn.execute(
                "SELECT report FROM sweep_report WHERE user_id = %s "
                "ORDER BY created_at",
                (user_id,),
            ).fetchall()
        ]


class TestANoteDeletedMidPatch:
    def test_a_delete_before_the_write_answers_not_found(self, client, auth):
        """The measured defect: this used to raise AttributeError, so 500."""
        headers = auth["headers"]
        store = get_runtime().store
        note_id = _note(client, headers, f"A {uuid.uuid4().hex[:6]}", "first body")

        real_update = store.update_note
        state: dict = {}

        def _delete_then_update(*args, **kwargs):
            if "deleted" not in state:
                state["deleted"] = client.delete(
                    f"/v1/notes/{note_id}", headers=headers
                ).status_code
            return real_update(*args, **kwargs)

        store.update_note = _delete_then_update
        try:
            resp = client.patch(f"/v1/notes/{note_id}", headers=headers,
                                json={"content": "second body"})
        finally:
            store.update_note = real_update

        assert state.get("deleted") == 200, "the delete never landed"
        assert resp.status_code == 404, resp.text
        assert resp.json()["error"]["code"] == "not_found", resp.text
        assert _count(
            "SELECT count(*) AS n FROM note WHERE id = %s", (note_id,)
        ) == 0

    def test_a_delete_before_the_request_answers_the_same_way(self, client, auth):
        """The ordering that already worked, so the two cannot diverge."""
        headers = auth["headers"]
        note_id = _note(client, headers, f"B {uuid.uuid4().hex[:6]}", "first body")
        assert client.delete(
            f"/v1/notes/{note_id}", headers=headers
        ).status_code == 200

        resp = client.patch(f"/v1/notes/{note_id}", headers=headers,
                            json={"content": "second body"})
        assert resp.status_code == 404, resp.text

    def test_a_delete_after_the_write_is_a_truthful_success(self, client, auth):
        """Not every interleaving is a failure.

        Here the update committed and the delete came second. Reporting that
        as success describes what happened; the note is gone because deleting
        it was the later act.
        """
        headers = auth["headers"]
        store = get_runtime().store
        note_id = _note(client, headers, f"C {uuid.uuid4().hex[:6]}", "first body")

        real_update = store.update_note
        state: dict = {}

        def _update_then_delete(*args, **kwargs):
            note = real_update(*args, **kwargs)
            if "deleted" not in state:
                state["deleted"] = client.delete(
                    f"/v1/notes/{note_id}", headers=headers
                ).status_code
            return note

        store.update_note = _update_then_delete
        try:
            resp = client.patch(f"/v1/notes/{note_id}", headers=headers,
                                json={"content": "second body"})
        finally:
            store.update_note = real_update

        assert state.get("deleted") == 200
        assert resp.status_code == 200, resp.text
        assert _count(
            "SELECT count(*) AS n FROM note WHERE id = %s", (note_id,)
        ) == 0, "the graph write resurrected the note"
        assert _count(
            "SELECT count(*) AS n FROM note_link "
            "WHERE src_note_id = %s OR dst_note_id = %s",
            (note_id, note_id),
        ) == 0


def _sweep_with_a_judgment(client, headers, user_id):
    """A sweep whose reason is drawn from what the model was shown."""
    import re

    runtime = get_runtime()
    real_generate = runtime.llm.generate

    def _witness(prompt, *args, **kwargs):
        found = re.search(r"codeword is (\S+)", str(prompt))
        tail = f" both describe the codeword {found.group(1)}" if found else " related"
        return {"content": f"EVOLVES -{tail}"}

    runtime.llm.generate = _witness
    try:
        return client.post("/v1/notes/sweep", headers=headers)
    finally:
        runtime.llm.generate = real_generate


class TestTheSweepArchiveIsAHistoricalSnapshot:
    """SPEC 19.6: self-contained, replayable, and not rewritten afterwards."""

    def _vault(self, client, headers):
        title = f"Migration plan {MARK}"
        note_id = _note(
            client, headers, title,
            "The migration plan covers the rollback path and the cutover "
            f"window. The agreed codeword is {MARK}. " * 10,
        )
        _note(
            client, headers, f"Cutover notes {uuid.uuid4().hex[:5]}",
            "The migration plan also covers the rollback path and cutover "
            "timing. A second account of the same work. " * 10,
        )
        return note_id, title

    def test_deleting_a_note_leaves_the_saved_report_untouched(
        self, client, auth
    ):
        headers = auth["headers"]
        note_id, title = self._vault(client, headers)
        sweep = _sweep_with_a_judgment(client, headers, auth["user_id"])
        assert sweep.status_code == 200, sweep.text

        reasons = [f.get("reason") for f in sweep.json()["data"]["findings"]]
        assert any(MARK in (r or "") for r in reasons), (
            f"the control failed: no judgment carried the marker: {reasons}"
        )
        before = _reports(auth["user_id"])
        assert before, "nothing was archived, so there is nothing to protect"

        assert client.delete(
            f"/v1/notes/{note_id}", headers=headers
        ).status_code == 200

        assert _reports(auth["user_id"]) == before, (
            "deleting a note rewrote an already-persisted sweep report"
        )
        listed = client.get("/v1/notes/sweeps", headers=headers)
        assert title in listed.text and MARK in listed.text, (
            "the archive stopped replaying what the witness judged"
        )

    def test_a_later_sweep_does_not_judge_the_deleted_note(self, client, auth):
        """The live vault is what moves on; the archive is what does not."""
        headers = auth["headers"]
        note_id, title = self._vault(client, headers)
        assert _sweep_with_a_judgment(
            client, headers, auth["user_id"]
        ).status_code == 200
        assert client.delete(
            f"/v1/notes/{note_id}", headers=headers
        ).status_code == 200

        again = _sweep_with_a_judgment(client, headers, auth["user_id"])
        assert again.status_code == 200, again.text
        data = again.json()["data"]
        named = {
            side.get("title")
            for finding in data["findings"]
            for side in (finding.get("a") or {}, finding.get("b") or {})
        }
        assert title not in named, (
            "a deleted note was judged by a sweep run after its deletion"
        )

    def test_erasing_the_account_removes_the_archive(self, client, auth):
        """The boundary that does take the history: the account itself."""
        import asyncio

        headers = auth["headers"]
        self._vault(client, headers)
        assert _sweep_with_a_judgment(
            client, headers, auth["user_id"]
        ).status_code == 200
        assert _reports(auth["user_id"]), "the control failed: nothing archived"

        runtime = get_runtime()
        assert asyncio.run(runtime.auth.delete_user(auth["user_id"]))

        assert _count(
            "SELECT count(*) AS n FROM sweep_report WHERE user_id = %s",
            (auth["user_id"],),
        ) == 0, "an erased account kept its sweep archive"
