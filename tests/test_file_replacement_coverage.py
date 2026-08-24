"""Replacing a file's bytes changes its generation, not its coverage.

A context covers a path. The bytes at that path are then replaced through the
ordinary upload endpoint, which is what a user does and which names no context.
Three things must hold afterwards, and they are one invariant seen from three
sides:

* the old text is no longer retrievable — a context that answers out of bytes
  the file no longer holds is answering out of something that does not exist;
* the new text is retrievable — the file did not silently leave the corpus;
* the context still covers the file — coverage is a property of the
  context/source relationship, and replacing bytes is not a statement about it.

Deliberately sequential, and deliberately black-box in its *actions*: every
step goes through the HTTP API, with no threads, no gate, no sleep, and no
reach into the engine. Concurrency was never needed to expose this. An earlier
attempt at the same invariant used two threads and a gated commit, and it hid
the defect on any machine where a directory listing happened to come back in a
convenient order — the test passed for a reason unrelated to its subject. This
one fails everywhere the invariant is broken.

Reads go to the store because the served surface has no chunk listing. That is
observation, not participation: the actions are the API's.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from liminallm.service.runtime import get_runtime

FIRST = b"# report\nTHE FIRST GENERATION\n" * 12
SECOND = b"# report\nTHE SECOND GENERATION\n" * 12


def _unique(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _account(client):
    email = f"{_unique('cov')}@example.com"
    resp = client.post(
        "/v1/auth/signup", json={"email": email, "password": "TestPassword123!"}
    )
    assert resp.status_code == 201, resp.text
    data = resp.json()["data"]
    return data["user_id"], {"Authorization": f"Bearer {data['access_token']}"}


def _context(client, headers) -> str:
    resp = client.post(
        "/v1/contexts",
        headers=headers,
        json={"name": _unique("ctx"), "description": "coverage"},
    )
    assert resp.status_code in (200, 201), resp.text
    return resp.json()["data"]["id"]


def _upload(client, headers, name, body, context_id=None):
    """The ordinary upload. `context_id` is absent unless a test means it."""
    data = {"context_id": context_id} if context_id else None
    return client.post(
        "/v1/files/upload",
        headers={**headers, "Idempotency-Key": _unique("k")},
        files={"file": (name, body, "text/markdown")},
        data=data,
    )


def _files_dir(runtime, user_id: str) -> Path:
    return Path(runtime.settings.shared_fs_root) / "users" / user_id / "files"


def _chunks_for(runtime, context_id: str, name: str):
    return [
        c
        for c in runtime.store.list_chunks(context_id, limit=500)
        if Path(c.fs_path or "").name == name
    ]


def _text_for(runtime, context_id: str, name: str) -> str:
    return " ".join(c.content or "" for c in _chunks_for(runtime, context_id, name))


def _cover_directory(client, headers, context_id, files_dir) -> None:
    resp = client.post(
        f"/v1/contexts/{context_id}/sources",
        headers=headers,
        json={"fs_path": str(files_dir), "recursive": False},
    )
    assert resp.status_code in (200, 201), resp.text


class TestReplacingBytesKeepsCoverage:
    def test_a_directory_source_still_describes_the_file_it_covers(self, client):
        """The whole invariant, against coverage acquired by directory source.

        This is the case the manifest cannot answer on its own: nothing about
        adding `files/` as a source writes an entry for `files/report.md`, so
        an implementation that asks the manifest which contexts cover the path
        is asking something that was never told.
        """
        runtime = get_runtime()
        user_id, headers = _account(client)
        context_id = _context(client, headers)

        assert _upload(client, headers, "report.md", FIRST).status_code == 200
        files_dir = _files_dir(runtime, user_id)
        _cover_directory(client, headers, context_id, files_dir)

        assert "THE FIRST GENERATION" in _text_for(runtime, context_id, "report.md"), (
            "the first generation was never indexed, so this test cannot say "
            "anything about the second"
        )

        assert _upload(client, headers, "report.md", SECOND).status_code == 200
        assert (files_dir / "report.md").read_bytes() == SECOND

        indexed = _text_for(runtime, context_id, "report.md")
        assert "THE FIRST GENERATION" not in indexed, (
            "the context still describes bytes the file no longer holds: "
            f"{indexed[:200]!r}"
        )
        assert "THE SECOND GENERATION" in indexed, (
            "the context stopped describing the file when it was replaced; "
            f"chunks now: {indexed[:200]!r}"
        )
        assert _chunks_for(runtime, context_id, "report.md"), (
            "the file left the context entirely"
        )

    def test_coverage_taken_by_naming_the_context_survives_a_later_replacement(
        self, client
    ):
        """The same invariant, against coverage acquired by naming a context.

        The second upload names nothing. Absence of a `context_id` is not a
        statement that the file now belongs to no context — it is the ordinary
        way to replace a file's bytes.
        """
        runtime = get_runtime()
        _user_id, headers = _account(client)
        context_id = _context(client, headers)

        assert (
            _upload(client, headers, "notes.md", FIRST, context_id=context_id
                    ).status_code == 200
        )
        assert "THE FIRST GENERATION" in _text_for(runtime, context_id, "notes.md")

        assert _upload(client, headers, "notes.md", SECOND).status_code == 200

        indexed = _text_for(runtime, context_id, "notes.md")
        assert "THE FIRST GENERATION" not in indexed, (
            f"stale generation retained: {indexed[:200]!r}"
        )
        assert "THE SECOND GENERATION" in indexed, (
            f"replacement not indexed: {indexed[:200]!r}"
        )

    def test_naming_a_context_on_replacement_adds_coverage_rather_than_narrowing_it(
        self, client
    ):
        """An explicit `context_id` may add a covering context. It may not
        remove the others."""
        runtime = get_runtime()
        user_id, headers = _account(client)
        covered_by_source = _context(client, headers)
        named_on_upload = _context(client, headers)

        assert _upload(client, headers, "shared.md", FIRST).status_code == 200
        files_dir = _files_dir(runtime, user_id)
        _cover_directory(client, headers, covered_by_source, files_dir)
        assert "THE FIRST GENERATION" in _text_for(
            runtime, covered_by_source, "shared.md"
        )

        assert (
            _upload(client, headers, "shared.md", SECOND,
                    context_id=named_on_upload).status_code == 200
        )

        added = _text_for(runtime, named_on_upload, "shared.md")
        assert "THE SECOND GENERATION" in added, (
            f"the named context did not receive the file: {added[:200]!r}"
        )
        kept = _text_for(runtime, covered_by_source, "shared.md")
        assert "THE SECOND GENERATION" in kept, (
            "naming one context on the upload narrowed coverage to it, "
            f"dropping the context that already covered the path: {kept[:200]!r}"
        )
        assert "THE FIRST GENERATION" not in kept, (
            f"stale generation retained in the covering context: {kept[:200]!r}"
        )
