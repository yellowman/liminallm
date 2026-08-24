"""Deleting a file must leave nothing that still describes its bytes.

The invariant, stated once: **after `DELETE /v1/files/{path}` returns success,
no retrievable state may describe the deleted bytes.** Chunks are the obvious
half and the one already handled. The rest is what this file is about.

Coverage is the interesting part, now that `context_source` is authoritative
for what a context covers. The two kinds of row mean different things and a
deletion must treat them differently:

* a **directory** source such as `files/` says "this context covers whatever
  is in here". Deleting one file inside it does not make that false, and the
  row must survive — if the name reappears, the context covers it again;
* an **exact-file** source says "this context covers this file". Deleting the
  file makes that a claim about something that no longer exists, so the row
  goes with it.

"Delete every source that covers this path" would satisfy the second and
destroy the first: one child deleted would collapse a whole directory's
coverage. So the distinction is the design, not an implementation detail.

Then there is the queue. A re-index job for a path that has just been deleted
must not be able to put its chunks back, and "must not" has to hold against a
job already *running* — past its own generation check, inside the extract and
embed, about to commit. Only the publication lock can order those two, which
is why deletion takes the same one an upload does.
"""

from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path

from liminallm.service import ingest_queue
from liminallm.service.fs import namespace_key, path_lock
from liminallm.service.runtime import get_runtime

BODY = b"# report\nTHE TEXT THAT WAS DELETED\n" * 12


def _unique(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _account(client):
    email = f"{_unique('del')}@example.com"
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
        json={"name": _unique("ctx"), "description": "deletion"},
    )
    assert resp.status_code in (200, 201), resp.text
    return resp.json()["data"]["id"]


def _upload(client, headers, name, body, context_id=None):
    data = {"context_id": context_id} if context_id else None
    return client.post(
        "/v1/files/upload",
        headers={**headers, "Idempotency-Key": _unique("k")},
        files={"file": (name, body, "text/markdown")},
        data=data,
    )


def _files_dir(runtime, user_id: str) -> Path:
    return Path(runtime.settings.shared_fs_root) / "users" / user_id / "files"


def _root(runtime) -> str:
    return str(runtime.settings.shared_fs_root)


def _add_source(client, headers, context_id, fs_path, *, recursive):
    return client.post(
        f"/v1/contexts/{context_id}/sources",
        headers=headers,
        json={"fs_path": str(fs_path), "recursive": recursive},
    )


def _sources(runtime, context_id: str) -> set:
    return {str(s.fs_path) for s in runtime.store.list_context_sources(context_id)}


def _chunks_under(runtime, context_id: str, prefix: str) -> list:
    return [
        c
        for c in runtime.store.list_chunks(context_id, limit=500)
        if str(c.fs_path or "") == prefix or str(c.fs_path or "").startswith(prefix + "/")
    ]


def _delete(client, headers, name):
    return client.delete(f"/v1/files/{name}", headers=headers)


class TestDeletionLeavesNoDescription:
    def test_a_directory_source_survives_the_deletion_of_one_file_in_it(
        self, client
    ):
        """The row says the context covers the directory. It still does.

        Removing it because a child was deleted would collapse coverage of
        every other file in that directory, and of every file later added to
        it — a deletion of one name silently un-indexing the rest.
        """
        runtime = get_runtime()
        user_id, headers = _account(client)
        context_id = _context(client, headers)

        assert _upload(client, headers, "report.md", BODY).status_code == 200
        assert _upload(client, headers, "keeper.md", b"# keeper\nSTILL HERE\n" * 12
                       ).status_code == 200
        files_dir = _files_dir(runtime, user_id)
        assert _add_source(
            client, headers, context_id, files_dir, recursive=False
        ).status_code in (200, 201)
        target = str(files_dir / "report.md")
        assert _chunks_under(runtime, context_id, target), (
            "the file was never indexed, so this test cannot say anything"
        )

        assert _delete(client, headers, "report.md").status_code == 200

        assert not (files_dir / "report.md").exists()
        assert not _chunks_under(runtime, context_id, target), (
            "the context still describes a file that no longer exists"
        )
        assert str(files_dir) in _sources(runtime, context_id), (
            "deleting one file removed the directory source, so the context "
            "has stopped covering every other file in that directory"
        )
        assert _chunks_under(runtime, context_id, str(files_dir / "keeper.md")), (
            "the other file in the directory lost its chunks too"
        )

    def test_an_exact_file_source_goes_with_the_file(self, client):
        """The row is a claim about one file. The file is gone."""
        runtime = get_runtime()
        user_id, headers = _account(client)
        context_id = _context(client, headers)

        assert _upload(client, headers, "report.md", BODY).status_code == 200
        files_dir = _files_dir(runtime, user_id)
        target = str(files_dir / "report.md")
        assert _add_source(
            client, headers, context_id, target, recursive=False
        ).status_code in (200, 201)
        assert target in _sources(runtime, context_id)

        assert _delete(client, headers, "report.md").status_code == 200

        assert not _chunks_under(runtime, context_id, target)
        assert target not in _sources(runtime, context_id), (
            "a source row still names a file that no longer exists, so the "
            "context claims to cover something deleted"
        )

    def test_a_queued_re_index_cannot_resurrect_a_deleted_file(self, client):
        """A job recorded before the deletion must not refill the index."""
        runtime = get_runtime()
        user_id, headers = _account(client)
        context_id = _context(client, headers)

        assert _upload(client, headers, "report.md", BODY).status_code == 200
        files_dir = _files_dir(runtime, user_id)
        target = str(files_dir / "report.md")
        assert _add_source(
            client, headers, context_id, files_dir, recursive=False
        ).status_code in (200, 201)

        runtime.store.enqueue_ingest_job(
            context_id, target, ingest_queue.generation_of(Path(target))
        )
        assert runtime.store.count_pending_ingest_jobs(target) == 1

        assert _delete(client, headers, "report.md").status_code == 200

        assert runtime.store.count_pending_ingest_jobs(target) == 0, (
            "a re-index is still owed for a file that no longer exists"
        )
        ingest_queue.drain_until_idle(
            runtime.store, runtime.rag, fs_root=_root(runtime)
        )
        assert not _chunks_under(runtime, context_id, target), (
            "the queue put back the contents of a deleted file"
        )


class TestDeletionSerializesAgainstTheQueue:
    """A job already inside its ingest is the case only a lock can order.

    The generation check at the top of a job stops a *queued* one from
    refilling a deleted path: it re-reads the file, finds nothing, and
    declines. It says nothing about a job that has already passed that check
    and is inside the extract and embed, holding bytes it read while the file
    still existed. That job's commit and the deletion's cleanup are two
    writes to the same rows, and whichever lands second decides whether a
    deleted file is still retrievable.

    So they must not overlap, and the only thing that can arrange it is the
    publication lock they are both required to take — on the *same* key.
    """

    def _gated_worker(self, runtime, monkeypatch):
        """Hold a job inside `ingest_file`, past its generation check."""
        inside = threading.Event()
        may_finish = threading.Event()
        real = runtime.rag.ingest_file

        def gated(*args, **kwargs):
            inside.set()
            assert may_finish.wait(30), "the deletion never released the gate"
            return real(*args, **kwargs)

        monkeypatch.setattr(runtime.rag, "ingest_file", gated)
        return inside, may_finish

    def test_a_root_file_deletion_waits_for_an_ingest_already_running(
        self, client, monkeypatch
    ):
        runtime = get_runtime()
        user_id, headers = _account(client)
        context_id = _context(client, headers)

        assert _upload(client, headers, "report.md", BODY).status_code == 200
        files_dir = _files_dir(runtime, user_id)
        target = str(files_dir / "report.md")
        assert _add_source(
            client, headers, context_id, files_dir, recursive=False
        ).status_code in (200, 201)
        runtime.store.enqueue_ingest_job(
            context_id, target, ingest_queue.generation_of(Path(target))
        )

        inside, may_finish = self._gated_worker(runtime, monkeypatch)
        worker = threading.Thread(
            target=ingest_queue.drain,
            args=(runtime.store, runtime.rag),
            kwargs={"fs_root": _root(runtime)},
            daemon=True,
        )
        worker.start()
        try:
            assert inside.wait(30), "the worker never reached its ingest"
            deleted: list = []

            def delete_it():
                deleted.append(_delete(client, headers, "report.md").status_code)

            deleter = threading.Thread(target=delete_it, daemon=True)
            deleter.start()
            time.sleep(0.5)
            assert not deleted, (
                "the deletion ran straight through while a job was mid-ingest "
                "on that path, so the two are not taking the same lock"
            )
        finally:
            may_finish.set()
            worker.join(timeout=30)
            deleter.join(timeout=30)

        assert deleted == [200], deleted
        assert not _chunks_under(runtime, context_id, target), (
            "the job's commit outlived the deletion that came after it"
        )

    def test_a_file_inside_a_tree_uses_the_same_key_the_tree_delete_uses(
        self, client, monkeypatch
    ):
        """The key is the tree, not the file — on both sides.

        `namespace_key` locks a name's *first component* precisely so that a
        recursive delete of `bundle` and a mutation of `bundle/inner.md` meet.
        A queue that keyed on the file's own directory would take a lock the
        delete never takes, and the two would run straight through each other.

        The tree is written directly because an extractor is what produces
        one, and this test is about what happens to it afterwards.
        """
        runtime = get_runtime()
        user_id, headers = _account(client)
        context_id = _context(client, headers)
        files_dir = _files_dir(runtime, user_id)
        files_dir.mkdir(parents=True, exist_ok=True)
        (files_dir / "bundle").mkdir(exist_ok=True)
        inner = files_dir / "bundle" / "inner.md"
        inner.write_bytes(BODY)

        assert _add_source(
            client, headers, context_id, files_dir / "bundle", recursive=True
        ).status_code in (200, 201)
        target = str(inner)
        assert _chunks_under(runtime, context_id, target), "the tree was not indexed"

        runtime.store.enqueue_ingest_job(
            context_id, target, ingest_queue.generation_of(inner)
        )
        inside, may_finish = self._gated_worker(runtime, monkeypatch)
        worker = threading.Thread(
            target=ingest_queue.drain,
            args=(runtime.store, runtime.rag),
            kwargs={"fs_root": _root(runtime)},
            daemon=True,
        )
        worker.start()
        try:
            assert inside.wait(30), "the worker never reached its ingest"
            deleted: list = []

            def delete_it():
                deleted.append(_delete(client, headers, "bundle").status_code)

            deleter = threading.Thread(target=delete_it, daemon=True)
            deleter.start()
            time.sleep(0.5)
            assert not deleted, (
                "deleting the tree ran straight through a job indexing a file "
                "inside it: the queue keyed the lock on the file's own "
                "directory, which is not the key the delete takes"
            )
        finally:
            may_finish.set()
            worker.join(timeout=30)
            deleter.join(timeout=30)

        assert deleted == [200], deleted
        assert not _chunks_under(runtime, context_id, target), (
            "a file inside a deleted tree is still retrievable"
        )
