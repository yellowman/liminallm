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
from liminallm.service.fs import path_lock, publication_key
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


def _publication_lock(runtime, fs_path: str, *, timeout: float = 0.1):
    """The lock an upload, a delete, or a queue job of this path would take."""
    return path_lock(
        _root(runtime), publication_key(_root(runtime), fs_path), timeout=timeout
    )


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


class TestDeletingATreeReachesInsideIt:
    """The recursive half, for the records the chunk test does not touch.

    The nested case above proves the lock key and that descendant *chunks* go.
    It says nothing about descendant source rows or descendant jobs, because
    its source names the tree itself and its job runs to completion before the
    deletion proceeds. So a future narrowing of either predicate — subtree
    match quietly becoming exact match — would leave `bundle/inner.md`'s own
    source row and its queued job behind while every other case here still
    passes.

    One tree, three records at three different depths, one delete.
    """

    def test_deleting_a_tree_takes_the_records_inside_it_and_not_above_it(
        self, client
    ):
        runtime = get_runtime()
        user_id, headers = _account(client)
        context_id = _context(client, headers)
        files_dir = _files_dir(runtime, user_id)
        files_dir.mkdir(parents=True, exist_ok=True)
        (files_dir / "bundle").mkdir(exist_ok=True)
        inner = files_dir / "bundle" / "inner.md"
        inner.write_bytes(BODY)
        keeper = files_dir / "keeper.md"
        keeper.write_bytes(b"# keeper\nSTILL HERE\n" * 12)

        # An ancestor directory source, which must survive.
        assert _add_source(
            client, headers, context_id, files_dir, recursive=True
        ).status_code in (200, 201)
        # And an exact-file source *inside* the tree, which must not.
        assert _add_source(
            client, headers, context_id, inner, recursive=False
        ).status_code in (200, 201)

        runtime.store.enqueue_ingest_job(
            context_id, str(inner), ingest_queue.generation_of(inner)
        )
        assert runtime.store.count_pending_ingest_jobs(str(inner)) == 1
        assert _chunks_under(runtime, context_id, str(inner)), "the tree was not indexed"

        assert _delete(client, headers, "bundle").status_code == 200

        sources = _sources(runtime, context_id)
        assert str(files_dir) in sources, (
            "deleting a tree removed the directory source above it, so the "
            "context has stopped covering everything else in that directory"
        )
        assert str(inner) not in sources, (
            "a source row inside the deleted tree survived it: the subtree "
            "match has narrowed to an exact match, so only the tree's own row "
            "would be taken"
        )
        assert runtime.store.count_pending_ingest_jobs(str(inner)) == 0, (
            "a re-index is still owed for a file inside a deleted tree"
        )
        assert not _chunks_under(runtime, context_id, str(inner))
        assert _chunks_under(runtime, context_id, str(keeper)), (
            "a file beside the deleted tree lost its chunks"
        )


class TestTheLockKeyIsAnchoredNotGuessed:
    """A tree may contain any names, including the ones the layout uses.

    The key has to identify the *user's* files directory, and a path shape is
    not an identification: an extracted archive is allowed to contain
    `users/x/files/`, and looking upward for the nearest thing shaped like one
    finds the archive's copy rather than the real root. The worker then locks
    a namespace inside the tree while a delete of the tree locks the tree, and
    the race this tranche closed is open again — reachable by unpacking an
    archive that happens to mirror the layout.

    So the root is not searched for. It is given.
    """

    def test_a_tree_containing_a_lookalike_layout_still_locks_the_tree(
        self, client, monkeypatch
    ):
        runtime = get_runtime()
        user_id, headers = _account(client)
        context_id = _context(client, headers)
        files_dir = _files_dir(runtime, user_id)
        # The nasty path: a real tree whose contents mirror the layout.
        inner = files_dir / "bundle" / "users" / "fake" / "files" / "inner.md"
        inner.parent.mkdir(parents=True, exist_ok=True)
        inner.write_bytes(BODY)

        assert _add_source(
            client, headers, context_id, files_dir / "bundle", recursive=True
        ).status_code in (200, 201)
        assert _chunks_under(runtime, context_id, str(inner)), "the tree was not indexed"
        runtime.store.enqueue_ingest_job(
            context_id, str(inner), ingest_queue.generation_of(inner)
        )

        indexing = threading.Event()
        may_finish = threading.Event()
        real = runtime.rag.ingest_file

        def gated(*args, **kwargs):
            indexing.set()
            assert may_finish.wait(30), "the deletion never released the gate"
            return real(*args, **kwargs)

        monkeypatch.setattr(runtime.rag, "ingest_file", gated)
        worker = threading.Thread(
            target=ingest_queue.drain,
            args=(runtime.store, runtime.rag),
            kwargs={"fs_root": _root(runtime)},
            daemon=True,
        )
        worker.start()
        try:
            assert indexing.wait(30), "the worker never reached its ingest"
            deleted: list = []

            def delete_it():
                deleted.append(_delete(client, headers, "bundle").status_code)

            deleter = threading.Thread(target=delete_it, daemon=True)
            deleter.start()
            time.sleep(0.5)
            assert not deleted, (
                "the delete ran straight through: the worker keyed its lock on "
                "the lookalike `users/fake/files` inside the tree rather than "
                "on the tree, so the two never met"
            )
        finally:
            may_finish.set()
            worker.join(timeout=30)
            deleter.join(timeout=30)

        assert deleted == [200], deleted
        assert not _chunks_under(runtime, context_id, str(inner))


class TestASupersededJobStaysSuperseded:
    """Standing aside must not revive a row somebody else has closed.

    A worker marks a job `running` when it claims it, and only then goes for
    the publication lock. A deletion holding that lock supersedes the job — it
    is entitled to, the path is going away — and the worker then times out and
    hands the job back. If handing back is an overwrite rather than a
    transition, it writes `queued` over `superseded` and the deletion's
    cancellation is undone by a worker that never touched the path.

    The job would normally then find the file missing and supersede itself, so
    this does not by itself restore deleted chunks. What it does is make the
    delete's guarantee false: if the same name with the same bytes reappears
    before that job runs, it ingests into a context whose exact source row the
    deletion already removed — derived state recreating itself with no
    authority behind it.
    """

    def test_a_worker_that_stands_aside_cannot_requeue_a_cancelled_job(
        self, client
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

        # The worker claims it: the row is `running` and the lock is not yet
        # taken, which is the window the deletion lands in.
        claimed = runtime.store.claim_ingest_jobs(1)
        assert len(claimed) == 1

        with _publication_lock(runtime, target, timeout=5.0):
            # What DELETE does while it holds the lock.
            assert runtime.store.cancel_ingest_jobs_under_path(user_id, target) == 1
            assert runtime.store.count_pending_ingest_jobs(target) == 0

            # And now the worker, which cannot have the lock, stands aside.
            assert ingest_queue.run_job(
                runtime.store, runtime.rag, claimed[0], fs_root=_root(runtime)
            ) is None

        assert runtime.store.count_pending_ingest_jobs(target) == 0, (
            "standing aside put a cancelled job back in the queue, so the "
            "deletion's cancellation was undone by a worker that never "
            "touched the path"
        )

    def test_a_failing_worker_cannot_requeue_a_cancelled_job_either(self, client):
        """The same rule on the failure path, not only the contention one.

        The deletion does its bookkeeping first and unlinks last — deliberately,
        so a half-failed delete leaves "nothing was deleted" rather than "the
        file is gone and still retrievable". That ordering leaves a real window
        where a job is already superseded and the file is still on disk, so a
        worker holding a claim gets past its generation check and into the
        ingest. When that ingest fails, putting the job back must be refused
        for the same reason standing aside is: knowing a row's id is not
        authority to revive it.
        """
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

        claimed = runtime.store.claim_ingest_jobs(1)
        assert len(claimed) == 1
        # The deletion's cancellation, at the moment before its unlink.
        assert runtime.store.cancel_ingest_jobs_under_path(user_id, target) == 1

        class Unreachable:
            def ingest_file(self, *args, **kwargs):
                raise RuntimeError("embedding backend unreachable")

        assert ingest_queue.run_job(
            runtime.store, Unreachable(), claimed[0], fs_root=_root(runtime)
        ) == 0
        assert runtime.store.count_pending_ingest_jobs(target) == 0, (
            "a failing worker put a cancelled job back in the queue"
        )
