"""Replacing a file's bytes changes its generation, not its coverage.

A context covers a path. The bytes at that path are then replaced through the
ordinary upload endpoint, which is what a user does and which names no context.
Three things must hold afterwards, and they are one invariant seen from three
sides:

* the old text is never retrievable again — a context that answers out of bytes
  the file no longer holds is answering out of something that does not exist;
* the new text becomes retrievable — the file did not silently leave the
  corpus;
* the context still covers the file — coverage is a property of the
  context/source relationship, and replacing bytes is not a statement about it.

The first and second hold at different times, on purpose. Dropping the old
chunks is cheap and happens during the request; re-reading and re-embedding the
file for every context that covers it is not bounded by anything the request
chose, so it happens out of band. In between, the path is *absent* from those
contexts. These tests therefore wait for the refresh rather than reading once —
but the wait is not a blind sleep: the old generation is checked for at every
observation, so an implementation that briefly reinstates it fails here even
though the end state looks right.

Deliberately sequential, and deliberately black-box in its *actions*: every
step goes through the HTTP API, with no threads, no gate, and no reach into the
engine. Concurrency was never needed to expose this. An earlier attempt at the
same invariant used two threads and a gated commit, and it hid the defect on
any machine where a directory listing happened to come back in a convenient
order — the test passed for a reason unrelated to its subject. This one fails
everywhere the invariant is broken.

Reads go to the store because the served surface has no chunk listing. That is
observation, not participation: the actions are the API's.
"""

from __future__ import annotations

import contextlib
import threading
import time
import uuid
from pathlib import Path

from liminallm.api import routes
from liminallm.service import ingest_queue
from liminallm.service.fs import PathLockTimeout, path_lock, publication_key
from liminallm.service.runtime import get_runtime

# Long enough that a slow box does not fail an implementation that is merely
# unhurried, short enough that one that never refreshes is reported quickly.
REFRESH_TIMEOUT_SECONDS = 20.0

FIRST = b"# report\nTHE FIRST GENERATION\n" * 12
SECOND = b"# report\nTHE SECOND GENERATION\n" * 12


def _root(runtime) -> str:
    """The shared filesystem root, which is what keys the publication lock."""
    return str(runtime.settings.shared_fs_root)


def _publication_lock(runtime, fs_path: str, *, timeout: float = 0.1):
    """The same lock, on the same key, that an upload of this path takes.

    Through `publication_key`, not a second derivation of it: a test that
    computes the key its own way can agree with production at one depth and
    diverge at another, which is exactly the bug this helper is used to catch.
    """
    return path_lock(
        _root(runtime), publication_key(_root(runtime), fs_path), timeout=timeout
    )


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


def _state(runtime, context_id: str, name: str) -> str:
    """Both halves of the invariant in one line, for the failure message.

    Whether the old text is gone and whether the new text arrived are separate
    facts with separate causes, and asserting them one after another reports
    only the first. A replacement that indexed without invalidating and one
    that did nothing at all both fail the same assertion, and they are
    different bugs — so every message carries both.
    """
    indexed = _text_for(runtime, context_id, name)
    return (
        f"[old_present={'THE FIRST GENERATION' in indexed} "
        f"new_present={'THE SECOND GENERATION' in indexed} "
        f"chunks={len(_chunks_for(runtime, context_id, name))}] "
        f"{indexed[:160]!r}"
    )


def _sources(runtime, context_id: str) -> set:
    """The context/source relationships, read from the table that holds them."""
    return {
        (str(s.fs_path), bool(s.recursive))
        for s in runtime.store.list_context_sources(context_id)
    }


def _wait_for_refresh(runtime, context_id: str, name: str) -> str:
    """Wait for the deferred re-index, holding the invariant the whole time.

    Absent is a legal intermediate state; wrong is not. The old generation is
    checked for at every observation, not only at the end, so an implementation
    that puts the stale chunks back — briefly, or by racing its own queue —
    fails here rather than passing on its final state.
    """
    deadline = time.monotonic() + REFRESH_TIMEOUT_SECONDS
    while True:
        indexed = _text_for(runtime, context_id, name)
        assert "THE FIRST GENERATION" not in indexed, (
            "the context described bytes the file no longer holds: "
            f"{_state(runtime, context_id, name)}"
        )
        if "THE SECOND GENERATION" in indexed or time.monotonic() >= deadline:
            return indexed
        time.sleep(0.05)


def _pending_job_id(runtime, fs_path: str) -> str:
    """The one queued job for a path. Asserts there is exactly one."""
    with runtime.store._connect() as conn:
        rows = conn.execute(
            "SELECT id FROM ingest_job WHERE fs_path = %s AND status = 'queued'",
            (fs_path,),
        ).fetchall()
    assert len(rows) == 1, f"expected one pending job, found {len(rows)}"
    return str(rows[0]["id"])


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

        sources_before = _sources(runtime, context_id)

        assert _upload(client, headers, "report.md", SECOND).status_code == 200
        assert (files_dir / "report.md").read_bytes() == SECOND

        indexed = _wait_for_refresh(runtime, context_id, "report.md")
        state = _state(runtime, context_id, "report.md")
        assert "THE SECOND GENERATION" in indexed, (
            f"the context never got the replaced file back: {state}"
        )
        assert sources_before <= _sources(runtime, context_id), (
            "replacing the bytes removed the context/source relationship that "
            "made this context cover the file"
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
        sources_before = _sources(runtime, context_id)

        assert _upload(client, headers, "notes.md", SECOND).status_code == 200

        indexed = _wait_for_refresh(runtime, context_id, "notes.md")
        assert "THE SECOND GENERATION" in indexed, (
            f"replacement never indexed: {_state(runtime, context_id, 'notes.md')}"
        )
        assert sources_before <= _sources(runtime, context_id), (
            "replacing the bytes removed the context/source relationship the "
            "first upload created"
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

        sources_before = _sources(runtime, covered_by_source)

        assert (
            _upload(client, headers, "shared.md", SECOND,
                    context_id=named_on_upload).status_code == 200
        )

        # The named context is the one ingest the request asked for, so it is
        # done by the time the response returns.
        added = _text_for(runtime, named_on_upload, "shared.md")
        assert "THE SECOND GENERATION" in added, (
            "the named context did not receive the file: "
            f"{_state(runtime, named_on_upload, 'shared.md')}"
        )

        kept = _wait_for_refresh(runtime, covered_by_source, "shared.md")
        assert "THE SECOND GENERATION" in kept, (
            "naming one context on the upload narrowed coverage to it, dropping "
            f"the context that already covered the path: "
            f"{_state(runtime, covered_by_source, 'shared.md')}"
        )
        assert sources_before <= _sources(runtime, covered_by_source), (
            "naming a different context removed the source relationship of the "
            "context that already covered the path"
        )


class TestTheWindowBetweenThem:
    """What the context holds while the re-read is still owed.

    The tests above wait for the refresh, so they see only the end state, and
    an implementation that re-indexed everything inside the request would
    satisfy them. It would also spend an amount of work no caller chose — one
    extract-and-embed per covering context — before answering an upload. So
    the deferral is deliberate, and it has its own requirement: during the
    window the path is *absent* from those contexts, never stale, and the work
    is *recorded*, never dropped.

    The drain is suspended here to hold that window open. That is not a
    contrivance: an un-drained queue is the ordinary state of the system
    between a replacement and the next worker poll.
    """

    def test_the_old_generation_goes_immediately_and_the_work_is_recorded(
        self, client, monkeypatch
    ):
        runtime = get_runtime()
        user_id, headers = _account(client)
        context_id = _context(client, headers)

        assert _upload(client, headers, "report.md", FIRST).status_code == 200
        files_dir = _files_dir(runtime, user_id)
        _cover_directory(client, headers, context_id, files_dir)
        assert "THE FIRST GENERATION" in _text_for(runtime, context_id, "report.md")
        sources_before = _sources(runtime, context_id)
        path = str(files_dir / "report.md")

        # Hold the window open: the request records the work but nothing runs
        # it. `drain` is kept by reference first, because suspending it by
        # module attribute suspends it for this test too.
        run_the_queue = ingest_queue.drain
        monkeypatch.setattr(ingest_queue, "drain", lambda *a, **k: 0)
        assert _upload(client, headers, "report.md", SECOND).status_code == 200

        state = _state(runtime, context_id, "report.md")
        assert not _chunks_for(runtime, context_id, "report.md"), (
            "the replaced file is still described by the chunks of a "
            f"generation that no longer exists: {state}"
        )
        assert runtime.store.count_pending_ingest_jobs(path) == 1, (
            "nothing owes this context a re-read, so the file is not "
            "temporarily absent — it is permanently gone"
        )
        assert sources_before <= _sources(runtime, context_id), (
            "the context stopped covering the path while waiting to re-read it"
        )

        # What the worker does on its next poll.
        assert run_the_queue(runtime.store, runtime.rag, fs_root=_root(runtime)) == 1
        assert runtime.store.count_pending_ingest_jobs(path) == 0
        indexed = _text_for(runtime, context_id, "report.md")
        assert "THE SECOND GENERATION" in indexed, f"the queue ran and lost the file: {indexed[:160]!r}"
        assert "THE FIRST GENERATION" not in indexed

    def test_a_job_queued_for_bytes_that_are_gone_declines_to_write(
        self, client, monkeypatch
    ):
        """Generation-awareness, which is what makes deferral safe.

        Two replacements land before the queue is drained. The pending slot
        holds one job, and it must index the *last* bytes: a job that indexed
        whatever it was queued for would put a superseded generation back into
        the context, which is the failure the immediate invalidation exists to
        prevent, arriving late instead of never.
        """
        runtime = get_runtime()
        user_id, headers = _account(client)
        context_id = _context(client, headers)

        assert _upload(client, headers, "report.md", FIRST).status_code == 200
        files_dir = _files_dir(runtime, user_id)
        _cover_directory(client, headers, context_id, files_dir)
        path = str(files_dir / "report.md")

        run_the_queue = ingest_queue.drain
        monkeypatch.setattr(ingest_queue, "drain", lambda *a, **k: 0)
        assert _upload(client, headers, "report.md", SECOND).status_code == 200
        third = b"# report\nTHE THIRD GENERATION\n" * 12
        assert _upload(client, headers, "report.md", third).status_code == 200

        assert runtime.store.count_pending_ingest_jobs(path) == 1, (
            "two replacements left two re-reads owed, one of them for bytes "
            "already gone"
        )
        assert run_the_queue(runtime.store, runtime.rag, fs_root=_root(runtime)) == 1
        indexed = _text_for(runtime, context_id, "report.md")
        assert "THE THIRD GENERATION" in indexed, f"the queue indexed the wrong generation: {indexed[:160]!r}"
        assert "THE SECOND GENERATION" not in indexed
        assert "THE FIRST GENERATION" not in indexed

    def test_a_job_whose_generation_is_gone_is_not_indexed_at_all(self, client):
        """A job holding a stale checksum writes nothing, rather than writing
        the current bytes under an old job's authority.

        Queued directly, because the route can only produce a job whose
        generation matches what it just wrote. This is the state a crash
        between the write and the enqueue leaves behind, and the state a
        replacement that happens while a job is already running leaves behind.
        """
        runtime = get_runtime()
        user_id, headers = _account(client)
        context_id = _context(client, headers)

        assert _upload(client, headers, "report.md", FIRST).status_code == 200
        files_dir = _files_dir(runtime, user_id)
        _cover_directory(client, headers, context_id, files_dir)
        path = str(files_dir / "report.md")
        runtime.store.replace_chunks_for_path(context_id, path, [])

        runtime.store.enqueue_ingest_job(context_id, path, "0" * 64)
        assert ingest_queue.drain(runtime.store, runtime.rag, fs_root=_root(runtime)) == 1
        assert not _chunks_for(runtime, context_id, "report.md"), (
            "a job queued for bytes that are not on disk indexed anyway"
        )

    def test_a_job_that_fails_goes_back_in_the_queue(self, client, monkeypatch):
        """A failed re-index is retried, because the alternative is losing it.

        Failures here are transient far more often than not — the database
        blinked, the encoder was briefly unreachable. Abandoning the job on
        the first one would leave the path missing from that context until
        somebody happened to replace the file again, which is precisely the
        permanent forgetting this design exists to prevent.
        """
        runtime = get_runtime()
        user_id, headers = _account(client)
        context_id = _context(client, headers)

        assert _upload(client, headers, "report.md", FIRST).status_code == 200
        files_dir = _files_dir(runtime, user_id)
        _cover_directory(client, headers, context_id, files_dir)
        path = str(files_dir / "report.md")

        run_the_queue = ingest_queue.drain
        monkeypatch.setattr(ingest_queue, "drain", lambda *a, **k: 0)
        assert _upload(client, headers, "report.md", SECOND).status_code == 200

        class FailsOnce:
            """The real engine, unavailable exactly once.

            It wraps the real service rather than standing in for it, so the
            run that succeeds is a real ingest and this test can tell the
            difference between recovering and merely not crashing.
            """

            def __init__(self, real):
                self.real = real
                self.failed = False

            def ingest_file(self, *args, **kwargs):
                if not self.failed:
                    self.failed = True
                    raise RuntimeError("embedding backend unreachable")
                return self.real.ingest_file(*args, **kwargs)

        flaky = FailsOnce(runtime.rag)
        assert run_the_queue(runtime.store, flaky, fs_root=_root(runtime)) == 1
        assert flaky.failed
        assert runtime.store.count_pending_ingest_jobs(path) == 1, (
            "the job was abandoned on its first failure, so this context has "
            "lost the file until someone replaces it again"
        )
        assert not _chunks_for(runtime, context_id, "report.md"), (
            "a failed job left chunks behind"
        )

        # Still owed, but not yet due: the retry is scheduled into the future
        # so a worker cannot spend the whole budget in one pass.
        assert run_the_queue(runtime.store, flaky, fs_root=_root(runtime)) == 0
        with runtime.store._connect() as conn:
            conn.execute(
                "UPDATE ingest_job SET next_attempt_at = now() WHERE fs_path = %s",
                (path,),
            )

        assert run_the_queue(runtime.store, flaky, fs_root=_root(runtime)) == 1
        assert "THE SECOND GENERATION" in _text_for(runtime, context_id, "report.md"), (
            f"the retry did not index the file: {_state(runtime, context_id, 'report.md')}"
        )
        assert runtime.store.count_pending_ingest_jobs(path) == 0


class TestWhatCoversWhat:
    """The boundary of `contexts_covering_path`, which decides who gets a job.

    Read straight from the store, because the subject is the predicate itself
    rather than the endpoint's use of it. Every context here is real and owned
    — created through the API — because owner scoping is one of the properties
    under test and a stand-in with no owner could not express it.

    Getting this wrong is not a cosmetic error. Too narrow, and a replaced file
    silently leaves a context that covers it; too wide, and a replacement
    re-indexes into contexts nobody added it to.
    """

    def _ctx(self, client, headers, base: Path, sub: str, recursive: bool) -> str:
        context_id = _context(client, headers)
        runtime = get_runtime()
        runtime.store.add_context_source(
            context_id=context_id, fs_path=str(base / sub), recursive=recursive
        )
        return context_id

    def test_a_non_recursive_directory_covers_its_files_and_stops_there(
        self, client
    ):
        runtime = get_runtime()
        user_id, headers = _account(client)
        base = _files_dir(runtime, user_id).parent
        shallow = self._ctx(client, headers, base, "files", recursive=False)
        deep = self._ctx(client, headers, base, "files", recursive=True)

        def covering(rel: str) -> list:
            return runtime.store.contexts_covering_path(
                str(base / rel), owner_user_id=user_id
            )

        assert shallow in covering("files/report.md"), (
            "a directory source does not cover the files in it, so replacing "
            "one of them would refresh nothing"
        )
        assert shallow not in covering("files/sub/deep.md"), (
            "a source added non-recursively reached a grandchild anyway; "
            "`recursive` is the depth the owner chose and it was ignored"
        )
        assert deep in covering("files/sub/deep.md"), (
            "a recursive source stopped at its immediate children"
        )

    def test_a_directory_is_not_covered_by_one_whose_name_it_starts_with(
        self, client
    ):
        """`files` and `files_archive` are siblings, not parent and child.

        The containment test is on path components, not on characters. String
        prefixes would put every `files_*` directory inside `files`, and the
        replacement of a file in one would rewrite contexts belonging to the
        other.
        """
        runtime = get_runtime()
        user_id, headers = _account(client)
        base = _files_dir(runtime, user_id).parent
        files = self._ctx(client, headers, base, "files", recursive=True)

        covering = runtime.store.contexts_covering_path(
            str(base / "files_archive" / "report.md"), owner_user_id=user_id
        )
        assert files not in covering, (
            "a source on `files` claimed a path in `files_archive`"
        )

    def test_a_path_holding_sql_wildcards_matches_only_itself(self, client):
        """`%` and `_` are ordinary characters in a filename.

        A SQL `LIKE` built from a path would read them as wildcards, and
        `a_b` would silently cover `axb`. The test is written in Python for
        exactly this reason, and this is the witness for it.
        """
        runtime = get_runtime()
        user_id, headers = _account(client)
        base = _files_dir(runtime, user_id).parent
        wild = self._ctx(client, headers, base, "a_b", recursive=True)

        def covering(rel: str) -> list:
            return runtime.store.contexts_covering_path(
                str(base / rel), owner_user_id=user_id
            )

        assert wild in covering("a_b/report.md"), "the literal path stopped matching"
        assert wild not in covering("axb/report.md"), (
            "`_` was treated as a wildcard, so a source on `a_b` claimed `axb`"
        )

    def test_another_owners_context_never_covers_this_owners_path(self, client):
        """Owner scoping, enforced where coverage is decided.

        `knowledge_context` is owned by one user, and a replacement must only
        ever refresh that user's own contexts. Left to the caller, this would
        be one forgotten argument away from letting an upload write into a
        stranger's index — so the scope is applied here, in the query, not
        by whoever calls it.
        """
        runtime = get_runtime()
        mine_id, mine_headers = _account(client)
        _other_id, other_headers = _account(client)
        base = _files_dir(runtime, mine_id).parent

        # A row saying another account's context covers this account's
        # directory. It is written to the store directly because the API
        # refuses to create one — asserted at the end — and what is under
        # test here is what the query does if such a row exists anyway.
        intruder = _context(client, other_headers)
        runtime.store.add_context_source(
            context_id=intruder, fs_path=str(base / "files"), recursive=True
        )
        mine = self._ctx(client, mine_headers, base, "files", recursive=True)

        covering = runtime.store.contexts_covering_path(
            str(base / "files" / "report.md"), owner_user_id=mine_id
        )
        assert mine in covering
        assert intruder not in covering, (
            "a context belonging to another account was returned as covering "
            "this account's path, so replacing the file would have written "
            "into that account's index"
        )

        # The scope above is the second line of defence. The first is that the
        # source cannot be added through the API at all, because the path is
        # resolved against the requesting account's own directory.
        refused = client.post(
            f"/v1/contexts/{intruder}/sources",
            headers=other_headers,
            json={"fs_path": str(base / "files"), "recursive": True},
        )
        assert refused.status_code in (400, 403), refused.text


class TestTheQueueDoesNotLoseWork:
    """Three ways a job could vanish, each of which breaks the same promise.

    The queue's whole reason to exist is that a path is *temporarily* absent
    from a context. Every one of these turns "temporarily" into "until someone
    happens to replace the file again", which is the defect this tranche
    started from, arriving by a different route.
    """

    def _replaced_with_drain_suspended(self, client, monkeypatch):
        """Set up a covering context, replace the bytes, hold the queue."""
        runtime = get_runtime()
        user_id, headers = _account(client)
        context_id = _context(client, headers)
        assert _upload(client, headers, "report.md", FIRST).status_code == 200
        files_dir = _files_dir(runtime, user_id)
        _cover_directory(client, headers, context_id, files_dir)

        run_the_queue = ingest_queue.drain
        monkeypatch.setattr(ingest_queue, "drain", lambda *a, **k: 0)
        assert _upload(client, headers, "report.md", SECOND).status_code == 200
        return runtime, context_id, str(files_dir / "report.md"), run_the_queue

    def test_a_job_claimed_by_a_process_that_died_is_taken_back(
        self, client, monkeypatch
    ):
        """Claiming marks a job running, and only queued jobs are claimed.

        A process killed after claiming would strand it there forever. Because
        the replacement has already dropped the chunks, that is not a delayed
        re-index — it is a file permanently absent from a context that covers
        it, caused by the bookkeeping meant to prevent exactly that.
        """
        runtime, context_id, path, _drain = self._replaced_with_drain_suspended(
            client, monkeypatch
        )

        claimed = runtime.store.claim_ingest_jobs(1)
        assert len(claimed) == 1
        assert runtime.store.count_pending_ingest_jobs(path) == 1  # 'running'
        # The process dies here: nothing ever calls finish_ingest_job.

        # Not yet: a job in flight is not a job abandoned.
        assert runtime.store.reclaim_stale_ingest_jobs(900, max_attempts=ingest_queue.MAX_ATTEMPTS) == 0
        with runtime.store._connect() as conn:
            conn.execute(
                "UPDATE ingest_job SET updated_at = now() - interval '1 hour' "
                "WHERE id = %s",
                (str(claimed[0]["id"]),),
            )

        # Through the worker's own entry point, not the store method it calls:
        # a reclaim nothing invokes is the same as no reclaim at all.
        monkeypatch.undo()
        assert ingest_queue.drain_until_idle(runtime.store, runtime.rag, fs_root=_root(runtime)) == 1, (
            "the worker's pass did not take back the abandoned job, so the "
            "file stays absent from a context that covers it"
        )
        assert "THE SECOND GENERATION" in _text_for(runtime, context_id, "report.md"), (
            "the reclaimed job did not re-index the file: "
            f"{_state(runtime, context_id, 'report.md')}"
        )

    def test_an_unreadable_file_is_retried_and_a_missing_one_is_not(
        self, client, monkeypatch
    ):
        """"Deleted" and "briefly unreadable" call for opposite responses.

        A bare `except OSError` cannot tell them apart, and treating both as
        "the file is gone" drops the re-index for good on an NFS blip.
        """
        runtime, context_id, path, run_the_queue = (
            self._replaced_with_drain_suspended(client, monkeypatch)
        )

        real_read = Path.read_bytes

        def unreadable(self, *args, **kwargs):
            if str(self) == path:
                raise OSError(5, "Input/output error")
            return real_read(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_bytes", unreadable)
        assert run_the_queue(runtime.store, runtime.rag, fs_root=_root(runtime)) == 1
        assert runtime.store.count_pending_ingest_jobs(path) == 1, (
            "a read error was read as a deletion, so the re-index was dropped"
        )

        monkeypatch.undo()
        with runtime.store._connect() as conn:
            conn.execute(
                "UPDATE ingest_job SET next_attempt_at = now() WHERE fs_path = %s",
                (path,),
            )
        assert run_the_queue(runtime.store, runtime.rag, fs_root=_root(runtime)) == 1
        assert "THE SECOND GENERATION" in _text_for(runtime, context_id, "report.md")

        # The other half: a file that is genuinely gone is not owed a retry.
        Path(path).unlink()
        runtime.store.enqueue_ingest_job(context_id, path, "0" * 64)
        assert run_the_queue(runtime.store, runtime.rag, fs_root=_root(runtime)) == 1
        assert runtime.store.count_pending_ingest_jobs(path) == 0, (
            "a deleted file was retried as though it might come back"
        )

    def test_a_path_already_being_indexed_is_left_to_whoever_holds_it(
        self, client, monkeypatch
    ):
        """One writer per (context, path), enforced across processes.

        Dropping a path's chunks and writing the new ones is two statements.
        A second worker running the same two for the same path can land its
        write between them, and the loser's rows then outlive the winner's
        delete: a generation that is gone, still in the index, with nothing
        queued to correct it. The lock is what makes that impossible; the job
        that cannot take it is owed another go rather than allowed to proceed.
        """
        runtime, context_id, path, run_the_queue = (
            self._replaced_with_drain_suspended(client, monkeypatch)
        )

        with _publication_lock(runtime, path, timeout=5.0):
            # Claimed and handed straight back, so nothing was attempted.
            assert run_the_queue(runtime.store, runtime.rag, fs_root=_root(runtime)) == 0
            assert not _chunks_for(runtime, context_id, "report.md"), (
                "a second worker rewrote a path another one was already "
                "indexing: "
                f"{_state(runtime, context_id, 'report.md')}"
            )
            assert runtime.store.count_pending_ingest_jobs(path) == 1, (
                "the job yielded the path and then forgot it was owed"
            )

        # Released. The job that stood aside now does the work.
        assert run_the_queue(runtime.store, runtime.rag, fs_root=_root(runtime)) == 1
        assert "THE SECOND GENERATION" in _text_for(runtime, context_id, "report.md")

    def test_waiting_for_another_worker_is_not_a_failed_attempt(
        self, client, monkeypatch
    ):
        """Standing aside is not the same as trying and failing.

        A worker's pass keeps draining until the queue is empty, so a job that
        yields a held path is re-claimed straight away — many times over while
        the holder is still embedding. If each of those spends one of the
        job's attempts, the budget is gone in milliseconds and the job is
        abandoned, which leaves the path missing from a context that covers
        it. Contention has to cost nothing.
        """
        runtime, context_id, path, _drain = self._replaced_with_drain_suspended(
            client, monkeypatch
        )
        monkeypatch.undo()

        with _publication_lock(runtime, path, timeout=5.0):
            # The worker's whole pass, looping against a path it cannot have.
            assert ingest_queue.drain_until_idle(runtime.store, runtime.rag, fs_root=_root(runtime)) == 0
            assert runtime.store.count_pending_ingest_jobs(path) == 1, (
                "waiting for the lock holder used up the job's attempts, so "
                "the re-index was abandoned and the file is now missing from "
                "a context that covers it"
            )

        assert ingest_queue.drain_until_idle(runtime.store, runtime.rag, fs_root=_root(runtime)) == 1
        assert "THE SECOND GENERATION" in _text_for(runtime, context_id, "report.md")

    def test_a_replacement_does_not_inherit_an_earlier_failures_backoff(
        self, client, monkeypatch
    ):
        """A new replacement is due now, not when the job it displaced was.

        Collapsing onto the pending slot resets the attempt count, so leaving
        the due time alone contradicts it: the new bytes would sit unindexed
        for as long as the previous failure had earned, up to the ceiling.
        """
        runtime, context_id, path, run_the_queue = (
            self._replaced_with_drain_suspended(client, monkeypatch)
        )
        # What a failure leaves behind, reached the way a failure reaches it:
        # a job is claimed, fails, and is put back with a delay. Requeueing is
        # a `running -> queued` transition, so the claim is not decoration —
        # pushing a never-claimed row's due time out would be arranging a
        # state the system does not produce.
        claimed = runtime.store.claim_ingest_jobs(1)
        assert len(claimed) == 1
        job_id = str(claimed[0]["id"])
        assert runtime.store.requeue_ingest_job(
            job_id, detail="embedding backend unreachable", delay_seconds=3600
        )
        assert run_the_queue(runtime.store, runtime.rag, fs_root=_root(runtime)) == 0, (
            "a job that is not due for an hour was claimed anyway, so this "
            "test cannot say anything about what a replacement does to it"
        )

        # What a replacement does: the same call the upload route makes.
        runtime.store.enqueue_ingest_job(
            context_id, path, ingest_queue.generation_of(Path(path))
        )

        assert run_the_queue(runtime.store, runtime.rag, fs_root=_root(runtime)) == 1, (
            "the replacement collapsed onto the delayed job and inherited its "
            "backoff, so the new bytes stay unindexed for an hour"
        )
        assert "THE SECOND GENERATION" in _text_for(runtime, context_id, "report.md")


class TestTheProducerIsAlsoAWriter:
    """A replacement is a rewrite of the same rows, so it takes the same lock.

    `run_job` serialises workers against each other. It is not enough. The
    upload route deletes a path's chunks and queues new work without holding
    that lock, and `ingest_file` extracts the file into memory well before it
    writes the chunks — so a replacement can land in between and have its
    invalidation undone by a write that was already in flight.
    """

    def test_a_replacement_cannot_be_undone_by_a_write_already_in_flight(
        self, client, monkeypatch
    ):
        """Gated, not raced: the interleave is forced, so this is decisive.

        The worker is held after it has read the old bytes and before it
        writes them as chunks. The replacement happens in that gap, through
        the ordinary endpoint. Then the worker is released. Afterwards the old
        generation must not be answerable — that is the whole promise.
        """
        import threading

        runtime = get_runtime()
        user_id, headers = _account(client)
        context_id = _context(client, headers)
        assert _upload(client, headers, "report.md", FIRST).status_code == 200
        files_dir = _files_dir(runtime, user_id)
        _cover_directory(client, headers, context_id, files_dir)
        path = str(files_dir / "report.md")

        run_the_queue = ingest_queue.drain
        monkeypatch.setattr(ingest_queue, "drain", lambda *a, **k: 0)

        # A job for the *current* bytes, so the worker's generation check
        # passes and it proceeds to write.
        runtime.store.replace_chunks_for_path(context_id, path, [])
        runtime.store.enqueue_ingest_job(
            context_id, path, ingest_queue.generation_of(Path(path))
        )

        extracted = threading.Event()
        may_write = threading.Event()
        real_ingest_text = runtime.rag.ingest_text

        def gated_ingest_text(*args, **kwargs):
            """The point after the file is read and before the chunks land."""
            extracted.set()
            assert may_write.wait(30), "the replacement never released the gate"
            return real_ingest_text(*args, **kwargs)

        monkeypatch.setattr(runtime.rag, "ingest_text", gated_ingest_text)
        worker = threading.Thread(
            target=run_the_queue,
            args=(runtime.store, runtime.rag),
            kwargs={"fs_root": _root(runtime)},
            daemon=True,
        )
        worker.start()
        assert extracted.wait(30), "the worker never reached the write"

        # The replacement, through the ordinary endpoint, while that write is
        # still in flight.
        replaced = threading.Thread(
            target=lambda: _upload(client, headers, "report.md", SECOND), daemon=True
        )
        replaced.start()
        replaced.join(30)
        may_write.set()
        worker.join(30)
        assert not worker.is_alive()

        monkeypatch.setattr(runtime.rag, "ingest_text", real_ingest_text)
        indexed = _text_for(runtime, context_id, "report.md")
        assert "THE FIRST GENERATION" not in indexed, (
            "a write that was already in flight put the replaced generation "
            "back after the upload had invalidated it, so the context answers "
            f"out of bytes the file no longer holds: {indexed[:160]!r}"
        )


class TestRetriesAreSpreadOverTime:
    def test_one_worker_pass_does_not_burn_the_whole_retry_budget(
        self, client, monkeypatch
    ):
        """Five attempts in five seconds is not a retry policy.

        A failed job goes straight back to `queued`, and a worker's pass keeps
        draining until the queue is empty — so a single pass can claim, fail
        and requeue the same job until its budget is gone. A thirty-second
        embedding outage would then permanently remove the file from the
        context, which is exactly what the retries are documented to prevent.
        """
        runtime = get_runtime()
        user_id, headers = _account(client)
        context_id = _context(client, headers)
        assert _upload(client, headers, "report.md", FIRST).status_code == 200
        files_dir = _files_dir(runtime, user_id)
        _cover_directory(client, headers, context_id, files_dir)
        path = str(files_dir / "report.md")

        monkeypatch.setattr(ingest_queue, "drain", lambda *a, **k: 0)
        assert _upload(client, headers, "report.md", SECOND).status_code == 200
        monkeypatch.undo()

        class Unavailable:
            """Not `FailsOnce`: an outage lasts longer than one call."""

            def __init__(self):
                self.calls = 0

            def ingest_file(self, *args, **kwargs):
                self.calls += 1
                raise RuntimeError("embedding backend unreachable")

        outage = Unavailable()
        ingest_queue.drain_until_idle(runtime.store, outage, fs_root=_root(runtime))

        assert outage.calls == 1, (
            f"one worker pass tried the same job {outage.calls} times; the "
            "retry budget is spent in seconds, not spread over an outage"
        )
        assert runtime.store.count_pending_ingest_jobs(path) == 1, (
            "the job was abandoned during a single pass, so the file is now "
            "permanently missing from a context that covers it"
        )

    def test_a_job_that_keeps_killing_the_process_is_eventually_abandoned(
        self, client, monkeypatch
    ):
        """The lease has to enforce the limit too.

        `_owed_another_go` counts attempts, but a hard-killed process never
        reaches it. If reclaim revives any abandoned claim regardless of how
        many it has already had, a job that crashes the worker every time
        revives forever and the limit means nothing.
        """
        runtime = get_runtime()
        user_id, headers = _account(client)
        context_id = _context(client, headers)
        assert _upload(client, headers, "report.md", FIRST).status_code == 200
        files_dir = _files_dir(runtime, user_id)
        _cover_directory(client, headers, context_id, files_dir)
        path = str(files_dir / "report.md")

        monkeypatch.setattr(ingest_queue, "drain", lambda *a, **k: 0)
        assert _upload(client, headers, "report.md", SECOND).status_code == 200

        claimed = runtime.store.claim_ingest_jobs(1)
        assert len(claimed) == 1
        with runtime.store._connect() as conn:
            conn.execute(
                "UPDATE ingest_job SET attempts = %s, "
                "updated_at = now() - interval '1 hour' WHERE id = %s",
                (ingest_queue.MAX_ATTEMPTS, str(claimed[0]["id"])),
            )

        assert runtime.store.reclaim_stale_ingest_jobs(900, max_attempts=ingest_queue.MAX_ATTEMPTS) == 0, (
            "a job that has already had every attempt was revived again, so "
            "one that kills the worker each time revives forever"
        )
        assert runtime.store.count_pending_ingest_jobs(path) == 0


class TestHoldersDoNotStarveEachOther:
    """Holding one connection and needing a second is a deadlock in waiting.

    These are not contending writers — each holds a *different* path and is
    entitled to run. Keeping waiters off the pool did nothing for them: the
    lock connection is held for the whole critical section, and the delete and
    the ingest inside it each need another. Once holders can take every
    connection, each waits for a second one that only another holder can give
    back, until the pool times out — and the upload's error path answers a
    timeout by deleting the file it just wrote.
    """


class TestOneLockForBothSides:
    """The upload and the queue worker contend on the *same* lock.

    This is the claim the whole design rests on once the database advisory
    lock is gone. Two locks that merely look alike would serialise nothing:
    the worker would re-index a path while an upload replaced its bytes, and
    the losing write would outlive the winner's — which is the defect the lock
    exists to prevent, reached through the machinery built to fix it.

    So the test does not inspect either side's code. It makes a worker hold
    the lock, has an ordinary upload of that same path go through the HTTP
    API, and asks whether the upload noticed.
    """

    def test_an_upload_cannot_publish_a_path_a_worker_is_indexing(
        self, client, monkeypatch
    ):
        runtime = get_runtime()
        user_id, headers = _account(client)
        context_id = _context(client, headers)

        assert _upload(client, headers, "report.md", FIRST).status_code == 200
        files_dir = _files_dir(runtime, user_id)
        _cover_directory(client, headers, context_id, files_dir)
        path = str(files_dir / "report.md")

        run_the_queue = ingest_queue.drain
        monkeypatch.setattr(ingest_queue, "drain", lambda *a, **k: 0)
        assert _upload(client, headers, "report.md", SECOND).status_code == 200
        assert runtime.store.count_pending_ingest_jobs(path) == 1

        # The upload waits the ordinary amount for a lock; shortened here so a
        # refusal takes a moment rather than half a minute. Only the timeout
        # changes — the lock, its key and its holder are the real ones.
        import functools

        monkeypatch.setattr(
            routes, "path_lock", functools.partial(path_lock, timeout=1.0)
        )

        indexing = threading.Event()
        may_finish = threading.Event()
        real_ingest_file = runtime.rag.ingest_file

        def gated_ingest_file(*args, **kwargs):
            indexing.set()
            assert may_finish.wait(30), "the upload never released the gate"
            return real_ingest_file(*args, **kwargs)

        monkeypatch.setattr(runtime.rag, "ingest_file", gated_ingest_file)
        worker = threading.Thread(
            target=run_the_queue,
            args=(runtime.store, runtime.rag),
            kwargs={"fs_root": _root(runtime)},
            daemon=True,
        )
        worker.start()
        try:
            assert indexing.wait(30), "the worker never reached its ingest"
            # The worker is inside the lock now. An ordinary upload of that
            # same name must not be able to publish over it.
            blocked = _upload(client, headers, "report.md", FIRST)
            assert blocked.status_code == 409, (
                "an upload published a path a worker was indexing, so the two "
                f"are not taking the same lock: {blocked.status_code} "
                f"{blocked.text[:200]}"
            )
        finally:
            may_finish.set()
            worker.join(timeout=30)
            assert not worker.is_alive()

        # And once the worker is done, the same upload goes through.
        assert _upload(client, headers, "report.md", FIRST).status_code == 200


class TestAFailedIngestStillOwesTheReRead:
    """The one path that skips the queue must not skip it silently.

    A named context is left out of the enqueue loop because it is about to be
    ingested in the request. When that ingest fails, the request has written a
    `context_source` row saying the context covers the path and has emptied
    what it said about it — a context covering a file it describes not at all,
    which is the coverage loss this whole queue exists to prevent, arriving
    through the one branch that does not use it.
    """

    def test_a_context_whose_ingest_failed_is_queued_for_a_re_read(
        self, client, monkeypatch
    ):
        runtime = get_runtime()
        user_id, headers = _account(client)
        context_id = _context(client, headers)

        assert _upload(client, headers, "report.md", FIRST,
                       context_id=context_id).status_code == 200
        files_dir = _files_dir(runtime, user_id)
        path = str(files_dir / "report.md")
        assert "THE FIRST GENERATION" in _text_for(runtime, context_id, "report.md")

        def unreachable(*args, **kwargs):
            raise RuntimeError("embedding backend unreachable")

        monkeypatch.setattr(runtime.rag, "ingest_file", unreachable)
        with contextlib.suppress(Exception):
            _upload(client, headers, "report.md", SECOND, context_id=context_id)

        # The claim survived the failure, as it should: the context was given
        # this file and still holds it as a source.
        assert any(str(s.fs_path) == path for s in
                   runtime.store.list_context_sources(context_id)), (
            "the source row is gone, so this test is not about what it says"
        )
        assert not _chunks_for(runtime, context_id, "report.md"), (
            "the failed generation was left in place"
        )
        assert runtime.store.count_pending_ingest_jobs(path) >= 1, (
            "the context covers this path and says nothing about it, and "
            "nothing is queued to fix that — the file is lost from a context "
            "that claims to hold it"
        )

        # And the queue does fix it, once the backend is reachable again.
        monkeypatch.undo()
        assert ingest_queue.drain_until_idle(
            runtime.store, runtime.rag, fs_root=_root(runtime)
        ) >= 1
        assert "THE SECOND GENERATION" in _text_for(runtime, context_id, "report.md")
