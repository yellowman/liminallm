"""Deferred re-indexing, so a file's coverage outlives its bytes.

Replacing a file changes its generation, not which contexts cover it. Both
halves are load-bearing, and they pull in opposite directions at the moment of
replacement:

* the old chunks describe bytes that no longer exist, so they must stop
  answering searches immediately — an index quoting a file that has moved on
  is worse than an index missing it;
* the contexts that cover the path keep covering it, so each of them owes the
  file a re-read.

The upload does the first, under the publication lock it already holds, and
records the second here. That split is the whole point: emptying is bounded
and cheap, while re-reading and re-embedding for every covering context is
work the request neither chose nor can bound. Between the two the path is
*absent* from those contexts, which is recoverable and honest, and it is never
permanently forgotten.

Two things keep the deferral safe.

`generation` is the checksum of the bytes that prompted a job. Before writing
anything the job re-reads the file and compares, so a job queued for an older
generation declines rather than reinstating it over a newer one.

The publication lock is the same `service.fs.path_lock` the upload takes, on
the same key. A worker that cannot get it is not failing — another publication
of that name is in progress, and it will queue its own job — so the worker
stands aside without spending an attempt.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Optional

from liminallm.logging import get_logger
from liminallm.service.fs import PathLockTimeout, namespace_key, path_lock

logger = get_logger(__name__)

# How many jobs one drain takes. The bound is on how much re-indexing a single
# caller does now, never on which contexts get re-indexed: every covering
# context has a durable row, so a bounded drain leaves work for the next one
# rather than dropping it. A file with a hundred consumers is refreshed over
# several passes; it does not lose ninety of them.
DRAIN_BATCH = 16

# How many the background worker takes per poll, across as many passes as it
# needs. It keeps going because the alternative is arithmetic no user would
# accept: one batch per poll turns a file with a hundred covering contexts
# into several minutes of being missing from most of them.
DRAIN_PER_POLL = 256

# How many times a job may fail before it is abandoned.
MAX_ATTEMPTS = 5

# Retry backoff, doubling. A worker drains until the queue is empty, so an
# unscheduled retry is re-claimed within a second of the first failure and
# covers none of the outages retries exist for.
RETRY_BASE_SECONDS = 30
RETRY_MAX_SECONDS = 900

# How long a claimed job may go unfinished before another poll takes it back.
# Generously longer than one file's extract-and-embed, because reviving a job
# that is merely slow does the work twice; short enough that a process killed
# mid-job costs minutes of a file being unsearchable, not the rest of time.
STALE_CLAIM_SECONDS = 900

# What a worker waits for the publication lock. Nearly nothing: whoever holds
# it is publishing that name and will queue the re-index the new bytes need,
# so this job has nothing to add by waiting.
LOCK_WAIT_SECONDS = 0.1


def _backoff_seconds(attempts: int) -> int:
    return min(RETRY_MAX_SECONDS, RETRY_BASE_SECONDS * (2 ** max(0, attempts - 1)))


def generation_of(path: Path) -> Optional[str]:
    """The checksum a job for these bytes would carry.

    None means the file is not there. Every other read error is raised, and
    the caller retries: "the file was deleted" and "the disk was briefly
    unavailable" look identical to a bare `except OSError` and call for
    opposite responses — one job is finished, the other is owed another go.
    """
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except FileNotFoundError:
        return None


def _owed_another_go(store, job: Dict[str, Any], detail: str) -> None:
    """Put a job back, or close it with the reason it is not coming back."""
    job_id = str(job["id"])
    attempts = int(job.get("attempts") or 0)
    if attempts >= MAX_ATTEMPTS:
        store.finish_ingest_job(
            job_id, "failed", detail=f"{detail} (abandoned after {attempts} attempts)"
        )
    elif not store.requeue_ingest_job(
        job_id, detail=detail, delay_seconds=_backoff_seconds(attempts)
    ):
        store.finish_ingest_job(
            job_id, "superseded", detail=f"{detail}; a newer job holds this path"
        )


def run_job(
    store, rag, job: Dict[str, Any], *, fs_root: str, chunk_size: Optional[int] = None
) -> Optional[int]:
    """Re-index one path into one context. Returns the chunks written.

    None means the job was not attempted, because another publication of that
    name holds the lock. That is not zero chunks written — a drain that
    treated the two alike would loop against a contended path until the job's
    budget ran out.
    """
    job_id = str(job["id"])
    context_id = str(job["context_id"])
    fs_path = str(job["fs_path"])
    path = Path(fs_path)

    try:
        with path_lock(
            fs_root,
            namespace_key(path.parent, path.name),
            timeout=LOCK_WAIT_SECONDS,
        ):
            return _reindex_under_lock(
                store, rag, job, path, chunk_size=chunk_size
            )
    except PathLockTimeout:
        # Standing aside, not failing: nothing was attempted, so this costs
        # the job neither an attempt nor its place in the queue. Whoever holds
        # the lock is publishing this name and queues the work its own bytes
        # need.
        detail = "another publication of this path is in progress"
        if not store.yield_ingest_job(job_id, detail=detail):
            store.finish_ingest_job(
                job_id, "superseded", detail=f"{detail}; a newer job holds it too"
            )
        return None


def _reindex_under_lock(
    store, rag, job: Dict[str, Any], path: Path, *, chunk_size: Optional[int]
) -> int:
    """The generation check and the commit, both inside the lock.

    The generation is read here rather than before the lock was taken, because
    waiting for it is exactly when a replacement is most likely to have
    happened.
    """
    job_id = str(job["id"])
    context_id = str(job["context_id"])
    fs_path = str(path)

    try:
        on_disk = generation_of(path)
    except OSError as exc:
        _owed_another_go(store, job, f"read failed: {type(exc).__name__}: {exc}")
        return 0
    if on_disk is None:
        # The file is gone. Its chunks were dropped when it was replaced, and
        # there is nothing to put back; the context still covers the path, so
        # a later file at that name is indexed by the job that upload queues.
        store.finish_ingest_job(job_id, "superseded", detail="file no longer exists")
        return 0
    if on_disk != str(job["generation"]):
        store.finish_ingest_job(
            job_id, "superseded", detail=f"on-disk generation {on_disk[:12]}"
        )
        return 0

    try:
        # One statement, not two. `ingest_file` commits through
        # `_commit_generation`, which replaces everything the context says
        # about this path in a single transaction — so there is no moment
        # where the path has neither the old chunks nor the new.
        written = rag.ingest_file(context_id, fs_path, chunk_size=chunk_size)
    except Exception as exc:
        logger.warning(
            "ingest_job_failed",
            job_id=job_id,
            context_id=context_id,
            fs_path=fs_path,
            attempts=int(job.get("attempts") or 0),
            error=str(exc),
        )
        _owed_another_go(store, job, f"{type(exc).__name__}: {exc}")
        return 0

    store.finish_ingest_job(job_id, "done", detail=f"{written} chunks")
    return written


def drain(
    store,
    rag,
    *,
    fs_root: str,
    limit: int = DRAIN_BATCH,
    chunk_size: Optional[int] = None,
) -> int:
    """Run up to `limit` due jobs. Returns how many were actually attempted.

    Jobs that stood aside for another publication are not counted, so a caller
    that keeps draining while this reports progress stops instead of looping
    over a path somebody else is busy with.
    """
    try:
        jobs = store.claim_ingest_jobs(limit)
    except Exception as exc:  # a drain is opportunistic; the rows outlive it
        logger.warning("ingest_drain_claim_failed", error=str(exc))
        return 0
    attempted = sum(
        1
        for job in jobs
        if run_job(store, rag, job, fs_root=fs_root, chunk_size=chunk_size) is not None
    )
    if jobs:
        logger.info(
            "ingest_drain_completed",
            jobs=attempted,
            stood_aside=len(jobs) - attempted,
        )
    return attempted


def drain_until_idle(store, rag, *, fs_root: str, max_jobs: int = DRAIN_PER_POLL) -> int:
    """Keep draining until nothing is due, or `max_jobs` have run.

    The cap is a backstop against one poll monopolising the worker, not a
    limit on what gets re-indexed: whatever it leaves is still queued and the
    next poll continues from there.
    """
    try:
        store.reclaim_stale_ingest_jobs(
            STALE_CLAIM_SECONDS, max_attempts=MAX_ATTEMPTS
        )
    except Exception as exc:  # a reclaim is opportunistic; the rows outlive it
        logger.warning("ingest_reclaim_failed", error=str(exc))
    done = 0
    while done < max_jobs:
        ran = drain(store, rag, fs_root=fs_root, limit=min(DRAIN_BATCH, max_jobs - done))
        if not ran:
            break
        done += ran
    return done
