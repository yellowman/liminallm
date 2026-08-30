"""Reclaiming the filesystem namespace of an account that no longer exists.

Deleting an account is one lifetime boundary, not a list of deletions. The
store's cascade takes the rows; everything the account owned on disk stayed
behind - `/users/<id>` holds its uploaded files and its content-addressed
attachment generations, `/.archive-staging/<id>` holds whole-tree extraction
work - because nothing on either side of the transaction was responsible for
it.

Removing the bytes inside the request has the failure the artifact sweep
already documents: a turn resolves a generation from Postgres and only then
reads it, so unlinking during the deletion lets a caller that legitimately
acquired the object read a filesystem where it is gone. So the request
revokes, the trigger records `user_namespace_retirement`, and this collects
afterwards.

The harder half is the clock. Three collectors already walk that namespace on
their own schedules, and each measures age from something on disk:

    sweep                       what it removes         its clock
    -----                       ---------------         ---------
    _sweep_tmp_dirs             users/<u>/tmp/*         file mtime
    sweep_generations           unreferenced blobs      blob mtime
    _sweep_archive_staging      .archive-staging/<u>/*  tree mtime

`sweep_generations` is the dangerous one. It marks from what the account's
conversations reference; once the rows are gone that mark set is empty, so
every generation the account ever made looks unreferenced and is judged by the
blob's own mtime, which is weeks old. The deletion's own grace period is
undercut by whichever cleanup pass runs next - often within the same minute.

Hence the rule these three now obey: while a retirement is pending for a user,
none of them touches that user at all. The account's lifetime outranks every
lifetime inside it, and when it comes due the whole identity-derived namespace
goes at once.

Both destructive targets are derived from the id and nothing else. There is
deliberately no per-subdirectory logic: deleting the whole namespace makes it
impossible to forget the next subdirectory somebody adds.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from liminallm.logging import get_logger
from liminallm.service.fs import PathTraversalError, safe_join

logger = get_logger(__name__)

#: The two trees the server derives from a user id alone. Not `schema.fs_dir`
#: and not anything a request named: these are the server's own layout, which
#: is the only authority that justifies destroying a directory.
NAMESPACE_DIRNAMES = ("users", ".archive-staging")

#: Longer than any request may legally take, so a turn that already resolved
#: one of this account's generations can finish reading it. The same hour the
#: artifact payload sweep uses, for the same reason.
DEFAULT_GRACE_SECONDS = 3600


def namespace_dirs(fs_root, user_id: str) -> list[Path]:
    """Every directory this server owns on behalf of one account."""
    base = Path(fs_root)
    return [safe_join(base, f"{name}/{user_id}") for name in NAMESPACE_DIRNAMES]


def enrol_unclaimed_namespaces(store, fs_root: str) -> int:
    """Record namespaces no account claims and no retirement covers.

    A ledger only collects what something puts in it, and the trigger that
    enrols deletions cannot reach backwards: a namespace left by an account
    deleted before any of this existed has no deletion left to fire. Those are
    recorded at first observation rather than removed on sight, which keeps
    discovery without inventing a retirement time nothing witnessed.

    The store refuses to enrol a namespace whose account still exists, so a
    directory created moments before its `app_user` row commits is not mistaken
    for debris.
    """
    enrolled = 0
    for dirname in NAMESPACE_DIRNAMES:
        parent = Path(fs_root) / dirname
        if not parent.is_dir():
            continue
        for child in parent.iterdir():
            if not child.is_dir():
                continue
            try:
                if store.enrol_user_namespace_retirement(child.name):
                    enrolled += 1
                    logger.info("user_namespace_orphan_enrolled", user_id=child.name)
            except Exception as exc:  # pragma: no cover - malformed name, or a race
                logger.warning(
                    "user_namespace_enrolment_failed",
                    user_id=child.name,
                    error=str(exc),
                )
    return enrolled


def sweep_user_namespaces(
    store, fs_root: str, *, grace_seconds: int = DEFAULT_GRACE_SECONDS
) -> int:
    """Reclaim the namespaces of retirements older than the grace period.

    Returns how many accounts were reclaimed.
    """
    removed = 0
    # Discovery first, so debris nothing enrolled starts its clock on the sweep
    # that finds it rather than never.
    try:
        enrol_unclaimed_namespaces(store, fs_root)
    except Exception as exc:  # pragma: no cover - filesystem unreadable
        logger.warning("user_namespace_enrolment_scan_failed", error=str(exc))
    try:
        due = store.due_user_namespace_retirements(grace_seconds=grace_seconds)
    except Exception as exc:  # pragma: no cover - the queue is unreadable
        logger.warning("user_namespace_queue_unreadable", error=str(exc))
        return 0

    for user_id in due:
        try:
            directories = namespace_dirs(fs_root, user_id)
        except PathTraversalError:  # pragma: no cover - malformed id
            logger.warning("user_namespace_path_refused", user_id=user_id)
            continue

        failed = False
        for directory in directories:
            try:
                if directory.is_dir():
                    shutil.rmtree(directory)
                    logger.info("user_namespace_reclaimed", path=str(directory))
            except OSError as exc:
                # Left in the queue, so the next sweep tries again - and so the
                # subordinate sweeps keep skipping this user until there is
                # nothing left of them to skip.
                logger.warning(
                    "user_namespace_sweep_failed",
                    path=str(directory),
                    error=str(exc),
                )
                failed = True
        if failed:
            continue
        store.clear_user_namespace_retirement(user_id)
        removed += 1
    return removed
