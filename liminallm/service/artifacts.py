"""Reclaiming the payloads of artifacts that no longer exist.

Deleting an artifact revokes a capability. Removing the bytes it referred to
is a different act with a different timing constraint, and doing both inside
the request produced state with no serial explanation:

    turn                        DELETE
    ----                        ------
    resolve adapter A
    (holds the capability)
                                delete the artifact row
                                commit
                                rmtree adapters/A
    params_path.stat()  ->  FileNotFoundError

The turn acquired A before the deletion and read the filesystem after it. If
the turn had run first it should have been able to finish; if the delete had
run first the turn should never have acquired A. Neither order produces what
happened.

So reclamation is delayed instead. The request revokes the capability and
returns; a sweep collects the directories afterwards, once they have been
orphans for longer than any request may live. Two other things improve for
free: an `rmtree` of a large checkpoint tree stops blocking an API worker, and
an I/O failure becomes a retry next sweep rather than an orphan that is logged
once and kept forever.

The delay is measured from a durable record written in the same transaction
as the deletion, not from anything on disk. A payload's own timestamps answer
a different question — an adapter trained a week ago and deleted a moment ago
is a week old by that measure, and the first version of this sweep collected
it immediately, putting the race straight back. "Retired at T" has to mean
"the capability stopped existing at T".

Artifact ids are never reused, so the rest is simple: take the retirements
whose grace has elapsed, remove only the directories derived from the id, and
clear the record once the bytes are gone.

The sweep also enrols what nothing enrolled for it. A ledger only collects
what something puts in it, and `create_artifact` writes its payload before
publishing the row — so a failed publication leaves a directory no artifact
ever named, with no deletion to trigger enrolment. Those are recorded at
first observation rather than removed on sight, which keeps discovery without
bringing back a clock that means the wrong thing.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable

from liminallm.logging import get_logger
from liminallm.service.fs import (
    PathTraversalError,
    server_owned_artifact_dirs,
)

logger = get_logger(__name__)

#: Where the server puts payloads it derives from an artifact's identity.
#: Never `schema.fs_dir`: that is user-editable, and a relocated tree has no
#: server-authoritative record of ownership to justify destroying it.
PAYLOAD_DIRNAMES = ("artifacts", "adapters")

#: Longer than any request may legally take, so a caller that already resolved
#: an artifact can finish reading its bytes.
DEFAULT_GRACE_SECONDS = 3600


def _candidate_dirs(root: Path) -> Iterable[tuple[str, str]]:
    """Every directory shaped like a payload this server produced.

    Yields `(artifact_id, artifact_type)`. The parent name is what tells the
    two apart, and it is the server's own layout rather than anything the
    artifact's schema claims.
    """
    for dirname in PAYLOAD_DIRNAMES:
        parent = root / dirname
        if not parent.is_dir():
            continue
        for child in parent.iterdir():
            if child.is_dir():
                yield child.name, ("adapter" if dirname == "adapters" else "artifact")


def enrol_unknown_payloads(store, fs_root: str) -> int:
    """Record payloads that no artifact claims and no retirement covers."""
    enrolled = 0
    for artifact_id, artifact_type in _candidate_dirs(Path(fs_root)):
        try:
            if store.enrol_artifact_retirement(artifact_id, artifact_type):
                enrolled += 1
                logger.info(
                    "artifact_payload_orphan_enrolled",
                    artifact_id=artifact_id,
                    artifact_type=artifact_type,
                )
        except Exception as exc:  # pragma: no cover - malformed name, or a race
            logger.warning(
                "artifact_payload_enrolment_failed",
                artifact_id=artifact_id,
                error=str(exc),
            )
    return enrolled


def sweep_artifact_payloads(
    store, fs_root: str, *, grace_seconds: int = DEFAULT_GRACE_SECONDS
) -> int:
    """Reclaim the payloads of retirements older than the grace period.

    Returns how many artifacts were reclaimed.
    """
    removed = 0
    # Discovery first, so an orphan nothing enrolled starts its clock on the
    # sweep that finds it rather than never.
    try:
        enrol_unknown_payloads(store, fs_root)
    except Exception as exc:  # pragma: no cover - filesystem unreadable
        logger.warning("artifact_payload_enrolment_scan_failed", error=str(exc))
    try:
        due = store.due_artifact_retirements(grace_seconds=grace_seconds)
    except Exception as exc:  # pragma: no cover - the queue is unreadable
        logger.warning("artifact_retirement_queue_unreadable", error=str(exc))
        return 0

    for artifact_id, artifact_type in due:
        # Ids are not reused, so this can only be true if the delete was rolled
        # back after its record was read. Cheap, and it is the one question
        # that must not be stale.
        if store.get_artifact(artifact_id) is not None:
            continue
        try:
            directories = server_owned_artifact_dirs(
                fs_root, artifact_id, artifact_type
            )
        except PathTraversalError:  # pragma: no cover - malformed id
            logger.warning("artifact_retirement_path_refused", artifact_id=artifact_id)
            continue

        failed = False
        for directory in directories:
            try:
                if directory.is_dir():
                    shutil.rmtree(directory)
                    logger.info("artifact_payload_reclaimed", path=str(directory))
            except OSError as exc:
                # Left in the queue, so the next sweep tries again. This is the
                # failure that used to become an orphan logged once and kept.
                logger.warning(
                    "artifact_payload_sweep_failed",
                    path=str(directory),
                    error=str(exc),
                )
                failed = True
        if failed:
            continue
        store.clear_artifact_retirement(artifact_id)
        removed += 1
    return removed
