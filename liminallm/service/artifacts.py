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

Artifact ids are never reused, so the rule is simple: the directory is named
like a server-owned payload, no artifact row has that id, and it has been that
way for longer than the grace period.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Iterable

from liminallm.service.fs import PathTraversalError, safe_join
from liminallm.logging import get_logger

logger = get_logger(__name__)

#: Where the server puts payloads it derives from an artifact's identity.
#: Never `schema.fs_dir`: that is user-editable, and a relocated tree has no
#: server-authoritative record of ownership to justify destroying it.
PAYLOAD_DIRNAMES = ("artifacts", "adapters")

#: Longer than any request may legally take, so a caller that already resolved
#: an artifact can finish reading its bytes.
DEFAULT_GRACE_SECONDS = 3600


def _candidate_dirs(root: Path) -> Iterable[Path]:
    for dirname in PAYLOAD_DIRNAMES:
        parent = root / dirname
        if not parent.is_dir():
            continue
        for child in parent.iterdir():
            if child.is_dir():
                yield child


def sweep_artifact_payloads(
    store, fs_root: str, *, grace_seconds: int = DEFAULT_GRACE_SECONDS
) -> int:
    """Remove payload directories no artifact claims. Returns how many went.

    `grace_seconds` is measured from the directory's own mtime, which is what
    a deletion leaves behind: the row goes, the directory stops changing.
    """
    root = Path(fs_root)
    cutoff = time.time() - max(grace_seconds, 0)
    removed = 0

    for directory in _candidate_dirs(root):
        artifact_id = directory.name
        try:
            # The name has to be one this server would have produced, so a
            # directory that merely sits in the tree is not swept on the
            # strength of its position.
            if safe_join(root, f"{directory.parent.name}/{artifact_id}") != directory:
                continue
        except PathTraversalError:
            continue
        try:
            if directory.stat().st_mtime > cutoff:
                continue
        except OSError:
            continue
        try:
            # The one authority check, taken immediately before removing
            # rather than during the scan: the scan's answer is stale by the
            # time the loop reaches this directory. A second, earlier copy of
            # this question would only be an optimization, and no test can
            # tell the two apart because artifact ids are never reused — so
            # there is one.
            if store.get_artifact(artifact_id) is not None:
                continue
            shutil.rmtree(directory)
        except OSError as exc:
            # Retried next sweep rather than lost. This is the failure mode
            # that used to become an orphan nothing would look at again.
            logger.warning(
                "artifact_payload_sweep_failed",
                path=str(directory),
                error=str(exc),
            )
            continue
        removed += 1
        logger.info("artifact_payload_reclaimed", path=str(directory))
    return removed
