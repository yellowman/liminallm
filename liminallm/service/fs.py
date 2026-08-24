import hashlib
import hmac
import os
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlencode


class PathTraversalError(ValueError):
    """Raised when a path escapes the intended base directory."""


# SPEC §18: Signed URL expiry time (10 minutes)
DEFAULT_URL_EXPIRY_SECONDS = 600


def generate_signed_url(
    file_path: str,
    user_id: str,
    secret_key: str,
    *,
    expiry_seconds: int = DEFAULT_URL_EXPIRY_SECONDS,
    base_url: str = "/v1/files/download",
) -> str:
    """Generate a signed download URL for secure file access.

    SPEC §18: Downloads use signed URLs with 10m expiry and content-disposition
    set to prevent inline execution.

    Args:
        file_path: Relative path to file within user's file storage
        user_id: User ID who owns the file
        secret_key: HMAC secret key for signing
        expiry_seconds: URL expiry time in seconds (default: 600 = 10 minutes)
        base_url: Base URL path for download endpoint

    Returns:
        Signed URL with signature and expiry parameters
    """
    expires_at = int(time.time()) + expiry_seconds
    # Create message to sign: path + user_id + expiry
    message = f"{file_path}|{user_id}|{expires_at}"
    signature = hmac.new(
        secret_key.encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    params = urlencode({
        "path": file_path,
        "expires": expires_at,
        "sig": signature,
    })
    return f"{base_url}?{params}"


def validate_signed_url(
    path: str,
    expires: str,
    signature: str,
    user_id: str,
    secret_key: str,
) -> Tuple[bool, Optional[str]]:
    """Validate a signed download URL.

    Args:
        path: File path from URL
        expires: Expiry timestamp from URL
        signature: HMAC signature from URL
        user_id: User ID making the request
        secret_key: HMAC secret key for validation

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        expires_at = int(expires)
    except (ValueError, TypeError):
        return False, "invalid expiry format"

    # Check expiry
    if time.time() > expires_at:
        return False, "URL has expired"

    # Recreate expected signature
    message = f"{path}|{user_id}|{expires_at}"
    expected_sig = hmac.new(
        secret_key.encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    # Constant-time comparison to prevent timing attacks
    if not hmac.compare_digest(signature, expected_sig):
        return False, "invalid signature"

    return True, None


def safe_join(base: Path, relative: str) -> Path:
    """Join ``relative`` to ``base`` while preventing path traversal.

    The resulting path must resolve within ``base``; absolute paths or ``..``
    segments that would escape the base directory raise ``PathTraversalError``.
    """

    base_resolved = base.resolve()
    rel_path = Path(relative)
    if rel_path.is_absolute():
        raise PathTraversalError("absolute paths not allowed")

    candidate = (base_resolved / rel_path).resolve()
    if candidate == base_resolved or base_resolved in candidate.parents:
        return candidate

    raise PathTraversalError("path traversal detected")


class PathAuthorityError(PermissionError):
    """This caller has nothing that entitles them to this filesystem path."""


def user_base(fs_root, user_id: str) -> Path:
    """The root SPEC §18 resolves an ordinary path against."""
    return Path(fs_root) / "users" / str(user_id)


def authorize_path(
    store,
    settings,
    fs_path: str,
    *,
    user_id: Optional[str],
    tenant_id: Optional[str],
) -> Path:
    """The resolved path this caller may have, or `PathAuthorityError`.

    SPEC §18 gives filesystem authority two sources and no third:

    * the caller's own area — `safe_join(base=/users/{user_id}, relative)`;
    * an artifact whose *persisted* visibility is `shared` or `global` and
      whose `fs_path` covers what is being asked for **under `/shared`**.

    The destination is part of the second rule, not decoration on it. An
    artifact is not a general-purpose grant that happens to name a path: a row
    covering `artifacts/{id}/v1.json`, or covering another user's files, must
    confer nothing, because §18 opened `/shared` and nowhere else.

    A pathname is not one of them. `POST /contexts/{id}/sources` used to accept
    anything underneath `shared_fs_root/shared` because it was underneath that
    directory, and then checked that the destination context belonged to the
    caller — which establishes who receives the content and never who was
    entitled to the source. Knowing a name became the whole of the authority.

    Authority is decided on where the path **resolves**, not how it reads:
    `..` is the escape everyone writes tests for, and a symlink is the same
    escape spelled so the string looks innocent.

    Every unprovable claim refuses, the same rule the workflow permission model
    follows: an ownerless `shared` artifact has no tenant to match, a caller
    with no tenant cannot match one, and a visibility nobody recognized grants
    exactly the values nobody considered.
    """
    if not user_id:
        raise PathAuthorityError("a filesystem path needs a caller to belong to")

    root = Path(settings.shared_fs_root).resolve()
    mine = user_base(settings.shared_fs_root, user_id)
    raw = Path(fs_path)

    if not raw.is_absolute():
        # Relative means "in my own area", and only that. Trying the relative
        # form against `/shared` as a fallback is how a name became a licence.
        try:
            return safe_join(mine, fs_path)
        except PathTraversalError as exc:
            raise PathAuthorityError(
                f"{fs_path!r} does not resolve inside your own files"
            ) from exc

    candidate = raw.resolve()
    mine_resolved = mine.resolve() if mine.exists() else mine
    if candidate == mine_resolved or mine_resolved in candidate.parents:
        return candidate

    shared_root = (root / "shared").resolve()
    if shared_root not in candidate.parents:
        # Not the caller's own area and not `/shared`, so there is no rule
        # left that could permit it. Checked before the artifact lookup rather
        # than inside it: an artifact row is only ever evidence about `/shared`,
        # so asking about any other path is asking the wrong question.
        raise PathAuthorityError(
            f"{fs_path!r} is neither in your own files nor under the shared area"
        )

    # Ask the ancestors, not the string: an artifact naming a corpus directory
    # authorizes the files in it, and naming one directory does not name the
    # one beside it.
    lineage = [str(candidate)] + [
        str(parent) for parent in candidate.parents if parent != parent.parent
    ]
    for artifact in store.artifacts_for_paths(lineage):
        if _artifact_authorizes(store, artifact, user_id=user_id, tenant_id=tenant_id):
            return candidate

    raise PathAuthorityError(
        f"{fs_path!r} is not covered by any artifact you may read"
    )


def _artifact_authorizes(store, artifact, *, user_id: str, tenant_id) -> bool:
    """Whether this artifact row entitles this caller to the path it names.

    Only `shared` and `global` — the two §18 names. `private` is deliberately
    absent: the caller's own authority is their `/users/{id}` root and is
    already spent there, so honouring a private row here would let an artifact
    widen a caller's filesystem reach beyond their own area, which is not one
    of the two sources the rule allows.
    """
    visibility = getattr(artifact, "visibility", "private")
    owner_id = getattr(artifact, "owner_user_id", None)
    if visibility == "shared":
        # `shared` is within one tenant, and the tenant is the owner's —
        # `artifact` has no tenant column of its own.
        if not owner_id or not tenant_id:
            return False
        owner = store.get_user(owner_id)
        return bool(owner) and owner.tenant_id == tenant_id
    if visibility == "global":
        return True
    # An unrecognized visibility is not a licence.
    return False


def adapter_dir_owner(path) -> str:
    """The adapter id a weights path claims to belong to, read from layout.

    One predicate for both ends of §5.5, because both ask the same question:
    training, before it writes a version into a directory, and serving, once
    it has resolved which `params.json` it would read. Handles every layout
    the resolver produces — `.../A`, `.../A/vNNNN/params.json` and the
    never-versioned `.../A/params.json`.
    """
    candidate = Path(str(path))
    if candidate.name == "params.json":
        candidate = candidate.parent
    if re.fullmatch(r"v\d+", candidate.name):
        candidate = candidate.parent
    return candidate.name


def server_owned_artifact_dirs(
    fs_root, artifact_id: str, artifact_type: str
) -> list[Path]:
    """The directories this server derives from an artifact's identity alone.

    Deliberately *not* `schema.fs_dir`. That is accepted by
    `adapter_root` when their final component matches the adapter's id, which
    is enough authority to stop adapter A serving adapter B's weights — it is
    not authority to destroy. The schema is user-editable, so a value like
    `<shared>/something-important/<their-own-artifact-id>` satisfies that rule
    while naming somebody else's data, and cleanup that trusted it would
    delete a path the artifact merely mentions.

    Derived from the id and joined safely, so a malformed identifier cannot
    reach outside the shared root either.
    """
    base = Path(fs_root)
    dirs = [safe_join(base, f"artifacts/{artifact_id}")]
    if artifact_type == "adapter":
        dirs.append(safe_join(base, f"adapters/{artifact_id}"))
    return dirs


def adapter_root(base: Path, adapter_id: str, explicit=None) -> Path:
    """The directory holding one adapter's versions, bound to its identity.

    An explicit ``fs_dir`` says **where** an adapter's directory
    lives — a per-user root, a different mount — never **whose** it is. Its
    final component must therefore be the adapter's own id, which both
    documented layouts already satisfy: ``adapters/{adapter_id}`` and
    ``/users/{user_id}/adapters/{adapter_id}``.

    Containment under ``base`` alone was not enough. It proved the path was
    inside the shared root, which every adapter's directory is, so an artifact
    whose schema named ``adapters/B`` had B's ``v0001/params.json`` served as
    A's version 1 — the same substitution as ``A/latest → B/v0001``, one level
    earlier, and reachable through ordinary artifact creation because the
    adapter schema accepts additional properties.

    Raises ``PathTraversalError`` for an escape or an identity mismatch.
    """
    identity = str(adapter_id or "").strip()
    if not identity:
        raise PathTraversalError("adapter path resolution requires an adapter id")
    if explicit is None or explicit == "":
        return safe_join(base, f"adapters/{identity}")

    base_resolved = base.resolve()
    candidate = Path(str(explicit))
    resolved = (
        candidate if candidate.is_absolute() else base_resolved / candidate
    ).resolve()
    if not (resolved == base_resolved or base_resolved in resolved.parents):
        raise PathTraversalError("adapter path must reside within fs_root")
    owner = adapter_dir_owner(resolved)
    if owner != identity:
        raise PathTraversalError(
            f"adapter {identity!r} declares directory {str(explicit)!r}, which "
            f"belongs to {owner!r}; an explicit root may relocate an adapter, "
            "never rename one adapter's weights to another's"
        )
    return resolved


# ---------------------------------------------------------------------------
# publication locks (SPEC §22: shared_fs_root is common across replicas)


class PathLockTimeout(RuntimeError):
    """Another publication of this path is in progress and did not finish."""


#: Where lock files live. Under the shared root rather than beside the file
#: they guard, so no user's directory listing grows an artefact, and one flat
#: directory rather than a mirrored tree, so a lock never needs a path to be
#: creatable before it can be taken.
LOCK_DIRNAME = ".locks"

_LOCK_POLL_SECONDS = 0.02
DEFAULT_LOCK_TIMEOUT_SECONDS = 30.0


def _lock_file(fs_root: Path, key: str) -> Path:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return Path(fs_root) / LOCK_DIRNAME / f"{digest}.lock"


DIGEST_BLOCK_BYTES = 1024 * 1024


def file_digest(path: str | Path) -> Optional[str]:
    """The SHA-256 of the bytes at `path`, or None if it cannot be read.

    Streamed, because the same function answers for a 200 MB upload and a
    12 KB attachment, and reading either one whole to hash it is a cost the
    caller did not ask for.

    None is "no answer", not "no match". Every caller compares the result
    with an expected digest, and None fails that comparison — which is the
    safe direction for all of them: an unreadable file is not a confirmed
    dedupe hit and is not a verified attachment.
    """
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(DIGEST_BLOCK_BYTES), b""):
                digest.update(block)
    except OSError:
        return None
    return digest.hexdigest()


def namespace_key(files_dir: str | Path, relative_name: str) -> str:
    """The lock key for anything that publishes or removes `relative_name`.

    The top-level entry, not the exact path. Publication and deletion do not
    always name the same depth: extraction of `outer/dir/inner.zip` publishes
    into `outer/dir/inner`, while a recursive delete targets `outer`. Locking
    each side's own path gives two keys that never meet, and the delete walks
    through a tree the extractor is still filling — later members recreate the
    ancestry with `mkdir(parents=True, exist_ok=True)`, so both requests report
    success over a partial tree. Measured before this rule existed.

    So every mutation of a persistent name takes the lock on the name's first
    component. For a root file that is the file itself, which is what upload
    already used; for anything inside a tree it is the tree. Coarser than an
    exact path, and the coarseness is the point: two operations on one tree
    must be able to see each other.
    """
    parts = Path(relative_name).parts
    return str(Path(files_dir) / (parts[0] if parts else relative_name))


def publication_key(fs_root: str | Path, fs_path: str | Path) -> str:
    """`namespace_key` for an absolute path, found rather than assumed.

    A worker holds an absolute path and the shared root; a route holds the
    user's files directory and a relative name. Both have to arrive at the
    same key or they take different locks and never see each other — which is
    not hypothetical: a queue that keyed on the file's own parent let a
    recursive delete of a tree run straight through a job indexing a file
    inside it, and the job then failed on a file removed underneath it.

    So the files directory is located in the path rather than guessed at, and
    the rest is handed to `namespace_key`. A path that is not under one — an
    adapter, a shared object — has no tree to belong to, and keying it on
    itself is both stable and correct for something nothing else contends on.
    """
    target = Path(fs_path)
    for parent in target.parents:
        # `.../users/<id>/files`, which is the base every namespace is
        # relative to. Matched by shape because that is what it is.
        if parent.name == "files" and parent.parent.parent.name == "users":
            return namespace_key(parent, str(target.relative_to(parent)))
    return str(target)


@contextmanager
def path_lock(
    fs_root: str | Path,
    key: str,
    *,
    timeout: float = DEFAULT_LOCK_TIMEOUT_SECONDS,
):
    """Serialise everything that publishes one path, across replicas.

    A publication is not one write. An upload puts bytes on disk, reads them
    back to index them, and records a checksum — three artefacts that have to
    describe the same generation, and three moments another request can land
    between. Making each step atomic does not help: measured, two uploads of
    one name left the second upload's bytes on disk, the second upload's
    chunks in the index, and the *first* upload's checksum in the manifest,
    with both requests returning 200.

    `flock`, for two reasons that rule out the alternatives. It is held by an
    open file description rather than by a process, so two threads in one API
    process serialise on it exactly as two replicas do — measured both ways;
    an in-process `threading.Lock` would be blind to the other replica, and
    §22 puts `shared_fs_root` in common between them deliberately. And the
    kernel drops it when the descriptor closes, so a replica that dies holding
    one does not wedge the name forever, which is the failure mode of a lock
    built out of `O_EXCL` and a stale file.

    Blocking, so call it off the event loop. `key` is any stable string naming
    what is being published — for a file, its path.

    Raises `PathLockTimeout` rather than proceeding unserialised.
    """
    import fcntl

    path = _lock_file(Path(fs_root), key)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    deadline = time.monotonic() + timeout
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise PathLockTimeout(
                        f"another publication of {key!r} is still in progress"
                    )
                time.sleep(_LOCK_POLL_SECONDS)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
