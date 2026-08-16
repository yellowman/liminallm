import hashlib
import hmac
import re
import time
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
      whose `fs_path` covers what is being asked for.

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

    if root not in candidate.parents and candidate != root:
        raise PathAuthorityError(
            f"{fs_path!r} is outside the managed filesystem root"
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
    """Whether this artifact row entitles this caller to the path it names."""
    visibility = getattr(artifact, "visibility", "private")
    owner_id = getattr(artifact, "owner_user_id", None)
    if visibility == "private":
        # Ownerless too: an artifact nobody owns cannot be shown to be this
        # caller's, and a check that only refuses when an owner is present
        # serves everyone a null owner.
        return bool(owner_id) and owner_id == user_id
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


def adapter_root(base: Path, adapter_id: str, explicit=None) -> Path:
    """The directory holding one adapter's versions, bound to its identity.

    An explicit ``fs_dir``/``cephfs_dir`` says **where** an adapter's directory
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
