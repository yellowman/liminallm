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
