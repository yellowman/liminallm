import hashlib
import hmac
import time
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import parse_qs, urlencode


class PathTraversalError(ValueError):
    """Raised when a path escapes the intended base directory."""


class SignedURLError(ValueError):
    """Raised when signed URL validation fails."""


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
