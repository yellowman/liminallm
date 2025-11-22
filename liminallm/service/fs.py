from pathlib import Path


class PathTraversalError(ValueError):
    """Raised when a path escapes the intended base directory."""


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
