"""File upload policy (SPEC §17), shared by the API and the interpreter.

Kept in its own module so the tool sandbox can apply the same extension policy
as /files/upload without importing the API layer (which would be circular).
"""
from __future__ import annotations

ALLOWED_UPLOAD_EXTENSIONS = {
    ".txt",
    ".md",
    ".pdf",
    ".json",
    ".csv",
    ".tsv",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".mp3",
    ".wav",
    ".ogg",
    # Archives are stored as-is and expanded only via POST /files/{name}/extract
    # (never auto-ingested); extraction enforces zip-bomb and zip-slip limits.
    ".zip",
    ".tar",
    ".tgz",
    ".gz",
}
