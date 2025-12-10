from __future__ import annotations

from datetime import datetime, timezone
from typing import Tuple


def encode_artifact_cursor(created_at: datetime, artifact_id: str) -> str:
    """Encode a cursor combining created_at and artifact id for keyset paging."""

    ts = created_at if created_at.tzinfo else created_at.replace(tzinfo=timezone.utc)
    return f"{ts.isoformat()}|{artifact_id}"


def decode_artifact_cursor(cursor: str) -> Tuple[datetime, str]:
    """Decode artifact keyset cursor into timestamp and id."""

    parts = cursor.split("|", 1)
    if len(parts) != 2:
        raise ValueError("invalid artifact cursor")
    ts = datetime.fromisoformat(parts[0])
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts, parts[1]

