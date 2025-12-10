from __future__ import annotations

from datetime import datetime, timezone
from typing import Tuple


def encode_time_id_cursor(created_at: datetime, identifier: str) -> str:
    """Encode a cursor combining a timestamp and identifier for keyset paging."""

    ts = created_at if created_at.tzinfo else created_at.replace(tzinfo=timezone.utc)
    return f"{ts.isoformat()}|{identifier}"


def decode_time_id_cursor(cursor: str) -> Tuple[datetime, str]:
    """Decode a time/id cursor into timestamp and identifier."""

    parts = cursor.split("|", 1)
    if len(parts) != 2:
        raise ValueError("invalid artifact cursor")
    ts = datetime.fromisoformat(parts[0])
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts, parts[1]


def encode_index_cursor(index: int, identifier: str) -> str:
    """Encode a cursor for ascending index/id pagination (e.g., chunks)."""

    return f"{index}|{identifier}"


def decode_index_cursor(cursor: str) -> Tuple[int, str]:
    """Decode an index/id cursor used for chunk pagination."""

    parts = cursor.split("|", 1)
    if len(parts) != 2:
        raise ValueError("invalid index cursor")
    return int(parts[0]), parts[1]


def encode_artifact_cursor(created_at: datetime, artifact_id: str) -> str:
    """Backward-compatible artifact cursor helper."""

    return encode_time_id_cursor(created_at, artifact_id)


def decode_artifact_cursor(cursor: str) -> Tuple[datetime, str]:
    """Backward-compatible artifact cursor decoder."""

    return decode_time_id_cursor(cursor)

