"""Utilities for normalizing structured message content.

The runtime stores `content_struct` alongside linearized message text to
preserve rich segments (code blocks, citations, tool calls, attachments,
redactions). This module offers light-weight validation and normalization so
callers can persist structured content without over-constraining the shape.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, TypedDict

SegmentType = Literal[
    "text", "code", "citation", "tool_call", "attachment", "redaction"
]


class ContentSegment(TypedDict, total=False):
    """Typed view of a single segment inside ``content_struct``.

    The shape is intentionally permissive; only a small set of keys is preserved
    so downstream clients can rely on stable fields without blocking custom
    annotations in ``meta``.
    """

    type: SegmentType
    text: str
    start: int
    end: int
    tags: List[str]
    # code
    language: str
    # citations / RAG
    source_id: str
    chunk_id: str
    score: float
    locator: str
    # tool calls
    name: str
    arguments: Dict[str, Any]
    result: Any
    status: str
    duration_ms: float
    # attachments
    kind: str
    uri: str
    mime: str
    description: str
    # redactions / filtering
    reason: str
    policy: str
    meta: Dict[str, Any]


class ContentStruct(TypedDict, total=False):
    segments: List[ContentSegment]
    summary: Dict[str, Any]


_SEGMENT_KEYS: Dict[SegmentType, List[str]] = {
    "text": ["text", "start", "end", "tags", "meta"],
    "code": ["text", "language", "start", "end", "tags", "meta"],
    "citation": [
        "text",
        "source_id",
        "chunk_id",
        "score",
        "locator",
        "start",
        "end",
        "tags",
        "meta",
    ],
    "tool_call": [
        "name",
        "arguments",
        "result",
        "status",
        "duration_ms",
        "start",
        "end",
        "tags",
        "meta",
    ],
    "attachment": [
        "kind",
        "uri",
        "mime",
        "description",
        "start",
        "end",
        "tags",
        "meta",
    ],
    "redaction": ["text", "reason", "policy", "start", "end", "tags", "meta"],
}


def _coerce_segment(segment: Any) -> Optional[ContentSegment]:
    if not isinstance(segment, dict):
        return None
    seg_type = segment.get("type")
    if seg_type not in _SEGMENT_KEYS:
        return None
    allowed_keys = _SEGMENT_KEYS[seg_type]
    normalized: ContentSegment = {"type": seg_type}
    for key in allowed_keys:
        if key in segment:
            normalized[key] = segment[key]
    return normalized


def normalize_content_struct(
    content_struct: Optional[dict], content: Optional[str] = None
) -> Optional[ContentStruct]:
    """Return a sanitized ``content_struct`` payload or ``None``.

    - Accepts dictionaries with a ``segments`` list and filters segments down to
      a stable set of keys per type.
    - Drops invalid structures to keep storage lean and JSON-serializable.
    - If no valid segments remain but ``content`` is provided, fall back to a
      single ``text`` segment so callers can rely on a consistent shape.
    """

    if content_struct is None:
        return None
    if not isinstance(content_struct, dict):
        return None
    segments = content_struct.get("segments")
    if not isinstance(segments, list):
        return None
    normalized_segments: List[ContentSegment] = []
    for raw in segments:
        normalized = _coerce_segment(raw)
        if normalized:
            normalized_segments.append(normalized)
    if not normalized_segments:
        if content:
            normalized_segments.append({"type": "text", "text": content})
        else:
            return None
    normalized_struct: ContentStruct = {"segments": normalized_segments}
    if isinstance(content_struct.get("summary"), dict):
        normalized_struct["summary"] = content_struct["summary"]
    return normalized_struct
