"""The only format a process we do not trust may speak to us in.

Two boundaries in this codebase declare the child hostile. SPEC §18 makes the
tool worker the untrusted half of the broker boundary — it runs model-chosen
control flow over attacker-controlled bytes. §19.5 puts parsers in a disposable
child because "assume the parsers are compromisable".

Both spoke `multiprocessing.Connection.send()`/`recv()`, and `recv()` unpickles.
Python's own documentation warns that this is unsafe from a peer you do not
trust: unpickling executes `__reduce__`, so the dangerous operation runs **in
the parent**, during decoding, before any check the parent might make. Measured
before this module existed: a sandbox child returning an object with a hostile
`__reduce__` had its callback run in the API process. That is not a bug in a
check — it is the absence of a boundary, because the decoder was the hole.

So the wire format is JSON over `send_bytes`/`recv_bytes`: data with no
callable in its grammar, and no way to name a type. Errors cross as
`{"type": ..., "message": ...}` rather than as exception objects, because an
exception object is a class the sender chooses and the receiver constructs.

Frames are bounded. A child is rlimited but its limit is generous, and without
a cap it can turn its own permitted memory into an allocation in the API
process. Each caller passes a bound derived from its own budget, because only
the caller knows what its results can legitimately be; there is a conservative
default here for callers with no budget of their own.
"""
from __future__ import annotations

import json
from typing import Any, Mapping, Optional

__all__ = [
    "WireError",
    "recv_frame",
    "send_frame",
    "error_payload",
    "DEFAULT_MAX_FRAME_BYTES",
    "ERROR_FRAME_BYTES",
    "MAX_ERROR_MESSAGE_CHARS",
]

#: For callers with no budget to derive one from. Large enough for a page of
#: results, small enough that a hostile child cannot allocate its way through
#: the parent.
DEFAULT_MAX_FRAME_BYTES = 4 * 1024 * 1024

#: An error message is for a log line and a user-facing reason, so it is
#: truncated rather than trusted: a child that fails on a 20MB input can put
#: the whole input in a `str(exc)`.
MAX_ERROR_MESSAGE_CHARS = 1000
_MAX_ERROR_TYPE_CHARS = 100

#: What an error frame can weigh, worst case: both fields are truncated above,
#: and JSON escapes at most six bytes per character (`\uXXXX`), so
#: (1000 + 100) * 6 plus keys and braces stays well inside this. Callers read
#: with at least this much headroom, so a failure is reportable even when the
#: caller's result budget is smaller than the report.
ERROR_FRAME_BYTES = 8 * 1024


class WireError(ValueError):
    """A frame was absent, oversized, or not the data shape we accept."""


def error_payload(exc: BaseException) -> dict:
    """Describe a failure as data.

    The type crosses as its *name*, never as a class. The receiver decides
    whether that name means anything to it, from a vocabulary the receiver
    owns; nothing here is imported, resolved, or constructed from the sender's
    string. That is the whole difference between an error channel and a
    second code path.
    """
    return {
        "type": type(exc).__name__[:_MAX_ERROR_TYPE_CHARS],
        "message": str(exc)[:MAX_ERROR_MESSAGE_CHARS],
    }


def send_frame(
    conn: Any,
    payload: Mapping[str, Any],
    *,
    max_bytes: Optional[int] = DEFAULT_MAX_FRAME_BYTES,
) -> int:
    """Put one JSON object on the wire; returns the bytes written.

    Raises rather than degrading when the payload is not representable:
    a `default=repr` fallback would silently turn an object into its repr and
    hide the fact that something non-data was being returned.

    The byte count is returned because a receiver that bounds its peer by what
    it has itself supplied needs to know how much that was.

    `max_bytes=None` sends unbounded, for a sender whose receiver holds the
    only cap that counts. Two caps on one frame is two numbers to keep in
    step, and the one that can be wrong quietly is the sender's.
    """
    try:
        raw = json.dumps(payload, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise WireError(f"payload is not representable as data: {exc}") from exc
    if max_bytes is not None and len(raw) > max_bytes:
        raise WireError(f"frame of {len(raw)} bytes exceeds the {max_bytes} allowed")
    conn.send_bytes(raw)
    return len(raw)


def recv_frame(conn: Any, *, max_bytes: Optional[int] = DEFAULT_MAX_FRAME_BYTES) -> dict:
    """Read one JSON object, or raise `WireError`.

    `recv_bytes(maxlength)` refuses an oversized message rather than reading it,
    which is the point: the cap is applied before the bytes are ours. The
    connection is unusable afterwards, which is correct — a peer that sent one
    has nothing further to say that we would believe.

    `max_bytes=None` reads unbounded, and is for the one case where the peer is
    the trusted end: a cap there would only defend an untrusted, disposable,
    already-rlimited process against the authority that started it.

    `EOFError` passes through rather than becoming a `WireError`. A peer that
    closed is not a peer that misbehaved, and every caller here says something
    different about the two — "the worker died" against "the worker sent
    something we will not decode".
    """
    try:
        raw = conn.recv_bytes(max_bytes)
    except OSError as exc:
        raise WireError(f"frame exceeded {max_bytes} bytes or the channel failed: {exc}") from exc
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise WireError(f"frame is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise WireError(f"frame is a {type(payload).__name__}, not an object")
    return payload
