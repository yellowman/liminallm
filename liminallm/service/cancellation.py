"""Cancelling an in-flight stream, wherever in the cluster it runs.

The cancel event lives in the process holding that request's WebSocket, so a
replica can only stop its own streams locally. Unknown requests are put to the
peers over the cluster bus; silence means no replica owns it.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

from liminallm.logging import get_logger

logger = get_logger(__name__)

#: Logical cluster-bus channel used to reach the replica that owns a request.
CANCEL_CHANNEL = "chat.cancel"

# request_id -> (cancel_event, registered_at, owner_user_id). The owner is kept
# so one user cannot cancel another's stream.
_active_requests: Dict[str, tuple[asyncio.Event, datetime, str]] = {}
# Created on first use: at import time there is no running event loop.
_active_requests_lock: Optional[asyncio.Lock] = None
_active_requests_last_cleanup = datetime.now(timezone.utc)
_ACTIVE_REQUEST_TTL_SECONDS = 30 * 60


def _get_active_requests_lock() -> asyncio.Lock:
    global _active_requests_lock
    if _active_requests_lock is None:
        _active_requests_lock = asyncio.Lock()
    return _active_requests_lock


async def _cleanup_stale() -> int:
    """Drop entries whose request died without unregistering. Runs at most
    every five minutes, so the registry cannot grow without bound."""
    global _active_requests_last_cleanup
    now = datetime.now(timezone.utc)
    if (now - _active_requests_last_cleanup).total_seconds() < 300:
        return 0

    _active_requests_last_cleanup = now
    threshold = now - timedelta(seconds=_ACTIVE_REQUEST_TTL_SECONDS)
    stale_keys = [
        req_id
        for req_id, (_, created_at, _) in _active_requests.items()
        if created_at < threshold
    ]
    for req_id in stale_keys:
        event, _, _ = _active_requests.pop(req_id, (None, None, None))
        if event and not event.is_set():
            event.set()
    if stale_keys:
        logger.warning(
            "active_requests_stale_cleanup",
            cleaned=len(stale_keys),
            remaining=len(_active_requests),
        )
    return len(stale_keys)


async def register(request_id: str, cancel_event: asyncio.Event, user_id: str) -> None:
    """Claim a request id for this process, owned by ``user_id``."""
    async with _get_active_requests_lock():
        _active_requests[request_id] = (
            cancel_event,
            datetime.now(timezone.utc),
            user_id,
        )
        await _cleanup_stale()


async def unregister(request_id: str) -> None:
    """Release a request id once its stream has finished."""
    async with _get_active_requests_lock():
        _active_requests.pop(request_id, None)
        await _cleanup_stale()


async def cancel(request_id: str, user_id: str) -> tuple[bool, str]:
    """Cancel here, or ask the replica that owns it. Returns
    ``(cancelled, reason)``: cancelled / already_cancelled / not_owner /
    request_not_found."""
    cancelled, reason = await cancel_local(request_id, user_id)
    if reason != "request_not_found":
        return cancelled, reason
    return await _cancel_remote(request_id, user_id)


async def cancel_local(request_id: str, user_id: str) -> tuple[bool, str]:
    """Cancel a request owned by this process, checking ownership first."""
    async with _get_active_requests_lock():
        entry = _active_requests.get(request_id)
        if not entry:
            return False, "request_not_found"
        cancel_event, _, owner_id = entry
        if owner_id != user_id:
            logger.warning(
                "cancel_request_ownership_mismatch",
                request_id=request_id,
                owner_id=owner_id,
                requester_id=user_id,
            )
            return False, "not_owner"
        if not cancel_event.is_set():
            cancel_event.set()
            return True, "cancelled"
        return False, "already_cancelled"


async def _cancel_remote(request_id: str, user_id: str) -> tuple[bool, str]:
    """Ask peer replicas to cancel a request this process doesn't own."""
    from liminallm.service.runtime import get_runtime

    bus = getattr(get_runtime(), "bus", None)
    if bus is None:
        return False, "request_not_found"
    try:
        ack = await bus.request(
            CANCEL_CHANNEL, {"request_id": request_id, "user_id": user_id}
        )
    except Exception as exc:  # never let coordination break the endpoint
        logger.warning(
            "cancel_request_bus_failed", request_id=request_id, error=str(exc)
        )
        return False, "request_not_found"
    if not ack:
        return False, "request_not_found"
    reason = str(ack.get("reason") or "request_not_found")
    return bool(ack.get("cancelled")), reason


async def handle_remote_cancel(data: dict) -> Optional[dict]:
    """Cluster-bus handler: cancel locally, or stay silent if we don't own it,
    so the asker can tell a verdict from "nobody has it".

    Ownership is re-checked here even though the peer authenticated the
    user_id, so the trust boundary is the coordination datastore, not a caller.
    """
    request_id = str(data.get("request_id") or "")
    user_id = str(data.get("user_id") or "")
    if not request_id or not user_id:
        return None
    cancelled, reason = await cancel_local(request_id, user_id)
    if reason == "request_not_found":
        return None
    return {"cancelled": cancelled, "reason": reason}
