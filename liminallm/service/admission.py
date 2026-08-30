"""Per-user concurrency caps (SPEC §18), named in one place.

They were named in two, and the two disagreed: the WebSocket path never took
the inference slot, so ``max_concurrent_inference`` applied only to the
endpoint the UI does not use.

Without Redis there is nowhere to count, so every acquire succeeds. The cap is
a fairness control; failing closed would make Redis load-bearing.
"""

from __future__ import annotations

import contextlib
from typing import AsyncIterator

from liminallm.service.errors import ConflictError

#: Slot kind -> the setting that caps it.
_LIMIT_SETTING = {
    "workflow": "max_concurrent_workflows",
    "inference": "max_concurrent_inference",
    "websocket": "max_websocket_connections_per_user",
}

#: The slots a chat turn occupies, whatever transport carried it.
CHAT_SLOTS = ("workflow", "inference")


class AtCapacity(ConflictError):
    """A per-user concurrency cap is full (409, code ``busy``)."""

    error_code = "busy"


async def acquire(runtime, kind: str, user_id: str) -> bool:
    """Take one slot of ``kind`` for ``user_id``, or raise :class:`AtCapacity`."""
    if not runtime.cache:
        return True
    limit = getattr(runtime.settings, _LIMIT_SETTING[kind])
    acquired, _current = await runtime.cache.acquire_concurrency_slot(
        kind, user_id, limit
    )
    if not acquired:
        raise AtCapacity(f"concurrent {kind} limit ({limit}) exceeded")
    return True


async def release(runtime, kind: str, user_id: str) -> None:
    """Give back one slot of ``kind``. Safe to call for a slot never taken."""
    if runtime.cache:
        await runtime.cache.release_concurrency_slot(kind, user_id)


@contextlib.asynccontextmanager
async def slots(runtime, user_id: str, *kinds: str) -> AsyncIterator[None]:
    """Hold several slots for a block, releasing everything actually taken -
    including when a later acquire is what fails."""
    taken: list[str] = []
    try:
        for kind in kinds:
            await acquire(runtime, kind, user_id)
            taken.append(kind)
        yield
    finally:
        for kind in reversed(taken):
            with contextlib.suppress(Exception):
                await release(runtime, kind, user_id)
