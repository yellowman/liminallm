"""Per-user concurrency caps for expensive work.

One place that knows which slots a chat turn takes. It used to be two: the HTTP
route took a workflow slot and an inference slot, the WebSocket route took a
websocket slot and a workflow slot and never took the inference slot at all —
so ``max_concurrent_inference`` was enforced only on the path the UI does not
use. Naming the set of slots once is what stops that from recurring.

Without Redis there is nowhere to count across replicas, so every acquire
succeeds. That is deliberate: the cap is a fairness control, not a correctness
one, and failing closed would make Redis load-bearing for serving traffic.
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
    """Hold several slots for the duration of a block.

    Acquires in order and releases everything actually taken, including when a
    later acquire is the thing that fails — the leak that made hand-rolled
    acquire/release pairs worth replacing.
    """
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
