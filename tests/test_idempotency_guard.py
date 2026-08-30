"""What a retry with the same Idempotency-Key gets back.

Asserted through the guard itself rather than by reading the stored record:
the record's shape is an implementation detail, and what callers depend on is
that a completed request replays and a failed one is allowed to run again.
"""

import pytest

from liminallm.api.idempotency import IdempotencyGuard
from liminallm.api.schemas import Envelope
from liminallm.service.runtime import get_runtime


@pytest.mark.asyncio
async def test_a_failed_request_may_be_retried():
    """The failure is recorded so the retry is not deadlocked behind an
    "in progress" record that never resolves - but it must not replay."""
    runtime = get_runtime()
    runtime._local_idempotency.clear()
    user_id, key = "user-test", "idem-failure"

    with pytest.raises(RuntimeError):
        async with IdempotencyGuard("route", user_id, key, require=True):
            raise RuntimeError("boom")

    async with IdempotencyGuard("route", user_id, key, require=True) as retry:
        assert retry.cached is None, "a failed attempt was replayed"


@pytest.mark.asyncio
async def test_a_completed_request_replays_instead_of_repeating():
    runtime = get_runtime()
    runtime._local_idempotency.clear()
    user_id, key = "user-test", "idem-success"

    async with IdempotencyGuard("route", user_id, key, require=True) as guard:
        await guard.store_result(
            Envelope(status="ok", data={"ok": True}, request_id=guard.request_id)
        )

    async with IdempotencyGuard("route", user_id, key, require=True) as retry:
        assert retry.cached is not None, "a completed request did not replay"
        assert retry.cached.data == {"ok": True}
        assert retry.cached.status == "ok"


@pytest.mark.asyncio
async def test_a_concurrent_retry_is_refused_rather_than_run_twice():
    """The slot is claimed atomically, so the loser gets a 409 instead of a
    second upload or a second turn."""
    from fastapi import HTTPException

    runtime = get_runtime()
    runtime._local_idempotency.clear()
    user_id, key = "user-test", "idem-concurrent"

    async with IdempotencyGuard("route", user_id, key, require=True):
        with pytest.raises(HTTPException) as caught:
            async with IdempotencyGuard("route", user_id, key, require=True):
                pass
        assert caught.value.status_code == 409


@pytest.mark.asyncio
async def test_a_different_key_is_a_different_request():
    runtime = get_runtime()
    runtime._local_idempotency.clear()

    async with IdempotencyGuard("route", "user-test", "key-a", require=True) as first:
        await first.store_result(
            Envelope(status="ok", data={"n": 1}, request_id=first.request_id)
        )

    async with IdempotencyGuard("route", "user-test", "key-b", require=True) as second:
        assert second.cached is None
