import pytest

from liminallm.api.routes import Envelope, IdempotencyGuard, _get_cached_idempotency_record
from liminallm.service.runtime import get_runtime


@pytest.mark.asyncio
async def test_idempotency_guard_records_failures():
    runtime = get_runtime()
    runtime._local_idempotency.clear()
    user_id = "user-test"
    key = "idem-failure"

    guard = IdempotencyGuard("route", user_id, key, require=True)
    with pytest.raises(RuntimeError):
        async with guard:
            raise RuntimeError("boom")

    record = await _get_cached_idempotency_record(runtime, "route", user_id, key)
    assert record["status"] == "failed"
    assert record["response"]["error"]["message"] == "boom"


@pytest.mark.asyncio
async def test_idempotency_guard_keeps_completed_status():
    runtime = get_runtime()
    runtime._local_idempotency.clear()
    user_id = "user-test"
    key = "idem-success"

    guard = IdempotencyGuard("route", user_id, key, require=True)
    async with guard:
        envelope = Envelope(status="ok", data={"ok": True}, request_id=guard.request_id)
        await guard.store_result(envelope)

    record = await _get_cached_idempotency_record(runtime, "route", user_id, key)
    assert record["status"] == "completed"
    assert record["response"]["data"] == {"ok": True}
