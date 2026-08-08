"""The suite talks to real services, and says so when it cannot.

"Green" has twice meant less than it looked like: an in-memory store that hid
three Postgres-only bugs, and a suite running without Redis while production
runs with it. A silent fall back to the in-process path looks identical to
success, so these skip when Redis is absent rather than passing quietly.
"""

from __future__ import annotations

import os

import pytest

from liminallm.service.runtime import get_runtime


def _redis_expected() -> bool:
    """Was a real server started or supplied for this run?"""
    import shutil

    return bool(os.environ.get("TEST_REDIS_URL")) or bool(shutil.which("redis-server"))


def test_the_store_is_the_one_production_runs():
    from liminallm.storage.postgres import PostgresStore

    assert isinstance(get_runtime().store, PostgresStore)


def test_redis_is_actually_connected():
    if not _redis_expected():
        pytest.skip("no redis-server available in this environment")
    cache = get_runtime().cache
    assert cache is not None, (
        "The runtime fell back to in-process state. Rate limits, idempotency, "
        "the session cache and concurrency slots are then not the code "
        "production runs, and a green suite does not mean they work."
    )


def test_redis_is_the_async_client_production_uses():
    if not _redis_expected():
        pytest.skip("no redis-server available in this environment")
    from liminallm.storage.redis_cache import RedisCache

    assert isinstance(get_runtime().cache, RedisCache)


@pytest.mark.asyncio
async def test_a_rate_limit_round_trips_through_redis():
    if not _redis_expected():
        pytest.skip("no redis-server available in this environment")
    runtime = get_runtime()
    key = "harness:probe"
    allowed, remaining, _ = await runtime.cache.check_rate_limit(
        key, limit=2, window_seconds=60, return_remaining=True
    )
    assert allowed and remaining >= 0
    await runtime.cache.check_rate_limit(key, 2, 60, return_remaining=True)
    allowed, _, _ = await runtime.cache.check_rate_limit(
        key, 2, 60, return_remaining=True
    )
    assert not allowed, "the bucket never emptied — is this the real client?"


def test_redis_url_has_no_environment_variable():
    """Exporting REDIS_URL does nothing, which is why conftest moves the
    default instead. If this ever gains an env var, conftest should use it."""
    from liminallm.config import Settings

    extra = Settings.model_fields["redis_url"].json_schema_extra or {}
    assert "env" not in extra, (
        "redis_url gained an environment variable; point the harness at it "
        "rather than patching the field default."
    )
