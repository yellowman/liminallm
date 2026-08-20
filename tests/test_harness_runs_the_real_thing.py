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


# --- One schema authority, and it is the one production runs -----------------
#
# The same principle as the store and the cache above, one layer down. The
# schema a deploy ends up with is only trustworthy if exactly one thing writes
# it, with the parameter it needs. Two of these were live defects: Docker had a
# second executor that ran first and won, and the migration container did not
# receive the embedding width, so `EMBEDDING_VECTOR_DIM=64 docker compose up`
# built a 1536-wide database that then refused to boot.


def _compose() -> dict:
    import pathlib

    import yaml

    root = pathlib.Path(__file__).resolve().parent.parent
    return yaml.safe_load((root / "docker-compose.yaml").read_text())


def test_docker_has_one_schema_authority():
    """`scripts/migrate.sh` applies the schema. Nothing else may.

    Mounting `sql/` into `/docker-entrypoint-initdb.d` makes Postgres apply
    `schema.sql` itself on first boot — before the migrate service runs, and
    without the `-v embedding_dim` that only `migrate.sh` passes. Every
    `CREATE TABLE IF NOT EXISTS` in the real run is then a no-op, so the
    entrypoint's defaults are what the database keeps.
    """
    volumes = _compose()["services"]["postgres"].get("volumes") or []
    offenders = [v for v in volumes if "docker-entrypoint-initdb.d" in str(v)]
    assert not offenders, (
        f"postgres mounts {offenders}: a second thing applies the schema, and "
        "it runs first. scripts/migrate.sh is the only schema authority."
    )


def test_the_migrate_container_is_told_the_embedding_width():
    """`migrate.sh` reads EMBEDDING_VECTOR_DIM and defaults to 1536.

    A migrate service that is handed only DATABASE_URL therefore pins the
    vector column at 1536 whatever the operator configured, and the app —
    which checks the column against the encoder at startup — refuses to boot
    with no indication that the width came from a container that never saw
    the setting.
    """
    environment = _compose()["services"]["migrate"].get("environment") or {}
    keys = environment if isinstance(environment, dict) else [
        str(entry).split("=", 1)[0] for entry in environment
    ]
    assert "EMBEDDING_VECTOR_DIM" in keys, (
        "the migrate service does not receive EMBEDDING_VECTOR_DIM, so it "
        "builds the vector column at the 1536 default regardless of what the "
        "operator set, and the app then refuses to start."
    )


def test_ci_applies_the_schema_the_way_production_does():
    """CI must run the command a deploy runs, not reimplement it.

    Calling psql on sql/schema.sql directly means scripts/migrate.sh — the
    command SPEC §13.6 names and Docker invokes — is never executed by CI, so
    a break in it is found by an operator instead.
    """
    import pathlib

    import yaml

    root = pathlib.Path(__file__).resolve().parent.parent
    workflow = yaml.safe_load((root / ".github" / "workflows" / "tests.yml").read_text())
    steps = workflow["jobs"]["test"]["steps"]
    schema_steps = [s for s in steps if "schema.sql" in str(s.get("run", ""))]
    assert not schema_steps, (
        "CI applies sql/schema.sql with its own psql invocation; it should "
        "run scripts/migrate.sh so the deploy path is what gets exercised."
    )
    assert any("migrate.sh" in str(s.get("run", "")) for s in steps), (
        "no CI step runs scripts/migrate.sh"
    )
