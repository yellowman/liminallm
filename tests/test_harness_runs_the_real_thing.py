"""The suite talks to real services, and says so when it cannot.

"Green" has twice meant less than it looked like: an in-memory store that hid
three Postgres-only bugs, and a suite running without Redis while production
runs with it. A silent fall back to the in-process path looks identical to
success, so these skip when Redis is absent rather than passing quietly.
"""

from __future__ import annotations

import functools
import os
import re

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


#: The production stack and the QA stack. Both declare the deployment, so a
#: dead variable in either is a deployment that says it configured something.
COMPOSE_FILES = ("docker-compose.yaml", "docker-compose.test.yml")


def _compose(filename: str = COMPOSE_FILES[0]) -> dict:
    import pathlib

    import yaml

    root = pathlib.Path(__file__).resolve().parent.parent
    return yaml.safe_load((root / filename).read_text())


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


def _service_env(service: str) -> dict:
    """A compose service's environment, from either of the two YAML forms."""
    environment = _compose()["services"][service].get("environment") or {}
    if isinstance(environment, dict):
        return {str(k): str(v) for k, v in environment.items()}
    pairs = (str(entry).split("=", 1) for entry in environment)
    return {p[0]: (p[1] if len(p) > 1 else "") for p in pairs}


def test_the_migrate_container_is_told_the_same_embedding_width_as_the_app():
    """One writes the column, the other checks it. They cannot disagree.

    `migrate.sh` reads EMBEDDING_VECTOR_DIM and defaults to 1536, so a migrate
    service handed only DATABASE_URL pins the vector column at 1536 whatever
    the operator configured. Asserting the key is merely *present* is not
    enough either: hard-coding the migrate service to "1536" passes that test
    and rebuilds the same bug for anyone running at 64. The two services have
    to resolve the setting the same way, so compare the values.
    """
    migrate = _service_env("migrate").get("EMBEDDING_VECTOR_DIM")
    app = _service_env("app").get("EMBEDDING_VECTOR_DIM")
    assert migrate is not None, (
        "the migrate service does not receive EMBEDDING_VECTOR_DIM, so it "
        "builds the vector column at the 1536 default regardless of what the "
        "operator set, and the app then refuses to start."
    )
    assert migrate == app, (
        f"migrate builds the vector column from {migrate!r} while the app "
        f"checks it against {app!r}. Whatever the operator sets, these two "
        "must resolve to one number or the app cannot boot."
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


def test_ci_does_not_let_pytest_repair_the_migrated_database():
    """CI must run the tests against the database migrate.sh produced.

    Running the deploy command proves the command executes; it does not prove
    it built anything. `conftest` applying the schema unconditionally closes
    that gap in the wrong direction — gut migrate.sh to `exit 0` and the
    schema step still passes, conftest builds the schema on the empty database
    it left, and the suite goes green.
    """
    import pathlib

    import yaml

    root = pathlib.Path(__file__).resolve().parent.parent
    workflow = yaml.safe_load((root / ".github" / "workflows" / "tests.yml").read_text())
    steps = workflow["jobs"]["test"]["steps"]

    migrated = next(
        (i for i, s in enumerate(steps) if "migrate.sh" in str(s.get("run", ""))), None
    )
    assert migrated is not None, "no CI step runs scripts/migrate.sh"
    # `pytest` as a command, not `pip install pytest` in the setup step.
    invokes_pytest = re.compile(r"^\s*pytest\b", re.MULTILINE)
    testing = next(
        (i for i, s in enumerate(steps) if invokes_pytest.search(str(s.get("run", "")))),
        None,
    )
    assert testing is not None, "no CI step runs pytest"
    assert migrated < testing, "CI runs pytest before it applies the schema"
    assert (steps[testing].get("env") or {}).get("TEST_SCHEMA_PREPARED"), (
        "the pytest step does not set TEST_SCHEMA_PREPARED, so conftest "
        "reapplies sql/schema.sql over whatever migrate.sh did — or did not — "
        "produce, and the suite cannot fail because of it."
    )


@pytest.mark.slow
def test_a_prepared_database_that_is_empty_stops_the_suite():
    """The other half: the flag has to actually suppress the repair.

    Setting TEST_SCHEMA_PREPARED in CI is only worth anything if the harness
    honours it. Against an empty database that claims to be prepared, the
    suite must refuse — that refusal is exactly what a no-op migrate.sh would
    produce in CI, and it is the signal the previous arrangement swallowed.
    """
    import subprocess
    import sys

    from tests.harness import ScratchPostgres

    pg = ScratchPostgres()
    if not pg.available:
        pytest.skip("initdb not available; cannot build an unprepared database")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    url = pg.start()
    try:
        env = {
            k: v for k, v in os.environ.items()
            if k not in {"DATABASE_URL", "TEST_PG_PORT", "SHARED_FS_ROOT"}
        }
        env.update(
            TEST_DATABASE_URL=url,
            TEST_SCHEMA_PREPARED="true",
            EMBEDDING_VECTOR_DIM="64",
            ALLOW_REDIS_FALLBACK_DEV="true",
        )
        done = subprocess.run(
            [sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider",
             "tests/test_harness_runs_the_real_thing.py"
             "::test_redis_url_has_no_environment_variable"],
            cwd=root, env=env, capture_output=True, text=True, timeout=600,
        )
    finally:
        pg.stop()

    assert done.returncode != 0, (
        "the suite passed against an empty database it was told was already "
        "prepared. TEST_SCHEMA_PREPARED is not suppressing the repair, so a "
        "migrate.sh that built nothing would still go green in CI."
    )
    # Not "does the word migrate appear" — the tracebacks of this very file
    # would satisfy that. The refusal has to be the store's schema check.
    output = done.stdout + done.stderr
    assert "Missing required Postgres tables" in output, (
        "the suite failed, but not because the schema was absent, so this "
        f"proves nothing about TEST_SCHEMA_PREPARED:\n{output[-3000:]}"
    )
    assert "scripts/migrate.sh" in output, (
        f"the refusal does not name the command that fixes it:\n{output[-3000:]}"
    )


# --- The shared store must be the store production would have built ---------
#
# One store now serves the whole session instead of being rebuilt twice per
# test. That is worth 23% of the suite's wall clock, and it moved two facts
# about the environment that no assertion was watching: which filesystem root
# the store writes under, and whether the bootstrap artifacts a fresh boot
# seeds are still there after the per-test TRUNCATE.


def test_the_store_writes_where_the_runtime_thinks_it_does():
    """`store.fs_root` and `settings.shared_fs_root` are one directory.

    A runtime-built store is handed `settings.shared_fs_root`, so the two
    agreed by construction. The shared store is built by the harness, and it
    used to mint a second temporary directory of its own — leaving artifact
    payloads written under one root while filesystem authority, adapters,
    archive staging and the interpreter all resolved paths under another.

    Nothing failed, because almost nothing reads both. Artifact retirement
    reads both.
    """
    from pathlib import Path

    runtime = get_runtime()
    assert (
        Path(runtime.store.fs_root).resolve()
        == Path(runtime.settings.shared_fs_root).resolve()
    ), (
        f"store writes under {runtime.store.fs_root} while the runtime "
        f"resolves paths under {runtime.settings.shared_fs_root}"
    )


def test_the_bootstrap_artifacts_survive_the_per_test_truncate():
    """Every test starts from the state a fresh boot produces.

    `_ensure_default_artifacts` runs in `PostgresStore.__init__` and seeds the
    default chat workflow and tool specs. While the store was rebuilt per
    test, the per-test TRUNCATE was undone by the next construction. With one
    store for the session that construction happens once, so the first
    TRUNCATE removed the defaults and the remaining tests ran without them —
    exercising fallbacks where production runs on seeded rows.
    """
    artifacts = get_runtime().store.list_artifacts()
    assert any(a.name == "default_chat_workflow" for a in artifacts), (
        "the default chat workflow is missing, so this test — and every test "
        "after the first — runs in a boot state production never has"
    )
    tools = {
        a.schema.get("name")
        for a in artifacts
        if isinstance(a.schema, dict) and a.schema.get("kind") == "tool.spec"
    }
    assert tools, "the seeded tool specs are missing for the same reason"


# The one piece of per-test state the shared store does carry.
# `PostgresStore.sessions` is an in-memory dictionary that `_truncate_all`
# cannot reach, so with a session-wide store it accumulated across the whole
# run. Not the primary read path today, which is why this is isolation rather
# than a product bug — but a cache whose contents depend on test order is a bad
# thing to leave in place. These two run in file order: the first dirties the
# cache, the second requires it to have been cleared between them.


def test_a_session_cached_here_dirties_the_shared_store():
    store = get_runtime().store
    with store._session_lock:
        store.sessions["leaked-from-the-previous-test"] = object()
    assert store.sessions


def test_the_stores_session_cache_does_not_leak_between_tests():
    assert get_runtime().store.sessions == {}, (
        "a session cached by the previous test is still here, so the cache "
        "grows with test order across the whole run"
    )


def test_compose_does_not_seed_the_filesystem_root_through_settings():
    """`shared_fs_root` is environment-only; Compose has to agree.

    It is no longer a managed setting, so a `shared_fs_root` key in
    INSTANCE_SETTINGS_JSON is filtered out as unknown — silently. The stack
    kept working only because the environment default happened to equal the
    mounted path.
    """
    import json as _json

    app = _compose()["services"]["app"]
    seed = (app.get("environment") or {}).get("INSTANCE_SETTINGS_JSON")
    if seed:
        assert "shared_fs_root" not in _json.loads(seed), (
            "compose still seeds shared_fs_root as a managed setting, which "
            "is now silently ignored"
        )


def test_compose_mounts_the_volume_where_the_app_will_look_for_it():
    """One root, declared once, used by both the environment and the mount."""
    app = _compose()["services"]["app"]
    root = _service_env("app").get("SHARED_FS_ROOT")
    assert root, (
        "the app container is not told SHARED_FS_ROOT, so the documented way "
        "to move the data root does nothing under Compose"
    )
    # Split on the first colon only: a target like `${VAR:-/default}` has
    # colons of its own.
    targets = [str(v).split(":", 1)[1] for v in (app.get("volumes") or [])
               if ":" in str(v)]
    assert root in targets, (
        f"the app resolves its data root at {root} while the volume is "
        f"mounted at {targets}"
    )


@functools.lru_cache(maxsize=1)
def _sources() -> tuple[str, ...]:
    """Everything in this repository that could read an environment variable.

    Read once: the caller asks about thirty names, and re-walking the package
    for each one costs seconds in a lane whose wall clock is measured.
    """
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    texts = []
    for path in (*root.glob("liminallm/**/*.py"), *root.glob("scripts/*")):
        if not path.is_file():
            continue
        try:
            texts.append(path.read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            continue
    return tuple(texts)


def _anything_reads(name: str) -> bool:
    """Does this repository read this environment variable anywhere?"""
    needles = (f'"{name}"', f"'{name}'", f"${name}", "${%s}" % name)
    return any(needle in text for text in _sources() for needle in needles)


@pytest.mark.parametrize("filename", COMPOSE_FILES)
def test_no_compose_variable_reaches_nothing(filename):
    """A variable nothing reads is a configuration that only looks applied.

    `REDIS_URL` sat in both files beside the settings that do work. But
    `redis_url` is a managed setting with no environment variable of its own,
    so the deployment ran on the in-process fallback — rate limits,
    idempotency, the session cache and the concurrency slots all on their
    fallback path — while its own compose file said otherwise. A dead name is
    indistinguishable from a live one until something measures the behaviour
    it claims to set, which is what this does.

    Only services this repository builds are checked. A service that names an
    `image:` runs somebody else's entrypoint, and `POSTGRES_PASSWORD` is read
    by code this repository cannot see.
    """
    services = _compose(filename).get("services") or {}
    dead = {}
    for name, body in services.items():
        if body.get("image"):
            continue
        environment = body.get("environment") or {}
        declared = (
            list(environment)
            if isinstance(environment, dict)
            else [str(entry).split("=", 1)[0] for entry in environment]
        )
        unread = sorted(key for key in declared if not _anything_reads(key))
        if unread:
            dead[name] = unread
    assert not dead, (
        f"{filename} declares variables nothing reads: {dead}. Either the "
        "setting is managed and belongs in INSTANCE_SETTINGS_JSON, or the "
        "name is stale and should go — leaving it makes the file claim a "
        "configuration the deployment never receives."
    )
