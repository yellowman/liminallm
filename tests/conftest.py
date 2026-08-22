import asyncio
import inspect
import os
import shutil
import sys
import tempfile
from pathlib import Path

# A throwaway root for everything this run writes: artifact payloads,
# adapters, uploaded files, lock files. `shared_fs_root` is environment-only
# (see config.env_field), so setting it here before any import genuinely takes
# effect — for a while it did not, because the field was database-managed and
# this line was inert, and the suite wrote into /srv/liminallm, which is where
# a real install keeps its data.
#
# Named for the worker that owns it, because under xdist "which root is this"
# is a question somebody will need answered from a directory listing, and
# because a test can then assert that workers were actually given different
# ones rather than that the code meant to.
from tests.harness import run_id, worker_id  # noqa: E402

_worker = worker_id()
_test_tmp_dir = tempfile.mkdtemp(
    prefix=f"liminallm_test_{_worker}_" if _worker else "liminallm_test_"
)
os.environ["SHARED_FS_ROOT"] = _test_tmp_dir
# Stored in the controller's environment at import, which is before execnet
# spawns any worker, so every worker in this invocation reads the same one.
run_id()
os.environ.setdefault("TEST_MODE", "true")
# Tests run against a real Postgres. See tests/harness.py: the in-memory
# store used to double the storage layer, so every storage feature was written
# twice and verified once — and the untested half was the one production runs.
os.environ.setdefault("EMBEDDING_VECTOR_DIM", "64")  # matches the hash encoder
os.environ.setdefault("ALLOW_REDIS_FALLBACK_DEV", "true")
import pytest  # noqa: E402

from tests.harness import (  # noqa: E402
    REDIS_LEASE_TTL,
    ScratchPostgres,
    ScratchRedis,
    _base_in_use_error,
    apply_schema,
    claim_redis_database,
    close_test_store,
    create_worker_database,
    drop_worker_database,
    get_test_store,
    redis_database_index,
    release_redis_database,
    renew_redis_database,
    reserve_base_database,
    reset_shared_store,
)

# Tests use the same async RedisCache production does, against a real
# redis-server. A second, synchronous implementation used to exist for the
# suite alone; it drifted eight methods behind and broke the attachment agent
# the moment Redis was present. Running without Redis has the same shape of
# problem one layer down — rate limits, idempotency, the session cache and the
# concurrency slots all take their fallback path, so the code production runs
# is the code the suite does not.
_REDIS = None
_PG = None
_WORKER_DB: str | None = None
_OWNED_REDIS_DB = False
_REDIS_LEASE: tuple[str, int, str] | None = None
#: The externally supplied database this run was pointed at, when there is
#: one. Held so the per-test hook can keep its reservation alive; a serial run
#: has no lease to renew but still needs the number kept out of circulation.
_REDIS_BASE: str | None = None


def _provision(worker: str) -> None:
    """Start or derive every service this process will actually use.

    Called from `pytest_configure`, and not from module import, because of one
    measured fact: under xdist the controller imports this file but never
    imports a test module. Provisioning at import gave the controller a
    Postgres cluster and a redis-server it had no use for — and worse, a
    connection pool on the database the workers were about to clone, which
    `CREATE DATABASE ... TEMPLATE` refuses while any session holds it.

    Serial runs reach this the same way they always did, one process doing all
    of it, and nothing below behaves differently for them.
    """
    global _REDIS, _PG, _WORKER_DB, _OWNED_REDIS_DB, _REDIS_LEASE, _REDIS_BASE

    # Tests use the same async RedisCache production does, against a real
    # redis-server. A second, synchronous implementation used to exist for the
    # suite alone; it drifted eight methods behind and broke the attachment
    # agent the moment Redis was present. Running without Redis has the same
    # shape of problem one layer down — rate limits, idempotency, the session
    # cache and the concurrency slots all take their fallback path, so the code
    # production runs is the code the suite does not.
    redis_url = os.environ.get("TEST_REDIS_URL")
    if redis_url:
        if worker:
            # A numbered database nobody else is using. Claimed rather than
            # derived from the worker id: that derivation is a function of
            # `gw0`, so two pytest invocations at once would pick the same
            # number and each would flush it believing it owned it.
            holder = f"{os.environ['LIMINALLM_TEST_RUN']}:{worker}"
            leased, index = claim_redis_database(redis_url, holder)
            _REDIS_LEASE = (redis_url, index, holder)
            _REDIS_BASE = redis_url
            redis_url = leased
            _OWNED_REDIS_DB = True
        else:
            # Serial, against somebody's Redis. This run leases nothing, so
            # nothing else was recording that the database it is about to use
            # is spoken for — and an xdist run starting alongside it would
            # lease that very number and flush it before every test. One
            # terminal running `make test` and another running the parallel
            # lane is an ordinary pair.
            #
            # Nothing is claimed here: the database is the caller's and this
            # run does not empty it. The reservation only keeps it out of the
            # candidate list.
            if not reserve_base_database(redis_url):
                raise _base_in_use_error(redis_database_index(redis_url))
            _REDIS_BASE = redis_url
    else:
        _REDIS = ScratchRedis()
        if _REDIS.available:
            redis_url = _REDIS.start()
            _OWNED_REDIS_DB = True
        else:
            # No redis-server here. The suite still runs on the documented
            # fallback, which is what a Redis outage does in production — but
            # say so, because a green run then means less than it looks like.
            _REDIS = None
            print("redis-server not found: running on the in-process fallback")
    if redis_url:
        # redis_url is a database-managed setting with no environment variable
        # of its own, so exporting REDIS_URL does nothing — which is what
        # conftest used to do, and why the suite ran on the fallback while
        # looking configured. Move the *default* instead of storing a value:
        # seeding through INSTANCE_SETTINGS_JSON would spend the instance's one
        # first boot, and writing a row would make every "has an operator
        # configured anything" check answer yes.
        from liminallm.config import SYSTEM_SETTINGS_DEFAULTS, Settings

        Settings.model_fields["redis_url"].default = redis_url
        Settings.model_rebuild(force=True)
        SYSTEM_SETTINGS_DEFAULTS["redis_url"] = redis_url

    base_url = os.environ.get("TEST_DATABASE_URL")
    prepared = bool(os.environ.get("TEST_SCHEMA_PREPARED"))
    if not base_url:
        # A cluster of its own, so an xdist worker needs nothing derived: it
        # already owns the whole server.
        _PG = ScratchPostgres()
        if not _PG.available:
            raise RuntimeError(
                "The test suite needs Postgres (initdb not found). Install "
                "postgresql-16 + postgresql-16-pgvector, or set TEST_DATABASE_URL."
            )
        os.environ["DATABASE_URL"] = _PG.start()
    elif worker:
        # Shared server. Each worker gets a database of its own, because the
        # per-test TRUNCATE below assumes exclusive ownership — four workers
        # truncating one database is not flakiness, it is every test deleting
        # every other test's rows.
        _WORKER_DB = create_worker_database(
            base_url, worker, os.environ["LIMINALLM_TEST_RUN"], prepared=prepared
        )
        os.environ["DATABASE_URL"] = _WORKER_DB
    else:
        os.environ["DATABASE_URL"] = base_url

    # TEST_SCHEMA_PREPARED says "something already applied the schema to this
    # database; do not touch it". CI sets it after running scripts/migrate.sh,
    # so the suite runs against the database the deploy command actually
    # produced.
    #
    # Without it, applying the schema here unconditionally means the suite
    # proves nothing about migrate.sh: gut the script to `exit 0` and CI's
    # schema step still succeeds, conftest then builds the whole schema from
    # scratch on the empty database it left behind, and the suite goes green
    # over a deploy command that does nothing. A scratch cluster this file
    # started has no such ambiguity — nothing else could have prepared it — so
    # it still applies the schema itself.
    #
    # A worker database is already in the right state either way:
    # `create_worker_database` clones a prepared one and builds an unprepared
    # one, which is the same distinction made one level up.
    if not prepared and _WORKER_DB is None:
        apply_schema(os.environ["DATABASE_URL"], embedding_dim=64)

    use_shared_store(get_test_store())


from fastapi.dependencies import utils as fastapi_dep_utils  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from liminallm.service.runtime import (  # noqa: E402
    reset_runtime_for_tests,
    use_shared_store,
)

# One store per process. The runtime is rebuilt before and after every test,
# and building one used to construct a connection pool and re-verify the whole
# schema each time — measured at about a quarter of the suite's wall clock.
# What per-test state the store does carry — its in-memory session cache, and
# the bootstrap artifacts TRUNCATE removes — is restored by
# `reset_shared_store` below, which costs a fraction of rebuilding it.
#
# Created in `_provision`, not here, so the xdist controller does not open a
# pool on a database its workers are about to clone.

# Avoid import-time failures for routes that rely on python-multipart in constrained test environments.
fastapi_dep_utils.ensure_multipart_is_installed = lambda: None


@pytest.fixture
def store():
    return get_test_store()


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from liminallm import app as app_module

    return TestClient(app_module.app)


def _signup(client, prefix, *, admin=False):
    import uuid

    from liminallm.service.runtime import get_runtime

    email = f"{prefix}_{uuid.uuid4().hex[:8]}@example.com"
    password = "TestPassword123!"
    resp = client.post("/v1/auth/signup", json={"email": email, "password": password})
    assert resp.status_code == 201, resp.text
    if admin:
        get_runtime().store.update_user_role(resp.json()["data"]["user_id"], role="admin")
        # Re-login: the role is in the token, so the signup one is stale.
        resp = client.post(
            "/v1/auth/login", json={"email": email, "password": password}
        )
        assert resp.status_code == 200, resp.text
    return {"Authorization": f"Bearer {resp.json()['data']['access_token']}"}


@pytest.fixture
def auth_headers(client):
    return _signup(client, "user")


@pytest.fixture
def admin_headers(client):
    return _signup(client, "admin", admin=True)


def _flush_owned_redis() -> None:
    """Empty this process's Redis database between tests.

    Only when we own it — a scratch server we started, or the numbered
    database derived for this worker. An externally supplied base database in
    a serial run is not ours to empty, and flushing it is the Redis-side
    version of four workers truncating one Postgres.

    Without this, isolation between tests rested on every key carrying a fresh
    UUID and on TTLs expiring. That holds until one test asserts something
    about a key another test's name happened to collide with.
    """
    if _REDIS_BASE is not None and _REDIS_LEASE is None:
        # A serial run against somebody's Redis: no lease to renew, but the
        # reservation that keeps its database out of other runs' candidate
        # lists still has to outlive the suite. The measured serial lane is
        # 881s against a 900s TTL, so this is not a margin to leave to one
        # write at provisioning time.
        if not reserve_base_database(_REDIS_BASE):
            raise RuntimeError(
                f"Redis database {redis_database_index(_REDIS_BASE)} is no "
                "longer reserved for this run — another run has leased it and "
                "is emptying it between its own tests. Re-run; if this "
                "repeats, the two runs need different databases."
            )
    if not _OWNED_REDIS_DB:
        return
    if _REDIS_LEASE is not None:
        # Renewed here rather than on a timer: this runs before every test, so
        # a live run keeps its claim and a dead one stops renewing and gives
        # its database back on its own.
        #
        # And if the claim is gone, this stops. A lease that expired has very
        # likely been taken by another run that is using that database right
        # now, and the next statement in this function empties it. Losing
        # ownership is not something to carry on best-effort through.
        if not renew_redis_database(*_REDIS_LEASE):
            raise RuntimeError(
                f"this run no longer holds Redis database {_REDIS_LEASE[1]}. "
                "Another run has most likely claimed it, and continuing would "
                "empty a database in use. Re-run; if this repeats, a test is "
                f"outrunning the {REDIS_LEASE_TTL}s lease."
            )
    from liminallm.config import get_settings

    url = get_settings().redis_url
    if not url:
        return
    try:
        from redis import Redis

        client = Redis.from_url(url, decode_responses=True)
        try:
            client.flushdb()
        finally:
            client.close()
    except Exception:  # pragma: no cover - a Redis that went away mid-run
        pass


def _truncate_all() -> None:
    """Wipe every table between tests.

    One database serves this process; this is what makes tests independent of
    each other without a cluster per test. Under xdist that database belongs
    to one worker — see `_provision` — because this statement assumes nothing
    else is looking at it.
    """
    pg = get_test_store()
    with pg.pool.connection() as conn:
        rows = conn.execute(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
        ).fetchall()
        tables = [row["tablename"] for row in rows]
        if tables:
            conn.execute(
                f"TRUNCATE {', '.join(tables)} RESTART IDENTITY CASCADE"
            )
        conn.commit()


@pytest.fixture(autouse=True)
def reset_runtime_state():
    _truncate_all()
    _flush_owned_redis()
    # TRUNCATE takes the bootstrap artifacts with everything else, and it
    # cannot reach the store's in-memory session cache. The store is built
    # once for the session, so nothing else puts either back.
    reset_shared_store(get_test_store())
    reset_runtime_for_tests()
    yield
    reset_runtime_for_tests()


def pytest_pyfunc_call(pyfuncitem):
    if inspect.iscoroutinefunction(pyfuncitem.obj):
        call_kwargs = {
            name: pyfuncitem.funcargs[name]
            for name in pyfuncitem._fixtureinfo.argnames
            if name in pyfuncitem.funcargs
        }
        asyncio.run(pyfuncitem.obj(**call_kwargs))
        return True
    return None


def pytest_configure(config):
    config.addinivalue_line("markers", "asyncio: mark test as async")
    # `slow` is declared in pyproject.toml. Declaring it a second time here
    # with different words is how two descriptions of one marker drift.

    # The xdist controller imports this file and then never imports a test
    # module — measured — so it has no use for a database, a Redis or a store,
    # and holding a pool on the database its workers are about to clone would
    # stop them cloning it. `workerinput` is xdist's own answer to "am I a
    # worker"; `dist` is its answer to "is this run parallel at all".
    #
    # The worker id comes from `config`, not from the environment variable it
    # was set in. A serial pytest launched from inside a worker — which the
    # harness's own tests do — inherits `PYTEST_XDIST_WORKER` and would
    # otherwise provision itself as a worker of a run it is not part of.
    is_worker = hasattr(config, "workerinput")
    is_controller = not is_worker and config.getoption("dist", "no") != "no"
    if is_controller:
        return
    _provision(config.workerinput["workerid"] if is_worker else "")


def pytest_sessionfinish(session, exitstatus):
    close_test_store()
    # After the pool is closed, so nothing is connected to it.
    if _WORKER_DB is not None:
        drop_worker_database(os.environ["TEST_DATABASE_URL"], _WORKER_DB)
    if _REDIS_LEASE is not None:
        release_redis_database(*_REDIS_LEASE)
    shutil.rmtree(_test_tmp_dir, ignore_errors=True)
    if _PG is not None:
        _PG.stop()
    if _REDIS is not None:
        _REDIS.stop()
