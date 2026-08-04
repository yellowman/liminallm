import asyncio
import inspect
import os
import sys
import tempfile
from pathlib import Path

# Create temp directory for tests before any imports that might initialize runtime
_test_tmp_dir = tempfile.mkdtemp(prefix="liminallm_test_")
os.environ.setdefault("SHARED_FS_ROOT", _test_tmp_dir)
os.environ.setdefault("TEST_MODE", "true")
# Tests run against a real Postgres. See tests/harness.py: the in-memory
# store used to double the storage layer, so every storage feature was written
# twice and verified once — and the untested half was the one production runs.
os.environ.setdefault("EMBEDDING_VECTOR_DIM", "64")  # matches the hash encoder
os.environ.setdefault("ALLOW_REDIS_FALLBACK_DEV", "true")
os.environ.setdefault("JWT_SECRET", "Test-Secret-Key-4-Testing-Only-Do-Not-Use-In-Production!")
import pytest  # noqa: E402

from tests.harness import (  # noqa: E402
    ScratchPostgres,
    ScratchRedis,
    apply_schema,
    close_test_store,
    get_test_store,
)

# Tests use the same async RedisCache production does, against a real
# redis-server. A second, synchronous implementation used to exist for the
# suite alone; it drifted eight methods behind and broke the attachment agent
# the moment Redis was present. Running without Redis has the same shape of
# problem one layer down — rate limits, idempotency, the session cache and the
# concurrency slots all take their fallback path, so the code production runs
# is the code the suite does not.
_REDIS = None
_redis_url = os.environ.get("TEST_REDIS_URL")
if not _redis_url:
    _REDIS = ScratchRedis()
    if _REDIS.available:
        _redis_url = _REDIS.start()
    else:
        # No redis-server here. The suite still runs on the documented
        # fallback, which is what a Redis outage does in production — but say
        # so, because a green run then means less than it looks like.
        _REDIS = None
        print("redis-server not found: running on the in-process fallback")
if _redis_url:
    # redis_url is a database-managed setting with no environment variable of
    # its own, so exporting REDIS_URL does nothing — which is what conftest
    # used to do, and why the suite ran on the fallback while looking
    # configured. Move the *default* instead of storing a value: seeding
    # through INSTANCE_SETTINGS_JSON would spend the instance's one first boot,
    # and writing a row would make every "has an operator configured anything"
    # check answer yes.
    from liminallm.config import SYSTEM_SETTINGS_DEFAULTS, Settings  # noqa: E402

    Settings.model_fields["redis_url"].default = _redis_url
    Settings.model_rebuild(force=True)
    SYSTEM_SETTINGS_DEFAULTS["redis_url"] = _redis_url

_PG = None
if not os.environ.get("TEST_DATABASE_URL"):
    _PG = ScratchPostgres()
    if not _PG.available:
        raise RuntimeError(
            "The test suite needs Postgres (initdb not found). Install "
            "postgresql-16 + postgresql-16-pgvector, or set TEST_DATABASE_URL."
        )
    os.environ["DATABASE_URL"] = _PG.start()
else:
    os.environ["DATABASE_URL"] = os.environ["TEST_DATABASE_URL"]
apply_schema(os.environ["DATABASE_URL"], embedding_dim=64)
from fastapi.dependencies import utils as fastapi_dep_utils  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from liminallm.service.runtime import reset_runtime_for_tests  # noqa: E402

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


def _truncate_all() -> None:
    """Wipe every table between tests.

    One database serves the whole session; this is what makes tests independent
    of each other without a cluster per test.
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


def pytest_sessionfinish(session, exitstatus):
    close_test_store()
    if _PG is not None:
        _PG.stop()
    if _REDIS is not None:
        _REDIS.stop()
