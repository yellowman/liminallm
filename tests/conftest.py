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
# Tests run against a real Postgres. See tests/pgharness.py: the in-memory
# store used to double the storage layer, so every storage feature was written
# twice and verified once — and the untested half was the one production runs.
os.environ.setdefault("EMBEDDING_VECTOR_DIM", "64")  # matches the hash encoder
os.environ.setdefault("ALLOW_REDIS_FALLBACK_DEV", "true")
os.environ.setdefault("JWT_SECRET", "Test-Secret-Key-4-Testing-Only-Do-Not-Use-In-Production!")
# Tests use the same async RedisCache production does. A second, synchronous
# implementation used to exist for the test suite; it silently drifted eight
# methods behind and broke the attachment agent whenever Redis was present.
# Falls back to in-process state if Redis is absent (ALLOW_REDIS_FALLBACK_DEV).
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/1")

import pytest  # noqa: E402

from tests.pgharness import (  # noqa: E402
    ScratchPostgres,
    apply_schema,
    close_test_store,
    get_test_store,
)

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
