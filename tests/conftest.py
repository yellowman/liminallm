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
os.environ.setdefault("USE_MEMORY_STORE", "true")
os.environ.setdefault("ALLOW_REDIS_FALLBACK_DEV", "true")
os.environ.setdefault("JWT_SECRET", "test-secret-key-for-testing-only-do-not-use-in-production")
# Use Redis in tests via SyncRedisCache to avoid async event loop issues
# Falls back to in-memory if Redis is not available (via ALLOW_REDIS_FALLBACK_DEV)
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/1")

import pytest  # noqa: E402
from fastapi.dependencies import utils as fastapi_dep_utils  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from liminallm.service.runtime import reset_runtime_for_tests  # noqa: E402


# Avoid import-time failures for routes that rely on python-multipart in constrained test environments.
fastapi_dep_utils.ensure_multipart_is_installed = lambda: None


@pytest.fixture(autouse=True)
def reset_runtime_state():
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
