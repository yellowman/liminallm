import threading
import uuid
from pathlib import Path

from liminallm.storage.models import Session
from liminallm.storage.postgres import PostgresStore


class DummyPool:
    def connection(self):
        raise AssertionError("database access should be stubbed in unit tests")


def test_postgres_store_cache_helpers(tmp_path: Path):
    store: PostgresStore = PostgresStore.__new__(PostgresStore)
    store.pool = DummyPool()
    store.fs_root = tmp_path
    store.sessions = {}
    store._session_lock = threading.Lock()  # Required for thread-safe cache ops

    session = Session.new(str(uuid.uuid4()))
    cached = store._cache_session(session)
    assert cached is session
    assert store.sessions[session.id] is session

    store._evict_session(session.id)
    assert session.id not in store.sessions

    store._cache_session(session)
    store._update_cached_session(session.id, user_agent="agent")
    assert store.sessions[session.id].user_agent == "agent"
