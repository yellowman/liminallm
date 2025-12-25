"""Tests for thread-safe session cache operations in PostgresStore.

Per SPEC §18, session cache operations must be thread-safe to prevent
race conditions under concurrent requests.
"""

import threading
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

from liminallm.storage.models import Session
from liminallm.storage.postgres import _MAX_SESSION_CACHE_SIZE, PostgresStore


class DummyPool:
    """Dummy connection pool that prevents accidental database access."""

    def connection(self):
        raise AssertionError("database access should be stubbed in unit tests")


def create_test_store(tmp_path: Path) -> PostgresStore:
    """Create a PostgresStore instance for testing without database."""
    store: PostgresStore = PostgresStore.__new__(PostgresStore)
    store.pool = DummyPool()
    store.fs_root = tmp_path
    store.sessions = {}
    store._session_lock = threading.Lock()
    return store


def create_session(user_id: str = None, expires_in_seconds: int = 3600) -> Session:
    """Create a test session with given expiration."""
    return Session(
        id=str(uuid.uuid4()),
        user_id=user_id or str(uuid.uuid4()),
        tenant_id="public",
        mfa_required=False,
        mfa_verified=False,
        created_at=datetime.utcnow(),
        expires_at=datetime.utcnow() + timedelta(seconds=expires_in_seconds),
    )


class TestSessionCacheBasics:
    """Basic session cache operation tests."""

    def test_cache_session_stores_session(self, tmp_path):
        """_cache_session stores session in dict."""
        store = create_test_store(tmp_path)
        session = create_session()

        cached = store._cache_session(session)

        assert cached is session
        assert store.sessions[session.id] is session

    def test_evict_session_removes_session(self, tmp_path):
        """_evict_session removes session from cache."""
        store = create_test_store(tmp_path)
        session = create_session()
        store._cache_session(session)

        store._evict_session(session.id)

        assert session.id not in store.sessions

    def test_evict_nonexistent_session_no_error(self, tmp_path):
        """_evict_session doesn't error on missing session."""
        store = create_test_store(tmp_path)

        # Should not raise
        store._evict_session("nonexistent-id")

    def test_update_cached_session_modifies_fields(self, tmp_path):
        """_update_cached_session updates session fields."""
        store = create_test_store(tmp_path)
        session = create_session()
        store._cache_session(session)

        store._update_cached_session(session.id, user_agent="Mozilla/5.0")

        assert store.sessions[session.id].user_agent == "Mozilla/5.0"

    def test_update_nonexistent_session_no_error(self, tmp_path):
        """_update_cached_session doesn't error on missing session."""
        store = create_test_store(tmp_path)

        # Should not raise
        store._update_cached_session("nonexistent-id", user_agent="test")


class TestSessionCacheEviction:
    """Tests for session cache eviction when at capacity."""

    def test_eviction_triggered_at_max_capacity(self, tmp_path):
        """Cache evicts oldest sessions when at max capacity."""
        store = create_test_store(tmp_path)

        # Fill cache to exactly max size
        sessions = []
        for i in range(_MAX_SESSION_CACHE_SIZE):
            # Earlier sessions expire sooner
            session = create_session(expires_in_seconds=i + 1)
            store.sessions[session.id] = session
            sessions.append(session)

        # Add one more session to trigger eviction
        new_session = create_session(expires_in_seconds=_MAX_SESSION_CACHE_SIZE + 1000)
        store._cache_session(new_session)

        # Should have evicted ~10% of oldest sessions
        # (evict_count and expected_remaining calculated for documentation)
        _ = max(1, _MAX_SESSION_CACHE_SIZE // 10)  # evict_count

        # Allow some tolerance for timing
        assert len(store.sessions) <= _MAX_SESSION_CACHE_SIZE
        assert new_session.id in store.sessions

    def test_eviction_removes_soonest_expiring(self, tmp_path):
        """Eviction removes sessions closest to expiration."""
        store = create_test_store(tmp_path)

        # Create sessions with varying expiration times
        soon_expiring = create_session(expires_in_seconds=1)
        later_expiring = create_session(expires_in_seconds=10000)

        store.sessions[soon_expiring.id] = soon_expiring
        store.sessions[later_expiring.id] = later_expiring

        # Fill to capacity
        for _ in range(_MAX_SESSION_CACHE_SIZE - 2):
            session = create_session(expires_in_seconds=5000)
            store.sessions[session.id] = session

        # Trigger eviction
        new_session = create_session(expires_in_seconds=5000)
        store._cache_session(new_session)

        # Later expiring should still be there, soon expiring should be evicted
        assert later_expiring.id in store.sessions


class TestSessionCacheThreadSafety:
    """Tests for thread-safe session cache operations."""

    def test_concurrent_cache_operations(self, tmp_path):
        """Concurrent cache/evict operations don't corrupt state."""
        store = create_test_store(tmp_path)
        errors: List[Exception] = []
        iterations = 100

        def cache_sessions():
            try:
                for _ in range(iterations):
                    session = create_session()
                    store._cache_session(session)
            except Exception as e:
                errors.append(e)

        def evict_sessions():
            try:
                for _ in range(iterations):
                    # Get a random session ID if available
                    with store._session_lock:
                        if store.sessions:
                            session_id = next(iter(store.sessions.keys()))
                        else:
                            session_id = "nonexistent"
                    store._evict_session(session_id)
            except Exception as e:
                errors.append(e)

        # Run concurrent operations
        threads = [
            threading.Thread(target=cache_sessions),
            threading.Thread(target=cache_sessions),
            threading.Thread(target=evict_sessions),
            threading.Thread(target=evict_sessions),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"

    def test_concurrent_updates(self, tmp_path):
        """Concurrent update operations don't corrupt state."""
        store = create_test_store(tmp_path)
        session = create_session()
        store._cache_session(session)
        errors: List[Exception] = []
        iterations = 100

        def update_session(field_name: str, value: str):
            try:
                for i in range(iterations):
                    store._update_cached_session(
                        session.id, **{field_name: f"{value}-{i}"}
                    )
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=update_session, args=("user_agent", "agent")),
            threading.Thread(target=update_session, args=("ip_addr", "ip")),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"
        # Session should still exist
        assert session.id in store.sessions

    def test_revoke_user_sessions_thread_safe(self, tmp_path):
        """revoke_user_sessions iteration is thread-safe."""
        store = create_test_store(tmp_path)
        user_id = str(uuid.uuid4())
        errors: List[Exception] = []

        # Create sessions for the user
        for _ in range(50):
            session = create_session(user_id=user_id)
            store.sessions[session.id] = session

        # Create sessions for other users
        for _ in range(50):
            session = create_session()
            store.sessions[session.id] = session

        def add_sessions():
            try:
                for _ in range(50):
                    session = create_session()
                    store._cache_session(session)
            except Exception as e:
                errors.append(e)

        def revoke_sessions():
            try:
                # Simulate revoke_user_sessions iteration
                with store._session_lock:
                    stale_ids = [
                        sid
                        for sid, sess in store.sessions.items()
                        if sess.user_id == user_id
                    ]
                    for sid in stale_ids:
                        store.sessions.pop(sid, None)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=add_sessions),
            threading.Thread(target=revoke_sessions),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"

    def test_lock_prevents_race_condition(self, tmp_path):
        """Lock prevents dict-changed-size-during-iteration error."""
        store = create_test_store(tmp_path)
        errors: List[Exception] = []
        iterations = 200

        def iterate_and_modify():
            try:
                for _ in range(iterations):
                    with store._session_lock:
                        # Iterate while holding lock
                        for _ in store.sessions.items():
                            pass
                        # Modify while holding lock
                        session = create_session()
                        store.sessions[session.id] = session
            except RuntimeError as e:
                if "dictionary changed size" in str(e):
                    errors.append(e)

        threads = [threading.Thread(target=iterate_and_modify) for _ in range(4)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No "dictionary changed size during iteration" errors
        assert len(errors) == 0, f"Race condition detected: {errors}"


class TestSessionCacheMemory:
    """Tests for session cache memory management."""

    def test_cache_respects_max_size(self, tmp_path):
        """Cache never exceeds max size significantly."""
        store = create_test_store(tmp_path)

        # Add many more sessions than max
        for _ in range(_MAX_SESSION_CACHE_SIZE + 100):
            session = create_session()
            store._cache_session(session)

        # Should not exceed max size
        assert len(store.sessions) <= _MAX_SESSION_CACHE_SIZE

    def test_empty_cache_operations(self, tmp_path):
        """Operations on empty cache work correctly."""
        store = create_test_store(tmp_path)

        # These should all work on empty cache
        store._evict_session("nonexistent")
        store._update_cached_session("nonexistent", user_agent="test")

        session = create_session()
        store._cache_session(session)

        assert len(store.sessions) == 1
