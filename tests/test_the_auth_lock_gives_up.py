"""Waiting for one user's auth state is bounded, and failing is failing closed.

`hold_user_auth_state` serializes password proof, session publication,
credential rotation, revocation and role changes for a single user across
replicas. It retried `pg_try_advisory_lock` forever.

Postgres drops an advisory lock when the holding session dies, so a crashed
replica frees itself. A *live* wedged holder does not, and every later auth
operation for that user queued behind it with no deadline and nothing to
show for the wait - a login, a refresh, a password change, all stalled on
one request that was never coming back.

The repair has two halves and the second is the one that matters. Bounding
the wait is easy; what the deadline must not do is fall through and run the
operation unlocked. Continuing without the lock is the exact outcome the
lock exists to prevent - a login that proved the old password publishing a
session after a reset had revoked everything and rotated the credential - so
the timeout raises. A refused request is recoverable; that is not.
"""

from __future__ import annotations

import asyncio
import time
import uuid

import pytest

from liminallm.storage.errors import AuthStateLockTimeout

PASSWORD = "TestPassword123!"
NEW_PASSWORD = "Another-1234!"


async def _held_by_someone_else(store, user_id):
    """Start a holder of `user_id`'s auth state and wait until it has it."""
    started, release = asyncio.Event(), asyncio.Event()

    async def hold():
        async with store.hold_user_auth_state(user_id):
            started.set()
            await release.wait()

    holder = asyncio.create_task(hold())
    await asyncio.wait_for(started.wait(), timeout=10)
    return holder, release


async def _attempt(store, user_id, *, deadline, body=None):
    """Try to acquire, and never hang doing it.

    The outer `wait_for` is the point. Without it, an implementation that
    retries for ever fails this file by hanging the lane until the job's own
    timeout kills it - which reads as infrastructure trouble rather than as
    the defect it is. Measured: removing the deadline from the store turned
    this file from a failure into a 120-second hang.
    """

    async def once():
        async with store.hold_user_auth_state(user_id, timeout=deadline):
            if body is not None:
                body()
            return "acquired"

    return await asyncio.wait_for(once(), timeout=deadline + 10)



@pytest.fixture
def runtime(client):
    from liminallm.service.runtime import get_runtime

    return get_runtime()


@pytest.fixture
def user(client, runtime):
    email = f"lock_{uuid.uuid4().hex[:8]}@example.com"
    resp = client.post(
        "/v1/auth/signup", json={"email": email, "password": PASSWORD}
    )
    assert resp.status_code == 201, resp.text
    data = resp.json()["data"]
    return {
        "id": data["user_id"],
        "email": email,
        "headers": {"Authorization": f"Bearer {data['access_token']}"},
    }


class TestTheWaitIsBounded:
    @pytest.mark.asyncio
    async def test_a_held_lock_is_given_up_on(self, runtime, user):
        """A second holder for the same user does not wait for ever."""
        store = runtime.store
        holder, release = await _held_by_someone_else(store, user["id"])
        began = time.monotonic()
        try:
            with pytest.raises(AuthStateLockTimeout):
                await _attempt(store, user["id"], deadline=0.4)
            waited = time.monotonic() - began
        finally:
            release.set()
            await holder

        assert 0.3 <= waited < 5, (
            f"gave up after {waited:.2f}s, which is not the deadline asked for"
        )

    @pytest.mark.asyncio
    async def test_giving_up_does_not_run_the_body(self, runtime, user):
        """The half that matters. A deadline that fell through to the body
        would satisfy "bounded" while removing the serialization entirely."""
        store = runtime.store
        ran = []
        holder, release = await _held_by_someone_else(store, user["id"])
        try:
            with pytest.raises(AuthStateLockTimeout):
                await _attempt(
                    store, user["id"], deadline=0.3, body=lambda: ran.append("body")
                )
        finally:
            release.set()
            await holder

        assert ran == [], (
            "the operation ran without the lock it asked for, which is the "
            "race the lock exists to prevent"
        )

    @pytest.mark.asyncio
    async def test_the_lock_is_usable_again_afterwards(self, runtime, user):
        """A timeout must not leave the waiter holding a pool connection or
        a half-taken lock behind it."""
        store = runtime.store
        holder, release = await _held_by_someone_else(store, user["id"])
        with pytest.raises(AuthStateLockTimeout):
            await _attempt(store, user["id"], deadline=0.3)
        release.set()
        await holder

        assert await _attempt(store, user["id"], deadline=10) == "acquired"


class TestAResetThatCouldNotGetTheLock:
    @pytest.mark.asyncio
    async def test_the_same_token_still_works_after_the_holder_releases(
        self, runtime, user, monkeypatch
    ):
        """Observation chooses the user's lock; only consumption authorizes.

        SPEC §12.1 permits observing a one-time token and says explicitly that
        observation grants nothing. If lock acquisition times out before the
        consuming read, the token must therefore still be live. This is the
        regression Bugbot found in the first bounded-lock patch: it consumed
        the token first, returned a transient failure, and made the retry
        impossible.
        """
        account = runtime.store.get_user(user["id"])
        assert account is not None
        token = await runtime.auth.initiate_password_reset(account)
        assert token

        store = runtime.store
        monkeypatch.setattr(store, "_AUTH_STATE_LOCK_TIMEOUT_SECONDS", 0.25)
        holder, release = await _held_by_someone_else(store, user["id"])
        try:
            with pytest.raises(AuthStateLockTimeout):
                await asyncio.wait_for(
                    runtime.auth.complete_password_reset_with_revocation(
                        token, NEW_PASSWORD
                    ),
                    timeout=3,
                )
        finally:
            release.set()
            await holder

        completed = await asyncio.wait_for(
            runtime.auth.complete_password_reset_with_revocation(
                token, NEW_PASSWORD
            ),
            timeout=5,
        )
        assert completed == (True, True), (
            "the timeout spent the one-time token even though no reset action "
            f"was authorized: {completed}"
        )
        assert runtime.auth.verify_password(user["id"], NEW_PASSWORD)
        assert not runtime.auth.verify_password(user["id"], PASSWORD)


class TestTheUncontendedPathIsUnchanged:
    """Controls. A change that always timed out, or always refused, would
    satisfy the class above and break every authentication in the product."""

    @pytest.mark.asyncio
    async def test_an_uncontended_lock_is_taken_immediately(self, runtime, user):
        store = runtime.store
        began = time.monotonic()
        async with store.hold_user_auth_state(user["id"]):
            pass
        assert time.monotonic() - began < 2

    @pytest.mark.asyncio
    async def test_two_different_users_do_not_wait_on_each_other(
        self, client, runtime, user
    ):
        """The lock is per user. Serializing across users would turn every
        login into a queue behind every other login."""
        other = client.post(
            "/v1/auth/signup",
            json={
                "email": f"lock_{uuid.uuid4().hex[:8]}@example.com",
                "password": PASSWORD,
            },
        )
        assert other.status_code == 201, other.text
        other_id = other.json()["data"]["user_id"]
        store = runtime.store

        holder, release = await _held_by_someone_else(store, user["id"])
        try:
            assert await _attempt(store, other_id, deadline=5) == "acquired"
        finally:
            release.set()
            await holder

    @pytest.mark.asyncio
    async def test_same_user_nesting_is_still_reentrant(self, runtime, user):
        """The deadline must not turn the documented reentrant case into a
        self-deadlock that now also times out."""
        store = runtime.store
        async with store.hold_user_auth_state(user["id"]):
            async with store.hold_user_auth_state(user["id"], timeout=1):
                pass


class TestARefusalReachesTheCaller:
    def test_the_timeout_is_served_as_a_conflict(
        self, client, runtime, user, monkeypatch
    ):
        """Handled centrally, with the code/status pair SPEC §13.0 defines.

        The request conflicts with another authentication-state operation.
        It is not a made-up 503/server_error pairing, and the handler never
        falls through to execute the protected mutation unlocked."""
        import contextlib

        @contextlib.asynccontextmanager
        async def always_times_out(*_args, **_kwargs):
            raise AuthStateLockTimeout("held by another operation")
            yield  # pragma: no cover - unreachable, keeps this a generator

        monkeypatch.setattr(
            runtime.store, "hold_user_auth_state", always_times_out
        )

        resp = client.post(
            "/v1/auth/password/change",
            headers=user["headers"],
            json={"current_password": PASSWORD, "new_password": NEW_PASSWORD},
        )

        assert resp.status_code == 409, resp.text
        body = resp.json()
        assert (body.get("error") or {}).get("code") == "conflict", body

    def test_the_password_was_not_changed_by_a_refused_request(
        self, client, runtime, user, monkeypatch
    ):
        """Failing closed means the credential is untouched, not that the
        response merely said so."""
        import contextlib

        @contextlib.asynccontextmanager
        async def always_times_out(*_args, **_kwargs):
            raise AuthStateLockTimeout("held by another operation")
            yield  # pragma: no cover

        monkeypatch.setattr(
            runtime.store, "hold_user_auth_state", always_times_out
        )
        client.post(
            "/v1/auth/password/change",
            headers=user["headers"],
            json={"current_password": PASSWORD, "new_password": NEW_PASSWORD},
        )

        monkeypatch.undo()
        still_works = client.post(
            "/v1/auth/login", json={"email": user["email"], "password": PASSWORD}
        )
        assert still_works.status_code == 200, (
            "the refused request changed the password anyway"
        )


class TestAPoolThatCannotGiveAConnection:
    """The deadline covers waiting for a pool slot, and must end the same way.

    Every holder keeps one Postgres connection for the whole protected
    operation and the pool holds ten, so contention on this lock is also
    contention on the pool - the two arrive together, not separately. An
    attempt can therefore spend its entire remaining budget waiting for a
    slot and never reach `pg_try_advisory_lock` at all.

    That path used to end differently from every other way of running out of
    time. `pool.connection(...)` was entered outside the `try` that converts
    failures, and the retry loop only handles cancellation, so
    `psycopg_pool.PoolTimeout` travelled out of the store untouched and the
    API's catch-all served 500/server_error - where the contention this
    feature exists to bound is specified as 409/conflict.

    The distinction matters to a caller: a conflict says try again, a server
    error says something is broken. Nothing in the existing file could see
    it, because every witness either contends on the advisory lock, which
    needs a connection to do at all, or injects `AuthStateLockTimeout`
    directly - which is the outcome under test, not the cause.
    """

    @staticmethod
    def _pool_is_exhausted(monkeypatch, store):
        """Make only the lock path's acquisition time out.

        `timeout=` is passed by nothing else in the store, so this leaves
        ordinary queries working - the request under test still has to reach
        the endpoint and authenticate.
        """
        from psycopg_pool import PoolTimeout

        calls = {"n": 0}
        original = store.pool.connection

        class _TimesOutOnEnter:
            """Where psycopg actually raises.

            `pool.connection(...)` returns its context manager at once and
            the wait happens inside `__enter__`, so an injection that raised
            from the call would exercise a line that cannot fail in
            production - and a fix written against it would guard nothing.
            """

            def __enter__(self):
                calls["n"] += 1
                raise PoolTimeout("pool exhausted")

            def __exit__(self, *_exc):
                return False

        def connection(*args, **kwargs):
            if "timeout" in kwargs:
                return _TimesOutOnEnter()
            return original(*args, **kwargs)

        monkeypatch.setattr(store.pool, "connection", connection)
        return calls

    @pytest.mark.asyncio
    async def test_it_is_a_lock_timeout_and_not_a_raw_pool_error(
        self, runtime, user, monkeypatch
    ):
        store = runtime.store
        calls = self._pool_is_exhausted(monkeypatch, store)
        ran = []

        with pytest.raises(AuthStateLockTimeout):
            await _attempt(
                store, user["id"], deadline=0.3, body=lambda: ran.append(1)
            )

        # The control. If the injection never fired, the assertion above
        # would be satisfied by ordinary lock contention and this test would
        # prove nothing about the pool at all.
        assert calls["n"] > 0, "the pool injection never ran"
        assert ran == [], "the protected body ran without the lock"

    def test_the_wire_says_conflict_and_not_server_error(
        self, client, runtime, user, monkeypatch
    ):
        store = runtime.store
        calls = self._pool_is_exhausted(monkeypatch, store)
        # The default deadline is fifteen seconds and this path now spends
        # all of it retrying. The bound under test is the outcome, not its
        # length.
        monkeypatch.setattr(
            store, "_AUTH_STATE_LOCK_TIMEOUT_SECONDS", 0.3, raising=False
        )

        resp = client.post(
            "/v1/auth/password/change",
            headers=user["headers"],
            json={"current_password": PASSWORD, "new_password": NEW_PASSWORD},
        )

        assert calls["n"] > 0, "the pool injection never ran"
        assert resp.status_code == 409, resp.text
        body = resp.json()
        assert (body.get("error") or {}).get("code") == "conflict", body

        # Failing closed: the credential the refused request would have
        # changed is untouched. The injection comes off first - it is not
        # specific to the lock path's caller, so leaving it on would refuse
        # the login too and the check would pass without meaning anything.
        monkeypatch.undo()
        again = client.post(
            "/v1/auth/login", json={"email": user["email"], "password": PASSWORD}
        )
        assert again.status_code == 200, again.text
