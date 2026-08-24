"""A one-time token is consumed, not observed.

SPEC §12.1 calls the password reset token single-use, and the code enforced
that by deleting the token *after* changing the password:

    GET reset:T
    ...
    save_password
    ...
    DELETE reset:T

Between the read and the delete the token is still there, so two requests
holding the same token both read a subject and both proceed. The second one
wins, and the password ends up being whichever of the two arrived last —
which for a token delivered by email is a window an attacker who has read the
message can use deliberately, and a window an ordinary double-click can hit by
accident.

`pop_oauth_state` already solved this for OAuth state: GETDEL, with a Lua
fallback for a Redis older than 6.2. This is the same primitive applied to the
two identity tokens that were still reading first.

The in-process fallback was already correct — its `pop()` under the state lock
*is* the atomic consume — so the work there is to leave it alone, and to have
a test that says so.
"""

from __future__ import annotations

import asyncio
import threading
import uuid

import pytest

from liminallm.service.runtime import get_runtime


def _unique(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _account(client, *, admin=False):
    email = f"{_unique('tok')}@example.com"
    password = "TestPassword123!"
    resp = client.post("/v1/auth/signup", json={"email": email, "password": password})
    assert resp.status_code == 201, resp.text
    data = resp.json()["data"]
    user_id = data["user_id"]
    if admin:
        get_runtime().store.update_user_role(user_id, role="admin")
        resp = client.post(
            "/v1/auth/login", json={"email": email, "password": password}
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()["data"]
    return user_id, email, {"Authorization": f"Bearer {data['access_token']}"}


class TestTheConsumePrimitiveIsAtomic:
    @pytest.mark.asyncio
    async def test_exactly_one_caller_gets_the_value(self, client):
        """The primitive itself, not the flow that uses it.

        Every other red here pauses a caller after its consume returned, so it
        tests the order the service does things in. This one asks whether the
        read and the removal are one step: eight callers, one key. Measured, a
        `GET` followed by a `DELETE` hands the subject to all eight, because
        every await is a point where the others run.
        """
        runtime = get_runtime()
        if runtime.cache is None:
            pytest.skip("no Redis in this environment")
        token = _unique("t")
        await runtime.cache.client.set(f"probe:{token}", "the-subject", ex=60)

        got = await asyncio.gather(
            *(runtime.cache.consume_identity_token("probe", token) for _ in range(8))
        )
        assert sum(1 for g in got if g) == 1, (
            f"one token was handed to {sum(1 for g in got if g)} callers"
        )
        assert await runtime.cache.client.get(f"probe:{token}") is None

    @pytest.mark.asyncio
    async def test_oauth_state_uses_the_same_primitive(self, client):
        """It is where this guarantee was already correct.

        Consolidating the two implementations is only an improvement if the
        one that worked keeps working, so this is the regression that says the
        OAuth path still consumes rather than reads.
        """
        from datetime import datetime, timedelta, timezone

        runtime = get_runtime()
        if runtime.cache is None:
            pytest.skip("no Redis in this environment")
        state = _unique("s")
        await runtime.cache.set_oauth_state(
            state, "google", datetime.now(timezone.utc) + timedelta(minutes=5), None
        )
        assert await runtime.cache.pop_oauth_state(state) is not None
        assert await runtime.cache.pop_oauth_state(state) is None


class TestAResetTokenIsConsumedOnce:
    @pytest.mark.asyncio
    async def test_a_second_holder_of_the_token_gets_nothing(self, client):
        """Forced, not two calls that happen to serialize.

        The second attempt is made while the first is between consuming the
        token and writing the password — the whole window the old order
        created. Two ordinary concurrent calls would usually miss it.
        """
        runtime = get_runtime()
        if runtime.cache is None:
            pytest.skip("no Redis in this environment")
        user_id, _, _ = _account(client)
        token = await runtime.auth.initiate_password_reset(
            runtime.store.get_user(user_id)
        )

        consumed = threading.Event()
        release = threading.Event()
        real_save = runtime.auth.store.save_password
        second: dict = {}

        def pause_before_the_password_write(*args, **kwargs):
            # Once. On a build where the token is still readable the second
            # attempt reaches here too, and pausing it as well would deadlock
            # the test rather than fail it.
            if not consumed.is_set():
                consumed.set()
                assert release.wait(timeout=30), "the first reset was not released"
            return real_save(*args, **kwargs)

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(
            runtime.auth.store, "save_password", pause_before_the_password_write
        )

        first: dict = {}

        def run_first():
            first["ok"] = asyncio.run(
                runtime.auth.complete_password_reset(token, "FirstPassword123!")
            )

        winner = threading.Thread(target=run_first, daemon=True)
        try:
            winner.start()
            assert consumed.wait(timeout=30), "the first reset never consumed"
            # The token has been consumed and the password not yet written.
            # This is the entire window the old implementation left open.
            second["ok"] = await runtime.auth.complete_password_reset(
                token, "SecondPassword123!"
            )
            release.set()
            winner.join(timeout=30)
        finally:
            release.set()
            monkeypatch.undo()

        assert not winner.is_alive()
        assert second["ok"] is False, (
            "a reset token was still readable after it had been handed to "
            "another request, so both could change the password"
        )
        assert first["ok"] is True
        assert runtime.auth.verify_password(user_id, "FirstPassword123!")
        assert not runtime.auth.verify_password(user_id, "SecondPassword123!")
        assert await runtime.cache.client.get(f"reset:{token}") is None

    @pytest.mark.asyncio
    async def test_a_consumed_token_stays_consumed_when_the_reset_fails(
        self, client
    ):
        """One-time means one attempt, not one success.

        The tempting repair for a failed reset is to put the token back so the
        user can retry. That is replayability with a friendlier name: the
        token is out, and the next holder of it must get nothing whatever
        happened to the first.
        """
        runtime = get_runtime()
        if runtime.cache is None:
            pytest.skip("no Redis in this environment")
        user_id, _, _ = _account(client)
        token = await runtime.auth.initiate_password_reset(
            runtime.store.get_user(user_id)
        )

        def refuse(*args, **kwargs):
            raise RuntimeError("the password write failed")

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(runtime.auth.store, "save_password", refuse)
        try:
            with pytest.raises(RuntimeError):
                await runtime.auth.complete_password_reset(token, "FirstPassword123!")
        finally:
            monkeypatch.undo()

        assert await runtime.cache.client.get(f"reset:{token}") is None, (
            "a failed reset put its token back, so the token is replayable"
        )
        assert await runtime.auth.complete_password_reset(
            token, "SecondPassword123!"
        ) is False
        assert not runtime.auth.verify_password(user_id, "SecondPassword123!")

    @pytest.mark.asyncio
    async def test_the_in_process_fallback_consumes_atomically_too(self, client):
        """Its `pop()` under the state lock already is the atomic consume.

        The work here is to leave that alone rather than 'normalize' it into a
        read-then-pop that matches the Redis code's old shape. So the property
        is asserted directly: many concurrent completions, exactly one True.
        """
        from liminallm.service.auth import AuthService

        runtime = get_runtime()
        auth = AuthService(runtime.store, None, runtime.settings)
        user_id, _, _ = _account(client)
        token = await auth.initiate_password_reset(runtime.store.get_user(user_id))

        results = await asyncio.gather(
            *(auth.complete_password_reset(token, f"Password{i}23!") for i in range(8))
        )
        assert sum(1 for r in results if r) == 1, (
            f"the in-process fallback handed one token to {sum(results)} resets"
        )


class TestAVerificationTokenIsConsumedOnce:
    @pytest.mark.asyncio
    async def test_a_second_holder_of_the_token_gets_nothing(self, client):
        """Marking a mailbox verified twice is harmless. Drift is not.

        This is the same one-time primitive, and leaving one of its two users
        reading first is how the next reader concludes that reading first is
        the house pattern.
        """
        runtime = get_runtime()
        if runtime.cache is None:
            pytest.skip("no Redis in this environment")
        user_id, _, _ = _account(client)
        token = await runtime.auth.request_email_verification(
            runtime.store.get_user(user_id)
        )

        consumed = threading.Event()
        release = threading.Event()
        real_mark = runtime.auth.store.mark_email_verified

        def pause_before_marking(*args, **kwargs):
            if not consumed.is_set():
                consumed.set()
                assert release.wait(timeout=30), "the first was not released"
            return real_mark(*args, **kwargs)

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(
            runtime.auth.store, "mark_email_verified", pause_before_marking
        )
        first: dict = {}

        def run_first():
            first["ok"] = asyncio.run(
                runtime.auth.complete_email_verification(token)
            )

        winner = threading.Thread(target=run_first, daemon=True)
        try:
            winner.start()
            assert consumed.wait(timeout=30), "the first verification never consumed"
            second = await runtime.auth.complete_email_verification(token)
            release.set()
            winner.join(timeout=30)
        finally:
            release.set()
            monkeypatch.undo()

        assert not winner.is_alive()
        assert second is False, (
            "a verification token was still readable after it had been handed "
            "to another request"
        )
        assert first["ok"] is True
        assert await runtime.cache.client.get(f"verify:{token}") is None


class TestIssuanceBelongsToTheAccountLifetime:
    @pytest.mark.asyncio
    async def test_a_token_issued_under_an_erasure_does_not_outlive_it(
        self, client
    ):
        """`/auth/reset/request` resolves the account, then writes the token.

        The erasure can commit and purge in between, leaving a fresh identity
        token naming an account that no longer exists. Inert — completion
        re-resolves the immutable id and finds nothing — but the erasure's own
        contract is that its identifiable state is gone when it returns, and
        this is the last write that could put some back.
        """
        runtime = get_runtime()
        if runtime.cache is None:
            pytest.skip("no Redis in this environment")
        user_id, _, _ = _account(client)
        _, _, admin_headers = _account(client, admin=True)
        user = runtime.store.get_user(user_id)

        reached = threading.Event()
        release = threading.Event()
        real = runtime.store.hold_live_user

        import contextlib as contextlib_

        @contextlib_.contextmanager
        def pause_inside(target_user_id):
            with real(target_user_id) as live:
                if target_user_id == user_id:
                    reached.set()
                    assert release.wait(timeout=30), "issuance was never released"
                yield live

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(runtime.store, "hold_live_user", pause_inside)
        issued: dict = {}
        deletion: dict = {}

        def issue():
            issued["token"] = asyncio.run(
                runtime.auth.initiate_password_reset(user)
            )

        def delete_the_account():
            deletion["status"] = client.delete(
                f"/v1/admin/users/{user_id}", headers=admin_headers
            ).status_code

        issuer = threading.Thread(target=issue, daemon=True)
        deleter = threading.Thread(target=delete_the_account, daemon=True)
        try:
            issuer.start()
            assert reached.wait(timeout=30), "issuance never reached the guard"
            deleter.start()
            deleter.join(timeout=1)
            assert deleter.is_alive() and "status" not in deletion, (
                "the account was erased and purged while a token naming it "
                "was already being issued"
            )
            release.set()
            issuer.join(timeout=30)
            deleter.join(timeout=30)
        finally:
            release.set()
            monkeypatch.undo()

        assert not issuer.is_alive() and not deleter.is_alive()
        assert deletion.get("status") in (200, 204), deletion
        assert issued.get("token")
        assert await runtime.cache.client.get(f"reset:{issued['token']}") is None, (
            "an identity token for an erased account survived the purge"
        )

    @pytest.mark.asyncio
    async def test_issuance_after_the_erasure_writes_nothing(self, client):
        """The other history: the deletion took the lock first.

        The caller still holds the `User` it resolved a moment ago, which is
        exactly what `/auth/reset/request` does.
        """
        runtime = get_runtime()
        if runtime.cache is None:
            pytest.skip("no Redis in this environment")
        user_id, _, _ = _account(client)
        _, _, admin_headers = _account(client, admin=True)
        stale = runtime.store.get_user(user_id)

        assert client.delete(
            f"/v1/admin/users/{user_id}", headers=admin_headers
        ).status_code in (200, 204)

        assert await runtime.auth.initiate_password_reset(stale) is None, (
            "a reset token was issued for an account that no longer exists"
        )
        assert await runtime.auth.request_email_verification(stale) is None
        assert [
            k
            async for k in runtime.cache.client.scan_iter(match="reset:*", count=500)
        ] == []

    def test_no_mail_is_sent_when_issuance_declined(self, client):
        """The route's half of the erased-mid-request case.

        Deleting before the request does not reach this: the route's own
        lookup fails first and the branch is skipped. What has to be covered
        is the account that was live at the lookup and gone at the write, and
        the observable part of that is a `None` token — so this drives the
        route by the contract rather than by re-staging the race, which
        `test_a_token_issued_under_an_erasure_does_not_outlive_it` already
        does for the service.
        """
        runtime = get_runtime()
        _, email, _ = _account(client)

        sent: list = []
        monkeypatch = pytest.MonkeyPatch()

        async def declined(user):
            return None

        monkeypatch.setattr(runtime.auth, "initiate_password_reset", declined)
        monkeypatch.setattr(
            runtime.email,
            "send_password_reset",
            lambda address, token: sent.append((address, token)),
        )
        try:
            resp = client.post("/v1/auth/reset/request", json={"email": email})
        finally:
            monkeypatch.undo()

        assert resp.status_code == 200, resp.text
        assert resp.json()["data"] == {"status": "sent"}
        assert sent == [], (
            f"the route mailed a token it was never given: {sent}"
        )

    @pytest.mark.asyncio
    async def test_the_request_route_still_hides_whether_the_account_exists(
        self, client
    ):
        """Anti-enumeration is the reason this route answers the way it does.

        An account erased mid-request must produce no token and no email, and
        the same 200 either way.
        """
        runtime = get_runtime()
        user_id, email, _ = _account(client)
        _, live_email, _ = _account(client)
        _, _, admin_headers = _account(client, admin=True)
        assert client.delete(
            f"/v1/admin/users/{user_id}", headers=admin_headers
        ).status_code in (200, 204)

        sent: list = []
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(
            runtime.email,
            "send_password_reset",
            lambda address, token: sent.append((address, token)),
        )
        try:
            gone = client.post("/v1/auth/reset/request", json={"email": email})
            present = client.post(
                "/v1/auth/reset/request", json={"email": live_email}
            )
        finally:
            monkeypatch.undo()

        # Indistinguishable from outside, which is what the route is for.
        assert gone.status_code == present.status_code == 200
        assert gone.json()["data"] == present.json()["data"]
        # And nothing was actually sent for the account that is gone.
        assert [address for address, _ in sent] == [live_email], (
            f"a reset mail was sent for an erased account: {sent}"
        )
