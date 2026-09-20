"""Telling somebody their other sessions are gone requires them to be gone.

`revoke_all_user_sessions` wraps the store delete in a try/except, logs a
warning, and returned 0 on failure and -1 on success - a distinction every
one of its five callers discarded. So `POST /auth/password/change` answered
`{"status": "changed"}` and `POST /auth/mfa/disable` answered
`{"status": "disabled"}` whether or not any session had been revoked.

The row is the mechanism, not a cache of one. `_authenticate_access_token`
reads `auth_session` and refuses the token when the row is missing, and no
token version or epoch stands behind it - `save_password` writes
`last_updated_at` and nothing reads that column. So a failed delete leaves
a stolen session working for its full lifetime, which defaults to seven
days, while the person who changed their password because it was stolen has
been told it is gone.

Raising is the wrong repair and was measured to be: the password is already
committed by the time revocation runs, so a failure here would report that
the change did not happen when it did. This asserts both halves - the
password really did change, and the caller is told the revocation did not.
"""

from __future__ import annotations

import asyncio
import uuid

import psycopg
import pytest

PASSWORD = "TestPassword123!"
NEW_PASSWORD = "TestPassword456!"


@pytest.fixture
def runtime(client):
    from liminallm.service.runtime import get_runtime

    return get_runtime()


@pytest.fixture
def account(client):
    """A signed-up user with a second, separate session to steal."""
    email = f"rev_{uuid.uuid4().hex[:8]}@example.com"
    signup = client.post(
        "/v1/auth/signup", json={"email": email, "password": PASSWORD}
    )
    assert signup.status_code == 201, signup.text
    mine = signup.json()["data"]

    other = client.post(
        "/v1/auth/login", json={"email": email, "password": PASSWORD}
    )
    assert other.status_code == 200, other.text
    stolen = other.json()["data"]["access_token"]
    return {
        "user_id": mine["user_id"],
        "email": email,
        "headers": {"Authorization": f"Bearer {mine['access_token']}"},
        "stolen": {"Authorization": f"Bearer {stolen}"},
    }


def _alive(client, headers) -> bool:
    return client.get("/v1/me", headers=headers).status_code == 200


class TestWhenRevocationWorks:
    """The control. A change that always reported failure would satisfy the
    class below while telling everybody their sessions survived."""

    def test_the_other_session_dies_and_the_answer_says_so(
        self, client, account
    ):
        assert _alive(client, account["stolen"]), "the fixture had no session"

        resp = client.post(
            "/v1/auth/password/change",
            headers=account["headers"],
            json={"current_password": PASSWORD, "new_password": NEW_PASSWORD},
        )

        assert resp.status_code == 200, resp.text
        assert resp.json()["data"]["other_sessions_revoked"] is True
        assert not _alive(client, account["stolen"]), (
            "the control could not observe revocation, so the test below "
            "proves nothing"
        )


class TestWhenRevocationFails:
    def test_the_password_change_says_the_sessions_survived(
        self, client, runtime, account, monkeypatch
    ):
        called: list = []

        def boom(*args, **kwargs):
            called.append(args)
            raise RuntimeError("the delete could not reach the database")

        monkeypatch.setattr(runtime.store, "revoke_user_sessions", boom)

        resp = client.post(
            "/v1/auth/password/change",
            headers=account["headers"],
            json={"current_password": PASSWORD, "new_password": NEW_PASSWORD},
        )

        assert called, "the mutation never applied, so this measures nothing"
        assert resp.status_code == 200, resp.text
        assert resp.json()["data"]["other_sessions_revoked"] is False, (
            "the response claimed the other sessions were revoked while the "
            "delete had failed"
        )

    def test_the_stolen_session_really_does_survive(
        self, client, runtime, account, monkeypatch
    ):
        """Why the report matters rather than being bookkeeping."""
        monkeypatch.setattr(
            runtime.store,
            "revoke_user_sessions",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")),
        )

        client.post(
            "/v1/auth/password/change",
            headers=account["headers"],
            json={"current_password": PASSWORD, "new_password": NEW_PASSWORD},
        )

        assert _alive(client, account["stolen"]), (
            "the fixture expected the failure to leave the session alive"
        )

    def test_password_reset_does_not_complete_if_sessions_survive(
        self, client, runtime, account, monkeypatch
    ):
        """SPEC §§12.1/13.2 make credential rotation and revocation one reset.

        If the canonical session delete fails, consume the one-time token but
        leave the credential unchanged and report that the reset did not
        complete.
        """
        user = runtime.store.get_user(account["user_id"])
        assert user is not None
        token = asyncio.run(runtime.auth.initiate_password_reset(user))
        assert token

        monkeypatch.setattr(
            runtime.store,
            "revoke_user_sessions",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")),
        )

        resp = client.post(
            "/v1/auth/reset/confirm",
            json={"token": token, "new_password": NEW_PASSWORD},
        )

        assert resp.status_code == 503, resp.text
        assert resp.json()["error"]["code"] == "reset_incomplete"
        assert runtime.auth.verify_password(account["user_id"], PASSWORD)
        assert not runtime.auth.verify_password(account["user_id"], NEW_PASSWORD)
        assert _alive(client, account["stolen"]), (
            "the fixture expected the failed revoke to leave the old session alive"
        )

    @pytest.mark.asyncio
    async def test_role_change_is_not_committed_if_sessions_survive(
        self, runtime, account, monkeypatch
    ):
        """A stale refresh token must not inherit a role upgrade."""
        before = runtime.store.get_user(account["user_id"])
        assert before is not None and before.role != "admin"

        monkeypatch.setattr(
            runtime.store,
            "revoke_user_sessions",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")),
        )

        with pytest.raises(RuntimeError, match="sessions could not be revoked"):
            await runtime.auth.set_user_role(account["user_id"], "admin")

        after = runtime.store.get_user(account["user_id"])
        assert after is not None
        assert after.role == before.role, (
            "the role upgrade committed even though the bearer sessions that "
            "would inherit it could not be revoked"
        )

    def test_single_session_login_fails_closed_if_prior_sessions_survive(
        self, client, runtime, account, monkeypatch
    ):
        """A single-session login must not create session N+1 after failing
        to revoke sessions 1..N."""
        with psycopg.connect(runtime.store.dsn, autocommit=True) as conn:
            conn.execute(
                "UPDATE app_user SET meta = jsonb_set("
                "COALESCE(meta, '{}'::jsonb), '{single_session}', 'true') "
                "WHERE id = %s",
                (account["user_id"],),
            )

        monkeypatch.setattr(
            runtime.store,
            "revoke_user_sessions",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")),
        )

        resp = client.post(
            "/v1/auth/login",
            json={"email": account["email"], "password": PASSWORD},
        )

        assert resp.status_code == 401, (
            "single-session mode created another session even though the "
            f"prior sessions could not be revoked: {resp.text}"
        )
        assert _alive(client, account["stolen"]), (
            "the fixture expected the failed revoke to leave the old session "
            "alive, so the fail-closed assertion proved nothing"
        )

    def test_the_password_still_changed(
        self, client, runtime, account, monkeypatch
    ):
        """Which is why the route must not raise. The password is committed
        before revocation runs, so failing the request would report that the
        change did not happen when it did."""
        monkeypatch.setattr(
            runtime.store,
            "revoke_user_sessions",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")),
        )

        client.post(
            "/v1/auth/password/change",
            headers=account["headers"],
            json={"current_password": PASSWORD, "new_password": NEW_PASSWORD},
        )

        fresh = client.post(
            "/v1/auth/login",
            json={"email": account["email"], "password": NEW_PASSWORD},
        )
        assert fresh.status_code == 200, (
            "the new password was refused, so the route had reported a change "
            "it did not make"
        )
        stale = client.post(
            "/v1/auth/login",
            json={"email": account["email"], "password": PASSWORD},
        )
        assert stale.status_code == 401, "the old password still works"
