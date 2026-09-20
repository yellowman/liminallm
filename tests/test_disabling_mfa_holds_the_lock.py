"""Disabling MFA revokes sessions, so it belongs in the same order as login.

`hold_user_auth_state` is what gives one user's authentication operations a
single cross-replica order: password proof, credential rotation, role
change, session publication and session revocation all take it, so none of
them can interleave with another.

Disabling MFA did not. It wrote the MFA flag and then called
`revoke_all_user_sessions` with no lock at all, while `_maybe_rotate_session`
creates a successor *under* the lock, after re-reading the predecessor to
check it still exists.

That re-read is the whole defence on the rotation side, and an unlocked
revocation walks straight past it. Rotation takes the lock, reads session S
and finds it live; the disable deletes S without waiting for anything and
reports that it revoked; rotation then publishes S', a successor of the
session that was just revoked. Somebody who disabled MFA because a session
was stolen is told the stolen session is gone, and a descendant of it is
live.

The property is structural, so it is tested structurally: the revocation
must happen while this user's auth-state lock is held. A timing test for
this would have to lose a race on purpose to fail, and one that passes
because the interleaving did not happen this run is a witness to nothing.
"""

from __future__ import annotations

import contextlib
import uuid

import pytest

PASSWORD = "TestPassword123!"
NEW_PASSWORD = "Another-1234!"


@pytest.fixture
def runtime(client):
    from liminallm.service.runtime import get_runtime

    return get_runtime()


@pytest.fixture
def user(client):
    email = f"mfa_{uuid.uuid4().hex[:8]}@example.com"
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


def _watch_the_lock(monkeypatch, runtime):
    """Record how deep the auth-state lock is when the revocation runs.

    Depth rather than a flag, because the lock is reentrant for the same
    user in one task: a caller that already holds it is still holding it.
    """
    seen = {"depth_at_revoke": None, "entered": 0}
    original_hold = runtime.store.hold_user_auth_state
    depth = {"n": 0}

    @contextlib.asynccontextmanager
    async def watched_hold(user_id, **kwargs):
        async with original_hold(user_id, **kwargs):
            depth["n"] += 1
            seen["entered"] += 1
            try:
                yield
            finally:
                depth["n"] -= 1

    original_revoke = runtime.auth.revoke_all_user_sessions

    async def watched_revoke(*args, **kwargs):
        seen["depth_at_revoke"] = depth["n"]
        return await original_revoke(*args, **kwargs)

    monkeypatch.setattr(runtime.store, "hold_user_auth_state", watched_hold)
    monkeypatch.setattr(runtime.auth, "revoke_all_user_sessions", watched_revoke)
    return seen


class TestTheRevocationRunsUnderTheLock:
    def test_the_probe_sees_a_lock_that_is_held(
        self, client, runtime, user, monkeypatch
    ):
        """The control.

        Changing a password revokes sessions under the lock and has done
        since that path was written, so it is the known-good case. If this
        reported no lock, the measurement below would be meaningless rather
        than a finding - "no lock held" and "cannot see a lock" would be the
        same reading.
        """
        seen = _watch_the_lock(monkeypatch, runtime)
        resp = client.post(
            "/v1/auth/password/change",
            headers=user["headers"],
            json={"current_password": PASSWORD, "new_password": NEW_PASSWORD},
        )
        assert resp.status_code == 200, resp.text
        assert seen["entered"] > 0, "the lock wrapper never ran"
        assert seen["depth_at_revoke"] is not None, "nothing revoked"
        assert seen["depth_at_revoke"] > 0, (
            "the control path revoked outside the lock, so this probe cannot "
            "tell held from unheld"
        )

    def test_disabling_mfa_revokes_under_the_lock(
        self, client, runtime, user, monkeypatch
    ):
        secret = "JBSWY3DPEHPK3PXP"
        runtime.store.set_user_mfa_secret(user["id"], secret, enabled=True)
        # The TOTP arithmetic is not what this is about, and a real code
        # would make the test depend on the clock.
        monkeypatch.setattr(
            runtime.auth, "_verify_totp", lambda *_args, **_kwargs: True
        )

        seen = _watch_the_lock(monkeypatch, runtime)
        resp = client.post(
            "/v1/auth/mfa/disable",
            headers=user["headers"],
            json={"code": "000000"},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["data"]["status"] == "disabled", resp.text

        assert seen["depth_at_revoke"] is not None, "nothing revoked"
        assert seen["depth_at_revoke"] > 0, (
            "MFA disable revoked this user's sessions without holding their "
            "auth-state lock, so a session rotation that already passed its "
            "existence re-read can publish a successor of a session this "
            "call reported as revoked"
        )
