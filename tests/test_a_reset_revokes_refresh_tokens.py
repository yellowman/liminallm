"""A completed password reset refuses the refresh tokens minted before it.

SPEC 12.1 and 13.2 both say the same thing about completing a reset: it
"consumes the token atomically, rotates credentials, revokes sessions and
refresh tokens". The session half already has a witness
(`test_a_reset_revokes_every_existing_session`). This is the refresh half,
which is held somewhere else entirely: a refresh token is a JWT carrying
`sid` and `jti`, revoked through a Redis denylist rather than by a row, so
deleting the session reaches it only through the lookup the refresh flow
performs on `sid`.

That indirection is the reason to pin it at the route. The promise holds
today because `refresh_tokens` refuses when `get_session` finds nothing; a
later change that served the session from cache, or stopped consulting the
row, would keep every existing test passing and quietly leave a refresh
token usable after the reset that was supposed to revoke it.

Both tokens are asserted, because a caller who has refreshed once is holding
the rotated one, not the token login handed them.

Scope is the SPEC's own words: sessions and refresh tokens. API keys are a
separate credential class - they authenticate only the agent surfaces and
skip session and MFA machinery (13.2) - and this deliberately says nothing
about them. Their survival is a consequence of that architecture, not a
promise anyone has made.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

from liminallm.service.runtime import get_runtime

PASSWORD = "OldPassword123!"
NEW_PASSWORD = "NewPassword456!"


@pytest.fixture
def account(client):
    email = f"rt_{uuid.uuid4().hex[:8]}@example.com"
    resp = client.post("/v1/auth/signup",
                       json={"email": email, "password": PASSWORD})
    assert resp.status_code == 201, resp.text
    data = resp.json()["data"]
    assert data.get("refresh_token"), (
        "signup returned no refresh token, so this test cannot reach the "
        "credential it exists to revoke"
    )
    return {"email": email, "user_id": data["user_id"],
            "refresh": data["refresh_token"]}


def _refresh(client, token):
    """The route a caller uses, not the service beneath it."""
    return client.post("/v1/auth/refresh", json={"refresh_token": token})


def _sessions(user_id):
    with get_runtime().store._connect() as conn:
        return conn.execute(
            "SELECT count(*) AS n FROM auth_session WHERE user_id = %s",
            (user_id,),
        ).fetchone()["n"]


def test_a_reset_refuses_the_refresh_tokens_minted_before_it(client, account):
    original = account["refresh"]

    # The control. Without it, a refusal after the reset would not
    # distinguish a revoked token from a refresh path that never worked.
    first = _refresh(client, original)
    assert first.status_code == 200, first.text
    rotated = first.json()["data"].get("refresh_token")
    assert rotated and rotated != original, (
        "the refresh did not rotate, so there is only one credential to test"
    )
    assert _sessions(account["user_id"]) > 0

    # Issued through the service because no route hands the token out: the
    # request endpoint answers identically for a known and an unknown
    # address (12.1). The reset itself goes through the route.
    runtime = get_runtime()
    user = runtime.store.get_user(account["user_id"])
    token = asyncio.run(runtime.auth.initiate_password_reset(user))
    assert token, "no reset token was issued"

    confirmed = client.post("/v1/auth/reset/confirm",
                            json={"token": token, "new_password": NEW_PASSWORD})
    assert confirmed.status_code == 200, confirmed.text

    assert _sessions(account["user_id"]) == 0, (
        "the reset left a session behind"
    )

    for name, credential in (("original", original), ("rotated", rotated)):
        replayed = _refresh(client, credential)
        assert replayed.status_code != 200, (
            f"the {name} refresh token still refreshes after the reset that "
            f"was supposed to revoke it: {replayed.text}"
        )
        body = replayed.json()
        assert not (body.get("data") or {}).get("access_token"), (
            f"the {name} refusal still handed back an access token: {body}"
        )
