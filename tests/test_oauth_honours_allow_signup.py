"""Signup disabled means disabled at every door that can create an account.

`allow_signup` off made the password route answer 403 `signup disabled` and
left the OAuth path untouched: an unknown identity completing the provider
round trip got an account, a session and tokens. Measured before the fix.
The two doors now ask the same question and give the same answer, and the
one that was open is closed at the point of creation rather than at a route,
so no new route can reopen it by forgetting.
"""

from __future__ import annotations

import uuid

import pytest

from liminallm.service.runtime import get_runtime
from tests.test_auth_flows import _stub_exchange


@pytest.fixture
def oauth_ready(monkeypatch):
    settings = get_runtime().settings
    monkeypatch.setattr(settings, "oauth_google_client_id", "client-id")
    monkeypatch.setattr(settings, "oauth_google_client_secret", "client-secret")
    monkeypatch.setattr(settings, "oauth_redirect_uri", "https://example.com/callback")
    return settings


async def _callback(client, email: str, *, uid: str | None = None):
    """The real route, with only the provider's half stubbed."""
    runtime = get_runtime()
    start = await runtime.auth.start_oauth(
        "google", tenant_id=runtime.settings.default_tenant_id
    )
    code = uuid.uuid4().hex
    _stub_exchange(runtime.auth, {
        ("google", code): {"provider_uid": uid or uuid.uuid4().hex,
                           "email": email, "handle": "someone"},
    })
    return client.get(
        "/v1/auth/oauth/google/callback", params={"code": code, "state": start["state"]}
    )


def _error(resp) -> dict:
    body = resp.json()
    return body.get("error") or (body.get("detail") or {}).get("error") or {}


class TestOneAnswerAtEveryDoor:
    @pytest.mark.asyncio
    async def test_oauth_refuses_an_unknown_identity_the_way_password_does(
        self, client, oauth_ready, monkeypatch
    ):
        runtime = get_runtime()
        monkeypatch.setattr(oauth_ready, "allow_signup", False)
        email = f"newcomer_{uuid.uuid4().hex[:8]}@example.com"

        password = client.post(
            "/v1/auth/signup", json={"email": email, "password": "TestPassword123!"}
        )
        oauth = await _callback(client, email)

        assert password.status_code == oauth.status_code == 403, (password.text, oauth.text)
        assert _error(password)["code"] == _error(oauth)["code"] == "forbidden"
        assert _error(password)["message"] == _error(oauth)["message"]
        assert runtime.store.get_user_by_email(email) is None, "a row was written"

    @pytest.mark.asyncio
    async def test_signup_off_is_not_login_off(self, client, oauth_ready, monkeypatch):
        runtime = get_runtime()
        email = f"regular_{uuid.uuid4().hex[:8]}@example.com"
        uid = uuid.uuid4().hex
        first = await _callback(client, email, uid=uid)
        assert first.status_code == 200, first.text
        monkeypatch.setattr(oauth_ready, "allow_signup", False)

        again = await _callback(client, email, uid=uid)

        assert again.status_code == 200, again.text
        assert again.json()["data"]["user_id"] == first.json()["data"]["user_id"]
        assert runtime.store.get_user_by_email(email) is not None

    @pytest.mark.asyncio
    async def test_turning_signup_back_on_needs_no_restart(
        self, client, oauth_ready, monkeypatch
    ):
        monkeypatch.setattr(oauth_ready, "allow_signup", False)
        email = f"later_{uuid.uuid4().hex[:8]}@example.com"
        assert (await _callback(client, email)).status_code == 403

        monkeypatch.setattr(oauth_ready, "allow_signup", True)

        assert (await _callback(client, email)).status_code == 200
