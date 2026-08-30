"""Telling the account holder that a second factor was added.

`send_mfa_setup_confirmation` was written, styled and tested, and no code path
ever called it. Enabling MFA is the change an account takeover makes to lock
the real owner out, so the notice is how they find out it was not them.

It goes out once, at enrolment - not on every subsequent login, which uses the
same verify endpoint.
"""

from __future__ import annotations

import time
import uuid

import pytest

from liminallm.service.runtime import get_runtime
from tests.test_auth_flows import enrolment, reference_totp


@pytest.fixture
def sent(monkeypatch):
    """Capture the notice instead of opening an SMTP connection."""
    calls = []
    monkeypatch.setattr(
        get_runtime().email,
        "send_mfa_setup_confirmation",
        lambda to_email: calls.append(to_email) or True,
    )
    return calls


def _signup(client):
    email = f"mfa_{uuid.uuid4().hex[:8]}@example.com"
    resp = client.post(
        "/v1/auth/signup", json={"email": email, "password": "Passw0rd!Passw0rd"}
    )
    data = resp.json()["data"]
    return email, data["session_id"], data.get("csrf_token")


def _cookie_call(client, path, body, session_id, csrf):
    return client.post(
        path,
        json=body,
        cookies={"session_id": session_id, "csrf_token": csrf},
        headers={"X-CSRF-Token": csrf},
    )


def _code_for(client, session_id, csrf):
    resp = _cookie_call(
        client, "/v1/auth/mfa/request", {"session_id": session_id}, session_id, csrf
    )
    uri = resp.json()["data"].get("otpauth_uri")
    if not uri:
        pytest.skip("MFA disabled in this instance")
    app = enrolment(uri)
    return reference_totp(
        app["secret"], time.time(), app["algorithm"],
        interval=app["interval"], digits=app["digits"],
    )


def test_enrolment_tells_the_account_holder(client, sent):
    email, session_id, csrf = _signup(client)
    code = _code_for(client, session_id, csrf)

    resp = _cookie_call(
        client, "/v1/auth/mfa/verify",
        {"session_id": session_id, "code": code}, session_id, csrf,
    )

    assert resp.status_code == 200
    assert sent == [email], "MFA was enabled and nobody was told"


def test_a_later_login_does_not_send_it_again(client, sent):
    """Same endpoint, so the enabled-before check is the only thing keeping
    this from mailing the user on every sign-in."""
    _, session_id, csrf = _signup(client)
    code = _code_for(client, session_id, csrf)
    _cookie_call(
        client, "/v1/auth/mfa/verify",
        {"session_id": session_id, "code": code}, session_id, csrf,
    )
    assert len(sent) == 1

    _cookie_call(
        client, "/v1/auth/mfa/verify",
        {"session_id": session_id, "code": code}, session_id, csrf,
    )
    assert len(sent) == 1, "the notice repeated on an ordinary MFA check"


def test_a_failed_code_tells_nobody_anything(client, sent):
    _, session_id, csrf = _signup(client)
    _code_for(client, session_id, csrf)

    resp = _cookie_call(
        client, "/v1/auth/mfa/verify",
        {"session_id": session_id, "code": "000000"}, session_id, csrf,
    )

    assert resp.status_code == 401
    assert sent == []


def test_a_broken_mailer_does_not_cost_the_user_their_enrolment(client, monkeypatch):
    """The notice is best effort. An SMTP outage must not turn a successful
    enrolment into a 500 the user reads as "it did not work"."""
    def explode(to_email):
        raise OSError("smtp is down")

    monkeypatch.setattr(get_runtime().email, "send_mfa_setup_confirmation", explode)
    _, session_id, csrf = _signup(client)
    code = _code_for(client, session_id, csrf)

    resp = _cookie_call(
        client, "/v1/auth/mfa/verify",
        {"session_id": session_id, "code": code}, session_id, csrf,
    )

    assert resp.status_code == 200
    assert get_runtime().store.get_user_mfa_secret(
        resp.json()["data"]["user_id"]
    ).enabled is True
