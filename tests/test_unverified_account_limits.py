"""An unverified address is a claim, and the account lives on a short leash.

SPEC §12.1: "unverified accounts are limited to 24h and low rate limits until
verified or the grace period expires."

Before this, `email_verified` was written on signup and set by `verify_email`,
and then read by nothing at all. An unverified account got the same seven-day
session and the same rate as a verified one, which made the whole verification
flow ceremonial: it changed a boolean and nothing else.

Two consequences, and both are deliberately small. A session minted for an
unverified account is capped at twenty-four hours - the number is the SPEC's,
so it is a constant rather than a setting. And its rate limits are the
ordinary per-plan ones times one modifier, which *is* a setting, because "low"
is a number the SPEC does not give and an operator has to choose. The clause
about a grace period is not covered: what expiring is meant to do to an
account is not stated anywhere, and guessing would mean inventing a lockout.

The cap is applied where the lifetime is decided rather than at each caller.
Four seams mint a session - signup, OAuth, login, and the rotation a long-
running session performs - and every one asks `_get_session_ttl`; refresh
tokens ask `_get_refresh_ttl`. Capping in those two is what makes a fifth
seam, whenever somebody writes it, correct without being told - and each of
the four is pinned separately below, because a cap that reached signup but not
rotation would look identical from the outside for a day.

The rate half is pinned through all three chat transports for the same reason.
They share one budget, so an account held to a lower rate on `/v1/chat` alone
would simply send the same turns over the Responses API or the socket.
"""

from __future__ import annotations

import time
import uuid
from datetime import timedelta

import pytest
from fastapi import HTTPException

from liminallm.api import routes
from liminallm.api.limits import enforce_per_plan, plan_rate_multiplier
from liminallm.service.auth import UNVERIFIED_SESSION_MAX_MINUTES
from liminallm.service.runtime import get_runtime

PASSWORD = "Str0ng!passw0rd"
DAY_MINUTES = 24 * 60


def _email():
    return f"unv_{uuid.uuid4().hex[:10]}@example.com"


def _minutes(session):
    """How long this session was minted for."""
    return (session.expires_at - session.created_at).total_seconds() / 60


@pytest.fixture
def account(client):
    """A fresh unverified account, with what each transport needs to sign in."""
    email = _email()
    resp = client.post("/v1/auth/signup", json={"email": email, "password": PASSWORD})
    assert resp.status_code == 201, resp.text
    data = resp.json()["data"]
    return {
        "user_id": data["user_id"],
        "token": data["access_token"],
        "headers": {"Authorization": f"Bearer {data['access_token']}"},
    }


def _chat(client, account):
    resp = client.post(
        "/v1/chat",
        headers=account["headers"],
        json={"message": {"content": "hello"}, "stream": False},
    )
    assert resp.status_code == 200, resp.text


def _responses(client, account):
    resp = client.post(
        "/v1/responses", headers=account["headers"], json={"input": "hello"}
    )
    assert resp.status_code == 200, resp.text


def _websocket(client, account):
    with client.websocket_connect("/v1/chat/stream") as ws:
        ws.send_json(
            {
                "access_token": account["token"],
                "message": "hello",
                "stream": False,
            }
        )
        envelope = ws.receive_json()
    assert envelope["status"] == "ok", envelope


class TestTheSessionIsCappedAtADay:
    """Every seam that mints one, because a cap on the obvious path is a cap
    somebody routes around."""

    @pytest.mark.asyncio
    async def test_signup_mints_a_short_session(self, store):
        auth = get_runtime().auth
        _user, session, _tokens = await auth.signup(_email(), PASSWORD)

        assert _minutes(store.get_session(session.id)) <= DAY_MINUTES

    @pytest.mark.asyncio
    async def test_login_mints_a_short_session(self, store):
        auth = get_runtime().auth
        email = _email()
        await auth.signup(email, PASSWORD)

        _user, session, _tokens = await auth.login(email, PASSWORD)

        assert session is not None
        assert _minutes(store.get_session(session.id)) <= DAY_MINUTES

    @pytest.mark.asyncio
    async def test_a_refresh_cannot_lengthen_the_leash(self, store):
        """The seam a short signup session invites somebody to try.

        Refreshing rotates the session and issues a new refresh token. If
        either came back at the ordinary length, the cap above would be one
        request from being undone.
        """
        auth = get_runtime().auth
        email = _email()
        _u, _s, issued = await auth.signup(email, PASSWORD)

        _user, session, tokens = await auth.refresh_tokens(
            issued["refresh_token"]
        )

        assert session is not None
        assert _minutes(store.get_session(session.id)) <= DAY_MINUTES
        # And the credential itself, read the way the service reads it: a
        # capped session with an uncapped refresh token beside it is the cap
        # one request from undone.
        payload = auth._decode_jwt(tokens["refresh_token"])
        assert payload is not None
        assert payload["exp"] - time.time() <= DAY_MINUTES * 60 + 60

    @pytest.mark.asyncio
    async def test_rotation_mints_a_short_session_too(self, store, monkeypatch):
        """The fourth seam, and the one an unverified account reaches by
        waiting rather than by asking.

        A session in continuous use is rotated after `session_rotation_hours`,
        which mints a fresh one. Uncapped, that turns the day into a treadmill
        an account never has to step off.
        """
        auth = get_runtime().auth
        if auth.cache is None:
            pytest.skip("session rotation needs the Redis cache")
        user, sess, _tokens = await auth.signup(_email(), PASSWORD)

        # The real rotation path, with only the clock made to say "long ago".
        stale = auth._now() - timedelta(
            hours=get_runtime().settings.session_rotation_hours + 1
        )

        async def _long_ago(_session_id):
            return stale

        monkeypatch.setattr(auth.cache, "get_session_activity", _long_ago)
        rotated = await auth._maybe_rotate_session(sess, user)

        assert rotated is not None, "rotation did not happen, so this proves nothing"
        assert _minutes(store.get_session(rotated.id)) <= DAY_MINUTES

    @pytest.mark.asyncio
    async def test_an_oauth_signin_mints_a_short_session(
        self, store, monkeypatch
    ):
        """The third seam, and the one that carries the consequence.

        Nothing records what the provider asserted about the address - the
        identity parser drops Google's `verified_email`, and Microsoft returns
        no equivalent - so an account created this way is unverified by this
        definition and is held to the day. That is the fail-closed reading and
        it is deliberate, but it is the half of this tranche a reader should
        see stated rather than discover.
        """
        runtime = get_runtime()
        auth = runtime.auth
        settings = runtime.settings
        before = (
            settings.oauth_google_client_id,
            settings.oauth_google_client_secret,
            settings.oauth_redirect_uri,
        )
        settings.oauth_google_client_id = "test-client-id"
        settings.oauth_google_client_secret = "test-client-secret"
        settings.oauth_redirect_uri = "https://example.com/callback"
        try:
            start = await auth.start_oauth("google")
            code = uuid.uuid4().hex

            async def exchange(_provider, _code):
                # A provider that says the address is fine. Nothing reads it.
                return {
                    "provider_uid": uuid.uuid4().hex,
                    "email": _email(),
                    "handle": "someone",
                    "verified_email": True,
                }

            monkeypatch.setattr(auth, "_exchange_oauth_code", exchange)
            user, session, _tokens = await auth.complete_oauth(
                "google", code, start["state"]
            )
        finally:
            (
                settings.oauth_google_client_id,
                settings.oauth_google_client_secret,
                settings.oauth_redirect_uri,
            ) = before

        assert user is not None, "the round trip produced no account"
        assert user.email_verified is False
        assert _minutes(store.get_session(session.id)) <= DAY_MINUTES

    def test_the_two_lifetime_helpers_carry_the_cap(self):
        """Where the rule lives, stated directly: a seam that has not been
        written yet gets this for free by asking either helper.

        Equality rather than an upper bound, so this also pins the number:
        the SPEC says a day, and a cap tightened past it would lock out
        accounts that are behaving normally.
        """
        auth = get_runtime().auth

        for device in ("web", "mobile"):
            assert auth._get_session_ttl(device, verified=False) == DAY_MINUTES
            assert auth._get_refresh_ttl(device, verified=False) == DAY_MINUTES
        assert UNVERIFIED_SESSION_MAX_MINUTES == DAY_MINUTES


class TestAVerifiedAccountIsUnchanged:
    """The other half. This tranche shortens one thing and must not shorten
    anything else."""

    def test_the_ordinary_lifetimes_are_untouched(self):
        auth = get_runtime().auth
        settings = get_runtime().settings

        assert auth._get_session_ttl("web", verified=True) == (
            settings.session_ttl_minutes_web
        )
        assert auth._get_session_ttl("mobile", verified=True) == (
            settings.session_ttl_minutes_mobile
        )
        assert auth._get_refresh_ttl("web", verified=True) == (
            settings.refresh_token_ttl_minutes_web
        )

    @pytest.mark.asyncio
    async def test_verifying_lengthens_the_next_session_not_the_current_one(
        self, store
    ):
        """Immediate for what is issued next, and nothing is rewritten.

        The account's existing session keeps the length it was minted with -
        this tranche adds no revocation, and reissuing tokens somebody already
        holds is a different decision from capping the ones they get next.
        """
        auth = get_runtime().auth
        email = _email()
        user, first, _tokens = await auth.signup(email, PASSWORD)
        short = store.get_session(first.id)
        assert _minutes(short) <= DAY_MINUTES

        store.mark_email_verified(user.id)

        _u, later, _t = await auth.login(email, PASSWORD)
        assert _minutes(store.get_session(later.id)) > DAY_MINUTES
        # Untouched, not retroactively lengthened or revoked.
        assert _minutes(store.get_session(short.id)) <= DAY_MINUTES


class TestTheRateIsLowerUntilVerified:
    """One modifier over the existing per-plan multiplier, not a second
    limiter."""

    @pytest.mark.parametrize(
        "drive", [_chat, _responses, _websocket],
        ids=["chat", "responses", "websocket"],
    )
    def test_every_chat_transport_reads_it_and_reads_it_again(
        self, drive, client, account, store, monkeypatch
    ):
        """All three, because they share one budget and one abuse surface.

        Nothing above this pins the endpoints: `plan_rate_multiplier` could
        be perfectly correct while every caller passed a literal. And an
        account held to a lower rate on `/v1/chat` alone would simply send
        the same turns over the Responses API or the socket.

        So this watches what each transport hands the limiter over a real
        request, before and after the address is confirmed.
        """
        seen: list[bool] = []
        real = routes.enforce_per_plan

        async def recording(*args, verified: bool, **kwargs):
            seen.append(verified)
            return await real(*args, verified=verified, **kwargs)

        monkeypatch.setattr(routes, "enforce_per_plan", recording)

        drive(client, account)
        assert seen == [False], "this transport did not consult the limiter"

        store.mark_email_verified(account["user_id"])

        drive(client, account)
        # Immediately, on the next request, with the same token.
        assert seen == [False, True]

    def test_it_layers_over_the_plan_rather_than_replacing_it(self):
        """A paid but unverified account keeps its plan's shape - the
        modifier scales what the tier already earned, rather than standing in
        for the tier."""
        runtime = get_runtime()

        for tier in ("free", "paid", "enterprise"):
            plan = plan_rate_multiplier(runtime, tier, verified=True)
            unverified = plan_rate_multiplier(runtime, tier, verified=False)

            assert unverified < plan
            assert unverified == (
                plan * runtime.settings.unverified_rate_limit_multiplier
            )

    @pytest.mark.asyncio
    async def test_scaling_down_never_reaches_unlimited(self):
        """Scaling must not arrive at the one value that means no limit.

        A limit of 0 is the operator's way of disabling one, and `enforce`
        honors it. Three requests a minute scaled by an unverified quarter is
        `int(0.75)`, so the account whose rate was meant to be lowest would
        get no limit at all - through arithmetic, at shipped defaults, with
        nothing misconfigured.
        """
        runtime = get_runtime()
        key = f"unverified_scale:{uuid.uuid4().hex}"

        await enforce_per_plan(runtime, key, 3, 60, "free", verified=False)
        with pytest.raises(HTTPException) as raised:
            await enforce_per_plan(runtime, key, 3, 60, "free", verified=False)

        assert raised.value.status_code == 429

    def test_verifying_restores_the_ordinary_rate(self):
        runtime = get_runtime()
        assert plan_rate_multiplier(runtime, "free", verified=True) == (
            runtime.settings.rate_limit_multiplier_free
        )

    def test_the_modifier_is_an_operator_setting(self):
        """The SPEC says "low" and gives no number, so somebody has to choose
        one and be able to change it."""
        from liminallm.config import managed_settings_schema  # noqa: PLC0415

        entry = next(
            item for item in managed_settings_schema()
            if item["name"] == "unverified_rate_limit_multiplier"
        )
        assert entry["group"] == "Rate limits"
        assert entry["secret"] is False
        assert 0 < entry["default"] < 1
