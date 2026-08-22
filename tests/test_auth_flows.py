"""Every way in other than email and password: OAuth, MFA, reset, verification.

Against the real service, store and Redis — the MFA lockout and the reset
tokens are atomic cache operations, and the fallback beside them is a
different implementation.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import struct
import time
import uuid
from datetime import timedelta

import pytest

from liminallm.service.runtime import get_runtime

# ---------------------------------------------------------------------------
# A reference TOTP, checked against RFC 6238's own vectors
# ---------------------------------------------------------------------------


def reference_totp(secret_b32: str, timestamp: float, algorithm, *, interval=30, digits=6) -> str:
    """TOTP exactly as RFC 6238 defines it, for use as an oracle.

    Verified against Appendix B: seed "12345678901234567890" at t=59 gives
    94287082 under SHA-1, and the SHA-256 seed gives 46119246. An authenticator
    app implements this; if the server disagrees, the server is wrong.
    """
    key = base64.b32decode(secret_b32 + "=" * ((8 - len(secret_b32) % 8) % 8), True)
    counter = struct.pack(">Q", int(timestamp // interval))
    digest = hmac.new(key, counter, algorithm).digest()
    offset = digest[-1] & 0x0F
    code = (int.from_bytes(digest[offset : offset + 4], "big") & 0x7FFFFFFF) % 10**digits
    return str(code).zfill(digits)


def test_the_reference_matches_rfc_6238():
    """If this drifts, every assertion below means nothing."""
    seed = base64.b32encode(b"12345678901234567890").decode().rstrip("=")
    assert reference_totp(seed, 59, hashlib.sha1, digits=8) == "94287082"
    seed256 = base64.b32encode(b"12345678901234567890123456789012").decode().rstrip("=")
    assert reference_totp(seed256, 59, hashlib.sha256, digits=8) == "46119246"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def auth():
    return get_runtime().auth


@pytest.fixture
def store():
    return get_runtime().store


@pytest.fixture
def user(store):
    return store.create_user(
        email=f"auth_{uuid.uuid4().hex[:8]}@example.com", tenant_id="public"
    )


# ---------------------------------------------------------------------------
# MFA
# ---------------------------------------------------------------------------


_HASHES = {"SHA1": hashlib.sha1, "SHA256": hashlib.sha256, "SHA512": hashlib.sha512}


def enrolment(uri: str) -> dict:
    """What an authenticator app reads out of the QR code.

    Anything the URI omits takes the Key Uri Format default — SHA1, 6 digits,
    30 seconds — because that is what the app will assume.
    """
    from urllib.parse import parse_qs, urlparse

    q = {k: v[0] for k, v in parse_qs(urlparse(uri).query).items()}
    return {
        "secret": q["secret"],
        "algorithm": _HASHES[q.get("algorithm", "SHA1").upper()],
        "digits": int(q.get("digits", 6)),
        "interval": int(q.get("period", 30)),
    }


@pytest.mark.asyncio
async def test_an_authenticator_apps_code_is_accepted(auth, user):
    """The server must verify whatever its own QR code promises.

    It promised nothing and verified HMAC-SHA-256, so every app computed
    HMAC-SHA-1 — the Key Uri Format default — and enrolment could never
    complete. The user scanned the code, typed a correct number, and was told
    it was wrong, with nothing logged to say why.
    """
    uri = (await auth.issue_mfa_challenge(user.id))["otpauth_uri"]
    app = enrolment(uri)

    code = reference_totp(
        app["secret"],
        time.time(),
        app["algorithm"],
        interval=app["interval"],
        digits=app["digits"],
    )
    assert await auth.verify_mfa_challenge(user.id, code), (
        f"An authenticator following {uri} was rejected. Nobody can enable MFA."
    )


@pytest.mark.asyncio
async def test_enrolment_uri_carries_what_an_app_needs(auth, user):
    uri = (await auth.issue_mfa_challenge(user.id))["otpauth_uri"]
    assert uri.startswith("otpauth://totp/")
    assert "secret=" in uri and "issuer=" in uri
    app = enrolment(uri)
    # RFC 4226 §4 R6: 160-bit shared secret. Base32 of 20 bytes is 32 chars.
    assert len(app["secret"]) >= 32, "shared secret is shorter than RFC 4226 asks for"


@pytest.mark.asyncio
async def test_a_wrong_code_is_rejected(auth, user):
    await auth.issue_mfa_challenge(user.id)
    assert not await auth.verify_mfa_challenge(user.id, "000000")


@pytest.mark.asyncio
async def test_mfa_without_enrolment_is_rejected(auth):
    assert not await auth.verify_mfa_challenge(str(uuid.uuid4()), "123456")


@pytest.mark.asyncio
async def test_five_wrong_codes_lock_the_account(auth, user):
    """SPEC §18: five failed codes locks MFA for five minutes. The lockout is
    an atomic Redis operation precisely so two racing attempts cannot both
    slip past the fifth."""
    app = enrolment((await auth.issue_mfa_challenge(user.id))["otpauth_uri"])

    for _ in range(5):
        assert not await auth.verify_mfa_challenge(user.id, "000000")

    good = reference_totp(app["secret"], time.time(), app["algorithm"])
    assert not await auth.verify_mfa_challenge(user.id, good), (
        "A correct code was accepted while the account should be locked out."
    )


@pytest.mark.asyncio
async def test_a_good_code_enables_the_secret(auth, user, store):
    app = enrolment((await auth.issue_mfa_challenge(user.id))["otpauth_uri"])
    assert store.get_user_mfa_secret(user.id).enabled is False

    code = reference_totp(app["secret"], time.time(), app["algorithm"])
    assert await auth.verify_mfa_challenge(user.id, code)
    assert store.get_user_mfa_secret(user.id).enabled is True


@pytest.mark.asyncio
async def test_verifying_marks_the_session(auth, user, store):
    session = store.create_session(user.id, mfa_required=True, tenant_id="public")
    app = enrolment((await auth.issue_mfa_challenge(user.id))["otpauth_uri"])

    code = reference_totp(app["secret"], time.time(), app["algorithm"])
    assert await auth.verify_mfa_challenge(user.id, code, session_id=session.id)
    assert store.get_session(session.id).mfa_verified is True


# ---------------------------------------------------------------------------
# Password reset
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_reset_token_rotates_the_password(auth, user):
    auth.save_password(user.id, "OldPassword123!")
    assert auth.verify_password(user.id, "OldPassword123!")

    token = await auth.initiate_password_reset(user)
    assert await auth.complete_password_reset(token, "NewPassword456!")

    assert auth.verify_password(user.id, "NewPassword456!")
    assert not auth.verify_password(user.id, "OldPassword123!")


@pytest.mark.asyncio
async def test_a_reset_revokes_every_existing_session(auth, user, store):
    """Whoever forced the reset must not keep a session they already had."""
    auth.save_password(user.id, "OldPassword123!")
    session = store.create_session(user.id, tenant_id="public")

    token = await auth.initiate_password_reset(user)
    assert await auth.complete_password_reset(token, "NewPassword456!")

    assert await auth.resolve_session(session.id) is None


@pytest.mark.asyncio
async def test_a_reset_token_works_once(auth, user):
    auth.save_password(user.id, "OldPassword123!")
    token = await auth.initiate_password_reset(user)
    assert await auth.complete_password_reset(token, "NewPassword456!")
    assert not await auth.complete_password_reset(token, "ThirdPassword789!")
    assert not auth.verify_password(user.id, "ThirdPassword789!")


@pytest.mark.asyncio
async def test_an_unknown_reset_token_is_refused(auth):
    assert not await auth.complete_password_reset("not-a-real-token", "Whatever123!")


@pytest.mark.asyncio
async def test_a_reset_for_a_deleted_user_is_refused(auth, user, store):
    token = await auth.initiate_password_reset(user)
    store.delete_user(user.id)
    assert not await auth.complete_password_reset(token, "NewPassword456!")


# ---------------------------------------------------------------------------
# Email verification
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_verification_token_marks_the_user(auth, user, store):
    token = await auth.request_email_verification(user)
    assert await auth.complete_email_verification(token)
    assert (store.get_user(user.id).meta or {}).get("email_verified") is True


@pytest.mark.asyncio
async def test_a_verification_token_works_once(auth, user):
    token = await auth.request_email_verification(user)
    assert await auth.complete_email_verification(token)
    assert not await auth.complete_email_verification(token)


@pytest.mark.asyncio
async def test_an_unknown_verification_token_is_refused(auth):
    assert not await auth.complete_email_verification("not-a-real-token")


# ---------------------------------------------------------------------------
# OAuth
# ---------------------------------------------------------------------------


@pytest.fixture
def oauth_configured():
    """Enough config for start_oauth, undone afterwards."""
    settings = get_runtime().settings
    before = (
        settings.oauth_google_client_id,
        settings.oauth_google_client_secret,
        settings.oauth_redirect_uri,
    )
    settings.oauth_google_client_id = "test-client-id"
    settings.oauth_google_client_secret = "test-client-secret"
    settings.oauth_redirect_uri = "https://example.com/callback"
    try:
        yield settings
    finally:
        (
            settings.oauth_google_client_id,
            settings.oauth_google_client_secret,
            settings.oauth_redirect_uri,
        ) = before


def _stub_exchange(auth, payloads):
    """Stub the provider round trip — the seam tests are meant to use.

    `_exchange_oauth_code` is the trust boundary; production has no way to
    pre-register a payload, so the stub lives here and consumes each code
    once, the way a real provider treats an authorization code.
    """

    async def exchange(provider, code):
        return payloads.pop((provider, code), None)

    auth._exchange_oauth_code = exchange
    return payloads


async def _round_trip(auth, *, tenant_id=None, email=None, uid=None):
    """start_oauth then complete_oauth, with the provider stubbed offline."""
    start = await auth.start_oauth("google", tenant_id=tenant_id)
    code = uuid.uuid4().hex
    _stub_exchange(auth, {
        ("google", code): {
            "provider_uid": uid or uuid.uuid4().hex,
            "email": email or f"oauth_{uuid.uuid4().hex[:8]}@example.com",
            "handle": "someone",
        },
    })
    return await auth.complete_oauth("google", code, start["state"])


@pytest.mark.asyncio
async def test_oauth_start_builds_a_provider_url(auth, oauth_configured):
    start = await auth.start_oauth("google")
    assert start["authorization_url"].startswith(
        "https://accounts.google.com/o/oauth2/v2/auth?"
    )
    assert f"state={start['state']}" in start["authorization_url"]
    assert "client_id=test-client-id" in start["authorization_url"]


@pytest.mark.asyncio
async def test_oauth_start_refuses_an_unconfigured_provider(auth):
    settings = get_runtime().settings
    before = settings.oauth_github_client_id
    settings.oauth_github_client_id = ""
    try:
        with pytest.raises(ValueError):
            await auth.start_oauth("github")
    finally:
        settings.oauth_github_client_id = before


@pytest.mark.asyncio
async def test_oauth_start_refuses_an_unknown_provider(auth):
    with pytest.raises(ValueError):
        await auth.start_oauth("myspace")


@pytest.mark.asyncio
async def test_oauth_creates_a_user_and_a_session(auth, oauth_configured):
    user, session, tokens = await _round_trip(auth)
    assert user is not None and session is not None
    assert tokens.get("access_token")
    assert await auth.resolve_session(session.id) is not None


@pytest.mark.asyncio
async def test_oauth_returning_the_same_identity_reuses_the_account(
    auth, oauth_configured
):
    uid = uuid.uuid4().hex
    first, _, _ = await _round_trip(auth, uid=uid)
    second, _, _ = await _round_trip(auth, uid=uid)
    assert first.id == second.id


@pytest.mark.asyncio
async def test_oauth_lands_the_user_in_the_tenant_bound_at_start(
    auth, oauth_configured
):
    """The tenant the site resolved at /start is carried in the server-created
    state and read back at the callback. Nothing in between can change it."""
    user, _, _ = await _round_trip(auth, tenant_id="acme")
    assert user.tenant_id == "acme"


@pytest.mark.asyncio
async def test_an_oauth_state_cannot_be_replayed(auth, oauth_configured):
    start = await auth.start_oauth("google")
    code = uuid.uuid4().hex
    payload = {"provider_uid": uuid.uuid4().hex, "email": "replay@example.com"}
    payloads = _stub_exchange(auth, {("google", code): payload})
    user, _, _ = await auth.complete_oauth("google", code, start["state"])
    assert user is not None

    payloads[("google", code)] = payload
    again, _, _ = await auth.complete_oauth("google", code, start["state"])
    assert again is None, "a spent OAuth state was accepted a second time"


@pytest.mark.asyncio
async def test_an_oauth_state_is_bound_to_its_provider(auth, oauth_configured):
    start = await auth.start_oauth("google")
    code = uuid.uuid4().hex
    _stub_exchange(auth, {("github", code): {"provider_uid": "x"}})
    user, _, _ = await auth.complete_oauth("github", code, start["state"])
    assert user is None


@pytest.mark.asyncio
async def test_an_expired_oauth_state_is_refused(auth, oauth_configured):
    start = await auth.start_oauth("google")
    state = start["state"]
    provider, _expires, tenant = auth._oauth_states[state]
    auth._oauth_states[state] = (provider, auth._now() - timedelta(minutes=1), tenant)
    if auth.cache:
        await auth.cache.pop_oauth_state(state)

    code = uuid.uuid4().hex
    _stub_exchange(auth, {("google", code): {"provider_uid": "x"}})
    user, _, _ = await auth.complete_oauth("google", code, state)
    assert user is None


@pytest.mark.asyncio
async def test_an_unknown_oauth_state_is_refused(auth, oauth_configured):
    user, _, _ = await auth.complete_oauth("google", "somecode", uuid.uuid4().hex)
    assert user is None


@pytest.mark.asyncio
async def test_an_oauth_account_cannot_be_signed_into_with_a_password(
    auth, oauth_configured
):
    """An unusable password marker is stored so the password path cannot
    succeed for an account that only ever authenticated via a provider."""
    user, _, _ = await _round_trip(auth)
    assert not auth.verify_password(user.id, "")
    assert not auth.verify_password(user.id, "password")


@pytest.mark.parametrize(
    "provider, userinfo, expected_uid, expected_email",
    [
        ("google", {"id": "g1", "email": "a@x.com", "name": "A"}, "g1", "a@x.com"),
        ("github", {"id": 7, "login": "gh", "email": "b@x.com"}, "7", "b@x.com"),
        (
            "microsoft",
            {"id": "m1", "userPrincipalName": "c@x.com", "displayName": "C"},
            "m1",
            "c@x.com",
        ),
        # Microsoft puts the address in `mail` when it has one.
        ("microsoft", {"id": "m2", "mail": "d@x.com"}, "m2", "d@x.com"),
    ],
)
def test_each_provider_maps_onto_one_identity_shape(
    auth, provider, userinfo, expected_uid, expected_email
):
    parsed = auth._parse_oauth_userinfo(provider, userinfo)
    assert parsed["provider_uid"] == expected_uid
    assert parsed["email"] == expected_email
