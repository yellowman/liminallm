"""An unverified account has one 24-hour grace period, not one per login.

SPEC 12.1: "unverified accounts are limited to 24h and low rate limits until
verified or the grace period expires". The grace period is the subject of that
last clause, so it belongs to the account, not to each session it opens.

The implementation capped each session's lifetime at 24 hours instead. Every
login minted another full day, so an account could stay unverified indefinitely
in 24-hour increments and "the grace period expires" described nothing. An API
key minted inside the window had no relationship to the deadline at all.

The rule pinned here:

    deadline = max(created_at, migration_floor) + 24h

Before it, an unverified account works under the reduced rate policy, and no
credential outlasts the deadline. At or after it, nothing authenticates: not
password login, not OAuth, not refresh, not an API key. What a refused attempt
does instead is send a fresh verification message, because `/auth/verify_email`
needs only the token - so proving the password or the provider identity is
enough to get the mail that restores the account, and expiry is dormancy rather
than a lockout.

The floor exists because every OAuth account in an existing database is
unverified through an implementation gap. Deriving their deadline from
`created_at` would expire all of them the moment this ships.

Provider attestation is the other half. Authenticating with a provider proves
the identity, not the address. Only an explicit claim about that exact address
marks the account verified: Google's `verified_email`, and GitHub's
`/user/emails` entry that is both primary and verified. Microsoft Graph `/me`
carries no such claim, so those accounts verify through the mailbox like
everyone else.
"""

from __future__ import annotations

import uuid
from datetime import timedelta

import psycopg
import pytest

from liminallm.service.auth import VERIFICATION_GRACE_FLOOR
from liminallm.service.errors import ServiceError
from liminallm.service.runtime import get_runtime

PASSWORD = "GracePassword123!"


def _age_account(store, user_id, hours):
    """Move an account's birthday back, which moves its deadline back."""
    with psycopg.connect(store.dsn, autocommit=True) as conn:
        conn.execute(
            "UPDATE app_user SET created_at = created_at - "
            "make_interval(hours => %s) WHERE id = %s",
            (hours, user_id),
        )


def _age_floor(store, hours):
    """Age the migration floor too, or it holds every deadline open.

    The floor is resolved once and kept in memory, so the stored row is moved
    and the resolved copy dropped - otherwise this writes a value nothing
    reads, and every assertion below would be measuring the floor recorded
    when this test started.
    """
    runtime = get_runtime()
    current = runtime.auth._grace_floor()  # also ensures the row exists
    # Aged in Python and written back as Python writes it. Doing the
    # arithmetic in SQL and storing `::text` renders Postgres's own format,
    # whose two-digit offset (`+00`) `datetime.fromisoformat` rejects on
    # Python 3.10 - it wants `+00:00`, and only 3.11 relaxed that. The parse
    # then failed, the floor was re-established at now, nothing was expired,
    # and every assertion below failed on one interpreter and passed on the
    # other two. Production never sees that format: the only writer stores
    # `now.isoformat()`.
    aged = (current - timedelta(hours=hours)).isoformat()
    with psycopg.connect(store.dsn, autocommit=True) as conn:
        conn.execute(
            "UPDATE instance_config SET config = "
            "jsonb_set(config, '{recorded_at}', to_jsonb(%s::text)) "
            "WHERE name = %s",
            (aged, VERIFICATION_GRACE_FLOOR),
        )
    runtime.auth._grace_floor_cache = None


def _expire(store, user_id, hours=25):
    _age_account(store, user_id, hours)
    _age_floor(store, hours)


@pytest.fixture
def account(client, store):
    email = f"grace_{uuid.uuid4().hex[:8]}@example.com"
    resp = client.post("/v1/auth/signup",
                       json={"email": email, "password": PASSWORD})
    assert resp.status_code == 201, resp.text
    data = resp.json()["data"]
    assert store.get_user(data["user_id"]).email_verified is False
    return {"email": email, "user_id": data["user_id"],
            "refresh": data["refresh_token"], "session_id": data.get("session_id")}


def _login(client, email):
    return client.post("/v1/auth/login",
                       json={"email": email, "password": PASSWORD})


def _refresh(client, token):
    return client.post("/v1/auth/refresh", json={"refresh_token": token})


class TestInsideTheGracePeriod:
    def test_a_fresh_unverified_account_works(self, client, account):
        """The control. Every refusal below needs this to have succeeded."""
        assert _login(client, account["email"]).status_code == 200
        assert _refresh(client, account["refresh"]).status_code == 200

    def test_no_session_outlives_the_deadline(self, client, store, account):
        """The cap is the deadline, not a rolling day from each login."""
        _age_account(store, account["user_id"], 23)
        resp = _login(client, account["email"])
        assert resp.status_code == 200, resp.text

        runtime = get_runtime()
        user = store.get_user(account["user_id"])
        deadline = runtime.auth._verification_deadline(user)
        session = store.get_session(resp.json()["data"]["session_id"])
        assert session.expires_at <= deadline, (
            "a session minted with an hour of grace left runs past the "
            "verification deadline"
        )


class TestAfterTheDeadline:
    def test_password_login_is_refused_and_sends_a_new_message(
        self, client, store, account
    ):
        _expire(store, account["user_id"])
        runtime = get_runtime()
        sent: list = []
        real = runtime.auth.request_email_verification

        async def _record(user):
            sent.append(user.id)
            return await real(user)

        runtime.auth.request_email_verification = _record
        try:
            resp = _login(client, account["email"])
        finally:
            runtime.auth.request_email_verification = real

        assert resp.status_code == 403, resp.text
        assert resp.json()["error"]["code"] == "verification_required", resp.text
        assert sent == [account["user_id"]], (
            "the refusal did not send a fresh verification message, so the "
            "account has no way back"
        )

    def test_a_wrong_password_is_still_just_unauthorized(
        self, client, store, account
    ):
        """The refusal must not become an account-existence oracle."""
        _expire(store, account["user_id"])
        resp = client.post("/v1/auth/login",
                           json={"email": account["email"], "password": "wrong"})
        assert resp.status_code == 401, resp.text
        assert resp.json()["error"]["code"] != "verification_required"

    def test_refresh_is_refused(self, client, store, account):
        _expire(store, account["user_id"])
        resp = _refresh(client, account["refresh"])
        assert resp.status_code != 200
        assert not (resp.json().get("data") or {}).get("access_token")

    def test_an_api_key_goes_dormant_rather_than_dying(
        self, client, store, account
    ):
        runtime = get_runtime()
        store.mark_email_verified(account["user_id"])
        _record, plaintext = runtime.auth.mint_api_key(
            account["user_id"], name="probe"
        )
        _unverify(store, account["user_id"])
        assert runtime.auth.authenticate_api_key(
            f"Bearer {plaintext}"
        ) is not None, "the control failed: the key never authenticated"

        _expire(store, account["user_id"])
        assert runtime.auth.authenticate_api_key(f"Bearer {plaintext}") is None, (
            "an API key still carries authority after the grace period ended"
        )
        assert [k for k in store.list_api_keys(account["user_id"])
                if k.revoked_at is None], "the key was destroyed, not made dormant"

        store.mark_email_verified(account["user_id"])
        assert runtime.auth.authenticate_api_key(f"Bearer {plaintext}") is not None, (
            "verifying did not wake the key"
        )

    def test_verifying_restores_login(self, client, store, account):
        _expire(store, account["user_id"])
        assert _login(client, account["email"]).status_code == 403
        store.mark_email_verified(account["user_id"])
        assert _login(client, account["email"]).status_code == 200


def _unverify(store, user_id):
    with psycopg.connect(store.dsn, autocommit=True) as conn:
        conn.execute(
            "UPDATE app_user SET meta = jsonb_set("
            "COALESCE(meta, '{}'::jsonb), '{email_verified}', 'false') "
            "WHERE id = %s",
            (user_id,),
        )


class TestMinting:
    def test_an_unverified_account_cannot_mint_a_key(self, account):
        """A long-lived credential does not belong to a temporary trust state."""
        with pytest.raises(ServiceError) as caught:
            get_runtime().auth.mint_api_key(account["user_id"], name="probe")
        assert "verif" in str(caught.value).lower(), caught.value

    def test_a_verified_account_still_can(self, store, account):
        """The control for the refusal above."""
        store.mark_email_verified(account["user_id"])
        record, plaintext = get_runtime().auth.mint_api_key(
            account["user_id"], name="probe"
        )
        assert record.id and plaintext


class TestAVerifiedAccountIsUnaffected:
    def test_age_does_not_matter_once_verified(self, client, store, account):
        store.mark_email_verified(account["user_id"])
        _age_account(store, account["user_id"], 500)
        assert _login(client, account["email"]).status_code == 200
        assert _refresh(client, account["refresh"]).status_code == 200


class TestTheMigrationFloor:
    def test_an_account_older_than_the_floor_still_gets_a_day(
        self, client, store, account
    ):
        """Shipping this must not expire every account that predates it.

        Every OAuth account in an existing database is unverified through an
        implementation gap, so a deadline derived from `created_at` alone would
        lock all of them out on deploy.
        """
        _age_account(store, account["user_id"], 500)

        assert _login(client, account["email"]).status_code == 200, (
            "an account older than the grace period was expired by its own "
            "creation date rather than by the deployment floor"
        )


class TestProviderAttestation:
    """Authenticating with a provider proves the identity, not the address."""

    def _identity(self, provider, userinfo):
        return get_runtime().auth._parse_oauth_userinfo(provider, userinfo)

    def test_google_is_trusted_only_when_it_says_verified(self):
        assert self._identity("google", {
            "id": "1", "email": "a@example.com", "verified_email": True,
        })["email_verified"] is True
        for claim in ({"verified_email": False}, {}, {"verified_email": "true"}):
            got = self._identity("google", {"id": "1", "email": "a@example.com", **claim})
            assert got["email_verified"] is False, claim

    def test_microsoft_is_never_trusted(self):
        """Graph `/me` carries no verified-address claim, and the mail value
        is mutable."""
        assert self._identity("microsoft", {
            "id": "1", "mail": "a@example.com", "displayName": "A",
        })["email_verified"] is False

    def test_an_unknown_provider_is_never_trusted(self):
        assert self._identity("whoever", {"id": "1"}).get("email_verified") is False


def test_verification_committing_during_the_password_check_is_seen(
    client, store, account
):
    """The snapshot race: verified between the read and the grace decision.

    `login` reads the account, verifies the password, then judges the grace.
    A verification committing inside that window must not be judged from the
    row read before it - which would refuse an account that is now verified,
    or, with the states reversed, issue credentials calculated from a state
    that no longer exists.
    """
    _expire(store, account["user_id"])
    runtime = get_runtime()
    real_verify = runtime.auth.verify_password
    landed: list = []

    def _verify_then_verify_email(user_id, password):
        ok = real_verify(user_id, password)
        if ok and not landed:
            store.mark_email_verified(user_id)
            landed.append(user_id)
        return ok

    runtime.auth.verify_password = _verify_then_verify_email
    try:
        resp = _login(client, account["email"])
    finally:
        runtime.auth.verify_password = real_verify

    assert landed == [account["user_id"]], "the interleaving never happened"
    assert resp.status_code == 200, (
        "the grace was judged from a snapshot taken before verification "
        f"committed: {resp.text}"
    )


class TestTheOAuthMatrix:
    """Provider proof establishes login and verification, not renaming."""

    def _identity(self, **kw):
        base = {"provider_uid": "gh-1", "email": None, "email_verified": False}
        base.update(kw)
        return base

    def test_a_trusted_claim_for_this_account_verifies_it(self, store, account):
        runtime = get_runtime()
        user = store.get_user(account["user_id"])
        _expire(store, account["user_id"])

        identity = self._identity(email=user.email, email_verified=True)
        assert runtime.auth._should_mark_verified(user, identity) is True

    def test_a_trusted_claim_for_another_address_does_not(self, store, account):
        """An attestation about one mailbox says nothing about another."""
        runtime = get_runtime()
        user = store.get_user(account["user_id"])

        identity = self._identity(
            email="someone-else@example.com", email_verified=True
        )
        assert runtime.auth._should_mark_verified(user, identity) is False

    def test_an_untrusted_claim_never_verifies(self, store, account):
        runtime = get_runtime()
        user = store.get_user(account["user_id"])

        identity = self._identity(email=user.email, email_verified=False)
        assert runtime.auth._should_mark_verified(user, identity) is False

    def test_an_already_verified_account_is_left_alone(self, store, account):
        runtime = get_runtime()
        store.mark_email_verified(account["user_id"])
        user = store.get_user(account["user_id"])

        identity = self._identity(email=user.email, email_verified=True)
        assert runtime.auth._should_mark_verified(user, identity) is False


class TestGithubEmailResolution:
    def test_the_verified_primary_becomes_the_address(self):
        """And the public profile address does not carry the attestation."""
        identity = get_runtime().auth._apply_github_emails(
            {"provider_uid": "1", "email": "public@example.com",
             "email_verified": False},
            [{"email": "public@example.com", "primary": False, "verified": False},
             {"email": "real@example.com", "primary": True, "verified": True}],
        )
        assert identity["email"] == "real@example.com"
        assert identity["email_verified"] is True

    def test_no_verified_primary_leaves_it_unproven(self):
        identity = get_runtime().auth._apply_github_emails(
            {"provider_uid": "1", "email": "public@example.com",
             "email_verified": False},
            [{"email": "public@example.com", "primary": True, "verified": False}],
        )
        assert identity["email"] == "public@example.com"
        assert identity["email_verified"] is False

    def test_a_malformed_payload_is_not_an_attestation(self):
        for payload in ({"message": "Bad credentials"}, [], ["x"], None):
            identity = get_runtime().auth._apply_github_emails(
                {"provider_uid": "1", "email": "p@example.com",
                 "email_verified": False},
                payload,
            )
            assert identity["email_verified"] is False, payload


class TestTheMigrationFloorIsEstablishedOnce:
    """The floor decides when every unverified account expires, so two
    workers establishing it must not each get their own."""

    def test_racing_workers_converge_on_one_floor(self, store):
        """Both boot together, each with its own idea of `now`.

        The danger is not that one is wrong - it is that they disagree, which
        would give the same account two different deadlines depending on which
        worker answered the request.
        """
        import threading

        from liminallm.service.auth import VERIFICATION_GRACE_FLOOR
        from liminallm.service.runtime import Runtime

        workers = [Runtime().auth for _ in range(4)]
        seen: list = []
        barrier = threading.Barrier(len(workers))

        def _establish(auth):
            barrier.wait()
            seen.append(auth._grace_floor())

        threads = [
            threading.Thread(target=_establish, args=(auth,)) for auth in workers
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(seen) == len(workers), "a worker never established a floor"
        assert len(set(seen)) == 1, (
            f"workers disagreed about the verification floor: {sorted(set(seen))}"
        )
        stored = store.get_instance_config(VERIFICATION_GRACE_FLOOR)
        assert stored.get("recorded_at") == seen[0].isoformat(), (
            "the value the workers agreed on is not the one that was committed"
        )

    def test_a_later_worker_cannot_move_it(self, store):
        """Arriving afterwards reads the first commit, and writing does not
        replace it."""
        from liminallm.service.auth import VERIFICATION_GRACE_FLOOR
        from liminallm.service.runtime import Runtime

        first = Runtime().auth._grace_floor()

        later = Runtime().auth
        assert later._grace_floor() == first

        # Even asking directly, with a value of its own.
        written = store.record_instance_config_default(
            VERIFICATION_GRACE_FLOOR, {"recorded_at": "2099-01-01T00:00:00+00:00"}
        )
        assert written["recorded_at"] == first.isoformat(), (
            "a later write moved a floor that is supposed to be immutable"
        )


def test_the_refresh_credential_never_outlives_the_deadline(client, store, account):
    """The session cap alone would leave the longer credential beside it."""
    _age_account(store, account["user_id"], 23)
    resp = _login(client, account["email"])
    assert resp.status_code == 200, resp.text

    runtime = get_runtime()
    user = store.get_user(account["user_id"])
    deadline = runtime.auth._verification_deadline(user)
    payload = runtime.auth._decode_jwt(resp.json()["data"]["refresh_token"])
    assert payload is not None
    assert payload["exp"] <= deadline.timestamp() + 60, (
        "the refresh token outlives the verification deadline, so the cap on "
        "the session is one request from being undone"
    )


class TestOAuthRefusesWithoutProof:
    def test_a_failed_exchange_says_nothing(self, store, account):
        """No provider proof, so no verification state is disclosed.

        The expired account exists and is unverified. A caller who cannot
        complete the exchange must not learn either fact.
        """
        import asyncio

        runtime = get_runtime()
        auth = runtime.auth
        _expire(store, account["user_id"])
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
            start = asyncio.run(auth.start_oauth("google"))

            async def _fails(_provider, _code):
                return None

            real = auth._exchange_oauth_code
            auth._exchange_oauth_code = _fails
            try:
                user, session, tokens = asyncio.run(
                    auth.complete_oauth("google", "code", start["state"])
                )
            finally:
                auth._exchange_oauth_code = real
        finally:
            (
                settings.oauth_google_client_id,
                settings.oauth_google_client_secret,
                settings.oauth_redirect_uri,
            ) = before

        assert (user, session, tokens) == (None, None, {}), (
            "a failed exchange produced something other than the generic refusal"
        )

    def test_an_unknown_state_says_nothing(self):
        import asyncio
        import uuid as _uuid

        auth = get_runtime().auth
        assert asyncio.run(
            auth.complete_oauth("google", "code", _uuid.uuid4().hex)
        ) == (None, None, {})
