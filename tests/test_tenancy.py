"""A tenant is the site you visited, and nothing a caller can say.

Two halves: the hostname decides, and no request field or header overrides it.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from liminallm import app as app_module
from liminallm.service import tenancy
from liminallm.service.errors import NotFoundError


def _settings(**over):
    base = dict(
        default_tenant_id="public", tenant_domains={}, trust_forwarded_host=False
    )
    base.update(over)
    return SimpleNamespace(**base)


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Acme.Example.COM", "acme.example.com"),
        ("acme.example.com:8443", "acme.example.com"),
        ("acme.example.com.", "acme.example.com"),
        ("  acme.example.com  ", "acme.example.com"),
        # A proxy may append; the first hop is the one the client visited.
        ("acme.example.com, internal.lb", "acme.example.com"),
        ("[::1]:8000", "[::1]"),
        (None, ""),
    ],
)
def test_host_normalization(raw, expected):
    assert tenancy.normalize_host(raw) == expected


def test_no_mapping_means_one_tenant():
    s = _settings()
    assert tenancy.tenant_for_host("anything.example.com", s) == "public"


def test_mapped_host_gets_its_tenant():
    s = _settings(tenant_domains={"acme.example.com": "acme"})
    assert tenancy.tenant_for_host("acme.example.com", s) == "acme"


def test_unmapped_host_is_refused_once_a_mapping_exists():
    """Serving the default tenant here would mean any DNS name pointed at the
    box gets that tenant's login page - the spoofing this module prevents."""
    s = _settings(tenant_domains={"acme.example.com": "acme"})
    with pytest.raises(NotFoundError):
        tenancy.tenant_for_host("evil.example.com", s)


def test_a_bare_address_is_not_a_site_either():
    """No host is exempt once a mapping exists.

    Letting localhost through to the default tenant meant anyone who could
    reach the port named their own tenant by sending Host: localhost. Probes
    do not authenticate and never ask for a tenant, so nothing legitimate
    needed the exemption.
    """
    s = _settings(tenant_domains={"acme.example.com": "acme"})
    for host in ("", "localhost", "127.0.0.1", "::1", "testserver"):
        with pytest.raises(NotFoundError):
            tenancy.tenant_for_host(host, s)


def test_a_bare_address_still_works_on_a_single_tenant_install():
    """The strictness only applies once an operator maps a domain."""
    s = _settings()
    for host in ("", "localhost", "testserver"):
        assert tenancy.tenant_for_host(host, s) == "public"


# ---------------------------------------------------------------------------
# The account half
# ---------------------------------------------------------------------------


def test_both_halves_must_agree():
    """A session is a bearer credential; it stays valid wherever it is sent."""
    assert tenancy.user_belongs_to_site("acme", "acme")
    assert not tenancy.user_belongs_to_site("acme", "globex")


def test_a_blank_on_either_side_is_a_mismatch_not_a_pass():
    """Skipping the comparison when a value is missing is how it goes missing.

    The caller with nothing to compare is the one that resolved no site - the
    case least safe to wave through.
    """
    assert not tenancy.user_belongs_to_site("acme", "")
    assert not tenancy.user_belongs_to_site("", "acme")
    assert not tenancy.user_belongs_to_site(None, None)


# ---------------------------------------------------------------------------
# Which header is believed
# ---------------------------------------------------------------------------


def test_forwarded_host_is_ignored_unless_the_operator_opts_in():
    """Off by default: with no proxy in front, X-Forwarded-Host is just a
    header the client picked, and believing it hands the caller its tenant."""
    headers = {"host": "acme.example.com", "x-forwarded-host": "globex.example.com"}
    assert tenancy.host_of(headers, _settings()) == "acme.example.com"


def test_forwarded_host_wins_when_trusted():
    headers = {"host": "internal-lb", "x-forwarded-host": "globex.example.com"}
    s = _settings(trust_forwarded_host=True)
    assert tenancy.host_of(headers, s) == "globex.example.com"


def test_trusted_but_absent_forwarded_host_falls_back_to_host():
    headers = {"host": "acme.example.com"}
    s = _settings(trust_forwarded_host=True)
    assert tenancy.host_of(headers, s) == "acme.example.com"


# ---------------------------------------------------------------------------
# The API surface refuses to take a tenant
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    return TestClient(app_module.app)


def test_signup_rejects_a_supplied_tenant(client):
    resp = client.post(
        "/v1/auth/signup",
        json={
            "email": f"t_{uuid.uuid4().hex[:8]}@example.com",
            "password": "TestPassword123!",
            "tenant_id": "somewhere-else",
        },
    )
    assert resp.status_code == 422, resp.text


def test_oauth_start_rejects_a_supplied_tenant(client):
    resp = client.post(
        "/v1/auth/oauth/google/start",
        json={"redirect_uri": "https://example.com/cb", "tenant_id": "somewhere-else"},
    )
    assert resp.status_code == 422, resp.text


def test_login_no_longer_declares_a_tenant_field():
    """It used to accept one and silently discard it, which reads as working."""
    from liminallm.api.schemas import LoginRequest, TokenRefreshRequest

    assert "tenant_id" not in LoginRequest.model_fields
    assert "tenant_id" not in TokenRefreshRequest.model_fields


def test_no_route_reads_a_tenant_header():
    """X-Tenant-ID let a client pick the tenant its own session was checked
    against; the only outcomes were 'matches' and 401."""
    import inspect

    from liminallm.api import routes

    assert "X-Tenant-ID" not in inspect.getsource(routes)


def test_signup_lands_in_the_tenant_serving_the_host(client):
    """The default install has no mapping, so this is default_tenant_id - but
    it arrives via tenancy, not via a hardcoded None."""
    email = f"t_{uuid.uuid4().hex[:8]}@example.com"
    resp = client.post(
        "/v1/auth/signup", json={"email": email, "password": "TestPassword123!"}
    )
    assert resp.status_code == 201, resp.text
    token = resp.json()["data"]["access_token"]
    me = client.get("/v1/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200, me.text
    from liminallm.service.runtime import get_runtime

    assert me.json()["data"]["tenant_id"] == get_runtime().settings.default_tenant_id


# ---------------------------------------------------------------------------
# Two sites on one box
# ---------------------------------------------------------------------------


@pytest.fixture
def two_sites():
    """Serve acme and globex from the same install for one test."""
    from liminallm.service.runtime import get_runtime

    settings = get_runtime().settings
    before = settings.tenant_domains
    settings.tenant_domains = {
        "acme.example.com": "acme",
        "globex.example.com": "globex",
    }
    try:
        yield
    finally:
        settings.tenant_domains = before


def _signup(client, host):
    email = f"t_{uuid.uuid4().hex[:8]}@example.com"
    resp = client.post(
        "/v1/auth/signup",
        json={"email": email, "password": "TestPassword123!"},
        headers={"Host": host},
    )
    assert resp.status_code == 201, resp.text
    return email, resp.json()["data"]


def test_the_site_decides_which_tenant_an_account_joins(client, two_sites):
    _, acme = _signup(client, "acme.example.com")
    me = client.get(
        "/v1/me",
        headers={"Authorization": f"Bearer {acme['access_token']}", "Host": "acme.example.com"},
    )
    assert me.status_code == 200, me.text
    assert me.json()["data"]["tenant_id"] == "acme"


def test_an_account_cannot_sign_in_at_another_tenants_site(client, two_sites):
    email, _ = _signup(client, "acme.example.com")
    body = {"email": email, "password": "TestPassword123!"}

    at_home = client.post("/v1/auth/login", json=body, headers={"Host": "acme.example.com"})
    assert at_home.status_code == 200, at_home.text

    elsewhere = client.post(
        "/v1/auth/login", json=body, headers={"Host": "globex.example.com"}
    )
    assert elsewhere.status_code == 401, elsewhere.text


def test_a_session_cannot_be_replayed_at_another_tenants_site(client, two_sites):
    _, acme = _signup(client, "acme.example.com")
    token = {"Authorization": f"Bearer {acme['access_token']}"}

    assert client.get("/v1/me", headers={**token, "Host": "acme.example.com"}).status_code == 200
    assert client.get("/v1/me", headers={**token, "Host": "globex.example.com"}).status_code == 401


def test_an_unmapped_host_is_not_served_a_tenant(client, two_sites):
    resp = client.post(
        "/v1/auth/signup",
        json={"email": f"t_{uuid.uuid4().hex[:8]}@example.com", "password": "TestPassword123!"},
        headers={"Host": "evil.example.com"},
    )
    assert resp.status_code == 404, resp.text
    assert resp.json()["error"]["code"] == "not_found"


def test_a_bare_host_cannot_reach_the_default_tenant(client, two_sites):
    """The hole this closed: Host is chosen by whoever reaches the port.

    localhost and the test client's own testserver used to resolve to the
    default tenant, so a caller who could reach the service directly picked
    that tenant - and with signup open, registered an account on it.
    """
    for host in ("localhost", "testserver", "127.0.0.1"):
        resp = client.post(
            "/v1/auth/signup",
            json={
                "email": f"t_{uuid.uuid4().hex[:8]}@example.com",
                "password": "TestPassword123!",
            },
            headers={"Host": host},
        )
        assert resp.status_code == 404, (host, resp.text)


def test_a_client_cannot_name_its_tenant_with_a_forwarded_header(client, two_sites):
    """trust_forwarded_host is off, so this is just a header the caller chose."""
    _, acme = _signup(client, "acme.example.com")
    resp = client.get(
        "/v1/me",
        headers={
            "Authorization": f"Bearer {acme['access_token']}",
            "Host": "globex.example.com",
            "X-Forwarded-Host": "acme.example.com",
        },
    )
    assert resp.status_code == 401, resp.text


# ---------------------------------------------------------------------------
# The console can actually set it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, typed",
    [
        # Each of these declares a validator accepting a friendlier form than
        # its annotation. The per-field pass used to judge them by the raw
        # annotation alone and reject the friendly form, so all three were
        # unsettable from the admin console.
        ("tenant_domains", '{"Acme.Example.COM:443": "acme"}'),
        ("cors_allow_origins", "http://a.example.com,http://b.example.com"),
        ("tool_network_allowlist", "a.example.com,b.example.com"),
    ],
)
def test_a_setting_accepts_what_its_own_validator_accepts(name, typed):
    from liminallm.config import SYSTEM_SETTINGS_DEFAULTS, validate_managed_settings

    errors = validate_managed_settings({name: typed}, dict(SYSTEM_SETTINGS_DEFAULTS))
    assert errors == {}, errors


def test_a_bad_tenant_map_says_why():
    from liminallm.config import SYSTEM_SETTINGS_DEFAULTS, validate_managed_settings

    current = dict(SYSTEM_SETTINGS_DEFAULTS)
    assert "JSON object" in validate_managed_settings(
        {"tenant_domains": "not json"}, current
    )["tenant_domains"]
    assert "host and a tenant" in validate_managed_settings(
        {"tenant_domains": '{"acme.example.com": ""}'}, current
    )["tenant_domains"]


def test_tenancy_settings_have_a_home_in_the_console():
    from liminallm.config import managed_settings_schema

    groups = {e["name"]: e["group"] for e in managed_settings_schema()}
    for name in ("default_tenant_id", "tenant_domains", "trust_forwarded_host"):
        assert groups[name] == "Tenancy", f"{name} is under {groups.get(name)}"


def test_oauth_cannot_sign_an_account_in_at_another_tenants_site(client, two_sites):
    """The provider proves who you are, not where you belong.

    Email is globally unique, so the lookup finds the account whatever site
    the flow began at. Without the check, starting Google at globex minted
    acme's tokens - while the password path refused the same thing.
    """
    from liminallm.service.runtime import get_runtime

    auth = get_runtime().auth
    email, _ = _signup(client, "acme.example.com")
    user = auth.store.get_user_by_email(email)
    assert user.tenant_id == "acme"

    assert auth._site_matches(user, "acme") is True
    assert auth._site_matches(user, "globex") is False


def test_login_uses_the_same_rule_as_every_other_entry_point():
    """login kept a truthy check after the others were converted.

    A blank site tenant short-circuited it to False, so login would admit any
    user in any tenant while refresh and authenticate rejected the same
    request. Four entry points, one rule.
    """
    from types import SimpleNamespace

    from liminallm.service.auth import AuthService

    matches = AuthService._site_matches
    user = SimpleNamespace(tenant_id="acme")

    assert matches(None, user, "acme") is True
    assert matches(None, user, "globex") is False
    assert matches(None, user, "") is False
    # None is not "unresolved", it is "not a tenanted decision" - logout
    # revoking your own session needs no opinion about where you belong.
    assert matches(None, user, None) is True


def test_the_default_tenant_cannot_be_cleared():
    """Blank-is-mismatch turned an empty default into an unrecoverable lockout.

    Every user's tenant_id would be blank too, so every request 401s -
    including the admin call that would put the value back.
    """
    from liminallm.config import SYSTEM_SETTINGS_DEFAULTS, validate_managed_settings

    errors = validate_managed_settings(
        {"default_tenant_id": ""}, dict(SYSTEM_SETTINGS_DEFAULTS)
    )
    assert "default_tenant_id" in errors


def test_an_ipv6_host_can_be_mapped_and_then_matched():
    """Two normalizers disagreed on brackets, so ::1 was unmappable.

    Settings stripped at the first colon; the request path kept the brackets.
    Once bare addresses stopped being exempt, that made IPv6 loopback a 404
    with no spelling an operator could use to fix it.
    """
    from liminallm.config import Settings

    # The real model, because the normalization under test is the config
    # validator's - a SimpleNamespace would skip the very code being checked.
    s = Settings(default_tenant_id="public", tenant_domains={"::1": "v6"})

    assert s.tenant_domains == {"[::1]": "v6"}, "settings and requests must agree"
    for spelling in ("[::1]:8000", "::1", "[::1]"):
        host = tenancy.normalize_host(spelling)
        assert tenancy.tenant_for_host(host, s) == "v6", spelling
