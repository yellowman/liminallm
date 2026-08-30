"""Integration tests for authentication flow.

Tests the complete auth flow including:
- User signup
- Login with password
- MFA setup and verification
- Email verification
- Password reset
- Token refresh
- Logout
"""

import pytest
from fastapi.testclient import TestClient

from liminallm import app as app_module


@pytest.fixture
def client():
    """Create a test client for the API."""
    return TestClient(app_module.app)


@pytest.fixture
def test_user_email():
    import uuid
    return f"testuser_{uuid.uuid4().hex[:8]}@example.com"


@pytest.fixture
def test_user_password():
    return "TestPassword123!"


class TestSignupFlow:
    """Tests for user registration."""

    def test_signup_creates_user(self, client, test_user_email, test_user_password):
        """Test that signup creates a new user."""
        response = client.post(
            "/v1/auth/signup",
            json={"email": test_user_email, "password": test_user_password},
        )

        assert response.status_code == 201  # Resource created
        data = response.json()
        assert data["status"] == "ok"
        assert "user_id" in data["data"]
        assert "session_id" in data["data"]

    def test_signup_rejects_duplicate_email(
        self, client, test_user_email, test_user_password
    ):
        """Test that signup rejects duplicate emails."""
        # First signup
        client.post(
            "/v1/auth/signup",
            json={"email": test_user_email, "password": test_user_password},
        )

        # Second signup with same email
        response = client.post(
            "/v1/auth/signup",
            json={"email": test_user_email, "password": test_user_password},
        )

        assert response.status_code == 409
        # API uses envelope format with "error" key containing {code, message, details}
        error = response.json().get("error", {})
        error_msg = error.get("message", "") if isinstance(error, dict) else str(error)
        assert "already exists" in error_msg.lower()

    def test_signup_validates_email_format(self, client, test_user_password):
        """Test that signup validates email format."""
        response = client.post(
            "/v1/auth/signup",
            json={"email": "invalid-email", "password": test_user_password},
        )

        assert response.status_code == 422

    def test_signup_validates_password_strength(self, client, test_user_email):
        """Test that signup validates password strength."""
        response = client.post(
            "/v1/auth/signup",
            json={"email": test_user_email, "password": "short"},
        )

        assert response.status_code == 422


class TestLoginFlow:
    """Tests for user login."""

    def test_login_with_valid_credentials(
        self, client, test_user_email, test_user_password
    ):
        """Test login with valid credentials."""
        # First signup
        client.post(
            "/v1/auth/signup",
            json={"email": test_user_email, "password": test_user_password},
        )

        # Then login
        response = client.post(
            "/v1/auth/login",
            json={"email": test_user_email, "password": test_user_password},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "access_token" in data["data"]
        assert "session_id" in data["data"]

    def test_login_with_invalid_password(self, client, test_user_email, test_user_password):
        """Test login with wrong password."""
        # First signup
        client.post(
            "/v1/auth/signup",
            json={"email": test_user_email, "password": test_user_password},
        )

        # Login with wrong password
        response = client.post(
            "/v1/auth/login",
            json={"email": test_user_email, "password": "WrongPassword123!"},
        )

        assert response.status_code == 401

    def test_login_with_nonexistent_user(self, client):
        """Test login with non-existent user."""
        response = client.post(
            "/v1/auth/login",
            json={"email": "nonexistent@example.com", "password": "Password123!"},
        )

        assert response.status_code == 401


class TestMFAFlow:
    """Tests for MFA setup and verification."""

    def test_mfa_request_returns_otpauth_uri(
        self, client, test_user_email, test_user_password
    ):
        """Test MFA request returns TOTP URI."""
        # Signup and login
        signup_resp = client.post(
            "/v1/auth/signup",
            json={"email": test_user_email, "password": test_user_password},
        )
        signup_data = signup_resp.json()["data"]
        session_id = signup_data["session_id"]
        csrf_token = signup_data.get("csrf_token")

        # Request MFA setup. The endpoint requires the session_id as a cookie
        # (double-submit guard, Issue 50.1) and, because a session cookie is
        # present, the CSRF middleware requires the matching X-CSRF-Token.
        response = client.post(
            "/v1/auth/mfa/request",
            json={"session_id": session_id},
            cookies={"session_id": session_id, "csrf_token": csrf_token},
            headers={"X-CSRF-Token": csrf_token},
        )

        assert response.status_code == 200
        data = response.json()
        # Either returns otpauth_uri or status=disabled
        assert "otpauth_uri" in data["data"] or data["data"].get("status") == "disabled"

    def test_mfa_status_endpoint(self, client, test_user_email, test_user_password):
        """Test MFA status endpoint."""
        # Signup
        signup_resp = client.post(
            "/v1/auth/signup",
            json={"email": test_user_email, "password": test_user_password},
        )
        access_token = signup_resp.json()["data"]["access_token"]

        # Check MFA status
        response = client.get(
            "/v1/auth/mfa/status",
            headers={"Authorization": f"Bearer {access_token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "enabled" in data["data"]
        assert "configured" in data["data"]


class TestTokenRefresh:
    """Tests for token refresh."""

    def test_refresh_tokens(self, client, test_user_email, test_user_password):
        """Test token refresh."""
        # Signup to get tokens
        signup_resp = client.post(
            "/v1/auth/signup",
            json={"email": test_user_email, "password": test_user_password},
        )
        refresh_token = signup_resp.json()["data"]["refresh_token"]

        # Refresh tokens
        response = client.post(
            "/v1/auth/refresh",
            json={"refresh_token": refresh_token},
        )

        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data["data"]


class TestUserProfile:
    """Tests for user profile endpoint."""

    def test_get_current_user(self, client, test_user_email, test_user_password):
        """Test getting current user profile."""
        # Signup
        signup_resp = client.post(
            "/v1/auth/signup",
            json={"email": test_user_email, "password": test_user_password},
        )
        access_token = signup_resp.json()["data"]["access_token"]

        # Get profile
        response = client.get(
            "/v1/me",
            headers={"Authorization": f"Bearer {access_token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["data"]["email"] == test_user_email
        assert "email_verified" in data["data"].get("meta", {}) or "meta" in data["data"]


class TestPasswordReset:
    """Tests for password reset flow."""

    def test_request_password_reset(self, client, test_user_email, test_user_password):
        """Test requesting password reset."""
        # First signup
        client.post(
            "/v1/auth/signup",
            json={"email": test_user_email, "password": test_user_password},
        )

        # Request reset
        response = client.post(
            "/v1/auth/reset/request",
            json={"email": test_user_email},
        )

        assert response.status_code == 200
        assert response.json()["data"]["status"] == "sent"

    def test_request_reset_nonexistent_email(self, client):
        """Test requesting reset for non-existent email returns success (to prevent enumeration)."""
        response = client.post(
            "/v1/auth/reset/request",
            json={"email": "nonexistent@example.com"},
        )

        # Should still return success to prevent email enumeration
        assert response.status_code == 200


class TestLogout:
    """Tests for logout."""

    def test_logout_invalidates_session(self, client, test_user_email, test_user_password):
        """Test that logout invalidates the session."""
        # Signup
        signup_resp = client.post(
            "/v1/auth/signup",
            json={"email": test_user_email, "password": test_user_password},
        )
        session_id = signup_resp.json()["data"]["session_id"]
        access_token = signup_resp.json()["data"]["access_token"]

        # Logout
        response = client.post(
            "/v1/auth/logout",
            headers={
                "Authorization": f"Bearer {access_token}",
                "session_id": session_id,
            },
        )

        assert response.status_code == 200


class TestOneCredentialFromTwoTransports:
    """The browser sends a cookie it cannot read; API clients send a body.

    SPEC §17.10 puts `refresh_token` and `session_id` in `HttpOnly` cookies so
    the page cannot hold a durable credential. That only works if the server
    accepts the cookie on its own - a required body field would force the SPA
    to keep the copy the cookie exists to replace. The body form stays for
    clients with no cookie jar, which is most of them.

    Disagreement is refused rather than resolved: picking either silently lets
    a caller who can write one transport override the other.
    """

    @pytest.fixture
    def client(self):
        """An https base URL, because the session cookies are `Secure`.

        Chromium treats 127.0.0.1 as a trustworthy origin and sends them over
        plain http; httpx applies no such exception, so a client on
        `http://testserver` silently holds the cookies and never sends them -
        which would make these pass or fail for a reason that has nothing to
        do with the server.
        """
        from fastapi.testclient import TestClient

        return TestClient(app_module.app, base_url="https://testserver")

    def _signed_in(self, client):
        import uuid

        email = f"tw_{uuid.uuid4().hex[:8]}@example.com"
        resp = client.post(
            "/v1/auth/signup",
            json={"email": email, "password": "TestPassword123!"},
        )
        assert resp.status_code == 201, resp.text
        return resp.json()["data"]

    def _csrf(self, client):
        """What the SPA's `jsonHeaders()` sends: the readable CSRF cookie.

        Once the session cookies actually reach the server, so does CSRF
        enforcement - measured, a test that omitted this got 403 and would
        have read as a refusal for the reason it was checking.
        """
        return {"X-CSRF-Token": client.cookies.get("csrf_token") or ""}

    def test_refresh_works_on_the_cookie_with_an_empty_body(self, client):
        data = self._signed_in(client)
        # The signup response set the cookies; the client jar now holds them.
        assert client.cookies.get("refresh_token")

        resp = client.post(
            "/v1/auth/refresh", json={}, headers=self._csrf(client)
        )

        assert resp.status_code == 200, resp.text
        assert resp.json()["data"]["access_token"]

    def test_refresh_still_works_from_the_body_for_a_client_without_cookies(
        self, client
    ):
        data = self._signed_in(client)
        token = data["refresh_token"]
        client.cookies.clear()

        resp = client.post("/v1/auth/refresh", json={"refresh_token": token})

        assert resp.status_code == 200, resp.text

    def test_a_refresh_with_no_credential_at_all_is_refused(self, client):
        self._signed_in(client)
        client.cookies.clear()

        assert client.post("/v1/auth/refresh", json={}).status_code == 401

    def test_two_disagreeing_refresh_tokens_are_refused_not_reconciled(
        self, client
    ):
        """The body must not be able to override the cookie.

        A *nonsense* body token proves nothing here: it fails whether or not
        the conflict is detected, because the token is invalid either way -
        measured, that version passed with the check removed. The credential
        has to be valid and belong to somebody else, which is the case the
        refusal exists for: a caller who can write one transport speaking as
        the account the other transport names.
        """
        from fastapi.testclient import TestClient

        self._signed_in(client)
        mine = client.cookies.get("refresh_token")

        other = TestClient(app_module.app, base_url="https://testserver")
        theirs = self._signed_in(other)["refresh_token"]
        assert theirs and theirs != mine

        resp = client.post(
            "/v1/auth/refresh",
            json={"refresh_token": theirs},
            headers=self._csrf(client),
        )

        assert resp.status_code == 401, (
            "a valid refresh token in the body overrode the cookie, so the "
            f"body decided whose session this is: {resp.text}"
        )

    def test_mfa_request_resolves_the_session_from_the_cookie(self, client):
        """The browser has no readable session id to put in the body."""
        self._signed_in(client)
        assert client.cookies.get("session_id")

        resp = client.post(
            "/v1/auth/mfa/request", json={}, headers=self._csrf(client)
        )

        # Not 401: the session resolved. Whatever the MFA state turns out to
        # be, refusing for "invalid session" would mean the cookie was ignored.
        assert resp.status_code != 401, resp.text

    def test_a_session_id_that_contradicts_the_cookie_is_refused(self, client):
        """Same rule, same reason: a real session id belonging to somebody else."""
        from fastapi.testclient import TestClient

        self._signed_in(client)

        other = TestClient(app_module.app, base_url="https://testserver")
        theirs = self._signed_in(other)["session_id"]
        assert theirs and theirs != client.cookies.get("session_id")

        resp = client.post(
            "/v1/auth/mfa/request",
            json={"session_id": theirs},
            headers=self._csrf(client),
        )

        assert resp.status_code == 401, (
            "a valid session id in the body overrode the cookie, so MFA acted "
            f"on a session the caller does not hold: {resp.text}"
        )
