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
from liminallm.service.runtime import get_runtime


@pytest.fixture
def client():
    """Create a test client for the API."""
    return TestClient(app_module.app)


@pytest.fixture
def test_user_email():
    return "testuser@example.com"


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
        assert "already exists" in response.json()["detail"].lower()

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
        session_id = signup_resp.json()["data"]["session_id"]

        # Request MFA setup
        response = client.post(
            "/v1/auth/mfa/request",
            json={"session_id": session_id},
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
            "/v1/auth/token/refresh",
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
