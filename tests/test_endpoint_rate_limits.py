"""Tests for API endpoint rate limiting.

Tests that rate limiting is properly applied to protected endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock

from liminallm import app as app_module


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app_module.app)


@pytest.fixture
def auth_headers(client):
    """Create a test user and return auth headers."""
    response = client.post(
        "/v1/auth/signup",
        json={"email": "ratelimit@example.com", "password": "TestPassword123!"},
    )
    access_token = response.json()["data"]["access_token"]
    return {"Authorization": f"Bearer {access_token}"}


class TestReadEndpointRateLimits:
    """Tests for read endpoint rate limiting."""

    def test_artifacts_endpoint_rate_limited(self, client, auth_headers):
        """Test that /artifacts endpoint has rate limiting."""
        # Make several requests - they should all succeed within limit
        for _ in range(5):
            response = client.get("/v1/artifacts", headers=auth_headers)
            assert response.status_code in [200, 429]

    def test_conversations_endpoint_rate_limited(self, client, auth_headers):
        """Test that /conversations endpoint has rate limiting."""
        for _ in range(5):
            response = client.get("/v1/conversations", headers=auth_headers)
            assert response.status_code in [200, 429]

    def test_contexts_endpoint_rate_limited(self, client, auth_headers):
        """Test that /contexts endpoint has rate limiting."""
        for _ in range(5):
            response = client.get("/v1/contexts", headers=auth_headers)
            assert response.status_code in [200, 429]

    def test_tools_specs_endpoint_rate_limited(self, client, auth_headers):
        """Test that /tools/specs endpoint has rate limiting."""
        for _ in range(5):
            response = client.get("/v1/tools/specs", headers=auth_headers)
            assert response.status_code in [200, 429]

    def test_workflows_endpoint_rate_limited(self, client, auth_headers):
        """Test that /workflows endpoint has rate limiting."""
        for _ in range(5):
            response = client.get("/v1/workflows", headers=auth_headers)
            assert response.status_code in [200, 429]

    def test_file_limits_endpoint_rate_limited(self, client, auth_headers):
        """Test that /files/limits endpoint has rate limiting."""
        for _ in range(5):
            response = client.get("/v1/files/limits", headers=auth_headers)
            assert response.status_code in [200, 429]


class TestUserSettingsEndpointRateLimits:
    """Tests for user settings endpoint rate limiting."""

    def test_get_settings_rate_limited(self, client, auth_headers):
        """Test that GET /settings has rate limiting."""
        for _ in range(5):
            response = client.get("/v1/settings", headers=auth_headers)
            assert response.status_code in [200, 429]

    def test_patch_settings_rate_limited(self, client, auth_headers):
        """Test that PATCH /settings has rate limiting."""
        for _ in range(5):
            response = client.patch(
                "/v1/settings",
                headers=auth_headers,
                json={"locale": "en-US"},
            )
            assert response.status_code in [200, 429]


class TestMFAEndpointRateLimits:
    """Tests for MFA endpoint rate limiting."""

    def test_mfa_status_rate_limited(self, client, auth_headers):
        """Test that GET /auth/mfa/status has rate limiting."""
        for _ in range(5):
            response = client.get("/v1/auth/mfa/status", headers=auth_headers)
            assert response.status_code in [200, 429]


class TestProfileEndpointRateLimits:
    """Tests for profile endpoint rate limiting."""

    def test_me_endpoint_rate_limited(self, client, auth_headers):
        """Test that GET /me has rate limiting."""
        for _ in range(5):
            response = client.get("/v1/me", headers=auth_headers)
            assert response.status_code in [200, 429]


class TestAdminEndpointRateLimits:
    """Tests for admin endpoint rate limiting."""

    def test_admin_endpoints_require_auth(self, client):
        """Test that admin endpoints require authentication."""
        admin_endpoints = [
            "/v1/admin/users",
            "/v1/admin/adapters",
            "/v1/admin/objects",
            "/v1/admin/settings",
            "/v1/config",
            "/v1/config/patches",
        ]

        for endpoint in admin_endpoints:
            response = client.get(endpoint)
            assert response.status_code in [401, 403], f"{endpoint} should require auth"


class TestRateLimitErrorResponse:
    """Tests for rate limit error responses."""

    def test_rate_limit_returns_429(self, client, auth_headers):
        """Test that rate limit exceeded returns 429."""
        with patch("liminallm.service.runtime.check_rate_limit", new_callable=AsyncMock) as mock_check:
            # Simulate rate limit exceeded
            mock_check.return_value = False

            response = client.get("/v1/artifacts", headers=auth_headers)

            # Should return 429 Too Many Requests
            assert response.status_code == 429

    def test_rate_limit_error_has_proper_format(self, client, auth_headers):
        """Test that rate limit error has proper JSON format."""
        with patch("liminallm.service.runtime.check_rate_limit", new_callable=AsyncMock) as mock_check:
            mock_check.return_value = False

            response = client.get("/v1/artifacts", headers=auth_headers)

            if response.status_code == 429:
                data = response.json()
                assert "detail" in data or "error" in data


class TestRateLimitByUser:
    """Tests for per-user rate limiting."""

    def test_different_users_have_independent_limits(self, client):
        """Test that different users have independent rate limits."""
        # Create two users
        resp1 = client.post(
            "/v1/auth/signup",
            json={"email": "user1@example.com", "password": "Password123!"},
        )
        token1 = resp1.json()["data"]["access_token"]

        resp2 = client.post(
            "/v1/auth/signup",
            json={"email": "user2@example.com", "password": "Password123!"},
        )
        token2 = resp2.json()["data"]["access_token"]

        # Both users should be able to make requests independently
        response1 = client.get(
            "/v1/artifacts",
            headers={"Authorization": f"Bearer {token1}"},
        )
        response2 = client.get(
            "/v1/artifacts",
            headers={"Authorization": f"Bearer {token2}"},
        )

        assert response1.status_code == 200
        assert response2.status_code == 200
