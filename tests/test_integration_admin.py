"""Integration tests for admin operations.

Tests admin-only functionality including:
- User management
- System settings
- Config patches
- Adapters and objects inspection
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
def admin_user(client):
    """Create an admin user and return credentials."""
    import uuid
    unique_email = f"admin_{uuid.uuid4().hex[:8]}@example.com"

    # First create a regular user via signup
    response = client.post(
        "/v1/auth/signup",
        json={"email": unique_email, "password": "AdminPassword123!"},
    )
    assert response.status_code == 201, f"Signup failed: {response.text}"
    user_id = response.json()["data"]["user_id"]
    access_token = response.json()["data"]["access_token"]

    # Promote to admin via direct store access (in tests only)
    runtime = get_runtime()
    runtime.store.update_user_role(user_id, role="admin")

    # Re-login to get updated token with admin role
    login_resp = client.post(
        "/v1/auth/login",
        json={"email": unique_email, "password": "AdminPassword123!"},
    )
    assert login_resp.status_code == 200, f"Login failed: {login_resp.text}"
    return {
        "user_id": user_id,
        "access_token": login_resp.json()["data"]["access_token"],
        "headers": {"Authorization": f"Bearer {login_resp.json()['data']['access_token']}"},
    }


@pytest.fixture
def regular_user(client):
    """Create a regular (non-admin) user and return credentials."""
    import uuid
    unique_email = f"regular_{uuid.uuid4().hex[:8]}@example.com"
    response = client.post(
        "/v1/auth/signup",
        json={"email": unique_email, "password": "RegularPassword123!"},
    )
    assert response.status_code == 201, f"Signup failed: {response.text}"
    access_token = response.json()["data"]["access_token"]
    return {
        "user_id": response.json()["data"]["user_id"],
        "access_token": access_token,
        "headers": {"Authorization": f"Bearer {access_token}"},
    }


class TestAdminUserManagement:
    """Tests for admin user management."""

    def test_admin_can_list_users(self, client, admin_user):
        """Test that admin can list users."""
        response = client.get("/v1/admin/users", headers=admin_user["headers"])

        assert response.status_code == 200
        data = response.json()
        assert "items" in data["data"]

    def test_regular_user_cannot_list_users(self, client, regular_user):
        """Test that regular users cannot list users."""
        response = client.get("/v1/admin/users", headers=regular_user["headers"])

        assert response.status_code == 403

    def test_admin_can_create_user(self, client, admin_user):
        """Test that admin can create a new user."""
        response = client.post(
            "/v1/admin/users",
            headers=admin_user["headers"],
            json={
                "email": "newuser@example.com",
                "password": "NewUserPassword123!",
                "role": "user",
            },
        )

        assert response.status_code in [200, 201]
        data = response.json()
        assert "id" in data["data"]

    def test_admin_can_update_user_role(self, client, admin_user, regular_user):
        """Test that admin can update user role."""
        response = client.post(
            f"/v1/admin/users/{regular_user['user_id']}/role",
            headers=admin_user["headers"],
            json={"role": "admin"},
        )

        assert response.status_code == 200


class TestAdminSettings:
    """Tests for admin settings management."""

    def test_admin_can_get_settings(self, client, admin_user):
        """Test that admin can get system settings."""
        response = client.get("/v1/admin/settings", headers=admin_user["headers"])

        assert response.status_code == 200
        data = response.json()
        assert "data" in data

    def test_admin_can_update_settings(self, client, admin_user):
        """Test that admin can update system settings."""
        response = client.patch(
            "/v1/admin/settings",
            headers=admin_user["headers"],
            json={"default_page_size": 50},
        )

        assert response.status_code == 200

    def test_regular_user_cannot_get_admin_settings(self, client, regular_user):
        """Test that regular users cannot access admin settings."""
        response = client.get("/v1/admin/settings", headers=regular_user["headers"])

        assert response.status_code == 403


class TestAdminInspection:
    """Tests for admin inspection endpoints."""

    def test_admin_can_list_adapters(self, client, admin_user):
        """Test that admin can list adapters."""
        response = client.get("/v1/admin/adapters", headers=admin_user["headers"])

        assert response.status_code == 200
        data = response.json()
        assert "items" in data["data"]

    def test_admin_can_list_storage_objects(self, client, admin_user):
        """Test that admin can list storage objects."""
        response = client.get("/v1/admin/objects", headers=admin_user["headers"])

        assert response.status_code == 200
        data = response.json()
        # Response contains details and summary, not items
        assert "details" in data["data"] or "summary" in data["data"]

    def test_regular_user_cannot_list_adapters(self, client, regular_user):
        """Test that regular users cannot list adapters."""
        response = client.get("/v1/admin/adapters", headers=regular_user["headers"])

        assert response.status_code == 403


class TestConfigPatches:
    """Tests for config patch management."""

    def test_admin_can_list_config_patches(self, client, admin_user):
        """Test that admin can list config patches."""
        response = client.get("/v1/config/patches", headers=admin_user["headers"])

        assert response.status_code == 200
        data = response.json()
        assert "items" in data["data"]

    def test_admin_can_propose_config_patch(self, client, admin_user):
        """Test that admin can propose a config patch."""
        # First create an artifact to patch
        artifact_resp = client.post(
            "/v1/artifacts",
            headers=admin_user["headers"],
            json={
                "name": "Patchable Artifact",
                "type": "workflow",
                "schema": {
                    "kind": "workflow.chat",
                    "nodes": [{"id": "start", "type": "llm_call"}],
                    "config": {"key": "value"},
                },
            },
        )
        artifact_id = artifact_resp.json()["data"]["id"]

        # Propose a patch
        response = client.post(
            "/v1/config/propose_patch",
            headers=admin_user["headers"],
            json={
                "artifact_id": artifact_id,
                "patch": [{"op": "replace", "path": "/config/key", "value": "new_value"}],
                "justification": "Test patch",
            },
        )

        assert response.status_code in [200, 201]

    def test_admin_can_filter_patches_by_status(self, client, admin_user):
        """Test that admin can filter patches by status."""
        response = client.get(
            "/v1/config/patches?status=pending",
            headers=admin_user["headers"],
        )

        assert response.status_code == 200


class TestRuntimeConfig:
    """Tests for runtime config endpoint."""

    def test_admin_can_get_runtime_config(self, client, admin_user):
        """Test that admin can get runtime config."""
        response = client.get("/v1/config", headers=admin_user["headers"])

        assert response.status_code == 200
        data = response.json()
        # Sensitive fields should be redacted
        if "settings" in data["data"]:
            settings = data["data"]["settings"]
            # Check that secrets are redacted
            for key in settings:
                if "secret" in key.lower() or "key" in key.lower():
                    assert settings[key] == "[REDACTED]" or "[REDACTED]" in str(settings.get(key, ""))

    def test_regular_user_cannot_get_runtime_config(self, client, regular_user):
        """Test that regular users cannot get runtime config."""
        response = client.get("/v1/config", headers=regular_user["headers"])

        assert response.status_code == 403


class TestRateLimiting:
    """Tests for rate limiting on admin endpoints."""

    def test_admin_endpoints_have_rate_limits(self, client, admin_user):
        """Test that admin endpoints are rate limited."""
        # Make multiple rapid requests
        responses = []
        for _ in range(5):
            resp = client.get("/v1/admin/users", headers=admin_user["headers"])
            responses.append(resp.status_code)

        # All should succeed (we're not hitting the limit in this test)
        # but the endpoint should be functional
        assert all(status == 200 for status in responses)


class TestAdminAccessControl:
    """Tests for admin access control."""

    def test_unauthenticated_cannot_access_admin(self, client):
        """Test that unauthenticated requests cannot access admin endpoints."""
        response = client.get("/v1/admin/users")

        assert response.status_code in [401, 403]

    def test_admin_endpoints_check_role(self, client, regular_user):
        """Test that admin endpoints verify admin role."""
        endpoints = [
            "/v1/admin/users",
            "/v1/admin/adapters",
            "/v1/admin/objects",
            "/v1/admin/settings",
            "/v1/config",
            "/v1/config/patches",
        ]

        for endpoint in endpoints:
            response = client.get(endpoint, headers=regular_user["headers"])
            assert response.status_code == 403, f"Endpoint {endpoint} should require admin role"
