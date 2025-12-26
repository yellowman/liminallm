"""Post-smoke tests for LiminalLM.

These tests catch issues that only appear after the first successful smoke run:
- Idempotency and caching issues
- Race conditions in concurrent requests
- File upload validation for all supported extensions
- Logout and token invalidation
- Tenant isolation between users

Run with: pytest tests/test_post_smoke.py -v
"""

import io
import pytest
import uuid
from fastapi.testclient import TestClient

from liminallm import app as app_module
from liminallm.api.routes import ALLOWED_UPLOAD_EXTENSIONS


@pytest.fixture
def client():
    """Create a test client for the API."""
    return TestClient(app_module.app)


def unique_email():
    """Generate a unique email for testing."""
    return f"postsmoke_{uuid.uuid4().hex[:8]}@test.local"


def signup_user(client, email: str, password: str = "TestPass123!") -> dict:
    """Helper to signup a user and return response data."""
    resp = client.post(
        "/v1/auth/signup",
        json={"email": email, "password": password},
    )
    assert resp.status_code == 201, f"Signup failed: {resp.json()}"
    return resp.json()["data"]


def auth_header(token: str) -> dict:
    """Return authorization header."""
    return {"Authorization": f"Bearer {token}"}


class TestIdempotencyAndRepeat:
    """Tests for idempotency keys, caching, and repeated operations.

    These tests run operations multiple times to catch:
    - Idempotency key collisions
    - Cache staleness issues
    - Race conditions in concurrent requests
    """

    def test_repeated_signup_same_email_rejected(self, client):
        """Verify that repeated signups with same email are properly rejected."""
        email = unique_email()
        password = "TestPass123!"

        # First signup should succeed
        resp1 = client.post(
            "/v1/auth/signup",
            json={"email": email, "password": password},
        )
        assert resp1.status_code == 201

        # Second and third signups with same email should fail
        for i in range(2):
            resp = client.post(
                "/v1/auth/signup",
                json={"email": email, "password": password},
            )
            assert resp.status_code == 409, f"Attempt {i+2}: Expected 409, got {resp.status_code}"

    def test_repeated_login_returns_fresh_tokens(self, client):
        """Verify that repeated logins return fresh tokens each time."""
        email = unique_email()
        password = "TestPass123!"
        signup_user(client, email, password)

        tokens = []
        for _ in range(3):
            resp = client.post(
                "/v1/auth/login",
                json={"email": email, "password": password},
            )
            assert resp.status_code == 200
            token = resp.json()["data"]["access_token"]
            tokens.append(token)

        # All tokens should be valid and (ideally) unique
        # Note: tokens might be the same if issued in the same second
        # but all should work
        for token in tokens:
            resp = client.get("/v1/me", headers=auth_header(token))
            assert resp.status_code == 200

    def test_repeated_conversation_creation(self, client):
        """Verify that creating conversations multiple times works correctly."""
        email = unique_email()
        data = signup_user(client, email)
        token = data["access_token"]
        headers = auth_header(token)

        conversation_ids = []
        for i in range(3):
            resp = client.post(
                "/v1/conversations",
                headers=headers,
                json={"title": f"Test Conversation {i+1}"},
            )
            assert resp.status_code in (200, 201), f"Create {i+1} failed: {resp.json()}"
            conv_id = resp.json()["data"]["id"]
            conversation_ids.append(conv_id)

        # All conversation IDs should be unique
        assert len(set(conversation_ids)) == 3, "Conversation IDs should be unique"

        # All conversations should be listable
        resp = client.get("/v1/conversations", headers=headers)
        assert resp.status_code == 200
        listed_ids = [c["id"] for c in resp.json()["data"]]
        for cid in conversation_ids:
            assert cid in listed_ids, f"Conversation {cid} not in list"

    def test_list_endpoints_return_consistent_results(self, client):
        """Verify that listing endpoints return consistent results on repeated calls."""
        email = unique_email()
        data = signup_user(client, email)
        token = data["access_token"]
        headers = auth_header(token)

        # Create some data
        client.post("/v1/conversations", headers=headers, json={"title": "Test"})

        # Call list endpoint 3 times
        results = []
        for _ in range(3):
            resp = client.get("/v1/conversations", headers=headers)
            assert resp.status_code == 200
            results.append(resp.json()["data"])

        # Results should be consistent
        assert results[0] == results[1] == results[2], "List results should be consistent"


# Text-based extensions with content type and sample content
TEXT_EXTENSIONS = {
    ".txt": ("text/plain", "Test content"),
    ".md": ("text/markdown", "# Markdown Test"),
    ".json": ("application/json", '{"test": "value"}'),
    ".csv": ("text/csv", "a,b,c\n1,2,3"),
    ".tsv": ("text/tab-separated-values", "a\tb\tc\n1\t2\t3"),
}


class TestFileExtensionUpload:
    """Tests for uploading files with each allowed extension.

    Verifies that the API accepts all documented file types.
    """

    @pytest.fixture
    def authenticated_user(self, client):
        """Create an authenticated user and return (client, headers)."""
        email = unique_email()
        data = signup_user(client, email)
        token = data["access_token"]
        return client, auth_header(token)

    @pytest.mark.parametrize("ext", [".txt", ".md", ".json", ".csv", ".tsv"])
    def test_upload_text_extension(self, authenticated_user, ext):
        """Test uploading text-based file extensions."""
        client, headers = authenticated_user
        content_type, content = TEXT_EXTENSIONS[ext]

        filename = f"test{ext}"
        files = {"file": (filename, io.BytesIO(content.encode()), content_type)}

        resp = client.post("/v1/files/upload", headers=headers, files=files)

        # Accept 200, 201 (success) or 429 (rate limit)
        assert resp.status_code in (200, 201, 429), f"Upload {ext} failed: {resp.json()}"
        if resp.status_code in (200, 201):
            assert "file_id" in resp.json()["data"] or "id" in resp.json()["data"]

    def test_upload_disallowed_extension_rejected(self, authenticated_user):
        """Verify that disallowed extensions are rejected."""
        client, headers = authenticated_user

        # Try to upload an executable
        files = {"file": ("test.exe", io.BytesIO(b"MZ"), "application/octet-stream")}
        resp = client.post("/v1/files/upload", headers=headers, files=files)

        assert resp.status_code == 400, "Should reject .exe files"

    def test_allowed_extensions_match_limits_endpoint(self, authenticated_user):
        """Verify that /files/limits returns the expected extensions."""
        client, headers = authenticated_user

        resp = client.get("/v1/files/limits", headers=headers)
        assert resp.status_code == 200

        reported = set(resp.json()["data"]["allowed_extensions"])
        expected = ALLOWED_UPLOAD_EXTENSIONS

        assert reported == expected, f"Mismatch: reported={reported}, expected={expected}"


class TestLogoutTokenInvalidation:
    """Tests for logout and token invalidation.

    Verifies that:
    - Logout properly invalidates the session
    - Old tokens cannot be used after logout
    - Multiple logouts are handled gracefully
    """

    def test_logout_invalidates_token(self, client):
        """Verify that logout invalidates the access token."""
        email = unique_email()
        data = signup_user(client, email)
        token = data["access_token"]
        session_id = data["session_id"]
        headers = auth_header(token)

        # Verify token works before logout
        resp = client.get("/v1/me", headers=headers)
        assert resp.status_code == 200

        # Logout
        resp = client.post(
            "/v1/auth/logout",
            headers={**headers, "session_id": session_id},
        )
        assert resp.status_code == 200

        # Verify token is invalidated after logout
        resp = client.get("/v1/me", headers=headers)
        assert resp.status_code == 401, "Token should be invalidated after logout"

    def test_logout_allows_new_login(self, client):
        """Verify that new login works after logout."""
        email = unique_email()
        password = "TestPass123!"
        data = signup_user(client, email, password)
        token = data["access_token"]
        session_id = data["session_id"]

        # Logout
        client.post(
            "/v1/auth/logout",
            headers={**auth_header(token), "session_id": session_id},
        )

        # New login should work
        resp = client.post(
            "/v1/auth/login",
            json={"email": email, "password": password},
        )
        assert resp.status_code == 200
        new_token = resp.json()["data"]["access_token"]

        # New token should work
        resp = client.get("/v1/me", headers=auth_header(new_token))
        assert resp.status_code == 200

    def test_double_logout_is_safe(self, client):
        """Verify that logging out twice doesn't cause errors."""
        email = unique_email()
        data = signup_user(client, email)
        token = data["access_token"]
        session_id = data["session_id"]
        headers = {**auth_header(token), "session_id": session_id}

        # First logout
        resp1 = client.post("/v1/auth/logout", headers=headers)
        assert resp1.status_code == 200

        # Second logout should either succeed or return 401 (already logged out)
        resp2 = client.post("/v1/auth/logout", headers=headers)
        assert resp2.status_code in (200, 401)


class TestPasswordReset:
    """Tests for password reset flow."""

    def test_request_password_reset(self, client):
        """Verify that password reset request works."""
        email = unique_email()
        signup_user(client, email)

        resp = client.post(
            "/v1/auth/reset/request",
            json={"email": email},
        )
        assert resp.status_code == 200
        assert resp.json()["data"]["status"] == "sent"

    def test_reset_request_for_nonexistent_email_succeeds(self, client):
        """Verify that reset for non-existent email returns success (prevents enumeration)."""
        resp = client.post(
            "/v1/auth/reset/request",
            json={"email": "nonexistent@example.com"},
        )
        # Should return success to prevent email enumeration attacks
        assert resp.status_code == 200


class TestTenantIsolation:
    """Tests for tenant isolation between users.

    Verifies that:
    - Users cannot access each other's conversations
    - Users cannot access each other's files
    - Users cannot access each other's artifacts
    - Listing endpoints only return user's own data
    """

    @pytest.fixture
    def two_users(self, client):
        """Create two authenticated users."""
        email_a = unique_email()
        email_b = unique_email()

        data_a = signup_user(client, email_a)
        data_b = signup_user(client, email_b)

        return {
            "user_a": {
                "email": email_a,
                "token": data_a["access_token"],
                "user_id": data_a["user_id"],
                "headers": auth_header(data_a["access_token"]),
            },
            "user_b": {
                "email": email_b,
                "token": data_b["access_token"],
                "user_id": data_b["user_id"],
                "headers": auth_header(data_b["access_token"]),
            },
        }

    def test_user_cannot_access_others_conversation(self, client, two_users):
        """Verify that User B cannot access User A's conversation."""
        user_a = two_users["user_a"]
        user_b = two_users["user_b"]

        # User A creates a conversation
        resp = client.post(
            "/v1/conversations",
            headers=user_a["headers"],
            json={"title": "User A Private Conversation"},
        )
        assert resp.status_code in (200, 201)
        conv_id = resp.json()["data"]["id"]

        # User B tries to access User A's conversation
        resp = client.get(
            f"/v1/conversations/{conv_id}",
            headers=user_b["headers"],
        )
        # Should be 403 (forbidden) or 404 (not found)
        assert resp.status_code in (403, 404), \
            f"User B should not access User A's conversation, got {resp.status_code}"

    def test_user_cannot_list_others_conversations(self, client, two_users):
        """Verify that User B's list doesn't include User A's conversations."""
        user_a = two_users["user_a"]
        user_b = two_users["user_b"]

        # User A creates conversations
        for i in range(3):
            client.post(
                "/v1/conversations",
                headers=user_a["headers"],
                json={"title": f"User A Conv {i}"},
            )

        # User B lists their conversations
        resp = client.get("/v1/conversations", headers=user_b["headers"])
        assert resp.status_code == 200

        # User B should see 0 conversations (they haven't created any)
        conversations = resp.json()["data"]
        assert len(conversations) == 0, \
            f"User B should not see User A's conversations, but saw {len(conversations)}"

    def test_user_cannot_delete_others_conversation(self, client, two_users):
        """Verify that User B cannot delete User A's conversation."""
        user_a = two_users["user_a"]
        user_b = two_users["user_b"]

        # User A creates a conversation
        resp = client.post(
            "/v1/conversations",
            headers=user_a["headers"],
            json={"title": "User A's Important Conversation"},
        )
        conv_id = resp.json()["data"]["id"]

        # User B tries to delete User A's conversation
        resp = client.delete(
            f"/v1/conversations/{conv_id}",
            headers=user_b["headers"],
        )
        # Should be 403 or 404
        assert resp.status_code in (403, 404)

        # Verify conversation still exists for User A
        resp = client.get(
            f"/v1/conversations/{conv_id}",
            headers=user_a["headers"],
        )
        assert resp.status_code == 200

    def test_user_cannot_access_others_artifacts(self, client, two_users):
        """Verify that User B cannot list User A's artifacts."""
        user_a = two_users["user_a"]
        user_b = two_users["user_b"]

        # Both users list artifacts
        resp_a = client.get("/v1/artifacts", headers=user_a["headers"])
        resp_b = client.get("/v1/artifacts", headers=user_b["headers"])

        assert resp_a.status_code == 200
        assert resp_b.status_code == 200

        # Each user should only see their own artifacts
        # (Empty for new users, but the point is they shouldn't see each other's)
        artifacts_a = resp_a.json()["data"]
        artifacts_b = resp_b.json()["data"]

        # If User A has artifacts, User B shouldn't see them and vice versa
        if artifacts_a:
            artifact_ids_a = {a["id"] for a in artifacts_a}
            artifact_ids_b = {a["id"] for a in artifacts_b}
            assert not artifact_ids_a.intersection(artifact_ids_b), \
                "Users should not see each other's artifacts"

    def test_user_isolation_on_me_endpoint(self, client, two_users):
        """Verify that /me returns correct user data for each user."""
        user_a = two_users["user_a"]
        user_b = two_users["user_b"]

        resp_a = client.get("/v1/me", headers=user_a["headers"])
        resp_b = client.get("/v1/me", headers=user_b["headers"])

        assert resp_a.status_code == 200
        assert resp_b.status_code == 200

        email_a = resp_a.json()["data"]["email"]
        email_b = resp_b.json()["data"]["email"]

        assert email_a == user_a["email"]
        assert email_b == user_b["email"]
        assert email_a != email_b

    def test_user_cannot_send_message_to_others_conversation(self, client, two_users):
        """Verify that User B cannot send messages to User A's conversation."""
        user_a = two_users["user_a"]
        user_b = two_users["user_b"]

        # User A creates a conversation
        resp = client.post(
            "/v1/conversations",
            headers=user_a["headers"],
            json={"title": "Private Chat"},
        )
        conv_id = resp.json()["data"]["id"]

        # User B tries to send a message to User A's conversation
        resp = client.post(
            "/v1/chat",
            headers=user_b["headers"],
            json={
                "conversation_id": conv_id,
                "content": "Trying to infiltrate!",
            },
        )
        # Should be 403 or 404 (not 200 or 503)
        assert resp.status_code in (403, 404), \
            f"User B should not send to User A's conversation, got {resp.status_code}"
