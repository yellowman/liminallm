"""Integration tests for chat flow.

Tests the complete chat flow including:
- Conversations management
- Message sending
- Contexts/knowledge management
- Artifacts
- Preferences
"""

import pytest
from fastapi.testclient import TestClient

from liminallm import app as app_module


@pytest.fixture
def client():
    """Create a test client for the API."""
    return TestClient(app_module.app)


@pytest.fixture
def auth_headers(client):
    """Create a test user and return auth headers."""
    import uuid
    unique_email = f"chattest_{uuid.uuid4().hex[:8]}@example.com"
    response = client.post(
        "/v1/auth/signup",
        json={"email": unique_email, "password": "TestPassword123!"},
    )
    assert response.status_code == 201, f"Signup failed: {response.text}"
    access_token = response.json()["data"]["access_token"]
    return {"Authorization": f"Bearer {access_token}"}


class TestConversations:
    """Tests for conversation management."""

    def test_create_conversation_via_chat(self, client, auth_headers):
        """Test that sending a message creates a new conversation."""
        response = client.post(
            "/v1/chat",
            headers=auth_headers,
            json={"message": {"content": "Hello, this is a test message"}, "stream": False},
        )

        assert response.status_code == 200
        data = response.json()
        assert "conversation_id" in data["data"]
        assert "message_id" in data["data"]

    def test_list_conversations(self, client, auth_headers):
        """Test listing conversations."""
        # First create a conversation
        client.post(
            "/v1/chat",
            headers=auth_headers,
            json={"message": {"content": "Test message"}, "stream": False},
        )

        # List conversations
        response = client.get("/v1/conversations", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert "items" in data["data"]
        assert len(data["data"]["items"]) >= 1

    def test_get_single_conversation(self, client, auth_headers):
        """Test getting a single conversation."""
        # Create a conversation
        chat_resp = client.post(
            "/v1/chat",
            headers=auth_headers,
            json={"message": {"content": "Test message"}, "stream": False},
        )
        conv_id = chat_resp.json()["data"]["conversation_id"]

        # Get the conversation
        response = client.get(f"/v1/conversations/{conv_id}", headers=auth_headers)

        assert response.status_code == 200
        assert response.json()["data"]["id"] == conv_id


class TestMessages:
    """Tests for message management."""

    def test_list_messages_in_conversation(self, client, auth_headers):
        """Test listing messages in a conversation."""
        # Create a conversation with a message
        chat_resp = client.post(
            "/v1/chat",
            headers=auth_headers,
            json={"message": {"content": "Test message"}, "stream": False},
        )
        conv_id = chat_resp.json()["data"]["conversation_id"]

        # List messages
        response = client.get(
            f"/v1/conversations/{conv_id}/messages",
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert "messages" in data["data"]
        # Should have at least the user message
        assert len(data["data"]["messages"]) >= 1

    def test_continue_conversation(self, client, auth_headers):
        """Test continuing an existing conversation."""
        # Create a conversation
        chat_resp = client.post(
            "/v1/chat",
            headers=auth_headers,
            json={"message": {"content": "First message"}, "stream": False},
        )
        conv_id = chat_resp.json()["data"]["conversation_id"]

        # Send another message to same conversation
        response = client.post(
            "/v1/chat",
            headers=auth_headers,
            json={"message": {"content": "Second message"}, "conversation_id": conv_id, "stream": False},
        )

        assert response.status_code == 200
        assert response.json()["data"]["conversation_id"] == conv_id


class TestKnowledgeContexts:
    """Tests for knowledge context management."""

    def test_create_context(self, client, auth_headers):
        """Test creating a knowledge context."""
        response = client.post(
            "/v1/contexts",
            headers=auth_headers,
            json={"name": "Test Context", "description": "A test knowledge context"},
        )

        assert response.status_code == 201
        data = response.json()
        assert "id" in data["data"]
        assert data["data"]["name"] == "Test Context"

    def test_list_contexts(self, client, auth_headers):
        """Test listing knowledge contexts."""
        # Create a context
        client.post(
            "/v1/contexts",
            headers=auth_headers,
            json={"name": "Test Context", "description": "Test description"},
        )

        # List contexts
        response = client.get("/v1/contexts", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert "items" in data["data"]
        assert len(data["data"]["items"]) >= 1

    def test_create_context_with_text(self, client, auth_headers):
        """Test creating a context with initial text."""
        response = client.post(
            "/v1/contexts",
            headers=auth_headers,
            json={
                "name": "Text Context",
                "description": "Context with text content",
                "text": "This is some sample text that will be chunked and embedded.",
            },
        )

        assert response.status_code == 201

    def test_list_chunks_in_context(self, client, auth_headers):
        """Test listing chunks in a context."""
        # Create context with text
        ctx_resp = client.post(
            "/v1/contexts",
            headers=auth_headers,
            json={
                "name": "Chunk Test Context",
                "description": "Context for chunk testing",
                "text": "This is text for chunk testing.",
            },
        )
        ctx_id = ctx_resp.json()["data"]["id"]

        # List chunks
        response = client.get(
            f"/v1/contexts/{ctx_id}/chunks",
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert "items" in data["data"]


class TestArtifacts:
    """Tests for artifact management."""

    def test_create_artifact(self, client, auth_headers):
        """Test creating an artifact."""
        response = client.post(
            "/v1/artifacts",
            headers=auth_headers,
            json={
                "name": "Test Artifact",
                "type": "workflow",
                "schema": {
                    "kind": "workflow.chat",
                    "nodes": [{"id": "start", "type": "llm_call"}],
                },
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert "id" in data["data"]
        assert data["data"]["name"] == "Test Artifact"

    def test_list_artifacts(self, client, auth_headers):
        """Test listing artifacts."""
        # Create an artifact
        client.post(
            "/v1/artifacts",
            headers=auth_headers,
            json={
                "name": "Test Artifact",
                "type": "workflow",
                "schema": {
                    "kind": "workflow.chat",
                    "nodes": [{"id": "start", "type": "llm_call"}],
                },
            },
        )

        # List artifacts
        response = client.get("/v1/artifacts", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert "items" in data["data"]

    def test_get_artifact(self, client, auth_headers):
        """Test getting a single artifact."""
        # Create artifact
        create_resp = client.post(
            "/v1/artifacts",
            headers=auth_headers,
            json={
                "name": "Test Artifact",
                "type": "tool",
                "schema": {
                    "kind": "tool.spec",
                    "name": "test_tool",
                    "handler": "test.handler",
                },
            },
        )
        artifact_id = create_resp.json()["data"]["id"]

        # Get artifact
        response = client.get(f"/v1/artifacts/{artifact_id}", headers=auth_headers)

        assert response.status_code == 200
        assert response.json()["data"]["id"] == artifact_id

    def test_list_artifact_versions(self, client, auth_headers):
        """Test listing artifact versions."""
        # Create artifact
        create_resp = client.post(
            "/v1/artifacts",
            headers=auth_headers,
            json={
                "name": "Versioned Artifact",
                "type": "workflow",
                "schema": {
                    "kind": "workflow.chat",
                    "nodes": [{"id": "start", "type": "llm_call"}],
                },
            },
        )
        artifact_id = create_resp.json()["data"]["id"]

        # List versions
        response = client.get(
            f"/v1/artifacts/{artifact_id}/versions",
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert "items" in data["data"]


class TestToolSpecs:
    """Tests for tool spec management."""

    def test_list_tool_specs(self, client, auth_headers):
        """Test listing tool specs."""
        # Create a tool spec
        client.post(
            "/v1/artifacts",
            headers=auth_headers,
            json={
                "name": "Test Tool",
                "type": "tool",
                "schema": {
                    "kind": "tool.spec",
                    "name": "test_tool",
                    "handler": "test.handler",
                },
            },
        )

        # List tool specs
        response = client.get("/v1/tools/specs", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert "items" in data["data"]


class TestWorkflows:
    """Tests for workflow management."""

    def test_list_workflows(self, client, auth_headers):
        """Test listing workflows."""
        # Create a workflow
        client.post(
            "/v1/artifacts",
            headers=auth_headers,
            json={
                "name": "Test Workflow",
                "type": "workflow",
                "schema": {
                    "kind": "workflow.chat",
                    "nodes": [{"id": "start", "type": "llm_call"}],
                },
            },
        )

        # List workflows
        response = client.get("/v1/workflows", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert "items" in data["data"]


class TestPreferences:
    """Tests for preference events."""

    def test_record_preference(self, client, auth_headers):
        """Test recording a preference event."""
        # First create a conversation
        chat_resp = client.post(
            "/v1/chat",
            headers=auth_headers,
            json={"message": {"content": "Test message"}, "stream": False},
        )
        data = chat_resp.json()["data"]
        conv_id = data["conversation_id"]
        msg_id = data["message_id"]

        # Record preference
        response = client.post(
            "/v1/preferences",
            headers=auth_headers,
            json={
                "conversation_id": conv_id,
                "message_id": msg_id,
                "feedback": "positive",
            },
        )

        assert response.status_code in [200, 201]

    def test_get_preference_insights(self, client, auth_headers):
        """Test getting preference insights."""
        response = client.get("/v1/preferences/insights", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        # Should have summary stats
        assert "data" in data


class TestFileUpload:
    """Tests for file upload."""

    def test_get_file_limits(self, client, auth_headers):
        """Test getting file upload limits."""
        response = client.get("/v1/files/limits", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert "max_upload_bytes" in data["data"]


class TestHealth:
    """Tests for health check."""

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/healthz")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "ok"]


class TestChatCancel:
    """Tests for chat cancel endpoint per SPEC §18."""

    def test_cancel_nonexistent_request(self, client, auth_headers):
        """Test cancelling a request that doesn't exist."""
        response = client.post(
            "/v1/chat/cancel",
            headers=auth_headers,
            json={"request_id": "nonexistent-request-id-12345"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["data"]["cancelled"] is False
        assert "not found" in data["data"]["message"].lower()

    def test_cancel_requires_auth(self, client):
        """Test that cancel endpoint requires authentication."""
        response = client.post(
            "/v1/chat/cancel",
            json={"request_id": "some-request-id"},
        )

        assert response.status_code in [401, 403]

    def test_cancel_request_id_required(self, client, auth_headers):
        """Test that request_id is required."""
        response = client.post(
            "/v1/chat/cancel",
            headers=auth_headers,
            json={},
        )

        assert response.status_code == 422  # Validation error

    def test_cancel_response_structure(self, client, auth_headers):
        """Test the cancel response follows SPEC §18 envelope format."""
        response = client.post(
            "/v1/chat/cancel",
            headers=auth_headers,
            json={"request_id": "test-request-id"},
        )

        assert response.status_code == 200
        data = response.json()
        # SPEC §18: Envelope with status, data
        assert "status" in data
        assert "data" in data
        assert "request_id" in data["data"]
        assert "cancelled" in data["data"]
        assert "message" in data["data"]
