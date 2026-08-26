"""Additional unit tests for storage methods.

Tests for:
- User settings CRUD
- User CRUD operations
- Conversation operations
- Context operations
- Artifact operations
"""


import pytest


@pytest.fixture
def test_user(store):
    """Create a test user."""
    return store.create_user("test@example.com", handle="testuser")


class TestUserSettings:
    """Tests for user settings storage methods."""

    def test_get_user_settings_defaults_for_new_user(self, store, test_user):
        """New users are seeded with default (empty) settings (Issue 21.4)."""
        settings = store.get_user_settings(test_user.id)
        assert settings is not None
        assert settings.user_id == test_user.id
        assert settings.locale is None
        assert settings.timezone is None
        assert settings.flags is None

    def test_set_user_settings_creates_new(self, store, test_user):
        """set_user_settings should create settings for user."""
        settings = store.set_user_settings(
            test_user.id,
            locale="en-US",
            timezone="America/New_York",
        )

        assert settings.user_id == test_user.id
        assert settings.locale == "en-US"
        assert settings.timezone == "America/New_York"

    def test_get_user_settings_returns_saved(self, store, test_user):
        """get_user_settings should return saved settings."""
        store.set_user_settings(
            test_user.id,
            locale="fr-FR",
            timezone="Europe/Paris",
            default_voice="nova",
        )

        settings = store.get_user_settings(test_user.id)

        assert settings is not None
        assert settings.locale == "fr-FR"
        assert settings.timezone == "Europe/Paris"
        assert settings.default_voice == "nova"

    def test_set_user_settings_updates_existing(self, store, test_user):
        """set_user_settings should update existing settings."""
        store.set_user_settings(test_user.id, locale="en-US")

        # Update with new values
        settings = store.set_user_settings(
            test_user.id,
            locale="de-DE",
            timezone="Europe/Berlin",
        )

        assert settings.locale == "de-DE"
        assert settings.timezone == "Europe/Berlin"

    def test_set_user_settings_with_style_dict(self, store, test_user):
        """set_user_settings should handle style dict."""
        style = {"theme": "dark", "font_size": 14}
        settings = store.set_user_settings(
            test_user.id,
            default_style=style,
        )

        assert settings.default_style == style

    def test_set_user_settings_with_flags_dict(self, store, test_user):
        """set_user_settings should handle flags dict."""
        flags = {"beta_features": True, "experimental": False}
        settings = store.set_user_settings(
            test_user.id,
            flags=flags,
        )

        assert settings.flags == flags


class TestUserOperations:
    """Tests for user CRUD operations."""

    def test_create_user_basic(self, store):
        """Test basic user creation."""
        user = store.create_user("new@example.com")

        assert user.email == "new@example.com"
        assert user.id is not None
        assert user.role == "user"
        assert user.is_active is True

    def test_create_user_with_handle(self, store):
        """Test user creation with handle."""
        user = store.create_user("user@example.com", handle="myhandle")

        assert user.handle == "myhandle"

    def test_create_user_with_role(self, store):
        """Test user creation with admin role."""
        user = store.create_user("admin@example.com", role="admin")

        assert user.role == "admin"

    def test_get_user_by_id(self, store, test_user):
        """Test getting user by ID."""
        fetched = store.get_user(test_user.id)

        assert fetched is not None
        assert fetched.id == test_user.id
        assert fetched.email == test_user.email

    def test_get_user_by_email(self, store, test_user):
        """Test getting user by email."""
        fetched = store.get_user_by_email(test_user.email)

        assert fetched is not None
        assert fetched.id == test_user.id

    def test_get_user_returns_none_for_missing(self, store):
        """Test getting non-existent user."""
        assert store.get_user("nonexistent") is None
        assert store.get_user_by_email("missing@example.com") is None

    def test_update_user_role(self, store, test_user):
        """Test updating user role."""
        updated = store.update_user_role(test_user.id, role="admin")

        assert updated is not None
        assert updated.role == "admin"

    def test_list_users(self, store):
        """Test listing users."""
        store.create_user("user1@example.com")
        store.create_user("user2@example.com")

        users = store.list_users()

        assert len(users) >= 2


class TestConversationOperations:
    """Tests for conversation operations."""

    def test_create_conversation(self, store, test_user):
        """Test conversation creation."""
        conv = store.create_conversation(test_user.id, title="Test Chat")

        assert conv.id is not None
        assert conv.user_id == test_user.id
        assert conv.title == "Test Chat"

    def test_get_conversation(self, store, test_user):
        """Test getting conversation by ID."""
        conv = store.create_conversation(test_user.id, title="Test")

        fetched = store.get_conversation(conv.id)

        assert fetched is not None
        assert fetched.id == conv.id

    def test_list_conversations_for_user(self, store, test_user):
        """Test listing user's conversations."""
        store.create_conversation(test_user.id, title="Conv 1")
        store.create_conversation(test_user.id, title="Conv 2")

        convs = store.list_conversations(test_user.id)

        assert len(convs) >= 2

    def test_append_message(self, store, test_user):
        """Test appending message to conversation."""
        conv = store.create_conversation(test_user.id)

        msg = store.append_message(
            conv.id,
            sender=test_user.id,
            role="user",
            content="Hello!",
        )

        assert msg.id is not None
        assert msg.conversation_id == conv.id
        assert msg.content == "Hello!"
        assert msg.role == "user"

    def test_list_messages(self, store, test_user):
        """Test listing messages in conversation."""
        conv = store.create_conversation(test_user.id)
        store.append_message(conv.id, sender=test_user.id, role="user", content="Hi")
        store.append_message(conv.id, sender="assistant", role="assistant", content="Hello!")

        messages = store.list_messages(conv.id)

        assert len(messages) >= 2


class TestContextOperations:
    """Tests for knowledge context operations."""

    def test_create_context(self, store, test_user):
        """Test context creation."""
        ctx = store.upsert_context(
            owner_user_id=test_user.id,
            name="My Knowledge",
            description="Test context",
        )

        assert ctx.id is not None
        assert ctx.name == "My Knowledge"
        assert ctx.owner_user_id == test_user.id

    def test_get_context(self, store, test_user):
        """Test getting context by ID."""
        ctx = store.upsert_context(
            owner_user_id=test_user.id,
            name="Test",
            description="Test description",
        )

        fetched = store.get_context(ctx.id)

        assert fetched is not None
        assert fetched.id == ctx.id

    def test_list_contexts_for_user(self, store, test_user):
        """Test listing user's contexts."""
        store.upsert_context(owner_user_id=test_user.id, name="Context 1", description="Desc 1")
        store.upsert_context(owner_user_id=test_user.id, name="Context 2", description="Desc 2")

        contexts = store.list_contexts(test_user.id)

        assert len(contexts) >= 2

class TestArtifactOperations:
    """Tests for artifact operations."""

    def test_create_artifact(self, store, test_user):
        """Test artifact creation."""
        artifact = store.create_artifact(
            type_="workflow",
            name="Test Artifact",
            schema={
                "kind": "workflow.chat",
                "nodes": [{"id": "start", "type": "llm_call"}],
            },
            owner_user_id=test_user.id,
        )

        assert artifact.id is not None
        assert artifact.name == "Test Artifact"
        assert artifact.type == "workflow"

    def test_get_artifact(self, store, test_user):
        """Test getting artifact by ID."""
        artifact = store.create_artifact(
            type_="tool",
            name="Test",
            schema={"kind": "tool.spec", "name": "test_tool", "handler": "test.handler"},
            owner_user_id=test_user.id,
        )

        fetched = store.get_artifact(artifact.id)

        assert fetched is not None
        assert fetched.id == artifact.id

    def test_list_artifacts_by_type(self, store, test_user):
        """Test listing artifacts filtered by type."""
        store.create_artifact(
            type_="workflow",
            name="Workflow 1",
            schema={
                "kind": "workflow.chat",
                "nodes": [{"id": "start", "type": "llm_call"}],
            },
            owner_user_id=test_user.id,
        )
        store.create_artifact(
            type_="tool",
            name="Tool 1",
            schema={"kind": "tool.spec", "name": "tool1", "handler": "tool.handler"},
            owner_user_id=test_user.id,
        )

        # Pass owner_user_id: private artifacts are only visible to their owner
        # under the visibility ACL (an owner-less list returns global only).
        workflows = store.list_artifacts(
            type_filter="workflow", owner_user_id=test_user.id
        )
        tools = store.list_artifacts(
            type_filter="tool", owner_user_id=test_user.id
        )

        assert any(a.name == "Workflow 1" for a in workflows)
        assert any(a.name == "Tool 1" for a in tools)

    def test_update_artifact(self, store, test_user):
        """Test updating artifact schema."""
        artifact = store.create_artifact(
            type_="workflow",
            name="Updateable",
            schema={
                "kind": "workflow.chat",
                "nodes": [{"id": "start", "type": "llm_call"}],
                "version": 1,
            },
            owner_user_id=test_user.id,
        )

        new_schema = {
            "kind": "workflow.chat",
            "nodes": [{"id": "start", "type": "llm_call"}],
            "version": 2,
        }
        updated = store.update_artifact(artifact.id, lambda _locked: new_schema)

        assert updated.schema["version"] == 2

    def test_list_artifact_versions(self, store, test_user):
        """Test listing artifact versions."""
        artifact = store.create_artifact(
            type_="workflow",
            name="Versioned",
            schema={
                "kind": "workflow.chat",
                "nodes": [{"id": "start", "type": "llm_call"}],
                "version": 1,
            },
            owner_user_id=test_user.id,
        )
        # The builder ignores the locked schema here because these two are
        # whole-document replacements, which is what a versioning test wants.
        for version in (2, 3):
            store.update_artifact(artifact.id, lambda _locked, v=version: {
                "kind": "workflow.chat",
                "nodes": [{"id": "start", "type": "llm_call"}],
                "version": v,
            })

        versions = store.list_artifact_versions(artifact.id)

        assert len(versions) >= 3


class TestMFAOperations:
    """Tests for MFA-related storage operations."""

    def test_set_and_get_mfa_secret(self, store, test_user):
        """Test setting and getting MFA secret."""
        store.set_user_mfa_secret(test_user.id, "TESTSECRET123", enabled=False)

        config = store.get_user_mfa_secret(test_user.id)

        assert config is not None
        assert config.secret == "TESTSECRET123"
        assert config.enabled is False

    def test_enable_mfa(self, store, test_user):
        """Test enabling MFA."""
        store.set_user_mfa_secret(test_user.id, "TESTSECRET123", enabled=False)
        store.set_user_mfa_secret(test_user.id, "TESTSECRET123", enabled=True)

        config = store.get_user_mfa_secret(test_user.id)

        assert config.enabled is True

    def test_get_mfa_secret_returns_none_for_new_user(self, store, test_user):
        """Test that new users have no MFA config."""
        config = store.get_user_mfa_secret(test_user.id)

        assert config is None
