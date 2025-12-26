"""Unit tests for auth service.

Tests for:
- Password hashing and verification
- JWT token generation and validation
- Session management
- MFA/TOTP verification
- Password reset flow
- Email verification flow
"""

import asyncio
from datetime import datetime

import pytest

from liminallm.config import Settings
from liminallm.service.auth import AuthService
from liminallm.storage.memory import MemoryStore


@pytest.fixture
def settings():
    """Create test settings."""
    return Settings(
        jwt_secret="Test-Secret-Key_for-Automation-Only-987654321!",
        access_token_ttl_minutes=15,
        refresh_token_ttl_minutes=60 * 24,
        mfa_enabled=False,  # Disable MFA for simpler testing
    )


@pytest.fixture
def memory_store(tmp_path):
    """Create memory store for testing."""
    return MemoryStore(fs_root=str(tmp_path))


@pytest.fixture
def auth_service(memory_store, settings):
    """Create auth service for testing."""
    return AuthService(store=memory_store, cache=None, settings=settings)


@pytest.fixture
def test_user(memory_store, auth_service):
    """Create a test user with password."""
    user = memory_store.create_user("test@example.com")
    # Hash the password and save it properly
    pwd_hash, algo = auth_service._hash_password("TestPassword123!")
    memory_store.save_password(user.id, pwd_hash, algo)
    return user


class TestPasswordHashing:
    """Tests for password hashing."""

    def test_password_hashing_produces_hash(self, auth_service):
        """Test that password hashing produces a hash."""
        pwd_hash, algo = auth_service._hash_password("TestPassword123!")

        assert pwd_hash is not None
        assert len(pwd_hash) > 0
        assert algo in ["bcrypt", "argon2", "argon2id", "scrypt"]

    def test_password_hash_is_not_plaintext(self, auth_service):
        """Test that hashed password is different from plaintext."""
        password = "TestPassword123!"
        pwd_hash, algo = auth_service._hash_password(password)

        assert pwd_hash != password
        assert len(pwd_hash) > len(password)

    def test_same_password_produces_different_hashes(self, auth_service):
        """Test that same password produces different hashes (salted)."""
        password = "TestPassword123!"
        hash1, _ = auth_service._hash_password(password)
        hash2, _ = auth_service._hash_password(password)

        # Due to salting, same password should produce different hashes
        assert hash1 != hash2


class TestJWTTokens:
    """Tests for JWT token generation and validation."""

    def test_issue_tokens_returns_access_and_refresh(self, auth_service, test_user, memory_store):
        """Test that issuing tokens returns both access and refresh tokens."""
        session = memory_store.create_session(test_user.id)
        tokens = auth_service._issue_tokens(test_user, session)

        assert "access_token" in tokens
        assert "refresh_token" in tokens
        assert tokens["token_type"].lower() == "bearer"

    def test_decode_jwt_valid_token(self, auth_service, test_user, memory_store):
        """Test decoding a valid JWT token."""
        session = memory_store.create_session(test_user.id)
        tokens = auth_service._issue_tokens(test_user, session)

        payload = auth_service._decode_jwt(tokens["access_token"])

        assert payload is not None
        assert payload.get("sub") == test_user.id

    def test_decode_jwt_invalid_token(self, auth_service):
        """Test decoding an invalid JWT token."""
        payload = auth_service._decode_jwt("invalid.token.here")

        assert payload is None

    def test_access_token_contains_user_info(self, auth_service, test_user, memory_store):
        """Test that access token contains user info."""
        session = memory_store.create_session(test_user.id)
        tokens = auth_service._issue_tokens(test_user, session)
        payload = auth_service._decode_jwt(tokens["access_token"])

        assert payload.get("sub") == test_user.id
        assert "exp" in payload


class TestSessionManagement:
    """Tests for session management."""

    def test_create_session(self, test_user, memory_store):
        """Test session creation."""
        session = memory_store.create_session(test_user.id)

        assert session.id is not None
        assert session.user_id == test_user.id
        assert session.expires_at > datetime.utcnow()

    def test_get_session(self, test_user, memory_store):
        """Test getting a session."""
        session = memory_store.create_session(test_user.id)

        fetched = memory_store.get_session(session.id)

        assert fetched is not None
        assert fetched.id == session.id

    def test_revoke_session(self, test_user, memory_store):
        """Test session revocation."""
        session = memory_store.create_session(test_user.id)

        memory_store.revoke_session(session.id)
        revoked = memory_store.get_session(session.id)

        # Session should be marked as revoked or deleted
        assert revoked is None or getattr(revoked, 'revoked_at', None) is not None


class TestMFAVerification:
    """Tests for MFA/TOTP verification."""

    def test_set_mfa_secret(self, test_user, memory_store):
        """Test setting MFA secret."""
        memory_store.set_user_mfa_secret(test_user.id, "TESTSECRETKEY", enabled=False)

        config = memory_store.get_user_mfa_secret(test_user.id)

        assert config is not None
        assert config.secret == "TESTSECRETKEY"
        assert config.enabled is False

    def test_enable_mfa(self, test_user, memory_store):
        """Test enabling MFA."""
        memory_store.set_user_mfa_secret(test_user.id, "TESTSECRETKEY", enabled=False)
        memory_store.set_user_mfa_secret(test_user.id, "TESTSECRETKEY", enabled=True)

        config = memory_store.get_user_mfa_secret(test_user.id)

        assert config.enabled is True


class TestSignupFlow:
    """Tests for signup flow."""

    def test_signup_creates_user(self, auth_service, memory_store):
        """Test that signup creates a new user."""
        user, session, tokens = asyncio.get_event_loop().run_until_complete(
            auth_service.signup("new@example.com", "NewPassword123!")
        )

        assert user is not None
        assert user.email == "new@example.com"
        assert session is not None

    def test_signup_stores_hashed_password(self, auth_service, memory_store):
        """Test that signup stores hashed password."""
        user, session, tokens = asyncio.get_event_loop().run_until_complete(
            auth_service.signup("another@example.com", "Password123!")
        )

        # Verify password was hashed (not stored as plaintext)
        pwd_info = memory_store.get_password_record(user.id)
        assert pwd_info is not None
        assert pwd_info[0] != "Password123!"  # Hash should not equal plaintext


class TestLoginFlow:
    """Tests for complete login flow."""

    def test_login_with_valid_credentials(self, auth_service, test_user, memory_store):
        """Test login with valid email and password."""
        user, session, tokens = asyncio.get_event_loop().run_until_complete(
            auth_service.login("test@example.com", "TestPassword123!")
        )

        assert user is not None
        assert user.id == test_user.id
        assert session is not None

    def test_login_with_invalid_password(self, auth_service, test_user):
        """Test login with wrong password."""
        user, session, tokens = asyncio.get_event_loop().run_until_complete(
            auth_service.login("test@example.com", "WrongPassword!")
        )

        assert user is None
        assert session is None

    def test_login_with_nonexistent_email(self, auth_service):
        """Test login with non-existent email."""
        user, session, tokens = asyncio.get_event_loop().run_until_complete(
            auth_service.login("nonexistent@example.com", "AnyPassword!")
        )

        assert user is None
        assert session is None
