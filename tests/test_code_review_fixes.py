"""Tests for code review fixes - edge cases identified during comprehensive analysis."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from liminallm.api.schemas import Envelope, ErrorBody

# ==============================================================================
# Zero-Weight Adapter Handling Tests
# ==============================================================================


class TestZeroWeightAdapterHandling:
    """Test that weight=0.0 is handled correctly (not treated as falsy)."""

    def test_zero_weight_not_replaced(self):
        """weight=0.0 should NOT be replaced with default 1.0."""
        from liminallm.service.model_backend import ApiAdapterBackend

        backend = ApiAdapterBackend("meta-llama", provider="together")

        # Adapter with explicit weight=0.0 should keep that weight
        adapters = [
            {"id": "disabled", "remote_adapter_id": "lora-disabled", "weight": 0.0},
            {"id": "enabled", "remote_adapter_id": "lora-enabled", "weight": 0.8},
        ]

        # Use _select_best_adapter which relies on weight extraction
        result = backend._select_best_adapter(adapters, max_count=2)

        # Enabled (0.8) should come first, disabled (0.0) second
        assert result[0]["id"] == "enabled"
        assert result[1]["id"] == "disabled"

    def test_zero_weight_in_format_remote_adapters(self):
        """weight=0.0 reaches the gate, unconditionally.

        This was `if extra_body and "adapter_weights" in extra_body:` around
        the assertion, which passes exactly when the weight is missing -
        measured, stopping the backend from sending `adapter_weights` at all
        left this green. Together advertises `gate_weights`, so the key is
        required rather than optional.
        """
        from liminallm.service.model_backend import ApiAdapterBackend

        backend = ApiAdapterBackend("meta-llama", provider="together")

        adapters = [
            {"id": "a1", "remote_adapter_id": "lora-1", "weight": 0.0},
        ]

        _model, extra_body, _applied, _dropped = backend._format_remote_adapters(
            adapters
        )

        assert extra_body and "adapter_weights" in extra_body, extra_body
        weights = extra_body["adapter_weights"]
        assert (0.0 in weights) if isinstance(weights, list) else (weights == 0.0)

    def test_none_weight_defaults_to_one(self):
        """None weight should default to 1.0 (not 0.0)."""
        from liminallm.service.model_backend import ApiAdapterBackend

        backend = ApiAdapterBackend("meta-llama", provider="together")

        adapters = [
            {"id": "no_weight", "remote_adapter_id": "lora-1"},  # No weight field
            {"id": "explicit", "remote_adapter_id": "lora-2", "weight": 0.5},
        ]

        result = backend._select_best_adapter(adapters, max_count=2)

        # no_weight (default 1.0) should come first
        assert result[0]["id"] == "no_weight"


# ==============================================================================
# Token-Based RAG Chunking Tests
# ==============================================================================


class TestTokenBasedRAGChunking:
    """Test token-based chunking with overlap per SPEC §2.5."""

    def test_simple_tokenize(self):
        """_simple_tokenize should split text into word tokens."""
        from liminallm.service.rag import _simple_tokenize

        tokens = _simple_tokenize("Hello, world! How are you?")

        assert "Hello" in tokens
        assert "world" in tokens
        assert "," in tokens
        assert "?" in tokens

    def test_detokenize(self):
        """_detokenize should reconstruct text from tokens."""
        from liminallm.service.rag import _detokenize

        tokens = ["Hello", ",", "world", "!"]
        result = _detokenize(tokens)

        assert "Hello" in result
        assert "world" in result

    def test_chunking_creates_overlap(self):
        """Chunks should have overlapping tokens."""
        from liminallm.service.rag import RAGService
        from tests.harness import get_test_store

        store = get_test_store()
        rag = RAGService(store, default_chunk_size=50)

        # Create a test context
        user = store.create_user("test@test.com", tenant_id="test")
        ctx = store.upsert_context(
            name="test",
            description="test context",
            owner_user_id=user.id,
        )

        # Ingest long text
        long_text = " ".join([f"word{i}" for i in range(200)])
        chunk_count = rag.ingest_text(ctx.id, long_text, chunk_size=50, overlap_tokens=20)

        assert chunk_count > 1

        # Verify chunks were created with metadata
        chunks = store.list_chunks(ctx.id)
        assert len(chunks) > 1

        # Every chunk after the first records the overlap it carries. Not
        # `if chunk.meta`: skipping when the metadata is absent passes
        # exactly when it should fail.
        for i, chunk in enumerate(chunks[1:], start=1):
            assert chunk.meta and "overlap_tokens" in chunk.meta, (i, chunk.meta)

    def test_chunk_metadata_includes_token_info(self):
        """Chunks should have token count metadata."""
        from liminallm.service.rag import RAGService
        from tests.harness import get_test_store

        store = get_test_store()
        rag = RAGService(store, default_chunk_size=100)

        # Create context
        user = store.create_user("test@test.com")
        ctx = store.upsert_context(
            name="test",
            description="test",
            owner_user_id=user.id,
        )

        text = "This is a test text with multiple words for chunking."
        rag.ingest_text(ctx.id, text, chunk_size=100)

        chunks = store.list_chunks(ctx.id)
        assert len(chunks) >= 1

        for chunk in chunks:
            assert chunk.meta, "a chunk with no metadata records neither"
            assert "token_count" in chunk.meta
            assert "embedding_model_id" in chunk.meta


# ==============================================================================
# ErrorBody Code Validation Tests
# ==============================================================================


class TestErrorBodyValidation:
    """Test ErrorBody.code validation per SPEC §18."""

    def test_valid_error_codes(self):
        """Valid SPEC §18 error codes should be accepted."""
        valid_codes = [
            "unauthorized",
            "forbidden",
            "not_found",
            "rate_limited",
            "validation_error",
            "conflict",
            "server_error",
        ]

        for code in valid_codes:
            error = ErrorBody(code=code, message="test")
            assert error.code == code

    def test_invalid_error_code_rejected(self):
        """Invalid error codes should be rejected."""
        with pytest.raises(ValueError) as exc_info:
            ErrorBody(code="invalid_code", message="test")

        assert "Invalid error code" in str(exc_info.value)

    def test_envelope_with_valid_error(self):
        """Envelope should accept valid ErrorBody."""
        error = ErrorBody(code="not_found", message="Resource not found")
        envelope = Envelope(status="error", error=error)

        assert envelope.status == "error"
        assert envelope.error.code == "not_found"


# ==============================================================================
# ChatMessage.mode Validation Tests
# ==============================================================================


class TestChatMessageModeValidation:
    """Test that ChatMessage.mode should be validated."""

    def test_valid_modes(self):
        """Valid modes should be accepted."""
        from liminallm.api.schemas import ChatMessage

        # text mode
        msg_text = ChatMessage(content="hello", mode="text")
        assert msg_text.mode == "text"

        # voice mode
        msg_voice = ChatMessage(content="base64audio", mode="voice")
        assert msg_voice.mode == "voice"

    def test_default_mode_is_text(self):
        """Default mode should be text."""
        from liminallm.api.schemas import ChatMessage

        msg = ChatMessage(content="hello")
        assert msg.mode == "text"


# ==============================================================================
# Refresh Token Revocation Security Tests
# ==============================================================================


class TestRefreshTokenRevocationSecurity:
    """Test that refresh token revocation defaults to safe behavior."""

    @pytest.mark.asyncio
    async def test_cache_failure_defaults_to_revoked(self):
        """When Redis cache fails, should assume token is revoked (safe default)."""
        from unittest.mock import AsyncMock

        from liminallm.config import Settings
        from liminallm.service.auth import AuthService

        # Create auth service with failing cache
        mock_store = MagicMock()
        mock_cache = MagicMock()
        mock_cache.is_refresh_revoked = AsyncMock(side_effect=Exception("Redis unavailable"))
        mock_settings = MagicMock(spec=Settings)
        mock_settings.jwt_secret = "test-secret-key-at-least-32-chars"
        mock_settings.jwt_algorithm = "HS256"

        auth = AuthService(store=mock_store, cache=mock_cache, settings=mock_settings)
        auth.revoked_refresh_tokens = set()  # Empty local set

        # Should return True (revoked) when cache fails
        result = await auth._is_refresh_revoked("test-jti")
        assert result is True  # Safe default: assume revoked

    @pytest.mark.asyncio
    async def test_cache_success_returns_actual_value(self):
        """When cache succeeds, should return actual revocation status."""
        from unittest.mock import AsyncMock

        from liminallm.config import Settings
        from liminallm.service.auth import AuthService

        mock_store = MagicMock()
        mock_cache = MagicMock()
        mock_cache.is_refresh_revoked = AsyncMock(return_value=False)
        mock_settings = MagicMock(spec=Settings)
        mock_settings.jwt_secret = "test-secret-key-at-least-32-chars"
        mock_settings.jwt_algorithm = "HS256"

        auth = AuthService(store=mock_store, cache=mock_cache, settings=mock_settings)
        auth.revoked_refresh_tokens = set()

        result = await auth._is_refresh_revoked("test-jti")
        assert result is False  # Token is not revoked
