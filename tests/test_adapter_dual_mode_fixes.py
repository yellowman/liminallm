"""Tests for dual-mode adapter fixes per SPEC §5.

This module tests the improvements made to handle local self-hosted
and external API modes:
1. Weighted adapter blending (respects router gate weights)
2. Base model compatibility validation
3. Deterministic tokenizer fallback (FNV-1a hash)

NOTE: Tests that use LocalJaxLoRABackend require JAX and are skipped
when JAX is not installed. TestValidateAdapterBaseModel runs without JAX.
"""

from __future__ import annotations

import json

import pytest

# Import validate_adapter_base_model unconditionally (doesn't require JAX)
from liminallm.service.model_backend import validate_adapter_base_model

# Check if JAX is available for LocalJaxLoRABackend tests
try:
    import jax
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

# Conditionally import LocalJaxLoRABackend only if JAX is available
if HAS_JAX:
    from liminallm.service.model_backend import LocalJaxLoRABackend

# Decorator to skip tests that require JAX
requires_jax = pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")


# ==============================================================================
# Base Model Validation Tests
# ==============================================================================


class TestValidateAdapterBaseModel:
    """Test base model compatibility validation per SPEC §5.1."""

    def test_exact_match(self):
        """Should return valid for exact base model match."""
        adapter = {"id": "a1", "base_model": "llama-7b"}
        is_valid, warning = validate_adapter_base_model(adapter, "llama-7b")

        assert is_valid is True
        assert warning is None

    def test_normalized_match(self):
        """Should normalize model names for comparison."""
        adapter = {"id": "a1", "base_model": "models/llama-7b"}
        is_valid, warning = validate_adapter_base_model(adapter, "llama-7b")

        assert is_valid is True
        assert warning is None

    def test_case_insensitive(self):
        """Should match case-insensitively."""
        adapter = {"id": "a1", "base_model": "Llama-7B"}
        is_valid, warning = validate_adapter_base_model(adapter, "llama-7b")

        assert is_valid is True
        assert warning is None

    def test_variant_match_with_warning(self):
        """Should match model variants with warning."""
        adapter = {"id": "a1", "base_model": "llama-7b"}
        is_valid, warning = validate_adapter_base_model(adapter, "llama-7b-chat")

        assert is_valid is True
        assert warning is not None
        assert "variant" in warning.lower()

    def test_family_match_variants(self):
        """Should recognize model family variants."""
        # Base vs chat variant
        adapter = {"id": "a1", "base_model": "llama-7b-base"}
        is_valid, warning = validate_adapter_base_model(adapter, "llama-7b-chat")

        assert is_valid is True
        assert "llama-7b" in warning.lower()

    def test_mismatch_different_models(self):
        """Should detect incompatible base models."""
        adapter = {"id": "a1", "base_model": "mistral-7b"}
        is_valid, warning = validate_adapter_base_model(adapter, "llama-7b")

        assert is_valid is True  # Non-strict mode allows but warns
        assert warning is not None
        assert "mistral-7b" in warning

    def test_mismatch_strict_mode(self):
        """Should reject mismatched models in strict mode."""
        adapter = {"id": "a1", "base_model": "mistral-7b"}
        is_valid, warning = validate_adapter_base_model(
            adapter, "llama-7b", strict=True
        )

        assert is_valid is False
        assert "incompatible" in warning.lower()

    def test_missing_base_model_non_strict(self):
        """Should warn but allow missing base_model in non-strict mode."""
        adapter = {"id": "a1"}
        is_valid, warning = validate_adapter_base_model(adapter, "llama-7b")

        assert is_valid is True
        assert warning is not None
        assert "unverified" in warning.lower()

    def test_missing_base_model_strict(self):
        """Should reject missing base_model in strict mode."""
        adapter = {"id": "a1"}
        is_valid, warning = validate_adapter_base_model(
            adapter, "llama-7b", strict=True
        )

        assert is_valid is False
        assert "missing" in warning.lower()

    def test_base_model_in_schema(self):
        """Should check schema dict for base_model."""
        adapter = {"id": "a1", "schema": {"base_model": "llama-7b"}}
        is_valid, warning = validate_adapter_base_model(adapter, "llama-7b")

        assert is_valid is True
        assert warning is None

    def test_model_field_as_fallback(self):
        """Should check 'model' field as alternative to base_model."""
        adapter = {"id": "a1", "model": "llama-7b"}
        is_valid, warning = validate_adapter_base_model(adapter, "llama-7b")

        assert is_valid is True
        assert warning is None

    def test_empty_adapter(self):
        """Should handle empty/None adapter gracefully."""
        is_valid, warning = validate_adapter_base_model(None, "llama-7b")
        assert is_valid is True
        assert warning is None

        # Empty dict is falsy, treated same as None - nothing to validate
        is_valid2, warning2 = validate_adapter_base_model({}, "llama-7b")
        assert is_valid2 is True
        assert warning2 is None

    def test_version_suffix_normalization(self):
        """Should normalize version suffixes."""
        adapter = {"id": "a1", "base_model": "llama-7b-v1.0"}
        is_valid, warning = validate_adapter_base_model(adapter, "llama-7b-v2.0")

        # Same base, different version - should still match
        assert is_valid is True

    def test_prefix_normalization(self):
        """Should strip common prefixes."""
        adapter = {"id": "a1", "base_model": "hf://meta-llama/llama-7b"}
        is_valid, warning = validate_adapter_base_model(adapter, "meta-llama/llama-7b")

        assert is_valid is True


# ==============================================================================
# Deterministic Tokenizer Tests
# ==============================================================================


@requires_jax
class TestDeterministicTokenHash:
    """Test FNV-1a hash-based tokenization for determinism."""

    @pytest.fixture
    def backend(self, tmp_path):
        """Create a LocalJaxLoRABackend for testing."""
        return LocalJaxLoRABackend("test-model", str(tmp_path))

    def test_consistent_across_calls(self, backend):
        """Same token should produce same hash."""
        token = "hello"
        vocab_size = 50000

        hash1 = backend._deterministic_token_hash(token, vocab_size)
        hash2 = backend._deterministic_token_hash(token, vocab_size)

        assert hash1 == hash2

    def test_different_tokens_different_hashes(self, backend):
        """Different tokens should usually produce different hashes."""
        vocab_size = 50000

        hash1 = backend._deterministic_token_hash("hello", vocab_size)
        hash2 = backend._deterministic_token_hash("world", vocab_size)

        # Very unlikely to collide with good hash
        assert hash1 != hash2

    def test_respects_vocab_size(self, backend):
        """Hash should be within vocab_size bounds."""
        token = "test"

        for vocab_size in [100, 1000, 50000, 100000]:
            h = backend._deterministic_token_hash(token, vocab_size)
            assert 0 <= h < vocab_size

    def test_unicode_tokens(self, backend):
        """Should handle unicode tokens correctly."""
        vocab_size = 50000

        # Various unicode strings
        tokens = ["hello", "世界", "привет", "🎉", "café"]
        hashes = [backend._deterministic_token_hash(t, vocab_size) for t in tokens]

        # All should be valid
        assert all(0 <= h < vocab_size for h in hashes)
        # All should be different (very unlikely to collide)
        assert len(set(hashes)) == len(hashes)

    def test_empty_string(self, backend):
        """Should handle empty string."""
        h = backend._deterministic_token_hash("", 50000)
        assert isinstance(h, int)
        assert 0 <= h < 50000

    def test_fnv1a_known_values(self, backend):
        """Test FNV-1a produces expected values for known inputs."""
        # FNV-1a for empty string should be offset basis
        # After mod, this may vary based on vocab_size
        vocab_size = 2**32  # Use max to see raw hash

        # The hash for empty string in FNV-1a 32-bit is 0x811c9dc5
        h = backend._deterministic_token_hash("", vocab_size)
        assert h == 0x811C9DC5


@requires_jax
class TestTokenizeFallback:
    """Test tokenize fallback behavior."""

    @pytest.fixture
    def backend(self, tmp_path):
        """Create a LocalJaxLoRABackend without tokenizer."""
        b = LocalJaxLoRABackend("test-model", str(tmp_path))
        # Force tokenizer to be unavailable
        b._tokenizer = None
        b._tokenizer_error = "test: tokenizer unavailable"
        return b

    def test_fallback_tokenization(self, backend):
        """Should use whitespace tokenization as fallback."""
        text = "hello world test"

        ids, attention = backend._tokenize(text)

        assert len(ids) == 3  # Three tokens
        assert len(attention) == 3
        assert all(a == 1 for a in attention)

    def test_fallback_deterministic(self, backend):
        """Fallback tokenization should be deterministic."""
        text = "the quick brown fox"

        ids1, _ = backend._tokenize(text)
        ids2, _ = backend._tokenize(text)

        assert ids1 == ids2

    def test_fallback_max_seq_len(self, backend):
        """Should respect max_seq_len in fallback mode."""
        backend.max_seq_len = 5
        text = " ".join(["word"] * 20)

        ids, attention = backend._tokenize(text)

        assert len(ids) <= 5
        assert len(attention) <= 5


# ==============================================================================
# Weighted Adapter Blending Tests
# ==============================================================================


@requires_jax
class TestWeightedAdapterBlending:
    """Test adapter weight blending per SPEC §5.2."""

    @pytest.fixture
    def backend_with_adapters(self, tmp_path):
        """Create backend with mock adapter weights."""
        backend = LocalJaxLoRABackend("test-model", str(tmp_path))

        # Create adapter directories with weight files
        adapter1_dir = tmp_path / "adapters" / "adapter1"
        adapter1_dir.mkdir(parents=True)
        (adapter1_dir / "params.json").write_text(
            json.dumps(
                {
                    "layer0.A": [[1.0, 0.0], [0.0, 1.0]],
                    "layer0.B": [[1.0, 0.0], [0.0, 1.0]],
                }
            )
        )

        adapter2_dir = tmp_path / "adapters" / "adapter2"
        adapter2_dir.mkdir(parents=True)
        (adapter2_dir / "params.json").write_text(
            json.dumps(
                {
                    "layer0.A": [[2.0, 0.0], [0.0, 2.0]],
                    "layer0.B": [[2.0, 0.0], [0.0, 2.0]],
                }
            )
        )

        return backend, tmp_path

    def test_single_adapter_full_weight(self, backend_with_adapters):
        """Single adapter with weight 1.0 should use full weights."""
        backend, tmp_path = backend_with_adapters

        adapters = [
            {
                "id": "adapter1",
                "weight": 1.0,
                "fs_dir": str(tmp_path / "adapters" / "adapter1"),
            }
        ]

        weights = backend._blend_adapter_weights(adapters, user_id="test-user")

        # Should be identity matrices (1.0 weight * original)
        assert "layer0.A" in weights
        # Values should be approximately [1, 0], [0, 1]
        a_values = weights["layer0.A"].tolist()
        assert abs(a_values[0][0] - 1.0) < 0.01
        assert abs(a_values[1][1] - 1.0) < 0.01

    def test_weighted_blend_respects_weights(self, backend_with_adapters):
        """Blending should respect individual adapter weights."""
        backend, tmp_path = backend_with_adapters

        # adapter1 has weight 0.3, adapter2 has weight 0.7
        # adapter1 values are 1.0, adapter2 values are 2.0
        # Expected blend: (0.3 * 1.0 + 0.7 * 2.0) / (0.3 + 0.7) = 1.7
        adapters = [
            {
                "id": "adapter1",
                "weight": 0.3,
                "fs_dir": str(tmp_path / "adapters" / "adapter1"),
            },
            {
                "id": "adapter2",
                "weight": 0.7,
                "fs_dir": str(tmp_path / "adapters" / "adapter2"),
            },
        ]

        weights = backend._blend_adapter_weights(adapters, user_id="test-user")

        # Check blended value
        a_values = weights["layer0.A"].tolist()
        expected = 1.7  # (0.3 * 1.0 + 0.7 * 2.0) / 1.0
        assert abs(a_values[0][0] - expected) < 0.01

    def test_gate_weight_field_recognized(self, backend_with_adapters):
        """Should recognize gate_weight as alternative to weight."""
        backend, tmp_path = backend_with_adapters

        adapters = [
            {
                "id": "adapter1",
                "gate_weight": 0.5,
                "fs_dir": str(tmp_path / "adapters" / "adapter1"),
            },
        ]

        weights = backend._blend_adapter_weights(adapters, user_id="test-user")

        # Should have loaded weights
        assert "layer0.A" in weights

    def test_weight_in_schema_recognized(self, backend_with_adapters):
        """Should check schema dict for weight."""
        backend, tmp_path = backend_with_adapters

        adapters = [
            {
                "id": "adapter1",
                "schema": {"weight": 0.5},
                "fs_dir": str(tmp_path / "adapters" / "adapter1"),
            },
        ]

        weights = backend._blend_adapter_weights(adapters, user_id="test-user")

        assert "layer0.A" in weights

    def test_zero_weight_adapter_skipped(self, backend_with_adapters):
        """Adapters with zero weight should be skipped.

        Note: This tests that weight=0.0 is correctly recognized as zero
        (not treated as falsy and defaulted to 1.0).
        """
        backend, tmp_path = backend_with_adapters
        # Clear cache to ensure fresh load
        backend._adapter_cache.clear()

        adapters = [
            {
                "id": "adapter1_zero",  # Unique ID to avoid cache issues
                "weight": 0.0,
                "fs_dir": str(tmp_path / "adapters" / "adapter1"),
            },
            {
                "id": "adapter2_full",  # Unique ID
                "weight": 1.0,
                "fs_dir": str(tmp_path / "adapters" / "adapter2"),
            },
        ]

        weights = backend._blend_adapter_weights(adapters, user_id="test-user")

        # Should only contain adapter2's values (2.0) since adapter1 has weight=0
        a_values = weights["layer0.A"].tolist()
        assert abs(a_values[0][0] - 2.0) < 0.01

    def test_weight_clamping(self, backend_with_adapters):
        """Weights should be clamped to [0, 1]."""
        backend, tmp_path = backend_with_adapters

        adapters = [
            {
                "id": "adapter1",
                "weight": 1.5,  # Should be clamped to 1.0
                "fs_dir": str(tmp_path / "adapters" / "adapter1"),
            },
            {
                "id": "adapter2",
                "weight": -0.5,  # Should be clamped to 0.0
                "fs_dir": str(tmp_path / "adapters" / "adapter2"),
            },
        ]

        weights = backend._blend_adapter_weights(adapters, user_id="test-user")

        # adapter2 should be skipped (weight clamped to 0)
        # adapter1 should use weight 1.0
        a_values = weights["layer0.A"].tolist()
        assert abs(a_values[0][0] - 1.0) < 0.01

    def test_default_weight(self, backend_with_adapters):
        """Adapters without weight should default to 1.0."""
        backend, tmp_path = backend_with_adapters

        adapters = [
            {
                "id": "adapter1",
                # No weight specified
                "fs_dir": str(tmp_path / "adapters" / "adapter1"),
            },
        ]

        weights = backend._blend_adapter_weights(adapters, user_id="test-user")

        # Should use full weight (1.0 * 1.0 = 1.0)
        a_values = weights["layer0.A"].tolist()
        assert abs(a_values[0][0] - 1.0) < 0.01

    def test_empty_adapters_returns_empty(self, backend_with_adapters):
        """Empty adapter list should return empty weights."""
        backend, _ = backend_with_adapters

        weights = backend._blend_adapter_weights([], user_id="test-user")

        assert weights == {}

    def test_normalization_preserves_scale(self, backend_with_adapters):
        """Weight normalization should preserve expected scale."""
        backend, tmp_path = backend_with_adapters

        # Two adapters with equal weights
        # adapter1: 1.0, adapter2: 2.0
        # Equal blend: (0.5 * 1.0 + 0.5 * 2.0) / 1.0 = 1.5
        adapters = [
            {
                "id": "adapter1",
                "weight": 0.5,
                "fs_dir": str(tmp_path / "adapters" / "adapter1"),
            },
            {
                "id": "adapter2",
                "weight": 0.5,
                "fs_dir": str(tmp_path / "adapters" / "adapter2"),
            },
        ]

        weights = backend._blend_adapter_weights(adapters, user_id="test-user")

        a_values = weights["layer0.A"].tolist()
        expected = 1.5  # Average of 1.0 and 2.0
        assert abs(a_values[0][0] - expected) < 0.01


# ==============================================================================
# Integration Tests
# ==============================================================================


@requires_jax
class TestDualModeIntegration:
    """Integration tests for dual-mode adapter handling."""

    def test_base_model_validation_in_load(self, tmp_path):
        """Base model validation should be called during weight loading."""
        backend = LocalJaxLoRABackend("llama-7b", str(tmp_path))

        # Create adapter with mismatched base model
        adapter_dir = tmp_path / "adapters" / "mismatch"
        adapter_dir.mkdir(parents=True)
        (adapter_dir / "params.json").write_text(json.dumps({"layer0.A": [[1.0]]}))

        adapter = {
            "id": "mismatch",
            "base_model": "mistral-7b",  # Different from backend's llama-7b
            "fs_dir": str(adapter_dir),
        }

        # Non-strict mode should log warning but still load
        weights = backend._load_adapter_weights(adapter, user_id="test-user")
        assert "layer0.A" in weights

    def test_strict_base_model_rejects_mismatch(self, tmp_path):
        """Strict mode should reject mismatched base models."""
        backend = LocalJaxLoRABackend("llama-7b", str(tmp_path))

        adapter_dir = tmp_path / "adapters" / "mismatch"
        adapter_dir.mkdir(parents=True)
        (adapter_dir / "params.json").write_text(json.dumps({"layer0.A": [[1.0]]}))

        adapter = {
            "id": "mismatch",
            "base_model": "mistral-7b",
            "fs_dir": str(adapter_dir),
        }

        with pytest.raises(ValueError, match="incompatible"):
            backend._load_adapter_weights(
                adapter, user_id="test-user", strict_base_model=True
            )
