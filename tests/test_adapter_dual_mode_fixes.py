"""Tests for dual-mode adapter fixes per SPEC §5.

This module tests the improvements made to handle local self-hosted
and external API modes:
1. Weighted adapter blending (respects router gate weights)
2. Base model compatibility validation
3. Deterministic tokenizer fallback (FNV-1a hash)

NOTE: Tests that use LocalJaxLoRABackend require JAX and are skipped
when JAX is not installed.
"""

from __future__ import annotations

import json

import pytest

# Check if JAX is available for LocalJaxLoRABackend tests. The import is the
# probe — find_spec would say a broken install is present.
try:
    import jax  # noqa: F401
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


@requires_jax
class TestBaseIdentityGovernsTheWeightPath:
    """SPEC §5.1: weights load onto the base they were fitted to, or not at all.

    This class replaces a suite of fuzzy-match tests that pinned a helper the
    weight path no longer consults. That helper accepted a mistral adapter on
    a llama backend with a warning, treated "-chat" and "-base" as one model,
    and stripped version suffixes so v1 weights loaded onto v2. Every one of
    those is a different frozen W.
    """

    def _backend(self, tmp_path, base="llama-7b"):
        return LocalJaxLoRABackend(base, str(tmp_path))

    def test_the_same_name_is_the_same_base(self, tmp_path):
        backend = self._backend(tmp_path)
        backend._assert_exact_base_model({"base_model": "llama-7b"}, "a1")
        # Case and a trailing separator are spelling, not identity.
        backend._assert_exact_base_model({"base_model": "Llama-7B"}, "a1")
        # A checkout path and the bare name of the same checkpoint agree.
        backend._assert_exact_base_model({"base_model": "models/llama-7b"}, "a1")

    def test_the_declaration_may_live_in_the_schema(self, tmp_path):
        backend = self._backend(tmp_path)
        backend._assert_exact_base_model({"schema": {"base_model": "llama-7b"}}, "a1")
        backend._assert_exact_base_model({"model": "llama-7b"}, "a1")

    @pytest.mark.parametrize(
        "declared",
        [
            "mistral-7b",      # a different model
            "llama-7b-chat",   # a different fine-tune of the same family
            "llama-13b",       # a different size
            "llama-7b-v2",     # a different release
            None,              # no declaration at all
            "",
        ],
    )
    def test_anything_else_refuses(self, tmp_path, declared):
        backend = self._backend(tmp_path)
        with pytest.raises(ValueError, match="serves 'llama-7b'"):
            backend._assert_exact_base_model({"base_model": declared}, "a1")

    def test_a_backend_with_no_base_serves_no_weights(self, tmp_path):
        """Fail closed at both ends: an unnamed serving base cannot be
        matched, so it cannot be shown to be the base anything was fitted to."""
        backend = self._backend(tmp_path, base="")
        with pytest.raises(ValueError, match="serves"):
            backend._assert_exact_base_model({"base_model": ""}, "a1")


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
        """Backend with adapter weights in the SPEC §5.2 name shape.

        These matrices used to be keyed `layer0.A` / `layer0.B`, a name shape
        the spec does not define. That was invisible while composition
        accepted anything ending in `.A`; now that a malformed name refuses
        the whole adapter, the fixture has to speak the format the product
        does — otherwise it tests blending against inputs no real adapter can
        have.
        """
        backend = LocalJaxLoRABackend("test-model", str(tmp_path))

        # Create adapter directories with weight files
        adapter1_dir = tmp_path / "adapters" / "adapter1"
        adapter1_dir.mkdir(parents=True)
        (adapter1_dir / "params.json").write_text(
            json.dumps(
                {
                    "layers.0.attn_q.A": [[1.0, 0.0], [0.0, 1.0]],
                    "layers.0.attn_q.B": [[1.0, 0.0], [0.0, 1.0]],
                }
            )
        )

        adapter2_dir = tmp_path / "adapters" / "adapter2"
        adapter2_dir.mkdir(parents=True)
        (adapter2_dir / "params.json").write_text(
            json.dumps(
                {
                    "layers.0.attn_q.A": [[2.0, 0.0], [0.0, 2.0]],
                    "layers.0.attn_q.B": [[2.0, 0.0], [0.0, 2.0]],
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
                "base_model": "test-model",
                "weight": 1.0,
                "fs_dir": str(tmp_path / "adapters" / "adapter1"),
            }
        ]

        weights = backend._blend_adapter_weights(adapters, user_id="test-user")

        # Should be identity matrices (1.0 weight * original)
        assert "layers.0.attn_q.A" in weights
        # Values should be approximately [1, 0], [0, 1]
        a_values = weights["layers.0.attn_q.A"].tolist()
        assert abs(a_values[0][0] - 1.0) < 0.01
        assert abs(a_values[1][1] - 1.0) < 0.01

    def test_weighted_blend_respects_weights(self, backend_with_adapters):
        """Composition follows SPEC §5.2: ΔW = Σ_j g_j·α_j·B_j A_j.

        This used to assert the average (0.3·1 + 0.7·2)/1.0 = 1.7 on the A
        matrix alone, which described the implementation rather than the
        SPEC — and that implementation cancelled the gate for a single
        adapter and produced cross terms for two. The composed *delta* is
        what the equation is about, so that is what is checked.
        """
        backend, tmp_path = backend_with_adapters

        adapters = [
            {
                "id": "adapter1",  # A = B = I
                "base_model": "test-model",
                "weight": 0.3,
                "fs_dir": str(tmp_path / "adapters" / "adapter1"),
            },
            {
                "id": "adapter2",  # A = B = 2I
                "base_model": "test-model",
                "weight": 0.7,
                "fs_dir": str(tmp_path / "adapters" / "adapter2"),
            },
        ]

        weights = backend._blend_adapter_weights(adapters, user_id="test-user")
        delta = (weights["layers.0.attn_q.B"] @ weights["layers.0.attn_q.A"]).tolist()

        # 0.3·(I·I) + 0.7·(2I·2I) = 0.3·I + 2.8·I = 3.1·I
        assert abs(delta[0][0] - 3.1) < 0.01
        assert abs(delta[1][1] - 3.1) < 0.01
        assert abs(delta[0][1]) < 0.01  # no cross terms off the diagonal

    def test_gate_weight_field_recognized(self, backend_with_adapters):
        """Should recognize gate_weight as alternative to weight."""
        backend, tmp_path = backend_with_adapters

        adapters = [
            {
                "id": "adapter1",
                "base_model": "test-model",
                "gate_weight": 0.5,
                "fs_dir": str(tmp_path / "adapters" / "adapter1"),
            },
        ]

        weights = backend._blend_adapter_weights(adapters, user_id="test-user")

        # Should have loaded weights
        assert "layers.0.attn_q.A" in weights

    def test_weight_in_schema_recognized(self, backend_with_adapters):
        """Should check schema dict for weight."""
        backend, tmp_path = backend_with_adapters

        adapters = [
            {
                "id": "adapter1",
                "base_model": "test-model",
                "schema": {"weight": 0.5},
                "fs_dir": str(tmp_path / "adapters" / "adapter1"),
            },
        ]

        weights = backend._blend_adapter_weights(adapters, user_id="test-user")

        assert "layers.0.attn_q.A" in weights

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
                "id": "adapter1",  # matches its directory (§5.5)
                "base_model": "test-model",
                "weight": 0.0,
                "fs_dir": str(tmp_path / "adapters" / "adapter1"),
            },
            {
                "id": "adapter2",  # matches its directory (§5.5)
                "base_model": "test-model",
                "weight": 1.0,
                "fs_dir": str(tmp_path / "adapters" / "adapter2"),
            },
        ]

        weights = backend._blend_adapter_weights(adapters, user_id="test-user")

        # Should only contain adapter2's values (2.0) since adapter1 has weight=0
        a_values = weights["layers.0.attn_q.A"].tolist()
        assert abs(a_values[0][0] - 2.0) < 0.01

    def test_weight_clamping(self, backend_with_adapters):
        """Weights should be clamped to [0, 1]."""
        backend, tmp_path = backend_with_adapters

        adapters = [
            {
                "id": "adapter1",
                "base_model": "test-model",
                "weight": 1.5,  # Should be clamped to 1.0
                "fs_dir": str(tmp_path / "adapters" / "adapter1"),
            },
            {
                "id": "adapter2",
                "base_model": "test-model",
                "weight": -0.5,  # Should be clamped to 0.0
                "fs_dir": str(tmp_path / "adapters" / "adapter2"),
            },
        ]

        weights = backend._blend_adapter_weights(adapters, user_id="test-user")

        # adapter2 should be skipped (weight clamped to 0)
        # adapter1 should use weight 1.0
        a_values = weights["layers.0.attn_q.A"].tolist()
        assert abs(a_values[0][0] - 1.0) < 0.01

    def test_default_weight(self, backend_with_adapters):
        """Adapters without weight should default to 1.0."""
        backend, tmp_path = backend_with_adapters

        adapters = [
            {
                "id": "adapter1",
                "base_model": "test-model",
                # No weight specified
                "fs_dir": str(tmp_path / "adapters" / "adapter1"),
            },
        ]

        weights = backend._blend_adapter_weights(adapters, user_id="test-user")

        # Should use full weight (1.0 * 1.0 = 1.0)
        a_values = weights["layers.0.attn_q.A"].tolist()
        assert abs(a_values[0][0] - 1.0) < 0.01

    def test_empty_adapters_returns_empty(self, backend_with_adapters):
        """Empty adapter list should return empty weights."""
        backend, _ = backend_with_adapters

        weights = backend._blend_adapter_weights([], user_id="test-user")

        assert weights == {}

    def test_gates_are_not_normalized_away(self, backend_with_adapters):
        """A gate must scale the delta, not cancel itself.

        The old implementation divided the accumulated matrices by the total
        gate weight, so a lone adapter computed (g·A)/g = A and every gate
        produced an identical delta. Halving the gate must halve the delta.
        """
        backend, tmp_path = backend_with_adapters
        path = str(tmp_path / "adapters" / "adapter1")  # A = B = I

        def delta_at(gate):
            weights = backend._blend_adapter_weights(
                [{"id": "adapter1", "base_model": "test-model", "weight": gate, "fs_dir": path}],
                user_id="test-user",
            )
            return (weights["layers.0.attn_q.B"] @ weights["layers.0.attn_q.A"]).tolist()[0][0]

        assert abs(delta_at(1.0) - 1.0) < 0.01
        assert abs(delta_at(0.5) - 0.5) < 0.01
        assert abs(delta_at(0.25) - 0.25) < 0.01


# ==============================================================================
# Integration Tests
# ==============================================================================


@requires_jax
class TestDualModeIntegration:
    """Integration tests for dual-mode adapter handling."""

    def test_a_mismatched_base_model_refuses_the_weights(self, tmp_path):
        """SPEC §5.1: an adapter is fitted to the model that will serve it.

        This test used to assert the opposite — that a mistral adapter loads
        onto a llama backend because non-strict mode "logs a warning but
        still loads". That recorded the implementation, not the contract:
        B·A was optimized against one particular frozen W, and an eval gate
        passed on that W says nothing about a different one. Training already
        refuses this mismatch; serving now holds the same line.
        """
        backend = LocalJaxLoRABackend("llama-7b", str(tmp_path))

        adapter_dir = tmp_path / "adapters" / "mismatch"
        adapter_dir.mkdir(parents=True)
        (adapter_dir / "params.json").write_text(json.dumps({"layers.0.attn_q.A": [[1.0]]}))

        adapter = {
            "id": "mismatch",
            "base_model": "mistral-7b",  # Different from backend's llama-7b
            "fs_dir": str(adapter_dir),
        }

        with pytest.raises(ValueError, match="serves"):
            backend._load_adapter_weights(adapter, user_id="test-user")

        # The matching adapter loads, so this is a contract and not a wall.
        matching = {**adapter, "base_model": "llama-7b"}
        assert "layers.0.attn_q.A" in backend._load_adapter_weights(
            matching, user_id="test-user"
        )

    def test_a_prompt_rung_adapter_is_exempt(self, tmp_path):
        """The base rule guards weights, not instructions.

        A prompt-rung adapter carries no tensors, so it has no base to
        disagree with — and refusing it would silence the ladder's first
        rung on any deployment that renamed its checkpoint.
        """
        backend = LocalJaxLoRABackend("llama-7b", str(tmp_path))

        adapter_dir = tmp_path / "adapters" / "prompted"
        adapter_dir.mkdir(parents=True)
        (adapter_dir / "params.json").write_text(json.dumps({"layers.0.attn_q.A": [[1.0]]}))

        assert backend._load_adapter_weights(
            {
                "id": "prompted",
                "mode": "prompt",
                "base_model": "mistral-7b",
                "fs_dir": str(adapter_dir),
            },
            user_id="test-user",
        ) == {}
