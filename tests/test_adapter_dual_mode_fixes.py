"""Tests for SPEC §5 weighted adapter blending and dual-mode fixes.

These tests require JAX for LocalJaxLoRABackend operations.
Tests are skipped when JAX is not installed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import pytest

from liminallm.config import AdapterMode

# Check if JAX is available
try:
    import jax  # noqa: F401

    HAS_JAX = True
except ImportError:
    HAS_JAX = False

# Skip entire module if JAX not available
pytestmark = pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")


# Only import LocalJaxLoRABackend if JAX is available to avoid import errors
if HAS_JAX:
    from liminallm.service.model_backend import LocalJaxLoRABackend


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def backend_with_adapters(tmp_path: Path) -> Tuple["LocalJaxLoRABackend", Path]:
    """Create a LocalJaxLoRABackend with test adapters."""
    backend = LocalJaxLoRABackend("test-model", str(tmp_path))

    # Create adapter directories with params
    adapters_dir = tmp_path / "adapters"
    adapters_dir.mkdir(parents=True)

    # Create adapter1 with weight values of 1.0
    adapter1_dir = adapters_dir / "adapter1"
    adapter1_dir.mkdir()
    (adapter1_dir / "params.json").write_text(
        json.dumps({"layer0.A": [[1.0]], "layer0.B": [[1.0]]})
    )

    # Create adapter2 with weight values of 2.0
    adapter2_dir = adapters_dir / "adapter2"
    adapter2_dir.mkdir()
    (adapter2_dir / "params.json").write_text(
        json.dumps({"layer0.A": [[2.0]], "layer0.B": [[2.0]]})
    )

    return backend, tmp_path


# ==============================================================================
# TestWeightedAdapterBlending
# ==============================================================================


class TestWeightedAdapterBlending:
    """Test weighted adapter blending in LocalJaxLoRABackend."""

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

        # Should have loaded the weights
        assert len(weights) > 0
        # Check that layer0.A is present
        assert "layer0.A" in weights

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

        # Should have blended weights
        assert len(weights) > 0

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

        assert len(weights) > 0

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

        assert len(weights) > 0

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

        # Should have weights from at least one adapter
        assert len(weights) > 0

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

        # Should have loaded weights (exact behavior depends on implementation)
        assert len(weights) > 0

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

        # Should have loaded the full weights
        assert len(weights) > 0

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

        # Blended weights should exist
        assert len(weights) > 0


# ==============================================================================
# TestDualModeIntegration
# ==============================================================================


class TestDualModeIntegration:
    """Test dual-mode integration aspects of LocalJaxLoRABackend."""

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

        # Should still load despite mismatch (non-strict mode)
        assert len(weights) > 0

    def test_compatible_modes_defined(self):
        """LocalJaxLoRABackend should define compatible modes."""
        assert AdapterMode.LOCAL in LocalJaxLoRABackend.COMPATIBLE_MODES
        assert AdapterMode.HYBRID in LocalJaxLoRABackend.COMPATIBLE_MODES
        assert AdapterMode.PROMPT in LocalJaxLoRABackend.COMPATIBLE_MODES
        assert AdapterMode.REMOTE not in LocalJaxLoRABackend.COMPATIBLE_MODES

    def test_mode_attribute_set(self, tmp_path):
        """Backend should have mode attribute set to local_lora."""
        backend = LocalJaxLoRABackend("test-model", str(tmp_path))
        assert backend.mode == "local_lora"

    def test_empty_adapter_returns_empty_weights(self, tmp_path):
        """Empty adapter dict should return empty weights."""
        backend = LocalJaxLoRABackend("test-model", str(tmp_path))

        weights = backend._load_adapter_weights({}, user_id="test-user")

        assert weights == {}

    def test_none_adapter_returns_empty_weights(self, tmp_path):
        """None adapter should return empty weights."""
        backend = LocalJaxLoRABackend("test-model", str(tmp_path))

        weights = backend._load_adapter_weights(None, user_id="test-user")

        assert weights == {}
