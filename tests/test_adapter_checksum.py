"""Tests for adapter checksum validation in model_backend.py.

Per SPEC §18, adapter checksums must be verified against schema.checksum
before activation to prevent loading tampered weights.
"""

import hashlib
import json
from unittest.mock import patch, MagicMock

import pytest


class TestAdapterChecksumValidation:
    """Tests for _load_adapter_weights checksum validation."""

    @pytest.fixture
    def mock_backend(self, tmp_path):
        """Create a LocalJaxLoRABackend for testing."""
        from liminallm.service.model_backend import LocalJaxLoRABackend

        backend = LocalJaxLoRABackend.__new__(LocalJaxLoRABackend)
        backend._fs_root = str(tmp_path)
        backend._base_model = "test-model"
        backend._adapter_cache = {}
        backend._jnp = None
        backend._jax = None
        backend._vocab_size = 32000
        return backend

    @pytest.fixture
    def valid_weights(self):
        """Create valid adapter weights JSON."""
        return {"layer1": [1.0, 2.0, 3.0], "layer2": [4.0, 5.0, 6.0]}

    def create_adapter_file(self, tmp_path, weights, subdir="test_adapter"):
        """Create adapter params.json file and return its path and checksum."""
        adapter_dir = tmp_path / subdir
        adapter_dir.mkdir(parents=True, exist_ok=True)

        params_path = adapter_dir / "params.json"
        payload = json.dumps(weights).encode()
        params_path.write_bytes(payload)

        checksum = hashlib.sha256(payload).hexdigest()
        return adapter_dir, checksum

    def test_valid_checksum_loads_weights(self, mock_backend, tmp_path, valid_weights):
        """Adapter with valid checksum loads successfully."""
        adapter_dir, checksum = self.create_adapter_file(tmp_path, valid_weights)

        adapter = {
            "id": "test-adapter",
            "checksum": checksum,
        }

        # Mock _adapter_path to return our test directory
        with patch.object(mock_backend, "_adapter_path", return_value=str(adapter_dir)):
            with patch.object(
                mock_backend,
                "_resolve_params_path",
                return_value=adapter_dir / "params.json",
            ):
                with patch.object(mock_backend, "_ensure_jax"):
                    # Mock jnp.array to just return the value
                    mock_jnp = MagicMock()
                    mock_jnp.array = lambda v, dtype: v
                    mock_jnp.float32 = "float32"
                    mock_backend._jnp = mock_jnp

                    weights = mock_backend._load_adapter_weights(adapter)

                    assert weights is not None
                    assert "layer1" in weights
                    assert "layer2" in weights

    def test_invalid_checksum_raises_error(self, mock_backend, tmp_path, valid_weights):
        """Adapter with invalid checksum raises ValueError."""
        adapter_dir, _ = self.create_adapter_file(tmp_path, valid_weights)

        adapter = {
            "id": "test-adapter",
            "checksum": "invalid_checksum_that_wont_match",
        }

        with patch.object(mock_backend, "_adapter_path", return_value=str(adapter_dir)):
            with patch.object(
                mock_backend,
                "_resolve_params_path",
                return_value=adapter_dir / "params.json",
            ):
                with pytest.raises(ValueError, match="checksum mismatch"):
                    mock_backend._load_adapter_weights(adapter)

    def test_checksum_mismatch_logs_error(self, mock_backend, tmp_path, valid_weights):
        """Checksum mismatch logs detailed error before raising."""
        adapter_dir, correct_checksum = self.create_adapter_file(
            tmp_path, valid_weights
        )

        adapter = {
            "id": "test-adapter",
            "checksum": "wrong_checksum",
        }

        with patch.object(mock_backend, "_adapter_path", return_value=str(adapter_dir)):
            with patch.object(
                mock_backend,
                "_resolve_params_path",
                return_value=adapter_dir / "params.json",
            ):
                with patch("liminallm.service.model_backend.logger") as mock_logger:
                    with pytest.raises(ValueError):
                        mock_backend._load_adapter_weights(adapter)

                    # Verify error was logged with details
                    mock_logger.error.assert_called_once()
                    call_args = mock_logger.error.call_args
                    assert call_args[0][0] == "adapter_checksum_mismatch"
                    assert "expected" in call_args[1]
                    assert "actual" in call_args[1]
                    assert call_args[1]["expected"] == "wrong_checksum"
                    assert call_args[1]["actual"] == correct_checksum

    def test_missing_checksum_logs_warning(self, mock_backend, tmp_path, valid_weights):
        """Adapter without checksum logs security warning."""
        adapter_dir, _ = self.create_adapter_file(tmp_path, valid_weights)

        adapter = {
            "id": "test-adapter",
            # No checksum provided
        }

        with patch.object(mock_backend, "_adapter_path", return_value=str(adapter_dir)):
            with patch.object(
                mock_backend,
                "_resolve_params_path",
                return_value=adapter_dir / "params.json",
            ):
                with patch.object(mock_backend, "_ensure_jax"):
                    mock_jnp = MagicMock()
                    mock_jnp.array = lambda v, dtype: v
                    mock_jnp.float32 = "float32"
                    mock_backend._jnp = mock_jnp

                    with patch("liminallm.service.model_backend.logger") as mock_logger:
                        mock_backend._load_adapter_weights(adapter)

                        # Should log warning about missing checksum
                        mock_logger.warning.assert_called_once()
                        call_args = mock_logger.warning.call_args
                        assert call_args[0][0] == "adapter_checksum_missing"
                        assert "message" in call_args[1]
                        assert "production" in call_args[1]["message"].lower()

    def test_checksum_from_schema_field(self, mock_backend, tmp_path, valid_weights):
        """Checksum can be provided in schema.checksum field."""
        adapter_dir, checksum = self.create_adapter_file(tmp_path, valid_weights)

        adapter = {
            "id": "test-adapter",
            "schema": {"checksum": checksum},
        }

        with patch.object(mock_backend, "_adapter_path", return_value=str(adapter_dir)):
            with patch.object(
                mock_backend,
                "_resolve_params_path",
                return_value=adapter_dir / "params.json",
            ):
                with patch.object(mock_backend, "_ensure_jax"):
                    mock_jnp = MagicMock()
                    mock_jnp.array = lambda v, dtype: v
                    mock_jnp.float32 = "float32"
                    mock_backend._jnp = mock_jnp

                    weights = mock_backend._load_adapter_weights(adapter)

                    assert weights is not None

    def test_top_level_checksum_preferred(self, mock_backend, tmp_path, valid_weights):
        """Top-level checksum is preferred over schema.checksum."""
        adapter_dir, correct_checksum = self.create_adapter_file(
            tmp_path, valid_weights
        )

        adapter = {
            "id": "test-adapter",
            "checksum": correct_checksum,
            "schema": {"checksum": "wrong_in_schema"},
        }

        with patch.object(mock_backend, "_adapter_path", return_value=str(adapter_dir)):
            with patch.object(
                mock_backend,
                "_resolve_params_path",
                return_value=adapter_dir / "params.json",
            ):
                with patch.object(mock_backend, "_ensure_jax"):
                    mock_jnp = MagicMock()
                    mock_jnp.array = lambda v, dtype: v
                    mock_jnp.float32 = "float32"
                    mock_backend._jnp = mock_jnp

                    # Should succeed because top-level checksum is correct
                    weights = mock_backend._load_adapter_weights(adapter)
                    assert weights is not None

    def test_empty_adapter_returns_empty_dict(self, mock_backend):
        """Empty adapter dict returns empty weights."""
        weights = mock_backend._load_adapter_weights({})
        assert weights == {}

    def test_none_adapter_returns_empty_dict(self, mock_backend):
        """None adapter returns empty weights."""
        weights = mock_backend._load_adapter_weights(None)
        assert weights == {}

    def test_cache_hit_skips_checksum(self, mock_backend, tmp_path, valid_weights):
        """Cached weights skip checksum verification."""
        adapter_dir, checksum = self.create_adapter_file(tmp_path, valid_weights)
        params_path = adapter_dir / "params.json"

        adapter = {
            "id": "test-adapter",
            "checksum": checksum,
        }

        # Pre-populate cache
        cached_weights = {"cached": True}
        mock_backend._adapter_cache["test-adapter"] = (
            params_path.stat().st_mtime,
            cached_weights,
        )

        with patch.object(mock_backend, "_adapter_path", return_value=str(adapter_dir)):
            with patch.object(
                mock_backend, "_resolve_params_path", return_value=params_path
            ):
                weights = mock_backend._load_adapter_weights(adapter)

                # Should return cached weights
                assert weights == cached_weights

    def test_modified_file_invalidates_cache(
        self, mock_backend, tmp_path, valid_weights
    ):
        """Modified file invalidates cache and re-verifies checksum."""
        adapter_dir, checksum = self.create_adapter_file(tmp_path, valid_weights)
        params_path = adapter_dir / "params.json"

        adapter = {
            "id": "test-adapter",
            "checksum": checksum,
        }

        # Pre-populate cache with old mtime
        mock_backend._adapter_cache["test-adapter"] = (0, {"old": True})

        with patch.object(mock_backend, "_adapter_path", return_value=str(adapter_dir)):
            with patch.object(
                mock_backend, "_resolve_params_path", return_value=params_path
            ):
                with patch.object(mock_backend, "_ensure_jax"):
                    mock_jnp = MagicMock()
                    mock_jnp.array = lambda v, dtype: v
                    mock_jnp.float32 = "float32"
                    mock_backend._jnp = mock_jnp

                    weights = mock_backend._load_adapter_weights(adapter)

                    # Should have loaded new weights, not cached
                    assert "layer1" in weights


class TestChecksumComputation:
    """Tests for checksum computation helpers."""

    def test_sha256_checksum_format(self):
        """Checksum is SHA-256 hex digest."""
        payload = b'{"test": "data"}'
        expected = hashlib.sha256(payload).hexdigest()

        assert len(expected) == 64  # SHA-256 produces 64 hex chars
        assert all(c in "0123456789abcdef" for c in expected)

    def test_checksum_changes_with_content(self):
        """Different content produces different checksums."""
        payload1 = b'{"a": 1}'
        payload2 = b'{"a": 2}'

        checksum1 = hashlib.sha256(payload1).hexdigest()
        checksum2 = hashlib.sha256(payload2).hexdigest()

        assert checksum1 != checksum2

    def test_checksum_consistent_for_same_content(self):
        """Same content always produces same checksum."""
        payload = b'{"consistent": "data"}'

        checksum1 = hashlib.sha256(payload).hexdigest()
        checksum2 = hashlib.sha256(payload).hexdigest()

        assert checksum1 == checksum2
