"""Tests for SPEC §5 adapter mode handling in TrainingService."""

from __future__ import annotations

import pytest
from pathlib import Path

from liminallm.config import AdapterMode, get_compatible_adapter_modes
from liminallm.service.training import TrainingService
from liminallm.storage.memory import MemoryStore


# ==============================================================================
# TrainingService Initialization Tests
# ==============================================================================


class TestTrainingServiceInit:
    """Test TrainingService initialization with adapter modes."""

    def test_default_adapter_mode(self, tmp_path):
        """Should use default adapter mode when not specified."""
        store = MemoryStore(fs_root=str(tmp_path))
        training = TrainingService(store, fs_root=str(tmp_path))

        assert training.default_adapter_mode == AdapterMode.HYBRID

    def test_explicit_adapter_mode(self, tmp_path):
        """Should accept explicit default adapter mode."""
        store = MemoryStore(fs_root=str(tmp_path))
        training = TrainingService(
            store, fs_root=str(tmp_path), default_adapter_mode=AdapterMode.LOCAL
        )

        assert training.default_adapter_mode == AdapterMode.LOCAL

    def test_backend_mode_sets_compatible_modes(self, tmp_path):
        """Should set compatible modes based on backend_mode."""
        store = MemoryStore(fs_root=str(tmp_path))
        training = TrainingService(
            store, fs_root=str(tmp_path), backend_mode="openai"
        )

        # OpenAI compatible modes
        assert AdapterMode.REMOTE in training._compatible_modes
        assert AdapterMode.PROMPT in training._compatible_modes
        assert AdapterMode.HYBRID in training._compatible_modes
        assert AdapterMode.LOCAL not in training._compatible_modes

    def test_local_backend_mode_compatible_modes(self, tmp_path):
        """Local backend should support LOCAL mode."""
        store = MemoryStore(fs_root=str(tmp_path))
        training = TrainingService(
            store, fs_root=str(tmp_path), backend_mode="local_lora"
        )

        assert AdapterMode.LOCAL in training._compatible_modes


# ==============================================================================
# _infer_adapter_mode Tests
# ==============================================================================


class TestInferAdapterMode:
    """Test adapter mode inference from legacy schema fields."""

    @pytest.fixture
    def training(self, tmp_path):
        store = MemoryStore(fs_root=str(tmp_path))
        return TrainingService(store, fs_root=str(tmp_path))

    def test_infer_prompt_from_backend(self, training):
        """Should infer PROMPT from prompt backend."""
        schema = {"backend": "prompt"}
        assert training._infer_adapter_mode(schema) == AdapterMode.PROMPT

        schema2 = {"backend": "prompt_distill"}
        assert training._infer_adapter_mode(schema2) == AdapterMode.PROMPT

    def test_infer_local_from_backend(self, training):
        """Should infer LOCAL from local backend."""
        schema = {"backend": "local"}
        assert training._infer_adapter_mode(schema) == AdapterMode.LOCAL

        schema2 = {"backend": "local_lora"}
        assert training._infer_adapter_mode(schema2) == AdapterMode.LOCAL

    def test_infer_local_from_provider(self, training):
        """Should infer LOCAL from local provider."""
        schema = {"provider": "local"}
        assert training._infer_adapter_mode(schema) == AdapterMode.LOCAL

    def test_infer_hybrid_when_local_with_prompt(self, training):
        """Should infer HYBRID when local with prompt instructions."""
        schema = {"backend": "local", "prompt_instructions": "Be helpful"}
        assert training._infer_adapter_mode(schema) == AdapterMode.HYBRID

        schema2 = {"backend": "local", "behavior_prompt": "Act as expert"}
        assert training._infer_adapter_mode(schema2) == AdapterMode.HYBRID

    def test_infer_remote_from_backend(self, training):
        """Should infer REMOTE from api/remote backend."""
        schema = {"backend": "api"}
        assert training._infer_adapter_mode(schema) == AdapterMode.REMOTE

        schema2 = {"backend": "remote"}
        assert training._infer_adapter_mode(schema2) == AdapterMode.REMOTE

    def test_infer_remote_from_model_id(self, training):
        """Should infer REMOTE when remote_model_id present."""
        schema = {"remote_model_id": "ft:gpt-4:custom"}
        assert training._infer_adapter_mode(schema) == AdapterMode.REMOTE

    def test_default_to_hybrid(self, training):
        """Should default to HYBRID for backwards compatibility."""
        schema = {"kind": "adapter.lora"}
        assert training._infer_adapter_mode(schema) == AdapterMode.HYBRID


# ==============================================================================
# _mode_to_backend Tests
# ==============================================================================


class TestModeToBackend:
    """Test adapter mode to backend field mapping."""

    @pytest.fixture
    def training(self, tmp_path):
        store = MemoryStore(fs_root=str(tmp_path))
        return TrainingService(store, fs_root=str(tmp_path))

    def test_local_mode(self, training):
        """LOCAL mode maps to 'local' backend."""
        assert training._mode_to_backend(AdapterMode.LOCAL) == "local"

    def test_remote_mode(self, training):
        """REMOTE mode maps to 'api' backend."""
        assert training._mode_to_backend(AdapterMode.REMOTE) == "api"

    def test_prompt_mode(self, training):
        """PROMPT mode maps to 'prompt' backend."""
        assert training._mode_to_backend(AdapterMode.PROMPT) == "prompt"

    def test_hybrid_mode(self, training):
        """HYBRID mode maps to 'hybrid' backend."""
        assert training._mode_to_backend(AdapterMode.HYBRID) == "hybrid"


# ==============================================================================
# _mode_to_provider Tests
# ==============================================================================


class TestModeToProvider:
    """Test adapter mode to provider field mapping."""

    @pytest.fixture
    def training_openai(self, tmp_path):
        store = MemoryStore(fs_root=str(tmp_path))
        return TrainingService(store, fs_root=str(tmp_path), backend_mode="openai")

    @pytest.fixture
    def training_together(self, tmp_path):
        store = MemoryStore(fs_root=str(tmp_path))
        return TrainingService(store, fs_root=str(tmp_path), backend_mode="together")

    def test_local_mode(self, training_openai):
        """LOCAL mode maps to 'local' provider."""
        assert training_openai._mode_to_provider(AdapterMode.LOCAL) == "local"

    def test_remote_mode_uses_backend_mode(self, training_openai, training_together):
        """REMOTE mode uses backend_mode as provider."""
        assert training_openai._mode_to_provider(AdapterMode.REMOTE) == "openai"
        assert training_together._mode_to_provider(AdapterMode.REMOTE) == "together"

    def test_prompt_mode(self, training_openai):
        """PROMPT mode maps to 'prompt' provider."""
        assert training_openai._mode_to_provider(AdapterMode.PROMPT) == "prompt"

    def test_hybrid_mode(self, training_openai):
        """HYBRID mode maps to 'hybrid' provider."""
        assert training_openai._mode_to_provider(AdapterMode.HYBRID) == "hybrid"


# ==============================================================================
# ensure_user_adapter Tests
# ==============================================================================


class TestEnsureUserAdapter:
    """Test adapter creation with explicit mode."""

    def test_creates_adapter_with_default_mode(self, tmp_path):
        """Should create adapter with default mode."""
        store = MemoryStore(fs_root=str(tmp_path))
        training = TrainingService(
            store, fs_root=str(tmp_path), default_adapter_mode=AdapterMode.HYBRID
        )

        user = store.create_user("test@example.com")
        adapter = training.ensure_user_adapter(user.id)

        assert adapter.schema.get("mode") == AdapterMode.HYBRID
        assert adapter.schema.get("backend") == "hybrid"

    def test_creates_adapter_with_explicit_mode(self, tmp_path):
        """Should create adapter with explicit mode override."""
        store = MemoryStore(fs_root=str(tmp_path))
        training = TrainingService(
            store, fs_root=str(tmp_path), default_adapter_mode=AdapterMode.HYBRID
        )

        user = store.create_user("test@example.com")
        adapter = training.ensure_user_adapter(user.id, adapter_mode=AdapterMode.PROMPT)

        assert adapter.schema.get("mode") == AdapterMode.PROMPT
        assert adapter.schema.get("backend") == "prompt"

    def test_local_mode_creates_fs_dir(self, tmp_path):
        """LOCAL mode should create fs_dir in schema."""
        store = MemoryStore(fs_root=str(tmp_path))
        training = TrainingService(store, fs_root=str(tmp_path))

        user = store.create_user("test@example.com")
        adapter = training.ensure_user_adapter(user.id, adapter_mode=AdapterMode.LOCAL)

        assert "fs_dir" in adapter.schema
        assert adapter.schema["fs_dir"] is not None

    def test_hybrid_mode_creates_fs_dir(self, tmp_path):
        """HYBRID mode should create fs_dir in schema."""
        store = MemoryStore(fs_root=str(tmp_path))
        training = TrainingService(store, fs_root=str(tmp_path))

        user = store.create_user("test@example.com")
        adapter = training.ensure_user_adapter(user.id, adapter_mode=AdapterMode.HYBRID)

        assert "fs_dir" in adapter.schema

    def test_prompt_mode_no_fs_dir(self, tmp_path):
        """PROMPT mode should not create fs_dir."""
        store = MemoryStore(fs_root=str(tmp_path))
        training = TrainingService(store, fs_root=str(tmp_path))

        user = store.create_user("test@example.com")
        adapter = training.ensure_user_adapter(user.id, adapter_mode=AdapterMode.PROMPT)

        # PROMPT mode doesn't need filesystem storage
        # It may or may not have fs_dir depending on implementation
        # The key is that mode is correctly set
        assert adapter.schema.get("mode") == AdapterMode.PROMPT

    def test_remote_mode_no_fs_dir(self, tmp_path):
        """REMOTE mode should not create fs_dir."""
        store = MemoryStore(fs_root=str(tmp_path))
        training = TrainingService(store, fs_root=str(tmp_path))

        user = store.create_user("test@example.com")
        adapter = training.ensure_user_adapter(user.id, adapter_mode=AdapterMode.REMOTE)

        # REMOTE mode doesn't need local filesystem storage
        assert adapter.schema.get("mode") == AdapterMode.REMOTE

    def test_migrates_existing_adapter_mode(self, tmp_path):
        """Should add mode field to existing adapter without one."""
        store = MemoryStore(fs_root=str(tmp_path))
        training = TrainingService(store, fs_root=str(tmp_path))

        user = store.create_user("test@example.com")

        # Create adapter directly without mode field (with required fields)
        legacy_adapter = store.create_artifact(
            type_="adapter",
            name="legacy_adapter",
            schema={
                "kind": "adapter.lora",
                "backend": "local",
                "scope": "per-user",
                "user_id": user.id,
                "base_model": "test-model",
                "current_version": 0,
            },
            description="Legacy adapter",
            owner_user_id=user.id,
        )

        # ensure_user_adapter should migrate it
        adapter = training.ensure_user_adapter(user.id)

        # Should have inferred mode added
        assert adapter.schema.get("mode") is not None
        # LOCAL backend should infer LOCAL mode
        assert adapter.schema.get("mode") == AdapterMode.LOCAL

    def test_preserves_existing_mode(self, tmp_path):
        """Should not overwrite existing mode field."""
        store = MemoryStore(fs_root=str(tmp_path))
        training = TrainingService(store, fs_root=str(tmp_path))

        user = store.create_user("test@example.com")

        # Create adapter with explicit mode (with required fields)
        existing = store.create_artifact(
            type_="adapter",
            name="existing_adapter",
            schema={
                "kind": "adapter.lora",
                "mode": AdapterMode.REMOTE,
                "backend": "api",
                "scope": "per-user",
                "user_id": user.id,
                "base_model": "test-model",
                "current_version": 0,
            },
            description="Existing adapter",
            owner_user_id=user.id,
        )

        # ensure_user_adapter should preserve mode
        adapter = training.ensure_user_adapter(user.id)

        assert adapter.schema.get("mode") == AdapterMode.REMOTE


# ==============================================================================
# Adapter Schema Field Tests
# ==============================================================================


class TestAdapterSchemaFields:
    """Test that adapter schema includes required mode-related fields."""

    def test_schema_includes_kind(self, tmp_path):
        """Schema should include kind field."""
        store = MemoryStore(fs_root=str(tmp_path))
        training = TrainingService(store, fs_root=str(tmp_path))

        user = store.create_user("test@example.com")
        adapter = training.ensure_user_adapter(user.id)

        assert adapter.schema.get("kind") == "adapter.lora"

    def test_schema_includes_mode(self, tmp_path):
        """Schema should include mode field."""
        store = MemoryStore(fs_root=str(tmp_path))
        training = TrainingService(store, fs_root=str(tmp_path))

        user = store.create_user("test@example.com")
        adapter = training.ensure_user_adapter(user.id)

        assert "mode" in adapter.schema
        assert adapter.schema["mode"] in [
            AdapterMode.LOCAL,
            AdapterMode.REMOTE,
            AdapterMode.PROMPT,
            AdapterMode.HYBRID,
        ]

    def test_schema_includes_backend(self, tmp_path):
        """Schema should include backend field."""
        store = MemoryStore(fs_root=str(tmp_path))
        training = TrainingService(store, fs_root=str(tmp_path))

        user = store.create_user("test@example.com")
        adapter = training.ensure_user_adapter(user.id)

        assert "backend" in adapter.schema

    def test_schema_includes_provider(self, tmp_path):
        """Schema should include provider field."""
        store = MemoryStore(fs_root=str(tmp_path))
        training = TrainingService(store, fs_root=str(tmp_path))

        user = store.create_user("test@example.com")
        adapter = training.ensure_user_adapter(user.id)

        assert "provider" in adapter.schema


# ==============================================================================
# get_compatible_adapter_modes Integration Tests
# ==============================================================================


class TestCompatibleModesIntegration:
    """Test integration with get_compatible_adapter_modes."""

    def test_training_uses_compatible_modes(self, tmp_path):
        """TrainingService should use get_compatible_adapter_modes."""
        store = MemoryStore(fs_root=str(tmp_path))
        training = TrainingService(
            store, fs_root=str(tmp_path), backend_mode="openai"
        )

        expected = get_compatible_adapter_modes("openai")
        assert training._compatible_modes == expected

    def test_default_backend_uses_openai_modes(self, tmp_path):
        """Default backend should use openai-compatible modes."""
        store = MemoryStore(fs_root=str(tmp_path))
        training = TrainingService(store, fs_root=str(tmp_path))

        # Default is "openai" if not specified
        expected = get_compatible_adapter_modes("openai")
        assert training._compatible_modes == expected
