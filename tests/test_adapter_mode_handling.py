"""Tests for SPEC §5 adapter mode handling in model backends."""

from __future__ import annotations

import pytest

from liminallm.config import AdapterMode, RemoteStyle, get_provider_capabilities
from liminallm.service.model_backend import (
    ApiAdapterBackend,
    filter_adapters_by_mode,
    get_adapter_mode,
)


# ==============================================================================
# get_adapter_mode Tests
# ==============================================================================


class TestGetAdapterMode:
    """Test adapter mode inference from schema."""

    def test_explicit_mode_field(self):
        """Should use explicit mode field when present."""
        adapter = {"id": "a1", "mode": AdapterMode.REMOTE}
        assert get_adapter_mode(adapter) == AdapterMode.REMOTE

    def test_explicit_mode_in_schema(self):
        """Should check schema dict for mode field."""
        adapter = {"id": "a1", "schema": {"mode": AdapterMode.LOCAL}}
        assert get_adapter_mode(adapter) == AdapterMode.LOCAL

    def test_infer_prompt_from_backend(self):
        """Should infer PROMPT from prompt backend."""
        adapter = {"id": "a1", "backend": "prompt"}
        assert get_adapter_mode(adapter) == AdapterMode.PROMPT

        adapter2 = {"id": "a2", "backend": "prompt_distill"}
        assert get_adapter_mode(adapter2) == AdapterMode.PROMPT

    def test_infer_local_from_backend(self):
        """Should infer LOCAL from local backend."""
        adapter = {"id": "a1", "backend": "local"}
        assert get_adapter_mode(adapter) == AdapterMode.LOCAL

        adapter2 = {"id": "a2", "backend": "local_lora"}
        assert get_adapter_mode(adapter2) == AdapterMode.LOCAL

    def test_infer_local_from_provider(self):
        """Should infer LOCAL from local provider."""
        adapter = {"id": "a1", "provider": "local"}
        assert get_adapter_mode(adapter) == AdapterMode.LOCAL

    def test_infer_hybrid_from_local_with_prompt(self):
        """Should infer HYBRID when local with prompt instructions."""
        adapter = {"id": "a1", "backend": "local", "prompt_instructions": "Be helpful"}
        assert get_adapter_mode(adapter) == AdapterMode.HYBRID

        adapter2 = {"id": "a2", "provider": "local", "behavior_prompt": "Act as expert"}
        assert get_adapter_mode(adapter2) == AdapterMode.HYBRID

    def test_infer_remote_from_backend(self):
        """Should infer REMOTE from api/remote backend."""
        adapter = {"id": "a1", "backend": "api"}
        assert get_adapter_mode(adapter) == AdapterMode.REMOTE

        adapter2 = {"id": "a2", "backend": "remote"}
        assert get_adapter_mode(adapter2) == AdapterMode.REMOTE

    def test_infer_remote_from_model_id(self):
        """Should infer REMOTE when remote_model_id present."""
        adapter = {"id": "a1", "remote_model_id": "ft:gpt-4:custom"}
        assert get_adapter_mode(adapter) == AdapterMode.REMOTE

    def test_infer_hybrid_from_backend_field(self):
        """Should recognize explicit hybrid backend."""
        adapter = {"id": "a1", "backend": "hybrid"}
        assert get_adapter_mode(adapter) == AdapterMode.HYBRID

    def test_default_to_hybrid(self):
        """Should default to HYBRID for backwards compatibility."""
        adapter = {"id": "a1"}
        assert get_adapter_mode(adapter) == AdapterMode.HYBRID

    def test_empty_adapter_returns_fallback(self):
        """Empty/None adapter should return fallback mode."""
        # None returns PROMPT (guard at beginning)
        assert get_adapter_mode(None) == AdapterMode.PROMPT
        # Empty dict goes through inference and finds no fields to match,
        # so check what it actually returns (implementation may vary)
        mode = get_adapter_mode({})
        assert mode in [AdapterMode.PROMPT, AdapterMode.HYBRID]

    def test_case_insensitive(self):
        """Backend/provider matching should be case-insensitive."""
        adapter = {"id": "a1", "backend": "LOCAL"}
        assert get_adapter_mode(adapter) == AdapterMode.LOCAL

        adapter2 = {"id": "a2", "backend": "Prompt"}
        assert get_adapter_mode(adapter2) == AdapterMode.PROMPT


# ==============================================================================
# filter_adapters_by_mode Tests
# ==============================================================================


class TestFilterAdaptersByMode:
    """Test adapter filtering by compatible modes."""

    def test_filter_keeps_compatible(self):
        """Should keep adapters with compatible modes."""
        adapters = [
            {"id": "a1", "mode": AdapterMode.REMOTE},
            {"id": "a2", "mode": AdapterMode.PROMPT},
        ]
        compatible = {AdapterMode.REMOTE, AdapterMode.PROMPT, AdapterMode.HYBRID}

        result = filter_adapters_by_mode(adapters, compatible)

        assert len(result) == 2
        assert result[0]["id"] == "a1"
        assert result[1]["id"] == "a2"

    def test_filter_removes_incompatible(self):
        """Should remove adapters with incompatible modes."""
        adapters = [
            {"id": "a1", "mode": AdapterMode.LOCAL},  # incompatible with API
            {"id": "a2", "mode": AdapterMode.REMOTE},
        ]
        compatible = {AdapterMode.REMOTE, AdapterMode.PROMPT, AdapterMode.HYBRID}

        result = filter_adapters_by_mode(adapters, compatible)

        assert len(result) == 1
        assert result[0]["id"] == "a2"

    def test_filter_infers_mode(self):
        """Should infer mode for adapters without explicit mode."""
        adapters = [
            {"id": "a1", "backend": "local"},  # inferred LOCAL
            {"id": "a2", "backend": "api"},  # inferred REMOTE
        ]
        compatible = {AdapterMode.REMOTE, AdapterMode.PROMPT}

        result = filter_adapters_by_mode(adapters, compatible)

        assert len(result) == 1
        assert result[0]["id"] == "a2"

    def test_empty_adapters(self):
        """Should return empty list for empty input."""
        result = filter_adapters_by_mode([], {AdapterMode.PROMPT})
        assert result == []


# ==============================================================================
# ApiAdapterBackend Tests
# ==============================================================================


class TestApiAdapterBackend:
    """Test ApiAdapterBackend capability-aware processing."""

    def test_infer_provider_from_mode(self):
        """Should infer provider from adapter_mode string."""
        backend = ApiAdapterBackend("gpt-4", adapter_mode="openai")
        assert backend.provider == "openai"

        backend2 = ApiAdapterBackend("meta-llama", adapter_mode="together")
        assert backend2.provider == "together"

        backend3 = ApiAdapterBackend("custom", adapter_mode="lorax")
        assert backend3.provider == "lorax"

    def test_explicit_provider_overrides_inference(self):
        """Explicit provider should override inference."""
        backend = ApiAdapterBackend("gpt-4", adapter_mode="api", provider="azure")
        assert backend.provider == "azure"

    def test_capabilities_loaded(self):
        """Should load capabilities for provider."""
        backend = ApiAdapterBackend("gpt-4", provider="openai")
        assert backend.capabilities.remote_style == RemoteStyle.MODEL_ID
        assert backend.capabilities.max_adapters == 1

        backend2 = ApiAdapterBackend("meta-llama", provider="together")
        assert backend2.capabilities.remote_style == RemoteStyle.ADAPTER_PARAM
        assert backend2.capabilities.max_adapters == 3

    def test_compatible_modes(self):
        """Should define compatible modes for API backend."""
        assert AdapterMode.REMOTE in ApiAdapterBackend.COMPATIBLE_MODES
        assert AdapterMode.PROMPT in ApiAdapterBackend.COMPATIBLE_MODES
        assert AdapterMode.HYBRID in ApiAdapterBackend.COMPATIBLE_MODES
        assert AdapterMode.LOCAL not in ApiAdapterBackend.COMPATIBLE_MODES


# ==============================================================================
# _process_adapters_for_provider Tests
# ==============================================================================


class TestProcessAdaptersForProvider:
    """Test adapter processing based on provider capabilities."""

    def test_local_adapters_dropped_in_api_mode(self):
        """LOCAL adapters should be dropped for API backends."""
        backend = ApiAdapterBackend("gpt-4", provider="openai")

        adapters = [{"id": "a1", "mode": AdapterMode.LOCAL}]
        result = backend._process_adapters_for_provider(adapters)

        assert "a1" in result["dropped"]
        assert len(result["applied"]) == 0

    def test_prompt_adapters_inject_instructions(self):
        """PROMPT adapters should inject prompt instructions."""
        backend = ApiAdapterBackend("gpt-4", provider="openai")

        adapters = [
            {
                "id": "a1",
                "mode": AdapterMode.PROMPT,
                "prompt_instructions": "Be concise and helpful",
            }
        ]
        result = backend._process_adapters_for_provider(adapters)

        assert "a1:prompt" in result["applied"]
        assert "Be concise and helpful" in result["prompt_injections"]

    def test_hybrid_adapters_extract_prompt(self):
        """HYBRID adapters should extract prompt and optionally add to remote."""
        backend = ApiAdapterBackend("gpt-4", provider="openai")

        adapters = [
            {
                "id": "a1",
                "mode": AdapterMode.HYBRID,
                "prompt_instructions": "Expert mode",
            }
        ]
        result = backend._process_adapters_for_provider(adapters)

        # Prompt should be extracted
        assert "Expert mode" in result["prompt_injections"]
        # Without remote_model_id, should only apply as prompt
        assert "a1:prompt" in result["applied"]

    def test_hybrid_with_remote_model_id(self):
        """HYBRID adapter with remote_model_id should be added to remote list."""
        backend = ApiAdapterBackend("gpt-4", provider="openai")

        adapters = [
            {
                "id": "a1",
                "mode": AdapterMode.HYBRID,
                "prompt_instructions": "Expert mode",
                "remote_model_id": "ft:gpt-4:custom",
            }
        ]
        result = backend._process_adapters_for_provider(adapters)

        assert "Expert mode" in result["prompt_injections"]
        assert "a1:hybrid" in result["applied"]

    def test_remote_adapters_added_to_list(self):
        """REMOTE adapters should be added to remote adapter list."""
        backend = ApiAdapterBackend("gpt-4", provider="openai")

        adapters = [
            {
                "id": "a1",
                "mode": AdapterMode.REMOTE,
                "remote_model_id": "ft:gpt-4:org",
            }
        ]
        result = backend._process_adapters_for_provider(adapters)

        assert "a1:model_id" in result["applied"]
        assert result["model"] == "ft:gpt-4:org"


# ==============================================================================
# _format_remote_adapters Tests
# ==============================================================================


class TestFormatRemoteAdapters:
    """Test remote adapter formatting based on provider remote_style."""

    def test_model_id_style_single_adapter(self):
        """MODEL_ID style should use remote_model_id as model."""
        backend = ApiAdapterBackend("gpt-4", provider="openai")

        adapters = [{"id": "a1", "remote_model_id": "ft:gpt-4:custom"}]
        model, extra_body, applied, dropped = backend._format_remote_adapters(adapters)

        assert model == "ft:gpt-4:custom"
        assert extra_body is None
        assert "a1:model_id" in applied
        assert len(dropped) == 0

    def test_model_id_style_multiple_drops_extra(self):
        """MODEL_ID style should drop extra adapters beyond first."""
        backend = ApiAdapterBackend("gpt-4", provider="openai")

        adapters = [
            {"id": "a1", "remote_model_id": "ft:gpt-4:first", "weight": 0.9},
            {"id": "a2", "remote_model_id": "ft:gpt-4:second", "weight": 0.8},
        ]
        model, extra_body, applied, dropped = backend._format_remote_adapters(adapters)

        # Should pick highest weight
        assert model == "ft:gpt-4:first"
        assert "a1:model_id" in applied
        assert "a2" in dropped

    def test_adapter_param_style_single(self):
        """ADAPTER_PARAM style should use adapter_id parameter."""
        backend = ApiAdapterBackend("meta-llama", provider="together")

        adapters = [{"id": "a1", "remote_adapter_id": "lora-123"}]
        model, extra_body, applied, dropped = backend._format_remote_adapters(adapters)

        assert model == "meta-llama"
        assert extra_body is not None
        assert extra_body["adapter_id"] == "lora-123"
        assert "a1:adapter_param" in applied

    def test_adapter_param_style_multiple(self):
        """ADAPTER_PARAM style should support multiple adapters."""
        backend = ApiAdapterBackend("meta-llama", provider="together")

        adapters = [
            {"id": "a1", "remote_adapter_id": "lora-1", "weight": 0.7},
            {"id": "a2", "remote_adapter_id": "lora-2", "weight": 0.3},
        ]
        model, extra_body, applied, dropped = backend._format_remote_adapters(adapters)

        # Together supports multi-adapter
        assert extra_body is not None
        assert isinstance(extra_body["adapter_id"], list)
        assert "lora-1" in extra_body["adapter_id"]
        assert "lora-2" in extra_body["adapter_id"]
        # With gate weights
        assert "adapter_weights" in extra_body

    def test_adapter_param_respects_max_adapters(self):
        """Should respect max_adapters limit."""
        backend = ApiAdapterBackend("meta-llama", provider="together")

        # Together has max_adapters=3
        adapters = [
            {"id": f"a{i}", "remote_adapter_id": f"lora-{i}", "weight": 1.0 / (i + 1)}
            for i in range(5)
        ]
        model, extra_body, applied, dropped = backend._format_remote_adapters(adapters)

        # Should only use top 3
        assert len([a for a in applied if "adapter_param" in a]) == 3
        assert len(dropped) == 2

    def test_empty_adapters_returns_base_model(self):
        """Empty adapter list should return base model."""
        backend = ApiAdapterBackend("gpt-4", provider="openai")

        model, extra_body, applied, dropped = backend._format_remote_adapters([])

        assert model == "gpt-4"
        assert extra_body is None
        assert applied == []
        assert dropped == []

    def test_no_valid_remote_id_falls_back(self):
        """Adapters without remote IDs should fall back to base model."""
        backend = ApiAdapterBackend("gpt-4", provider="openai")

        adapters = [{"id": "a1"}]  # No remote_model_id
        model, extra_body, applied, dropped = backend._format_remote_adapters(adapters)

        assert model == "gpt-4"
        assert "a1" in dropped


# ==============================================================================
# _select_best_adapter Tests
# ==============================================================================


class TestSelectBestAdapter:
    """Test adapter selection by weight."""

    def test_sorts_by_weight_descending(self):
        """Should sort adapters by weight, highest first."""
        backend = ApiAdapterBackend("gpt-4", provider="openai")

        adapters = [
            {"id": "low", "weight": 0.2},
            {"id": "high", "weight": 0.9},
            {"id": "mid", "weight": 0.5},
        ]
        result = backend._select_best_adapter(adapters, max_count=3)

        assert result[0]["id"] == "high"
        assert result[1]["id"] == "mid"
        assert result[2]["id"] == "low"

    def test_respects_max_count(self):
        """Should limit to max_count adapters."""
        backend = ApiAdapterBackend("gpt-4", provider="openai")

        adapters = [
            {"id": "a1", "weight": 0.9},
            {"id": "a2", "weight": 0.8},
            {"id": "a3", "weight": 0.7},
        ]
        result = backend._select_best_adapter(adapters, max_count=2)

        assert len(result) == 2
        assert result[0]["id"] == "a1"
        assert result[1]["id"] == "a2"

    def test_handles_gate_weight_field(self):
        """Should recognize gate_weight as alternative to weight."""
        backend = ApiAdapterBackend("gpt-4", provider="openai")

        adapters = [
            {"id": "a1", "gate_weight": 0.3},
            {"id": "a2", "gate_weight": 0.7},
        ]
        result = backend._select_best_adapter(adapters, max_count=2)

        assert result[0]["id"] == "a2"  # Higher gate_weight

    def test_defaults_weight_to_1(self):
        """Should default weight to 1.0 if not specified."""
        backend = ApiAdapterBackend("gpt-4", provider="openai")

        adapters = [
            {"id": "explicit", "weight": 0.5},
            {"id": "implicit"},  # No weight
        ]
        result = backend._select_best_adapter(adapters, max_count=2)

        # Implicit (1.0) should be first
        assert result[0]["id"] == "implicit"


# ==============================================================================
# _extract_prompt_instructions Tests
# ==============================================================================


class TestExtractPromptInstructions:
    """Test prompt instruction extraction from adapter.

    Tests SPEC §5.0.1 prompt instruction handling with explicit priority order.
    """

    def test_extract_prompt_instructions(self):
        """Should extract from prompt_instructions field."""
        backend = ApiAdapterBackend("gpt-4", provider="openai")

        adapter = {"prompt_instructions": "Be helpful"}
        result = backend._extract_prompt_instructions(adapter)

        assert result == "Be helpful"

    def test_extract_behavior_prompt(self):
        """Should extract from behavior_prompt field."""
        backend = ApiAdapterBackend("gpt-4", provider="openai")

        adapter = {"behavior_prompt": "Act as expert"}
        result = backend._extract_prompt_instructions(adapter)

        assert result == "Act as expert"

    def test_extract_system_prompt(self):
        """Should extract from system_prompt field."""
        backend = ApiAdapterBackend("gpt-4", provider="openai")

        adapter = {"system_prompt": "You are a coding assistant"}
        result = backend._extract_prompt_instructions(adapter)

        assert result == "You are a coding assistant"

    def test_extract_from_schema(self):
        """Should check schema dict for instructions."""
        backend = ApiAdapterBackend("gpt-4", provider="openai")

        adapter = {"schema": {"prompt_instructions": "From schema"}}
        result = backend._extract_prompt_instructions(adapter)

        assert result == "From schema"

    def test_extract_from_applicability(self):
        """Should extract from applicability.natural_language."""
        backend = ApiAdapterBackend("gpt-4", provider="openai")

        adapter = {"applicability": {"natural_language": "For technical questions"}}
        result = backend._extract_prompt_instructions(adapter)

        assert result == "For technical questions"

    def test_no_fallback_to_description_without_flag(self):
        """Should NOT fall back to description without explicit flag.

        This is a key change from the original behavior - generic description
        fields should not be used for prompt injection unless explicitly opted in.
        """
        backend = ApiAdapterBackend("gpt-4", provider="openai")

        adapter = {"description": "A helpful assistant"}
        result = backend._extract_prompt_instructions(adapter)

        # Description should NOT be used without opt-in flag
        assert result is None

    def test_description_used_with_explicit_flag(self):
        """Should use description when use_description_as_prompt=True."""
        backend = ApiAdapterBackend("gpt-4", provider="openai")

        adapter = {
            "description": "A helpful assistant",
            "use_description_as_prompt": True,
        }
        result = backend._extract_prompt_instructions(adapter)

        assert result == "A helpful assistant"

    def test_description_flag_in_schema(self):
        """Should check schema for use_description_as_prompt flag."""
        backend = ApiAdapterBackend("gpt-4", provider="openai")

        adapter = {
            "schema": {
                "description": "Expert coder",
                "use_description_as_prompt": True,
            }
        }
        result = backend._extract_prompt_instructions(adapter)

        assert result == "Expert coder"

    def test_priority_order(self):
        """Should respect priority order: prompt_instructions > applicability > description."""
        backend = ApiAdapterBackend("gpt-4", provider="openai")

        # When all fields present, prompt_instructions wins
        adapter = {
            "prompt_instructions": "Priority 1",
            "applicability": {"natural_language": "Priority 2"},
            "description": "Priority 3",
            "use_description_as_prompt": True,
        }
        result = backend._extract_prompt_instructions(adapter)
        assert result == "Priority 1"

        # Without prompt_instructions, applicability wins
        adapter2 = {
            "applicability": {"natural_language": "Priority 2"},
            "description": "Priority 3",
            "use_description_as_prompt": True,
        }
        result2 = backend._extract_prompt_instructions(adapter2)
        assert result2 == "Priority 2"

    def test_returns_none_for_empty(self):
        """Should return None if no instructions found."""
        backend = ApiAdapterBackend("gpt-4", provider="openai")

        result = backend._extract_prompt_instructions({})
        assert result is None

    def test_strips_whitespace(self):
        """Should strip whitespace from instructions."""
        backend = ApiAdapterBackend("gpt-4", provider="openai")

        adapter = {"prompt_instructions": "  padded text  "}
        result = backend._extract_prompt_instructions(adapter)

        assert result == "padded text"

    def test_ignores_empty_strings(self):
        """Should ignore empty or whitespace-only values."""
        backend = ApiAdapterBackend("gpt-4", provider="openai")

        adapter = {
            "prompt_instructions": "   ",
            "behavior_prompt": "",
            "applicability": {"natural_language": "Valid"},
        }
        result = backend._extract_prompt_instructions(adapter)

        assert result == "Valid"


# ==============================================================================
# _inject_adapter_prompts Tests
# ==============================================================================


class TestInjectAdapterPrompts:
    """Test adapter prompt injection into messages."""

    def test_injects_into_existing_system_message(self):
        """Should append to existing system message."""
        backend = ApiAdapterBackend("gpt-4", provider="openai")

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        prompts = ["Be concise"]

        result = backend._inject_adapter_prompts(messages, prompts)

        assert len(result) == 2
        assert "You are helpful." in result[0]["content"]
        assert "Adapter guidance" in result[0]["content"]
        assert "Be concise" in result[0]["content"]

    def test_prepends_system_message_if_missing(self):
        """Should prepend system message if none exists."""
        backend = ApiAdapterBackend("gpt-4", provider="openai")

        messages = [{"role": "user", "content": "Hello"}]
        prompts = ["Be helpful"]

        result = backend._inject_adapter_prompts(messages, prompts)

        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert "Adapter guidance" in result[0]["content"]
        assert "Be helpful" in result[0]["content"]

    def test_multiple_prompts_combined(self):
        """Should combine multiple prompt instructions."""
        backend = ApiAdapterBackend("gpt-4", provider="openai")

        messages = [{"role": "user", "content": "Hello"}]
        prompts = ["Be concise", "Use examples", "Stay on topic"]

        result = backend._inject_adapter_prompts(messages, prompts)

        content = result[0]["content"]
        assert "Be concise" in content
        assert "Use examples" in content
        assert "Stay on topic" in content

    def test_no_injection_for_empty_prompts(self):
        """Should not modify messages if no prompts."""
        backend = ApiAdapterBackend("gpt-4", provider="openai")

        messages = [{"role": "user", "content": "Hello"}]

        result = backend._inject_adapter_prompts(messages, [])

        assert result == messages

    def test_does_not_modify_original(self):
        """Should not modify original messages list."""
        backend = ApiAdapterBackend("gpt-4", provider="openai")

        messages = [
            {"role": "system", "content": "Original"},
            {"role": "user", "content": "Hello"},
        ]
        prompts = ["Added"]

        result = backend._inject_adapter_prompts(messages, prompts)

        # Original should be unchanged
        assert messages[0]["content"] == "Original"
        # Result should be modified
        assert "Added" in result[0]["content"]
