"""Tests for SPEC §5.0.2 provider capabilities registry."""

from __future__ import annotations


from liminallm.config import (
    AdapterMode,
    BACKEND_ADAPTER_COMPATIBILITY,
    DEFAULT_PROVIDER_CAPABILITIES,
    PROVIDER_CAPABILITIES,
    ProviderCapabilities,
    RemoteStyle,
    get_compatible_adapter_modes,
    get_provider_capabilities,
)


# ==============================================================================
# RemoteStyle Enum Tests
# ==============================================================================


class TestRemoteStyle:
    """Test RemoteStyle enum values."""

    def test_model_id_style(self):
        """MODEL_ID style is for fine-tuned model endpoints."""
        assert RemoteStyle.MODEL_ID == "model_id"

    def test_adapter_param_style(self):
        """ADAPTER_PARAM style is for adapter_id in request body."""
        assert RemoteStyle.ADAPTER_PARAM == "adapter_param"

    def test_none_style(self):
        """NONE style is for local-only providers."""
        assert RemoteStyle.NONE == "none"

    def test_all_styles_are_strings(self):
        """All RemoteStyle values should be string enum members."""
        for style in RemoteStyle:
            assert isinstance(style.value, str)


# ==============================================================================
# AdapterMode Enum Tests
# ==============================================================================


class TestAdapterMode:
    """Test AdapterMode enum values."""

    def test_local_mode(self):
        """LOCAL mode for filesystem-backed weights."""
        assert AdapterMode.LOCAL == "local"

    def test_remote_mode(self):
        """REMOTE mode for externally hosted adapters."""
        assert AdapterMode.REMOTE == "remote"

    def test_prompt_mode(self):
        """PROMPT mode for system prompt injection."""
        assert AdapterMode.PROMPT == "prompt"

    def test_hybrid_mode(self):
        """HYBRID mode for local + prompt fallback."""
        assert AdapterMode.HYBRID == "hybrid"


# ==============================================================================
# ProviderCapabilities Tests
# ==============================================================================


class TestProviderCapabilities:
    """Test ProviderCapabilities dataclass."""

    def test_model_id_provider_capabilities(self):
        """MODEL_ID providers should be single-adapter."""
        caps = ProviderCapabilities(
            remote_style=RemoteStyle.MODEL_ID,
            multi_adapter=False,
            gate_weights=False,
            max_adapters=1,
        )
        assert caps.remote_style == RemoteStyle.MODEL_ID
        assert caps.multi_adapter is False
        assert caps.max_adapters == 1

    def test_adapter_param_provider_capabilities(self):
        """ADAPTER_PARAM providers can support multi-adapter."""
        caps = ProviderCapabilities(
            remote_style=RemoteStyle.ADAPTER_PARAM,
            multi_adapter=True,
            gate_weights=True,
            max_adapters=3,
            adapter_param_name="lora_id",
        )
        assert caps.remote_style == RemoteStyle.ADAPTER_PARAM
        assert caps.multi_adapter is True
        assert caps.gate_weights is True
        assert caps.max_adapters == 3
        assert caps.adapter_param_name == "lora_id"

    def test_default_values(self):
        """Test default field values."""
        caps = ProviderCapabilities(
            remote_style=RemoteStyle.MODEL_ID,
            multi_adapter=False,
            gate_weights=False,
            max_adapters=1,
        )
        # Check defaults
        assert caps.adapter_param_name == "adapter_id"
        assert caps.supports_streaming is True
        assert caps.model_id_prefix == ""


# ==============================================================================
# PROVIDER_CAPABILITIES Registry Tests
# ==============================================================================


class TestProviderCapabilitiesRegistry:
    """Test the provider capabilities registry."""

    def test_openai_capabilities(self):
        """OpenAI uses MODEL_ID style, single adapter."""
        caps = PROVIDER_CAPABILITIES["openai"]
        assert caps.remote_style == RemoteStyle.MODEL_ID
        assert caps.multi_adapter is False
        assert caps.max_adapters == 1
        assert caps.model_id_prefix == "ft:"

    def test_azure_capabilities(self):
        """Azure uses MODEL_ID style, single adapter."""
        caps = PROVIDER_CAPABILITIES["azure"]
        assert caps.remote_style == RemoteStyle.MODEL_ID
        assert caps.multi_adapter is False
        assert caps.max_adapters == 1

    def test_azure_openai_capabilities(self):
        """Azure OpenAI uses MODEL_ID style, single adapter."""
        caps = PROVIDER_CAPABILITIES["azure_openai"]
        assert caps.remote_style == RemoteStyle.MODEL_ID
        assert caps.multi_adapter is False

    def test_vertex_capabilities(self):
        """Vertex uses MODEL_ID style."""
        caps = PROVIDER_CAPABILITIES["vertex"]
        assert caps.remote_style == RemoteStyle.MODEL_ID
        assert caps.multi_adapter is False

    def test_gemini_capabilities(self):
        """Gemini uses MODEL_ID style."""
        caps = PROVIDER_CAPABILITIES["gemini"]
        assert caps.remote_style == RemoteStyle.MODEL_ID
        assert caps.multi_adapter is False

    def test_bedrock_capabilities(self):
        """Bedrock uses MODEL_ID style."""
        caps = PROVIDER_CAPABILITIES["bedrock"]
        assert caps.remote_style == RemoteStyle.MODEL_ID
        assert caps.multi_adapter is False

    def test_together_capabilities(self):
        """Together uses ADAPTER_PARAM style with multi-adapter."""
        caps = PROVIDER_CAPABILITIES["together"]
        assert caps.remote_style == RemoteStyle.ADAPTER_PARAM
        assert caps.multi_adapter is True
        assert caps.gate_weights is True
        assert caps.max_adapters == 3
        assert caps.adapter_param_name == "adapter_id"

    def test_together_ai_alias(self):
        """together.ai should have same capabilities as together."""
        caps = PROVIDER_CAPABILITIES["together.ai"]
        assert caps.remote_style == RemoteStyle.ADAPTER_PARAM
        assert caps.multi_adapter is True

    def test_lorax_capabilities(self):
        """LoRAX uses ADAPTER_PARAM style with multi-adapter."""
        caps = PROVIDER_CAPABILITIES["lorax"]
        assert caps.remote_style == RemoteStyle.ADAPTER_PARAM
        assert caps.multi_adapter is True
        assert caps.gate_weights is True
        assert caps.max_adapters == 5

    def test_adapter_server_capabilities(self):
        """Adapter server uses ADAPTER_PARAM style."""
        caps = PROVIDER_CAPABILITIES["adapter_server"]
        assert caps.remote_style == RemoteStyle.ADAPTER_PARAM
        assert caps.multi_adapter is True

    def test_sagemaker_capabilities(self):
        """SageMaker uses ADAPTER_PARAM style, single adapter."""
        caps = PROVIDER_CAPABILITIES["sagemaker"]
        assert caps.remote_style == RemoteStyle.ADAPTER_PARAM
        assert caps.multi_adapter is False
        assert caps.max_adapters == 1

    def test_local_lora_capabilities(self):
        """Local LoRA uses NONE remote style (local weights)."""
        caps = PROVIDER_CAPABILITIES["local_lora"]
        assert caps.remote_style == RemoteStyle.NONE
        assert caps.multi_adapter is True
        assert caps.gate_weights is True

    def test_local_gpu_lora_capabilities(self):
        """Local GPU LoRA uses NONE remote style."""
        caps = PROVIDER_CAPABILITIES["local_gpu_lora"]
        assert caps.remote_style == RemoteStyle.NONE
        assert caps.multi_adapter is True


# ==============================================================================
# get_provider_capabilities Tests
# ==============================================================================


class TestGetProviderCapabilities:
    """Test get_provider_capabilities function."""

    def test_known_provider(self):
        """Should return capabilities for known provider."""
        caps = get_provider_capabilities("openai")
        assert caps.remote_style == RemoteStyle.MODEL_ID

    def test_case_insensitive(self):
        """Provider lookup should be case-insensitive."""
        caps_lower = get_provider_capabilities("openai")
        caps_upper = get_provider_capabilities("OPENAI")
        caps_mixed = get_provider_capabilities("OpenAI")

        assert caps_lower == caps_upper == caps_mixed

    def test_unknown_provider_returns_default(self):
        """Unknown provider should return conservative defaults."""
        caps = get_provider_capabilities("unknown_provider_xyz")
        assert caps == DEFAULT_PROVIDER_CAPABILITIES
        assert caps.remote_style == RemoteStyle.MODEL_ID
        assert caps.multi_adapter is False
        assert caps.max_adapters == 1


# ==============================================================================
# BACKEND_ADAPTER_COMPATIBILITY Tests
# ==============================================================================


class TestBackendAdapterCompatibility:
    """Test backend-to-adapter-mode compatibility mapping."""

    def test_openai_compatible_modes(self):
        """OpenAI should support REMOTE, PROMPT, HYBRID."""
        modes = BACKEND_ADAPTER_COMPATIBILITY["openai"]
        assert AdapterMode.REMOTE in modes
        assert AdapterMode.PROMPT in modes
        assert AdapterMode.HYBRID in modes
        assert AdapterMode.LOCAL not in modes

    def test_api_backends_dont_support_local(self):
        """API backends shouldn't support LOCAL mode."""
        api_backends = ["openai", "azure", "vertex", "gemini", "bedrock", "together"]
        for backend in api_backends:
            modes = BACKEND_ADAPTER_COMPATIBILITY[backend]
            assert AdapterMode.LOCAL not in modes

    def test_local_backends_support_local(self):
        """Local backends should support LOCAL mode."""
        local_backends = ["local_lora", "local_gpu_lora"]
        for backend in local_backends:
            modes = BACKEND_ADAPTER_COMPATIBILITY[backend]
            assert AdapterMode.LOCAL in modes
            assert AdapterMode.REMOTE not in modes

    def test_all_backends_support_prompt(self):
        """All backends should support PROMPT mode as fallback."""
        for backend, modes in BACKEND_ADAPTER_COMPATIBILITY.items():
            assert AdapterMode.PROMPT in modes, f"{backend} should support PROMPT mode"

    def test_all_backends_support_hybrid(self):
        """All backends should support HYBRID mode."""
        for backend, modes in BACKEND_ADAPTER_COMPATIBILITY.items():
            assert AdapterMode.HYBRID in modes, f"{backend} should support HYBRID mode"


# ==============================================================================
# get_compatible_adapter_modes Tests
# ==============================================================================


class TestGetCompatibleAdapterModes:
    """Test get_compatible_adapter_modes function."""

    def test_known_backend(self):
        """Should return modes for known backend."""
        modes = get_compatible_adapter_modes("openai")
        assert AdapterMode.REMOTE in modes
        assert AdapterMode.PROMPT in modes

    def test_case_insensitive(self):
        """Backend lookup should be case-insensitive."""
        modes_lower = get_compatible_adapter_modes("openai")
        modes_upper = get_compatible_adapter_modes("OPENAI")
        assert modes_lower == modes_upper

    def test_unknown_backend_defaults_to_prompt(self):
        """Unknown backend should default to PROMPT mode only."""
        modes = get_compatible_adapter_modes("unknown_backend")
        assert modes == {AdapterMode.PROMPT}


# ==============================================================================
# Cross-Module Consistency Tests
# ==============================================================================


class TestProviderConsistency:
    """Test consistency between capability and compatibility definitions."""

    def test_model_id_providers_are_single_adapter(self):
        """Providers with MODEL_ID style must have max_adapters=1."""
        for provider, caps in PROVIDER_CAPABILITIES.items():
            if caps.remote_style == RemoteStyle.MODEL_ID:
                assert caps.max_adapters == 1, f"{provider} should have max_adapters=1"
                assert (
                    caps.multi_adapter is False
                ), f"{provider} should have multi_adapter=False"

    def test_adapter_param_providers_have_param_name(self):
        """ADAPTER_PARAM providers should have adapter_param_name set."""
        for provider, caps in PROVIDER_CAPABILITIES.items():
            if caps.remote_style == RemoteStyle.ADAPTER_PARAM:
                assert (
                    caps.adapter_param_name
                ), f"{provider} should have adapter_param_name"

    def test_local_providers_have_none_style(self):
        """Local providers should have NONE remote style."""
        local_providers = ["local_lora", "local_gpu_lora"]
        for provider in local_providers:
            caps = PROVIDER_CAPABILITIES[provider]
            assert caps.remote_style == RemoteStyle.NONE

    def test_gate_weights_requires_multi_adapter(self):
        """gate_weights only makes sense with multi_adapter."""
        for provider, caps in PROVIDER_CAPABILITIES.items():
            if caps.gate_weights:
                # If gate_weights is True, should also have multi_adapter or max_adapters > 1
                assert (
                    caps.multi_adapter or caps.max_adapters > 1
                ), f"{provider} has gate_weights but no multi-adapter support"
