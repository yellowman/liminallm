"""The model catalogue: context windows, reasoning families, provider wiring.

The table is a *fallback* — provider discovery and the model_context_window
setting both outrank it — so these tests pin the properties that make it safe
to fall back on, not a frozen copy of every published number.
"""

from __future__ import annotations

import pytest

from liminallm.config import (
    PROVIDER_ENDPOINTS,
    ModelBackend,
    get_provider_capabilities,
    resolve_provider_endpoint,
)
from liminallm.service.model_backend import (
    KNOWN_CONTEXT_WINDOWS,
    context_window_from_table,
    is_reasoning_model,
)


class TestContextWindows:
    @pytest.mark.parametrize("model,window", [
        # Google — verified against a live ListModels call.
        ("gemini-3.6-flash", 1_000_000),
        ("gemini-flash-latest", 1_000_000),
        ("gemini-2.5-pro", 1_000_000),
        ("gemini-2.5-flash-image", 32_768),
        ("gemini-3-pro-image-preview", 131_072),
        # OpenAI — tier splits within one version.
        ("gpt-5.2", 400_000),
        ("gpt-5.6-sol", 1_050_000),
        ("gpt-5.6-terra", 1_050_000),
        ("gpt-5.5-pro", 1_050_000),
        ("gpt-5.4", 1_050_000),
        ("gpt-5.4-pro", 1_050_000),
        ("gpt-5.4-mini", 400_000),
        ("gpt-5.3-codex", 400_000),
        ("gpt-5.2-chat-latest", 128_000),
        ("gpt-4.1-mini", 1_000_000),
        ("gpt-4o-mini", 128_000),
        ("o3-mini", 200_000),
        # Anthropic.
        ("claude-opus-5", 1_000_000),
        ("claude-sonnet-5", 1_000_000),
        ("claude-haiku-4-5", 200_000),
        ("claude-3-5-sonnet-20241022", 200_000),
        # The rest.
        ("grok-4.5", 500_000),
        ("grok-4.3", 1_000_000),
        ("grok-4.20-0309-reasoning", 1_000_000),
        ("grok-build-0.1", 256_000),
        ("deepseek-reasoner", 128_000),
        ("glm-5.2", 1_000_000),
        ("kimi-k2.6", 256_000),
        ("qwen3.8-max", 1_000_000),
        ("qwen3.7-plus", 1_000_000),
        ("qwen-plus-latest", 1_000_000),
        ("qwen-long", 10_000_000),
        ("qwen3-max", 262_144),
        ("qwen3.6-max-preview", 262_144),
        ("baichuan4-turbo", 32_768),
        ("baichuan3-turbo-128k", 128_000),
    ])
    def test_the_family_resolves_to_its_published_window(self, model, window):
        assert context_window_from_table(model) == window

    def test_an_unknown_model_gets_no_guess(self):
        """None means "ask the provider or use the default" — a wrong guess
        here would silently mis-budget every turn."""
        assert context_window_from_table("some-unreleased-model") is None
        assert context_window_from_table("") is None

    def test_a_longer_prefix_always_beats_a_shorter_one(self):
        """The whole table depends on this: families are listed alongside the
        specific models that diverge from them."""
        assert context_window_from_table("claude-opus-4-8") == 1_000_000
        assert context_window_from_table("claude-opus-4-5") == 200_000
        assert context_window_from_table("moonshot-v1-8k") == 8_192
        assert context_window_from_table("moonshot-v1-128k") == 131_072

    def test_a_newer_name_is_not_assumed_to_be_a_bigger_window(self):
        """Three families break the intuition, and each one would overflow if
        the version prefix were allowed to answer for the whole tier."""
        # 5.4 is 1.05M, but its mini/nano tiers are 400K.
        assert context_window_from_table("gpt-5.4") == 1_050_000
        assert context_window_from_table("gpt-5.4-nano") == 400_000
        # Grok 4.5 is newer than 4.3 and half the size.
        assert context_window_from_table("grok-4.5") < context_window_from_table("grok-4.3")
        # qwen3.6-max-preview is a 256K model inside a 1M generation.
        assert context_window_from_table("qwen3.6-flash") == 1_000_000
        assert context_window_from_table("qwen3.6-max-preview") == 262_144

    def test_no_family_claims_a_window_it_cannot_serve(self):
        """Over-guessing overflows the window and fails the turn; the ceiling
        here is the largest window any listed provider actually ships."""
        for prefix, window in KNOWN_CONTEXT_WINDOWS:
            assert 4096 <= window <= 10_000_000, prefix

    def test_the_table_holds_no_duplicate_prefixes(self):
        prefixes = [prefix for prefix, _ in KNOWN_CONTEXT_WINDOWS]
        assert len(prefixes) == len(set(prefixes))


class TestReasoningModels:
    @pytest.mark.parametrize("model", [
        "o3-mini", "o4-mini", "gpt-5.2", "gemini-3-pro-preview",
        "claude-opus-5", "claude-sonnet-5", "claude-opus-4-8", "deepseek-reasoner",
    ])
    def test_models_that_reject_temperature_are_flagged(self, model):
        assert is_reasoning_model(model)

    @pytest.mark.parametrize("model", [
        "gpt-4o", "gpt-4.1", "claude-opus-4-6", "claude-sonnet-4-6",
        "claude-haiku-4-5", "deepseek-chat", "gemini-2.5-flash",
    ])
    def test_models_that_accept_temperature_are_not(self, model):
        assert not is_reasoning_model(model)


class TestProviderWiring:
    def test_every_backend_mode_that_names_a_provider_resolves(self):
        """A mode an admin can pick from the dropdown must reach an endpoint,
        or picking it produces a backend with no base URL."""
        non_api = {
            "stub", "local_lora", "local_gpu_lora", "adapter_server",
            "gemini_native", "lorax", "sagemaker", "aws_sagemaker",
            "vertex", "bedrock", "azure", "azure_openai",
        }
        for mode in ModelBackend:
            if mode.value in non_api:
                continue
            assert resolve_provider_endpoint(mode.value), mode.value

    def test_each_endpoint_names_a_key_env_and_a_reachable_base(self):
        for mode, entry in PROVIDER_ENDPOINTS.items():
            assert entry["api_key_env"], mode
            assert entry["provider"], mode
            base = entry["base_url"]
            # OpenAI itself is the one provider that keeps the SDK default.
            assert base is None or base.startswith("https://"), mode

    def test_aliases_agree_with_their_canonical_provider(self):
        for alias, canonical in (("grok", "xai"), ("kimi", "moonshot"),
                                 ("dashscope", "qwen"), ("glm", "zhipu"),
                                 ("google", "gemini")):
            assert PROVIDER_ENDPOINTS[alias] == PROVIDER_ENDPOINTS[canonical], alias

    def test_a_new_provider_gets_workable_default_capabilities(self):
        """These providers serve one model per request with no adapter
        multiplexing — the default — so none of them needs a bespoke entry."""
        for provider in ("xai", "deepseek", "moonshot", "qwen", "baichuan"):
            caps = get_provider_capabilities(provider)
            assert caps.max_adapters == 1
            assert not caps.multi_adapter
