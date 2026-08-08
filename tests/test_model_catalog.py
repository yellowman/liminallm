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
    HOSTED_CONTEXT_WINDOWS,
    KNOWN_CONTEXT_WINDOWS,
    context_window_from_table,
    is_reasoning_model,
)


class TestContextWindows:
    @pytest.mark.parametrize("model,window", [
        # Google — verified against a live ListModels call.
        ("gemini-3.6-flash", 1_048_576),
        ("gemini-flash-latest", 1_048_576),
        ("gemini-2.5-pro", 1_048_576),
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
        ("glm-5.1", 200_000),
        ("glm-4.7", 200_000),
        ("kimi-k2.6", 256_000),
        ("kimi-k3", 1_000_000),
        ("minimax-m3", 1_000_000),
        ("minimax-m2.7", 204_800),
        ("mistral-medium-3-5", 256_000),
        ("mistral-small-latest", 128_000),
        ("command-a-03-2025", 256_000),
        ("command-a-plus-05-2026", 128_000),
        ("muse-spark-1.2", 1_048_576),
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

    def test_no_table_holds_a_duplicate_prefix(self):
        """A duplicate is dead weight at best and two disagreeing answers at
        worst — longest-prefix picks whichever it reaches first."""
        for name, table in [("known", KNOWN_CONTEXT_WINDOWS),
                            *HOSTED_CONTEXT_WINDOWS.items()]:
            prefixes = [prefix for prefix, _ in table]
            assert len(prefixes) == len(set(prefixes)), name


class TestHostedWindows:
    """A reseller's serving limit is smaller than the model's native ceiling,
    and using the native number would overflow the host's real window."""

    @pytest.mark.parametrize("provider,model,hosted,native", [
        ("together", "MiniMaxAI/MiniMax-M3", 524_288, 1_000_000),
        ("together", "deepseek-ai/DeepSeek-V4-Pro", 512_000, 1_000_000),
        ("together", "zai-org/GLM-5.2", 262_144, 1_000_000),
        ("together", "moonshotai/Kimi-K2.6", 262_144, 256_000),
        ("fireworks", "accounts/fireworks/models/minimax-m3", 512_000, 1_000_000),
        ("fireworks", "accounts/fireworks/models/kimi-k3", 1_000_000, 1_000_000),
        ("groq", "minimaxai/minimax-m2.7", 196_608, 204_800),
    ])
    def test_the_host_answers_before_the_model_family(self, provider, model, hosted, native):
        assert context_window_from_table(model, provider=provider) == hosted
        del native  # documented above for contrast, not asserted here

    def test_the_native_window_still_answers_without_a_host(self):
        assert context_window_from_table("minimax-m3") == 1_000_000
        assert context_window_from_table("glm-5.2") == 1_000_000

    def test_an_unlisted_host_falls_through_to_the_family(self):
        """A provider with no hosted table is not an error — the model family
        still answers."""
        assert context_window_from_table("glm-5.2", provider="openai") == 1_000_000
        assert context_window_from_table("glm-5.2", provider="") == 1_000_000

    def test_the_meta_model_api_serves_muse_spark(self):
        """Meta's hosted family, reachable since the Llama API host retired."""
        assert context_window_from_table("muse-spark-1.2", provider="meta") == 1_048_576
        assert PROVIDER_ENDPOINTS["meta"]["base_url"] == "https://api.meta.ai/v1"

    def test_a_host_that_does_not_serve_the_model_falls_through(self):
        """Together has no gpt-5 row; the OpenAI family should still answer
        rather than the lookup returning nothing."""
        assert context_window_from_table("gpt-5.2", provider="together") == 400_000

    def test_cerebras_takes_the_free_tier_figure(self):
        """The window is account-tier dependent, so the table takes the floor
        and lets discovery or model_context_window raise it."""
        assert context_window_from_table("gpt-oss-120b", provider="cerebras") == 65_000
        assert context_window_from_table("openai/gpt-oss-120b", provider="groq") == 131_072

    def test_the_backend_passes_its_provider_into_the_lookup(self):
        """Without this the hosted tables are unreachable — the whole point of
        keying them by provider."""
        import inspect

        from liminallm.service.model_backend import ApiAdapterBackend

        src = inspect.getsource(ApiAdapterBackend.context_window.fget)
        assert "provider=self.provider" in src


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
        for provider in ("xai", "deepseek", "moonshot", "qwen", "baichuan",
                         "minimax", "mistral", "cohere", "groq", "cerebras",
                         "fireworks", "meta"):
            caps = get_provider_capabilities(provider)
            assert caps.max_adapters == 1
            assert not caps.multi_adapter


class TestListingDiscovery:
    """Resolving a window out of a /models listing.

    A reseller's listing holds hundreds of models. Reading a window from it
    without checking which model it belongs to is an over-guess — the one
    direction the whole fallback chain is built to avoid.
    """

    TOGETHER_LISTING = {
        "object": "list",
        "data": [
            {"id": "moonshotai/Kimi-K3", "context_length": 1_000_000},
            {"id": "Qwen/Qwen2.5-7B-Instruct-Turbo", "context_length": 32_768},
            {"id": "zai-org/GLM-5.2", "context_length": 262_144},
        ],
    }

    def test_the_listing_answers_for_the_model_that_was_asked_about(self):
        from liminallm.service.model_backend import _window_from_json

        assert _window_from_json(
            self.TOGETHER_LISTING, model="Qwen/Qwen2.5-7B-Instruct-Turbo"
        ) == 32_768
        assert _window_from_json(
            self.TOGETHER_LISTING, model="zai-org/GLM-5.2"
        ) == 262_144

    def test_it_does_not_hand_back_an_unrelated_models_window(self):
        """Without the id check this returns 1,000,000 — the first entry —
        for a 32K model, and the turn overflows."""
        from liminallm.service.model_backend import _window_from_json

        window = _window_from_json(
            self.TOGETHER_LISTING, model="Qwen/Qwen2.5-7B-Instruct-Turbo"
        )
        assert window != 1_000_000

    def test_a_listing_that_never_names_the_model_says_nothing(self):
        from liminallm.service.model_backend import _window_from_json

        assert _window_from_json(self.TOGETHER_LISTING, model="acme-llm-9") is None

    def test_a_host_qualified_id_matches_on_its_trailing_segment(self):
        """Gemini names models `models/x`; hosts prefix with an org."""
        from liminallm.service.model_backend import _window_from_json

        payload = {"data": [{"name": "models/gemini-3.6-flash",
                             "inputTokenLimit": 1_048_576}]}
        assert _window_from_json(payload, model="gemini-3.6-flash") == 1_048_576

    def test_a_single_model_response_still_scans_freely(self):
        """One-model servers (vLLM, LM Studio) answer without a listing, and
        their payload may not echo the id at all."""
        from liminallm.service.model_backend import _window_from_json

        assert _window_from_json({"max_model_len": 32_768}, model="whatever") == 32_768
        assert _window_from_json(
            {"id": "served-model", "max_model_len": 8_192}, model="requested-name"
        ) == 8_192

    def test_the_probe_passes_the_model_into_the_listing_lookup(self):
        import inspect

        from liminallm.service.model_backend import probe_context_window

        src = inspect.getsource(probe_context_window)
        assert "_window_from_json(resp.json(), model=model)" in src


def test_an_unknown_model_warns_rather_than_budgeting_8192_silently(monkeypatch):
    """Falling to the default is a real degradation, not routine info: every
    turn is then budgeted against 8192, and nothing else says so."""
    from unittest.mock import MagicMock

    from liminallm.service import model_backend as mb

    fake = MagicMock()
    monkeypatch.setattr(mb, "logger", fake)
    monkeypatch.setattr(mb, "probe_context_window", lambda **kw: None)

    backend = mb.ApiAdapterBackend("acme-llm-9", adapter_mode="openai", api_key=None)
    assert backend.context_window == mb.DEFAULT_CONTEXT_WINDOW
    fake.warning.assert_called_once()
    assert fake.warning.call_args.kwargs["source"] == "default"
    assert fake.warning.call_args.kwargs["model"] == "acme-llm-9"


def test_a_recognised_model_stays_at_info(monkeypatch):
    """The warning has to mean something — a model the table knows is not an
    operator problem."""
    from unittest.mock import MagicMock

    from liminallm.service import model_backend as mb

    fake = MagicMock()
    monkeypatch.setattr(mb, "logger", fake)
    monkeypatch.setattr(mb, "probe_context_window", lambda **kw: None)

    backend = mb.ApiAdapterBackend("gpt-4o-mini", adapter_mode="openai", api_key=None)
    assert backend.context_window == 128_000
    fake.warning.assert_not_called()
    assert fake.info.call_args.kwargs["source"] == "table"
