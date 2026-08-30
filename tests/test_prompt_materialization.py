"""SPEC §5.0.1: an adapter's instructions are materialized once, by the service.

Two layers both believed they owned this. `LLMService` placed the
instructions, and then `ApiAdapterBackend` and `GeminiBackend` placed them
again - so `generate` and `generate_stream` sent the text twice, while
`generate_with_tools` and `stream_messages`, which never passed through the
service's message builder, sent it once. Deleting the backend copy alone
would have taken that pair to zero; the repair was one service primitive that
every entry point uses.

These tests count occurrences on the surface that actually reaches a
provider, because "the instruction is present" was true throughout the
defect. Each is paired with the closed-gate case, and the file ends with a
check that neutralizing the service materializer turns the positive cases
red - the false-green shape this audit has found repeatedly.
"""

import json

import pytest

from liminallm.service.gemini_backend import GeminiBackend
from liminallm.service.llm import LLMService
from liminallm.service.model_backend import ApiAdapterBackend

INSTRUCTION = "prefer tabs"
PROMPT_ADAPTER = {
    "id": "skill",
    "mode": "prompt",
    "prompt_instructions": INSTRUCTION,
}
CALLER_MESSAGES = [{"role": "user", "content": "hi"}]


class _Capturing:
    """Records exactly the messages a backend is handed.

    An API-shaped backend: it applies no LoRA weights, so §5.0.1 makes the
    prompt its representation of an adapter - the case where a second
    materializer showed up.
    """

    applies_lora_weights = False
    supports_tools = True

    def __init__(self):
        self.messages = None

    def _record(self, messages):
        self.messages = list(messages)
        return {"content": "", "usage": {}}

    def generate(self, messages, adapters, **kwargs):
        return self._record(messages)

    def generate_with_tools(self, messages, tools, adapters, **kwargs):
        return self._record(messages)

    def generate_stream(self, messages, adapters, **kwargs):
        self._record(messages)
        return iter(())


def _count(messages) -> int:
    return sum(str(m.get("content") or "").count(INSTRUCTION) for m in messages or [])


def _drive(path: str, adapter: dict) -> int:
    """Run one LLMService entry point and count what the backend received."""
    backend = _Capturing()
    service = LLMService(base_model="m", backend=backend)
    adapters = [adapter] if adapter else []
    if path == "generate":
        service.generate("hi", adapters, [])
    elif path == "generate_stream":
        list(service.generate_stream("hi", adapters, []))
    elif path == "generate_with_tools":
        service.generate_with_tools(list(CALLER_MESSAGES), [], adapters)
    elif path == "stream_messages":
        list(service.stream_messages(list(CALLER_MESSAGES), adapters))
    else:  # pragma: no cover - guards a typo in the parametrization
        raise AssertionError(f"unknown path {path!r}")
    return _count(backend.messages)


PATHS = ["generate", "generate_stream", "generate_with_tools", "stream_messages"]


class TestEveryServicePathMaterializesExactlyOnce:
    @pytest.mark.parametrize("path", PATHS)
    @pytest.mark.parametrize("gate", [0.01, 0.5, 1.0])
    def test_a_positive_gate_sends_one_unchanged_copy(self, path, gate):
        assert _drive(path, {**PROMPT_ADAPTER, "weight": gate}) == 1

    @pytest.mark.parametrize("path", PATHS)
    def test_an_ungated_adapter_sends_one_copy(self, path):
        """The common case: no explicit weight is a fully open gate."""
        assert _drive(path, dict(PROMPT_ADAPTER)) == 1

    @pytest.mark.parametrize("path", PATHS)
    @pytest.mark.parametrize("gate", [0.0, -1.0])
    def test_a_closed_gate_sends_none(self, path, gate):
        assert _drive(path, {**PROMPT_ADAPTER, "weight": gate}) == 0

    @pytest.mark.parametrize("path", PATHS)
    def test_the_callers_messages_are_not_edited(self, path):
        """The service prepares a copy; a caller that iterates a tool loop
        would otherwise accumulate a guidance block per round."""
        if path in {"generate", "generate_stream"}:
            pytest.skip("these build their own message list")
        caller = [dict(m) for m in CALLER_MESSAGES]
        service = LLMService(base_model="m", backend=_Capturing())
        if path == "generate_with_tools":
            service.generate_with_tools(caller, [], [dict(PROMPT_ADAPTER)])
        else:
            list(service.stream_messages(caller, [dict(PROMPT_ADAPTER)]))
        assert caller == CALLER_MESSAGES


class TestAStatedModeMaterializes:
    """Every prompt-carrying mode reaches the prompt, in either dict shape.

    This was originally about *inferred* modes: `get_adapter_mode` returned a
    raw string when the artifact stated one and an `AdapterMode` member when
    it inferred, and the service compared `str(mode)` - "AdapterMode.HYBRID"
    for a member, matching nothing - so every adapter that did not state a
    mode was skipped. Inference is retired (Pass C), so the cases state their
    modes; the enum-versus-string comparison it guards is still live.

    It went unnoticed because the API backends were injecting the prompt
    themselves; the same bug, with the backends no longer doing that, would
    have dropped those instructions entirely. Moving a responsibility is how
    you find out what was covering for it.
    """

    @pytest.mark.parametrize(
        "adapter",
        [
            # Hybrid carries the prompt as well as weights.
            {"id": "legacy", "mode": "hybrid", "prompt_instructions": INSTRUCTION},
            # Instructions and mode nested in the schema shape.
            {"id": "legacy", "schema": {"mode": "hybrid",
                                        "prompt_instructions": INSTRUCTION}},
            # A stated prompt mode, top-level instructions.
            {"id": "legacy", "mode": "prompt", "prompt_instructions": INSTRUCTION},
            # And nested instructions with a stated mode.
            {"id": "legacy", "mode": "prompt", "schema": {"prompt_instructions": INSTRUCTION}},
        ],
    )
    def test_the_instructions_are_placed(self, adapter):
        assert _drive("generate", adapter) == 1


class TestBackendsDoNotMaterializeAgain:
    def test_the_api_backend_adds_nothing_to_prepared_messages(self):
        """Given messages the service already prepared, the backend's job is
        transport, remote selection and accounting - not a second copy."""
        api = ApiAdapterBackend(base_model="gpt-x", api_key=None, provider="openai")
        service = LLMService(base_model="gpt-x", backend=api)
        prepared, adapters = service._prepare_backend_messages(
            list(CALLER_MESSAGES), [dict(PROMPT_ADAPTER)]
        )
        assert _count(prepared) == 1

        processed = api._process_adapters_for_provider(adapters)
        # It still reports the adapter as applied - absent from the text is
        # not absent from the accounting.
        assert processed["applied"] == ["skill:prompt"]
        assert "prompt_injections" not in processed

    def test_the_gemini_request_body_carries_one_copy(self):
        backend = GeminiBackend(base_model="gemini-2.0-flash", api_key="k")
        service = LLMService(base_model="gemini-2.0-flash", backend=backend)

        prepared, adapters = service._prepare_backend_messages(
            list(CALLER_MESSAGES), [dict(PROMPT_ADAPTER)]
        )
        body, applied = backend._request_body(prepared, adapters)
        assert json.dumps(body).count(INSTRUCTION) == 1
        assert applied == ["skill:prompt"]

    def test_the_gemini_request_body_is_empty_on_a_closed_gate(self):
        backend = GeminiBackend(base_model="gemini-2.0-flash", api_key="k")
        service = LLMService(base_model="gemini-2.0-flash", backend=backend)
        prepared, adapters = service._prepare_backend_messages(
            list(CALLER_MESSAGES), [{**PROMPT_ADAPTER, "weight": 0.0}]
        )
        body, applied = backend._request_body(prepared, adapters)
        assert json.dumps(body).count(INSTRUCTION) == 0
        assert applied == []


class TestAppliedNamesMechanismsNotModes:
    """`applied` claims the adapter affected inference (§5.0.1).

    It was built from the mode: a hybrid adapter with neither instructions
    nor a remote id was reported as `:prompt` because it was hybrid, while
    nothing was injected, no adapter was sent and the model was unchanged.
    The artifact schema permits that shape - neither field is required - so
    it is a valid artifact producing a false claim.
    """

    def _api(self, provider="openai", model="gpt-x"):
        return ApiAdapterBackend(base_model=model, api_key=None, provider=provider)

    def test_a_hybrid_with_no_usable_representation_is_dropped(self):
        processed = self._api()._process_adapters_for_provider(
            [{"id": "empty", "mode": "hybrid", "current_version": 0, "weight": 1.0}]
        )
        assert processed["applied"] == []
        assert processed["dropped"] == ["empty"]
        assert processed["model"] == "gpt-x", "the model was changed by nothing"

    def test_a_prompt_adapter_with_no_instructions_is_dropped(self):
        processed = self._api()._process_adapters_for_provider(
            [{"id": "silent", "mode": "prompt"}]
        )
        assert processed["applied"] == []
        assert processed["dropped"] == ["silent"]

    def test_each_mechanism_present_gets_its_own_entry(self):
        """Two mechanisms, two entries - not one marker for the mode. The
        `:hybrid` marker was appended before the provider had decided whether
        it would select the model at all."""
        processed = self._api()._process_adapters_for_provider(
            [
                {
                    "id": "both",
                    "mode": "hybrid",
                    "prompt_instructions": INSTRUCTION,
                    "remote_model_id": "ft:gpt-4:custom",
                }
            ]
        )
        assert processed["applied"] == ["both:prompt", "both:model_id"]
        assert processed["model"] == "ft:gpt-4:custom"

    def test_a_hybrid_carried_only_by_its_prompt_says_so(self):
        processed = self._api()._process_adapters_for_provider(
            [{"id": "p", "mode": "hybrid", "prompt_instructions": INSTRUCTION}]
        )
        assert processed["applied"] == ["p:prompt"]
        assert processed["dropped"] == []

    def test_a_hybrid_carried_only_by_a_remote_adapter_says_so(self):
        processed = self._api(
            provider="together", model="mixtral"
        )._process_adapters_for_provider(
            [{"id": "r", "mode": "hybrid", "remote_adapter_id": "ra-1"}]
        )
        assert processed["applied"] == ["r:adapter_param"]
        assert processed["dropped"] == []


class TestTheServiceIsTheMaterializer:
    """If the service stops placing the text, nothing else puts it back.

    This is the check that would have failed during the defect: back then a
    disabled service materializer still produced one copy, because the
    backends were making their own. A test that only counted "exactly one"
    could not tell the two arrangements apart.
    """

    @pytest.mark.parametrize("path", PATHS)
    def test_defeating_it_leaves_nothing(self, path, monkeypatch):
        monkeypatch.setattr(
            LLMService, "_build_adapter_prompts", lambda self, adapters: []
        )
        assert _drive(path, dict(PROMPT_ADAPTER)) == 0

    def test_defeating_it_empties_the_gemini_body(self, monkeypatch):
        backend = GeminiBackend(base_model="gemini-2.0-flash", api_key="k")
        service = LLMService(base_model="gemini-2.0-flash", backend=backend)
        monkeypatch.setattr(
            LLMService, "_build_adapter_prompts", lambda self, adapters: []
        )
        prepared, adapters = service._prepare_backend_messages(
            list(CALLER_MESSAGES), [dict(PROMPT_ADAPTER)]
        )
        body, _ = backend._request_body(prepared, adapters)
        assert json.dumps(body).count(INSTRUCTION) == 0
