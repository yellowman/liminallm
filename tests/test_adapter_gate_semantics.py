"""SPEC §5.0.1: the gate activates first and modulates second.

`g = clamp(g_router, 0, 1)`. `g == 0` means the adapter is absent from the
request — from every mechanism, not just the one whose equation happens to
name `g`. Before this rule was written down, three surfaces disagreed:
composition dropped the zero-gated term, prompt injection never read the gate
at all, and the KV signature hashed an adapter contributing nothing.

The tests are grouped by surface because that is where the disagreement was.
Each closed-gate assertion is paired with the open-gate case, so a change
that makes an adapter inert for some unrelated reason cannot leave them
passing.
"""

import json

import pytest

from liminallm.service.llm import LLMService
from liminallm.service.model_backend import (
    ApiAdapterBackend,
    active_adapters,
    effective_gate,
)

PROMPT_ADAPTER = {
    "id": "skill",
    "mode": "prompt",
    "prompt_instructions": "prefer tabs",
}


class _LocalBackend:
    """A backend that applies weights, for the hybrid branch of §5.0.1."""

    applies_lora_weights = True

    def generate(self, messages, adapters, **kwargs):
        return {"messages": messages, "adapters": adapters}


class _ApiBackend:
    applies_lora_weights = False

    def generate(self, messages, adapters, **kwargs):
        return {"messages": messages, "adapters": adapters}


def _system_text(backend, adapter):
    service = LLMService(base_model="m", backend=backend)
    messages, kept = service._prepare_generation("hi", [adapter], [])
    joined = "\n".join(m["content"] for m in messages if m["role"] == "system")
    return joined, kept


class TestTheGateIsClampedBeforeItIsRead:
    @pytest.mark.parametrize(
        "adapter, expected",
        [
            ({"id": "a"}, 1.0),  # unset means fully active
            ({"id": "a", "weight": 0.0}, 0.0),
            ({"id": "a", "weight": -1.0}, 0.0),  # §8.1 clamp: negative is closed
            ({"id": "a", "weight": 5.0}, 1.0),  # and > 1 is fully open
            ({"id": "a", "weight": "0.25"}, 0.25),  # user-authored strings
            ({"id": "a", "weight": "nonsense"}, 1.0),  # unparseable is not "off"
            ({"id": "a", "gate_weight": 0.0}, 0.0),
            ({"id": "a", "schema": {"weight": 0.0}}, 0.0),
            ({"id": "a", "schema": "not-a-dict"}, 1.0),
        ],
    )
    def test_effective_gate(self, adapter, expected):
        assert effective_gate(adapter) == pytest.approx(expected)

    def test_active_adapters_keeps_every_positive_gate(self):
        adapters = [
            {"id": "off", "weight": 0.0},
            {"id": "negative", "weight": -3},
            {"id": "tiny", "weight": 0.01},
            {"id": "default"},
        ]
        assert [a["id"] for a in active_adapters(adapters)] == ["tiny", "default"]

    def test_no_threshold_rounds_a_small_gate_to_off(self):
        """A threshold would be a second routing policy, downstream of the
        one that owns the decision."""
        assert active_adapters([{"id": "a", "weight": 0.01}])
        assert active_adapters([{"id": "a", "weight": 1e-9}])


class TestPromptExecutionIsBinary:
    def test_a_closed_gate_injects_nothing(self):
        text, kept = _system_text(_ApiBackend(), {**PROMPT_ADAPTER, "weight": 0.0})
        assert "prefer tabs" not in text
        assert kept == [], "a zero-gated adapter reached the backend"

    def test_a_negative_gate_injects_nothing(self):
        text, _ = _system_text(_ApiBackend(), {**PROMPT_ADAPTER, "weight": -1.0})
        assert "prefer tabs" not in text

    @pytest.mark.parametrize("gate", [0.01, 0.5, 1.0])
    def test_any_positive_gate_injects_once_and_unchanged(self, gate):
        """There is no analogue of multiplying a sentence by g, so a
        fractional gate must not shorten, repeat or paraphrase it."""
        text, kept = _system_text(_ApiBackend(), {**PROMPT_ADAPTER, "weight": gate})
        assert text.count("prefer tabs") == 1
        assert [a["id"] for a in kept] == ["skill"]

    def test_the_local_backend_agrees(self):
        """The prompt rung injects on every backend (§5.5), so the gate has
        to close it on every backend too."""
        closed, _ = _system_text(_LocalBackend(), {**PROMPT_ADAPTER, "weight": 0.0})
        open_, _ = _system_text(_LocalBackend(), {**PROMPT_ADAPTER, "weight": 1.0})
        assert "prefer tabs" not in closed
        assert "prefer tabs" in open_


class TestHybridFollowsTheRepresentationInUse:
    HYBRID = {
        "id": "skill",
        "mode": "hybrid",
        "current_version": 0,  # nothing promoted: the fallback is all there is
        "prompt_instructions": "prefer tabs",
    }

    def test_an_unpromoted_hybrid_is_silenced_by_a_closed_gate(self):
        closed, _ = _system_text(_LocalBackend(), {**self.HYBRID, "weight": 0.0})
        open_, _ = _system_text(_LocalBackend(), {**self.HYBRID, "weight": 0.7})
        assert "prefer tabs" not in closed
        assert open_.count("prefer tabs") == 1

    def test_a_promoted_hybrid_keeps_its_api_fallback_gated(self):
        promoted = {**self.HYBRID, "current_version": 3}
        closed, _ = _system_text(_ApiBackend(), {**promoted, "weight": 0.0})
        open_, _ = _system_text(_ApiBackend(), {**promoted, "weight": 0.3})
        assert "prefer tabs" not in closed
        assert open_.count("prefer tabs") == 1


class TestRemotePassthroughAndAccounting:
    def _api(self, provider="together", model="mixtral"):
        return ApiAdapterBackend(base_model=model, api_key=None, provider=provider)

    def test_a_closed_gate_is_absent_rather_than_dropped(self):
        """`dropped` records an adapter the backend could not honour. A
        zero-gated adapter was never asked for, so it belongs in neither
        list — reporting it as applied would claim it affected the answer."""
        processed = self._api()._process_adapters_for_provider(
            [{**PROMPT_ADAPTER, "weight": 0.0}]
        )
        assert processed["applied"] == []
        assert processed["dropped"] == []
        assert processed["prompt_injections"] == []

    def test_a_closed_gate_is_not_sent_to_the_provider(self):
        remote = {"id": "r", "mode": "remote", "remote_adapter_id": "ra-1"}
        api = self._api()

        closed = api._process_adapters_for_provider([{**remote, "weight": 0.0}])
        assert closed["extra_body"] in (None, {})
        assert closed["applied"] == []

        # Control: the same adapter is passed through when active, and the
        # numeric gate survives because this mechanism scales continuously.
        active = api._process_adapters_for_provider([{**remote, "weight": 0.5}])
        assert active["extra_body"]["adapter_id"] == "ra-1"
        assert active["extra_body"]["adapter_weights"] == pytest.approx(0.5)
        assert active["applied"] == ["r:adapter_param"]

    def test_gemini_native_agrees(self):
        from liminallm.service.gemini_backend import GeminiBackend

        backend = GeminiBackend(base_model="gemini-2.0-flash", api_key="k")
        closed, applied = backend._prompt_injections(
            [{**PROMPT_ADAPTER, "weight": 0.0}]
        )
        assert closed == [] and applied == []
        open_, applied = backend._prompt_injections(
            [{**PROMPT_ADAPTER, "weight": 0.2}]
        )
        assert open_ == ["prefer tabs"] and applied == ["skill:prompt"]


class TestAccountingAndAuditSayDifferentThings:
    def test_a_zero_gated_adapter_is_traced_but_not_reported_as_applied(self, client):
        """"the router assigned this a zero gate" and "this adapter shaped the
        answer" are different facts, so they live in different fields.

        Driven through the runtime's real router, store and policy engine:
        the value has to survive the actual hand-offs, which is where the
        earlier gate defect lived.
        """
        import asyncio
        import uuid

        from liminallm.service.runtime import get_runtime

        runtime = get_runtime()
        user = runtime.store.create_user(email=f"gate0_{uuid.uuid4().hex[:8]}@t.local")
        adapter = runtime.store.create_artifact(
            "adapter",
            f"muted_{uuid.uuid4().hex[:6]}",
            {
                "kind": "adapter.lora",
                "mode": "prompt",
                "backend": "prompt",
                "base_model": "test-base",
                "prompt_instructions": "be terse",
                "current_version": 0,
            },
            owner_user_id=user.id,
        )
        runtime.store.create_artifact(
            "policy",
            "default_routing",
            {
                "kind": "policy.routing",
                # The router's own `weight_floor` (default 0.05) would drop a
                # zero gate before selection ever saw it, and then this test
                # would pass without exercising the rule. Lower the floor so
                # the gate genuinely arrives at 0.0.
                "weight_floor": 0,
                "rules": [
                    {
                        "when": "true",
                        "action": {
                            "type": "activate_adapter_by_id",
                            "adapter_id": adapter.id,
                            "weight": 0.0,
                        },
                    }
                ],
            },
            visibility="global",
        )

        adapters, _, gates = asyncio.run(
            runtime.workflow._select_adapters(
                user_message="anything at all",
                user_id=user.id,
                context_id=None,
                tenant_id=user.tenant_id,
            )
        )

        assert adapter.id not in {a.get("id") for a in adapters}, (
            "a zero-gated adapter reached the request"
        )
        assert adapter.id in {g.get("id") for g in gates}, (
            "the router's own decision was erased from the audit trail"
        )


jax = pytest.importorskip("jax")
pytest.importorskip("safetensors")

from liminallm.service.model_backend import LocalJaxLoRABackend  # noqa: E402
from tests.test_local_transformer import _build_checkpoint  # noqa: E402


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    return _build_checkpoint(tmp_path_factory.mktemp("gate_semantics_model"))


class TestTheEffectiveStackHashesTheSame:
    def test_an_adapter_at_zero_is_the_same_model_as_no_adapter(
        self, tmp_path, checkpoint
    ):
        """§5.3 keys cached KV by the effective stack. `[X @ 0]` and `[]` are
        the same effective model, so they must be the same key — hashing the
        zero-gated adapter cost a legitimate reuse on every turn where the
        router happened to close a gate."""
        backend = LocalJaxLoRABackend(str(checkpoint), str(tmp_path))
        closed = {"id": "x", "current_version": 2, "weight": 0.0}

        assert backend._adapter_signature([closed]) == backend._adapter_signature([])
        assert backend._adapter_signature([closed]) == "base"
        # And the gates that do change the model still key apart.
        assert backend._adapter_signature(
            [{**closed, "weight": 0.2}]
        ) != backend._adapter_signature([{**closed, "weight": 0.8}])

    def test_a_closed_gate_does_not_perturb_an_open_one(self, tmp_path, checkpoint):
        backend = LocalJaxLoRABackend(str(checkpoint), str(tmp_path))
        served = {"id": "y", "current_version": 1, "weight": 0.6}
        closed = {"id": "x", "current_version": 2, "weight": 0.0}
        assert backend._adapter_signature([served, closed]) == (
            backend._adapter_signature([served])
        )

    def test_a_closed_gate_reuses_the_prefix_built_without_it(
        self, tmp_path, checkpoint
    ):
        """The cache is the point, not the string: a turn that closes a gate
        must be able to continue the prefix computed when the adapter was
        never routed."""
        config = __import__(
            "liminallm.service.transformer", fromlist=["x"]
        ).load_config(checkpoint)
        version = tmp_path / "x" / "v0001"
        version.mkdir(parents=True)
        (version / "params.json").write_text(
            json.dumps(
                {
                    "layers.0.attn_q.A": [[0.05] * config.hidden_size] * 2,
                    "layers.0.attn_q.B": [[0.05, 0.05]]
                    * (config.num_heads * config.head_dim),
                }
            )
        )
        backend = LocalJaxLoRABackend(
            str(checkpoint), str(tmp_path), max_new_tokens=2
        )
        messages = [{"role": "user", "content": "hello there friend"}]

        backend.generate(messages, [], user_id="u")
        second = backend.generate(
            messages,
            [
                {
                    "id": "x",
                    "mode": "local",
                    "fs_dir": "x",
                    "current_version": 1,
                    "weight": 0.0,
                    "base_model": "some-other-model",
                }
            ],
            user_id="u",
        )
        assert second["usage"].get("cached_tokens", 0) > 0
