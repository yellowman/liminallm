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

    def test_the_effective_set_carries_the_canonical_magnitude(self):
        """Membership is half the answer. A consumer that reads the adapter's
        weight must read the same number composition scales by — otherwise it
        re-derives it, and re-derivation is what sent a provider 5.0 for an
        adapter already clamped to 1.0."""
        adapters = [
            {"id": "over", "weight": 5.0},
            {"id": "schema_only", "schema": {"weight": 0.25}},
            {"id": "unparseable", "weight": "nonsense"},
            {"id": "legacy", "gate_weight": 0.4},
        ]
        assert [a["weight"] for a in active_adapters(adapters)] == [
            1.0,
            0.25,
            1.0,
            0.4,
        ]

    def test_the_callers_adapters_are_not_edited(self):
        original = {"id": "over", "weight": 5.0}
        (canonical,) = active_adapters([original])
        assert canonical is not original
        assert original["weight"] == 5.0, "the caller's dict was rewritten"

    def test_canonicalizing_twice_changes_nothing(self):
        """The set passes through several layers on the way to a backend."""
        once = active_adapters([{"id": "a", "weight": 5.0}])
        assert active_adapters(once) == once

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
        assert "prompt_injections" not in processed

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

    @pytest.mark.parametrize(
        "adapter_fields, expected",
        [
            ({"weight": 0.5}, 0.5),
            ({"weight": 5.0}, 1.0),  # §8.1 clamp, not the raw number
            ({"schema": {"weight": 0.25}}, 0.25),  # same precedence as §5.2
            ({"weight": "nonsense"}, 1.0),  # one interpretation, not a crash
            ({"gate_weight": 0.4}, 0.4),
        ],
    )
    def test_the_provider_receives_the_canonical_gate(self, adapter_fields, expected):
        """§5.0.1: a mechanism with a continuous weight applies `g` exactly.

        The formatter used to re-read the raw dict, so it sent 5.0 for an
        adapter the kernel clamps to 1.0, sent 1.0 for one whose gate lived in
        `schema.weight`, and raised on a weight the canonical rule reads as
        1.0 — a malformed number became a failed request rather than a
        defaulted one.
        """
        adapter = {
            "id": "r",
            "mode": "remote",
            "remote_adapter_id": "ra-1",
            **adapter_fields,
        }
        processed = self._api()._process_adapters_for_provider([adapter])
        assert processed["extra_body"]["adapter_weights"] == pytest.approx(expected)

    def test_ranking_uses_the_canonical_gate(self):
        """MODEL_ID providers serve exactly one adapter, so the ranking is
        the choice. An out-of-range 5.0 must not outrank a legitimate 1.0,
        and a schema-held gate must not rank as if it were unset."""
        api = self._api(provider="openai", model="gpt-x")
        loud = {"id": "loud", "mode": "remote", "remote_model_id": "ft-loud", "weight": 5.0}
        quiet = {"id": "quiet", "mode": "remote", "remote_model_id": "ft-quiet",
                 "schema": {"weight": 0.9}}
        # Both clamp below/at 1.0; `loud` at 1.0 still wins, but on 1.0 not 5.0.
        model, _, applied, _ = api._format_remote_adapters([quiet, loud])
        assert model == "ft-loud" and applied == ["loud:model_id"]
        # And a malformed weight ranks as 1.0 rather than raising.
        broken = {"id": "b", "mode": "remote", "remote_model_id": "ft-b",
                  "weight": "nonsense"}
        model, _, applied, _ = api._format_remote_adapters([broken])
        assert model == "ft-b"

    def test_gemini_native_agrees(self):
        """Accounting only. This used to assert the backend returned the
        instruction text, which is the second-materializer contract §5.0.1
        replaced — `_request_body` then prepended it on top of the copy
        LLMService had already placed."""
        from liminallm.service.gemini_backend import GeminiBackend

        backend = GeminiBackend(base_model="gemini-2.0-flash", api_key="k")
        assert backend._applied_prompt_adapters(
            [{**PROMPT_ADAPTER, "weight": 0.0}]
        ) == []
        assert backend._applied_prompt_adapters(
            [{**PROMPT_ADAPTER, "weight": 0.2}]
        ) == ["skill:prompt"]


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

    def test_the_key_names_the_mechanisms_that_ran(self, tmp_path, checkpoint):
        """§5.3 keys cached KV by the *effective* stack, and on this backend
        the mechanism is weights. An adapter that applies none — nothing
        promoted, or a prompt rung whose text is already in the tokens the
        key covers — describes the same local model as no adapter at all, so
        it must not key apart from it. Safe either way (a mismatch only costs
        a reuse), but one definition of "effective" is the point."""
        backend = LocalJaxLoRABackend(str(checkpoint), str(tmp_path))
        base = backend._adapter_signature([])

        assert backend._adapter_signature(
            [{"id": "u", "mode": "local", "current_version": 0}]
        ) == base
        assert backend._adapter_signature(
            [{"id": "p", "mode": "prompt", "current_version": 3}]
        ) == base
        # And the promoted ones still key apart by gate, as §5.3 requires.
        assert backend._adapter_signature(
            [{"id": "a", "current_version": 2, "weight": 0.2}]
        ) != backend._adapter_signature(
            [{"id": "a", "current_version": 2, "weight": 0.8}]
        )

    def test_a_closed_gate_does_not_perturb_an_open_one(self, tmp_path, checkpoint):
        backend = LocalJaxLoRABackend(str(checkpoint), str(tmp_path))
        served = {"id": "y", "current_version": 1, "weight": 0.6}
        closed = {"id": "x", "current_version": 2, "weight": 0.0}
        assert backend._adapter_signature([served, closed]) == (
            backend._adapter_signature([served])
        )

    def test_a_closed_gate_is_not_reported_as_applied(self, tmp_path, checkpoint):
        """§5.0.1 omits a zero-gated adapter from inference accounting, not
        only from the LoRA sum and the cache key.

        The backend applied no weights and hashed as the base model, then
        returned `usage.adapter_id == "X"` — a turn that claimed an adapter
        shaped an answer it had no part in. `generate()` canonicalizes the
        list at its own entry now, so this holds for a direct call and not
        only downstream of LLMService.
        """
        config = __import__(
            "liminallm.service.transformer", fromlist=["x"]
        ).load_config(checkpoint)
        version = tmp_path / "X" / "v0001"
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
        adapter = {
            "id": "X",
            "mode": "local",
            "fs_dir": "X",
            "current_version": 1,
            "base_model": str(checkpoint),
        }
        messages = [{"role": "user", "content": "hello"}]

        closed = backend.generate(messages, [{**adapter, "weight": 0.0}], user_id="u")
        assert closed["usage"].get("adapter_id") is None

        # Control: the same adapter, open, is reported — so the absence above
        # is the gate and not a backend that never reports anything.
        open_ = backend.generate(messages, [{**adapter, "weight": 1.0}], user_id="u")
        assert open_["usage"].get("adapter_id") == "X"

    def test_a_closed_first_adapter_does_not_size_the_tokenizer(
        self, tmp_path, checkpoint
    ):
        """`adapters[0]` fed `_apply_adapter_vocab_size` before any filtering,
        so an absent adapter could still reconfigure tokenization.

        Both adapters here are promoted, because vocabulary is weight-specific
        state and §5.5 only lets a promoted adapter carry weights — an
        unpromoted one applies no mechanism and so has no say either.
        """
        config = __import__(
            "liminallm.service.transformer", fromlist=["x"]
        ).load_config(checkpoint)
        version = tmp_path / "adapters" / "Y" / "v0001"
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
        seen = []
        backend._apply_adapter_vocab_size = lambda adapter: seen.append(adapter)
        backend.generate(
            [{"role": "user", "content": "hi"}],
            [
                {"id": "X", "weight": 0.0, "vocab_size": 99, "current_version": 1},
                {
                    "id": "Y",
                    "weight": 1.0,
                    "current_version": 1,
                    "base_model": str(checkpoint),
                },
            ],
            user_id="u",
        )
        assert [a.get("id") for a in seen] == ["Y"]

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
