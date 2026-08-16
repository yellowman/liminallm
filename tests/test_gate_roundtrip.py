"""Router gates reaching the model, and the promoted version reaching a turn.

The composition equation can be exactly right inside the backend and still
never see a real gate, because the gate is computed in the router and has to
survive three hand-offs to get there. These tests follow the value, not the
formula.
"""

import json

import pytest

from liminallm.service import transformer

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("safetensors")

from liminallm.service.model_backend import LocalJaxLoRABackend  # noqa: E402
from tests.test_local_transformer import _build_checkpoint  # noqa: E402

BASE = ""
"""The serving base identity these fixtures' adapters declare.

SPEC §5.1 ties LoRA weights to one frozen base, so serving refuses an adapter
that does not declare the base it is being applied to — a fixture without one
describes an adapter the artifact schema could not store either.
"""


@pytest.fixture(scope="module", autouse=True)
def checkpoint(tmp_path_factory):
    """autouse so BASE is set before any test in the module reads it, whatever
    order they run in — an unset BASE would refuse weights for the wrong
    reason."""
    global BASE
    directory = _build_checkpoint(tmp_path_factory.mktemp("gate_model"))
    BASE = str(directory)
    return directory


def _write_adapter(directory, config, *, value, rank=2):
    """An adapter whose ΔW is a known constant, so gating is readable."""
    import numpy as np

    width = config.num_heads * config.head_dim
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "params.json").write_text(
        json.dumps(
            {
                "layers.0.attn_q.A": np.full(
                    (rank, config.hidden_size), value, dtype=float
                ).tolist(),
                "layers.0.attn_q.B": np.full((width, rank), value, dtype=float).tolist(),
            }
        )
    )


class TestRouterGatesReachTheModel:
    def test_select_adapters_carries_the_gate_onto_the_adapter(self, client):
        """The router's weight must travel ON the adapter dict.

        `_select_adapters` used to rebuild the activated list from the
        candidate lookup and return the gates separately for tracing, so
        every routed turn composed at 1.0 regardless of policy.

        Run against the runtime's real engine, real store and real router —
        the value has to survive the actual hand-offs, which is the whole
        point of the finding.
        """
        import asyncio
        import uuid

        from liminallm.service.runtime import get_runtime

        runtime = get_runtime()
        user = runtime.store.create_user(email=f"gate_{uuid.uuid4().hex[:8]}@t.local")
        adapter = runtime.store.create_artifact(
            "adapter",
            f"persona_{uuid.uuid4().hex[:6]}",
            {
                "kind": "adapter.lora",
                # hybrid so the runtime's configured backend accepts it; the
                # mode filter is a separate concern from gate propagation.
                "backend": "hybrid",
                "mode": "hybrid",
                "base_model": "test-base",
                "prompt_instructions": "be terse",
                "current_version": 3,
                "weight": 1.0,
            },
            owner_user_id=user.id,
        )
        # A policy that activates the adapter at a distinctly non-default gate.
        runtime.store.create_artifact(
            "policy",
            "default_routing",
            {
                "kind": "policy.routing",
                "rules": [
                    {
                        "when": "true",
                        "action": {
                            "type": "activate_adapter_by_id",
                            "adapter_id": adapter.id,
                            "weight": 0.18,
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

        activated = {a["id"]: a for a in adapters}
        assert adapter.id in activated, "the routed adapter never reached serving"
        assert activated[adapter.id]["weight"] == pytest.approx(0.18)
        # The rest of the adapter payload still travels with it.
        assert activated[adapter.id]["current_version"] == 3
        assert gates  # still reported for tracing

    def test_a_routed_gate_scales_the_served_delta(self, tmp_path, checkpoint):
        """End of the chain: the gate the router picked is the gate applied."""
        config = transformer.load_config(checkpoint)
        _write_adapter(tmp_path / "ad" / "v0001", config, value=0.05)
        backend = LocalJaxLoRABackend(str(checkpoint), str(tmp_path))
        adapter = {"id": "ad", "base_model": BASE, "backend": "local", "fs_dir": "ad",
                   "current_version": 1}

        full = backend._blend_adapter_weights(
            [{**adapter, "weight": 1.0}], user_id="u"
        )
        gated = backend._blend_adapter_weights(
            [{**adapter, "weight": 0.18}], user_id="u"
        )
        full_delta = full["layers.0.attn_q.B"] @ full["layers.0.attn_q.A"]
        gated_delta = gated["layers.0.attn_q.B"] @ gated["layers.0.attn_q.A"]
        assert float(jnp.max(jnp.abs(gated_delta - 0.18 * full_delta))) < 1e-6


class TestGateAwareKvCache:
    def test_the_same_adapter_at_two_gates_is_two_cache_identities(
        self, tmp_path, checkpoint
    ):
        """Gates are per-request (SPEC §5.3), so every cached K/V tensor was
        computed under one of them. An id+version key would let a 0.2 request
        continue a prefix built at 0.8."""
        backend = LocalJaxLoRABackend(str(checkpoint), str(tmp_path))
        adapter = {"id": "ad", "current_version": 2}

        quiet = backend._adapter_signature([{**adapter, "weight": 0.2}])
        loud = backend._adapter_signature([{**adapter, "weight": 0.8}])
        assert quiet != loud
        # Same gate, same identity — the cache must still be usable.
        assert quiet == backend._adapter_signature([{**adapter, "weight": 0.2}])
        # And the base (no adapters) stays distinct from any gated stack.
        assert backend._adapter_signature([]) == "base"

    def test_a_gate_change_does_not_reuse_the_other_gates_prefix(
        self, tmp_path, checkpoint
    ):
        config = transformer.load_config(checkpoint)
        _write_adapter(tmp_path / "ad" / "v0001", config, value=0.05)
        backend = LocalJaxLoRABackend(
            str(checkpoint), str(tmp_path), max_new_tokens=2
        )
        adapter = {"id": "ad", "base_model": BASE, "backend": "local", "fs_dir": "ad",
                   "current_version": 1}
        messages = [{"role": "user", "content": "hello there friend"}]

        backend.generate(messages, [{**adapter, "weight": 0.8}], user_id="u")
        # Same text, different gate: nothing may be reused.
        second = backend.generate(
            messages, [{**adapter, "weight": 0.2}], user_id="u"
        )
        assert second["usage"].get("cached_tokens", 0) == 0
        # Same text, same gate: reuse is expected.
        third = backend.generate(
            messages, [{**adapter, "weight": 0.2}], user_id="u"
        )
        assert third["usage"].get("cached_tokens", 0) > 0


class TestCompositionRefusesRatherThanPartiallyApplies:
    def test_disagreeing_shapes_refuse_the_stack(self, tmp_path, checkpoint):
        """SPEC §5.2 forbids partial application; dropping the odd adapter
        and applying the rest serves a stack the router never chose."""
        config = transformer.load_config(checkpoint)
        _write_adapter(tmp_path / "ok" / "v0001", config, value=0.05)
        # Same target, incompatible projection width.
        (tmp_path / "bad" / "v0001").mkdir(parents=True)
        (tmp_path / "bad" / "v0001" / "params.json").write_text(
            json.dumps(
                {
                    "layers.0.attn_q.A": [[0.1] * (config.hidden_size + 3)] * 2,
                    "layers.0.attn_q.B": [[0.1, 0.1]] * 5,
                }
            )
        )
        backend = LocalJaxLoRABackend(str(checkpoint), str(tmp_path))
        with pytest.raises(ValueError, match="refusing the adapter stack"):
            backend._blend_adapter_weights(
                [
                    {"id": "ok", "base_model": BASE, "backend": "local", "fs_dir": "ok", "current_version": 1},
                    {"id": "bad", "base_model": BASE, "backend": "local", "fs_dir": "bad", "current_version": 1},
                ],
                user_id="u",
            )

    def test_an_a_without_its_b_refuses(self, tmp_path, checkpoint):
        (tmp_path / "half" / "v0001").mkdir(parents=True)
        (tmp_path / "half" / "v0001" / "params.json").write_text(
            json.dumps({"layers.0.attn_q.A": [[0.1, 0.2]]})
        )
        backend = LocalJaxLoRABackend(str(checkpoint), str(tmp_path))
        with pytest.raises(ValueError, match="refusing the adapter stack"):
            backend._blend_adapter_weights(
                [{"id": "half", "base_model": BASE, "backend": "local", "fs_dir": "half",
                  "current_version": 1}], user_id="u"
            )


class TestPromotedVersionIsWhatServes:
    def test_the_promoted_version_wins_over_a_newer_directory(
        self, tmp_path, checkpoint
    ):
        """A decoy vNNNN+1 on disk must lose to the promoted current_version.

        An un-promoted version left by a gate-rejected run is exactly what a
        "newest directory" scan would pick.
        """
        config = transformer.load_config(checkpoint)
        adapter_dir = tmp_path / "adapters" / "skill"
        _write_adapter(adapter_dir / "v0001", config, value=0.05)  # promoted
        _write_adapter(adapter_dir / "v0002", config, value=0.9)  # rejected decoy

        backend = LocalJaxLoRABackend(str(checkpoint), str(tmp_path))
        adapter = {
            "id": "skill",
            "base_model": BASE,
            "backend": "local",
            "fs_dir": str(adapter_dir),
            "current_version": 1,
        }
        blended = backend._blend_adapter_weights([adapter], user_id="u")
        # v0001's constant is 0.05; v0002's would be 0.9.
        assert float(jnp.max(jnp.abs(blended["layers.0.attn_q.A"] - 0.05))) < 1e-6

    def test_the_cache_is_keyed_by_version_not_id_alone(self, tmp_path, checkpoint):
        """SPEC §5.3 keys the adapter RAM cache by (adapter_id, version)."""
        config = transformer.load_config(checkpoint)
        adapter_dir = tmp_path / "adapters" / "skill"
        _write_adapter(adapter_dir / "v0001", config, value=0.05)
        _write_adapter(adapter_dir / "v0002", config, value=0.9)
        backend = LocalJaxLoRABackend(str(checkpoint), str(tmp_path))
        base = {"id": "skill", "base_model": BASE, "backend": "local", "fs_dir": str(adapter_dir)}

        first = backend._blend_adapter_weights(
            [{**base, "current_version": 1}], user_id="u"
        )
        second = backend._blend_adapter_weights(
            [{**base, "current_version": 2}], user_id="u"
        )
        assert float(jnp.max(jnp.abs(first["layers.0.attn_q.A"] - 0.05))) < 1e-6
        assert float(jnp.max(jnp.abs(second["layers.0.attn_q.A"] - 0.9))) < 1e-6
