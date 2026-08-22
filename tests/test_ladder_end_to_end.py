"""The local ladder, from a preference event to the tokens that answer next.

Every previous test stopped at a seam: `_select_adapters` on one side,
`_blend_adapter_weights` on the other. The defects that survived those tests
lived precisely between them — a gate computed and dropped, a version pinned
by a caller rather than by the artifact, weights loaded for an adapter still
on the prompt rung.

So this file follows one value the whole way:

    preference events → training → eval gate → promotion → router gate
    → LLMService → LocalJaxLoRABackend → the model that answers

with a rejected decoy version on disk and a routed gate that is not 1.0.
"""

import json
import uuid

import pytest

from liminallm.service import local_format, transformer

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("optax")
pytest.importorskip("safetensors")

from liminallm.service.llm import LLMService  # noqa: E402
from liminallm.service.model_backend import LocalJaxLoRABackend  # noqa: E402
from liminallm.service.training import TrainingService  # noqa: E402
from tests.harness import get_test_store  # noqa: E402
from tests.test_local_transformer import _build_checkpoint  # noqa: E402

# Every test in this module runs the real thing — training steps, a forward
# pass, an eval gate — so it is measured in seconds rather than milliseconds.
# `make test-fast` skips these; `make test` and the pre-commit gate do not,
# because what they exercise is not covered anywhere else.
pytestmark = pytest.mark.slow



@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    return _build_checkpoint(tmp_path_factory.mktemp("ladder_model"))


def _seed_events(store, cluster_id, count=10):
    user = store.create_user(email=f"ladder_{uuid.uuid4().hex[:8]}@t.local")
    convo = store.create_conversation(user.id, title="t")
    for index in range(count):
        store.append_message(convo.id, "user", "user", f"question {index}")
        reply = store.append_message(convo.id, "assistant", "assistant", "answer")
        store.record_preference_event(
            user.id,
            convo.id,
            reply.id,
            "positive",
            corrected_text="use tabs",
            context_embedding=[0.1] * 64,
            cluster_id=cluster_id,
            context_text=f"context {index}",
        )
    return user


class TestLadderEndToEnd:
    def test_preference_to_promoted_weights_to_a_generated_answer(
        self, tmp_path, checkpoint, client, monkeypatch
    ):
        """The promoted half of the ladder, deterministically.

        Written with an `if promoted: … else: …` first, this test passed
        while exercising nothing: the fixture's tiny random model improves
        its holdout by well under the 1% production bar, so every run took
        the refusal branch and the decoy, the gate scaling and the generated
        turn were never reached. The threshold is a product tuning constant,
        so the test relaxes *it* — the training, the gate logic, the
        promotion and the serving path all stay real, and the improvement is
        still asserted to be real.
        """
        import liminallm.service.training as training_module
        from liminallm.service.clustering import SemanticClusterer

        monkeypatch.setattr(
            training_module, "EVAL_MIN_RELATIVE_IMPROVEMENT", 1e-9
        )
        store = get_test_store()
        cluster = store.upsert_semantic_cluster(
            user_id=None, centroid=[0.1] * 64, size=10, label="tabs"
        )
        user = _seed_events(store, cluster.id)

        training = TrainingService(
            store, str(tmp_path), runtime_base_model=str(checkpoint)
        )
        # The skill is BORN on the prompt rung, the way §7.3 creates it —
        # not conjured as an already-trained artifact by the test.
        clusterer = SemanticClusterer(store, llm=None, training=training)
        born = clusterer.promote_skill_adapters(min_size=5, weights_min_events=10)
        assert born, "no skill adapter was created from the qualifying cluster"
        skill = store.get_artifact(born[0])
        assert skill.schema["mode"] == "prompt"
        assert skill.schema.get("current_version", 0) == 0

        result = training.train_from_preferences(user.id, adapter_id=skill.id)
        assert result["jax_trace"]["status"] == "ok", result["jax_trace"].get("reason")
        gate = result["eval_gate"]
        # Real improvement, not merely a relaxed bar.
        assert gate["eval_after"] < gate["eval_before"]
        assert gate["promoted"] is True

        adapter = store.get_artifact(result["adapter_id"])
        promoted_version = adapter.schema.get("current_version", 0)
        assert promoted_version >= 1
        # §5.5: graduation flips the rung and keeps the prompt as fallback.
        assert adapter.schema["mode"] == "hybrid"
        assert adapter.schema.get("prompt_instructions")
        adapter_dir = training._adapter_dir(user.id, adapter.id, adapter.schema)
        # A decoy the gate never approved, newer on disk than the promoted one.
        decoy = adapter_dir / f"v{promoted_version + 1:04d}"
        decoy.mkdir(parents=True, exist_ok=True)
        config = transformer.load_config(checkpoint)
        width = config.num_heads * config.head_dim
        (decoy / "params.json").write_text(
            json.dumps(
                {
                    "layers.0.attn_q.A": [[9.0] * config.hidden_size] * 2,
                    "layers.0.attn_q.B": [[9.0, 9.0]] * width,
                }
            )
        )

        # The 0.4 comes from a routing policy through the real RouterEngine
        # and WorkflowEngine, not from the test's hand.
        import asyncio

        from liminallm.service.runtime import get_runtime

        runtime = get_runtime()
        store.create_artifact(
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
                            "weight": 0.4,
                        },
                    }
                ],
            },
            visibility="global",
        )
        routed, _, _ = asyncio.run(
            runtime.workflow._select_adapters(
                user_message="should I use tabs?",
                user_id=user.id,
                context_id=None,
                tenant_id=user.tenant_id,
            )
        )
        served = next((a for a in routed if a["id"] == adapter.id), None)
        assert served is not None, "the promoted adapter was not routed"
        assert served["weight"] == pytest.approx(0.4)

        backend = LocalJaxLoRABackend(
            str(checkpoint), str(tmp_path), max_new_tokens=4
        )
        blended = backend._blend_adapter_weights([served], user_id=user.id)
        assert blended, "the promoted adapter produced no weights"
        # The decoy's constant is 9.0; nothing that large may appear.
        assert float(jnp.max(jnp.abs(blended["layers.0.attn_q.A"]))) < 1.0

        # The routed gate scales what the model actually receives.
        full = backend._blend_adapter_weights(
            [{**served, "weight": 1.0}], user_id=user.id
        )
        gated_delta = blended["layers.0.attn_q.B"] @ blended["layers.0.attn_q.A"]
        full_delta = full["layers.0.attn_q.B"] @ full["layers.0.attn_q.A"]
        assert float(jnp.max(jnp.abs(gated_delta - 0.4 * full_delta))) < 1e-6

        # And a whole turn runs through LLMService into the local backend,
        # carrying a context snippet so the local serialization is exercised.
        llm = LLMService(base_model=str(checkpoint), backend=backend)
        messages, _ = llm._prepare_generation(
            "should I use tabs?", [served], ["the style guide says tabs"]
        )
        rendered = local_format.format_conversation(messages)
        # One representation (§5.1), and no prompt fallback beside the
        # weights on a local backend (§5.0.1).
        assert f"{local_format.CONTEXT_ROLE}: the style guide says tabs" in rendered
        assert adapter.schema["prompt_instructions"] not in rendered

        reply = llm.generate(
            "should I use tabs?",
            [served],
            ["the style guide says tabs"],
            history=None,
            user_id=user.id,
        )
        assert isinstance(reply.get("content"), str)
        assert reply["usage"]["prompt_tokens"] > 0

    def test_a_prompt_rung_adapter_never_loads_weights(
        self, tmp_path, checkpoint
    ):
        """SPEC §5.5: weights arrive on graduation, not before — even if a
        training job has already written a version directory."""
        adapter_dir = tmp_path / "adapters" / "skill"
        version_dir = adapter_dir / "v0001"
        version_dir.mkdir(parents=True)
        config = transformer.load_config(checkpoint)
        (version_dir / "params.json").write_text(
            json.dumps(
                {
                    "layers.0.attn_q.A": [[0.5] * config.hidden_size] * 2,
                    "layers.0.attn_q.B": [[0.5, 0.5]]
                    * (config.num_heads * config.head_dim),
                }
            )
        )
        backend = LocalJaxLoRABackend(str(checkpoint), str(tmp_path))

        prompt_rung = {
            "id": "skill",
            "mode": "prompt",
            # Declared because a real artifact declares it (§5.1); the prompt
            # rung itself is exempt, carrying no weights to be fitted.
            "base_model": str(checkpoint),
            "fs_dir": str(adapter_dir),
            "current_version": 0,
        }
        assert backend._blend_adapter_weights([prompt_rung], user_id="u") == {}

        # Version 0 alone is also unservable, whatever the mode says.
        unpromoted = {**prompt_rung, "mode": "local"}
        assert backend._blend_adapter_weights([unpromoted], user_id="u") == {}

        # And once promoted, the same files load.
        promoted = {**unpromoted, "current_version": 1}
        assert backend._blend_adapter_weights([promoted], user_id="u")


class TestBrokenCheckpointFailsClosed:
    def test_a_broken_checkpoint_never_degrades_to_the_stand_in(
        self, tmp_path, checkpoint
    ):
        """`_model_state is None` used to mean both "dev box, use the
        stand-in" and "misconfigured, refuse" — so a refused request was
        followed by one answered from the synthetic model."""
        import shutil

        broken = tmp_path / "broken"
        shutil.copytree(checkpoint, broken)
        (broken / "tokenizer.json").unlink()
        (broken / "tokenizer_config.json").unlink()

        backend = LocalJaxLoRABackend(str(broken), str(tmp_path))
        messages = [{"role": "user", "content": "hello there"}]

        for attempt in range(3):
            with pytest.raises(ValueError, match="local model refused"):
                backend.generate(messages, [], user_id="u")
            assert backend._checkpoint_state == "broken", f"attempt {attempt}"

    def test_an_absent_checkpoint_still_uses_the_stand_in(self, tmp_path):
        """The dev/CI lane must stay open — that is the whole reason the
        states are distinguished rather than merged."""
        backend = LocalJaxLoRABackend(str(tmp_path / "nothing"), str(tmp_path))
        result = backend.generate(
            [{"role": "user", "content": "hello"}], [], user_id="u"
        )
        assert backend._checkpoint_state == "absent"
        assert "content" in result
