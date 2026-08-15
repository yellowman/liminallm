"""Contracts that held only where the earlier tests happened to look.

Each of these survived a round of review because the mechanism was right and
the rule was enforced one step too late, or from the wrong field:

* validation ran on the *composed* pair, which cannot see two adapters whose
  ranks only agree after concatenation;
* a promoted adapter whose weights would not load simply vanished from the
  stack instead of refusing it;
* a positive `current_version` still accepted `latest` without checking where
  it pointed;
* the worker defaulted a missing gate decision to "promoted";
* prompt injection keyed off `backend` while SPEC calls `mode` authoritative;
* training and serving agreed on the context marker and disagreed on where
  the context goes, which for a raw decoder is a different input.
"""

import json
import uuid

import pytest

from liminallm.service import local_format, transformer

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("safetensors")

from liminallm.service.llm import LLMService  # noqa: E402
from liminallm.service.model_backend import LocalJaxLoRABackend  # noqa: E402
from liminallm.service.training import TrainingService  # noqa: E402
from tests.harness import get_test_store  # noqa: E402
from tests.test_local_transformer import _build_checkpoint  # noqa: E402


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    return _build_checkpoint(tmp_path_factory.mktemp("contract2_model"))


@pytest.fixture(scope="module")
def config(checkpoint):
    return transformer.load_config(checkpoint)


def _write(directory, weights):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "params.json").write_text(json.dumps(weights))


class TestValidationHappensPerAdapter:
    def test_ranks_that_only_agree_after_concatenation_are_refused(
        self, tmp_path, checkpoint, config
    ):
        """Two individually invalid adapters can compose into a pair whose
        totals look right: rank 2+1 of A and 1+2 of B both sum to 3, so a
        post-composition check passes while every row pairs with the wrong
        column."""
        width = config.num_heads * config.head_dim
        _write(
            tmp_path / "a1" / "v0001",
            {
                "layers.0.attn_q.A": [[0.1] * config.hidden_size] * 2,  # rank 2
                "layers.0.attn_q.B": [[0.1]] * width,  # rank 1
            },
        )
        _write(
            tmp_path / "a2" / "v0001",
            {
                "layers.0.attn_q.A": [[0.1] * config.hidden_size] * 1,  # rank 1
                "layers.0.attn_q.B": [[0.1, 0.1]] * width,  # rank 2
            },
        )
        backend = LocalJaxLoRABackend(str(checkpoint), str(tmp_path))
        stack = [
            {"id": "a1", "mode": "local", "fs_dir": "a1", "current_version": 1},
            {"id": "a2", "mode": "local", "fs_dir": "a2", "current_version": 1},
        ]
        with pytest.raises(ValueError, match="rank"):
            backend._blend_adapter_weights(stack, user_id="u", config=config)
        # Even with no config to check dimensions against, self-consistency is
        # still knowable.
        with pytest.raises(ValueError, match="rank"):
            backend._blend_adapter_weights(stack, user_id="u")

    def test_a_foreign_key_is_caught_before_composition_discards_it(
        self, tmp_path, checkpoint, config
    ):
        """Composition only carries A/B pairs forward, so a foreign key never
        reached a validator that ran afterwards."""
        _write(
            tmp_path / "mixed" / "v0001",
            {
                "layers.0.attn_q.A": [[0.1] * config.hidden_size] * 2,
                "layers.0.attn_q.B": [[0.1, 0.1]] * (config.num_heads * config.head_dim),
                "encoder.block.0.layer.0.SelfAttention.q.weight": [[0.1]],
            },
        )
        backend = LocalJaxLoRABackend(str(checkpoint), str(tmp_path))
        with pytest.raises(ValueError, match="outside the declared shape"):
            backend._blend_adapter_weights(
                [{"id": "mixed", "mode": "local", "fs_dir": "mixed", "current_version": 1}],
                user_id="u",
                config=config,
            )


class TestOrphanScaleCannotHide:
    def test_a_scale_only_adapter_is_refused(self, tmp_path, checkpoint, config):
        """The raw dict is non-empty, so the "never silently leaves the
        stack" check did not fire — and composition, which iterates `.A`,
        produced nothing. A promoted adapter contributed exactly nothing
        while looking present."""
        _write(tmp_path / "scaly" / "v0001", {"layers.0.attn_q.scale": 0.5})
        backend = LocalJaxLoRABackend(str(checkpoint), str(tmp_path))
        with pytest.raises(ValueError, match="missing its A matrix"):
            backend._blend_adapter_weights(
                [
                    {
                        "id": "scaly",
                        "mode": "local",
                        "fs_dir": "scaly",
                        "current_version": 1,
                        "weight": 0.5,
                    }
                ],
                user_id="u",
                config=config,
            )

    def test_a_scale_for_a_projection_with_no_matrices_is_refused(self, config):
        weights = {
            "layers.0.attn_q.A": [[0.1] * config.hidden_size] * 2,
            "layers.0.attn_q.B": [[0.1, 0.1]] * (config.num_heads * config.head_dim),
            "layers.0.attn_v.scale": 0.5,  # orphan: no attn_v matrices
        }
        with pytest.raises(ValueError, match="attn_v"):
            transformer.validate_lora_weights(config, weights)

    def test_a_non_scalar_scale_is_refused(self, config):
        weights = {
            "layers.0.attn_q.A": [[0.1] * config.hidden_size] * 2,
            "layers.0.attn_q.B": [[0.1, 0.1]] * (config.num_heads * config.head_dim),
            "layers.0.attn_q.scale": [[0.5]],
        }
        with pytest.raises(ValueError, match="must be a scalar"):
            transformer.validate_lora_weights(config, weights)

    def test_the_same_rules_hold_without_a_model_config(self):
        """One validator, not two: the no-checkpoint lane checks everything
        knowable from the weights alone."""
        with pytest.raises(ValueError, match="outside the declared shape"):
            transformer.validate_lora_weights(None, {"foreign.0.x.A": [[0.1]]})
        with pytest.raises(ValueError, match="missing its A matrix"):
            transformer.validate_lora_weights(None, {"layers.0.attn_q.scale": 0.5})
        with pytest.raises(ValueError, match="rank"):
            transformer.validate_lora_weights(
                None,
                {
                    "layers.0.attn_q.A": [[0.1, 0.2]] * 2,
                    "layers.0.attn_q.B": [[0.1]] * 4,
                },
            )


class TestASelectedAdapterNeverVanishes:
    def test_promoted_but_unloadable_refuses_the_stack(self, tmp_path, checkpoint):
        """The router chose it; serving without it is serving a different
        stack. (Version 3 is promoted, but only v0001 exists on disk.)"""
        _write(tmp_path / "skill" / "v0001", {})
        backend = LocalJaxLoRABackend(str(checkpoint), str(tmp_path))
        with pytest.raises(ValueError, match="could not be loaded"):
            backend._blend_adapter_weights(
                [
                    {
                        "id": "skill",
                        "mode": "local",
                        "fs_dir": "skill",
                        "current_version": 3,
                        "weight": 0.5,
                    }
                ],
                user_id="u",
            )

    def test_weightless_by_design_is_still_fine(self, tmp_path, checkpoint):
        backend = LocalJaxLoRABackend(str(checkpoint), str(tmp_path))
        # Prompt rung.
        assert backend._blend_adapter_weights(
            [{"id": "p", "mode": "prompt", "fs_dir": "nothing", "current_version": 0}],
            user_id="u",
        ) == {}
        # Nothing promoted yet.
        assert backend._blend_adapter_weights(
            [{"id": "l", "mode": "local", "fs_dir": "nothing", "current_version": 0}],
            user_id="u",
        ) == {}
        # Closed gate.
        assert backend._blend_adapter_weights(
            [
                {
                    "id": "l",
                    "mode": "local",
                    "fs_dir": "nothing",
                    "current_version": 2,
                    "weight": 0.0,
                }
            ],
            user_id="u",
        ) == {}


class TestVersionAuthorityIsAbsolute:
    def test_latest_takes_no_part_in_authoritative_resolution(
        self, tmp_path, checkpoint
    ):
        """`latest` proved only that its target was *named* vNNNN, so
        `A/latest -> B/v0001` served another adapter's weights as A's version
        1. And it enabled nothing: a pointer legitimately aimed at A/vNNNN
        means A/vNNNN exists, which the exact path already returns."""
        other = tmp_path / "other"
        _write(other / "v0001", {"layers.0.attn_q.A": [[0.9]]})
        adapter_dir = tmp_path / "skill"
        adapter_dir.mkdir(parents=True)
        (adapter_dir / "latest").symlink_to(other / "v0001")

        backend = LocalJaxLoRABackend(str(checkpoint), str(tmp_path))
        # Version 1 is promoted, skill/v0001 does not exist, and `latest`
        # points into a different adapter entirely.
        assert backend._resolve_params_path(adapter_dir, current_version=1) is None

        # Even a same-adapter pointer is not consulted; the real directory is.
        (adapter_dir / "latest").unlink()
        (adapter_dir / "latest").symlink_to(adapter_dir / "v0001")
        assert backend._resolve_params_path(adapter_dir, current_version=1) is None
        _write(adapter_dir / "v0001", {"layers.0.attn_q.A": [[0.1]]})
        resolved = backend._resolve_params_path(adapter_dir, current_version=1)
        assert resolved == adapter_dir / "v0001" / "params.json"

    def test_a_direct_file_cannot_satisfy_a_versioned_artifact(
        self, tmp_path, checkpoint
    ):
        """A bare params.json cannot demonstrate which version it is. This
        used to be accepted for any positive version — the earlier test even
        asserted it."""
        params = tmp_path / "loose" / "params.json"
        _write(tmp_path / "loose", {"layers.0.attn_q.A": [[0.1]]})
        backend = LocalJaxLoRABackend(str(checkpoint), str(tmp_path))
        assert backend._resolve_params_path(params, current_version=1) is None
        assert backend._resolve_params_path(params, current_version=0) is None
        # An artifact that was never versioned may still use a direct path.
        assert backend._resolve_params_path(params, current_version=None) == params


class TestTheWorkerCarriesTheGateDecision:
    def test_describe_run_keeps_eval_gate(self, tmp_path):
        service = TrainingService(store=None, fs_root=str(tmp_path))
        summary = service.describe_run(
            {
                "loss": 1.0,
                "new_version": 2,
                "jax_trace": {"status": "ok"},
                "eval_gate": {"promoted": False, "reason": "regressed"},
            }
        )
        assert summary["eval_gate"] == {"promoted": False, "reason": "regressed"}

    def test_a_rejected_run_is_not_recorded_as_a_rollout(self, tmp_path):
        """Executed, not inspected.

        The first version of this test read the worker's source for the
        literal default — which proves the character is present, not that a
        rejected run leaves the adapter uncredited. This drives the real
        `_process_job` against a gate-rejected result.
        """
        import asyncio

        from liminallm.service.training_worker import TrainingWorker

        store = get_test_store()
        user = store.create_user(email=f"worker_{uuid.uuid4().hex[:8]}@t.local")
        adapter = store.create_artifact(
            "adapter",
            f"a_{uuid.uuid4().hex[:6]}",
            {
                "kind": "adapter.lora",
                "mode": "local",
                "backend": "local",
                "base_model": "b",
                "current_version": 0,
            },
            owner_user_id=user.id,
        )
        job = store.create_training_job(user.id, adapter.id)

        service = TrainingService(store=store, fs_root=str(tmp_path))
        credited = []

        def rejected_run(**kwargs):
            return {
                "loss": 1.0,
                "new_version": 1,
                "jax_trace": {"status": "ok"},
                "eval_gate": {"promoted": False, "reason": "regressed"},
            }

        service.train_from_preferences = rejected_run  # type: ignore[assignment]
        service.record_training_outcome = lambda *a, **k: credited.append(a)  # type: ignore[assignment]

        worker = TrainingWorker(store, service)
        asyncio.run(worker._process_job(job))

        refreshed = store.get_training_job(job.id)
        assert refreshed.status == "gate_rejected", refreshed.status
        assert credited == [], "a gate-rejected run was credited as a rollout"
        assert store.get_artifact(adapter.id).schema.get("current_version", 0) == 0


class TestModeIsAuthoritative:
    class _Local:
        applies_lora_weights = True

        def generate(self, messages, adapters, **kwargs):
            return {"messages": messages}

    def _prompts(self, adapter):
        service = LLMService(base_model="m", backend=self._Local())
        messages, _ = service._prepare_generation("hi", [adapter], [])
        return "\n".join(m["content"] for m in messages if m["role"] == "system")

    def test_mode_wins_over_a_disagreeing_backend_field(self):
        """`mode: hybrid, backend: prompt` used to get the prompt AND the
        weights once promoted, because injection read `backend`."""
        adapter = {
            "id": "s",
            "mode": "hybrid",
            "backend": "prompt",
            "current_version": 2,
            "prompt_instructions": "prefer tabs",
        }
        assert "prefer tabs" not in self._prompts(adapter)

    def test_prompt_mode_with_a_local_backend_field_still_injects(self):
        """And the mirror: `mode: prompt, backend: local` got neither weights
        (prompt rung) nor prompt (not a prompt backend)."""
        adapter = {
            "id": "s",
            "mode": "prompt",
            "backend": "local",
            "prompt_instructions": "be terse",
        }
        assert "be terse" in self._prompts(adapter)


class TestTrainingAndServingSerializeIdentically:
    def test_the_canonical_conversation_serialization_matches(self, tmp_path):
        """Not "context: appears somewhere" — the same conversation and
        snippet must produce the same string on both sides, ordering
        included.

        The comparison is of the canonical *conversation* serialization:
        roles, the retrieved-context representation and its placement. The
        runtime system instruction serving prepends is application-level
        content, not part of that convention, so it is excluded rather than
        retroactively made an invariant.
        """
        store = get_test_store()
        user = store.create_user(email=f"fmt_{uuid.uuid4().hex[:8]}@t.local")
        convo = store.create_conversation(user.id, title="c")
        store.append_message(convo.id, "user", "user", "how do I indent?")
        target = store.append_message(convo.id, "assistant", "assistant", "use tabs")
        event = store.record_preference_event(
            user.id,
            convo.id,
            target.id,
            "positive",
            corrected_text="use tabs",
            context_embedding=[0.1] * 64,
            context_text="the style guide says tabs",
        )
        service = TrainingService(store=store, fs_root=str(tmp_path))
        (example,) = list(service._build_examples([event]))

        llm = LLMService(base_model="m", backend=TestModeIsAuthoritative._Local())
        messages, _ = llm._prepare_generation(
            "how do I indent?", [], ["the style guide says tabs"]
        )
        # Drop the serving-only system preamble; compare the conversation.
        conversation = [m for m in messages if m["role"] != "system"]
        served = local_format.format_conversation(conversation)

        assert example["prompt"] == served, (
            "training and serving disagree on the local representation:\n"
            f"training: {example['prompt']!r}\nserving : {served!r}"
        )
