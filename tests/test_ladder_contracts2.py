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
    def test_latest_pointing_elsewhere_is_refused(self, tmp_path, checkpoint, config):
        adapter_dir = tmp_path / "skill"
        _write(adapter_dir / "v0002", {"layers.0.attn_q.A": [[0.9]]})
        (adapter_dir / "latest").symlink_to(adapter_dir / "v0002")
        backend = LocalJaxLoRABackend(str(checkpoint), str(tmp_path))
        # v0001 is promoted but absent; `latest` resolves to v0002 and must
        # not be substituted for it.
        assert (
            backend._resolve_params_path(adapter_dir, current_version=1) is None
        )
        # Pointing at the pinned version, it is acceptable.
        (adapter_dir / "latest").unlink()
        _write(adapter_dir / "v0001", {"layers.0.attn_q.A": [[0.1]]})
        (adapter_dir / "latest").symlink_to(adapter_dir / "v0001")
        assert backend._resolve_params_path(adapter_dir, current_version=1) is not None

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

    def test_a_missing_decision_is_not_approval(self):
        """The worker read `gate.get("promoted", True)` while the summary it
        was reading had dropped eval_gate entirely, so every run was credited
        as promoted."""
        import inspect

        from liminallm.service import training_worker

        source = inspect.getsource(training_worker.TrainingWorker._process_job)
        assert 'gate.get("promoted", False)' in source
        assert 'gate.get("promoted", True)' not in source


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
    def test_the_exact_prefix_matches(self, tmp_path):
        """Not "context: appears somewhere" — the same conversation and
        snippet must produce the same string on both sides, ordering
        included."""
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
