"""LoRA training against the real base model, per SPEC §5.2/§5.4/§5.5.

Before this, training minimised a loss over a synthetic sine table with no
attention: the optimizer worked, the gate ran, versions were written, and the
resulting weights fitted no model that exists. These tests pin the parts that
make the ladder true instead of ceremonial.

The strongest one is `test_training_teaches_the_target`. Everything else can
pass while the adapter still learns nothing useful — loss curves go down for
all sorts of wrong reasons. Asking whether the trained target actually became
more likely *in the model that will serve it* is the claim the README makes,
so it is the claim under test.
"""

import json

import pytest

from liminallm.service import transformer
from liminallm.service.training import TrainingService

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("optax")
pytest.importorskip("safetensors")

from tests.test_local_transformer import _build_checkpoint  # noqa: E402


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    return _build_checkpoint(tmp_path_factory.mktemp("train_model"))


def _service(tmp_path, base_model=None):
    return TrainingService(
        store=None, fs_root=str(tmp_path), runtime_base_model=base_model
    )


def _batch(pairs):
    """Causal-LM SFT batch: prompt+target, next-token labels, target-only mask."""
    seqs = [(prompt + target, len(prompt)) for prompt, target in pairs]
    seq_len = max(len(full) - 1 for full, _ in seqs)
    input_ids, labels, mask = [], [], []
    for full, prompt_len in seqs:
        inp, lab = full[:-1], full[1:]
        row = [1.0 if (j + 1) >= prompt_len else 0.0 for j in range(len(lab))]
        pad = seq_len - len(inp)
        input_ids.append(inp + [0] * pad)
        labels.append(lab + [0] * pad)
        mask.append(row + [0.0] * pad)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": mask,
        "shape": {"batch": len(pairs), "seq_len": seq_len},
    }


PROMPT = [5, 9, 12]
TARGET = [31, 32]


class TestLoraInitialization:
    def test_shapes_come_from_the_checkpoint(self, tmp_path, checkpoint):
        """SPEC §5.2: A is [r, d_in] and B is [d_out, r], per projection."""
        service = _service(tmp_path, str(checkpoint))
        weights = service._init_lora_weights(
            rank=4, layers=[0, 1], matrices=["attn_q", "attn_v"]
        )
        config = transformer.load_config(checkpoint)

        assert set(weights) == {
            f"layers.{layer}.{target}.{slot}"
            for layer in (0, 1)
            for target in ("attn_q", "attn_v")
            for slot in ("A", "B")
        }
        q_b = weights["layers.0.attn_q.B"]
        v_b = weights["layers.0.attn_v.B"]
        assert (len(q_b), len(q_b[0])) == (config.num_heads * config.head_dim, 4)
        # Grouped-query attention: v is narrower than q. A single "hidden
        # width" for both is the shape mistake this asserts against.
        assert (len(v_b), len(v_b[0])) == (config.num_kv_heads * config.head_dim, 4)
        assert len(v_b) < len(q_b)
        a = weights["layers.0.attn_q.A"]
        assert (len(a), len(a[0])) == (4, config.hidden_size)

    def test_b_starts_at_zero_so_a_new_adapter_is_the_identity(
        self, tmp_path, checkpoint
    ):
        """SPEC §5.5: an adapter on the prompt rung must not perturb the model
        before the data has earned any weights."""
        service = _service(tmp_path, str(checkpoint))
        weights = service._init_lora_weights(
            rank=4, layers=[0], matrices=["attn_q"]
        )
        assert all(
            value == 0.0
            for name, matrix in weights.items()
            if name.endswith(".B")
            for row in matrix
            for value in row
        )

        config, params = transformer.load_checkpoint(checkpoint)
        tokens = jnp.array([PROMPT], dtype=jnp.int32)
        baseline, _ = transformer.forward(jnp, config, params, tokens)
        fresh, _ = transformer.forward(
            jnp,
            config,
            params,
            tokens,
            lora=transformer.lora_by_layer(jnp, weights, config.num_layers),
        )
        assert float(jnp.max(jnp.abs(fresh - baseline))) == 0.0

    def test_no_base_model_means_no_weights(self, tmp_path):
        """LoRA hooks the base model's matrices; with no checkpoint there is
        nothing to hook, so the adapter stays prompt-only."""
        service = _service(tmp_path, None)
        assert service._init_lora_weights(4, [0], ["attn_q"]) == {}

    def test_layers_outside_the_model_are_dropped(self, tmp_path, checkpoint):
        service = _service(tmp_path, str(checkpoint))
        weights = service._init_lora_weights(4, [0, 99], ["attn_q"])
        assert set(weights) == {"layers.0.attn_q.A", "layers.0.attn_q.B"}


class TestTrainingRuns:
    def _train(self, tmp_path, checkpoint, *, epochs=30):
        service = _service(tmp_path, str(checkpoint))
        weights = service._init_lora_weights(
            rank=4, layers=[0, 1], matrices=["attn_q", "attn_v"]
        )
        params_path = tmp_path / "params.json"
        trace = service._run_jax_optax_training(
            weights,
            [_batch([(PROMPT, TARGET), (PROMPT + [13], TARGET)])],
            eval_batches=[_batch([(PROMPT, TARGET)])],
            params_path=params_path,
            epochs=epochs,
        )
        return service, trace, params_path

    def test_training_reduces_holdout_loss(self, tmp_path, checkpoint):
        _, trace, _ = self._train(tmp_path, checkpoint)
        assert trace["status"] == "ok"
        assert trace["eval_after"] < trace["eval_before"]

    def test_gradients_reach_b(self, tmp_path, checkpoint):
        """B is zero at init, so any nonzero value proves the gradient flowed
        through the real attention projections."""
        _, _, params_path = self._train(tmp_path, checkpoint)
        trained = json.loads(params_path.read_text())
        assert any(
            value != 0.0
            for name, matrix in trained.items()
            if name.endswith(".B")
            for row in matrix
            for value in row
        )

    def test_training_teaches_the_target(self, tmp_path, checkpoint):
        """The claim the README makes: the trained target becomes more likely
        in the model that will serve it."""
        _, _, params_path = self._train(tmp_path, checkpoint)
        trained = json.loads(params_path.read_text())
        config, params = transformer.load_checkpoint(checkpoint)
        tokens = jnp.array([PROMPT], dtype=jnp.int32)

        before, _ = transformer.forward(jnp, config, params, tokens)
        after, _ = transformer.forward(
            jnp,
            config,
            params,
            tokens,
            lora=transformer.lora_by_layer(jnp, trained, config.num_layers),
        )
        before_lp = jax.nn.log_softmax(before[0, -1])
        after_lp = jax.nn.log_softmax(after[0, -1])
        assert float(after_lp[TARGET[0]]) > float(before_lp[TARGET[0]])
        rank_before = int((before_lp > before_lp[TARGET[0]]).sum())
        rank_after = int((after_lp > after_lp[TARGET[0]]).sum())
        assert rank_after < rank_before, "training did not improve the target's rank"

    def test_the_base_model_is_never_updated(self, tmp_path, checkpoint):
        """SPEC §5.1: 'only on adapters, never on the base model'. The base
        params are closed over, not differentiated — so they must come out of
        training bit-identical."""
        service = _service(tmp_path, str(checkpoint))
        config, params = service._base_checkpoint()
        before = {
            "embed": jnp.array(params["embed"]),
            "q0": jnp.array(params["layers"][0]["attn_q"]),
        }
        weights = service._init_lora_weights(4, [0], ["attn_q"])
        service._run_jax_optax_training(
            weights,
            [_batch([(PROMPT, TARGET)])],
            params_path=tmp_path / "p.json",
            epochs=5,
        )
        _, after_params = service._base_checkpoint()
        assert float(jnp.max(jnp.abs(after_params["embed"] - before["embed"]))) == 0.0
        assert (
            float(
                jnp.max(jnp.abs(after_params["layers"][0]["attn_q"] - before["q0"]))
            )
            == 0.0
        )

    def test_an_explicit_scale_survives_training_unchanged(
        self, tmp_path, checkpoint
    ):
        """α is a hyperparameter, not a parameter.

        It used to enter the optimizer tree — the L2 term alone gives it a
        gradient — so Optax moved it while the evaluation still used the
        original value. The gate then passed under one α and serving got
        another.
        """
        service = _service(tmp_path, str(checkpoint))
        weights = service._init_lora_weights(4, [0], ["attn_q"])
        weights["layers.0.attn_q.scale"] = 0.75
        params_path = tmp_path / "scaled.json"
        trace = service._run_jax_optax_training(
            weights,
            [_batch([(PROMPT, TARGET)])],
            eval_batches=[_batch([(PROMPT, TARGET)])],
            params_path=params_path,
            epochs=10,
        )
        assert trace["status"] == "ok"
        trained = json.loads(params_path.read_text())
        assert trained["layers.0.attn_q.scale"] == 0.75
        # And the matrices did move, so this is not a no-op run.
        assert any(
            value != 0.0
            for row in trained["layers.0.attn_q.B"]
            for value in row
        )

    def test_l2_regularization_is_in_the_loss(self, tmp_path, checkpoint, monkeypatch):
        """SPEC §5.4.4 adds an L2 term. A large lambda must visibly shrink the
        learned weights; if the term were missing, nothing would change."""
        import liminallm.service.training as training_module

        def norm_after(lambda_value):
            monkeypatch.setattr(training_module, "LORA_L2_LAMBDA", lambda_value)
            service = _service(tmp_path, str(checkpoint))
            weights = service._init_lora_weights(4, [0], ["attn_q"])
            path = tmp_path / f"p{lambda_value}.json"
            service._run_jax_optax_training(
                weights,
                [_batch([(PROMPT, TARGET)])],
                params_path=path,
                epochs=20,
            )
            trained = json.loads(path.read_text())
            return sum(
                value * value
                for matrix in trained.values()
                for row in matrix
                for value in row
            )

        assert norm_after(10.0) < norm_after(1e-6)


class TestTheLadderRefusesWhatItCannotTrain:
    def test_without_a_checkpoint_the_run_is_skipped(self, tmp_path):
        """SPEC §5.4.6: a skipped run never promotes, so the adapter stays on
        the prompt rung. Training against something else and reporting success
        is the failure this prevents."""
        service = _service(tmp_path, None)
        trace = service._run_jax_optax_training(
            {"layers.0.attn_q.A": [[0.1]], "layers.0.attn_q.B": [[0.0]]},
            [_batch([(PROMPT, TARGET)])],
            params_path=tmp_path / "p.json",
        )
        assert trace["status"] == "skipped"
        assert "checkpoint" in trace["reason"]

        gate = service._promotion_gate(trace, holdout_count=5)
        assert gate["promoted"] is False

    def test_an_adapter_with_no_matching_matrices_is_skipped(
        self, tmp_path, checkpoint
    ):
        """Weights trained for another architecture must fail visibly rather
        than train a partially-applied adapter."""
        service = _service(tmp_path, str(checkpoint))
        trace = service._run_jax_optax_training(
            {"transformer.h.0.attn.c_attn.A": [[0.1]]},
            [_batch([(PROMPT, TARGET)])],
            params_path=tmp_path / "p.json",
        )
        assert trace["status"] == "skipped"
        assert service._promotion_gate(trace, holdout_count=5)["promoted"] is False
