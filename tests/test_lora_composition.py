"""Adapter composition and tokenizer fidelity — SPEC §5.2 and §5.4.

Every test here covers a case the earlier suite could not distinguish. The
old composition averaged A and B and divided by the total gate weight, which
is *correct* for exactly one input — a single adapter at weight 1.0 — and
that was the only input anything tested. So:

* gates other than 1.0, where the old code cancelled the gate entirely;
* two adapters at once, where averaging produces cross terms B₁A₂ + B₂A₁
  that appear in no SPEC sum;
* a closed gate, which must contribute nothing rather than be normalized
  back into existence;
* a real checkpoint whose tokenizer will not load, where gradients through
  the right weights still teach the wrong token space.
"""

import json

import numpy as np
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
    directory = _build_checkpoint(tmp_path_factory.mktemp("compose_model"))
    BASE = str(directory)
    return directory


@pytest.fixture(scope="module")
def config(checkpoint):
    return transformer.load_config(checkpoint)


@pytest.fixture(scope="module")
def checkpoint_without_tokenizer(tmp_path_factory):
    """Weights present, tokenizer absent — a misconfigured deployment."""
    directory = _build_checkpoint(tmp_path_factory.mktemp("no_tokenizer"))
    for name in ("tokenizer.json", "tokenizer_config.json"):
        (directory / name).unlink()
    return directory


def _adapter_on_disk(directory, config, *, seed, rank=4, scale=None):
    """A real params.json for one adapter, sized to the model."""
    rng = np.random.default_rng(seed)
    width = config.num_heads * config.head_dim
    weights = {
        "layers.0.attn_q.A": rng.normal(0, 0.2, (rank, config.hidden_size)).tolist(),
        "layers.0.attn_q.B": rng.normal(0, 0.2, (width, rank)).tolist(),
    }
    if scale is not None:
        weights["layers.0.attn_q.scale"] = scale
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "params.json").write_text(json.dumps(weights))
    return weights


def _effective_delta(blended, key="layers.0.attn_q"):
    """The ΔW the composition applies: `B @ A`.

    SPEC §5.2 is an equation about weights, and weights compose exactly.
    Asserting on logits instead would be asserting that the whole network is
    linear in the adapter delta — it is not, because the delta passes through
    a softmax and a SwiGLU, so logit changes are only proportional to first
    order. Testing ΔW tests the actual claim, exactly.
    """
    return blended[f"{key}.B"] @ blended[f"{key}.A"]


def _logit_delta(config, params, blended, tokens):
    """How much the composed adapter moves the logits (direction, not scale)."""
    base, _ = transformer.forward(jnp, config, params, tokens)
    lora = transformer.lora_by_layer(jnp, blended, config.num_layers)
    adapted, _ = transformer.forward(jnp, config, params, tokens, lora=lora)
    return adapted - base


TOKENS = jnp.array([[3, 9, 27]], dtype=jnp.int32)


class TestGateWeights:
    def test_the_gate_actually_scales_the_delta(self, tmp_path, checkpoint, config):
        """SPEC §5.2: the delta is g·α·BA. Averaging then normalizing gave
        (gA)/g = A, so every gate produced an identical delta."""
        backend = LocalJaxLoRABackend(str(checkpoint), str(tmp_path))
        _adapter_on_disk(tmp_path / "one", config, seed=1)
        _, params = transformer.load_checkpoint(checkpoint)
        adapter = {"id": "one", "base_model": BASE, "backend": "local", "fs_dir": "one"}

        full = _effective_delta(
            backend._blend_adapter_weights([{**adapter, "weight": 1.0}], user_id="u")
        )
        quarter = _effective_delta(
            backend._blend_adapter_weights([{**adapter, "weight": 0.25}], user_id="u")
        )
        assert float(jnp.max(jnp.abs(full))) > 1e-6
        # A quarter gate applies a quarter of the weight delta, exactly.
        assert float(jnp.max(jnp.abs(quarter - 0.25 * full))) < 1e-6
        # And it reaches the model: a smaller gate moves the logits less.
        moved_full = _logit_delta(
            config,
            params,
            backend._blend_adapter_weights([{**adapter, "weight": 1.0}], user_id="u"),
            TOKENS,
        )
        moved_quarter = _logit_delta(
            config,
            params,
            backend._blend_adapter_weights([{**adapter, "weight": 0.25}], user_id="u"),
            TOKENS,
        )
        assert float(jnp.max(jnp.abs(moved_quarter))) < float(
            jnp.max(jnp.abs(moved_full))
        )

    def test_a_closed_gate_contributes_nothing(self, tmp_path, checkpoint, config):
        backend = LocalJaxLoRABackend(str(checkpoint), str(tmp_path))
        _adapter_on_disk(tmp_path / "off", config, seed=2)
        blended = backend._blend_adapter_weights(
            [{"id": "off", "base_model": BASE, "backend": "local", "fs_dir": "off", "weight": 0.0}],
            user_id="u",
        )
        assert blended == {}

    def test_explicit_scale_is_honored(self, tmp_path, checkpoint, config):
        """α in the params (SPEC §5.2) reaches the delta. It used to be
        classified as an unmatched name, so the declared serialization
        contract was not round-trippable."""
        backend = LocalJaxLoRABackend(str(checkpoint), str(tmp_path))
        _adapter_on_disk(tmp_path / "plain", config, seed=3)
        _adapter_on_disk(tmp_path / "scaled", config, seed=3, scale=0.5)
        _, params = transformer.load_checkpoint(checkpoint)

        plain = _effective_delta(
            backend._blend_adapter_weights(
                [{"id": "plain", "base_model": BASE, "backend": "local", "fs_dir": "plain"}], user_id="u"
            )
        )
        scaled = _effective_delta(
            backend._blend_adapter_weights(
                [{"id": "scaled", "base_model": BASE, "backend": "local", "fs_dir": "scaled"}], user_id="u"
            )
        )
        assert float(jnp.max(jnp.abs(plain))) > 1e-6
        assert float(jnp.max(jnp.abs(scaled - 0.5 * plain))) < 1e-6

    def test_a_scale_name_is_not_reported_as_unmatched(self, config):
        weights = {
            "layers.0.attn_q.A": [[0.1] * config.hidden_size] * 2,
            "layers.0.attn_q.B": [[0.1, 0.1]] * (config.num_heads * config.head_dim),
            "layers.0.attn_q.scale": 0.5,
        }
        lora = transformer.lora_by_layer(jnp, weights, config.num_layers)
        assert lora is not None
        assert lora[0]["attn_q.scale"] is not None


class TestTwoAdaptersCompose:
    def test_sum_of_deltas_not_a_product_of_averages(
        self, tmp_path, checkpoint, config
    ):
        """The composed delta must equal the sum of the individual deltas.

        Averaging A and B produces B̄Ā, whose expansion contains B₁A₂ and
        B₂A₁ — one adapter's up-projection against another's down-projection.
        No term of the SPEC sum has that shape.
        """
        backend = LocalJaxLoRABackend(str(checkpoint), str(tmp_path))
        _adapter_on_disk(tmp_path / "a1", config, seed=11)
        _adapter_on_disk(tmp_path / "a2", config, seed=22, rank=2)  # differing rank
        _, params = transformer.load_checkpoint(checkpoint)
        first = {"id": "a1", "base_model": BASE, "backend": "local", "fs_dir": "a1", "weight": 0.6}
        second = {"id": "a2", "base_model": BASE, "backend": "local", "fs_dir": "a2", "weight": 0.4}

        alone_first = _effective_delta(
            backend._blend_adapter_weights([first], user_id="u")
        )
        alone_second = _effective_delta(
            backend._blend_adapter_weights([second], user_id="u")
        )
        together = _effective_delta(
            backend._blend_adapter_weights([first, second], user_id="u")
        )

        # ΔW_together == g₁α₁B₁A₁ + g₂α₂B₂A₂, exactly — no cross terms.
        assert float(jnp.max(jnp.abs(together - (alone_first + alone_second)))) < 1e-6
        # And the composition is not the degenerate "one of them wins".
        assert float(jnp.max(jnp.abs(together - alone_first))) > 1e-6
        # Both adapters actually reach the model.
        assert (
            float(
                jnp.max(
                    jnp.abs(
                        _logit_delta(
                            config,
                            params,
                            backend._blend_adapter_weights([first, second], user_id="u"),
                            TOKENS,
                        )
                    )
                )
            )
            > 1e-6
        )

    def test_ranks_concatenate_rather_than_collide(self, tmp_path, checkpoint, config):
        backend = LocalJaxLoRABackend(str(checkpoint), str(tmp_path))
        _adapter_on_disk(tmp_path / "r4", config, seed=5, rank=4)
        _adapter_on_disk(tmp_path / "r2", config, seed=6, rank=2)
        blended = backend._blend_adapter_weights(
            [
                {"id": "r4", "base_model": BASE, "backend": "local", "fs_dir": "r4"},
                {"id": "r2", "base_model": BASE, "backend": "local", "fs_dir": "r2"},
            ],
            user_id="u",
        )
        assert blended["layers.0.attn_q.A"].shape[0] == 6  # 4 + 2 ranks
        assert blended["layers.0.attn_q.B"].shape[1] == 6


class TestTokenizerFidelity:
    def test_serving_refuses_a_checkpoint_whose_tokenizer_is_missing(
        self, tmp_path, checkpoint_without_tokenizer
    ):
        """Real weights reached through an invented token space are not the
        real model: every embedding lookup would be arbitrary."""
        backend = LocalJaxLoRABackend(
            str(checkpoint_without_tokenizer), str(tmp_path)
        )
        backend._ensure_model()
        assert backend._model_state is None
        assert "tokenizer" in (backend._model_error or "")

    def test_serving_accepts_a_complete_checkpoint(self, tmp_path, checkpoint):
        """The same check must not refuse a checkout that has its tokenizer."""
        backend = LocalJaxLoRABackend(str(checkpoint), str(tmp_path))
        backend._ensure_model()
        assert backend._model_state is not None
        assert backend._model_error is None

    def test_training_skips_when_the_real_tokenizer_is_missing(
        self, tmp_path, checkpoint_without_tokenizer
    ):
        """Gradients through the right weights still teach the wrong token
        space, and the holdout — tokenized the same wrong way — agrees."""
        from liminallm.service.training import TrainingService

        checkpoint = checkpoint_without_tokenizer
        service = TrainingService(
            store=None, fs_root=str(tmp_path), runtime_base_model=str(checkpoint)
        )
        service._ensure_tokenizer(str(checkpoint))
        assert service.tokenizer is None  # this fixture ships no tokenizer

        weights = service._init_lora_weights(4, [0], ["attn_q"])
        trace = service._run_jax_optax_training(
            weights,
            [
                {
                    "input_ids": [[1, 2]],
                    "labels": [[2, 3]],
                    "attention_mask": [[0.0, 1.0]],
                    "shape": {"batch": 1, "seq_len": 2},
                }
            ],
            params_path=tmp_path / "p.json",
        )
        assert trace["status"] == "skipped"
        assert "tokenizer" in trace["reason"]
        assert service._promotion_gate(trace, holdout_count=5)["promoted"] is False

    @pytest.mark.parametrize(
        ("input_ids", "labels"),
        [
            ([[1, 99_999]], [[99_999, 3]]),  # above the vocabulary
            ([[1, -1]], [[-1, 3]]),  # below it — and negative indexing does
            ([[1, 2]], [[-5, 3]]),  # not crash, it reads from the far end
        ],
    )
    def test_training_refuses_ids_outside_the_checkpoint_vocabulary(
        self, tmp_path, checkpoint, monkeypatch, input_ids, labels
    ):
        """A mismatch is refused, not clipped into a token nobody wrote.

        Both ends of [0, vocab_size): a negative id is as far outside the
        vocabulary as an oversized one, and it is the quieter of the two —
        array indexing resolves it against the end of the embedding table
        instead of failing.
        """
        from liminallm.service.training import TrainingService

        service = TrainingService(
            store=None, fs_root=str(tmp_path), runtime_base_model=str(checkpoint)
        )
        # Stand in for a loaded tokenizer so the check under test is reached.
        monkeypatch.setattr(service, "tokenizer", object())
        weights = service._init_lora_weights(4, [0], ["attn_q"])
        trace = service._run_jax_optax_training(
            weights,
            [
                {
                    "input_ids": input_ids,
                    "labels": labels,
                    "attention_mask": [[0.0, 1.0]],
                    "shape": {"batch": 1, "seq_len": 2},
                }
            ],
            params_path=tmp_path / "p.json",
        )
        assert trace["status"] == "skipped"
        assert "vocabulary" in trace["reason"]


class TestSftSequenceConstruction:
    def _service(self, tmp_path, tokenizer):
        from liminallm.service.training import TrainingService

        service = TrainingService(store=None, fs_root=str(tmp_path))
        service.tokenizer = tokenizer
        service._tokenizer_error = None
        return service

    def test_truncation_keeps_the_newest_prompt_context(self, tmp_path):
        """The *newest* turn must survive, not the oldest.

        The tokenizer was previously asked to truncate first, and tokenizers
        truncate from the right — so a long prompt was already reduced to its
        oldest tokens before the deliberate "keep the tail" slice ran, which
        could then only pick among tokens that had lost the recent context.
        This tokenizer does not truncate at all, so the code under test has to
        do it correctly itself.
        """

        class HonestTokenizer:
            """Never truncates: truncation is the caller's decision."""

            def encode(self, text, truncation=False, add_special_tokens=True, **kw):
                mapping = {"EARLY": 40, "LATEST": 41, "ANSWER": 42}
                ids = [mapping.get(word, 7) for word in text.split()]
                return ([1] + ids) if add_special_tokens else ids

        service = self._service(tmp_path, HonestTokenizer())
        prompt = "EARLY " + ("filler " * 300) + "LATEST"
        entries = [{"prompt": prompt, "target": "ANSWER"}]
        (batch,) = list(service._tokenize_batches(entries, batch_size=1, max_length=32))

        sequence = batch["input_ids"][0] + [batch["labels"][0][-1]]
        assert 41 in sequence, "the newest prompt context was truncated away"
        assert 40 not in sequence, "the oldest context should have been trimmed"
        assert sum(batch["attention_mask"][0]) > 0

    def test_an_empty_target_is_dropped_not_invented(self, tmp_path):
        """`or [0]` turned a missing correction into supervision teaching the
        model to emit token 0 — with positive mask weight, so no later
        zero-mask check could catch it."""

        class WordTokenizer:
            def encode(self, text, truncation=False, add_special_tokens=True, **kw):
                ids = [(len(word) % 30) + 2 for word in text.split()]
                return ([1] + ids) if add_special_tokens else ids

        service = self._service(tmp_path, WordTokenizer())
        entries = [
            {"prompt": "a question", "target": ""},
            {"prompt": "another question", "target": "   "},
        ]
        assert list(service._tokenize_batches(entries, batch_size=2, max_length=32)) == []

        # A real target alongside an empty one keeps only the real one.
        mixed = [
            {"prompt": "a question", "target": ""},
            {"prompt": "another question", "target": "real answer"},
        ]
        (batch,) = list(service._tokenize_batches(mixed, batch_size=2, max_length=32))
        assert len(batch["input_ids"]) == 1  # only one usable example
        # The shape describes the rows actually emitted. It used to report the
        # source slice, so a consumer preallocating from it would size for
        # examples that were dropped — and this assertion used to pin that.
        assert batch["shape"]["batch"] == 1

    def test_a_long_prompt_never_erases_the_target(self, tmp_path):
        """Slicing the head of prompt+target could drop the whole target,
        leaving an all-zero mask: a zero loss that reads as a perfect
        example."""

        class WordTokenizer:
            def encode(self, text, truncation=True, max_length=None, add_special_tokens=True):
                ids = [(len(word) % 50) + 1 for word in text.split()]
                if add_special_tokens:
                    ids = [1] + ids
                return ids[:max_length] if max_length else ids

        service = self._service(tmp_path, WordTokenizer())
        entries = [{"prompt": "word " * 400, "target": "the wanted answer"}]
        batches = list(service._tokenize_batches(entries, batch_size=1, max_length=64))

        assert batches, "the example was dropped entirely"
        mask = batches[0]["attention_mask"][0]
        assert sum(mask) > 0, "no supervised tokens survived truncation"
        # The target's tokens are the ones being predicted.
        assert sum(mask) >= 3

    def test_text_to_batches_to_training_to_serving(self, tmp_path, checkpoint):
        """The whole path with no hand-written token ids anywhere.

        Every other training test starts from token lists, which cannot catch
        a tokenizer that disagrees with the model — the batches would be
        internally consistent and wrong. Here the text goes through the
        production tokenizer, the batches feed the real trainer, and the
        trained file is loaded by the serving backend.
        """
        from liminallm.service.training import TrainingService

        service = TrainingService(
            store=None, fs_root=str(tmp_path), runtime_base_model=str(checkpoint)
        )
        entries = [
            {"prompt": "question the context", "target": "use tabs"} for _ in range(4)
        ]
        batches = list(service._tokenize_batches(entries, batch_size=2, max_length=32))
        assert batches, "the production tokenizer produced no batches"
        # Real ids from the model's own tokenizer, inside its vocabulary.
        config = transformer.load_config(checkpoint)
        for batch in batches:
            for sequence in batch["input_ids"]:
                assert max(sequence) < config.vocab_size
            assert any(sum(row) > 0 for row in batch["attention_mask"])

        weights = service._init_lora_weights(4, [0, 1], ["attn_q", "attn_v"])
        params_path = tmp_path / "e2e.json"
        trace = service._run_jax_optax_training(
            weights, batches, eval_batches=batches[:1],
            params_path=params_path, epochs=20,
        )
        assert trace["status"] == "ok", trace.get("reason")
        assert trace["eval_after"] < trace["eval_before"]

        # And serving consumes exactly this file, through gate composition.
        backend = LocalJaxLoRABackend(str(checkpoint), str(tmp_path))
        # The directory carries the adapter's identity (§5.5), so it is named
        # for the adapter rather than for its role in this test.
        adapter_dir = tmp_path / "e2e"
        adapter_dir.mkdir()
        (adapter_dir / "params.json").write_text(params_path.read_text())
        blended = backend._blend_adapter_weights(
            [{"id": "e2e", "base_model": BASE, "backend": "local", "fs_dir": "e2e", "weight": 1.0}],
            user_id="u",
        )
        assert transformer.lora_by_layer(jnp, blended, config.num_layers) is not None
        _, params = transformer.load_checkpoint(checkpoint)
        moved = _logit_delta(config, params, blended, TOKENS)
        assert float(jnp.max(jnp.abs(moved))) > 1e-6

    def test_the_target_does_not_get_a_second_bos(self, tmp_path):
        class BosTokenizer:
            def encode(self, text, truncation=True, max_length=None, add_special_tokens=True):
                ids = [(len(word) % 50) + 2 for word in text.split()]
                return ([1] + ids) if add_special_tokens else ids

        service = self._service(tmp_path, BosTokenizer())
        entries = [{"prompt": "alpha beta", "target": "gamma"}]
        (batch,) = list(service._tokenize_batches(entries, batch_size=1, max_length=64))
        # full = [BOS, alpha, beta, gamma]; inputs drop the last token.
        sequence = batch["input_ids"][0]
        assert sequence.count(1) == 1, f"duplicate BOS spliced mid-sequence: {sequence}"
