"""The local lane's real forward pass, exercised against a real checkpoint.

The model here is small (2 layers, hidden 32, vocab 64) but it is not a
stub: real safetensors on disk, real config.json, the same loader and the
same forward pass production uses. That is the point - a hand-made stand-in
would encode what I believe the forward pass does, and the bugs worth
catching live exactly in the gap between that belief and the arithmetic.

What is pinned here, and why each one:

* cached decode == full recompute - a KV cache that drifts degrades answers
  silently, the longer the conversation the worse.
* causality - a mask off by one makes the model see its own future.
* LoRA at B=0 is the identity - that is how every adapter initializes, so a
  fresh adapter changing logits means the adapter maths is wrong.
* cold generation == warm generation - the prefix cache must be a speedup
  and nothing else. If reuse changed a single token it would be a
  correctness bug wearing a performance costume.
"""

import json

import pytest

from liminallm.service import transformer

# Every test in this module runs the real thing - training steps, a forward
# pass, an eval gate - so it is measured in seconds rather than milliseconds.
# `make test-fast` skips these; `make test` and the pre-commit gate do not,
# because what they exercise is not covered anywhere else.
pytestmark = pytest.mark.slow


# `numpy` guarded alongside `jax` rather than imported plainly above, for the
# same reason the two below are: it is in the `train` extra, and the only CI
# lane that installs it does so by naming `jax` on a `pip install` line.
# Imported at module scope it took down the browser lane - whose install set
# is the narrowest in CI - at collection, which deselects nothing because
# collection never finishes. The same defect as the undeclared `httpx`: a
# module-scope import satisfied by somebody else's requirement.
np = pytest.importorskip("numpy")
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
save_file = pytest.importorskip("safetensors.numpy").save_file

CONFIG = {
    "vocab_size": 64,
    "hidden_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    # Deliberately grouped-query (4 query heads over 2 kv heads): the head
    # repeat is where an einsum mistake hides.
    "num_key_value_heads": 2,
    "intermediate_size": 64,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "tie_word_embeddings": True,
    "max_position_embeddings": 128,
}


#: Words the fixture tokenizer knows. Anything else becomes [UNK], which is
#: what a real tokenizer does and what the training path must survive.
_TOKENIZER_WORDS = (
    "hello there friend question answer use tabs context the a and now "
    "another entirely follow up word wanted alpha beta gamma delta"
).split()


def _build_tokenizer(directory, vocab_size):
    """A real tokenizer beside the weights, as a real checkout has.

    Without one, the local lane refuses the checkpoint outright: a hashed
    token space is not the model's own, and an adapter fitted to hashed ids
    would be scored against tokens serving never emits. The fixture ships a
    genuine tokenizer so the tests exercise the text → tokenizer → batch path
    production uses, rather than hand-written token ids.
    """
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {"[UNK]": 0, "[BOS]": 1}
    for index, word in enumerate(_TOKENIZER_WORDS):
        if len(vocab) >= vocab_size:
            break
        vocab.setdefault(word, index + 2)
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(directory / "tokenizer.json"))
    (directory / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "unk_token": "[UNK]",
                "bos_token": "[BOS]",
                "model_max_length": 512,
            }
        )
    )


def _build_checkpoint(directory, seed=0):
    """A real HF-layout checkpoint: config, weights, and its own tokenizer."""
    rng = np.random.default_rng(seed)
    hidden = CONFIG["hidden_size"]
    heads = CONFIG["num_attention_heads"]
    kv_heads = CONFIG["num_key_value_heads"]
    head_dim = hidden // heads
    inter = CONFIG["intermediate_size"]

    def r(*shape):
        return rng.normal(0, 0.05, shape).astype(np.float32)

    tensors = {"model.embed_tokens.weight": r(CONFIG["vocab_size"], hidden)}
    for index in range(CONFIG["num_hidden_layers"]):
        prefix = f"model.layers.{index}"
        tensors[f"{prefix}.input_layernorm.weight"] = np.ones(hidden, np.float32)
        tensors[f"{prefix}.self_attn.q_proj.weight"] = r(heads * head_dim, hidden)
        tensors[f"{prefix}.self_attn.k_proj.weight"] = r(kv_heads * head_dim, hidden)
        tensors[f"{prefix}.self_attn.v_proj.weight"] = r(kv_heads * head_dim, hidden)
        tensors[f"{prefix}.self_attn.o_proj.weight"] = r(hidden, heads * head_dim)
        tensors[f"{prefix}.post_attention_layernorm.weight"] = np.ones(
            hidden, np.float32
        )
        tensors[f"{prefix}.mlp.gate_proj.weight"] = r(inter, hidden)
        tensors[f"{prefix}.mlp.up_proj.weight"] = r(inter, hidden)
        tensors[f"{prefix}.mlp.down_proj.weight"] = r(hidden, inter)
    tensors["model.norm.weight"] = np.ones(hidden, np.float32)

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(CONFIG))
    save_file(tensors, str(directory / "model.safetensors"))
    _build_tokenizer(directory, CONFIG["vocab_size"])
    return directory


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    return _build_checkpoint(tmp_path_factory.mktemp("tiny_model"))


@pytest.fixture(scope="module")
def loaded(checkpoint):
    return transformer.load_checkpoint(checkpoint)


class TestCheckpointLoading:
    def test_loads_a_real_checkpoint(self, checkpoint, loaded):
        assert transformer.checkpoint_available(checkpoint)
        config, params = loaded
        assert config.num_layers == 2
        assert config.num_heads == 4 and config.num_kv_heads == 2
        assert config.head_dim == 8
        assert len(params["layers"]) == 2
        # Tied embeddings: the head IS the embedding table, not a copy.
        assert params["lm_head"].shape == params["embed"].shape

    def test_absent_checkpoint_is_reported_not_raised(self, tmp_path):
        assert not transformer.checkpoint_available(tmp_path)

    def test_missing_tensor_raises_rather_than_defaulting(self, tmp_path):
        """A half-loaded model answers confidently and wrongly."""
        directory = _build_checkpoint(tmp_path / "broken")
        tensors = {
            k: v
            for k, v in __import__(
                "safetensors.numpy", fromlist=["load_file"]
            ).load_file(str(directory / "model.safetensors")).items()
            if k != "model.layers.1.mlp.down_proj.weight"
        }
        save_file(tensors, str(directory / "model.safetensors"))
        with pytest.raises(KeyError, match="down_proj"):
            transformer.load_checkpoint(directory)

    def test_unusable_config_is_refused(self, tmp_path):
        directory = tmp_path / "empty"
        directory.mkdir()
        (directory / "config.json").write_text(json.dumps({"vocab_size": 8}))
        save_file({"x": np.zeros(2, np.float32)}, str(directory / "m.safetensors"))
        with pytest.raises(ValueError, match="usable"):
            transformer.load_checkpoint(directory)


class TestForwardPass:
    def test_shapes(self, loaded):
        config, params = loaded
        tokens = jnp.array([[3, 9, 27, 5]], dtype=jnp.int32)
        logits, cache = transformer.forward(jnp, config, params, tokens)
        assert logits.shape == (1, 4, config.vocab_size)
        assert len(cache) == config.num_layers
        # KV is stored per kv-head, not per query head (that is the GQA win).
        assert cache[0][0].shape == (1, 4, config.num_kv_heads, config.head_dim)

    def test_incremental_decode_matches_full_recompute(self, loaded):
        config, params = loaded
        tokens = jnp.array([[3, 9, 27, 5, 11, 2]], dtype=jnp.int32)
        full, _ = transformer.forward(jnp, config, params, tokens)

        cache = None
        stepwise = []
        for position in range(tokens.shape[1]):
            step, cache = transformer.forward(
                jnp,
                config,
                params,
                tokens[:, position : position + 1],
                cache=cache,
            )
            stepwise.append(step[0, -1])
        assert float(jnp.max(jnp.abs(jnp.stack(stepwise) - full[0]))) < 1e-5

    def test_prefill_then_decode_block_matches(self, loaded):
        config, params = loaded
        tokens = jnp.array([[3, 9, 27, 5, 11, 2]], dtype=jnp.int32)
        full, _ = transformer.forward(jnp, config, params, tokens)
        _, cache = transformer.forward(jnp, config, params, tokens[:, :4])
        tail, _ = transformer.forward(
            jnp, config, params, tokens[:, 4:], cache=cache
        )
        assert float(jnp.max(jnp.abs(tail[0] - full[0, 4:]))) < 1e-5

    def test_attention_is_causal(self, loaded):
        config, params = loaded
        tokens = jnp.array([[3, 9, 27, 5, 11, 2]], dtype=jnp.int32)
        baseline, _ = transformer.forward(jnp, config, params, tokens)
        altered, _ = transformer.forward(
            jnp, config, params, tokens.at[0, -1].set(41)
        )
        # Changing the last token cannot reach backwards...
        assert float(jnp.max(jnp.abs(altered[0, :-1] - baseline[0, :-1]))) == 0.0
        # ...but it must change its own position.
        assert float(jnp.max(jnp.abs(altered[0, -1] - baseline[0, -1]))) > 0.0

    def test_position_matters(self, loaded):
        """RoPE applied: the same token at a different position differs."""
        config, params = loaded
        a, _ = transformer.forward(jnp, config, params, jnp.array([[7, 7]], jnp.int32))
        assert float(jnp.max(jnp.abs(a[0, 0] - a[0, 1]))) > 0.0


class TestLoRA:
    def _adapter(self, config, *, b_scale):
        rank = 4
        width = config.num_heads * config.head_dim
        rng = np.random.default_rng(3)
        return {
            "layers.0.attn_q.A": rng.normal(0, 0.1, (rank, config.hidden_size)).astype(
                np.float32
            ),
            "layers.0.attn_q.B": (
                np.zeros((width, rank), np.float32)
                if b_scale == 0
                else rng.normal(0, b_scale, (width, rank)).astype(np.float32)
            ),
        }

    def test_zero_b_is_exactly_the_identity(self, loaded):
        config, params = loaded
        tokens = jnp.array([[3, 9, 27, 5]], dtype=jnp.int32)
        baseline, _ = transformer.forward(jnp, config, params, tokens)
        lora = transformer.lora_by_layer(
            jnp, self._adapter(config, b_scale=0), config.num_layers
        )
        adapted, _ = transformer.forward(
            jnp, config, params, tokens, lora=lora
        )
        assert float(jnp.max(jnp.abs(adapted - baseline))) == 0.0

    def test_trained_weights_change_the_output(self, loaded):
        config, params = loaded
        tokens = jnp.array([[3, 9, 27, 5]], dtype=jnp.int32)
        baseline, _ = transformer.forward(jnp, config, params, tokens)
        lora = transformer.lora_by_layer(
            jnp, self._adapter(config, b_scale=0.1), config.num_layers
        )
        adapted, _ = transformer.forward(jnp, config, params, tokens, lora=lora)
        assert float(jnp.max(jnp.abs(adapted - baseline))) > 1e-6

    def test_foreign_matrix_names_do_not_half_apply(self, loaded):
        """An adapter trained for another architecture must not partially
        land - it reports nothing matched instead."""
        config, _ = loaded
        assert (
            transformer.lora_by_layer(
                jnp, {"transformer.h.0.attn.c_attn.A": np.zeros((2, 2), np.float32)},
                config.num_layers,
            )
            is None
        )
        assert transformer.lora_by_layer(jnp, {}, config.num_layers) is None


class TestLocalBackendGeneration:
    def _backend(self, checkpoint, **kwargs):
        from liminallm.service.model_backend import LocalJaxLoRABackend

        return LocalJaxLoRABackend(
            str(checkpoint), "/tmp", max_seq_len=64, max_new_tokens=6, **kwargs
        )

    def test_generates_from_the_real_model(self, checkpoint):
        backend = self._backend(checkpoint)
        result = backend.generate([{"role": "user", "content": "hello there"}], [])
        assert backend._model_state is not None, "real checkpoint was not used"
        usage = result["usage"]
        assert usage["prompt_tokens"] > 0
        assert usage["completion_tokens"] == 6
        assert usage["total_tokens"] == usage["prompt_tokens"] + 6
        # Nothing cached on the first turn - the number must be earned.
        assert "cached_tokens" not in usage

    def test_continuation_reuses_the_prefix_and_reports_it(self, checkpoint):
        backend = self._backend(checkpoint)
        first = [{"role": "user", "content": "hello there"}]
        opening = backend.generate(first, [])
        second = first + [
            {"role": "assistant", "content": opening["content"]},
            {"role": "user", "content": "another question entirely"},
        ]
        follow_up = backend.generate(second, [])
        assert follow_up["usage"]["cached_tokens"] > 0

    def test_reuse_never_changes_the_answer(self, checkpoint):
        """The correctness property: a warm cache is a speedup, nothing else."""
        messages = [
            {"role": "user", "content": "hello there"},
            {"role": "assistant", "content": "some earlier reply"},
            {"role": "user", "content": "and now a follow up"},
        ]
        warm = self._backend(checkpoint)
        warm.generate(messages[:1], [])  # seed the cache
        warmed = warm.generate(messages, [])
        assert warmed["usage"].get("cached_tokens", 0) > 0

        cold = self._backend(checkpoint)
        chilled = cold.generate(messages, [])
        assert "cached_tokens" not in chilled["usage"]
        assert warmed["content"] == chilled["content"]

    def test_a_divergent_prompt_reuses_only_what_it_shares(self, checkpoint):
        backend = self._backend(checkpoint)
        backend.generate([{"role": "user", "content": "alpha beta gamma delta"}], [])
        other = backend.generate(
            [{"role": "user", "content": "zeta eta theta iota"}], []
        )
        # Only the shared leading tokens (the role preamble) may be reused,
        # never the diverging content.
        assert other["usage"].get("cached_tokens", 0) <= 2

    def test_repeated_prompt_leaves_a_token_to_run(self, checkpoint):
        """A fully cached prompt still needs one token forward to produce
        logits - otherwise there is nothing to sample from."""
        backend = self._backend(checkpoint)
        messages = [{"role": "user", "content": "identical prompt"}]
        backend.generate(messages, [])
        again = backend.generate(messages, [])
        assert again["usage"]["cached_tokens"] == again["usage"]["prompt_tokens"] - 1
        assert again["usage"]["completion_tokens"] == 6

    def test_cache_is_bounded(self, checkpoint):
        backend = self._backend(checkpoint, max_cached_tokens=24)
        for index in range(6):
            backend.generate(
                [{"role": "user", "content": f"unrelated prompt number {index}"}], []
            )
        cached = sum(len(entry[1]) for entry in backend._prefix_cache)
        assert cached <= 24 or len(backend._prefix_cache) == 1

    def test_no_checkpoint_falls_back_without_crashing(self, tmp_path):
        backend = self._backend(tmp_path)
        result = backend.generate([{"role": "user", "content": "hello"}], [])
        assert backend._model_state is None
        assert backend._model_error
        assert "content" in result and "usage" in result
