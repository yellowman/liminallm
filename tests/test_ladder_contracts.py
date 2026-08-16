"""The contracts SPEC states and the ordinary paths had stopped honoring.

Each of these passed review at the level of "the mechanism is right" and
failed at the level of "the rule holds everywhere it is written":

* §5.2 — a malformed adapter must not partially apply. Refusing the pairs
  and dimensions still let a *foreign name* through, because assembly
  skipped what it did not recognize and applied the rest.
* §5.0.1 — a hybrid adapter is weights on a local backend, prompt on an API
  one. Injecting both gave the model an input the eval gate never scored.
* §5.4.2 — the prompt ends at the target's sequence. Searching a 200-message
  window for the target silently disabled the bound whenever the target was
  older than the window.
* §5.1 — one local representation. Serving still appended its own
  ``Context:`` marker inside the user turn.
* §5.5 — version authority over path shape. A direct ``params.json`` path
  short-circuited the promotion check entirely.
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
    directory = _build_checkpoint(tmp_path_factory.mktemp("contract_model"))
    BASE = str(directory)
    return directory


@pytest.fixture(scope="module")
def config(checkpoint):
    return transformer.load_config(checkpoint)


def _valid_pair(config, rank=2, value=0.05):
    width = config.num_heads * config.head_dim
    return {
        "layers.0.attn_q.A": [[value] * config.hidden_size] * rank,
        "layers.0.attn_q.B": [[value] * rank] * width,
    }


class TestMalformedAdaptersNeverPartiallyApply:
    def test_a_foreign_name_beside_valid_ones_refuses_the_whole_adapter(
        self, config
    ):
        """The recognized half used to be applied and the rest merely logged."""
        weights = _valid_pair(config)
        weights["foreign_arch.0.foo.A"] = [[0.1]]
        weights["foreign_arch.0.foo.B"] = [[0.1]]
        with pytest.raises(ValueError, match="outside the declared shape"):
            transformer.validate_lora_weights(config, weights)

    def test_an_unknown_target_refuses(self, config):
        weights = _valid_pair(config)
        weights["layers.0.mlp_gate.A"] = [[0.1] * config.hidden_size]
        weights["layers.0.mlp_gate.B"] = [[0.1]]
        with pytest.raises(ValueError, match="not one of"):
            transformer.validate_lora_weights(config, weights)

    def test_a_layer_outside_the_model_refuses(self, config):
        weights = _valid_pair(config)
        weights[f"layers.{config.num_layers + 5}.attn_q.A"] = [[0.1] * config.hidden_size]
        weights[f"layers.{config.num_layers + 5}.attn_q.B"] = [[0.1]]
        with pytest.raises(ValueError, match="outside the model"):
            transformer.validate_lora_weights(config, weights)

    def test_wrong_projection_width_refuses(self, config):
        weights = {
            "layers.0.attn_q.A": [[0.1] * (config.hidden_size + 4)] * 2,
            "layers.0.attn_q.B": [[0.1, 0.1]] * (config.num_heads * config.head_dim),
        }
        with pytest.raises(ValueError, match="projection is"):
            transformer.validate_lora_weights(config, weights)

    def test_a_valid_adapter_passes(self, config):
        transformer.validate_lora_weights(config, _valid_pair(config))
        # α is part of the contract, not a foreign name.
        with_scale = _valid_pair(config)
        with_scale["layers.0.attn_q.scale"] = 0.5
        transformer.validate_lora_weights(config, with_scale)

    def test_serving_refuses_at_generate_time(self, tmp_path, checkpoint, config):
        version_dir = tmp_path / "mixed" / "v0001"
        version_dir.mkdir(parents=True)
        weights = _valid_pair(config)
        weights["foreign_arch.0.foo.A"] = [[0.1]]
        weights["foreign_arch.0.foo.B"] = [[0.1]]
        (version_dir / "params.json").write_text(json.dumps(weights))
        backend = LocalJaxLoRABackend(str(checkpoint), str(tmp_path), max_new_tokens=2)
        with pytest.raises(ValueError, match="refusing the adapter"):
            backend.generate(
                [{"role": "user", "content": "hello"}],
                [{
                    "id": "mixed",
                    "base_model": BASE,
                    "backend": "local",
                    "mode": "local",
                    "fs_dir": "mixed",
                    "current_version": 1,
                }],
                user_id="u",
            )


class TestHybridIsWeightsOrPromptNotBoth:
    class _Local:
        applies_lora_weights = True

        def generate(self, messages, adapters, **kwargs):
            return {"messages": messages}

    class _Api:
        applies_lora_weights = False

        def generate(self, messages, adapters, **kwargs):
            return {"messages": messages}

    def _prompts(self, backend, adapter):
        service = LLMService(base_model="m", backend=backend)
        messages, _ = service._prepare_generation("hi", [adapter], [])
        return "\n".join(m["content"] for m in messages if m["role"] == "system")

    def test_a_promoted_hybrid_adapter_is_weights_only_on_local(self):
        """SPEC §5.0.1: the prompt is the API fallback, not a second copy of
        the behaviour the weights already carry."""
        adapter = {
            "id": "skill",
            "base_model": BASE,
            "backend": "hybrid",
            "mode": "hybrid",
            "current_version": 2,
            "prompt_instructions": "prefer tabs",
        }
        assert "prefer tabs" not in self._prompts(self._Local(), adapter)
        # The same adapter on an API backend keeps the portable fallback.
        assert "prefer tabs" in self._prompts(self._Api(), adapter)

    def test_an_unpromoted_hybrid_adapter_keeps_its_prompt_locally(self):
        """Nothing is promoted, so no weights will load: the fallback is all
        the adapter has, and dropping it would silence it entirely."""
        adapter = {
            "id": "skill",
            "base_model": BASE,
            "backend": "hybrid",
            "mode": "hybrid",
            "current_version": 0,
            "prompt_instructions": "prefer tabs",
        }
        assert "prefer tabs" in self._prompts(self._Local(), adapter)

    def test_prompt_mode_injects_on_every_backend(self):
        adapter = {
            "id": "p",
            "base_model": BASE,
            "backend": "prompt",
            "mode": "prompt",
            "prompt_instructions": "be terse",
        }
        assert "be terse" in self._prompts(self._Local(), adapter)
        assert "be terse" in self._prompts(self._Api(), adapter)


class TestOneLocalRepresentation:
    def test_context_reaches_the_local_model_as_training_wrote_it(self):
        service = LLMService(base_model="m", backend=TestHybridIsWeightsOrPromptNotBoth._Local())
        messages, _ = service._prepare_generation("what is it?", [], ["the blue cabinet"])
        rendered = local_format.format_conversation(messages)
        assert f"{local_format.CONTEXT_ROLE}: the blue cabinet" in rendered
        assert "Context:" not in rendered  # the serving-only marker is gone

    def test_api_backends_keep_the_provider_representation(self):
        service = LLMService(base_model="m", backend=TestHybridIsWeightsOrPromptNotBoth._Api())
        messages, _ = service._prepare_generation("what is it?", [], ["the blue cabinet"])
        joined = "\n".join(m["content"] for m in messages)
        assert "Context: the blue cabinet" in joined


class TestSftPromptEndsAtTheTarget:
    def test_an_old_target_still_bounds_the_prompt(self, tmp_path):
        """The bound used to be searched for inside the newest 200 messages,
        so a target older than the window disabled it silently."""
        store = get_test_store()
        user = store.create_user(email=f"seq_{uuid.uuid4().hex[:8]}@t.local")
        convo = store.create_conversation(user.id, title="long")

        store.append_message(convo.id, "user", "user", "BEFORE the target")
        target = store.append_message(convo.id, "assistant", "assistant", "the answer")
        # Comfortably more than the 200-message fetch window.
        for index in range(230):
            store.append_message(convo.id, "user", "user", f"AFTER {index}")

        event = store.record_preference_event(
            user.id,
            convo.id,
            target.id,
            "positive",
            corrected_text="",
            context_embedding=[0.1] * 64,
        )
        service = TrainingService(store=store, fs_root=str(tmp_path))
        (example,) = list(service._build_examples([event]))

        assert "BEFORE the target" in example["prompt"]
        assert "AFTER" not in example["prompt"], "future turns leaked into the prompt"
        # The target's own text is still available as the label.
        assert example["target"] == "the answer"

    def test_an_unresolvable_target_drops_the_example(self, tmp_path):
        store = get_test_store()
        user = store.create_user(email=f"drop_{uuid.uuid4().hex[:8]}@t.local")
        convo = store.create_conversation(user.id, title="c")
        message = store.append_message(convo.id, "assistant", "assistant", "a")
        event = store.record_preference_event(
            user.id, convo.id, message.id, "positive", context_embedding=[0.1] * 64
        )
        event.message_id = str(uuid.uuid4())  # a target that no longer exists
        service = TrainingService(store=store, fs_root=str(tmp_path))
        assert list(service._build_examples([event])) == []


class TestVersionAuthorityOverPathShape:
    def test_a_direct_params_file_cannot_satisfy_a_versioned_artifact(
        self, tmp_path, checkpoint, config
    ):
        """A bare params.json cannot demonstrate which version it is.

        This test previously asserted the opposite for a positive version —
        it blessed a file being served as "version 1" on nothing but the
        artifact's say-so, which is the escape hatch §5.5's version authority
        exists to close. It then allowed a never-versioned artifact to use a
        direct path, which was the same hatch with a different key: a
        versionless hybrid took weights from the file while the service,
        reading metadata alone, injected its prompt. A direct file authorizes
        nothing, at any version.
        """
        adapter_dir = tmp_path / "direct"
        adapter_dir.mkdir()
        params = adapter_dir / "params.json"
        params.write_text(json.dumps(_valid_pair(config)))
        backend = LocalJaxLoRABackend(str(checkpoint), str(tmp_path))
        adapter = {
            "id": "direct",
            "base_model": BASE,
            "backend": "local",
            "mode": "local",
            "fs_dir": str(params),
        }

        assert backend._resolve_params_path(params, current_version=0) is None
        assert backend._resolve_params_path(params, current_version=1) is None
        assert backend._resolve_params_path(params, current_version=None) is None
        # A promoted artifact pointing at a bare file refuses rather than
        # serving it: the adapter was selected and cannot be honored.
        with pytest.raises(ValueError, match="could not be loaded"):
            backend._blend_adapter_weights(
                [{**adapter, "current_version": 1, "weight": 0.5}], user_id="u"
            )
