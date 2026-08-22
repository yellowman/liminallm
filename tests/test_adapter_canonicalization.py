"""Pass C: every adapter carries an explicit mode; legacy spellings are dead.

The oracle below was frozen from the *old* resolvers — `get_adapter_mode`'s
inference chain and `extract_prompt_instructions`' five-alias sweep — by
running them over every legacy shape before they were deleted, in the same
working tree. The repair in `sql/schema.sql` must give each shape the same
meaning under the new, trivial resolvers. That proves semantic equivalence,
not merely that an UPDATE added a key.

Two doors are then held shut: the validator requires `mode` and rejects every
retired spelling, and the writers emit canonical fields only.
"""

from __future__ import annotations

import json
import uuid

import pytest

from liminallm.service.model_backend import get_adapter_mode
from liminallm.service.prompt_utils import extract_prompt_instructions
from tests.harness import get_test_store

#: name -> (legacy schema, meaning under the old resolvers).
#: "meaning" is what serving actually consumed: the mode, the effective
#: prompt text, the weights directory, and the remote selection ids.
LEGACY = {
    "explicit_local": ({"mode": "local"}, ("local", None, None, None, None)),
    "explicit_remote": ({"mode": "remote"}, ("remote", None, None, None, None)),
    "explicit_prompt": ({"mode": "prompt"}, ("prompt", None, None, None, None)),
    "explicit_hybrid": ({"mode": "hybrid"}, ("hybrid", None, None, None, None)),
    "backend_prompt": ({"backend": "prompt"}, ("prompt", None, None, None, None)),
    "backend_prompt_distill": (
        {"backend": "prompt_distill"}, ("prompt", None, None, None, None)),
    "backend_local": ({"backend": "local"}, ("local", None, None, None, None)),
    "backend_local_lora": (
        {"backend": "local_lora"}, ("local", None, None, None, None)),
    "provider_local": ({"provider": "local"}, ("local", None, None, None, None)),
    "local_with_prompt": (
        {"backend": "local", "prompt_instructions": "be terse"},
        ("hybrid", "be terse", None, None, None)),
    "local_with_behavior": (
        {"backend": "local", "behavior_prompt": "be kind"},
        ("hybrid", "be kind", None, None, None)),
    "backend_api": ({"backend": "api"}, ("remote", None, None, None, None)),
    "backend_remote": ({"backend": "remote"}, ("remote", None, None, None, None)),
    "bare_remote_model_id": (
        {"remote_model_id": "ft:gpt-x"},
        ("remote", None, None, "ft:gpt-x", None)),
    "backend_hybrid": ({"backend": "hybrid"}, ("hybrid", None, None, None, None)),
    "kind_only": ({}, ("hybrid", None, None, None, None)),
    "cephfs_only": (
        {"cephfs_dir": "/mnt/adapters/a1"},
        ("hybrid", None, "/mnt/adapters/a1", None, None)),
    "dir_conflict": (
        # The old readers said `cephfs_dir or fs_dir`, so on conflict the
        # cephfs spelling won and must keep winning through the repair.
        {"fs_dir": "/new/a1", "cephfs_dir": "/old/a1"},
        ("hybrid", None, "/old/a1", None, None)),
    "alias_behavior": (
        {"behavior_prompt": "alias b"}, ("hybrid", "alias b", None, None, None)),
    "alias_system": (
        {"system_prompt": "alias s"}, ("hybrid", "alias s", None, None, None)),
    "alias_instructions": (
        {"instructions": "alias i"}, ("hybrid", "alias i", None, None, None)),
    "alias_template": (
        {"prompt_template": "alias t"}, ("hybrid", "alias t", None, None, None)),
    "alias_collision": (
        # The old extractor's order: behavior_prompt before system_prompt.
        {"behavior_prompt": "b wins", "system_prompt": "s loses"},
        ("hybrid", "b wins", None, None, None)),
    "canonical_beats_alias": (
        {"prompt_instructions": "canon", "behavior_prompt": "alias"},
        ("hybrid", "canon", None, None, None)),
    "nonstring_alias": (
        # The old extractor skipped non-strings.
        {"behavior_prompt": 5, "system_prompt": "real one"},
        ("hybrid", "real one", None, None, None)),
    "blank_canonical": (
        # And skipped blank strings, so the alias spoke.
        {"prompt_instructions": "   ", "behavior_prompt": "fallback"},
        ("hybrid", "fallback", None, None, None)),
    "model_id_alias": (
        {"mode": "remote", "model_id": "ft:alias"},
        ("remote", None, None, "ft:alias", None)),
    "adapter_id_alias": (
        {"mode": "remote", "adapter_id": "lora-alias"},
        ("remote", None, None, None, "lora-alias")),
}

RETIRED_KEYS = (
    "backend", "provider", "cephfs_dir", "behavior_prompt", "system_prompt",
    "instructions", "prompt_template", "model_id", "adapter_id",
)


def _meaning(schema: dict) -> tuple:
    """What the new resolvers say a repaired schema means."""
    return (
        get_adapter_mode(schema),
        extract_prompt_instructions(schema),
        schema.get("fs_dir"),
        schema.get("remote_model_id"),
        schema.get("remote_adapter_id"),
    )


@pytest.fixture()
def repaired(client):
    """Insert every legacy shape as an old build's rows, then run the repair.

    Raw SQL inserts, because that is what an old database is: rows that never
    met today's validator. The repair is the block in sql/schema.sql, applied
    the way migrate.sh applies it.
    """
    import psycopg

    from tests.harness import apply_schema

    store = get_test_store()
    ids = {}
    with psycopg.connect(store.dsn, autocommit=True) as conn:
        for name, (schema, _) in LEGACY.items():
            row_id = str(uuid.uuid4())
            ids[name] = row_id
            conn.execute(
                "INSERT INTO artifact (id, type, name, schema) "
                "VALUES (%s, 'adapter', %s, %s)",
                (row_id, f"legacy-{name}", json.dumps({"kind": "adapter.lora", **schema})),
            )
    apply_schema(store.dsn.replace("//", "//", 1), embedding_dim=64)
    yield ids
    with psycopg.connect(store.dsn, autocommit=True) as conn:
        conn.execute(
            "DELETE FROM artifact WHERE id = ANY(%s)", (list(ids.values()),)
        )


class TestTheRepairPreservesEveryLegacyMeaning:
    def test_every_legacy_shape_means_what_it_used_to(self, repaired):
        store = get_test_store()
        failures = []
        for name, (_, expected) in LEGACY.items():
            artifact = store.get_artifact(repaired[name])
            got = _meaning(artifact.schema)
            if got != expected:
                failures.append(f"{name}: {expected} -> {got}")
        assert not failures, "\n".join(failures)

    def test_no_retired_spelling_survives_the_repair(self, repaired):
        store = get_test_store()
        for name in LEGACY:
            schema = store.get_artifact(repaired[name]).schema
            leftovers = [key for key in RETIRED_KEYS if key in schema]
            assert not leftovers, f"{name} still carries {leftovers}"
            assert schema.get("mode") in {"local", "remote", "prompt", "hybrid"}, (
                f"{name} has no valid mode after repair: {schema.get('mode')!r}"
            )

    def test_the_repair_is_repeat_safe(self, repaired):
        from tests.harness import apply_schema

        store = get_test_store()
        before = {n: store.get_artifact(i).schema for n, i in repaired.items()}
        apply_schema(store.dsn, embedding_dim=64)
        after = {n: store.get_artifact(i).schema for n, i in repaired.items()}
        assert before == after


class TestTheDoorIsShut:
    """Old shapes must not be creatable again tomorrow."""

    def _adapter(self, **extra):
        return {
            "kind": "adapter.lora",
            "mode": "prompt",
            "base_model": "jax-base",
            "current_version": 0,
            **extra,
        }

    def test_a_new_adapter_without_mode_is_refused(self, client):
        from liminallm.service.artifact_validation import (
            ArtifactValidationError,
            validate_artifact,
        )

        schema = self._adapter()
        del schema["mode"]
        with pytest.raises(ArtifactValidationError):
            validate_artifact("adapter", schema)

    def test_a_mode_outside_the_four_is_refused(self, client):
        from liminallm.service.artifact_validation import (
            ArtifactValidationError,
            validate_artifact,
        )

        with pytest.raises(ArtifactValidationError):
            validate_artifact("adapter", self._adapter(mode="clever-new-mode"))

    @pytest.mark.parametrize("retired", RETIRED_KEYS)
    def test_every_retired_spelling_is_refused(self, client, retired):
        from liminallm.service.artifact_validation import (
            ArtifactValidationError,
            validate_artifact,
        )

        with pytest.raises(ArtifactValidationError):
            validate_artifact("adapter", self._adapter(**{retired: "x"}))

    def test_the_canonical_form_passes(self, client):
        from liminallm.service.artifact_validation import validate_artifact

        validate_artifact(
            "adapter",
            self._adapter(
                fs_dir="/adapters/a1",
                prompt_instructions="be terse",
                remote_model_id="ft:x",
                remote_adapter_id="lora-x",
            ),
        )


class TestTheWritersEmitCanonicalFieldsOnly:
    def test_a_trained_adapter_is_born_canonical(self, client):
        from liminallm.service.runtime import get_runtime

        runtime = get_runtime()
        user = runtime.store.create_user(
            email=f"canon_{uuid.uuid4().hex[:8]}@example.com"
        )
        adapter = runtime.training.ensure_user_adapter(user.id)
        schema = adapter.schema
        assert schema.get("mode") in {"local", "remote", "prompt", "hybrid"}
        leftovers = [key for key in RETIRED_KEYS if key in schema]
        assert not leftovers, f"training still writes {leftovers}"
