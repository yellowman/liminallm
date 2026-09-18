"""An approved config patch is a writer too, so the role rule binds it.

`adapter_role` decides which authority a training run needs: a skill must be
authorized by a pinned job, a persona may select live preference events.
`update_artifact` refuses to change or remove it, and its comment says the
check lives in the store "because every writer - the route, config ops, the
training service - arrives through this one method".

Config ops does not. `apply_config_patch` is a second read-modify-write with
its own transaction, and it carries the schema validator and the tool-reference
check but not the role rule. So the one path whose content is model-authored is
the path the rule does not cover: propose a patch that rewrites the role, get it
approved, and the classification changes.

The last test is the other half. The rule must refuse the role and nothing
else, or config ops stops being able to edit adapters at all.
"""

from __future__ import annotations

import uuid

import pytest

from liminallm.service.runtime import get_runtime
from liminallm.storage.errors import ConstraintViolation

TENANT = "adapter-role-patch"


@pytest.fixture
def ops():
    return get_runtime().config_ops


@pytest.fixture
def store():
    return get_runtime().store


@pytest.fixture
def skill(store):
    """A skill adapter, owned by an administrable user in this tenant."""
    user = store.create_user(
        email=f"ar_{uuid.uuid4().hex[:8]}@example.com", tenant_id=TENANT
    )
    artifact = store.create_artifact(
        type_="adapter",
        name=f"skill-{uuid.uuid4().hex[:6]}",
        schema={
            "kind": "adapter.lora",
            "base_model": "jax-base",
            "mode": "prompt",
            "scope": "per-user",
            "current_version": 0,
            "rank": 4,
            "layers": [0],
            "matrices": ["attn_q"],
            "adapter_role": "skill",
        },
        description="a skill adapter under test",
        owner_user_id=user.id,
    )
    assert artifact.schema["adapter_role"] == "skill", "the fixture is not a skill"
    return artifact


def _propose(store, artifact, ops_list):
    return store.record_config_patch(
        artifact_id=artifact.id,
        proposer="system_llm",
        patch={"ops": ops_list},
        justification="test",
    )


def _role(store, artifact_id):
    return store.get_artifact(artifact_id).schema.get("adapter_role")


def test_the_ordinary_edit_path_refuses_the_flip(store, skill):
    """The control.

    Without it, a refusal below would not distinguish a rule that covers
    config ops from a fixture this test never managed to make a skill.
    """
    with pytest.raises(ConstraintViolation, match="adapter_role"):
        store.update_private_artifact(
            skill.id,
            lambda locked: {**locked, "adapter_role": "persona"},
            owner_user_id=skill.owner_user_id,
            version_author=skill.owner_user_id,
        )
    assert _role(store, skill.id) == "skill"


def test_an_approved_patch_cannot_flip_the_role(ops, store, skill):
    patch = _propose(
        store, skill,
        [{"op": "replace", "path": "/adapter_role", "value": "persona"}],
    )
    ops.decide_patch(patch.id, "approve", tenant_id=TENANT)

    with pytest.raises(ConstraintViolation, match="adapter_role"):
        ops.apply_patch(patch.id, tenant_id=TENANT)

    assert _role(store, skill.id) == "skill", (
        "an approved config patch recast a skill as a persona, which moves it "
        "onto the training path that may select live preference events"
    )


def test_an_approved_patch_cannot_remove_the_role(ops, store, skill):
    """Removing it is the same act: what is left trains as a persona."""
    patch = _propose(
        store, skill, [{"op": "remove", "path": "/adapter_role"}]
    )
    ops.decide_patch(patch.id, "approve", tenant_id=TENANT)

    with pytest.raises(ConstraintViolation, match="adapter_role"):
        ops.apply_patch(patch.id, tenant_id=TENANT)

    assert _role(store, skill.id) == "skill"


def test_a_refused_patch_writes_no_version_and_stays_unapplied(
    ops, store, skill
):
    """A refusal has to leave the audit trail honest, not half-applied."""
    before = store.get_artifact_current_version(skill.id)
    patch = _propose(
        store, skill,
        [{"op": "replace", "path": "/adapter_role", "value": "persona"}],
    )
    ops.decide_patch(patch.id, "approve", tenant_id=TENANT)

    with pytest.raises(ConstraintViolation):
        ops.apply_patch(patch.id, tenant_id=TENANT)

    assert store.get_artifact_current_version(skill.id) == before
    assert store.get_config_patch(patch.id).status == "approved", (
        "the refused patch marked itself applied"
    )


def test_an_approved_patch_may_still_edit_the_rest_of_the_adapter(
    ops, store, skill
):
    """The rule is about one member, not about adapters."""
    patch = _propose(
        store, skill, [{"op": "replace", "path": "/rank", "value": 8}]
    )
    ops.decide_patch(patch.id, "approve", tenant_id=TENANT)
    result = ops.apply_patch(patch.id, tenant_id=TENANT)

    assert result["artifact"].schema["rank"] == 8
    assert _role(store, skill.id) == "skill"


def test_a_patch_that_restates_the_same_role_is_not_a_change(
    ops, store, skill
):
    """Writing `skill` onto a skill changes nothing and must not be refused.

    A patch that rewrites the whole schema would otherwise be refused for
    carrying the role it is keeping.
    """
    patch = _propose(
        store, skill,
        [{"op": "replace", "path": "/adapter_role", "value": "skill"},
         {"op": "replace", "path": "/rank", "value": 16}],
    )
    ops.decide_patch(patch.id, "approve", tenant_id=TENANT)
    result = ops.apply_patch(patch.id, tenant_id=TENANT)

    assert result["artifact"].schema["rank"] == 16
    assert _role(store, skill.id) == "skill"
