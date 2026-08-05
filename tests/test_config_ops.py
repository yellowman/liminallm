"""ConfigOps: the model proposing edits to its own configuration (SPEC §10).

The patch body is model-authored, so malformed shapes are the normal case; and
approval is the only thing between a proposal and a live config change.
"""

from __future__ import annotations

import uuid

import pytest

from liminallm.service.errors import BadRequestError, NotFoundError
from liminallm.service.runtime import get_runtime


@pytest.fixture
def ops():
    return get_runtime().config_ops


@pytest.fixture
def store():
    return get_runtime().store


@pytest.fixture
def artifact(store):
    user = store.create_user(email=f"co_{uuid.uuid4().hex[:8]}@example.com")
    return store.create_artifact(
        type_="artifact",
        name=f"cfg-{uuid.uuid4().hex[:6]}",
        schema={"kind": "routing.policy", "rules": [], "meta": {"weight": 1}},
        description="a policy under test",
        owner_user_id=user.id,
    )


def _propose(store, artifact, patch):
    return store.record_config_patch(
        artifact_id=artifact.id,
        proposer="system_llm",
        patch=patch,
        justification="test",
    )


# ---------------------------------------------------------------------------
# Approval is a gate, not bookkeeping
# ---------------------------------------------------------------------------


def test_a_pending_patch_cannot_be_applied(ops, store, artifact):
    patch = _propose(store, artifact, {"ops": []})
    with pytest.raises(BadRequestError, match="approved"):
        ops.apply_patch(patch.id)


def test_a_rejected_patch_cannot_be_applied(ops, store, artifact):
    patch = _propose(store, artifact, {"ops": []})
    ops.decide_patch(patch.id, "reject")
    with pytest.raises(BadRequestError, match="approved"):
        ops.apply_patch(patch.id)


def test_an_approved_patch_rewrites_the_schema(ops, store, artifact):
    patch = _propose(
        store, artifact, {"ops": [{"op": "replace", "path": "/meta/weight", "value": 7}]}
    )
    ops.decide_patch(patch.id, "approve")
    result = ops.apply_patch(patch.id)

    assert result["artifact"].schema["meta"]["weight"] == 7
    assert store.get_artifact(artifact.id).schema["meta"]["weight"] == 7


def test_applying_records_a_new_artifact_version(ops, store, artifact):
    before = store.get_artifact_current_version(artifact.id)
    patch = _propose(
        store, artifact, {"ops": [{"op": "add", "path": "/meta/note", "value": "x"}]}
    )
    ops.decide_patch(patch.id, "approve")
    ops.apply_patch(patch.id)
    assert store.get_artifact_current_version(artifact.id) > before


def test_a_patch_is_decided_once(ops, store, artifact):
    patch = _propose(store, artifact, {"ops": []})
    ops.decide_patch(patch.id, "approve")
    with pytest.raises(BadRequestError, match="pending"):
        ops.decide_patch(patch.id, "reject")


@pytest.mark.parametrize(
    "word, expected",
    [("approve", "approved"), ("approved", "approved"), ("APPROVE", "approved"),
     ("reject", "rejected"), ("rejected", "rejected"), ("Reject", "rejected")],
)
def test_a_decision_word_normalizes(ops, store, artifact, word, expected):
    patch = _propose(store, artifact, {"ops": []})
    assert ops.decide_patch(patch.id, word).status == expected


@pytest.mark.parametrize("word", ["maybe", "", "approve-ish", "yes"])
def test_an_unrecognised_decision_is_refused(ops, store, artifact, word):
    patch = _propose(store, artifact, {"ops": []})
    with pytest.raises(BadRequestError, match="decision"):
        ops.decide_patch(patch.id, word)


def test_deciding_an_unknown_patch_is_a_not_found(ops):
    with pytest.raises(NotFoundError):
        ops.decide_patch(999_999_999, "approve")


def test_applying_an_unknown_patch_is_a_not_found(ops):
    with pytest.raises(NotFoundError):
        ops.apply_patch(999_999_999)


def test_a_reason_is_kept_with_the_decision(ops, store, artifact):
    patch = _propose(store, artifact, {"ops": []})
    decided = ops.decide_patch(patch.id, "reject", reason="changes the wrong knob")
    assert (decided.meta or {}).get("reason") == "changes the wrong knob"


def test_auto_generate_needs_a_real_artifact(ops):
    with pytest.raises(NotFoundError):
        ops.auto_generate_patch(str(uuid.uuid4()), user_id=None)


# ---------------------------------------------------------------------------
# Applying ops to a schema
# ---------------------------------------------------------------------------


def test_add_replace_and_remove(ops):
    schema = {"a": 1, "keep": True, "nested": {"b": 2}}
    out = ops._apply_patch_to_schema(
        schema,
        {
            "ops": [
                {"op": "add", "path": "/added", "value": "new"},
                {"op": "replace", "path": "/a", "value": 9},
                {"op": "remove", "path": "/nested/b"},
            ]
        },
    )
    assert out == {"a": 9, "keep": True, "nested": {}, "added": "new"}


def test_the_original_schema_is_not_mutated(ops):
    """apply_patch computes the new schema before touching the database, so a
    caller holding the artifact must not see it change underneath them."""
    schema = {"meta": {"weight": 1}}
    ops._apply_patch_to_schema(
        schema, {"ops": [{"op": "replace", "path": "/meta/weight", "value": 99}]}
    )
    assert schema == {"meta": {"weight": 1}}


def test_a_missing_intermediate_is_created(ops):
    out = ops._apply_patch_to_schema(
        {}, {"ops": [{"op": "add", "path": "/a/b/c", "value": 1}]}
    )
    assert out == {"a": {"b": {"c": 1}}}


def test_appending_to_a_list(ops):
    out = ops._apply_patch_to_schema(
        {"rules": [1, 2]}, {"ops": [{"op": "add", "path": "/rules/-", "value": 3}]}
    )
    assert out["rules"] == [1, 2, 3]


def test_replacing_and_removing_a_list_entry(ops):
    out = ops._apply_patch_to_schema(
        {"rules": ["a", "b", "c"]},
        {"ops": [{"op": "replace", "path": "/rules/1", "value": "B"},
                 {"op": "remove", "path": "/rules/2"}]},
    )
    assert out["rules"] == ["a", "B"]


def test_a_patch_without_ops_deep_merges(ops):
    out = ops._apply_patch_to_schema(
        {"meta": {"weight": 1, "keep": True}, "other": 1},
        {"meta": {"weight": 5}, "new": "x"},
    )
    assert out == {"meta": {"weight": 5, "keep": True}, "other": 1, "new": "x"}


def test_a_deep_merge_replaces_rather_than_merges_a_non_dict(ops):
    out = ops._apply_patch_to_schema({"a": {"b": 1}}, {"a": "scalar"})
    assert out == {"a": "scalar"}


def test_an_empty_patch_changes_nothing(ops):
    schema = {"a": 1}
    assert ops._apply_patch_to_schema(schema, {}) == schema
    assert ops._apply_patch_to_schema(schema, None) == schema


@pytest.mark.parametrize(
    "op", [{}, {"op": "add"}, {"path": "/a"}, {"op": "", "path": ""}]
)
def test_an_op_missing_its_parts_is_skipped(ops, op):
    assert ops._apply_patch_to_schema({"a": 1}, {"ops": [op]}) == {"a": 1}


# ---------------------------------------------------------------------------
# Malformed patches: the model writes these, so they are the normal case
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "schema, path",
    [
        ({"a": "hello"}, "/a/b"),        # traverse into a string
        ({"a": {"b": 5}}, "/a/b/c"),     # traverse into an int
        ({"a": None}, "/a/b"),           # traverse into null
        ({"a": True}, "/a/b/c"),         # traverse into a bool
    ],
)
def test_a_path_through_a_scalar_is_a_bad_request(ops, schema, path):
    """It used to raise a bare TypeError, which the API turned into a 500.

    The patch body is model-authored, so this shape arrives routinely — and an
    admin reviewing a proposal deserves to be told which path is wrong rather
    than getting an opaque internal error.
    """
    with pytest.raises(BadRequestError):
        ops._apply_patch_to_schema(schema, {"ops": [{"op": "add", "path": path, "value": 1}]})


def test_a_root_path_is_a_bad_request(ops):
    """`/` addresses the whole document; treating it as a key wrote an
    empty-string entry into the schema and said nothing."""
    with pytest.raises(BadRequestError):
        ops._apply_patch_to_schema({"a": 1}, {"ops": [{"op": "add", "path": "/", "value": 9}]})


def test_a_negative_list_index_is_refused(ops):
    with pytest.raises(BadRequestError, match="negative"):
        ops._apply_patch_to_schema(
            {"rules": [1]}, {"ops": [{"op": "add", "path": "/rules/-5", "value": 1}]}
        )


def test_a_huge_list_index_is_refused(ops):
    """Padding a list out to the requested index is how a one-line patch
    becomes a memory exhaustion."""
    with pytest.raises(BadRequestError, match="too large"):
        ops._apply_patch_to_schema(
            {"rules": []},
            {"ops": [{"op": "add", "path": "/rules/999999999", "value": 1}]},
        )


def test_a_failing_op_abandons_the_whole_patch(ops):
    """Ops are applied to a copy, so a patch that fails halfway must not leave
    the artifact holding the first half."""
    schema = {"rules": [], "meta": {"weight": 1}}
    with pytest.raises(BadRequestError):
        ops._apply_patch_to_schema(
            schema,
            {
                "ops": [
                    {"op": "replace", "path": "/meta/weight", "value": 99},
                    {"op": "add", "path": "/rules/999999999", "value": 1},
                ]
            },
        )
    assert schema["meta"]["weight"] == 1


def test_a_non_numeric_list_index_is_refused_not_skipped(ops):
    """This used to be silently dropped, which meant a patch op that did
    nothing reported success — the admin approved a change that never
    happened. The reviewer is told which path is wrong instead."""
    with pytest.raises(BadRequestError, match="list index"):
        ops._apply_patch_to_schema(
            {"rules": [1]}, {"ops": [{"op": "add", "path": "/rules/x", "value": 2}]}
        )


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def test_a_small_schema_is_sent_whole(ops):
    assert ops._safe_truncate_json({"a": 1}, 2000) == '{"a":1}'


def test_a_large_schema_is_truncated_to_valid_json(ops):
    import json

    big = {f"key_{i}": "x" * 200 for i in range(50)}
    out = ops._safe_truncate_json(big, 2000)
    assert len(out) <= 2000
    parsed = json.loads(out)  # must still parse — a prompt gets this verbatim
    assert parsed["_truncated"] is True


def test_the_prompt_names_the_artifact_and_the_goal(ops, store, artifact):
    prompt = ops._build_prompt(store.get_artifact(artifact.id), "make routing sharper")
    assert artifact.name in prompt
    assert "make routing sharper" in prompt


def test_the_prompt_has_a_goal_even_when_none_is_given(ops, store, artifact):
    prompt = ops._build_prompt(store.get_artifact(artifact.id), None)
    assert "Goal:" in prompt


# ---------------------------------------------------------------------------
# The artifact type routing-as-data needs
# ---------------------------------------------------------------------------


def test_a_routing_policy_can_be_created(store):
    """SPEC §6.1 / §8.1: routing lives in `policy` artifacts, not in code.

    The workflow engine already queried for them and §13.4 documents the type
    on the list endpoint, but validate_artifact had no schema for it — so
    POST /v1/artifacts {type: "policy"} answered "unknown artifact type" and
    the data half of routing-as-data could not be written at all.
    """
    artifact = store.create_artifact(
        type_="policy",
        # The engine looks for this exact name (workflow.py), globally visible
        # because a routing policy applies to everyone until a user overrides it.
        name="default_routing",
        description="chooses adapters by cluster",
        visibility="global",
        schema={
            "kind": "policy.routing",
            "name": "default_user_routing",
            "rules": [
                {
                    "id": "always_persona",
                    "when": "true",
                    "action": {
                        "type": "activate_adapter_by_type",
                        "adapter_type": "persona",
                        "weight": 0.5,
                    },
                }
            ],
        },
    )
    assert artifact.type == "policy"

    # The engine's own query, verbatim: this is what it means for the policy to
    # be reachable rather than merely stored.
    found = [
        a for a in store.list_artifacts(type_filter="policy")
        if a.name == "default_routing"
    ]
    assert found, "the workflow engine's policy lookup cannot see it"
    assert found[0].schema["rules"][0]["id"] == "always_persona"


@pytest.mark.parametrize(
    "schema, why",
    [
        ({"kind": "policy.routing"}, "rules"),
        ({"kind": "workflow.graph", "rules": []}, "policy.routing"),
        ({"kind": "policy.routing", "rules": [{"id": "x"}]}, "when"),
    ],
)
def test_a_malformed_routing_policy_is_refused(schema, why):
    from liminallm.service.artifact_validation import (
        ArtifactValidationError,
        validate_artifact,
    )

    with pytest.raises(ArtifactValidationError) as caught:
        validate_artifact("policy", schema)
    assert any(why in err for err in caught.value.errors)
