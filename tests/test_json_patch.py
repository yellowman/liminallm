"""One JSON Patch implementation, exercised through both of its callers.

Config patches had the hardened traversal but no move/copy/test; artifact
PATCH had the full verb set but swallowed every failure with ``pass`` — a
patch that did nothing reported success. These tests pin the union.
"""

from __future__ import annotations

import uuid

import pytest

from liminallm.service import json_patch
from liminallm.service.errors import BadRequestError

# ---------------------------------------------------------------------------
# The verbs
# ---------------------------------------------------------------------------


def test_add_and_replace_set_dict_keys():
    out = json_patch.apply_ops({}, [
        {"op": "add", "path": "/a/b", "value": 1},
        {"op": "replace", "path": "/a/b", "value": 2},
    ])
    assert out == {"a": {"b": 2}}


def test_add_inserts_into_a_list_replace_overwrites():
    out = json_patch.apply_ops({"xs": [1, 3]}, [
        {"op": "add", "path": "/xs/1", "value": 2},
    ])
    assert out["xs"] == [1, 2, 3]
    out = json_patch.apply_ops({"xs": [1, 3]}, [
        {"op": "replace", "path": "/xs/1", "value": 2},
    ])
    assert out["xs"] == [1, 2]


def test_dash_appends():
    out = json_patch.apply_ops({"xs": [1]}, [{"op": "add", "path": "/xs/-", "value": 2}])
    assert out["xs"] == [1, 2]


def test_remove_tolerates_a_missing_key():
    out = json_patch.apply_ops({"a": 1}, [{"op": "remove", "path": "/ghost"}])
    assert out == {"a": 1}


def test_move_takes_the_value_with_it():
    out = json_patch.apply_ops({"a": {"x": 1}, "b": {}}, [
        {"op": "move", "path": "/b/x", "from": "/a/x"},
    ])
    assert out == {"a": {}, "b": {"x": 1}}


def test_copy_leaves_the_source_and_severs_the_alias():
    doc = {"a": {"deep": [1]}}
    out = json_patch.apply_ops(doc, [{"op": "copy", "path": "/b", "from": "/a"}])
    out["b"]["deep"].append(2)
    assert out["a"]["deep"] == [1], "copy aliased instead of deep-copying"


def test_a_passing_test_op_is_silent_and_a_failing_one_is_a_400():
    json_patch.apply_ops({"v": 1}, [{"op": "test", "path": "/v", "value": 1}])
    with pytest.raises(BadRequestError, match="test operation failed"):
        json_patch.apply_ops({"v": 1}, [{"op": "test", "path": "/v", "value": 2}])


def test_the_original_document_is_never_mutated():
    doc = {"a": [1]}
    json_patch.apply_ops(doc, [{"op": "add", "path": "/a/-", "value": 2}])
    assert doc == {"a": [1]}


# ---------------------------------------------------------------------------
# The hardening — every silent ``pass`` in the old artifact path, now a 400
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "op, match",
    [
        ({"op": "add", "path": "/", "value": 1}, "document root"),
        ({"op": "add", "path": "/rules/x", "value": 1}, "list index"),
        ({"op": "add", "path": "/name/deep", "value": 1}, "non-container"),
        ({"op": "add", "path": "/rules/9999", "value": 1}, "too large"),
        ({"op": "add", "path": "/rules/-1/x", "value": 1}, "negative"),
        ({"op": "move", "path": "/b", "from": "/ghost"}, "source path not found"),
        ({"op": "copy", "path": "/b", "from": "/rules/99"}, "source path not found"),
        ({"op": "sing", "path": "/a", "value": 1}, "unknown patch operation"),
    ],
    ids=["root", "bad-index", "scalar-walk", "huge-index", "negative-index",
         "move-missing", "copy-missing", "unknown-verb"],
)
def test_a_malformed_op_names_its_problem(op, match):
    """The old artifact implementation did nothing and reported success for
    every one of these."""
    doc = {"rules": [1, 2], "name": "scalar"}
    with pytest.raises(BadRequestError, match=match):
        json_patch.apply_ops(doc, [op])


def test_an_op_without_action_or_path_is_ignored():
    """Half-formed entries are dropped rather than guessed at — both callers
    validated shape upstream and relied on this."""
    assert json_patch.apply_ops({"a": 1}, [{}, {"op": "add"}, {"path": "/x"}]) == {"a": 1}


# ---------------------------------------------------------------------------
# deep_merge
# ---------------------------------------------------------------------------


def test_deep_merge_merges_dicts_and_replaces_scalars():
    out = json_patch.deep_merge({"a": {"b": 1, "keep": 2}}, {"a": {"b": 9}})
    assert out == {"a": {"b": 9, "keep": 2}}
    assert json_patch.deep_merge({"a": {"b": 1}}, {"a": "flat"}) == {"a": "flat"}


def test_deep_merge_skip_keys_excludes_the_ops_envelope():
    out = json_patch.deep_merge({"a": 1}, {"ops": [1], "b": 2}, skip_keys=("ops",))
    assert out == {"a": 1, "b": 2}


def test_deep_merge_detaches_the_patch_values():
    patch = {"a": {"list": [1]}}
    out = json_patch.deep_merge({}, patch)
    out["a"]["list"].append(2)
    assert patch["a"]["list"] == [1]


# ---------------------------------------------------------------------------
# Through the artifact PATCH route
# ---------------------------------------------------------------------------


@pytest.fixture
def owned_artifact(client, auth_headers):
    resp = client.post(
        "/v1/artifacts",
        json={
            "type": "policy",
            "name": f"patch-target-{uuid.uuid4().hex[:6]}",
            "schema": {"kind": "policy.routing",
                       "rules": [{"when": "a", "action": {"tool": "x"}}]},
        },
        headers=auth_headers,
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["data"]


def test_the_route_applies_move_copy_and_test(client, auth_headers, owned_artifact):
    resp = client.patch(
        f"/v1/artifacts/{owned_artifact['id']}",
        json={"patch": [
            {"op": "test", "path": "/kind", "value": "policy.routing"},
            {"op": "copy", "path": "/rules/-", "from": "/rules/0"},
            {"op": "replace", "path": "/rules/1/when", "value": "b"},
        ]},
        headers=auth_headers,
    )
    assert resp.status_code == 200, resp.text
    rules = resp.json()["data"]["schema"]["rules"]
    assert [r["when"] for r in rules] == ["a", "b"]


def test_the_route_refuses_a_malformed_patch_instead_of_pretending(
    client, auth_headers, owned_artifact
):
    """The old route implementation would have returned 200 with nothing
    changed — the client had no way to know the patch was dropped."""
    resp = client.patch(
        f"/v1/artifacts/{owned_artifact['id']}",
        json={"patch": [{"op": "add", "path": "/rules/notanumber", "value": {}}]},
        headers=auth_headers,
    )
    assert resp.status_code == 400
    assert "list index" in resp.json()["error"]["message"]


def test_a_failed_test_op_aborts_the_whole_patch(client, auth_headers, owned_artifact):
    resp = client.patch(
        f"/v1/artifacts/{owned_artifact['id']}",
        json={"patch": [
            {"op": "test", "path": "/kind", "value": "policy.wrong"},
            {"op": "replace", "path": "/kind", "value": "policy.other"},
        ]},
        headers=auth_headers,
    )
    assert resp.status_code == 400
    check = client.get(f"/v1/artifacts/{owned_artifact['id']}", headers=auth_headers)
    assert check.json()["data"]["schema"]["kind"] == "policy.routing"
