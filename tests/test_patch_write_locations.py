"""An RFC 6902 operation applies where it is permitted, or changes nothing.

Two halves of one rule:

* traversal never manufactures missing intermediate structure, and
* an operation that requires an existing target never turns absence into
  success.

The engine used to do the opposite of both. `add` and `replace` shared a
creating walk, so `replace /a/b` on a document with no `a` invented one and
reported success; `remove` treated a missing target as nothing to do. The
measurable consequence was on published configuration: a ConfigOps patch
whose path did not exist was applied, marked `applied` and given a new
`artifact_version`, while the configuration serving actually consumed stayed
as it was. The operator was told the change landed. It had not.

The positive controls matter as much as the refusals. "Reject every absent
location" is a fix that breaks `add`, whose whole job is naming a member that
is not there yet.
"""

from __future__ import annotations

import json
import uuid

import pytest

from liminallm.service import json_patch
from liminallm.service.errors import BadRequestError
from liminallm.service.runtime import get_runtime

WORKFLOW = {
    "kind": "workflow.chat",
    "entrypoint": "only",
    "spare": "keep",
    "nodes": [
        {"id": "only", "type": "tool_call", "tool": "llm.generic", "next": "end"},
        {"id": "end", "type": "end"},
    ],
}


def apply(doc, *ops):
    return json_patch.apply_ops(doc, list(ops))


# ---------------------------------------------------------------------------
# replace
# ---------------------------------------------------------------------------


class TestReplaceRequiresAnExistingTarget:
    """RFC 6902 §4.3: the target location must exist for the operation to be
    successful."""

    def test_a_missing_member_is_refused(self):
        with pytest.raises(BadRequestError):
            apply({"a": 1}, {"op": "replace", "path": "/ghost", "value": 2})

    def test_a_missing_parent_is_refused_and_not_invented(self):
        """The measured defect, at its smallest.

        `/schema/foo` was the documented spelling while the engine's document
        *is* the schema, so this is the shape every caller following the docs
        produced: a new nested object, the intended value untouched, and a
        success.
        """
        with pytest.raises(BadRequestError):
            apply({"spare": "keep"},
                  {"op": "replace", "path": "/schema/spare", "value": "CHANGED"})

    def test_an_out_of_range_index_is_refused(self):
        with pytest.raises(BadRequestError):
            apply({"xs": [1, 2]}, {"op": "replace", "path": "/xs/9", "value": 0})

    def test_replacing_something_that_is_there_still_works(self):
        assert apply({"a": 1}, {"op": "replace", "path": "/a", "value": 2}) == {"a": 2}
        assert apply({"xs": [1, 2]},
                     {"op": "replace", "path": "/xs/1", "value": 9})["xs"] == [1, 9]


# ---------------------------------------------------------------------------
# remove
# ---------------------------------------------------------------------------


class TestRemoveRequiresAnExistingTarget:
    def test_a_missing_member_is_refused(self):
        with pytest.raises(BadRequestError):
            apply({"a": 1}, {"op": "remove", "path": "/ghost"})

    def test_a_missing_parent_is_refused(self):
        with pytest.raises(BadRequestError):
            apply({}, {"op": "remove", "path": "/a/b"})

    def test_an_out_of_range_index_is_refused(self):
        with pytest.raises(BadRequestError):
            apply({"xs": []}, {"op": "remove", "path": "/xs/3"})

    def test_removing_something_that_is_there_still_works(self):
        assert apply({"a": 1, "b": 2}, {"op": "remove", "path": "/b"}) == {"a": 1}
        assert apply({"xs": [1, 2]},
                     {"op": "remove", "path": "/xs/0"})["xs"] == [2]

    def test_a_refused_removal_leaves_the_document_alone(self):
        """The other half of the rule: refusing is not enough if the walk
        already wrote the containers it was refusing to remove from.

        Driven through `apply_op`, which edits in place. `apply_ops` copies
        first, so asserting on the caller's document there proves only that
        `copy.deepcopy` works — the risk lives on the mutating entry point.
        """
        doc = {"keep": 1}
        with pytest.raises(BadRequestError):
            json_patch.apply_op(doc, {"op": "remove", "path": "/a/b/c"})
        assert doc == {"keep": 1}


# ---------------------------------------------------------------------------
# add
# ---------------------------------------------------------------------------


class TestAddMayNameANewMemberButNotANewParent:
    def test_a_new_member_of_an_existing_object_is_the_point(self):
        assert apply({"a": 1}, {"op": "add", "path": "/newkey", "value": 2}) == {
            "a": 1, "newkey": 2,
        }

    def test_a_new_member_of_an_existing_nested_object_works(self):
        assert apply({"a": {"b": 1}},
                     {"op": "add", "path": "/a/c", "value": 2}) == {
            "a": {"b": 1, "c": 2},
        }

    def test_a_missing_parent_is_refused(self):
        with pytest.raises(BadRequestError):
            apply({}, {"op": "add", "path": "/a/b", "value": 1})

    def test_a_deeply_missing_parent_is_refused(self):
        with pytest.raises(BadRequestError):
            apply({"a": {}}, {"op": "add", "path": "/a/b/c", "value": 1})


class TestArrayBoundsFollowTheSpec:
    def test_appending_at_the_end_is_allowed(self):
        assert apply({"xs": [1, 2]},
                     {"op": "add", "path": "/xs/2", "value": 3})["xs"] == [1, 2, 3]

    def test_the_dash_target_appends(self):
        assert apply({"xs": [1]},
                     {"op": "add", "path": "/xs/-", "value": 2})["xs"] == [1, 2]

    def test_inserting_inside_the_array_still_shifts(self):
        assert apply({"xs": [1, 3]},
                     {"op": "add", "path": "/xs/1", "value": 2})["xs"] == [1, 2, 3]

    def test_an_index_past_the_end_is_refused(self):
        with pytest.raises(BadRequestError):
            apply({"xs": [1, 2]}, {"op": "add", "path": "/xs/5", "value": 3})

    def test_replace_can_address_an_existing_large_array_index(self):
        """The bound is the list's own length, not a constant.

        A fixed ceiling made position 1024 addressable by `remove`, `test` and
        both source reads while `replace` and `add` refused it — the same
        location existing for one verb and not another, which is the
        inconsistency this file exists to remove.
        """
        doc = {"xs": list(range(1025))}
        assert apply(doc, {"op": "replace", "path": "/xs/1024",
                           "value": "changed"})["xs"][1024] == "changed"

    def test_add_can_append_to_a_large_existing_array(self):
        doc = {"xs": list(range(1025))}
        assert apply(doc, {"op": "add", "path": "/xs/1025",
                           "value": "tail"})["xs"][-1] == "tail"

    def test_a_huge_gap_is_still_refused_without_allocating(self):
        """What the ceiling was really for, now carried by the length check.

        `/xs/999999999` on a two-element list would allocate a billion
        placeholders. It is refused because 999999999 is past the end, which
        is the same reason `/xs/5` is — one rule instead of two.
        """
        with pytest.raises(BadRequestError):
            apply({"xs": [1, 2]}, {"op": "add", "path": "/xs/999999999", "value": 3})


class TestMoveAndCopyDestinationsFollowAdd:
    def test_a_copy_cannot_conjure_its_destination_parent(self):
        with pytest.raises(BadRequestError):
            apply({"a": {"x": 1}},
                  {"op": "copy", "from": "/a/x", "path": "/b/x"})

    def test_a_move_cannot_conjure_its_destination_parent(self):
        with pytest.raises(BadRequestError):
            apply({"a": {"x": 1}},
                  {"op": "move", "from": "/a/x", "path": "/b/x"})

    def test_a_refused_move_does_not_lose_the_value(self):
        """A move is a remove and an add. If the destination is refused after
        the source has been taken, the value has been deleted by an operation
        that failed."""
        doc = {"a": {"x": 1}}
        with pytest.raises(BadRequestError):
            json_patch.apply_op(doc, {"op": "move", "from": "/a/x", "path": "/b/x"})
        assert doc == {"a": {"x": 1}}, "the source was taken and never delivered"

    def test_move_into_its_own_child_refuses_without_taking_the_source(self):
        """RFC 6902 §4.4: `from` must not be a proper prefix of `path`.

        Checking the destination before the removal is not enough, because
        this destination is only invalid *because of* the removal. `/a` is a
        perfectly good parent until `/a` is the thing being taken, and then
        there is nowhere to put it back.
        """
        doc = {"a": {"x": 1}}
        with pytest.raises(BadRequestError):
            json_patch.apply_op(doc, {"op": "move", "from": "/a", "path": "/a/child"})
        assert doc == {"a": {"x": 1}}, "the source was taken and never delivered"

    def test_move_to_an_index_the_removal_invalidates_refuses(self):
        """The same shape without any prefix relationship.

        `/xs/3` is a legal append target on a three-element list and an
        out-of-range one on the two-element list the removal leaves behind.
        RFC 6902 defines `move` as a remove followed by an add, so the
        destination has to be valid in the document the remove produces.
        """
        doc = {"xs": ["a", "b", "c"]}
        with pytest.raises(BadRequestError):
            json_patch.apply_op(doc, {"op": "move", "from": "/xs/0", "path": "/xs/3"})
        assert doc == {"xs": ["a", "b", "c"]}, "the list was shortened by a failure"

    def test_a_move_within_one_list_still_works(self):
        """The positive control the two above could otherwise break.

        Moving inside a list is exactly the case where the destination must
        be judged after the removal rather than before it — index 2 is the
        end of the shortened list, not of the original.
        """
        assert json_patch.apply_ops(
            {"xs": ["a", "b", "c"]},
            [{"op": "move", "from": "/xs/0", "path": "/xs/2"}],
        )["xs"] == ["b", "c", "a"]

    def test_a_destination_whose_parent_exists_still_works(self):
        assert apply({"a": {"x": 1}, "b": {}},
                     {"op": "move", "from": "/a/x", "path": "/b/x"}) == {
            "a": {}, "b": {"x": 1},
        }


class TestAPatchIsAllOrNothing:
    def test_a_later_op_on_a_missing_target_yields_no_document(self):
        """`apply_ops` either returns a fully patched document or raises.

        It works on a copy, so the caller's own document is safe by
        construction; what this pins is that no half-applied result is handed
        back, which is what a caller would otherwise persist.
        """
        doc = {"a": 1, "b": 2}
        with pytest.raises(BadRequestError):
            apply(doc,
                  {"op": "replace", "path": "/a", "value": 99},
                  {"op": "replace", "path": "/ghost", "value": 0})
        assert doc == {"a": 1, "b": 2}

    def test_the_mutating_entry_point_stops_at_the_failing_op(self):
        """`apply_op` edits in place, so a caller looping over ops itself
        keeps whatever landed before the failure. The engine's own callers
        copy first; this states the boundary rather than leaving it to be
        rediscovered."""
        doc = {"a": 1}
        json_patch.apply_op(doc, {"op": "replace", "path": "/a", "value": 99})
        with pytest.raises(BadRequestError):
            json_patch.apply_op(doc, {"op": "replace", "path": "/ghost", "value": 0})
        assert doc == {"a": 99}


# ---------------------------------------------------------------------------
# both callers inherit it
# ---------------------------------------------------------------------------


def _published(client, admin_headers):
    made = client.post("/v1/artifacts", headers=admin_headers, json={
        "type": "workflow", "name": f"wl-{uuid.uuid4().hex[:6]}",
        "schema": json.loads(json.dumps(WORKFLOW)), "visibility": "global",
    })
    assert made.status_code in (200, 201), made.text
    return made.json()["data"]["id"]


def _schema(artifact_id):
    return get_runtime().store.get_artifact(artifact_id).schema


def _versions(artifact_id):
    return len(get_runtime().store.list_artifact_versions(artifact_id))


class TestConfigOpsInheritsIt:
    def test_a_patch_to_a_location_that_does_not_exist_does_not_apply(
        self, client, admin_headers
    ):
        """The consequence that made this worth finding.

        Before: HTTP 200, status `applied`, a new artifact_version, and the
        configuration serving consumes unchanged — an audit trail asserting a
        change that did not happen.
        """
        artifact = _published(client, admin_headers)
        before, before_versions = _schema(artifact), _versions(artifact)

        proposed = client.post("/v1/config/propose_patch", headers=admin_headers, json={
            "artifact_id": artifact,
            "patch": [{"op": "replace", "path": "/schema/spare", "value": "CHANGED"}],
            "justification": "a path that does not exist",
        })
        assert proposed.status_code in (200, 201), proposed.text
        patch_id = proposed.json()["data"]["id"]
        client.post(f"/v1/config/patches/{patch_id}/decide",
                    headers=admin_headers, json={"decision": "approve"})

        applied = client.post(f"/v1/config/patches/{patch_id}/apply",
                              headers=admin_headers, json={})

        assert applied.status_code == 400, applied.text
        assert _schema(artifact) == before, "the artifact changed anyway"
        assert _versions(artifact) == before_versions, "a version was written"
        status = get_runtime().store.get_config_patch(patch_id).status
        assert status == "approved", f"the patch was marked {status}"

    def test_a_patch_to_a_location_that_exists_still_applies(
        self, client, admin_headers
    ):
        artifact = _published(client, admin_headers)
        before_versions = _versions(artifact)

        proposed = client.post("/v1/config/propose_patch", headers=admin_headers, json={
            "artifact_id": artifact,
            "patch": [{"op": "replace", "path": "/spare", "value": "CHANGED"}],
            "justification": "a path that exists",
        })
        patch_id = proposed.json()["data"]["id"]
        client.post(f"/v1/config/patches/{patch_id}/decide",
                    headers=admin_headers, json={"decision": "approve"})

        applied = client.post(f"/v1/config/patches/{patch_id}/apply",
                              headers=admin_headers, json={})

        assert applied.status_code == 200, applied.text
        assert _schema(artifact)["spare"] == "CHANGED"
        assert _versions(artifact) == before_versions + 1


class TestArtifactPatchInheritsIt:
    def test_a_private_artifact_refuses_a_location_that_does_not_exist(
        self, client, admin_headers
    ):
        made = client.post("/v1/artifacts", headers=admin_headers, json={
            "type": "workflow", "name": f"pv-{uuid.uuid4().hex[:6]}",
            "schema": json.loads(json.dumps(WORKFLOW)), "visibility": "private",
        })
        artifact = made.json()["data"]["id"]
        before, before_versions = _schema(artifact), _versions(artifact)

        resp = client.patch(f"/v1/artifacts/{artifact}", headers=admin_headers, json={
            "patch": [{"op": "replace", "path": "/schema/spare", "value": "CHANGED"}],
        })

        assert resp.status_code == 400, resp.text
        assert _schema(artifact) == before
        assert _versions(artifact) == before_versions
