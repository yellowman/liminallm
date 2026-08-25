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

    def test_a_negative_write_index_is_not_counted_from_the_end(self):
        """`list.insert(-1, v)` writes before the last element, silently.

        So a negative final segment does not fail loudly on the write path —
        it lands somewhere the caller did not name: `add /xs/-1` on [1, 2]
        produces [1, 9, 2], and `/xs/-2` produces [9, 1, 2]. `replace` is
        already covered, because requiring an existing target reads the index
        first. `add` has no target to require, so it is the one verb where the
        index itself has to be refused.

        Found by a mutation that survived, not by review.
        """
        doc = {"xs": [1, 2]}
        with pytest.raises(BadRequestError, match="negative"):
            json_patch.apply_op(doc, {"op": "add", "path": "/xs/-1", "value": 9})
        assert doc == {"xs": [1, 2]}, "a negative index wrote from the end"

    def test_a_huge_gap_is_still_refused_without_allocating(self):
        """What the ceiling was really for, now carried by the length check.

        `/xs/999999999` on a two-element list would allocate a billion
        placeholders. It is refused because 999999999 is past the end, which
        is the same reason `/xs/5` is — one rule instead of two.
        """
        with pytest.raises(BadRequestError):
            apply({"xs": [1, 2]}, {"op": "add", "path": "/xs/999999999", "value": 3})


class TestAPointerNamesTheKeyItSpells:
    """RFC 6901. The same invariant as the rest of this file, one step
    earlier: before an operation can land where it was aimed, the pointer has
    to survive being read.

    Tokenizing with `strip("/")`, `split("/")` and "drop the empty segments"
    rewrites the caller's path four separate ways, and every one of them is a
    silent change of address rather than a refusal.
    """

    def test_an_escaped_slash_addresses_the_key_it_spells(self):
        """`~1` is a `/`, so `/a~1b` names one key called `a/b`.

        Undecoded it names a *different, also-valid* key spelled `a~1b` — so
        a document holding both gets the wrong one written and the right one
        left alone, with nothing raised.
        """
        doc = {"a/b": "RIGHT", "a~1b": "WRONG"}
        json_patch.apply_op(doc, {"op": "replace", "path": "/a~1b", "value": "X"})
        assert doc == {"a/b": "X", "a~1b": "WRONG"}

    def test_an_escaped_tilde_addresses_the_key_it_spells(self):
        doc = {"a~b": "RIGHT", "a~0b": "WRONG"}
        json_patch.apply_op(doc, {"op": "replace", "path": "/a~0b", "value": "X"})
        assert doc == {"a~b": "X", "a~0b": "WRONG"}

    def test_the_escapes_decode_in_the_order_the_rfc_gives(self):
        """§4: `~1` first, then `~0`. `~01` is the two-character key `~1`.

        Decoding `~0` first would turn it into `~1` and then into `/`, which
        is a third key again.
        """
        doc = {"~1": "RIGHT", "/": "WRONG", "~01": "ALSO WRONG"}
        json_patch.apply_op(doc, {"op": "replace", "path": "/~01", "value": "X"})
        assert doc == {"~1": "X", "/": "WRONG", "~01": "ALSO WRONG"}

    def test_an_empty_reference_token_is_a_real_key(self):
        """`/a//b` is three tokens — `a`, ``, `b` — not two.

        Dropping the empty one addressed `a.b`, a sibling of the key named.
        """
        doc = {"a": {"": {"b": "RIGHT"}, "b": "WRONG"}}
        json_patch.apply_op(doc, {"op": "replace", "path": "/a//b", "value": "X"})
        assert doc == {"a": {"": {"b": "X"}, "b": "WRONG"}}

    def test_a_trailing_slash_names_the_empty_key(self):
        """`/a/` names `a`'s empty-string member, not `a` itself.

        Stripping it replaced the whole object with the value.
        """
        doc = {"a": {"": "RIGHT", "keep": "WRONG"}}
        json_patch.apply_op(doc, {"op": "replace", "path": "/a/", "value": "X"})
        assert doc == {"a": {"": "X", "keep": "WRONG"}}

    def test_a_lone_slash_names_the_empty_key_at_the_top(self):
        doc = {"": "RIGHT", "keep": "WRONG"}
        json_patch.apply_op(doc, {"op": "replace", "path": "/", "value": "X"})
        assert doc == {"": "X", "keep": "WRONG"}

    def test_a_pointer_must_begin_with_a_slash(self):
        """`a/b` is not a pointer. Accepting it as `/a/b` guesses."""
        doc = {"a": {"b": "RIGHT"}}
        with pytest.raises(BadRequestError):
            json_patch.apply_op(doc, {"op": "replace", "path": "a/b", "value": "X"})
        assert doc == {"a": {"b": "RIGHT"}}

    def test_an_escape_that_escapes_nothing_is_refused(self):
        """`~` must be followed by `0` or `1`. Passing `~2` through would make
        it a literal, which is a fifth silent reinterpretation."""
        doc = {"a~2b": "RIGHT"}
        with pytest.raises(BadRequestError):
            json_patch.apply_op(doc, {"op": "replace", "path": "/a~2b", "value": "X"})
        assert doc == {"a~2b": "RIGHT"}

    def test_a_trailing_tilde_is_refused(self):
        doc = {"a~": "RIGHT"}
        with pytest.raises(BadRequestError):
            json_patch.apply_op(doc, {"op": "replace", "path": "/a~", "value": "X"})
        assert doc == {"a~": "RIGHT"}

    def test_the_whole_document_pointer_is_refused_not_ignored(self):
        """`""` is the whole document (§5). Every verb here edits a member of
        a container, so there is nothing to serve — but it is refused out
        loud. It used to return quietly, which reports success."""
        doc = {"k": 1}
        with pytest.raises(BadRequestError):
            json_patch.apply_op(doc, {"op": "replace", "path": "", "value": "X"})
        assert doc == {"k": 1}

    def test_an_op_that_omits_its_path_is_still_skipped(self):
        """Structurally incomplete, which is not the same as naming the whole
        document. The existing contract for these is unchanged."""
        doc = {"k": 1}
        json_patch.apply_op(doc, {"op": "replace", "value": "X"})
        assert doc == {"k": 1}

    def test_a_source_pointer_is_read_the_same_way(self):
        """`from` goes through the same tokenizer, or `move` and `copy` get
        their own address book."""
        doc = {"a/b": "RIGHT", "a~1b": "WRONG"}
        json_patch.apply_op(doc, {"op": "move", "from": "/a~1b", "path": "/moved"})
        assert doc == {"a~1b": "WRONG", "moved": "RIGHT"}


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


def _published(client, admin_headers, extra=None):
    schema = json.loads(json.dumps(WORKFLOW))
    schema.update(extra or {})
    made = client.post("/v1/artifacts", headers=admin_headers, json={
        "type": "workflow", "name": f"wl-{uuid.uuid4().hex[:6]}",
        "schema": schema, "visibility": "global",
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

    def test_an_escaped_pointer_lands_on_the_key_it_names(
        self, client, admin_headers
    ):
        """The pointer form of the defect that started this tranche.

        `/a~1b` names the key `a/b`. Undecoded it names `a~1b`, which is a
        real and different key. So this patch succeeds — 200, `applied`, a new
        artifact_version — while changing a key the operator did not name and
        leaving the one they did name alone. Same audit trail asserting a
        change that did not happen, reached through the pointer instead of
        through the path root.
        """
        artifact = _published(client, admin_headers,
                              extra={"a/b": "RIGHT", "a~1b": "WRONG"})

        proposed = client.post("/v1/config/propose_patch", headers=admin_headers, json={
            "artifact_id": artifact,
            "patch": [{"op": "replace", "path": "/a~1b", "value": "CHANGED"}],
            "justification": "an escaped slash in the pointer",
        })
        assert proposed.status_code in (200, 201), proposed.text
        patch_id = proposed.json()["data"]["id"]
        client.post(f"/v1/config/patches/{patch_id}/decide",
                    headers=admin_headers, json={"decision": "approve"})

        applied = client.post(f"/v1/config/patches/{patch_id}/apply",
                              headers=admin_headers, json={})
        assert applied.status_code == 200, applied.text

        schema = _schema(artifact)
        assert schema["a/b"] == "CHANGED", "the named key was not the one written"
        assert schema["a~1b"] == "WRONG", "a key the pointer did not name was written"

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
