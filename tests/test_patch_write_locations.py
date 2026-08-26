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
import threading
import time
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

    def test_a_huge_gap_must_not_silently_mean_the_end(self):
        """`/xs/999999999` must not quietly become `/xs/2`.

        Nothing here pads a list, so deleting the length check does not
        allocate a billion entries — measured, it falls through to one
        `append` and lands at index 2. That is the failure: an address the
        caller never named, reported as success. Same rule as `/xs/5`.
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

    def test_a_move_without_a_source_says_so(self):
        """Found while fixing the pointer, same class as the pointer.

        A missing `from` used to default to "", go through the tokenizer, and
        come back as "addresses the whole document" — a true sentence about
        an operand the caller never wrote. It is now one entry in the operand
        table rather than a special case here.
        """
        doc = {"k": 1}
        with pytest.raises(BadRequestError, match="from"):
            json_patch.apply_op(doc, {"op": "move", "path": "/b"})
        assert doc == {"k": 1}

    def test_a_source_pointer_is_read_the_same_way(self):
        """`from` goes through the same tokenizer, or `move` and `copy` get
        their own address book."""
        doc = {"a/b": "RIGHT", "a~1b": "WRONG"}
        json_patch.apply_op(doc, {"op": "move", "from": "/a~1b", "path": "/moved"})
        assert doc == {"a~1b": "WRONG", "moved": "RIGHT"}


class TestAnOperationCarriesItsOperands:
    """RFC 6902 §4: an operation carries `op` and `path`, and each verb names
    the further members it needs.

    Treating an absent member as a default is the same defect one level up
    from a mis-parsed pointer: the engine acts on an operand nobody supplied.
    A half-formed op used to be skipped in silence, which through the artifact
    route still wrote a new version — an audit entry for a patch that did
    nothing.
    """

    def test_a_replace_without_a_value_does_not_write_none(self):
        """The sharpest of these: it does not no-op, it destroys.

        `{"op": "replace", "path": "/k"}` produced `{"k": None}` — the value
        overwritten on behalf of an operand the caller never wrote.
        """
        doc = {"k": "ORIGINAL"}
        with pytest.raises(BadRequestError, match="value"):
            json_patch.apply_op(doc, {"op": "replace", "path": "/k"})
        assert doc == {"k": "ORIGINAL"}, "the value was destroyed by a malformed op"

    def test_an_add_without_a_value_is_refused(self):
        doc = {}
        with pytest.raises(BadRequestError, match="value"):
            json_patch.apply_op(doc, {"op": "add", "path": "/new"})
        assert doc == {}

    def test_a_test_without_a_value_is_a_shape_error_not_a_comparison(self):
        """It used to compare against `None` and report the *test* as failed,
        which sends the reader looking at their document."""
        with pytest.raises(BadRequestError, match="value"):
            json_patch.apply_op({"k": 1}, {"op": "test", "path": "/k"})

    def test_an_explicit_null_value_is_still_a_legal_operand(self):
        """The control that stops the fix from being "reject falsy values".

        `value: null` is a perfectly good JSON Patch operand. The question is
        whether the member is *present*, never whether it is truthy.
        """
        doc = {"k": "ORIGINAL"}
        json_patch.apply_op(doc, {"op": "replace", "path": "/k", "value": None})
        assert doc == {"k": None}

    def test_an_op_without_a_path_is_refused(self):
        doc = {"k": 1}
        with pytest.raises(BadRequestError, match="path"):
            json_patch.apply_op(doc, {"op": "replace", "value": "X"})
        assert doc == {"k": 1}

    def test_an_op_without_a_verb_is_refused(self):
        doc = {"k": 1}
        with pytest.raises(BadRequestError, match="op"):
            json_patch.apply_op(doc, {"path": "/k", "value": "X"})
        assert doc == {"k": 1}

    def test_an_empty_operation_is_refused(self):
        doc = {"k": 1}
        with pytest.raises(BadRequestError):
            json_patch.apply_op(doc, {})
        assert doc == {"k": 1}

    def test_a_patch_that_names_no_operation_is_refused(self):
        """The same defect one level up, found by grepping the class.

        An empty list is well-formed JSON and still names no change. Both
        callers accepted it and wrote a version anyway: the artifact route
        guarded `apply_ops` behind `if ops:` and went straight to the store —
        measured, `{"patch": []}` returned 200 and took the artifact from
        version 1 to 2 — and ConfigOps looped zero times and marked the patch
        applied.
        """
        with pytest.raises(BadRequestError, match="no operation"):
            json_patch.apply_ops({"a": 1}, [])

    def test_a_patch_that_is_not_a_list_is_refused(self):
        with pytest.raises(BadRequestError):
            json_patch.apply_ops({"a": 1}, {"op": "remove", "path": "/a"})

    @pytest.mark.parametrize(
        "bad", [None, 42, 1.5, ["/a"], {"a": 1}, True],
        ids=["null", "number", "float", "array", "object", "bool"],
    )
    def test_a_path_that_is_not_a_string_is_refused(self, bad):
        """Presence was required, type was not.

        `_segments_or_raise` reaches straight for `path.startswith("/")`, so a
        non-string pointer left as an uncaught AttributeError — a 500 for what
        is plainly a bad request. Both API models accept nested arbitrary
        dicts, so this arrives over the wire.
        """
        with pytest.raises(BadRequestError):
            json_patch.apply_op({"k": 1}, {"op": "replace", "path": bad, "value": "X"})

    @pytest.mark.parametrize(
        "bad", [None, 42, ["/a"], {"a": 1}],
        ids=["null", "number", "array", "object"],
    )
    def test_a_from_that_is_not_a_string_is_refused(self, bad):
        with pytest.raises(BadRequestError):
            json_patch.apply_op({"k": 1}, {"op": "move", "path": "/dest", "from": bad})

    def test_a_remove_needs_no_further_operand(self):
        """The control for the operand table: `remove` is complete with
        `op` and `path` alone."""
        doc = {"k": 1, "j": 2}
        json_patch.apply_op(doc, {"op": "remove", "path": "/k"})
        assert doc == {"j": 2}


class TestArrayIndexGrammar:
    """RFC 6901 §4: an array index is `0` or a non-zero digit run. ASCII, no
    leading zeros.

    `str.isdigit()` is a far larger set, so several distinct pointer spellings
    named one position — the normalization this whole file exists to stop.
    """

    def test_a_leading_zero_is_not_the_same_index(self):
        doc = {"xs": ["a", "b", "c"]}
        with pytest.raises(BadRequestError):
            json_patch.apply_op(doc, {"op": "replace", "path": "/xs/01", "value": "X"})
        assert doc == {"xs": ["a", "b", "c"]}

    @pytest.mark.parametrize(
        "token, name",
        [("١", "arabic-indic one"), ("０", "fullwidth zero")],
        ids=["arabic-indic", "fullwidth"],
    )
    def test_a_non_ascii_digit_is_not_an_index(self, token, name):
        """`"١".isdigit()` is True and `int("١")` is 1, so these reached a real
        position under a spelling RFC 6901 does not define."""
        doc = {"xs": ["a", "b", "c"]}
        with pytest.raises(BadRequestError):
            json_patch.apply_op(
                doc, {"op": "replace", "path": f"/xs/{token}", "value": "X"}
            )
        assert doc == {"xs": ["a", "b", "c"]}, f"{name} addressed a position"

    @pytest.mark.parametrize("token", ["²", "-²"], ids=["sup2", "neg-sup2"])
    def test_a_digit_int_cannot_read_is_a_bad_request_not_a_crash(self, token):
        """`"²".isdigit()` is True and `int("²")` raises. The pair let a
        malformed pointer leave as an uncaught ValueError instead of a 400 —
        the one case here that was a 500 rather than a wrong write.
        """
        with pytest.raises(BadRequestError):
            json_patch.apply_op(
                {"xs": ["a"]}, {"op": "replace", "path": f"/xs/{token}", "value": "X"}
            )

    def test_ordinary_indices_still_work(self):
        """The control. Refusing every unusual spelling must not refuse the
        ordinary ones, including a two-digit index and the append token."""
        assert apply({"xs": ["a", "b"]},
                     {"op": "replace", "path": "/xs/0", "value": "X"})["xs"] == ["X", "b"]
        assert apply({"xs": list(range(11))},
                     {"op": "replace", "path": "/xs/10", "value": "X"})["xs"][10] == "X"
        assert apply({"xs": ["a"]},
                     {"op": "add", "path": "/xs/-", "value": "b"})["xs"] == ["a", "b"]


class TestTestComparesJsonValues:
    """RFC 6902 §4.6 compares JSON values. Python `==` does not.

    Python makes `True == 1` and `False == 0`, and carries that equivalence
    recursively through lists and dicts. JSON has no such rule: booleans and
    numbers are different value classes. `test` exists to guard the
    operations after it, so an equality that is too generous does not just
    misreport — it lets a mutation through on a precondition that was never
    actually met.
    """

    @pytest.mark.parametrize(
        "held, expected, why",
        [
            (True, 1, "a JSON boolean is not the number one"),
            (1, True, "and the same in the other direction"),
            (False, 0, "nor is false the number zero"),
            (0, False, "either way round"),
            ([True], [1], "arrays carry the confusion into their elements"),
            ({"a": True}, {"a": 1}, "and objects into their members"),
            ([1, [True]], [1, [1]], "at any depth"),
        ],
        ids=["true-1", "1-true", "false-0", "0-false", "array", "object", "nested"],
    )
    def test_a_boolean_is_not_a_number(self, held, expected, why):
        with pytest.raises(BadRequestError, match="test operation failed"):
            json_patch.apply_op({"k": held}, {"op": "test", "path": "/k",
                                              "value": expected})

    @pytest.mark.parametrize(
        "held, expected, why",
        [
            (1, 1.0, "JSON has one number type, so 1 and 1.0 are one value"),
            (True, True, "a boolean still equals itself"),
            ([1, {"a": "s"}], [1, {"a": "s"}], "and structures still match"),
            (None, None, "null equals null"),
        ],
        ids=["int-float", "bool-bool", "nested", "null"],
    )
    def test_equal_json_values_still_pass(self, held, expected, why):
        """The controls. A fix that just refused more would pass every case
        above and break `test` entirely."""
        json_patch.apply_op({"k": held}, {"op": "test", "path": "/k",
                                          "value": expected})

    @pytest.mark.parametrize(
        "held, expected",
        [("1", 1), (1, "1"), (None, 0), (0, None), ({"a": 1}, {"a": 1, "b": 2})],
        ids=["str-num", "num-str", "null-zero", "zero-null", "extra-key"],
    )
    def test_other_type_mismatches_still_fail(self, held, expected):
        with pytest.raises(BadRequestError, match="test operation failed"):
            json_patch.apply_op({"k": held}, {"op": "test", "path": "/k",
                                              "value": expected})


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


def _private(client, admin_headers, extra=None):
    schema = json.loads(json.dumps(WORKFLOW))
    schema.update(extra or {})
    made = client.post("/v1/artifacts", headers=admin_headers, json={
        "type": "workflow", "name": f"pv-{uuid.uuid4().hex[:6]}",
        "schema": schema, "visibility": "private",
    })
    assert made.status_code in (200, 201), made.text
    return made.json()["data"]["id"]


# Two tool_call nodes, so an index that should be refused still names a real
# `tool` when it is wrongly normalized. Against the single-tool_call fixture
# `/nodes/01/tool` would be refused for having no target, which proves
# nothing about the index.
TWO_TOOLS = {
    "entrypoint": "first",
    "nodes": [
        {"id": "first", "type": "tool_call", "tool": "llm.generic", "next": "second"},
        {"id": "second", "type": "tool_call", "tool": "llm.generic", "next": "end"},
        {"id": "end", "type": "end"},
    ],
}


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


class TestThePatchProducersEmitApplicablePatches:
    """A patch this system generates for itself must apply to the artifact it
    names.

    Two producers write a single key under `/meta`: the ConfigOps fallback
    patch and the adapter auto-prune proposer. A freshly created artifact has
    no `meta` in its schema, and traversal no longer invents one, so both now
    emit a patch that stores `pending`, approves cleanly, and then fails on
    apply — a dead end this branch introduced.

    Prepending an unconditional `add /meta` is not the fix: `add` on a member
    that is already there replaces it, so that trades a refused patch for a
    destroyed one. There is a control below holding exactly that line.
    """

    def test_the_generated_patch_is_one_leaf_op(self):
        """No `add /meta` alongside it, ever.

        A stored patch is applied later than it is written, so a parent
        creation decided at proposal time is a decision made against a
        document that may no longer be there.
        """
        from liminallm.service.json_patch import meta_ops

        assert [o["path"] for o in meta_ops("llm_autopatch", {"x": 1})] == [
            "/meta/llm_autopatch"
        ]

    def test_a_meta_that_appeared_since_the_proposal_survives(self):
        """The witness that made the first repair wrong.

        The patch is written against an artifact with no `meta` and applied
        after something else has put one there. The parent-creating version
        carried `add /meta {}` and wiped it; the leaf op adds its own key and
        leaves the rest alone.
        """
        from liminallm.service.json_patch import meta_ops

        proposed = meta_ops("llm_autopatch", {"x": 1})
        later = {"kind": "workflow.chat", "meta": {"landed_in_between": "KEEP"}}
        out = json_patch.apply_ops(later, proposed)
        assert out["meta"] == {"landed_in_between": "KEEP", "llm_autopatch": {"x": 1}}

    def test_an_existing_meta_keeps_its_other_members(self):
        from liminallm.service.json_patch import meta_ops

        out = json_patch.apply_ops(
            {"kind": "workflow.chat", "meta": {"keep": "ME"}},
            meta_ops("auto_prune", {"x": 1}),
        )
        assert out["meta"] == {"keep": "ME", "auto_prune": {"x": 1}}

    def test_a_bare_artifact_refuses_the_patch_rather_than_mangling_it(self):
        """What the leaf op gives up, held in place deliberately.

        An artifact with no `meta` refuses this patch. That is a visible dead
        end rather than silent damage, and it is the better half of the trade:
        closing it properly means version-gating stored patches or moving
        these annotations out of the schema document.
        """
        from liminallm.service.json_patch import meta_ops

        schema = {"kind": "workflow.chat", "nodes": []}
        with pytest.raises(BadRequestError):
            json_patch.apply_ops(schema, meta_ops("auto_prune", {"x": 1}))
        assert schema == {"kind": "workflow.chat", "nodes": []}

    def test_the_real_pruning_producer_lands_its_key_and_keeps_siblings(
        self, client, admin_headers
    ):
        """`recommend_adapter_pruning()` itself, not the helper it calls.

        This call site has changed twice in this branch and had no witness
        either time: the tests drove `meta_ops` directly and the ConfigOps
        fallback, and the periodic-worker test uses a fake training service,
        so it proves scheduling and not patch construction.

        A genuinely prune-eligible adapter, through propose → approve → apply,
        with a `meta` sibling that has to survive.
        """
        from datetime import datetime, timedelta, timezone

        runtime = get_runtime()
        store = runtime.store

        made = client.post("/v1/artifacts", headers=admin_headers, json={
            "type": "adapter", "name": f"ad-{uuid.uuid4().hex[:6]}", "visibility": "private",
            "schema": {
                "kind": "adapter.lora", "mode": "prompt", "base_model": "m",
                "current_version": 1, "meta": {"keep": "ME"},
            },
        })
        assert made.status_code in (200, 201), made.text
        adapter = made.json()["data"]["id"]

        # Eligible: below both thresholds and last used before the stale
        # cutoff. Read from the service so the fixture cannot drift from it.
        stale = datetime.now(timezone.utc) - timedelta(
            days=runtime.training.ADAPTER_PRUNE_STALE_DAYS + 1
        )
        store.update_adapter_router_state(
            adapter,
            success_score=runtime.training.ADAPTER_PRUNE_MAX_SUCCESS / 2,
            last_used_at=stale,
        )

        assert runtime.training.recommend_adapter_pruning() >= 1, "nothing proposed"

        pending = [
            p for p in store.list_config_patches("pending")
            if p.artifact_id == adapter
        ]
        assert len(pending) == 1, f"expected one proposal, got {len(pending)}"
        patch_id = pending[0].id
        assert [o["path"] for o in pending[0].patch["ops"]] == ["/meta/auto_prune"]

        client.post(f"/v1/config/patches/{patch_id}/decide",
                    headers=admin_headers, json={"decision": "approve"})
        applied = client.post(f"/v1/config/patches/{patch_id}/apply",
                              headers=admin_headers, json={})

        assert applied.status_code == 200, applied.text
        meta = store.get_artifact(adapter).schema["meta"]
        assert meta["auto_prune"]["recommended"] is True
        assert meta["keep"] == "ME", "the producer clobbered an existing meta"

    def test_the_configops_fallback_is_the_same_single_op(
        self, client, admin_headers
    ):
        """Through the real producer, with the model failing so the fallback
        is what gets stored — and applied against an artifact that has a
        `meta`, which is the case the fallback can actually serve."""
        artifact = _published(client, admin_headers, extra={"meta": {"keep": "ME"}})
        runtime = get_runtime()
        ops = runtime.config_ops
        original = ops.llm.generate

        def _fails(*args, **kwargs):
            raise RuntimeError("no model")

        ops.llm.generate = _fails
        try:
            audit = ops.auto_generate_patch(artifact, None, goal="probe")
        finally:
            ops.llm.generate = original

        assert [o["path"] for o in audit.patch["ops"]] == ["/meta/llm_autopatch"]

        client.post(f"/v1/config/patches/{audit.id}/decide",
                    headers=admin_headers, json={"decision": "approve"})
        applied = client.post(f"/v1/config/patches/{audit.id}/apply",
                              headers=admin_headers, json={})

        assert applied.status_code == 200, applied.text
        meta = _schema(artifact)["meta"]
        assert "llm_autopatch" in meta
        assert meta["keep"] == "ME", "the proposal clobbered an existing meta"


class TestApplyIsOneReadModifyWrite:
    """SPEC §10.1: applying a config patch loads the *current*
    `artifact.schema`, applies the patch, validates, writes the version, then
    marks the patch applied.

    "Current" is the load-bearing word. The service read the artifact and
    computed the new document before entering the store transaction, and the
    store then locked the artifact row and wrote that already-computed
    document — so the lock serialized the write without covering the read it
    came from. Anything committed in between was overwritten.

    The earlier staleness witness in this file asks the right question at the
    wrong altitude: it calls `apply_ops` on an already-later dictionary, which
    exercises the engine and never touches the lifecycle.
    """

    def _proposed(self, client, admin_headers, artifact, ops):
        proposed = client.post("/v1/config/propose_patch", headers=admin_headers, json={
            "artifact_id": artifact, "patch": ops, "justification": "race",
        })
        assert proposed.status_code in (200, 201), proposed.text
        patch_id = proposed.json()["data"]["id"]
        client.post(f"/v1/config/patches/{patch_id}/decide",
                    headers=admin_headers, json={"decision": "approve"})
        return patch_id

    def test_an_edit_committed_during_apply_is_not_lost(self, client, admin_headers):
        """The lost update, driven through the real lifecycle.

        A concurrent edit lands after ConfigOps has read the artifact and
        computed its result, and before the row is written. Either outcome is
        correct — apply against the newly locked schema and keep both changes,
        or refuse as stale and change nothing — but the edit must not vanish.
        """
        artifact = _private(client, admin_headers, extra={"meta": {"keep": "ME"}})
        patch_id = self._proposed(client, admin_headers, artifact, [
            {"op": "add", "path": "/meta/from_patch", "value": "PATCH"},
        ])

        runtime = get_runtime()
        store = runtime.store
        real = store.apply_config_patch

        def racing(*args, **kwargs):
            # The seam is here rather than inside the patch computation: once
            # the fix moves that computation under the artifact lock, an edit
            # attempted from inside it waits on a lock this very call holds.
            # This runs before the transaction opens, so it commits through
            # the ordinary mutation path exactly as another replica would —
            # and it is the same seam either way, because the unfixed service
            # has already computed its result by the time it gets here.
            edit = client.patch(f"/v1/artifacts/{artifact}", headers=admin_headers,
                                json={"patch": [{"op": "add", "path": "/meta/landed",
                                                 "value": "CONCURRENT"}]})
            assert edit.status_code == 200, edit.text
            return real(*args, **kwargs)

        store.apply_config_patch = racing
        try:
            applied = client.post(f"/v1/config/patches/{patch_id}/apply",
                                  headers=admin_headers, json={})
        finally:
            store.apply_config_patch = real

        meta = _schema(artifact).get("meta", {})
        assert meta.get("landed") == "CONCURRENT", (
            "the concurrent edit was overwritten by a schema computed before it"
        )
        if applied.status_code == 200:
            assert meta.get("from_patch") == "PATCH", "the patch reported success"
        else:
            assert "from_patch" not in meta, "a refused apply changed the artifact"
            status = get_runtime().store.get_config_patch(patch_id).status
            assert status == "approved", f"a refused apply left the patch {status}"

    def test_the_private_route_does_not_overwrite_a_concurrent_apply(
        self, client, admin_headers
    ):
        """The same defect in the other direction, found by grepping the class.

        Fixing ConfigOps closed the race for one writer. The ordinary private
        PATCH route still reads the artifact, computes the whole new schema,
        and hands it to a store method that locks the row and writes the
        document it was given — so with the two writers interleaved the other
        way round, it is the *applied* ConfigOps patch that disappears:

            private PATCH reads schema N, computes N + D
            ConfigOps locks, reads N, writes N + C, marks the patch applied
            private PATCH takes the lock, writes its precomputed N + D
            -> C is gone, and its audit trail still says applied

        Which is the campaign invariant exactly: the patch says applied, a
        version records it, and the serving configuration does not have it.
        """
        artifact = _private(client, admin_headers,
                            extra={"field_c": "ORIGINAL", "field_d": "ORIGINAL"})
        patch_id = self._proposed(client, admin_headers, artifact, [
            {"op": "replace", "path": "/field_c", "value": "C"},
        ])

        store = get_runtime().store
        real = store.update_private_artifact

        def racing(*args, **kwargs):
            # The route has computed its result and holds no lock here.
            applied = client.post(f"/v1/config/patches/{patch_id}/apply",
                                  headers=admin_headers, json={})
            assert applied.status_code == 200, applied.text
            return real(*args, **kwargs)

        store.update_private_artifact = racing
        try:
            resp = client.patch(f"/v1/artifacts/{artifact}", headers=admin_headers,
                                json={"patch": [{"op": "replace", "path": "/field_d",
                                                 "value": "D"}]})
        finally:
            store.update_private_artifact = real

        schema = _schema(artifact)
        patch_status = get_runtime().store.get_config_patch(patch_id).status
        assert schema["field_c"] == "C", (
            f"the applied ConfigOps patch was overwritten; it is still {patch_status!r}"
        )
        if resp.status_code == 200:
            assert schema["field_d"] == "D", "the route reported success"
        else:
            assert schema["field_d"] == "ORIGINAL", "a refused route call still wrote"

    def test_one_approved_patch_applies_exactly_once_in_sequence(
        self, client, admin_headers
    ):
        artifact = _private(client, admin_headers, extra={"meta": {"keep": "ME"}})
        before_versions = _versions(artifact)
        patch_id = self._proposed(client, admin_headers, artifact, [
            {"op": "add", "path": "/meta/from_patch", "value": "PATCH"},
        ])

        first = client.post(f"/v1/config/patches/{patch_id}/apply",
                            headers=admin_headers, json={})
        second = client.post(f"/v1/config/patches/{patch_id}/apply",
                             headers=admin_headers, json={})

        assert first.status_code == 200, first.text
        assert second.status_code == 400, second.text
        assert _versions(artifact) == before_versions + 1, "two versions for one patch"

    @pytest.mark.slow
    def test_one_approved_patch_applies_exactly_once_under_contention(
        self, client, admin_headers
    ):
        """Two applies overlapping, which is the case the sequential one
        cannot reach.

        The status check ran outside the transaction and the status write had
        no `approved` guard, so both callers could observe `approved` and each
        write a version for one patch. The sequential test never sees it —
        the first apply has already committed by the time the second reads.

        Determinism comes from the patch computation, which the fix runs
        inside the transaction: the first apply blocks there holding both row
        locks, so the second is queued on the patch row rather than racing.
        Without the fix, the first blocks before any transaction and the
        second runs to completion, which is what makes this witness fail.
        """
        artifact = _private(client, admin_headers, extra={"meta": {"keep": "ME"}})
        before_versions = _versions(artifact)
        patch_id = self._proposed(client, admin_headers, artifact, [
            {"op": "add", "path": "/meta/from_patch", "value": "PATCH"},
        ])

        ops = get_runtime().config_ops
        real = ops._apply_patch_to_schema
        reached, release = threading.Event(), threading.Event()
        first_call = threading.Lock()
        held = {"taken": False}

        def blocking(schema, patch):
            with first_call:
                mine = not held["taken"]
                held["taken"] = True
            if mine:
                reached.set()
                assert release.wait(timeout=30), "the second apply never arrived"
            return real(schema, patch)

        results = {}

        def apply_as(name):
            results[name] = client.post(f"/v1/config/patches/{patch_id}/apply",
                                        headers=admin_headers, json={}).status_code

        ops._apply_patch_to_schema = blocking
        try:
            a = threading.Thread(target=apply_as, args=("a",), daemon=True)
            a.start()
            assert reached.wait(timeout=30), "the first apply never reached the seam"
            b = threading.Thread(target=apply_as, args=("b",), daemon=True)
            b.start()
            # Long enough for the second to reach the lock it should queue on.
            time.sleep(1.0)
            release.set()
            a.join(timeout=60)
            b.join(timeout=60)
        finally:
            release.set()
            ops._apply_patch_to_schema = real

        assert sorted(results.values()) == [200, 400], f"both applied: {results}"
        assert _versions(artifact) == before_versions + 1, (
            f"two versions for one patch: {_versions(artifact) - before_versions}"
        )


class TestArtifactPatchInheritsIt:
    def test_a_half_formed_op_does_not_write_a_version(self, client, admin_headers):
        """The route's own consequence for a silently skipped operation.

        `{"op": "replace", "value": "CHANGED"}` has no `path`. The engine used
        to return quietly, so the route carried on into
        `update_private_artifact` and wrote an artifact update and a new
        version recording a patch that changed nothing. The request model does
        not check operand shape either, so nothing above the engine caught it.
        """
        artifact = _private(client, admin_headers)
        before, before_versions = _schema(artifact), _versions(artifact)

        resp = client.patch(f"/v1/artifacts/{artifact}", headers=admin_headers, json={
            "patch": [{"op": "replace", "value": "CHANGED"}],
        })

        assert resp.status_code == 400, resp.text
        assert _schema(artifact) == before, "the artifact changed anyway"
        assert _versions(artifact) == before_versions, "a version was written"

    def test_a_patch_with_no_operations_does_not_write_a_version(
        self, client, admin_headers
    ):
        """`if ops:` skipped the engine entirely, so an empty patch never met
        a rule — it went straight to the store and got a version."""
        artifact = _private(client, admin_headers)
        before, before_versions = _schema(artifact), _versions(artifact)

        resp = client.patch(f"/v1/artifacts/{artifact}", headers=admin_headers,
                            json={"patch": []})

        assert resp.status_code == 400, resp.text
        assert _schema(artifact) == before
        assert _versions(artifact) == before_versions, "a version was written"

    @pytest.mark.parametrize(
        "op, label",
        [
            ({"op": "replace", "path": 42, "value": "X"}, "path"),
            ({"op": "move", "path": "/dest", "from": 42}, "from"),
        ],
        ids=["path", "from"],
    )
    def test_a_non_string_pointer_is_a_bad_request_not_a_crash(
        self, client, admin_headers, op, label
    ):
        """It arrives over the wire as valid JSON, so it has to leave as 400.

        `ArtifactPatchRequest.patch` is `List[dict]`, which admits any JSON
        value in any member. `_segments_or_raise` then calls `.startswith`
        on it.
        """
        artifact = _private(client, admin_headers)
        before, before_versions = _schema(artifact), _versions(artifact)

        resp = client.patch(f"/v1/artifacts/{artifact}", headers=admin_headers,
                            json={"patch": [op]})

        assert resp.status_code == 400, f"{label}: {resp.text}"
        assert _schema(artifact) == before
        assert _versions(artifact) == before_versions

    def test_a_failed_test_stops_the_operations_after_it(
        self, client, admin_headers
    ):
        """What `test` is for, and what Python equality quietly gave away.

        The stored value is JSON `true` and the precondition asks for the
        number 1. Those are different JSON values, so the guard must fail and
        the replace behind it must never run. `True == 1` in Python, so it
        ran: `spare` changed and a version was written on a precondition that
        was never met.
        """
        artifact = _private(client, admin_headers, extra={"enabled": True})
        before, before_versions = _schema(artifact), _versions(artifact)

        resp = client.patch(f"/v1/artifacts/{artifact}", headers=admin_headers, json={
            "patch": [
                {"op": "test", "path": "/enabled", "value": 1},
                {"op": "replace", "path": "/spare", "value": "CHANGED"},
            ],
        })

        assert resp.status_code == 400, resp.text
        assert _schema(artifact)["spare"] == "keep", "the guarded op ran anyway"
        assert _schema(artifact) == before
        assert _versions(artifact) == before_versions, "a version was written"

    def test_a_numeric_field_is_not_guarded_by_a_boolean(
        self, client, admin_headers
    ):
        """The inverse, so a fix cannot special-case one direction."""
        artifact = _private(client, admin_headers, extra={"retries": 1})
        before_versions = _versions(artifact)

        resp = client.patch(f"/v1/artifacts/{artifact}", headers=admin_headers, json={
            "patch": [
                {"op": "test", "path": "/retries", "value": True},
                {"op": "replace", "path": "/spare", "value": "CHANGED"},
            ],
        })

        assert resp.status_code == 400, resp.text
        assert _schema(artifact)["spare"] == "keep"
        assert _versions(artifact) == before_versions

    def test_a_test_that_holds_still_lets_the_patch_through(
        self, client, admin_headers
    ):
        """The control at the caller: a precondition that is genuinely met
        must still apply the operations behind it."""
        artifact = _private(client, admin_headers, extra={"enabled": True})
        before_versions = _versions(artifact)

        resp = client.patch(f"/v1/artifacts/{artifact}", headers=admin_headers, json={
            "patch": [
                {"op": "test", "path": "/enabled", "value": True},
                {"op": "replace", "path": "/spare", "value": "CHANGED"},
            ],
        })

        assert resp.status_code == 200, resp.text
        assert _schema(artifact)["spare"] == "CHANGED"
        assert _versions(artifact) == before_versions + 1

    def test_a_leading_zero_index_does_not_reach_a_node(self, client, admin_headers):
        """The index grammar in the shape it actually arrives in.

        `/nodes/01/tool` is not a JSON Pointer array index. It used to
        normalize to 1 and rewrite the second node's tool, then return 200 and
        write a version — an operator's typo becoming a silent edit to a
        different node than the one they named.
        """
        artifact = _private(client, admin_headers, extra=TWO_TOOLS)
        before, before_versions = _schema(artifact), _versions(artifact)

        resp = client.patch(f"/v1/artifacts/{artifact}", headers=admin_headers, json={
            "patch": [{"op": "replace", "path": "/nodes/01/tool", "value": "CHANGED"}],
        })

        assert resp.status_code == 400, resp.text
        assert _schema(artifact) == before, "a node the pointer did not name changed"
        assert _versions(artifact) == before_versions, "a version was written"

    def test_a_private_artifact_refuses_a_location_that_does_not_exist(
        self, client, admin_headers
    ):
        artifact = _private(client, admin_headers)
        before, before_versions = _schema(artifact), _versions(artifact)

        resp = client.patch(f"/v1/artifacts/{artifact}", headers=admin_headers, json={
            "patch": [{"op": "replace", "path": "/schema/spare", "value": "CHANGED"}],
        })

        assert resp.status_code == 400, resp.text
        assert _schema(artifact) == before
        assert _versions(artifact) == before_versions
