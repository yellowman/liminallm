"""RFC 6902 JSON Patch — one implementation.

Two grew independently. ConfigOps had the hardened traversal — root path,
scalar traversal, and unbounded list indexes all answer 400 — but knew only
add/replace/remove. The artifact PATCH route knew move/copy/test but swallowed
every failure with ``pass``, so a patch that did nothing reported success.
This module is the union: full verb set, hardened everywhere. Patch bodies
are model-authored as often as human-authored, so the malformed shapes are
routine arrivals, and the reviewer must be told which path is wrong.
"""

from __future__ import annotations

import copy
import re
from typing import Any, Dict, Iterable, List

from liminallm.service.errors import BadRequestError

# RFC 6902 §4: every operation carries `op` and `path`; each verb names the
# further members it needs. `remove` needs none.
_OPERANDS: Dict[str, tuple] = {
    "add": ("value",),
    "replace": ("value",),
    "test": ("value",),
    "move": ("from",),
    "copy": ("from",),
    "remove": (),
}

# RFC 6901 §4 array index: `0`, or a non-zero digit followed by digits. ASCII,
# and no leading zeros. `str.isdigit()` accepted `01`, `007`, `١` and `０` as
# ordinary indices, so several spellings named one position — and `²`, which
# satisfies `isdigit()` but not `int()`, left as an uncaught ValueError.
_INDEX = re.compile(r"^(?:0|[1-9][0-9]*)$")
_NEGATIVE_INDEX = re.compile(r"^-(?:0|[1-9][0-9]*)$")


def validate_op(op: Any) -> None:
    """Refuse an operation whose required members are absent.

    The engine is the last boundary rather than the only one: request models
    call this too, so the API can refuse a malformed patch before it reaches
    a store, without either side keeping its own copy of the rule.

    Absence is the question, never truthiness — `value: null` is a legal
    operand, and reading it as "no value" is how `{"op": "replace",
    "path": "/k"}` came to write `None` over a value nobody asked to change.
    """
    if not isinstance(op, dict):
        raise BadRequestError(
            "patch operation is not an object",
            detail={"found": type(op).__name__},
        )
    action = op.get("op")
    if not action or not isinstance(action, str):
        raise BadRequestError("patch operation is missing its op", detail={"op": op})
    if action not in _OPERANDS:
        raise BadRequestError("unknown patch operation", detail={"op": action})
    if "path" not in op:
        raise BadRequestError(
            "patch operation is missing its path", detail={"op": action}
        )
    _require_pointer(op, "path", action)
    for member in _OPERANDS[action]:
        if member not in op:
            raise BadRequestError(
                f"patch operation is missing its {member}",
                detail={"op": action, "path": op.get("path")},
            )
    if "from" in _OPERANDS[action]:
        _require_pointer(op, "from", action)


def _require_pointer(op: Dict[str, Any], member: str, action: str) -> None:
    """A pointer operand is a string (RFC 6901 §3), and is not coerced.

    Presence was required and type was not, so `_segments_or_raise` reached
    for `.startswith` on whatever arrived and a number, null, array or object
    left as an uncaught AttributeError — a 500 for a plainly bad request, and
    reachable over the wire because both API models take `List[dict]`.

    Refused rather than coerced: `str(42)` is `"42"`, a pointer that is not
    the one anybody wrote, which is the failure this module exists to stop.
    `bool` is excluded explicitly — it is not a `str`, but saying so keeps
    the check honest next to the JSON-value rules below.
    """
    value = op[member]
    if not isinstance(value, str):
        raise BadRequestError(
            f"patch operation {member} is not a JSON Pointer string",
            detail={"op": action, member: value, "found": type(value).__name__},
        )


def meta_ops(key: str, value: Any) -> List[Dict[str, Any]]:
    """Ops that write ``value`` at ``/meta/<key>``, and nothing else.

    For the patches this system generates for itself. It exists to hold one
    decision in place: **the parent is never created here.**

    The tempting version inspects the artifact and prepends `add /meta {}`
    when `meta` is missing, so a proposal against a bare artifact still
    applies. That was written, and it is wrong. ConfigOps stores a patch and
    applies it later, and `add` on a member that is already present replaces
    it — so if anything puts a `meta` there in between (another pending
    patch, a direct edit, the second producer on the same artifact), the
    baked `add /meta {}` silently wipes it. The data loss is not avoided,
    only deferred across the propose/apply gap.

    RFC 6902 has no "add if absent" and no test for absence, so no
    proposal-time decision about the parent can be made stale-proof. The leaf
    op alone is both safer and better behaved under staleness:

    ==================  ==========================  ========================
    at apply time       parent-creating             leaf only
    ==================  ==========================  ========================
    `meta` absent       applies                     refused, nothing changed
    `meta` appeared     **destroys it**             applies, siblings kept
    ==================  ==========================  ========================

    What it gives up is the bare-artifact case, where the patch is refused
    instead of applying. That is a visible dead end rather than silent
    damage, and closing it properly means either version-gating stored
    patches or moving these annotations to the artifact's own `meta` column —
    both larger than the engine.
    """
    return [{"op": "add", "path": f"/meta/{key}", "value": value}]


def validate_ops(ops: Any) -> Any:
    """Check a whole patch's shape, and refuse one that names no operation.

    An empty list is well-formed JSON and still a request that changes
    nothing. Accepting it produced the same dishonest audit entry a
    half-formed op did, one level up: the artifact route guarded the engine
    behind ``if ops:`` and went straight to writing a version, and ConfigOps
    looped zero times and marked the patch applied.
    """
    if not isinstance(ops, list):
        raise BadRequestError(
            "patch is not a list of operations",
            detail={"found": type(ops).__name__},
        )
    if not ops:
        raise BadRequestError("patch names no operation", detail={"operations": 0})
    for op in ops:
        validate_op(op)
    return ops


def apply_ops(doc: dict, ops: List[Dict[str, Any]]) -> dict:
    """Apply a list of RFC 6902 operations to a copy of ``doc``."""
    validate_ops(ops)
    working = copy.deepcopy(doc)
    for op in ops:
        apply_op(working, op)
    return working


def deep_merge(base: dict, patch: dict, *, skip_keys: Iterable[str] = ()) -> dict:
    """Recursive merge for the legacy non-ops patch shape.

    Dicts merge; anything else replaces. ``skip_keys`` lets a caller exclude
    envelope keys (config patches carry their ops under "ops").
    """
    skip = set(skip_keys)
    merged = dict(base)
    for key, value in patch.items():
        if key in skip:
            continue
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merged[key] = deep_merge(base[key], value, skip_keys=skip)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def json_equal(left: Any, right: Any) -> bool:
    """RFC 6902 §4.6 equality: JSON values, not Python objects.

    Python makes `True == 1` and `False == 0`, and carries that through lists
    and dicts, so `test` passed on a value of a different JSON type. It is
    the one verb whose whole job is guarding the operations behind it, so a
    generous comparison does not merely misreport — it lets a mutation run on
    a precondition that was never met.

    JSON has one number type, so `1` and `1.0` are one value. Booleans are
    their own class and equal only booleans. Everything else compares within
    its own type.
    """
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return left == right
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            json_equal(a, b) for a, b in zip(left, right)
        )
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            json_equal(left[k], right[k]) for k in left
        )
    if type(left) is not type(right):
        return False
    return left == right


def _unescape(token: str, path: str) -> str:
    """RFC 6901 §4: `~1` becomes `/`, then `~0` becomes `~`.

    The order is not cosmetic. Decoding `~0` first turns `~01` into `~1` and
    then into `/`, which is a third key again; scanning once, two characters
    at a time, gives the specified order and also catches a `~` that escapes
    nothing.
    """
    if "~" not in token:
        return token
    out: List[str] = []
    i = 0
    while i < len(token):
        if token[i] != "~":
            out.append(token[i])
            i += 1
            continue
        escaped = token[i + 1 : i + 2]
        if escaped not in ("0", "1"):
            raise BadRequestError(
                "malformed JSON Pointer escape",
                detail={"path": path, "at": "~" + escaped},
            )
        out.append("~" if escaped == "0" else "/")
        i += 2
    return "".join(out)


def _segments_or_raise(path: str) -> List[str]:
    """Tokenize an RFC 6901 pointer without changing what it names.

    The previous reader was `strip("/")`, `split("/")` and "drop the empty
    ones", which is four separate rewrites of the caller's address: `/a//b`
    became `/a/b`, `/a/` became `/a`, `a/b` was taken for `/a/b`, and `~1`
    and `~0` were never decoded, so `/a~1b` addressed a key literally spelled
    `a~1b` rather than the key `a/b`. Both spellings can exist in one
    document, so that last pair does not fail — it writes to a real location
    nobody named. This module exists to stop exactly that.
    """
    if path == "":
        # §5: the empty pointer is the whole document. Every verb here edits
        # a member of a container, so there is nothing to serve — but it is
        # refused out loud, because returning quietly reports success.
        raise BadRequestError(
            "patch path addresses the whole document", detail={"path": path}
        )
    if not path.startswith("/"):
        raise BadRequestError(
            "patch path is not a JSON Pointer",
            detail={"path": path, "expected": "a pointer begins with '/'"},
        )
    # `"/"` is one empty token: the member keyed "". Not the document root —
    # that is `""`, handled above.
    return [_unescape(token, path) for token in path[1:].split("/")]


def _pointer(tokens: List[str]) -> str:
    """Render tokens back as a pointer, re-escaped.

    Error details name a location, so a key containing `/` has to go back out
    as `~1` or the detail reads as two tokens and misdirects the reader the
    same way the unescaped parse did.
    """
    return "".join(
        "/" + token.replace("~", "~0").replace("/", "~1") for token in tokens
    )


def _non_container(path: str, at_segments: List[str], found: Any) -> BadRequestError:
    return BadRequestError(
        "patch path traverses a non-container value",
        detail={
            "path": path,
            "at": _pointer(at_segments),
            "found": type(found).__name__,
        },
    )


def _missing(path: str, at_segments: List[str]) -> BadRequestError:
    return BadRequestError(
        "patch path not found",
        detail={"path": path, "at": _pointer(at_segments)},
    )


def _walk_parent(doc: Any, segments: List[str], path: str) -> Any:
    """Walk to the parent of the last segment, creating nothing.

    Every write verb goes through here, because a patch names a location in a
    document that already exists — it does not describe a document to build.
    The creating version of this walk is what let `replace /a/b` invent an
    `a`, report success, and leave the value the caller meant to change
    exactly as it was.

    Only the *parent* has to be there. `add` may still name a member that does
    not exist yet; that is the difference between the verbs, and it is decided
    below rather than here.
    """
    parent = doc
    for depth, seg in enumerate(segments[:-1]):
        if isinstance(parent, list):
            idx = _read_index(seg, path)
            if not 0 <= idx < len(parent):
                raise _missing(path, segments[: depth + 1])
            parent = parent[idx]
        elif isinstance(parent, dict):
            if seg not in parent:
                raise _missing(path, segments[: depth + 1])
            parent = parent[seg]
        else:
            raise _non_container(path, segments[: depth + 1], parent)
    if not isinstance(parent, (dict, list)):
        raise _non_container(path, segments[:-1], parent)
    return parent


def _require_target(parent: Any, key: str, path: str) -> None:
    """RFC 6902 §4.2/§4.3: `remove` and `replace` need the target to be there.

    Absence is the error, not a no-op. A patch that addressed the wrong path
    was otherwise indistinguishable from one that did its job.
    """
    if isinstance(parent, list):
        idx = _read_index(key, path)
        if not 0 <= idx < len(parent):
            raise _missing(path, [key])
    elif key not in parent:
        raise _missing(path, [key])


def _read_index(seg: str, path: str) -> int:
    """RFC 6902 array indices are non-negative digit runs. Python's list[-1]
    would otherwise quietly serve `/xs/-1` on the read paths (move/copy/test)
    while the write path refuses it — the same op legal on one side of a
    round trip and not the other.

    Both messages are direction-neutral, because this is reached from four
    callers and only one of them is reading a source. A malformed index is a
    malformed index wherever it appears; calling it a missing *source path*
    described `replace /xs/nope` as a problem with an operand it does not
    have.
    """
    if _INDEX.match(seg):
        return int(seg)
    # A negative index says something specific about the patch's author, so it
    # keeps its own message: Python would happily serve `/xs/-1` from the end,
    # and "not found" would send the reader looking for a missing element
    # rather than at the index they wrote. Matched strictly for the same
    # reason as the positive form — `"-²".lstrip("-").isdigit()` was true and
    # `int("-²")` was not.
    if _NEGATIVE_INDEX.match(seg):
        raise BadRequestError(
            "negative list index", detail={"path": path, "index": int(seg)}
        )
    raise BadRequestError(
        "list index is not a number", detail={"path": path, "at": seg}
    )


def _read(doc: Any, segments: List[str], path: str) -> Any:
    """Resolve a source path without creating anything along the way."""
    node = doc
    for depth, seg in enumerate(segments):
        if isinstance(node, dict):
            if seg not in node:
                raise BadRequestError(
                    "patch source path not found",
                    detail={"path": path, "at": _pointer(segments[: depth + 1])},
                )
            node = node[seg]
        elif isinstance(node, list):
            try:
                node = node[_read_index(seg, path)]
            except IndexError:
                raise BadRequestError(
                    "patch source path not found",
                    detail={"path": path, "at": _pointer(segments[: depth + 1])},
                )
        else:
            raise _non_container(path, segments[: depth + 1], node)
    return node


def _remove_at(doc: Any, segments: List[str], path: str) -> Any:
    """Remove and return the value at ``segments``.

    `move`'s helper only, now that `remove` requires its target through
    `_require_target` and pops in place. A `from` that is not there is an
    error: the value has to exist to be moved.
    """
    parent = _read(doc, segments[:-1], path) if len(segments) > 1 else doc
    key = segments[-1]
    if isinstance(parent, dict):
        if key not in parent:
            raise BadRequestError(
                "patch source path not found", detail={"path": path}
            )
        return parent.pop(key)
    if isinstance(parent, list):
        try:
            return parent.pop(_read_index(key, path))
        except IndexError:
            raise BadRequestError(
                "patch source path not found", detail={"path": path}
            )
    raise _non_container(path, segments[:-1], parent)


def _set_at(parent: Any, key: str, value: Any, path: str, *, insert: bool) -> None:
    if isinstance(parent, list):
        if key == "-":
            parent.append(value)
            return
        idx = _read_index(key, path)
        # RFC 6902 §4.1: an `add` index may equal the length, which appends;
        # anything beyond it is out of range.
        #
        # What this defends is the address, not the heap. Nothing here pads a
        # list, so without the check `/xs/999999999` on two elements falls
        # through to a single `append` and lands at index 2 — measured. The
        # failure is that it silently *means* `/xs/2`, which is the same
        # wrong-location bug as everything else in this module. A constant
        # ceiling used to sit here instead and got the ordinary case wrong,
        # refusing position 1024 on a list that had one while every read verb
        # served it.
        limit = len(parent) if insert else len(parent) - 1
        if idx > limit:
            raise BadRequestError(
                "list index out of range",
                detail={"path": path, "index": idx, "length": len(parent)},
            )
        if idx < len(parent):
            if insert:
                parent.insert(idx, value)
            else:
                parent[idx] = value
        else:
            parent.append(value)
    else:
        parent[key] = value


def apply_op(doc: dict, op: Dict[str, Any]) -> None:
    """Apply one operation to ``doc`` in place."""
    validate_op(op)
    action = op["op"]
    path = op["path"]
    value = op.get("value")

    segments = _segments_or_raise(path)
    key = segments[-1]

    if action in ("add", "replace"):
        # RFC add inserts into lists; replace overwrites. On dicts both set.
        # Only `replace` requires the target itself: naming a member that is
        # not there yet is what `add` is for.
        parent = _walk_parent(doc, segments, path)
        if action == "replace":
            _require_target(parent, key, path)
        _set_at(parent, key, value, path, insert=(action == "add"))

    elif action == "remove":
        parent = _walk_parent(doc, segments, path)
        _require_target(parent, key, path)
        if isinstance(parent, list):
            parent.pop(_read_index(key, path))
        else:
            parent.pop(key)

    elif action in ("move", "copy"):
        # Present because `validate_op` required it. Defaulting it to "" used
        # to send a verb with no source through the tokenizer, which reported
        # it as a patch that "addresses the whole document" — a true sentence
        # about an operand the caller never wrote.
        from_path = op["from"]
        from_segments = _segments_or_raise(from_path)
        if action == "copy":
            # Reading mutates nothing, so the first thing that can change the
            # document is the write, and a refused write leaves it alone.
            value = copy.deepcopy(_read(doc, from_segments, from_path))
            _set_at(_walk_parent(doc, segments, path), key, value, path, insert=True)
        else:
            # RFC 6902 §4.4: a move is a remove followed by an add, so the add
            # has to be legal in the document the remove *leaves behind*, not
            # the one it started from. Checking the destination first accepts
            # `/a` as a parent while `/a` is the value being taken, and accepts
            # `/xs/3` as an append target on a three-element list that the
            # removal shortens to two. Both then deleted the source on behalf
            # of an operation that went on to fail.
            #
            # So rehearse the whole move on a throwaway copy. Whatever it
            # raises is raised before the real document has been touched, and
            # if it raises nothing the replay below cannot fail either.
            shadow = copy.deepcopy(doc)
            rehearsed = _remove_at(shadow, from_segments, from_path)
            _set_at(_walk_parent(shadow, segments, path), key, rehearsed, path, insert=True)

            moved = _remove_at(doc, from_segments, from_path)
            _set_at(_walk_parent(doc, segments, path), key, moved, path, insert=True)

    elif action == "test":
        current = _read(doc, segments, path)
        if not json_equal(current, value):
            raise BadRequestError(
                "JSON Patch test operation failed",
                detail={"path": path, "expected": value, "actual": current},
            )

    else:
        raise BadRequestError(
            "unknown patch operation", detail={"op": action, "path": path}
        )
