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
from typing import Any, Dict, Iterable, List

from liminallm.service.errors import BadRequestError

# A patch may grow a list (add at index beyond the end), but an index like
# 10**9 would allocate that many placeholder entries. Bound it.
MAX_LIST_EXTENSION = 1024


def apply_ops(doc: dict, ops: List[Dict[str, Any]]) -> dict:
    """Apply a list of RFC 6902 operations to a copy of ``doc``."""
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


def _split(path: str) -> List[str]:
    return [seg for seg in (path or "").strip("/").split("/") if seg]


def _segments_or_raise(path: str) -> List[str]:
    segments = _split(path)
    if not segments:
        # "/" addresses the whole document. Treating it as a key wrote an
        # empty-string entry into the schema and reported success.
        raise BadRequestError(
            "patch path addresses the document root", detail={"path": path}
        )
    return segments


def _non_container(path: str, at_segments: List[str], found: Any) -> BadRequestError:
    return BadRequestError(
        "patch path traverses a non-container value",
        detail={
            "path": path,
            "at": "/" + "/".join(at_segments),
            "found": type(found).__name__,
        },
    )


def _ensure_list_capacity(idx: int, path: str) -> None:
    if idx < 0:
        raise BadRequestError(
            "negative list index", detail={"path": path, "index": idx}
        )
    if idx >= MAX_LIST_EXTENSION:
        raise BadRequestError(
            "list index too large",
            detail={"path": path, "index": idx, "max_index": MAX_LIST_EXTENSION - 1},
        )


def _missing(path: str, at_segments: List[str]) -> BadRequestError:
    return BadRequestError(
        "patch path not found",
        detail={"path": path, "at": "/" + "/".join(at_segments)},
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
    round trip and not the other."""
    if not seg.isdigit():
        # A negative index says something specific about the patch's author,
        # so it keeps its own message: Python would happily serve `/xs/-1`
        # from the end, and "not found" would send the reader looking for a
        # missing element rather than at the index they wrote.
        if seg.lstrip("-").isdigit():
            raise BadRequestError(
                "negative list index", detail={"path": path, "index": int(seg)}
            )
        raise BadRequestError(
            "patch source path not found", detail={"path": path, "at": seg}
        )
    return int(seg)


def _read(doc: Any, segments: List[str], path: str) -> Any:
    """Resolve a source path without creating anything along the way."""
    node = doc
    for depth, seg in enumerate(segments):
        if isinstance(node, dict):
            if seg not in node:
                raise BadRequestError(
                    "patch source path not found",
                    detail={"path": path, "at": "/" + "/".join(segments[: depth + 1])},
                )
            node = node[seg]
        elif isinstance(node, list):
            try:
                node = node[_read_index(seg, path)]
            except IndexError:
                raise BadRequestError(
                    "patch source path not found",
                    detail={"path": path, "at": "/" + "/".join(segments[: depth + 1])},
                )
        else:
            raise _non_container(path, segments[: depth + 1], node)
    return node


def _remove_at(doc: Any, segments: List[str], path: str) -> Any:
    """Remove and return the value at ``segments``. Missing is an error for
    move (the value must exist to be moved); remove tolerates it."""
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
        try:
            idx = int(key)
        except ValueError:
            raise BadRequestError(
                "list index is not a number", detail={"path": path, "at": key}
            )
        _ensure_list_capacity(idx, path)
        # RFC 6902 §4.1: an `add` index may equal the length, which appends;
        # anything beyond it is out of range. Silently appending instead let
        # `/xs/5` on a two-element array land at position 2 — a write to a
        # place the caller did not name.
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
    action = (op or {}).get("op")
    path = (op or {}).get("path", "")
    value = op.get("value") if op else None
    if not action or not path:
        return

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
        from_path = op.get("from", "")
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
        if current != value:
            raise BadRequestError(
                "JSON Patch test operation failed",
                detail={"path": path, "expected": value, "actual": current},
            )

    else:
        raise BadRequestError(
            "unknown patch operation", detail={"op": action, "path": path}
        )
