"""Whether a workflow graph matches itself.

Pure and shapeless-input tolerant, because two callers with different error
contracts need the same answer: artifact admission raises
``ArtifactValidationError`` and the engine raises ``BadRequestError``, so this
returns the problems and lets each of them say so in its own words.

The rule is that a workflow executes exactly the graph it declares. The engine
used to do the opposite at every turn, and quietly: a dangling `entrypoint`
fell back to `next(iter(node_map))`, a dangling edge hit
`if not node: continue`, and duplicate ids collapsed in the dict comprehension
that builds `node_map`. Each one runs a different graph from the one the
operator published and reports nothing.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Tuple

# Every field the executor treats as naming a node, measured from
# `WorkflowEngine` rather than taken from the artifact kind schema. `after`
# and `on_error` are not in that schema at all, so reading it would have given
# three of the five and looked complete.
#
#   entrypoint          run(), choosing where to start
#   next                _execute_node, scalar or list, and parallel children
#   branches[].next     switch
#   after               where a parallel fan-in continues
#   on_error            taken instead of `next` when a tool call fails
#
# The cardinality is measured too, and it is not uniform. `next` is the only
# field the executor reads as either a string or a list; it wraps `on_error`
# as a single next-node id and inserts `after` as a single pending id, so a
# list in either position arrives at `node_map.get(...)` as a list. Neither
# field is in the artifact kind schema, so nothing else pins their shape.
_EDGE_FIELDS: Dict[str, bool] = {
    "next": True,      # a list is legal here
    "after": False,
    "on_error": False,
}


def _nodes(schema: Any) -> List[Dict[str, Any]]:
    if not isinstance(schema, dict):
        return []
    nodes = schema.get("nodes")
    if not isinstance(nodes, list):
        return []
    return [n for n in nodes if isinstance(n, dict)]


def _references(node: Dict[str, Any]) -> Iterator[Tuple[str, Any, bool]]:
    """Every node reference this node declares: where, what, and whether a
    list is legal there."""
    node_id = node.get("id")
    for field, list_ok in _EDGE_FIELDS.items():
        value = node.get(field)
        if value is None:
            continue
        yield f"node {node_id!r} {field}", value, list_ok
    branches = node.get("branches")
    if isinstance(branches, list):
        for index, branch in enumerate(branches):
            if isinstance(branch, dict) and branch.get("next") is not None:
                # The switch executor appends `branch["next"]` as one value
                # and does not flatten, so a list here is not fan-out — that
                # is what `parallel` is for (SPEC §9).
                yield f"node {node_id!r} branch {index}", branch["next"], False


def graph_problems(schema: Any) -> List[str]:
    """Every way the declared graph fails to match itself.

    Returns all of them rather than the first: an operator fixing one dangling
    edge per deploy is a bad afternoon.

    Shape is the kind schema's job. Anything malformed enough that there is no
    graph to check yields no problems here, so admission reports the shape
    error it already has rather than a crash on the way to it.
    """
    nodes = _nodes(schema)
    if not nodes:
        return []

    problems: List[str] = []

    declared: List[str] = []
    seen: set[str] = set()
    for index, node in enumerate(nodes):
        node_id = node.get("id")
        if not isinstance(node_id, str) or not node_id:
            # `node_map` is keyed by id and drops the falsy ones, so a node
            # declared with an empty or non-string id disappears — the same
            # silent removal a duplicate causes, and reported the same way
            # rather than skipped.
            problems.append(
                f"node {index} has id {node_id!r}, which cannot name a node"
            )
            continue
        if node_id in seen:
            # The dict comprehension that builds `node_map` keeps the last of
            # these, so the earlier node becomes unreachable without anything
            # being said.
            problems.append(f"duplicate node id {node_id!r}")
        seen.add(node_id)
        declared.append(node_id)

    if not declared:
        return problems

    # Only an *explicitly named* entrypoint has to resolve. Omitting the key
    # and starting at the first node is the engine's own behaviour; refusing
    # that would be a different bug. Present-but-empty is not omitted, though:
    # the operator wrote something, and it names no node.
    if isinstance(schema, dict) and "entrypoint" in schema:
        entrypoint = schema["entrypoint"]
        if not isinstance(entrypoint, str) or entrypoint not in seen:
            problems.append(f"entrypoint {entrypoint!r} is not a declared node")

    for node in nodes:
        for where, value, list_ok in _references(node):
            targets = value if isinstance(value, list) else [value]
            if isinstance(value, list) and not list_ok:
                problems.append(
                    f"{where} is a list, and the executor reads one node id there"
                )
                continue
            for target in targets:
                if not isinstance(target, str) or target not in seen:
                    problems.append(
                        f"{where} names {target!r}, which is not a declared node"
                    )

    return problems
