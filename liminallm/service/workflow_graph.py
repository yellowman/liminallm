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
_EDGE_FIELDS = ("next", "after", "on_error")


def _nodes(schema: Any) -> List[Dict[str, Any]]:
    if not isinstance(schema, dict):
        return []
    nodes = schema.get("nodes")
    if not isinstance(nodes, list):
        return []
    return [n for n in nodes if isinstance(n, dict)]


def _targets(node: Dict[str, Any]) -> Iterator[Tuple[str, Any]]:
    """Every node reference this node declares, with where it was written."""
    node_id = node.get("id")
    for field in _EDGE_FIELDS:
        value = node.get(field)
        # `next` is a string on ordinary nodes and a list on a parallel fan-out,
        # and both spellings reach the executor, so both are edges here.
        for target in value if isinstance(value, list) else [value]:
            if target:
                yield f"node {node_id!r} {field}", target
    branches = node.get("branches")
    if isinstance(branches, list):
        for index, branch in enumerate(branches):
            if isinstance(branch, dict) and branch.get("next"):
                yield f"node {node_id!r} branch {index}", branch["next"]


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
    for node in nodes:
        node_id = node.get("id")
        if not isinstance(node_id, str) or not node_id:
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

    entrypoint = schema.get("entrypoint") if isinstance(schema, dict) else None
    # Only an *explicitly named* entrypoint has to resolve. Omitting it and
    # starting at the first node is the engine's own behaviour; refusing that
    # would be a different bug.
    if entrypoint and entrypoint not in seen:
        problems.append(f"entrypoint {entrypoint!r} is not a declared node")

    for node in nodes:
        for where, target in _targets(node):
            if not isinstance(target, str) or target not in seen:
                problems.append(f"{where} names {target!r}, which is not a declared node")

    return problems
