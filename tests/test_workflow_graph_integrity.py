"""A workflow executes exactly the graph it declares.

Three parts of one rule:

* every node id is unique,
* an explicitly named entrypoint resolves to a declared node, and
* every graph edge resolves to a declared node.

The engine did none of them, and each failure is silent rather than loud. A
dangling `entrypoint` fell back to `next(iter(node_map))` — whatever node
happened to be first — so a published workflow ran from a node the operator
never named. A dangling `next` hit `if not node: continue`, so the
continuation simply vanished. Duplicate ids collapsed in the `node_map` dict
comprehension, one node quietly replacing the other.

**The edge fields were measured, not read off the schema.** The executor
consumes five: `entrypoint`, `next` (scalar or list), `branches[].next`,
`after`, and `on_error`. The last two are not in the artifact kind schema at
all, so a validator written from the schema would have covered three of five
and looked complete.

Measuring them once was not enough, because *which* fields a node reads is
decided by its type. Asking the question globally let a graph declare
`end -> side`, have the validator confirm the edge resolves, and have
execution stop at `end` — a resolved edge that never runs is the same silent
divergence one level in. The node type is that shape again: `_execute_node`
runs anything it does not recognise as a `tool_call`, so a node typed
`"swich"` was admitted and then invoked its tool.

Validation sits at two altitudes on purpose. Admission stops new invalid
graphs entering; the engine checks again before it builds `node_map`, because
a row can predate the check or arrive by import, and "repaired silently at
execution" is the defect, not the fallback.

Execution has two altitudes of its own, and the same rule has to hold on
both. `run_streaming` streams three tools without calling the blocking
executor, so the decisions the blocking path makes *around* a tool call — the
circuit-breaker preflight and the `on_error` handoff — did not happen there
at all.
"""

from __future__ import annotations

import json
import uuid

import pytest

from liminallm.service.errors import BadRequestError
from liminallm.service.runtime import get_runtime
from liminallm.service.workflow_graph import graph_problems

# A graph that is valid under every rule here, and exercises all five edge
# kinds so the positive controls are not narrower than the refusals.
VALID = {
    "kind": "workflow.chat",
    "entrypoint": "start",
    "nodes": [
        {"id": "start", "type": "switch", "branches": [
            {"when": "true", "next": "fan"},
            {"when": "false", "next": "work"},
        ]},
        {"id": "fan", "type": "parallel", "next": ["work", "other"], "after": "join"},
        {"id": "work", "type": "tool_call", "tool": "llm.generic",
         "next": "join", "on_error": "sorry"},
        {"id": "other", "type": "tool_call", "tool": "llm.generic", "next": "join"},
        {"id": "join", "type": "tool_call", "tool": "llm.generic", "next": ["end"]},
        {"id": "sorry", "type": "end"},
        {"id": "end", "type": "end"},
    ],
}


def _graph(**changes):
    schema = json.loads(json.dumps(VALID))
    nodes = {n["id"]: n for n in schema["nodes"]}
    for key, value in changes.items():
        if key == "entrypoint":
            schema["entrypoint"] = value
            continue
        node_id, _, field = key.partition("__")
        nodes[node_id][field] = value
    return schema


class TestTheDeclaredGraphIsChecked:
    def test_the_valid_graph_is_accepted(self):
        """The control for everything below. It uses all five edge kinds, so
        a rule that is too strict fails here rather than passing quietly."""
        assert graph_problems(VALID) == []

    def test_an_absent_entrypoint_is_not_an_error(self):
        """Only an *explicitly named* entrypoint has to resolve. Omitting it
        and starting at the first node is the engine's documented behaviour,
        and refusing it would be a different bug."""
        schema = json.loads(json.dumps(VALID))
        del schema["entrypoint"]
        assert graph_problems(schema) == []

    def test_a_dangling_entrypoint_is_a_problem(self):
        problems = graph_problems(_graph(entrypoint="nowhere"))
        assert problems and any("entrypoint" in p for p in problems), problems

    def test_duplicate_node_ids_are_a_problem(self):
        schema = json.loads(json.dumps(VALID))
        schema["nodes"].append({"id": "work", "type": "end"})
        problems = graph_problems(schema)
        assert problems and any("work" in p for p in problems), problems

    def test_a_dangling_scalar_next_is_a_problem(self):
        assert graph_problems(_graph(work__next="nowhere"))

    def test_a_dangling_member_of_a_list_next_is_a_problem(self):
        """Every member, not just the first. A list whose head resolves is
        the shape a "check the first one" rule passes."""
        assert graph_problems(_graph(join__next=["end", "nowhere"]))

    def test_a_dangling_branch_target_is_a_problem(self):
        assert graph_problems(_graph(start__branches=[
            {"when": "true", "next": "nowhere"},
        ]))

    def test_a_dangling_after_is_a_problem(self):
        """`after` is not in the artifact kind schema, so nothing else looks
        at it. The executor queues it after a parallel fan-in."""
        assert graph_problems(_graph(fan__after="nowhere"))

    def test_a_dangling_on_error_is_a_problem(self):
        """Nor is `on_error`. The executor takes it instead of `next` when a
        tool call fails — which is exactly when a workflow can least afford to
        silently stop."""
        assert graph_problems(_graph(work__on_error="nowhere"))

    def test_every_problem_is_reported_not_just_the_first(self):
        """An operator fixing one dangling edge at a time, one deploy at a
        time, is a bad afternoon."""
        schema = _graph(entrypoint="nowhere", work__next="also_nowhere")
        assert len(graph_problems(schema)) >= 2, schema

    @pytest.mark.parametrize(
        "schema",
        [{}, {"nodes": []}, {"nodes": None}, {"nodes": [{"type": "end"}]}],
        ids=["empty", "no-nodes", "null-nodes", "node-without-id"],
    )
    def test_a_shapeless_graph_does_not_crash_the_validator(self, schema):
        """Shape is the kind schema's job; this must not raise on the way to
        saying so, or admission reports a TypeError instead of a bad request.
        """
        graph_problems(schema)


class TestAdmissionRefusesAnInvalidGraph:
    def test_creating_a_workflow_with_a_dangling_edge_is_refused(
        self, client, admin_headers
    ):
        made = client.post("/v1/artifacts", headers=admin_headers, json={
            "type": "workflow", "name": f"wg-{uuid.uuid4().hex[:6]}",
            "schema": _graph(work__next="nowhere"), "visibility": "private",
        })
        assert made.status_code == 400, made.text

    def test_creating_a_workflow_with_duplicate_ids_is_refused(
        self, client, admin_headers
    ):
        schema = json.loads(json.dumps(VALID))
        schema["nodes"].append({"id": "work", "type": "end"})
        made = client.post("/v1/artifacts", headers=admin_headers, json={
            "type": "workflow", "name": f"wg-{uuid.uuid4().hex[:6]}",
            "schema": schema, "visibility": "private",
        })
        assert made.status_code == 400, made.text

    def test_patching_a_workflow_into_an_invalid_graph_is_refused(
        self, client, admin_headers
    ):
        """The other way in. A valid graph can be patched into a broken one,
        and the patch engine only checks where the write lands."""
        made = client.post("/v1/artifacts", headers=admin_headers, json={
            "type": "workflow", "name": f"wg-{uuid.uuid4().hex[:6]}",
            "schema": VALID, "visibility": "private",
        })
        assert made.status_code in (200, 201), made.text
        artifact = made.json()["data"]["id"]

        resp = client.patch(f"/v1/artifacts/{artifact}", headers=admin_headers, json={
            "patch": [{"op": "replace", "path": "/entrypoint", "value": "nowhere"}],
        })
        assert resp.status_code == 400, resp.text
        assert get_runtime().store.get_artifact(artifact).schema["entrypoint"] == "start"

    def test_a_valid_workflow_still_goes_in(self, client, admin_headers):
        made = client.post("/v1/artifacts", headers=admin_headers, json={
            "type": "workflow", "name": f"wg-{uuid.uuid4().hex[:6]}",
            "schema": VALID, "visibility": "private",
        })
        assert made.status_code in (200, 201), made.text


class TestTheEngineRefusesRatherThanRepairs:
    """The second altitude, and the reason there are two.

    Admission stops new invalid graphs. It does nothing about a row that
    predates the check or arrived by import, and those are exactly the rows
    the engine used to repair on the operator's behalf without saying so.
    """

    @pytest.fixture
    def engine(self):
        from liminallm.service.workflow import WorkflowEngine
        from tests.test_workflow_retry_timeout import (
            MockLLM,
            MockRAG,
            MockRedisCache,
            MockRouter,
            MockStore,
        )

        return WorkflowEngine(MockStore(), MockLLM(), MockRouter(), MockRAG(),
                              cache=MockRedisCache())

    @pytest.mark.parametrize(
        "changes, what",
        [
            ({"entrypoint": "nowhere"}, "ran from a node the graph never named"),
            ({"work__next": "nowhere"}, "dropped the continuation"),
            ({"fan__after": "nowhere"}, "dropped the parallel continuation"),
            ({"work__on_error": "nowhere"}, "dropped the error transition"),
        ],
        ids=["entrypoint", "next", "after", "on_error"],
    )
    @pytest.mark.asyncio
    async def test_an_invalid_persisted_graph_fails_closed(
        self, engine, monkeypatch, changes, what
    ):
        monkeypatch.setattr(engine, "_load_workflow_for",
                            lambda *a, **k: _graph(**changes))
        with pytest.raises(BadRequestError):
            await engine.run("wf", None, "hello", None, user_id="u")

    @pytest.mark.asyncio
    async def test_duplicate_ids_fail_closed(self, engine, monkeypatch):
        schema = json.loads(json.dumps(VALID))
        schema["nodes"].append({"id": "work", "type": "end"})
        monkeypatch.setattr(engine, "_load_workflow_for", lambda *a, **k: schema)
        with pytest.raises(BadRequestError):
            await engine.run("wf", None, "hello", None, user_id="u")

    @pytest.mark.asyncio
    async def test_a_valid_graph_still_runs(self, engine, monkeypatch):
        """The control at this altitude. Refusing every graph would pass all
        five refusals above."""
        monkeypatch.setattr(engine, "_load_workflow_for", lambda *a, **k: VALID)
        out = await engine.run("wf", None, "hello", None, user_id="u")
        assert out.get("status") != "error", out

    @pytest.mark.asyncio
    async def test_the_workflows_this_system_builds_itself_are_not_refused(
        self, engine
    ):
        """No workflow id at all, so the engine synthesises one. If either
        built-in schema failed the new rule, every ordinary turn would.

        The assertion is that the graph is not *refused*, not that the turn
        succeeds: under these mocks the default workflow's node returns an
        error either way, measured on the unwired code as well. A graph
        problem raises; a tool problem comes back in the result. Asserting on
        the result would have made this a witness for the mock harness.
        """
        out = await engine.run(None, None, "hello", None, user_id="u")
        assert "graph" not in str(out.get("error", "")).lower(), out

    @pytest.mark.asyncio
    async def test_both_built_in_graphs_pass_the_rule_directly(self):
        """The same property without the harness in the way."""
        from liminallm.service.workflow import (
            WorkflowEngine,
            get_default_attachment_workflow_schema,
        )

        assert graph_problems(get_default_attachment_workflow_schema()) == []
        assert graph_problems(WorkflowEngine._default_workflow(None)) == []


class TestAReferenceHasTheShapeTheExecutorReads:
    """Referential integrity is not enough: the *cardinality* has to match too.

    The executor supports a list only for `next`. `after` is inserted as one
    pending node id, and `on_error` is wrapped as one next-node id, so a list
    in either position reaches `node_map.get(...)` as a list rather than a
    node id. Neither field is in the artifact kind schema, so JSON Schema does
    not reject the shape either — measured, `{"after": ["join"]}` passed both
    admission layers with zero problems and then failed at execution.

    This is the second half of the lesson from measuring fields the schema did
    not know about: their cardinality is unpinned for exactly the same reason
    their targets were.
    """

    # Each field on the node type that actually reads it, so this measures
    # cardinality and not "a tool_call has no `after`".
    @pytest.mark.parametrize("where", ["fan__after", "work__on_error"],
                             ids=["after", "on_error"])
    def test_a_list_where_the_executor_reads_one_id_is_a_problem(self, where):
        problems = graph_problems(_graph(**{where: ["join"]}))
        assert problems, f"{where} accepted a list the executor cannot use"

    def test_a_scalar_next_and_a_list_next_are_both_fine(self):
        """`next` is the one field where the executor really does take both,
        so narrowing it would be a different bug."""
        assert graph_problems(_graph(work__next="join")) == []
        assert graph_problems(_graph(work__next=["join", "end"])) == []

    def test_a_list_branch_target_is_a_problem(self):
        """The switch executor appends `branch["next"]` as one value and does
        not flatten. The kind schema advertised string-or-array, which is a
        contradiction the schema now no longer states."""
        assert graph_problems(_graph(start__branches=[
            {"when": "true", "next": ["fan"]},
        ]))

    def test_a_non_string_reference_is_a_problem(self):
        for value in (7, {"id": "join"}, True):
            assert graph_problems(_graph(work__next=value)), value


class TestAnEdgeIsReadByTheNodeTypeThatDeclaresIt:
    """An edge that resolves is not the same as an edge that executes.

    Which fields a node reads is decided by its type, and the validator asked
    the question globally: every node was allowed `next`, `after` and
    `on_error`, and every node was allowed `branches`. So a graph could
    declare an edge, have the validator confirm it resolves, and have
    execution never look at it. Measured before the fix, all four of these
    reported no problems at all:

    ==========================================  ==============================
    declared                                    what executes
    ==========================================  ==============================
    `end` with `next`                           nothing; `end` stops the run
    `switch` with `next`                        only `branches[].next`
    `tool_call` with `after`                    only `next` / `on_error`
    `parallel` with `on_error`                  only `next` / `after`
    ==========================================  ==============================

    The node type itself is the same shape one level up. SPEC §9 names four,
    the kind schema accepted any string, and `_execute_node` treats anything
    it does not recognise as a `tool_call` — so a node typed `"swich"` was
    admitted and then silently invoked its tool. Verified: it was accepted at
    admission and traced `{"node": "x", "status": "ok"}`.
    """

    def _one(self, node):
        """That node plus somewhere for its edges to point."""
        return {"kind": "workflow.chat", "entrypoint": node["id"],
                "nodes": [node, {"id": "side", "type": "end"}]}

    def test_an_end_node_declaring_next_is_a_problem(self):
        """The sharpest one: publish `end -> side`, validation says the edge
        resolves, execution stops at `end` and `side` never runs."""
        problems = graph_problems(
            self._one({"id": "stop", "type": "end", "next": "side"})
        )
        assert problems, "an `end` node advertised a continuation it never takes"

    def test_a_switch_declaring_next_is_a_problem(self):
        problems = graph_problems(self._one({
            "id": "choose", "type": "switch", "next": "side",
            "branches": [{"when": "true", "next": "side"}],
        }))
        assert problems, "a switch advertised an edge outside its branches"

    def test_a_tool_call_declaring_after_is_a_problem(self):
        """`after` is where a *parallel* fan-in continues. On a tool node the
        executor never reads it."""
        problems = graph_problems(self._one({
            "id": "t", "type": "tool_call", "tool": "llm.generic",
            "next": "side", "after": "side",
        }))
        assert problems, "a tool node advertised a parallel fan-in edge"

    def test_a_parallel_declaring_on_error_is_a_problem(self):
        """`on_error` is the tool-failure edge. A parallel node's failure
        handling is `_execute_parallel_nodes`, which never looks at it."""
        problems = graph_problems(self._one({
            "id": "fan", "type": "parallel", "next": ["side"],
            "after": "side", "on_error": "side",
        }))
        assert problems, "a parallel node advertised a tool-failure edge"

    def test_branches_on_a_node_that_is_not_a_switch_is_a_problem(self):
        """Same shape, the other direction: only `switch` reads `branches`."""
        problems = graph_problems(self._one({
            "id": "t", "type": "tool_call", "tool": "llm.generic",
            "next": "side", "branches": [{"when": "true", "next": "side"}],
        }))
        assert problems, "a tool node advertised branches nothing reads"

    def test_an_unknown_node_type_is_a_problem(self):
        """SPEC §9 names four. `_execute_node` recognises `switch`, `parallel`
        and `end`, and runs *everything else* as a tool call — so a typo does
        not fail, it invokes."""
        problems = graph_problems(self._one({
            "id": "x", "type": "swich", "tool": "llm.generic", "next": "side",
        }))
        assert problems and any("swich" in p for p in problems), problems

    @pytest.mark.parametrize("node", [
        {"id": "t", "type": "tool_call", "tool": "llm.generic",
         "next": "side", "on_error": "side"},
        {"id": "t", "type": "tool_call", "tool": "llm.generic", "next": ["side"]},
        {"id": "t", "type": "parallel", "next": ["side"], "after": "side"},
        {"id": "t", "type": "switch", "branches": [{"when": "true", "next": "side"}]},
        {"id": "t", "type": "end"},
    ], ids=["tool", "tool-list", "parallel", "switch", "end"])
    def test_each_type_may_declare_the_edges_it_reads(self, node):
        """The control. A table that refuses everything passes every refusal
        above, so each node type gets its own legal shape here."""
        assert graph_problems(self._one(node)) == [], node

    def test_a_node_with_no_type_is_read_as_the_executor_reads_it(self):
        """`_execute_node` defaults a missing `type` to `tool_call`, so the
        validator does too. Requiring the key is admission's job; this
        altitude exists to agree with execution, not to be stricter than it."""
        assert graph_problems(self._one(
            {"id": "t", "tool": "llm.generic", "next": "side"}
        )) == []

    def test_the_kind_schema_itself_names_the_four_node_types(self):
        """The enum, on its own.

        `graph_problems` refuses the same graph a moment after JSON Schema
        does, so an end-to-end admission test cannot tell which layer said
        no — measured, reverting the enum alone still returned 400. The kind
        schema is the published contract that external tooling reads and that
        SPEC §9 writes as an enum, so it gets a witness of its own.
        """
        from jsonschema import Draft202012Validator

        from liminallm.service.artifact_validation import _ARTIFACT_SCHEMAS

        validator = Draft202012Validator(_ARTIFACT_SCHEMAS["workflow"])
        bad = self._one({"id": "x", "type": "swich", "tool": "llm.generic"})
        assert list(validator.iter_errors(bad)), "the kind schema accepted 'swich'"
        for good in ("tool_call", "switch", "parallel", "end"):
            node = {"id": "x", "type": good, "tool": "llm.generic"}
            if good == "switch":
                node["branches"] = [{"when": "true", "next": "side"}]
            if good == "parallel":
                node["next"] = ["side"]
            assert not list(validator.iter_errors(self._one(node))), good

    def test_admission_refuses_an_unknown_node_type(self, client, admin_headers):
        """SPEC §9's schema sketch already writes this as an enum. The kind
        schema said `{"type": "string"}`."""
        schema = self._one({"id": "x", "type": "swich", "tool": "llm.generic",
                            "next": "side"})
        made = client.post("/v1/artifacts", headers=admin_headers, json={
            "type": "workflow", "name": f"wg-{uuid.uuid4().hex[:6]}",
            "schema": schema, "visibility": "private",
        })
        assert made.status_code == 400, made.text

    def test_admission_refuses_an_end_node_that_declares_next(
        self, client, admin_headers
    ):
        schema = self._one({"id": "stop", "type": "end", "next": "side"})
        made = client.post("/v1/artifacts", headers=admin_headers, json={
            "type": "workflow", "name": f"wg-{uuid.uuid4().hex[:6]}",
            "schema": schema, "visibility": "private",
        })
        assert made.status_code == 400, made.text


class TestTheEngineRefusesGraphsItsSchemaWouldNowRefuse:
    """The runtime altitude for the node-semantics rule.

    Schema tests alone would be the wrong evidence: the whole reason for a
    second altitude is rows that never passed today's schema — written before
    the enum existed, or imported. Those reach `run` directly.
    """

    @pytest.fixture
    def engine(self):
        from liminallm.service.workflow import WorkflowEngine
        from tests.test_workflow_retry_timeout import (
            MockLLM,
            MockRAG,
            MockRedisCache,
            MockRouter,
            MockStore,
        )

        return WorkflowEngine(MockStore(), MockLLM(), MockRouter(), MockRAG(),
                              cache=MockRedisCache())

    @pytest.mark.parametrize("node", [
        {"id": "x", "type": "swich", "tool": "llm.generic", "next": "side"},
        {"id": "x", "type": "end", "next": "side"},
    ], ids=["unknown-type", "ignored-field"])
    @pytest.mark.asyncio
    async def test_a_persisted_row_the_schema_would_refuse_fails_closed(
        self, engine, monkeypatch, node
    ):
        schema = {"kind": "workflow.chat", "entrypoint": node["id"],
                  "nodes": [node, {"id": "side", "type": "end"}]}
        monkeypatch.setattr(engine, "_load_workflow_for", lambda *a, **k: schema)
        with pytest.raises(BadRequestError):
            await engine.run("wf", None, "hello", None, user_id="u")

    @pytest.mark.asyncio
    async def test_streaming_refuses_them_too(self, engine, monkeypatch):
        schema = {"kind": "workflow.chat", "entrypoint": "x", "nodes": [
            {"id": "x", "type": "swich", "tool": "llm.generic", "next": "side"},
            {"id": "side", "type": "end"},
        ]}
        monkeypatch.setattr(engine, "_load_workflow_for", lambda *a, **k: schema)
        events = [e async for e in engine.run_streaming(
            "wf", None, "hello", None, user_id="u")]
        assert events[0].get("event") == "error", events[:3]
        assert events[0]["data"]["code"] == "validation_error", events[0]


class TestANodeIdIsUsableAsAnId:
    """`node_map` is keyed by id and drops falsy ones, so a declared node with
    an empty id disappears — the same silent-removal shape as a duplicate."""

    def test_an_empty_node_id_is_a_problem(self):
        schema = json.loads(json.dumps(VALID))
        schema["nodes"].append({"id": "", "type": "end"})
        assert graph_problems(schema)

    def test_a_non_string_node_id_is_a_problem(self):
        schema = json.loads(json.dumps(VALID))
        schema["nodes"].append({"id": 7, "type": "end"})
        assert graph_problems(schema)

    def test_an_explicitly_empty_entrypoint_is_a_problem(self):
        """Different from omitting it. Omitted means "start at the first
        node"; written as empty means the operator named something and it is
        not a node."""
        assert graph_problems(_graph(entrypoint=""))

    def test_admission_refuses_an_empty_node_id(self, client, admin_headers):
        schema = json.loads(json.dumps(VALID))
        schema["nodes"].append({"id": "", "type": "end"})
        made = client.post("/v1/artifacts", headers=admin_headers, json={
            "type": "workflow", "name": f"wg-{uuid.uuid4().hex[:6]}",
            "schema": schema, "visibility": "private",
        })
        assert made.status_code == 400, made.text


class TestStreamingRefusesTheSameGraphs:
    """The batch/streaming altitude split.

    `run_streaming` is a separate graph execution path with its own copy of
    the repair semantics: the same entrypoint fallback and the same
    `if not node: continue`. Blocking chat fails closed on an invalid row
    while streaming chat silently runs a different graph — which is the exact
    row this tranche exists to protect.
    """

    @pytest.fixture
    def engine(self):
        from liminallm.service.workflow import WorkflowEngine
        from tests.test_workflow_retry_timeout import (
            MockLLM,
            MockRAG,
            MockRedisCache,
            MockRouter,
            MockStore,
        )

        return WorkflowEngine(MockStore(), MockLLM(), MockRouter(), MockRAG(),
                              cache=MockRedisCache())

    @pytest.mark.parametrize(
        "changes",
        [{"entrypoint": "nowhere"}, {"work__next": "nowhere"},
         {"fan__after": "nowhere"}, {"work__on_error": "nowhere"}],
        ids=["entrypoint", "next", "after", "on_error"],
    )
    @pytest.mark.asyncio
    async def test_an_invalid_graph_is_refused_before_anything_is_emitted(
        self, engine, monkeypatch, changes
    ):
        monkeypatch.setattr(engine, "_load_workflow_for",
                            lambda *a, **k: _graph(**changes))
        events = [e async for e in engine.run_streaming(
            "wf", None, "hello", None, user_id="u")]

        assert events, "the stream produced nothing at all"
        first = events[0]
        assert first.get("event") == "error", (
            f"the graph was executed before it was checked: {events[:3]}"
        )
        assert first["data"]["code"] == "validation_error", first
        # Before any token, trace or node execution — the point of failing
        # closed is that nothing downstream ever saw the repaired graph.
        assert not any(e.get("event") in {"token", "trace"} for e in events), events

    @pytest.mark.asyncio
    async def test_streaming_refuses_duplicate_ids(self, engine, monkeypatch):
        schema = json.loads(json.dumps(VALID))
        schema["nodes"].append({"id": "work", "type": "end"})
        monkeypatch.setattr(engine, "_load_workflow_for", lambda *a, **k: schema)
        events = [e async for e in engine.run_streaming(
            "wf", None, "hello", None, user_id="u")]
        assert events[0].get("event") == "error", events[:3]

    @pytest.mark.asyncio
    async def test_streaming_still_runs_a_valid_graph(self, engine, monkeypatch):
        """The control at this altitude."""
        monkeypatch.setattr(engine, "_load_workflow_for", lambda *a, **k: VALID)
        events = [e async for e in engine.run_streaming(
            "wf", None, "hello", None, user_id="u")]
        assert events, "the stream produced nothing at all"
        codes = [e["data"].get("code") for e in events if e.get("event") == "error"]
        assert "validation_error" not in codes, events[:3]


class TestAFailedToolTakesItsErrorEdge:
    """`on_error` is the edge a tool node takes when the call fails, and the
    circuit-breaker path did not take it.

    The ordinary tool tail swaps `next` for `on_error` on an error result. The
    circuit-open branch builds its own error result, reads `next`, and returns
    before reaching that swap — so an open breaker sends the turn down the
    *success* path, into nodes that assume outputs the failed node never
    produced.

    This is the same class the rest of this file is about, one level in: the
    declared graph says `tool -> recover` on failure and the runtime does
    `tool -> normal`. The graph validator cannot see it, because the graph is
    valid; what was wrong is which edge execution chose.
    """

    BREAKER = {
        "kind": "workflow.chat",
        "entrypoint": "tool",
        "nodes": [
            {"id": "tool", "type": "tool_call", "tool": "llm.generic",
             "next": "normal", "on_error": "recover"},
            {"id": "normal", "type": "end"},
            {"id": "recover", "type": "end"},
        ],
    }

    @pytest.fixture
    def engine(self):
        from liminallm.service.workflow import WorkflowEngine
        from tests.test_workflow_retry_timeout import (
            MockLLM,
            MockRAG,
            MockRedisCache,
            MockRouter,
            MockStore,
        )

        return WorkflowEngine(MockStore(), MockLLM(), MockRouter(), MockRAG(),
                              cache=MockRedisCache())

    @pytest.mark.asyncio
    async def test_an_open_circuit_takes_on_error_not_next(
        self, engine, monkeypatch
    ):
        async def open_breaker(tool_name, *, tenant_id=None):
            return True, None

        monkeypatch.setattr(engine.cache, "check_circuit_breaker", open_breaker)
        monkeypatch.setattr(engine, "_load_workflow_for",
                            lambda *a, **k: self.BREAKER)

        out = await engine.run("wf", None, "hello", None, user_id="u")
        ran = [entry.get("node") for entry in out.get("workflow_trace") or []]

        assert "recover" in ran, f"the declared error edge was not taken: {ran}"
        assert "normal" not in ran, (
            f"an open breaker took the success edge into nodes that assume "
            f"outputs the failed node never produced: {ran}"
        )

    @pytest.mark.asyncio
    async def test_a_closed_circuit_still_takes_next(self, engine, monkeypatch):
        """The control. Routing every tool node to `on_error` would pass the
        witness above and break every successful turn."""
        monkeypatch.setattr(engine, "_load_workflow_for",
                            lambda *a, **k: self.BREAKER)
        out = await engine.run("wf", None, "hello", None, user_id="u")
        ran = [entry.get("node") for entry in out.get("workflow_trace") or []]
        assert "normal" in ran, f"a successful tool call took the error edge: {ran}"
        assert "recover" not in ran, ran

    @pytest.mark.asyncio
    async def test_an_ordinary_tool_failure_also_takes_on_error(
        self, engine, monkeypatch
    ):
        """The other caller of the chooser, which had no witness of its own.

        Found by a mutation rather than by review: removing `on_error` from
        the chooser entirely killed only the circuit-open witness, which meant
        the primary path — a tool that simply fails — was resting on the
        breaker case to notice. An unknown tool name is an ordinary error
        result, and tool names are not graph-validated yet, so this reaches
        the failure tail rather than the breaker branch.
        """
        schema = json.loads(json.dumps(self.BREAKER))
        schema["nodes"][0]["tool"] = "no.such.tool.v1"
        monkeypatch.setattr(engine, "_load_workflow_for", lambda *a, **k: schema)

        out = await engine.run("wf", None, "hello", None, user_id="u")
        ran = [entry.get("node") for entry in out.get("workflow_trace") or []]

        assert "recover" in ran, f"a failing tool did not take its error edge: {ran}"
        assert "normal" not in ran, ran


class TestAStreamedToolObeysTheSameControlPlane:
    """The same rule, on the path that produces tokens.

    `run_streaming` does not call `_execute_node_with_retry` for the three
    tools it streams — `llm.generic`, `llm.generic_chat_v1`, `agent.files_v1`
    — it enters `_stream_llm_node` directly. Both of the decisions the
    blocking path makes around a tool call therefore did not happen here:

    * the circuit-breaker preflight lives in `_execute_node`, so an open
      breaker did not stop a streamed LLM call at all, and
    * the continuation read `node["next"]` directly, so a graph declaring
      `tool -> recover` on failure ended the turn with an error event and
      never ran `recover`.

    Measured before the fix: with the breaker forced open, `generate_stream`
    was still called and the run traced `['tool', 'normal']` with
    `status: ok`. The same graph on the blocking path traced `recover`.

    Token production stays streaming-specific. What is shared is the control
    plane around it, because a second copy of a decision is how the first one
    went wrong.
    """

    BREAKER = TestAFailedToolTakesItsErrorEdge.BREAKER

    @pytest.fixture
    def engine(self):
        from liminallm.service.workflow import WorkflowEngine
        from tests.test_workflow_retry_timeout import (
            MockLLM,
            MockRAG,
            MockRedisCache,
            MockRouter,
            MockStore,
        )

        return WorkflowEngine(MockStore(), MockLLM(), MockRouter(), MockRAG(),
                              cache=MockRedisCache())

    @staticmethod
    def _stream(engine, monkeypatch, *, schema, opens_breaker=False, raises=False):
        """Install a streaming LLM and return the list its calls land in."""
        calls: list = []

        def generate_stream(*args, **kwargs):
            calls.append("generate_stream")
            if raises:
                raise RuntimeError("backend down")
            return iter([
                {"event": "token", "data": "hi"},
                {"event": "message_done",
                 "data": {"content": "hi", "usage": {}}},
            ])

        async def open_breaker(tool_name, *, tenant_id=None):
            return True, None

        monkeypatch.setattr(engine.llm, "generate_stream", generate_stream,
                            raising=False)
        if opens_breaker:
            monkeypatch.setattr(engine.cache, "check_circuit_breaker", open_breaker)
        monkeypatch.setattr(engine, "_load_workflow_for", lambda *a, **k: schema)
        return calls

    @staticmethod
    def _nodes_run(events):
        return [
            e["data"]["workflow_trace"].get("node")
            for e in events
            if e.get("event") == "trace" and "workflow_trace" in (e.get("data") or {})
        ]

    @pytest.mark.asyncio
    async def test_an_open_circuit_never_starts_the_stream(
        self, engine, monkeypatch
    ):
        """The preflight half, on its own so a mutation can reach it alone.

        An open breaker means "stop calling this tool". Streaming called it
        anyway, which is the failure the breaker exists to prevent — and it
        did so for the one tool every ordinary chat turn uses.
        """
        calls = self._stream(engine, monkeypatch, schema=self.BREAKER,
                             opens_breaker=True)
        [e async for e in engine.run_streaming("wf", None, "hi", None, user_id="u")]
        assert calls == [], (
            "an open breaker did not stop the streamed call: the tool ran anyway"
        )

    @pytest.mark.asyncio
    async def test_an_open_circuit_takes_on_error_not_next(
        self, engine, monkeypatch
    ):
        """The handoff half, for a failure the breaker produced."""
        self._stream(engine, monkeypatch, schema=self.BREAKER, opens_breaker=True)
        events = [e async for e in engine.run_streaming(
            "wf", None, "hi", None, user_id="u")]
        ran = self._nodes_run(events)
        assert "recover" in ran, f"the declared error edge was not taken: {ran}"
        assert "normal" not in ran, (
            f"a failed streamed node took the success edge: {ran}"
        )

    @pytest.mark.asyncio
    async def test_a_stream_that_fails_before_the_first_token_takes_on_error(
        self, engine, monkeypatch
    ):
        """The handoff half, for a failure the backend produced.

        Before the first token on purpose: what recovery means once partial
        output has already reached the client is a separate question, and
        this tranche does not answer it.
        """
        self._stream(engine, monkeypatch, schema=self.BREAKER, raises=True)
        events = [e async for e in engine.run_streaming(
            "wf", None, "hi", None, user_id="u")]
        ran = self._nodes_run(events)
        assert "recover" in ran, (
            f"a stream that failed before producing anything ended the turn "
            f"instead of taking its declared error edge: {ran}"
        )
        assert "normal" not in ran, ran

    @pytest.mark.asyncio
    async def test_a_streamed_tool_that_succeeds_still_takes_next(
        self, engine, monkeypatch
    ):
        """The control. Routing every streamed node to `on_error` would pass
        both witnesses above and break every successful turn."""
        calls = self._stream(engine, monkeypatch, schema=self.BREAKER)
        events = [e async for e in engine.run_streaming(
            "wf", None, "hi", None, user_id="u")]
        ran = self._nodes_run(events)
        assert calls == ["generate_stream"], calls
        assert "normal" in ran, f"a successful stream took the error edge: {ran}"
        assert "recover" not in ran, ran
        assert any(e.get("event") == "token" for e in events), (
            "tokens stopped reaching the client"
        )

    @pytest.mark.asyncio
    async def test_a_streamed_failure_with_no_error_edge_still_stops_the_stream(
        self, engine, monkeypatch
    ):
        """The other control, and the reason the fix is not "always call
        `_successors`".

        With no `on_error` declared, the chooser falls through to `next` — so
        handing every failure to it would send a failed node down the
        *success* path, into nodes that assume outputs it never produced.
        A graph that names nowhere to go on failure ends where it always did.
        """
        schema = json.loads(json.dumps(self.BREAKER))
        del schema["nodes"][0]["on_error"]
        self._stream(engine, monkeypatch, schema=schema, raises=True)
        events = [e async for e in engine.run_streaming(
            "wf", None, "hi", None, user_id="u")]
        assert any(e.get("event") == "error" for e in events), events
        assert "normal" not in self._nodes_run(events), self._nodes_run(events)
