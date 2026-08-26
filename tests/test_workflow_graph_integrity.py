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

Validation sits at two altitudes on purpose. Admission stops new invalid
graphs entering; the engine checks again before it builds `node_map`, because
a row can predate the check or arrive by import, and "repaired silently at
execution" is the defect, not the fallback.
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
