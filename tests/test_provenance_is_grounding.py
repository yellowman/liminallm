"""Provenance describes what grounded the answer, not what retrieval found.

Retrieval returns a shortlist; prompt budgeting decides how much of it the
model actually reads. Registering the whole shortlist would make a chunk the
budget dropped an eligible citation target - a citation to something the
model never saw. These pin the narrower set at each automatic path.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

from liminallm.service.provenance import SourceRegistry
from liminallm.service.runtime import get_runtime

#: Three chunks worth of distinct text, so budgeting has something to cut.
PARAGRAPHS = [
    f"Passage {n}. " + (f"turbine blade inspection detail {n} " * 60)
    for n in range(3)
]


def _grounded_context(store):
    """A real user and a real context holding three separable chunks."""
    user = store.create_user(email=f"prov_{uuid.uuid4().hex[:8]}@t.local")
    ctx = store.upsert_context(
        name=f"prov-{uuid.uuid4().hex[:6]}",
        description="three chunks",
        owner_user_id=user.id,
    )
    for text in PARAGRAPHS:
        written = get_runtime().rag.ingest_text(ctx.id, text)
        assert written > 0, "the fixture failed to index a paragraph"
    return user.id, ctx.id


def _budget_for(engine, monkeypatch, chunks, keep):
    """Force budgeting to retain exactly `keep` of the retrieved snippets."""
    counted = [len(c.content) // 4 for c in chunks]
    monkeypatch.setattr(
        engine, "prompt_budget", lambda: sum(counted[:keep]) + 8
    )


@pytest.fixture
def engine():
    return get_runtime().workflow


class TestOnlyGroundingThatReachedTheModelIsEligible:
    def test_batch_llm_generic_registers_what_survived_the_budget(
        self, store, engine, monkeypatch
    ):
        user_id, ctx_id = _grounded_context(store)
        registry = SourceRegistry()
        chunks = engine.rag.retrieve(
            [ctx_id], "turbine blade inspection", user_id=user_id, tenant_id=None
        )
        assert len(chunks) >= 2, "fixture must retrieve more than one chunk"
        _budget_for(engine, monkeypatch, chunks, keep=1)

        engine._tool_llm_generic(
            {"message": "turbine blade inspection"},
            [],
            [],
            ctx_id,
            None,
            "turbine blade inspection",
            user_id,
            None,
            source_registry=registry,
        )

        texts = {e.text for e in registry.evidence}
        assert texts == {chunks[0].content}, (
            "only the chunk the budget kept may become eligible provenance"
        )

    def test_the_binding_reaches_the_caller(self, store, engine, monkeypatch):
        """A registered source that nothing points at cannot support an
        answer: the relation is what says this evidence was actually used."""
        user_id, ctx_id = _grounded_context(store)
        registry = SourceRegistry()
        chunks = engine.rag.retrieve(
            [ctx_id], "turbine blade inspection", user_id=user_id, tenant_id=None
        )
        _budget_for(engine, monkeypatch, chunks, keep=1)

        # The sink, not the return value: a tool's declared `output_schema`
        # may forbid additional properties, so the bindings are the parent's
        # own record rather than part of what the tool produced.
        bindings = []
        engine._tool_llm_generic(
            {"message": "turbine blade inspection"},
            [],
            [],
            ctx_id,
            None,
            "turbine blade inspection",
            user_id,
            None,
            source_registry=registry,
            bindings_sink=bindings,
        )
        assert bindings, "the parent's sink was never filled"
        assert all(
            set(b) == {"context_id", "source_id", "evidence_id"} for b in bindings
        )
        assert {b["context_id"] for b in bindings} == {ctx_id}
        assert len(bindings) == 1

    def test_rag_answer_registers_everything_it_sends(
        self, store, engine, monkeypatch
    ):
        """The exception, and the reason it is one: `rag.answer` passes every
        retrieved chunk to the model, so every one of them did ground it."""
        user_id, ctx_id = _grounded_context(store)
        registry = SourceRegistry()
        chunks = engine.rag.retrieve(
            [ctx_id], "turbine blade inspection", user_id=user_id, tenant_id=None
        )
        engine._tool_rag_answer(
            {"question": "turbine blade inspection", "context_id": ctx_id},
            [],
            [],
            ctx_id,
            None,
            "turbine blade inspection",
            user_id,
            None,
            source_registry=registry,
        )
        assert {e.text for e in registry.evidence} == {c.content for c in chunks}


class TestTheStreamedPathAgrees:
    """Same rule on the streamed transport, which budgets in its own copy of
    the assembly and so can drift from the blocking one."""

    @pytest.mark.asyncio
    async def test_streaming_registers_what_survived_the_budget(
        self, store, monkeypatch
    ):
        import uuid as _uuid

        from liminallm.service.workflow import WorkflowEngine
        from liminallm.storage.models import KnowledgeChunk

        rt = get_runtime()
        user = store.create_user(email=f"sp_{_uuid.uuid4().hex[:8]}@t.local")
        schema = {
            "kind": "workflow.chat",
            "entrypoint": "call",
            "nodes": [
                {"id": "call", "type": "tool_call", "tool": "llm.generic",
                 "next": "fin"},
                {"id": "fin", "type": "end"},
            ],
        }
        artifact = store.create_artifact(
            "workflow", f"spwf-{_uuid.uuid4().hex[:6]}", schema,
            owner_user_id=user.id, visibility="private",
        )

        chunks = [
            KnowledgeChunk(
                context_id="ctx_s", fs_path=f"notes/p{n}.md",
                content=f"PASSAGE-{n} " + ("filler " * 40),
                embedding=[], chunk_index=n,
            )
            for n in range(3)
        ]

        class Grounded:
            def retrieve(self, ctx_ids, query, **kwargs):
                return list(chunks)

        engine = WorkflowEngine(store, rt.llm, rt.router, Grounded(), cache=rt.cache)
        engine.llm.generate_stream = lambda *a, **k: iter([
            {"event": "token", "data": "hi"},
            {"event": "message_done", "data": {"content": "hi", "usage": {}}},
        ])
        # Room for the first passage only.
        monkeypatch.setattr(
            engine, "prompt_budget", lambda: len(chunks[0].content) // 4 + 8
        )

        events = [
            e async for e in engine.run_streaming(
                artifact.id, None, "hi", "ctx_s",
                user_id=user.id, tenant_id=None,
            )
        ]
        done = [e for e in events if e.get("event") == "message_done"]
        assert done, events[-2:]
        snippets = done[-1]["data"].get("context_snippets") or []
        bindings = done[-1]["data"].get("provenance_bindings") or []
        assert len(snippets) == 1, f"fixture did not prune: {len(snippets)}"
        assert len(bindings) == 1, (
            "the streamed turn made a pruned chunk eligible provenance: "
            f"{bindings}"
        )


class TestTheAgentPathBindsWhatTheParentPlaced:
    """`_explicit_context_grounding` retrieves; `_build_agent_context` decides
    how much of it fits. The bindings must describe the second, and must be
    attached by the parent rather than round-tripped through the worker."""

    def test_only_the_grounding_that_fit_is_bound(self, store, engine, monkeypatch):
        user_id, ctx_id = _grounded_context(store)
        registry = SourceRegistry()
        _, grounding, chunks = engine._explicit_context_grounding(
            "turbine blade inspection", ctx_id, user_id=user_id, tenant_id=None
        )
        assert len(chunks) >= 2, "fixture must retrieve more than one chunk"
        _budget_for(engine, monkeypatch, chunks, keep=1)

        _, _, context, _ = engine._plan_invocation(
            "agent.files_v1",
            {"message": "turbine blade inspection"},
            adapters=[],
            history=[],
            context_id=ctx_id,
            conversation_id=None,
            user_message="turbine blade inspection",
            user_id=user_id,
            tenant_id=None,
            source_registry=registry,
        )
        assert len(context.provenance_bindings) == 1, (
            f"bound {len(context.provenance_bindings)} of {len(chunks)} retrieved"
        )
        assert {e.text for e in registry.evidence} == {chunks[0].content}

    def test_the_worker_never_supplies_them(self, store, engine, monkeypatch):
        """A worker that could name what supported the answer could name a
        source it never read."""
        user_id, ctx_id = _grounded_context(store)
        _, _, context, _ = engine._plan_invocation(
            "agent.files_v1",
            {"message": "turbine blade inspection"},
            adapters=[],
            history=[],
            context_id=ctx_id,
            conversation_id=None,
            user_message="turbine blade inspection",
            user_id=user_id,
            tenant_id=None,
            source_registry=SourceRegistry(),
        )
        assert context.provenance_bindings, "the parent computed none"
        # The plan is the only thing the worker reads.
        _, plan, _, _ = engine._plan_invocation(
            "agent.files_v1",
            {"message": "turbine blade inspection"},
            adapters=[],
            history=[],
            context_id=ctx_id,
            conversation_id=None,
            user_message="turbine blade inspection",
            user_id=user_id,
            tenant_id=None,
            source_registry=SourceRegistry(),
        )
        assert "provenance_bindings" not in plan
        assert "source_registry" not in plan


class TestTheBindingsSurviveTheSeamsTheyCross:
    """The parent computes the agent path's bindings before the worker runs
    and attaches them after it returns. Both halves need a witness: the
    computation is not the delivery."""

    @pytest.mark.asyncio
    async def test_the_parent_attaches_them_to_the_agent_result(
        self, store, engine, monkeypatch
    ):
        user_id, ctx_id = _grounded_context(store)
        registry = SourceRegistry()

        # The worker's whole contribution, and it names no provenance.
        def _served(invocation, worker_tool, plan, context, limits, **kw):
            return {"content": "answered", "usage": {}}

        monkeypatch.setattr(engine, "_serve_invocation", _served)
        result = await engine._invoke_tool(
            "agent.files_v1",
            {"message": "turbine blade inspection"},
            [],
            [],
            ctx_id,
            None,
            "turbine blade inspection",
            source_registry=registry,
            user_id=user_id,
            tenant_id=None,
        )
        assert result.get("provenance_bindings"), (
            "the parent computed grounding and then dropped it at the seam"
        )
        assert all(
            set(b) == {"context_id", "source_id", "evidence_id"}
            for b in result["provenance_bindings"]
        )

    @pytest.mark.asyncio
    async def test_a_failed_parallel_child_grounds_nothing(self, engine, monkeypatch):
        """Its retrieval is consulted, not supporting. The registry keeps it;
        the bindings must not."""
        ok = {
            "status": "ok",
            "content": "fine",
            "provenance_bindings": [
                {"context_id": "c1", "source_id": "src_1", "evidence_id": "ev_1"}
            ],
        }
        failed = {
            "status": "error",
            "error": "boom",
            "provenance_bindings": [
                {"context_id": "c2", "source_id": "src_2", "evidence_id": "ev_2"}
            ],
        }

        async def _fake_node(node, **kwargs):
            return (ok if node.get("id") == "good" else failed), []

        monkeypatch.setattr(engine, "_execute_node_with_retry", _fake_node)
        from liminallm.service.workflow_limits import ExecutionBudget

        result = await engine._execute_parallel_nodes(
            ["good", "bad"],
            {"good": {"id": "good"}, "bad": {"id": "bad"}},
            budget=ExecutionBudget(10),
            user_message="m",
            context_id=None,
            conversation_id=None,
            adapters=[],
            history=[],
            vars_scope={},
            source_registry=SourceRegistry(),
            user_id=None,
            tenant_id=None,
            workflow_start_time=0.0,
            workflow_timeout_ms=60_000,
        )
        assert result.merged_bindings == ok["provenance_bindings"], (
            f"a failed child's grounding leaked: {result.merged_bindings}"
        )

    @pytest.mark.asyncio
    async def test_a_refused_result_carries_no_bindings(
        self, store, engine, monkeypatch
    ):
        """A refusal is not an answer, so nothing grounded it. Attaching
        bindings there would offer citations for a reply the user never got."""
        user_id, ctx_id = _grounded_context(store)

        def _served(invocation, worker_tool, plan, context, limits, **kw):
            return {"content": "answered", "usage": {}}

        monkeypatch.setattr(engine, "_serve_invocation", _served)
        monkeypatch.setattr(
            engine,
            "tool_postflight",
            lambda result, spec, *, tool_name: (
                result,
                {"status": "error", "content": "refused", "error": "validation_error"},
            ),
        )
        result = await engine._invoke_tool(
            "agent.files_v1",
            {"message": "turbine blade inspection"},
            [],
            [],
            ctx_id,
            None,
            "turbine blade inspection",
            source_registry=SourceRegistry(),
            user_id=user_id,
            tenant_id=None,
        )
        assert result["status"] == "error"
        assert "provenance_bindings" not in result


class TestTheTurnReportsWhatGroundedIt:
    """The node knows; the turn is what the API and storage read."""

    def test_the_blocking_turn_result_carries_the_bindings(
        self, store, engine, monkeypatch
    ):
        user_id, ctx_id = _grounded_context(store)
        chunks = engine.rag.retrieve(
            [ctx_id], "turbine blade inspection", user_id=user_id, tenant_id=None
        )
        _budget_for(engine, monkeypatch, chunks, keep=1)
        result = asyncio.run(
            engine.run(None, None, "turbine blade inspection", ctx_id, user_id)
        )
        bindings = result.get("provenance_bindings")
        assert bindings, (
            "the turn recorded grounding and then reported none: "
            f"{sorted(result)}"
        )
        assert {b["context_id"] for b in bindings} == {ctx_id}
        assert len(bindings) == 1, "a pruned chunk reached the turn's provenance"
