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


def _budget_for(engine, monkeypatch, chunks, keep, headroom=8):
    """Force budgeting to retain exactly `keep` of the retrieved snippets.

    `headroom` covers whatever else shares the budget - the agent path
    assembles a system block and budgets the grounding against it.
    """
    counted = [len(c.content) // 4 for c in chunks]
    monkeypatch.setattr(
        engine, "prompt_budget", lambda: sum(counted[:keep]) + headroom
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


class TestWhicheverPromptAnswersOwnsTheBindings:
    """`_plan_invocation` assembles an agent prompt, then may abandon it for
    a plain one. The assembly that produces the answer owns the grounding;
    the abandoned one leaves its retrieval in the registry as consulted."""

    def _plan(self, engine, ctx_id, user_id, registry):
        return engine._plan_invocation(
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

    def test_an_abandoned_agent_plan_binds_nothing(self, store, engine):
        """A backend that cannot call tools answers the ordinary way. The
        agent prompt was built and thrown away, so it grounded nothing - and
        its bindings must not ride along into the prompt that replaced it."""
        user_id, ctx_id = _grounded_context(store)
        registry = SourceRegistry()
        tool, _, context, _ = self._plan(engine, ctx_id, user_id, registry)
        assert tool == "llm.generic", "fixture did not take the fallback"
        assert context.provenance_bindings == []
        assert registry.evidence, "the retrieval is still consulted"

    def test_the_agent_plan_binds_when_it_is_the_answer_path(
        self, store, engine, monkeypatch
    ):
        user_id, ctx_id = _grounded_context(store)
        registry = SourceRegistry()
        chunks = engine.rag.retrieve(
            [ctx_id], "turbine blade inspection", user_id=user_id, tenant_id=None
        )
        assert len(chunks) >= 2, "fixture must retrieve more than one chunk"
        # Headroom for the agent system block, which is budgeted with the
        # grounding rather than ahead of it. Measured: 200 keeps exactly one.
        _budget_for(engine, monkeypatch, chunks, keep=1, headroom=200)
        engine._budget_cache = None
        monkeypatch.setattr(type(engine.llm), "supports_tools", True, raising=False)

        tool, plan, context, _ = self._plan(engine, ctx_id, user_id, registry)
        assert tool == "agent.files_v1", "fixture did not take the agent path"
        assert len(context.provenance_bindings) == 1, (
            f"bound {len(context.provenance_bindings)} of {len(chunks)} retrieved"
        )
        # The plan is the only thing the worker reads.
        assert "provenance_bindings" not in plan
        assert "source_registry" not in plan


class TestTheBindingsSurviveTheSeamsTheyCross:
    """The parent computes the agent path's bindings before the worker runs
    and attaches them after it returns. Both halves need a witness: the
    computation is not the delivery."""

    @pytest.mark.asyncio
    async def test_the_parent_attaches_them_to_the_result(
        self, store, engine
    ):
        """Whichever body ran, the bindings reach the caller through the
        parent rather than through the tool's own validated output."""
        user_id, ctx_id = _grounded_context(store)
        registry = SourceRegistry()
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


class TestAFailedAttemptGroundsNothing:
    """A retry re-retrieves, and the streamed body records its grounding
    before the provider stream begins. So a list shared across attempts lets
    an attempt that died mid-answer contribute to the one that succeeded."""

    @pytest.mark.asyncio
    async def test_only_the_succeeding_attempt_binds(self, store, monkeypatch):
        import uuid as _uuid

        from liminallm.service.workflow import WorkflowEngine
        from liminallm.storage.models import KnowledgeChunk

        rt = get_runtime()
        user = store.create_user(email=f"rt_{_uuid.uuid4().hex[:8]}@t.local")
        artifact = store.create_artifact(
            "workflow", f"rtwf-{_uuid.uuid4().hex[:6]}",
            {
                "kind": "workflow.chat",
                "entrypoint": "call",
                "nodes": [
                    {"id": "call", "type": "tool_call", "tool": "llm.generic",
                     "next": "fin"},
                    {"id": "fin", "type": "end"},
                ],
            },
            owner_user_id=user.id, visibility="private",
        )

        def _chunk(tag):
            return KnowledgeChunk(
                context_id="ctx_r", fs_path=f"notes/{tag}.md",
                content=f"PASSAGE-{tag}", embedding=[], chunk_index=0,
            )

        attempts = {"n": 0}

        class Grounded:
            def retrieve(self, ctx_ids, query, **kwargs):
                # A retrieves on the first attempt, B on the second.
                return [_chunk("A" if attempts["n"] == 0 else "B")]

        engine = WorkflowEngine(store, rt.llm, rt.router, Grounded(), cache=rt.cache)

        def _stream(*a, **k):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise RuntimeError("provider died before completing")
            return iter([
                {"event": "token", "data": "hi"},
                {"event": "message_done", "data": {"content": "hi", "usage": {}}},
            ])

        engine.llm.generate_stream = _stream
        events = [
            e async for e in engine.run_streaming(
                artifact.id, None, "hi", "ctx_r",
                user_id=user.id, tenant_id=None,
            )
        ]
        done = [e for e in events if e.get("event") == "message_done"]
        assert done, (f"attempts={attempts['n']}", [e.get("event") for e in events])
        assert attempts["n"] >= 2, "the fixture did not retry"
        bindings = done[-1]["data"].get("provenance_bindings") or []
        assert len(bindings) == 1, (
            f"the failed attempt's grounding rode along: {bindings}"
        )


class TestTheAnswersOwnSourcesAreTheEligibleOnes:
    """Sequential nodes replace content. A union would let a citation
    validator accept a reference to an earlier node's source for a later
    node's answer."""

    def test_the_last_node_to_speak_owns_the_bindings(
        self, store, engine, monkeypatch
    ):
        import uuid as _uuid

        user = store.create_user(email=f"own_{_uuid.uuid4().hex[:8]}@t.local")
        first = store.upsert_context(
            name=f"a-{_uuid.uuid4().hex[:6]}", description="a",
            owner_user_id=user.id,
        )
        second = store.upsert_context(
            name=f"b-{_uuid.uuid4().hex[:6]}", description="b",
            owner_user_id=user.id,
        )
        engine.rag.ingest_text(
            first.id, "Alpha turbine blade inspection record. " * 60
        )
        engine.rag.ingest_text(
            second.id, "Beta turbine blade inspection record. " * 60
        )

        artifact = store.create_artifact(
            "workflow", f"ownwf-{_uuid.uuid4().hex[:6]}",
            {
                "kind": "workflow.chat",
                "entrypoint": "a",
                "nodes": [
                    {"id": "a", "type": "tool_call", "tool": "llm.generic",
                     "inputs": {"context_id": first.id}, "next": "b"},
                    {"id": "b", "type": "tool_call", "tool": "llm.generic",
                     "inputs": {"context_id": second.id}, "next": "fin"},
                    {"id": "fin", "type": "end"},
                ],
            },
            owner_user_id=user.id, visibility="private",
        )
        result = asyncio.run(
            engine.run(
                artifact.id, None, "turbine blade inspection", None, user.id
            )
        )
        bindings = result.get("provenance_bindings") or []
        assert bindings, "the turn reported no grounding at all"
        assert {b["context_id"] for b in bindings} == {second.id}, (
            "the earlier node's sources are still eligible for the later "
            f"node's answer: {sorted({b['context_id'] for b in bindings})}"
        )


def _stream_workflow(store, user, nodes):
    import uuid as _uuid

    return store.create_artifact(
        "workflow", f"swf-{_uuid.uuid4().hex[:6]}",
        {"kind": "workflow.chat", "entrypoint": nodes[0]["id"], "nodes": nodes},
        owner_user_id=user.id, visibility="private",
    )


def _two_contexts(store, engine, user):
    import uuid as _uuid

    first = store.upsert_context(
        name=f"a-{_uuid.uuid4().hex[:6]}", description="a", owner_user_id=user.id
    )
    second = store.upsert_context(
        name=f"b-{_uuid.uuid4().hex[:6]}", description="b", owner_user_id=user.id
    )
    engine.rag.ingest_text(first.id, "Alpha turbine blade inspection record. " * 60)
    engine.rag.ingest_text(second.id, "Beta turbine blade inspection record. " * 60)
    return first.id, second.id


def _plain_stream(*a, **k):
    return iter([
        {"event": "token", "data": "hi"},
        {"event": "message_done", "data": {"content": "hi", "usage": {}}},
    ])


class TestTheStreamedTurnObeysTheSameOwnership:
    """Every rule the blocking driver follows, on the transport that has its
    own copy of the driver."""

    @pytest.mark.asyncio
    async def test_the_last_streamed_node_owns_the_bindings(self, store, engine):
        import uuid as _uuid

        user = store.create_user(email=f"so_{_uuid.uuid4().hex[:8]}@t.local")
        first, second = _two_contexts(store, engine, user)
        artifact = _stream_workflow(store, user, [
            {"id": "a", "type": "tool_call", "tool": "llm.generic",
             "inputs": {"context_id": first}, "next": "b"},
            {"id": "b", "type": "tool_call", "tool": "llm.generic",
             "inputs": {"context_id": second}, "next": "fin"},
            {"id": "fin", "type": "end"},
        ])
        engine.llm.generate_stream = _plain_stream
        events = [
            e async for e in engine.run_streaming(
                artifact.id, None, "turbine blade inspection", None,
                user_id=user.id, tenant_id=None,
            )
        ]
        done = [e for e in events if e.get("event") == "message_done"]
        assert done, [e.get("event") for e in events]
        bindings = done[-1]["data"].get("provenance_bindings") or []
        assert bindings, "the streamed turn reported no grounding"
        assert {b["context_id"] for b in bindings} == {second}, (
            "an earlier streamed node's sources are still eligible: "
            f"{sorted({b['context_id'] for b in bindings})}"
        )

    @pytest.mark.asyncio
    async def test_an_abandoned_streamed_agent_plan_binds_nothing(
        self, store, engine
    ):
        """The agent assembly is built, then handed to the plain body when
        the backend cannot call tools. Committing both binds the same
        evidence twice - once for a prompt that never ran - so the sink must
        hold exactly what the body that answered put there.
        """
        import uuid as _uuid

        from liminallm.service.invocation import InvocationRegistry

        user = store.create_user(email=f"sa_{_uuid.uuid4().hex[:8]}@t.local")
        ctx = store.upsert_context(
            name=f"sa-{_uuid.uuid4().hex[:6]}", description="c",
            owner_user_id=user.id,
        )
        engine.rag.ingest_text(ctx.id, "Gamma turbine blade inspection record. " * 60)
        engine.llm.generate_stream = _plain_stream
        assert not engine.llm.supports_tools, "fixture must take the fallback"

        registry = SourceRegistry()
        invocation = InvocationRegistry().open(
            _uuid.uuid4().hex, tool="agent.files_v1", user_id=user.id,
            tenant_id=None,
        )
        sink: list = []
        async for _ in engine._stream_agent_files_node(
            {"id": "files", "type": "tool_call", "tool": "agent.files_v1"},
            user_message="turbine blade inspection",
            context_id=ctx.id,
            conversation_id=None,
            adapters=[],
            history=[],
            vars_scope={},
            source_registry=registry,
            bindings_sink=sink,
            user_id=user.id,
            tenant_id=None,
            invocation=invocation,
        ):
            pass
        assert sink, "the fallback body recorded nothing"
        seen = [(b["source_id"], b["evidence_id"]) for b in sink]
        assert len(seen) == len(set(seen)), (
            f"the abandoned agent plan bound the same evidence again: {seen}"
        )

    @pytest.mark.asyncio
    async def test_a_blocking_bodied_streamed_attempt_keeps_its_bindings(
        self, store, engine
    ):
        """A backend that cannot be stopped is run through the blocking body,
        whose result carries the bindings in the result rather than through
        the sink. That branch used to copy everything but them."""
        import uuid as _uuid

        user = store.create_user(email=f"sb_{_uuid.uuid4().hex[:8]}@t.local")
        ctx = store.upsert_context(
            name=f"sb-{_uuid.uuid4().hex[:6]}", description="c",
            owner_user_id=user.id,
        )
        engine.rag.ingest_text(ctx.id, "Delta turbine blade inspection record. " * 60)
        artifact = _stream_workflow(store, user, [
            {"id": "call", "type": "tool_call", "tool": "llm.generic",
             "next": "fin"},
            {"id": "fin", "type": "end"},
        ])
        streamed = []
        engine.llm.generate_stream = lambda *a, **k: (
            streamed.append(1) or _plain_stream()
        )
        engine.llm.backend.supports_stream_cancel = False
        try:
            events = [
                e async for e in engine.run_streaming(
                    artifact.id, None, "turbine blade inspection", ctx.id,
                    user_id=user.id, tenant_id=None,
                )
            ]
        finally:
            del engine.llm.backend.supports_stream_cancel
        assert not streamed, "the fixture streamed after all"
        done = [e for e in events if e.get("event") == "message_done"]
        assert done, [e.get("event") for e in events]
        assert done[-1]["data"].get("provenance_bindings"), (
            "a streamed turn through a non-streamable backend lost its "
            "provenance entirely"
        )


class TestAFailedResultIsNoAuthority:
    """`status="error"` with ordinary valid output is not a refusal - it
    passes postflight. So the producer boundary has to ask whether the tool
    succeeded, not only whether its output validated."""

    @pytest.mark.asyncio
    async def test_a_failed_host_tool_carries_no_bindings(
        self, store, engine, monkeypatch
    ):
        user_id, ctx_id = _grounded_context(store)
        registry = SourceRegistry()
        real = engine._tool_llm_generic

        def _fails(*args, **kwargs):
            result = real(*args, **kwargs)
            # Grounding happened; the tool then failed on its own terms.
            return {**result, "status": "error", "content": "failed"}

        monkeypatch.setattr(engine, "_tool_llm_generic", _fails)
        result = await engine._invoke_tool(
            "llm.generic",
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
        assert result.get("status") == "error"
        assert registry.evidence, "the retrieval is still consulted"
        assert "provenance_bindings" not in result, (
            "a failed tool became citation authority for whatever follows"
        )

    def test_a_failed_node_does_not_own_the_turns_bindings(self, store, engine):
        """`on_error` hands off to a node that answers. The failed node's
        content is what the turn keeps only because the recovery produced
        none - and its sources are not authority for that sentence."""
        import uuid as _uuid

        user = store.create_user(email=f"fn_{_uuid.uuid4().hex[:8]}@t.local")
        ctx = store.upsert_context(
            name=f"fn-{_uuid.uuid4().hex[:6]}", description="c",
            owner_user_id=user.id,
        )
        engine.rag.ingest_text(ctx.id, "Epsilon turbine blade inspection record. " * 60)
        # Your scenario: the failed node's content is what the turn keeps,
        # because the node that recovered produced none of its own.
        # `llm.intent_classifier_v1` returns `{"intent": ...}` and no content.
        artifact = _stream_workflow(store, user, [
            {"id": "a", "type": "tool_call", "tool": "llm.generic",
             "inputs": {"context_id": ctx.id},
             "next": "fin", "on_error": "rec"},
            {"id": "rec", "type": "tool_call",
             "tool": "llm.intent_classifier_v1", "next": "fin"},
            {"id": "fin", "type": "end"},
        ])
        real = engine._tool_llm_generic

        def _fails(*args, **kwargs):
            result = real(*args, **kwargs)
            return {**result, "status": "error", "content": "failed"}

        engine._tool_llm_generic = _fails
        try:
            result = asyncio.run(
                engine.run(
                    artifact.id, None, "turbine blade inspection", None, user.id
                )
            )
        finally:
            engine._tool_llm_generic = real
        # Not vacuous: the failed node's content is what the turn kept, and
        # the retrieval did happen - only the authority is withheld.
        assert result.get("content") == "failed", result.get("content")
        assert result.get("context_snippets"), "the fixture never retrieved"
        assert not (result.get("provenance_bindings") or []), (
            "a failed node owns the turn's provenance: "
            f"{result.get('provenance_bindings')}"
        )


class TestTheStreamedParallelBlockOwnsItsOwn:
    @pytest.mark.asyncio
    async def test_streamed_parallel_children_replace_earlier_bindings(
        self, store, engine
    ):
        """The blocking driver got this rule; the streamed one did not, so a
        parallel block's concatenated answer kept an earlier node's sources."""
        import uuid as _uuid

        user = store.create_user(email=f"sp2_{_uuid.uuid4().hex[:8]}@t.local")
        first, second = _two_contexts(store, engine, user)
        artifact = _stream_workflow(store, user, [
            {"id": "a", "type": "tool_call", "tool": "llm.generic",
             "inputs": {"context_id": first}, "next": "fan"},
            {"id": "fan", "type": "parallel", "next": ["b"], "after": "fin"},
            {"id": "b", "type": "tool_call", "tool": "llm.generic",
             "inputs": {"context_id": second}},
            {"id": "fin", "type": "end"},
        ])
        engine.llm.generate_stream = _plain_stream
        events = [
            e async for e in engine.run_streaming(
                artifact.id, None, "turbine blade inspection", None,
                user_id=user.id, tenant_id=None,
            )
        ]
        done = [e for e in events if e.get("event") == "message_done"]
        assert done, [e.get("event") for e in events]
        bindings = done[-1]["data"].get("provenance_bindings") or []
        assert bindings, "the streamed parallel block reported no grounding"
        assert {b["context_id"] for b in bindings} == {second}, (
            "the node before the parallel block still owns the answer's "
            f"sources: {sorted({b['context_id'] for b in bindings})}"
        )


class TestASentenceTheServerWroteHasNoSources:
    @pytest.mark.asyncio
    async def test_the_synthetic_empty_answer_inherits_nothing(
        self, store, engine
    ):
        """`No response generated.` is server-authored. Retrieval happened,
        but that sentence rests on none of it."""
        import uuid as _uuid

        user = store.create_user(email=f"sy_{_uuid.uuid4().hex[:8]}@t.local")
        ctx = store.upsert_context(
            name=f"sy-{_uuid.uuid4().hex[:6]}", description="c",
            owner_user_id=user.id,
        )
        engine.rag.ingest_text(ctx.id, "Zeta turbine blade inspection record. " * 60)
        artifact = _stream_workflow(store, user, [
            {"id": "call", "type": "tool_call", "tool": "llm.generic",
             "next": "fin"},
            {"id": "fin", "type": "end"},
        ])
        engine.llm.generate_stream = lambda *a, **k: iter([
            {"event": "message_done", "data": {"content": "", "usage": {}}},
        ])
        events = [
            e async for e in engine.run_streaming(
                artifact.id, None, "turbine blade inspection", ctx.id,
                user_id=user.id, tenant_id=None,
            )
        ]
        done = [e for e in events if e.get("event") == "message_done"]
        assert done, [e.get("event") for e in events]
        data = done[-1]["data"]
        assert data.get("content") == "No response generated."
        assert not (data.get("provenance_bindings") or []), (
            "a sentence the server wrote inherited the model's grounding: "
            f"{data.get('provenance_bindings')}"
        )


class TestAWorkerNeverSuppliesItsOwnAuthority:
    """`provenance_bindings` is parent-owned. A worker runs model-chosen
    control flow over attacker-controlled bytes; if it could name what
    supported the answer, it could name a source it never read - and once S3
    validates citations against these, that is an authority bypass rather
    than bookkeeping."""

    FORGED = [{"context_id": "ctx_x", "source_id": "src_9", "evidence_id": "ev_9"}]

    @pytest.mark.asyncio
    async def test_a_successful_worker_cannot_supply_them(
        self, store, engine, monkeypatch
    ):
        """The parent has no grounding of its own here, so nothing overwrites
        the forged field - it has to be refused or stripped instead."""
        import uuid as _uuid

        user = store.create_user(email=f"wf_{_uuid.uuid4().hex[:8]}@t.local")

        def _served(invocation, worker_tool, plan, context, limits, **kw):
            return {
                "content": "answered",
                "usage": {},
                "provenance_bindings": list(self.FORGED),
            }

        monkeypatch.setattr(engine, "_serve_invocation", _served)
        result = await engine._invoke_tool(
            "agent.files_v1",
            {"message": "hi"},
            [],
            [],
            None,
            None,
            "hi",
            source_registry=SourceRegistry(),
            user_id=user.id,
            tenant_id=None,
        )
        # Refused, not quietly stripped. A worker that sent this is either
        # compromised or speaking a protocol this parent does not have, and
        # continuing from it would hide both.
        assert result.get("status") == "error", result
        assert result.get("error") == "validation_error"
        assert "reserved field" in (result.get("content") or "")
        assert result.get("provenance_bindings") != self.FORGED, (
            "a worker named what supported the answer"
        )

    @pytest.mark.asyncio
    async def test_a_failed_worker_cannot_either(self, store, engine, monkeypatch):
        """The nastier shape: the parent correctly declines to attach its own
        bindings to a failed result, so a forged set has nothing competing
        with it and survives into whatever the graph recovers with."""
        import uuid as _uuid

        user = store.create_user(email=f"wff_{_uuid.uuid4().hex[:8]}@t.local")

        def _served(invocation, worker_tool, plan, context, limits, **kw):
            return {
                "status": "error",
                "content": "failed",
                "usage": {},
                "provenance_bindings": list(self.FORGED),
            }

        monkeypatch.setattr(engine, "_serve_invocation", _served)
        result = await engine._invoke_tool(
            "agent.files_v1",
            {"message": "hi"},
            [],
            [],
            None,
            None,
            "hi",
            source_registry=SourceRegistry(),
            user_id=user.id,
            tenant_id=None,
        )
        assert result.get("provenance_bindings") != self.FORGED, (
            "a failed worker's forged authority survived"
        )

    def test_a_forged_field_never_reaches_the_turn(self, store, engine):
        """End to end, through `on_error` to a node that answers with no
        content of its own - the path that makes the failed node's content,
        and its claimed sources, the turn's final state."""
        import uuid as _uuid

        user = store.create_user(email=f"wft_{_uuid.uuid4().hex[:8]}@t.local")
        artifact = _stream_workflow(store, user, [
            {"id": "a", "type": "tool_call", "tool": "llm.generic",
             "next": "fin", "on_error": "rec"},
            {"id": "rec", "type": "tool_call",
             "tool": "llm.intent_classifier_v1", "next": "fin"},
            {"id": "fin", "type": "end"},
        ])
        real = engine._tool_llm_generic

        def _forges(*args, **kwargs):
            result = real(*args, **kwargs)
            return {
                **result,
                "status": "error",
                "content": "failed",
                "provenance_bindings": list(self.FORGED),
            }

        engine._tool_llm_generic = _forges
        try:
            result = asyncio.run(engine.run(artifact.id, None, "hi", None, user.id))
        finally:
            engine._tool_llm_generic = real
        assert (result.get("provenance_bindings") or []) != self.FORGED, (
            f"forged authority became the turn's: {result.get('provenance_bindings')}"
        )

    @pytest.mark.asyncio
    async def test_the_control_still_gets_the_parents_own(
        self, store, engine, monkeypatch
    ):
        """The guard must refuse the worker's field without costing the
        parent its own: real grounding still reaches the caller."""
        user_id, ctx_id = _grounded_context(store)
        registry = SourceRegistry()
        result = await engine._invoke_tool(
            "llm.generic",
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
        bindings = result.get("provenance_bindings") or []
        assert bindings, "the parent's own grounding was lost with the guard"
        assert {b["context_id"] for b in bindings} == {ctx_id}
