"""Provenance from a capability the worker asked for.

Explicit `file_search` runs parent-side, behind the broker. Two things have
to hold at that seam and neither is automatic: the ids that confer citation
authority must not travel back to the worker, and they must survive a replay
- the ledger returns a committed result to a replacement attempt without
running the handler again, so a sidecar that only exists in the handler's
return value is lost exactly when a retry needs it.
"""

from __future__ import annotations

import uuid

import pytest

from liminallm.service.broker import CapabilityBroker, InvocationContext
from liminallm.service.invocation import InvocationRegistry
from liminallm.service.provenance import SourceRegistry
from liminallm.service.runtime import get_runtime

MANUAL = "Turbine blade inspection happens every 400 flight hours. " * 40


@pytest.fixture
def store():
    return get_runtime().store


def _grounded(store, files=None):
    user = store.create_user(email=f"cap_{uuid.uuid4().hex[:8]}@t.local")
    ctx = store.upsert_context(
        name=f"cap-{uuid.uuid4().hex[:6]}", description="c",
        owner_user_id=user.id,
    )
    for path, text in (files or {"reports/turbines.md": MANUAL}).items():
        assert get_runtime().rag.ingest_text(ctx.id, text, source_path=path) > 0
    return user.id, ctx.id


def _broker(user_id, ctx_id, registry, bindings):
    context = InvocationContext(
        user_id=user_id,
        context_id=ctx_id,
        source_registry=registry,
        provenance_bindings=bindings,
    )
    return CapabilityBroker(get_runtime().workflow, context), context


def _ask(broker, invocation, seq=1, query="turbine blade inspection"):
    return broker._answer(
        invocation,
        {
            "capability": "rag.retrieve",
            "operation_seq": seq,
            "payload": {"query": query, "limit": 4},
        },
    )


def _open():
    return InvocationRegistry().open(
        uuid.uuid4().hex, tool="agent.files_v1", user_id="u", tenant_id=None
    )


class TestTheWorkerGetsTheTextAndNotTheAuthority:
    def test_the_capability_reply_carries_no_ids(self, store):
        """`BrokerClient.call` hands this straight to the worker. An id in it
        is an id the untrusted side can quote back as its own citation."""
        user_id, ctx_id = _grounded(store)
        registry, bindings = SourceRegistry(), []
        broker, _ = _broker(user_id, ctx_id, registry, bindings)
        reply = _ask(broker, _open())

        assert reply["ok"], reply
        result = reply["result"]
        assert result.get("text"), "the worker got no excerpts"
        flat = repr(result)
        for forbidden in ("provenance_bindings", "source_id", "evidence_id"):
            assert forbidden not in flat, (
                f"{forbidden} crossed the pipe: {flat[:200]}"
            )

    def test_the_parent_keeps_the_binding(self, store):
        user_id, ctx_id = _grounded(store)
        registry, bindings = SourceRegistry(), []
        broker, context = _broker(user_id, ctx_id, registry, bindings)
        _ask(broker, _open())

        assert context.provenance_bindings, "the parent recorded nothing"
        assert all(
            set(b) == {"context_id", "source_id", "evidence_id"}
            for b in context.provenance_bindings
        )
        assert registry.evidence, "no evidence was registered"
        for binding in context.provenance_bindings:
            assert registry.get_source(binding["source_id"]) is not None
            assert registry.get_evidence(binding["evidence_id"]) is not None


class TestAReplayedCapabilityStillCarriesItsProvenance:
    """The ledger returns a committed result to a replacement attempt without
    running the handler. If the bindings only existed in the handler's return
    value, the replayed attempt would receive the search text and no record of
    what it rested on - the exact defect this work removes, recreated by a
    retry."""

    def test_the_replacement_attempt_gets_the_original_bindings(self, store):
        user_id, ctx_id = _grounded(store)
        invocation = _open()

        first_registry, first_bindings = SourceRegistry(), []
        broker_a, ctx_a = _broker(user_id, ctx_id, first_registry, first_bindings)
        reply_a = _ask(broker_a, invocation)
        assert ctx_a.provenance_bindings, "attempt A recorded nothing"

        # Attempt B: the same invocation and ledger, a fresh worker and so a
        # fresh broker, asking for the same capability at the same position.
        second_registry, second_bindings = SourceRegistry(), []
        broker_b, ctx_b = _broker(user_id, ctx_id, second_registry, second_bindings)
        ran = {"handler": False}
        real = broker_b._rag_retrieve

        def _tripwire(*args, **kwargs):
            ran["handler"] = True
            return real(*args, **kwargs)

        broker_b._rag_retrieve = _tripwire
        reply_b = _ask(broker_b, invocation)

        assert reply_b.get("replayed"), "the fixture did not exercise a replay"
        assert not ran["handler"], "the handler ran again on replay"
        assert reply_b["result"] == reply_a["result"], "the text differed"
        assert ctx_b.provenance_bindings == ctx_a.provenance_bindings, (
            "the replayed attempt received the text with no provenance: "
            f"{ctx_b.provenance_bindings}"
        )


class TestOneRelationHoweverManyTimesItIsReached:
    def test_an_explicit_search_does_not_duplicate_the_opening_prompt(
        self, store
    ):
        """A selected context may have put a chunk in the opening prompt, and
        the model may then search and reach the same chunk. That is one
        eligible relation, not two copies of it."""
        user_id, ctx_id = _grounded(store)
        registry, bindings = SourceRegistry(), []
        broker, context = _broker(user_id, ctx_id, registry, bindings)

        _ask(broker, _open(), seq=1)
        first = [dict(b) for b in context.provenance_bindings]
        assert first, "the first search recorded nothing"

        # The same passages again, at a different position in the worker's
        # control flow so the ledger does not replay it.
        _ask(broker, _open(), seq=2)
        assert context.provenance_bindings == first, (
            f"the same relation was recorded twice: {context.provenance_bindings}"
        )


class TestParallelSearchesFoldInCallOrder:
    def test_the_slower_first_call_still_comes_first(self, store, monkeypatch):
        """Two searches run concurrently. Which finishes first is not the
        order the model asked in, and the relation list is read in order."""
        import time as _time

        engine = get_runtime().workflow
        user_id, ctx_id = _grounded(
            store,
            {
                "reports/alpha.md": "Alpha turbine inspection record. " * 40,
                "reports/beta.md": "Beta turbine inspection record. " * 40,
            },
        )
        registry = SourceRegistry()
        real = engine._run_file_search

        def _slow_first(query, limit, **kwargs):
            # The first call finishes last.
            if "alpha" in query:
                _time.sleep(0.15)
            return real(query, limit, **kwargs)

        monkeypatch.setattr(engine, "_run_file_search", _slow_first)
        collected: list = []
        engine._run_round_tools(
            [("id0", "file_search", {"query": "alpha turbine inspection"}),
             ("id1", "file_search", {"query": "beta turbine inspection"})],
            conversation_id=None,
            context_id=ctx_id,
            user_id=user_id,
            tenant_id=None,
            session={},
            snippets=[],
            fallback_query="turbine",
            source_registry=registry,
            bindings=collected,
        )
        assert collected, "the round recorded nothing"
        # Ids follow registration, which is completion order - that is fine,
        # the registry is a set. The binding list is what is read in order,
        # so it must follow the calls: alpha was asked first and finished
        # last.
        titles = [
            registry.get_source(b["source_id"]).title for b in collected
        ]
        assert titles[0] == "alpha.md", (
            f"bindings were folded in completion order: {titles}"
        )
        assert "beta.md" in titles, f"the second call recorded nothing: {titles}"
        for binding in collected:
            assert registry.get_source(binding["source_id"]) is not None
            assert registry.get_evidence(binding["evidence_id"]) is not None


class TestTheStreamedAssemblyOwnsItsCapabilityBindings:
    """A search the worker asked for is recorded on the invocation's context.
    Whether it becomes authority is the same question the assembly's own
    grounding answers: only if this assembly produced the answer."""

    @staticmethod
    def _served(recorded):
        """Stand in for a worker that ran one `file_search` and answered.

        The broker writes served bindings onto the context, so appending here
        is what that looks like from the streamed node's side.
        """

        def serve(invocation, tool, plan, context, limits, **kwargs):
            context.provenance_bindings.extend(recorded)
            return {"content": "answered", "usage": {}, "context_snippets": []}

        return serve

    async def _run(self, store, engine, monkeypatch, recorded, fail):
        import uuid as _uuid

        from liminallm.service.invocation import InvocationRegistry

        user_id, ctx_id = _grounded(store)
        monkeypatch.setattr(type(engine.llm), "supports_tools", True, raising=False)
        engine.llm.generate_stream = lambda *a, **k: iter([
            {"event": "token", "data": "hi"},
            {"event": "message_done", "data": {"content": "hi", "usage": {}}},
        ])
        if fail:
            def serve(*a, **k):
                raise RuntimeError("the worker died after searching")
            monkeypatch.setattr(engine, "_serve_invocation", serve)
        else:
            monkeypatch.setattr(engine, "_serve_invocation", self._served(recorded))

        sink: list = []
        invocation = InvocationRegistry().open(
            _uuid.uuid4().hex, tool="agent.files_v1", user_id=user_id,
            tenant_id=None,
        )
        async for _ in engine._stream_agent_files_node(
            {"id": "files", "type": "tool_call", "tool": "agent.files_v1"},
            user_message="turbine blade inspection",
            context_id=ctx_id,
            conversation_id=None,
            adapters=[],
            history=[],
            vars_scope={},
            source_registry=SourceRegistry(),
            bindings_sink=sink,
            user_id=user_id,
            tenant_id=None,
            invocation=invocation,
        ):
            pass
        return sink

    @pytest.mark.asyncio
    async def test_a_served_search_reaches_the_turn(
        self, store, monkeypatch
    ):
        searched = [
            {"context_id": "ctx_w", "source_id": "src_9", "evidence_id": "ev_9"}
        ]
        sink = await self._run(
            store, get_runtime().workflow, monkeypatch, searched, fail=False
        )
        assert searched[0] in sink, (
            f"the worker's own search grounded nothing: {sink}"
        )

    @pytest.mark.asyncio
    async def test_a_failed_assembly_takes_them_with_it(
        self, store, monkeypatch
    ):
        searched = [
            {"context_id": "ctx_w", "source_id": "src_9", "evidence_id": "ev_9"}
        ]
        sink = await self._run(
            store, get_runtime().workflow, monkeypatch, searched, fail=True
        )
        assert searched[0] not in sink, (
            f"a failed assembly's searches became authority: {sink}"
        )
