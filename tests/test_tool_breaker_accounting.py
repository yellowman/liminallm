"""Breaker accounting is one ledger, written at the tool boundary.

The rule, frozen:

    Every tool invocation that actually starts records exactly one breaker
    outcome against the resolved tool and tenant. Tool-level failure
    increments; tool-level success clears. Transport and retry path do not
    change the ledger. A call refused before invocation records nothing.

An invocation *starts* when its serve begins — the worker is spawned, or the
stream's producer runs. Resolution, admission, the breaker check itself,
input validation and plan assembly all happen before that point, so their
failures are refusals, not tool health. Two deliberate boundary rulings, both
measured against the previous behaviour first:

* An attempt that starts and is then cancelled at the node's own deadline
  records a failure. A hung backend whose `timeout_seconds` exceeds the node
  budget would otherwise never record an outcome at all, and the breaker
  could not protect against exactly the hang it exists for.
* An attempt abandoned by its caller — a cancel, a revoked lease — records
  nothing. The tool was not proven unhealthy, and a user's cancel habit must
  not open their tenant's breaker.

Measured on the previous code (each row one attempt, real engine, real
ledger): streamed failures recorded 0 and streamed successes cleared
nothing; a healthy tool whose *consumer's* `output_schema` refused the node
was charged a tool failure; so were input-validation refusals, unresolved
references, plan-phase crashes and caller revocations — none of which ran
the tool. All keys were the node's reference spelling, so two specs that
share a spelling shared a breaker across scopes.
"""

from __future__ import annotations

import asyncio
import threading
import time
import uuid

import pytest

from liminallm.service.invocation import LeaseRevoked
from liminallm.service.llm import LLMService
from liminallm.service.runtime import get_runtime
from liminallm.service.workflow import WorkflowEngine


def _u(p):
    return f"{p}_{uuid.uuid4().hex[:8]}"


def _wf(tool_name, *, max_retries=0, timeout_ms=None, backoff_ms=None):
    node = {"id": "call", "type": "tool_call", "next": "fin",
            "max_retries": max_retries}
    if tool_name is not None:
        node["tool"] = tool_name
    if timeout_ms is not None:
        node["timeout_ms"] = timeout_ms
    if backoff_ms is not None:
        node["backoff_ms"] = backoff_ms
    return {"kind": "workflow.chat", "entrypoint": "call", "nodes": [
        node, {"id": "fin", "type": "end"},
    ]}


async def _count(cache, tenant, *idents):
    """Failure count summed over the given breaker identities.

    Summing lets a witness of "exactly one failure" hold across the key
    migration: before the fix the write landed on the reference spelling,
    after it on the resolved identity, and a pin of the *amount* recorded
    must not care which key carried it.
    """
    total = 0
    for ident in idents:
        total += int(
            await cache.client.zcard(f"circuit:{tenant}:{ident}:failures:v2") or 0
        )
    return total


async def _is_open(cache, tenant, ident):
    return bool(await cache.client.exists(f"circuit:{tenant}:{ident}:open"))


class _Ctx:
    """One tenant, one engine, a clean ledger."""

    def __init__(self, *, llm=None, tenant=None, user=None):
        rt = get_runtime()
        self.store = rt.store
        self.cache = rt.cache
        self.tenant = tenant or _u("bk")
        self.user = user or self.store.create_user(
            email=f"{_u('bk')}@t.local", tenant_id=self.tenant
        )
        self.engine = WorkflowEngine(
            self.store, llm or rt.llm, rt.router, rt.rag, cache=rt.cache
        )

    def wf(self, tool_name, **kw):
        return self.store.create_artifact(
            "workflow", _u("wf"), _wf(tool_name, **kw),
            owner_user_id=self.user.id, visibility="private",
        )

    def tool(self, name, **extra):
        return self.store.create_artifact(
            "tool", _u("t"),
            {"kind": "tool.spec", "name": name, "handler": "llm.generic", **extra},
            owner_user_id=self.user.id, visibility="private",
        )

    async def run(self, wf_art):
        return await self.engine.run(
            wf_art.id, None, "hi", None,
            user_id=self.user.id, tenant_id=self.tenant,
        )

    async def run_streaming(self, wf_art):
        return [e async for e in self.engine.run_streaming(
            wf_art.id, None, "hi", None,
            user_id=self.user.id, tenant_id=self.tenant,
        )]

    async def seed_failures(self, ident, n):
        for _ in range(n):
            await self.cache.record_tool_failure(ident, tenant_id=self.tenant)

    async def count(self, *idents):
        return await _count(self.cache, self.tenant, *idents)

    async def is_open(self, ident):
        return await _is_open(self.cache, self.tenant, ident)

    def ident(self, name="llm.generic"):
        """The breaker identity `name` resolves to for this user's scope.

        Derived through the engine's own resolution, because the real store
        seeds the default tools as artifact rows: `llm.generic` resolves to
        a persisted artifact, and the identity is that row — not the
        spelling. Witnesses that assert an exact count read the spelling key
        *too*, so a write that regresses to the reference spelling is a
        miscount, not a miss.
        """
        from liminallm.service.tool_namespace import ToolResolutionScope

        d = self.engine._resolve_tool(
            name, ToolResolutionScope("private", self.user.id, self.tenant)
        )
        assert d is not None, f"{name} did not resolve"
        return d.artifact_id or d.name


def _ok(*a, **k):
    return {"status": "ok", "content": "x"}


def _err(*a, **k):
    return {"status": "error", "content": "x", "error": "boom"}


def _raise(*a, **k):
    raise RuntimeError("kaboom")


class _StreamBackend:
    """A cancellable streaming backend: one token, then the answer — or a
    failure before any token when told to fail. `on_stream` runs at the top
    of each provider call, for witnesses that change the world mid-retry."""

    supports_stream_cancel = True

    def __init__(self, fail=False, stall=False, fail_after_token=False,
                 truncate=False):
        self.fail = fail
        self.stall = stall
        self.fail_after_token = fail_after_token
        #: A clean TCP-style EOF: one token, then the iterator returns with
        #: no `message_done` and no error.
        self.truncate = truncate
        self.stream_calls = 0
        self.on_stream = None
        #: When set, the stream blocks on this event after its first token —
        #: the shape of a provider mid-read when a cancel lands.
        self.block_after_token = None
        #: A pause between two tokens, so a cancel set on the first is
        #: discovered when the second arrives rather than by the stop path.
        self.gap = False

    def generate(self, messages, adapters, *, user_id=None):
        return {"content": "whole answer", "usage": {}}

    def generate_stream(self, messages, adapters, *, user_id=None):
        self.stream_calls += 1
        if self.on_stream is not None:
            self.on_stream()

        def gen():
            if self.fail:
                raise RuntimeError("stream boom")
            yield {"event": "token", "data": "x"}
            if self.truncate:
                return
            if self.fail_after_token:
                raise RuntimeError("post-token boom")
            if self.block_after_token is not None:
                self.block_after_token.wait(8.0)
            if self.gap:
                time.sleep(0.4)
                yield {"event": "token", "data": "y"}
            if self.stall:
                time.sleep(2)
            yield {"event": "message_done", "data": {"content": "x", "usage": {}}}
        return gen()


class _SpyCache:
    """The real cache, with the ledger writes on the record.

    Delegation rather than a hand-made stand-in: every read and the Lua
    arithmetic stay the real object's, and the spy only observes. Needed
    because the real ledger masks one class of defect — recording a failure
    while the breaker is open is a no-op in the atomic script — so "the
    counter did not move" cannot distinguish a refusal that records nothing
    from a refusal whose recording was silently swallowed.
    """

    def __init__(self, inner):
        self._inner = inner
        self.recorded = []

    def __getattr__(self, name):
        return getattr(self._inner, name)

    async def record_tool_failure(self, *a, **k):
        self.recorded.append(("failure", a, k))
        return await self._inner.record_tool_failure(*a, **k)

    async def record_tool_success(self, *a, **k):
        self.recorded.append(("success", a, k))
        return await self._inner.record_tool_success(*a, **k)


def _stream_ctx(fail=False, stall=False, fail_after_token=False, truncate=False):
    backend = _StreamBackend(
        fail=fail, stall=stall, fail_after_token=fail_after_token,
        truncate=truncate,
    )
    return _Ctx(llm=LLMService("test-model", backend=backend)), backend


class _SlowCheckCache:
    """The real cache with a stalled breaker check, for the witness that
    preparation time comes out of the attempt's budget, not on top of it.
    With `once=True` only the first check stalls — the shape of a transient
    stall whose retry must then get its retry."""

    def __init__(self, inner, delay, once=False):
        self._inner = inner
        self._delay = delay
        self._once = once
        self._checks = 0

    def __getattr__(self, name):
        return getattr(self._inner, name)

    async def check_circuit_breaker(self, *a, **k):
        self._checks += 1
        if not self._once or self._checks == 1:
            await asyncio.sleep(self._delay)
        return await self._inner.check_circuit_breaker(*a, **k)


# =============================================================================
# Tool-level outcomes, blocking transport. These pin what already held.
# =============================================================================


class TestToolLevelOutcomesAreTheLedger:
    @pytest.mark.asyncio
    async def test_a_raw_error_result_records_one_failure(self):
        c = _Ctx()
        c.engine._tool_llm_generic = _err
        r = await c.run(c.wf("llm.generic"))
        assert r.get("status") == "error"
        assert await c.count(c.ident(), "llm.generic") == 1

    @pytest.mark.asyncio
    async def test_a_success_clears_the_failure_count(self):
        c = _Ctx()
        await c.seed_failures(c.ident(), 3)
        c.engine._tool_llm_generic = _ok
        await c.run(c.wf("llm.generic"))
        assert await c.count(c.ident(), "llm.generic") == 0

    @pytest.mark.asyncio
    async def test_each_started_attempt_records_its_own_outcome(self):
        """The retry path does not change the ledger: three attempts that
        each ran the tool are three observations of tool health."""
        c = _Ctx()
        c.engine._tool_llm_generic = _raise
        await c.run(c.wf("llm.generic", max_retries=2))
        assert await c.count(c.ident(), "llm.generic") == 3

    @pytest.mark.asyncio
    async def test_a_tool_spec_timeout_records_a_failure(self):
        """The tool's own declared budget: exceeding it is tool health."""
        c = _Ctx()
        art = c.tool(name := _u("slow"), timeout_seconds=1)
        c.engine._tool_llm_generic = lambda *a, **k: (time.sleep(3), _ok())[1]
        r = await c.run(c.wf(name))
        assert r.get("status") == "error"
        assert await c.count(name, art.id) == 1


# =============================================================================
# Transport parity. Streaming recorded nothing at all.
# =============================================================================


class TestTheTransportDoesNotChangeTheLedger:
    @pytest.mark.asyncio
    async def test_a_streamed_failure_increments_the_breaker(self):
        c, _ = _stream_ctx(fail=True)
        await c.run_streaming(c.wf("llm.generic"))
        assert await c.count(c.ident(), "llm.generic") == 1, (
            "a tool failure on the streaming transport left the ledger "
            "untouched; the same failure on the blocking transport counts"
        )

    @pytest.mark.asyncio
    async def test_a_streamed_success_clears_the_failure_count(self):
        c, _ = _stream_ctx()
        await c.seed_failures(c.ident(), 3)
        events = await c.run_streaming(c.wf("llm.generic"))
        assert any(e.get("event") == "message_done" for e in events)
        assert await c.count(c.ident(), "llm.generic") == 0, (
            "a streamed success did not reset the failure counter; four "
            "sporadic failures spread across days would open the breaker"
        )

    @pytest.mark.asyncio
    async def test_streaming_failures_open_the_breaker_for_blocking(self):
        c, _ = _stream_ctx(fail=True)
        await c.seed_failures(c.ident(), 4)
        await c.run_streaming(c.wf("llm.generic"))
        assert await c.is_open(c.ident()), (
            "the fifth failure arrived on the streaming transport and the "
            "breaker did not trip"
        )

        blocking = _Ctx(tenant=c.tenant, user=c.user)
        ran = []
        blocking.engine._tool_llm_generic = lambda *a, **k: (ran.append(1), _ok())[1]
        r = await blocking.run(blocking.wf("llm.generic"))
        assert r.get("status") == "error" and not ran, (
            "an open breaker did not refuse the blocking invocation"
        )

    @pytest.mark.asyncio
    async def test_a_clean_truncation_is_a_failure(self):
        """A provider that returns mid-answer without an error event — a
        clean TCP EOF — started its serve and produced no completed result.
        That is a tool failure, distinguishable now from an interruption:
        the pump knows a stop cut the stream short, and only the stop stays
        silent. Five clean EOFs must open the breaker."""
        c, backend = _stream_ctx(truncate=True)
        await c.seed_failures(c.ident(), 4)
        events = await c.run_streaming(c.wf("llm.generic", max_retries=0))
        assert any(e.get("event") == "error" for e in events)
        assert await c.is_open(c.ident()), (
            "a provider stream that truncated cleanly recorded nothing; "
            "a backend that always dies mid-answer can never open the breaker"
        )

    @pytest.mark.asyncio
    async def test_blocking_failures_open_the_breaker_for_streaming(self):
        """The control: this direction already held, and must keep holding."""
        c = _Ctx()
        await c.seed_failures(c.ident(), 4)
        c.engine._tool_llm_generic = _err
        await c.run(c.wf("llm.generic"))
        assert await c.is_open(c.ident())

        streamed, backend = _stream_ctx()
        streamed.tenant, streamed.user = c.tenant, c.user
        r = await streamed.run_streaming(streamed.wf("llm.generic"))
        assert backend.stream_calls == 0, (
            "an open breaker did not stop the streamed invocation"
        )
        assert not any(e.get("event") == "message_done" and
                       (e.get("data") or {}).get("content") for e in r)


# =============================================================================
# Node correctness is not tool health.
# =============================================================================


class TestNodeCorrectnessIsNotToolHealth:
    @pytest.mark.asyncio
    async def test_an_output_schema_refusal_records_the_tools_success(self):
        """The tool answered; the consumer's schema refused the node. The
        node fails — and the ledger records a success, clearing the count."""
        c = _Ctx()
        art = c.tool(
            name := _u("strict"),
            output_schema={"type": "object", "required": ["definitely_absent"]},
        )
        await c.seed_failures(art.id, 2)
        c.engine._tool_llm_generic = _ok
        r = await c.run(c.wf(name))
        assert r.get("status") == "error"
        assert await c.count(art.id, name) == 0, (
            "a consumer's output_schema charged the tool's breaker; five "
            "runs of one misconfigured workflow would cut off every "
            "workflow of the tenant that uses the healthy tool"
        )

    @pytest.mark.asyncio
    async def test_a_streamed_output_schema_refusal_records_the_tools_success(self):
        c, _ = _stream_ctx()
        art = c.tool(
            name := _u("strict"),
            output_schema={"type": "object", "required": ["definitely_absent"]},
        )
        await c.seed_failures(art.id, 2)
        await c.run_streaming(c.wf(name))
        assert await c.count(art.id, name) == 0


# =============================================================================
# A call refused before invocation records nothing.
# =============================================================================


class TestARefusalBeforeInvocationRecordsNothing:
    @pytest.mark.asyncio
    async def test_a_circuit_open_refusal_records_nothing(self):
        """The control that already held: the breaker's own refusal must not
        feed the breaker. The spy is what makes it airtight — the real
        ledger no-ops a failure recorded while open, so the counter alone
        cannot see an engine that wrongly records on refusal."""
        c = _Ctx()
        await c.seed_failures(c.ident(), 5)
        assert await c.is_open(c.ident())
        spy = _SpyCache(c.cache)
        c.engine.cache = spy
        ran = []
        c.engine._tool_llm_generic = lambda *a, **k: (ran.append(1), _ok())[1]
        r = await c.run(c.wf("llm.generic"))
        assert r.get("status") == "error" and not ran
        assert await c.count(c.ident(), "llm.generic") == 0
        assert not spy.recorded, (
            f"an open-breaker refusal wrote to the ledger: {spy.recorded}"
        )

    @pytest.mark.asyncio
    async def test_an_input_validation_refusal_records_nothing(self):
        c = _Ctx()
        art = c.tool(
            name := _u("strictin"),
            input_schema={"type": "object", "required": ["impossible_key"]},
        )
        c.engine._tool_llm_generic = _ok
        r = await c.run(c.wf(name))
        assert r.get("status") == "error"
        assert await c.count(name, art.id) == 0, (
            "an input refused before the tool ran was charged to the tool"
        )

    @pytest.mark.asyncio
    async def test_an_unresolved_reference_records_nothing(self):
        c = _Ctx()
        art = c.tool(name := _u("gone"))
        wf_art = c.wf(name)
        with c.store._connect() as conn:
            conn.execute("DELETE FROM artifact WHERE id = %s", (art.id,))
        r = await c.run(wf_art)
        assert r.get("status") == "error"
        assert await c.count(name, art.id) == 0, (
            "a reference that resolved to nothing was charged to the "
            "breaker of whatever tool later takes that name"
        )

    @pytest.mark.asyncio
    async def test_the_invocation_backstop_refusal_records_nothing(self):
        """`_invoke_tool` keeps its own preflight as the last line of defense
        for a caller that skipped admission — the seams refuse first, so
        only a direct probe reaches it. Its refusal must be as silent in
        the ledger as the seams' are."""
        c = _Ctx()
        art = c.tool(
            name := _u("backstop"),
            input_schema={"type": "object", "required": ["impossible_key"]},
        )
        from liminallm.service.node_attempt import BreakerObservation
        from liminallm.service.tool_namespace import ToolResolutionScope

        d = c.engine._resolve_tool(
            name, ToolResolutionScope("private", c.user.id, c.tenant)
        )
        observation = BreakerObservation(identity=art.id)
        r = await c.engine._invoke_tool(
            name, {"message": "hi"}, [], [], None, None, None,
            user_id=c.user.id, tenant_id=c.tenant,
            descriptor=d, observation=observation,
        )
        assert r.get("error") == "validation_error", r
        assert observation.outcome is None and not observation.started
        await c.engine._record_breaker_outcome(observation, tenant_id=c.tenant)
        assert await c.count(name, art.id) == 0

    @pytest.mark.asyncio
    async def test_a_plan_phase_failure_records_nothing(self):
        """Plan assembly is engine work — attachments, context, budgets. A
        crash there proves nothing about the tool, whose body never ran."""
        c = _Ctx()

        def boom(*a, **k):
            raise RuntimeError("plan kaboom")

        c.engine._plan_invocation = boom
        r = await c.run(c.wf("llm.generic"))
        assert r.get("status") == "error"
        assert await c.count(c.ident(), "llm.generic") == 0


# =============================================================================
# The started-attempt boundary rulings.
# =============================================================================


class TestTheStartedAttemptBoundary:
    @pytest.mark.asyncio
    async def test_a_node_deadline_timeout_records_a_failure(self):
        """The attempt started and produced nothing inside the node's budget.
        Without this ruling a backend hung past every node budget never
        records an outcome, and the breaker cannot open for a hang."""
        c = _Ctx()
        c.engine._tool_llm_generic = lambda *a, **k: (time.sleep(3), _ok())[1]
        r = await c.run(c.wf("llm.generic", timeout_ms=300))
        assert r.get("status") == "error"
        assert await c.count(c.ident(), "llm.generic") == 1

    @pytest.mark.asyncio
    async def test_a_deadline_before_the_worker_spawns_records_nothing(self):
        """`started` means the worker actually started — not that the serve
        was scheduled into a thread pool. A deadline that expires while the
        serve is queued or stalled short of its spawn proves nothing about
        the tool."""
        c = _Ctx()

        def never_spawns(*a, **k):
            time.sleep(2)
            return _ok()

        c.engine._serve_invocation = never_spawns
        r = await c.run(c.wf("llm.generic", timeout_ms=300, max_retries=0))
        assert r.get("status") == "error"
        assert await c.count(c.ident(), "llm.generic") == 0, (
            "a deadline that expired before any worker spawned was charged "
            "to the tool"
        )

    @pytest.mark.asyncio
    async def test_a_deadline_during_the_ready_wait_records_the_failure(self):
        """The other edge of `started`: the worker has started and is
        registered — only the READY handshake is outstanding. A node
        deadline expiring there kills a real, running worker, and that is
        a started serve cut off: a breaker failure. Marking `started` only
        after `spawn()` returns leaves a window as long as the handshake
        timeout in which a killed worker was never `started`."""
        from unittest.mock import patch

        import liminallm.service.tool_worker as tool_worker_mod

        c = _Ctx()
        ready_entered = threading.Event()

        def held_ready(conn, process):
            ready_entered.set()
            time.sleep(6)
            return False

        with patch.object(tool_worker_mod, "_await_ready", held_ready):
            r = await c.run(c.wf("llm.generic", timeout_ms=2500, max_retries=0))
        assert ready_entered.is_set(), "the worker never reached registration"
        assert r.get("status") == "error"
        assert await c.count(c.ident(), "llm.generic") == 1, (
            "a registered worker was killed at the node deadline and the "
            "breaker recorded nothing: started was marked after the READY "
            "wait instead of at registration"
        )

    @pytest.mark.asyncio
    async def test_a_deadline_spent_in_planning_records_nothing(self):
        """The complement that positions the boundary: the same deadline,
        fired before the serve began, proves nothing about the tool."""
        c = _Ctx()

        def slow_plan(*a, **k):
            time.sleep(2)
            raise RuntimeError("abandoned")

        c.engine._plan_invocation = slow_plan
        r = await c.run(c.wf("llm.generic", timeout_ms=300))
        assert r.get("status") == "error"
        assert await c.count(c.ident(), "llm.generic") == 0

    @pytest.mark.asyncio
    async def test_preparation_cannot_extend_the_node_deadline(self):
        """The deadline is absolute and preparation spends it. A breaker
        check that stalls past the node budget must not hand the body a
        fresh clock afterwards: the attempt times out, the tool never
        starts, and nothing is recorded — preparation never `started`."""
        c = _Ctx()
        c.engine.cache = _SlowCheckCache(c.cache, 0.5)
        ran = []
        c.engine._tool_llm_generic = lambda *a, **k: (ran.append(1), _ok())[1]
        r = await c.run(c.wf("llm.generic", timeout_ms=300))
        assert r.get("status") == "error"
        assert not ran, (
            "preparation overran the node deadline and the body started on "
            "a fresh clock anyway"
        )
        assert await c.count(c.ident(), "llm.generic") == 0

    @pytest.mark.asyncio
    async def test_a_caller_revocation_records_nothing(self):
        """The caller walked away mid-serve. The tool was not proven
        unhealthy, and a cancel habit must not open the tenant's breaker."""
        c = _Ctx()

        def revoked(*a, **k):
            # Stands for a worker that spawned and whose lease was then
            # revoked mid-serve: the real serve marks `started` at the
            # spawn, so this double does too — otherwise the recorder's
            # started gate masks the very write this witness polices.
            if k.get("observation") is not None:
                k["observation"].started = True
            raise LeaseRevoked("caller cancelled")

        c.engine._serve_invocation = revoked
        r = await c.run(c.wf("llm.generic"))
        assert r.get("status") == "error"
        assert await c.count(c.ident(), "llm.generic") == 0


# =============================================================================
# The breaker binds each attempt, not the logical node.
# =============================================================================


class TestTheBreakerBindsEachAttempt:
    @pytest.mark.asyncio
    async def test_a_breaker_tripped_mid_execution_stops_the_next_attempt(self):
        """The fifth failure arrives on attempt one of a retrying node. The
        retry is a new invocation, and an open breaker refuses it before it
        starts — checking once at node entry lets retries walk past the trip
        their own first attempt caused."""
        c = _Ctx()
        await c.seed_failures(c.ident(), 4)
        calls = []
        c.engine._tool_llm_generic = lambda *a, **k: (calls.append(1), _err())[1]
        r = await c.run(c.wf("llm.generic", max_retries=2, backoff_ms=1))
        assert r.get("status") == "error"
        assert await c.is_open(c.ident())
        assert len(calls) == 1, (
            f"the first attempt tripped the breaker and the retries ran the "
            f"tool anyway: {len(calls)} calls"
        )

    @pytest.mark.asyncio
    async def test_a_streamed_trip_mid_execution_stops_the_next_attempt(self):
        c, backend = _stream_ctx(fail=True)
        await c.seed_failures(c.ident(), 4)
        await c.run_streaming(c.wf("llm.generic", max_retries=2, backoff_ms=1))
        assert await c.is_open(c.ident())
        assert backend.stream_calls == 1, (
            f"the first streamed attempt tripped the breaker and the retries "
            f"called the provider anyway: {backend.stream_calls} calls"
        )

    @pytest.mark.asyncio
    async def test_a_retry_resolves_current_canonical_state(self):
        """Tranche 2's frozen rule, per attempt: current canonical state is
        consulted at execution, and a captured descriptor is a process-local
        cache manufacturing authority. Attempt one retires the tool; attempt
        two must ask Postgres and refuse, not rerun the deleted spec."""
        c = _Ctx()
        name = _u("mut")
        art = c.tool(name)
        wf_art = c.wf(name, max_retries=1, backoff_ms=1)
        calls = []

        def delete_and_fail(*a, **k):
            calls.append(1)
            with c.store._connect() as conn:
                conn.execute("DELETE FROM artifact WHERE id = %s", (art.id,))
            return _err()

        c.engine._tool_llm_generic = delete_and_fail
        r = await c.run(wf_art)
        assert r.get("status") == "error"
        assert len(calls) == 1, (
            f"a retry ran a spec the store no longer holds: {len(calls)} calls"
        )
        assert await c.count(name, art.id) == 1

    @pytest.mark.asyncio
    async def test_a_streamed_retry_resolves_current_canonical_state(self):
        c, backend = _stream_ctx(fail=True)
        name = _u("smut")
        art = c.tool(name)
        wf_art = c.wf(name, max_retries=1, backoff_ms=1)

        def delete_art():
            with c.store._connect() as conn:
                conn.execute("DELETE FROM artifact WHERE id = %s", (art.id,))

        backend.on_stream = delete_art
        await c.run_streaming(wf_art)
        assert backend.stream_calls == 1, (
            f"a streamed retry ran a spec the store no longer holds: "
            f"{backend.stream_calls} calls"
        )
        assert await c.count(name, art.id) == 1


# =============================================================================
# A timed-out attempt is one attempt, not the whole execution.
# =============================================================================


class TestATimedOutAttemptIsNotTheExecution:
    """The driver's timeout revoke must scope to the attempt that timed out.
    A revoke that finds no current `Attempt` cancels the logical execution —
    fail-closed and right for a revoke racing the first spawn, wrong for a
    node timeout whose retry policy still owes the node a retry."""

    @pytest.mark.asyncio
    async def test_a_preparation_timeout_leaves_the_retry_its_retry(self):
        """Attempt one's breaker check stalls past the node deadline; attempt
        two is instant and its body succeeds. The node must succeed."""
        c = _Ctx()
        c.engine.cache = _SlowCheckCache(c.cache, 0.5, once=True)
        ran = []
        c.engine._tool_llm_generic = lambda *a, **k: (ran.append(1), _ok())[1]
        r = await c.run(
            c.wf("llm.generic", timeout_ms=300, max_retries=1, backoff_ms=1)
        )
        assert r.get("status") != "error", (
            f"a preparation timeout cancelled the logical execution and the "
            f"retry never ran: {r.get('error')}"
        )
        assert len(ran) == 1

    @pytest.mark.asyncio
    async def test_a_streamed_preparation_timeout_leaves_the_retry_its_retry(self):
        c, backend = _stream_ctx()
        c.engine.cache = _SlowCheckCache(c.cache, 0.5, once=True)
        events = await c.run_streaming(
            c.wf("llm.generic", timeout_ms=300, max_retries=1, backoff_ms=1)
        )
        done = [e for e in events if e.get("event") == "message_done"]
        assert done and done[-1]["data"].get("content") == "x", (
            "a streamed preparation timeout cancelled the logical execution "
            "and the retry never streamed"
        )
        assert backend.stream_calls == 1

    @pytest.mark.asyncio
    async def test_a_planning_timeout_leaves_the_retry_its_retry(self):
        """The same hole one stage later: blocking plan assembly times out
        before the worker spawn ever opened an `Attempt`."""
        c = _Ctx()
        real_plan = c.engine._plan_invocation
        calls = []

        def slow_once_plan(*a, **k):
            calls.append(1)
            if len(calls) == 1:
                time.sleep(0.6)
            return real_plan(*a, **k)

        c.engine._plan_invocation = slow_once_plan
        ran = []
        c.engine._tool_llm_generic = lambda *a, **k: (ran.append(1), _ok())[1]
        r = await c.run(
            c.wf("llm.generic", timeout_ms=300, max_retries=1, backoff_ms=1)
        )
        assert r.get("status") != "error", (
            f"a planning timeout cancelled the logical execution and the "
            f"retry never ran: {r.get('error')}"
        )
        assert len(ran) == 1


# =============================================================================
# Authority travels by exact attempt identity.
# =============================================================================


class TestAuthorityTravelsByExactAttempt:
    """A worker spawn joins the attempt it was created for — never whatever
    attempt happens to be current when an abandoned thread finally wakes up.
    Ambient authority by arrival time is the class tranche 2 removed."""

    @pytest.mark.asyncio
    async def test_a_late_serve_cannot_adopt_the_retry_attempt(self):
        """Attempt one's serve is still queued when the node times out and
        the retry begins. When that stale thread finally spawns, it must be
        refused — not adopt the retry's fresh attempt and run the old plan
        beside the retry's own worker."""
        c = _Ctx()
        real_serve = c.engine._serve_invocation
        b_started = threading.Event()
        a_done = threading.Event()
        serve_calls = []
        body_calls = []

        def gated_serve(*a, **k):
            n = len(serve_calls)
            serve_calls.append(1)
            if n == 0:
                try:
                    b_started.wait(8.0)
                    return real_serve(*a, **k)
                finally:
                    a_done.set()
            b_started.set()
            return real_serve(*a, **k)

        c.engine._serve_invocation = gated_serve
        c.engine._tool_llm_generic = (
            lambda *a, **k: (body_calls.append(1), time.sleep(0.5), _ok())[2]
        )
        r = await c.run(
            c.wf("llm.generic", timeout_ms=1500, max_retries=1, backoff_ms=1)
        )
        assert r.get("status") != "error", r
        assert a_done.wait(8.0), "attempt one's serve thread never finished"
        await asyncio.sleep(0.2)
        assert len(body_calls) == 1, (
            f"attempt one's abandoned serve adopted the retry's attempt and "
            f"ran the old plan under its authority: {len(body_calls)} runs"
        )

    def test_adoption_requires_the_exact_live_attempt(self):
        """The `Invocation` contract itself: adoption names the attempt the
        spawn belongs to, and anything else — a revoked attempt, a
        different current attempt, a cancelled execution — is refused."""
        from liminallm.service.invocation import Invocation, LeaseRevoked

        inv = Invocation(_u("inv"), tool="t")
        a = inv.begin_attempt()
        inv.revoke("node_timeout")
        inv.end_attempt(a)
        b = inv.begin_attempt()

        with pytest.raises(LeaseRevoked):
            inv.adopt_attempt(a)
        assert not b.adopted, (
            "a stale spawn's adoption attached to the retry's attempt"
        )
        assert inv.adopt_attempt(b) is b

        fresh = Invocation(_u("inv"), tool="t")
        own = fresh.adopt_attempt(None)
        assert own is fresh.current_attempt, (
            "a driverless spawn must still begin its own attempt"
        )

    @pytest.mark.asyncio
    async def test_a_spawn_failure_leaves_the_retry_prompt(self):
        """`process.start()` raising is an ordinary retryable failure. It
        must not strand a half-adopted attempt whose `finished` nobody will
        ever set — that turns a one-line OSError into the thirty-second
        unreaped path."""
        import liminallm.service.tool_worker as tool_worker_mod

        real_get_context = tool_worker_mod.multiprocessing.get_context
        state = {"n": 0}

        class FlakyContext:
            def __init__(self, real):
                self._real = real

            def __getattr__(self, name):
                return getattr(self._real, name)

            def Process(self, *a, **k):
                p = self._real.Process(*a, **k)
                state["n"] += 1
                if state["n"] == 1:
                    def boom():
                        raise OSError("spawn refused")
                    p.start = boom
                return p

        c = _Ctx()
        ran = []
        c.engine._tool_llm_generic = lambda *a, **k: (ran.append(1), _ok())[1]
        t0 = time.monotonic()
        from unittest.mock import patch

        with patch.object(
            tool_worker_mod.multiprocessing, "get_context",
            lambda kind: FlakyContext(real_get_context(kind)),
        ):
            r = await c.run(
                c.wf("llm.generic", max_retries=1, backoff_ms=1)
            )
        elapsed = time.monotonic() - t0
        assert r.get("status") != "error", r
        assert len(ran) == 1
        assert elapsed < 10, (
            f"a spawn failure stranded its attempt and the retry waited "
            f"{elapsed:.0f}s for a serve loop that never ran"
        )
        assert await c.count(c.ident(), "llm.generic") == 0, (
            "a worker that never started was charged to the tool"
        )

    @pytest.mark.asyncio
    async def test_a_revoke_cannot_land_inside_the_spawn_window(self):
        """Adoption, worker start and child registration are one
        linearization boundary: a revoke either refuses the spawn before it
        starts anything, or finds the started child registered and kills
        it. On the unlocked shape, a revoke slips between adoption and
        `process.start()`, its kill sweep finds nothing, and the child
        starts after revocation with nobody holding it."""
        import liminallm.service.tool_worker as tool_worker_mod
        from liminallm.service.invocation import Invocation

        in_window = threading.Event()
        release = threading.Event()
        registered = []
        kills = []

        class HeldProcess:
            pid = 4242

            def start(self):
                in_window.set()
                release.wait(8.0)

            def is_alive(self):
                return False

            def join(self, *a, **k):
                return None

        class HeldContext:
            def __init__(self, real):
                self._real = real

            def __getattr__(self, name):
                return getattr(self._real, name)

            def Process(self, *a, **k):
                return HeldProcess()

        inv = Invocation(_u("inv"), tool="t")
        inv.resources.add_child = (
            lambda *a, **k: registered.append(1)
        )
        inv.resources.kill_all = lambda: (kills.append(1), [])[1]

        real_get_context = tool_worker_mod.multiprocessing.get_context
        revoke_done_in_window = []

        def spawn_it():
            try:
                tool_worker_mod.spawn(
                    inv, "llm.generic", {}, limits={}, scratch=lambda: "/tmp"
                )
            except Exception:
                pass

        from unittest.mock import patch

        with patch.object(
            tool_worker_mod.multiprocessing, "get_context",
            lambda kind: HeldContext(real_get_context(kind)),
        ), patch.object(tool_worker_mod, "_READY_TIMEOUT_SECONDS", 0.2):
            t1 = threading.Thread(target=spawn_it, daemon=True)
            t1.start()
            assert in_window.wait(8.0), "the spawn never reached start()"

            def revoke_it():
                inv.revoke("node_timeout")

            t2 = threading.Thread(target=revoke_it, daemon=True)
            t2.start()
            t2.join(0.4)
            if not t2.is_alive():
                revoke_done_in_window.append(1)
            release.set()
            t1.join(8.0)
            t2.join(8.0)

        assert not revoke_done_in_window, (
            "a revoke completed between adoption and process.start(): its "
            "kill sweep ran before the child existed, and the child then "
            "started after revocation"
        )
        assert registered and kills, (
            "after the window closed, the revoke must find the registered "
            "child and sweep it"
        )


# =============================================================================
# A stale serve allocates nothing on its way to being refused.
# =============================================================================


class TestAStaleServeLeavesNothing:
    @pytest.mark.asyncio
    async def test_a_stale_serve_after_close_leaves_no_scratch(self):
        """The invocation is closed — cancelled, terminated, cleaned. A
        serve thread waking after that allocates its scratch outside the
        lock, then is refused at the adoption check inside it. What it must
        not do is leave that directory behind, because the second `close()`
        is an idempotent no-op with nobody left to remove it — so the
        refused spawn deletes the directory it made, itself."""
        import os
        import uuid as uuid_mod

        from liminallm.service.broker import InvocationContext
        from liminallm.service.node_attempt import BreakerObservation

        c = _Ctx()
        engine = c.engine
        inv = engine.invocations.open(
            uuid_mod.uuid4().hex, tool="llm.generic",
            user_id=c.user.id, tenant_id=c.tenant,
        )
        lease = inv.begin_attempt()
        inv.end_attempt(lease)
        await asyncio.to_thread(inv.close)

        created = []
        real_scratch = engine._worker_scratch

        def recording_scratch(invocation):
            path = real_scratch(invocation)
            created.append(path)
            return path

        engine._worker_scratch = recording_scratch
        obs = BreakerObservation(identity="x", attempt=lease)
        context = InvocationContext(
            user_id=c.user.id, tenant_id=c.tenant, conversation_id=None,
            context_id=None, adapters=[], history=[], user_message="hi",
        )
        with pytest.raises(LeaseRevoked):
            await asyncio.to_thread(
                engine._serve_invocation,
                inv, "llm.generic", {"inputs": {}, "message": "hi"},
                context, engine._worker_limits(None),
                expected_attempt=lease, observation=obs,
            )
        for path in created:
            assert not os.path.exists(path), (
                f"a stale serve left filesystem state behind after the "
                f"invocation closed, with nobody left to remove it: {path}"
            )
        assert obs.started is False


# =============================================================================
# A stalled scratch allocation cannot hold off revocation.
# =============================================================================


class TestAStalledScratchDoesNotBlockRevocation:
    @pytest.mark.asyncio
    async def test_a_stalled_scratch_allocation_does_not_hold_off_the_revoke(
        self,
    ):
        """A node deadline revokes through the invocation lock. If the
        scratch allocation runs while that lock is held, a slow `mkdtemp`
        holds the revoke off for its whole duration — and the hard
        wall-clock deadline earlier rounds established is gone, replaced by
        "the deadline, plus however long the filesystem took". Allocation
        runs outside the lock now; only the revalidated ownership transfer
        runs inside it, so the revoke lands at once."""
        import os

        from liminallm.service.broker import InvocationContext
        from liminallm.service.node_attempt import BreakerObservation

        c = _Ctx()
        engine = c.engine
        inv = engine.invocations.open(
            _u("stall"), tool="llm.generic",
            user_id=c.user.id, tenant_id=c.tenant,
        )
        lease = inv.begin_attempt()
        obs = BreakerObservation(identity="x", attempt=lease)

        entered = threading.Event()
        allocated = []
        real_scratch = engine._worker_scratch

        def blocking_scratch(invocation):
            entered.set()
            time.sleep(2.0)
            path = real_scratch(invocation)
            allocated.append(path)
            return path

        engine._worker_scratch = blocking_scratch
        context = InvocationContext(
            user_id=c.user.id, tenant_id=c.tenant, conversation_id=None,
            context_id=None, adapters=[], history=[], user_message="hi",
        )
        serve_err = {}

        def serve():
            try:
                engine._serve_invocation(
                    inv, "llm.generic", {"inputs": {}, "message": "hi"},
                    context, engine._worker_limits(None),
                    expected_attempt=lease, observation=obs,
                )
            except BaseException as exc:  # noqa: BLE001 - recorded for the assert
                serve_err["e"] = exc

        t = threading.Thread(target=serve, daemon=True)
        t.start()
        assert entered.wait(2.0), "the serve never reached scratch allocation"
        # The node deadline fires here, mid-allocation.
        t0 = time.monotonic()
        await asyncio.to_thread(inv.revoke, "node_timeout")
        revoke_elapsed = time.monotonic() - t0
        assert revoke_elapsed < 0.5, (
            f"revoke waited {revoke_elapsed:.2f}s for the stalled scratch "
            "allocation: a slow mkdtemp held the node deadline hostage"
        )
        t.join(5)
        assert isinstance(serve_err.get("e"), LeaseRevoked), (
            f"the spawn revalidated stale must be refused: {serve_err}"
        )
        # No worker started, so the breaker records nothing.
        assert obs.started is False
        # And the directory it allocated after the revoke was deleted by the
        # spawn itself: a refused spawn leaves nothing behind.
        for path in allocated:
            assert not os.path.exists(path), (
                f"a refused spawn left its scratch behind: {path}"
            )
        await asyncio.to_thread(inv.close)


# =============================================================================
# The window is a rolling window.
# =============================================================================


class TestTheWindowIsARollingWindow:
    @pytest.mark.asyncio
    async def test_a_slow_drip_outside_any_window_never_trips(self):
        """SPEC's contract is N failures inside one window. A per-failure
        TTL refresh turns that into "a chain with no gap over the window",
        so a slow drip — one failure every fifty seconds, forever — trips a
        breaker whose sixty-second window never held five failures."""
        rt = get_runtime()
        cache = rt.cache
        tenant = _u("win")
        ident = _u("drip")
        tripped = False
        for pause in (0.0, 0.6, 0.6):
            if pause:
                await asyncio.sleep(pause)
            tripped, _ = await cache.record_tool_failure(
                ident, failure_threshold=3, window_seconds=1,
                cooldown_seconds=60, tenant_id=tenant,
            )
        assert not tripped and not await _is_open(cache, tenant, ident), (
            "three failures that no one-second window contains together "
            "tripped the breaker: the TTL refresh makes the window a chain"
        )

    @pytest.mark.asyncio
    async def test_a_burst_inside_the_window_trips(self):
        """The control: the same three failures back to back must trip."""
        rt = get_runtime()
        cache = rt.cache
        tenant = _u("win")
        ident = _u("burst")
        tripped = False
        for _ in range(3):
            tripped, _ = await cache.record_tool_failure(
                ident, failure_threshold=3, window_seconds=1,
                cooldown_seconds=60, tenant_id=tenant,
            )
        assert tripped and await _is_open(cache, tenant, ident)


# =============================================================================
# A representation change is a coordinated reset, not a rolling mixed-version
# deploy. The version keeps the boundary crash-safe; it does not unify two
# independent ledgers, so v2 owns its own key and never reads or clears a
# stray legacy one. The cutover procedure itself — draining, then purging the
# superseded namespace — is pinned separately below.
# =============================================================================


class TestTheFailureKeyIsRolloutCompatible:
    @pytest.mark.asyncio
    async def test_a_legacy_string_failure_count_does_not_break_the_new_code(
        self,
    ):
        """Merged main stores the breaker count as a plain string at
        `:failures`, written with `INCR`. The rolling window makes that key a
        sorted set, and the ZSET commands raise `WRONGTYPE` against a string
        — which the breaker preflight does not mask, so every call for a tool
        with 1–4 recent failures would error the moment a new replica rolls
        out. The window uses a versioned key, so the legacy counter is left
        untouched and neither side reads the other's type."""
        rt = get_runtime()
        cache = rt.cache
        tenant = _u("legacy")
        ident = _u("tool")
        # Exactly what a pre-upgrade replica leaves behind.
        await cache.client.set(f"circuit:{tenant}:{ident}:failures", "2", ex=60)

        # Neither call may raise WRONGTYPE against the legacy string.
        is_open, count = await cache.check_circuit_breaker(
            ident, failure_threshold=5, window_seconds=60,
            cooldown_seconds=60, tenant_id=tenant,
        )
        assert is_open is False and count == 0
        tripped, _ = await cache.record_tool_failure(
            ident, failure_threshold=5, window_seconds=60,
            cooldown_seconds=60, tenant_id=tenant,
        )
        assert tripped is False

    @pytest.mark.asyncio
    async def test_v2_never_half_migrates_a_stray_legacy_key(self):
        """Representation isolation, not the cutover procedure. `:failures`
        and `:failures:v2` are independent ledgers, and the v2 code path must
        never opportunistically read a stray legacy counter into its window
        or clear one on success — half-migrating them would rebuild the "one
        success clears only one ledger" partition the two-ledger design
        forbids (SPEC §18.3). So the steady-state v2 code touches v2 alone.

        This is defense-in-depth on the code, independent of deployment. The
        cutover itself does not leave a stray legacy key lying around — it
        purges the superseded namespace, pinned by
        `test_a_reset_purges_the_superseded_history_against_rollback`. Here
        we only prove that if such a key is present, v2 ignores it."""
        rt = get_runtime()
        cache = rt.cache
        tenant = _u("iso")
        ident = _u("tool")
        legacy = f"circuit:{tenant}:{ident}:failures"
        # A stray legacy counter, four short of a trip.
        await cache.client.set(legacy, "4", ex=60)

        # The v2 ledger counts only its own entries, never the legacy four.
        await cache.record_tool_failure(
            ident, failure_threshold=5, window_seconds=60,
            cooldown_seconds=60, tenant_id=tenant,
        )
        is_open, count = await cache.check_circuit_breaker(
            ident, failure_threshold=5, window_seconds=60,
            cooldown_seconds=60, tenant_id=tenant,
        )
        assert count == 1, (
            f"the v2 ledger read the stray legacy counter into its window "
            f"(count={count}); the two must stay independent"
        )
        # A v2 success clears the v2 ledger and leaves the stray key alone.
        await cache.record_tool_success(ident, tenant_id=tenant)
        assert await cache.client.get(legacy) == "4", (
            "v2 code cleared a stray legacy `:failures` counter; the steady "
            "state touches v2 alone and does not half-migrate the two ledgers"
        )
        assert 0 < int(await cache.client.ttl(legacy)) <= 60

    @pytest.mark.asyncio
    async def test_a_reset_purges_the_superseded_history_against_rollback(self):
        """Abandoning the old counter to its TTL is not a reset if the old
        representation can come back inside that window. There is no
        mixed-version serving in this sequence — the deployment contract is
        followed perfectly — and the reset still fails: a rollback re-reads a
        still-live legacy `:failures` and resumes counting from it, opening a
        breaker the reset was supposed to have cleared. So the reset purges
        the superseded representation's failure history rather than trusting
        its TTL. The shared `:open` cooldown and other representations are
        left alone (SPEC §18.3)."""
        rt = get_runtime()
        cache = rt.cache
        tenant = _u("rollback")
        ident = _u("tool")
        legacy = f"circuit:{tenant}:{ident}:failures"
        open_key = f"circuit:{tenant}:{ident}:open"
        # A different tool already on the new representation, which the purge
        # of the *old* one must not touch.
        other = _u("other")
        other_v2 = f"circuit:{tenant}:{other}:failures:v2"

        # The old representation left four failures and an open cooldown.
        await cache.client.set(legacy, "4", ex=60)
        await cache.client.set(open_key, "1", ex=60)
        await cache.client.zadd(other_v2, {"m": 100.0})

        # The coordinated reset onto v2 purges the legacy failure history.
        removed = await cache.purge_breaker_failure_history("legacy")
        assert removed >= 1

        # Rollback to the old representation, well inside the old 60s TTL. Its
        # counter must be gone, so an old-style INCR starts fresh at one.
        resumed = await cache.client.incr(legacy)
        assert resumed == 1, (
            f"the legacy counter survived the reset and resurrected on "
            f"rollback (resumed at {resumed}, not 1): the reset abandoned it "
            f"to its TTL instead of purging it"
        )
        # The shared cooldown survives — an unhealthy tool stays open.
        assert await cache.client.exists(open_key) == 1, (
            "the reset purged the shared `:open` cooldown; it must survive a "
            "representation change"
        )
        # And another representation's history is untouched.
        assert await cache.client.exists(other_v2) == 1, (
            "the reset of one representation purged another representation's "
            "history"
        )

    @pytest.mark.asyncio
    async def test_the_purge_refuses_an_unknown_representation(self):
        """The purge is destructive, so it is fail-closed at the primitive,
        not only at the CLI. It takes a named representation, not a raw key
        suffix, and refuses anything else before building a glob. A raw `*`
        would otherwise expand to `circuit:*:*` and delete legacy history, v2
        history, and the shared `:open` cooldown together (SPEC §18.3)."""
        rt = get_runtime()
        cache = rt.cache
        tenant = _u("failclosed")
        ident = _u("tool")
        legacy = f"circuit:{tenant}:{ident}:failures"
        v2 = f"circuit:{tenant}:{ident}:failures:v2"
        open_key = f"circuit:{tenant}:{ident}:open"
        await cache.client.set(legacy, "4", ex=60)
        await cache.client.zadd(v2, {"m": 100.0})
        await cache.client.set(open_key, "1", ex=60)

        for bad in ("*", "failures*", "open", "", "failures:v2:x", "failures:v2"):
            with pytest.raises(ValueError):
                await cache.purge_breaker_failure_history(bad)

        # Nothing was touched by any refused call.
        assert await cache.client.get(legacy) == "4"
        assert await cache.client.exists(v2) == 1
        assert await cache.client.exists(open_key) == 1


# =============================================================================
# The window's clock is the ledger's, not the serving host's.
# =============================================================================


class TestTheWindowClockIsTheLedgers:
    @pytest.mark.asyncio
    async def test_replica_clock_skew_does_not_drop_a_failure_from_the_window(
        self, monkeypatch
    ):
        """The window is timestamped by the ledger's own clock, not the
        serving host's. Threshold 2, window 60s: two failures moments apart
        are inside one real minute and must trip. But if the first is
        recorded by a replica whose clock runs 100 seconds slow, it lands
        with a past score, and the next — normally-clocked — replica prunes
        everything older than its own now-minus-window before counting,
        dropping that failure early. The breaker then never sees two. Read
        against the ledger's own clock the skew is inert, and the two
        failures trip it.

        (The mirror direction, a fast replica scoring in the future, is
        masked by the set's own window-length TTL — the future entry is
        evicted by real time before it can outlast the window — so the
        early-prune direction is the one a witness can pin.)"""
        import liminallm.storage.redis_cache as rc_mod

        rt = get_runtime()
        cache = rt.cache
        tenant = _u("skew")
        ident = _u("tool")

        real_time = time.time
        slow = {"on": True}

        def maybe_slow():
            return real_time() - 100.0 if slow["on"] else real_time()

        # If the ledger scores against this process's clock, the skew lands
        # in the entry; if it scores against Redis's clock, this is inert.
        monkeypatch.setattr(rc_mod.time, "time", maybe_slow)

        tripped, _ = await cache.record_tool_failure(
            ident, failure_threshold=2, window_seconds=60,
            cooldown_seconds=60, tenant_id=tenant,
        )
        assert not tripped
        slow["on"] = False  # the normally-clocked replica records next
        tripped, _ = await cache.record_tool_failure(
            ident, failure_threshold=2, window_seconds=60,
            cooldown_seconds=60, tenant_id=tenant,
        )
        assert tripped and await _is_open(cache, tenant, ident), (
            "two failures inside one real minute failed to trip: a failure "
            "scored by a clock-skewed replica was pruned early by the next "
            "replica's window math, because the window is read against a "
            "process-local clock, not the ledger's"
        )


# =============================================================================
# Cancellation during preparation stops the provider before it starts.
# =============================================================================


class TestCancelDuringPreparationStopsTheProvider:
    @pytest.mark.asyncio
    async def test_a_cancel_set_during_preparation_never_starts_the_provider(self):
        """The cancel lands while preparation is blocked in the breaker
        check. Preparation then returns normally — and the provider must
        not start on a revoked attempt: the producer gate checks the exact
        attempt's liveness under the invocation lock before anything runs."""
        c, backend = _stream_ctx()
        spy = _SpyCache(_SlowCheckCache(c.cache, 0.5, once=True))
        c.engine.cache = spy
        wf_art = c.wf("llm.generic", max_retries=0)
        cancel_event = asyncio.Event()

        async def trip():
            await asyncio.sleep(0.15)
            cancel_event.set()

        task = asyncio.create_task(trip())
        events = []
        async for e in c.engine.run_streaming(
            wf_art.id, None, "hi", None,
            user_id=c.user.id, tenant_id=c.tenant, cancel_event=cancel_event,
        ):
            events.append(e)
        await task
        assert backend.stream_calls == 0, (
            "the provider started on an attempt the cancel had already "
            "revoked during preparation"
        )
        assert any(e.get("event") == "cancel_ack" for e in events), events[-2:]
        assert not spy.recorded, (
            f"a cancelled preparation wrote to the ledger: {spy.recorded}"
        )


# =============================================================================
# The dispatch resolver is not free clock.
# =============================================================================


class TestDispatchIsNotFreeClock:
    @pytest.mark.asyncio
    async def test_the_transport_decision_spends_the_node_deadline(self):
        """The streamed turn's only resolver runs inside the attempt's
        deadline. A resolver that stalls longer than the node budget must
        time the attempt out — not run on free clock and then hand the
        provider a fresh one."""
        c, backend = _stream_ctx()
        real_resolve = c.engine._resolve_tool
        calls = []

        def slow_once_resolve(name, scope):
            calls.append(1)
            if len(calls) == 1:
                time.sleep(0.6)
            return real_resolve(name, scope)

        c.engine._resolve_tool = slow_once_resolve
        events = await c.run_streaming(c.wf("llm.generic", timeout_ms=300))
        assert backend.stream_calls == 0, (
            "the dispatch resolver ran on free clock and the provider "
            "started after the node deadline was spent"
        )
        assert any(e.get("event") == "error" for e in events)
        assert await c.count(c.ident(), "llm.generic") == 0


# =============================================================================
# Fresh authority is freshly adjudicated.
# =============================================================================


class TestFreshAuthorityIsFreshlyAdjudicated:
    """Re-resolving per attempt is only half of tranche 2's rule. The other
    half is that everything a resolved tool must pass before any body runs —
    the privileged conjunction, the input schema — is decided against the
    spec the attempt actually resolved, not the one the first attempt saw."""

    @pytest.mark.asyncio
    async def test_a_retry_cannot_inherit_preflight_from_a_retired_spec(self):
        """Attempt one runs the caller's own non-privileged spec and fails;
        the spec is retired; attempt two falls through to an admin-owned
        *privileged* spec of the same name. An ordinary caller must be
        refused `forbidden` before that body starts — a preflight carried
        over from the retired spec is an authority bypass, not staleness."""
        rt = get_runtime()
        store = rt.store
        tenant = _u("auth")
        user = store.create_user(email=f"{_u('or')}@t.local", tenant_id=tenant)
        admin = store.create_user(
            email=f"{_u('ad')}@t.local", tenant_id=tenant, role="admin"
        )
        name = _u("authfoo")
        private_art = store.create_artifact(
            "tool", _u("t"),
            {"kind": "tool.spec", "name": name, "handler": "llm.generic"},
            owner_user_id=user.id, visibility="private",
        )
        store.create_artifact(
            "tool", _u("t"),
            {"kind": "tool.spec", "name": name, "handler": "llm.generic",
             "privileged": True},
            owner_user_id=admin.id, visibility="global",
        )
        c, backend = _stream_ctx(fail=True)
        c.tenant, c.user = tenant, user
        wf_art = c.wf(name, max_retries=1, backoff_ms=1)

        def retire_private():
            with store._connect() as conn:
                conn.execute(
                    "DELETE FROM artifact WHERE id = %s", (private_art.id,)
                )

        backend.on_stream = retire_private
        events = await c.run_streaming(wf_art)
        assert backend.stream_calls == 1, (
            f"an ordinary caller ran a privileged spec because the retry "
            f"kept the retired spec's preflight: {backend.stream_calls} calls"
        )
        errs = [e for e in events if e.get("event") == "error"]
        assert errs and errs[-1]["data"].get("code") == "forbidden", errs[-2:]

    @pytest.mark.asyncio
    async def test_a_retry_revalidates_inputs_against_the_current_spec(self):
        """The same shape without the authority load: the spec standing
        behind the retired one requires an input the turn does not carry,
        and the retry must refuse it rather than execute it."""
        rt = get_runtime()
        store = rt.store
        c, backend = _stream_ctx(fail=True)
        name = _u("valfoo")
        plain = c.tool(name)
        store.create_artifact(
            "tool", _u("t"),
            {"kind": "tool.spec", "name": name, "handler": "llm.generic",
             "input_schema": {"type": "object",
                              "required": ["impossible_key"]}},
            owner_user_id=c.user.id, visibility="global",
        )
        wf_art = c.wf(name, max_retries=1, backoff_ms=1)

        def retire_plain():
            with store._connect() as conn:
                conn.execute("DELETE FROM artifact WHERE id = %s", (plain.id,))

        backend.on_stream = retire_plain
        events = await c.run_streaming(wf_art)
        assert backend.stream_calls == 1, (
            f"a retry executed a spec whose input schema refuses this turn: "
            f"{backend.stream_calls} calls"
        )
        errs = [e for e in events if e.get("event") == "error"]
        assert errs and errs[-1]["data"].get("code") == "validation_error", (
            errs[-2:]
        )


# =============================================================================
# Recovery is not tool health.
# =============================================================================


class TestRecoveryIsNotToolHealth:
    @pytest.mark.asyncio
    async def test_a_cancel_mid_read_does_not_complete_the_answer(self):
        """Cancellation stops the pump, and the pump's terminal sentinel must
        not read as a natural end of stream: with the provider blocked
        mid-read, the agent otherwise falls out of its loop, completes the
        partial as a normal answer, and the caller's own cancel clears the
        breaker as a success. Caller abandonment records nothing."""
        c, backend = _stream_ctx()
        backend.generate_with_tools = lambda *a, **k: {}
        backend.block_after_token = threading.Event()
        agent_ident = c.ident("agent.files_v1")
        await c.seed_failures(agent_ident, 4)
        spy = _SpyCache(c.cache)
        c.engine.cache = spy
        wf_art = c.wf("agent.files_v1", max_retries=0)
        c.engine._serve_invocation = lambda *a, **k: {
            "messages": [], "usage": {}, "context_snippets": [],
            "tool_calls": [], "artifacts": [], "injection_findings": [],
        }
        c.engine._build_agent_context = lambda *a, **k: (
            [], [{"name": "t"}], "", [], []
        )
        cancel_event = asyncio.Event()
        events = []
        async for e in c.engine.run_streaming(
            wf_art.id, None, "hi", None,
            user_id=c.user.id, tenant_id=c.tenant, cancel_event=cancel_event,
        ):
            events.append(e)
            if e.get("event") == "token":
                cancel_event.set()
        assert not any(
            e.get("event") == "message_done" and (e.get("data") or {}).get("content")
            for e in events
        ), "a cancelled turn was completed as a normal answer"
        assert await c.count(agent_ident) == 4, (
            "the caller's own cancel cleared the tool's failure count"
        )
        assert not [r for r in spy.recorded if r[0] == "success"], (
            "a cancelled attempt recorded a breaker success"
        )

    @pytest.mark.asyncio
    async def test_a_cancel_between_events_does_not_record(self):
        """The other cancel arrival: the pump has started — `started` is
        marked — and the cancel is discovered when the next event lands,
        not by the stop cutting a blocked read. The drain then runs its
        no-answer tail with the acknowledgment in hand, and the caller's
        cancel must still leave the ledger untouched."""
        c, backend = _stream_ctx()
        backend.gap = True
        ident = c.ident()
        await c.seed_failures(ident, 4)
        spy = _SpyCache(c.cache)
        c.engine.cache = spy
        wf_art = c.wf("llm.generic", max_retries=0)
        cancel_event = asyncio.Event()
        events = []
        async for e in c.engine.run_streaming(
            wf_art.id, None, "hi", None,
            user_id=c.user.id, tenant_id=c.tenant, cancel_event=cancel_event,
        ):
            events.append(e)
            if e.get("event") == "token":
                cancel_event.set()
        assert any(e.get("event") == "cancel_ack" for e in events), events[-2:]
        assert await c.count(ident) == 4, (
            "a cancel discovered between events changed the ledger"
        )
        assert not spy.recorded, spy.recorded

    @pytest.mark.asyncio
    async def test_the_drain_distinguishes_interruption_from_truncation(self):
        """The attempt's own contract, probed at the seam: the running paths
        stop iterating at a forwarded `cancel_ack`, so the drain's tail
        only ever runs post-acknowledgment for a consumer that drains to
        the end — and for that consumer the distinction must hold: an
        acknowledged interruption records nothing, a truncation records
        the failure."""
        from liminallm.service.node_attempt import StreamedNodeAttempt

        async def stream(events):
            for e in events:
                yield e

        acked = StreamedNodeAttempt(
            stream([
                {"event": "token", "data": "x"},
                {"event": "cancel_ack", "data": {}},
            ]),
            finalize=lambda r: (r, None),
        )
        acked.breaker.started = True
        _ = [e async for e in acked.events()]
        assert acked.breaker.outcome is None, (
            "an acknowledged interruption was recorded as a truncation "
            "failure"
        )

        truncated = StreamedNodeAttempt(
            stream([{"event": "token", "data": "x"}]),
            finalize=lambda r: (r, None),
        )
        truncated.breaker.started = True
        _ = [e async for e in truncated.events()]
        assert truncated.breaker.outcome == "failure", (
            "a stream that ended with no answer and no acknowledgment "
            "recorded nothing"
        )

    @pytest.mark.asyncio
    async def test_a_salvaged_partial_answer_still_records_the_failure(self):
        """The agent keeps a partial answer when its final stream dies after
        a token — the user keeps what was on their screen. The breaker must
        still see the provider failure: the salvage is user-facing recovery,
        and a partial `tool_result` that overwrites the observation with
        success turns five provider deaths into a clean bill of health."""
        c, backend = _stream_ctx(fail_after_token=True)
        backend.generate_with_tools = lambda *a, **k: {}
        await c.seed_failures(c.ident("agent.files_v1"), 4)
        wf_art = c.wf("agent.files_v1", max_retries=0)
        c.engine._serve_invocation = lambda *a, **k: {
            "messages": [], "usage": {}, "context_snippets": [],
            "tool_calls": [], "artifacts": [], "injection_findings": [],
        }
        c.engine._build_agent_context = lambda *a, **k: (
            [], [{"name": "t"}], "", [], []
        )
        events = await c.run_streaming(wf_art)
        done = [e for e in events if e.get("event") == "message_done"]
        assert done and done[-1]["data"].get("content") == "x", (
            "the salvage itself must keep working: the partial answer "
            "reaches the client"
        )
        assert await c.is_open(c.ident("agent.files_v1")), (
            "a salvaged partial answer recorded the provider failure as a "
            "breaker success"
        )


# =============================================================================
# Streamed planning is not the serve.
# =============================================================================


class TestStreamedPlanningIsNotTheServe:
    @pytest.mark.asyncio
    async def test_a_streamed_provider_hang_records_a_failure(self):
        """The other side of the boundary: the provider was called and then
        hung past the node deadline. That is the hang the breaker exists to
        stop, and it must count exactly as the blocking sibling counts."""
        c, backend = _stream_ctx(stall=True)
        r = await c.run_streaming(c.wf("llm.generic", timeout_ms=300))
        assert any(e.get("event") == "error" for e in r)
        assert backend.stream_calls == 1
        assert await c.count(c.ident(), "llm.generic") == 1

    @pytest.mark.asyncio
    async def test_an_agent_serve_hang_records_a_failure(self):
        """The agent body reaches its worker serve — the backend declares
        tool support and the context assembly offers a tool, else the body
        delegates to the plain LLM node before any serve exists."""
        c, backend = _stream_ctx()
        # `LLMService.supports_tools` asks for a callable, not a flag; the
        # spy serve below means it is never actually called.
        backend.generate_with_tools = lambda *a, **k: {}
        wf_art = c.wf("agent.files_v1", timeout_ms=300, max_retries=0)

        def hung_serve(*a, **k):
            # Stands for a worker that spawned and then hung: the real serve
            # marks `started` once the spawn has registered the child, so
            # this double does the same before it stalls.
            if k.get("observation") is not None:
                k["observation"].started = True
            time.sleep(2)
            return {}

        c.engine._serve_invocation = hung_serve
        c.engine._build_agent_context = lambda *a, **k: (
            [], [{"name": "t"}], "", [], []
        )
        events = await c.run_streaming(wf_art)
        assert any(e.get("event") == "error" for e in events)
        assert await c.count(c.ident("agent.files_v1"), "agent.files_v1") == 1

    @pytest.mark.asyncio
    async def test_a_deadline_spent_in_streamed_planning_records_nothing(self):
        """The streamed complement of the blocking witness above: the agent
        body assembles grounding and context before any worker or provider
        runs, and a node deadline spent there proves nothing about the
        tool. `started` must mark the serve, not the body's first line."""
        c, backend = _stream_ctx()
        wf_art = c.wf("agent.files_v1", timeout_ms=300, max_retries=0)
        serve_calls = []

        def spy_serve(*a, **k):
            serve_calls.append(1)
            return {}

        def slow_grounding(*a, **k):
            time.sleep(2)
            return ([], "")

        c.engine._serve_invocation = spy_serve
        c.engine._explicit_context_grounding = slow_grounding
        events = await c.run_streaming(wf_art)
        assert any(e.get("event") == "error" for e in events)
        assert not serve_calls, "the worker serve ran despite the stalled plan"
        assert backend.stream_calls == 0, "the provider ran despite the stalled plan"
        assert await c.count(c.ident("agent.files_v1"), "agent.files_v1") == 0, (
            "a deadline spent in streamed planning was charged to the tool"
        )


# =============================================================================
# The direct invocation is the same ledger.
# =============================================================================


class TestTheDirectInvocationIsTheSameLedger:
    """`POST /v1/tools/{id}/invoke` starts tool invocations like any node
    attempt, so it checks the same breaker and records through the same
    writer — otherwise the direct API is an unmetered way to hammer a tool
    the breaker has already cut off for every workflow."""

    def _descriptor(self, c):
        from liminallm.service.tool_namespace import ToolResolutionScope

        return c.engine._resolve_tool(
            "llm.generic", ToolResolutionScope("private", c.user.id, c.tenant)
        )

    async def _invoke(self, c):
        return await c.engine.invoke_tool(
            self._descriptor(c), {"message": "hi"},
            user_id=c.user.id, tenant_id=c.tenant,
        )

    @pytest.mark.asyncio
    async def test_a_direct_raw_error_records_one_failure(self):
        c = _Ctx()
        c.engine._tool_llm_generic = _err
        r = await self._invoke(c)
        assert r.get("status") == "error"
        assert await c.count(c.ident(), "llm.generic") == 1

    @pytest.mark.asyncio
    async def test_a_direct_success_clears_the_failure_count(self):
        c = _Ctx()
        await c.seed_failures(c.ident(), 3)
        c.engine._tool_llm_generic = _ok
        r = await self._invoke(c)
        assert r.get("status") != "error"
        assert await c.count(c.ident(), "llm.generic") == 0

    @pytest.mark.asyncio
    async def test_a_direct_input_refusal_records_nothing(self):
        c = _Ctx()
        art = c.tool(
            name := _u("directin"),
            input_schema={"type": "object", "required": ["impossible_key"]},
        )
        ran = []
        c.engine._tool_llm_generic = lambda *a, **k: (ran.append(1), _ok())[1]
        from liminallm.service.tool_namespace import ToolResolutionScope

        d = c.engine._resolve_tool(
            name, ToolResolutionScope("private", c.user.id, c.tenant)
        )
        r = await c.engine.invoke_tool(
            d, {"message": "hi"}, user_id=c.user.id, tenant_id=c.tenant
        )
        assert r.get("error") == "validation_error", r
        assert not ran
        assert await c.count(name, art.id) == 0

    @pytest.mark.asyncio
    async def test_direct_admission_order_matches_the_transports(self):
        """One admission order everywhere: preflight, then breaker. With both
        grounds to refuse, the direct endpoint gives the same answer a
        workflow attempt gives — validation, not circuit-open."""
        c = _Ctx()
        art = c.tool(
            name := _u("ordfoo"),
            input_schema={"type": "object", "required": ["impossible_key"]},
        )
        await c.seed_failures(art.id, 5)
        assert await c.is_open(art.id)
        from liminallm.service.tool_namespace import ToolResolutionScope

        d = c.engine._resolve_tool(
            name, ToolResolutionScope("private", c.user.id, c.tenant)
        )
        r = await c.engine.invoke_tool(
            d, {"message": "hi"}, user_id=c.user.id, tenant_id=c.tenant
        )
        assert r.get("error") == "validation_error", (
            f"the direct endpoint consulted the breaker before validation: "
            f"{r.get('error')}"
        )

    @pytest.mark.asyncio
    async def test_an_open_breaker_refuses_the_direct_invocation(self):
        c = _Ctx()
        await c.seed_failures(c.ident(), 5)
        assert await c.is_open(c.ident())
        spy = _SpyCache(c.cache)
        c.engine.cache = spy
        ran = []
        c.engine._tool_llm_generic = lambda *a, **k: (ran.append(1), _ok())[1]
        r = await self._invoke(c)
        assert r.get("error") == "circuit_breaker_open", r
        assert not ran, "an open breaker did not stop the direct invocation"
        assert not spy.recorded, (
            f"a refused direct invocation wrote to the ledger: {spy.recorded}"
        )


# =============================================================================
# Breaker identity is the resolved tool, not the reference spelling.
# =============================================================================


class TestBreakerIdentityIsTheResolvedTool:
    @pytest.mark.asyncio
    async def test_two_specs_sharing_a_spelling_do_not_share_a_breaker(self):
        """Alice's private `foo` and Bob's private `foo` are different tools
        that happen to share a name. Alice's failures must not cut Bob off."""
        tenant = _u("bkid")
        rt = get_runtime()
        alice = rt.store.create_user(email=f"{_u('al')}@t.local", tenant_id=tenant)
        bob = rt.store.create_user(email=f"{_u('bo')}@t.local", tenant_id=tenant)
        ca = _Ctx(tenant=tenant, user=alice)
        cb = _Ctx(tenant=tenant, user=bob)
        name = _u("foo")
        ca.tool(name)
        cb.tool(name)
        wf_a = ca.wf(name)
        wf_b = cb.wf(name)

        ca.engine._tool_llm_generic = _err
        for _ in range(5):
            await ca.run(wf_a)

        ran_alice = []
        ca.engine._tool_llm_generic = lambda *a, **k: (ran_alice.append(1), _ok())[1]
        r_alice = await ca.run(wf_a)
        assert r_alice.get("status") == "error" and not ran_alice, (
            "five failures did not open the breaker for the spec that failed"
        )

        ran_bob = []
        cb.engine._tool_llm_generic = lambda *a, **k: (ran_bob.append(1), _ok())[1]
        r_bob = await cb.run(wf_b)
        assert ran_bob and r_bob.get("status") != "error", (
            "Alice's failing spec opened the breaker for Bob's healthy one; "
            "the ledger is keyed by spelling, not by the resolved tool"
        )

    @pytest.mark.asyncio
    async def test_two_spellings_of_one_resolved_tool_share_one_breaker(self):
        """A node with no `tool` runs the default LLM tool. Same resolved
        tool, same breaker: the implicit spelling is not a bypass.

        Admission requires `tool` on a `tool_call` node, so the schema is
        handed to the engine the way legacy rows reach it — through what the
        store vouches for — rather than created as a new artifact."""
        from unittest.mock import patch

        from liminallm.service.tool_namespace import (
            ResolvedWorkflow,
            ToolResolutionScope,
        )

        c = _Ctx()
        await c.seed_failures(c.ident(), 5)
        assert await c.is_open(c.ident())
        ran = []
        c.engine._tool_llm_generic = lambda *a, **k: (ran.append(1), _ok())[1]
        loaded = ResolvedWorkflow(
            _wf(None), ToolResolutionScope("private", c.user.id, c.tenant)
        )
        with patch.object(c.store, "get_latest_workflow", return_value=loaded):
            r = await c.engine.run(
                _u("legacy"), None, "hi", None,
                user_id=c.user.id, tenant_id=c.tenant,
            )
        assert r.get("status") == "error" and not ran, (
            "an open breaker refused the explicit spelling and ran the "
            "implicit one; both resolve to the same tool"
        )
