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

import time
import uuid

import pytest

from liminallm.service.invocation import LeaseRevoked
from liminallm.service.llm import LLMService
from liminallm.service.runtime import get_runtime
from liminallm.service.workflow import WorkflowEngine


def _u(p):
    return f"{p}_{uuid.uuid4().hex[:8]}"


def _wf(tool_name, *, max_retries=0, timeout_ms=None):
    node = {"id": "call", "type": "tool_call", "next": "fin",
            "max_retries": max_retries}
    if tool_name is not None:
        node["tool"] = tool_name
    if timeout_ms is not None:
        node["timeout_ms"] = timeout_ms
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
        raw = await cache.client.get(f"circuit:{tenant}:{ident}:failures")
        if isinstance(raw, bytes):
            raw = raw.decode()
        total += int(raw) if raw else 0
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
    failure before any token when told to fail."""

    supports_stream_cancel = True

    def __init__(self, fail=False):
        self.fail = fail
        self.stream_calls = 0

    def generate(self, messages, adapters, *, user_id=None):
        return {"content": "whole answer", "usage": {}}

    def generate_stream(self, messages, adapters, *, user_id=None):
        self.stream_calls += 1

        def gen():
            if self.fail:
                raise RuntimeError("stream boom")
            yield {"event": "token", "data": "x"}
            yield {"event": "message_done", "data": {"content": "x", "usage": {}}}
        return gen()


def _stream_ctx(fail=False):
    backend = _StreamBackend(fail=fail)
    return _Ctx(llm=LLMService("test-model", backend=backend)), backend


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
        feed the breaker."""
        c = _Ctx()
        await c.seed_failures(c.ident(), 5)
        assert await c.is_open(c.ident())
        ran = []
        c.engine._tool_llm_generic = lambda *a, **k: (ran.append(1), _ok())[1]
        r = await c.run(c.wf("llm.generic"))
        assert r.get("status") == "error" and not ran
        assert await c.count(c.ident(), "llm.generic") == 0

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
    async def test_a_caller_revocation_records_nothing(self):
        """The caller walked away mid-serve. The tool was not proven
        unhealthy, and a cancel habit must not open the tenant's breaker."""
        c = _Ctx()

        def revoked(*a, **k):
            raise LeaseRevoked("caller cancelled")

        c.engine._serve_invocation = revoked
        r = await c.run(c.wf("llm.generic"))
        assert r.get("status") == "error"
        assert await c.count(c.ident(), "llm.generic") == 0


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
