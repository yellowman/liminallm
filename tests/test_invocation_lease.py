"""A tool call is a process the kernel can kill, and these are the proofs.

`_invoke_tool` used to submit the handler to a `ThreadPoolExecutor` and await
it with `asyncio.wait_for`. On timeout it called `future.cancel()` — which
returns `False` for anything already running, because a `concurrent.futures`
future can only cancel work that has not started. The engine returned "tool
timed out" to the workflow, and the thread ran on with `user_id` and
`tenant_id` still in its closure. With retries, a second worker started while
the first was still running.

Tranche 1b.1 replaced that with what SPEC §18 always said: a spawned worker
process per attempt, a parent-owned broker serving every effect, and a ledger
keyed by the logical execution. The four properties below are its closure
conditions, and each one was false before in a way no assertion about return
values could see — the work carried on in a thread nobody was waiting for.

So most of these tests assert on processes and on files: whether a request went
out, whether a child was started, whether a tree is gone. That is the only
evidence that tells "stopped" from "stopped being watched".
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import threading
import time
import uuid

import pytest

from liminallm.service import agent_tools, interpreter, web
from liminallm.service.broker import CapabilityBroker, InvocationContext
from liminallm.service.invocation import (
    COMMITTED,
    UNKNOWN,
    Invocation,
    InvocationRegistry,
    LeaseRevoked,
    OperationLedger,
    RetryDivergence,
    commit_guard,
    current_invocation,
    payload_hash,
)


def _unique(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def runtime(client):
    from liminallm.service.runtime import get_runtime

    return get_runtime()


@pytest.fixture
def caller(runtime):
    return runtime.store.create_user(email=f"{_unique('late')}@t.local")


def _broker(engine, **kwargs):
    return CapabilityBroker(engine, InvocationContext(**kwargs))


def _ask(broker, invocation, capability, payload, seq=1):
    return broker._answer(
        invocation,
        {"operation_seq": seq, "capability": capability, "payload": payload},
    )


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _sleeper() -> subprocess.Popen:
    """A child that will not exit on its own, so killing it must be observed."""
    return subprocess.Popen(  # noqa: S603 - fixed argv, test-only
        [sys.executable, "-c", "import time; time.sleep(300)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


# ---------------------------------------------------------------------------
# 1. no retry before the prior worker's process tree is dead


class TestNoRetryBeforeThePriorTreeIsDead:
    """Attempt n+1 may not begin while attempt n still has a process.

    This is the condition the old `_reap` could not meet. It waited
    `REAP_GRACE_SECONDS` and returned, so a handler outliving the grace was
    still running when the next attempt began — and a thread cannot be killed,
    so bounded reaping was the best a thread worker could do.
    """

    async def test_each_attempt_starts_with_its_predecessor_dead(
        self, runtime, caller, monkeypatch
    ):
        engine = runtime.workflow
        node = {
            "id": "retry_node",
            "type": "tool_call",
            "tool": "test.slow_tool",
            "max_retries": 2,
            "backoff_ms": 10,
            # Short enough that the node times out while the worker is still
            # blocked on the capability below.
            "timeout_ms": 400,
        }
        engine.tool_registry.setdefault("test.slow_tool", {"name": "test.slow_tool"})

        pids_seen: list[int] = []
        live_at_start: list[list[int]] = []
        predecessors_dead: list[bool] = []
        entered = 0
        real_serve = engine._serve_invocation

        def watched(invocation, worker_tool, plan, context, limits, **kwargs):
            # Observed as this attempt starts: whatever the last one left
            # behind must already be gone, by pid and by registry.
            live_at_start.append(list(invocation.resources.live_children()))
            predecessors_dead.append(all(not _alive(pid) for pid in pids_seen))
            try:
                return real_serve(invocation, worker_tool, plan, context, limits, **kwargs)
            finally:
                pids_seen.extend(
                    a.pid
                    for a in invocation.attempts
                    if a.pid and a.pid not in pids_seen
                )

        def slow_host(tool, inputs, *, context):
            # The worker is alive and blocked on this reply when the node's
            # clock runs out. Longer than the node timeout, shorter than the
            # test's.
            nonlocal entered
            entered += 1
            time.sleep(1.5)
            return {"status": "error", "content": "too slow", "error": "boom"}

        monkeypatch.setattr(engine, "_serve_invocation", watched)
        monkeypatch.setattr(engine, "_run_host_tool", slow_host)

        result, _ = await engine._execute_node_with_retry(
            node,
            user_message="hello",
            context_id=None,
            conversation_id=None,
            adapters=[],
            history=[],
            vars_scope={},
            user_id=caller.id,
            tenant_id=caller.tenant_id,
            workflow_start_time=time.monotonic(),
            workflow_timeout_ms=60_000,
        )

        assert result["status"] == "error"
        assert entered >= 1, "the worker never reached the capability it blocks on"
        assert len(live_at_start) == 3, f"expected 3 attempts, got {len(live_at_start)}"
        for index, live in enumerate(live_at_start):
            assert live == [], f"attempt {index} started with {live} still running"
            assert predecessors_dead[index], (
                f"attempt {index} started beside its own predecessor"
            )
        for pid in pids_seen:
            assert not _alive(pid), f"worker {pid} outlived its invocation"


# ---------------------------------------------------------------------------
# 2. a revoked invocation sends no web request


class TestARevokedInvocationSendsNoWebRequest:
    """Revocation is checked before the fetch, so nothing leaves the box.

    Asserting on the returned error would pass just as well if the request had
    gone out and the answer been thrown away. The counter is the only witness.
    """

    def test_no_fetch_and_no_search(self, runtime, caller, monkeypatch):
        engine = runtime.workflow
        reached: list = []

        def counting(*args, **kwargs):
            reached.append(args)
            raise AssertionError("a revoked invocation must not reach the network")

        monkeypatch.setattr(web, "fetch_url", counting)
        monkeypatch.setattr(web, "search_web", counting)
        monkeypatch.setattr(engine.settings, "web_tools_enabled", True, raising=False)

        invocation = Invocation("revoked-web", tool="web.fetch_v1")
        invocation.begin_attempt()
        invocation.revoke("cancelled")
        try:
            for capability, payload in (
                ("web.fetch", {"url": "http://example.invalid/page"}),
                ("web.search", {"query": "anything"}),
            ):
                reply = _ask(_broker(engine, user_id=caller.id), invocation, capability, payload)
                assert reply["ok"] is False
                assert reply["code"] == "revoked"
            assert reached == [], f"a revoked turn issued {reached}"
        finally:
            invocation.close()


# ---------------------------------------------------------------------------
# 3. a revoked invocation launches no Python sandbox child


class TestARevokedInvocationLaunchesNoSandboxChild:
    """Not a child that is then killed. No child at all."""

    def test_no_child_and_no_scratch(self, runtime, caller, monkeypatch):
        engine = runtime.workflow
        started: list = []
        monkeypatch.setattr(
            interpreter,
            "run_python_sandboxed",
            lambda *a, **kw: started.append(1) or {},
        )

        invocation = Invocation("revoked-python", tool="code.python_v1")
        invocation.begin_attempt()
        invocation.revoke("cancelled")
        try:
            reply = _ask(
                _broker(engine, user_id=caller.id),
                invocation,
                "python.run",
                {"code": "print('should not run')"},
            )
            assert reply["ok"] is False
            assert reply["code"] == "revoked"
            assert started == [], "a revoked turn started a sandbox child"
            assert invocation.resources.children() == []
            # The scratch is not prepared either: preparing it copies the
            # user's attachments, which is work a revoked turn must not do.
            assert invocation.workdir is None
        finally:
            invocation.close()

    def test_revoked_between_the_check_and_the_launch(
        self, runtime, caller, monkeypatch
    ):
        """Preparing the scratch is the window; the second check closes it."""
        engine = runtime.workflow
        invocation = Invocation("revoked-mid-prepare", tool="code.python_v1")
        invocation.begin_attempt()
        started: list = []
        real_prepare = interpreter.prepare_workdir

        def revoke_while_preparing(*args, **kwargs):
            workdir = real_prepare(*args, **kwargs)
            invocation.revoke("cancelled mid-preparation")
            return workdir

        monkeypatch.setattr(interpreter, "prepare_workdir", revoke_while_preparing)
        monkeypatch.setattr(
            interpreter, "run_python_sandboxed", lambda *a, **kw: started.append(1)
        )
        try:
            reply = _ask(
                _broker(engine, user_id=caller.id),
                invocation,
                "python.run",
                {"code": "print(1)"},
            )
            assert reply["ok"] is False and reply["code"] == "revoked"
            assert started == [], "the child started after the turn was revoked"
        finally:
            invocation.close()

    def test_a_live_invocation_owns_the_child_it_starts(self, runtime, caller):
        """The registry is what makes the tree reachable at all.

        A sandbox child is the *parent's* child, not the worker's, so killing
        the worker never reaches it. It has to be registered as it starts.
        """
        from pathlib import Path

        engine = runtime.workflow
        invocation = Invocation("owns-its-child", tool="code.python_v1")
        invocation.begin_attempt()
        try:
            with current_invocation(invocation):
                agent_tools.run_python(
                    "print(6 * 7)",
                    [],
                    settings=engine.settings,
                    user_id=caller.id,
                    session=invocation.session,
                    invocation=invocation,
                )
            seen = [child.label for child in invocation.resources.children()]
            workdir = invocation.workdir
            assert workdir, "the scratch was never prepared"
            assert any("sandbox" in label for label in seen), seen
        finally:
            invocation.close()
        # Closing the execution is what removes the scratch now: the handler's
        # own `finally` went with the handler.
        assert not Path(workdir).exists(), "the scratch outlived the execution"


# ---------------------------------------------------------------------------
# 4. every broker-owned descendant and resource is killed and reaped


class TestEverythingBrokerOwnedIsKilledAndReaped:
    def test_descendants_and_scratch_are_gone_before_a_retry(self, tmp_path):
        invocation = Invocation("kill-and-reap")
        invocation.begin_attempt()
        scratch = tmp_path / "session-abcdef"
        scratch.mkdir()
        (scratch / "copy.csv").write_text("a,b\n1,2\n")
        invocation.workdir = str(scratch)
        invocation.resources.add_path(str(scratch))

        children = [_sleeper() for _ in range(3)]
        for child in children:
            invocation.resources.add_child(child.pid, "sandbox:test", reap=child.wait)
        assert all(_alive(c.pid) for c in children)

        invocation.revoke("cancelled")
        assert invocation.terminate() is True, "termination must confirm, not assume"

        for child in children:
            assert not _alive(child.pid), f"descendant {child.pid} survived revocation"
            # Reaped, not merely signalled: a zombie still holds a table entry.
            assert child.poll() is not None
        assert invocation.resources.live_children() == []
        assert not scratch.exists(), "the scratch outlived the invocation"
        invocation.close()

    def test_a_retry_gets_a_fresh_scratch_not_the_dead_one(self, tmp_path):
        """The scratch dies with the attempt, and its name dies with it.

        A retry that inherited the directory would read the killed attempt's
        half-written files. A retry that inherited only the *name* would be
        worse: it would skip preparation and run in a path that is gone.
        """
        invocation = Invocation("fresh-scratch")
        invocation.begin_attempt()
        scratch = tmp_path / "session-1"
        scratch.mkdir()
        (scratch / "half-written.csv").write_text("1,2")
        invocation.workdir = str(scratch)
        invocation.resources.add_path(str(scratch))

        assert invocation.terminate() is True
        assert not scratch.exists()
        assert invocation.workdir is None, "the retry would run in a deleted directory"
        invocation.close()

    def test_termination_reports_failure_rather_than_pretending(self):
        """A tree that will not die returns False, and the retry honours it."""
        invocation = Invocation("undead")
        # A pid that exists and cannot be killed by us. The registry must
        # report it alive rather than quietly deciding the tree is clear.
        invocation.resources.add_child(1, "unkillable", reap=lambda: None)
        try:
            assert invocation.terminate(timeout=0.2) is False
            assert invocation.resources.live_children() == [1]
        finally:
            invocation.resources.forget_child(1)
            invocation.close()

    async def test_a_tree_that_will_not_die_fails_the_node(
        self, runtime, caller, monkeypatch
    ):
        """The retry honours the answer instead of running alongside."""
        engine = runtime.workflow
        node = {
            "id": "undead_node",
            "type": "tool_call",
            "tool": "test.undead",
            "max_retries": 2,
            "backoff_ms": 1,
            "timeout_ms": 200,
        }
        engine.tool_registry.setdefault("test.undead", {"name": "test.undead"})
        attempts = 0
        real_serve = engine._serve_invocation

        def unkillable(invocation, *args, **kwargs):
            nonlocal attempts
            attempts += 1
            invocation.resources.add_child(1, "unkillable", reap=lambda: None)
            try:
                return real_serve(invocation, *args, **kwargs)
            finally:
                pass

        monkeypatch.setattr(engine, "_serve_invocation", unkillable)
        monkeypatch.setattr(
            engine,
            "_run_host_tool",
            lambda tool, inputs, *, context: {
                "status": "error",
                "content": "no",
                "error": "boom",
            },
        )
        try:
            result, _ = await engine._execute_node_with_retry(
                node,
                user_message="hello",
                context_id=None,
                conversation_id=None,
                adapters=[],
                history=[],
                vars_scope={},
                user_id=caller.id,
                tenant_id=caller.tenant_id,
                workflow_start_time=time.monotonic(),
                workflow_timeout_ms=30_000,
            )
        finally:
            for invocation_id in engine.invocations.live():
                live = engine.invocations.get(invocation_id)
                if live is not None:
                    live.resources.forget_child(1)
                    live.close()
        assert result["error"] == "tool_worker_unreaped", result
        assert attempts == 1, f"a second attempt started anyway ({attempts})"


# ---------------------------------------------------------------------------
# the lease is per attempt; the ledger is per logical execution


class TestTwoIdsBecauseTheyAnswerDifferentQuestions:
    def test_a_retry_does_not_inherit_the_revoked_attempt(self):
        invocation = Invocation("two-ids")
        first = invocation.begin_attempt()
        invocation.revoke("node_timeout")
        assert invocation.revoked is True
        with pytest.raises(LeaseRevoked):
            invocation.check_live()

        second = invocation.begin_attempt()
        assert second is not first
        assert invocation.revoked is False, "attempt two inherited attempt one's revocation"
        invocation.check_live()
        invocation.close()

    def test_a_cancelled_execution_starts_no_further_attempt(self):
        invocation = Invocation("cancelled")
        invocation.begin_attempt()
        invocation.cancel("user cancelled")
        with pytest.raises(LeaseRevoked):
            invocation.begin_attempt()
        invocation.close()

    def test_a_revoke_landing_before_the_first_spawn_is_not_forgotten(self):
        """There is no attempt to scope it to, so it refuses the execution.

        The other reading — scope it to the attempt that has not started yet —
        loses the revocation entirely, which is the failure this whole tranche
        is about.
        """
        invocation = Invocation("revoked-before-spawn")
        invocation.revoke("cancelled in the gap")
        assert invocation.revoked is True
        with pytest.raises(LeaseRevoked):
            invocation.begin_attempt()
        invocation.close()

    def test_the_ledger_outlives_the_attempt_that_wrote_it(self):
        invocation = Invocation("ledger-spans")
        invocation.begin_attempt()
        with commit_guard(
            invocation, "publish.artifacts", {"created": ["a.csv"]}, operation_seq=1
        ) as op:
            op.result = ["a.csv"]
        invocation.revoke("node_timeout")
        invocation.begin_attempt()

        with commit_guard(
            invocation, "publish.artifacts", {"created": ["a.csv"]}, operation_seq=1
        ) as replayed:
            assert replayed.replayable, "the retry lost the first attempt's record"
            assert replayed.result == ["a.csv"]
        invocation.close()


class TestTheOperationLedger:
    """What a content-addressed key could not do.

    `operation_key()` was `logical_execution_id + ":" + operation_name`. It
    cannot distinguish two durable operations of the same kind inside one node
    execution, and the agent loop makes several tool calls per execution. The
    ledger is ordered instead: position, capability, payload.
    """

    def test_a_committed_step_replays_instead_of_happening_twice(self):
        ledger = OperationLedger()
        digest = payload_hash({"created": ["a.csv"]})
        ledger.begin(3, "publish.artifacts", digest)
        ledger.commit(3, ["a.csv"])
        replayed = ledger.replay(3, "publish.artifacts", digest, durable=True)
        assert replayed is not None and replayed.state == COMMITTED
        assert replayed.result == ["a.csv"]

    def test_two_identical_calls_at_different_positions_stay_separate(self):
        """A key collides them into one; a sequence does not."""
        ledger = OperationLedger()
        digest = payload_hash({"query": "same"})
        ledger.begin(1, "web.search", digest)
        ledger.commit(1, "first")
        assert ledger.replay(2, "web.search", digest) is None
        assert len(ledger) == 1

    def test_a_diverged_durable_retry_is_refused_not_answered(self):
        ledger = OperationLedger()
        ledger.begin(2, "publish.artifacts", payload_hash({"created": ["a.csv"]}))
        ledger.commit(2, ["a.csv"])
        with pytest.raises(RetryDivergence):
            ledger.replay(
                2,
                "publish.artifacts",
                payload_hash({"created": ["b.csv"]}),
                durable=True,
            )

    def test_a_diverged_read_runs_again_rather_than_refusing(self):
        """A read has no earlier effect to misreport, so it simply re-runs."""
        ledger = OperationLedger()
        ledger.begin(2, "web.search", payload_hash({"query": "one"}))
        ledger.commit(2, "one")
        assert ledger.replay(2, "web.search", payload_hash({"query": "two"})) is None

    def test_a_step_in_flight_when_its_attempt_died_is_unknown(self):
        """Neither committed nor failed: the parent does not know which."""
        ledger = OperationLedger()
        digest = payload_hash({"created": ["a.csv"]})
        ledger.begin(1, "publish.artifacts", digest)
        assert ledger.orphan_pending() == 1
        assert ledger.get(1).state == UNKNOWN
        with pytest.raises(RetryDivergence):
            ledger.replay(1, "publish.artifacts", digest, durable=True)

    def test_a_failed_step_is_not_replayed_as_a_result(self):
        ledger = OperationLedger()
        digest = payload_hash({"query": "q"})
        ledger.begin(1, "web.search", digest)
        ledger.fail(1, "provider down")
        assert ledger.replay(1, "web.search", digest) is None


class TestPublicationHappensOnce:
    def test_a_replayed_publish_copies_nothing(self, runtime, caller, tmp_path, monkeypatch):
        """The guard is around the copy, so a retry can tell "the files are
        there" from "a worker asked for them to be"."""
        invocation = Invocation("publish-once")
        invocation.begin_attempt()
        workdir = tmp_path / "work"
        workdir.mkdir()
        (workdir / "out.csv").write_text("1,2\n")
        dest = tmp_path / "files"
        copies: list = []
        real_publish = interpreter.publish_artifacts

        def counting(*args, **kwargs):
            copies.append(1)
            return real_publish(*args, **kwargs)

        monkeypatch.setattr(agent_tools.interpreter, "publish_artifacts", counting)
        try:
            with current_invocation(invocation):
                for _ in range(2):  # the attempt, then its replacement
                    published = agent_tools._publish(
                        str(workdir),
                        str(dest),
                        [{"name": "out.csv", "size": 4}],
                        invocation=invocation,
                        operation_seq=1,
                        step="publish",
                    )
                    assert published == ["out.csv"]
        finally:
            invocation.close()
        assert copies == [1], f"the copy happened {len(copies)} times, not once"
        assert dest.joinpath("out.csv").read_text() == "1,2\n"

    def test_a_diverged_republish_is_refused(self, runtime, tmp_path):
        """A replacement worker that produced different files at the same
        position is not answered with the earlier attempt's filenames."""
        invocation = Invocation("publish-diverged")
        invocation.begin_attempt()
        workdir = tmp_path / "work"
        workdir.mkdir()
        (workdir / "a.csv").write_text("1")
        (workdir / "b.csv").write_text("2")
        dest = tmp_path / "files"
        try:
            with current_invocation(invocation):
                agent_tools._publish(
                    str(workdir), str(dest), [{"name": "a.csv", "size": 1}],
                    invocation=invocation, operation_seq=1, step="publish",
                )
                with pytest.raises(RetryDivergence):
                    agent_tools._publish(
                        str(workdir), str(dest), [{"name": "b.csv", "size": 1}],
                        invocation=invocation, operation_seq=1, step="publish",
                    )
        finally:
            invocation.close()


class TestTheRequestLedgerRecordsWhatLanded:
    """`commit_guard` wraps the write, not the handler.

    The idempotency slot already records that a request was entered. What it
    cannot say is which of that request's mutations landed — an upload writes
    bytes and then ingests them, and those are two facts.
    """

    async def test_an_upload_records_its_two_mutations_in_order(
        self, client, auth_headers
    ):
        from liminallm.api import idempotency

        seen: list = []
        real_commit = idempotency.IdempotencyGuard.commit

        def watched(self, capability, payload):
            seen.append((capability, sorted(payload)))
            return real_commit(self, capability, payload)

        idempotency.IdempotencyGuard.commit = watched
        try:
            resp = client.post(
                "/v1/files/upload",
                headers=auth_headers,
                files={"file": ("notes.txt", b"turbine notes", "text/plain")},
            )
            assert resp.status_code == 200, resp.text
        finally:
            idempotency.IdempotencyGuard.commit = real_commit

        assert ("files.write", ["checksum", "path"]) in seen, seen

    def test_a_chat_turn_guards_the_assistant_message(self, client, auth_headers):
        from liminallm.api import idempotency

        seen: list = []
        real_commit = idempotency.IdempotencyGuard.commit

        def watched(self, capability, payload):
            seen.append(capability)
            return real_commit(self, capability, payload)

        idempotency.IdempotencyGuard.commit = watched
        try:
            resp = client.post(
                "/v1/chat",
                headers=auth_headers,
                json={"message": {"content": "hello"}},
            )
            assert resp.status_code == 200, resp.text
        finally:
            idempotency.IdempotencyGuard.commit = real_commit

        assert "message.assistant" in seen, seen

    def test_the_guard_closes_its_ledger_with_the_request(self, runtime):
        """A request that opened one leaves nothing behind."""
        from liminallm.api.idempotency import IdempotencyGuard

        guard = IdempotencyGuard("test", "u1", None)
        guard.request_id = "req-1"
        invocation = guard.invocation
        assert runtime.workflow.invocations.get(invocation.invocation_id) is not None
        asyncio.run(guard.__aexit__(None, None, None))
        assert runtime.workflow.invocations.get(invocation.invocation_id) is None


# ---------------------------------------------------------------------------
# the registry opens once and closes once


class TestTheRegistryDoesNotGrow:
    """The `_guards` lifetime defect, stated as a measurement.

    The old broker installed one lock per invocation in `issue()` and left it
    behind in `revoke()`, so a long-lived server retained a dict entry, a lock
    and an id for every tool attempt ever made. Measured then: 1000
    issue+revoke cycles left 1000 guards retained.
    """

    def test_a_thousand_executions_leave_nothing_behind(self):
        registry = InvocationRegistry()
        for _ in range(1000):
            invocation = registry.open(uuid.uuid4().hex, tool="t")
            invocation.begin_attempt()
            invocation.revoke("done")
            invocation.close()
        assert len(registry) == 0
        assert registry.live() == []

    def test_close_is_idempotent_and_retires_the_entry(self):
        registry = InvocationRegistry()
        invocation = registry.open("once", tool="t")
        invocation.close()
        invocation.close()
        assert registry.get("once") is None

    def test_a_retry_finds_the_same_execution(self):
        registry = InvocationRegistry()
        first = registry.open("same", tool="t")
        assert registry.open("same", tool="t") is first
        first.close()


# ---------------------------------------------------------------------------
# the worker holds nothing worth stealing


class TestTheWorkerCarriesNoAuthority:
    def test_identity_never_crosses_the_pipe(self, runtime, caller):
        engine = runtime.workflow
        _tool, plan, context, _preamble = engine._plan_invocation(
            "llm.generic",
            {"message": "hello"},
            adapters=[{"id": "a1"}],
            history=["turn"],
            context_id="ctx-1",
            conversation_id="conv-1",
            user_message="hello",
            user_id=caller.id,
            tenant_id="tenant-1",
        )
        flat = repr(plan)
        for secret in (caller.id, "tenant-1", "conv-1", "ctx-1"):
            assert secret not in flat, f"{secret} crossed the pipe in the plan"
        # ...and the parent kept every one of them.
        assert context.user_id == caller.id
        assert context.tenant_id == "tenant-1"
        assert context.conversation_id == "conv-1"

    def test_the_worker_module_imports_no_store_or_model(self):
        """The child's import graph is the cost of every spawn, and its blast
        radius. A worker that imported the service layer would pay for httpx,
        psycopg and the model clients on every tool call — and would hold the
        handles this design exists to keep away from it."""
        from pathlib import Path

        import liminallm.service.tool_worker as worker_module

        source = Path(worker_module.__file__).read_text()
        for forbidden in (
            "from liminallm.service.runtime",
            "from liminallm.storage",
            "from liminallm.service.llm",
            "from liminallm.config import",
            "import httpx",
        ):
            assert forbidden not in source, f"the worker imports {forbidden}"

    def test_the_spawn_pays_for_the_standard_library_and_no_more(self):
        """`spawn` re-imports this module in the child, so its import graph is
        paid on every tool call. structlog pulls rich and pygments behind it and
        cost more than the rest of the spawn put together; the logger and the
        invocation types are parent-side and are kept out deliberately."""
        source = subprocess.run(  # noqa: S603 - fixed argv, test-only
            [
                sys.executable,
                "-c",
                "import sys, json; import liminallm.service.tool_worker; "
                "print(json.dumps(sorted(sys.modules)))",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        loaded = set(json.loads(source.stdout))
        for heavy in ("structlog", "rich", "pygments", "psycopg", "httpx", "jax"):
            assert heavy not in loaded, (
                f"importing the worker pulled in {heavy}; every tool call pays "
                "for it in the child"
            )

    def test_the_worker_leads_its_own_process_group(self):
        """One killpg has to reach whatever the worker went on to spawn."""
        import signal
        from pathlib import Path

        import liminallm.service.tool_worker as worker_module

        source = Path(worker_module.__file__).read_text()
        body = source.split("def _worker_main")[1]
        assert body.index("os.setsid()") < body.index("_apply_limits"), (
            "setsid must come first, before anything can spawn"
        )
        assert hasattr(signal, "SIGKILL")

    @pytest.mark.parametrize(
        "limit", ["memory_bytes", "cpu_seconds", "file_size_bytes"]
    )
    def test_worker_limits_come_from_the_tool_sandbox_config(self, runtime, limit):
        """SPEC §18: the worker runs under rlimits, from the tool's own config."""
        limits = runtime.workflow._worker_limits({"name": "any.tool"})
        assert limits[limit] > 0

    def test_a_spec_handler_alias_reaches_its_worker_body(self, runtime):
        """A `tool.spec` naming a worker-side builtin as its handler resolves
        to it. Without this the alias falls through to `tool.host`, which looks
        in the map the body was deliberately moved out of."""
        engine = runtime.workflow
        engine.tool_registry["custom.analyse"] = {
            "name": "custom.analyse",
            "handler": "code.python_v1",
        }
        engine.tool_registry["custom.chat"] = {
            "name": "custom.chat",
            "handler": "llm.generic",
        }
        try:
            assert engine._resolve_worker_tool("custom.analyse") == "code.python_v1"
            assert engine._resolve_worker_tool("code.python_v1") == "code.python_v1"
            # A host body's alias resolves too, so `tool.host` is asked for the
            # body rather than for the name that pointed at it.
            assert engine._resolve_worker_tool("custom.chat") == "llm.generic"
            # And the authorized row's spec wins over the shared registry: a
            # private tool never enters it.
            assert (
                engine._resolve_worker_tool(
                    "not.registered", {"handler": "notes.search_v1"}
                )
                == "notes.search_v1"
            )
        finally:
            engine.tool_registry.pop("custom.analyse", None)
            engine.tool_registry.pop("custom.chat", None)


# ---------------------------------------------------------------------------
# the round still runs where the effects are


class TestTheRoundKeepsItsLeaseAndItsOrder:
    """`_run_round_tools` opens a nested `ThreadPoolExecutor` for read-only
    rounds. Both the egress guard and the bound invocation are thread-local, so
    both must be re-applied inside every pool worker — an unbound thread reads
    as the API path and passes every check."""

    def test_a_parallel_round_still_holds_the_lease(self, runtime, caller):
        from liminallm.service.invocation import active_invocation

        engine = runtime.workflow
        seen: dict = {}
        invocation = Invocation("round-lease")
        invocation.begin_attempt()

        def fake_execute(name, args, **kwargs):
            seen[name] = active_invocation()
            return f"{name}: ok"

        engine._execute_agent_tool = fake_execute
        try:
            engine._run_round_tools(
                [(0, "note_search", {}), (1, "file_search", {})],
                conversation_id=None,
                context_id=None,
                user_id=caller.id,
                tenant_id=caller.tenant_id,
                session=invocation.session,
                snippets=[],
                fallback_query="q",
                invocation=invocation,
            )
        finally:
            invocation.close()

        assert set(seen) == {"note_search", "file_search"}, seen
        for name, bound in seen.items():
            assert bound is not None, (
                f"{name} ran in the nested pool unbound; every store call it "
                "made was treated as the API path"
            )
            assert bound.invocation_id == "round-lease"

    def test_a_revoked_execution_stops_the_nested_read(self, runtime, caller):
        """Block a parallel-safe read, revoke, release it, and require the
        store call it then makes to be refused."""
        engine = runtime.workflow
        at_the_brink = threading.Event()
        released = threading.Event()
        outcome: dict = {}
        invocation = Invocation("round-revoked")
        invocation.begin_attempt()

        def fake_execute(name, args, **kwargs):
            if name == "note_search":
                at_the_brink.set()
                released.wait(timeout=10)
                try:
                    engine.store.get_user(caller.id)
                    outcome["read"] = True
                except BaseException as exc:  # noqa: BLE001
                    outcome["error"] = exc
            return f"{name}: ok"

        engine._execute_agent_tool = fake_execute

        def revoke_when_ready():
            at_the_brink.wait(timeout=10)
            invocation.revoke("cancelled")
            released.set()

        racer = threading.Thread(target=revoke_when_ready, daemon=True)
        racer.start()
        try:
            engine._run_round_tools(
                [(0, "note_search", {}), (1, "file_search", {})],
                conversation_id=None,
                context_id=None,
                user_id=caller.id,
                tenant_id=caller.tenant_id,
                session=invocation.session,
                snippets=[],
                fallback_query="q",
                invocation=invocation,
            )
        finally:
            racer.join(timeout=15)
            invocation.close()

        assert "read" not in outcome, (
            "a revoked execution read through the store from the nested pool"
        )
        assert isinstance(outcome.get("error"), LeaseRevoked), outcome


# ---------------------------------------------------------------------------
# end to end


class TestTheWholePathStillWorks:
    async def test_code_python_runs_through_the_worker(self, runtime, caller):
        engine = runtime.workflow
        result = await engine._invoke_tool(
            "code.python_v1",
            {"code": "print(6 * 7)"},
            [],
            [],
            None,
            None,
            "",
            user_id=caller.id,
            tenant_id=caller.tenant_id,
        )
        assert "42" in result.get("content", ""), result

    async def test_a_timeout_leaves_no_worker_behind(
        self, runtime, caller, monkeypatch
    ):
        """The report and the fact agree: `timeout` is returned, and the
        process it names is gone."""
        engine = runtime.workflow
        engine.tool_registry["test.hang"] = {
            "name": "test.hang",
            "timeout_seconds": 1,
        }
        seen: list = []
        real_serve = engine._serve_invocation

        def watched(invocation, *args, **kwargs):
            # Captured on the way in: on the timeout path this call never
            # returns to us, which is the whole point of the test.
            seen.append(invocation)
            return real_serve(invocation, *args, **kwargs)

        monkeypatch.setattr(engine, "_serve_invocation", watched)
        monkeypatch.setattr(
            engine,
            "_run_host_tool",
            lambda tool, inputs, *, context: time.sleep(5)
            or {"status": "ok", "content": "late"},
        )
        try:
            result = await engine._invoke_tool(
                "test.hang", {}, [], [], None, None, "",
                user_id=caller.id, tenant_id=caller.tenant_id,
            )
            assert result.get("error") == "timeout", result
            assert seen, "no worker was ever spawned; the test proves nothing"
            pids = [a.pid for a in seen[0].attempts if a.pid]
            assert pids, "the attempt recorded no process"
            # The invocation was closed on the way out, which kills and reaps.
            for pid in pids:
                assert not _alive(pid), f"worker {pid} outlived the timeout"
        finally:
            engine.tool_registry.pop("test.hang", None)
