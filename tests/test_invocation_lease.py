"""A tool that outlives its node timeout keeps the caller's authority.

`_invoke_tool` submits the handler to a `ThreadPoolExecutor` and awaits it
with `asyncio.wait_for`. On timeout it called `future.cancel()` — which
returns `False` for anything already running, because a `concurrent.futures`
future can only cancel work that has not started. The engine logged
`tool_timeout_cancellation_failed`, returned `"tool timed out"` to the
workflow, and the thread ran on with `user_id` and `tenant_id` still in its
closure.

What that thread could still do was the whole of the tool surface: write to
the store, spend on the model, fetch a URL, publish files into the user's
file area. It did all of it *after* the request that authorized it was
abandoned and reported failed — and if the node was retried, a second worker
started while the first was still running. The first test below created an
artifact under the caller's id a full second after the node reported
`timeout`.

These tests state the property the tranche has to deliver: **the authority
ends when the invocation ends.** Not the thread — a Python thread cannot be
killed — the *lease*. Revoked first, then the worker is torn down, and the
worker is reaped before anything retries in its name.
"""

from __future__ import annotations

import asyncio
import threading
import time
import uuid

import pytest

from liminallm.service.lease import LeaseRevoked


def _unique(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def runtime(client):
    from liminallm.service.runtime import get_runtime

    return get_runtime()


@pytest.fixture
def caller(runtime):
    return runtime.store.create_user(email=f"{_unique('late')}@t.local")


class _LateWriter:
    """A handler that finishes well after the node gave up on it.

    It records what it managed to do, so a test can ask whether the write
    landed rather than whether the code looks like it would.
    """

    def __init__(self, store, *, delay: float):
        self.store = store
        self.delay = delay
        self.started = threading.Event()
        self.finished = threading.Event()
        self.wrote: str | None = None
        self.error: BaseException | None = None

    def __call__(
        self, inputs, adapters, history, context_id, conversation_id,
        user_message, user_id, tenant_id,
    ):
        self.started.set()
        time.sleep(self.delay)
        try:
            artifact = self.store.create_artifact(
                "tool",
                _unique("written_after_timeout"),
                {"kind": "tool.spec", "name": _unique("late"), "handler": "x"},
                owner_user_id=user_id,
            )
            self.wrote = artifact.id
        except BaseException as exc:  # noqa: BLE001 - the point is which one
            self.error = exc
        finally:
            self.finished.set()
        return {"content": "too late", "usage": {}}


def _install(monkeypatch, engine, name, handler):
    """Add one handler to the engine's table for the length of a test.

    Through monkeypatch rather than plain assignment: the engine is shared
    within a test and a leaked handler would be resolvable by a later one.
    """
    original = engine._builtin_tool_handlers

    def patched():
        table = original()
        table[name] = handler
        return table

    monkeypatch.setattr(engine, "_builtin_tool_handlers", patched)


def _call(engine, name, *, user_id, tenant_id, timeout_seconds):
    from liminallm.service.workflow import ToolDescriptor

    return asyncio.run(
        engine._invoke_tool(
            name, {}, [], [], None, None, "",
            user_id=user_id,
            tenant_id=tenant_id,
            descriptor=ToolDescriptor(
                name=name,
                schema={
                    "kind": "tool.spec",
                    "name": name,
                    "handler": name,
                    "timeout_seconds": timeout_seconds,
                },
                artifact_id=None,
                owner_user_id=None,
                owner_role=None,
            ),
        )
    )


class TestAuthorityEndsWithTheInvocation:
    def test_the_node_does_report_the_timeout(self, runtime, caller, monkeypatch):
        """The half that already works, so a failure below is about authority
        and not about the timeout never firing."""
        engine = runtime.workflow
        name = _unique("slow")
        # `engine.store`, not `runtime.store`: a handler is an engine method
        # and reaches the store through `self`. A double built on the raw
        # store would test a path no handler takes.
        late = _LateWriter(engine.store, delay=1.5)
        _install(monkeypatch, engine, name, late)

        result = _call(
            engine, name,
            user_id=caller.id, tenant_id=caller.tenant_id, timeout_seconds=1,
        )
        assert result.get("error") == "timeout", result
        assert late.started.is_set(), "the handler never ran; the test proves nothing"

    def test_a_write_after_the_timeout_does_not_land(
        self, runtime, caller, monkeypatch
    ):
        """The defect. The worker's store write happens after the node
        returned `timeout`, with the caller's id, and nothing stops it."""
        engine = runtime.workflow
        name = _unique("slow")
        # `engine.store`, not `runtime.store`: a handler is an engine method
        # and reaches the store through `self`. A double built on the raw
        # store would test a path no handler takes.
        late = _LateWriter(engine.store, delay=1.5)
        _install(monkeypatch, engine, name, late)

        result = _call(
            engine, name,
            user_id=caller.id, tenant_id=caller.tenant_id, timeout_seconds=1,
        )
        assert result.get("error") == "timeout", result

        assert late.finished.wait(timeout=15), "the worker never finished"
        assert late.wrote is None, (
            f"the abandoned worker created artifact {late.wrote} using the "
            "caller's authority after the invocation was reported timed out"
        )
        # Name the reason. Absence of a write is also what a broken handler
        # produces, and this test would pass on one.
        assert isinstance(late.error, LeaseRevoked), late.error


class TestTheCheckHappensAtTheCall:
    """Not at handler start.

    A lease read once on entry would leave the whole body of the handler
    running on a decision made before the timeout existed. The revocation
    here lands while the worker is already inside the handler, between its
    last statement and its write.
    """

    def test_a_revocation_mid_handler_stops_the_next_write(
        self, runtime, caller, monkeypatch
    ):
        engine = runtime.workflow
        name = _unique("racing")
        at_the_brink = threading.Event()
        released = threading.Event()
        outcome: dict = {}

        def handler(inputs, adapters, history, ctx, conv, msg, user_id, tenant_id):
            at_the_brink.set()
            released.wait(timeout=10)
            try:
                artifact = engine.store.create_artifact(
                    "tool", _unique("raced"),
                    {"kind": "tool.spec", "name": _unique("r"), "handler": "x"},
                    owner_user_id=user_id,
                )
                outcome["wrote"] = artifact.id
            except BaseException as exc:  # noqa: BLE001
                outcome["error"] = exc
            return {"content": "done", "usage": {}}

        _install(monkeypatch, engine, name, handler)

        # Catch the lease the engine issues, so the revocation below goes
        # through the same public call the timeout path uses.
        issued: list = []
        real_issue = engine.broker.issue

        def recording_issue(*args, **kwargs):
            invocation = real_issue(*args, **kwargs)
            issued.append(invocation)
            return invocation

        monkeypatch.setattr(engine.broker, "issue", recording_issue)

        def revoke_when_ready():
            at_the_brink.wait(timeout=10)
            engine.broker.revoke(issued[-1])
            released.set()

        racer = threading.Thread(target=revoke_when_ready, daemon=True)
        racer.start()
        _call(
            engine, name,
            user_id=caller.id, tenant_id=caller.tenant_id, timeout_seconds=30,
        )
        racer.join(timeout=15)

        assert "wrote" not in outcome, (
            f"a revoked worker committed artifact {outcome.get('wrote')}"
        )
        assert isinstance(outcome.get("error"), LeaseRevoked), outcome


class TestARetryDoesNotStackWorkers:
    """`_execute_node_with_retry` runs up to three attempts.

    Before reaping, attempt two started while attempt one's worker was still
    in the pool — one node holding three workers, and for `code.python_v1`
    three sandbox children. The lease already makes the stragglers harmless;
    this is about not leaving them running.
    """

    def test_the_worker_is_finished_before_the_call_returns(
        self, runtime, caller, monkeypatch
    ):
        engine = runtime.workflow
        name = _unique("straggler")
        live = threading.Semaphore(0)
        state = {"concurrent": 0, "peak": 0}
        lock = threading.Lock()

        def handler(inputs, adapters, history, ctx, conv, msg, user_id, tenant_id):
            with lock:
                state["concurrent"] += 1
                state["peak"] = max(state["peak"], state["concurrent"])
            try:
                time.sleep(1.5)
            finally:
                with lock:
                    state["concurrent"] -= 1
                live.release()
            return {"content": "late", "usage": {}}

        _install(monkeypatch, engine, name, handler)

        for _ in range(3):  # what the retry loop would do
            result = _call(
                engine, name,
                user_id=caller.id, tenant_id=caller.tenant_id, timeout_seconds=1,
            )
            assert result.get("error") == "timeout", result
            # The reap has already happened by the time the node returns.
            with lock:
                assert state["concurrent"] == 0, (
                    "a worker from the previous attempt is still running"
                )

        for _ in range(3):
            assert live.acquire(timeout=10)
        assert state["peak"] == 1, f"attempts overlapped: peak {state['peak']}"


class TestTheLeaseSurvivesTheNestedPool:
    """`_run_round_tools` opens its own `ThreadPoolExecutor`.

    Its `run_one` re-applies `tool_network_guard` inside every worker, and
    the docstring gives the reason: the guard is thread-local, so the
    ambient one does not follow work into a pool. The invocation lease is
    thread-local for the same reason and was not re-applied — so a parallel
    round of reads ran with `active_invocation() is None`, which `LeasedProxy`
    reads as "the API path" and waves through.

    That is a direct contradiction of the rule the SPEC now states: every
    store, model and retrieval call a worker makes checks the lease.
    """

    def _round(self, engine, seen, *, user_id, tenant_id):
        from liminallm.service.lease import active_invocation

        def fake_execute(name, args, **kwargs):
            seen[name] = active_invocation()
            return f"{name}: ok"

        engine._execute_agent_tool = fake_execute
        return engine._run_round_tools(
            [(0, "note_search", {}), (1, "file_search", {})],
            conversation_id=None,
            context_id=None,
            user_id=user_id,
            tenant_id=tenant_id,
            session={},
            snippets=[],
            fallback_query="q",
        )

    def test_a_parallel_round_still_holds_the_lease(
        self, runtime, caller, monkeypatch
    ):
        from liminallm.service.lease import current_invocation

        engine = runtime.workflow
        seen: dict = {}
        invocation = engine.broker.issue(
            "agent", user_id=caller.id, tenant_id=caller.tenant_id
        )
        # Stand where a leased handler stands, then fan out exactly as
        # `_tool_agent_files` does.
        with current_invocation(invocation):
            self._round(engine, seen, user_id=caller.id, tenant_id=caller.tenant_id)

        assert set(seen) == {"note_search", "file_search"}, seen
        for name, nested in seen.items():
            assert nested is not None, (
                f"{name} ran in the nested pool with no lease; every store "
                "call it made was treated as the API path"
            )
            assert nested.invocation_id == invocation.invocation_id

    def test_a_revoked_lease_stops_the_nested_read(
        self, runtime, caller, monkeypatch
    ):
        """The behavioral half: block a parallel-safe read, revoke the outer
        invocation, release it, and require its store call to be refused."""
        from liminallm.service.lease import current_invocation

        engine = runtime.workflow
        at_the_brink = threading.Event()
        released = threading.Event()
        outcome: dict = {}
        invocation = engine.broker.issue(
            "agent", user_id=caller.id, tenant_id=caller.tenant_id
        )

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
            engine.broker.revoke(invocation)
            released.set()

        racer = threading.Thread(target=revoke_when_ready, daemon=True)
        racer.start()
        with current_invocation(invocation):
            engine._run_round_tools(
                [(0, "note_search", {}), (1, "file_search", {})],
                conversation_id=None, context_id=None,
                user_id=caller.id, tenant_id=caller.tenant_id,
                session={}, snippets=[], fallback_query="q",
            )
        racer.join(timeout=15)

        assert "read" not in outcome, (
            "a revoked invocation read through the store from the nested pool"
        )
        assert isinstance(outcome.get("error"), LeaseRevoked), outcome


class TestRetriesShareALogicalExecutionButNotALease:
    """Two ids, because they answer different questions.

    The lease is keyed by `invocation_id`: attempt two must not inherit
    attempt one's authority. A durable idempotency key is keyed by the
    logical execution: killing attempt one does not recall an external
    operation it already submitted, so attempt two must be able to recognise
    it as the same operation.
    """

    def test_each_attempt_gets_its_own_lease(self, runtime, caller, monkeypatch):
        engine = runtime.workflow
        name = _unique("twice")
        seen: list = []
        real_issue = engine.broker.issue

        def recording_issue(*args, **kwargs):
            invocation = real_issue(*args, **kwargs)
            seen.append(invocation)
            return invocation

        monkeypatch.setattr(engine.broker, "issue", recording_issue)

        def handler(inputs, adapters, history, ctx, conv, msg, user_id, tenant_id):
            return {"content": "ok", "usage": {}}

        _install(monkeypatch, engine, name, handler)

        node_id = uuid.uuid4().hex
        for _ in range(2):  # two attempts of one logical node execution
            asyncio.run(
                engine._invoke_tool(
                    name, {}, [], [], None, None, "",
                    user_id=caller.id, tenant_id=caller.tenant_id,
                    logical_execution_id=node_id,
                )
            )

        assert len(seen) == 2, seen
        first, second = seen
        assert first.invocation_id != second.invocation_id, "a retry reused the lease"
        assert first.logical_execution_id == second.logical_execution_id == node_id
        assert first.operation_key("publish") == second.operation_key("publish"), (
            "attempt two would treat a submitted operation as a new one"
        )

    def test_a_call_with_no_node_is_its_own_execution(self, runtime, caller):
        """Direct invocation has no retry loop above it, so it must still get
        a usable key rather than an empty one."""
        engine = runtime.workflow
        invocation = engine.broker.issue(
            "direct", user_id=caller.id, tenant_id=caller.tenant_id
        )
        assert invocation.logical_execution_id
        assert invocation.operation_key("publish").endswith(":publish")


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Open: closing this needs the spawned per-invocation worker. `_reap` "
        "waits REAP_GRACE_SECONDS and returns, so a handler outliving the "
        "grace is still running when the next attempt begins. A thread cannot "
        "be killed, so bounded reaping is the best a thread worker can do — "
        "the fix is the killable process, not a longer grace."
    ),
)
class TestReapingIsNotAGracePeriod:
    """`_reap` waits `REAP_GRACE_SECONDS`, logs `tool_worker_unreaped`, and
    returns — so the retry machinery is free to start attempt two while
    attempt one is still executing.

        handler 30s, timeout 1s, grace 5s
        t=1  lease revoked
        t=6  reap gives up
        t>6  attempt 2 begins
        t=30 attempt 1 finally exits

    The earlier tests all used workers shorter than the grace, so none of
    them ever entered this branch. A worker that outlives the grace is the
    case that shows thread reaping cannot be perfected — the tranche needs a
    killable process.
    """

    def test_a_worker_outliving_the_grace_blocks_the_next_attempt(
        self, runtime, caller, monkeypatch
    ):
        engine = runtime.workflow
        name = _unique("stubborn")
        overrun = engine.REAP_GRACE_SECONDS + 3
        alive = threading.Semaphore(0)
        state = {"concurrent": 0, "peak": 0}
        lock = threading.Lock()

        def handler(inputs, adapters, history, ctx, conv, msg, user_id, tenant_id):
            with lock:
                state["concurrent"] += 1
                state["peak"] = max(state["peak"], state["concurrent"])
            try:
                time.sleep(overrun)
            finally:
                with lock:
                    state["concurrent"] -= 1
                alive.release()
            return {"content": "eventually", "usage": {}}

        _install(monkeypatch, engine, name, handler)

        for attempt in range(2):  # what _execute_node_with_retry would do
            result = _call(
                engine, name,
                user_id=caller.id, tenant_id=caller.tenant_id, timeout_seconds=1,
            )
            assert result.get("error") == "timeout", result
            with lock:
                assert state["concurrent"] == 0, (
                    f"attempt {attempt + 1} returned while a previous worker "
                    "was still running; the retry would stack on it"
                )

        for _ in range(2):
            assert alive.acquire(timeout=overrun * 3)
        assert state["peak"] == 1, f"attempts overlapped: peak {state['peak']}"


class TestPublishingAUserFileIsADurableOperation:
    """`agent_tools.run_python` copies into the user's persistent file area
    by calling `interpreter.publish_artifacts` directly.

    That path touches neither `store`, `llm` nor `rag`, so `LeasedProxy`
    never sees it and a revoked invocation can still put a file in the user's
    area. It has to become one of the broker's durable operations.
    """

    def test_a_revoked_invocation_publishes_nothing(
        self, runtime, caller, monkeypatch, tmp_path
    ):
        from liminallm.service import interpreter
        from liminallm.service.lease import current_invocation

        engine = runtime.workflow
        files_dir = tmp_path / "user_files"
        files_dir.mkdir()
        workdir = tmp_path / "work"
        workdir.mkdir()
        (workdir / "result.csv").write_text("a,b\n1,2\n")

        invocation = engine.broker.issue(
            "code.python_v1", user_id=caller.id, tenant_id=caller.tenant_id
        )
        engine.broker.revoke(invocation)

        # Refusing loudly, not returning an empty list: a durable operation
        # that quietly does nothing is indistinguishable from one that had
        # nothing to do, and the caller would report success.
        with current_invocation(invocation), pytest.raises(LeaseRevoked):
            interpreter.publish_artifacts(
                str(workdir),
                str(files_dir),
                [{"name": "result.csv", "size": 8}],
                allowed_extensions={".csv"},
            )

        assert list(files_dir.iterdir()) == [], (
            "a revoked invocation published a file into the user's area"
        )

    def test_a_live_invocation_still_publishes(
        self, runtime, caller, monkeypatch, tmp_path
    ):
        """The control. A check that refused everything would pass the test
        above and break the tool."""
        from liminallm.service import interpreter
        from liminallm.service.lease import current_invocation

        engine = runtime.workflow
        files_dir = tmp_path / "user_files"
        files_dir.mkdir()
        workdir = tmp_path / "work"
        workdir.mkdir()
        (workdir / "result.csv").write_text("a,b\n1,2\n")

        invocation = engine.broker.issue(
            "code.python_v1", user_id=caller.id, tenant_id=caller.tenant_id
        )
        with current_invocation(invocation):
            published = interpreter.publish_artifacts(
                str(workdir), str(files_dir),
                [{"name": "result.csv", "size": 8}],
                allowed_extensions={".csv"},
            )
        assert published == ["result.csv"]
        assert (files_dir / "result.csv").is_file()


# ---------------------------------------------------------------------------
# Acceptance tests for the brokered capabilities. These are not assertions
# about the thread design — they state what the spawned worker plus broker
# must deliver, and they are red until it exists.
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Open: web access is not yet a brokered capability. `_run_web_search` "
        "and `_run_web_fetch` hand raw `self.settings` to agent_tools, which "
        "no proxy sees, so a revoked worker still reaches the network with "
        "the deployment's proxy, allowlist and credentials. Closing it means "
        "the parent owning WEB_SEARCH/WEB_FETCH, not another in-thread check."
    ),
)
class TestWebAccessIsACapability:
    """`store`, `llm` and `rag` are proxied; `settings` never was.

    `_run_web_search` and `_run_web_fetch` hand `self.settings` straight to
    `agent_tools`, so a revoked worker can still reach the network with the
    deployment's configured proxy, allowlist and provider credentials. That
    is authority the invocation no longer holds, and no proxy sees it — the
    capability has to be owned by the parent broker.
    """

    def test_a_revoked_invocation_sends_no_fetch(
        self, runtime, caller, monkeypatch
    ):
        from liminallm.service import web
        from liminallm.service.lease import current_invocation

        engine = runtime.workflow
        reached: list = []
        monkeypatch.setattr(
            web, "fetch_url", lambda url, **kw: reached.append(url) or (_ for _ in ()).throw(
                web.WebFetchError("unreachable")
            )
        )

        invocation = engine.broker.issue(
            "web.fetch_v1", user_id=caller.id, tenant_id=caller.tenant_id
        )
        engine.broker.revoke(invocation)

        with current_invocation(invocation):
            try:
                engine._run_web_fetch("https://example.invalid/page")
            except LeaseRevoked:
                pass

        assert reached == [], (
            f"a revoked invocation reached the network: {reached}"
        )

    def test_a_revoked_invocation_sends_no_search(
        self, runtime, caller, monkeypatch
    ):
        from liminallm.service import web
        from liminallm.service.lease import current_invocation

        engine = runtime.workflow
        reached: list = []
        monkeypatch.setattr(
            web, "search_web", lambda *a, **kw: reached.append(a) or []
        )

        invocation = engine.broker.issue(
            "web.search_v1", user_id=caller.id, tenant_id=caller.tenant_id
        )
        engine.broker.revoke(invocation)

        with current_invocation(invocation):
            try:
                engine._run_web_search("anything", 3)
            except LeaseRevoked:
                pass

        assert reached == [], f"a revoked invocation searched the web: {reached}"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Open: publication is checked, the launch is not. A revoked worker "
        "still spawns a confined child and burns its budget before the "
        "publish refuses. Closing it means RUN_PYTHON being a broker-owned "
        "capability whose child the parent registers and can kill."
    ),
)
class TestRunPythonIsNotStartedForARevokedInvocation:
    """Publication is checked; the launch is not.

    A revoked worker can still spawn a confined sandbox child and burn its
    CPU and memory budget, only to have the publish refused at the end. The
    capability must refuse before the process starts.
    """

    def test_no_sandbox_child_is_spawned(self, runtime, caller, monkeypatch):
        from liminallm.service import agent_tools, interpreter
        from liminallm.service.lease import current_invocation

        engine = runtime.workflow
        spawned: list = []
        monkeypatch.setattr(
            interpreter,
            "run_python_sandboxed",
            lambda code, **kw: spawned.append(code) or {
                "ok": True, "stdout": "", "stderr": "", "created_files": []
            },
        )

        invocation = engine.broker.issue(
            "code.python_v1", user_id=caller.id, tenant_id=caller.tenant_id
        )
        engine.broker.revoke(invocation)

        with current_invocation(invocation):
            try:
                agent_tools.run_python(
                    "print(1)", [],
                    settings=engine.settings, user_id=caller.id, session={},
                )
            except LeaseRevoked:
                pass

        assert spawned == [], (
            "a revoked invocation launched a sandbox child before its "
            "publication was refused"
        )


class TestTheCommitGuardIsALinearizationPoint:
    """A bare `check()` before `COMMIT` still has a window.

        check says live
                              revoke lands here
        COMMIT

    `commit_guard` and `revoke` contend on the same per-invocation lock, so
    only two histories exist: the commit runs and revocation waits for it, or
    revocation completes and the commit is refused. There is no third in
    which the timeout path reports revocation done while a commit slips
    through behind it.
    """

    def test_a_revocation_racing_the_guard_never_lands_mid_commit(
        self, runtime, caller
    ):
        broker = runtime.workflow.broker
        for _ in range(200):  # a window this small needs repetition to catch
            invocation = broker.issue(
                "t", user_id=caller.id, tenant_id=caller.tenant_id
            )
            committed: list = []
            inside = threading.Event()

            def commit():
                try:
                    with broker.commit_guard(invocation):
                        inside.set()
                        committed.append(True)
                except LeaseRevoked:
                    pass

            worker = threading.Thread(target=commit, daemon=True)
            revoker = threading.Thread(
                target=lambda: broker.revoke(invocation), daemon=True
            )
            worker.start()
            revoker.start()
            worker.join(timeout=5)
            revoker.join(timeout=5)

            # Whichever won, the invariant holds: nothing is live afterwards,
            # and a commit that happened happened before revocation returned.
            assert not broker.is_live(invocation)
            assert len(committed) <= 1

    def test_a_commit_after_revocation_is_refused(self, runtime, caller):
        broker = runtime.workflow.broker
        invocation = broker.issue(
            "t", user_id=caller.id, tenant_id=caller.tenant_id
        )
        broker.revoke(invocation)
        with pytest.raises(LeaseRevoked):
            with broker.commit_guard(invocation):
                raise AssertionError("the guard admitted a revoked invocation")

    def test_revocation_waits_for_a_commit_already_inside(self, runtime, caller):
        """The other ordering, made deterministic: revocation cannot return
        while a commit holds the guard."""
        broker = runtime.workflow.broker
        invocation = broker.issue(
            "t", user_id=caller.id, tenant_id=caller.tenant_id
        )
        inside = threading.Event()
        finished_commit = threading.Event()
        revoke_returned = threading.Event()

        def commit():
            with broker.commit_guard(invocation):
                inside.set()
                time.sleep(0.5)
                finished_commit.set()

        def revoke():
            inside.wait(timeout=5)
            broker.revoke(invocation)
            revoke_returned.set()

        threads = [
            threading.Thread(target=commit, daemon=True),
            threading.Thread(target=revoke, daemon=True),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert finished_commit.is_set()
        assert revoke_returned.is_set()
        assert not broker.is_live(invocation)


class TestTheBrokerTravelsWithTheInvocation:
    def test_a_replaced_engine_does_not_answer_for_an_old_lease(
        self, runtime, caller, monkeypatch, tmp_path
    ):
        """Hot reload replaces the engine while in-flight workers finish.

        With a process-global broker, an old worker's durable operation asked
        the *new* engine's broker about a lease it never issued, and got a
        refusal that looks identical to a real revocation. The invocation
        carries its own broker instead.
        """
        from liminallm.service import interpreter
        from liminallm.service.lease import InvocationBroker, current_invocation

        old = runtime.workflow.broker
        invocation = old.issue(
            "code.python_v1", user_id=caller.id, tenant_id=caller.tenant_id
        )
        # Stand in for the reload: a wholly new engine and broker appear.
        runtime.workflow.broker = InvocationBroker()

        files_dir = tmp_path / "files"
        files_dir.mkdir()
        workdir = tmp_path / "work"
        workdir.mkdir()
        (workdir / "out.csv").write_text("x\n")

        with current_invocation(invocation):
            published = interpreter.publish_artifacts(
                str(workdir), str(files_dir),
                [{"name": "out.csv", "size": 2}],
                allowed_extensions={".csv"},
            )
        assert published == ["out.csv"], (
            "the old invocation's still-live lease was judged by the new "
            "engine's broker"
        )
