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
