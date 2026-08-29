"""One attempt at a node, and what it takes to be sure the attempt has stopped.

SPEC §9.2 makes retries, backoff, the per-node timeout and output validation
properties of a *node*; §18.3 fixes their numbers. None of that is a property
of how the node's answer is transported, so none of it belongs in a second
copy beside the streaming path. It had one anyway — measured on one aliased
tool that resolves to `llm.generic` and therefore streams:

    property           blocking      streaming
    max_retries: 1     2 attempts    1 attempt
    timeout_ms: 200    enforced      node ran 1.51s and completed
    output_schema      status error  tokens emitted, no error

What streaming does specialise is the body of one attempt: a producer of
tokens rather than a coroutine returning a result. So the attempt is the thing
that varies, and the policy around it is shared. `NodeAttempt` is that seam.

The hard half is stopping. A worker is a process and a kill ends it; a
streamed producer is a thread in this process, and nothing ends a Python
thread from outside. `asyncio.wait_for` cancels the *waiter*: the loop gets
control back and the thread is still inside `next()`, still producing the
answer the next attempt is about to replace. So `asyncio.to_thread(next, it)`
cannot be the timeout mechanism. Termination here is a request the producer
honours between events, plus a separate answer about whether it actually
returned — and an attempt that will not confirm its death stops the retry
rather than running beside its replacement.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Protocol,
    Tuple,
)

#: Put on the queue when the producer has finished, and again by `stop` so a
#: consumer waiting on an abandoned producer is not left waiting forever.
_DONE = object()


class StreamPump:
    """A synchronous producer, iterated off the event loop, that can be told to
    stop and asked whether it did.

    One thread owns the producer for its whole life. That is the difference
    from `asyncio.to_thread(next, iterator)`, which hands one item to one
    pool thread and leaves nobody owning the iterator between items — cancel
    the await and the thread is abandoned inside `next()` with no way to reach
    it, and the pool loses a worker per cancelled stream.

    Events cross on the loop's own queue via `call_soon_threadsafe`, so there
    is no thread hop per token and the consumer's await is ordinarily
    cancellable. The queue is unbounded: the stop flag is checked every
    iteration, so an abandoned producer contributes at most one more event.
    """

    def __init__(
        self,
        factory: Callable[[], Iterator[Dict[str, Any]]],
        *,
        label: str = "stream",
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> None:
        self._factory = factory
        self.label = label
        self._loop = loop or asyncio.get_running_loop()
        self._queue: asyncio.Queue = asyncio.Queue()
        self._stop = threading.Event()
        #: The producer's iterator, kept so `stop` can reach its `abort` — a
        #: cancellable backend stream can interrupt a read in flight, and the
        #: stop flag alone is only read *between* events.
        self._iterator: Optional[Iterator[Dict[str, Any]]] = None
        self._thread = threading.Thread(
            target=self._run, name=f"pump-{label}", daemon=True
        )

    def start(self) -> "StreamPump":
        self._thread.start()
        return self

    # -- producer side ----------------------------------------------------

    def _emit(self, item: Any) -> None:
        try:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, item)
        except RuntimeError:
            # The loop closed before the producer did. There is nobody to
            # deliver to, and the flag is what ends the thread.
            self._stop.set()

    def _run(self) -> None:
        iterator: Optional[Iterator[Dict[str, Any]]] = None
        try:
            # Called here, not by the caller, so a backend that raises on the
            # way in is reported as a failed attempt rather than as an
            # exception on the event loop.
            iterator = self._factory()
            self._iterator = iterator
            # `stop` may have run between the factory and the assignment, in
            # which case its abort found nothing: nothing has connected yet
            # either, so refusing to start iterating is the complete stop.
            if self._stop.is_set():
                return
            for event in iterator:
                if self._stop.is_set():
                    break
                self._emit(event)
        except BaseException as exc:  # noqa: BLE001 - reported as an event
            # After a stop this is the abort surfacing — the shutdown socket
            # raises out of the read — not a result anyone may act on.
            if not self._stop.is_set():
                self._emit(
                    {
                        "event": "error",
                        "data": {"code": "server_error", "message": str(exc)},
                    }
                )
        finally:
            # Closing from this thread is the safe direction: the generator is
            # suspended at its own yield, so `GeneratorExit` lands where its
            # cleanup can run. Closing it from the consumer's thread while it
            # is inside `next()` is what CPython refuses.
            close = getattr(iterator, "close", None)
            if callable(close):
                try:
                    close()
                except BaseException:  # noqa: BLE001 - already ending
                    pass
            self._emit(_DONE)

    # -- consumer side ----------------------------------------------------

    async def events(self) -> AsyncIterator[Dict[str, Any]]:
        while True:
            item = await self._queue.get()
            if item is _DONE:
                return
            yield item

    # -- termination ------------------------------------------------------

    def stop(self) -> None:
        """Stop the producer, and release whoever is waiting on it.

        Three parts, in this order. The flag ends the thread at its next
        iteration. The abort — when the backend's stream carries one —
        interrupts the read the thread is inside *right now*: the shipped
        network backends block in a synchronous read bounded only by the
        provider client's 30–60s timeout, and without the abort a
        `timeout_ms: 200` stopped the waiter while the provider request ran
        on. The sentinel ends the consumer immediately either way.
        """
        self._stop.set()
        iterator = self._iterator
        abort = getattr(iterator, "abort", None)
        if callable(abort):
            try:
                abort()
            except Exception:  # noqa: BLE001 - aborting is best effort
                pass
        self._emit(_DONE)

    def alive(self) -> bool:
        return self._thread.is_alive()

    def cancellation_proven(self) -> bool:
        """Whether this producer's death can be presumed prompt.

        True once the stream's abort handle is armed — an interrupt is in
        hand, so a stop reaches a read in flight and the thread returns in
        moments. Terminal teardown waits for exactly these producers: a
        proven claim is cashed, not forgotten. Unarmed producers (plain
        in-memory doubles, streams still connecting) are excluded from that
        wait as before, and the retry precondition still waits for
        everything.
        """
        return bool(getattr(self._iterator, "armed", False))

    async def wait_dead(self, timeout: float) -> bool:
        """True once the producer thread has actually returned."""
        self.stop()
        if not self._thread.is_alive():
            return True
        await asyncio.to_thread(self._thread.join, timeout)
        return not self._thread.is_alive()


@dataclass
class BreakerObservation:
    """What one attempt learned about the tool's health (SPEC §18).

    Deliberately not derived from the attempt's node-level result: a node can
    fail for reasons that say nothing about the tool — the consumer's
    `output_schema`, an input refused before anything ran — and the ledger
    must record what the *tool* did. `outcome` is set at the raw tool
    boundary; `started` marks that the tool's own work began, which is what
    lets a deadline that fired mid-serve count as a failure while one that
    fired during planning records nothing.
    """

    #: The resolved breaker identity this attempt runs under — the persisted
    #: artifact's id, or the builtin name when nothing is persisted behind
    #: it. On the observation rather than beside it, because resolution is
    #: per attempt: two attempts of one node can resolve different rows, and
    #: each outcome belongs to the row that produced it.
    identity: Optional[str] = None
    #: The tool's own work began: the worker's serve started, or the stream's
    #: producer ran. Resolution, admission, validation and planning all
    #: precede it.
    started: bool = False
    #: "success" | "failure", or None when the attempt proved nothing about
    #: the tool — refused before it started, or abandoned by its caller.
    outcome: Optional[str] = None


@dataclass
class NodeOutcome:
    """What one node execution produced, however it was transported.

    `emitted` is the streamed path's retry boundary and its recovery boundary
    at once: once a token is on the user's screen there is no second answer to
    give them, so a failure after the first token is terminal and takes no
    `on_error` edge.
    """

    result: Dict[str, Any]
    next_nodes: List[str] = field(default_factory=list)
    #: The producer's own error event, kept so a terminal failure reaches the
    #: client with the code the backend gave it rather than a generic one.
    failure_event: Optional[Dict[str, Any]] = None
    emitted: bool = False


class NodeAttempt(Protocol):
    """One try at a node. Never its own retry policy.

    Stopping is deliberately absent: an attempt registers what it started on
    the `Invocation`, and the driver tears the execution down through that.
    Two teardown paths is how one of them ends up not covering the streamed
    producer, which is the state this replaced.
    """

    #: What the driver reports when this kind of attempt will not confirm its
    #: death, so the log names the thing that would not stop.
    unreaped_error: str

    #: Whether `result()` is already computed once `events()` has ended.
    #: True for a streamed attempt — its work happened while the events
    #: drained, and `result()` only returns the stored outcome. False for a
    #: blocking attempt — its body *starts* inside `result()`. The driver
    #: may collect a ready result after the clock has crossed zero; it must
    #: never start un-begun work there.
    result_ready_after_events: bool

    #: Whether the driver must open an `Attempt` lease for this try (SPEC
    #: §18.3: authority is fresh per attempt). True for attempts whose work
    #: runs in this process; False when the body spawns a worker, because the
    #: spawn opens the lease itself and a second one here would double-count.
    needs_lease: bool

    #: This attempt's breaker observation. The attempt fills it in; the
    #: driver writes the ledger from it, exactly once per attempt.
    breaker: BreakerObservation

    def events(self) -> AsyncIterator[Dict[str, Any]]: ...

    async def result(self) -> NodeOutcome: ...


class BlockingNodeAttempt:
    """A node whose answer arrives whole.

    The body is the engine's ordinary node execution and the producer is a
    worker process, which the invocation already kills and reaps.
    """

    unreaped_error = "tool_worker_unreaped"
    #: The worker spawn calls `begin_attempt` itself, per §18.3.
    needs_lease = False
    #: The body runs inside `result()`; nothing exists before it is awaited.
    result_ready_after_events = False

    def __init__(
        self,
        run: Callable[[], Any],
        *,
        breaker: Optional[BreakerObservation] = None,
    ) -> None:
        self._run = run
        #: Shared with the body: the engine's tool invocation sets it at the
        #: raw tool boundary, inside `run`, and the driver reads it here.
        self.breaker = breaker or BreakerObservation()

    async def events(self) -> AsyncIterator[Dict[str, Any]]:
        return
        yield  # pragma: no cover - makes this an async generator

    async def result(self) -> NodeOutcome:
        result, next_nodes = await self._run()
        return NodeOutcome(result=result, next_nodes=list(next_nodes or []))


class StreamedNodeAttempt:
    """A node whose answer arrives as tokens.

    Buffers when the node's tool declares an `output_schema`. SPEC §9.2
    validates outputs as well as inputs, and a validated output cannot be
    incremental: tokens already on the user's screen cannot be withdrawn when
    the finished answer turns out to violate the schema. So a node with a
    schema streams nothing until its answer passes, and a node without one
    streams exactly as it did before.

    The completed tool result arrives as its own `tool_result` event, emitted
    by the streaming implementation and consumed here — never forwarded, and
    never reconstructed from the client-facing `message_done`. This class
    used to manufacture the raw result itself from the fields it knew about,
    which was exactly `llm.generic`'s four — so `agent.files_v1`'s
    `artifacts` and `injection_findings` vanished before validation, and a
    schema for the real result got a different verdict per transport. The
    handler that produced the result names its fields; a transport does not.

    The producer's `error` event is not forwarded. It is this attempt's
    outcome, and the driver may still have a retry to spend; emitting it would
    put a failure on the client's screen that the next attempt then contradicts.
    """

    unreaped_error = "stream_producer_unreaped"
    #: No worker spawn on this path, so nothing else opens the lease. Without
    #: it a streamed retry ran with no authority of its own — and worse:
    #: `revoke("retry")` found no current attempt, read that as "nothing has
    #: started", cancelled the whole execution, and the next attempt called
    #: the provider anyway because nothing here asked.
    needs_lease = True
    #: `_drain` stores the outcome before it finishes; `result()` reads it.
    result_ready_after_events = True

    def __init__(
        self,
        stream: AsyncIterator[Dict[str, Any]],
        *,
        finalize: Callable[
            [Dict[str, Any]], Tuple[Dict[str, Any], Optional[Dict[str, Any]]]
        ],
        buffer: bool = False,
        breaker: Optional[BreakerObservation] = None,
    ) -> None:
        self._stream = stream
        #: The postflight: `(sanitized, refusal)`. Always applied, exactly as
        #: the blocking path applies it — sanitizing is not conditional on a
        #: schema, and what proceeds downstream is the sanitized object.
        self._finalize = finalize
        self._buffer = buffer
        #: Shared with the streaming body, which marks `started` at its own
        #: serve boundary — the worker spawn or the provider pump, not this
        #: class's first pull: the body plans (retrieval, grounding, context
        #: assembly) before any tool work runs, and a deadline spent there
        #: must record nothing.
        self.breaker = breaker or BreakerObservation()
        self._outcome = NodeOutcome(
            result={"status": "error", "error": "stream produced no answer"}
        )

    async def events(self) -> AsyncIterator[Dict[str, Any]]:
        try:
            async for event in self._drain():
                yield event
        finally:
            # Explicitly, because closing *this* generator does not close the
            # one it was iterating: the attempt still holds a reference, so the
            # inner generator's cleanup — which stops the producer — would wait
            # for a collection instead of happening when the node ends.
            await self._stream.aclose()

    async def _drain(self) -> AsyncIterator[Dict[str, Any]]:
        held: List[Dict[str, Any]] = []
        emitted = False
        done: Optional[Dict[str, Any]] = None
        raw: Optional[Dict[str, Any]] = None

        async for event in self._stream:
            kind = event.get("event")
            if kind == "error":
                self.breaker.outcome = "failure"
                self._outcome = NodeOutcome(
                    result={
                        "status": "error",
                        "error": (event.get("data") or {}).get("message", ""),
                    },
                    failure_event=event,
                    emitted=emitted,
                )
                return
            if kind == "tool_result":
                # The canonical completed result. Consumed, never forwarded:
                # the client's contract is tokens and `message_done`.
                raw = dict(event.get("data") or {})
                # The breaker records what the *tool* did, so the observation
                # is taken here — before the postflight, whose refusal is the
                # consumer's schema speaking, not the tool (SPEC §18). A
                # failure already observed is sticky: a body that salvages a
                # partial answer after its provider died still emits a
                # well-formed `tool_result`, and user-facing recovery must
                # not rewrite tool health.
                if self.breaker.outcome != "failure":
                    self.breaker.outcome = (
                        "failure" if raw.get("status") == "error" else "success"
                    )
                continue
            if kind == "message_done":
                done = event
                break
            if kind == "token":
                if self._buffer:
                    held.append(event)
                    continue
                emitted = True
            yield event

        if done is None:
            # The producer stopped without an answer: a cancel, or a stream
            # that ended mid-sentence. Either way there is no result to
            # validate and nothing to record as success.
            self._outcome = NodeOutcome(result=dict(self._outcome.result), emitted=emitted)
            return

        if raw is None:
            # Fail closed rather than reconstruct: inventing the result here
            # from the client event is the defect this seam removes.
            self._outcome = NodeOutcome(
                result={
                    "status": "error",
                    "error": "stream completed without a tool result",
                },
                emitted=emitted,
            )
            return

        sanitized, refusal = self._finalize(raw)
        if refusal is not None:
            # Nothing held is released. That is the whole reason for
            # holding it.
            self._outcome = NodeOutcome(result=refusal, emitted=emitted)
            return

        for token_event in held:
            emitted = True
            yield token_event
        yield done
        self._outcome = NodeOutcome(
            result={"status": "ok", **sanitized}, emitted=emitted
        )

    async def result(self) -> NodeOutcome:
        return self._outcome


async def bounded(
    events: AsyncIterator[Dict[str, Any]], deadline: float
) -> AsyncIterator[Dict[str, Any]]:
    """Forward `events`, but raise `asyncio.TimeoutError` at `deadline`.

    A deadline rather than one `wait_for` around the whole drain, because the
    events have to keep flowing while it is being enforced: a stream that is
    forwarded only after it finishes is not a stream.
    """
    iterator = events.__aiter__()
    try:
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                # The deadline governs waiting for events, and a stream that
                # has already delivered its final event has nothing left to
                # time out. This raised unconditionally, so the pull that
                # would have ended a completed stream reported a node timeout
                # instead — and an empty completion was then retried, a
                # second answer after one the client had received. One short
                # grace distinguishes finished from late: `StopAsyncIteration`
                # ends cleanly; an event, or nothing, is late — the event is
                # dropped, exactly as if the deadline had caught it earlier.
                try:
                    await asyncio.wait_for(iterator.__anext__(), 0.001)
                except StopAsyncIteration:
                    return
                except asyncio.TimeoutError:
                    pass
                raise asyncio.TimeoutError()
            try:
                event = await asyncio.wait_for(iterator.__anext__(), remaining)
            except StopAsyncIteration:
                return
            yield event
    finally:
        aclose = getattr(iterator, "aclose", None)
        if aclose is not None:
            await aclose()
