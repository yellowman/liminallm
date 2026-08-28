"""One logical execution of one tool call: its ledger, its children, its life.

A tool call is not one attempt. A node times out, retries, and times out
again; the same logical execution has now run three times. What has to survive
those attempts is not the process — the whole point is that the process dies —
but the record of which effects already landed, and a grip on everything the
execution started.

Three properties are why this module owns a lock rather than leaving the state
as plain attributes:

* an effect checks liveness **before** it acts, never after. After is a report,
  not a control;
* a commit and a revoke are linearized, so a durable mutation cannot land in
  the gap between "the row was written" and "the ledger says it was";
* everything the execution spawned stays reachable, so it can be killed and
  reaped before anything is allowed to run again.

The record is an ordered **ledger**, not a content-addressed key. A key can
only answer "have I seen this payload before", which is the wrong question
twice over: two legitimate identical calls collide into one, and a retry that
varies a single byte of a model-written payload misses entirely. A sequence
answers "which step of this execution is this", which is what a replay needs.

`InvocationRegistry` holds the live executions, and an engine owns one. It is
deliberately not a module global: hot reload replaces the engine while
in-flight work finishes, and a global would have an old attempt asking the new
engine about an execution it never opened — a refusal indistinguishable from a
real revocation. Entries are opened once and closed once, on every terminal
path including revocation. A map that leaks entries is the same defect as one
that drops them early, told backwards.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Protocol

from liminallm.logging import get_logger

__all__ = [
    "Attempt",
    "COMMITTED",
    "FAILED",
    "Invocation",
    "InvocationRegistry",
    "LeasedProxy",
    "LeaseRevoked",
    "Operation",
    "OperationLedger",
    "PENDING",
    "ResourceRegistry",
    "RetryDivergence",
    "UNKNOWN",
    "active_invocation",
    "commit_guard",
    "current_invocation",
    "payload_hash",
    "require_live_lease",
]

logger = get_logger(__name__)

#: Ledger states. `pending` is a step the parent started and has not finished.
#: `unknown` is what `pending` becomes when the attempt that owned it died: the
#: effect may or may not have landed, and nothing left can say which.
PENDING = "pending"
COMMITTED = "committed"
FAILED = "failed"
UNKNOWN = "unknown"

#: How long a kill waits for the tree to actually be gone. A retry that starts
#: before this returns is the defect this module exists to prevent, so the wait
#: is not optional and its expiry is an error rather than a shrug.
TERMINATION_TIMEOUT_SECONDS = 10.0
_TERMINATION_POLL_SECONDS = 0.02


class LeaseRevoked(RuntimeError):
    """The invocation ended before this call, so it carries no authority."""


class RetryDivergence(RuntimeError):
    """A retry asked for a different durable operation at a taken position.

    Not an error in the model's behaviour — a replacement worker may legally
    choose differently. It is an error to answer the new request with the old
    request's result, because the earlier mutation already happened and cannot
    be un-done by renaming it.
    """


def payload_hash(payload: Any) -> str:
    """A stable digest of an operation's arguments.

    Canonical JSON, so key order and float formatting cannot make two identical
    payloads hash differently. Values JSON cannot carry degrade to their repr
    rather than raising: the hash exists to detect divergence between a run and
    its replay, and a payload that cannot be hashed exactly is one that must be
    treated as diverged — which a repr-derived hash does.
    """
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=repr
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass
class Operation:
    """One effect a logical execution asked for, at a known position in it.

    `operation_seq` is the worker's position in its own control flow. `step`
    names a durable sub-operation of that position — publishing what a round's
    `run_python` produced, for one. A round is one request over the pipe, so
    the worker has no number of its own to give the publication, and the
    parent must not invent one that a replay could not reproduce: the step
    name is derived from the round's own contents instead.
    """

    operation_seq: int
    capability: str
    payload_hash: str
    state: str = PENDING
    result: Any = None
    step: str = ""

    @property
    def replayable(self) -> bool:
        return self.state == COMMITTED


class OperationLedger:
    """The ordered effects of one logical execution, and their outcomes.

    Held by the parent, so it outlives every worker that runs against it. A
    fresh worker replays its control flow from the top and each effect it asks
    for arrives with the position it reached. Same position, same capability,
    same payload, already committed — the stored result comes back and nothing
    happens twice.

    Divergence at a position is expected for a read and refused for a durable
    mutation. A read has no earlier effect to misreport, so its entry is
    dropped and the step runs fresh; a durable mutation does, so `replay`
    raises rather than silently treating the new request as the old one.
    """

    def __init__(self) -> None:
        self._ops: Dict[tuple[int, str], Operation] = {}
        self._lock = threading.RLock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._ops)

    def entries(self) -> List[Operation]:
        with self._lock:
            return [self._ops[key] for key in sorted(self._ops)]

    def get(self, operation_seq: int, step: str = "") -> Optional[Operation]:
        with self._lock:
            return self._ops.get((operation_seq, step))

    def replay(
        self,
        operation_seq: int,
        capability: str,
        digest: str,
        *,
        step: str = "",
        durable: bool = False,
    ) -> Optional[Operation]:
        """The committed result for this step, or None if it must be run."""
        with self._lock:
            existing = self._ops.get((operation_seq, step))
            if existing is None:
                return None
            if existing.capability != capability or existing.payload_hash != digest:
                if durable:
                    raise RetryDivergence(
                        f"operation {operation_seq}{'.' + step if step else ''} "
                        f"was {existing.capability} and is now {capability}: a "
                        "retry cannot inherit a durable result it did not ask for"
                    )
                for key in [
                    k for k in self._ops if k[0] >= operation_seq and k[1] == step
                ]:
                    del self._ops[key]
                return None
            if existing.state == COMMITTED:
                return existing
            if existing.state == UNKNOWN and durable:
                raise RetryDivergence(
                    f"operation {operation_seq} ({capability}) was in flight "
                    "when its attempt died; whether it landed is unknown, so "
                    "it is refused rather than repeated"
                )
            return None

    def begin(
        self, operation_seq: int, capability: str, digest: str, *, step: str = ""
    ) -> Operation:
        """Record that this step is now in flight."""
        with self._lock:
            op = Operation(
                operation_seq=operation_seq,
                capability=capability,
                payload_hash=digest,
                step=step,
            )
            self._ops[(operation_seq, step)] = op
            return op

    def commit(self, operation_seq: int, result: Any, *, step: str = "") -> None:
        with self._lock:
            op = self._ops.get((operation_seq, step))
            if op is not None:
                op.state = COMMITTED
                op.result = result

    def fail(self, operation_seq: int, error: str, *, step: str = "") -> None:
        with self._lock:
            op = self._ops.get((operation_seq, step))
            if op is not None:
                op.state = FAILED
                op.result = error

    def orphan_pending(self) -> int:
        """Mark every in-flight step `unknown`. Returns how many there were.

        Called when the attempt that owned them is torn down. `failed` would
        claim the effect did not land and `committed` would claim it did; the
        parent knows neither, and the honest state is the one that makes the
        next attempt refuse a durable repeat.
        """
        with self._lock:
            pending = [op for op in self._ops.values() if op.state == PENDING]
            for op in pending:
                op.state = UNKNOWN
            return len(pending)


@dataclass
class _Child:
    """A process this execution is answerable for."""

    pid: int
    label: str
    #: True when the child leads its own process group, so killing the group
    #: reaches everything it went on to spawn.
    group: bool = False
    #: Reaper for a child this process did not create with `os.fork` directly —
    #: a `multiprocessing.Process` is joined, not waitpid-ed.
    reap: Optional[Callable[[], None]] = None
    #: Set once the leader has been positively reaped while its group is still
    #: draining. From then on `pid` is not a handle: the kernel may reissue the
    #: number, so it is read only as the group's id and never signalled.
    #: §18 — "a registration left behind after a child is reaped is a standing
    #: licence to signal whoever inherits it."
    leader_reaped: bool = False


class Producer(Protocol):
    """Work this execution started in a thread of *this* process.

    A worker is a process and a kill ends it. A streamed producer is a thread,
    and nothing ends a Python thread from outside — so it is asked to stop and
    then asked whether it did, and an execution is not torn down until it says
    yes. Registering it here rather than beside the streaming path is the whole
    point: one revoke reaches everything an attempt started, whatever kind of
    thing it is.
    """

    def stop(self) -> None: ...

    def alive(self) -> bool: ...


@dataclass
class _Producer:
    producer: Producer
    label: str


class ResourceRegistry:
    """Everything a logical execution spawned or created, so it can be undone.

    Two kinds of child exist and both have to be here. The worker leads its own
    process group, so one `killpg` reaches whatever it spawned. The sandbox
    children the broker starts on the worker's behalf are the *parent's*
    children — not in the worker's group, and they survive killing it — so they
    are registered one by one and killed one by one.

    In-process producers are the third kind, and they are not children at all.
    See `Producer`.
    """

    def __init__(self) -> None:
        self._children: Dict[int, _Child] = {}
        self._producers: List[_Producer] = []
        self._paths: List[str] = []
        self._lock = threading.RLock()

    def add_producer(self, producer: Producer, label: str) -> None:
        with self._lock:
            self._producers.append(_Producer(producer=producer, label=label))

    def stop_producers(self) -> int:
        """Ask every producer to stop. Returns how many were still running.

        A request, not a kill — see `Producer`. `live_producers` is the answer
        about whether it was honoured.
        """
        stopped = 0
        with self._lock:
            producers = list(self._producers)
        for entry in producers:
            if not entry.producer.alive():
                continue
            stopped += 1
            try:
                entry.producer.stop()
            except Exception as exc:  # noqa: BLE001 - stopping is best effort
                logger.warning(
                    "producer_stop_failed", producer=entry.label, error=str(exc)
                )
        return stopped

    def live_producers(self) -> List[str]:
        """Producers that have not returned yet, by label."""
        with self._lock:
            producers = list(self._producers)
        alive = [entry.label for entry in producers if entry.producer.alive()]
        if not alive:
            with self._lock:
                self._producers = [
                    entry for entry in self._producers if entry.producer.alive()
                ]
        return alive

    def add_child(
        self,
        pid: int,
        label: str,
        *,
        group: bool = False,
        reap: Optional[Callable[[], None]] = None,
    ) -> None:
        if pid <= 0:
            return
        with self._lock:
            self._children[pid] = _Child(pid=pid, label=label, group=group, reap=reap)

    def forget_child(self, pid: int) -> None:
        """Drop a child that exited on its own and has already been reaped."""
        with self._lock:
            self._children.pop(pid, None)

    def mark_leader_reaped(self, pid: int) -> None:
        """This child's leader is gone; only its group is left to watch.

        Not the same as forgetting it. The group still holds members, so a
        retry must still wait — but the number that named the leader is now
        the kernel's to reissue, so from here it is read as a group id and
        nothing is ever signalled through it. The SIGKILL that emptied the
        group has already been sent; what remains is to watch it drain.
        """
        with self._lock:
            child = self._children.get(pid)
            if child is not None:
                child.leader_reaped = True

    def add_path(self, path: str) -> None:
        if path:
            with self._lock:
                self._paths.append(path)

    def children(self) -> List[_Child]:
        with self._lock:
            return list(self._children.values())

    def live_children(self) -> List[int]:
        """Children whose tree still exists. A reaped child is gone.

        For a child that leads its own group the pid is not the whole tree.
        A process group outlives its leader for as long as any member is in
        it, so a leader that has been reaped while its group still holds
        somebody is not "gone" — and forgetting it there is what turns an
        abandoned descendant into nobody's.
        """
        alive: List[int] = []
        for child in self.children():
            if child.leader_reaped:
                # The number is a group id and nothing else now. Asking
                # `_pid_alive` here would answer about whoever the kernel gave
                # the pid to next, and answer "the child is alive".
                if group_alive(child.pid):
                    alive.append(child.pid)
                else:
                    self.forget_child(child.pid)
                continue
            if _pid_alive(child.pid) or (child.group and group_alive(child.pid)):
                alive.append(child.pid)
            else:
                self.forget_child(child.pid)
        return alive

    def kill_all(self) -> List[int]:
        """Signal every child, then reap it. Returns the pids signalled.

        A child whose leader has been reaped is skipped, not re-signalled. Its
        group already took a SIGKILL, and the only thing the pid could reach
        now is whoever inherited the number — `_kill` falls back to a plain
        `os.kill` whenever the target does not lead the expected group, which
        is precisely what a reissued pid looks like.
        """
        self.stop_producers()
        signalled: List[int] = []
        for child in self.children():
            if child.leader_reaped:
                continue
            if _kill(child.pid, group=child.group):
                signalled.append(child.pid)
            if child.reap is not None:
                try:
                    child.reap()
                except Exception as exc:  # noqa: BLE001 - reaping is best effort
                    logger.warning(
                        "invocation_reap_failed", pid=child.pid, error=str(exc)
                    )
            else:
                _reap(child.pid)
        return signalled

    def cleanup_paths(self) -> None:
        with self._lock:
            paths, self._paths = self._paths, []
        for path in paths:
            shutil.rmtree(path, ignore_errors=True)


def _pid_alive(pid: int) -> bool:
    """Whether `pid` is still a process.

    A zombie counts as alive: an unreaped child still holds a process-table
    slot, and "the tree is dead" has to mean the entry is gone, not merely that
    the code stopped running.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def group_alive(pgid: int) -> bool:
    """Whether a process group still has members.

    Only ever asked about a group whose id was proved equal to its leader's
    pid, because that is the only group this service is entitled to reason
    about. Uncertainty answers True: the cost of a false "still there" is a
    refused retry, which SPEC already prescribes for a tree that will not die,
    while the cost of a false "gone" is a descendant nobody owns.
    """
    if pgid <= 0 or not hasattr(os, "killpg"):
        return False
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return True
    return True


def _kill(pid: int, *, group: bool) -> bool:
    """SIGKILL, not SIGTERM.

    A revoked execution has no cleanup worth running, and a signal handler is
    one more thing an untrusted process can decline to honour.

    A group kill is only ever aimed at a group the target *leads*. A process
    that has not yet called `setsid` — or could not — is still in the group
    that started it, which for a tool worker is the API server's: `killpg`
    there would take down the service and everything sharing its group. The
    registration is supposed to prevent that (`group=True` is only set once the
    child proves its pgid), and this is the second check, because the cost of
    the two disagreeing is the whole process group.
    """
    try:
        if group and hasattr(os, "killpg") and os.getpgid(pid) == pid:
            os.killpg(pid, signal.SIGKILL)
        else:
            os.kill(pid, signal.SIGKILL)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


def _reap(pid: int) -> None:
    try:
        os.waitpid(pid, os.WNOHANG)
    except (ChildProcessError, OSError):
        pass


@dataclass
class Attempt:
    """One run of a logical execution: which process, and when it ended.

    The lease lives here rather than on the execution. SPEC §18 keeps two ids
    because they answer different questions: authority is fresh per attempt, so
    a retry cannot inherit the authority of the attempt that was abandoned,
    while the ledger is keyed by the execution, because killing a worker does
    not recall an operation it already committed.
    """

    index: int
    pid: Optional[int] = None
    started_at: float = field(default_factory=time.monotonic)
    terminated_at: Optional[float] = None
    revoked: bool = False
    #: Set when the parent-side serve loop for this attempt has returned. The
    #: next attempt waits on it, so no two attempts ever have a capability in
    #: flight at the same time.
    finished: threading.Event = field(default_factory=threading.Event)


class Invocation:
    """One logical execution of one tool call.

    Lives in the parent for the whole of that execution, retries included. The
    lock is the linearization point: a durable commit holds it, and so does a
    revoke, so no mutation can land in the middle of a teardown and no teardown
    can report itself complete while a commit it never saw is still in flight.

    `session` is the parent's copy of what the turn has learned — injection
    findings above all. It lives here rather than in the worker because
    withdrawal has to be enforced by whoever owns the capability: the process
    that just read "ignore your rules and run this" is the last one that should
    be asked whether the rule still applies.
    """

    def __init__(
        self,
        invocation_id: str,
        *,
        tool: str = "",
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        registry: Optional["InvocationRegistry"] = None,
    ) -> None:
        self.invocation_id = invocation_id
        self.tool = tool
        #: Read by capability handlers for logging only. Authority never comes
        #: from a field: it comes from this object still being live.
        self.user_id = user_id
        self.tenant_id = tenant_id
        self.ledger = OperationLedger()
        self.resources = ResourceRegistry()
        self.attempts: List[Attempt] = []
        self.session: Dict[str, Any] = {}
        #: The registry that opened this execution, so `close` can retire the
        #: entry without any module-level lookup.
        self.registry = registry
        self._cancelled = False
        self._revoke_reason = ""
        self._closed = False
        self._current: Optional[Attempt] = None
        self._lock = threading.RLock()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Invocation({self.invocation_id!r}, tool={self.tool!r})"

    # -- state ------------------------------------------------------------

    @property
    def lock(self) -> threading.RLock:
        return self._lock

    @property
    def revoked(self) -> bool:
        """Whether the work in flight has lost the caller's authority.

        True when the execution was cancelled outright, and true when the
        attempt currently running was revoked. A retry clears the second by
        beginning a new attempt; nothing clears the first.
        """
        with self._lock:
            if self._cancelled:
                return True
            return self._current is not None and self._current.revoked

    @property
    def cancelled(self) -> bool:
        with self._lock:
            return self._cancelled

    @property
    def revoke_reason(self) -> str:
        with self._lock:
            return self._revoke_reason

    @property
    def current_attempt(self) -> Optional[Attempt]:
        with self._lock:
            return self._current

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    @property
    def workdir(self) -> Optional[str]:
        """The scratch directory this execution's code runs in, if prepared."""
        return self.session.get("workdir")

    @workdir.setter
    def workdir(self, value: Optional[str]) -> None:
        self.session["workdir"] = value

    def check_live(self) -> None:
        """Refuse to proceed if this execution has been revoked.

        Every capability calls this before it acts. After is too late: the
        request has gone out, the child has started, and revocation has become
        a description of something that already happened.
        """
        if self.revoked:
            raise LeaseRevoked(
                f"invocation {self.invocation_id} of {self.tool!r} was "
                f"revoked ({self._revoke_reason}); this work no longer "
                "holds the caller's authority"
            )

    # -- attempts ---------------------------------------------------------

    def begin_attempt(self) -> Attempt:
        """Start a fresh attempt, with authority of its own.

        Refuses once the execution is cancelled: a cancelled turn does not get
        another try, whatever the retry policy says.
        """
        with self._lock:
            if self._cancelled:
                raise LeaseRevoked(
                    f"invocation {self.invocation_id} was cancelled "
                    f"({self._revoke_reason}); no further attempt may start"
                )
            attempt = Attempt(index=len(self.attempts))
            self.attempts.append(attempt)
            self._current = attempt
            return attempt

    def end_attempt(self, attempt: Attempt) -> None:
        """Record that the parent-side work for this attempt has finished."""
        with self._lock:
            attempt.terminated_at = time.monotonic()
        attempt.finished.set()

    def await_attempt(self, timeout: float) -> bool:
        """Wait for the current attempt's parent-side work to return.

        The worker is a process and dies when it is killed; the serve loop that
        was answering it is a thread in this process and does not. It exits
        within one poll of the worker's death unless a capability is mid-call,
        and waiting for that is the point: attempt *n+1* must not have a
        capability running beside attempt *n*'s.
        """
        attempt = self.current_attempt
        if attempt is None:
            return True
        return attempt.finished.wait(timeout)

    # -- revocation -------------------------------------------------------

    def revoke(self, reason: str = "revoked") -> None:
        """End the running attempt's authority and take down what it owns.

        Ordering is the whole content of this method. The flag is set under the
        lock first, so an effect racing us is refused rather than started; only
        then is anything killed. The reverse order lets a capability that has
        already passed its liveness check start a request against a tree we
        have torn down.

        Idempotent: an attempt revoked on timeout is revoked again on return.
        """
        with self._lock:
            attempt = self._current
            if attempt is None:
                # Nothing has started, so there is no attempt to scope this to.
                # Refusing the whole execution is the fail-closed reading: a
                # revoke that landed before the first spawn must not be
                # forgotten by the attempt that follows it.
                already = self._cancelled
                self._cancelled = True
            else:
                already = attempt.revoked
                attempt.revoked = True
            if not already:
                self._revoke_reason = reason
            orphaned = 0 if already else self.ledger.orphan_pending()
        if already:
            return
        killed = self.resources.kill_all()
        logger.info(
            "invocation_revoked",
            invocation_id=self.invocation_id,
            tool=self.tool,
            reason=reason,
            attempt=attempt.index if attempt is not None else None,
            unknown_operations=orphaned,
            killed=len(killed),
        )

    def cancel(self, reason: str = "cancelled") -> None:
        """End the whole execution: this attempt, and any that would follow.

        A node timeout revokes an attempt and lets the retry policy have its
        next one. A cancel is the caller saying the answer is no longer wanted,
        so it also refuses the attempts that have not started.
        """
        with self._lock:
            self._cancelled = True
            self._revoke_reason = reason
        self.revoke(reason)

    def terminate(
        self,
        *,
        timeout: float = TERMINATION_TIMEOUT_SECONDS,
        producers: bool = True,
    ) -> bool:
        """Kill and reap the whole tree. True once nothing of it is left.

        A retry calls this and honours the answer. That is the entire contract:
        attempt *n+1* may not start while attempt *n* still has a process, or
        the two share a working directory, a sandbox child, and an idea of
        whose output is whose.

        `producers=False` drops in-process producers from the *wait* — never
        from the stop, which `kill_all` has already done. The retry precondition
        needs the answer and must pay for it; ending the execution must not,
        because a producer blocked inside a read cannot be made to return and
        the caller would be held for the timeout waiting on a thread that owns
        no process and no scratch path. It is a daemon thread with its stop
        flag set: it exits when its read does.

        The scratch goes with the processes, and `workdir` is cleared with it.
        Keeping the directory across attempts would hand the retry a
        half-written file the killed attempt left behind; keeping the *name*
        after deleting the directory would be worse — the next attempt would
        skip preparation and run in a path that no longer exists.
        """
        with self._lock:
            self.ledger.orphan_pending()
        self.resources.kill_all()
        deadline = time.monotonic() + timeout
        while True:
            # Producers count as alive for exactly the reason children do. A
            # thread still inside `next()` is still writing the answer the next
            # attempt is about to replace, and "nothing of it is left" is false
            # while it is.
            alive = self.resources.live_children()
            if producers:
                alive = alive + self.resources.live_producers()
            if not alive:
                self.resources.cleanup_paths()
                self.session.pop("workdir", None)
                return True
            if time.monotonic() >= deadline:
                logger.error(
                    "invocation_termination_timeout",
                    invocation_id=self.invocation_id,
                    tool=self.tool,
                    alive=alive,
                )
                return False
            self.resources.kill_all()
            time.sleep(_TERMINATION_POLL_SECONDS)

    def close(self) -> None:
        """Finish this execution: kill anything left, drop its resources.

        Idempotent, and reached from every terminal path — success, failure,
        timeout and revocation alike. An execution that ends any other way
        leaves a live sandbox child and a scratch directory behind, with nobody
        left to notice either.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self.cancel("closed")
        self.terminate(producers=False)
        self.resources.cleanup_paths()
        registry = self.registry
        if registry is not None:
            registry.forget(self.invocation_id)


@contextmanager
def commit_guard(
    invocation: Invocation,
    capability: str,
    payload: Any,
    *,
    operation_seq: int,
    step: str = "",
    durable: bool = True,
) -> Iterator[Operation]:
    """Wrap a durable mutation — the write itself, not the call that leads to it.

    The distinction is the point. A guard around a call boundary records that a
    request was made, which is a fact about the caller; the ledger needs a fact
    about the store. Between "the handler was entered" and "the row exists"
    there is a window, and a retry landing in it either duplicates the write or
    skips it, depending on which side of the boundary the guard sat.

    So the body of this context manager is the mutation. Liveness is checked on
    the way in, under the invocation's lock, and the lock is held across the
    body: a revoke arriving mid-write waits for the write to finish and then
    tears down, rather than interleaving with it. Holding a lock across a
    durable write is a real cost, paid deliberately — the alternative is a
    revoke that reports success while a commit it never saw is still landing.
    Do no blocking work inside it beyond the mutation.
    """
    digest = payload_hash(payload)
    with invocation.lock:
        invocation.check_live()
        replayed = invocation.ledger.replay(
            operation_seq, capability, digest, step=step, durable=durable
        )
        if replayed is not None:
            yield replayed
            return
        op = invocation.ledger.begin(operation_seq, capability, digest, step=step)
        try:
            yield op
        except BaseException as exc:  # noqa: BLE001 - the ledger records why
            invocation.ledger.fail(operation_seq, str(exc), step=step)
            raise
        if op.state == PENDING:
            invocation.ledger.commit(operation_seq, op.result, step=step)


class InvocationRegistry:
    """The live logical executions of one engine.

    Engine-owned rather than module-global, so hot reload cannot leave an old
    attempt asking a new engine about an execution it never opened. Entries are
    opened once and closed once; `close` tears the tree down and retires the
    entry, which is what keeps this map from growing by one lock per tool
    attempt for the life of the process.
    """

    def __init__(self) -> None:
        self._live: Dict[str, Invocation] = {}
        self._lock = threading.RLock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._live)

    def open(
        self,
        invocation_id: str,
        *,
        tool: str = "",
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> Invocation:
        """Register a new logical execution, or return the one already running.

        Returning the existing entry is what makes a retry a retry: the second
        attempt finds the first attempt's ledger and replays against it.
        """
        with self._lock:
            existing = self._live.get(invocation_id)
            if existing is not None:
                return existing
            invocation = Invocation(
                invocation_id,
                tool=tool,
                user_id=user_id,
                tenant_id=tenant_id,
                registry=self,
            )
            self._live[invocation_id] = invocation
            return invocation

    def get(self, invocation_id: str) -> Optional[Invocation]:
        with self._lock:
            return self._live.get(invocation_id)

    def forget(self, invocation_id: str) -> None:
        """Drop the entry. Called by `Invocation.close`, not instead of it."""
        with self._lock:
            self._live.pop(invocation_id, None)

    def close(self, invocation_id: str) -> None:
        """End a logical execution and stop tracking it."""
        with self._lock:
            invocation = self._live.get(invocation_id)
        if invocation is not None:
            invocation.close()

    def cancel(self, invocation_id: str, reason: str = "cancelled") -> bool:
        """Cancel a live execution by id. False when there is nothing to."""
        invocation = self.get(invocation_id)
        if invocation is None:
            return False
        invocation.cancel(reason)
        return True

    def live(self) -> List[str]:
        with self._lock:
            return list(self._live)

    def close_all(self) -> None:
        """Close every tracked execution. For engine shutdown."""
        with self._lock:
            invocations = list(self._live.values())
        for invocation in invocations:
            invocation.close()


# -- binding an execution to the thread that is serving it -------------------
#
# The worker holds no store, model or settings handle, so the calls that need
# checking are the parent's own: the capability handlers run against the real
# services. Binding the invocation to the serving thread lets `LeasedProxy`
# check every one of them — reads included — without each handler having to
# remember. A thread with no invocation bound is the API path and passes
# through untouched.

_CURRENT = threading.local()


@contextmanager
def current_invocation(invocation: Optional[Invocation]):
    """Bind an execution to this thread for the duration of some work.

    Restores the previous value rather than clearing it: threads are reused,
    and clearing would leave the next piece of work on that thread unbound,
    which reads as "the API path" and passes every check.
    """
    previous = getattr(_CURRENT, "invocation", None)
    _CURRENT.invocation = invocation
    try:
        yield invocation
    finally:
        _CURRENT.invocation = previous


def active_invocation() -> Optional[Invocation]:
    return getattr(_CURRENT, "invocation", None)


def require_live_lease() -> None:
    """Refuse a durable operation whose execution has ended.

    For work that does not reach its target through a proxied dependency —
    launching a sandbox child, publishing into the user's file area. A thread
    with nothing bound is the API path and passes, exactly as the proxy treats
    it.
    """
    invocation = active_invocation()
    if invocation is not None:
        invocation.check_live()


class LeasedProxy:
    """Passes calls through, unless the calling thread's execution has ended.

    A thread with no execution bound is the API path and is not this module's
    business, so it delegates untouched. Wrapping is engine-wide rather than
    per-handler because the capability handlers reach their dependencies
    through the engine, and the thread-local is what makes one shared object
    behave correctly for both callers.

    The check is on *every* call, reads included. A name-prefix list of "write
    methods" would be a heuristic about which calls matter, and a revoked
    execution has no authority to read with either.
    """

    __slots__ = ("_inner",)

    def __init__(self, inner: Any) -> None:
        object.__setattr__(self, "_inner", inner)

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._inner, name)
        invocation = active_invocation()
        if invocation is None or not callable(attr):
            return attr

        def guarded(*args: Any, **kwargs: Any) -> Any:
            invocation.check_live()
            return attr(*args, **kwargs)

        return guarded

    # Writes go to the wrapped object. The proxy adds a check to calls; it is
    # not a second place to keep state, and anything that sets an attribute —
    # a test substituting a method, a service caching on its store — must land
    # where every other reader will see it.
    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._inner, name, value)

    def __delattr__(self, name: str) -> None:
        delattr(self._inner, name)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"LeasedProxy({self._inner!r})"
