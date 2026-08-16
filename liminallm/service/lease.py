"""Authority that ends when the invocation ends.

A tool handler runs on a `ThreadPoolExecutor` thread. When the node times
out, `asyncio.wait_for` gives up on the *future* — the thread keeps running,
because `concurrent.futures` can only cancel work that has not started, and
Python cannot kill a thread at all. The handler therefore carried on holding
the caller's `user_id` and `tenant_id`, and could still write to the store,
spend on the model, or publish into the user's file area, long after the
request that authorized it had been reported failed. With retries the engine
would start a second worker on top of the first.

So the thing that ends is not the thread. It is the **lease**:

* the worker never holds authority, only a `ToolInvocation` naming one;
* every authority-bearing call it makes goes through a proxy that asks the
  broker whether that lease is still live;
* on timeout the lease is revoked **before** the worker is abandoned, never
  after — the reverse order leaves exactly the window this exists to close;
* the worker is reaped before anything retries in its name.

The check is on *every* call a leased thread makes through the proxy, reads
included. A name-prefix list of "write methods" would be a heuristic about
which calls matter, and a revoked invocation has no authority to read with
either.
"""
from __future__ import annotations

import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Optional

__all__ = [
    "InvocationBroker",
    "LeaseRevoked",
    "LeasedProxy",
    "ToolInvocation",
    "active_invocation",
    "current_invocation",
    "require_live_lease",
]

_CURRENT = threading.local()


class LeaseRevoked(RuntimeError):
    """The invocation ended before this call, so it carries no authority."""


@dataclass(frozen=True)
class ToolInvocation:
    """One tool call's identity. Held by the worker; proves nothing by itself.

    The worker can read these fields, but reading them is not authority —
    only the broker says whether the lease behind `invocation_id` is live.
    """

    invocation_id: str
    tool_name: str
    user_id: Optional[str]
    tenant_id: Optional[str]
    artifact_id: Optional[str] = None
    # One node execution, stable across its retries. The lease is keyed by
    # `invocation_id` — attempt two must not inherit attempt one's authority —
    # but a durable idempotency key has to be keyed by the *logical* execution,
    # or attempt two duplicates an operation attempt one already submitted at
    # the boundary. Killing a worker does not recall what it sent.
    logical_execution_id: str = ""
    # The broker that answers for this lease, carried by the invocation
    # rather than looked up in a process global. Runtime hot reload replaces
    # the engine while in-flight workers finish, so a global would have an
    # old worker asking the *new* engine's broker about an invocation it
    # never issued. Not compared or repr'd: it is a channel, not identity.
    broker: Any = field(default=None, repr=False, compare=False)

    def operation_key(self, operation: str) -> str:
        """A key stable across retries of one logical node execution."""
        return f"{self.logical_execution_id or self.invocation_id}:{operation}"


class InvocationBroker:
    """The only thing that knows which leases are live.

    Kept deliberately small: issue, revoke, ask. It is consulted from worker
    threads and mutated from the event loop, so every operation takes the
    lock.
    """

    def __init__(self) -> None:
        self._live: set[str] = set()
        self._lock = threading.Lock()
        # One guard per live invocation. `commit_guard` and `revoke` contend
        # on it, which is what makes the two orderings the only two.
        self._guards: dict[str, threading.Lock] = {}

    def issue(
        self,
        tool_name: str,
        *,
        user_id: Optional[str],
        tenant_id: Optional[str],
        artifact_id: Optional[str] = None,
        logical_execution_id: Optional[str] = None,
    ) -> ToolInvocation:
        invocation = ToolInvocation(
            invocation_id=uuid.uuid4().hex,
            tool_name=tool_name,
            user_id=user_id,
            tenant_id=tenant_id,
            artifact_id=artifact_id,
            # A call with no node behind it is its own logical execution.
            logical_execution_id=logical_execution_id or uuid.uuid4().hex,
            broker=self,
        )
        with self._lock:
            self._live.add(invocation.invocation_id)
            self._guards[invocation.invocation_id] = threading.Lock()
        return invocation

    def _guard(self, invocation: ToolInvocation) -> threading.Lock:
        with self._lock:
            guard = self._guards.get(invocation.invocation_id)
            if guard is None:
                # Already revoked and cleaned up. A fresh lock is correct:
                # `check` inside the guard will refuse anyway.
                guard = threading.Lock()
                self._guards[invocation.invocation_id] = guard
            return guard

    def revoke(self, invocation: ToolInvocation) -> None:
        """Idempotent: a lease revoked on timeout is revoked again on return.

        Takes the invocation's commit guard first. A commit already inside
        the guard completes before this returns, and one that has not yet
        entered finds the lease dead when it does — so the timeout path never
        reports revocation complete while an authorized commit is still in
        flight.
        """
        with self._guard(invocation):
            with self._lock:
                self._live.discard(invocation.invocation_id)

    @contextmanager
    def commit_guard(self, invocation: ToolInvocation):
        """Hold the linearization point across a durable commit.

        A bare `check()` before `COMMIT` is not enough: revocation can land
        between the two. Holding this guard leaves exactly two histories --
        the commit wins and revocation waits for it, or revocation wins and
        the commit is refused. Do no blocking work inside it.
        """
        with self._guard(invocation):
            self.check(invocation)
            yield

    def is_live(self, invocation: ToolInvocation) -> bool:
        with self._lock:
            return invocation.invocation_id in self._live

    def check(self, invocation: ToolInvocation) -> None:
        if not self.is_live(invocation):
            raise LeaseRevoked(
                f"invocation {invocation.invocation_id} of {invocation.tool_name!r} "
                "was revoked; this worker no longer holds the caller's authority"
            )


@contextmanager
def current_invocation(invocation: Optional[ToolInvocation]):
    """Bind a lease to this thread for the duration of the handler.

    Restores the previous value rather than clearing it, because the pool
    reuses threads: clearing would leave the next invocation on that thread
    unleased, which reads as "the API path" and passes every check.
    """
    previous = getattr(_CURRENT, "invocation", None)
    _CURRENT.invocation = invocation
    try:
        yield invocation
    finally:
        _CURRENT.invocation = previous


def active_invocation() -> Optional[ToolInvocation]:
    return getattr(_CURRENT, "invocation", None)


def require_live_lease() -> None:
    """Refuse a durable operation whose invocation has ended.

    For work that does not reach its target through a proxied dependency —
    web access, launching a sandbox child, publishing into the user's file
    area. A thread with no lease is the API path and passes, exactly as the
    proxy treats it.

    The broker comes from the invocation, never from a process global: hot
    reload replaces the engine while in-flight workers finish, and a global
    would have an old worker asking a new engine's broker about a lease it
    never issued.
    """
    invocation = active_invocation()
    if invocation is None:
        return
    if invocation.broker is None:  # pragma: no cover - defensive
        raise LeaseRevoked(
            f"invocation {invocation.invocation_id} carries no broker, so "
            "nothing can say whether its lease is live"
        )
    invocation.broker.check(invocation)


class LeasedProxy:
    """Passes calls through, unless the calling thread holds a dead lease.

    A thread with no lease is the API path and is not this module's business,
    so it delegates untouched. Wrapping is engine-wide rather than per-worker
    because handlers reach their dependencies through the engine, and the
    thread-local is what makes one shared object behave correctly for both.
    """

    __slots__ = ("_inner", "_broker")

    def __init__(self, inner: Any, broker: InvocationBroker) -> None:
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_broker", broker)

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._inner, name)
        invocation = active_invocation()
        if invocation is None or not callable(attr):
            return attr

        def guarded(*args: Any, **kwargs: Any) -> Any:
            self._broker.check(invocation)
            return attr(*args, **kwargs)

        return guarded

    # Writes go to the wrapped object. The proxy adds a check to calls; it is
    # not a second place to keep state, and anything that sets an attribute —
    # a test substituting a method, a service caching on its store — must
    # land where every other reader will see it.
    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._inner, name, value)

    def __delattr__(self, name: str) -> None:
        delattr(self._inner, name)

    # Repr should describe the thing being wrapped: the proxy is a policy,
    # not a different object.
    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"LeasedProxy({self._inner!r})"
