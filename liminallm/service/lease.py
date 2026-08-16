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
from dataclasses import dataclass
from typing import Any, Optional

__all__ = [
    "InvocationBroker",
    "LeaseRevoked",
    "LeasedProxy",
    "ToolInvocation",
    "active_invocation",
    "current_invocation",
    "require_live_lease",
    "set_broker",
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
        )
        with self._lock:
            self._live.add(invocation.invocation_id)
        return invocation

    def revoke(self, invocation: ToolInvocation) -> None:
        """Idempotent: a lease revoked on timeout is revoked again on return."""
        with self._lock:
            self._live.discard(invocation.invocation_id)

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


# The broker that answers for leases in this process. Durable operations that
# do not reach their target through a proxied dependency — publishing into the
# user's file area is the one that exists today — ask it directly, and cannot
# be handed a proxy to go through instead.
_BROKER: Optional["InvocationBroker"] = None


def set_broker(broker: "InvocationBroker") -> None:
    global _BROKER
    _BROKER = broker


def require_live_lease() -> None:
    """Refuse a durable operation whose invocation has ended.

    A thread with no lease is the API path and passes, exactly as the proxy
    treats it. A leased thread with no registered broker is a wiring mistake
    rather than an authorization one, so it fails closed and says which.
    """
    invocation = active_invocation()
    if invocation is None:
        return
    if _BROKER is None:  # pragma: no cover - defensive
        raise LeaseRevoked(
            "a leased thread reached a durable operation but no broker is "
            "registered to say whether the lease is live"
        )
    _BROKER.check(invocation)


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
