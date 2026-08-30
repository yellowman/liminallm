"""Replaying a completed request instead of repeating its side effects.

A retry with the same Idempotency-Key gets the original response back, not a
second upload or a second turn. The slot is claimed atomically, so concurrent
retries cannot both start work - the loser gets a 409.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

from liminallm.api.errors import http_error
from liminallm.api.schemas import Envelope
from liminallm.service.invocation import Invocation, commit_guard
from liminallm.service.runtime import (
    IDEMPOTENCY_TTL_SECONDS,
    _acquire_idempotency_slot,
    _set_cached_idempotency_record,
    get_runtime,
)


async def resolve(
    route: str,
    user_id: str,
    idempotency_key: Optional[str],
    *,
    require: bool = False,
    request_id: Optional[str] = None,
) -> tuple[str, Optional[Envelope]]:
    """Claim the slot, or return the completed response to replay. A cached
    envelope means the work is done and must not be repeated."""
    request_id = request_id or str(uuid4())
    runtime = get_runtime()
    if not idempotency_key:
        if require:
            raise http_error(
                "validation_error", "Idempotency-Key header required", status_code=400
            )
        return request_id, None

    in_progress_record = {
        "status": "in_progress",
        "request_id": request_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    acquired, existing_record = await _acquire_idempotency_slot(
        runtime,
        route,
        user_id,
        idempotency_key,
        in_progress_record,
        ttl_seconds=IDEMPOTENCY_TTL_SECONDS,
    )
    if acquired:
        return request_id, None

    # A prior "failed" record is reclaimed atomically inside
    # _acquire_idempotency_slot (so `acquired` would be True), meaning a
    # non-acquired record here is either in progress or a completed result.
    if existing_record:
        status = existing_record.get("status")
        if status == "completed" and existing_record.get("response"):
            # Replay only successful results, so duplicate side effects are
            # avoided but a failure can be retried.
            response_payload = existing_record.get("response", {})
            if "request_id" not in response_payload:
                response_payload["request_id"] = existing_record.get(
                    "request_id", request_id
                )
            return existing_record.get("request_id", request_id), Envelope(
                **response_payload
            )

    raise http_error("conflict", "request in progress", status_code=409)


async def store(
    route: str,
    user_id: str,
    idempotency_key: Optional[str],
    envelope: Envelope,
    status: str = "completed",
) -> None:
    """Record the outcome so a retry can replay it."""
    if not idempotency_key:
        return
    runtime = get_runtime()
    await _set_cached_idempotency_record(
        runtime,
        route,
        user_id,
        idempotency_key,
        {
            "status": status,
            "request_id": envelope.request_id,
            # `mode="json"` because this record is JSON-encoded on the way to
            # the cache. A plain model_dump leaves datetimes as datetimes, so
            # every route whose response carries `created_at` answered 500 the
            # moment a client sent the Idempotency-Key SPEC §18 invites - and
            # the same request without the header succeeded, which is why it
            # went unnoticed.
            "response": envelope.model_dump(mode="json"),
        },
        ttl_seconds=IDEMPOTENCY_TTL_SECONDS,
    )


class IdempotencyGuard:
    """Scope a request to its idempotency slot.

    Entry claims it, or surfaces a completed result as ``.cached``. An
    exception records the failure, so a retry is not deadlocked behind an
    "in progress" record that never resolves.

    The slot is a fact about the *request*. What landed in the store is a fact
    about the store, and the two are not the same: between "the bytes were
    written" and "the response was recorded" there is a window, and the slot
    describes neither side of it. ``commit`` records each durable mutation as
    it happens, so this guard can go on describing the request.

    The two answer different questions and neither replaces the other. Replay
    across requests is the slot's job - it is in Redis, so it survives the
    process and the replica (§22). The ledger here is in memory and lives for
    one request: what it buys is that each mutation is recorded at the moment
    it lands, in order, so a route that makes several of them can say which
    ones did.
    """

    def __init__(
        self,
        route: str,
        user_id: str,
        idempotency_key: Optional[str],
        *,
        require: bool = False,
    ):
        self.route = route
        self.user_id = user_id
        self.idempotency_key = idempotency_key
        self.require = require
        self.request_id: Optional[str] = None
        self.cached: Optional[Envelope] = None
        self._stored = False
        #: The mutations this request has made, in the order it made them.
        #: Opened lazily: a request that mutates nothing needs no ledger.
        self._invocation: Optional[Invocation] = None
        self._operation_seq = 0

    @property
    def invocation(self) -> Invocation:
        """This request's ledger, opened on first use and keyed by request id.

        Taken from the engine's registry rather than a module global, for the
        same reason the tool path does: hot reload replaces the engine while
        in-flight work finishes (SPEC §18).
        """
        if self._invocation is None:
            self._invocation = get_runtime().workflow.invocations.open(
                f"request:{self.request_id}", tool=self.route, user_id=self.user_id
            )
            self._invocation.begin_attempt()
        return self._invocation

    def commit(self, capability: str, payload: Any):
        """Guard one durable mutation. Wrap the write, not the handler.

        Each mutation goes inside its own guard, so the ledger records what
        actually landed rather than what was attempted::

            with idem.commit("files.write", {"path": name}) as op:
                if not op.replayable:
                    dest.write_bytes(contents)

        A retry finds the committed step and skips it. A guard around the
        handler cannot do that: it only knows the request was entered.
        """
        self._operation_seq += 1
        return commit_guard(
            self.invocation, capability, payload, operation_seq=self._operation_seq
        )

    async def __aenter__(self) -> "IdempotencyGuard":
        self.request_id = self.request_id or str(uuid4())
        self.request_id, self.cached = await resolve(
            self.route,
            self.user_id,
            self.idempotency_key,
            require=self.require,
            request_id=self.request_id,
        )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        if exc and not self._stored and self.request_id:
            await store(
                self.route,
                self.user_id,
                self.idempotency_key,
                Envelope(
                    status="error",
                    error={"code": "server_error", "message": str(exc)},
                    request_id=self.request_id,
                ),
                status="failed",
            )
            self._stored = True
        if self._invocation is not None:
            # Synchronous on purpose: a request's ledger owns no processes and
            # no scratch, so closing it is a dict pop. The tool path closes off
            # the event loop because there it kills and reaps.
            self._invocation.close()
            self._invocation = None
        return False

    async def store_result(
        self, envelope: Envelope, *, status: str = "completed"
    ) -> None:
        """Record the response. Only a final status closes the slot."""
        await store(
            self.route, self.user_id, self.idempotency_key, envelope, status=status
        )
        if status in {"completed", "failed"}:
            self._stored = True
