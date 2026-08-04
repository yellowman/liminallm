"""Replaying a completed request instead of repeating its side effects.

A retry with the same Idempotency-Key gets the original response back, not a
second upload or a second turn. The slot is claimed atomically, so concurrent
retries cannot both start work — the loser gets a 409.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from liminallm.api.errors import http_error
from liminallm.api.schemas import Envelope
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
            "response": envelope.model_dump(),
        },
        ttl_seconds=IDEMPOTENCY_TTL_SECONDS,
    )


class IdempotencyGuard:
    """Scope a request to its idempotency slot.

    Entry claims it, or surfaces a completed result as ``.cached``. An
    exception records the failure, so a retry is not deadlocked behind an
    "in progress" record that never resolves.
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

    async def store_error(self, message: str) -> None:
        if not self.request_id:
            return
        await self.store_result(
            Envelope(
                status="error",
                error={"code": "server_error", "message": message},
                request_id=self.request_id,
            ),
            status="failed",
        )
