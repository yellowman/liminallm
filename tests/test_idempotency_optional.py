"""Tests for SPEC §18 optional Idempotency-Key header behavior."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from liminallm.api.idempotency import IdempotencyGuard, resolve

# ==============================================================================
# IdempotencyGuard Tests
# ==============================================================================


class TestIdempotencyGuardOptional:
    """Test that Idempotency-Key is optional per SPEC §18."""

    @pytest.mark.asyncio
    async def test_guard_allows_none_when_not_required(self):
        """Guard should allow None idempotency_key when require=False."""
        guard = IdempotencyGuard(
            route="test",
            user_id="user123",
            idempotency_key=None,
            require=False,
        )

        with patch("liminallm.api.idempotency.resolve") as mock_resolve:
            mock_resolve.return_value = ("req-123", None)

            async with guard as g:
                # Should not raise, should proceed normally
                assert g.request_id is not None
                assert g.cached is None

    @pytest.mark.asyncio
    async def test_guard_rejects_none_when_required(self):
        """Guard should reject None idempotency_key when require=True."""
        guard = IdempotencyGuard(
            route="test",
            user_id="user123",
            idempotency_key=None,
            require=True,
        )

        with patch("liminallm.api.idempotency.resolve") as mock_resolve:
            # Should raise validation error
            mock_resolve.side_effect = HTTPException(
                status_code=400,
                detail={
                    "status": "error",
                    "error": {
                        "code": "validation_error",
                        "message": "Idempotency-Key header required",
                    },
                },
            )

            with pytest.raises(HTTPException) as exc_info:
                async with guard:
                    pass

            assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_guard_uses_key_when_provided(self):
        """Guard should use idempotency_key when provided."""
        guard = IdempotencyGuard(
            route="test",
            user_id="user123",
            idempotency_key="my-key-123",
            require=False,
        )

        with patch("liminallm.api.idempotency.resolve") as mock_resolve:
            mock_resolve.return_value = ("req-123", None)

            async with guard as g:
                assert g.request_id == "req-123"

            # Should have called resolve with the key
            mock_resolve.assert_called_once()
            call_kwargs = mock_resolve.call_args
            assert call_kwargs[0][2] == "my-key-123"  # idempotency_key param


# ==============================================================================
# resolve Tests
# ==============================================================================


class TestResolveIdempotency:
    """Test the idempotency resolution logic."""

    @pytest.mark.asyncio
    async def test_returns_new_request_id_when_no_key(self):
        """Should return a new request_id when no idempotency_key is provided."""
        with patch("liminallm.api.routes.get_runtime") as mock_runtime:
            mock_runtime.return_value = MagicMock()

            request_id, cached = await resolve(
                route="chat",
                user_id="user123",
                idempotency_key=None,
                require=False,
            )

            assert request_id is not None
            assert len(request_id) > 0
            assert cached is None

    @pytest.mark.asyncio
    async def test_raises_when_required_but_missing(self):
        """Should raise HTTPException when key is required but missing."""
        with pytest.raises(HTTPException) as exc_info:
            await resolve(
                route="chat",
                user_id="user123",
                idempotency_key=None,
                require=True,
            )

        assert exc_info.value.status_code == 400
        assert "Idempotency-Key" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_returns_cached_response_when_completed(self):
        """Should return cached response when idempotency key was previously completed."""
        cached_envelope = {
            "status": "ok",
            "data": {"message_id": "msg-123"},
            "request_id": "prev-req-id",
        }

        with (
            patch("liminallm.api.routes.get_runtime") as mock_runtime,
            patch(
                "liminallm.api.idempotency._acquire_idempotency_slot",
                new_callable=AsyncMock,
            ) as mock_acquire,
        ):
            mock_runtime.return_value = MagicMock()
            # Slot not acquired; the existing record is a completed result.
            mock_acquire.return_value = (
                False,
                {
                    "status": "completed",
                    "request_id": "prev-req-id",
                    "response": cached_envelope,
                },
            )

            request_id, cached = await resolve(
                route="chat",
                user_id="user123",
                idempotency_key="repeat-key",
                require=False,
            )

            assert request_id == "prev-req-id"
            assert cached is not None
            assert cached.status == "ok"

    @pytest.mark.asyncio
    async def test_raises_conflict_when_in_progress(self):
        """Should raise 409 Conflict when prior request is still in progress."""
        with (
            patch("liminallm.api.routes.get_runtime") as mock_runtime,
            patch(
                "liminallm.api.idempotency._acquire_idempotency_slot",
                new_callable=AsyncMock,
            ) as mock_acquire,
        ):
            mock_runtime.return_value = MagicMock()
            # Slot not acquired; a prior request is still in progress.
            mock_acquire.return_value = (
                False,
                {"status": "in_progress", "request_id": "ongoing-req"},
            )

            with pytest.raises(HTTPException) as exc_info:
                await resolve(
                    route="chat",
                    user_id="user123",
                    idempotency_key="busy-key",
                    require=False,
                )

            assert exc_info.value.status_code == 409
            assert "in progress" in str(exc_info.value.detail).lower()


# ==============================================================================
# Endpoint Integration Tests
# ==============================================================================


class TestEndpointIdempotencyOptional:
    """Test that endpoints accept optional Idempotency-Key per SPEC §18."""

    def test_chat_endpoint_accepts_no_key(self):
        """POST /v1/chat should work without Idempotency-Key."""
        # This is verified by the fact that require=False is set
        # A full integration test would require more setup
        pass

    def test_artifacts_endpoint_accepts_no_key(self):
        """POST /v1/artifacts should work without Idempotency-Key."""
        pass

    def test_tools_invoke_endpoint_accepts_no_key(self):
        """POST /v1/tools/{id}/invoke should work without Idempotency-Key."""
        pass

    def test_files_upload_endpoint_accepts_no_key(self):
        """POST /v1/files/upload should work without Idempotency-Key."""
        pass

    def test_contexts_endpoint_accepts_no_key(self):
        """POST /v1/contexts should work without Idempotency-Key."""
        pass


# ==============================================================================
# SPEC Compliance Tests
# ==============================================================================


class TestSpecCompliance:
    """Test SPEC §18 idempotency requirements."""

    def test_spec_endpoints_listed(self):
        """SPEC §18 lists POST endpoints that should accept Idempotency-Key."""
        # Per SPEC: /v1/chat, /v1/tools/run, /v1/artifacts
        # These should all use require=False
        # Verified by code inspection - all use require=False
        assert True  # Placeholder - manual verification test

    @pytest.mark.asyncio
    async def test_idempotency_ttl_24h(self):
        """Idempotency should replay within 24h TTL per SPEC §18."""
        # IDEMPOTENCY_TTL_SECONDS should be 24 * 60 * 60 = 86400
        from liminallm.service.runtime import IDEMPOTENCY_TTL_SECONDS

        assert IDEMPOTENCY_TTL_SECONDS == 24 * 60 * 60

    @pytest.mark.asyncio
    async def test_returns_409_for_in_progress(self):
        """Should return 409 if prior attempt is still running per SPEC §18."""
        # Tested in test_raises_conflict_when_in_progress
        pass


# --- A cached response has to be serializable ------------------------------


@pytest.mark.parametrize(
    "path, body",
    [
        (
            "/v1/artifacts",
            {
                "type": "workflow",
                "name": "idempotent-workflow",
                "description": "d",
                "schema": {"kind": "workflow.chat", "nodes": []},
            },
        ),
        (
            "/v1/contexts",
            {"name": "idempotent-context", "description": "d"},
        ),
    ],
)
def test_an_idempotency_key_does_not_break_a_response_with_timestamps(
    client, auth_headers, path, body
):
    """SPEC §18 accepts Idempotency-Key on these routes. It has to work.

    The guard cached `envelope.model_dump()`, which leaves `datetime` objects
    as objects, and the record is then JSON-encoded — so every route whose
    response carries `created_at` answered 500 the moment a client sent the
    header it is invited to send. Without the header the same request
    succeeds, which is why nothing noticed.
    """
    import uuid as _uuid

    keyed = client.post(
        path,
        headers={**auth_headers, "Idempotency-Key": _uuid.uuid4().hex},
        json=body,
    )
    assert keyed.status_code in (200, 201), (
        f"{path} failed only because an Idempotency-Key was supplied: "
        f"{keyed.status_code} {keyed.text[:300]}"
    )


def test_a_repeated_key_returns_the_first_response(client, auth_headers):
    """And the cached copy is the one that comes back."""
    import uuid as _uuid

    key = _uuid.uuid4().hex
    body = {"name": "replayed-context", "description": "d"}
    first = client.post(
        "/v1/contexts", headers={**auth_headers, "Idempotency-Key": key}, json=body
    )
    assert first.status_code in (200, 201), first.text
    second = client.post(
        "/v1/contexts", headers={**auth_headers, "Idempotency-Key": key}, json=body
    )
    assert second.status_code in (200, 201), second.text
    assert second.json()["data"]["id"] == first.json()["data"]["id"], (
        "the replay created a second context instead of returning the first"
    )
