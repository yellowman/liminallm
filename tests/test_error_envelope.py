"""Tests for SPEC §18 error envelope format and error handling.

These tests verify that error responses conform to the stable API envelope format:
{
    "status": "error",
    "error": {
        "code": "<stable_code>",
        "message": "<human_readable>",
        "details": <object|array|null>
    },
    "request_id": "<uuid>"
}
"""

import pytest
from pydantic import ValidationError

from liminallm.api.error_handling import (
    _STATUS_TO_CODE,
    _error_code_for_status,
    _error_response,
)
from liminallm.api.schemas import Envelope, ErrorBody


class TestErrorBody:
    """Tests for the ErrorBody Pydantic model."""

    def test_error_body_required_fields(self):
        """ErrorBody requires code and message fields."""
        error = ErrorBody(code="unauthorized", message="Invalid credentials")
        assert error.code == "unauthorized"
        assert error.message == "Invalid credentials"
        assert error.details is None

    def test_error_body_with_details_dict(self):
        """ErrorBody accepts dict details."""
        error = ErrorBody(
            code="validation_error",
            message="Invalid input",
            details={"field": "email", "reason": "invalid format"},
        )
        assert error.details == {"field": "email", "reason": "invalid format"}

    def test_error_body_with_details_list(self):
        """ErrorBody accepts list details."""
        error = ErrorBody(
            code="validation_error",
            message="Multiple errors",
            details=[{"field": "email"}, {"field": "password"}],
        )
        assert len(error.details) == 2

    def test_error_body_with_details_null(self):
        """ErrorBody accepts explicit null details."""
        error = ErrorBody(code="not_found", message="Resource not found", details=None)
        assert error.details is None

    def test_error_body_missing_code_raises(self):
        """ErrorBody without code raises ValidationError."""
        with pytest.raises(ValidationError):
            ErrorBody(message="Error occurred")

    def test_error_body_missing_message_raises(self):
        """ErrorBody without message raises ValidationError."""
        with pytest.raises(ValidationError):
            ErrorBody(code="server_error")

    def test_error_body_serialization(self):
        """ErrorBody serializes to correct dict structure."""
        error = ErrorBody(
            code="forbidden",
            message="Access denied",
            details={"resource": "admin"},
        )
        dumped = error.model_dump()

        assert dumped["code"] == "forbidden"
        assert dumped["message"] == "Access denied"
        assert dumped["details"] == {"resource": "admin"}


class TestEnvelope:
    """Tests for the Envelope model with error support."""

    def test_envelope_error_status(self):
        """Envelope accepts 'error' status with ErrorBody."""
        error_body = ErrorBody(code="unauthorized", message="Invalid token")
        envelope = Envelope(status="error", error=error_body)

        assert envelope.status == "error"
        assert envelope.error is not None
        assert envelope.error.code == "unauthorized"
        assert envelope.data is None

    def test_envelope_ok_status(self):
        """Envelope accepts 'ok' status with data."""
        envelope = Envelope(status="ok", data={"user_id": "123"})

        assert envelope.status == "ok"
        assert envelope.data == {"user_id": "123"}
        assert envelope.error is None

    def test_envelope_request_id_auto_generated(self):
        """Envelope auto-generates request_id if not provided."""
        envelope = Envelope(status="ok")

        assert envelope.request_id is not None
        assert len(envelope.request_id) == 36  # UUID format

    def test_envelope_request_id_custom(self):
        """Envelope accepts custom request_id."""
        envelope = Envelope(status="ok", request_id="custom-id-123")

        assert envelope.request_id == "custom-id-123"

    def test_envelope_invalid_status_raises(self):
        """Envelope rejects invalid status values."""
        with pytest.raises(ValidationError):
            Envelope(status="pending")

        with pytest.raises(ValidationError):
            Envelope(status="success")

    def test_envelope_error_serialization(self):
        """Full error envelope serializes correctly for API response."""
        error_body = ErrorBody(
            code="rate_limited",
            message="Too many requests",
            details={"retry_after": 60},
        )
        envelope = Envelope(
            status="error",
            error=error_body,
            request_id="test-req-123",
        )
        dumped = envelope.model_dump()

        assert dumped["status"] == "error"
        assert dumped["error"]["code"] == "rate_limited"
        assert dumped["error"]["message"] == "Too many requests"
        assert dumped["error"]["details"]["retry_after"] == 60
        assert dumped["request_id"] == "test-req-123"
        assert dumped["data"] is None


class TestErrorCodeMapping:
    """Tests for HTTP status to SPEC §18 error code mapping."""

    def test_status_400_maps_to_validation_error(self):
        """400 Bad Request maps to 'validation_error'."""
        assert _error_code_for_status(400) == "validation_error"

    def test_status_401_maps_to_unauthorized(self):
        """401 Unauthorized maps to 'unauthorized'."""
        assert _error_code_for_status(401) == "unauthorized"

    def test_status_403_maps_to_forbidden(self):
        """403 Forbidden maps to 'forbidden'."""
        assert _error_code_for_status(403) == "forbidden"

    def test_status_404_maps_to_not_found(self):
        """404 Not Found maps to 'not_found'."""
        assert _error_code_for_status(404) == "not_found"

    def test_status_409_maps_to_conflict(self):
        """409 Conflict maps to 'conflict'."""
        assert _error_code_for_status(409) == "conflict"

    def test_status_429_maps_to_rate_limited(self):
        """429 Too Many Requests maps to 'rate_limited'."""
        assert _error_code_for_status(429) == "rate_limited"

    def test_status_500_maps_to_server_error(self):
        """500 Internal Server Error maps to 'server_error'."""
        assert _error_code_for_status(500) == "server_error"

    def test_unknown_status_defaults_to_server_error(self):
        """Unknown status codes default to 'server_error'."""
        assert _error_code_for_status(418) == "server_error"  # I'm a teapot
        assert _error_code_for_status(503) == "server_error"  # Service Unavailable
        assert _error_code_for_status(999) == "server_error"  # Invalid

    def test_all_spec_codes_covered(self):
        """All SPEC §18 error codes are in the mapping."""
        expected_codes = {
            "unauthorized",
            "forbidden",
            "not_found",
            "rate_limited",
            "validation_error",
            "conflict",
            "server_error",
        }
        actual_codes = set(_STATUS_TO_CODE.values())
        assert actual_codes == expected_codes


class TestErrorResponseFactory:
    """Tests for the _error_response helper function."""

    def test_error_response_basic(self):
        """_error_response creates JSONResponse with correct envelope."""
        response = _error_response(401, "Invalid credentials")

        assert response.status_code == 401
        body = response.body.decode()
        assert '"status":"error"' in body or '"status": "error"' in body
        assert "unauthorized" in body
        assert "Invalid credentials" in body

    def test_error_response_with_details(self):
        """_error_response includes details in envelope."""
        response = _error_response(
            400,
            "Validation failed",
            details={"field": "email", "error": "required"},
        )

        assert response.status_code == 400
        body = response.body.decode()
        assert "email" in body
        assert "required" in body

    def test_error_response_custom_code(self):
        """_error_response accepts custom error code override (must be valid SPEC §18 code)."""
        # Custom code must be a valid SPEC §18 error code
        response = _error_response(400, "Custom error", code="conflict")

        body = response.body.decode()
        assert "conflict" in body

    def test_error_response_includes_request_id(self):
        """_error_response includes request_id in envelope."""
        response = _error_response(500, "Server error")

        body = response.body.decode()
        assert "request_id" in body

    def test_error_response_null_details(self):
        """_error_response handles None details correctly."""
        response = _error_response(404, "Not found", details=None)

        assert response.status_code == 404
        body = response.body.decode()
        # Should still be valid JSON
        import json

        data = json.loads(body)
        assert data["error"]["details"] is None

    def test_error_response_list_details(self):
        """_error_response handles list details."""
        response = _error_response(
            400,
            "Multiple errors",
            details=[{"field": "a"}, {"field": "b"}],
        )

        import json

        data = json.loads(response.body.decode())
        assert isinstance(data["error"]["details"], list)
        assert len(data["error"]["details"]) == 2
