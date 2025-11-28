from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from liminallm.service.artifact_validation import ArtifactValidationError
from liminallm.service.errors import ServiceError
from liminallm.service.fs import PathTraversalError
from liminallm.storage.errors import ConstraintViolation
from liminallm.api.schemas import Envelope, ErrorBody
from liminallm.logging import get_logger

logger = get_logger(__name__)

# SPEC §18 stable error codes mapped to HTTP status codes
_STATUS_TO_CODE = {
    400: "validation_error",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    409: "conflict",
    429: "rate_limited",
    500: "server_error",
}


def _error_code_for_status(status_code: int) -> str:
    """Map HTTP status to SPEC §18 stable error code."""
    return _STATUS_TO_CODE.get(status_code, "server_error")


def _error_response(status_code: int, message: str, details: dict | list | None = None, code: str | None = None) -> JSONResponse:
    """Create a SPEC §18 compliant error response envelope."""
    error_code = code or _error_code_for_status(status_code)
    error_body = ErrorBody(code=error_code, message=message, details=details)
    envelope = Envelope(status="error", error=error_body)
    return JSONResponse(status_code=status_code, content=envelope.model_dump())


def register_exception_handlers(app: FastAPI) -> None:
    """Install consistent exception handlers for domain and storage errors per SPEC §18."""

    @app.exception_handler(ConstraintViolation)
    async def handle_constraint_violation(request, exc: ConstraintViolation):  # type: ignore[unused-argument]
        return _error_response(409, exc.message, exc.detail, code="conflict")

    @app.exception_handler(ServiceError)
    async def handle_service_error(request, exc: ServiceError):  # type: ignore[unused-argument]
        return _error_response(exc.status_code, exc.message, exc.detail)

    @app.exception_handler(ArtifactValidationError)
    async def handle_validation_error(request, exc: ArtifactValidationError):  # type: ignore[unused-argument]
        return _error_response(400, str(exc), exc.errors, code="validation_error")

    @app.exception_handler(PathTraversalError)
    async def handle_path_traversal_error(request, exc: PathTraversalError):  # type: ignore[unused-argument]
        return _error_response(400, str(exc), code="validation_error")

    @app.exception_handler(HTTPException)
    async def handle_http_exception(request: Request, exc: HTTPException):  # type: ignore[unused-argument]
        detail = exc.detail if isinstance(exc.detail, dict) else {"detail": exc.detail}
        message = detail.get("detail", "http error") if isinstance(detail, dict) else str(exc.detail)
        details = detail if isinstance(detail, dict) else None
        return _error_response(exc.status_code, message, details)

    @app.exception_handler(Exception)
    async def handle_uncaught(request: Request, exc: Exception):  # type: ignore[unused-argument]
        logger.exception("unhandled_exception", exc_info=exc)
        return _error_response(500, "internal server error", code="server_error")
