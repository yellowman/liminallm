from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from liminallm.service.artifact_validation import ArtifactValidationError
from liminallm.service.errors import ServiceError
from liminallm.service.fs import PathTraversalError
from liminallm.storage.errors import ConstraintViolation
from liminallm.api.schemas import Envelope
from liminallm.logging import get_logger

logger = get_logger(__name__)


def _error_response(status_code: int, message: str, detail: dict | None = None) -> JSONResponse:
    envelope = Envelope(status="error", error={"message": message, "detail": detail or {}})
    return JSONResponse(status_code=status_code, content=envelope.model_dump())


def register_exception_handlers(app: FastAPI) -> None:
    """Install consistent exception handlers for domain and storage errors."""

    @app.exception_handler(ConstraintViolation)
    async def handle_constraint_violation(request, exc: ConstraintViolation):  # type: ignore[unused-argument]
        return _error_response(409, exc.message, exc.detail)

    @app.exception_handler(ServiceError)
    async def handle_service_error(request, exc: ServiceError):  # type: ignore[unused-argument]
        return _error_response(exc.status_code, exc.message, exc.detail)

    @app.exception_handler(ArtifactValidationError)
    async def handle_validation_error(request, exc: ArtifactValidationError):  # type: ignore[unused-argument]
        return _error_response(400, str(exc), exc.errors)

    @app.exception_handler(PathTraversalError)
    async def handle_path_traversal_error(request, exc: PathTraversalError):  # type: ignore[unused-argument]
        return _error_response(400, str(exc))

    @app.exception_handler(HTTPException)
    async def handle_http_exception(request: Request, exc: HTTPException):  # type: ignore[unused-argument]
        detail = exc.detail if isinstance(exc.detail, dict) else {"detail": exc.detail}
        return _error_response(exc.status_code, detail.get("detail", "http error"), detail if isinstance(detail, dict) else {})

    @app.exception_handler(Exception)
    async def handle_uncaught(request: Request, exc: Exception):  # type: ignore[unused-argument]
        logger.exception("unhandled_exception", exc_info=exc)
        return _error_response(500, "internal server error")

