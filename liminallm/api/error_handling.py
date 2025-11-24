from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from liminallm.service.artifact_validation import ArtifactValidationError
from liminallm.service.errors import ServiceError
from liminallm.service.fs import PathTraversalError
from liminallm.storage.errors import ConstraintViolation


def register_exception_handlers(app: FastAPI) -> None:
    """Install consistent exception handlers for domain and storage errors."""

    @app.exception_handler(ConstraintViolation)
    async def handle_constraint_violation(request, exc: ConstraintViolation):  # type: ignore[unused-argument]
        return JSONResponse(status_code=409, content={"detail": exc.message, "errors": exc.detail})

    @app.exception_handler(ServiceError)
    async def handle_service_error(request, exc: ServiceError):  # type: ignore[unused-argument]
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.message, "errors": exc.detail})

    @app.exception_handler(ArtifactValidationError)
    async def handle_validation_error(request, exc: ArtifactValidationError):  # type: ignore[unused-argument]
        return JSONResponse(status_code=400, content={"detail": str(exc), "errors": exc.errors})

    @app.exception_handler(PathTraversalError)
    async def handle_path_traversal_error(request, exc: PathTraversalError):  # type: ignore[unused-argument]
        return JSONResponse(status_code=400, content={"detail": str(exc)})

