from __future__ import annotations

from typing import Optional


class ServiceError(Exception):
    """Base class for service-layer exceptions mapped to HTTP responses.

    Per SPEC §18, all errors should have stable error codes. Each exception class
    defines both an HTTP status_code and an error_code that maps to the SPEC codes:
    - unauthorized (401)
    - forbidden (403)
    - not_found (404)
    - rate_limited (429)
    - validation_error (400)
    - conflict (409)
    - server_error (500)
    """

    status_code: int = 400
    error_code: str = "validation_error"

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        detail: Optional[dict] = None,
        error_code: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        if error_code is not None:
            self.error_code = error_code
        self.detail = detail or {}


class ValidationError(ServiceError):
    """Request validation failed (400)."""
    status_code = 400
    error_code = "validation_error"


class BadRequestError(ValidationError):
    """Alias for ValidationError - request is malformed or invalid."""
    pass


class AuthenticationError(ServiceError):
    """Authentication failed or missing (401)."""
    status_code = 401
    error_code = "unauthorized"


class SessionExpiredError(AuthenticationError):
    """Session has expired (401)."""
    pass


class ForbiddenError(ServiceError):
    """Access denied - insufficient permissions (403)."""
    status_code = 403
    error_code = "forbidden"


class NotFoundError(ServiceError):
    """Requested resource not found (404)."""
    status_code = 404
    error_code = "not_found"


class ConflictError(ServiceError):
    """Resource conflict, e.g., duplicate creation (409)."""
    status_code = 409
    error_code = "conflict"


class RateLimitedError(ServiceError):
    """Rate limit exceeded (429)."""
    status_code = 429
    error_code = "rate_limited"


class ServerError(ServiceError):
    """Internal server error (500)."""
    status_code = 500
    error_code = "server_error"


__all__ = [
    "ServiceError",
    "ValidationError",
    "BadRequestError",
    "AuthenticationError",
    "SessionExpiredError",
    "ForbiddenError",
    "NotFoundError",
    "ConflictError",
    "RateLimitedError",
    "ServerError",
]
