from __future__ import annotations

from typing import Optional


class ServiceError(Exception):
    """Base class for service-layer exceptions mapped to HTTP responses."""

    status_code: int = 400

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        detail: Optional[dict] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code or self.status_code
        self.detail = detail or {}


class ConflictError(ServiceError):
    status_code = 409


class NotFoundError(ServiceError):
    status_code = 404


class BadRequestError(ServiceError):
    status_code = 400


class AuthenticationError(ServiceError):
    status_code = 401


class SessionExpiredError(ServiceError):
    status_code = 401


__all__ = [
    "ServiceError",
    "ConflictError",
    "NotFoundError",
    "BadRequestError",
    "AuthenticationError",
    "SessionExpiredError",
]
