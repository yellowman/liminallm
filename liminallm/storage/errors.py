from __future__ import annotations

from typing import Any, Dict, Optional


class ConstraintViolation(Exception):
    """Raised when a storage-layer uniqueness or FK constraint is violated."""

    def __init__(self, message: str, detail: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.detail = detail or {}


__all__ = ["ConstraintViolation"]
