"""The SPEC §18 error envelope, as an exception.

Separate from ``error_handling`` (which renders it) so modules that raise
errors need not import the app wiring.
"""

from __future__ import annotations

from typing import Optional

from fastapi import HTTPException


def http_error(
    code: str, message: str, status_code: int, details: Optional[dict | str] = None
) -> HTTPException:
    """An HTTPException whose detail is already a SPEC §18 error envelope."""
    payload: dict[str, object] = {
        "status": "error",
        "error": {"code": code, "message": message},
    }
    if details is not None:
        payload["error"]["details"] = details  # type: ignore[index]
    return HTTPException(status_code=status_code, detail=payload)
