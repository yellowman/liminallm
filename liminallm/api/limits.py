"""Named rate-limit policies.

An endpoint names a policy instead of restating the settings lookup, the window
and the key format - six lines that appeared fifty-seven times, and which an
endpoint therefore got written without. Naming them also makes "unlimited" a
visible choice; seventeen routes were unlimited by omission.
"""

from __future__ import annotations

from typing import Optional

from fastapi import Request, Response

from liminallm.api.errors import http_error
from liminallm.logging import get_logger
from liminallm.service.runtime import check_rate_limit

logger = get_logger(__name__)

#: Policy name -> (setting carrying the limit, window in seconds or the setting
#: carrying it).
RATE_POLICIES: dict[str, tuple[str, str | int]] = {
    "read": ("read_rate_limit_per_minute", 60),
    "write": ("write_rate_limit_per_minute", 60),
    "chat": ("chat_rate_limit_per_minute", "chat_rate_limit_window_seconds"),
    "admin:read": ("admin_rate_limit_per_minute", "admin_rate_limit_window_seconds"),
    "admin:write": ("admin_rate_limit_per_minute", "admin_rate_limit_window_seconds"),
    "mfa": ("mfa_rate_limit_per_minute", 60),
    "configops": ("configops_rate_limit_per_hour", 3600),
    "signup": ("signup_rate_limit_per_minute", 60),
    "login": ("login_rate_limit_per_minute", 60),
    "reset": ("reset_rate_limit_per_minute", 60),
    "refresh": ("refresh_rate_limit_per_minute", "refresh_rate_limit_window_seconds"),
    "files:upload": ("files_upload_rate_limit_per_minute", 60),
    "websocket:connect": ("websocket_connect_rate_limit_per_minute", 60),
}


class RateLimitInfo:
    """Rate limit state, for the response headers."""

    __slots__ = ("limit", "remaining", "reset_seconds")

    def __init__(self, limit: int, remaining: int, reset_seconds: int):
        self.limit = limit
        self.remaining = remaining
        self.reset_seconds = reset_seconds

    def apply_headers(self, response: Response) -> None:
        """Per IETF draft-polli-ratelimit-headers."""
        response.headers["X-RateLimit-Limit"] = str(self.limit)
        response.headers["X-RateLimit-Remaining"] = str(max(0, self.remaining))
        response.headers["X-RateLimit-Reset"] = str(self.reset_seconds)


def client_ip(request: Optional[Request]) -> str:
    """Best-effort client IP, so one caller cannot exhaust a shared limit
    (password reset, email verification, OAuth) for everyone else."""
    if request is None or request.client is None:
        return "unknown"
    return request.client.host


def plan_rate_multiplier(runtime, plan_tier: str, *, verified: bool) -> float:
    """Rate-limit multiplier for a plan tier (SPEC §18).

    An account whose address is not yet verified is scaled down again on top
    of its tier (SPEC §12.1). One modifier over the existing multiplier, not a
    second limiter: the tier still decides the shape, and verifying restores
    the ordinary rate on the next request.

    `verified` has no default deliberately. A caller that forgets it fails
    loudly rather than quietly granting the full rate.
    """
    settings = runtime.settings
    plan = {
        "free": settings.rate_limit_multiplier_free,
        "paid": settings.rate_limit_multiplier_paid,
        "enterprise": settings.rate_limit_multiplier_enterprise,
    }.get(plan_tier, 1.0)
    if verified:
        return plan
    return plan * settings.unverified_rate_limit_multiplier


def plan_upload_limit(runtime, plan_tier: str) -> int:
    """Per-plan upload size cap in bytes (SPEC §18: free 25MB, paid 200MB)."""
    limits = {
        "free": 25 * 1024 * 1024,
        "paid": 200 * 1024 * 1024,
        "enterprise": 200 * 1024 * 1024,
    }
    return limits.get(plan_tier, limits["free"])


async def rate_limit(
    runtime,
    policy: str,
    subject: str,
    *,
    response: Optional[Response] = None,
    cost: int = 1,
) -> RateLimitInfo:
    """Apply a named policy to a subject - usually principal.user_id, but an
    email for signup and an address for anonymous flows."""
    limit_attr, window = RATE_POLICIES[policy]
    window_seconds = (
        window if isinstance(window, int) else getattr(runtime.settings, window)
    )
    return await enforce(
        runtime,
        f"{policy}:{subject}",
        getattr(runtime.settings, limit_attr),
        window_seconds,
        response=response,
        cost=cost,
    )


async def enforce(
    runtime,
    key: str,
    limit: int,
    window_seconds: int,
    *,
    response: Optional[Response] = None,
    cost: int = 1,
) -> RateLimitInfo:
    """Consume ``cost`` against ``key``, raising 429 when the bucket is empty."""
    # limit <= 0 means "disabled/unlimited" per the admin-settings contract, and
    # check_rate_limit() honors that. Do NOT clamp to 1 here - that would turn
    # an operator's "unlimited" into the strictest possible limit.
    if window_seconds <= 0:
        logger.warning(
            "invalid_rate_limit_window", key=key, window_seconds=window_seconds
        )
        window_seconds = 60

    allowed, remaining, reset_after = await check_rate_limit(
        runtime, key, limit, window_seconds, return_remaining=True, cost=cost
    )
    # Use the calculated reset when available so a failure doesn't leak the
    # window (Issue 77.8).
    reset_seconds = reset_after if reset_after is not None else window_seconds
    info = RateLimitInfo(limit, remaining, reset_seconds)

    if response is not None:
        info.apply_headers(response)
    if not allowed:
        raise http_error("rate_limited", "rate limit exceeded", status_code=429)
    return info


async def enforce_per_plan(
    runtime,
    key: str,
    base_limit: int,
    window_seconds: int,
    plan_tier: str,
    *,
    verified: bool,
    response: Optional[Response] = None,
) -> RateLimitInfo:
    """Enforce a limit scaled by the user's plan tier (SPEC §18).

    `verified` says whether the account's email address has been confirmed;
    an unverified one is scaled down again (SPEC §12.1).
    """
    multiplier = plan_rate_multiplier(runtime, plan_tier, verified=verified)
    adjusted = int(base_limit * multiplier)
    # Scaling must not manufacture "unlimited". `enforce` reads a limit of 0
    # as disabled, so a small base times a multiplier below 1 - three requests
    # a minute scaled by an unverified quarter - would round down to the one
    # value that means no limit at all. 0 stays available as the operator's
    # own choice; it is not something arithmetic arrives at.
    if base_limit > 0:
        adjusted = max(1, adjusted)
    return await enforce(runtime, key, adjusted, window_seconds, response=response)
