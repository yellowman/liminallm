from __future__ import annotations

import asyncio
import base64
import json
from datetime import datetime, timezone
from pathlib import Path as FilePath
from typing import Any, Dict, Optional
from uuid import uuid4

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    Path,
    Query,
    Response,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)

from liminallm.api.schemas import (
    AdminCreateUserRequest,
    AdminCreateUserResponse,
    AdminInspectionResponse,
    AdminSettingsResponse,
    AdminSettingsUpdateRequest,
    ArtifactListResponse,
    ArtifactRequest,
    ArtifactResponse,
    ArtifactVersionListResponse,
    ArtifactVersionResponse,
    AuthResponse,
    AutoPatchRequest,
    ChatCancelRequest,
    ChatCancelResponse,
    ChatRequest,
    ChatResponse,
    ConfigPatchAuditResponse,
    ConfigPatchDecisionRequest,
    ConfigPatchListResponse,
    ConfigPatchRequest,
    ContextSourceListResponse,
    ContextSourceRequest,
    ContextSourceResponse,
    ConversationListResponse,
    ConversationMessagesResponse,
    ConversationSummary,
    CreateConversationRequest,
    CreateConversationResponse,
    EmailVerificationRequest,
    Envelope,
    FileUploadResponse,
    KnowledgeChunkListResponse,
    KnowledgeChunkResponse,
    KnowledgeContextListResponse,
    KnowledgeContextRequest,
    KnowledgeContextResponse,
    LoginRequest,
    MFADisableRequest,
    MFARequest,
    MFAStatusResponse,
    MFAVerifyRequest,
    OAuthStartRequest,
    OAuthStartResponse,
    PasswordChangeRequest,
    PasswordResetConfirm,
    PasswordResetRequest,
    PreferenceEventRequest,
    PreferenceEventResponse,
    PreferenceInsightsResponse,
    SignupRequest,
    TokenRefreshRequest,
    ToolInvokeRequest,
    ToolInvokeResponse,
    ToolSpecListResponse,
    UpdateUserRoleRequest,
    UserListResponse,
    UserResponse,
    UserSettingsRequest,
    UserSettingsResponse,
    VoiceSynthesisRequest,
    VoiceSynthesisResponse,
    VoiceTranscriptionResponse,
    WorkflowListResponse,
)
from liminallm.config import get_settings
from liminallm.content_struct import normalize_content_struct
from liminallm.logging import get_logger
from liminallm.service.auth import AuthContext
from liminallm.service.errors import BadRequestError, NotFoundError
from liminallm.service.fs import safe_join
from liminallm.service.runtime import (
    IDEMPOTENCY_TTL_SECONDS,
    _get_cached_idempotency_record,
    _set_cached_idempotency_record,
    check_rate_limit,
    get_runtime,
)
from liminallm.storage.models import Conversation, KnowledgeContext, Session

logger = get_logger(__name__)

router = APIRouter(prefix="/v1")

# Registry for active streaming requests - maps request_id to cancel_event
# Used by POST /chat/cancel to cancel in-progress streaming requests per SPEC §18
_active_requests: Dict[str, asyncio.Event] = {}
_active_requests_lock = asyncio.Lock()


async def _register_cancel_event(request_id: str, cancel_event: asyncio.Event) -> None:
    """Register a cancel event for an active streaming request."""
    async with _active_requests_lock:
        _active_requests[request_id] = cancel_event


async def _unregister_cancel_event(request_id: str) -> None:
    """Unregister a cancel event when request completes."""
    async with _active_requests_lock:
        _active_requests.pop(request_id, None)


async def _cancel_request(request_id: str) -> bool:
    """Cancel an active request by request_id. Returns True if cancelled."""
    async with _active_requests_lock:
        cancel_event = _active_requests.get(request_id)
        if cancel_event and not cancel_event.is_set():
            cancel_event.set()
            return True
        return False


def _http_error(
    code: str, message: str, status_code: int, details: Optional[dict | str] = None
) -> HTTPException:
    payload: dict[str, object] = {
        "status": "error",
        "error": {"code": code, "message": message},
    }
    if details is not None:
        payload["error"]["details"] = details  # type: ignore[index]
    return HTTPException(status_code=status_code, detail=payload)


class IdempotencyGuard:
    def __init__(
        self,
        route: str,
        user_id: str,
        idempotency_key: Optional[str],
        *,
        require: bool = False,
    ):
        self.route = route
        self.user_id = user_id
        self.idempotency_key = idempotency_key
        self.require = require
        self.request_id: Optional[str] = None
        self.cached: Optional[Envelope] = None
        self._stored = False

    async def __aenter__(self) -> "IdempotencyGuard":
        self.request_id = self.request_id or str(uuid4())
        self.request_id, self.cached = await _resolve_idempotency(
            self.route,
            self.user_id,
            self.idempotency_key,
            require=self.require,
            request_id=self.request_id,
        )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        if exc and not self._stored and self.request_id:
            await _store_idempotency_result(
                self.route,
                self.user_id,
                self.idempotency_key,
                Envelope(
                    status="error",
                    error={"code": "server_error", "message": str(exc)},
                    request_id=self.request_id,
                ),
                status="failed",
            )
            self._stored = True
        return False

    async def store_result(
        self, envelope: Envelope, *, status: str = "completed"
    ) -> None:
        """Store result in idempotency cache.

        Args:
            envelope: Response envelope to store
            status: Status to record. Only "completed" marks the operation as
                    fully done - intermediate statuses like "processing" still
                    allow error recording via __aexit__.
        """
        await _store_idempotency_result(
            self.route, self.user_id, self.idempotency_key, envelope, status=status
        )
        # Only mark as stored for final states to allow error recording for intermediate states
        if status in {"completed", "failed"}:
            self._stored = True

    async def store_error(self, message: str) -> None:
        if not self.request_id:
            return
        await self.store_result(
            Envelope(
                status="error",
                error={"code": "server_error", "message": message},
                request_id=self.request_id,
            ),
            status="failed",
        )


class RateLimitInfo:
    """Rate limit state for adding response headers."""

    __slots__ = ("limit", "remaining", "reset_seconds")

    def __init__(self, limit: int, remaining: int, reset_seconds: int):
        self.limit = limit
        self.remaining = remaining
        self.reset_seconds = reset_seconds

    def apply_headers(self, response: Response) -> None:
        """Apply rate limit headers to response per IETF draft-polli-ratelimit-headers."""
        response.headers["X-RateLimit-Limit"] = str(self.limit)
        response.headers["X-RateLimit-Remaining"] = str(max(0, self.remaining))
        response.headers["X-RateLimit-Reset"] = str(self.reset_seconds)


def _get_plan_rate_multiplier(runtime, plan_tier: str) -> float:
    """Get rate limit multiplier for a user's plan tier (SPEC §18).

    Args:
        runtime: Application runtime context
        plan_tier: User's plan tier (free, paid, enterprise)

    Returns:
        Multiplier to apply to base rate limits
    """
    multipliers = {
        "free": runtime.settings.rate_limit_multiplier_free,
        "paid": runtime.settings.rate_limit_multiplier_paid,
        "enterprise": runtime.settings.rate_limit_multiplier_enterprise,
    }
    return multipliers.get(plan_tier, 1.0)


async def _enforce_rate_limit(
    runtime, key: str, limit: int, window_seconds: int, *, response: Optional[Response] = None
) -> RateLimitInfo:
    """Enforce rate limit and optionally apply headers to response.

    Args:
        runtime: Application runtime context
        key: Rate limit key (e.g., "chat:{user_id}")
        limit: Maximum requests allowed in window
        window_seconds: Rate limit window in seconds
        response: Optional response to add rate limit headers to

    Returns:
        RateLimitInfo with current rate limit state

    Raises:
        HTTPException with 429 if rate limit exceeded
    """
    allowed, remaining = await check_rate_limit(runtime, key, limit, window_seconds, return_remaining=True)
    info = RateLimitInfo(limit, remaining, window_seconds)

    if response is not None:
        info.apply_headers(response)

    if not allowed:
        raise _http_error("rate_limited", "rate limit exceeded", status_code=429)

    return info


async def _enforce_rate_limit_per_plan(
    runtime,
    key: str,
    base_limit: int,
    window_seconds: int,
    plan_tier: str,
    *,
    response: Optional[Response] = None,
) -> RateLimitInfo:
    """Enforce rate limit adjusted for user's plan tier (SPEC §18).

    Args:
        runtime: Application runtime context
        key: Rate limit key (e.g., "chat:{user_id}")
        base_limit: Base rate limit (will be multiplied by plan tier)
        window_seconds: Rate limit window in seconds
        plan_tier: User's plan tier for multiplier lookup
        response: Optional response to add rate limit headers to

    Returns:
        RateLimitInfo with current rate limit state

    Raises:
        HTTPException with 429 if rate limit exceeded
    """
    multiplier = _get_plan_rate_multiplier(runtime, plan_tier)
    adjusted_limit = int(base_limit * multiplier)
    return await _enforce_rate_limit(runtime, key, adjusted_limit, window_seconds, response=response)


async def _acquire_workflow_slot(runtime, user_id: str) -> bool:
    """Acquire a workflow concurrency slot (SPEC §18: max 3 per user).

    Args:
        runtime: Application runtime context
        user_id: User ID

    Returns:
        True if slot acquired, raises 409 if at capacity

    Raises:
        HTTPException with 409 if concurrency cap exceeded
    """
    if not runtime.cache:
        # Without Redis, we can't track concurrency - allow the request
        return True

    acquired, current = await runtime.cache.acquire_concurrency_slot(
        "workflow",
        user_id,
        runtime.settings.max_concurrent_workflows,
    )
    if not acquired:
        raise _http_error(
            "busy",
            f"concurrent workflow limit ({runtime.settings.max_concurrent_workflows}) exceeded",
            status_code=409,
        )
    return True


async def _release_workflow_slot(runtime, user_id: str) -> None:
    """Release a workflow concurrency slot."""
    if runtime.cache:
        await runtime.cache.release_concurrency_slot("workflow", user_id)


async def _acquire_inference_slot(runtime, user_id: str) -> bool:
    """Acquire an inference concurrency slot (SPEC §18: max 2 per user).

    Args:
        runtime: Application runtime context
        user_id: User ID

    Returns:
        True if slot acquired, raises 409 if at capacity

    Raises:
        HTTPException with 409 if concurrency cap exceeded
    """
    if not runtime.cache:
        return True

    acquired, current = await runtime.cache.acquire_concurrency_slot(
        "inference",
        user_id,
        runtime.settings.max_concurrent_inference,
    )
    if not acquired:
        raise _http_error(
            "busy",
            f"concurrent inference limit ({runtime.settings.max_concurrent_inference}) exceeded",
            status_code=409,
        )
    return True


async def _release_inference_slot(runtime, user_id: str) -> None:
    """Release an inference concurrency slot."""
    if runtime.cache:
        await runtime.cache.release_concurrency_slot("inference", user_id)


async def _resolve_idempotency(
    route: str,
    user_id: str,
    idempotency_key: Optional[str],
    *,
    require: bool = False,
    request_id: Optional[str] = None,
) -> tuple[str, Optional[Envelope]]:
    request_id = request_id or str(uuid4())
    runtime = get_runtime()
    if not idempotency_key:
        if require:
            raise _http_error(
                "validation_error", "Idempotency-Key header required", status_code=400
            )
        return request_id, None
    record = await _get_cached_idempotency_record(
        runtime, route, user_id, idempotency_key
    )
    if record:
        status = record.get("status")
        if status == "in_progress":
            raise _http_error("conflict", "request in progress", status_code=409)
        if status in {"completed", "failed"} and record.get("response"):
            response_payload = record.get("response", {})
            if "request_id" not in response_payload:
                response_payload["request_id"] = record.get("request_id", request_id)
            return record.get("request_id", request_id), Envelope(**response_payload)
    await _set_cached_idempotency_record(
        runtime,
        route,
        user_id,
        idempotency_key,
        {
            "status": "in_progress",
            "request_id": request_id,
            "started_at": datetime.utcnow().isoformat(),
        },
        ttl_seconds=IDEMPOTENCY_TTL_SECONDS,
    )
    return request_id, None


async def _store_idempotency_result(
    route: str,
    user_id: str,
    idempotency_key: Optional[str],
    envelope: Envelope,
    status: str = "completed",
) -> None:
    if not idempotency_key:
        return
    runtime = get_runtime()
    payload = envelope.model_dump()
    await _set_cached_idempotency_record(
        runtime,
        route,
        user_id,
        idempotency_key,
        {"status": status, "request_id": envelope.request_id, "response": payload},
        ttl_seconds=IDEMPOTENCY_TTL_SECONDS,
    )


async def get_user(
    authorization: Optional[str] = Header(None),
    session_id: Optional[str] = Header(None, convert_underscores=False),
    x_tenant_id: Optional[str] = Header(
        None, convert_underscores=False, alias="X-Tenant-ID"
    ),
) -> AuthContext:
    runtime = get_runtime()
    ctx = await runtime.auth.authenticate(
        authorization,
        session_id,
        tenant_hint=x_tenant_id,
    )
    if not ctx:
        raise _http_error("unauthorized", "invalid session", status_code=401)
    return ctx


async def get_admin_user(
    authorization: Optional[str] = Header(None),
    session_id: Optional[str] = Header(None, convert_underscores=False),
    x_tenant_id: Optional[str] = Header(
        None, convert_underscores=False, alias="X-Tenant-ID"
    ),
) -> AuthContext:
    runtime = get_runtime()
    ctx = await runtime.auth.authenticate(
        authorization,
        session_id,
        tenant_hint=x_tenant_id,
        required_role="admin",
    )
    if not ctx:
        raise _http_error("forbidden", "admin access required", status_code=403)
    return ctx


def _get_owned_conversation(
    runtime, conversation_id: str, principal: AuthContext
) -> Conversation:
    conversation = runtime.store.get_conversation(
        conversation_id, user_id=principal.user_id
    )
    if not conversation:
        raise _http_error("not_found", "conversation not found", status_code=404)
    return conversation


def _stringify_adapters(adapters: Any) -> list[str]:
    adapter_list: list[str] = []
    if not isinstance(adapters, list):
        return adapter_list
    for adapter in adapters:
        if isinstance(adapter, dict):
            name = adapter.get("name") or adapter.get("id")
            adapter_list.append(str(name or adapter))
        else:
            adapter_list.append(str(adapter))
    return adapter_list


def _get_owned_context(
    runtime, context_id: str, principal: AuthContext
) -> KnowledgeContext:
    ctx = runtime.store.get_context(context_id)
    if not ctx:
        raise _http_error("not_found", "context not found", status_code=404)
    if ctx.owner_user_id != principal.user_id:
        raise _http_error(
            "forbidden", "context is owned by another user", status_code=403
        )
    return ctx


def _get_owned_artifact(runtime, artifact_id: str, principal: AuthContext):
    artifact = runtime.store.get_artifact(artifact_id)
    if not artifact:
        raise _http_error("not_found", "artifact not found", status_code=404)
    if artifact.owner_user_id and artifact.owner_user_id != principal.user_id:
        if principal.role != "admin":
            raise _http_error(
                "forbidden", "artifact is owned by another user", status_code=403
            )
    if not artifact.owner_user_id and principal.role != "admin":
        raise _http_error(
            "forbidden", "artifact access requires admin privileges", status_code=403
        )
    return artifact


def _user_to_response(user) -> UserResponse:
    return UserResponse(
        id=user.id,
        email=user.email,
        handle=user.handle,
        role=user.role,
        tenant_id=user.tenant_id,
        created_at=user.created_at,
        is_active=getattr(user, "is_active", True),
        plan_tier=getattr(user, "plan_tier", "free"),
        meta=getattr(user, "meta", None),
    )


def _get_artifact_kind(schema: Any) -> Optional[str]:
    """Extract 'kind' from artifact schema, handling None/non-dict safely."""
    if isinstance(schema, dict):
        return schema.get("kind")
    return None


def _get_pagination_settings(runtime) -> dict:
    """Get pagination settings from runtime config with fallback to env settings.

    Returns dict with: default_page_size, max_page_size, default_conversations_limit
    """
    settings = get_settings()
    runtime_config = (
        runtime.store.get_runtime_config()
        if hasattr(runtime.store, "get_runtime_config")
        else {}
    )
    return {
        "default_page_size": runtime_config.get("default_page_size", settings.default_page_size),
        "max_page_size": runtime_config.get("max_page_size", settings.max_page_size),
        "default_conversations_limit": runtime_config.get(
            "default_conversations_limit", settings.default_conversations_limit
        ),
    }


def _generate_conversation_title(message: str, max_length: int = 50) -> str:
    """Generate a conversation title from the first message.

    Creates a human-readable title by truncating the message content.
    Preserves word boundaries where possible.
    """
    if not message:
        return "New conversation"

    # Clean up whitespace
    cleaned = " ".join(message.split())

    # Handle whitespace-only input after cleanup
    if not cleaned:
        return "New conversation"

    if len(cleaned) <= max_length:
        return cleaned

    # Truncate at word boundary if possible
    truncated = cleaned[:max_length]
    last_space = truncated.rfind(" ")
    if last_space > max_length // 2:
        truncated = truncated[:last_space]

    return truncated.rstrip(".,!?;:") + "..."


def _apply_session_cookies(
    response: Response, session: Session, tokens: dict, *, refresh_ttl_minutes: int
) -> None:
    # Convert naive datetime to UTC-aware for cookie expiration
    expires_at = session.expires_at
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    response.set_cookie(
        "session_id",
        session.id,
        httponly=True,
        secure=True,
        samesite="lax",
        expires=expires_at,
        path="/",
    )
    refresh_token = tokens.get("refresh_token")
    if refresh_token:
        response.set_cookie(
            "refresh_token",
            refresh_token,
            httponly=True,
            secure=True,
            samesite="lax",
            max_age=refresh_ttl_minutes * 60,
            path="/",
        )


@router.post("/auth/signup", response_model=Envelope, status_code=201, tags=["auth"])
async def signup(body: SignupRequest, response: Response):
    """Create a new user account.

    Registers a new user with email and password credentials. Returns session
    tokens and sets authentication cookies. Rate limited per email address.

    Raises:
        403: If signup is disabled in settings
        429: If rate limit exceeded for this email
    """
    settings = get_settings()
    if not settings.allow_signup:
        raise _http_error("forbidden", "signup disabled", status_code=403)
    runtime = get_runtime()
    await _enforce_rate_limit(
        runtime,
        f"signup:{body.email.lower()}",
        runtime.settings.signup_rate_limit_per_minute,
        60,
    )
    user, session, tokens = await runtime.auth.signup(
        email=body.email,
        password=body.password,
        handle=body.handle,
        tenant_id=body.tenant_id,
    )
    _apply_session_cookies(
        response,
        session,
        tokens,
        refresh_ttl_minutes=runtime.settings.refresh_token_ttl_minutes,
    )
    return Envelope(
        status="ok",
        data=AuthResponse(
            user_id=user.id,
            session_id=session.id,
            session_expires_at=session.expires_at,
            mfa_required=session.mfa_required,
            access_token=tokens.get("access_token"),
            refresh_token=tokens.get("refresh_token"),
            token_type=tokens.get("token_type"),
            role=user.role,
            tenant_id=user.tenant_id,
        ),
    )


@router.post("/auth/login", response_model=Envelope, tags=["auth"])
async def login(body: LoginRequest, response: Response):
    """Authenticate user with email and password.

    Validates credentials and returns session tokens with authentication cookies.
    Supports optional MFA verification via mfa_code parameter.

    Raises:
        401: If credentials are invalid
        429: If rate limit exceeded for this email
    """
    runtime = get_runtime()
    await _enforce_rate_limit(
        runtime,
        f"login:{body.email.lower()}",
        runtime.settings.login_rate_limit_per_minute,
        60,
    )
    user, session, tokens = await runtime.auth.login(
        email=body.email,
        password=body.password,
        mfa_code=body.mfa_code,
        tenant_id=body.tenant_id,
    )
    if not user or not session:
        raise _http_error("unauthorized", "invalid credentials", status_code=401)
    _apply_session_cookies(
        response,
        session,
        tokens,
        refresh_ttl_minutes=runtime.settings.refresh_token_ttl_minutes,
    )
    return Envelope(
        status="ok",
        data=AuthResponse(
            user_id=user.id,
            session_id=session.id,
            session_expires_at=session.expires_at,
            mfa_required=session.mfa_required,
            access_token=tokens.get("access_token"),
            refresh_token=tokens.get("refresh_token"),
            token_type=tokens.get("token_type"),
            role=user.role,
            tenant_id=user.tenant_id,
        ),
    )


@router.post("/auth/oauth/{provider}/start", response_model=Envelope, tags=["auth"])
async def oauth_start(
    provider: str = Path(..., description="OAuth provider (google, github, etc.)"),
    body: OAuthStartRequest = ...,
):
    """Start OAuth authentication flow.

    Returns authorization URL for the specified OAuth provider.
    Client should redirect user to this URL to complete authentication.
    """
    runtime = get_runtime()
    # Rate limit OAuth start to prevent state token exhaustion
    await _enforce_rate_limit(
        runtime,
        f"oauth:start:{provider}",
        limit=20,
        window_seconds=60,
    )
    start = await runtime.auth.start_oauth(
        provider, redirect_uri=body.redirect_uri, tenant_id=body.tenant_id
    )
    return Envelope(
        status="ok",
        data=OAuthStartResponse(
            authorization_url=start["authorization_url"],
            state=start["state"],
            provider=provider,
        ),
    )


@router.get("/auth/oauth/{provider}/callback", response_model=Envelope, tags=["auth"])
async def oauth_callback(
    provider: str = Path(..., description="OAuth provider"),
    code: str = Query(..., max_length=512, description="Authorization code from OAuth provider"),
    state: str = Query(..., max_length=128, description="State parameter for CSRF protection"),
    response: Response = ...,
    tenant_id: Optional[str] = Query(None, max_length=128, description="Optional tenant ID"),
):
    """Complete OAuth authentication flow.

    Exchanges authorization code for tokens and creates user session.
    Called by OAuth provider after user authorizes the application.
    """
    runtime = get_runtime()
    # Rate limit OAuth callback to prevent code brute-forcing
    await _enforce_rate_limit(
        runtime,
        f"oauth:callback:{provider}",
        limit=10,
        window_seconds=60,
    )
    user, session, tokens = await runtime.auth.complete_oauth(
        provider, code, state, tenant_id=tenant_id
    )
    if not user or not session:
        raise _http_error("unauthorized", "oauth verification failed", status_code=401)
    _apply_session_cookies(
        response,
        session,
        tokens,
        refresh_ttl_minutes=runtime.settings.refresh_token_ttl_minutes,
    )
    return Envelope(
        status="ok",
        data=AuthResponse(
            user_id=user.id,
            session_id=session.id,
            session_expires_at=session.expires_at,
            mfa_required=session.mfa_required,
            access_token=tokens.get("access_token"),
            refresh_token=tokens.get("refresh_token"),
            token_type=tokens.get("token_type"),
            role=user.role,
            tenant_id=user.tenant_id,
        ),
    )


@router.post("/auth/refresh", response_model=Envelope, tags=["auth"])
async def refresh_tokens(
    body: TokenRefreshRequest,
    response: Response,
    authorization: Optional[str] = Header(None),
    x_tenant_id: Optional[str] = Header(
        None, convert_underscores=False, alias="X-Tenant-ID"
    ),
):
    runtime = get_runtime()
    tenant_hint = body.tenant_id or x_tenant_id
    user, session, tokens = await runtime.auth.refresh_tokens(
        body.refresh_token, tenant_hint=tenant_hint
    )
    if not user or not session:
        raise _http_error("unauthorized", "invalid refresh", status_code=401)
    _apply_session_cookies(
        response,
        session,
        tokens,
        refresh_ttl_minutes=runtime.settings.refresh_token_ttl_minutes,
    )
    return Envelope(
        status="ok",
        data=AuthResponse(
            user_id=user.id,
            session_id=session.id,
            session_expires_at=session.expires_at,
            mfa_required=session.mfa_required,
            access_token=tokens.get("access_token"),
            refresh_token=tokens.get("refresh_token"),
            token_type=tokens.get("token_type"),
            role=user.role,
            tenant_id=user.tenant_id,
        ),
    )


@router.get("/admin/users", response_model=Envelope, tags=["admin"])
async def admin_list_users(
    tenant_id: Optional[str] = None,
    limit: Optional[int] = Query(None, ge=1, description="Maximum users to return"),
    principal: AuthContext = Depends(get_admin_user),
):
    """List users in the admin's tenant.

    Returns:
        List of users with their roles and metadata.
    """
    runtime = get_runtime()
    await _enforce_rate_limit(
        runtime,
        f"admin:read:{principal.user_id}",
        runtime.settings.admin_rate_limit_per_minute,
        runtime.settings.admin_rate_limit_window_seconds,
    )
    paging = _get_pagination_settings(runtime)
    resolved_limit = min(limit or paging["default_page_size"], paging["max_page_size"])
    # Admin can only see users in their own tenant (prevent cross-tenant access)
    target_tenant = tenant_id or principal.tenant_id
    if target_tenant != principal.tenant_id:
        raise _http_error(
            "forbidden", "cannot access other tenant users", status_code=403
        )
    users = runtime.auth.list_users(tenant_id=target_tenant, limit=resolved_limit)
    return Envelope(
        status="ok", data=UserListResponse(items=[_user_to_response(u) for u in users])
    )


@router.post("/admin/users", response_model=Envelope, tags=["admin"])
async def admin_create_user(
    body: AdminCreateUserRequest, principal: AuthContext = Depends(get_admin_user)
):
    runtime = get_runtime()
    await _enforce_rate_limit(
        runtime,
        f"admin:create_user:{principal.user_id}",
        runtime.settings.admin_rate_limit_per_minute,
        runtime.settings.admin_rate_limit_window_seconds,
    )
    target_tenant = body.tenant_id or principal.tenant_id
    if target_tenant != principal.tenant_id:
        raise _http_error(
            "forbidden", "cannot create users in other tenant", status_code=403
        )
    user, password = await runtime.auth.admin_create_user(
        email=body.email,
        password=body.password,
        handle=body.handle,
        tenant_id=target_tenant,
        role=body.role,
        plan_tier=body.plan_tier,
        is_active=body.is_active,
        meta=body.meta,
    )
    return Envelope(
        status="ok",
        data=AdminCreateUserResponse(
            **_user_to_response(user).model_dump(), password=password
        ),
    )


@router.post("/admin/users/{user_id}/role", response_model=Envelope, tags=["admin"])
async def admin_set_role(
    user_id: str,
    body: UpdateUserRoleRequest,
    principal: AuthContext = Depends(get_admin_user),
):
    runtime = get_runtime()
    await _enforce_rate_limit(
        runtime,
        f"admin:set_role:{principal.user_id}",
        runtime.settings.admin_rate_limit_per_minute,
        runtime.settings.admin_rate_limit_window_seconds,
    )
    user = runtime.auth.set_user_role(user_id, body.role)
    if not user:
        raise NotFoundError("user not found", detail={"user_id": user_id})
    return Envelope(status="ok", data=_user_to_response(user))


@router.delete("/admin/users/{user_id}", response_model=Envelope, tags=["admin"])
async def admin_delete_user(
    user_id: str, principal: AuthContext = Depends(get_admin_user)
):
    runtime = get_runtime()
    await _enforce_rate_limit(
        runtime,
        f"admin:delete_user:{principal.user_id}",
        runtime.settings.admin_rate_limit_per_minute,
        runtime.settings.admin_rate_limit_window_seconds,
    )
    removed = runtime.auth.delete_user(user_id)
    if not removed:
        raise NotFoundError("user not found", detail={"user_id": user_id})
    return Envelope(status="ok", data={"deleted": True, "user_id": user_id})


@router.get("/admin/adapters", response_model=Envelope, tags=["admin"])
async def admin_list_adapters(principal: AuthContext = Depends(get_admin_user)):
    runtime = get_runtime()
    await _enforce_rate_limit(
        runtime,
        f"admin:read:{principal.user_id}",
        runtime.settings.admin_rate_limit_per_minute,
        runtime.settings.admin_rate_limit_window_seconds,
    )
    # Filter adapters by tenant to prevent cross-tenant data exposure
    adapters = list(runtime.store.list_artifacts(
        type_filter="adapter", tenant_id=principal.tenant_id
    ))

    # Batch fetch current versions
    artifact_ids = [a.id for a in adapters]
    versions = runtime.store.get_artifact_current_versions(artifact_ids)

    return Envelope(
        status="ok",
        data=ArtifactListResponse(
            items=[
                ArtifactResponse(
                    id=a.id,
                    type=a.type,
                    kind=a.schema.get("kind") if isinstance(a.schema, dict) else None,
                    name=a.name,
                    description=a.description,
                    schema=a.schema,
                    owner_user_id=a.owner_user_id,
                    visibility=getattr(a, "visibility", "private"),
                    version=versions.get(a.id, 1),
                    created_at=a.created_at,
                    updated_at=a.updated_at,
                )
                for a in adapters
            ]
        ),
    )


@router.get("/admin/objects", response_model=Envelope, tags=["admin"])
async def admin_inspect_objects(
    kind: Optional[str] = None,
    limit: Optional[int] = Query(None, ge=1, description="Maximum objects to return"),
    principal: AuthContext = Depends(get_admin_user),
):
    runtime = get_runtime()
    await _enforce_rate_limit(
        runtime,
        f"admin:read:{principal.user_id}",
        runtime.settings.admin_rate_limit_per_minute,
        runtime.settings.admin_rate_limit_window_seconds,
    )
    paging = _get_pagination_settings(runtime)
    resolved_limit = min(limit or paging["default_conversations_limit"], paging["max_page_size"])
    if not hasattr(runtime.store, "inspect_state"):
        raise BadRequestError("inspect not supported")
    details = runtime.store.inspect_state(
        kind=kind, tenant_id=principal.tenant_id, limit=resolved_limit
    )
    summary = {k: len(v) for k, v in details.items()}
    return Envelope(
        status="ok", data=AdminInspectionResponse(summary=summary, details=details)
    )


@router.get("/admin/settings", response_model=Envelope, tags=["admin"])
async def get_admin_settings(principal: AuthContext = Depends(get_admin_user)):
    """Get current runtime settings configurable via admin UI.

    Returns pagination limits, model settings, and other runtime configuration
    that can be modified without restarting the server.
    """
    runtime = get_runtime()
    await _enforce_rate_limit(
        runtime,
        f"admin:read:{principal.user_id}",
        runtime.settings.admin_rate_limit_per_minute,
        runtime.settings.admin_rate_limit_window_seconds,
    )
    settings = get_settings()
    runtime_config = (
        runtime.store.get_runtime_config()
        if hasattr(runtime.store, "get_runtime_config")
        else {}
    )

    # Merge runtime config with env defaults (runtime config takes precedence)
    return Envelope(
        status="ok",
        data=AdminSettingsResponse(
            default_page_size=runtime_config.get("default_page_size", settings.default_page_size),
            max_page_size=runtime_config.get("max_page_size", settings.max_page_size),
            default_conversations_limit=runtime_config.get(
                "default_conversations_limit", settings.default_conversations_limit
            ),
            model_backend=runtime_config.get("model_backend"),
            model_path=runtime_config.get("model_path"),
        ),
    )


@router.patch("/admin/settings", response_model=Envelope, tags=["admin"])
async def update_admin_settings(
    body: AdminSettingsUpdateRequest,
    principal: AuthContext = Depends(get_admin_user),
):
    """Update runtime settings via admin UI.

    Only provided fields are updated. Changes take effect immediately
    without requiring a server restart.
    """
    runtime = get_runtime()
    await _enforce_rate_limit(
        runtime,
        f"admin:settings:{principal.user_id}",
        runtime.settings.admin_rate_limit_per_minute,
        runtime.settings.admin_rate_limit_window_seconds,
    )

    if not hasattr(runtime.store, "set_runtime_config"):
        raise BadRequestError("runtime config update not supported by storage backend")

    # Build update dict from non-None fields
    update = {}
    if body.default_page_size is not None:
        update["default_page_size"] = body.default_page_size
    if body.max_page_size is not None:
        update["max_page_size"] = body.max_page_size
    if body.default_conversations_limit is not None:
        update["default_conversations_limit"] = body.default_conversations_limit
    if body.model_backend is not None:
        update["model_backend"] = body.model_backend
    if body.model_path is not None:
        update["model_path"] = body.model_path

    if not update:
        raise BadRequestError("no settings provided to update")

    updated_config = runtime.store.set_runtime_config(update)
    settings = get_settings()

    return Envelope(
        status="ok",
        data=AdminSettingsResponse(
            default_page_size=updated_config.get("default_page_size", settings.default_page_size),
            max_page_size=updated_config.get("max_page_size", settings.max_page_size),
            default_conversations_limit=updated_config.get(
                "default_conversations_limit", settings.default_conversations_limit
            ),
            model_backend=updated_config.get("model_backend"),
            model_path=updated_config.get("model_path"),
        ),
    )


@router.post("/auth/mfa/request", response_model=Envelope, tags=["auth"])
async def request_mfa(body: MFARequest):
    runtime = get_runtime()
    auth_ctx = await runtime.auth.resolve_session(
        body.session_id, allow_pending_mfa=True
    )
    if not auth_ctx:
        raise _http_error("unauthorized", "invalid session", status_code=401)
    await _enforce_rate_limit(
        runtime,
        f"mfa:request:{auth_ctx.user_id}",
        runtime.settings.mfa_rate_limit_per_minute,
        60,
    )
    challenge = await runtime.auth.issue_mfa_challenge(user_id=auth_ctx.user_id)
    return Envelope(status="ok", data=challenge)


@router.post("/auth/mfa/verify", response_model=Envelope, tags=["auth"])
async def verify_mfa(body: MFAVerifyRequest, response: Response):
    runtime = get_runtime()
    auth_ctx = await runtime.auth.resolve_session(
        body.session_id, allow_pending_mfa=True
    )
    if not auth_ctx:
        raise _http_error("unauthorized", "invalid session", status_code=401)
    await _enforce_rate_limit(
        runtime,
        f"mfa:verify:{auth_ctx.user_id}",
        runtime.settings.mfa_rate_limit_per_minute,
        60,
    )
    ok = await runtime.auth.verify_mfa_challenge(
        user_id=auth_ctx.user_id, code=body.code, session_id=body.session_id
    )
    if not ok:
        raise _http_error("unauthorized", "invalid mfa", status_code=401)
    user, session, tokens = runtime.auth.issue_tokens_for_session(body.session_id)
    resp: dict = {"status": "verified"}
    if user and session:
        resp.update(
            AuthResponse(
                user_id=user.id,
                session_id=session.id,
                session_expires_at=session.expires_at,
                mfa_required=False,
                access_token=tokens.get("access_token"),
                refresh_token=tokens.get("refresh_token"),
                token_type=tokens.get("token_type"),
                role=user.role,
                tenant_id=user.tenant_id,
            ).model_dump()
        )
        _apply_session_cookies(
            response,
            session,
            tokens,
            refresh_ttl_minutes=runtime.settings.refresh_token_ttl_minutes,
        )
    return Envelope(status="ok", data=resp)


@router.get("/auth/mfa/status", response_model=Envelope, tags=["auth"])
async def get_mfa_status(principal: AuthContext = Depends(get_user)):
    """Get the current MFA status for the authenticated user."""
    runtime = get_runtime()
    await _enforce_rate_limit(
        runtime,
        f"mfa:status:{principal.user_id}",
        runtime.settings.read_rate_limit_per_minute,
        60,
    )
    mfa_cfg = runtime.store.get_user_mfa_secret(principal.user_id)
    return Envelope(
        status="ok",
        data=MFAStatusResponse(
            enabled=mfa_cfg.enabled if mfa_cfg else False,
            configured=mfa_cfg is not None,
        ),
    )


@router.post("/auth/mfa/disable", response_model=Envelope, tags=["auth"])
async def disable_mfa(body: MFADisableRequest, principal: AuthContext = Depends(get_user)):
    """Disable MFA for the authenticated user. Requires current TOTP code."""
    runtime = get_runtime()
    await _enforce_rate_limit(
        runtime,
        f"mfa:disable:{principal.user_id}",
        runtime.settings.mfa_rate_limit_per_minute,
        60,
    )
    mfa_cfg = runtime.store.get_user_mfa_secret(principal.user_id)
    if not mfa_cfg or not mfa_cfg.enabled:
        raise _http_error("validation_error", "MFA not enabled", status_code=400)
    # Verify the code before disabling
    if not runtime.auth._verify_totp(mfa_cfg.secret, body.code):
        raise _http_error("unauthorized", "invalid MFA code", status_code=401)
    # Disable MFA by setting enabled=False
    runtime.store.set_user_mfa_secret(principal.user_id, mfa_cfg.secret, enabled=False)

    # SECURITY: Revoke all other sessions to force re-authentication
    await runtime.auth.revoke_all_user_sessions(
        principal.user_id, except_session_id=principal.session_id
    )

    return Envelope(status="ok", data={"status": "disabled"})


@router.post("/auth/reset/request", response_model=Envelope, tags=["auth"])
async def request_reset(body: PasswordResetRequest):
    runtime = get_runtime()
    await _enforce_rate_limit(
        runtime,
        f"reset:{body.email.lower()}",
        runtime.settings.reset_rate_limit_per_minute,
        60,
    )
    # Check if user exists before generating token (don't reveal if user exists)
    user = runtime.store.get_user_by_email(body.email)
    if user:
        token = await runtime.auth.initiate_password_reset(body.email)
        # Run blocking SMTP in thread to avoid blocking event loop
        await asyncio.to_thread(runtime.email.send_password_reset, body.email, token)
    # Always return success to prevent email enumeration
    return Envelope(status="ok", data={"status": "sent"})


@router.post("/auth/reset/confirm", response_model=Envelope, tags=["auth"])
async def confirm_reset(body: PasswordResetConfirm):
    runtime = get_runtime()
    # Rate limit to prevent token brute-forcing
    await _enforce_rate_limit(
        runtime,
        "reset:confirm",
        limit=5,
        window_seconds=300,
    )
    ok = await runtime.auth.complete_password_reset(body.token, body.new_password)
    if not ok:
        raise _http_error("validation_error", "invalid token", status_code=400)
    return Envelope(status="ok", data={"status": "reset"})


@router.get("/me", response_model=Envelope, tags=["auth"])
async def get_current_user(principal: AuthContext = Depends(get_user)):
    """Get the current user's profile.

    Returns user details including email verification status.
    """
    runtime = get_runtime()
    await _enforce_rate_limit(
        runtime,
        f"read:{principal.user_id}",
        runtime.settings.read_rate_limit_per_minute,
        60,
    )
    user = runtime.store.get_user(principal.user_id)
    if not user:
        raise _http_error("not_found", "user not found", status_code=404)
    return Envelope(
        status="ok",
        data=UserResponse(
            id=user.id,
            email=user.email,
            handle=user.handle,
            role=user.role,
            tenant_id=user.tenant_id or "global",
            created_at=user.created_at,
            is_active=user.is_active,
            plan_tier=user.plan_tier or "free",
            meta=user.meta,
        ),
    )


@router.get("/settings", response_model=Envelope, tags=["settings"])
async def get_user_settings(principal: AuthContext = Depends(get_user)):
    """Get the current user's settings.

    Returns user preferences like locale, timezone, voice settings.
    """
    runtime = get_runtime()
    await _enforce_rate_limit(
        runtime,
        f"read:{principal.user_id}",
        runtime.settings.read_rate_limit_per_minute,
        60,
    )
    settings = runtime.store.get_user_settings(principal.user_id)
    if not settings:
        # Return empty settings if none exist
        return Envelope(
            status="ok",
            data=UserSettingsResponse(),
        )
    return Envelope(
        status="ok",
        data=UserSettingsResponse(
            locale=settings.locale,
            timezone=settings.timezone,
            default_voice=settings.default_voice,
            default_style=settings.default_style,
            flags=settings.flags,
        ),
    )


@router.patch("/settings", response_model=Envelope, tags=["settings"])
async def update_user_settings(
    body: UserSettingsRequest,
    principal: AuthContext = Depends(get_user),
):
    """Update the current user's settings.

    Only provided fields will be updated; omitted fields remain unchanged.
    """
    runtime = get_runtime()
    await _enforce_rate_limit(
        runtime,
        f"write:{principal.user_id}",
        runtime.settings.chat_rate_limit_per_minute,
        60,
    )
    settings = runtime.store.set_user_settings(
        principal.user_id,
        locale=body.locale,
        timezone=body.timezone,
        default_voice=body.default_voice,
        default_style=body.default_style,
        flags=body.flags,
    )
    return Envelope(
        status="ok",
        data=UserSettingsResponse(
            locale=settings.locale,
            timezone=settings.timezone,
            default_voice=settings.default_voice,
            default_style=settings.default_style,
            flags=settings.flags,
        ),
    )


@router.post("/auth/request_email_verification", response_model=Envelope, tags=["auth"])
async def request_email_verification(principal: AuthContext = Depends(get_user)):
    runtime = get_runtime()
    # Rate limit to prevent email spam
    await _enforce_rate_limit(
        runtime,
        f"verify:request:{principal.user_id}",
        limit=5,
        window_seconds=300,
    )
    user = runtime.store.get_user(principal.user_id)
    if not user:
        raise _http_error("not_found", "user not found", status_code=404)
    token = await runtime.auth.request_email_verification(user)
    # Run blocking SMTP in thread to avoid blocking event loop
    await asyncio.to_thread(runtime.email.send_email_verification, user.email, token)
    return Envelope(status="ok", data={"status": "sent"})


@router.post("/auth/verify_email", response_model=Envelope, tags=["auth"])
async def verify_email(body: EmailVerificationRequest):
    runtime = get_runtime()
    # Rate limit to prevent token brute-forcing
    await _enforce_rate_limit(
        runtime,
        "verify:email",
        limit=10,
        window_seconds=300,
    )
    ok = await runtime.auth.complete_email_verification(body.token)
    if not ok:
        raise _http_error("validation_error", "invalid token", status_code=400)
    return Envelope(status="ok", data={"status": "verified"})


@router.post("/auth/password/change", response_model=Envelope, tags=["auth"])
async def change_password(
    body: PasswordChangeRequest,
    principal: AuthContext = Depends(get_user),
):
    """Change the current user's password.

    Requires the current password for verification.
    """
    runtime = get_runtime()
    await _enforce_rate_limit(
        runtime,
        f"password:change:{principal.user_id}",
        limit=5,
        window_seconds=300,
    )

    # Verify current password
    if not runtime.auth.verify_password(principal.user_id, body.current_password):
        raise _http_error("unauthorized", "current password is incorrect", status_code=401)

    # Save new password
    runtime.auth.save_password(principal.user_id, body.new_password)

    # SECURITY: Revoke all other sessions to force re-authentication
    await runtime.auth.revoke_all_user_sessions(
        principal.user_id, except_session_id=principal.session_id
    )

    return Envelope(status="ok", data={"status": "changed"})


@router.post("/auth/logout", response_model=Envelope, tags=["auth"])
async def logout(
    response: Response,
    session_id: Optional[str] = Header(None, convert_underscores=False),
    authorization: Optional[str] = Header(None),
):
    runtime = get_runtime()
    if authorization or session_id:
        ctx = await runtime.auth.authenticate(
            authorization, session_id, allow_pending_mfa=True
        )
        if not ctx:
            raise _http_error("unauthorized", "invalid session", status_code=401)
        # SECURITY: Only allow revoking the authenticated user's own session
        target_session = session_id or ctx.session_id
        if target_session and target_session != ctx.session_id:
            # Verify session ownership before revoking
            sess = runtime.store.get_session(target_session)
            if not sess or sess.user_id != ctx.user_id:
                raise _http_error("forbidden", "cannot revoke other user sessions", status_code=403)
        if target_session:
            await runtime.auth.revoke(target_session)
    response.delete_cookie("session_id", path="/", secure=True, samesite="lax")
    response.delete_cookie("refresh_token", path="/", secure=True, samesite="lax")
    return Envelope(status="ok", data={"message": "session revoked"})


@router.post("/chat", response_model=Envelope, tags=["chat"])
async def chat(
    body: ChatRequest,
    principal: AuthContext = Depends(get_user),
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
):
    """Process a chat message and generate a response.

    Routes the message through the workflow engine, applies relevant adapters,
    and returns the assistant's response. Supports voice transcription, RAG
    context, and idempotency via the Idempotency-Key header.

    Raises:
        401: If authentication fails
        404: If conversation or context not found
        409: If concurrent workflow limit exceeded (SPEC §18)
        429: If rate limit exceeded
    """
    runtime = get_runtime()
    user_id = principal.user_id

    # Get user's plan tier for per-plan rate limits (SPEC §18)
    user = runtime.store.get_user(user_id)
    plan_tier = user.plan_tier if user else "free"

    # SPEC §18: Accept Idempotency-Key when provided (optional)
    async with IdempotencyGuard(
        "chat", user_id, idempotency_key, require=False
    ) as idem:
        if idem.cached:
            return idem.cached

        # SPEC §18: Per-plan adjustable rate limits
        await _enforce_rate_limit_per_plan(
            runtime,
            f"chat:{user_id}",
            runtime.settings.chat_rate_limit_per_minute,
            runtime.settings.chat_rate_limit_window_seconds,
            plan_tier,
        )

        # SPEC §18: Concurrency cap - max 3 concurrent workflows per user
        await _acquire_workflow_slot(runtime, user_id)
        try:
            conversation: Conversation | None = None
            context_id = body.context_id
            validated_context_id: str | None = None
            if body.conversation_id:
                conversation = _get_owned_conversation(
                    runtime, body.conversation_id, principal
                )
            else:
                if context_id:
                    _get_owned_context(runtime, context_id, principal)
                    validated_context_id = context_id
                # Generate title from first message for new conversations
                auto_title = _generate_conversation_title(body.message.content)
                conversation = runtime.store.create_conversation(
                    user_id=user_id, active_context_id=body.context_id, title=auto_title
                )
            conversation_id = conversation.id
            context_id = context_id or conversation.active_context_id
            if context_id and context_id != validated_context_id:
                _get_owned_context(runtime, context_id, principal)
            user_content = body.message.content
            voice_meta: dict = {}
            if body.message.mode == "voice":
                try:
                    audio_bytes = base64.b64decode(body.message.content)
                except Exception as exc:
                    raise _http_error(
                        "bad_request",
                        "invalid base64-encoded audio payload",
                        status_code=400,
                    ) from exc
                transcript = await runtime.voice.transcribe(audio_bytes, user_id=user_id) or {}
                user_content = transcript.get("transcript") or transcript.get("text")
                if not user_content:
                    raise _http_error(
                        "bad_request", "unable to transcribe audio", status_code=400
                    )
                voice_meta = {"mode": "voice", "transcript": transcript}
            user_content_struct = normalize_content_struct(
                body.message.content_struct, user_content
            )
            runtime.store.append_message(
                conversation_id,
                sender="user",
                role="user",
                content=user_content,
                meta=voice_meta or None,
                content_struct=user_content_struct,
            )
            orchestration = await runtime.workflow.run(
                body.workflow_id,
                conversation_id,
                user_content,
                context_id,
                user_id,
                tenant_id=principal.tenant_id,
            )
            orchestration_dict: dict[str, Any] = (
                orchestration if isinstance(orchestration, dict) else {}
            )
            adapter_names = _stringify_adapters(orchestration_dict.get("adapters", []))
            assistant_content_struct = normalize_content_struct(
                orchestration_dict.get("content_struct"),
                orchestration_dict.get("content"),
            )
            assistant_content = orchestration_dict.get("content", "No response generated.")
            assistant_msg = runtime.store.append_message(
                conversation_id,
                sender="assistant",
                role="assistant",
                content=assistant_content,
                content_struct=assistant_content_struct,
                meta={
                    "adapters": orchestration_dict.get("adapters", []),
                    "adapter_gates": orchestration_dict.get("adapter_gates", []),
                    "routing_trace": orchestration_dict.get("routing_trace", []),
                    "workflow_trace": orchestration_dict.get("workflow_trace", []),
                    "usage": orchestration_dict.get("usage", {}),
                },
            )
            resp = ChatResponse(
                message_id=assistant_msg.id,
                conversation_id=conversation_id,
                content=assistant_msg.content,
                content_struct=assistant_msg.content_struct,
                workflow_id=body.workflow_id,
                adapters=adapter_names,
                adapter_gates=orchestration_dict.get("adapter_gates", []),
                usage=orchestration_dict.get("usage", {}),
                context_snippets=orchestration_dict.get("context_snippets", []),
                routing_trace=orchestration_dict.get("routing_trace", []),
                workflow_trace=orchestration_dict.get("workflow_trace", []),
            )
            envelope = Envelope(
                status="ok", data=resp.model_dump(), request_id=idem.request_id
            )
            if runtime.cache:
                history = runtime.store.list_messages(
                    conversation_id, user_id=principal.user_id
                )
                await runtime.workflow.cache_conversation_state(conversation_id, history)
            await idem.store_result(envelope)
            return envelope
        finally:
            # Always release workflow slot, even on error
            await _release_workflow_slot(runtime, user_id)
    # Exceptions bubble through the guard which records failed states


@router.post("/chat/cancel", response_model=Envelope, tags=["chat"])
async def cancel_chat(
    body: ChatCancelRequest,
    principal: AuthContext = Depends(get_user),
):
    """Cancel an in-progress chat request per SPEC §18.

    Cancellation signals the orchestrator to abort decode, free KV cache and
    adapter refs, and emit cancel_ack with partial tokens if any.

    Returns:
        cancelled: True if request was found and cancelled
        message: Human-readable status
    """
    runtime = get_runtime()

    # Rate limit cancellation requests (5 per minute per user)
    await _enforce_rate_limit(
        runtime,
        f"chat_cancel:{principal.user_id}",
        limit=5,
        window_seconds=60,
    )

    request_id = body.request_id

    # Try to cancel in-memory active request
    cancelled = await _cancel_request(request_id)

    if cancelled:
        logger.info("chat_request_cancelled", request_id=request_id, user_id=principal.user_id)
        return Envelope(
            status="ok",
            data=ChatCancelResponse(
                request_id=request_id,
                cancelled=True,
                message="Request cancelled successfully",
            ).model_dump(),
        )

    # Request not found in active requests - may have already completed or never existed
    # Per SPEC §18, this is not an error condition
    logger.info(
        "chat_cancel_request_not_found",
        request_id=request_id,
        user_id=principal.user_id,
    )
    return Envelope(
        status="ok",
        data=ChatCancelResponse(
            request_id=request_id,
            cancelled=False,
            message="Request not found or already completed",
        ).model_dump(),
    )


@router.post("/preferences", response_model=Envelope, tags=["preferences"])
async def record_preference(
    body: PreferenceEventRequest,
    response: Response,
    principal: AuthContext = Depends(get_user),
):
    """Record user preference feedback for an assistant message.

    Rate limited to 30 requests per minute to protect clustering resources.
    """
    runtime = get_runtime()

    # Rate limit preference recording (triggers expensive clustering operations)
    await _enforce_rate_limit(
        runtime,
        f"preferences:{principal.user_id}",
        limit=30,
        window_seconds=60,
        response=response,
    )

    event = runtime.training.record_feedback_event(
        user_id=principal.user_id,
        conversation_id=body.conversation_id,
        message_id=body.message_id,
        feedback=body.feedback,
        score=body.score,
        context_text=body.context_text,
        corrected_text=body.corrected_text,
        weight=body.weight,
        explicit_signal=body.explicit_signal,
        routing_trace=body.routing_trace,
        adapter_gates=body.adapter_gates,
        notes=body.notes,
    )
    await runtime.clusterer.cluster_user_preferences(principal.user_id)
    runtime.clusterer.promote_skill_adapters()
    resp = PreferenceEventResponse(
        id=event.id,
        cluster_id=event.cluster_id,
        feedback=event.feedback,
        created_at=event.created_at,
    )
    return Envelope(status="ok", data=resp)


@router.post("/preferences/routing_feedback", response_model=Envelope, tags=["preferences"])
async def record_routing_feedback(
    body: PreferenceEventRequest,
    response: Response,
    principal: AuthContext = Depends(get_user),
):
    """Record routing-specific feedback for adapter selection improvement.

    Rate limited to 30 requests per minute to protect clustering resources.
    """
    runtime = get_runtime()

    # Rate limit routing feedback (shares limit with general preferences)
    await _enforce_rate_limit(
        runtime,
        f"preferences:{principal.user_id}",
        limit=30,
        window_seconds=60,
        response=response,
    )

    event = runtime.training.record_feedback_event(
        user_id=principal.user_id,
        conversation_id=body.conversation_id,
        message_id=body.message_id,
        feedback=body.feedback,
        explicit_signal=body.explicit_signal or "routing_feedback",
        score=body.score,
        context_text=body.context_text,
        corrected_text=body.corrected_text,
        weight=body.weight,
        routing_trace=body.routing_trace,
        adapter_gates=body.adapter_gates,
        notes=body.notes,
    )
    resp = PreferenceEventResponse(
        id=event.id,
        cluster_id=event.cluster_id,
        feedback=event.feedback,
        created_at=event.created_at,
    )
    return Envelope(status="ok", data=resp)


@router.get("/preferences/insights", response_model=Envelope, tags=["preferences"])
async def preference_insights(
    response: Response,
    principal: AuthContext = Depends(get_user),
):
    """Get user preference insights and cluster summaries.

    Rate limited to 60 requests per minute.
    """
    runtime = get_runtime()

    # Rate limit insights queries
    await _enforce_rate_limit(
        runtime,
        f"preferences_insights:{principal.user_id}",
        limit=60,
        window_seconds=60,
        response=response,
    )

    summary = runtime.training.summarize_preferences(principal.user_id)
    return Envelope(status="ok", data=PreferenceInsightsResponse(**summary))


@router.get("/artifacts", response_model=Envelope, tags=["artifacts"])
async def list_artifacts(
    type: Optional[str] = None,
    kind: Optional[str] = None,
    visibility: Optional[str] = Query(None, pattern="^(private|shared|global)$", description="Filter by visibility"),
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: Optional[int] = Query(None, ge=1, description="Items per page"),
    limit: Optional[int] = Query(None, ge=1, description="Alias for page_size (for frontend compatibility)"),
    principal: AuthContext = Depends(get_user),
):
    """List artifacts owned by the current user.

    Supports filtering by type, kind, and visibility, with pagination.

    Args:
        type: Filter by artifact type (e.g., 'adapter', 'workflow')
        kind: Filter by artifact kind (e.g., 'adapter.lora', 'workflow.linear')
        visibility: Filter by visibility ('private', 'shared', 'global')
        page: Page number (1-indexed)
        page_size: Number of items per page

    Returns:
        Paginated list of artifacts with metadata.
    """
    runtime = get_runtime()
    await _enforce_rate_limit(
        runtime,
        f"read:{principal.user_id}",
        runtime.settings.read_rate_limit_per_minute,
        60,
    )
    paging = _get_pagination_settings(runtime)
    kind_filter = kind or (type if type and "." in type else None)
    type_filter = type if type and "." not in type else None
    if not type_filter and kind_filter:
        type_filter = kind_filter.split(".", 1)[0]
    # Accept 'limit' as alias for 'page_size' for frontend compatibility
    effective_page_size = limit if limit is not None else (page_size or paging["default_page_size"])
    resolved_page_size = min(max(effective_page_size, 1), paging["max_page_size"])

    # Get one extra item to determine if there are more pages
    raw_items = list(runtime.store.list_artifacts(
        type_filter=type_filter,
        kind_filter=kind_filter,
        owner_user_id=principal.user_id,
        page=page,
        page_size=resolved_page_size + 1,
        visibility=visibility,
    ))

    has_next = len(raw_items) > resolved_page_size
    page_items = raw_items[:resolved_page_size]

    # Batch fetch current versions for all artifacts
    artifact_ids = [a.id for a in page_items]
    versions = runtime.store.get_artifact_current_versions(artifact_ids)

    items = [
        ArtifactResponse(
            id=a.id,
            type=a.type,
            kind=a.schema.get("kind") if isinstance(a.schema, dict) else None,
            name=a.name,
            description=a.description,
            schema=a.schema,
            owner_user_id=a.owner_user_id,
            visibility=getattr(a, "visibility", "private"),
            version=versions.get(a.id, 1),
            created_at=a.created_at,
            updated_at=a.updated_at,
        )
        for a in page_items
    ]

    return Envelope(
        status="ok",
        data=ArtifactListResponse(
            items=items,
            has_next=has_next,
            next_page=page + 1 if has_next else None,
            page_size=resolved_page_size,
        ),
    )


@router.get("/artifacts/{artifact_id}", response_model=Envelope, tags=["artifacts"])
async def get_artifact(
    artifact_id: str = Path(..., max_length=255, description="Artifact identifier"),
    principal: AuthContext = Depends(get_user),
):
    runtime = get_runtime()
    await _enforce_rate_limit(
        runtime,
        f"read:{principal.user_id}",
        runtime.settings.read_rate_limit_per_minute,
        60,
    )
    artifact = _get_owned_artifact(runtime, artifact_id, principal)
    current_version = runtime.store.get_artifact_current_version(artifact_id)
    resp = ArtifactResponse(
        id=artifact.id,
        type=artifact.type,
        kind=artifact.schema.get("kind") if isinstance(artifact.schema, dict) else None,
        name=artifact.name,
        description=artifact.description,
        schema=artifact.schema,
        owner_user_id=artifact.owner_user_id,
        visibility=getattr(artifact, "visibility", "private"),
        version=current_version,
        created_at=artifact.created_at,
        updated_at=artifact.updated_at,
    )
    return Envelope(status="ok", data=resp)


@router.get("/tools/specs", response_model=Envelope, tags=["tools"])
async def list_tool_specs(
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(50, ge=1, le=200, description="Items per page"),
    principal: AuthContext = Depends(get_user),
):
    runtime = get_runtime()
    await _enforce_rate_limit(
        runtime,
        f"read:{principal.user_id}",
        runtime.settings.read_rate_limit_per_minute,
        60,
    )
    resolved_page_size = min(max(page_size, 1), 200)
    artifacts = list(runtime.store.list_artifacts(
        type_filter="tool",
        kind_filter="tool.spec",
        owner_user_id=principal.user_id,
        page=page,
        page_size=resolved_page_size,
    ))

    # Batch fetch current versions
    artifact_ids = [a.id for a in artifacts]
    versions = runtime.store.get_artifact_current_versions(artifact_ids)

    items = [
        ArtifactResponse(
            id=a.id,
            type=a.type,
            kind=a.schema.get("kind") if isinstance(a.schema, dict) else None,
            name=a.name,
            description=a.description,
            schema=a.schema,
            owner_user_id=a.owner_user_id,
            visibility=getattr(a, "visibility", "private"),
            version=versions.get(a.id, 1),
            created_at=a.created_at,
            updated_at=a.updated_at,
        )
        for a in artifacts
    ]
    next_page = page + 1 if len(items) == resolved_page_size else None
    return Envelope(
        status="ok",
        data=ToolSpecListResponse(
            items=items, next_page=next_page, page_size=resolved_page_size
        ),
    )


@router.get("/tools/specs/{artifact_id}", response_model=Envelope, tags=["tools"])
async def get_tool_spec(
    artifact_id: str = Path(..., max_length=255, description="Tool spec artifact ID"),
    principal: AuthContext = Depends(get_user),
):
    runtime = get_runtime()
    await _enforce_rate_limit(
        runtime,
        f"read:{principal.user_id}",
        runtime.settings.read_rate_limit_per_minute,
        60,
    )
    artifact = _get_owned_artifact(runtime, artifact_id, principal)
    if not (
        isinstance(artifact.schema, dict) and artifact.schema.get("kind") == "tool.spec"
    ):
        raise NotFoundError("tool spec not found", detail={"artifact_id": artifact_id})
    current_version = runtime.store.get_artifact_current_version(artifact_id)
    resp = ArtifactResponse(
        id=artifact.id,
        type=artifact.type,
        kind=artifact.schema.get("kind") if isinstance(artifact.schema, dict) else None,
        name=artifact.name,
        description=artifact.description,
        schema=artifact.schema,
        owner_user_id=artifact.owner_user_id,
        visibility=getattr(artifact, "visibility", "private"),
        version=current_version,
        created_at=artifact.created_at,
        updated_at=artifact.updated_at,
    )
    return Envelope(status="ok", data=resp)


@router.get("/workflows", response_model=Envelope, tags=["workflows"])
async def list_workflows(
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(50, ge=1, le=200, description="Items per page"),
    principal: AuthContext = Depends(get_user),
):
    runtime = get_runtime()
    await _enforce_rate_limit(
        runtime,
        f"read:{principal.user_id}",
        runtime.settings.read_rate_limit_per_minute,
        60,
    )
    resolved_page_size = min(max(page_size, 1), 200)
    artifacts = list(runtime.store.list_artifacts(
        type_filter="workflow",
        kind_filter="workflow.chat",
        owner_user_id=principal.user_id,  # Filter by owner
        page=page,
        page_size=resolved_page_size,
    ))

    # Batch fetch current versions
    artifact_ids = [a.id for a in artifacts]
    versions = runtime.store.get_artifact_current_versions(artifact_ids)

    items = [
        ArtifactResponse(
            id=a.id,
            type=a.type,
            kind=a.schema.get("kind") if isinstance(a.schema, dict) else None,
            name=a.name,
            description=a.description,
            schema=a.schema,
            owner_user_id=a.owner_user_id,
            visibility=getattr(a, "visibility", "private"),
            version=versions.get(a.id, 1),
            created_at=a.created_at,
            updated_at=a.updated_at,
        )
        for a in artifacts
    ]
    next_page = page + 1 if len(items) == resolved_page_size else None
    return Envelope(
        status="ok",
        data=WorkflowListResponse(
            items=items, next_page=next_page, page_size=resolved_page_size
        ),
    )


@router.get("/artifacts/{artifact_id}/versions", response_model=Envelope, tags=["artifacts"])
async def list_artifact_versions(
    artifact_id: str = Path(..., max_length=255, description="Artifact identifier"),
    limit: Optional[int] = Query(None, ge=1, description="Maximum versions to return"),
    principal: AuthContext = Depends(get_user),
):
    runtime = get_runtime()
    await _enforce_rate_limit(
        runtime,
        f"read:{principal.user_id}",
        runtime.settings.read_rate_limit_per_minute,
        60,
    )
    paging = _get_pagination_settings(runtime)
    resolved_limit = min(limit or paging["default_page_size"], paging["max_page_size"])
    # Verify ownership before listing versions
    _get_owned_artifact(runtime, artifact_id, principal)
    versions = runtime.store.list_artifact_versions(artifact_id, limit=resolved_limit)
    if not versions:
        artifact = runtime.store.get_artifact(artifact_id)
        if not artifact:
            raise NotFoundError(
                "artifact not found", detail={"artifact_id": artifact_id}
            )
    items = [
        ArtifactVersionResponse(
            id=v.id,
            artifact_id=v.artifact_id,
            version=v.version,
            schema=v.schema,
            created_by=v.created_by,
            change_note=v.change_note,
            created_at=v.created_at,
            fs_path=v.fs_path,
            meta=v.meta,
        )
        for v in versions
    ]
    return Envelope(status="ok", data=ArtifactVersionListResponse(items=items))


@router.post("/tools/{tool_id}/invoke", response_model=Envelope, tags=["tools"])
async def invoke_tool(
    tool_id: str,
    body: ToolInvokeRequest,
    principal: AuthContext = Depends(get_user),
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
):
    runtime = get_runtime()
    # SPEC §18: Accept Idempotency-Key when provided (optional)
    async with IdempotencyGuard(
        f"tools:{tool_id}:invoke", principal.user_id, idempotency_key, require=False
    ) as idem:
        if idem.cached:
            return idem.cached
        artifact = _get_owned_artifact(runtime, tool_id, principal)
        if not (
            isinstance(artifact.schema, dict)
            and artifact.schema.get("kind") == "tool.spec"
        ):
            raise NotFoundError("tool spec not found", detail={"artifact_id": tool_id})
        result = runtime.workflow.invoke_tool(
            artifact.schema,
            body.inputs or {},
            conversation_id=body.conversation_id,
            context_id=body.context_id,
            user_message=body.user_message,
            user_id=principal.user_id,
            tenant_id=principal.tenant_id,
        )
        response = ToolInvokeResponse(
            status=result.get("status", "ok") if isinstance(result, dict) else "ok",
            outputs=result.get("outputs", {}) if isinstance(result, dict) else {},
            content=result.get("content") if isinstance(result, dict) else None,
            usage=result.get("usage") if isinstance(result, dict) else None,
            context_snippets=(
                result.get("context_snippets") if isinstance(result, dict) else None
            ),
        )
        envelope = Envelope(status="ok", data=response, request_id=idem.request_id)
        await idem.store_result(envelope)
        return envelope


@router.post("/artifacts", response_model=Envelope, status_code=201, tags=["artifacts"])
async def create_artifact(
    body: ArtifactRequest,
    principal: AuthContext = Depends(get_user),
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
):
    runtime = get_runtime()
    # SPEC §18: Accept Idempotency-Key when provided (optional)
    async with IdempotencyGuard(
        "artifacts:create", principal.user_id, idempotency_key, require=False
    ) as idem:
        if idem.cached:
            return idem.cached
        if not isinstance(body.schema, dict):
            raise BadRequestError(
                "artifact schema must be an object",
                detail={"provided_type": type(body.schema).__name__},
            )
        if any(not isinstance(key, str) for key in body.schema):
            raise BadRequestError("artifact schema keys must be strings")
        try:
            json.dumps(body.schema)
        except (TypeError, ValueError):
            # SECURITY: Don't expose internal error details
            raise BadRequestError("artifact schema must be JSON serializable")
        schema_kind = body.schema.get("kind")
        if schema_kind is not None and not isinstance(schema_kind, str):
            raise BadRequestError(
                "artifact kind must be a string", detail={"kind": schema_kind}
            )
        type_prefix = body.type
        if schema_kind and not type_prefix:
            type_prefix = schema_kind.split(".", 1)[0]
        if (
            schema_kind
            and type_prefix
            and not schema_kind.startswith(f"{type_prefix}.")
        ):
            raise BadRequestError(
                "kind must start with the type prefix",
                detail={"kind": schema_kind, "type": type_prefix},
            )
        if not type_prefix:
            raise BadRequestError("type or kind is required")
        artifact_schema = dict(body.schema)
        if schema_kind:
            artifact_schema["kind"] = schema_kind
        artifact = runtime.store.create_artifact(
            type_=type_prefix,
            name=body.name,
            description=body.description or "",
            schema=artifact_schema,
            owner_user_id=principal.user_id,
            version_author=principal.user_id,
        )
        # New artifact starts at version 1
        resp = ArtifactResponse(
            id=artifact.id,
            type=artifact.type,
            kind=(
                artifact.schema.get("kind")
                if isinstance(artifact.schema, dict)
                else None
            ),
            name=artifact.name,
            description=artifact.description,
            schema=artifact.schema,
            owner_user_id=artifact.owner_user_id,
            visibility=getattr(artifact, "visibility", "private"),
            version=1,
            created_at=artifact.created_at,
            updated_at=artifact.updated_at,
        )
        envelope = Envelope(status="ok", data=resp, request_id=idem.request_id)
        await idem.store_result(envelope)
        return envelope


@router.patch("/artifacts/{artifact_id}", response_model=Envelope, tags=["artifacts"])
async def patch_artifact(
    artifact_id: str, body: ArtifactRequest, principal: AuthContext = Depends(get_user)
):
    runtime = get_runtime()
    current = _get_owned_artifact(runtime, artifact_id, principal)
    if not isinstance(body.schema, dict):
        raise BadRequestError(
            "artifact schema must be an object",
            detail={"provided_type": type(body.schema).__name__},
        )
    schema_kind = body.schema.get("kind")
    artifact_schema = dict(body.schema)
    if schema_kind:
        if not schema_kind.startswith(f"{current.type}."):
            raise BadRequestError(
                "kind must start with the type prefix",
                detail={"kind": schema_kind, "type": current.type},
            )
        artifact_schema["kind"] = schema_kind
    artifact = runtime.store.update_artifact(
        artifact_id,
        schema=artifact_schema,
        description=body.description,
        version_author=principal.user_id,
    )
    if not artifact:
        raise NotFoundError("artifact not found", detail={"artifact_id": artifact_id})
    # Get the new version after update
    current_version = runtime.store.get_artifact_current_version(artifact_id)
    resp = ArtifactResponse(
        id=artifact.id,
        type=artifact.type,
        kind=artifact.schema.get("kind") if isinstance(artifact.schema, dict) else None,
        name=artifact.name,
        description=artifact.description,
        schema=artifact.schema,
        owner_user_id=artifact.owner_user_id,
        visibility=getattr(artifact, "visibility", "private"),
        version=current_version,
        created_at=artifact.created_at,
        updated_at=artifact.updated_at,
    )
    return Envelope(status="ok", data=resp)


@router.post("/config/propose_patch", response_model=Envelope, tags=["config"])
async def propose_patch(
    body: ConfigPatchRequest, principal: AuthContext = Depends(get_admin_user)
):
    runtime = get_runtime()
    # Rate limit configops per SPEC §18: 30 req/hour
    await _enforce_rate_limit(
        runtime,
        f"configops:{principal.user_id}",
        runtime.settings.configops_rate_limit_per_hour,
        3600,
    )
    proposer = "human_admin" if principal.role == "admin" else "user"
    audit = runtime.store.record_config_patch(
        artifact_id=body.artifact_id,
        proposer=proposer,
        patch=body.patch,
        justification=body.justification,
    )
    resp = ConfigPatchAuditResponse(
        id=audit.id,
        artifact_id=audit.artifact_id,
        proposer=audit.proposer,
        justification=audit.justification,
        status=audit.status,
        patch=audit.patch,
        created_at=audit.created_at,
        decided_at=audit.decided_at,
        applied_at=audit.applied_at,
        meta=audit.meta,
    )
    return Envelope(status="ok", data=resp)


@router.get("/config/patches", response_model=Envelope, tags=["config"])
async def list_config_patches(
    status: Optional[str] = None, principal: AuthContext = Depends(get_admin_user)
):
    runtime = get_runtime()
    # Rate limit configops per SPEC §18: 30 req/hour
    await _enforce_rate_limit(
        runtime,
        f"configops:{principal.user_id}",
        runtime.settings.configops_rate_limit_per_hour,
        3600,
    )
    patches = runtime.store.list_config_patches(status)
    items = [
        ConfigPatchAuditResponse(
            id=p.id,
            artifact_id=p.artifact_id,
            proposer=p.proposer,
            justification=p.justification,
            status=p.status,
            patch=p.patch,
            created_at=p.created_at,
            decided_at=p.decided_at,
            applied_at=p.applied_at,
            meta=p.meta,
        )
        for p in patches
    ]
    return Envelope(status="ok", data=ConfigPatchListResponse(items=items))


@router.post("/config/patches/{patch_id}/decide", response_model=Envelope, tags=["config"])
async def decide_config_patch(
    patch_id: int,
    body: ConfigPatchDecisionRequest,
    principal: AuthContext = Depends(get_admin_user),
):
    runtime = get_runtime()
    # Rate limit configops per SPEC §18: 30 req/hour
    await _enforce_rate_limit(
        runtime,
        f"configops:{principal.user_id}",
        runtime.settings.configops_rate_limit_per_hour,
        3600,
    )
    decision = runtime.config_ops.decide_patch(patch_id, body.decision, body.reason)
    resp = ConfigPatchAuditResponse(
        id=decision.id,
        artifact_id=decision.artifact_id,
        proposer=decision.proposer,
        justification=decision.justification,
        status=decision.status,
        patch=decision.patch,
        created_at=decision.created_at,
        decided_at=decision.decided_at,
        applied_at=decision.applied_at,
        meta=decision.meta,
    )
    return Envelope(status="ok", data=resp)


@router.post("/config/patches/{patch_id}/apply", response_model=Envelope, tags=["config"])
async def apply_config_patch(
    patch_id: int, principal: AuthContext = Depends(get_admin_user)
):
    runtime = get_runtime()
    # Rate limit configops per SPEC §18: 30 req/hour
    await _enforce_rate_limit(
        runtime,
        f"configops:{principal.user_id}",
        runtime.settings.configops_rate_limit_per_hour,
        3600,
    )
    result = runtime.config_ops.apply_patch(
        patch_id, approver_user_id=principal.user_id
    )
    patch = result.get("patch")
    resp = ConfigPatchAuditResponse(
        id=patch.id,
        artifact_id=patch.artifact_id,
        proposer=patch.proposer,
        justification=patch.justification,
        status="applied",
        patch=patch.patch,
        created_at=patch.created_at,
        decided_at=patch.decided_at,
        applied_at=patch.applied_at or patch.decided_at or patch.created_at,
        meta=patch.meta,
    )
    # Include warning if status update failed (partial success)
    envelope_data = resp.model_dump()
    if result.get("warning"):
        envelope_data["warning"] = result["warning"]
    return Envelope(status="ok", data=envelope_data)


@router.post("/config/auto_patch", response_model=Envelope, tags=["config"])
async def auto_patch(
    body: AutoPatchRequest, principal: AuthContext = Depends(get_admin_user)
):
    runtime = get_runtime()
    # Rate limit configops per SPEC §18: 30 req/hour
    await _enforce_rate_limit(
        runtime,
        f"configops:{principal.user_id}",
        runtime.settings.configops_rate_limit_per_hour,
        3600,
    )
    audit = runtime.config_ops.auto_generate_patch(
        body.artifact_id, principal.user_id, goal=body.goal
    )
    resp = ConfigPatchAuditResponse(
        id=audit.id,
        artifact_id=audit.artifact_id,
        proposer=audit.proposer,
        justification=audit.justification,
        status=audit.status,
        patch=audit.patch,
        created_at=audit.created_at,
        decided_at=audit.decided_at,
        applied_at=audit.applied_at,
        meta=audit.meta,
    )
    return Envelope(status="ok", data=resp)


@router.get("/config", response_model=Envelope, tags=["config"])
async def get_config(principal: AuthContext = Depends(get_admin_user)):
    """Expose runtime configuration for the admin console."""
    runtime = get_runtime()
    await _enforce_rate_limit(
        runtime,
        f"admin:read:{principal.user_id}",
        runtime.settings.admin_rate_limit_per_minute,
        runtime.settings.admin_rate_limit_window_seconds,
    )

    def _sanitize_dict(data: dict) -> dict:
        """Recursively sanitize sensitive fields in config dictionaries."""
        sensitive_tokens = (
            "secret",
            "token",
            "key",
            "password",
            "credential",
            "api_key",
            "private",
            "auth",
            "dsn",
            "connection_string",
            "bearer",
            "access_token",
            "refresh_token",
            "signing",
            "encryption",
            "smtp_password",
            "oauth_client_secret",
            "jwt_secret",
            "hash_salt",
        )
        sanitized = {}
        for k, v in data.items():
            k_lower = k.lower()
            if any(tok in k_lower for tok in sensitive_tokens):
                sanitized[k] = "[REDACTED]"
            elif isinstance(v, dict):
                sanitized[k] = _sanitize_dict(v)
            else:
                sanitized[k] = v
        return sanitized

    runtime = get_runtime()
    settings = get_settings().model_dump()
    sanitized_settings = _sanitize_dict(settings)
    # Also sanitize runtime_config to prevent data leakage
    runtime_config = (
        runtime.store.get_runtime_config()
        if hasattr(runtime.store, "get_runtime_config")
        else {}
    )
    sanitized_runtime_config = (
        _sanitize_dict(runtime_config)
        if isinstance(runtime_config, dict)
        else runtime_config
    )
    return Envelope(
        status="ok",
        data={
            "runtime_config": sanitized_runtime_config,
            "settings": sanitized_settings,
        },
    )


@router.get("/files/limits", response_model=Envelope, tags=["files"])
async def get_file_limits(principal: AuthContext = Depends(get_user)):
    runtime = get_runtime()
    await _enforce_rate_limit(
        runtime,
        f"read:{principal.user_id}",
        runtime.settings.read_rate_limit_per_minute,
        60,
    )
    return Envelope(
        status="ok",
        data={
            "max_upload_bytes": runtime.settings.max_upload_bytes,
        },
    )


@router.post("/files/upload", response_model=Envelope, tags=["files"])
async def upload_file(
    file: UploadFile = File(...),
    context_id: Optional[str] = Form(None, max_length=255),
    chunk_size: Optional[int] = Form(None, ge=64, le=4000),
    principal: AuthContext = Depends(get_user),
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
):
    import re

    runtime = get_runtime()
    # Rate limit file uploads per SPEC §18: 10 req/min
    await _enforce_rate_limit(
        runtime,
        f"files:upload:{principal.user_id}",
        runtime.settings.files_upload_rate_limit_per_minute,
        60,
    )
    # SPEC §18: Accept Idempotency-Key when provided (optional)
    async with IdempotencyGuard(
        "files:upload", principal.user_id, idempotency_key, require=False
    ) as idem:
        if idem.cached:
            return idem.cached
        # Sanitize filename to prevent path traversal and other attacks
        raw_filename = file.filename or "untitled"
        # Remove path separators and keep only safe characters
        safe_filename = re.sub(r"[^\w\-_\. ]", "_", raw_filename)
        safe_filename = safe_filename.lstrip(".")  # Prevent hidden files
        safe_filename = safe_filename[:255]  # Limit length
        if not safe_filename:
            safe_filename = "untitled"
        dest_dir = (
            FilePath(runtime.settings.shared_fs_root)
            / "users"
            / principal.user_id
            / "files"
        )
        max_bytes = max(1, runtime.settings.max_upload_bytes)
        contents = await file.read(max_bytes + 1)
        if len(contents) > max_bytes:
            raise _http_error("validation_error", "file too large", status_code=413)
        dest_path = safe_join(dest_dir, safe_filename)
        resolved_dest = dest_path.resolve()
        if (
            dest_dir.resolve() not in resolved_dest.parents
            and dest_dir.resolve() != resolved_dest
        ):
            raise _http_error("validation_error", "invalid file path", status_code=400)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Atomic idempotency: record intent BEFORE writing file
        # This prevents duplicate file writes if idempotency storage fails after file write
        # Per SPEC §18: Idempotency must work reliably for POST endpoints
        pending_envelope = Envelope(
            status="pending",
            data={"fs_path": safe_filename, "context_id": context_id},
            request_id=idem.request_id,
        )
        try:
            await idem.store_result(pending_envelope, status="processing")
        except Exception:
            # If we can't record idempotency, log but continue (degraded mode)
            # SECURITY: Don't log exception details that may contain sensitive info
            logger.warning(
                "file_upload_idempotency_pre_record_failed",
                request_id=idem.request_id,
            )

        # Now write the file
        dest_path.write_bytes(contents)
        chunk_count = None
        if context_id:
            try:
                _get_owned_context(runtime, context_id, principal)
                chunk_count = runtime.rag.ingest_file(
                    context_id, str(dest_path), chunk_size=chunk_size
                )
            except Exception:
                # Clean up file on any error (not just ConstraintViolation)
                dest_path.unlink(missing_ok=True)
                raise

        # Update idempotency with final result
        resp = FileUploadResponse(
            fs_path=safe_filename, context_id=context_id, chunk_count=chunk_count
        )
        envelope = Envelope(status="ok", data=resp, request_id=idem.request_id)
        await idem.store_result(envelope)
        return envelope


@router.get("/conversations/{conversation_id}/messages", response_model=Envelope, tags=["conversations"])
async def list_messages(
    conversation_id: str,
    limit: Optional[int] = Query(None, ge=1, description="Maximum messages to return"),
    principal: AuthContext = Depends(get_user),
):
    runtime = get_runtime()
    await _enforce_rate_limit(
        runtime,
        f"read:{principal.user_id}",
        runtime.settings.read_rate_limit_per_minute,
        60,
    )
    paging = _get_pagination_settings(runtime)
    resolved_limit = min(limit or paging["default_page_size"], paging["max_page_size"])
    _get_owned_conversation(runtime, conversation_id, principal)
    msgs = runtime.store.list_messages(
        conversation_id, limit=resolved_limit, user_id=principal.user_id
    )
    payload = [
        {
            "id": m.id,
            "role": m.role,
            "sender": m.sender,
            "content": m.content,
            "content_struct": m.content_struct,
            "seq": m.seq,
            "created_at": m.created_at,
            "meta": m.meta,
        }
        for m in msgs
    ]
    return Envelope(
        status="ok",
        data=ConversationMessagesResponse(
            conversation_id=conversation_id, messages=payload
        ),
    )


@router.post("/conversations", response_model=Envelope, status_code=201, tags=["conversations"])
async def create_conversation(
    body: CreateConversationRequest,
    principal: AuthContext = Depends(get_user),
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
):
    """Create a new conversation.

    Allows creating an empty conversation that can later be populated
    with messages via the chat endpoint.
    """
    runtime = get_runtime()
    async with IdempotencyGuard(
        "conversations:create", principal.user_id, idempotency_key, require=False
    ) as idem:
        if idem.cached:
            return idem.cached
        await _enforce_rate_limit(
            runtime,
            f"write:{principal.user_id}",
            runtime.settings.write_rate_limit_per_minute,
            60,
        )
        # Validate context_id if provided
        if body.context_id:
            _get_owned_context(runtime, body.context_id, principal)
        conversation = runtime.store.create_conversation(
            user_id=principal.user_id,
            title=body.title,
            active_context_id=body.context_id,
        )
        response = Envelope(
            status="ok",
            data=CreateConversationResponse(
                id=conversation.id,
                created_at=conversation.created_at,
                updated_at=conversation.updated_at,
                title=conversation.title,
                status=conversation.status,
                active_context_id=conversation.active_context_id,
            ),
        )
        idem.result = response
        return response


@router.get("/conversations", response_model=Envelope, tags=["conversations"])
async def list_conversations(
    limit: Optional[int] = Query(None, ge=1, description="Maximum conversations to return"),
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    principal: AuthContext = Depends(get_user),
):
    runtime = get_runtime()
    await _enforce_rate_limit(
        runtime,
        f"read:{principal.user_id}",
        runtime.settings.read_rate_limit_per_minute,
        60,
    )
    paging = _get_pagination_settings(runtime)
    resolved_limit = min(limit or paging["default_conversations_limit"], paging["max_page_size"])
    # Fetch one extra to determine if more items exist
    convs = runtime.store.list_conversations(
        principal.user_id, limit=resolved_limit + 1, offset=(page - 1) * resolved_limit
    )
    has_next = len(convs) > resolved_limit
    if has_next:
        convs = convs[:resolved_limit]
    items = [
        ConversationSummary(
            id=c.id,
            created_at=c.created_at,
            updated_at=c.updated_at,
            title=c.title,
            status=c.status,
            active_context_id=c.active_context_id,
        )
        for c in convs
    ]
    return Envelope(
        status="ok",
        data=ConversationListResponse(
            items=items,
            has_next=has_next,
            next_page=page + 1 if has_next else None,
        ),
    )


@router.get("/conversations/{conversation_id}", response_model=Envelope, tags=["conversations"])
async def get_conversation(
    conversation_id: str = Path(..., max_length=255, description="Conversation identifier"),
    principal: AuthContext = Depends(get_user),
):
    """Get a single conversation by ID.

    Returns conversation details including title, status, and metadata.
    Only the conversation owner can access it.
    """
    runtime = get_runtime()
    await _enforce_rate_limit(
        runtime,
        f"read:{principal.user_id}",
        runtime.settings.read_rate_limit_per_minute,
        60,
    )
    conversation = _get_owned_conversation(runtime, conversation_id, principal)
    return Envelope(
        status="ok",
        data=ConversationSummary(
            id=conversation.id,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
            title=conversation.title,
            status=conversation.status,
            active_context_id=conversation.active_context_id,
        ),
    )


@router.post("/contexts", response_model=Envelope, status_code=201, tags=["knowledge"])
async def create_context(
    body: KnowledgeContextRequest,
    principal: AuthContext = Depends(get_user),
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
):
    runtime = get_runtime()
    # SPEC §18: Accept Idempotency-Key when provided (optional)
    async with IdempotencyGuard(
        "contexts:create", principal.user_id, idempotency_key, require=False
    ) as idem:
        if idem.cached:
            return idem.cached
        ctx_meta = {"embedding_model_id": runtime.rag.embedding_model_id}
        ctx = runtime.store.upsert_context(
            owner_user_id=principal.user_id,
            name=body.name,
            description=body.description,
            meta=ctx_meta,
        )
        if body.text:
            runtime.rag.ingest_text(ctx.id, body.text, chunk_size=body.chunk_size)
        envelope = Envelope(
            status="ok",
            data=KnowledgeContextResponse(
                id=ctx.id,
                name=ctx.name,
                description=ctx.description,
                created_at=ctx.created_at,
                updated_at=ctx.updated_at,
                owner_user_id=ctx.owner_user_id,
                meta=ctx.meta,
            ),
            request_id=idem.request_id,
        )
        await idem.store_result(envelope)
        return envelope


@router.get("/contexts", response_model=Envelope, tags=["knowledge"])
async def list_contexts(
    limit: Optional[int] = Query(None, ge=1, description="Maximum contexts to return"),
    principal: AuthContext = Depends(get_user),
):
    runtime = get_runtime()
    await _enforce_rate_limit(
        runtime,
        f"read:{principal.user_id}",
        runtime.settings.read_rate_limit_per_minute,
        60,
    )
    paging = _get_pagination_settings(runtime)
    resolved_limit = min(limit or paging["default_page_size"], paging["max_page_size"])
    contexts = runtime.store.list_contexts(owner_user_id=principal.user_id, limit=resolved_limit)
    items = [
        KnowledgeContextResponse(
            id=c.id,
            name=c.name,
            description=c.description,
            created_at=c.created_at,
            updated_at=c.updated_at,
            owner_user_id=c.owner_user_id,
            meta=c.meta,
        )
        for c in contexts
    ]
    return Envelope(status="ok", data=KnowledgeContextListResponse(items=items))


@router.get("/contexts/{context_id}/chunks", response_model=Envelope, tags=["knowledge"])
async def list_chunks(
    context_id: str = Path(..., max_length=255, description="Knowledge context ID"),
    limit: Optional[int] = Query(None, ge=1, description="Maximum chunks to return"),
    principal: AuthContext = Depends(get_user),
):
    runtime = get_runtime()
    await _enforce_rate_limit(
        runtime,
        f"read:{principal.user_id}",
        runtime.settings.read_rate_limit_per_minute,
        60,
    )
    paging = _get_pagination_settings(runtime)
    resolved_limit = min(limit or paging["default_page_size"], paging["max_page_size"])
    _get_owned_context(runtime, context_id, principal)
    chunks = runtime.store.list_chunks(context_id, owner_user_id=principal.user_id, limit=resolved_limit)
    for ch in chunks:
        if ch.id is None:
            raise _http_error(
                "server_error", "chunk id missing for context", status_code=500
            )
    data = [
        KnowledgeChunkResponse(
            id=int(ch.id),
            context_id=ch.context_id,
            fs_path=ch.fs_path,
            content=ch.content,
            chunk_index=ch.chunk_index,
        )
        for ch in chunks
    ]
    return Envelope(status="ok", data=KnowledgeChunkListResponse(items=data))


@router.post("/contexts/{context_id}/sources", response_model=Envelope, status_code=201, tags=["knowledge"])
async def add_context_source(
    context_id: str = Path(..., max_length=255, description="Knowledge context ID"),
    body: ContextSourceRequest = ...,
    principal: AuthContext = Depends(get_user),
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
):
    """Add a source path to a knowledge context for indexing.

    The source path will be indexed and its content split into chunks
    that can be retrieved during RAG queries.
    """
    runtime = get_runtime()

    async with IdempotencyGuard(
        "context_sources:create", principal.user_id, idempotency_key, require=False
    ) as idem:
        if idem.cached:
            return idem.cached

        # Verify context ownership
        _get_owned_context(runtime, context_id, principal)

        # Add the source
        source = runtime.store.add_context_source(
            context_id=context_id,
            fs_path=body.fs_path,
            recursive=body.recursive,
        )

        # Trigger indexing via RAG service
        try:
            runtime.rag.ingest_path(
                context_id=context_id,
                fs_path=body.fs_path,
                recursive=body.recursive,
            )
        except Exception as exc:
            # Clean up the source record since ingestion failed
            logger.warning(
                "context_source_ingest_failed",
                context_id=context_id,
                fs_path=body.fs_path,
                error=str(exc),
            )
            try:
                runtime.store.delete_context_source(source.id)
            except Exception:
                pass  # Best effort cleanup
            raise _http_error(
                "ingest_failed",
                f"Failed to index source: {exc}",
                status_code=500,
            )

        envelope = Envelope(
            status="ok",
            data=ContextSourceResponse(
                id=source.id,
                context_id=source.context_id,
                fs_path=source.fs_path,
                recursive=source.recursive,
                meta=source.meta,
            ),
            request_id=idem.request_id,
        )
        await idem.store_result(envelope)
        return envelope


@router.get("/contexts/{context_id}/sources", response_model=Envelope, tags=["knowledge"])
async def list_context_sources(
    context_id: str = Path(..., max_length=255, description="Knowledge context ID"),
    principal: AuthContext = Depends(get_user),
):
    """List all source paths for a knowledge context."""
    runtime = get_runtime()
    await _enforce_rate_limit(
        runtime,
        f"read:{principal.user_id}",
        runtime.settings.read_rate_limit_per_minute,
        60,
    )

    # Verify context ownership
    _get_owned_context(runtime, context_id, principal)

    sources = runtime.store.list_context_sources(context_id)
    items = [
        ContextSourceResponse(
            id=s.id,
            context_id=s.context_id,
            fs_path=s.fs_path,
            recursive=s.recursive,
            meta=s.meta,
        )
        for s in sources
    ]
    return Envelope(status="ok", data=ContextSourceListResponse(items=items))


@router.post("/voice/transcribe", response_model=Envelope, tags=["voice"])
async def transcribe_voice(
    file: UploadFile = File(...), principal: AuthContext = Depends(get_user)
):
    runtime = get_runtime()
    # Rate limit voice transcription (resource-intensive)
    await _enforce_rate_limit(
        runtime,
        f"voice:transcribe:{principal.user_id}",
        runtime.settings.chat_rate_limit_per_minute,  # Use chat rate limit as default
        60,
    )
    # Limit audio file size to 10MB
    max_audio_bytes = 10 * 1024 * 1024
    audio_bytes = await file.read(max_audio_bytes + 1)
    if len(audio_bytes) > max_audio_bytes:
        raise _http_error(
            "validation_error", "audio file too large (max 10MB)", status_code=413
        )
    result = await runtime.voice.transcribe(audio_bytes, user_id=principal.user_id)
    return Envelope(status="ok", data=VoiceTranscriptionResponse(**result))


@router.post("/voice/synthesize", response_model=Envelope, tags=["voice"])
async def synthesize_voice(
    body: VoiceSynthesisRequest, principal: AuthContext = Depends(get_user)
):
    runtime = get_runtime()
    # Rate limit voice synthesis (resource-intensive)
    await _enforce_rate_limit(
        runtime,
        f"voice:synthesize:{principal.user_id}",
        runtime.settings.chat_rate_limit_per_minute,  # Use chat rate limit as default
        60,
    )
    # Limit text length to prevent resource exhaustion
    if len(body.text) > 5000:
        raise _http_error(
            "validation_error",
            "text too long for synthesis (max 5000 chars)",
            status_code=400,
        )
    audio = await runtime.voice.synthesize(
        body.text, user_id=principal.user_id, voice=body.voice, speed=body.speed
    )
    return Envelope(status="ok", data=VoiceSynthesisResponse(**audio))


@router.websocket("/chat/stream")
async def websocket_chat(ws: WebSocket):
    """Handle WebSocket chat connections for streaming responses."""
    runtime = get_runtime()
    await ws.accept()
    user_id: Optional[str] = None
    idempotency_key: Optional[str] = None
    # Generate request_id immediately to ensure traceability even on early errors
    request_id: str = str(uuid4())
    convo_id: Optional[str] = None
    workflow_slot_acquired: bool = False
    try:
        init = await ws.receive_json()
        idempotency_key = init.get("idempotency_key")
        # Use client-provided request_id if available, otherwise keep generated one
        request_id = init.get("request_id") or request_id
        session_id = init.get("session_id")
        access_token = init.get("access_token")
        auth_ctx = await runtime.auth.authenticate(
            f"Bearer {access_token}" if access_token else None,
            session_id,
        )
        if not auth_ctx:
            await ws.close(code=4401)
            return
        user_id = auth_ctx.user_id

        # Get user's plan tier for per-plan rate limits (SPEC §18)
        user = runtime.store.get_user(user_id)
        plan_tier = user.plan_tier if user else "free"

        # SPEC §18: Accept Idempotency-Key when provided (optional)
        request_id, cached = await _resolve_idempotency(
            "chat:ws", user_id, idempotency_key, require=False, request_id=request_id
        )
        if cached:
            await ws.send_json(cached.model_dump())
            return

        # SPEC §18: Per-plan adjustable rate limits
        await _enforce_rate_limit_per_plan(
            runtime,
            f"chat:{user_id}",
            runtime.settings.chat_rate_limit_per_minute,
            runtime.settings.chat_rate_limit_window_seconds,
            plan_tier,
        )

        # SPEC §18: Concurrency cap - max 3 concurrent workflows per user
        # Check if we can acquire a slot; if not, close with 409-equivalent code
        if runtime.cache:
            acquired, _ = await runtime.cache.acquire_concurrency_slot(
                "workflow",
                user_id,
                runtime.settings.max_concurrent_workflows,
            )
            if not acquired:
                await ws.send_json({
                    "event": "error",
                    "data": {
                        "error": "busy",
                        "message": f"concurrent workflow limit ({runtime.settings.max_concurrent_workflows}) exceeded",
                    },
                    "request_id": request_id,
                })
                await ws.close(code=4409)  # Custom code for "busy"
                return
            workflow_slot_acquired = True

        convo_id = init.get("conversation_id")
        if convo_id:
            conversation = _get_owned_conversation(runtime, convo_id, auth_ctx)
        else:
            # Generate title from first message for new conversations
            user_message = init.get("message", "")
            auto_title = _generate_conversation_title(user_message)
            conversation = runtime.store.create_conversation(
                user_id=user_id, title=auto_title
            )
            convo_id = conversation.id
        context_id = init.get("context_id") or conversation.active_context_id
        if context_id:
            _get_owned_context(runtime, context_id, auth_ctx)
        runtime.store.append_message(
            convo_id, sender="user", role="user", content=init.get("message", "")
        )

        # SPEC §18: Check if streaming is requested (default True for WebSocket)
        stream_enabled = init.get("stream", True)

        if stream_enabled:
            # Streaming mode: emit token, trace, message_done, error events
            cancel_event = asyncio.Event()
            # Register cancel event so POST /chat/cancel can cancel this request
            await _register_cancel_event(request_id, cancel_event)
            full_content = ""
            orchestration_dict: dict[str, Any] = {}

            # Concurrent task to listen for cancel requests while streaming
            async def listen_for_cancel():
                try:
                    while not cancel_event.is_set():
                        try:
                            msg = await asyncio.wait_for(ws.receive_json(), timeout=0.5)
                            if msg.get("action") == "cancel":
                                cancel_event.set()
                                return
                            elif msg.get("action") == "ping":
                                await ws.send_json({"event": "pong", "data": None})
                        except asyncio.TimeoutError:
                            continue
                        except WebSocketDisconnect:
                            cancel_event.set()
                            return
                except Exception:
                    pass  # Listener task should exit silently on errors

            cancel_listener = asyncio.create_task(listen_for_cancel())

            try:
                async for event in runtime.workflow.run_streaming(
                    init.get("workflow_id"),
                    convo_id,
                    init.get("message", ""),
                    context_id,
                    user_id,
                    tenant_id=auth_ctx.tenant_id,
                    cancel_event=cancel_event,
                ):
                    event_type = event.get("event")
                    event_data = event.get("data")

                    # SPEC §18: WebSockets wrap as {"event": "token", "data": "..."}
                    await ws.send_json({"event": event_type, "data": event_data})

                    if event_type == "token":
                        full_content += event_data if isinstance(event_data, str) else ""
                    elif event_type == "message_done":
                        orchestration_dict = event_data if isinstance(event_data, dict) else {}
                    elif event_type == "error":
                        # Error already sent, close connection
                        cancel_listener.cancel()
                        await ws.close(code=1011)
                        return
                    elif event_type == "cancel_ack":
                        cancel_listener.cancel()
                        await ws.close(code=1000)
                        return

            except WebSocketDisconnect:
                cancel_event.set()
                cancel_listener.cancel()
                await _unregister_cancel_event(request_id)
                return
            finally:
                cancel_listener.cancel()
                await _unregister_cancel_event(request_id)

            # Save assistant message after streaming completes
            adapter_names = _stringify_adapters(orchestration_dict.get("adapters", []))
            assistant_content = orchestration_dict.get("content", full_content or "No response generated.")
            assistant_content_struct = normalize_content_struct(
                orchestration_dict.get("content_struct"),
                assistant_content,
            )
            assistant_msg = runtime.store.append_message(
                convo_id,
                sender="assistant",
                role="assistant",
                content=assistant_content,
                content_struct=assistant_content_struct,
                meta={
                    "adapters": orchestration_dict.get("adapters", []),
                    "adapter_gates": orchestration_dict.get("adapter_gates", []),
                    "routing_trace": orchestration_dict.get("routing_trace", []),
                    "workflow_trace": orchestration_dict.get("workflow_trace", []),
                    "usage": orchestration_dict.get("usage", {}),
                },
            )
            # Store idempotency result for completed streaming
            envelope = Envelope(
                status="ok",
                data=ChatResponse(
                    message_id=assistant_msg.id,
                    conversation_id=convo_id,
                    content=assistant_msg.content,
                    content_struct=assistant_msg.content_struct,
                    workflow_id=init.get("workflow_id"),
                    adapters=adapter_names,
                    adapter_gates=orchestration_dict.get("adapter_gates", []),
                    usage=orchestration_dict.get("usage", {}),
                    context_snippets=orchestration_dict.get("context_snippets", []),
                    routing_trace=orchestration_dict.get("routing_trace", []),
                    workflow_trace=orchestration_dict.get("workflow_trace", []),
                ).model_dump(),
                request_id=request_id,
            )
            await _store_idempotency_result("chat:ws", user_id, idempotency_key, envelope)

            # Send final event with message_id and conversation_id to client
            await ws.send_json({
                "event": "streaming_complete",
                "data": {
                    "message_id": assistant_msg.id,
                    "conversation_id": convo_id,
                    "adapters": adapter_names,
                    "adapter_gates": orchestration_dict.get("adapter_gates", []),
                    "usage": orchestration_dict.get("usage", {}),
                    "context_snippets": orchestration_dict.get("context_snippets", []),
                    "routing_trace": orchestration_dict.get("routing_trace", []),
                    "workflow_trace": orchestration_dict.get("workflow_trace", []),
                }
            })

        else:
            # Non-streaming mode: single response (legacy behavior)
            orchestration = await runtime.workflow.run(
                init.get("workflow_id"),
                convo_id,
                init.get("message", ""),
                context_id,
                user_id,
                tenant_id=auth_ctx.tenant_id,
            )
            orchestration_dict = (
                orchestration if isinstance(orchestration, dict) else {}
            )
            adapter_names = _stringify_adapters(orchestration_dict.get("adapters", []))
            assistant_content_struct = normalize_content_struct(
                orchestration_dict.get("content_struct"),
                orchestration_dict.get("content"),
            )
            assistant_content = orchestration_dict.get("content", "No response generated.")
            assistant_msg = runtime.store.append_message(
                convo_id,
                sender="assistant",
                role="assistant",
                content=assistant_content,
                content_struct=assistant_content_struct,
                meta={
                    "adapters": orchestration_dict.get("adapters", []),
                    "adapter_gates": orchestration_dict.get("adapter_gates", []),
                    "routing_trace": orchestration_dict.get("routing_trace", []),
                    "workflow_trace": orchestration_dict.get("workflow_trace", []),
                    "usage": orchestration_dict.get("usage", {}),
                },
            )
            envelope = Envelope(
                status="ok",
                data=ChatResponse(
                    message_id=assistant_msg.id,
                    conversation_id=convo_id,
                    content=assistant_msg.content,
                    content_struct=assistant_msg.content_struct,
                    workflow_id=init.get("workflow_id"),
                    adapters=adapter_names,
                    adapter_gates=orchestration_dict.get("adapter_gates", []),
                    usage=orchestration_dict.get("usage", {}),
                    context_snippets=orchestration_dict.get("context_snippets", []),
                    routing_trace=orchestration_dict.get("routing_trace", []),
                    workflow_trace=orchestration_dict.get("workflow_trace", []),
                ).model_dump(),
                request_id=request_id,
            )
            await _store_idempotency_result("chat:ws", user_id, idempotency_key, envelope)
            await ws.send_json(envelope.model_dump())
    except HTTPException as exc:
        detail = exc.detail if isinstance(exc.detail, dict) else None
        error_payload = (
            detail.get("error")
            if detail and "error" in detail
            else {"code": "server_error", "message": str(exc.detail)}
        )
        error_env = Envelope(
            status="error", error=error_payload, request_id=request_id
        )
        if user_id:
            await _store_idempotency_result(
                "chat:ws", user_id, idempotency_key, error_env, status="failed"
            )
        await ws.send_json(error_env.model_dump())
        status_code = getattr(exc, "status_code", 500)
        await ws.close(code=4429 if status_code == 429 else 1011)
    except WebSocketDisconnect:
        return
    except json.JSONDecodeError:
        # Handle invalid JSON in initial message
        logger.warning(
            "websocket_invalid_json",
            request_id=request_id,
        )
        error_env = Envelope(
            status="error",
            error={"code": "invalid_json", "message": "Invalid JSON in request"},
            request_id=request_id,
        )
        try:
            await ws.send_json(error_env.model_dump())
        except Exception:
            pass
        await ws.close(code=1003)
    except Exception as exc:
        # SECURITY: Use logger.error instead of logger.exception to avoid
        # exposing full stack traces that may reveal implementation details
        logger.error(
            "unhandled_websocket_error",
            user_id=user_id,
            conversation_id=convo_id,
            request_id=request_id,
            error_type=type(exc).__name__,
        )
        error_env = Envelope(
            status="error",
            error={"code": "server_error", "message": "An internal error occurred"},
            request_id=request_id,
        )
        if user_id:
            await _store_idempotency_result(
                "chat:ws", user_id, idempotency_key, error_env, status="failed"
            )
        # Send error envelope to client before closing
        try:
            await ws.send_json(error_env.model_dump())
        except Exception:
            pass  # Connection may already be closed
        await ws.close(code=1011)
    finally:
        # Always release workflow slot, even on error
        if workflow_slot_acquired and user_id and runtime.cache:
            await runtime.cache.release_concurrency_slot("workflow", user_id)
