from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import math
import mimetypes
import os
import shutil
import stat
import time
from contextlib import suppress
from datetime import datetime, timedelta, timezone
from pathlib import Path as FilePath
from typing import Any, Optional
from urllib.parse import quote
from uuid import uuid4

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Body,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    Path,
    Query,
    Request,
    Response,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import JSONResponse, StreamingResponse

from liminallm.api import chat_turn, idempotency, mcp
from liminallm.api.errors import http_error
from liminallm.api.limits import (
    client_ip,
    enforce,
    enforce_per_plan,
    plan_upload_limit,
    rate_limit,
)
from liminallm.api.schemas import (
    AdminCreateUserRequest,
    AdminCreateUserResponse,
    AdminInspectionResponse,
    ApiKeyCreatedResponse,
    ApiKeyCreateRequest,
    ApiKeyListResponse,
    ApiKeyRevokeResponse,
    ApiKeySummary,
    ArtifactListResponse,
    ArtifactPatchRequest,
    ArtifactRequest,
    ArtifactResponse,
    ArtifactVersionListResponse,
    ArtifactVersionResponse,
    AuthResponse,
    AutoPatchRequest,
    ChatCancelRequest,
    ChatCancelResponse,
    ChatRequest,
    ConfigPatchAuditResponse,
    ConfigPatchDecisionRequest,
    ConfigPatchListResponse,
    ConfigPatchRequest,
    ContextSourceListResponse,
    ContextSourceRequest,
    ContextSourceResponse,
    ConversationListResponse,
    ConversationMessagesResponse,
    ConversationShareRequest,
    ConversationSummary,
    ConversationUpdateRequest,
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
    KnowledgeContextUpdateRequest,
    LoginRequest,
    MFADisableRequest,
    MFARequest,
    MFAStatusResponse,
    MFAVerifyRequest,
    NoteCreateRequest,
    NoteFromFileRequest,
    NoteSearchRequest,
    NoteUpdateRequest,
    NoteWitnessRequest,
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
from liminallm.config import (
    get_settings,
    managed_settings_schema,
    secret_setting_names,
    validate_managed_settings,
)
from liminallm.logging import get_logger
from liminallm.service import (
    admission,
    cancellation,
    ingest_queue,
    json_patch,
    tenancy,
)
from liminallm.service import extract as extract_service
from liminallm.service import notes as notes_service
from liminallm.service.archive import (
    ArchiveExtractionError,
    archive_stem,
    extract_archive_sandboxed,
    is_archive_filename,
)
from liminallm.service.attachments import (
    classify_attachment,
    ensure_conversation_context,
    find_conversation_context_id,
    generation_key,
    generation_lock,
    is_auto_context,
    list_attachments,
    record_attachment,
    store_generation,
)
from liminallm.service.auth import AuthContext
from liminallm.service.errors import BadRequestError, NotFoundError, ServiceError
from liminallm.service.fs import (
    PathAuthorityError,
    PathLockTimeout,
    PathTraversalError,
    authorize_path,
    file_digest,
    generate_signed_url,
    is_internal_path,
    namespace_key,
    path_lock,
    safe_join,
    validate_signed_url,
)
from liminallm.service.runtime import MODEL_AFFECTING_SETTINGS, get_runtime
from liminallm.service.sandbox import SandboxError
from liminallm.service.upload_policy import ALLOWED_UPLOAD_EXTENSIONS
from liminallm.storage.cursors import (
    encode_artifact_cursor,
    encode_index_cursor,
    encode_time_id_cursor,
)
from liminallm.storage.errors import ConstraintViolation
from liminallm.storage.models import Conversation, KnowledgeContext, Session

logger = get_logger(__name__)

#: Where archives are extracted before they are published. Outside every
#: user's path authority on purpose: a hidden sibling of the destination
#: would still be walked by a context source covering `files/`.
ARCHIVE_STAGING_DIRNAME = ".archive-staging"

# ALLOWED_UPLOAD_EXTENSIONS (SPEC §17) is imported above from
# service.upload_policy so the tool sandbox can apply the same policy; it stays
# importable from this module because callers and tests reference it here.

router = APIRouter(prefix="/v1")

async def get_user(
    request: Request,
    authorization: Optional[str] = Header(None),
    session_id: Optional[str] = Header(None, convert_underscores=False),
) -> AuthContext:
    runtime = get_runtime()
    ctx = await runtime.auth.authenticate(
        authorization,
        session_id,
        tenant_hint=tenancy.tenant_of(request.headers, runtime.settings),
    )
    if not ctx:
        raise http_error("unauthorized", "invalid session", status_code=401)
    return ctx


async def get_admin_user(
    request: Request,
    authorization: Optional[str] = Header(None),
    session_id: Optional[str] = Header(None, convert_underscores=False),
) -> AuthContext:
    runtime = get_runtime()
    ctx = await runtime.auth.authenticate(
        authorization,
        session_id,
        tenant_hint=tenancy.tenant_of(request.headers, runtime.settings),
        required_role="admin",
    )
    if not ctx:
        raise http_error("forbidden", "admin access required", status_code=403)
    return ctx


async def get_user_or_api_key(
    request: Request,
    authorization: Optional[str] = Header(None),
    session_id: Optional[str] = Header(None, convert_underscores=False),
) -> AuthContext:
    """Principal for the agent surfaces (/v1/responses, /v1/mcp): an API key,
    or any session auth.

    Only this dependency reads API keys. Native routes authenticate through
    get_user, which cannot — so a leaked key can drive the agent surfaces and
    nothing else, and in particular cannot mint or revoke keys.
    """
    runtime = get_runtime()
    ctx = runtime.auth.authenticate_api_key(
        authorization,
        tenant_hint=tenancy.tenant_of(request.headers, runtime.settings),
    )
    if ctx:
        return ctx
    return await get_user(request, authorization, session_id)


def _get_owned_conversation(
    runtime, conversation_id: str, principal: AuthContext
) -> Conversation:
    conversation = runtime.store.get_conversation(
        conversation_id, user_id=principal.user_id
    )
    if not conversation:
        raise http_error("not_found", "conversation not found", status_code=404)
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
    """A context this caller may name, or an error.

    A conversation's implicit index is never one of them. Being one is
    load-bearing — the stale-generation sweep skips these contexts, and
    retrieval from them is filtered to what the conversation's records still
    name — and both of those key on the conversation, not on the context. So
    a caller that can *name* one gets an index nothing scopes: measured, a
    second chat read the first chat's attachment by naming its id, which
    §19.5 scopes to the chat that received it.

    Reported as absent before ownership is even considered, so the answer is
    the same for every caller and the id says nothing. Enforcement does not
    depend on the id being hard to obtain; it depends on nothing accepting
    it.
    """
    ctx = runtime.store.get_context(context_id)
    if not ctx or is_auto_context(ctx):
        raise http_error("not_found", "context not found", status_code=404)
    if ctx.owner_user_id != principal.user_id:
        raise http_error(
            "forbidden", "context is owned by another user", status_code=403
        )
    return ctx


def _get_private_artifact(runtime, artifact_id: str, principal: AuthContext):
    """An artifact this caller may *mutate*, or an error.

    Not `_get_owned_artifact`. That one is a read capability: it lets an admin
    through to another user's artifact and to ownerless system artifacts,
    which is right for viewing and wrong as a mutation rule. Used for PATCH,
    it made `/artifacts/{id}` a way to edit a global system workflow directly
    — the change ConfigOps exists to review — as an accidental consequence of
    being an admin on an ordinary user route.

    SPEC §12.3 scopes user CRUD to *private* artifacts. Visibility is part of
    the rule, not just ownership: publishing an artifact binds it into other
    people's work, so retiring or editing it stops being a private decision.
    Shared, global and system artifacts change through ConfigOps.

    Reported as absent to anyone who could not already read it, so the id
    says nothing about what exists. The owner of a *published* artifact is the
    one exception: they can read it, so "not found" would only be confusing
    where the real answer is that publishing moved it out of their sole
    control.
    """
    artifact = runtime.store.get_private_artifact(
        artifact_id, owner_user_id=principal.user_id
    )
    if artifact is not None:
        return artifact

    existing = runtime.store.get_artifact(artifact_id)
    if existing is not None and existing.owner_user_id == principal.user_id:
        raise http_error(
            "forbidden",
            f"this artifact is {existing.visibility}; published artifacts are "
            "changed and retired through config ops, not here",
            status_code=403,
        )
    raise http_error("not_found", "artifact not found", status_code=404)


def _get_owned_artifact(runtime, artifact_id: str, principal: AuthContext):
    artifact = runtime.store.get_artifact(artifact_id)
    if not artifact:
        raise http_error("not_found", "artifact not found", status_code=404)
    if artifact.owner_user_id and artifact.owner_user_id != principal.user_id:
        if principal.role != "admin":
            raise http_error(
                "forbidden", "artifact is owned by another user", status_code=403
            )
    if not artifact.owner_user_id and principal.role != "admin":
        raise http_error(
            "forbidden", "artifact access requires admin privileges", status_code=403
        )
    return artifact


def _get_pagination_settings(runtime) -> dict:
    """Get pagination settings from database-managed system settings.

    Returns dict with: default_page_size, max_page_size, default_conversations_limit
    """
    settings = runtime.settings
    return {
        "default_page_size": settings.default_page_size,
        "max_page_size": settings.max_page_size,
        # Keep conversation pagination consistent with general defaults (SPEC pagination guidance)
        "default_conversations_limit": settings.default_conversations_limit,
    }


def _apply_session_cookies(
    response: Response, session: Session, tokens: dict, *, refresh_ttl_minutes: Optional[int] = None
) -> None:
    # Cookie expiry must be UTC-aware: a naive stamp is UTC by convention here,
    # and an aware one may carry some other zone.
    expires_at = session.expires_at
    expires_at = (
        expires_at.replace(tzinfo=timezone.utc)
        if expires_at.tzinfo is None
        else expires_at.astimezone(timezone.utc)
    )
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
        # Get refresh TTL from database settings if not provided
        if refresh_ttl_minutes is None:
            runtime = get_runtime()
            refresh_ttl_minutes = runtime.settings.refresh_token_ttl_minutes
        response.set_cookie(
            "refresh_token",
            refresh_token,
            httponly=True,
            secure=True,
            samesite="lax",
            max_age=refresh_ttl_minutes * 60,
            path="/",
        )
    csrf_token = tokens.get("csrf_token")
    if csrf_token:
        response.set_cookie(
            "csrf_token",
            csrf_token,
            httponly=False,
            secure=True,
            samesite="lax",
            expires=expires_at,
            path="/",
        )
        response.headers.setdefault("X-CSRF-Token", csrf_token)


@router.post("/auth/signup", response_model=Envelope, status_code=201, tags=["auth"])
async def signup(body: SignupRequest, response: Response, request: Request):
    """Create a new user account.

    Registers a new user with email and password credentials. Returns session
    tokens and sets authentication cookies. Rate limited per email address.

    Raises:
        403: If signup is disabled in settings
        429: If rate limit exceeded for this email
    """
    runtime = get_runtime()
    if not runtime.settings.allow_signup:
        raise http_error("forbidden", "signup disabled", status_code=403)
    await rate_limit(runtime, "signup", body.email.lower())
    # Per CLAUDE.md, tenant_id is never taken from the request. It is the tenant
    # serving the site this signup arrived at (service/tenancy.py).
    user, session, tokens = await runtime.auth.signup(
        email=body.email,
        password=body.password,
        handle=body.handle,
        tenant_id=tenancy.tenant_of(request.headers, runtime.settings),
    )
    logger.info("user_signup_completed", user_id=user.id, email=body.email)
    _apply_session_cookies(
        response,
        session,
        tokens,
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
            csrf_token=tokens.get("csrf_token"),
            role=user.role,
            tenant_id=user.tenant_id,
        ),
    )


@router.post("/auth/login", response_model=Envelope, tags=["auth"])
async def login(body: LoginRequest, request: Request, response: Response):
    """Authenticate user with email and password.

    Validates credentials and returns session tokens with authentication cookies.
    Supports optional MFA verification via mfa_code parameter.

    Raises:
        401: If credentials are invalid
        429: If rate limit exceeded for this email
    """
    runtime = get_runtime()
    await rate_limit(runtime, "login", body.email.lower())
    # Bug fix: Pass user_agent and ip_addr for session metadata
    user_agent = request.headers.get("user-agent")
    ip_addr = request.client.host if request.client else None
    # Per CLAUDE.md the tenant is never taken from the request body. It is the
    # tenant serving the site this login arrived at, and it must match the one
    # on the user's record — an account from another site cannot sign in here.
    user, session, tokens = await runtime.auth.login(
        email=body.email,
        password=body.password,
        mfa_code=body.mfa_code,
        tenant_id=tenancy.tenant_of(request.headers, runtime.settings),
        user_agent=user_agent,
        ip_addr=ip_addr,
        device_type=body.device_type,
    )
    if not user or not session:
        logger.warning("login_failed", email=body.email, ip=ip_addr)
        raise http_error("unauthorized", "invalid credentials", status_code=401)
    logger.info("user_login_completed", user_id=user.id, ip=ip_addr)
    _apply_session_cookies(
        response,
        session,
        tokens,
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
            csrf_token=tokens.get("csrf_token"),
            role=user.role,
            tenant_id=user.tenant_id,
        ),
    )


@router.post("/auth/oauth/{provider}/start", response_model=Envelope, tags=["auth"])
async def oauth_start(
    request: Request,
    provider: str = Path(..., description="OAuth provider (google, github, etc.)"),
    body: OAuthStartRequest = ...,
):
    """Start OAuth authentication flow.

    Returns authorization URL for the specified OAuth provider.
    Client should redirect user to this URL to complete authentication.
    """
    runtime = get_runtime()
    # Rate limit OAuth start per client to prevent state token exhaustion
    await enforce(
        runtime,
        f"oauth:start:{provider}:{client_ip(request)}",
        limit=20,
        window_seconds=60,
    )
    # The tenant is bound into the server-created state here, so the callback
    # lands the user on the site they started from and nowhere else.
    try:
        start = await runtime.auth.start_oauth(
            provider,
            redirect_uri=body.redirect_uri,
            tenant_id=tenancy.tenant_of(request.headers, runtime.settings),
        )
    except ValueError as exc:
        # Unsupported or unconfigured provider is a client-visible condition,
        # not a server fault.
        raise http_error("invalid_request", str(exc), status_code=400)
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
    request: Request,
    provider: str = Path(..., description="OAuth provider"),
    code: str = Query(..., max_length=512, description="Authorization code from OAuth provider"),
    state: str = Query(..., max_length=128, description="State parameter for CSRF protection"),
    response: Response = ...,
    # SECURITY: tenant_id is derived from OAuth state, not from query params
    # This prevents tenant spoofing attacks per CLAUDE.md guidelines
):
    """Complete OAuth authentication flow.

    Exchanges authorization code for tokens and creates user session.
    Called by OAuth provider after user authorizes the application.

    Note: tenant_id is securely derived from the OAuth state token that was
    set during oauth_start, not from user-controllable query parameters.
    """
    runtime = get_runtime()
    # Rate limit OAuth callback per client to prevent code brute-forcing
    await enforce(
        runtime,
        f"oauth:callback:{provider}:{client_ip(request)}",
        limit=10,
        window_seconds=60,
    )
    # SECURITY: Don't pass tenant_id - it's derived from validated state
    user, session, tokens = await runtime.auth.complete_oauth(
        provider, code, state
    )
    if not user or not session:
        raise http_error("unauthorized", "oauth verification failed", status_code=401)
    _apply_session_cookies(
        response,
        session,
        tokens,
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
            csrf_token=tokens.get("csrf_token"),
            role=user.role,
            tenant_id=user.tenant_id,
        ),
    )


def _credential_from_body_or_cookie(
    body_value: Optional[str], cookie_value: Optional[str], *, name: str
) -> Optional[str]:
    """One credential from two transports, or a refusal.

    The browser sends it as an `HttpOnly` cookie it cannot read (§17.10); API
    and mobile clients, which have no cookie jar, send it in the body. Both
    remain supported, because narrowing the public API is not what moving the
    browser's copy out of reach is for.

    Disagreement is refused rather than resolved. Picking either one silently
    lets a caller who can write one transport override the other, and there
    is no reading of "here are two different credentials" that is safe to
    guess at.
    """
    if body_value and cookie_value and body_value != cookie_value:
        logger.warning("auth_credential_conflict", credential=name)
        raise http_error(
            "unauthorized",
            f"conflicting {name}: the body and the cookie disagree",
            status_code=401,
        )
    return body_value or cookie_value


@router.post("/auth/refresh", response_model=Envelope, tags=["auth"])
async def refresh_tokens(
    body: TokenRefreshRequest,
    response: Response,
    request: Request,
    authorization: Optional[str] = Header(None),
):
    runtime = get_runtime()
    tenant_hint = tenancy.tenant_of(request.headers, runtime.settings)
    client_ip = request.client.host if request.client else "unknown"
    await rate_limit(runtime, "refresh", client_ip)
    refresh_token = _credential_from_body_or_cookie(
        body.refresh_token,
        request.cookies.get("refresh_token"),
        name="refresh token",
    )
    if not refresh_token:
        raise http_error("unauthorized", "invalid refresh", status_code=401)
    user, session, tokens = await runtime.auth.refresh_tokens(
        refresh_token, tenant_hint=tenant_hint
    )
    if not user or not session:
        logger.warning(
            "refresh_invalid",
            tenant_hint=tenant_hint,
            has_header=bool(authorization),
        )
        raise http_error("unauthorized", "invalid refresh", status_code=401)
    _apply_session_cookies(
        response,
        session,
        tokens,
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
            csrf_token=tokens.get("csrf_token"),
            role=user.role,
            tenant_id=user.tenant_id,
        ),
    )


@router.get("/admin/users", response_model=Envelope, tags=["admin"])
async def admin_list_users(
    tenant_id: Optional[str] = None,
    # Issue 46.5: Add upper bound to prevent integer overflow/DoS
    limit: Optional[int] = Query(None, ge=1, le=10000, description="Maximum users to return"),
    principal: AuthContext = Depends(get_admin_user),
):
    """List users in the admin's tenant.

    Returns:
        List of users with their roles and metadata.
    """
    runtime = get_runtime()
    await rate_limit(runtime, "admin:read", principal.user_id)
    paging = _get_pagination_settings(runtime)
    resolved_limit = min(limit or paging["default_page_size"], paging["max_page_size"])
    # Admin can only see users in their own tenant (prevent cross-tenant access)
    target_tenant = tenant_id or principal.tenant_id
    if target_tenant != principal.tenant_id:
        raise http_error(
            "forbidden", "cannot access other tenant users", status_code=403
        )
    users = runtime.auth.list_users(tenant_id=target_tenant, limit=resolved_limit)
    return Envelope(
        status="ok", data=UserListResponse(items=[UserResponse.model_validate(u) for u in users])
    )


@router.post("/admin/users", response_model=Envelope, tags=["admin"])
async def admin_create_user(
    body: AdminCreateUserRequest, principal: AuthContext = Depends(get_admin_user)
):
    runtime = get_runtime()
    await enforce(
        runtime,
        f"admin:create_user:{principal.user_id}",
        runtime.settings.admin_rate_limit_per_minute,
        runtime.settings.admin_rate_limit_window_seconds,
    )
    # An admin creates users in their own tenant, which is the site they are
    # administering. There is deliberately no override: a tenant is a site, and
    # reaching another one means visiting it.
    target_tenant = principal.tenant_id
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
    # Issue 51.2: Audit logging for admin user creation (GDPR/SOC2 compliance)
    logger.info(
        "admin_user_created",
        admin_id=principal.user_id,
        created_user_id=user.id,
        created_email=user.email,
        created_role=user.role,
        tenant_id=target_tenant,
    )
    return Envelope(
        status="ok",
        data=AdminCreateUserResponse(
            **UserResponse.model_validate(user).model_dump(), password=password
        ),
    )


@router.post("/admin/users/{user_id}/role", response_model=Envelope, tags=["admin"])
async def admin_set_role(
    user_id: str = Path(..., max_length=255, description="User identifier"),
    body: UpdateUserRoleRequest = Body(...),
    principal: AuthContext = Depends(get_admin_user),
):
    runtime = get_runtime()
    await enforce(
        runtime,
        f"admin:set_role:{principal.user_id}",
        runtime.settings.admin_rate_limit_per_minute,
        runtime.settings.admin_rate_limit_window_seconds,
    )
    # Issue 50.3: Validate tenant isolation - admin can only modify users in their tenant
    target_user = runtime.store.get_user(user_id)
    if not target_user:
        raise NotFoundError("user not found", detail={"user_id": user_id})
    if target_user.tenant_id != principal.tenant_id:
        raise http_error(
            "forbidden", "cannot modify users in other tenant", status_code=403
        )
    # Issue 22.1: set_user_role is now async to revoke sessions on role change
    old_role = target_user.role
    user = await runtime.auth.set_user_role(user_id, body.role)
    if not user:
        raise NotFoundError("user not found", detail={"user_id": user_id})
    # Issue 51.4: Audit logging for permission changes (SOC2 compliance)
    logger.info(
        "admin_role_changed",
        admin_id=principal.user_id,
        target_user_id=user_id,
        old_role=old_role,
        new_role=body.role,
        tenant_id=principal.tenant_id,
    )
    return Envelope(status="ok", data=UserResponse.model_validate(user))


@router.delete("/admin/users/{user_id}", response_model=Envelope, tags=["admin"])
async def admin_delete_user(
    user_id: str, principal: AuthContext = Depends(get_admin_user)
):
    runtime = get_runtime()
    await enforce(
        runtime,
        f"admin:delete_user:{principal.user_id}",
        runtime.settings.admin_rate_limit_per_minute,
        runtime.settings.admin_rate_limit_window_seconds,
    )
    # Issue 50.4: Validate tenant isolation - admin can only delete users in their tenant
    target_user = runtime.store.get_user(user_id)
    if not target_user:
        raise NotFoundError("user not found", detail={"user_id": user_id})
    if target_user.tenant_id != principal.tenant_id:
        raise http_error(
            "forbidden", "cannot delete users in other tenant", status_code=403
        )
    # The refusal is inside `delete_user`'s transaction, where it can hold the
    # account, its artifacts and their unfinished jobs. Asking here first
    # would be a check-then-act, which is the race this exists to remove; the
    # store raises TrainingInProgress and the handler turns that into 409.
    removed = await runtime.auth.delete_user(user_id)
    if not removed:
        raise NotFoundError("user not found", detail={"user_id": user_id})
    # Issue 51.3: Audit logging for user deletion (GDPR/SOC2 compliance)
    #
    # By id. The address is the identifier this request exists to remove, and
    # writing it into the audit log copies it into a store with its own
    # retention and its own readers — an erasure that ends by re-recording
    # what it erased. The id correlates the entry with everything else about
    # the account, which is what an audit trail is for.
    logger.info(
        "admin_user_deleted",
        admin_id=principal.user_id,
        deleted_user_id=user_id,
        tenant_id=principal.tenant_id,
    )
    return Envelope(status="ok", data={"deleted": True, "user_id": user_id})


@router.get("/admin/adapters", response_model=Envelope, tags=["admin"])
async def admin_list_adapters(principal: AuthContext = Depends(get_admin_user)):
    runtime = get_runtime()
    await rate_limit(runtime, "admin:read", principal.user_id)
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
    await rate_limit(runtime, "admin:read", principal.user_id)
    paging = _get_pagination_settings(runtime)
    resolved_limit = min(limit or paging["default_conversations_limit"], paging["max_page_size"])
    details = runtime.store.inspect_state(
        kind=kind, tenant_id=principal.tenant_id, limit=resolved_limit
    )
    summary = {k: len(v) for k, v in details.items()}
    return Envelope(
        status="ok", data=AdminInspectionResponse(summary=summary, details=details)
    )


@router.post("/auth/mfa/request", response_model=Envelope, tags=["auth"])
async def request_mfa(body: MFARequest, request: Request):
    runtime = get_runtime()
    client_ip = request.client.host if request.client else "unknown"
    session_cookie = request.cookies.get("session_id")
    # Issue 50.1: Rate limit per IP first to prevent session enumeration attacks
    await enforce(
        runtime,
        f"mfa:request:ip:{client_ip}",
        runtime.settings.mfa_rate_limit_per_minute,
        60,
    )
    # The cookie is the browser's authority and the body is the API client's;
    # a disagreement is refused rather than resolved. Neither means there is
    # no session to act on.
    session_id = _credential_from_body_or_cookie(
        body.session_id, session_cookie, name="session id"
    )
    if not session_id:
        logger.warning(
            "mfa_request_missing_cookie",
            ip=client_ip,
            cookie_present=bool(session_cookie),
        )
        raise http_error("unauthorized", "invalid session", status_code=401)
    auth_ctx = await runtime.auth.resolve_session(
        session_id, allow_pending_mfa=True
    )
    if not auth_ctx:
        logger.warning("mfa_request_invalid_session", ip=client_ip)
        raise http_error("unauthorized", "invalid session", status_code=401)
    # Issue 50.1: Validate session IP matches requester IP for security
    session = runtime.store.get_session(session_id)
    if session and session.ip_addr and str(session.ip_addr) != client_ip:
        logger.warning(
            "mfa_request_ip_mismatch",
            session_id=session_id,
            session_ip=str(session.ip_addr),
            request_ip=client_ip,
        )
        raise http_error("unauthorized", "session bound to different IP", status_code=401)
    await enforce(
        runtime,
        f"mfa:request:{auth_ctx.user_id}",
        runtime.settings.mfa_rate_limit_per_minute,
        60,
    )
    challenge = await runtime.auth.issue_mfa_challenge(user_id=auth_ctx.user_id)
    logger.info("mfa_challenge_issued", user_id=auth_ctx.user_id, ip=client_ip)
    return Envelope(status="ok", data=challenge)


@router.post("/auth/mfa/verify", response_model=Envelope, tags=["auth"])
async def verify_mfa(body: MFAVerifyRequest, request: Request, response: Response):
    runtime = get_runtime()
    client_ip = request.client.host if request.client else "unknown"
    session_cookie = request.cookies.get("session_id")
    # Issue 50.2: Rate limit per IP first to prevent brute force attacks
    await enforce(
        runtime,
        f"mfa:verify:ip:{client_ip}",
        runtime.settings.mfa_rate_limit_per_minute,
        60,
    )
    # The cookie is the browser's authority and the body is the API client's;
    # a disagreement is refused rather than resolved. Neither means there is
    # no session to act on.
    session_id = _credential_from_body_or_cookie(
        body.session_id, session_cookie, name="session id"
    )
    if not session_id:
        logger.warning(
            "mfa_verify_missing_cookie",
            ip=client_ip,
            cookie_present=bool(session_cookie),
        )
        raise http_error("unauthorized", "invalid session", status_code=401)
    auth_ctx = await runtime.auth.resolve_session(
        session_id, allow_pending_mfa=True
    )
    if not auth_ctx:
        logger.warning("mfa_verify_invalid_session", ip=client_ip)
        raise http_error("unauthorized", "invalid session", status_code=401)
    # Issue 50.2: Validate session IP matches requester IP for security
    session = runtime.store.get_session(session_id)
    if session and session.ip_addr and str(session.ip_addr) != client_ip:
        logger.warning(
            "mfa_verify_ip_mismatch",
            session_id=session_id,
            session_ip=str(session.ip_addr),
            request_ip=client_ip,
            user_id=auth_ctx.user_id,
        )
        raise http_error("unauthorized", "session bound to different IP", status_code=401)
    await enforce(
        runtime,
        f"mfa:verify:{auth_ctx.user_id}",
        runtime.settings.mfa_rate_limit_per_minute,
        60,
    )
    # Read before verifying: a successful verify flips the secret to enabled,
    # so this is the only point where enrolment is distinguishable from an
    # ordinary login check.
    was_enabled = bool(getattr(runtime.store.get_user_mfa_secret(auth_ctx.user_id), "enabled", False))
    ok = await runtime.auth.verify_mfa_challenge(
        user_id=auth_ctx.user_id, code=body.code, session_id=session_id
    )
    if not ok:
        logger.warning("mfa_verify_failed", user_id=auth_ctx.user_id, ip=client_ip)
        raise http_error("unauthorized", "invalid mfa", status_code=401)
    user, session, tokens = runtime.auth.issue_tokens_for_session(session_id)
    if not was_enabled and user and user.email:
        # Enrolment just completed. Telling the account holder that a second
        # factor now guards their login is how they find out if it was not
        # them who added it. Never block the response on the SMTP round trip.
        try:
            await asyncio.to_thread(
                runtime.email.send_mfa_setup_confirmation, user.email
            )
        except Exception as exc:  # noqa: BLE001 - notification is best effort
            logger.warning(
                "mfa_setup_email_failed", user_id=auth_ctx.user_id, error=str(exc)
            )
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
                csrf_token=tokens.get("csrf_token"),
                role=user.role,
                tenant_id=user.tenant_id,
            ).model_dump()
        )
        _apply_session_cookies(
            response,
            session,
            tokens,
                    )
    return Envelope(status="ok", data=resp)


@router.get("/auth/mfa/status", response_model=Envelope, tags=["auth"])
async def get_mfa_status(principal: AuthContext = Depends(get_user)):
    """Get the current MFA status for the authenticated user."""
    runtime = get_runtime()
    await enforce(
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
    await enforce(
        runtime,
        f"mfa:disable:{principal.user_id}",
        runtime.settings.mfa_rate_limit_per_minute,
        60,
    )
    mfa_cfg = runtime.store.get_user_mfa_secret(principal.user_id)
    if not mfa_cfg or not mfa_cfg.enabled:
        raise http_error("validation_error", "MFA not enabled", status_code=400)
    # Verify the code before disabling
    if not runtime.auth._verify_totp(mfa_cfg.secret, body.code):
        raise http_error("unauthorized", "invalid MFA code", status_code=401)
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
    await rate_limit(runtime, "reset", body.email.lower())
    # Check if user exists before generating token (don't reveal if user exists)
    user = runtime.store.get_user_by_email(body.email)
    if user:
        # No token when the account was erased between the lookup above and
        # the write. Nothing is sent, and the answer below is the same one an
        # address that never existed gets — which is the point of that answer.
        token = await runtime.auth.initiate_password_reset(user)
        if token:
            # Run blocking SMTP in thread to avoid blocking event loop
            await asyncio.to_thread(
                runtime.email.send_password_reset, body.email, token
            )
    # Always return success to prevent email enumeration
    return Envelope(status="ok", data={"status": "sent"})


@router.post("/auth/reset/confirm", response_model=Envelope, tags=["auth"])
async def confirm_reset(body: PasswordResetConfirm, request: Request):
    runtime = get_runtime()
    # Rate limit per client to prevent token brute-forcing without letting one
    # client lock out password reset for everyone.
    await enforce(
        runtime,
        f"reset:confirm:{client_ip(request)}",
        limit=5,
        window_seconds=300,
    )
    ok = await runtime.auth.complete_password_reset(body.token, body.new_password)
    if not ok:
        raise http_error("validation_error", "invalid token", status_code=400)
    return Envelope(status="ok", data={"status": "reset"})


@router.get("/me", response_model=Envelope, tags=["auth"])
async def get_current_user(principal: AuthContext = Depends(get_user)):
    """Get the current user's profile.

    Returns user details including email verification status.
    """
    runtime = get_runtime()
    await rate_limit(runtime, "read", principal.user_id)
    user = runtime.store.get_user(principal.user_id)
    if not user:
        raise http_error("not_found", "user not found", status_code=404)
    return Envelope(
        status="ok",
        data=UserResponse.model_validate(user),
    )


@router.get("/settings", response_model=Envelope, tags=["settings"])
async def get_user_settings(principal: AuthContext = Depends(get_user)):
    """Get the current user's settings.

    Returns user preferences like locale, timezone, voice settings.
    """
    runtime = get_runtime()
    await rate_limit(runtime, "read", principal.user_id)
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
    await enforce(
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
    await enforce(
        runtime,
        f"verify:request:{principal.user_id}",
        limit=5,
        window_seconds=300,
    )
    user = runtime.store.get_user(principal.user_id)
    if not user:
        raise http_error("not_found", "user not found", status_code=404)
    token = await runtime.auth.request_email_verification(user)
    if not token:
        # Erased between the lookup above and the write. The caller is holding
        # a credential for an account that no longer exists.
        raise http_error("not_found", "user not found", status_code=404)
    # Run blocking SMTP in thread to avoid blocking event loop
    await asyncio.to_thread(runtime.email.send_email_verification, user.email, token)
    return Envelope(status="ok", data={"status": "sent"})


@router.post("/auth/verify_email", response_model=Envelope, tags=["auth"])
async def verify_email(body: EmailVerificationRequest, request: Request):
    runtime = get_runtime()
    # Rate limit per client to prevent token brute-forcing without letting one
    # client block email verification platform-wide.
    await enforce(
        runtime,
        f"verify:email:{client_ip(request)}",
        limit=10,
        window_seconds=300,
    )
    ok = await runtime.auth.complete_email_verification(body.token)
    if not ok:
        raise http_error("validation_error", "invalid token", status_code=400)
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
    await enforce(
        runtime,
        f"password:change:{principal.user_id}",
        limit=5,
        window_seconds=300,
    )

    # Verify current password
    if not runtime.auth.verify_password(principal.user_id, body.current_password):
        raise http_error("unauthorized", "current password is incorrect", status_code=401)

    # Save new password
    runtime.auth.save_password(principal.user_id, body.new_password)

    # SECURITY: Revoke all other sessions to force re-authentication
    await runtime.auth.revoke_all_user_sessions(
        principal.user_id, except_session_id=principal.session_id
    )

    # Issue 51.6: Audit logging for password change (GDPR/SOC2 compliance)
    logger.info(
        "user_password_changed",
        user_id=principal.user_id,
        session_id=principal.session_id,
    )
    return Envelope(status="ok", data={"status": "changed"})


@router.post("/auth/logout", response_model=Envelope, tags=["auth"])
async def logout(
    request: Request,
    response: Response,
    session_id: Optional[str] = Header(None, convert_underscores=False),
    authorization: Optional[str] = Header(None),
):
    runtime = get_runtime()
    # By address: logout takes an unauthenticated session id, so an attacker
    # could otherwise walk guesses at it to revoke other people's sessions.
    await rate_limit(runtime, "write", client_ip(request))
    if authorization or session_id:
        ctx = await runtime.auth.authenticate(
            authorization, session_id, allow_pending_mfa=True
        )
        if not ctx:
            raise http_error("unauthorized", "invalid session", status_code=401)
        # SECURITY: Only allow revoking the authenticated user's own session
        target_session = session_id or ctx.session_id
        if target_session and target_session != ctx.session_id:
            # Verify session ownership before revoking
            sess = runtime.store.get_session(target_session)
            if not sess or sess.user_id != ctx.user_id:
                raise http_error("forbidden", "cannot revoke other user sessions", status_code=403)
        if target_session:
            await runtime.auth.revoke(target_session)
            # Issue 51.9: Audit logging for session revocation (SOC2 compliance)
            logger.info(
                "session_revoked",
                user_id=ctx.user_id,
                session_id=target_session,
            )
    response.delete_cookie("session_id", path="/", secure=True, samesite="lax")
    response.delete_cookie("refresh_token", path="/", secure=True, samesite="lax")
    response.delete_cookie("csrf_token", path="/", secure=True, samesite="lax")
    return Envelope(status="ok", data={"message": "session revoked"})


# Bounded so a compromised session can't stockpile credentials that each
# need their own revocation.
MAX_ACTIVE_API_KEYS = 20


def _api_key_summary(record) -> ApiKeySummary:
    return ApiKeySummary(
        id=record.id,
        name=record.name,
        prefix=record.prefix,
        created_at=record.created_at,
        last_used_at=record.last_used_at,
        revoked_at=record.revoked_at,
    )


@router.post("/auth/api-keys", response_model=Envelope, status_code=201, tags=["auth"])
async def create_api_key(
    body: ApiKeyCreateRequest,
    principal: AuthContext = Depends(get_user),
):
    """Mint a bearer key for the served Responses API (SPEC §13.1).

    Session-authenticated on purpose — get_user cannot read keys, so a key
    can never mint another and outlive its own revocation.
    """
    runtime = get_runtime()
    await rate_limit(runtime, "write", principal.user_id)
    if runtime.store.count_active_api_keys(principal.user_id) >= MAX_ACTIVE_API_KEYS:
        raise http_error(
            "validation_error",
            f"at most {MAX_ACTIVE_API_KEYS} active API keys; revoke one first",
            status_code=400,
        )
    record, plaintext = runtime.auth.mint_api_key(
        principal.user_id, name=body.name.strip()
    )
    logger.info("api_key_created", user_id=principal.user_id, key_id=record.id)
    return Envelope(
        status="ok",
        data=ApiKeyCreatedResponse(
            id=record.id,
            name=record.name,
            prefix=record.prefix,
            created_at=record.created_at,
            api_key=plaintext,
        ),
    )


@router.get("/auth/api-keys", response_model=Envelope, tags=["auth"])
async def list_api_keys(principal: AuthContext = Depends(get_user)):
    """Every key the user holds, revoked ones included: that is the audit view."""
    runtime = get_runtime()
    await rate_limit(runtime, "read", principal.user_id)
    records = runtime.store.list_api_keys(principal.user_id)
    return Envelope(
        status="ok",
        data=ApiKeyListResponse(items=[_api_key_summary(r) for r in records]),
    )


@router.delete("/auth/api-keys/{key_id}", response_model=Envelope, tags=["auth"])
async def revoke_api_key(
    key_id: str = Path(..., max_length=64),
    principal: AuthContext = Depends(get_user),
):
    runtime = get_runtime()
    await rate_limit(runtime, "write", principal.user_id)
    revoked = runtime.store.revoke_api_key(key_id, user_id=principal.user_id)
    if not revoked:
        raise http_error("not_found", "API key not found", status_code=404)
    logger.info("api_key_revoked", user_id=principal.user_id, key_id=key_id)
    return Envelope(status="ok", data=ApiKeyRevokeResponse(revoked=True))


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
    logger.info(
        "chat_request",
        user_id=user_id,
        conversation_id=body.conversation_id,
        context_id=body.context_id,
        workflow_id=body.workflow_id,
        plan_tier=plan_tier,
        idempotency_key=bool(idempotency_key),
    )

    # SPEC §18: Accept Idempotency-Key when provided (optional)
    async with idempotency.IdempotencyGuard(
        "chat", user_id, idempotency_key, require=False
    ) as idem:
        if idem.cached:
            return idem.cached

        # SPEC §18: Per-plan adjustable rate limits
        await enforce_per_plan(
            runtime,
            f"chat:{user_id}",
            runtime.settings.chat_rate_limit_per_minute,
            runtime.settings.chat_rate_limit_window_seconds,
            plan_tier,
        )

        # SPEC §18: concurrency caps - limit decode pressure per user. The same
        # slots the streaming path takes, named once in service.admission.
        async with admission.slots(runtime, user_id, *admission.CHAT_SLOTS):
            turn = await chat_turn.begin(
                runtime,
                principal,
                conversation_id=body.conversation_id,
                context_id=body.context_id,
                workflow_id=body.workflow_id,
                content=body.message.content,
                content_struct=body.message.content_struct,
                mode=body.message.mode,
                owned_conversation=_get_owned_conversation,
                owned_context=_get_owned_context,
            )
            orchestration = await runtime.workflow.run(
                body.workflow_id,
                turn.conversation_id,
                turn.user_content,
                turn.context_id,
                user_id,
                tenant_id=principal.tenant_id,
            )
            assistant_msg = await chat_turn.finish(
                runtime, turn, orchestration, commit=idem.commit
            )
            adapter_names = _stringify_adapters(turn.orchestration.get("adapters", []))
            envelope = Envelope(
                status="ok",
                data=chat_turn.response(turn, assistant_msg, adapter_names).model_dump(),
                request_id=idem.request_id,
            )
            usage = turn.orchestration.get("usage") or {}
            logger.info(
                "chat_response",
                user_id=user_id,
                conversation_id=turn.conversation_id,
                context_id=turn.context_id,
                workflow_id=body.workflow_id,
                adapters=adapter_names,
                total_tokens=usage.get("total_tokens"),
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                request_id=idem.request_id,
            )
            await idem.store_result(envelope)
            return envelope
    # Exceptions bubble through the guard which records failed states


# ---------------------------------------------------------------------------
# Served Responses API (OpenAI-compatible surface over liminallm turns)
# ---------------------------------------------------------------------------
#
# The point of serving this shape: any client that speaks the Responses API
# gets a weak model wrapped in this kernel's enrichment — hybrid RAG,
# personas, skill adapters, routing, compaction — because a request here runs
# the SAME turn pipeline as /chat. Responses rather than chat/completions
# because Responses is stateful by design: previous_response_id maps directly
# onto conversations, which carry exactly the state a completions shim would
# throw away.
#
# Two shape rules this route must never break, because SDKs parse by them:
# success bodies are the bare Responses object, never the Envelope; error
# bodies are OpenAI's {"error": {...}}, never ours. The one seam left open is
# auth: a 401 from the dependency still arrives Envelope-shaped (SPEC notes
# it), because rewriting the app-wide auth path for one route is worse.

_RESPONSES_ID_PREFIX = "resp_"
# OpenAI's own metadata bounds; enforced so an echo cannot become a blob store.
_RESPONSES_METADATA_MAX_KEYS = 16
_RESPONSES_METADATA_KEY_CHARS = 64
_RESPONSES_METADATA_VALUE_CHARS = 512
# The same DoS cap /chat enforces (ChatMessage.content max_length); this
# surface validates by hand, so the guard has to be written by hand too.
_RESPONSES_INPUT_MAX_CHARS = 100_000


class _ResponsesReject(Exception):
    """A request shape this surface refuses, in OpenAI's error vocabulary."""

    def __init__(self, message: str, *, param: Optional[str] = None, status: int = 400):
        super().__init__(message)
        self.message = message
        self.param = param
        self.status = status


def _openai_error(
    status: int, message: str, *, param: Optional[str] = None, code: Optional[str] = None
) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={
            "error": {
                "message": message,
                "type": "invalid_request_error" if status < 500 else "server_error",
                "param": param,
                "code": code,
            }
        },
    )


def _responses_reject_unsupported(body: dict) -> None:
    """The v1 scope line, drawn where an SDK will read it.

    Each rejection names why in terms of what this kernel already does —
    "unsupported" alone reads as "broken", and every one of these has a
    deliberate reason.
    """
    if body.get("tools"):
        raise _ResponsesReject(
            "caller-supplied tools are not supported: this surface runs the "
            "kernel's own tool loop (retrieval, files, notes) server-side.",
            param="tools",
        )
    if body.get("instructions"):
        raise _ResponsesReject(
            "instructions are not supported: the system prompt here belongs "
            "to per-user personas and skill adapters, which is the point of "
            "this server.",
            param="instructions",
        )
    if body.get("store") is False:
        raise _ResponsesReject(
            "store=false is not supported: turns persist to the conversation, "
            "which is what previous_response_id continues.",
            param="store",
        )


def _responses_input_text(payload: Any) -> str:
    """The user text of a Responses ``input``, or a named rejection.

    A string passes through. A list accepts message items carrying user text
    — role shorthand or typed parts — and refuses the rest by name: system
    and developer items would fight the personas that own the system prompt
    here, and function_call_output belongs to caller tools, which this
    surface does not take.
    """
    if isinstance(payload, str):
        if len(payload) > _RESPONSES_INPUT_MAX_CHARS:
            raise _ResponsesReject(
                f"input must be at most {_RESPONSES_INPUT_MAX_CHARS} characters.",
                param="input",
            )
        text = payload.strip()
        if not text:
            raise _ResponsesReject("input must not be empty.", param="input")
        return text
    if not isinstance(payload, list) or not payload:
        raise _ResponsesReject(
            "input must be a string or a non-empty list of input items.",
            param="input",
        )
    total_chars = 0
    texts: list[str] = []

    def _take(chunk: str) -> None:
        # Bail before the join: the cap has to stop the accumulation itself,
        # not measure the giant string after it exists.
        nonlocal total_chars
        total_chars += len(chunk)
        if total_chars > _RESPONSES_INPUT_MAX_CHARS:
            raise _ResponsesReject(
                f"input must be at most {_RESPONSES_INPUT_MAX_CHARS} characters "
                "in total.",
                param="input",
            )
        texts.append(chunk)

    for position, item in enumerate(payload):
        where = f"input[{position}]"
        if not isinstance(item, dict):
            raise _ResponsesReject(f"{where} must be an object.", param=where)
        item_type = item.get("type", "message")
        if item_type != "message":
            raise _ResponsesReject(
                f"{where}.type {item_type!r} is not supported on this surface.",
                param=where,
            )
        role = item.get("role")
        if role in ("system", "developer"):
            raise _ResponsesReject(
                f"{where}: system/developer items are not supported — the "
                "system prompt belongs to per-user personas and adapters here.",
                param=where,
            )
        if role != "user":
            raise _ResponsesReject(
                f"{where}: only user message items are supported.", param=where
            )
        content = item.get("content")
        if isinstance(content, str):
            _take(content)
            continue
        if isinstance(content, list):
            for part_at, part in enumerate(content):
                if not isinstance(part, dict) or part.get("type") != "input_text":
                    raise _ResponsesReject(
                        f"{where}.content[{part_at}] must be input_text.",
                        param=where,
                    )
                _take(str(part.get("text") or ""))
            continue
        raise _ResponsesReject(
            f"{where}.content must be a string or a list of input_text parts.",
            param=where,
        )
    combined = "\n\n".join(t for t in (t.strip() for t in texts) if t)
    if not combined:
        raise _ResponsesReject("input must not be empty.", param="input")
    return combined


def _responses_metadata(payload: Any) -> dict:
    if payload is None:
        return {}
    if not isinstance(payload, dict) or len(payload) > _RESPONSES_METADATA_MAX_KEYS:
        raise _ResponsesReject(
            f"metadata must be an object of at most "
            f"{_RESPONSES_METADATA_MAX_KEYS} string pairs.",
            param="metadata",
        )
    for key, value in payload.items():
        if (
            not isinstance(key, str)
            or not isinstance(value, str)
            or len(key) > _RESPONSES_METADATA_KEY_CHARS
            or len(value) > _RESPONSES_METADATA_VALUE_CHARS
        ):
            raise _ResponsesReject(
                "metadata keys and values must be bounded strings.",
                param="metadata",
            )
    return dict(payload)


def _responses_message_item(message_id: str, text: str, *, status: str = "completed") -> dict:
    return {
        "type": "message",
        "id": f"msg_{message_id}",
        "status": status,
        "role": "assistant",
        "content": (
            [{"type": "output_text", "text": text, "annotations": []}]
            if status == "completed"
            else []
        ),
    }


def _responses_usage(usage: dict) -> dict:
    """The turn's usage in the Responses shape, details included.

    The compat layer carries reasoning_tokens and cached_tokens all the way
    into the turn's usage precisely so a consumer can see them; dropping them
    here was the one gap in that chain. The details objects are always
    present (zeros when unknown) because typed SDKs require the fields, and
    the total falls back to input+output for backends — the local tokenizer
    path historically — that report the parts without the sum.
    """
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    reasoning = int(usage.get("reasoning_tokens") or 0)
    total = int(usage.get("total_tokens") or 0)

    # In this shape reasoning_tokens is a detail *within* output_tokens, so a
    # client may compute visible output as output - reasoning and expect
    # input + output == total. Backends disagree about that: OpenAI counts
    # reasoning inside its output count, Gemini counts thoughts alongside
    # candidates. Measured on our own fixture — prompt 10 + candidates 5 = 15
    # against a reported total of 22, which only reconciles once the 7 thought
    # tokens are added.
    #
    # Decided from the provider's own total rather than from a per-backend
    # flag, because the total is the one number every backend reports and it
    # is what makes the parts checkable at all: if they only add up once
    # reasoning is included, reasoning was counted separately and belongs
    # inside the output count we publish.
    if reasoning and total and prompt + completion + reasoning == total:
        completion += reasoning
    if not total:
        total = prompt + completion
    return {
        "input_tokens": prompt,
        "input_tokens_details": {
            "cached_tokens": int(usage.get("cached_tokens") or 0),
            # Required by the dialect's own type since openai 3.x, which is
            # what an unpinned install resolves. Read from the turn's usage
            # like its sibling rather than hard-coded to zero, so a backend
            # that starts reporting cache writes needs no change here; today
            # none does, and the zero is the "present but unknown" the
            # docstring above describes.
            "cache_write_tokens": int(usage.get("cache_write_tokens") or 0),
        },
        "output_tokens": completion,
        "output_tokens_details": {"reasoning_tokens": reasoning},
        "total_tokens": total,
    }


#: Server-side tool runs served as the dialect's own output items — only the
#: types the dialect defines, so typed SDK parsers never meet an unknown
#: discriminator. Everything else (note_search, history_search, run_python,
#: web_fetch) appears in the liminallm extension's tool_trace instead of
#: being dressed up as something it is not.
_RESPONSES_TOOL_ITEM_TYPES = {
    "file_search": "file_search_call",
    "web_search": "web_search_call",
}


def _web_search_action(query: Any) -> dict:
    """What a `web_search_call` says it did. Required by the dialect.

    The action distinguishes a search from opening a page or finding within
    one, and the search variant requires the query — so an item that omitted
    it said a web search happened without saying what for, and failed the
    generated type outright. `run_web_search` is always a search, and the
    trace carries the query, so nothing is invented here.

    An unrecorded query is the empty string rather than an absent field, for
    the reason §16 already gives for the usage detail objects: typed SDKs
    require the field, so it is present and empty when unknown.
    """
    return {"type": "search", "query": str(query) if query else ""}


def _responses_tool_items(tool_trace: list) -> list:
    items = []
    for call in tool_trace or []:
        if not isinstance(call, dict):
            continue
        item_type = _RESPONSES_TOOL_ITEM_TYPES.get(call.get("tool"))
        if not item_type:
            continue
        item = {
            "type": item_type,
            "id": f"{item_type[:2]}_{uuid4().hex}",
            "status": "completed",
        }
        query = (call.get("arguments") or {}).get("query")
        if item_type == "file_search_call":
            item["queries"] = [str(query)] if query else []
        else:
            item["action"] = _web_search_action(query)
        items.append(item)
    return items


def _responses_enrich_tool_items(items: list, tool_trace: list) -> None:
    """Fill streamed tool items from the trace, once the trace exists.

    A streamed item opens on a trace event that carries no arguments, so it
    is emitted with the empty-when-unknown query — which is honest at that
    moment and is what the dialect requires. By the time the turn finishes
    the arguments are known, and the finished response is where a caller
    reads what the run actually did, so it says so rather than staying empty
    because of when the item happened to open. Measured before this existed:
    a streamed `file_search_call` reported `queries: []` and a
    `web_search_call` an empty query, for a run whose trace named one.

    The already-emitted `output_item.done` keeps the empty form: it was
    serialized when it was true. Ids are untouched, so a caller correlating
    the finished item with the one it saw open finds the same item.

    Matched by position among the dialect-native calls, which is the order
    both lists are built in.
    """
    calls = [
        call
        for call in tool_trace or []
        if isinstance(call, dict)
        and _RESPONSES_TOOL_ITEM_TYPES.get(call.get("tool"))
    ]
    for item, call in zip(items, calls):
        if _RESPONSES_TOOL_ITEM_TYPES.get(call.get("tool")) != item.get("type"):
            continue
        query = (call.get("arguments") or {}).get("query")
        if item["type"] == "file_search_call":
            item["queries"] = [str(query)] if query else []
        else:
            item["action"] = _web_search_action(query)


def _responses_extension(orchestration: dict) -> dict:
    """The enrichment the dialect has no slot for, under one namespaced key.

    Extra top-level keys survive the OpenAI SDKs (their models allow unknown
    fields) and stay invisible to strict typed readers. Citations are NOT
    faked into annotations: an annotation needs a character anchor and a file
    identity this surface cannot honestly provide, so provenance rides here
    as plain snippets instead.
    """
    return {
        "context_snippets": list(orchestration.get("context_snippets") or []),
        "tool_trace": list(orchestration.get("tool_calls") or []),
        "adapters": _stringify_adapters(orchestration.get("adapters", [])),
    }


def _responses_payload(
    *,
    response_id: str,
    created_at: int,
    status: str,
    model: str,
    output: list,
    previous_response_id: Optional[str],
    metadata: dict,
    usage: Optional[dict],
    error: Optional[dict] = None,
    extension: Optional[dict] = None,
) -> dict:
    payload = {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "status": status,
        "error": error,
        "model": model,
        "output": output,
        "previous_response_id": previous_response_id,
        "store": True,
        "metadata": metadata,
        "usage": usage,
        # Required by the dialect, and all three describe the caller-supplied
        # tool surface — which this endpoint refuses by name, because it runs
        # the kernel's own loop server-side. So: no caller tools were in
        # effect, none were available to choose between, and none were
        # emitted in parallel. What the server ran appears where §16 says it
        # does, as dialect-native `output` items and the `liminallm` trace.
        "tools": [],
        "tool_choice": "none",
        "parallel_tool_calls": False,
    }
    if extension is not None:
        payload["liminallm"] = extension
    return payload


def _sse_event(event: str, sequence_number: int, data: dict) -> str:
    payload = {"type": event, "sequence_number": sequence_number, **data}
    return f"event: {event}\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n"


async def _responses_stream(
    runtime,
    principal: AuthContext,
    turn,
    *,
    message_id: str,
    previous_response_id: Optional[str],
    metadata: dict,
    held_slots: list,
):
    """The turn as OpenAI ``response.*`` server-sent events.

    The reply's id is minted before the first event, so ``response.created``
    and ``response.completed`` carry the SAME id and the assistant message is
    persisted under it at finish. A client disconnect cancels generation;
    failures become a ``response.failed`` event, never a broken stream or an
    envelope-shaped body.
    """
    seq = 0

    def ev(event: str, data: dict) -> str:
        nonlocal seq
        seq += 1
        return _sse_event(event, seq, data)

    resp_id = f"{_RESPONSES_ID_PREFIX}{message_id}"
    item_id = f"msg_{message_id}"
    created_at = int(time.time())
    model = runtime.llm.serving_model or runtime.settings.model_path

    def payload(status: str, output: list, usage=None, error=None, extension=None) -> dict:
        return _responses_payload(
            response_id=resp_id,
            created_at=created_at,
            status=status,
            model=model,
            output=output,
            previous_response_id=previous_response_id,
            metadata=metadata,
            usage=usage,
            error=error,
            extension=extension,
        )

    cancel_event = asyncio.Event()
    tokens: list[str] = []
    orchestration: dict = {}
    # Server-side tool runs become dialect items as their traces arrive; the
    # message item is emitted lazily so its output_index lands after them.
    open_items: list = []
    closed_items: list = []
    next_output_index = 0
    message_index: Optional[int] = None

    def close_open_items() -> list:
        chunks = []
        for tool_item, index in open_items:
            tool_item["status"] = "completed"
            chunks.append(
                ev(
                    "response.output_item.done",
                    {"output_index": index, "item": tool_item},
                )
            )
            closed_items.append(tool_item)
        open_items.clear()
        return chunks

    def start_message() -> list:
        # Tools ran before the text started; say so before saying the text.
        nonlocal message_index, next_output_index
        chunks = close_open_items()
        if message_index is None:
            message_index = next_output_index
            next_output_index += 1
            chunks.append(
                ev(
                    "response.output_item.added",
                    {
                        "output_index": message_index,
                        "item": _responses_message_item(
                            message_id, "", status="in_progress"
                        ),
                    },
                )
            )
            chunks.append(
                ev(
                    "response.content_part.added",
                    {
                        "item_id": item_id,
                        "output_index": message_index,
                        "content_index": 0,
                        "part": {"type": "output_text", "text": "", "annotations": []},
                    },
                )
            )
        return chunks

    try:
        yield ev("response.created", {"response": payload("in_progress", [])})
        yield ev("response.in_progress", {"response": payload("in_progress", [])})
        finished = False
        async for event in runtime.workflow.run_streaming(
            None,
            turn.conversation_id,
            turn.user_content,
            turn.context_id,
            principal.user_id,
            tenant_id=principal.tenant_id,
            cancel_event=cancel_event,
        ):
            kind = event.get("event")
            data = event.get("data")
            if kind == "token":
                if isinstance(data, str) and data:
                    tokens.append(data)
                    for chunk in start_message():
                        yield chunk
                    yield ev(
                        "response.output_text.delta",
                        {
                            "item_id": item_id,
                            "output_index": message_index,
                            "content_index": 0,
                            "delta": data,
                            # Required by the dialect on both text events, and
                            # read by the SDK's own stream accumulator. There
                            # are no token logprobs on this surface, so it is
                            # present and empty rather than absent.
                            "logprobs": [],
                        },
                    )
            elif kind == "trace":
                tool = data.get("tool") if isinstance(data, dict) else None
                item_type = _RESPONSES_TOOL_ITEM_TYPES.get(tool)
                if item_type and message_index is None:
                    tool_item = {
                        "type": item_type,
                        "id": f"{item_type[:2]}_{uuid4().hex}",
                        "status": "in_progress",
                    }
                    # The trace event that opens the item does not carry the
                    # arguments, so both of these are the empty-when-unknown
                    # form the dialect requires rather than an omission.
                    if item_type == "file_search_call":
                        tool_item["queries"] = []
                    else:
                        tool_item["action"] = _web_search_action(None)
                    yield ev(
                        "response.output_item.added",
                        {"output_index": next_output_index, "item": dict(tool_item)},
                    )
                    open_items.append((tool_item, next_output_index))
                    next_output_index += 1
            elif kind == "message_done":
                finished = True
                orchestration = data if isinstance(data, dict) else {}
            elif kind == "error":
                message = data.get("message") if isinstance(data, dict) else None
                yield ev(
                    "response.failed",
                    {
                        "response": payload(
                            "failed",
                            [],
                            error={
                                "code": "server_error",
                                "message": message or "generation failed",
                            },
                        )
                    },
                )
                return
            # trace and other kernel events are not part of this dialect.
        if not finished:
            yield ev(
                "response.failed",
                {
                    "response": payload(
                        "failed",
                        [],
                        error={
                            "code": "server_error",
                            "message": "generation ended early",
                        },
                    )
                },
            )
            return
        assistant_msg = await chat_turn.finish(
            runtime,
            turn,
            orchestration,
            content="".join(tokens),
            assistant_message_id=message_id,
        )
        text = assistant_msg.content
        # An agent-loop answer arrives whole (zero deltas): the message item
        # opens here instead, after the tool items it waited on close.
        for chunk in start_message():
            yield chunk
        yield ev(
            "response.output_text.done",
            {
                "item_id": item_id,
                "output_index": message_index,
                "content_index": 0,
                "text": text,
                "logprobs": [],
            },
        )
        yield ev(
            "response.content_part.done",
            {
                "item_id": item_id,
                "output_index": message_index,
                "content_index": 0,
                "part": {"type": "output_text", "text": text, "annotations": []},
            },
        )
        item = _responses_message_item(assistant_msg.id, text)
        yield ev(
            "response.output_item.done",
            {"output_index": message_index, "item": item},
        )
        usage = turn.orchestration.get("usage") or {}
        # The trace only exists now, so this is where the items it describes
        # stop saying "unknown".
        _responses_enrich_tool_items(
            closed_items, turn.orchestration.get("tool_calls")
        )
        yield ev(
            "response.completed",
            {
                "response": payload(
                    "completed",
                    closed_items + [item],
                    usage=_responses_usage(usage),
                    extension=_responses_extension(turn.orchestration),
                )
            },
        )
        logger.info(
            "responses_turn",
            user_id=principal.user_id,
            conversation_id=turn.conversation_id,
            previous_response_id=previous_response_id,
            total_tokens=usage.get("total_tokens"),
            stream=True,
        )
    except GeneratorExit:
        # The client went away; stop generating. Nothing more can be sent.
        cancel_event.set()
        raise
    except Exception:  # noqa: BLE001 — a failed event beats a broken stream
        logger.exception("responses_stream_failed", user_id=principal.user_id)
        yield ev(
            "response.failed",
            {
                "response": payload(
                    "failed",
                    [],
                    error={"code": "server_error", "message": "internal server error"},
                )
            },
        )
    finally:
        for kind in reversed(held_slots):
            await admission.release(runtime, kind, principal.user_id)


@router.post("/responses", tags=["responses"])
async def create_response(
    request: Request,
    principal: AuthContext = Depends(get_user_or_api_key),
):
    """One enriched turn, in the Responses API shape.

    The body is read raw and validated by hand so both the success and the
    error wire shapes are exactly OpenAI's — response_model wrapping, a typed
    body parameter, and FastAPI's 422s would each break an SDK's parser in a
    different way.
    """
    runtime = get_runtime()
    user_id = principal.user_id
    try:
        try:
            body = await request.json()
        except ValueError:
            raise _ResponsesReject("request body must be valid JSON.") from None
        if not isinstance(body, dict):
            raise _ResponsesReject("request body must be a JSON object.")
        _responses_reject_unsupported(body)
        content = _responses_input_text(body.get("input"))
        metadata = _responses_metadata(body.get("metadata"))

        conversation_id: Optional[str] = None
        previous_response_id = body.get("previous_response_id")
        if previous_response_id is not None:
            if not isinstance(previous_response_id, str) or not (
                previous_response_id.startswith(_RESPONSES_ID_PREFIX)
            ):
                raise _ResponsesReject(
                    "previous_response_id must be an id returned by this "
                    "endpoint.",
                    param="previous_response_id",
                )
            conversation_id = runtime.store.get_message_conversation(
                previous_response_id[len(_RESPONSES_ID_PREFIX):]
            )
            if not conversation_id:
                raise _ResponsesReject(
                    f"No response found with id {previous_response_id!r}.",
                    param="previous_response_id",
                    status=404,
                )

        # Liminallm extension, documented in SPEC: bind a knowledge context on
        # the FIRST turn so retrieval grounds the whole thread. Continuations
        # inherit the conversation's binding, as they do on /chat.
        context_id = body.get("context_id")
        if context_id is not None and not isinstance(context_id, str):
            raise _ResponsesReject("context_id must be a string.", param="context_id")

        user = runtime.store.get_user(user_id)
        plan_tier = user.plan_tier if user else "free"
        # The same budget pool as /chat, deliberately: this IS a chat turn,
        # and a second bucket would be a second limit to misconfigure.
        await enforce_per_plan(
            runtime,
            f"chat:{user_id}",
            runtime.settings.chat_rate_limit_per_minute,
            runtime.settings.chat_rate_limit_window_seconds,
            plan_tier,
        )

        if body.get("stream") is True:
            # Everything that can refuse the request refuses it HERE, as a
            # proper HTTP error — once the stream starts the status is spent.
            # Slots are taken in the route and handed to the generator, whose
            # finally releases them however the stream ends.
            held_slots: list[str] = []
            try:
                for kind in admission.CHAT_SLOTS:
                    await admission.acquire(runtime, kind, user_id)
                    held_slots.append(kind)
                turn = await chat_turn.begin(
                    runtime,
                    principal,
                    conversation_id=conversation_id,
                    context_id=context_id,
                    workflow_id=None,
                    content=content,
                    content_struct=None,
                    mode="text",
                    conversation_meta={"source": "responses"},
                    owned_conversation=_get_owned_conversation,
                    owned_context=_get_owned_context,
                )
            except BaseException:
                for kind in reversed(held_slots):
                    await admission.release(runtime, kind, user_id)
                raise
            return StreamingResponse(
                _responses_stream(
                    runtime,
                    principal,
                    turn,
                    message_id=str(uuid4()),
                    previous_response_id=previous_response_id,
                    metadata=metadata,
                    held_slots=held_slots,
                ),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
            )

        async with admission.slots(runtime, user_id, *admission.CHAT_SLOTS):
            turn = await chat_turn.begin(
                runtime,
                principal,
                conversation_id=conversation_id,
                context_id=context_id,
                workflow_id=None,
                content=content,
                content_struct=None,
                mode="text",
                # The native UI badges these; the thread itself is ordinary.
                conversation_meta={"source": "responses"},
                owned_conversation=_get_owned_conversation,
                owned_context=_get_owned_context,
            )
            orchestration = await runtime.workflow.run(
                None,
                turn.conversation_id,
                turn.user_content,
                turn.context_id,
                user_id,
                tenant_id=principal.tenant_id,
            )
            assistant_msg = await chat_turn.finish(runtime, turn, orchestration)

        usage = turn.orchestration.get("usage") or {}
        logger.info(
            "responses_turn",
            user_id=user_id,
            conversation_id=turn.conversation_id,
            previous_response_id=previous_response_id,
            total_tokens=usage.get("total_tokens"),
        )
        output = _responses_tool_items(turn.orchestration.get("tool_calls") or [])
        output.append(
            _responses_message_item(assistant_msg.id, assistant_msg.content)
        )
        return JSONResponse(
            status_code=200,
            content=_responses_payload(
                response_id=f"{_RESPONSES_ID_PREFIX}{assistant_msg.id}",
                created_at=int(time.time()),
                status="completed",
                model=runtime.llm.serving_model or runtime.settings.model_path,
                output=output,
                previous_response_id=previous_response_id,
                metadata=metadata,
                usage=_responses_usage(usage),
                extension=_responses_extension(turn.orchestration),
            ),
        )
    except _ResponsesReject as reject:
        return _openai_error(reject.status, reject.message, param=reject.param)
    except HTTPException as exc:
        # Ownership misses, rate limits and admission raise our Envelope-styled
        # HTTPException; on this surface they must leave OpenAI-shaped.
        detail = exc.detail if isinstance(exc.detail, dict) else {}
        error = detail.get("error") if isinstance(detail.get("error"), dict) else {}
        message = error.get("message") or str(exc.detail) or "request failed"
        return _openai_error(exc.status_code, message, code=error.get("code"))
    except ConstraintViolation as exc:
        return _openai_error(409, exc.message, code="conflict")
    except ServiceError as exc:
        # A provider or service failure mid-turn: without this, the app-wide
        # handler would answer in the Envelope and break the SDK parsing it.
        return _openai_error(exc.status_code, exc.message, code=exc.error_code)
    except Exception:  # noqa: BLE001 — the wire-shape rule outranks the global handler
        logger.exception("responses_turn_failed", user_id=user_id)
        return _openai_error(500, "internal server error", code="server_error")


@router.post("/mcp", tags=["mcp"])
async def mcp_endpoint(
    request: Request,
    principal: AuthContext = Depends(get_user_or_api_key),
):
    """Model Context Protocol over Streamable HTTP (JSON responses only).

    The same credentials as /v1/responses: an API key or a session. See
    liminallm/api/mcp.py for what is spoken and what is deliberately not.
    """
    runtime = get_runtime()
    await rate_limit(runtime, "read", principal.user_id)
    try:
        body = await request.json()
    except ValueError:
        return JSONResponse(status_code=200, content=mcp.parse_error())
    reply = mcp.handle_message(runtime, principal, body)
    if reply is None:
        # A notification: accepted, nothing to say (spec: 202, empty body).
        return Response(status_code=202)
    return JSONResponse(status_code=200, content=reply)


@router.get("/mcp", tags=["mcp"], include_in_schema=False)
async def mcp_get():
    """No server-initiated stream; the spec allows refusing GET outright."""
    return Response(status_code=405)


@router.post("/chat/cancel", response_model=Envelope, tags=["chat"])
async def cancel_chat(
    body: ChatCancelRequest,
    principal: AuthContext = Depends(get_user),
):
    """Cancel an in-progress chat request per SPEC §13.1.

    Cancellation signals the orchestrator to abort decode, free KV cache and
    adapter refs, and emit cancel_ack with partial tokens if any.

    Returns:
        cancelled: True if request was found and cancelled
        message: Human-readable status
    """
    runtime = get_runtime()

    # Rate limit cancellation requests (5 per minute per user)
    await enforce(
        runtime,
        f"chat_cancel:{principal.user_id}",
        limit=5,
        window_seconds=60,
    )

    request_id = body.request_id

    # Try to cancel in-memory active request (Issue 50.5: validates ownership)
    cancelled, reason = await cancellation.cancel(request_id, principal.user_id)

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

    # Issue 50.5: Return 403 if user doesn't own the request
    if reason == "not_owner":
        raise http_error(
            "forbidden", "cannot cancel requests owned by other users", status_code=403
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


# ---------------------------------------------------------------------------
# Notes vault (SPEC: linked notes + the witness)


def _require_notes_enabled(runtime) -> None:
    """403 when the vault is switched off. settings carries the admin value."""
    if not runtime.settings.notes_enabled:
        raise http_error("notes_disabled", "the notes vault is disabled", status_code=403)


def _get_owned_note(runtime, note_id: str, principal: AuthContext):
    note = runtime.store.get_note(note_id, user_id=principal.user_id)
    if not note:
        raise http_error("not_found", "note not found", status_code=404)
    return note


def _note_payload(note, *, content: bool = True) -> dict:
    payload = {
        "id": note.id,
        "title": note.title,
        "created_at": note.created_at.isoformat(),
        "updated_at": note.updated_at.isoformat(),
    }
    if content:
        payload["content"] = note.content
    return payload


def _save_note_graph(runtime, principal: AuthContext, note) -> None:
    """Re-derive links + embedding after any content change."""
    dangling = notes_service.resolve_links(
        runtime.store, principal.user_id, note.id, note.content
    )
    runtime.store.update_note_meta(note.id, {"dangling": dangling})
    embedding = notes_service.embed_note(
        runtime.embeddings, note.title, note.content
    )
    if embedding:
        runtime.store.update_note(note.id, embedding=embedding)
    notes_service.connect_dangling_links(runtime.store, principal.user_id, note)


@router.post("/notes", response_model=Envelope, status_code=201, tags=["notes"])
async def create_note(body: NoteCreateRequest, principal: AuthContext = Depends(get_user)):
    runtime = get_runtime()
    await rate_limit(runtime, "write", principal.user_id)
    _require_notes_enabled(runtime)
    title = notes_service.normalize_title(body.title)
    if not title:
        raise http_error("bad_request", "title required", status_code=400)
    try:
        note = runtime.store.create_note(principal.user_id, title, body.content)
    except ConstraintViolation:
        raise http_error(
            "conflict", "a note with this title already exists", status_code=409
        )
    _save_note_graph(runtime, principal, note)
    return Envelope(status="ok", data=_note_payload(note))


@router.get("/notes", response_model=Envelope, tags=["notes"])
async def list_notes(
    limit: int = 200,
    offset: int = 0,
    principal: AuthContext = Depends(get_user),
):
    runtime = get_runtime()
    await rate_limit(runtime, "read", principal.user_id)
    _require_notes_enabled(runtime)
    notes = runtime.store.list_notes(
        principal.user_id, limit=min(max(limit, 1), 500), offset=max(offset, 0)
    )
    return Envelope(
        status="ok",
        data={
            "notes": [_note_payload(n, content=False) for n in notes],
            "total": runtime.store.count_notes(principal.user_id),
        },
    )


@router.get("/notes/graph", response_model=Envelope, tags=["notes"])
async def notes_graph(principal: AuthContext = Depends(get_user)):
    """Nodes and edges of the user's vault, for the graph view."""
    runtime = get_runtime()
    await rate_limit(runtime, "read", principal.user_id)
    _require_notes_enabled(runtime)
    notes = runtime.store.list_notes(principal.user_id, limit=10_000)
    edges = runtime.store.list_note_edges(principal.user_id)
    degree: dict[str, int] = {}
    for src, dst in edges:
        degree[src] = degree.get(src, 0) + 1
        degree[dst] = degree.get(dst, 0) + 1
    return Envelope(
        status="ok",
        data={
            "nodes": [
                {**_note_payload(n, content=False), "degree": degree.get(n.id, 0)}
                for n in notes
            ],
            "edges": [{"src": src, "dst": dst} for src, dst in edges],
        },
    )


@router.get("/notes/sweeps", response_model=Envelope, tags=["notes"])
async def list_sweep_reports(
    limit: int = 10, principal: AuthContext = Depends(get_user)
):
    """Past sweep reports, newest first — replay without re-spending calls."""
    runtime = get_runtime()
    await rate_limit(runtime, "read", principal.user_id)
    _require_notes_enabled(runtime)
    rows = runtime.store.list_sweep_reports(
        principal.user_id, limit=max(1, min(limit, 50))
    )
    return Envelope(
        status="ok",
        data={"sweeps": [
            {
                "id": r["id"],
                "created_at": r["created_at"].isoformat(),
                "contradictions": (r["report"] or {}).get("contradictions", 0),
                "evolutions": (r["report"] or {}).get("evolutions", 0),
                "judged": (r["report"] or {}).get("judged", 0),
                "report": r["report"],
            }
            for r in rows
        ]},
    )


@router.get("/notes/{note_id}", response_model=Envelope, tags=["notes"])
async def get_note(note_id: str, principal: AuthContext = Depends(get_user)):
    runtime = get_runtime()
    await rate_limit(runtime, "read", principal.user_id)
    _require_notes_enabled(runtime)
    note = _get_owned_note(runtime, note_id, principal)
    links = [
        _note_payload(n, content=False)
        for nid in runtime.store.list_note_links_from(note.id)
        if (n := runtime.store.get_note(nid))
    ]
    backlinks = [
        _note_payload(n, content=False)
        for nid in runtime.store.list_backlinks(note.id)
        if (n := runtime.store.get_note(nid))
    ]
    dangling = (note.meta or {}).get("dangling", [])
    return Envelope(
        status="ok",
        data={
            **_note_payload(note),
            "links": links,
            "backlinks": backlinks,
            "dangling": dangling,
        },
    )


@router.patch("/notes/{note_id}", response_model=Envelope, tags=["notes"])
async def update_note(
    note_id: str, body: NoteUpdateRequest, principal: AuthContext = Depends(get_user)
):
    runtime = get_runtime()
    await rate_limit(runtime, "write", principal.user_id)
    _require_notes_enabled(runtime)
    existing = _get_owned_note(runtime, note_id, principal)
    old_title = existing.title
    title = notes_service.normalize_title(body.title) if body.title else None
    try:
        note = runtime.store.update_note(note_id, title=title, content=body.content)
    except ConstraintViolation:
        raise http_error(
            "conflict", "a note with this title already exists", status_code=409
        )
    _save_note_graph(runtime, principal, note)
    if title and title.lower() != old_title.lower():
        # [[Old Title]] in other notes no longer resolves here; rebuild their
        # edges from their text so the graph matches what the notes say.
        notes_service.reresolve_note_sources(
            runtime.store, principal.user_id, runtime.store.list_backlinks(note.id)
        )
    return Envelope(status="ok", data=_note_payload(note))


@router.delete("/notes/{note_id}", response_model=Envelope, tags=["notes"])
async def delete_note(note_id: str, principal: AuthContext = Depends(get_user)):
    runtime = get_runtime()
    await rate_limit(runtime, "write", principal.user_id)
    _require_notes_enabled(runtime)
    _get_owned_note(runtime, note_id, principal)
    sources = runtime.store.list_backlinks(note_id)
    runtime.store.delete_note(note_id)
    # Sources' [[links]] to this title now dangle; record that, so recreating
    # the title later reconnects them.
    notes_service.reresolve_note_sources(runtime.store, principal.user_id, sources)
    return Envelope(status="ok", data={"deleted": True})


@router.post("/notes/search", response_model=Envelope, tags=["notes"])
async def search_notes(
    body: NoteSearchRequest, principal: AuthContext = Depends(get_user)
):
    runtime = get_runtime()
    await rate_limit(runtime, "read", principal.user_id)
    _require_notes_enabled(runtime)
    results = notes_service.search_notes(
        runtime.store,
        runtime.embeddings,
        principal.user_id,
        body.query,
        limit=body.limit,
    )
    return Envelope(
        status="ok",
        data={
            "results": [
                {
                    **_note_payload(note, content=False),
                    # Position, not a score. What search_notes returns is a
                    # fused rank value that tops out near 0.016 and packs the
                    # whole result set into a hair's breadth of itself — a
                    # number no client can render and none should try to.
                    "rank": position,
                    "excerpt": " ".join(note.content.split())[:300],
                }
                for position, (note, _score) in enumerate(results, start=1)
            ]
        },
    )


@router.post("/notes/sweep", response_model=Envelope, tags=["notes"])
async def sweep_vault(principal: AuthContext = Depends(get_user)):
    """Run the witness across the whole vault: the strongest pairs, judged.

    Deliberately expensive (up to 30 model calls), so the rate limit is
    stricter than the per-note witness.
    """
    runtime = get_runtime()
    _require_notes_enabled(runtime)
    await enforce(
        runtime, f"notes_sweep:{principal.user_id}", limit=2, window_seconds=600
    )
    report = await asyncio.to_thread(
        notes_service.vault_sweep,
        runtime.store,
        runtime.embeddings,
        runtime.llm,
        principal.user_id,
    )
    # A sweep costs up to 30 model calls; keep the result so a reload replays
    # it for free and the user has a record of what moved (SPEC 19.6).
    try:
        saved = await asyncio.to_thread(
            runtime.store.save_sweep_report, principal.user_id, report
        )
        report = {**report, "id": saved["id"],
                  "created_at": saved["created_at"].isoformat()}
    except Exception as exc:  # noqa: BLE001 - archiving must not fail a sweep
        logger.warning("sweep_report_save_failed", error=str(exc))
    return Envelope(status="ok", data=report)


@router.post("/notes/from-file", response_model=Envelope, status_code=201, tags=["notes"])
async def note_from_file(
    body: NoteFromFileRequest, principal: AuthContext = Depends(get_user)
):
    """Promote an uploaded file into the vault, explicitly.

    Per-conversation RAG over uploads is automatic; joining the user's
    permanent cross-conversation corpus is a deliberate act. This endpoint is
    that act: it copies the file's text into a note (provenance in meta), so
    the witness and note_search treat it like anything else the user wrote.
    """
    runtime = get_runtime()
    await rate_limit(runtime, "write", principal.user_id)
    _require_notes_enabled(runtime)
    from liminallm.service.attachments import attachment_path

    path = attachment_path(
        runtime.settings.shared_fs_root, principal.user_id, body.name
    )
    if not path or not path.is_file():
        raise http_error("not_found", "file not found", status_code=404)
    try:
        extracted = await asyncio.to_thread(
            extract_service.extract_text,
            path,
            llm=runtime.llm,
            readers=extract_service.parse_reader_order(
                runtime.settings.extract_readers
            ),
        )
    except extract_service.ExtractError as exc:
        raise http_error("bad_request", exc.reason, status_code=400)
    except OSError:
        raise http_error("bad_request", "file is not readable", status_code=400)
    text = extracted["text"]
    truncated = len(text.encode("utf-8")) > notes_service.NOTE_FROM_FILE_MAX_BYTES
    if truncated:
        text = text.encode("utf-8")[: notes_service.NOTE_FROM_FILE_MAX_BYTES].decode(
            "utf-8", errors="ignore"
        )

    base_title = notes_service.normalize_title(FilePath(body.name).stem) or "Imported file"
    title = base_title
    for suffix in range(2, 20):
        if not runtime.store.get_note_by_title(principal.user_id, title):
            break
        title = f"{base_title} ({suffix})"
    try:
        note = runtime.store.create_note(
            principal.user_id,
            title,
            text,
            meta={
                "source": "upload",
                "filename": body.name,
                "truncated": truncated,
                # "pdf"/"vision" tell the reader this is an extraction of the
                # file, not the file itself.
                "method": extracted["method"],
            },
        )
    except ConstraintViolation:
        raise http_error(
            "conflict", "a note with this title already exists", status_code=409
        )
    _save_note_graph(runtime, principal, note)
    return Envelope(
        status="ok",
        data={
            **_note_payload(note, content=False),
            "truncated": truncated,
            "method": extracted["method"],
        },
    )


@router.post("/notes/{note_id}/witness", response_model=Envelope, tags=["notes"])
async def witness_note(
    note_id: str,
    body: NoteWitnessRequest,
    principal: AuthContext = Depends(get_user),
):
    """Judge this note against the vault: one model call per candidate."""
    runtime = get_runtime()
    _require_notes_enabled(runtime)
    note = _get_owned_note(runtime, note_id, principal)
    await enforce(
        runtime, f"notes_witness:{principal.user_id}", limit=5, window_seconds=60
    )
    report = await asyncio.to_thread(
        notes_service.witness_report,
        runtime.store,
        runtime.embeddings,
        runtime.llm,
        principal.user_id,
        note,
        limit=body.limit,
    )
    return Envelope(status="ok", data=report)


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
    await enforce(
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
    # Clustering + skill promotion are best-effort: they call the LLM (cluster
    # labeling), which can be transiently unavailable, and the background
    # training worker performs the same work on a schedule. The feedback event
    # is already durably recorded above, so a downstream failure here must not
    # turn a successful POST into a 500.
    try:
        await runtime.clusterer.cluster_user_preferences(principal.user_id)
        runtime.clusterer.promote_skill_adapters()
    except Exception as exc:
        logger.warning(
            "preference_clustering_deferred",
            user_id=principal.user_id,
            error=str(exc),
        )
    refreshed = runtime.store.get_preference_event(event.id) or event
    resp = PreferenceEventResponse(
        id=event.id,
        cluster_id=refreshed.cluster_id,
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
    await enforce(
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
    await enforce(
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
    # Issue 46.5: Add upper bounds to prevent integer overflow/DoS
    page: int = Query(1, ge=1, le=100000, description="Page number (1-indexed)"),
    page_size: Optional[int] = Query(None, ge=1, le=1000, description="Items per page"),
    limit: Optional[int] = Query(None, ge=1, le=1000, description="Alias for page_size (for frontend compatibility)"),
    cursor: Optional[str] = Query(
        None, description="Keyset cursor returned by a previous response"
    ),
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
    await rate_limit(runtime, "read", principal.user_id)
    paging = _get_pagination_settings(runtime)
    kind_filter = kind or (type if type and "." in type else None)
    type_filter = type if type and "." not in type else None
    if not type_filter and kind_filter:
        type_filter = kind_filter.split(".", 1)[0]
    # Accept 'limit' as alias for 'page_size' for frontend compatibility
    effective_page_size = limit if limit is not None else (page_size or paging["default_page_size"])
    resolved_page_size = min(max(effective_page_size, 1), paging["max_page_size"])

    # Get one extra item to determine if there are more pages
    # Pass tenant_id for proper shared artifact visibility filtering
    raw_items = list(runtime.store.list_artifacts(
        type_filter=type_filter,
        kind_filter=kind_filter,
        owner_user_id=principal.user_id,
        tenant_id=principal.tenant_id,
        page=page,
        page_size=resolved_page_size,
        cursor=cursor,
        include_sentinel=True,
        visibility=visibility,
    ))

    has_next = len(raw_items) > resolved_page_size
    page_items = raw_items[:resolved_page_size]
    next_cursor = None
    if has_next and page_items:
        last_item = page_items[-1]
        next_cursor = encode_artifact_cursor(last_item.created_at, last_item.id)
    next_page = None if cursor else (page + 1 if has_next else None)

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
            next_page=next_page,
            next_cursor=next_cursor,
            page_size=resolved_page_size,
        ),
    )


@router.get("/artifacts/{artifact_id}", response_model=Envelope, tags=["artifacts"])
async def get_artifact(
    artifact_id: str = Path(..., max_length=255, description="Artifact identifier"),
    principal: AuthContext = Depends(get_user),
):
    runtime = get_runtime()
    await rate_limit(runtime, "read", principal.user_id)
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
    await rate_limit(runtime, "read", principal.user_id)
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
    await rate_limit(runtime, "read", principal.user_id)
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
    await rate_limit(runtime, "read", principal.user_id)
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
    await rate_limit(runtime, "read", principal.user_id)
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
    tool_id: str = Path(..., max_length=255, description="Tool artifact identifier"),
    body: ToolInvokeRequest = Body(...),
    principal: AuthContext = Depends(get_user),
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
):
    runtime = get_runtime()
    await rate_limit(runtime, "write", principal.user_id)
    # SPEC §18: Accept Idempotency-Key when provided (optional)
    async with idempotency.IdempotencyGuard(
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
        result = await runtime.workflow.invoke_tool(
            # The exact row this request authorized, not its name.
            runtime.workflow._describe_tool(artifact),
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
    await rate_limit(runtime, "write", principal.user_id)
    # SPEC §18: Accept Idempotency-Key when provided (optional)
    async with idempotency.IdempotencyGuard(
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
        # Publishing is a privilege, and the role comes off the authenticated
        # token — never from the body, which is the same rule `tenant_id`
        # lives under. A `shared` artifact is every account in the tenant's,
        # and a `global` one is the installation's: a globally visible `tool`
        # enters the registry every turn resolves against, and a globally
        # visible `mcp` server becomes a capability of every turn. Private
        # stays open to everybody, or this becomes an admin-only endpoint by
        # accident.
        if body.visibility != "private" and principal.role != "admin":
            raise http_error(
                "forbidden",
                "publishing an artifact requires admin access",
                status_code=403,
            )
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
            visibility=body.visibility,
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


@router.delete("/artifacts/{artifact_id}", response_model=Envelope, tags=["artifacts"])
async def delete_artifact(
    artifact_id: str = Path(..., max_length=255, description="Artifact identifier"),
    principal: AuthContext = Depends(get_user),
):
    """Retire a private artifact the caller owns.

    Its versions, config patches, router state and training jobs go with it by
    cascade. An adapter being trained right now is refused with 409 instead:
    the worker is writing weights and will try to promote a version onto this
    row, so the deletion has to lose that race rather than win it.

    The payloads are deliberately *not* removed here. A turn resolves an
    adapter from Postgres and only then touches disk, so unlinking inside this
    request lets a caller that legitimately acquired the artifact read a
    filesystem where it no longer exists — a state no serial order of the two
    produces. `service/artifacts.sweep_artifact_payloads` collects them once
    they have been orphans for longer than any request may live, which also
    keeps an `rmtree` of a large checkpoint tree off an API worker and makes
    an I/O failure a retry rather than a permanent orphan.
    """
    runtime = get_runtime()
    await rate_limit(runtime, "write", principal.user_id)
    _get_private_artifact(runtime, artifact_id, principal)

    artifact = runtime.store.delete_private_artifact(
        artifact_id, owner_user_id=principal.user_id
    )
    if artifact is None:
        raise http_error("not_found", "artifact not found", status_code=404)

    logger.info(
        "artifact_deleted",
        user_id=principal.user_id,
        artifact_id=artifact_id,
        artifact_type=artifact.type,
    )
    return Envelope(status="ok", data={"deleted": True})


@router.patch("/artifacts/{artifact_id}", response_model=Envelope, tags=["artifacts"])
async def patch_artifact(
    artifact_id: str = Path(..., max_length=255, description="Artifact identifier"),
    body: ArtifactPatchRequest = Body(...),
    principal: AuthContext = Depends(get_user),
):
    """Update an artifact via RFC 6902 JSON Patch or legacy schema update.

    Paths address the artifact's schema document itself — `/entrypoint`, not
    `/schema/entrypoint`. See `ArtifactPatchRequest` for why the second
    spelling was wrong here and is now refused rather than misapplied.

    Accepts:
    - RFC 6902 format: {"patch": [{"op": "replace", "path": "/foo", "value": "bar"}]}
    - Legacy format: {"schema": {...}, "description": "..."} (for backward compatibility)
    """
    runtime = get_runtime()
    await rate_limit(runtime, "write", principal.user_id)
    current = _get_private_artifact(runtime, artifact_id, principal)

    normalized = body.get_normalized_patch()

    def build_schema(locked: dict) -> dict:
        """Applied to the schema the store reads under the artifact lock.

        `current` above answers the request; it is not what the write is
        derived from. Computing the whole document here and handing it over
        made this lock serialize the write without covering the read behind
        it, so a change committed in between was replaced by a result built
        from the older row.

        The kind check moved in here with it, which also means a refusal
        happens inside the transaction and writes nothing. `current.type` is
        safe to close over: type is immutable through every mutation path.
        """
        if "ops" in normalized:
            # No `if ops:` guard: a falsy list used to skip the engine and go
            # straight to writing a version for a patch that named nothing.
            candidate = json_patch.apply_ops(locked, normalized["ops"])
        elif "schema_update" in normalized:
            candidate = json_patch.deep_merge(locked, normalized["schema_update"])
        else:
            candidate = locked

        schema_kind = candidate.get("kind")
        if schema_kind and not schema_kind.startswith(f"{current.type}."):
            raise BadRequestError(
                "kind must start with the type prefix",
                detail={"kind": schema_kind, "type": current.type},
            )
        return candidate

    # None when the request did not ask for one, which the store reads as
    # "leave it alone". Passing back the description we read would revert a
    # concurrent change to it — the same staleness as the schema, one column
    # over.
    new_description = normalized.get("description")

    # The private-owner predicate travels into the statement that takes the
    # lock. `_get_private_artifact` above answers the request; this is what
    # makes the answer still true at the moment of the write.
    artifact = runtime.store.update_private_artifact(
        artifact_id,
        build_schema,
        new_description,
        owner_user_id=principal.user_id,
        version_author=principal.user_id,
    )
    if not artifact:
        raise NotFoundError("artifact not found", detail={"artifact_id": artifact_id})

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
    # Rate limit configops (database-managed setting, SPEC §18.6)
    await rate_limit(runtime, "configops", principal.user_id)
    proposer = "human_admin" if principal.role == "admin" else "user"
    audit = runtime.store.record_config_patch(
        artifact_id=body.artifact_id,
        proposer=proposer,
        patch=body.patch,
        justification=body.justification,
    )
    resp = ConfigPatchAuditResponse.model_validate(audit)
    return Envelope(status="ok", data=resp)


@router.get("/config/patches", response_model=Envelope, tags=["config"])
async def list_config_patches(
    status: Optional[str] = None, principal: AuthContext = Depends(get_admin_user)
):
    runtime = get_runtime()
    # Rate limit configops (database-managed setting, SPEC §18.6)
    await rate_limit(runtime, "configops", principal.user_id)
    patches = runtime.store.list_config_patches(status)
    items = [
        ConfigPatchAuditResponse.model_validate(p)
        for p in patches
    ]
    return Envelope(status="ok", data=ConfigPatchListResponse(items=items))


@router.post("/config/patches/{patch_id}/decide", response_model=Envelope, tags=["config"])
async def decide_config_patch(
    patch_id: int = Path(..., ge=1, description="Config patch identifier"),
    body: ConfigPatchDecisionRequest = Body(...),
    principal: AuthContext = Depends(get_admin_user),
):
    runtime = get_runtime()
    # Rate limit configops (database-managed setting, SPEC §18.6)
    await rate_limit(runtime, "configops", principal.user_id)
    decision = runtime.config_ops.decide_patch(patch_id, body.decision, body.reason)
    resp = ConfigPatchAuditResponse.model_validate(decision)
    return Envelope(status="ok", data=resp)


@router.post("/config/patches/{patch_id}/apply", response_model=Envelope, tags=["config"])
async def apply_config_patch(
    patch_id: int = Path(..., ge=1, description="Config patch identifier"),
    principal: AuthContext = Depends(get_admin_user)
):
    runtime = get_runtime()
    # Rate limit configops (database-managed setting, SPEC §18.6)
    await rate_limit(runtime, "configops", principal.user_id)
    result = runtime.config_ops.apply_patch(
        patch_id, approver_user_id=principal.user_id
    )
    patch = result.get("patch")
    resp = ConfigPatchAuditResponse.model_validate(patch).model_copy(
        update={
            # True by definition here, and the stored row may not have caught up.
            "status": "applied",
            "applied_at": patch.applied_at or patch.decided_at or patch.created_at,
        }
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
    # Rate limit configops (database-managed setting, SPEC §18.6)
    await rate_limit(runtime, "configops", principal.user_id)
    audit = runtime.config_ops.auto_generate_patch(
        body.artifact_id, principal.user_id, goal=body.goal
    )
    resp = ConfigPatchAuditResponse.model_validate(audit)
    return Envelope(status="ok", data=resp)


@router.get("/config", response_model=Envelope, tags=["config"])
async def get_config(principal: AuthContext = Depends(get_admin_user)):
    """Expose runtime configuration for the admin console."""
    runtime = get_runtime()
    await rate_limit(runtime, "admin:read", principal.user_id)

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

    settings = get_settings().model_dump()
    return Envelope(status="ok", data={"settings": _sanitize_dict(settings)})


@router.get("/admin/settings", response_model=Envelope, tags=["admin"])
async def get_system_settings(principal: AuthContext = Depends(get_admin_user)):
    """Get admin-managed system settings.

    Returns settings for session rotation, concurrency caps, and rate limit
    multipliers that can be modified via the admin UI.
    """
    runtime = get_runtime()
    await rate_limit(runtime, "admin:read", principal.user_id)
    return Envelope(status="ok", data=runtime.store.get_system_settings())


@router.get("/admin/settings/schema", response_model=Envelope, tags=["admin"])
async def get_system_settings_schema(
    principal: AuthContext = Depends(get_admin_user),
):
    """Describe every managed setting: type, bounds, choices, default, group.

    The admin console renders itself from this rather than hard-coding the
    field list, so a setting cannot exist in the model and be missing from the
    console, and the control's limits are the ones the API enforces.
    """
    runtime = get_runtime()
    await rate_limit(runtime, "admin:read", principal.user_id)
    stored = runtime.store.get_system_settings_overrides()
    fields = managed_settings_schema()
    for entry in fields:
        # "Changed from the shipped default" is what an operator wants to see
        # at a glance; it is also what actually lives in the database.
        entry["overridden"] = entry["name"] in stored
        if entry.get("secret"):
            # Whether a secret is set is useful; its value never leaves here.
            entry["default"] = ""
            entry["is_set"] = bool(stored.get(entry["name"]))
        if entry["name"] == "rag_rerank" and (
            getattr(runtime.rag.rerank, "transport", None) == "prose"
        ):
            # A runtime fact, not schema: reranking is on, and this backend
            # cannot carry the verdict out-of-band, so it is parsed from
            # reply text — the degraded transport. The one person who can fix
            # that is the admin reading this screen.
            entry["warning"] = (
                "Active on a backend without tool calling: the verdict is "
                "parsed from reply text, the degraded transport. Use a "
                "tool-calling backend, or turn reranking off."
            )
    return Envelope(status="ok", data={"fields": fields})


@router.put("/admin/settings", response_model=Envelope, tags=["admin"])
async def update_system_settings(
    body: dict,
    principal: AuthContext = Depends(get_admin_user),
):
    """Update admin-managed system settings.

    Allows admins to modify operational parameters without restarting the application.

    Supported settings (all have sensible defaults):
    - Session: session_rotation_hours (24), session_rotation_grace_seconds (300)
    - Concurrency: max_concurrent_workflows (3), max_concurrent_inference (2)
    - Rate multipliers: rate_limit_multiplier_free (1.0), _paid (2.0), _enterprise (5.0)
    - Rate limits: chat_rate_limit_per_minute (60), chat_rate_limit_window_seconds (60),
      login_rate_limit_per_minute (10), signup_rate_limit_per_minute (5),
      reset_rate_limit_per_minute (5), mfa_rate_limit_per_minute (5),
      admin_rate_limit_per_minute (30), admin_rate_limit_window_seconds (60),
      files_upload_rate_limit_per_minute (10), configops_rate_limit_per_hour (30),
      read_rate_limit_per_minute (120)
    - Pagination: default_page_size (100), max_page_size (500), default_conversations_limit (50)
    - Files: max_upload_bytes (10485760), rag_chunk_size (400)
    - Token TTL: access_token_ttl_minutes (30), refresh_token_ttl_minutes (1440)
    - Feature flags: enable_mfa (true), allow_signup (true)
    - Training worker: training_worker_enabled (true), training_worker_poll_interval (60)
    - SMTP: smtp_host, smtp_port (587), smtp_user, smtp_password, smtp_security (starttls),
      email_from_address, email_from_name ("LiminalLM")
    - URL settings: oauth_redirect_uri, app_base_url ("http://localhost:8000")
    - Voice: voice_transcription_model (whisper-1), voice_synthesis_model (tts-1, tts-1-hd),
      voice_default_voice (alloy, echo, fable, onyx, nova, shimmer)
    - Model: embedding_model_id (text-embedding, text-embedding-3-small,
      text-embedding-3-large, text-embedding-ada-002), model_path ("gpt-4o-mini" with suggestions),
      model_backend (openai, azure, together, etc.), default_adapter_mode (local, remote, prompt, hybrid)
    - Tenant: default_tenant_id ("public"), jwt_issuer ("liminallm"), jwt_audience ("liminal-clients")
    """
    runtime = get_runtime()
    await rate_limit(runtime, "admin:write", principal.user_id)

    # A blank secret means the operator did not retype it, not that they want
    # it cleared. The console cannot show the current value, so it cannot
    # round-trip one either.
    secrets = secret_setting_names()
    body = {
        key: value
        for key, value in body.items()
        if not (key in secrets and value == "")
    }
    if not body:
        return Envelope(status="ok", data=runtime.store.get_system_settings())

    # Types, bounds and allowed values are declared on the Settings fields;
    # this checks the patch against them rather than restating them here.
    errors = validate_managed_settings(
        body, runtime.store.get_system_settings_raw()
    )
    if errors:
        raise http_error(
            "validation_error",
            "; ".join(f"{key}: {message}" for key, message in sorted(errors.items())),
            status_code=400,
            details={"fields": errors},
        )

    updated = runtime.store.set_system_settings(body)

    # Rebuild backend-dependent services so a changed model_backend / model_path
    # (or rag/embedding setting) takes effect without a process restart.
    # The same list the rebuild signature is built from, so the two cannot
    # disagree about which settings need one (service/runtime.py).
    if set(MODEL_AFFECTING_SETTINGS) & set(body.keys()):
        try:
            # Off the event loop: the rebuild is synchronous and takes the
            # reload lock, which the background watcher may briefly hold.
            await asyncio.to_thread(runtime.reload_model_services)
        except Exception as exc:
            # Settings are persisted, but the live stack could not be rebuilt.
            # Surface this instead of reporting success so the caller knows the
            # change is not yet active. The persisted settings make a retry
            # (or restart) apply them; the runtime keeps the previous backend.
            logger.error(
                "model_services_reload_failed",
                admin_user_id=principal.user_id,
                error=str(exc),
            )
            raise http_error(
                "server_error",
                "Settings were saved but the model services failed to reload; the "
                "previous backend remains active. Retry, or restart to apply them.",
                status_code=500,
            )
    else:
        # Non-structural settings still have to reach THIS worker's
        # runtime.settings before the response returns. The polling watcher
        # exists for the *other* workers; leaning on it here made "the change
        # is live" true only after an interval nobody promised the admin —
        # the reranker reads rag_rerank per retrieval, but per-retrieval
        # reads of a stale object are still stale.
        runtime.refresh_settings()

    logger.info(
        "system_settings_updated",
        admin_user_id=principal.user_id,
        updated_keys=list(body.keys()),
    )
    return Envelope(status="ok", data=updated)


@router.patch("/admin/settings", response_model=Envelope, tags=["admin"])
async def patch_system_settings(
    body: dict,
    principal: AuthContext = Depends(get_admin_user),
):
    """Partial update of admin-managed system settings.

    Alias for PUT - both methods perform partial updates (only provided keys are changed).
    PATCH is provided for REST convention compatibility with frontend clients.
    """
    return await update_system_settings(body, principal)


@router.get("/files/limits", response_model=Envelope, tags=["files"])
async def get_file_limits(principal: AuthContext = Depends(get_user)):
    runtime = get_runtime()
    await rate_limit(runtime, "read", principal.user_id)
    # Issue 4.3: Return per-plan upload limits (SPEC §18)
    user = runtime.store.get_user(principal.user_id)
    plan_tier = user.plan_tier if user else "free"
    return Envelope(
        status="ok",
        data={
            "max_upload_bytes": plan_upload_limit(runtime, plan_tier),
            "plan_tier": plan_tier,
            "allowed_extensions": sorted(ALLOWED_UPLOAD_EXTENSIONS),
        },
    )


@router.post("/files/upload", response_model=Envelope, tags=["files"])
async def upload_file(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    context_id: Optional[str] = Form(None, max_length=255),
    conversation_id: Optional[str] = Form(None, max_length=255),
    chunk_size: Optional[int] = Form(None, ge=64, le=4000),
    principal: AuthContext = Depends(get_user),
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
):
    """Upload a file, optionally attaching it to a conversation.

    With `conversation_id` the file becomes usable in that chat immediately —
    small text files are injected into the prompt, larger documents are chunked
    into the conversation's implicit context for the model's file_search tool,
    and everything is readable from the code interpreter. No context needs to
    be created or selected by the user.
    """
    import re

    runtime = get_runtime()
    # SPEC §18: Accept Idempotency-Key when provided (optional)
    async with idempotency.IdempotencyGuard(
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
        ext = FilePath(safe_filename).suffix.lower()
        if ext not in ALLOWED_UPLOAD_EXTENSIONS:
            raise http_error(
                "validation_error",
                f"unsupported file extension: {ext or 'none'}",
                status_code=400,
            )
        dest_dir = (
            FilePath(runtime.settings.shared_fs_root)
            / "users"
            / principal.user_id
            / "files"
        )
        # Issue 4.3: Enforce per-plan file size limits (SPEC §18)
        user = runtime.store.get_user(principal.user_id)
        plan_tier = user.plan_tier if user else "free"
        max_bytes = max(1, plan_upload_limit(runtime, plan_tier))
        # Rate limit before buffering the entire payload to avoid pre-limit memory spikes
        declared_size = None
        if file.headers:
            try:
                declared_size = int(file.headers.get("content-length"))
            except (TypeError, ValueError):
                declared_size = None
        # Pre-charge from the declared size when the client sends one; otherwise
        # charge a nominal 1 and top up by the real cost after reading (below).
        # Multipart part headers usually omit content-length, so defaulting to
        # max_bytes here would make every such upload exceed the bucket.
        approx_cost = (
            max(1, math.ceil(declared_size / (256 * 1024))) if declared_size else 1
        )
        await enforce(
            runtime,
            f"files:upload:{principal.user_id}",
            runtime.settings.files_upload_rate_limit_per_minute,
            60,
            cost=approx_cost,
        )
        contents = await file.read(max_bytes + 1)
        actual_cost = max(1, math.ceil(len(contents) / (256 * 1024)))
        if actual_cost > approx_cost:
            await enforce(
                runtime,
                f"files:upload:{principal.user_id}",
                runtime.settings.files_upload_rate_limit_per_minute,
                60,
                cost=actual_cost - approx_cost,
            )
        if len(contents) > max_bytes:
            raise http_error("validation_error", f"file too large (max {max_bytes // 1024 // 1024}MB for {plan_tier} plan)", status_code=413)
        detected_mime = file.content_type or mimetypes.guess_type(safe_filename)[0]
        if not detected_mime or detected_mime == "application/octet-stream":
            raise http_error("validation_error", "unknown or unsupported MIME type", status_code=400)
        checksum = hashlib.sha256(contents).hexdigest()
        dest_path = safe_join(dest_dir, safe_filename)
        resolved_dest = dest_path.resolve()
        if (
            dest_dir.resolve() not in resolved_dest.parents
            and dest_dir.resolve() != resolved_dest
        ):
            raise http_error("validation_error", "invalid file path", status_code=400)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        manifest_path = dest_dir / ".checksums.json"

        def _read_manifest() -> tuple[dict, Optional[str], set[str]]:
            """The manifest as of now. Only meaningful under the lock."""
            existing: dict[str, Any] = {}
            if manifest_path.exists():
                try:
                    existing = json.loads(manifest_path.read_text())
                except ValueError:
                    # Corrupt content reads as empty, and the write below
                    # rebuilds the file from this upload. An OSError is not
                    # corruption: it says nothing about what the manifest
                    # holds, and treating it as an empty manifest would make
                    # the write replace every other name's entry with
                    # nothing. `UnicodeDecodeError` is a `ValueError`, so
                    # binary rubbish counts as corrupt too.
                    logger.warning(
                        "file_checksum_manifest_read_failed", path=str(manifest_path)
                    )
                if not isinstance(existing, dict):
                    existing = {}
            entry = existing.get(safe_filename)
            checksum_before: Optional[str] = None
            contexts: set[str] = set()
            if isinstance(entry, dict):
                checksum_before = entry.get("checksum")
                try:
                    contexts = set(
                        str(c) for c in entry.get("contexts", []) if c is not None
                    )
                except Exception:
                    contexts = set()
            elif isinstance(entry, str):
                checksum_before = entry
            return existing, checksum_before, contexts
        # Archives are binary; they are stored but never text-ingested.
        # POST /files/{name}/extract expands them (with ingestion) instead.
        # (Attaching an archive to a conversation is fine — it is handed to the
        # code interpreter, never chunked, so context_id stays unset below.)
        if context_id and is_archive_filename(safe_filename):
            raise http_error(
                "validation_error",
                "archives cannot be ingested directly; upload without context_id, "
                "then extract",
                status_code=400,
            )

        # Before anything durable happens. `_publish` used to reach this
        # check inside its ingestion step, which runs after the pathname has
        # already been replaced — so a request refused for naming a context
        # that does not exist had overwritten the file first, and the failure
        # handler then unlinked it. Measured: the pathname was gone while the
        # manifest and the chunks still described the generation it used to
        # hold, and the user was told their request was refused.
        #
        # A parameter the route will refuse is knowable before any mutation,
        # so it is refused there. The conversation branch below creates its
        # own context, which needs no checking.
        if context_id:
            _get_owned_context(runtime, context_id, principal)

        # Attach-to-conversation: classify the file, and route searchable ones
        # into the conversation's implicit context so file_search can find them.
        attachment_caps: Optional[dict] = None
        #: True when `context_id` below is a conversation's implicit index
        #: rather than something the caller named.
        implicit_context = False
        if conversation_id and not context_id:
            _get_owned_conversation(runtime, conversation_id, principal)
            attachment_caps = classify_attachment(safe_filename, len(contents))
            if attachment_caps["searchable"]:
                context_id = ensure_conversation_context(
                    runtime.store,
                    user_id=principal.user_id,
                    conversation_id=conversation_id,
                ).id
                implicit_context = True

        def _record(chunks: Optional[int]) -> Optional[dict]:
            """Persist the attachment record on the conversation."""
            if not (conversation_id and attachment_caps):
                return None
            records = record_attachment(
                runtime.store,
                conversation_id=conversation_id,
                user_id=principal.user_id,
                name=safe_filename,
                size=len(contents),
                capabilities=attachment_caps,
                chunk_count=chunks,
                # Which bytes this chat was given. The name it was given
                # them under belongs to every later upload of that name too.
                checksum=checksum,
                # Recording and retiring are one transaction under the
                # conversation's row lock.
                fs_root=runtime.settings.shared_fs_root,
                prune_context_id=find_conversation_context_id(
                    runtime.store,
                    user_id=principal.user_id,
                    conversation_id=conversation_id,
                ),
            )
            logger.info(
                "conversation_attachment_added",
                user_id=principal.user_id,
                conversation_id=conversation_id,
                filename=safe_filename,
                inline=attachment_caps["inline"],
                searchable=attachment_caps["searchable"],
                analyzable=attachment_caps["analyzable"],
            )
            return next((r for r in records if r.get("name") == safe_filename), None)
        # Contexts this upload owes a re-read, filled in under the lock and
        # drained after the response. A set, because the same context can be
        # reached by more than one source row.
        queued_reindex: set = set()

        def _publish() -> Optional[int]:
            """Put this upload on disk, in the index, and in the manifest.

            One critical section, because the three are one generation. Each
            step alone was already atomic and that was not enough: measured,
            two uploads of one name left the second's bytes on disk, the
            second's chunks in the index, and the *first's* checksum in the
            manifest, with both requests returning 200. Between this upload's
            write and its own read-back another can replace the file, and the
            re-read is what indexes somebody else's bytes under this one's
            provenance.

            The manifest is read *here* rather than earlier for the same
            reason: it is a read-modify-write, and a copy taken before the
            lock describes a generation that may already be gone.

            Synchronous, and run in one thread, so the lock is taken and
            released off the event loop and never spans an await.
            """
            existing_checksums, prior_checksum, prior_contexts = _read_manifest()
            # The manifest nominates a dedupe hit; the disk confirms it. The
            # record can outlive the bytes it describes — a publication that
            # failed after writing them and was never retried leaves exactly
            # that — and when the record alone decides, re-uploading the bytes
            # it names skips the write and reports success over a file holding
            # something else. That is the false dedupe hit 2E.1 removed,
            # reached this time by abandoning a failed request rather than by
            # racing. Hashing only happens when the record already claims a
            # match, so an ordinary upload of new bytes pays nothing.
            deduped = (
                prior_checksum == checksum
                and dest_path.exists()
                and file_digest(dest_path) == checksum
            )
            chunks: Optional[int] = None

            # A conversation attachment keeps its own copy of these bytes.
            # The chat was given *this* generation, and `/files/{name}` is a
            # name every later upload of it also owns — so what the chat can
            # read has to stop depending on that name. Stored before anything
            # names it, and unreferenced copies are reclaimed by the sweep.
            generation: Optional[FilePath] = None
            if attachment_caps:
                generation = store_generation(
                    runtime.settings.shared_fs_root,
                    principal.user_id,
                    contents,
                    checksum,
                )
            # What this upload's chunks will claim to be the contents of. For
            # an attachment that is the immutable generation, so a later
            # upload of the same filename cannot make the conversation's
            # index describe something the chat was never given.
            indexed_path = generation if generation is not None else dest_path
            # What this upload's chunks are called. For an attachment that is
            # a *reading* of the object rather than the object itself: the
            # same bytes attached under two names parse two ways, and keying
            # the index by the object made the second reading replace the
            # first.
            indexed_identity = (
                generation_key(checksum, safe_filename)
                if generation is not None
                else str(dest_path)
            )

            if not deduped:
                # The IdempotencyGuard already claimed an in-progress slot on
                # entry, so a concurrent duplicate request is rejected with 409
                # while this write proceeds. The slot records that the request
                # was entered; the guards here record which of its mutations
                # actually landed — this upload writes bytes and then ingests
                # them, and those are two facts, not one.
                with idem.commit(
                    "files.write", {"path": safe_filename, "checksum": checksum}
                ) as operation:
                    if not operation.replayable:
                        # Staged beside the destination, then renamed onto it.
                        # `write_bytes` truncates the file in place, so a
                        # download that already opened it saw the old bytes
                        # end mid-stream — measured, a reader got eight bytes
                        # and then EOF, which is neither generation. A rename
                        # replaces the *name*: an open descriptor keeps the
                        # inode it has, and the next open gets the new one.
                        # The signed URL naming a path rather than a
                        # generation is the standing reading; it may resolve
                        # to either, and never to a torn one.
                        staged = dest_path.with_name(
                            f".{dest_path.name}.{uuid4().hex[:8]}.part"
                        )
                        try:
                            staged.write_bytes(contents)
                            os.replace(staged, dest_path)
                        finally:
                            staged.unlink(missing_ok=True)

                # The bytes changed, so every context that described the
                # previous ones is now describing a file that is gone. Upload
                # already stops *recording* them — the manifest's context set
                # starts empty for a new checksum — and left their chunks in
                # place, which is the record forgetting them while the index
                # does not. Measured, with no race at all: two uploads of one
                # name into two contexts left the first context answering
                # with the first generation's text.
                #
                # Asked of the database, not of `prior_contexts`. The manifest
                # records only the contexts an upload named, and a context can
                # acquire a path through `POST /contexts/{id}/sources` without
                # ever appearing there — so a sweep driven by the manifest
                # walked straight past it. The chunks are what claim to be
                # this path's contents, so they are the reverse index.
                #
                # Emptied here, refreshed out of band. `_commit_generation`
                # states the reason for the emptying: these chunks claim to be
                # the contents of this path, so once new bytes exist the claim
                # is false, and "this path has nothing to say" is an answer
                # about the current bytes.
                #
                # But emptying alone answers only half of it. Replacing bytes
                # changes the file's generation, not which contexts cover it,
                # and a context that keeps its `context_source` row while its
                # chunks are gone has not been corrected — it has lost the
                # file. Re-ingesting for every covering context inside this
                # lock is genuinely unbounded work the request never chose, so
                # what happens here is the bounded half: empty now, record the
                # re-read, and let the queue do it. The path is briefly absent
                # from those contexts and then back, rather than gone.
                runtime.store.invalidate_path_in_other_contexts(
                    principal.user_id,
                    str(dest_path),
                    keep_context_id=context_id,
                )
                for target in runtime.store.contexts_covering_path(
                    str(dest_path), owner_user_id=principal.user_id
                ):
                    if target == context_id:
                        continue  # ingested in this request, below
                    runtime.store.enqueue_ingest_job(
                        target, str(dest_path), checksum
                    )
                    queued_reindex.add(target)

            def _persist(contexts: set) -> None:
                """Record this generation's checksum and its context set.

                Under a second lock, and re-read rather than reusing the copy
                above, because this file is one JSON object for every name in
                the directory: an upload of *another* name takes a different
                file lock, runs alongside, and its read-modify-write is built
                from a snapshot taken before this one landed. Measured, that
                dropped the other upload's entry entirely — and a missing
                entry is a dedupe miss, so the next upload of that name
                re-ingests a file that never changed.

                Always file lock then manifest lock, never the reverse: one
                order for two locks is what stops two uploads from each
                holding what the other is waiting for.

                Not best effort. The manifest is one of the three records
                that have to describe one generation, and a swallowed failure
                leaves it naming the previous checksum — recoverable only
                because the dedupe check now confirms against the disk.
                """
                with path_lock(runtime.settings.shared_fs_root, str(manifest_path)):
                    current, _, _ = _read_manifest()
                    current[safe_filename] = {
                        "checksum": checksum,
                        "contexts": sorted(contexts),
                    }
                    manifest_path.write_text(json.dumps(current, indent=2))

            wants_ingest = bool(context_id) and (
                not deduped or context_id not in prior_contexts
            )
            if context_id and generation is None:
                # Naming a context on an upload is how a file joins one, so
                # record it where coverage is read from. Left implicit, the
                # relationship would survive only as long as the chunks this
                # ingest happens to leave behind. Attachments are excluded:
                # their context is a conversation's implicit index, which
                # §19.5 keeps out of path-following coverage entirely.
                runtime.store.ensure_context_source(context_id, str(dest_path))
            if wants_ingest:
                try:
                    # Ingestion is its own mutation: the bytes can be on disk
                    # with the chunks not yet written, and a retry must be
                    # able to tell.
                    with idem.commit(
                        "files.ingest",
                        {"path": safe_filename, "context_id": context_id},
                    ) as operation:
                        if operation.replayable:
                            chunks = operation.result
                        else:
                            chunks = runtime.rag.ingest_file(
                                context_id,
                                str(indexed_path),
                                chunk_size=chunk_size,
                                source_identity=indexed_identity,
                                # The generation is named by its digest, so
                                # the extractor is told what the file is
                                # called separately from where it is kept.
                                format_name=safe_filename,
                            )
                            operation.result = chunks
                except Exception:
                    # The new bytes stay. Unlinking them does not restore what
                    # they replaced — those bytes are already gone — so it only
                    # removes the pathname while the manifest and the index go
                    # on describing a generation no file has. What exists by
                    # here is this generation, so it is what gets recorded,
                    # with the context whose ingestion failed left out of the
                    # set. A retry under the same key finds the bytes in place
                    # and re-runs only the ingestion.
                    logger.warning(
                        "file_upload_ingest_failed",
                        user_id=principal.user_id,
                        filename=safe_filename,
                        context_id=context_id,
                    )
                    # The target context is the one case the invalidation
                    # above deliberately skips, because it was about to
                    # receive this generation. It did not, so whatever it
                    # still says about this path describes bytes that are
                    # gone.
                    runtime.store.replace_chunks_for_path(
                        context_id, indexed_identity, []
                    )
                    # And record the re-read, for the same reason every other
                    # covering context gets one. This context was skipped in
                    # the enqueue above because it was about to be ingested
                    # here; it was not, and the source row written before the
                    # attempt still says it covers this path. A context that
                    # covers a path and says nothing about it is the silent
                    # coverage loss this queue exists to prevent, so leaving
                    # it out would reintroduce that defect through the one
                    # path that skips the queue.
                    #
                    # The worker's poll is what refills it: this request is
                    # about to abort, so nothing it scheduled will run.
                    # Attachments are excluded exactly as above — no source
                    # row was written for one, so nothing claims coverage.
                    if generation is None:
                        with contextlib.suppress(Exception):
                            runtime.store.enqueue_ingest_job(
                                context_id, str(dest_path), checksum
                            )
                    _persist(set(prior_contexts) if deduped else set())
                    raise

            # Persist checksum manifest for deduplication (SPEC §2.5).
            if not deduped or wants_ingest:
                contexts = set(prior_contexts) if deduped else set()
                if context_id:
                    contexts.add(context_id)
                _persist(contexts)
            # Inside the lock, because the record says how large this file is
            # and therefore how the conversation may use it — §19.5 makes
            # inline/searchable/analyzable part of that. Written after the
            # lock was released, the loser's classification could land last,
            # and `read_inline_contents` would then open the winner's bytes
            # under the loser's rules. Measured: the conversation said 6000
            # bytes while the disk held 24000.
            return chunks, _record(chunks)

        def _locked_publish() -> tuple[Optional[int], Optional[dict]]:
            with path_lock(
                runtime.settings.shared_fs_root,
                namespace_key(dest_dir, safe_filename),
            ):
                if not attachment_caps:
                    return _publish()
                # An attachment also holds its generation still, from before
                # the object is created or reused until the record naming it
                # is durable. `store_generation` adopts an existing object
                # without touching its age, so an object old enough to be
                # swept can be adopted — and the sweep then unlinked it
                # during the very attachment that was adopting it. Namespace
                # lock first, generation lock second; the sweep takes only
                # the second, so the two orders cannot meet.
                with generation_lock(
                    runtime.settings.shared_fs_root, principal.user_id, checksum
                ):
                    return _publish()

        try:
            chunk_count, attachment = await asyncio.to_thread(_locked_publish)
        except PathLockTimeout as exc:
            raise http_error("conflict", str(exc), status_code=409)

        # Out of band: after the response, off the request's critical path,
        # and bounded per pass. The worker drains whatever this leaves.
        #
        # No `chunk_size`: a drain takes whatever jobs are due, including
        # other uploads' jobs, and this request's tuning parameter is not
        # theirs. The queue uses the configured default, so a job produces
        # the same chunks whether this request drains it or the worker does.
        if queued_reindex:
            background.add_task(
                ingest_queue.drain,
                runtime.store,
                runtime.rag,
                fs_root=runtime.settings.shared_fs_root,
            )

        # Update idempotency with final result
        resp = FileUploadResponse(
            fs_path=safe_filename,
            # Not the implicit one. Nothing accepts a conversation's index as
            # a named context any more, so publishing its id buys a client
            # nothing — `chunk_count` already says the file was indexed — and
            # an identifier nobody needs is one more thing to keep refusing.
            context_id=None if implicit_context else context_id,
            chunk_count=chunk_count,
            attachment=attachment,
        )
        envelope = Envelope(status="ok", data=resp, request_id=idem.request_id)
        await idem.store_result(envelope)
        return envelope


@router.get(
    "/conversations/{conversation_id}/attachments", response_model=Envelope, tags=["conversations"]
)
async def list_conversation_attachments(
    conversation_id: str = Path(..., max_length=255),
    principal: AuthContext = Depends(get_user),
):
    """Files attached to a conversation, with how the model can reach each."""
    runtime = get_runtime()
    await rate_limit(runtime, "read", principal.user_id)
    conversation = _get_owned_conversation(runtime, conversation_id, principal)
    return Envelope(status="ok", data={"items": list_attachments(conversation)})


# =========================================================================
# File Download Endpoints (SPEC §13.3, §18)
# =========================================================================


@router.get("/files", response_model=Envelope, tags=["files"])
async def list_files(
    limit: Optional[int] = Query(None, ge=1, le=100, description="Maximum files to return"),
    offset: Optional[int] = Query(0, ge=0, description="Offset for pagination"),
    principal: AuthContext = Depends(get_user),
):
    """List user's uploaded files with pagination.

    SPEC §13.3: GET /v1/files - list user files (paginated)
    """
    runtime = get_runtime()
    await rate_limit(runtime, "read", principal.user_id)
    # Get user's file directory
    files_dir = (
        FilePath(runtime.settings.shared_fs_root)
        / "users"
        / principal.user_id
        / "files"
    )

    if not files_dir.exists():
        return Envelope(
            status="ok",
            data={"files": [], "total": 0, "limit": limit or 50, "offset": offset},
        )

    # List files with pagination, walking extracted-archive subdirectories.
    # Hidden components are internal bookkeeping (e.g. the .checksums.json
    # upload manifest) — uploads and extraction strip leading dots, so users
    # can never own a dotfile here.
    all_files = []
    try:
        for f in files_dir.rglob("*"):
            # One stat, and a disappearance is not an error. A listing is
            # observational: `is_file()` followed by `stat()` is two questions
            # about a name, and a delete landing between them raised
            # FileNotFoundError out of a route that only caught
            # PermissionError. Nothing here needs a lock — it needs to accept
            # that what it saw a moment ago may be gone.
            try:
                info = f.stat()
            except (FileNotFoundError, NotADirectoryError, PermissionError, OSError):
                continue
            if not stat.S_ISREG(info.st_mode):
                continue
            rel = f.relative_to(files_dir)
            if is_internal_path(rel):
                continue
            all_files.append({
                "name": rel.as_posix(),
                "size": info.st_size,
                "created_at": datetime.fromtimestamp(info.st_ctime, tz=timezone.utc).isoformat(),
                "modified_at": datetime.fromtimestamp(info.st_mtime, tz=timezone.utc).isoformat(),
            })
    except PermissionError:
        logger.warning("files_list_permission_denied", user_id=principal.user_id)
        return Envelope(status="ok", data={"files": [], "total": 0, "limit": limit or 50, "offset": offset})

    # Sort by modified_at descending
    all_files.sort(key=lambda x: x["modified_at"], reverse=True)

    # Apply pagination
    resolved_limit = limit or 50
    paginated = all_files[offset:offset + resolved_limit]

    return Envelope(
        status="ok",
        data={
            "files": paginated,
            "total": len(all_files),
            "limit": resolved_limit,
            "offset": offset,
            "has_next": offset + resolved_limit < len(all_files),
        },
    )


def _is_hidden_relpath(filename: str) -> bool:
    """True when any component of a user-supplied relative path is hidden.

    Kept as a name this module reads well with; the rule itself is
    `service.fs.is_internal_path`, shared with corpus ingestion so that a
    path this API treats as absent cannot become a document.
    """
    return is_internal_path(filename)


@router.get("/files/{filename:path}/url", response_model=Envelope, tags=["files"])
async def get_file_download_url(
    filename: str = Path(..., max_length=512, description="File name to download"),
    principal: AuthContext = Depends(get_user),
):
    """Get a signed download URL for a file.

    SPEC §18: Downloads use signed URLs with 10m expiry and content-disposition
    set to prevent inline execution.
    """
    runtime = get_runtime()
    await rate_limit(runtime, "read", principal.user_id)

    # Verify file exists
    files_dir = (
        FilePath(runtime.settings.shared_fs_root)
        / "users"
        / principal.user_id
        / "files"
    )
    try:
        file_path = safe_join(files_dir, filename)
    except PathTraversalError:
        raise http_error("validation_error", "invalid filename", status_code=400)

    # Hidden files are internal bookkeeping; report them as absent.
    if _is_hidden_relpath(filename) or not file_path.exists() or not file_path.is_file():
        raise http_error("not_found", "file not found", status_code=404)

    # Generate signed URL with 10-minute expiry
    signed_url = generate_signed_url(
        file_path=filename,
        user_id=principal.user_id,
        secret_key=runtime.settings.jwt_secret,
        expiry_seconds=600,  # SPEC §18: 10 minute expiry
    )

    return Envelope(
        status="ok",
        data={
            "download_url": signed_url,
            # §13.3 names `expires_at`. `expires_in` is kept beside it rather
            # than replaced, because removing a field clients may already read
            # is a break the SPEC does not ask for.
            "expires_at": (
                datetime.now(timezone.utc) + timedelta(seconds=600)
            ).isoformat(),
            "expires_in": 600,
            "filename": filename,
        },
    )


@router.get("/files/download", tags=["files"])
async def download_file(
    path: str = Query(..., max_length=255, description="File path"),
    expires: str = Query(..., description="Expiry timestamp"),
    sig: str = Query(..., max_length=128, description="URL signature"),
    principal: AuthContext = Depends(get_user),
):
    """Download a file using a signed URL.

    SPEC §18: Downloads use signed URLs with 10m expiry and content-disposition
    set to prevent inline execution.
    """
    runtime = get_runtime()
    await rate_limit(runtime, "read", principal.user_id)

    # Validate signed URL
    is_valid, error_msg = validate_signed_url(
        path=path,
        expires=expires,
        signature=sig,
        user_id=principal.user_id,
        secret_key=runtime.settings.jwt_secret,
    )

    if not is_valid:
        raise http_error("unauthorized", error_msg or "invalid signature", status_code=401)

    # Build file path and validate
    files_dir = (
        FilePath(runtime.settings.shared_fs_root)
        / "users"
        / principal.user_id
        / "files"
    )
    try:
        file_path = safe_join(files_dir, path)
    except PathTraversalError:
        raise http_error("validation_error", "invalid file path", status_code=400)

    # Opened once, here, and streamed from that descriptor. `FileResponse`
    # takes a pathname and opens it later, so the route's checks described a
    # moment that had passed by the time anything was read: a delete landing
    # in between turned a valid request into an internal failure rather than
    # into either the file or a clean 404. Holding the object instead means a
    # delete afterwards unlinks the name while this download finishes, which
    # is what POSIX already promises.
    #
    # `O_NOFOLLOW` for the same reason it is used when publishing: the check
    # and the open are then one operation on one object.
    try:
        fd = os.open(file_path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError:
        raise http_error("not_found", "file not found", status_code=404)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            os.close(fd)
            raise http_error("not_found", "file not found", status_code=404)
    except OSError:
        os.close(fd)
        raise http_error("not_found", "file not found", status_code=404)

    # Determine content type
    content_type, _ = mimetypes.guess_type(str(file_path))
    if not content_type:
        content_type = "application/octet-stream"

    # SPEC §18: content-disposition set to prevent inline execution.
    # The header is *encoded*, never interpolated: a filename is
    # attacker-influenced the moment model-written code chose it, and
    # `filename="{path}"` let a name containing a quote close the string and
    # add a second `filename=` parameter — a browser taking the last one saves
    # the file under a name and extension the injected page picked. This is
    # the same RFC 5987 rule `FileResponse` applies, kept when the body moved
    # to a descriptor; `tests/test_signed_download.py` is what holds it.
    quoted = quote(path)
    disposition = (
        f"attachment; filename*=utf-8''{quoted}"
        if quoted != path
        else f'attachment; filename="{path}"'
    )

    def _stream():
        try:
            while True:
                block = os.read(fd, 64 * 1024)
                if not block:
                    return
                yield block
        finally:
            os.close(fd)

    return StreamingResponse(
        _stream(),
        media_type=content_type,
        headers={
            "content-disposition": disposition,
            "content-length": str(info.st_size),
            "X-Content-Type-Options": "nosniff",
            "Cache-Control": "private, no-cache, no-store, must-revalidate",
        },
    )


@router.delete("/files/{filename:path}", response_model=Envelope, tags=["files"])
async def delete_file(
    filename: str = Path(..., max_length=512, description="File or folder to delete"),
    principal: AuthContext = Depends(get_user),
):
    """Delete a user's file, or a folder produced by archive extraction."""
    runtime = get_runtime()
    await rate_limit(runtime, "write", principal.user_id)

    files_dir = (
        FilePath(runtime.settings.shared_fs_root)
        / "users"
        / principal.user_id
        / "files"
    )
    try:
        file_path = safe_join(files_dir, filename)
    except PathTraversalError:
        raise http_error("validation_error", "invalid filename", status_code=400)

    if _is_hidden_relpath(filename):
        # Hidden files are internal bookkeeping (upload strips leading dots,
        # so a user can never own one); report them as absent.
        raise http_error("not_found", "file not found", status_code=404)

    # The name a publisher claims, not the one being deleted. Upload locks
    # `report.txt`; extraction locks `bundle` and then writes a whole tree
    # underneath it. Deleting `bundle/subdir` has to conflict with the
    # extraction that owns `bundle`, or a publisher finishes reporting success
    # over a tree something else removed part of.
    namespace = namespace_key(files_dir, filename)
    manifest_path = files_dir / ".checksums.json"

    def _locked_delete() -> None:
        """Remove the target and its manifest entry as one act.

        Deletion used to take no lock while upload and extraction each treat a
        name as one critical section. Two things followed, both measured.

        A delete landing inside an upload's transaction left the file absent
        while the manifest and the index still described it, with both
        requests returning 200 — a state no ordering of those two requests
        produces.

        And the manifest is one object for every name in the directory, so an
        unlocked read-modify-write here dropped an entry belonging to a
        concurrent upload of a *different* file, which is the false dedupe hit
        2E.1 removed, reintroduced from the other side.

        Namespace lock first, manifest lock second — the same order upload
        uses, so the two can never each hold what the other waits for.

        Inside the lock the order is bookkeeping first and the pathname last.
        No transaction spans Postgres and the filesystem, so one of the two
        halves can fail with the other already done, and the two ways of
        failing are not equally bad. Unlinking first and cleaning up
        afterwards leaves "the file is gone, its contents are still
        retrievable, and the request failed" — the user is told the deletion
        did not happen while the thing they wanted deleted is still readable
        through any grounded conversation. Doing the durable work first
        leaves "nothing was deleted and the request failed", which is a
        state the user can act on by asking again.
        """
        with path_lock(runtime.settings.shared_fs_root, namespace):
            # Re-checked under the lock: existence answered before it is an
            # answer about a moment that has passed.
            if not file_path.exists() or file_path == files_dir.resolve():
                raise http_error("not_found", "file not found", status_code=404)

            # Every context this user owns, not one named context: the same
            # file uploaded to a second context is ingested again, and an
            # extracted tree's members are recorded nowhere the route could
            # enumerate. For a directory this covers everything under it.
            runtime.store.delete_chunks_under_path(principal.user_id, str(file_path))

            # The chunks were what described the bytes; these two are what
            # would put them back or go on claiming them.
            #
            # A source row naming this path, or anything inside it, is a claim
            # about a name that is about to stop existing. A row naming an
            # *ancestor* is not: `files/` still covers that directory, and
            # will cover the name again if it reappears — so containment is
            # the test rather than coverage, or one deleted child would
            # un-index every other file beside it.
            runtime.store.delete_context_sources_under_path(
                principal.user_id, str(file_path)
            )
            # And any re-read still owed for this path is owed for nothing.
            runtime.store.cancel_ingest_jobs_under_path(
                principal.user_id, str(file_path)
            )

            with path_lock(runtime.settings.shared_fs_root, str(manifest_path)):
                if manifest_path.exists():
                    # Not best effort. The manifest is one of the three
                    # records that have to describe one generation, and a
                    # swallowed failure here leaves it naming a file that is
                    # about to stop existing — the next upload of that name
                    # then dedupes against a checksum no file has.
                    manifest = json.loads(manifest_path.read_text())
                    if manifest.pop(filename, None) is not None:
                        manifest_path.write_text(json.dumps(manifest, indent=2))

            if file_path.is_dir():
                shutil.rmtree(file_path)
            else:
                file_path.unlink()

    try:
        await asyncio.to_thread(_locked_delete)
    except PathLockTimeout as exc:
        raise http_error("conflict", str(exc), status_code=409)

    logger.info("file_deleted", user_id=principal.user_id, filename=filename)
    # §13.3 states the body exactly: {"deleted": true}. The filename is
    # already the request path, so returning it said nothing extra.
    return Envelope(status="ok", data={"deleted": True})


@router.post("/files/{filename:path}/extract", response_model=Envelope, tags=["files"])
async def extract_uploaded_archive(
    filename: str = Path(..., max_length=512, description="Archive to extract"),
    context_id: Optional[str] = Query(None, max_length=255),
    chunk_size: Optional[int] = Query(None, ge=64, le=4000),
    principal: AuthContext = Depends(get_user),
):
    """Extract an uploaded archive into a folder next to it.

    SPEC §18: extraction is hardened against zip bombs (streamed size,
    entry-count, and compression-ratio budgets), zip-slip (component-wise
    path sanitization + safe_join), and hostile members (symlinks, devices,
    and hardlinks are skipped; nested archives are never auto-expanded), and
    the whole job runs in a resource-limited sandbox subprocess. Extracted
    files can optionally be ingested into a knowledge context.
    """
    runtime = get_runtime()
    await rate_limit(runtime, "write", principal.user_id)

    files_dir = (
        FilePath(runtime.settings.shared_fs_root)
        / "users"
        / principal.user_id
        / "files"
    )
    try:
        archive_path = safe_join(files_dir, filename)
    except PathTraversalError:
        raise http_error("validation_error", "invalid filename", status_code=400)
    if (
        _is_hidden_relpath(filename)
        or not archive_path.exists()
        or not archive_path.is_file()
    ):
        raise http_error("not_found", "file not found", status_code=404)
    if not is_archive_filename(filename):
        raise http_error(
            "validation_error",
            "not an archive (.zip, .tar, .tar.gz, .tgz, .gz)",
            status_code=400,
        )
    # Before anything is published. The check used to sit after the
    # extraction, so a request refused for naming a context that does not
    # exist had already written the whole tree — and the corrected retry then
    # got 409, because the destination the refused request created was in the
    # way. Same ordering rule as the upload route: a parameter this will
    # refuse is knowable before any mutation.
    if context_id:
        _get_owned_context(runtime, context_id, principal)

    # Destination folder sits next to the archive, named after its stem.
    dest_rel = FilePath(filename).parent / archive_stem(archive_path.name)
    try:
        dest_path = safe_join(files_dir, dest_rel.as_posix())
    except PathTraversalError:
        raise http_error("validation_error", "invalid filename", status_code=400)
    # Budgets scale with the user's plan: per-member = upload limit,
    # total = 10x upload limit. Ratio and entry caps stop crafted bombs.
    user = runtime.store.get_user(principal.user_id)
    plan_tier = user.plan_tier if user else "free"
    member_bytes = max(1, plan_upload_limit(runtime, plan_tier))
    limits = {
        "max_entries": 1000,
        "max_member_bytes": member_bytes,
        "max_total_bytes": member_bytes * 10,
        "max_ratio": 100,
        "allowed_extensions": sorted(ALLOWED_UPLOAD_EXTENSIONS),
    }

    def _extract_into_destination() -> tuple[dict, Optional[int]]:
        """Claim the destination, then fill it, without letting go between.

        The check and the extraction are one act. `bundle.zip` and
        `bundle.tar.gz` are different requests for different archives whose
        only shared state is where they land, so two of them could both pass
        an exists-check taken in the API process and both write into one
        tree — measured, `bundle/` ended up holding `zip.txt` and `tar.txt`
        with both requests returning 200.

        Worse on the failure path: `extract_archive` removes the destination
        when it refuses, so a corrupt archive's cleanup deleted a tree the
        other request had already published. Measured, `bundle/` was gone and
        the successful request had reported 200.

        Keyed on the destination rather than on the archive, because the
        archive names are deliberately different and the destination is the
        thing being competed for. A waiter that arrives afterwards finds the
        finished tree and gets the ordinary 409.
        """
        with path_lock(
            runtime.settings.shared_fs_root,
            namespace_key(files_dir, dest_rel.as_posix()),
        ):
            if dest_path.exists():
                raise http_error(
                    "conflict",
                    f"'{dest_rel.as_posix()}' already exists; delete it first",
                    status_code=409,
                )
            # Filled somewhere nobody can reach, then moved into place. The
            # extractor creates each member at its final path and streams
            # into it, so the destination existed under its real name from
            # the first member onward: measured, a member of an unfinished
            # tree was signable, and a download of it would have returned a
            # short file with a content-length that agreed. The unit that
            # has to appear at once is the tree — a listing showing half a
            # bundle describes something that never existed.
            #
            # Staged outside the user's area rather than as a hidden sibling
            # of the destination. `ingest_path` walks `**/*` and does not
            # skip hidden components, so a context source covering `files/`
            # would find the half-written members. Nothing here is under any
            # user's path authority.
            staging_root = (
                FilePath(runtime.settings.shared_fs_root)
                / ARCHIVE_STAGING_DIRNAME
                / principal.user_id
            )
            staging_root.mkdir(parents=True, exist_ok=True)
            staging = staging_root / uuid4().hex
            try:
                report = extract_archive_sandboxed(
                    str(archive_path), str(staging), limits
                )
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                # One filesystem, so this is a rename and not a copy. The
                # destination was checked absent under this lock and nothing
                # else may claim it while the lock is held, so the rename has
                # nothing to overwrite.
                os.rename(staging, dest_path)
            finally:
                shutil.rmtree(staging, ignore_errors=True)
            # Ingestion stays inside the lock. Releasing it here and indexing
            # afterwards leaves the tree deletable mid-traversal, and
            # `ingest_path` catches per-file errors and returns the partial
            # count rather than failing — so the request reports success with
            # the folder gone and only the files read before the delete
            # indexed. "Extract with a context" is one operation; the
            # destination has to stay this operation's until it is finished
            # with it.
            chunks = None
            if context_id:
                chunks = runtime.rag.ingest_path(
                    context_id,
                    str(dest_path),
                    recursive=True,
                    chunk_size=chunk_size,
                    allowed_base=files_dir,
                )
            return report, chunks

    try:
        report, chunk_count = await asyncio.to_thread(_extract_into_destination)
    except PathLockTimeout as exc:
        raise http_error("conflict", str(exc), status_code=409)
    except ArchiveExtractionError as exc:
        raise http_error("validation_error", str(exc), status_code=413)
    except SandboxError as exc:
        logger.warning(
            "archive_extraction_sandbox_failed",
            user_id=principal.user_id,
            filename=filename,
            error=str(exc),
        )
        raise http_error(
            "validation_error", f"extraction failed: {exc}", status_code=422
        )

    logger.info(
        "archive_extracted",
        user_id=principal.user_id,
        filename=filename,
        files=len(report["extracted"]),
        skipped=len(report["skipped"]),
        total_bytes=report["total_bytes"],
    )
    return Envelope(
        status="ok",
        data={
            "extracted_to": dest_rel.as_posix(),
            "files": report["extracted"],
            "skipped": report["skipped"],
            "total_bytes": report["total_bytes"],
            "chunk_count": chunk_count,
        },
    )


@router.get("/conversations/{conversation_id}/messages", response_model=Envelope, tags=["conversations"])
async def list_messages(
    conversation_id: str = Path(
        ..., max_length=255, description="Conversation identifier"
    ),
    limit: Optional[int] = Query(None, ge=1, description="Maximum messages to return"),
    principal: AuthContext = Depends(get_user),
):
    runtime = get_runtime()
    await rate_limit(runtime, "read", principal.user_id)
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
    async with idempotency.IdempotencyGuard(
        "conversations:create", principal.user_id, idempotency_key, require=False
    ) as idem:
        if idem.cached:
            return idem.cached
        await rate_limit(runtime, "write", principal.user_id)
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
        await idem.store_result(response)
        return response


@router.get("/conversations", response_model=Envelope, tags=["conversations"])
async def list_conversations(
    limit: Optional[int] = Query(None, ge=1, description="Maximum conversations to return"),
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    principal: AuthContext = Depends(get_user),
):
    runtime = get_runtime()
    await rate_limit(runtime, "read", principal.user_id)
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
            public=bool((c.meta or {}).get("public")),
            source=(c.meta or {}).get("source") or "chat",
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
    await rate_limit(runtime, "read", principal.user_id)
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
            public=bool((conversation.meta or {}).get("public")),
            source=(conversation.meta or {}).get("source") or "chat",
        ),
    )


@router.patch(
    "/conversations/{conversation_id}", response_model=Envelope, tags=["conversations"]
)
async def update_conversation(
    body: ConversationUpdateRequest,
    conversation_id: str = Path(..., max_length=255, description="Conversation identifier"),
    principal: AuthContext = Depends(get_user),
):
    """Rename or archive a conversation. Owner-only.

    Only `title` and `status` are editable here; the request model forbids
    anything else rather than ignoring it, so a caller sending `meta` or
    `active_context_id` is told they were refused instead of being told their
    edit succeeded when it silently did not.
    """
    runtime = get_runtime()
    await rate_limit(runtime, "write", principal.user_id)
    _get_owned_conversation(runtime, conversation_id, principal)
    conversation = runtime.store.update_conversation(
        conversation_id,
        user_id=principal.user_id,
        title=body.title,
        status=body.status,
    )
    if not conversation:
        raise NotFoundError(
            "conversation not found", detail={"conversation_id": conversation_id}
        )
    return Envelope(
        status="ok",
        data=ConversationSummary(
            id=conversation.id,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
            title=conversation.title,
            status=conversation.status,
            active_context_id=conversation.active_context_id,
            public=bool((conversation.meta or {}).get("public")),
            source=(conversation.meta or {}).get("source") or "chat",
        ),
    )


@router.delete(
    "/conversations/{conversation_id}", response_model=Envelope, tags=["conversations"]
)
async def delete_conversation(
    conversation_id: str = Path(..., max_length=255, description="Conversation identifier"),
    principal: AuthContext = Depends(get_user),
):
    """Delete a conversation and everything scoped to it. Owner-only.

    That includes the chat's messages and its implicit attachment index, whose
    chunks hold the text of files attached to this chat and nowhere else
    (SPEC §19.5). The index is removed by the cascading foreign key on
    `knowledge_context.conversation_id`, in the same transaction.

    The stored file objects are not unlinked here. They are content-addressed
    and another conversation may name the same checksum, so releasing them is
    left to the sweep, which reclaims what no conversation references any
    more.
    """
    runtime = get_runtime()
    await rate_limit(runtime, "write", principal.user_id)
    _get_owned_conversation(runtime, conversation_id, principal)
    deleted = runtime.store.delete_conversation(
        conversation_id, user_id=principal.user_id
    )
    if not deleted:
        raise NotFoundError(
            "conversation not found", detail={"conversation_id": conversation_id}
        )

    # After the database commits, and best effort. Redis caches this chat's
    # recent messages under its own hour-long TTL, so without this the text of
    # a deleted conversation stays readable there after every trace of it has
    # gone from Postgres. It runs second because the database is the record:
    # a cache that cannot be reached must not turn a completed deletion into a
    # reported failure the user would retry against a chat that is already
    # gone. What survives a failure here is an entry that expires on its own.
    if runtime.cache is not None:
        try:
            await runtime.cache.delete_conversation_summary(conversation_id)
        except Exception as exc:  # pragma: no cover - cache outage
            logger.warning(
                "conversation_cache_retire_failed",
                conversation_id=conversation_id,
                error=str(exc),
            )

    logger.info(
        "conversation_deleted",
        user_id=principal.user_id,
        conversation_id=conversation_id,
    )
    return Envelope(status="ok", data={"deleted": True})


@router.post(
    "/conversations/{conversation_id}/share", response_model=Envelope, tags=["conversations"]
)
async def share_conversation(
    body: ConversationShareRequest,
    conversation_id: str = Path(..., max_length=255, description="Conversation identifier"),
    principal: AuthContext = Depends(get_user),
):
    """Toggle public sharing for a conversation.

    Conversations are private by default; the owner can publish one to the
    read-only /share/{id} page and unpublish it again at any time.
    """
    runtime = get_runtime()
    await rate_limit(runtime, "write", principal.user_id)
    conversation = runtime.store.set_conversation_public(
        conversation_id, user_id=principal.user_id, public=body.public
    )
    if not conversation:
        raise NotFoundError(
            "conversation not found", detail={"conversation_id": conversation_id}
        )
    logger.info(
        "conversation_share_toggled",
        user_id=principal.user_id,
        conversation_id=conversation_id,
        public=body.public,
    )
    return Envelope(
        status="ok",
        data={
            "conversation_id": conversation.id,
            "public": bool((conversation.meta or {}).get("public")),
            "share_path": f"/share/{conversation.id}",
        },
    )


# =========================================================================
# Public (unauthenticated) read-only access to shared conversations.
# The share directory and pages are noindex by default (SPEC §18): both the
# API and page responses carry X-Robots-Tag, and robots.txt disallows /share/.
# =========================================================================

_PUBLIC_NOINDEX = {"X-Robots-Tag": "noindex, nofollow"}


@router.get("/public/conversations", response_model=Envelope, tags=["public"])
async def list_public_conversations(
    response: Response,
    limit: Optional[int] = Query(None, ge=1, le=100),
):
    """Directory of publicly shared conversations (unauthenticated)."""
    runtime = get_runtime()
    await enforce(
        runtime,
        "public:read",
        runtime.settings.read_rate_limit_per_minute,
        60,
    )
    response.headers.update(_PUBLIC_NOINDEX)
    items = runtime.store.list_public_conversations(limit=limit or 50)
    return Envelope(
        status="ok",
        data={
            "items": [
                {
                    "id": c.id,
                    "title": c.title or "Untitled conversation",
                    "updated_at": c.updated_at,
                }
                for c in items
            ]
        },
    )


@router.get(
    "/public/conversations/{conversation_id}", response_model=Envelope, tags=["public"]
)
async def get_public_conversation(
    response: Response,
    conversation_id: str = Path(..., max_length=255),
):
    """Read a publicly shared conversation (unauthenticated).

    Only role, content, and timestamps are exposed — message metadata,
    adapter traces, and owner identity stay private.
    """
    runtime = get_runtime()
    await enforce(
        runtime,
        "public:read",
        runtime.settings.read_rate_limit_per_minute,
        60,
    )
    response.headers.update(_PUBLIC_NOINDEX)
    conversation = runtime.store.get_public_conversation(conversation_id)
    if not conversation:
        raise NotFoundError(
            "conversation not found", detail={"conversation_id": conversation_id}
        )
    messages = runtime.store.list_messages(
        conversation_id, limit=500, user_id=conversation.user_id
    )
    ordered = sorted(messages, key=lambda m: m.created_at)
    return Envelope(
        status="ok",
        data={
            "id": conversation.id,
            "title": conversation.title or "Shared conversation",
            "updated_at": conversation.updated_at,
            "messages": [
                {"role": m.role, "content": m.content, "created_at": m.created_at}
                for m in ordered
            ],
        },
    )


@router.post("/contexts", response_model=Envelope, status_code=201, tags=["knowledge"])
async def create_context(
    body: KnowledgeContextRequest,
    principal: AuthContext = Depends(get_user),
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
):
    runtime = get_runtime()
    await rate_limit(runtime, "write", principal.user_id)
    # SPEC §18: Accept Idempotency-Key when provided (optional)
    async with idempotency.IdempotencyGuard(
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
    page: int = Query(1, ge=1, description="Page number (ignored when cursor is set)"),
    page_size: Optional[int] = Query(
        None, ge=1, description="Number of contexts per page (alias: limit)"
    ),
    cursor: Optional[str] = Query(
        None, description="Keyset cursor returned by a previous response"
    ),
    principal: AuthContext = Depends(get_user),
):
    runtime = get_runtime()
    await rate_limit(runtime, "read", principal.user_id)
    paging = _get_pagination_settings(runtime)
    effective_page_size = page_size or limit or paging["default_page_size"]
    resolved_page_size = min(max(effective_page_size, 1), paging["max_page_size"])
    # Per-conversation attachment contexts are an implementation detail of
    # the attach-to-conversation flow, so they are not part of what this
    # listing pages over. Excluded in the query rather than from the result:
    # the ordering and the limit happen in the store, so dropping them
    # afterwards spent page slots on rows the caller never sees and reported
    # no next page with ordinary contexts still unreached.
    contexts = runtime.store.list_contexts(
        owner_user_id=principal.user_id,
        page=page,
        page_size=resolved_page_size,
        cursor=cursor,
        include_sentinel=True,
        include_auto=False,
    )
    has_next = len(contexts) > resolved_page_size
    page_contexts = contexts[:resolved_page_size]
    next_cursor = None
    if has_next and page_contexts:
        last_ctx = page_contexts[-1]
        next_cursor = encode_time_id_cursor(last_ctx.created_at, last_ctx.id)
    next_page = None if cursor else (page + 1 if has_next else None)
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
        for c in page_contexts
    ]
    return Envelope(
        status="ok",
        data=KnowledgeContextListResponse(
            items=items,
            has_next=has_next,
            next_cursor=next_cursor,
            next_page=next_page,
            page_size=resolved_page_size,
        ),
    )


@router.get("/contexts/{context_id}/chunks", response_model=Envelope, tags=["knowledge"])
async def list_chunks(
    context_id: str = Path(..., max_length=255, description="Knowledge context ID"),
    limit: Optional[int] = Query(None, ge=1, description="Maximum chunks to return"),
    page: int = Query(1, ge=1, description="Page number (ignored when cursor is set)"),
    page_size: Optional[int] = Query(
        None, ge=1, description="Number of chunks per page (alias: limit)"
    ),
    cursor: Optional[str] = Query(
        None, description="Cursor returned by a previous response"
    ),
    principal: AuthContext = Depends(get_user),
):
    runtime = get_runtime()
    await rate_limit(runtime, "read", principal.user_id)
    paging = _get_pagination_settings(runtime)
    effective_page_size = page_size or limit or paging["default_page_size"]
    resolved_page_size = min(max(effective_page_size, 1), paging["max_page_size"])
    _get_owned_context(runtime, context_id, principal)
    chunks = runtime.store.list_chunks(
        context_id,
        owner_user_id=principal.user_id,
        page=page,
        page_size=resolved_page_size,
        cursor=cursor,
        include_sentinel=True,
    )
    for ch in chunks:
        if ch.id is None:
            raise http_error(
                "server_error", "chunk id missing for context", status_code=500
            )
    has_next = len(chunks) > resolved_page_size
    page_chunks = chunks[:resolved_page_size]
    next_cursor = None
    if has_next and page_chunks:
        last_chunk = page_chunks[-1]
        next_cursor = encode_index_cursor(last_chunk.chunk_index, str(last_chunk.id))
    next_page = None if cursor else (page + 1 if has_next else None)
    data = [
        KnowledgeChunkResponse(
            id=int(ch.id),
            context_id=ch.context_id,
            fs_path=ch.fs_path,
            content=ch.content,
            chunk_index=ch.chunk_index,
        )
        for ch in page_chunks
    ]
    return Envelope(
        status="ok",
        data=KnowledgeChunkListResponse(
            items=data,
            has_next=has_next,
            next_cursor=next_cursor,
            next_page=next_page,
            page_size=resolved_page_size,
        ),
    )


def _context_payload(ctx) -> KnowledgeContextResponse:
    return KnowledgeContextResponse(
        id=ctx.id,
        name=ctx.name,
        description=ctx.description,
        created_at=ctx.created_at,
        updated_at=ctx.updated_at,
        owner_user_id=ctx.owner_user_id,
        meta=ctx.meta,
    )


@router.get("/contexts/{context_id}", response_model=Envelope, tags=["knowledge"])
async def get_context(
    context_id: str = Path(..., max_length=255, description="Knowledge context ID"),
    principal: AuthContext = Depends(get_user),
):
    """Read one context the caller owns.

    An identity lookup, not the first page of a listing. `/contexts` pages in
    SQL, so before this route existed a caller holding an id could not reach
    the row behind it once the account had a page of newer contexts.
    """
    runtime = get_runtime()
    await rate_limit(runtime, "read", principal.user_id)
    _get_owned_context(runtime, context_id, principal)
    ctx = runtime.store.get_ordinary_context(
        context_id, owner_user_id=principal.user_id
    )
    if ctx is None:
        raise http_error("not_found", "context not found", status_code=404)
    return Envelope(status="ok", data=_context_payload(ctx))


@router.patch("/contexts/{context_id}", response_model=Envelope, tags=["knowledge"])
async def update_context(
    body: KnowledgeContextUpdateRequest,
    context_id: str = Path(..., max_length=255, description="Knowledge context ID"),
    principal: AuthContext = Depends(get_user),
):
    """Rename or redescribe a context the caller owns."""
    runtime = get_runtime()
    await rate_limit(runtime, "write", principal.user_id)
    _get_owned_context(runtime, context_id, principal)
    ctx = runtime.store.update_context(
        context_id,
        owner_user_id=principal.user_id,
        name=body.name,
        description=body.description,
    )
    if ctx is None:
        raise http_error("not_found", "context not found", status_code=404)
    return Envelope(status="ok", data=_context_payload(ctx))


@router.delete("/contexts/{context_id}", response_model=Envelope, tags=["knowledge"])
async def delete_context(
    context_id: str = Path(..., max_length=255, description="Knowledge context ID"),
    principal: AuthContext = Depends(get_user),
):
    """Retire a context the caller owns.

    Its source records, chunks and segment vectors go with it by cascade, and
    conversations bound to it through `active_context_id` are released rather
    than deleted. The files it indexed are untouched: a context references
    paths, it does not own them.
    """
    runtime = get_runtime()
    await rate_limit(runtime, "write", principal.user_id)
    _get_owned_context(runtime, context_id, principal)
    if not runtime.store.delete_context(context_id, owner_user_id=principal.user_id):
        raise http_error("not_found", "context not found", status_code=404)
    logger.info(
        "context_deleted", user_id=principal.user_id, context_id=context_id
    )
    return Envelope(status="ok", data={"deleted": True})


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
    await rate_limit(runtime, "write", principal.user_id)

    # SPEC §18: a path resolves under `/users/{user_id}`, or an artifact whose
    # persisted visibility is `shared`/`global` covers it. This used to accept
    # anything underneath `shared_fs_root/shared` because it was underneath
    # that directory, and then check that the destination *context* belonged to
    # the caller — which says who receives the content and never who was
    # entitled to the source. See service/fs.authorize_path.
    try:
        validated_path = authorize_path(
            runtime.store,
            runtime.settings,
            body.fs_path,
            user_id=principal.user_id,
            tenant_id=principal.tenant_id,
        )
    except PathAuthorityError as exc:
        logger.warning(
            "context_source_denied",
            user_id=principal.user_id,
            fs_path=body.fs_path,
            reason=str(exc),
        )
        raise http_error(
            "forbidden",
            "you do not have authority over that path",
            status_code=403,
        ) from exc

    async with idempotency.IdempotencyGuard(
        "context_sources:create", principal.user_id, idempotency_key, require=False
    ) as idem:
        if idem.cached:
            return idem.cached

        # Verify context ownership
        # `_get_owned_context` refuses a conversation's implicit index, so
        # a path-following source cannot be added to one.
        _get_owned_context(runtime, context_id, principal)

        # Add the source with validated path
        source = runtime.store.add_context_source(
            context_id=context_id,
            fs_path=str(validated_path),
            recursive=body.recursive,
        )

        # Trigger indexing via RAG service with validated path
        # Pass allowed_base for defense-in-depth path traversal protection
        def _ingest() -> None:
            """Read each file and commit what was read, without letting go.

            Reading and committing are two moments. Upload, extraction and
            deletion all treat a pathname as one critical section, and this
            route took part in none of it — so its commit could land after a
            newer generation had already replaced the bytes, the manifest
            entry and the chunks. Measured: the disk and the manifest
            described the upload while the index described what the source
            had read, with both requests returning success. No serial
            ordering produces that.

            The guard is per file, not one lock over the whole source. A
            source rooted at `files/` takes a key nothing else takes, while
            an upload of `files/report.md` takes that name's key, so a single
            source-wide lock let the same interleaving straight through one
            level up — measured again, with the source being the directory
            rather than the file.

            Only for a path inside the caller's own files. That is where the
            lock means something, because every writer of those names takes
            it. A shared corpus may have writers outside this application
            entirely, and a lock no one else holds would only look like
            protection.
            """
            files_dir = (
                FilePath(runtime.settings.shared_fs_root)
                / "users"
                / principal.user_id
                / "files"
            )

            def guard(candidate: FilePath):
                try:
                    relative = candidate.relative_to(files_dir)
                except ValueError:
                    return contextlib.nullcontext()
                return path_lock(
                    runtime.settings.shared_fs_root,
                    namespace_key(files_dir, relative.as_posix()),
                )

            runtime.rag.ingest_path(
                context_id=context_id,
                fs_path=str(validated_path),
                recursive=body.recursive,
                allowed_base=runtime.settings.shared_fs_root,
                file_guard=guard,
            )

        try:
            # In a thread, both because the lock must not be taken on the
            # event loop and because walking a tree was already blocking it.
            await asyncio.to_thread(_ingest)
        except PathLockTimeout as exc:
            try:
                runtime.store.delete_context_source(source.id)
            except Exception:
                pass  # Best effort cleanup
            raise http_error("conflict", str(exc), status_code=409)
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
            raise http_error(
                "ingest_failed",
                f"Failed to index source: {exc}",
                status_code=500,
            )

        # Recording the source and reading it are two moments, and the owner
        # may retire the context in between. The database already refuses the
        # work — chunks reference the context, the source row went with it by
        # cascade — but `ingest_path` treats a failed file as a warning and
        # carries on, which is right for one unreadable file in a tree and
        # wrong for the whole context being gone. Without this the request
        # reported 201 and returned a source record that no longer existed.
        if runtime.store.get_ordinary_context(
            context_id, owner_user_id=principal.user_id
        ) is None:
            logger.info(
                "context_source_abandoned",
                context_id=context_id,
                user_id=principal.user_id,
            )
            raise http_error(
                "conflict",
                "the context was deleted while its source was being indexed",
                status_code=409,
            )

        envelope = Envelope(
            status="ok",
            data=ContextSourceResponse.model_validate(source),
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
    await rate_limit(runtime, "read", principal.user_id)

    # Verify context ownership
    _get_owned_context(runtime, context_id, principal)

    sources = runtime.store.list_context_sources(context_id)
    items = [
        ContextSourceResponse.model_validate(s) for s in sources
    ]
    return Envelope(status="ok", data=ContextSourceListResponse(items=items))


@router.post("/voice/transcribe", response_model=Envelope, tags=["voice"])
async def transcribe_voice(
    file: UploadFile = File(...), principal: AuthContext = Depends(get_user)
):
    runtime = get_runtime()
    # Rate limit voice transcription (resource-intensive)
    await enforce(
        runtime,
        f"voice:transcribe:{principal.user_id}",
        runtime.settings.chat_rate_limit_per_minute,  # Use chat rate limit as default
        60,
    )
    # Limit audio file size to 10MB
    max_audio_bytes = 10 * 1024 * 1024
    audio_bytes = await file.read(max_audio_bytes + 1)
    if len(audio_bytes) > max_audio_bytes:
        raise http_error(
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
    await enforce(
        runtime,
        f"voice:synthesize:{principal.user_id}",
        runtime.settings.chat_rate_limit_per_minute,  # Use chat rate limit as default
        60,
    )
    # Limit text length to prevent resource exhaustion
    if len(body.text) > 5000:
        raise http_error(
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
    client_host = ws.client.host if ws.client else "unknown"
    # The declared policy, not a second lookup with its own hardcoded default —
    # that copy ignored the admin's setting whenever it was left unset.
    try:
        await rate_limit(runtime, "websocket:connect", client_host)
    except HTTPException:
        await ws.accept()
        await ws.close(code=4429)
        return
    await ws.accept()
    user_id: Optional[str] = None
    idempotency_key: Optional[str] = None
    # Generate request_id immediately to ensure traceability even on early errors
    request_id: str = str(uuid4())
    convo_id: Optional[str] = None
    held_slots: list[str] = []
    try:
        init = await ws.receive_json()
        idempotency_key = init.get("idempotency_key")
        # Use client-provided request_id if available, otherwise keep generated one
        request_id = init.get("request_id") or request_id
        session_id = init.get("session_id")
        access_token = init.get("access_token")
        if session_id and access_token:
            await ws.send_json(
                {
                    "event": "error",
                    "data": {
                        "error": "fresh_session_required",
                        "message": "provide a single authentication method (refresh session before switching transports)",
                    },
                    "request_id": request_id,
                }
            )
            await ws.close(code=4401)
            return
        auth_ctx = await runtime.auth.authenticate(
            f"Bearer {access_token}" if access_token else None,
            session_id,
            # The site the socket was opened against, same as every HTTP route.
            # A session from one tenant's site cannot be replayed at another's.
            tenant_hint=tenancy.tenant_of(ws.headers, runtime.settings),
        )
        if not auth_ctx:
            await ws.close(code=4401)
            return
        user_id = auth_ctx.user_id

        # Issue 5.2: Per-user WebSocket connection limits
        try:
            await admission.acquire(runtime, "websocket", user_id)
        except admission.AtCapacity as exc:
            await ws.send_json({
                "event": "error",
                "data": {"error": "connection_limit", "message": exc.message},
                "request_id": request_id,
            })
            await ws.close(code=4429)  # Custom code for "too many connections"
            return
        held_slots.append("websocket")

        # Get user's plan tier for per-plan rate limits (SPEC §18)
        user = runtime.store.get_user(user_id)
        plan_tier = user.plan_tier if user else "free"

        # SPEC §18: Accept Idempotency-Key when provided (optional)
        request_id, cached = await idempotency.resolve(
            "chat:ws", user_id, idempotency_key, require=False, request_id=request_id
        )
        if cached:
            await ws.send_json(cached.model_dump())
            return

        # SPEC §18: Per-plan adjustable rate limits
        await enforce_per_plan(
            runtime,
            f"chat:{user_id}",
            runtime.settings.chat_rate_limit_per_minute,
            runtime.settings.chat_rate_limit_window_seconds,
            plan_tier,
        )

        # SPEC §18: concurrency caps. CHAT_SLOTS is what a turn costs on either
        # transport; this path used to take the workflow slot alone and skip the
        # inference cap entirely, which made max_concurrent_inference a setting
        # that only applied to the endpoint the UI does not use.
        for kind in admission.CHAT_SLOTS:
            try:
                await admission.acquire(runtime, kind, user_id)
            except admission.AtCapacity as exc:
                await ws.send_json({
                    "event": "error",
                    "data": {"error": "busy", "message": exc.message},
                    "request_id": request_id,
                })
                await ws.close(code=4409)  # Custom code for "busy"
                return
            held_slots.append(kind)

        convo_id = init.get("conversation_id")
        turn = await chat_turn.begin(
            runtime,
            auth_ctx,
            conversation_id=convo_id,
            context_id=init.get("context_id"),
            workflow_id=init.get("workflow_id"),
            content=init.get("message", ""),
            content_struct=init.get("content_struct"),
            owned_conversation=_get_owned_conversation,
            owned_context=_get_owned_context,
        )
        convo_id = turn.conversation_id
        context_id = turn.context_id

        # SPEC §18: Check if streaming is requested (default True for WebSocket)
        stream_enabled = init.get("stream", True)

        if stream_enabled:
            # Streaming mode: emit token, trace, message_done, error events
            cancel_event = asyncio.Event()
            # Register cancel event so POST /chat/cancel can cancel this request
            # Issue 50.5: Include user_id for ownership validation
            await cancellation.register(request_id, cancel_event, user_id)
            # Issue 38.5: Use list for O(n) token accumulation instead of string += O(n²)
            content_tokens: list[str] = []
            orchestration_dict: dict[str, Any] = {}
            message_done_received = False

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
                                await ws.send_json({"event": "pong", "data": None, "request_id": request_id})
                        except asyncio.TimeoutError:
                            continue
                        except WebSocketDisconnect:
                            cancel_event.set()
                            return
                except Exception as exc:
                    # Listener task exits silently on errors; log for debugging
                    logger.debug("websocket_listener_exit", error=str(exc))

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

                    # SPEC §18: WebSockets wrap as {"event": "token", "data": "...", "request_id": "..."}
                    # The workflow's own message_done is a control signal for this
                    # route (it lacks message_id/conversation_id); the client gets
                    # exactly one message_done - the final one sent after the
                    # assistant message is persisted. Forwarding both made clients
                    # bind to an ID-less completion and lose the conversation id.
                    if event_type != "message_done":
                        try:
                            await ws.send_json(
                                {"event": event_type, "data": event_data, "request_id": request_id}
                            )
                        except WebSocketDisconnect:
                            cancel_event.set()
                            break
                        except RuntimeError as exc:
                            cancel_event.set()
                            logger.warning(
                                "websocket_send_failed", request_id=request_id, error=str(exc)
                            )
                            break

                    if event_type == "token":
                        if isinstance(event_data, str):
                            content_tokens.append(event_data)
                    elif event_type == "message_done":
                        message_done_received = True
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
                with suppress(asyncio.CancelledError):
                    await cancel_listener
                await cancellation.unregister(request_id)
                return
            finally:
                cancel_listener.cancel()
                with suppress(asyncio.CancelledError):
                    await cancel_listener
                await cancellation.unregister(request_id)

            # Save assistant message after streaming completes (only if successful)
            if cancel_event.is_set() or not message_done_received:
                return
            # Issue 38.5: join tokens at the end for O(n) rather than O(n^2).
            assistant_msg = await chat_turn.finish(
                runtime, turn, orchestration_dict, content="".join(content_tokens)
            )
            adapter_names = _stringify_adapters(turn.orchestration.get("adapters", []))
            assistant_content = assistant_msg.content
            envelope = Envelope(
                status="ok",
                data=chat_turn.response(turn, assistant_msg, adapter_names).model_dump(),
                request_id=request_id,
            )
            await idempotency.store("chat:ws", user_id, idempotency_key, envelope)

            # Send final event with message_id and conversation_id to client
            # SPEC §18: Valid events are token, message_done, error, cancel_ack, trace
            await ws.send_json({
                "event": "message_done",
                "data": {
                    "message_id": assistant_msg.id,
                    "conversation_id": convo_id,
                    # The full reply, so a client that received no token events
                    # can still render it. Tool-calling nodes (the attachment
                    # agent) return their answer at once rather than streaming.
                    "content": assistant_content,
                    "adapters": adapter_names,
                    "adapter_gates": orchestration_dict.get("adapter_gates", []),
                    "usage": orchestration_dict.get("usage", {}),
                    "context_snippets": orchestration_dict.get("context_snippets", []),
                    "routing_trace": orchestration_dict.get("routing_trace", []),
                    "workflow_trace": orchestration_dict.get("workflow_trace", []),
                },
                "request_id": request_id,
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
            # Same finish as the streaming branch: this one used to persist the
            # reply and schedule nothing, so `{"stream": false}` produced a turn
            # with no label, no title, no digest and no embedding backfill.
            assistant_msg = await chat_turn.finish(runtime, turn, orchestration)
            adapter_names = _stringify_adapters(turn.orchestration.get("adapters", []))
            envelope = Envelope(
                status="ok",
                data=chat_turn.response(turn, assistant_msg, adapter_names).model_dump(),
                request_id=request_id,
            )
            await idempotency.store("chat:ws", user_id, idempotency_key, envelope)
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
            await idempotency.store(
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
            error={"code": "validation_error", "message": "Invalid JSON in request"},
            request_id=request_id,
        )
        try:
            await ws.send_json(error_env.model_dump())
        except Exception as send_exc:
            logger.debug("websocket_error_send_failed", error=str(send_exc))
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
            await idempotency.store(
                "chat:ws", user_id, idempotency_key, error_env, status="failed"
            )
        # Send error envelope to client before closing
        try:
            await ws.send_json(error_env.model_dump())
        except Exception as send_exc:
            # Connection may already be closed
            logger.debug("websocket_error_send_failed", error=str(send_exc))
        await ws.close(code=1011)
    finally:
        # Always release slots, even on error
        for kind in reversed(held_slots):
            with contextlib.suppress(Exception):
                await admission.release(runtime, kind, user_id)
