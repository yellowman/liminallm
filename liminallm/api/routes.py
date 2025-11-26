from __future__ import annotations

import base64
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Response, UploadFile, WebSocket, WebSocketDisconnect

from liminallm.content_struct import normalize_content_struct
from liminallm.api.schemas import (
    ArtifactListResponse,
    ArtifactRequest,
    ArtifactResponse,
    ArtifactVersionListResponse,
    ArtifactVersionResponse,
    AuthResponse,
    ChatRequest,
    ChatResponse,
    ConversationMessagesResponse,
    ConversationListResponse,
    ConversationSummary,
    ConfigPatchAuditResponse,
    ConfigPatchDecisionRequest,
    ConfigPatchListResponse,
    ConfigPatchRequest,
    Envelope,
    LoginRequest,
    MFARequest,
    MFAVerifyRequest,
    AutoPatchRequest,
    OAuthStartRequest,
    OAuthStartResponse,
    PasswordResetConfirm,
    PasswordResetRequest,
    SignupRequest,
    TokenRefreshRequest,
    KnowledgeChunkListResponse,
    KnowledgeChunkResponse,
    KnowledgeContextRequest,
    KnowledgeContextListResponse,
    KnowledgeContextResponse,
    FileUploadResponse,
    PreferenceEventRequest,
    PreferenceEventResponse,
    PreferenceInsightsResponse,
    VoiceSynthesisRequest,
    VoiceSynthesisResponse,
    VoiceTranscriptionResponse,
    UserListResponse,
    UserResponse,
    AdminCreateUserRequest,
    UpdateUserRoleRequest,
    AdminInspectionResponse,
    ToolInvokeRequest,
    ToolInvokeResponse,
    ToolSpecListResponse,
    WorkflowListResponse,
)
from liminallm.service.auth import AuthContext
from liminallm.service.errors import BadRequestError, NotFoundError
from liminallm.storage.errors import ConstraintViolation
from liminallm.storage.models import Conversation, KnowledgeContext, Session
from liminallm.config import get_settings
from liminallm.service.fs import safe_join
from liminallm.service.runtime import (
    IDEMPOTENCY_TTL_SECONDS,
    _get_cached_idempotency_record,
    _set_cached_idempotency_record,
    check_rate_limit,
    get_runtime,
)

router = APIRouter(prefix="/v1")


class IdempotencyGuard:
    def __init__(self, route: str, user_id: str, idempotency_key: Optional[str], *, require: bool = False):
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
            self.route, self.user_id, self.idempotency_key, require=self.require, request_id=self.request_id
        )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        if exc and not self._stored and self.request_id:
            await _store_idempotency_result(
                self.route,
                self.user_id,
                self.idempotency_key,
                Envelope(status="error", error={"message": str(exc)}, request_id=self.request_id),
                status="failed",
            )
            self._stored = True
        return False

    async def store_result(self, envelope: Envelope, *, status: str = "completed") -> None:
        await _store_idempotency_result(self.route, self.user_id, self.idempotency_key, envelope, status=status)
        self._stored = True

    async def store_error(self, message: str) -> None:
        if not self.request_id:
            return
        await self.store_result(
            Envelope(status="error", error={"message": message}, request_id=self.request_id),
            status="failed",
        )


async def _enforce_rate_limit(runtime, key: str, limit: int, window_seconds: int) -> None:
    allowed = await check_rate_limit(runtime, key, limit, window_seconds)
    if not allowed:
        raise HTTPException(status_code=429, detail="rate limit exceeded")


async def _resolve_idempotency(
    route: str, user_id: str, idempotency_key: Optional[str], *, require: bool = False, request_id: Optional[str] = None
) -> tuple[str, Optional[Envelope]]:
    request_id = request_id or str(uuid4())
    runtime = get_runtime()
    if not idempotency_key:
        if require:
            raise HTTPException(status_code=400, detail="Idempotency-Key header required")
        return request_id, None
    record = await _get_cached_idempotency_record(runtime, route, user_id, idempotency_key)
    if record:
        status = record.get("status")
        if status == "in_progress":
            raise HTTPException(status_code=409, detail="request in progress")
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
        {"status": "in_progress", "request_id": request_id, "started_at": datetime.utcnow().isoformat()},
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
    x_tenant_id: Optional[str] = Header(None, convert_underscores=False, alias="X-Tenant-ID"),
) -> AuthContext:
    runtime = get_runtime()
    ctx = await runtime.auth.authenticate(
        authorization,
        session_id,
        tenant_hint=x_tenant_id,
    )
    if not ctx:
        raise HTTPException(status_code=401, detail="invalid session")
    return ctx


async def get_admin_user(
    authorization: Optional[str] = Header(None),
    session_id: Optional[str] = Header(None, convert_underscores=False),
    x_tenant_id: Optional[str] = Header(None, convert_underscores=False, alias="X-Tenant-ID"),
) -> AuthContext:
    runtime = get_runtime()
    ctx = await runtime.auth.authenticate(
        authorization,
        session_id,
        tenant_hint=x_tenant_id,
        required_role="admin",
    )
    if not ctx:
        raise HTTPException(status_code=403, detail="admin access required")
    return ctx


def _get_owned_conversation(runtime, conversation_id: str, principal: AuthContext) -> Conversation:
    conversation = runtime.store.get_conversation(conversation_id, user_id=principal.user_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="conversation not found")
    if conversation.user_id != principal.user_id:
        raise HTTPException(status_code=403, detail="forbidden")
    return conversation


def _get_owned_context(runtime, context_id: str, principal: AuthContext) -> KnowledgeContext:
    ctx = runtime.store.get_context(context_id)
    if not ctx:
        raise HTTPException(status_code=404, detail="context not found")
    if ctx.owner_user_id != principal.user_id:
        raise HTTPException(status_code=403, detail="forbidden")
    return ctx


def _get_owned_artifact(runtime, artifact_id: str, principal: AuthContext):
    artifact = runtime.store.get_artifact(artifact_id)
    if not artifact:
        raise HTTPException(status_code=404, detail="artifact not found")
    if artifact.owner_user_id and artifact.owner_user_id != principal.user_id:
        raise HTTPException(status_code=403, detail="forbidden")
    if not artifact.owner_user_id and principal.role != "admin":
        raise HTTPException(status_code=403, detail="forbidden")
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


def _apply_session_cookies(response: Response, session: Session, tokens: dict, *, refresh_ttl_minutes: int) -> None:
    response.set_cookie(
        "session_id",
        session.id,
        httponly=True,
        secure=True,
        samesite="lax",
        expires=session.expires_at,
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


@router.post("/auth/signup", response_model=Envelope, status_code=201)
async def signup(body: SignupRequest, response: Response):
    settings = get_settings()
    if not settings.allow_signup:
        raise HTTPException(status_code=403, detail="signup disabled")
    runtime = get_runtime()
    await _enforce_rate_limit(
        runtime,
        f"signup:{body.email.lower()}",
        runtime.settings.signup_rate_limit_per_minute,
        60,
    )
    user, session, tokens = await runtime.auth.signup(
        email=body.email, password=body.password, handle=body.handle, tenant_id=body.tenant_id
    )
    _apply_session_cookies(response, session, tokens, refresh_ttl_minutes=runtime.settings.refresh_token_ttl_minutes)
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


@router.post("/auth/login", response_model=Envelope)
async def login(body: LoginRequest, response: Response):
    runtime = get_runtime()
    await _enforce_rate_limit(
        runtime,
        f"login:{body.email.lower()}",
        runtime.settings.login_rate_limit_per_minute,
        60,
    )
    user, session, tokens = await runtime.auth.login(
        email=body.email, password=body.password, mfa_code=body.mfa_code, tenant_id=body.tenant_id
    )
    if not user or not session:
        raise HTTPException(status_code=401, detail="invalid credentials")
    _apply_session_cookies(response, session, tokens, refresh_ttl_minutes=runtime.settings.refresh_token_ttl_minutes)
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


@router.post("/auth/oauth/{provider}/start", response_model=Envelope)
async def oauth_start(provider: str, body: OAuthStartRequest):
    runtime = get_runtime()
    start = await runtime.auth.start_oauth(provider, redirect_uri=body.redirect_uri, tenant_id=body.tenant_id)
    return Envelope(
        status="ok",
        data=OAuthStartResponse(authorization_url=start["authorization_url"], state=start["state"], provider=provider),
    )


@router.get("/auth/oauth/{provider}/callback", response_model=Envelope)
async def oauth_callback(provider: str, code: str, state: str, tenant_id: Optional[str] = None, response: Response = None):
    runtime = get_runtime()
    user, session, tokens = await runtime.auth.complete_oauth(provider, code, state, tenant_id=tenant_id)
    if not user or not session:
        raise HTTPException(status_code=401, detail="oauth verification failed")
    if response:
        _apply_session_cookies(response, session, tokens, refresh_ttl_minutes=runtime.settings.refresh_token_ttl_minutes)
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


@router.post("/auth/refresh", response_model=Envelope)
async def refresh_tokens(
    body: TokenRefreshRequest,
    authorization: Optional[str] = Header(None),
    x_tenant_id: Optional[str] = Header(None, convert_underscores=False, alias="X-Tenant-ID"),
    response: Response = None,
):
    runtime = get_runtime()
    tenant_hint = body.tenant_id or x_tenant_id
    user, session, tokens = await runtime.auth.refresh_tokens(body.refresh_token, tenant_hint=tenant_hint)
    if not user or not session:
        raise HTTPException(status_code=401, detail="invalid refresh")
    if response:
        _apply_session_cookies(response, session, tokens, refresh_ttl_minutes=runtime.settings.refresh_token_ttl_minutes)
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


@router.get("/admin/users", response_model=Envelope)
async def admin_list_users(
    tenant_id: Optional[str] = None, limit: int = 100, principal: AuthContext = Depends(get_admin_user)
):
    runtime = get_runtime()
    users = runtime.auth.list_users(tenant_id=tenant_id or principal.tenant_id, limit=min(limit, 500))
    return Envelope(status="ok", data=UserListResponse(items=[_user_to_response(u) for u in users]))


@router.post("/admin/users", response_model=Envelope)
async def admin_create_user(body: AdminCreateUserRequest, principal: AuthContext = Depends(get_admin_user)):
    runtime = get_runtime()
    user, password = await runtime.auth.admin_create_user(
        email=body.email,
        password=body.password,
        handle=body.handle,
        tenant_id=body.tenant_id or principal.tenant_id,
        role=body.role,
        plan_tier=body.plan_tier,
        is_active=body.is_active,
        meta=body.meta,
    )
    return Envelope(status="ok", data=_user_to_response(user))


@router.post("/admin/users/{user_id}/role", response_model=Envelope)
async def admin_set_role(user_id: str, body: UpdateUserRoleRequest, principal: AuthContext = Depends(get_admin_user)):
    runtime = get_runtime()
    user = runtime.auth.set_user_role(user_id, body.role)
    if not user:
        raise NotFoundError("user not found", detail={"user_id": user_id})
    return Envelope(status="ok", data=_user_to_response(user))


@router.delete("/admin/users/{user_id}", response_model=Envelope)
async def admin_delete_user(user_id: str, principal: AuthContext = Depends(get_admin_user)):
    runtime = get_runtime()
    removed = runtime.auth.delete_user(user_id)
    if not removed:
        raise NotFoundError("user not found", detail={"user_id": user_id})
    return Envelope(status="ok", data={"deleted": True, "user_id": user_id})


@router.get("/admin/adapters", response_model=Envelope)
async def admin_list_adapters(principal: AuthContext = Depends(get_admin_user)):
    runtime = get_runtime()
    adapters = runtime.store.list_artifacts(type_filter="adapter")
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
                    created_at=a.created_at,
                    updated_at=a.updated_at,
                )
                for a in adapters
            ]
        ),
    )


@router.get("/admin/objects", response_model=Envelope)
async def admin_inspect_objects(
    kind: Optional[str] = None, limit: int = 50, principal: AuthContext = Depends(get_admin_user)
):
    runtime = get_runtime()
    if not hasattr(runtime.store, "inspect_state"):
        raise BadRequestError("inspect not supported")
    details = runtime.store.inspect_state(kind=kind, tenant_id=principal.tenant_id, limit=min(limit, 500))
    summary = {k: len(v) for k, v in details.items()}
    return Envelope(status="ok", data=AdminInspectionResponse(summary=summary, details=details))


@router.post("/auth/mfa/request", response_model=Envelope)
async def request_mfa(body: MFARequest):
    runtime = get_runtime()
    auth_ctx = await runtime.auth.resolve_session(body.session_id, allow_pending_mfa=True)
    if not auth_ctx:
        raise HTTPException(status_code=401, detail="invalid session")
    challenge = await runtime.auth.issue_mfa_challenge(user_id=auth_ctx.user_id)
    return Envelope(status="ok", data=challenge)


@router.post("/auth/mfa/verify", response_model=Envelope)
async def verify_mfa(body: MFAVerifyRequest, response: Response):
    runtime = get_runtime()
    auth_ctx = await runtime.auth.resolve_session(body.session_id, allow_pending_mfa=True)
    if not auth_ctx:
        raise HTTPException(status_code=401, detail="invalid session")
    ok = await runtime.auth.verify_mfa_challenge(user_id=auth_ctx.user_id, code=body.code, session_id=body.session_id)
    if not ok:
        raise HTTPException(status_code=401, detail="invalid mfa")
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
        _apply_session_cookies(response, session, tokens, refresh_ttl_minutes=runtime.settings.refresh_token_ttl_minutes)
    return Envelope(status="ok", data=resp)


@router.post("/auth/reset/request", response_model=Envelope)
async def request_reset(body: PasswordResetRequest):
    runtime = get_runtime()
    await _enforce_rate_limit(
        runtime,
        f"reset:{body.email.lower()}",
        runtime.settings.reset_rate_limit_per_minute,
        60,
    )
    token = await runtime.auth.initiate_password_reset(body.email)
    # Token delivery should be handled out-of-band; avoid exposing it directly.
    return Envelope(status="ok", data={"status": "sent"})


@router.post("/auth/reset/confirm", response_model=Envelope)
async def confirm_reset(body: PasswordResetConfirm):
    runtime = get_runtime()
    ok = await runtime.auth.complete_password_reset(body.token, body.new_password)
    if not ok:
        raise HTTPException(status_code=400, detail="invalid token")
    return Envelope(status="ok", data={"status": "reset"})


@router.post("/auth/logout", response_model=Envelope)
async def logout(
    session_id: Optional[str] = Header(None, convert_underscores=False),
    authorization: Optional[str] = Header(None),
    response: Response = None,
):
    runtime = get_runtime()
    if authorization or session_id:
        ctx = await runtime.auth.authenticate(authorization, session_id, allow_pending_mfa=True)
        target_session = session_id or (ctx.session_id if ctx else None)
        if not target_session:
            raise HTTPException(status_code=401, detail="invalid session")
        await runtime.auth.revoke(target_session)
    if response:
        response.delete_cookie("session_id", path="/")
        response.delete_cookie("refresh_token", path="/")
    return Envelope(status="ok", data={"message": "session revoked"})


@router.post("/chat", response_model=Envelope)
async def chat(
    body: ChatRequest,
    principal: AuthContext = Depends(get_user),
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
):
    runtime = get_runtime()
    user_id = principal.user_id
    async with IdempotencyGuard("chat", user_id, idempotency_key, require=True) as idem:
        if idem.cached:
            return idem.cached
        await _enforce_rate_limit(
            runtime,
            f"chat:{user_id}",
            runtime.settings.chat_rate_limit_per_minute,
            runtime.settings.chat_rate_limit_window_seconds,
        )
        conversation: Conversation | None = None
        if body.context_id:
            _get_owned_context(runtime, body.context_id, principal)
        if body.conversation_id:
            conversation = _get_owned_conversation(runtime, body.conversation_id, principal)
            conversation_id = conversation.id
        else:
            conversation = runtime.store.create_conversation(user_id=user_id, active_context_id=body.context_id)
            conversation_id = conversation.id
        if conversation and conversation.active_context_id:
            _get_owned_context(runtime, conversation.active_context_id, principal)
        user_content = body.message.content
        voice_meta: dict = {}
        if body.message.mode == "voice":
            try:
                audio_bytes = base64.b64decode(body.message.content)
            except Exception:
                audio_bytes = body.message.content.encode()
            transcript = runtime.voice.transcribe(audio_bytes, user_id=user_id)
            user_content = transcript.get("transcript", body.message.content)
            voice_meta = {"mode": "voice", "transcript": transcript}
        user_content_struct = normalize_content_struct(body.message.content_struct, user_content)
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
            body.context_id,
            user_id,
            tenant_id=principal.tenant_id,
        )
        assistant_content_struct = normalize_content_struct(
            orchestration.get("content_struct") if isinstance(orchestration, dict) else None,
            orchestration.get("content") if isinstance(orchestration, dict) else None,
        )
        assistant_msg = runtime.store.append_message(
            conversation_id,
            sender="assistant",
            role="assistant",
            content=orchestration["content"],
            content_struct=assistant_content_struct,
            meta={
                "adapters": orchestration.get("adapters", []),
                "adapter_gates": orchestration.get("adapter_gates", []),
                "routing_trace": orchestration.get("routing_trace", []),
                "workflow_trace": orchestration.get("workflow_trace", []),
                "usage": orchestration.get("usage", {}),
            },
        )
        resp = ChatResponse(
            message_id=assistant_msg.id,
            conversation_id=conversation_id,
            content=assistant_msg.content,
            content_struct=assistant_msg.content_struct,
            workflow_id=body.workflow_id,
            adapters=orchestration.get("adapters", []),
            adapter_gates=orchestration.get("adapter_gates", []),
            usage=orchestration.get("usage", {}),
            context_snippets=orchestration.get("context_snippets", []),
            routing_trace=orchestration.get("routing_trace", []),
            workflow_trace=orchestration.get("workflow_trace", []),
        )
        envelope = Envelope(status="ok", data=resp.model_dump(), request_id=idem.request_id)
        if runtime.cache:
            history = runtime.store.list_messages(conversation_id, user_id=principal.user_id)
            await runtime.workflow.cache_conversation_state(conversation_id, history)
        await idem.store_result(envelope)
        return envelope
    # Exceptions bubble through the guard which records failed states


@router.post("/preferences", response_model=Envelope)
async def record_preference(body: PreferenceEventRequest, principal: AuthContext = Depends(get_user)):
    runtime = get_runtime()
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
    )
    runtime.clusterer.cluster_user_preferences(principal.user_id)
    runtime.clusterer.promote_skill_adapters()
    resp = PreferenceEventResponse(id=event.id, cluster_id=event.cluster_id, feedback=event.feedback, created_at=event.created_at)
    return Envelope(status="ok", data=resp)


@router.post("/preferences/routing_feedback", response_model=Envelope)
async def record_routing_feedback(body: PreferenceEventRequest, principal: AuthContext = Depends(get_user)):
    runtime = get_runtime()
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
    )
    resp = PreferenceEventResponse(id=event.id, cluster_id=event.cluster_id, feedback=event.feedback, created_at=event.created_at)
    return Envelope(status="ok", data=resp)


@router.get("/preferences/insights", response_model=Envelope)
async def preference_insights(principal: AuthContext = Depends(get_user)):
    runtime = get_runtime()
    summary = runtime.training.summarize_preferences(principal.user_id)
    return Envelope(status="ok", data=PreferenceInsightsResponse(**summary))


@router.get("/artifacts", response_model=Envelope)
async def list_artifacts(
    type: Optional[str] = None,
    kind: Optional[str] = None,
    page: int = 1,
    page_size: int = 100,
    principal: AuthContext = Depends(get_user),
):
    runtime = get_runtime()
    kind_filter = kind or (type if type and "." in type else None)
    type_filter = type if type and "." not in type else None
    if not type_filter and kind_filter:
        type_filter = kind_filter.split(".", 1)[0]
    resolved_page_size = min(max(page_size, 1), 500)
    items = [
        ArtifactResponse(
            id=a.id,
            type=a.type,
            kind=a.schema.get("kind") if isinstance(a.schema, dict) else None,
            name=a.name,
            description=a.description,
            schema=a.schema,
            owner_user_id=a.owner_user_id,
            created_at=a.created_at,
            updated_at=a.updated_at,
        )
        for a in runtime.store.list_artifacts(
            type_filter=type_filter,
            kind_filter=kind_filter,
            owner_user_id=principal.user_id,
            page=page,
            page_size=resolved_page_size,
        )
    ]
    return Envelope(status="ok", data=ArtifactListResponse(items=items))


@router.get("/artifacts/{artifact_id}", response_model=Envelope)
async def get_artifact(artifact_id: str, principal: AuthContext = Depends(get_user)):
    runtime = get_runtime()
    artifact = _get_owned_artifact(runtime, artifact_id, principal)
    resp = ArtifactResponse(
        id=artifact.id,
        type=artifact.type,
        kind=artifact.schema.get("kind") if isinstance(artifact.schema, dict) else None,
        name=artifact.name,
        description=artifact.description,
        schema=artifact.schema,
        owner_user_id=artifact.owner_user_id,
        created_at=artifact.created_at,
        updated_at=artifact.updated_at,
    )
    return Envelope(status="ok", data=resp)


@router.get("/tools/specs", response_model=Envelope)
async def list_tool_specs(
    page: int = 1, page_size: int = 50, principal: AuthContext = Depends(get_user)
):
    runtime = get_runtime()
    resolved_page_size = min(max(page_size, 1), 200)
    artifacts = runtime.store.list_artifacts(
        type_filter="tool",
        kind_filter="tool.spec",
        owner_user_id=principal.user_id,
        page=page,
        page_size=resolved_page_size,
    )
    items = [
        ArtifactResponse(
            id=a.id,
            type=a.type,
            kind=a.schema.get("kind") if isinstance(a.schema, dict) else None,
            name=a.name,
            description=a.description,
            schema=a.schema,
            owner_user_id=a.owner_user_id,
            created_at=a.created_at,
            updated_at=a.updated_at,
        )
        for a in artifacts
    ]
    next_page = page + 1 if len(items) == resolved_page_size else None
    return Envelope(status="ok", data=ToolSpecListResponse(items=items, next_page=next_page, page_size=resolved_page_size))


@router.get("/tools/specs/{artifact_id}", response_model=Envelope)
async def get_tool_spec(artifact_id: str, principal: AuthContext = Depends(get_user)):
    runtime = get_runtime()
    artifact = _get_owned_artifact(runtime, artifact_id, principal)
    if not (isinstance(artifact.schema, dict) and artifact.schema.get("kind") == "tool.spec"):
        raise NotFoundError("tool spec not found", detail={"artifact_id": artifact_id})
    resp = ArtifactResponse(
        id=artifact.id,
        type=artifact.type,
        kind=artifact.schema.get("kind") if isinstance(artifact.schema, dict) else None,
        name=artifact.name,
        description=artifact.description,
        schema=artifact.schema,
        owner_user_id=artifact.owner_user_id,
        created_at=artifact.created_at,
        updated_at=artifact.updated_at,
    )
    return Envelope(status="ok", data=resp)


@router.get("/workflows", response_model=Envelope)
async def list_workflows(
    page: int = 1, page_size: int = 50, principal: AuthContext = Depends(get_user)
):
    runtime = get_runtime()
    resolved_page_size = min(max(page_size, 1), 200)
    artifacts = runtime.store.list_artifacts(
        type_filter="workflow", kind_filter="workflow.chat", page=page, page_size=resolved_page_size
    )
    items = [
        ArtifactResponse(
            id=a.id,
            type=a.type,
            kind=a.schema.get("kind") if isinstance(a.schema, dict) else None,
            name=a.name,
            description=a.description,
            schema=a.schema,
            owner_user_id=a.owner_user_id,
            created_at=a.created_at,
            updated_at=a.updated_at,
        )
        for a in artifacts
    ]
    next_page = page + 1 if len(items) == resolved_page_size else None
    return Envelope(status="ok", data=WorkflowListResponse(items=items, next_page=next_page, page_size=resolved_page_size))


@router.get("/artifacts/{artifact_id}/versions", response_model=Envelope)
async def list_artifact_versions(artifact_id: str, principal: AuthContext = Depends(get_user)):
    runtime = get_runtime()
    versions = runtime.store.list_artifact_versions(artifact_id)
    if not versions:
        artifact = runtime.store.get_artifact(artifact_id)
        if not artifact:
            raise NotFoundError("artifact not found", detail={"artifact_id": artifact_id})
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


@router.post("/tools/{tool_id}/invoke", response_model=Envelope)
async def invoke_tool(
    tool_id: str,
    body: ToolInvokeRequest,
    principal: AuthContext = Depends(get_user),
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
):
    runtime = get_runtime()
    async with IdempotencyGuard(
        f"tools:{tool_id}:invoke", principal.user_id, idempotency_key, require=True
    ) as idem:
        if idem.cached:
            return idem.cached
        artifact = _get_owned_artifact(runtime, tool_id, principal)
        if not (isinstance(artifact.schema, dict) and artifact.schema.get("kind") == "tool.spec"):
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
            context_snippets=result.get("context_snippets") if isinstance(result, dict) else None,
        )
        envelope = Envelope(status="ok", data=response, request_id=idem.request_id)
        await idem.store_result(envelope)
        return envelope


@router.post("/artifacts", response_model=Envelope, status_code=201)
async def create_artifact(
    body: ArtifactRequest,
    principal: AuthContext = Depends(get_user),
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
):
    runtime = get_runtime()
    async with IdempotencyGuard("artifacts:create", principal.user_id, idempotency_key, require=True) as idem:
        if idem.cached:
            return idem.cached
        if not isinstance(body.schema, dict):
            raise BadRequestError("artifact schema must be an object", detail={"provided_type": type(body.schema).__name__})
        schema_kind = body.schema.get("kind")
        type_prefix = body.type
        if schema_kind and not type_prefix:
            type_prefix = schema_kind.split(".", 1)[0]
        if schema_kind and type_prefix and not schema_kind.startswith(f"{type_prefix}."):
            raise BadRequestError("kind must start with the type prefix", detail={"kind": schema_kind, "type": type_prefix})
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
        resp = ArtifactResponse(
            id=artifact.id,
            type=artifact.type,
            kind=artifact.schema.get("kind") if isinstance(artifact.schema, dict) else None,
            name=artifact.name,
            description=artifact.description,
            schema=artifact.schema,
            owner_user_id=artifact.owner_user_id,
            created_at=artifact.created_at,
            updated_at=artifact.updated_at,
        )
        envelope = Envelope(status="ok", data=resp, request_id=idem.request_id)
        await idem.store_result(envelope)
        return envelope


@router.patch("/artifacts/{artifact_id}", response_model=Envelope)
async def patch_artifact(artifact_id: str, body: ArtifactRequest, principal: AuthContext = Depends(get_user)):
    runtime = get_runtime()
    current = _get_owned_artifact(runtime, artifact_id, principal)
    if not isinstance(body.schema, dict):
        raise BadRequestError("artifact schema must be an object", detail={"provided_type": type(body.schema).__name__})
    schema_kind = body.schema.get("kind")
    artifact_schema = dict(body.schema)
    if schema_kind:
        if not schema_kind.startswith(f"{current.type}."):
            raise BadRequestError("kind must start with the type prefix", detail={"kind": schema_kind, "type": current.type})
        artifact_schema["kind"] = schema_kind
    artifact = runtime.store.update_artifact(
        artifact_id,
        schema=artifact_schema,
        description=body.description,
        version_author=principal.user_id,
    )
    if not artifact:
        raise NotFoundError("artifact not found", detail={"artifact_id": artifact_id})
    resp = ArtifactResponse(
        id=artifact.id,
        type=artifact.type,
        kind=artifact.schema.get("kind") if isinstance(artifact.schema, dict) else None,
        name=artifact.name,
        description=artifact.description,
        schema=artifact.schema,
        owner_user_id=artifact.owner_user_id,
        created_at=artifact.created_at,
        updated_at=artifact.updated_at,
    )
    return Envelope(status="ok", data=resp)


@router.post("/config/propose_patch", response_model=Envelope)
async def propose_patch(body: ConfigPatchRequest, principal: AuthContext = Depends(get_admin_user)):
    runtime = get_runtime()
    proposer = "human_admin" if principal.role == "admin" else "user"
    audit = runtime.store.record_config_patch(
        artifact_id=body.artifact_id, proposer=proposer, patch=body.patch, justification=body.justification
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


@router.get("/config/patches", response_model=Envelope)
async def list_config_patches(status: Optional[str] = None, principal: AuthContext = Depends(get_admin_user)):
    runtime = get_runtime()
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


@router.post("/config/patches/{patch_id}/decide", response_model=Envelope)
async def decide_config_patch(patch_id: int, body: ConfigPatchDecisionRequest, principal: AuthContext = Depends(get_admin_user)):
    runtime = get_runtime()
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


@router.post("/config/patches/{patch_id}/apply", response_model=Envelope)
async def apply_config_patch(patch_id: int, principal: AuthContext = Depends(get_admin_user)):
    runtime = get_runtime()
    result = runtime.config_ops.apply_patch(patch_id, approver_user_id=principal.user_id)
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
    return Envelope(status="ok", data=resp)


@router.post("/config/auto_patch", response_model=Envelope)
async def auto_patch(body: AutoPatchRequest, principal: AuthContext = Depends(get_admin_user)):
    runtime = get_runtime()
    audit = runtime.config_ops.auto_generate_patch(body.artifact_id, principal.user_id, goal=body.goal)
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


@router.post("/files/upload", response_model=Envelope)
async def upload_file(
    file: UploadFile = File(...),
    context_id: Optional[str] = Form(None),
    chunk_size: Optional[int] = Form(None),
    principal: AuthContext = Depends(get_user),
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
):
    runtime = get_runtime()
    async with IdempotencyGuard("files:upload", principal.user_id, idempotency_key, require=True) as idem:
        if idem.cached:
            return idem.cached
        dest_dir = Path(runtime.settings.shared_fs_root) / "users" / principal.user_id / "files"
        max_bytes = max(1, runtime.settings.max_upload_bytes)
        contents = await file.read(max_bytes + 1)
        if len(contents) > max_bytes:
            raise HTTPException(status_code=413, detail="file too large")
        dest_path = safe_join(dest_dir, file.filename)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(contents)
        chunk_count = None
        if context_id:
            _get_owned_context(runtime, context_id, principal)
            try:
                chunk_count = runtime.rag.ingest_file(context_id, str(dest_path), chunk_size=chunk_size)
            except ConstraintViolation:
                dest_path.unlink(missing_ok=True)
                raise
        resp = FileUploadResponse(fs_path=file.filename, context_id=context_id, chunk_count=chunk_count)
        envelope = Envelope(status="ok", data=resp, request_id=idem.request_id)
        await idem.store_result(envelope)
        return envelope


@router.get("/conversations/{conversation_id}/messages", response_model=Envelope)
async def list_messages(conversation_id: str, principal: AuthContext = Depends(get_user)):
    runtime = get_runtime()
    _get_owned_conversation(runtime, conversation_id, principal)
    msgs = runtime.store.list_messages(conversation_id, user_id=principal.user_id)
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
    return Envelope(status="ok", data=ConversationMessagesResponse(conversation_id=conversation_id, messages=payload))


@router.get("/conversations", response_model=Envelope)
async def list_conversations(principal: AuthContext = Depends(get_user)):
    runtime = get_runtime()
    convs = runtime.store.list_conversations(principal.user_id)
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
    return Envelope(status="ok", data=ConversationListResponse(items=items))


@router.post("/contexts", response_model=Envelope, status_code=201)
async def create_context(
    body: KnowledgeContextRequest,
    principal: AuthContext = Depends(get_user),
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
):
    runtime = get_runtime()
    async with IdempotencyGuard("contexts:create", principal.user_id, idempotency_key, require=True) as idem:
        if idem.cached:
            return idem.cached
        ctx_meta = {"embedding_model_id": runtime.rag.embedding_model_id}
        ctx = runtime.store.upsert_context(
            owner_user_id=principal.user_id, name=body.name, description=body.description, meta=ctx_meta
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


@router.get("/contexts", response_model=Envelope)
async def list_contexts(principal: AuthContext = Depends(get_user)):
    runtime = get_runtime()
    contexts = runtime.store.list_contexts(owner_user_id=principal.user_id)
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


@router.get("/contexts/{context_id}/chunks", response_model=Envelope)
async def list_chunks(context_id: str, principal: AuthContext = Depends(get_user)):
    runtime = get_runtime()
    _get_owned_context(runtime, context_id, principal)
    chunks = runtime.store.list_chunks(context_id, owner_user_id=principal.user_id)
    data = [
        KnowledgeChunkResponse(
            id=ch.id if ch.id is not None else 0,
            context_id=ch.context_id,
            fs_path=ch.fs_path,
            content=ch.content,
            chunk_index=ch.chunk_index,
        )
        for ch in chunks
    ]
    return Envelope(status="ok", data=KnowledgeChunkListResponse(items=data))


@router.post("/voice/transcribe", response_model=Envelope)
async def transcribe_voice(file: UploadFile = File(...), principal: AuthContext = Depends(get_user)):
    runtime = get_runtime()
    audio_bytes = await file.read()
    result = runtime.voice.transcribe(audio_bytes, user_id=principal.user_id)
    return Envelope(status="ok", data=VoiceTranscriptionResponse(**result))


@router.post("/voice/synthesize", response_model=Envelope)
async def synthesize_voice(body: VoiceSynthesisRequest, principal: AuthContext = Depends(get_user)):
    runtime = get_runtime()
    audio = runtime.voice.synthesize(body.text, user_id=principal.user_id, voice=body.voice)
    return Envelope(status="ok", data=VoiceSynthesisResponse(**audio))


@router.websocket("/v1/chat/stream")
async def websocket_chat(ws: WebSocket):
    runtime = get_runtime()
    await ws.accept()
    try:
        init = await ws.receive_json()
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
        if runtime.cache:
            allowed = await runtime.cache.check_rate_limit(
                f"chat:{user_id}", runtime.settings.chat_rate_limit_per_minute, runtime.settings.chat_rate_limit_window_seconds
            )
            if not allowed:
                await ws.close(code=4429)
                return
        convo_id = init.get("conversation_id")
        if convo_id:
            _get_owned_conversation(runtime, convo_id, auth_ctx)
        else:
            convo_id = runtime.store.create_conversation(user_id=user_id).id
        if init.get("context_id"):
            _get_owned_context(runtime, init.get("context_id"), auth_ctx)
        runtime.store.append_message(convo_id, sender="user", role="user", content=init.get("message", ""))
        orchestration = await runtime.workflow.run(
            init.get("workflow_id"),
            convo_id,
            init.get("message", ""),
            init.get("context_id"),
            user_id,
            tenant_id=auth_ctx.tenant_id,
        )
        assistant_content_struct = normalize_content_struct(
            orchestration.get("content_struct") if isinstance(orchestration, dict) else None,
            orchestration.get("content") if isinstance(orchestration, dict) else None,
        )
        assistant_msg = runtime.store.append_message(
            convo_id,
            sender="assistant",
            role="assistant",
            content=orchestration["content"],
            content_struct=assistant_content_struct,
            meta={
                "adapters": orchestration.get("adapters", []),
                "adapter_gates": orchestration.get("adapter_gates", []),
                "routing_trace": orchestration.get("routing_trace", []),
                "workflow_trace": orchestration.get("workflow_trace", []),
                "usage": orchestration.get("usage", {}),
            },
        )
        await ws.send_json(
            Envelope(
                status="ok",
                data=ChatResponse(
                    message_id=assistant_msg.id,
                    conversation_id=convo_id,
                    content=assistant_msg.content,
                    content_struct=assistant_msg.content_struct,
                    workflow_id=init.get("workflow_id"),
                    adapters=orchestration.get("adapters", []),
                    adapter_gates=orchestration.get("adapter_gates", []),
                    usage=orchestration.get("usage", {}),
                    context_snippets=orchestration.get("context_snippets", []),
                    routing_trace=orchestration.get("routing_trace", []),
                    workflow_trace=orchestration.get("workflow_trace", []),
                ).model_dump(),
            ).model_dump()
        )
    except WebSocketDisconnect:
        return
