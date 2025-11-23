from __future__ import annotations

import base64
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile, WebSocket, WebSocketDisconnect

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
    PasswordResetConfirm,
    PasswordResetRequest,
    SignupRequest,
    KnowledgeChunkResponse,
    KnowledgeContextRequest,
    KnowledgeContextResponse,
    FileUploadResponse,
    PreferenceEventRequest,
    PreferenceEventResponse,
    PreferenceInsightsResponse,
    VoiceSynthesisRequest,
    VoiceSynthesisResponse,
    VoiceTranscriptionResponse,
)
from liminallm.storage.errors import ConstraintViolation
from liminallm.config import get_settings
from liminallm.service.fs import PathTraversalError, safe_join
from liminallm.service.runtime import get_runtime

router = APIRouter(prefix="/v1")


# Dependency placeholder for auth: replace with signed tokens + role scopes
# once the API surface is hardened beyond demo mode.
async def get_user(session_id: Optional[str] = Header(None, convert_underscores=False)) -> str:
    runtime = get_runtime()
    user_id = await runtime.auth.resolve_session(session_id)
    if not user_id:
        raise HTTPException(status_code=401, detail="invalid session")
    return user_id


@router.post("/auth/signup", response_model=Envelope)
async def signup(body: SignupRequest):
    settings = get_settings()
    if not settings.allow_signup:
        raise HTTPException(status_code=403, detail="signup disabled")
    runtime = get_runtime()
    try:
        user, session = await runtime.auth.signup(email=body.email, password=body.password, handle=body.handle)
    except ConstraintViolation as err:
        raise HTTPException(status_code=409, detail=err.message)
    return Envelope(
        status="ok",
        data=AuthResponse(
            user_id=user.id, session_id=session.id, session_expires_at=session.expires_at, mfa_required=session.mfa_required
        ),
    )


@router.post("/auth/login", response_model=Envelope)
async def login(body: LoginRequest):
    runtime = get_runtime()
    user, session = await runtime.auth.login(email=body.email, password=body.password, mfa_code=body.mfa_code)
    if not user or not session:
        raise HTTPException(status_code=401, detail="invalid credentials")
    return Envelope(
        status="ok",
        data=AuthResponse(
            user_id=user.id, session_id=session.id, session_expires_at=session.expires_at, mfa_required=session.mfa_required
        ),
    )


@router.post("/auth/mfa/request", response_model=Envelope)
async def request_mfa(body: MFARequest):
    runtime = get_runtime()
    user_id = await runtime.auth.resolve_session(body.session_id, allow_pending_mfa=True)
    if not user_id:
        raise HTTPException(status_code=401, detail="invalid session")
    challenge = await runtime.auth.issue_mfa_challenge(user_id=user_id)
    return Envelope(status="ok", data=challenge)


@router.post("/auth/mfa/verify", response_model=Envelope)
async def verify_mfa(body: MFAVerifyRequest):
    runtime = get_runtime()
    user_id = await runtime.auth.resolve_session(body.session_id, allow_pending_mfa=True)
    if not user_id:
        raise HTTPException(status_code=401, detail="invalid session")
    ok = await runtime.auth.verify_mfa_challenge(user_id=user_id, code=body.code, session_id=body.session_id)
    if not ok:
        raise HTTPException(status_code=401, detail="invalid mfa")
    return Envelope(status="ok", data={"status": "verified"})


@router.post("/auth/reset/request", response_model=Envelope)
async def request_reset(body: PasswordResetRequest):
    runtime = get_runtime()
    token = await runtime.auth.initiate_password_reset(body.email)
    return Envelope(status="ok", data={"token": token})


@router.post("/auth/reset/confirm", response_model=Envelope)
async def confirm_reset(body: PasswordResetConfirm):
    runtime = get_runtime()
    ok = await runtime.auth.complete_password_reset(body.token, body.new_password)
    if not ok:
        raise HTTPException(status_code=400, detail="invalid token")
    return Envelope(status="ok", data={"status": "reset"})


@router.post("/auth/logout", response_model=Envelope)
async def logout(session_id: Optional[str] = Header(None, convert_underscores=False)):
    if session_id:
        runtime = get_runtime()
        await runtime.auth.revoke(session_id)
    return Envelope(status="ok", data={"message": "session revoked"})


@router.post("/chat", response_model=Envelope)
async def chat(body: ChatRequest, user_id: str = Depends(get_user), idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key")):
    runtime = get_runtime()
    if runtime.cache:
        allowed = await runtime.cache.check_rate_limit(
            f"chat:{user_id}", runtime.settings.chat_rate_limit_per_minute, runtime.settings.chat_rate_limit_window_seconds
        )
        if not allowed:
            raise HTTPException(status_code=429, detail="rate limit exceeded")
    if body.conversation_id:
        conversation_id = body.conversation_id
    else:
        conversation = runtime.store.create_conversation(user_id=user_id, active_context_id=body.context_id)
        conversation_id = conversation.id
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
    runtime.store.append_message(conversation_id, sender="user", role="user", content=user_content, meta=voice_meta or None)
    orchestration = runtime.workflow.run(body.workflow_id, conversation_id, user_content, body.context_id, user_id)
    assistant_msg = runtime.store.append_message(
        conversation_id,
        sender="assistant",
        role="assistant",
        content=orchestration["content"],
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
        workflow_id=body.workflow_id,
        adapters=orchestration.get("adapters", []),
        adapter_gates=orchestration.get("adapter_gates", []),
        usage=orchestration.get("usage", {}),
        context_snippets=orchestration.get("context_snippets", []),
        routing_trace=orchestration.get("routing_trace", []),
        workflow_trace=orchestration.get("workflow_trace", []),
    )
    return Envelope(status="ok", data=resp.model_dump())


@router.post("/preferences", response_model=Envelope)
async def record_preference(body: PreferenceEventRequest, user_id: str = Depends(get_user)):
    runtime = get_runtime()
    try:
        event = runtime.training.record_feedback_event(
            user_id=user_id,
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
    except ConstraintViolation as err:
        raise HTTPException(status_code=409, detail=err.message)
    runtime.clusterer.cluster_user_preferences(user_id)
    runtime.clusterer.promote_skill_adapters()
    resp = PreferenceEventResponse(id=event.id, cluster_id=event.cluster_id, feedback=event.feedback, created_at=event.created_at)
    return Envelope(status="ok", data=resp)


@router.post("/preferences/routing_feedback", response_model=Envelope)
async def record_routing_feedback(body: PreferenceEventRequest, user_id: str = Depends(get_user)):
    runtime = get_runtime()
    event = runtime.training.record_feedback_event(
        user_id=user_id,
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
async def preference_insights(user_id: str = Depends(get_user)):
    runtime = get_runtime()
    summary = runtime.training.summarize_preferences(user_id)
    return Envelope(status="ok", data=PreferenceInsightsResponse(**summary))


@router.get("/artifacts", response_model=Envelope)
async def list_artifacts(
    type: Optional[str] = None, kind: Optional[str] = None, user_id: str = Depends(get_user)
):
    runtime = get_runtime()
    kind_filter = kind or (type if type and "." in type else None)
    type_filter = type if type and "." not in type else None
    if not type_filter and kind_filter:
        type_filter = kind_filter.split(".", 1)[0]
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
        for a in runtime.store.list_artifacts(type_filter=type_filter, kind_filter=kind_filter)
    ]
    return Envelope(status="ok", data=ArtifactListResponse(items=items))


@router.get("/artifacts/{artifact_id}", response_model=Envelope)
async def get_artifact(artifact_id: str, user_id: str = Depends(get_user)):
    runtime = get_runtime()
    artifact = runtime.store.get_artifact(artifact_id)
    if not artifact:
        raise HTTPException(status_code=404, detail="artifact not found")
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


@router.get("/artifacts/{artifact_id}/versions", response_model=Envelope)
async def list_artifact_versions(artifact_id: str, user_id: str = Depends(get_user)):
    runtime = get_runtime()
    versions = runtime.store.list_artifact_versions(artifact_id)
    if not versions:
        artifact = runtime.store.get_artifact(artifact_id)
        if not artifact:
            raise HTTPException(status_code=404, detail="artifact not found")
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


@router.post("/artifacts", response_model=Envelope)
async def create_artifact(body: ArtifactRequest, user_id: str = Depends(get_user)):
    runtime = get_runtime()
    if not isinstance(body.schema, dict):
        raise HTTPException(status_code=400, detail="artifact schema must be an object")
    schema_kind = body.kind or body.schema.get("kind")
    type_prefix = body.type
    if schema_kind and not type_prefix:
        type_prefix = schema_kind.split(".", 1)[0]
    if schema_kind and type_prefix and not schema_kind.startswith(f"{type_prefix}."):
        raise HTTPException(status_code=400, detail="kind must start with the type prefix")
    if not type_prefix:
        raise HTTPException(status_code=400, detail="type or kind is required")
    artifact_schema = dict(body.schema)
    if schema_kind:
        artifact_schema["kind"] = schema_kind
    try:
        artifact = runtime.store.create_artifact(
            type_=type_prefix,
            name=body.name,
            description=body.description or "",
            schema=artifact_schema,
            owner_user_id=user_id,
            created_by=user_id,
        )
    except ConstraintViolation as err:
        raise HTTPException(status_code=409, detail=err.message)
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


@router.patch("/artifacts/{artifact_id}", response_model=Envelope)
async def patch_artifact(artifact_id: str, body: ArtifactRequest, user_id: str = Depends(get_user)):
    runtime = get_runtime()
    current = runtime.store.get_artifact(artifact_id)
    if not current:
        raise HTTPException(status_code=404, detail="artifact not found")
    if not isinstance(body.schema, dict):
        raise HTTPException(status_code=400, detail="artifact schema must be an object")
    schema_kind = body.kind or body.schema.get("kind")
    artifact_schema = dict(body.schema)
    if schema_kind:
        if not schema_kind.startswith(f"{current.type}."):
            raise HTTPException(status_code=400, detail="kind must start with the type prefix")
        artifact_schema["kind"] = schema_kind
    artifact = runtime.store.update_artifact(
        artifact_id,
        schema=artifact_schema,
        description=body.description,
        created_by=user_id,
    )
    if not artifact:
        raise HTTPException(status_code=404, detail="artifact not found")
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
async def propose_patch(body: ConfigPatchRequest, user_id: str = Depends(get_user)):
    runtime = get_runtime()
    audit = runtime.store.record_config_patch(
        artifact_id=body.artifact_id, proposer_user_id=user_id, patch=body.patch, justification=body.justification
    )
    resp = ConfigPatchAuditResponse(
        id=audit.id,
        artifact_id=audit.artifact_id,
        justification=audit.justification,
        status=audit.status,
        patch=audit.patch,
        decided_at=audit.decided_at,
        applied_at=audit.applied_at,
        meta=audit.meta,
    )
    return Envelope(status="ok", data=resp)


@router.get("/config/patches", response_model=Envelope)
async def list_config_patches(status: Optional[str] = None, user_id: str = Depends(get_user)):
    runtime = get_runtime()
    patches = runtime.store.list_config_patches(status)
    items = [
        ConfigPatchAuditResponse(
            id=p.id,
            artifact_id=p.artifact_id,
            justification=p.justification,
            status=p.status,
            patch=p.patch,
            decided_at=p.decided_at,
            applied_at=p.applied_at,
            meta=p.meta,
        )
        for p in patches
    ]
    return Envelope(status="ok", data=ConfigPatchListResponse(items=items))


@router.post("/config/patches/{patch_id}/decide", response_model=Envelope)
async def decide_config_patch(patch_id: str, body: ConfigPatchDecisionRequest, user_id: str = Depends(get_user)):
    runtime = get_runtime()
    decision = runtime.config_ops.decide_patch(patch_id, body.decision, body.reason)
    if not decision:
        raise HTTPException(status_code=404, detail="patch not found")
    resp = ConfigPatchAuditResponse(
        id=decision.id,
        artifact_id=decision.artifact_id,
        justification=decision.justification,
        status=decision.status,
        patch=decision.patch,
        decided_at=decision.decided_at,
        applied_at=decision.applied_at,
        meta=decision.meta,
    )
    return Envelope(status="ok", data=resp)


@router.post("/config/patches/{patch_id}/apply", response_model=Envelope)
async def apply_config_patch(patch_id: str, user_id: str = Depends(get_user)):
    runtime = get_runtime()
    result = runtime.config_ops.apply_patch(patch_id, approver_user_id=user_id)
    if not result:
        raise HTTPException(status_code=404, detail="patch not found")
    patch = result.get("patch")
    resp = ConfigPatchAuditResponse(
        id=patch.id,
        artifact_id=patch.artifact_id,
        justification=patch.justification,
        status="applied",
        patch=patch.patch,
        decided_at=patch.decided_at,
        applied_at=patch.applied_at or patch.updated_at,
        meta=patch.meta,
    )
    return Envelope(status="ok", data=resp)


@router.post("/config/auto_patch", response_model=Envelope)
async def auto_patch(body: AutoPatchRequest, user_id: str = Depends(get_user)):
    runtime = get_runtime()
    audit = runtime.config_ops.auto_generate_patch(body.artifact_id, user_id, goal=body.goal)
    resp = ConfigPatchAuditResponse(
        id=audit.id,
        artifact_id=audit.artifact_id,
        justification=audit.justification,
        status=audit.status,
        patch=audit.patch,
        decided_at=audit.decided_at,
        applied_at=audit.applied_at,
        meta=audit.meta,
    )
    return Envelope(status="ok", data=resp)


@router.post("/files/upload", response_model=Envelope)
async def upload_file(
    file: UploadFile = File(...),
    context_id: Optional[str] = Form(None),
    user_id: str = Depends(get_user),
):
    runtime = get_runtime()
    dest_dir = Path(runtime.settings.shared_fs_root) / "users" / user_id / "files"
    contents = await file.read()
    try:
        dest_path = safe_join(dest_dir, file.filename)
    except PathTraversalError as err:
        raise HTTPException(status_code=400, detail=str(err))
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_bytes(contents)
    chunk_count = None
    if context_id:
        try:
            chunk_count = runtime.rag.ingest_file(context_id, str(dest_path))
        except ConstraintViolation as err:
            dest_path.unlink(missing_ok=True)
            raise HTTPException(status_code=409, detail=err.message)
    resp = FileUploadResponse(fs_path=str(dest_path), context_id=context_id, chunk_count=chunk_count)
    return Envelope(status="ok", data=resp)


@router.get("/conversations/{conversation_id}/messages", response_model=Envelope)
async def list_messages(conversation_id: str, user_id: str = Depends(get_user)):
    runtime = get_runtime()
    msgs = runtime.store.list_messages(conversation_id)
    payload = [
        {
            "id": m.id,
            "role": m.role,
            "sender": m.sender,
            "content": m.content,
            "seq": m.seq,
            "created_at": m.created_at,
            "meta": m.meta,
        }
        for m in msgs
    ]
    return Envelope(status="ok", data=ConversationMessagesResponse(conversation_id=conversation_id, messages=payload))


@router.get("/conversations", response_model=Envelope)
async def list_conversations(user_id: str = Depends(get_user)):
    runtime = get_runtime()
    convs = runtime.store.list_conversations(user_id)
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


@router.post("/contexts", response_model=Envelope)
async def create_context(body: KnowledgeContextRequest, user_id: str = Depends(get_user)):
    runtime = get_runtime()
    try:
        ctx = runtime.store.upsert_context(owner_user_id=user_id, name=body.name, description=body.description)
    except ConstraintViolation as err:
        raise HTTPException(status_code=409, detail=err.message)
    if body.text:
        runtime.rag.ingest_text(ctx.id, body.text)
    return Envelope(
        status="ok",
        data=KnowledgeContextResponse(
            id=ctx.id,
            name=ctx.name,
            description=ctx.description,
            created_at=ctx.created_at,
            updated_at=ctx.updated_at,
            owner_user_id=ctx.owner_user_id,
        ),
    )


@router.get("/contexts", response_model=Envelope)
async def list_contexts(user_id: str = Depends(get_user)):
    runtime = get_runtime()
    contexts = runtime.store.list_contexts()
    items = [
        KnowledgeContextResponse(
            id=c.id,
            name=c.name,
            description=c.description,
            created_at=c.created_at,
            updated_at=c.updated_at,
            owner_user_id=c.owner_user_id,
        )
        for c in contexts
    ]
    return Envelope(status="ok", data=items)


@router.get("/contexts/{context_id}/chunks", response_model=Envelope)
async def list_chunks(context_id: str, user_id: str = Depends(get_user)):
    runtime = get_runtime()
    chunks = runtime.store.search_chunks(context_id, None)
    data = [
        KnowledgeChunkResponse(id=ch.id, context_id=ch.context_id, text=ch.text, seq=ch.seq)
        for ch in chunks
    ]
    return Envelope(status="ok", data=data)


@router.post("/voice/transcribe", response_model=Envelope)
async def transcribe_voice(file: UploadFile = File(...), user_id: str = Depends(get_user)):
    runtime = get_runtime()
    audio_bytes = await file.read()
    result = runtime.voice.transcribe(audio_bytes, user_id=user_id)
    return Envelope(status="ok", data=VoiceTranscriptionResponse(**result))


@router.post("/voice/synthesize", response_model=Envelope)
async def synthesize_voice(body: VoiceSynthesisRequest, user_id: str = Depends(get_user)):
    runtime = get_runtime()
    audio = runtime.voice.synthesize(body.text, user_id=user_id, voice=body.voice)
    return Envelope(status="ok", data=VoiceSynthesisResponse(**audio))


@router.websocket("/v1/chat/stream")
async def websocket_chat(ws: WebSocket):
    runtime = get_runtime()
    await ws.accept()
    try:
        init = await ws.receive_json()
        session_id = init.get("session_id")
        user_id = await runtime.auth.resolve_session(session_id)
        if not user_id:
            await ws.close(code=4401)
            return
        if runtime.cache:
            allowed = await runtime.cache.check_rate_limit(
                f"chat:{user_id}", runtime.settings.chat_rate_limit_per_minute, runtime.settings.chat_rate_limit_window_seconds
            )
            if not allowed:
                await ws.close(code=4429)
                return
        convo_id = init.get("conversation_id") or runtime.store.create_conversation(user_id=user_id).id
        runtime.store.append_message(convo_id, sender="user", role="user", content=init.get("message", ""))
        orchestration = runtime.workflow.run(
            init.get("workflow_id"), convo_id, init.get("message", ""), init.get("context_id"), user_id
        )
        assistant_msg = runtime.store.append_message(
            convo_id,
            sender="assistant",
            role="assistant",
            content=orchestration["content"],
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
