from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile, WebSocket, WebSocketDisconnect

from liminallm.api.schemas import (
    ArtifactListResponse,
    ArtifactRequest,
    ArtifactResponse,
    AuthResponse,
    ChatRequest,
    ChatResponse,
    ConversationMessagesResponse,
    ConversationListResponse,
    ConversationSummary,
    ConfigPatchAuditResponse,
    ConfigPatchRequest,
    Envelope,
    LoginRequest,
    MFARequest,
    MFAVerifyRequest,
    PasswordResetConfirm,
    PasswordResetRequest,
    SignupRequest,
    KnowledgeChunkResponse,
    KnowledgeContextRequest,
    KnowledgeContextResponse,
    FileUploadResponse,
)
from liminallm.storage.errors import ConstraintViolation
from liminallm.config import get_settings
from liminallm.service.fs import PathTraversalError, safe_join
from liminallm.service.runtime import get_runtime

router = APIRouter(prefix="/v1")


# Dependency placeholder for auth
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
    return Envelope(status="ok", data=AuthResponse(user_id=user.id, session_id=session.id, session_expires_at=session.expires_at))


@router.post("/auth/login", response_model=Envelope)
async def login(body: LoginRequest):
    runtime = get_runtime()
    user, session = await runtime.auth.login(email=body.email, password=body.password)
    if not user or not session:
        raise HTTPException(status_code=401, detail="invalid credentials")
    return Envelope(status="ok", data=AuthResponse(user_id=user.id, session_id=session.id, session_expires_at=session.expires_at))


@router.post("/auth/mfa/request", response_model=Envelope)
async def request_mfa(body: MFARequest):
    runtime = get_runtime()
    user_id = await runtime.auth.resolve_session(body.session_id)
    if not user_id:
        raise HTTPException(status_code=401, detail="invalid session")
    challenge = runtime.auth.issue_mfa_challenge(user_id=user_id)
    return Envelope(status="ok", data={"challenge": challenge})


@router.post("/auth/mfa/verify", response_model=Envelope)
async def verify_mfa(body: MFAVerifyRequest):
    runtime = get_runtime()
    user_id = await runtime.auth.resolve_session(body.session_id)
    if not user_id:
        raise HTTPException(status_code=401, detail="invalid session")
    ok = runtime.auth.verify_mfa_challenge(user_id=user_id, code=body.code, expected=body.code)
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
    runtime.store.append_message(conversation_id, sender="user", role="user", content=body.message.content)
    orchestration = runtime.workflow.run(body.workflow_id, conversation_id, body.message.content, body.context_id)
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


@router.get("/artifacts", response_model=Envelope)
async def list_artifacts(type: Optional[str] = None, user_id: str = Depends(get_user)):
    runtime = get_runtime()
    items = [
        ArtifactResponse(
            id=a.id,
            type=a.type,
            name=a.name,
            description=a.description,
            schema=a.schema,
            owner_user_id=a.owner_user_id,
            created_at=a.created_at,
            updated_at=a.updated_at,
        )
        for a in runtime.store.list_artifacts(type_filter=type)
    ]
    return Envelope(status="ok", data=ArtifactListResponse(items=items))


@router.post("/artifacts", response_model=Envelope)
async def create_artifact(body: ArtifactRequest, user_id: str = Depends(get_user)):
    runtime = get_runtime()
    try:
        artifact = runtime.store.create_artifact(type_=body.type, name=body.name, description=body.description or "", schema=body.schema)
    except ConstraintViolation as err:
        raise HTTPException(status_code=409, detail=err.message)
    resp = ArtifactResponse(
        id=artifact.id,
        type=artifact.type,
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
    artifact = runtime.store.update_artifact(artifact_id, schema=body.schema, description=body.description)
    if not artifact:
        raise HTTPException(status_code=404, detail="artifact not found")
    resp = ArtifactResponse(
        id=artifact.id,
        type=artifact.type,
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
            raise HTTPException(status_code=404, detail=err.message)
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
        orchestration = runtime.workflow.run(init.get("workflow_id"), convo_id, init.get("message", ""), init.get("context_id"))
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
