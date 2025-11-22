from __future__ import annotations

from datetime import datetime
from typing import Any, List, Optional

from pydantic import BaseModel, Field


class Envelope(BaseModel):
    status: str = Field(..., pattern="^(ok|error)$")
    data: Optional[Any] = None
    error: Optional[dict] = None


class SignupRequest(BaseModel):
    email: str
    password: str
    handle: Optional[str] = None


class AuthResponse(BaseModel):
    user_id: str
    session_id: str
    session_expires_at: datetime


class LoginRequest(BaseModel):
    email: str
    password: str


class MFARequest(BaseModel):
    session_id: str


class MFAVerifyRequest(BaseModel):
    session_id: str
    code: str


class PasswordResetRequest(BaseModel):
    email: str


class PasswordResetConfirm(BaseModel):
    token: str
    new_password: str


class ChatMessage(BaseModel):
    content: str
    mode: str = "text"


class ChatRequest(BaseModel):
    conversation_id: Optional[str] = None
    message: ChatMessage
    context_id: Optional[str] = None
    workflow_id: Optional[str] = None
    stream: bool = True
    client_request_id: Optional[str] = None


class ChatResponse(BaseModel):
    message_id: str
    conversation_id: str
    content: str
    workflow_id: Optional[str]
    adapters: List[str] = []
    usage: dict = {}
    context_snippets: List[str] = []


class ArtifactRequest(BaseModel):
    type: str
    name: str
    description: Optional[str] = ""
    schema: dict


class ArtifactResponse(BaseModel):
    id: str
    type: str
    name: str
    description: str
    schema: dict
    owner_user_id: Optional[str]
    created_at: datetime
    updated_at: datetime


class ArtifactListResponse(BaseModel):
    items: List[ArtifactResponse]


class ConfigPatchAuditResponse(BaseModel):
    id: str
    artifact_id: str
    justification: Optional[str]
    status: str
    patch: dict


class ConfigPatchRequest(BaseModel):
    artifact_id: str
    patch: dict
    justification: str


class ConversationMessagesResponse(BaseModel):
    conversation_id: str
    messages: List[dict]


class ConversationSummary(BaseModel):
    id: str
    created_at: datetime
    updated_at: datetime
    title: Optional[str]
    status: str
    active_context_id: Optional[str]


class ConversationListResponse(BaseModel):
    items: List[ConversationSummary]


class KnowledgeContextRequest(BaseModel):
    name: str
    description: str
    text: Optional[str] = None


class KnowledgeContextResponse(BaseModel):
    id: str
    name: str
    description: str
    created_at: datetime
    updated_at: datetime
    owner_user_id: Optional[str]


class KnowledgeChunkResponse(BaseModel):
    id: str
    context_id: str
    text: str
    seq: int


class FileUploadResponse(BaseModel):
    fs_path: str
    context_id: Optional[str] = None
    chunk_count: Optional[int] = None
