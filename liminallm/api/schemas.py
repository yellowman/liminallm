from __future__ import annotations

from datetime import datetime
from typing import Any, List, Optional

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
    mfa_required: bool = False


class LoginRequest(BaseModel):
    email: str
    password: str
    mfa_code: Optional[str] = None


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
    workflow_id: Optional[str] = None
    adapters: List[str] = Field(default_factory=list)
    adapter_gates: List[dict] = Field(default_factory=list)
    usage: dict = Field(default_factory=dict)
    context_snippets: List[str] = Field(default_factory=list)
    routing_trace: List[dict] = Field(default_factory=list)
    workflow_trace: List[dict] = Field(default_factory=list)


class ArtifactRequest(BaseModel):
    type: Optional[str] = None
    kind: Optional[str] = None
    name: str
    description: Optional[str] = ""
    schema: dict


class ArtifactResponse(BaseModel):
    id: str
    type: str
    kind: Optional[str]
    name: str
    description: str
    schema: dict
    owner_user_id: Optional[str]
    created_at: datetime
    updated_at: datetime


class ArtifactListResponse(BaseModel):
    items: List[ArtifactResponse]


class ArtifactVersionResponse(BaseModel):
    id: int
    artifact_id: str
    version: int
    schema: dict
    created_by: str
    change_note: Optional[str] = None
    created_at: datetime
    fs_path: Optional[str] = None
    meta: Optional[dict] = None


class ArtifactVersionListResponse(BaseModel):
    items: List[ArtifactVersionResponse]


class ConfigPatchAuditResponse(BaseModel):
    id: str
    artifact_id: str
    justification: Optional[str]
    status: str
    patch: dict
    decided_at: Optional[datetime] = None
    applied_at: Optional[datetime] = None
    meta: Optional[dict] = None


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


class PreferenceEventRequest(BaseModel):
    conversation_id: str
    message_id: str
    feedback: str = Field(..., pattern="^(positive|negative|neutral|like|dislike)$")
    explicit_signal: Optional[str] = None
    score: Optional[float] = None
    context_text: Optional[str] = None
    corrected_text: Optional[str] = None
    weight: Optional[float] = None
    routing_trace: Optional[List[dict]] = None
    adapter_gates: Optional[List[dict]] = None
    notes: Optional[str] = None


class PreferenceEventResponse(BaseModel):
    id: str
    cluster_id: Optional[str] = None
    feedback: str
    created_at: datetime


class PreferenceInsightsResponse(BaseModel):
    totals: dict
    clusters: List[dict] = Field(default_factory=list)
    adapters: List[dict] = Field(default_factory=list)
    routing_feedback: dict = Field(default_factory=dict)
    events: List[dict] = Field(default_factory=list)


class ConfigPatchDecisionRequest(BaseModel):
    decision: str
    reason: Optional[str] = None


class ConfigPatchListResponse(BaseModel):
    items: List[ConfigPatchAuditResponse]


class AutoPatchRequest(BaseModel):
    artifact_id: str
    goal: Optional[str] = None


class VoiceTranscriptionResponse(BaseModel):
    transcript: str
    duration_ms: int
    user_id: Optional[str] = None


class VoiceSynthesisRequest(BaseModel):
    text: str
    voice: Optional[str] = None


class VoiceSynthesisResponse(BaseModel):
    audio_path: str
    format: str
    sample_rate: int
    duration_ms: int
    voice: str
