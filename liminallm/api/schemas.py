from __future__ import annotations

from datetime import datetime
from typing import Any, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Envelope(BaseModel):
    status: str = Field(..., pattern="^(ok|error)$")
    data: Optional[Any] = None
    error: Optional[dict] = None
    request_id: str = Field(default_factory=lambda: str(uuid4()))


def _validate_email(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("email must be a string")
    normalized = value.strip()
    local, sep, domain = normalized.partition("@")
    if not sep or not local or not domain or "." not in domain:
        raise ValueError("invalid email address")
    return normalized


class SignupRequest(BaseModel):
    email: str
    password: str
    handle: Optional[str] = None
    tenant_id: Optional[str] = None

    @field_validator("email")
    @classmethod
    def _validate_signup_email(cls, value: str) -> str:
        return _validate_email(value)

    @field_validator("password")
    @classmethod
    def _validate_password(cls, value: str) -> str:
        if len(value) < 8:
            raise ValueError("password must be at least 8 characters")
        if not any(c.islower() for c in value) or not any(c.isupper() for c in value):
            raise ValueError("password must include upper and lower case letters")
        if not any(c.isdigit() for c in value):
            raise ValueError("password must include a digit")
        return value


class AuthResponse(BaseModel):
    user_id: str
    session_id: str
    session_expires_at: datetime
    mfa_required: bool = False
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    token_type: Optional[str] = None
    role: str = "user"
    tenant_id: str = "public"


class LoginRequest(BaseModel):
    email: str
    password: str
    mfa_code: Optional[str] = None
    tenant_id: Optional[str] = None

    @field_validator("email")
    @classmethod
    def _validate_login_email(cls, value: str) -> str:
        return _validate_email(value)


class TokenRefreshRequest(BaseModel):
    refresh_token: str
    tenant_id: Optional[str] = None


class OAuthStartRequest(BaseModel):
    redirect_uri: Optional[str] = None
    tenant_id: Optional[str] = None


class OAuthStartResponse(BaseModel):
    authorization_url: str
    state: str
    provider: str


class MFARequest(BaseModel):
    session_id: str


class MFAVerifyRequest(BaseModel):
    session_id: str
    code: str


class PasswordResetRequest(BaseModel):
    email: str

    @field_validator("email")
    @classmethod
    def _validate_password_reset_email(cls, value: str) -> str:
        return _validate_email(value)


class PasswordResetConfirm(BaseModel):
    token: str
    new_password: str

    @field_validator("new_password")
    @classmethod
    def _validate_new_password(cls, value: str) -> str:
        if len(value) < 8:
            raise ValueError("password must be at least 8 characters")
        if not any(c.islower() for c in value) or not any(c.isupper() for c in value):
            raise ValueError("password must include upper and lower case letters")
        if not any(c.isdigit() for c in value):
            raise ValueError("password must include a digit")
        return value


class EmailVerificationRequest(BaseModel):
    token: str


class ChatMessage(BaseModel):
    content: str = Field(..., max_length=100000)  # 100KB max to prevent DoS
    mode: str = "text"
    content_struct: Optional[dict] = None


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
    content_struct: Optional[dict] = None
    workflow_id: Optional[str] = None
    adapters: List[str] = Field(default_factory=list)
    adapter_gates: List[dict] = Field(default_factory=list)
    usage: dict = Field(default_factory=dict)
    context_snippets: List[str] = Field(default_factory=list)
    routing_trace: List[dict] = Field(default_factory=list)
    workflow_trace: List[dict] = Field(default_factory=list)


class _SchemaPayload(BaseModel):
    """Common payload with a JSON schema, avoiding BaseModel.schema clash."""

    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    schema_: dict = Field(alias="schema")

    @property
    def schema(self) -> dict:  # pragma: no cover - compatibility shim
        return self.schema_


class ArtifactRequest(_SchemaPayload):
    type: Optional[str] = None
    name: str
    description: Optional[str] = ""


class ArtifactResponse(_SchemaPayload):
    id: str
    type: str
    kind: Optional[str]
    name: str
    description: str
    owner_user_id: Optional[str]
    created_at: datetime
    updated_at: datetime


class ArtifactListResponse(BaseModel):
    items: List[ArtifactResponse]


class ArtifactVersionResponse(_SchemaPayload):
    id: int
    artifact_id: str
    version: int
    created_by: str
    change_note: Optional[str] = None
    created_at: datetime
    fs_path: Optional[str] = None
    meta: Optional[dict] = None


class ArtifactVersionListResponse(BaseModel):
    items: List[ArtifactVersionResponse]


class ConfigPatchAuditResponse(BaseModel):
    id: int
    artifact_id: str
    proposer: str
    justification: Optional[str]
    status: str
    patch: dict
    created_at: datetime
    decided_at: Optional[datetime] = None
    applied_at: Optional[datetime] = None
    meta: Optional[dict] = None


class ConfigPatchRequest(BaseModel):
    artifact_id: str = Field(..., max_length=255)
    patch: dict
    justification: str = Field(..., max_length=2000)


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
    name: str = Field(..., max_length=255)
    description: str = Field(..., max_length=2000)
    text: Optional[str] = Field(default=None, max_length=10_000_000)  # 10MB max
    chunk_size: Optional[int] = Field(default=None, ge=64, le=4000)


class KnowledgeContextResponse(BaseModel):
    id: str
    name: str
    description: str
    created_at: datetime
    updated_at: datetime
    owner_user_id: str
    meta: Optional[dict] = None


class KnowledgeContextListResponse(BaseModel):
    items: List[KnowledgeContextResponse]


class KnowledgeChunkResponse(BaseModel):
    id: int
    context_id: str
    fs_path: str
    content: str
    chunk_index: int


class KnowledgeChunkListResponse(BaseModel):
    items: List[KnowledgeChunkResponse]


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
    status: Optional[str] = None
    error: Optional[str] = None
    totals: dict
    clusters: List[dict] = Field(default_factory=list)
    clusters_status: Optional[str] = None
    clusters_error: Optional[str] = None
    adapters: List[dict] = Field(default_factory=list)
    routing_feedback: dict = Field(default_factory=dict)
    events: List[dict] = Field(default_factory=list)
    adapter_router_state: dict = Field(default_factory=dict)


class ConfigPatchDecisionRequest(BaseModel):
    decision: Literal["approve", "reject"]
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
    text: str = Field(..., max_length=5000)  # Reasonable limit for TTS
    voice: Optional[str] = None


class VoiceSynthesisResponse(BaseModel):
    audio_path: str
    format: str
    sample_rate: int
    duration_ms: int
    voice: str


class UserResponse(BaseModel):
    id: str
    email: str
    handle: Optional[str] = None
    role: str
    tenant_id: str
    created_at: datetime
    is_active: bool = True
    plan_tier: str = "free"
    meta: Optional[dict] = None


class UserListResponse(BaseModel):
    items: List[UserResponse]


class AdminCreateUserResponse(UserResponse):
    password: str


class AdminCreateUserRequest(BaseModel):
    email: str
    password: Optional[str] = None
    handle: Optional[str] = None
    role: Optional[str] = None
    tenant_id: Optional[str] = None
    plan_tier: Optional[str] = None
    is_active: Optional[bool] = None
    meta: Optional[dict] = None

    @field_validator("email")
    @classmethod
    def _validate_admin_email(cls, value: str) -> str:
        return _validate_email(value)


class UpdateUserRoleRequest(BaseModel):
    role: Literal["admin", "user"]


class AdminInspectionResponse(BaseModel):
    summary: dict
    details: dict


class ToolInvokeRequest(BaseModel):
    inputs: dict = Field(default_factory=dict)
    conversation_id: Optional[str] = None
    context_id: Optional[str] = None
    user_message: Optional[str] = None


class ToolInvokeResponse(BaseModel):
    status: str = "ok"
    outputs: dict = Field(default_factory=dict)
    content: Optional[str] = None
    usage: Optional[dict] = None
    context_snippets: Optional[List[str]] = None


class ToolSpecListResponse(BaseModel):
    items: List[ArtifactResponse]
    next_page: Optional[int] = None
    page_size: int = 50


class WorkflowListResponse(BaseModel):
    items: List[ArtifactResponse]
    next_page: Optional[int] = None
    page_size: int = 50
