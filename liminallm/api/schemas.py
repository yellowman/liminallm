from __future__ import annotations

import re
from datetime import datetime
from typing import Any, List, Literal, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


_VALID_ERROR_CODES = frozenset({
    "unauthorized",
    "forbidden",
    "not_found",
    "rate_limited",
    "validation_error",
    "conflict",
    "server_error",
})


class ErrorBody(BaseModel):
    """SPEC §18 error envelope body with stable code values."""

    code: str = Field(
        ...,
        description="Stable error code per SPEC §18",
    )
    message: str
    details: Optional[Any] = None  # object, array, or null

    @field_validator("code")
    @classmethod
    def _validate_error_code(cls, value: str) -> str:
        if value not in _VALID_ERROR_CODES:
            raise ValueError(
                f"Invalid error code '{value}'. Must be one of: {', '.join(sorted(_VALID_ERROR_CODES))}"
            )
        return value


class Envelope(BaseModel):
    """SPEC §18 API envelope format."""

    status: str = Field(..., pattern="^(ok|error)$")
    data: Optional[Any] = None
    error: Optional[ErrorBody] = None
    request_id: str = Field(default_factory=lambda: str(uuid4()))


_EMAIL_REGEX = re.compile(
    r"^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?"
    r"(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)+$"
)


def _validate_email(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("email must be a string")
    normalized = value.strip().lower()
    if len(normalized) > 254:
        raise ValueError("email address too long")
    if len(normalized) < 3:
        raise ValueError("email address too short")
    local, sep, domain = normalized.partition("@")
    if not sep or not local or not domain:
        raise ValueError("invalid email address")
    if len(local) > 64:
        raise ValueError("email local part too long")
    if not _EMAIL_REGEX.match(normalized):
        raise ValueError("invalid email address format")
    return normalized


_HANDLE_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")


def _validate_handle(value: Optional[str]) -> Optional[str]:
    """Validate handle format: alphanumeric with underscores/hyphens, max 64 chars."""
    if value is None:
        return None
    if len(value) > 64:
        raise ValueError("handle must be at most 64 characters")
    if len(value) < 1:
        raise ValueError("handle must be at least 1 character")
    if not _HANDLE_PATTERN.match(value):
        raise ValueError("handle must contain only alphanumeric characters, underscores, and hyphens")
    return value


def _validate_password_strength(value: str) -> str:
    """Validate password meets minimum requirements."""
    if len(value) < 8:
        raise ValueError("password must be at least 8 characters")
    if len(value) > 128:
        raise ValueError("password must be at most 128 characters")
    return value


class SignupRequest(BaseModel):
    email: str
    password: str
    handle: Optional[str] = Field(default=None, max_length=64)
    tenant_id: Optional[str] = None

    @field_validator("email")
    @classmethod
    def _validate_signup_email(cls, value: str) -> str:
        return _validate_email(value)

    @field_validator("password")
    @classmethod
    def _validate_password(cls, value: str) -> str:
        return _validate_password_strength(value)

    @field_validator("handle")
    @classmethod
    def _validate_handle(cls, value: Optional[str]) -> Optional[str]:
        return _validate_handle(value)


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
        if len(value) > 128:
            raise ValueError("password must be at most 128 characters")
        return value


class EmailVerificationRequest(BaseModel):
    token: str


class ChatMessage(BaseModel):
    content: str = Field(..., min_length=1, max_length=100000)  # 100KB max to prevent DoS
    mode: str = Field(default="text", pattern="^(text|voice|structured)$")
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
    total_count: Optional[int] = Field(default=None, description="Total number of matching artifacts (when available)")
    has_next: Optional[bool] = Field(default=None, description="Whether more items exist beyond this page")
    next_page: Optional[int] = Field(default=None, description="Next page number, if available")
    page_size: int = Field(default=100, description="Number of items per page")


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


_JSON_PATCH_OPS = frozenset({"add", "remove", "replace", "move", "copy", "test"})


class ConfigPatchRequest(BaseModel):
    artifact_id: str = Field(..., max_length=255)
    patch: Union[dict, List[dict]]  # RFC 6902: array of ops, or wrapper dict with "ops"/"operations"
    justification: str = Field(..., min_length=1, max_length=2000)

    @field_validator("patch")
    @classmethod
    def _validate_patch_format(cls, value: Union[dict, List[dict]]) -> dict:
        """Validate patch is a valid JSON Patch format (RFC 6902).

        Accepts:
        - RFC 6902 array of operations: [{"op": "add", "path": "/foo", "value": "bar"}]
        - Wrapper dict with "ops" or "operations" key: {"ops": [...]}
        - Single operation dict: {"op": "add", "path": "/foo", "value": "bar"}

        Returns normalized dict with "ops" key for consistent internal storage.
        """
        # Handle RFC 6902 array format directly
        if isinstance(value, list):
            ops = value
        else:
            # Dict format - check for ops/operations wrapper or single operation
            ops = value.get("ops") or value.get("operations")
            if ops is None:
                if "op" in value and "path" in value:
                    # Single operation dict
                    ops = [value]
                else:
                    raise ValueError(
                        "patch must be an RFC 6902 array, contain 'ops'/'operations' array, "
                        "or be a valid JSON Patch operation with 'op' and 'path' fields"
                    )

        if not isinstance(ops, list):
            raise ValueError("patch operations must be an array")

        for i, op in enumerate(ops):
            if not isinstance(op, dict):
                raise ValueError(f"patch operation {i} must be an object")
            if "op" not in op:
                raise ValueError(f"patch operation {i} missing 'op' field")
            if op["op"] not in _JSON_PATCH_OPS:
                raise ValueError(f"patch operation {i} has invalid 'op': {op['op']}")
            if "path" not in op:
                raise ValueError(f"patch operation {i} missing 'path' field")

        # Normalize to dict with "ops" key for consistent internal storage
        return {"ops": ops}


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
    total_count: Optional[int] = Field(default=None, description="Total number of conversations")
    has_next: Optional[bool] = Field(default=None, description="Whether more items exist")
    next_page: Optional[int] = Field(default=None, description="Next page number")


class KnowledgeContextRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
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
    total_count: Optional[int] = Field(default=None, description="Total number of contexts")
    has_next: Optional[bool] = Field(default=None, description="Whether more items exist")


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
    score: Optional[float] = Field(default=None, ge=-1.0, le=1.0)  # SPEC §2.6 bounds
    context_text: Optional[str] = None
    corrected_text: Optional[str] = None
    weight: Optional[float] = Field(default=None, ge=0.0, le=1.0)  # SPEC §8.1 gate weight bounds
    routing_trace: Optional[List[dict]] = None
    adapter_gates: Optional[List[dict]] = None
    notes: Optional[str] = Field(default=None, max_length=2000)


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
    password: Optional[str] = Field(default=None, description="If not provided, a random password will be generated")
    handle: Optional[str] = Field(default=None, max_length=64)
    role: Optional[Literal["admin", "user"]] = Field(default=None)
    tenant_id: Optional[str] = None
    plan_tier: Optional[Literal["free", "pro", "enterprise"]] = Field(default=None)
    is_active: Optional[bool] = None
    meta: Optional[dict] = None

    @field_validator("email")
    @classmethod
    def _validate_admin_email(cls, value: str) -> str:
        return _validate_email(value)

    @field_validator("password")
    @classmethod
    def _validate_admin_password(cls, value: Optional[str]) -> Optional[str]:
        if value is not None:
            return _validate_password_strength(value)
        return value

    @field_validator("handle")
    @classmethod
    def _validate_admin_handle(cls, value: Optional[str]) -> Optional[str]:
        return _validate_handle(value)


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
