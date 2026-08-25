from __future__ import annotations

import json
import re
import unicodedata
from datetime import datetime
from typing import Any, List, Literal, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from liminallm.service.tokenizer_utils import (
    MAX_SINGLE_MESSAGE_TOKENS,
    estimate_token_count,
)

# Issue 46.1: Maximum nested JSON depth to prevent deserialization bombs
MAX_JSON_DEPTH = 20
# Issue 46.2: Maximum array items to prevent memory exhaustion
MAX_ARRAY_ITEMS = 1000


def _validate_json_depth(obj: Any, max_depth: int = MAX_JSON_DEPTH, current_depth: int = 0) -> None:
    """Validate nested JSON depth to prevent deserialization bombs (Issue 46.1).

    Args:
        obj: Object to validate
        max_depth: Maximum allowed nesting depth
        current_depth: Current depth (for recursion)

    Raises:
        ValueError: If depth exceeds maximum
    """
    if current_depth > max_depth:
        raise ValueError(f"JSON nesting depth exceeds maximum of {max_depth}")

    if isinstance(obj, dict):
        for value in obj.values():
            _validate_json_depth(value, max_depth, current_depth + 1)
    elif isinstance(obj, list):
        # Issue 46.2: Also check array length
        if len(obj) > MAX_ARRAY_ITEMS:
            raise ValueError(f"Array length {len(obj)} exceeds maximum of {MAX_ARRAY_ITEMS}")
        for item in obj:
            _validate_json_depth(item, max_depth, current_depth + 1)


def _validate_dict_field(value: Optional[dict], field_name: str = "field") -> Optional[dict]:
    """Validate a dict field for depth and size (Issue 46.1).

    Args:
        value: Dict to validate
        field_name: Name for error messages

    Returns:
        Validated dict

    Raises:
        ValueError: If validation fails
    """
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a dict")
    _validate_json_depth(value)
    return value


def _normalize_unicode(value: str) -> str:
    """Normalize Unicode string using NFKC (Issue 46.4).

    This handles:
    - Combining diacritics
    - Compatibility characters
    - Zero-width characters

    Args:
        value: String to normalize

    Returns:
        NFKC normalized string
    """
    # Remove zero-width characters that could be used for spoofing
    # U+200B ZERO WIDTH SPACE, U+200C ZERO WIDTH NON-JOINER,
    # U+200D ZERO WIDTH JOINER, U+FEFF ZERO WIDTH NO-BREAK SPACE
    zero_width = '\u200b\u200c\u200d\ufeff'
    cleaned = ''.join(c for c in value if c not in zero_width)

    # Remove RTL/LTR override characters that could be used for spoofing
    # U+202A-U+202E, U+2066-U+2069
    bidi_overrides = set(chr(c) for c in range(0x202A, 0x202F))
    bidi_overrides.update(chr(c) for c in range(0x2066, 0x206A))
    cleaned = ''.join(c for c in cleaned if c not in bidi_overrides)

    # Apply NFKC normalization
    return unicodedata.normalize('NFKC', cleaned)

_VALID_ERROR_CODES = frozenset({
    "unauthorized",
    "forbidden",
    "not_found",
    "rate_limited",
    "validation_error",
    "conflict",
    "server_error",
    # Codes raised by route handlers (concurrency caps, ingestion, 501s).
    "busy",
    "ingest_failed",
    "not_implemented",
})


class ErrorBody(BaseModel):
    """SPEC §18 error envelope body with stable code values."""

    code: str = Field(
        ...,
        description="Stable error code per SPEC §13.0",
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


_EMAIL_LOCAL_PART = re.compile(r"^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+$")
_EMAIL_DOMAIN_LABEL = re.compile(r"^[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?$")


def _validate_email(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("email must be a string")
    # Issue 46.4: Apply Unicode normalization before validation
    normalized = _normalize_unicode(value.strip().lower())
    if len(normalized) > 254:
        raise ValueError("email address too long")
    if len(normalized) < 3:
        raise ValueError("email address too short")
    local, sep, domain = normalized.partition("@")
    if not sep or not local or not domain:
        raise ValueError("invalid email address")
    if len(local) > 64:
        raise ValueError("email local part too long")
    if not _EMAIL_LOCAL_PART.match(local):
        raise ValueError("invalid email address format")
    domain_parts = domain.split(".")
    if len(domain_parts) < 2:
        raise ValueError("invalid email address format")
    for label in domain_parts:
        if len(label) > 63 or not _EMAIL_DOMAIN_LABEL.match(label):
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
    tenant_id: Optional[str] = Field(default=None, max_length=128)

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

    @model_validator(mode="after")
    def _reject_tenant_id(self):
        # Issue 53.2: tenant_id must be derived from server config per SPEC §12
        if getattr(self, "tenant_id", None):
            raise ValueError("tenant_id is managed server-side and cannot be provided")
        self.tenant_id = None
        return self


class AuthResponse(BaseModel):
    user_id: str
    session_id: str
    session_expires_at: datetime
    mfa_required: bool = False
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    token_type: Optional[str] = None
    csrf_token: Optional[str] = None
    role: str = "user"
    tenant_id: str = "public"


class LoginRequest(BaseModel):
    # No tenant_id: the tenant comes from the hostname the login arrived at.
    # This field used to be accepted and then silently discarded by the route,
    # which is worse than the raising validators below — it looked like it worked.
    email: str
    password: str
    mfa_code: Optional[str] = Field(default=None, max_length=10)
    device_type: str = Field(default="web", max_length=16)

    @field_validator("email")
    @classmethod
    def _validate_login_email(cls, value: str) -> str:
        return _validate_email(value)

    @field_validator("device_type")
    @classmethod
    def _normalize_device_type(cls, value: str) -> str:
        normalized = (value or "web").lower()
        if normalized not in {"web", "mobile"}:
            raise ValueError("device_type must be 'web' or 'mobile'")
        return normalized


class TokenRefreshRequest(BaseModel):
    #: Optional because a browser does not have one to send. §17.10 puts the
    #: refresh token in an `HttpOnly` cookie precisely so the page cannot read
    #: it, and a field the SPA must fill would undo that. API and mobile
    #: clients, which have no cookie jar, still send it here.
    refresh_token: Optional[str] = Field(default=None, max_length=2048)


class OAuthStartRequest(BaseModel):
    redirect_uri: Optional[str] = Field(default=None, max_length=2048)
    tenant_id: Optional[str] = Field(default=None, max_length=128)

    @model_validator(mode="after")
    def _reject_oauth_tenant(self):
        # Issue 53.3: tenant_id must be bound to server-created OAuth state
        if getattr(self, "tenant_id", None):
            raise ValueError("tenant_id is derived from OAuth state, not request input")
        self.tenant_id = None
        return self


class OAuthStartResponse(BaseModel):
    authorization_url: str
    state: str
    provider: str


class MFARequest(BaseModel):
    #: Optional for the same reason as `TokenRefreshRequest.refresh_token`:
    #: the browser's session id is an `HttpOnly` cookie it cannot read, and a
    #: field the SPA must fill would require keeping a readable copy.
    session_id: Optional[str] = Field(default=None, max_length=128)


class MFAVerifyRequest(BaseModel):
    session_id: Optional[str] = Field(default=None, max_length=128)
    code: str = Field(..., max_length=10)


class MFADisableRequest(BaseModel):
    code: str = Field(..., max_length=10, description="Current TOTP code to verify identity")


class MFAStatusResponse(BaseModel):
    enabled: bool = Field(..., description="Whether MFA is currently enabled")
    configured: bool = Field(..., description="Whether MFA secret is configured (pending verification)")


class ApiKeyCreateRequest(BaseModel):
    name: str = Field(default="", max_length=100, description="Label for the key")


class ApiKeySummary(BaseModel):
    id: str
    name: str
    #: Enough of the key to recognize it, never enough to use it.
    prefix: str
    created_at: datetime
    last_used_at: Optional[datetime] = None
    revoked_at: Optional[datetime] = None


class ApiKeyCreatedResponse(ApiKeySummary):
    #: The plaintext, present only in this one response.
    api_key: str


class ApiKeyListResponse(BaseModel):
    items: list[ApiKeySummary]


class ApiKeyRevokeResponse(BaseModel):
    revoked: bool


class UserSettingsRequest(BaseModel):
    """Request to update user settings."""
    locale: Optional[str] = Field(None, max_length=10, description="Locale code (e.g., en-US)")
    timezone: Optional[str] = Field(None, max_length=64, description="Timezone (e.g., America/New_York)")
    default_voice: Optional[str] = Field(None, max_length=64, description="Default voice ID for TTS")
    default_style: Optional[dict] = Field(None, description="Default style preferences")
    flags: Optional[dict] = Field(None, description="Feature flags")

    # Issue 46.1: Validate nested dict depth
    @field_validator("default_style", "flags")
    @classmethod
    def _validate_dict_depth(cls, value: Optional[dict]) -> Optional[dict]:
        return _validate_dict_field(value)


class UserSettingsResponse(BaseModel):
    """User settings response."""
    locale: Optional[str] = None
    timezone: Optional[str] = None
    default_voice: Optional[str] = None
    default_style: Optional[dict] = None
    flags: Optional[dict] = None


class PasswordResetRequest(BaseModel):
    email: str

    @field_validator("email")
    @classmethod
    def _validate_password_reset_email(cls, value: str) -> str:
        return _validate_email(value)


class PasswordResetConfirm(BaseModel):
    token: str = Field(..., max_length=256)
    new_password: str

    @field_validator("new_password")
    @classmethod
    def _validate_new_password(cls, value: str) -> str:
        if len(value) < 8:
            raise ValueError("password must be at least 8 characters")
        if len(value) > 128:
            raise ValueError("password must be at most 128 characters")
        return value


class PasswordChangeRequest(BaseModel):
    """Request to change password (requires current password)."""
    current_password: str = Field(..., min_length=1, max_length=128)
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
    token: str = Field(..., max_length=256)


class ChatMessage(BaseModel):
    content: str = Field(..., min_length=1, max_length=100000)  # 100KB max to prevent DoS
    mode: str = Field(default="text", pattern="^(text|voice|structured)$")
    content_struct: Optional[dict] = None

    # Issue 46.1: Validate nested dict depth
    @field_validator("content_struct")
    @classmethod
    def _validate_content_struct(cls, value: Optional[dict]) -> Optional[dict]:
        return _validate_dict_field(value, "content_struct")

    @field_validator("content")
    @classmethod
    def _validate_token_budget(cls, value: str) -> str:
        # A sanity cap on one message, not the model's budget: capping at
        # MAX_GENERATION_TOKENS rejected long pastes that a million-token
        # model handles easily. The real budget is enforced per model in the
        # workflow, which can prune; validation can only reject.
        if estimate_token_count(value) > MAX_SINGLE_MESSAGE_TOKENS:
            raise ValueError(
                f"message exceeds the single-message limit of "
                f"{MAX_SINGLE_MESSAGE_TOKENS} tokens"
            )
        return value


class ChatRequest(BaseModel):
    # Issue 46.3: Add max_length to string fields
    conversation_id: Optional[str] = Field(None, max_length=128)
    message: ChatMessage
    context_id: Optional[str] = Field(None, max_length=128)
    workflow_id: Optional[str] = Field(None, max_length=128)
    stream: bool = True
    client_request_id: Optional[str] = Field(None, max_length=128)


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

    # Issue 46.1: Validate nested dict depth
    @field_validator("schema_")
    @classmethod
    def _validate_schema_depth(cls, value: dict) -> dict:
        if value is None:
            return value
        _validate_json_depth(value)
        try:
            json.dumps(value)
        except TypeError as exc:
            raise ValueError("schema must be JSON-serializable") from exc
        return value


class ArtifactRequest(_SchemaPayload):
    # Issue 46.3: Add max_length to string fields
    type: str = Field(..., max_length=64)
    name: str = Field(..., max_length=256)
    description: Optional[str] = Field("", max_length=4096)
    #: Who this artifact is for. `private` — the default, and what every
    #: caller that omits the field keeps getting — is the owner's own. The
    #: other two publish, and the route admits them only for an admin: a
    #: globally visible `tool` or `mcp` artifact is a capability of every
    #: turn in the installation, not a preference of the account that wrote
    #: it. Constrained here rather than in the route so an unknown value is a
    #: 422 naming the field, not a 500 from the store's enum.
    visibility: Literal["private", "shared", "global"] = "private"


def _validated_ops(ops):
    """Check the patch's shape with the engine's own rule, not a copy of it.

    Imported here rather than at module scope: schemas is imported by the
    service layer, and a top-level import would close the ring.
    """
    from liminallm.service.json_patch import validate_ops

    return validate_ops(ops)


class ArtifactPatchRequest(BaseModel):
    """RFC 6902 JSON Patch request for artifact updates.

    Paths address the artifact's **schema document**, not a wrapper around it:
    a workflow's entrypoint is `/entrypoint`, not `/schema/entrypoint`. This
    example used to say `/schema/foo`, which named a key *inside* the schema —
    the engine created it, reported success, and left the value the caller
    meant to change alone. Patch traversal no longer invents missing parents,
    so that spelling is now refused rather than silently misapplied.

    Accepts:
    - RFC 6902 array of operations: [{"op": "replace", "path": "/foo", "value": "bar"}]
    - Wrapper dict with "ops" key: {"ops": [...]}
    - Legacy format for backward compatibility: {"schema": {...}, "description": "..."}
    """
    patch: Optional[Union[List[dict], dict]] = None
    # Legacy fields for backward compatibility
    schema_: Optional[dict] = Field(default=None, alias="schema")
    description: Optional[str] = None

    model_config = {"populate_by_name": True}

    @model_validator(mode="after")
    def _validate_request(self) -> "ArtifactPatchRequest":
        """Validate that either patch or legacy fields are provided."""
        has_patch = self.patch is not None
        has_legacy = self.schema_ is not None or self.description is not None
        if not has_patch and not has_legacy:
            raise ValueError("Must provide either 'patch' (RFC 6902) or 'schema'/'description' fields")
        return self

    def get_normalized_patch(self) -> dict:
        """Get patch in normalized format for processing.

        Operation shape is checked here, against the patch engine's own
        predicate, so a malformed op is refused before the route reaches the
        store. The engine checks again when it applies — it is the boundary
        every caller crosses — but by then this route has already decided to
        write a version, and a half-formed op that the engine skipped in
        silence used to get one.
        """
        if self.patch is not None:
            # RFC 6902 format
            if isinstance(self.patch, list):
                return {"ops": _validated_ops(self.patch)}
            elif isinstance(self.patch, dict):
                if "ops" in self.patch or "operations" in self.patch:
                    ops = self.patch.get("ops") or self.patch.get("operations")
                    return {"ops": _validated_ops(ops)}
                # Dict without ops key - treat as legacy schema update
                return {"schema_update": self.patch}
            return {}
        # Legacy format - convert to schema update
        result = {}
        if self.schema_ is not None:
            result["schema_update"] = self.schema_
        if self.description is not None:
            result["description"] = self.description
        return result


class ArtifactResponse(_SchemaPayload):
    id: str
    type: str
    kind: Optional[str]
    name: str
    description: str
    owner_user_id: Optional[str]
    visibility: str = "private"
    version: int = 1
    created_at: datetime
    updated_at: datetime


class ArtifactListResponse(BaseModel):
    items: List[ArtifactResponse]
    total_count: Optional[int] = Field(default=None, description="Total number of matching artifacts (when available)")
    has_next: Optional[bool] = Field(default=None, description="Whether more items exist beyond this page")
    next_page: Optional[int] = Field(default=None, description="Next page number, if available")
    next_cursor: Optional[str] = Field(
        default=None, description="Cursor token for keyset pagination"
    )
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
    # Field-for-field mirror of the storage model, so it is built with
    # model_validate(obj) rather than restated at each call site. Ten
    # hand-written mappings had already drifted: /me reported
    # `tenant_id or "global"` where the admin listing reported the
    # stored value. tests/test_response_mirrors.py keeps the two in step.
    model_config = ConfigDict(from_attributes=True)

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


# SPEC §2.6: explicit_signal allowed values ('like','dislike','always','never', etc.)
# Extended with practical UI values (thumbs_up/thumbs_down) and routing signals
_VALID_EXPLICIT_SIGNALS = frozenset({
    # SPEC §2.6 core values
    "like", "dislike", "always", "never",
    # UI-friendly aliases
    "thumbs_up", "thumbs_down",
    # Routing and system signals
    "routing_feedback", "correction", "edit",
})

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

        # SECURITY: Limit number of operations to prevent DoS via resource exhaustion
        MAX_PATCH_OPS = 100
        if len(ops) > MAX_PATCH_OPS:
            raise ValueError(f"patch contains too many operations (max {MAX_PATCH_OPS})")

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
    # Sharing state (SPEC §18): conversations are private by default.
    public: bool = False
    #: "chat" for the native surface, "responses" when an agent made it via
    #: the served Responses API — the UI badges the latter.
    source: str = "chat"


class ConversationUpdateRequest(BaseModel):
    """The fields a user may edit on their own conversation.

    Deliberately two. `meta` carries the public-share flag and the attachment
    records, and `active_context_id` names a knowledge context whose ownership
    is checked where contexts are chosen — so accepting either here would let
    a rename request publish a chat, rewrite its attachment list, or point it
    at a context the caller does not own. Sharing has its own endpoint that
    says what it does.
    """

    model_config = ConfigDict(extra="forbid")

    title: Optional[str] = Field(default=None, max_length=500)
    status: Optional[Literal["open", "archived"]] = None


class ConversationShareRequest(BaseModel):
    public: bool


class ConversationListResponse(BaseModel):
    items: List[ConversationSummary]
    total_count: Optional[int] = Field(default=None, description="Total number of conversations")
    has_next: Optional[bool] = Field(default=None, description="Whether more items exist")
    next_page: Optional[int] = Field(default=None, description="Next page number")


class CreateConversationRequest(BaseModel):
    title: Optional[str] = Field(default=None, max_length=255)
    context_id: Optional[str] = Field(default=None, max_length=255)


class CreateConversationResponse(BaseModel):
    id: str
    created_at: datetime
    updated_at: datetime
    title: Optional[str]
    status: str
    active_context_id: Optional[str]


class KnowledgeContextRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: str = Field(..., max_length=2000)
    text: Optional[str] = Field(default=None, max_length=10_000_000)  # 10MB max
    chunk_size: Optional[int] = Field(default=None, ge=64, le=4000)


class KnowledgeContextUpdateRequest(BaseModel):
    """The fields a user may edit on a context they own.

    Deliberately two. `meta` and `conversation_id` are how a row would claim
    to be a conversation's implicit index, which has a different lifetime and
    a different authorization rule; `fs_path` and `text` are ingestion, which
    is a separate mutation with its own path authority. Unknown fields are
    refused rather than dropped, so a caller is never told an edit succeeded
    when part of it did not happen.
    """

    model_config = ConfigDict(extra="forbid")

    name: Optional[str] = Field(default=None, min_length=1, max_length=255)
    description: Optional[str] = Field(default=None, max_length=2000)


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
    next_cursor: Optional[str] = Field(default=None, description="Cursor for the next page")
    next_page: Optional[int] = Field(default=None, description="Next page number when using page/page_size")
    page_size: Optional[int] = Field(default=None, description="Resolved page size")


class KnowledgeChunkResponse(BaseModel):
    id: int
    context_id: str
    fs_path: str
    content: str
    chunk_index: int


class KnowledgeChunkListResponse(BaseModel):
    items: List[KnowledgeChunkResponse]
    has_next: Optional[bool] = Field(default=None, description="Whether more chunks exist")
    next_cursor: Optional[str] = Field(default=None, description="Cursor for the next page")
    next_page: Optional[int] = Field(default=None, description="Next page number when using page/page_size")
    page_size: Optional[int] = Field(default=None, description="Resolved page size")


class ContextSourceRequest(BaseModel):
    """Request to add a source path to a knowledge context."""
    fs_path: str = Field(..., min_length=1, max_length=4096, description="Source path to index")
    recursive: bool = Field(default=True, description="Whether to recursively index subdirectories")


class ContextSourceResponse(BaseModel):
    # Field-for-field mirror of the storage model, so it is built with
    # model_validate(obj) rather than restated at each call site. Ten
    # hand-written mappings had already drifted: /me reported
    # `tenant_id or "global"` where the admin listing reported the
    # stored value. tests/test_response_mirrors.py keeps the two in step.
    model_config = ConfigDict(from_attributes=True)

    """Response containing context source details."""
    id: str
    context_id: str
    fs_path: str
    recursive: bool
    meta: Optional[dict] = None


class ContextSourceListResponse(BaseModel):
    """Response containing a list of context sources."""
    items: List[ContextSourceResponse]


class FileUploadResponse(BaseModel):
    fs_path: str
    context_id: Optional[str] = None
    chunk_count: Optional[int] = None
    # Set when the upload was attached to a conversation: how the model can
    # reach this file (inline / searchable / analyzable).
    attachment: Optional[dict[str, Any]] = None


class PreferenceEventRequest(BaseModel):
    conversation_id: str
    message_id: str
    feedback: str = Field(..., pattern="^(positive|negative|neutral|like|dislike)$")
    explicit_signal: Optional[str] = Field(default=None, max_length=64)
    score: Optional[float] = Field(default=None, ge=-1.0, le=1.0)  # SPEC §2.6 bounds
    context_text: Optional[str] = None
    corrected_text: Optional[str] = None
    weight: Optional[float] = Field(default=None, ge=0.0, le=1.0)  # SPEC §8.1 gate weight bounds
    routing_trace: Optional[List[dict]] = None
    adapter_gates: Optional[List[dict]] = None
    notes: Optional[str] = Field(default=None, max_length=2000)

    @field_validator("explicit_signal")
    @classmethod
    def _validate_explicit_signal(cls, value: Optional[str]) -> Optional[str]:
        """Validate explicit_signal per SPEC §2.6."""
        if value is None:
            return None
        if value not in _VALID_EXPLICIT_SIGNALS:
            raise ValueError(
                f"Invalid explicit_signal '{value}'. Must be one of: {', '.join(sorted(_VALID_EXPLICIT_SIGNALS))}"
            )
        return value


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


class ChatCancelRequest(BaseModel):
    """Request to cancel an in-progress chat request per SPEC §13.1."""
    request_id: str = Field(..., max_length=128, description="The request_id of the chat request to cancel")


class ChatCancelResponse(BaseModel):
    """Response from chat cancellation."""
    request_id: str
    cancelled: bool = Field(..., description="Whether the request was successfully cancelled")
    message: str = Field(..., description="Human-readable status message")


class VoiceTranscriptionResponse(BaseModel):
    transcript: str
    duration_ms: int
    user_id: Optional[str] = None
    model: Optional[str] = None
    language: Optional[str] = None


class VoiceSynthesisRequest(BaseModel):
    text: str = Field(..., max_length=5000)  # Reasonable limit for TTS
    voice: Optional[str] = None
    speed: float = Field(default=1.0, ge=0.25, le=4.0)


class VoiceSynthesisResponse(BaseModel):
    audio_path: str
    audio_url: Optional[str] = None
    format: str
    sample_rate: int
    duration_ms: int
    voice: str
    model: Optional[str] = None


class UserResponse(BaseModel):
    # Field-for-field mirror of the storage model, so it is built with
    # model_validate(obj) rather than restated at each call site. Ten
    # hand-written mappings had already drifted: /me reported
    # `tenant_id or "global"` where the admin listing reported the
    # stored value. tests/test_response_mirrors.py keeps the two in step.
    model_config = ConfigDict(from_attributes=True)

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


class NoteCreateRequest(BaseModel):
    """Create a note in the user's vault."""

    title: str = Field(..., min_length=1, max_length=200)
    content: str = Field("", max_length=200_000)


class NoteUpdateRequest(BaseModel):
    """Patch a note; omitted fields are left alone."""

    title: Optional[str] = Field(None, min_length=1, max_length=200)
    content: Optional[str] = Field(None, max_length=200_000)


class NoteSearchRequest(BaseModel):
    """Search the vault."""

    query: str = Field(..., min_length=1, max_length=2000)
    limit: int = Field(8, ge=1, le=25)


class NoteWitnessRequest(BaseModel):
    """Ask the witness to check a note against the rest of the vault."""

    limit: int = Field(6, ge=1, le=6)


class NoteFromFileRequest(BaseModel):
    """Promote an uploaded file into the vault as a note."""

    name: str = Field(..., min_length=1, max_length=512)
