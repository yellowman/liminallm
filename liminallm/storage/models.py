from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from ipaddress import IPv4Address, IPv6Address, ip_address
from typing import Dict, List, Optional


def _utcnow() -> datetime:
    """Timezone-aware UTC. Naive stamps do not compare with what Postgres returns."""
    return datetime.now(timezone.utc)



POSITIVE_FEEDBACK_VALUES = {"positive", "like"}


@dataclass
class User:
    id: str
    email: str
    handle: Optional[str] = None
    role: str = "user"
    tenant_id: str = "public"
    created_at: datetime = field(default_factory=_utcnow)
    is_active: bool = True
    plan_tier: str = "free"
    meta: Dict | None = None


@dataclass
class Session:
    id: str
    user_id: str
    created_at: datetime
    expires_at: datetime
    user_agent: Optional[str] = None
    ip_addr: IPv4Address | IPv6Address | None = None
    mfa_required: bool = False
    mfa_verified: bool = False
    tenant_id: str = "public"
    meta: Dict | None = None
    allow_expired: bool = field(default=False, repr=False, compare=False)
    enforce_future_expiry: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.expires_at <= self.created_at:
            raise ValueError("session expiration must be after creation time")
        if (
            self.enforce_future_expiry
            and not self.allow_expired
            and self.expires_at <= datetime.now(timezone.utc)
        ):
            raise ValueError("session expiration must be in the future")

    @classmethod
    def new(
        cls,
        user_id: str,
        ttl_minutes: int = 60 * 24,
        user_agent: str | None = None,
        ip_addr: IPv4Address | IPv6Address | str | None = None,
        *,
        mfa_required: bool = False,
        tenant_id: str = "public",
        meta: Dict | None = None,
    ) -> "Session":
        now = datetime.now(timezone.utc)
        parsed_ip = None
        if isinstance(ip_addr, str):
            try:
                parsed_ip = ip_address(ip_addr)
            except ValueError:
                # Invalid IP address (e.g., "testclient" in tests) - leave as None
                parsed_ip = None
        elif ip_addr is not None:
            parsed_ip = ip_addr
        return cls(
            id=str(uuid.uuid4()),
            user_id=user_id,
            created_at=now,
            expires_at=now + timedelta(minutes=ttl_minutes),
            user_agent=user_agent,
            ip_addr=parsed_ip,
            mfa_required=mfa_required,
            mfa_verified=not mfa_required,
            tenant_id=tenant_id,
            meta=meta,
            enforce_future_expiry=True,
        )


@dataclass
class UserSettings:
    """User preferences and settings (maps to user_settings table)."""
    user_id: str
    locale: Optional[str] = None
    timezone: Optional[str] = None
    default_voice: Optional[str] = None
    default_style: Optional[dict] = None
    flags: Optional[dict] = None


@dataclass
class Conversation:
    id: str
    user_id: str
    created_at: datetime
    updated_at: datetime
    title: Optional[str] = None
    status: str = "open"
    active_context_id: Optional[str] = None
    meta: Dict | None = None


@dataclass
class Message:
    id: str
    conversation_id: str
    sender: str
    role: str
    content: str
    seq: int
    created_at: datetime
    content_struct: Optional[dict] = None
    token_count_in: Optional[int] = None
    token_count_out: Optional[int] = None
    meta: Dict | None = None


@dataclass
class Artifact:
    id: str
    type: str
    name: str
    schema: dict
    description: str = ""
    owner_user_id: Optional[str] = None
    visibility: str = "private"
    created_at: datetime = field(default_factory=_utcnow)
    updated_at: datetime = field(default_factory=_utcnow)
    fs_path: Optional[str] = None
    base_model: Optional[str] = None
    meta: Dict | None = None


@dataclass
class ArtifactVersion:
    id: int
    artifact_id: str
    version: int
    schema: dict
    created_by: str = "system_llm"
    change_note: Optional[str] = None
    created_at: datetime = field(default_factory=_utcnow)
    fs_path: Optional[str] = None
    base_model: Optional[str] = None
    meta: Dict | None = None


@dataclass
class ConfigPatchAudit:
    id: int
    artifact_id: str
    proposer: str
    patch: dict
    justification: Optional[str]
    status: str = "pending"
    created_at: datetime = field(default_factory=_utcnow)
    decided_at: Optional[datetime] = None
    applied_at: Optional[datetime] = None
    meta: Dict | None = None


@dataclass
class KnowledgeContext:
    id: str
    owner_user_id: str
    name: str
    description: str
    created_at: datetime = field(default_factory=_utcnow)
    updated_at: datetime = field(default_factory=_utcnow)
    fs_path: Optional[str] = None
    meta: Dict | None = None
    #: Set for a conversation's implicit attachment index, and the authority
    #: on what that context is. `meta.auto` describes the same thing for the
    #: UI, but only this is a foreign key: it cascades on delete and it is
    #: what every exclusion filter keys on.
    conversation_id: Optional[str] = None


@dataclass
class ContextSource:
    id: str
    context_id: str
    fs_path: str
    recursive: bool = True
    meta: Dict | None = None


@dataclass
class KnowledgeChunk:
    context_id: str
    fs_path: str
    content: str
    embedding: List[float]
    chunk_index: int
    id: Optional[int] = None
    created_at: datetime = field(default_factory=_utcnow)
    meta: Dict | None = None


@dataclass
class PreferenceEvent:
    id: str
    user_id: str
    conversation_id: str
    message_id: str
    feedback: str
    score: Optional[float] = None
    explicit_signal: Optional[str] = None
    context_embedding: List[float] = field(default_factory=list)
    cluster_id: Optional[str] = None
    context_text: Optional[str] = None
    corrected_text: Optional[str] = None
    created_at: datetime = field(default_factory=_utcnow)
    weight: float = 1.0
    meta: Dict | None = None


@dataclass
class AdapterRouterState:
    artifact_id: str
    base_model: Optional[str] = None
    centroid_vec: Optional[List[float]] = None
    usage_count: int = 0
    success_score: float = 0.0
    last_used_at: Optional[datetime] = None
    last_trained_at: Optional[datetime] = None
    meta: Dict | None = None


@dataclass
class TrainingJob:
    id: str
    user_id: str
    adapter_id: str
    status: str = "queued"
    num_events: Optional[int] = None
    created_at: datetime = field(default_factory=_utcnow)
    updated_at: datetime = field(default_factory=_utcnow)
    loss: Optional[float] = None
    preference_event_ids: List[str] = field(default_factory=list)
    dataset_path: Optional[str] = None
    new_version: Optional[int] = None
    meta: Dict | None = None


@dataclass
class SemanticCluster:
    id: str
    user_id: Optional[str]
    centroid: List[float]
    size: int
    label: Optional[str] = None
    description: Optional[str] = None
    sample_message_ids: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=_utcnow)
    updated_at: datetime = field(default_factory=_utcnow)
    meta: Dict | None = None


@dataclass
class UserMFAConfig:
    user_id: str
    secret: str
    enabled: bool = False
    created_at: datetime = field(default_factory=_utcnow)
    meta: Dict | None = None


@dataclass
class ApiKey:
    """A long-lived bearer credential (user_api_key table).

    Deliberately hashless: the SHA-256 stays in the store, the plaintext is
    returned exactly once by the route that minted it.
    """

    id: str
    user_id: str
    name: str
    prefix: str
    created_at: datetime = field(default_factory=_utcnow)
    last_used_at: Optional[datetime] = None
    revoked_at: Optional[datetime] = None


@dataclass
class Note:
    id: str
    user_id: str
    title: str
    content: str = ""
    embedding: List[float] | None = None
    created_at: datetime = field(default_factory=_utcnow)
    updated_at: datetime = field(default_factory=_utcnow)
    meta: Dict | None = None
