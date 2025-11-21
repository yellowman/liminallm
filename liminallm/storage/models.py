from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional


@dataclass
class User:
    id: str
    email: str
    handle: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
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
    ip_addr: Optional[str] = None
    meta: Dict | None = None

    @classmethod
    def new(cls, user_id: str, ttl_minutes: int = 60 * 24, user_agent: str | None = None, ip_addr: str | None = None) -> "Session":
        now = datetime.utcnow()
        return cls(
            id=str(uuid.uuid4()),
            user_id=user_id,
            created_at=now,
            expires_at=now + timedelta(minutes=ttl_minutes),
            user_agent=user_agent,
            ip_addr=ip_addr,
        )


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
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    fs_path: Optional[str] = None
    meta: Dict | None = None


@dataclass
class ArtifactVersion:
    id: int
    artifact_id: str
    version: int
    schema: dict
    created_at: datetime = field(default_factory=datetime.utcnow)
    fs_path: Optional[str] = None
    meta: Dict | None = None


@dataclass
class ConfigPatchAudit:
    id: str
    artifact_id: str
    proposer_user_id: Optional[str]
    patch: dict
    justification: Optional[str]
    status: str = "pending"
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    meta: Dict | None = None


@dataclass
class KnowledgeContext:
    id: str
    owner_user_id: Optional[str]
    name: str
    description: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    fs_path: Optional[str] = None
    meta: Dict | None = None


@dataclass
class KnowledgeChunk:
    id: str
    context_id: str
    text: str
    embedding: List[float]
    seq: int
    created_at: datetime = field(default_factory=datetime.utcnow)
    meta: Dict | None = None
