from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from liminallm.storage.models import (
    Artifact,
    ArtifactVersion,
    ConfigPatchAudit,
    Conversation,
    KnowledgeChunk,
    KnowledgeContext,
    Message,
    Session,
    User,
)


class MemoryStore:
    """Minimal in-memory backing store for the initial prototype."""

    def __init__(self, fs_root: str = "/tmp/liminallm") -> None:
        self.users: Dict[str, User] = {}
        self.sessions: Dict[str, Session] = {}
        self.conversations: Dict[str, Conversation] = {}
        self.messages: Dict[str, List[Message]] = {}
        self.credentials: Dict[str, tuple[str, str]] = {}
        self.artifacts: Dict[str, Artifact] = {}
        self.artifact_versions: Dict[str, List[ArtifactVersion]] = {}
        self.config_patches: Dict[str, ConfigPatchAudit] = {}
        self.contexts: Dict[str, KnowledgeContext] = {}
        self.chunks: Dict[str, List[KnowledgeChunk]] = {}
        self.fs_root = Path(fs_root)
        self.fs_root.mkdir(parents=True, exist_ok=True)
        self.default_artifacts()

    def default_artifacts(self) -> None:
        chat_workflow_id = str(uuid.uuid4())
        default_schema = {
            "kind": "workflow.chat",
            "entrypoint": "classify",
            "nodes": [
                {"id": "classify", "tool": "llm.generic", "outputs": ["intent"]},
                {"id": "plain_chat", "tool": "llm.generic"},
            ],
        }
        payload_path = self.persist_artifact_payload(chat_workflow_id, default_schema)
        self.artifacts[chat_workflow_id] = Artifact(
            id=chat_workflow_id,
            type="workflow",
            name="default_chat_workflow",
            description="LLM-only chat workflow defined as data.",
            schema=default_schema,
            fs_path=payload_path,
        )
        self.artifact_versions[chat_workflow_id] = [
            ArtifactVersion(
                id=1,
                artifact_id=chat_workflow_id,
                version=1,
                schema=self.artifacts[chat_workflow_id].schema,
                fs_path=payload_path,
            )
        ]

    # user / auth
    def create_user(self, email: str, handle: Optional[str] = None) -> User:
        user_id = str(uuid.uuid4())
        user = User(id=user_id, email=email, handle=handle)
        self.users[user_id] = user
        return user

    def get_user_by_email(self, email: str) -> Optional[User]:
        return next((u for u in self.users.values() if u.email == email), None)

    def save_password(self, user_id: str, password_hash: str, password_algo: str) -> None:
        self.credentials[user_id] = (password_hash, password_algo)

    def get_password_record(self, user_id: str) -> Optional[tuple[str, str]]:
        return self.credentials.get(user_id)

    def create_session(self, user_id: str, ttl_minutes: int = 60 * 24, user_agent: str | None = None, ip_addr: str | None = None) -> Session:
        sess = Session.new(user_id=user_id, ttl_minutes=ttl_minutes, user_agent=user_agent, ip_addr=ip_addr)
        self.sessions[sess.id] = sess
        return sess

    def get_session(self, session_id: str) -> Optional[Session]:
        return self.sessions.get(session_id)

    def revoke_session(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)

    # chat
    def create_conversation(self, user_id: str, title: Optional[str] = None) -> Conversation:
        conv_id = str(uuid.uuid4())
        now = datetime.utcnow()
        conv = Conversation(id=conv_id, user_id=user_id, created_at=now, updated_at=now, title=title)
        self.conversations[conv_id] = conv
        self.messages[conv_id] = []
        return conv

    def append_message(self, conversation_id: str, sender: str, role: str, content: str, meta: Optional[Dict] = None) -> Message:
        seq = len(self.messages.get(conversation_id, []))
        msg = Message(
            id=str(uuid.uuid4()),
            conversation_id=conversation_id,
            sender=sender,
            role=role,
            content=content,
            seq=seq,
            created_at=datetime.utcnow(),
            meta=meta,
        )
        self.messages.setdefault(conversation_id, []).append(msg)
        if conversation_id in self.conversations:
            self.conversations[conversation_id].updated_at = msg.created_at
        return msg

    def list_messages(self, conversation_id: str, limit: int = 10) -> List[Message]:
        msgs = self.messages.get(conversation_id, [])
        return msgs[-limit:]

    def list_conversations(self, user_id: str, limit: int = 20) -> List[Conversation]:
        convs = [c for c in self.conversations.values() if c.user_id == user_id]
        convs.sort(key=lambda c: c.updated_at, reverse=True)
        return convs[:limit]

    # artifacts
    def list_artifacts(self, type_filter: Optional[str] = None) -> List[Artifact]:
        artifacts = list(self.artifacts.values())
        if type_filter:
            artifacts = [a for a in artifacts if a.type == type_filter]
        return artifacts

    def get_artifact(self, artifact_id: str) -> Optional[Artifact]:
        return self.artifacts.get(artifact_id)

    def create_artifact(self, type_: str, name: str, schema: dict, description: str = "", owner_user_id: Optional[str] = None) -> Artifact:
        artifact_id = str(uuid.uuid4())
        fs_path = self.persist_artifact_payload(artifact_id, schema)
        artifact = Artifact(
            id=artifact_id,
            type=type_,
            name=name,
            schema=schema,
            description=description,
            owner_user_id=owner_user_id,
            fs_path=fs_path,
        )
        self.artifacts[artifact_id] = artifact
        self.artifact_versions.setdefault(artifact_id, []).append(
            ArtifactVersion(
                id=len(self.artifact_versions.get(artifact_id, [])) + 1,
                artifact_id=artifact_id,
                version=1,
                schema=schema,
                fs_path=fs_path,
            )
        )
        return artifact

    def update_artifact(self, artifact_id: str, schema: dict, description: Optional[str] = None) -> Optional[Artifact]:
        artifact = self.artifacts.get(artifact_id)
        if not artifact:
            return None
        fs_path = self.persist_artifact_payload(artifact_id, schema)
        artifact.schema = schema
        if description is not None:
            artifact.description = description
        artifact.updated_at = datetime.utcnow()
        artifact.fs_path = fs_path
        versions = self.artifact_versions.setdefault(artifact_id, [])
        versions.append(
            ArtifactVersion(
                id=len(versions) + 1,
                artifact_id=artifact_id,
                version=(versions[-1].version + 1 if versions else 1),
                schema=schema,
                fs_path=fs_path,
            )
        )
        return artifact

    def persist_artifact_payload(self, artifact_id: str, schema: dict) -> str:
        artifact_dir = self.fs_root / "artifacts" / artifact_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        version = len(self.artifact_versions.get(artifact_id, [])) + 1
        payload_path = artifact_dir / f"v{version}.json"
        payload_path.write_text(json.dumps(schema, indent=2))
        return str(payload_path)

    def record_config_patch(self, artifact_id: str, proposer_user_id: Optional[str], patch: dict, justification: Optional[str]) -> ConfigPatchAudit:
        audit = ConfigPatchAudit(
            id=str(uuid.uuid4()),
            artifact_id=artifact_id,
            proposer_user_id=proposer_user_id,
            patch=patch,
            justification=justification,
        )
        self.config_patches[audit.id] = audit
        return audit

    def get_latest_workflow(self, workflow_id: str) -> Optional[dict]:
        versions = self.artifact_versions.get(workflow_id)
        if not versions:
            return None
        return versions[-1].schema

    def upsert_context(self, owner_user_id: Optional[str], name: str, description: str, fs_path: Optional[str] = None) -> KnowledgeContext:
        ctx_id = str(uuid.uuid4())
        ctx = KnowledgeContext(id=ctx_id, owner_user_id=owner_user_id, name=name, description=description, fs_path=fs_path)
        self.contexts[ctx.id] = ctx
        return ctx

    def list_contexts(self) -> List[KnowledgeContext]:
        return list(self.contexts.values())

    def add_chunks(self, context_id: str, chunks: Iterable[KnowledgeChunk]) -> None:
        existing = self.chunks.setdefault(context_id, [])
        existing.extend(chunks)

    def search_chunks(self, context_id: Optional[str], limit: int = 4) -> List[KnowledgeChunk]:
        if context_id and context_id in self.chunks:
            return self.chunks[context_id][:limit]
        # flatten all contexts for global search fallback
        all_chunks: List[KnowledgeChunk] = []
        for vals in self.chunks.values():
            all_chunks.extend(vals)
        return all_chunks[:limit]
