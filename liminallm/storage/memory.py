from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from liminallm.storage.errors import ConstraintViolation
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

        if not self._load_state():
            self.default_artifacts()
            self._persist_state()

    def _state_path(self) -> Path:
        state_dir = self.fs_root / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        return state_dir / "memory_store.json"

    @staticmethod
    def _serialize_datetime(dt: datetime) -> str:
        return dt.isoformat()

    @staticmethod
    def _deserialize_datetime(raw: str) -> datetime:
        return datetime.fromisoformat(raw)

    def default_artifacts(self) -> None:
        chat_workflow_id = str(uuid.uuid4())
        default_schema = {
            "kind": "workflow.chat",
            "entrypoint": "classify",
            "nodes": [
                {
                    "id": "classify",
                    "type": "tool_call",
                    "tool": "llm.intent_classifier_v1",
                    "outputs": ["intent"],
                    "next": "route",
                },
                {
                    "id": "route",
                    "type": "switch",
                    "branches": [
                        {"when": "vars.intent == 'qa_with_docs'", "next": "rag"},
                        {"when": "vars.intent == 'code_edit'", "next": "code"},
                        {"when": "true", "next": "plain_chat"},
                    ],
                },
                {"id": "rag", "type": "tool_call", "tool": "rag.answer_with_context_v1", "next": "end"},
                {"id": "code", "type": "tool_call", "tool": "agent.code_v1", "next": "end"},
                {"id": "plain_chat", "type": "tool_call", "tool": "llm.generic", "next": "end"},
                {"id": "end", "type": "end"},
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
        self._persist_state()

    # user / auth
    def create_user(self, email: str, handle: Optional[str] = None) -> User:
        if any(existing.email == email for existing in self.users.values()):
            raise ConstraintViolation("email already exists", {"field": "email"})
        user_id = str(uuid.uuid4())
        user = User(id=user_id, email=email, handle=handle)
        self.users[user_id] = user
        self._persist_state()
        return user

    def get_user_by_email(self, email: str) -> Optional[User]:
        return next((u for u in self.users.values() if u.email == email), None)

    def save_password(self, user_id: str, password_hash: str, password_algo: str) -> None:
        if user_id not in self.users:
            raise ConstraintViolation("user not found for credentials", {"user_id": user_id})
        self.credentials[user_id] = (password_hash, password_algo)
        self._persist_state()

    def get_password_record(self, user_id: str) -> Optional[tuple[str, str]]:
        return self.credentials.get(user_id)

    def create_session(self, user_id: str, ttl_minutes: int = 60 * 24, user_agent: str | None = None, ip_addr: str | None = None) -> Session:
        if user_id not in self.users:
            raise ConstraintViolation("user does not exist", {"user_id": user_id})
        sess = Session.new(user_id=user_id, ttl_minutes=ttl_minutes, user_agent=user_agent, ip_addr=ip_addr)
        self.sessions[sess.id] = sess
        self._persist_state()
        return sess

    def get_session(self, session_id: str) -> Optional[Session]:
        return self.sessions.get(session_id)

    def revoke_session(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)
        self._persist_state()

    # chat
    def create_conversation(self, user_id: str, title: Optional[str] = None, active_context_id: Optional[str] = None) -> Conversation:
        if user_id not in self.users:
            raise ConstraintViolation("conversation owner missing", {"user_id": user_id})
        if active_context_id and active_context_id not in self.contexts:
            raise ConstraintViolation("active context missing", {"context_id": active_context_id})
        conv_id = str(uuid.uuid4())
        now = datetime.utcnow()
        conv = Conversation(id=conv_id, user_id=user_id, created_at=now, updated_at=now, title=title, active_context_id=active_context_id)
        self.conversations[conv_id] = conv
        self.messages[conv_id] = []
        self._persist_state()
        return conv

    def append_message(self, conversation_id: str, sender: str, role: str, content: str, meta: Optional[Dict] = None) -> Message:
        if conversation_id not in self.conversations:
            raise ConstraintViolation("conversation not found", {"conversation_id": conversation_id})
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
        self._persist_state()
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
        if owner_user_id and owner_user_id not in self.users:
            raise ConstraintViolation("artifact owner missing", {"owner_user_id": owner_user_id})
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
        self._persist_state()
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
        self._persist_state()
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
        self._persist_state()
        return audit

    def get_latest_workflow(self, workflow_id: str) -> Optional[dict]:
        versions = self.artifact_versions.get(workflow_id)
        if not versions:
            return None
        return versions[-1].schema

    def upsert_context(self, owner_user_id: Optional[str], name: str, description: str, fs_path: Optional[str] = None) -> KnowledgeContext:
        if owner_user_id and owner_user_id not in self.users:
            raise ConstraintViolation("context owner missing", {"owner_user_id": owner_user_id})
        ctx_id = str(uuid.uuid4())
        ctx = KnowledgeContext(id=ctx_id, owner_user_id=owner_user_id, name=name, description=description, fs_path=fs_path)
        self.contexts[ctx.id] = ctx
        self._persist_state()
        return ctx

    def list_contexts(self) -> List[KnowledgeContext]:
        return list(self.contexts.values())

    def add_chunks(self, context_id: str, chunks: Iterable[KnowledgeChunk]) -> None:
        if context_id not in self.contexts:
            raise ConstraintViolation("context not found", {"context_id": context_id})
        existing = self.chunks.setdefault(context_id, [])
        existing.extend(chunks)
        self._persist_state()

    def search_chunks(
        self, context_id: Optional[str], query_embedding: Optional[List[float]], limit: int = 4
    ) -> List[KnowledgeChunk]:
        def _cosine(a: List[float], b: List[float]) -> float:
            if not a or not b:
                return 0.0
            length = min(len(a), len(b))
            dot = sum(a[i] * b[i] for i in range(length))
            norm_a = sum(x * x for x in a) ** 0.5 or 1.0
            norm_b = sum(x * x for x in b) ** 0.5 or 1.0
            return dot / (norm_a * norm_b)

        if context_id:
            if context_id not in self.chunks:
                return []
            candidates = list(self.chunks[context_id])
        else:
            candidates: List[KnowledgeChunk] = []
            for vals in self.chunks.values():
                candidates.extend(vals)

        if query_embedding:
            ranked = sorted(candidates, key=lambda ch: _cosine(query_embedding, ch.embedding), reverse=True)
            return ranked[:limit]
        if context_id:
            return candidates[:limit]
        return candidates[:limit]

    def _persist_state(self) -> None:
        state = {
            "users": [self._serialize_user(u) for u in self.users.values()],
            "sessions": [self._serialize_session(s) for s in self.sessions.values()],
            "credentials": [
                {"user_id": user_id, "password_hash": creds[0], "password_algo": creds[1]}
                for user_id, creds in self.credentials.items()
            ],
            "conversations": [self._serialize_conversation(c) for c in self.conversations.values()],
            "messages": [self._serialize_message(m) for msgs in self.messages.values() for m in msgs],
            "artifacts": [self._serialize_artifact(a) for a in self.artifacts.values()],
            "artifact_versions": [
                self._serialize_artifact_version(v) for versions in self.artifact_versions.values() for v in versions
            ],
            "config_patches": [self._serialize_config_patch(cp) for cp in self.config_patches.values()],
            "contexts": [self._serialize_context(ctx) for ctx in self.contexts.values()],
            "chunks": [self._serialize_chunk(ch) for chs in self.chunks.values() for ch in chs],
        }
        path = self._state_path()
        path.write_text(json.dumps(state, indent=2))

    def _load_state(self) -> bool:
        path = self._state_path()
        if not path.exists():
            return False
        data = json.loads(path.read_text())
        self.users = {u["id"]: self._deserialize_user(u) for u in data.get("users", [])}
        self.sessions = {s["id"]: self._deserialize_session(s) for s in data.get("sessions", [])}
        self.credentials = {
            entry["user_id"]: (entry["password_hash"], entry.get("password_algo", ""))
            for entry in data.get("credentials", [])
        }
        self.conversations = {c["id"]: self._deserialize_conversation(c) for c in data.get("conversations", [])}
        self.messages = {}
        for msg_data in data.get("messages", []):
            msg = self._deserialize_message(msg_data)
            self.messages.setdefault(msg.conversation_id, []).append(msg)
        for convo_messages in self.messages.values():
            convo_messages.sort(key=lambda m: m.seq)
        self.artifacts = {a["id"]: self._deserialize_artifact(a) for a in data.get("artifacts", [])}
        self.artifact_versions = {}
        for version_data in data.get("artifact_versions", []):
            version = self._deserialize_artifact_version(version_data)
            versions = self.artifact_versions.setdefault(version.artifact_id, [])
            versions.append(version)
        for versions in self.artifact_versions.values():
            versions.sort(key=lambda v: v.version)
        self.config_patches = {cp["id"]: self._deserialize_config_patch(cp) for cp in data.get("config_patches", [])}
        self.contexts = {ctx["id"]: self._deserialize_context(ctx) for ctx in data.get("contexts", [])}
        self.chunks = {}
        for chunk_data in data.get("chunks", []):
            chunk = self._deserialize_chunk(chunk_data)
            self.chunks.setdefault(chunk.context_id, []).append(chunk)
        return True

    def _serialize_user(self, user: User) -> dict:
        return {
            "id": user.id,
            "email": user.email,
            "handle": user.handle,
            "created_at": self._serialize_datetime(user.created_at),
            "is_active": user.is_active,
            "plan_tier": user.plan_tier,
            "meta": user.meta,
        }

    def _deserialize_user(self, data: dict) -> User:
        return User(
            id=data["id"],
            email=data["email"],
            handle=data.get("handle"),
            created_at=self._deserialize_datetime(data["created_at"]),
            is_active=data.get("is_active", True),
            plan_tier=data.get("plan_tier", "free"),
            meta=data.get("meta"),
        )

    def _serialize_session(self, session: Session) -> dict:
        return {
            "id": session.id,
            "user_id": session.user_id,
            "created_at": self._serialize_datetime(session.created_at),
            "expires_at": self._serialize_datetime(session.expires_at),
            "user_agent": session.user_agent,
            "ip_addr": session.ip_addr,
            "meta": session.meta,
        }

    def _deserialize_session(self, data: dict) -> Session:
        return Session(
            id=data["id"],
            user_id=data["user_id"],
            created_at=self._deserialize_datetime(data["created_at"]),
            expires_at=self._deserialize_datetime(data["expires_at"]),
            user_agent=data.get("user_agent"),
            ip_addr=data.get("ip_addr"),
            meta=data.get("meta"),
        )

    def _serialize_conversation(self, conversation: Conversation) -> dict:
        return {
            "id": conversation.id,
            "user_id": conversation.user_id,
            "created_at": self._serialize_datetime(conversation.created_at),
            "updated_at": self._serialize_datetime(conversation.updated_at),
            "title": conversation.title,
            "status": conversation.status,
            "active_context_id": conversation.active_context_id,
            "meta": conversation.meta,
        }

    def _deserialize_conversation(self, data: dict) -> Conversation:
        return Conversation(
            id=data["id"],
            user_id=data["user_id"],
            created_at=self._deserialize_datetime(data["created_at"]),
            updated_at=self._deserialize_datetime(data["updated_at"]),
            title=data.get("title"),
            status=data.get("status", "open"),
            active_context_id=data.get("active_context_id"),
            meta=data.get("meta"),
        )

    def _serialize_message(self, message: Message) -> dict:
        return {
            "id": message.id,
            "conversation_id": message.conversation_id,
            "sender": message.sender,
            "role": message.role,
            "content": message.content,
            "seq": message.seq,
            "created_at": self._serialize_datetime(message.created_at),
            "token_count_in": message.token_count_in,
            "token_count_out": message.token_count_out,
            "meta": message.meta,
        }

    def _deserialize_message(self, data: dict) -> Message:
        return Message(
            id=data["id"],
            conversation_id=data["conversation_id"],
            sender=data["sender"],
            role=data["role"],
            content=data["content"],
            seq=data["seq"],
            created_at=self._deserialize_datetime(data["created_at"]),
            token_count_in=data.get("token_count_in"),
            token_count_out=data.get("token_count_out"),
            meta=data.get("meta"),
        )

    def _serialize_artifact(self, artifact: Artifact) -> dict:
        return {
            "id": artifact.id,
            "type": artifact.type,
            "name": artifact.name,
            "schema": artifact.schema,
            "description": artifact.description,
            "owner_user_id": artifact.owner_user_id,
            "visibility": artifact.visibility,
            "created_at": self._serialize_datetime(artifact.created_at),
            "updated_at": self._serialize_datetime(artifact.updated_at),
            "fs_path": artifact.fs_path,
            "meta": artifact.meta,
        }

    def _deserialize_artifact(self, data: dict) -> Artifact:
        return Artifact(
            id=data["id"],
            type=data["type"],
            name=data["name"],
            schema=data.get("schema", {}),
            description=data.get("description", ""),
            owner_user_id=data.get("owner_user_id"),
            visibility=data.get("visibility", "private"),
            created_at=self._deserialize_datetime(data["created_at"]),
            updated_at=self._deserialize_datetime(data["updated_at"]),
            fs_path=data.get("fs_path"),
            meta=data.get("meta"),
        )

    def _serialize_artifact_version(self, version: ArtifactVersion) -> dict:
        return {
            "id": version.id,
            "artifact_id": version.artifact_id,
            "version": version.version,
            "schema": version.schema,
            "created_at": self._serialize_datetime(version.created_at),
            "fs_path": version.fs_path,
            "meta": version.meta,
        }

    def _deserialize_artifact_version(self, data: dict) -> ArtifactVersion:
        return ArtifactVersion(
            id=data["id"],
            artifact_id=data["artifact_id"],
            version=data["version"],
            schema=data.get("schema", {}),
            created_at=self._deserialize_datetime(data["created_at"]),
            fs_path=data.get("fs_path"),
            meta=data.get("meta"),
        )

    def _serialize_config_patch(self, patch: ConfigPatchAudit) -> dict:
        return {
            "id": patch.id,
            "artifact_id": patch.artifact_id,
            "proposer_user_id": patch.proposer_user_id,
            "patch": patch.patch,
            "justification": patch.justification,
            "status": patch.status,
            "created_at": self._serialize_datetime(patch.created_at),
            "updated_at": self._serialize_datetime(patch.updated_at),
            "meta": patch.meta,
        }

    def _deserialize_config_patch(self, data: dict) -> ConfigPatchAudit:
        return ConfigPatchAudit(
            id=data["id"],
            artifact_id=data["artifact_id"],
            proposer_user_id=data.get("proposer_user_id"),
            patch=data.get("patch", {}),
            justification=data.get("justification"),
            status=data.get("status", "pending"),
            created_at=self._deserialize_datetime(data["created_at"]),
            updated_at=self._deserialize_datetime(data["updated_at"]),
            meta=data.get("meta"),
        )

    def _serialize_context(self, ctx: KnowledgeContext) -> dict:
        return {
            "id": ctx.id,
            "owner_user_id": ctx.owner_user_id,
            "name": ctx.name,
            "description": ctx.description,
            "created_at": self._serialize_datetime(ctx.created_at),
            "updated_at": self._serialize_datetime(ctx.updated_at),
            "fs_path": ctx.fs_path,
            "meta": ctx.meta,
        }

    def _deserialize_context(self, data: dict) -> KnowledgeContext:
        return KnowledgeContext(
            id=data["id"],
            owner_user_id=data.get("owner_user_id"),
            name=data["name"],
            description=data.get("description", ""),
            created_at=self._deserialize_datetime(data["created_at"]),
            updated_at=self._deserialize_datetime(data["updated_at"]),
            fs_path=data.get("fs_path"),
            meta=data.get("meta"),
        )

    def _serialize_chunk(self, chunk: KnowledgeChunk) -> dict:
        return {
            "id": chunk.id,
            "context_id": chunk.context_id,
            "text": chunk.text,
            "embedding": chunk.embedding,
            "seq": chunk.seq,
            "created_at": self._serialize_datetime(chunk.created_at),
            "meta": chunk.meta,
        }

    def _deserialize_chunk(self, data: dict) -> KnowledgeChunk:
        return KnowledgeChunk(
            id=data["id"],
            context_id=data["context_id"],
            text=data.get("text", ""),
            embedding=data.get("embedding", []),
            seq=data.get("seq", 0),
            created_at=self._deserialize_datetime(data["created_at"]),
            meta=data.get("meta"),
        )
