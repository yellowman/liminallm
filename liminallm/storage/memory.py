from __future__ import annotations

import base64
import json
import os
import secrets
import threading
import uuid
import hashlib
import shutil
from datetime import datetime
from ipaddress import ip_address
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from cryptography.fernet import Fernet, InvalidToken
from liminallm.content_struct import normalize_content_struct
from liminallm.logging import get_logger
from liminallm.service.artifact_validation import (
    ArtifactValidationError,
    validate_artifact,
)
from liminallm.service.bm25 import (
    tokenize_text as _tokenize_text,
    compute_bm25_scores as _compute_bm25_scores,
)
from liminallm.service.embeddings import cosine_similarity
from liminallm.storage.errors import ConstraintViolation
from liminallm.storage.models import (
    Artifact,
    ArtifactVersion,
    AdapterRouterState,
    ConfigPatchAudit,
    ContextSource,
    Conversation,
    KnowledgeChunk,
    KnowledgeContext,
    Message,
    PreferenceEvent,
    SemanticCluster,
    Session,
    TrainingJob,
    User,
    UserAuthProvider,
    UserMFAConfig,
)


class MemoryStore:
    """Minimal in-memory backing store for the initial prototype."""

    def __init__(
        self, fs_root: str = "/tmp/liminallm", *, mfa_encryption_key: str | None = None
    ) -> None:
        self.logger = get_logger(__name__)
        self.users: Dict[str, User] = {}
        self.sessions: Dict[str, Session] = {}
        self.conversations: Dict[str, Conversation] = {}
        self.messages: Dict[str, List[Message]] = {}
        self.credentials: Dict[str, tuple[str, str]] = {}
        self.providers: List[UserAuthProvider] = []
        self.artifacts: Dict[str, Artifact] = {}
        self.artifact_versions: Dict[str, List[ArtifactVersion]] = {}
        self.config_patches: Dict[int, ConfigPatchAudit] = {}
        self.runtime_config: Dict[str, str] = {}
        self.contexts: Dict[str, KnowledgeContext] = {}
        self.context_sources: Dict[str, List[ContextSource]] = {}
        self.chunks: Dict[str, List[KnowledgeChunk]] = {}
        self._chunk_id_seq: int = 1
        self._artifact_version_seq: int = 1
        # Thread lock for sequence counters to prevent race conditions
        self._seq_lock = threading.Lock()
        # RLock for all data operations to ensure thread safety
        # Using RLock to allow nested acquisitions within the same thread
        self._data_lock = threading.RLock()
        self.preference_events: Dict[str, PreferenceEvent] = {}
        self.training_jobs: Dict[str, TrainingJob] = {}
        self.semantic_clusters: Dict[str, SemanticCluster] = {}
        self.adapter_router_state: Dict[str, AdapterRouterState] = {}
        self.mfa_secrets: Dict[str, UserMFAConfig] = {}
        self.fs_root = Path(fs_root)
        self.fs_root.mkdir(parents=True, exist_ok=True)
        self._mfa_cipher = self._build_mfa_cipher(mfa_encryption_key)

        if not self._load_state():
            self.default_artifacts()
            self._persist_state()
        elif self._backfill_default_classifier_inputs():
            self._persist_state()

    def _state_path(self) -> Path:
        state_dir = self.fs_root / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        return state_dir / "memory_store.json"

    @staticmethod
    def _derive_cipher_key(key_material: str) -> bytes:
        return base64.urlsafe_b64encode(hashlib.sha256(key_material.encode()).digest())

    def _build_mfa_cipher(self, key_material: str | None) -> Fernet:
        material = (
            key_material or os.getenv("MFA_SECRET_KEY") or os.getenv("JWT_SECRET")
        )
        if not material:
            shared_fs = Path(os.getenv("SHARED_FS_ROOT", "/srv/liminallm"))
            secret_path = shared_fs / ".jwt_secret"
            fallback_path = self.fs_root / ".jwt_secret"
            for candidate in (secret_path, fallback_path):
                try:
                    if candidate.exists():
                        material = candidate.read_text().strip()
                        if material:
                            break
                except Exception:
                    continue
            if not material:
                generated = secrets.token_urlsafe(64)
                try:
                    secret_path.parent.mkdir(parents=True, exist_ok=True)
                    secret_path.write_text(generated)
                    os.chmod(secret_path, 0o600)
                    material = generated
                except Exception as exc:
                    raise RuntimeError("Unable to persist MFA encryption key") from exc
        try:
            return Fernet(self._derive_cipher_key(material))
        except Exception as exc:
            raise RuntimeError("Unable to initialize MFA cipher") from exc

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
                    "inputs": {"message": "${input.message}"},
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
                {
                    "id": "rag",
                    "type": "tool_call",
                    "tool": "rag.answer_with_context_v1",
                    "inputs": {"message": "${input.message}"},
                    "next": "end",
                },
                {
                    "id": "code",
                    "type": "tool_call",
                    "tool": "agent.code_v1",
                    "inputs": {"message": "${input.message}"},
                    "next": "end",
                },
                {
                    "id": "plain_chat",
                    "type": "tool_call",
                    "tool": "llm.generic",
                    "inputs": {"message": "${input.message}"},
                    "next": "end",
                },
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
                id=self._next_artifact_version_id(),
                artifact_id=chat_workflow_id,
                version=1,
                schema=self.artifacts[chat_workflow_id].schema,
                fs_path=payload_path,
            )
        ]
        self._persist_state()

    def _next_artifact_version_id(self) -> int:
        with self._seq_lock:
            next_id = self._artifact_version_seq
            self._artifact_version_seq += 1
            return next_id

    def _next_chunk_id(self) -> int:
        with self._seq_lock:
            next_id = self._chunk_id_seq
            self._chunk_id_seq += 1
            return next_id

    def _backfill_default_classifier_inputs(self) -> bool:
        """Ensure default workflow classifier passes the user message to the tool."""

        def _ensure_inputs(schema: dict) -> bool:
            if not isinstance(schema, dict):
                return False
            nodes = schema.get("nodes", [])
            changed = False
            for node in nodes:
                if not isinstance(node, dict):
                    continue
                if (
                    node.get("id") != "classify"
                    or node.get("tool") != "llm.intent_classifier_v1"
                ):
                    continue
                inputs = node.get("inputs") or {}
                if "message" not in inputs:
                    new_inputs = dict(inputs)
                    new_inputs["message"] = "${input.message}"
                    node["inputs"] = new_inputs
                    changed = True
            return changed

        changed = False
        for artifact in self.artifacts.values():
            if artifact.name != "default_chat_workflow":
                continue
            if _ensure_inputs(artifact.schema):
                artifact.updated_at = datetime.utcnow()
                changed = True
            for version in self.artifact_versions.get(artifact.id, []):
                if _ensure_inputs(version.schema):
                    version.created_at = datetime.utcnow()
                    changed = True
        return changed

    # user / auth
    def create_user(
        self,
        email: str,
        handle: Optional[str] = None,
        *,
        tenant_id: str = "public",
        role: str = "user",
        plan_tier: str = "free",
        is_active: bool = True,
        meta: Optional[Dict] = None,
    ) -> User:
        with self._data_lock:
            if any(existing.email == email for existing in self.users.values()):
                raise ConstraintViolation("email already exists", {"field": "email"})
            user_id = str(uuid.uuid4())
            normalized_meta = meta.copy() if meta else {}
            normalized_meta.setdefault("email_verified", False)
            user = User(
                id=user_id,
                email=email,
                handle=handle,
                tenant_id=tenant_id,
                role=role,
                plan_tier=plan_tier,
                is_active=is_active,
                meta=normalized_meta,
            )
            self.users[user_id] = user
            self._persist_state()
            return user

    def link_user_auth_provider(
        self, user_id: str, provider: str, provider_uid: str
    ) -> None:
        with self._data_lock:
            # Check for existing mapping (avoid duplicates like postgres ON CONFLICT DO NOTHING)
            for existing in self.providers:
                if existing.provider == provider and existing.provider_uid == provider_uid:
                    return  # Already linked, do nothing
            # Generate unique ID
            max_id = max((p.id for p in self.providers), default=0)
            mapping = UserAuthProvider(
                id=max_id + 1, user_id=user_id, provider=provider, provider_uid=provider_uid
            )
            self.providers.append(mapping)
            self._persist_state()

    def get_user_by_provider(self, provider: str, provider_uid: str) -> Optional[User]:
        with self._data_lock:
            for mapping in self.providers:
                if mapping.provider == provider and mapping.provider_uid == provider_uid:
                    return self.users.get(mapping.user_id)
            return None

    def get_user_by_email(self, email: str) -> Optional[User]:
        with self._data_lock:
            return next((u for u in self.users.values() if u.email == email), None)

    def get_user(self, user_id: str) -> Optional[User]:
        with self._data_lock:
            return self.users.get(user_id)

    def list_users(
        self, tenant_id: Optional[str] = None, limit: int = 100
    ) -> List[User]:
        with self._data_lock:
            results = [
                u for u in self.users.values() if not tenant_id or u.tenant_id == tenant_id
            ]
            return sorted(results, key=lambda u: u.created_at, reverse=True)[:limit]

    def update_user_role(self, user_id: str, role: str) -> Optional[User]:
        with self._data_lock:
            user = self.users.get(user_id)
            if not user:
                return None
            user.role = role
            self._persist_state()
            return user

    def mark_email_verified(self, user_id: str) -> Optional[User]:
        with self._data_lock:
            user = self.users.get(user_id)
            if not user:
                return None
            meta = user.meta or {}
            meta["email_verified"] = True
            user.meta = meta
            self._persist_state()
            return user

    def delete_user(self, user_id: str) -> bool:
        with self._data_lock:
            if user_id not in self.users:
                return False
            self.users.pop(user_id, None)
            self.mfa_secrets.pop(user_id, None)
            # Clean up user credentials
            self.credentials.pop(user_id, None)
            # Clean up auth providers for this user
            self.providers = [p for p in self.providers if p.user_id != user_id]
            for sess_id, sess in list(self.sessions.items()):
                if sess.user_id == user_id:
                    self.sessions.pop(sess_id, None)
            for conv_id, conv in list(self.conversations.items()):
                if conv.user_id == user_id:
                    self.messages.pop(conv_id, None)
                    self.conversations.pop(conv_id, None)
            for ctx_id, ctx in list(self.contexts.items()):
                if ctx.owner_user_id == user_id:
                    self.contexts.pop(ctx_id, None)
                    self.context_sources.pop(ctx_id, None)
                    self.chunks.pop(ctx_id, None)
            user_artifacts: list[str] = []
            for art_id, art in list(self.artifacts.items()):
                if art.owner_user_id == user_id:
                    self.artifacts.pop(art_id, None)
                    self.artifact_versions.pop(art_id, None)
                    user_artifacts.append(art_id)
            for evt_id, evt in list(self.preference_events.items()):
                if evt.user_id == user_id:
                    self.preference_events.pop(evt_id, None)
            for job_id, job in list(self.training_jobs.items()):
                if job.user_id == user_id:
                    self.training_jobs.pop(job_id, None)
            for cluster_id, cluster in list(self.semantic_clusters.items()):
                if cluster.user_id == user_id:
                    self.semantic_clusters.pop(cluster_id, None)
            for state_id, state in list(self.adapter_router_state.items()):
                if (
                    getattr(state, "artifact_id", None) in user_artifacts
                    or getattr(state, "user_id", None) == user_id
                ):
                    self.adapter_router_state.pop(state_id, None)
            for art_id in user_artifacts:
                shutil.rmtree(self.fs_root / "artifacts" / art_id, ignore_errors=True)
            shutil.rmtree(self.fs_root / "users" / user_id, ignore_errors=True)
            self._persist_state()
            return True

    def save_password(
        self, user_id: str, password_hash: str, password_algo: str
    ) -> None:
        with self._data_lock:
            if user_id not in self.users:
                raise ConstraintViolation(
                    "user not found for credentials", {"user_id": user_id}
                )
            self.credentials[user_id] = (password_hash, password_algo)
            self._persist_state()

    def get_password_record(self, user_id: str) -> Optional[tuple[str, str]]:
        with self._data_lock:
            return self.credentials.get(user_id)

    def _encrypt_mfa_secret(self, secret: str) -> str:
        if not secret:
            return secret
        if not self._mfa_cipher:
            raise RuntimeError("MFA cipher unavailable; secret cannot be stored")
        return self._mfa_cipher.encrypt(secret.encode()).decode()

    def _decrypt_mfa_secret(self, secret: str) -> str:
        if not secret:
            return secret
        if not self._mfa_cipher:
            raise RuntimeError("MFA cipher unavailable; cannot decrypt secret")
        try:
            return self._mfa_cipher.decrypt(secret.encode()).decode()
        except InvalidToken:
            self.logger.warning("mfa_secret_decrypt_failed")
            return secret

    def set_user_mfa_secret(
        self, user_id: str, secret: str, enabled: bool = False
    ) -> UserMFAConfig:
        with self._data_lock:
            if user_id not in self.users:
                raise ConstraintViolation("user not found for mfa", {"user_id": user_id})
            encrypted_secret = self._encrypt_mfa_secret(secret)
            record = UserMFAConfig(
                user_id=user_id, secret=encrypted_secret, enabled=enabled
            )
            self.mfa_secrets[user_id] = record
            self._persist_state()
            return UserMFAConfig(
                user_id=user_id,
                secret=secret,
                enabled=enabled,
                created_at=record.created_at,
                meta=record.meta,
            )

    def get_user_mfa_secret(self, user_id: str) -> Optional[UserMFAConfig]:
        with self._data_lock:
            cfg = self.mfa_secrets.get(user_id)
            if not cfg:
                return None
            decrypted = self._decrypt_mfa_secret(cfg.secret)
            return UserMFAConfig(
                user_id=cfg.user_id,
                secret=decrypted,
                enabled=cfg.enabled,
                created_at=cfg.created_at,
                meta=cfg.meta,
            )

    def create_session(
        self,
        user_id: str,
        ttl_minutes: int = 60 * 24,
        user_agent: str | None = None,
        ip_addr: str | None = None,
        *,
        mfa_required: bool = False,
        tenant_id: str = "public",
        meta: Optional[Dict] = None,
    ) -> Session:
        with self._data_lock:
            if user_id not in self.users:
                raise ConstraintViolation("user does not exist", {"user_id": user_id})
            sess = Session.new(
                user_id=user_id,
                ttl_minutes=ttl_minutes,
                user_agent=user_agent,
                ip_addr=ip_addr,
                mfa_required=mfa_required,
                tenant_id=tenant_id,
                meta=meta,
            )
            self.sessions[sess.id] = sess
            self._persist_state()
            return sess

    def get_session(self, session_id: str) -> Optional[Session]:
        with self._data_lock:
            return self.sessions.get(session_id)

    def set_session_meta(self, session_id: str, meta: Dict) -> None:
        with self._data_lock:
            sess = self.sessions.get(session_id)
            if not sess:
                return
            sess.meta = meta
            self.sessions[session_id] = sess
            self._persist_state()

    def revoke_session(self, session_id: str) -> None:
        with self._data_lock:
            self.sessions.pop(session_id, None)
            self._persist_state()

    def revoke_user_sessions(self, user_id: str) -> None:
        with self._data_lock:
            stale = [sid for sid, sess in self.sessions.items() if sess.user_id == user_id]
            for sid in stale:
                self.sessions.pop(sid, None)
            if stale:
                self._persist_state()

    def mark_session_verified(self, session_id: str) -> None:
        with self._data_lock:
            sess = self.sessions.get(session_id)
            if not sess:
                return
            sess.mfa_verified = True
            self._persist_state()

    # chat
    def create_conversation(
        self,
        user_id: str,
        title: Optional[str] = None,
        active_context_id: Optional[str] = None,
    ) -> Conversation:
        if user_id not in self.users:
            raise ConstraintViolation(
                "conversation owner missing", {"user_id": user_id}
            )
        if active_context_id and active_context_id not in self.contexts:
            raise ConstraintViolation(
                "active context missing", {"context_id": active_context_id}
            )
        conv_id = str(uuid.uuid4())
        now = datetime.utcnow()
        conv = Conversation(
            id=conv_id,
            user_id=user_id,
            created_at=now,
            updated_at=now,
            title=title,
            active_context_id=active_context_id,
        )
        self.conversations[conv_id] = conv
        self.messages[conv_id] = []
        self._persist_state()
        return conv

    def get_conversation(
        self, conversation_id: str, *, user_id: Optional[str] = None
    ) -> Optional[Conversation]:
        conv = self.conversations.get(conversation_id)
        if not conv:
            return None
        if user_id and conv.user_id != user_id:
            return None
        return conv

    def append_message(
        self,
        conversation_id: str,
        sender: str,
        role: str,
        content: str,
        meta: Optional[Dict] = None,
        content_struct: Optional[dict] = None,
    ) -> Message:
        with self._data_lock:
            if conversation_id not in self.conversations:
                raise ConstraintViolation(
                    "conversation not found", {"conversation_id": conversation_id}
                )
            normalized_content_struct = normalize_content_struct(content_struct, content)
            # Thread-safe sequence number generation
            seq = len(self.messages.get(conversation_id, []))
            msg = Message(
                id=str(uuid.uuid4()),
                conversation_id=conversation_id,
                sender=sender,
                role=role,
                content=content,
                content_struct=normalized_content_struct,
                seq=seq,
                created_at=datetime.utcnow(),
                meta=meta,
            )
            self.messages.setdefault(conversation_id, []).append(msg)
            if conversation_id in self.conversations:
                self.conversations[conversation_id].updated_at = msg.created_at
            self._persist_state()
            return msg

    # preference events
    def _text_embedding(self, text: Optional[str]) -> List[float]:
        if not text:
            return []
        tokens = text.lower().split()
        dim = 64
        vec = [0.0] * dim
        for tok in tokens:
            h = int(hashlib.sha256(tok.encode()).hexdigest(), 16)
            vec[h % dim] += 1.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    def record_preference_event(
        self,
        user_id: str,
        conversation_id: str,
        message_id: str,
        feedback: str,
        *,
        score: Optional[float] = None,
        explicit_signal: Optional[str] = None,
        corrected_text: Optional[str] = None,
        weight: Optional[float] = None,
        context_embedding: Optional[List[float]] = None,
        cluster_id: Optional[str] = None,
        context_text: Optional[str] = None,
        meta: Optional[Dict] = None,
    ) -> PreferenceEvent:
        with self._data_lock:
            if user_id not in self.users:
                raise ConstraintViolation("preference user missing", {"user_id": user_id})
            if conversation_id not in self.conversations:
                raise ConstraintViolation(
                    "preference conversation missing", {"conversation_id": conversation_id}
                )
            message = None
            message_conversation_id = None
            for cid, messages in self.messages.items():
                for msg in messages:
                    if msg.id == message_id:
                        message = msg
                        message_conversation_id = cid
                        break
                if message:
                    break
            if not message:
                raise ConstraintViolation(
                    "preference message missing",
                    {"message_id": message_id, "conversation_id": conversation_id},
                )
            if message_conversation_id != conversation_id:
                raise ConstraintViolation(
                    "preference message conversation mismatch",
                    {"message_id": message_id, "conversation_id": conversation_id},
                )
            event_id = str(uuid.uuid4())
            normalized_weight = weight if weight is not None else 1.0
            embedding = context_embedding or self._text_embedding(
                context_text or message.content
            )
            event = PreferenceEvent(
                id=event_id,
                user_id=user_id,
                conversation_id=conversation_id,
                message_id=message_id,
                feedback=feedback,
                score=score,
                explicit_signal=explicit_signal,
                context_embedding=embedding,
                cluster_id=cluster_id,
                context_text=context_text,
                corrected_text=corrected_text,
                weight=normalized_weight,
                meta=meta,
            )
            self.preference_events[event_id] = event
            self._persist_state()
            return event

    def list_preference_events(
        self,
        user_id: Optional[str] = None,
        feedback: Optional[Iterable[str] | str] = None,
        cluster_id: Optional[str] = None,
        *,
        tenant_id: Optional[str] = None,
    ) -> List[PreferenceEvent]:
        events = list(self.preference_events.values())
        if tenant_id:
            events = [
                e
                for e in events
                if e.user_id in self.users
                and self.users[e.user_id].tenant_id == tenant_id
            ]
        if user_id:
            events = [e for e in events if e.user_id == user_id]
        if feedback:
            feedback_values = {feedback} if isinstance(feedback, str) else set(feedback)
            events = [e for e in events if e.feedback in feedback_values]
        if cluster_id:
            events = [e for e in events if e.cluster_id == cluster_id]
        return sorted(events, key=lambda e: e.created_at)

    def get_preference_event(self, event_id: str) -> Optional[PreferenceEvent]:
        with self._data_lock:
            return self.preference_events.get(event_id)

    def update_preference_event(
        self, event_id: str, *, cluster_id: Optional[str] = None
    ) -> Optional[PreferenceEvent]:
        with self._data_lock:
            event = self.preference_events.get(event_id)
            if not event:
                return None
            if cluster_id:
                event.cluster_id = cluster_id
            self._persist_state()
            return event

    # semantic clusters
    def upsert_semantic_cluster(
        self,
        *,
        cluster_id: Optional[str] = None,
        user_id: Optional[str],
        centroid: List[float],
        size: int,
        label: Optional[str] = None,
        description: Optional[str] = None,
        sample_message_ids: Optional[List[str]] = None,
        meta: Optional[Dict] = None,
    ) -> SemanticCluster:
        cid = cluster_id or str(uuid.uuid4())
        now = datetime.utcnow()
        existing = self.semantic_clusters.get(cid)
        created_at = existing.created_at if existing else now
        cluster = SemanticCluster(
            id=cid,
            user_id=user_id,
            centroid=list(centroid),
            size=size,
            label=(
                label if label is not None else (existing.label if existing else None)
            ),
            description=(
                description
                if description is not None
                else (existing.description if existing else None)
            ),
            sample_message_ids=sample_message_ids
            or (existing.sample_message_ids if existing else []),
            created_at=created_at,
            updated_at=now,
            meta=meta or (existing.meta if existing else None),
        )
        self.semantic_clusters[cid] = cluster
        self._persist_state()
        return cluster

    def update_semantic_cluster(
        self,
        cluster_id: str,
        *,
        label: Optional[str] = None,
        description: Optional[str] = None,
        centroid: Optional[List[float]] = None,
        size: Optional[int] = None,
        meta: Optional[Dict] = None,
    ) -> Optional[SemanticCluster]:
        cluster = self.semantic_clusters.get(cluster_id)
        if not cluster:
            return None
        if label is not None:
            cluster.label = label
        if description is not None:
            cluster.description = description
        if centroid is not None:
            cluster.centroid = list(centroid)
        if size is not None:
            cluster.size = size
        if meta is not None:
            cluster.meta = meta
        cluster.updated_at = datetime.utcnow()
        self._persist_state()
        return cluster

    def list_semantic_clusters(
        self, user_id: Optional[str] = None
    ) -> List[SemanticCluster]:
        clusters = list(self.semantic_clusters.values())
        if user_id:
            clusters = [c for c in clusters if c.user_id == user_id]
        return sorted(clusters, key=lambda c: c.updated_at, reverse=True)

    def get_semantic_cluster(self, cluster_id: str) -> Optional[SemanticCluster]:
        return self.semantic_clusters.get(cluster_id)

    # training jobs
    def create_training_job(
        self,
        user_id: str,
        adapter_id: str,
        preference_event_ids: Optional[List[str]] = None,
        dataset_path: Optional[str] = None,
        meta: Optional[Dict] = None,
    ) -> TrainingJob:
        if user_id not in self.users:
            raise ConstraintViolation("training user missing", {"user_id": user_id})
        job_id = str(uuid.uuid4())
        pref_ids = preference_event_ids or []
        job = TrainingJob(
            id=job_id,
            user_id=user_id,
            adapter_id=adapter_id,
            num_events=len(pref_ids) if pref_ids else None,
            preference_event_ids=pref_ids,
            dataset_path=dataset_path,
            meta=meta,
        )
        self.training_jobs[job_id] = job
        self._persist_state()
        return job

    def get_training_job(self, job_id: str) -> Optional[TrainingJob]:
        return self.training_jobs.get(job_id)

    def update_training_job(
        self,
        job_id: str,
        *,
        status: Optional[str] = None,
        loss: Optional[float] = None,
        new_version: Optional[int] = None,
        dataset_path: Optional[str] = None,
        meta: Optional[Dict] = None,
    ) -> Optional[TrainingJob]:
        job = self.training_jobs.get(job_id)
        if not job:
            return None
        if status:
            job.status = status
        if loss is not None:
            job.loss = loss
        if new_version is not None:
            job.new_version = new_version
        if dataset_path is not None:
            job.dataset_path = dataset_path
        if meta is not None:
            job.meta = meta
        job.updated_at = datetime.utcnow()
        self._persist_state()
        return job

    def list_training_jobs(
        self,
        user_id: Optional[str] = None,
        status: Optional[str] = None,
        *,
        limit: Optional[int] = None,
        tenant_id: Optional[str] = None,
    ) -> List[TrainingJob]:
        jobs = list(self.training_jobs.values())
        if tenant_id:
            jobs = [
                j
                for j in jobs
                if j.user_id in self.users
                and self.users[j.user_id].tenant_id == tenant_id
            ]
        if user_id:
            jobs = [j for j in jobs if j.user_id == user_id]
        if status:
            jobs = [j for j in jobs if j.status == status]
        jobs.sort(key=lambda j: j.updated_at or j.created_at, reverse=True)
        return jobs if limit is None else jobs[:limit]

    def inspect_state(
        self,
        *,
        tenant_id: Optional[str] = None,
        kind: Optional[str] = None,
        limit: int = 50,
    ) -> dict:
        def _serialize(obj: Any) -> dict:
            if hasattr(obj, "__dict__"):
                data = dict(obj.__dict__)
            elif isinstance(obj, dict):
                data = dict(obj)
            else:
                return {"value": str(obj)}
            for k, v in list(data.items()):
                if isinstance(v, datetime):
                    data[k] = v.isoformat()
            return data

        sections: dict[str, list] = {}
        if kind in (None, "users"):
            sections["users"] = [
                _serialize(u) for u in self.list_users(tenant_id=tenant_id, limit=limit)
            ]
        if kind in (None, "sessions"):
            sessions = [
                s
                for s in self.sessions.values()
                if not tenant_id or s.tenant_id == tenant_id
            ]
            sections["sessions"] = [
                _serialize(s)
                for s in sorted(sessions, key=lambda s: s.created_at, reverse=True)[
                    :limit
                ]
            ]
        if kind in (None, "conversations"):
            convs = [
                c
                for c in self.conversations.values()
                if not tenant_id
                or (
                    self.users.get(c.user_id)
                    and self.users[c.user_id].tenant_id == tenant_id
                )
            ]
            sections["conversations"] = [
                _serialize(c)
                for c in sorted(convs, key=lambda c: c.updated_at, reverse=True)[:limit]
            ]
        if kind in (None, "messages"):
            flattened: list[Message] = []
            for msgs in self.messages.values():
                for msg in msgs:
                    conv = self.conversations.get(msg.conversation_id)
                    if tenant_id and conv:
                        user = self.users.get(conv.user_id)
                        if not user or user.tenant_id != tenant_id:
                            continue
                    flattened.append(msg)
            sections["messages"] = [
                _serialize(m)
                for m in sorted(flattened, key=lambda m: m.created_at, reverse=True)[
                    :limit
                ]
            ]
        if kind in (None, "artifacts"):
            sections["artifacts"] = [
                _serialize(a) for a in list(self.artifacts.values())[:limit]
            ]
        if kind in (None, "contexts"):
            contexts = [
                c
                for c in self.contexts.values()
                if not tenant_id
                or (
                    c.owner_user_id
                    and self.users.get(c.owner_user_id)
                    and self.users[c.owner_user_id].tenant_id == tenant_id
                )
            ]
            sections["contexts"] = [_serialize(c) for c in contexts[:limit]]
        if kind in (None, "chunks"):
            flattened_chunks: list[KnowledgeChunk] = []
            for context_id, chunk_list in self.chunks.items():
                ctx = self.contexts.get(context_id)
                if tenant_id and ctx and ctx.owner_user_id:
                    user = self.users.get(ctx.owner_user_id)
                    if not user or user.tenant_id != tenant_id:
                        continue
                flattened_chunks.extend(chunk_list)
            sections["chunks"] = [_serialize(ch) for ch in flattened_chunks[:limit]]
        if kind in (None, "training_jobs"):
            jobs = [
                t
                for t in self.training_jobs.values()
                if not tenant_id
                or (
                    self.users.get(t.user_id)
                    and self.users[t.user_id].tenant_id == tenant_id
                )
            ]
            sections["training_jobs"] = [_serialize(t) for t in jobs[:limit]]
        if kind in (None, "config_patches"):
            sections["config_patches"] = [
                _serialize(p) for p in list(self.config_patches.values())[:limit]
            ]
        return sections

    def list_messages(
        self,
        conversation_id: str,
        limit: Optional[int] = None,
        *,
        user_id: Optional[str] = None,
        **_: Any,
    ) -> List[Message]:
        conv = (
            self.get_conversation(conversation_id, user_id=user_id)
            if user_id
            else self.conversations.get(conversation_id)
        )
        if not conv:
            return []
        msgs = self.messages.get(conversation_id, [])
        if limit is None:
            return list(msgs)
        return msgs[-limit:]

    def list_conversations(self, user_id: str, limit: int = 20) -> List[Conversation]:
        convs = [c for c in self.conversations.values() if c.user_id == user_id]
        convs.sort(key=lambda c: c.updated_at, reverse=True)
        return convs[:limit]

    # artifacts
    def list_artifacts(
        self,
        type_filter: Optional[str] = None,
        kind_filter: Optional[str] = None,
        *,
        page: int = 1,
        page_size: int = 100,
        owner_user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> List[Artifact]:
        artifacts = list(self.artifacts.values())
        if tenant_id:
            artifacts = [
                a
                for a in artifacts
                if a.owner_user_id
                and a.owner_user_id in self.users
                and self.users[a.owner_user_id].tenant_id == tenant_id
            ]
        if owner_user_id:
            artifacts = [a for a in artifacts if a.owner_user_id == owner_user_id]
        if type_filter:
            artifacts = [a for a in artifacts if a.type == type_filter]
        if kind_filter:
            artifacts = [
                a
                for a in artifacts
                if isinstance(a.schema, dict) and a.schema.get("kind") == kind_filter
            ]
        start = max(page - 1, 0) * max(page_size, 1)
        end = start + max(page_size, 1)
        return artifacts[start:end]

    def get_artifact(self, artifact_id: str) -> Optional[Artifact]:
        return self.artifacts.get(artifact_id)

    def create_artifact(
        self,
        type_: str,
        name: str,
        schema: dict,
        description: str = "",
        owner_user_id: Optional[str] = None,
        *,
        version_author: Optional[str] = None,
        change_note: Optional[str] = None,
    ) -> Artifact:
        try:
            validate_artifact(type_, schema)
        except ArtifactValidationError as exc:
            self.logger.warning("artifact_validation_failed", errors=exc.errors)
            raise
        if owner_user_id and owner_user_id not in self.users:
            raise ConstraintViolation(
                "artifact owner missing", {"owner_user_id": owner_user_id}
            )
        artifact_id = str(uuid.uuid4())
        fs_path = self.persist_artifact_payload(artifact_id, schema)
        author = version_author or owner_user_id or "system_llm"
        artifact = Artifact(
            id=artifact_id,
            type=type_,
            name=name,
            schema=schema,
            description=description,
            owner_user_id=owner_user_id,
            fs_path=fs_path,
            base_model=schema.get("base_model"),
        )
        self.artifacts[artifact_id] = artifact
        self.artifact_versions.setdefault(artifact_id, []).append(
            ArtifactVersion(
                id=self._next_artifact_version_id(),
                artifact_id=artifact_id,
                version=1,
                schema=schema,
                created_by=author,
                change_note=change_note,
                fs_path=fs_path,
                base_model=schema.get("base_model"),
            )
        )
        self._persist_state()
        return artifact

    def update_artifact(
        self,
        artifact_id: str,
        schema: dict,
        description: Optional[str] = None,
        *,
        version_author: Optional[str] = None,
        change_note: Optional[str] = None,
    ) -> Optional[Artifact]:
        schema_kind = schema.get("kind")
        if schema_kind == "workflow.chat":
            validator_type = "workflow"
        elif schema_kind == "tool.spec":
            validator_type = "tool"
        elif schema_kind == "adapter.lora":
            validator_type = "adapter"
        else:
            validator_type = "artifact"
        try:
            validate_artifact(validator_type, schema)  # type: ignore[arg-type]
        except ArtifactValidationError as exc:
            self.logger.warning("artifact_validation_failed", errors=exc.errors)
            raise
        artifact = self.artifacts.get(artifact_id)
        if not artifact:
            return None
        fs_path = self.persist_artifact_payload(artifact_id, schema)
        author = version_author or artifact.owner_user_id or "system_llm"
        artifact.schema = schema
        if description is not None:
            artifact.description = description
        artifact.updated_at = datetime.utcnow()
        artifact.fs_path = fs_path
        artifact.base_model = schema.get("base_model")
        versions = self.artifact_versions.setdefault(artifact_id, [])
        versions.append(
            ArtifactVersion(
                id=self._next_artifact_version_id(),
                artifact_id=artifact_id,
                version=(versions[-1].version + 1 if versions else 1),
                schema=schema,
                created_by=author,
                change_note=change_note,
                fs_path=fs_path,
                base_model=schema.get("base_model"),
            )
        )
        self._persist_state()
        return artifact

    def list_artifact_versions(self, artifact_id: str) -> List[ArtifactVersion]:
        versions = list(self.artifact_versions.get(artifact_id, []))
        return sorted(versions, key=lambda v: v.version, reverse=True)

    def get_artifact_current_version(self, artifact_id: str) -> int:
        """Get the current (highest) version number for an artifact."""
        versions = self.artifact_versions.get(artifact_id, [])
        if not versions:
            return 1
        return max(v.version for v in versions)

    def get_artifact_current_versions(self, artifact_ids: List[str]) -> Dict[str, int]:
        """Get current versions for multiple artifacts efficiently."""
        result: Dict[str, int] = {}
        for artifact_id in artifact_ids:
            versions = self.artifact_versions.get(artifact_id, [])
            result[artifact_id] = max((v.version for v in versions), default=1)
        return result

    def _persist_payload(self, artifact_id: str, schema: dict) -> str:
        return self.persist_artifact_payload(artifact_id, schema)

    def persist_artifact_payload(self, artifact_id: str, schema: dict) -> str:
        artifact_dir = self.fs_root / "artifacts" / artifact_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        # Thread-safe version calculation to prevent race conditions
        with self._seq_lock:
            version = len(self.artifact_versions.get(artifact_id, [])) + 1
            payload_path = artifact_dir / f"v{version}.json"
            payload_path.write_text(json.dumps(schema, indent=2))
        return str(payload_path)

    def record_config_patch(
        self, artifact_id: str, proposer: str, patch: dict, justification: Optional[str]
    ) -> ConfigPatchAudit:
        # Thread-safe ID generation to prevent collisions
        with self._seq_lock:
            audit_id = max(self.config_patches.keys(), default=0) + 1
            audit = ConfigPatchAudit(
                id=audit_id,
                artifact_id=artifact_id,
                proposer=proposer,
                patch=patch,
                justification=justification,
            )
            self.config_patches[audit.id] = audit
        self._persist_state()
        return audit

    def get_config_patch(self, patch_id: int) -> Optional[ConfigPatchAudit]:
        return self.config_patches.get(patch_id)

    def list_config_patches(
        self, status: Optional[str] = None
    ) -> List[ConfigPatchAudit]:
        patches = list(self.config_patches.values())
        if status:
            patches = [p for p in patches if p.status == status]
        return sorted(patches, key=lambda p: p.created_at, reverse=True)

    def update_config_patch_status(
        self,
        patch_id: int,
        status: str,
        *,
        meta: Optional[Dict] = None,
        mark_decided: bool = False,
        mark_applied: bool = False,
    ) -> Optional[ConfigPatchAudit]:
        patch = self.config_patches.get(patch_id)
        if not patch:
            return None
        patch.status = status
        now = datetime.utcnow()
        if mark_decided and not patch.decided_at:
            patch.decided_at = now
        if mark_applied:
            patch.applied_at = now
        if meta:
            merged = dict(patch.meta or {})
            merged.update(meta)
            patch.meta = merged
        self._persist_state()
        return patch

    def get_runtime_config(self) -> dict:
        """Return the runtime configuration persisted for the web admin UI."""

        return dict(self.runtime_config)

    def get_latest_workflow(self, workflow_id: str) -> Optional[dict]:
        versions = self.artifact_versions.get(workflow_id)
        if not versions:
            return None
        return versions[-1].schema

    def list_adapter_router_state(
        self, user_id: Optional[str] = None
    ) -> list[AdapterRouterState]:
        """Return synthetic router state for adapters in memory."""

        adapters = [a for a in self.artifacts.values() if a.type == "adapter"]
        if user_id:
            adapters = [a for a in adapters if a.owner_user_id == user_id]
        return [
            AdapterRouterState(
                artifact_id=a.id,
                base_model=a.schema.get("base_model"),
                centroid_vec=a.schema.get("embedding_centroid") or [],
                usage_count=0,
                success_score=0.0,
                meta=None,
            )
            for a in adapters
        ]

    def upsert_context(
        self,
        owner_user_id: Optional[str],
        name: str,
        description: str,
        fs_path: Optional[str] = None,
        meta: Optional[dict] = None,
    ) -> KnowledgeContext:
        if not owner_user_id:
            raise ConstraintViolation(
                "context owner required", {"owner_user_id": owner_user_id}
            )
        if owner_user_id not in self.users:
            raise ConstraintViolation(
                "context owner missing", {"owner_user_id": owner_user_id}
            )
        ctx_id = str(uuid.uuid4())
        ctx = KnowledgeContext(
            id=ctx_id,
            owner_user_id=owner_user_id,
            name=name,
            description=description,
            fs_path=fs_path,
            meta=meta,
        )
        self.contexts[ctx.id] = ctx
        self._persist_state()
        return ctx

    def get_context(self, context_id: str) -> Optional[KnowledgeContext]:
        return self.contexts.get(context_id)

    def list_contexts(
        self, owner_user_id: Optional[str] = None
    ) -> List[KnowledgeContext]:
        if not owner_user_id:
            return []
        return [
            ctx for ctx in self.contexts.values() if ctx.owner_user_id == owner_user_id
        ]

    def add_context_source(
        self,
        context_id: str,
        fs_path: str,
        recursive: bool = True,
        meta: Optional[Dict] = None,
    ) -> ContextSource:
        if context_id not in self.contexts:
            raise ConstraintViolation("context not found", {"context_id": context_id})
        if not fs_path or not fs_path.strip():
            raise ConstraintViolation(
                "fs_path required for context_source", {"fs_path": fs_path}
            )
        source = ContextSource(
            id=str(uuid.uuid4()),
            context_id=context_id,
            fs_path=fs_path,
            recursive=recursive,
            meta=meta,
        )
        sources = self.context_sources.setdefault(context_id, [])
        sources.append(source)
        self._persist_state()
        return source

    def list_context_sources(
        self, context_id: Optional[str] = None
    ) -> List[ContextSource]:
        if context_id:
            return list(self.context_sources.get(context_id, []))
        all_sources: List[ContextSource] = []
        for sources in self.context_sources.values():
            all_sources.extend(sources)
        return all_sources

    def add_chunks(self, context_id: str, chunks: Iterable[KnowledgeChunk]) -> None:
        if context_id not in self.contexts:
            raise ConstraintViolation("context not found", {"context_id": context_id})
        existing = self.chunks.setdefault(context_id, [])
        for chunk in chunks:
            if not chunk.fs_path or not str(chunk.fs_path).strip():
                raise ConstraintViolation(
                    "fs_path required for knowledge_chunk", {"fs_path": chunk.fs_path}
                )
            chunk_id = getattr(chunk, "id", None)
            if isinstance(chunk_id, str):
                try:
                    chunk_id = int(chunk_id)
                except ValueError:
                    chunk_id = None
            if not chunk_id or int(chunk_id) <= 0:
                # Use thread-safe method for ID assignment
                chunk.id = self._next_chunk_id()
            else:
                chunk.id = int(chunk_id)
                # Update sequence counter safely
                with self._seq_lock:
                    self._chunk_id_seq = max(self._chunk_id_seq, chunk.id + 1)
            existing.append(chunk)
        self._persist_state()

    def list_chunks(
        self, context_id: Optional[str] = None, *, owner_user_id: Optional[str] = None
    ) -> List[KnowledgeChunk]:
        if context_id:
            ctx = self.contexts.get(context_id)
            if owner_user_id and (not ctx or ctx.owner_user_id != owner_user_id):
                return []
            return list(self.chunks.get(context_id, []))
        if not owner_user_id:
            return []
        chunks: List[KnowledgeChunk] = []
        for vals in self.chunks.values():
            if vals:
                ctx = self.contexts.get(vals[0].context_id)
                if ctx and ctx.owner_user_id == owner_user_id:
                    chunks.extend(vals)
        return chunks

    def search_chunks(
        self,
        context_id: Optional[str],
        query: str,
        query_embedding: Optional[List[float]],
        limit: int = 4,
    ) -> List[KnowledgeChunk]:
        candidates = self.list_chunks(context_id)
        if not candidates:
            return []

        query_tokens = _tokenize_text(query)
        doc_tokens = [_tokenize_text(ch.content) for ch in candidates]
        bm25_scores = _compute_bm25_scores(query_tokens, doc_tokens)
        semantic_scores = []
        for ch in candidates:
            if not query_embedding or not ch.embedding:
                semantic_scores.append(0.0)
                continue
            dim = min(len(query_embedding), len(ch.embedding))
            semantic_scores.append(
                cosine_similarity(query_embedding[:dim], ch.embedding[:dim])
            )
        max_bm25 = max(bm25_scores) or 1.0
        combined: Dict[str, tuple[KnowledgeChunk, float]] = {}
        for chunk, lex, sem in zip(candidates, bm25_scores, semantic_scores):
            hybrid = 0.45 * (lex / max_bm25) + 0.55 * sem
            key = " ".join(chunk.content.split()).lower() or str(chunk.id or "")
            existing = combined.get(key)
            if not existing or hybrid > existing[1]:
                combined[key] = (chunk, hybrid)
        ranked = sorted(combined.values(), key=lambda pair: pair[1], reverse=True)
        return [pair[0] for pair in ranked[:limit]]

    def search_chunks_pgvector(
        self,
        context_ids: Optional[Sequence[str]],
        query: str,
        query_embedding: List[float],
        limit: int = 4,
        filters: Optional[dict] = None,
        *,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> List[KnowledgeChunk]:
        if not context_ids or not query_embedding:
            return []

        allowed_chunks: List[KnowledgeChunk] = []
        for ctx_id in context_ids:
            ctx = self.contexts.get(ctx_id)
            if not ctx:
                continue
            if user_id and ctx.owner_user_id != user_id:
                continue
            if tenant_id:
                if not ctx.owner_user_id:
                    continue
                owner = self.users.get(ctx.owner_user_id)
                if not owner or owner.tenant_id != tenant_id:
                    continue
            allowed_chunks.extend(self.list_chunks(ctx_id))

        if not allowed_chunks:
            return []

        if filters and filters.get("embedding_model_id"):
            allowed_chunks = [
                c
                for c in allowed_chunks
                if (c.meta or {}).get("embedding_model_id")
                == filters.get("embedding_model_id")
            ]
        if not allowed_chunks:
            return []
        if filters and filters.get("fs_path"):
            allowed_chunks = [
                c for c in allowed_chunks if c.fs_path == filters["fs_path"]
            ]
        if not allowed_chunks:
            return []

        # pgvector mode: pure vector similarity (no BM25) per SPEC §3
        # Sort by cosine similarity to emulate pgvector ORDER BY embedding <-> query
        scored: List[tuple[KnowledgeChunk, float]] = []
        for ch in allowed_chunks:
            if not ch.embedding:
                scored.append((ch, 0.0))
                continue
            dim = min(len(query_embedding), len(ch.embedding))
            sim = cosine_similarity(query_embedding[:dim], ch.embedding[:dim])
            scored.append((ch, sim))
        ranked = sorted(scored, key=lambda pair: pair[1], reverse=True)
        return [chunk for chunk, _ in ranked[:limit]]

    def _persist_state(self) -> None:
        state = {
            "users": [self._serialize_user(u) for u in self.users.values()],
            "sessions": [self._serialize_session(s) for s in self.sessions.values()],
            "credentials": [
                {
                    "user_id": user_id,
                    "password_hash": creds[0],
                    "password_algo": creds[1],
                }
                for user_id, creds in self.credentials.items()
            ],
            "providers": [self._serialize_provider(p) for p in self.providers],
            "conversations": [
                self._serialize_conversation(c) for c in self.conversations.values()
            ],
            "messages": [
                self._serialize_message(m)
                for msgs in self.messages.values()
                for m in msgs
            ],
            "artifacts": [self._serialize_artifact(a) for a in self.artifacts.values()],
            "artifact_versions": [
                self._serialize_artifact_version(v)
                for versions in self.artifact_versions.values()
                for v in versions
            ],
            "config_patches": [
                self._serialize_config_patch(cp) for cp in self.config_patches.values()
            ],
            "runtime_config": self.runtime_config,
            "contexts": [
                self._serialize_context(ctx) for ctx in self.contexts.values()
            ],
            "context_sources": [
                self._serialize_context_source(src)
                for srcs in self.context_sources.values()
                for src in srcs
            ],
            "chunks": [
                self._serialize_chunk(ch) for chs in self.chunks.values() for ch in chs
            ],
            "preference_events": [
                self._serialize_preference_event(e)
                for e in self.preference_events.values()
            ],
            "training_jobs": [
                self._serialize_training_job(j) for j in self.training_jobs.values()
            ],
            "semantic_clusters": [
                self._serialize_semantic_cluster(c)
                for c in self.semantic_clusters.values()
            ],
            "mfa_secrets": [
                self._serialize_mfa_config(cfg) for cfg in self.mfa_secrets.values()
            ],
        }
        path = self._state_path()
        try:
            path.write_text(json.dumps(state, indent=2))
        except Exception as exc:
            raise RuntimeError(f"failed to persist in-memory state: {exc}")

    def _load_state(self) -> bool:
        path = self._state_path()
        # Use try-except instead of exists() to avoid TOCTOU race condition
        try:
            data = json.loads(path.read_text())
        except FileNotFoundError:
            return False
        self.users = {u["id"]: self._deserialize_user(u) for u in data.get("users", [])}
        self.sessions = {
            s["id"]: self._deserialize_session(s) for s in data.get("sessions", [])
        }
        self.credentials = {
            entry["user_id"]: (entry["password_hash"], entry.get("password_algo", ""))
            for entry in data.get("credentials", [])
        }
        self.providers = [
            self._deserialize_provider(p) for p in data.get("providers", [])
        ]
        self.conversations = {
            c["id"]: self._deserialize_conversation(c)
            for c in data.get("conversations", [])
        }
        self.messages = {}
        for msg_data in data.get("messages", []):
            msg = self._deserialize_message(msg_data)
            self.messages.setdefault(msg.conversation_id, []).append(msg)
        for convo_messages in self.messages.values():
            convo_messages.sort(key=lambda m: m.seq)
        self.artifacts = {
            a["id"]: self._deserialize_artifact(a) for a in data.get("artifacts", [])
        }
        self.artifact_versions = {}
        for version_data in data.get("artifact_versions", []):
            version = self._deserialize_artifact_version(version_data)
            versions = self.artifact_versions.setdefault(version.artifact_id, [])
            versions.append(version)
        for versions in self.artifact_versions.values():
            versions.sort(key=lambda v: v.version)
        max_artifact_version_id = max(
            (
                version.id
                for versions in self.artifact_versions.values()
                for version in versions
            ),
            default=0,
        )
        self._artifact_version_seq = max_artifact_version_id + 1
        self.runtime_config = data.get("runtime_config", {})
        self.config_patches = {}
        for cp in data.get("config_patches", []):
            deserialized = self._deserialize_config_patch(cp)
            self.config_patches[deserialized.id] = deserialized
        self.contexts = {
            ctx["id"]: self._deserialize_context(ctx)
            for ctx in data.get("contexts", [])
        }
        self.context_sources = {}
        for src_data in data.get("context_sources", []):
            source = self._deserialize_context_source(src_data)
            self.context_sources.setdefault(source.context_id, []).append(source)
        self.chunks = {}
        for chunk_data in data.get("chunks", []):
            chunk = self._deserialize_chunk(chunk_data)
            self.chunks.setdefault(chunk.context_id, []).append(chunk)
        max_chunk_id = max(
            (chunk.id or 0 for chunks in self.chunks.values() for chunk in chunks),
            default=0,
        )
        self._chunk_id_seq = max_chunk_id + 1
        self.preference_events = {
            e["id"]: self._deserialize_preference_event(e)
            for e in data.get("preference_events", [])
        }
        self.training_jobs = {
            j["id"]: self._deserialize_training_job(j)
            for j in data.get("training_jobs", [])
        }
        self.semantic_clusters = {
            c["id"]: self._deserialize_semantic_cluster(c)
            for c in data.get("semantic_clusters", [])
        }
        self.mfa_secrets = {
            cfg["user_id"]: self._deserialize_mfa_config(cfg)
            for cfg in data.get("mfa_secrets", [])
        }
        return True

    def _serialize_user(self, user: User) -> dict:
        return {
            "id": user.id,
            "email": user.email,
            "handle": user.handle,
            "role": user.role,
            "tenant_id": user.tenant_id,
            "created_at": self._serialize_datetime(user.created_at),
            "is_active": user.is_active,
            "plan_tier": user.plan_tier,
            "meta": user.meta,
        }

    def _deserialize_user(self, data: dict) -> User:
        return User(
            id=str(data["id"]),
            email=data["email"],
            handle=data.get("handle"),
            role=data.get("role", "user"),
            tenant_id=data.get("tenant_id", "public"),
            created_at=self._deserialize_datetime(data["created_at"]),
            is_active=data.get("is_active", True),
            plan_tier=data.get("plan_tier", "free"),
            meta=data.get("meta"),
        )

    def _serialize_provider(self, provider: UserAuthProvider) -> dict:
        return {
            "id": provider.id,
            "user_id": provider.user_id,
            "provider": provider.provider,
            "provider_uid": provider.provider_uid,
            "created_at": self._serialize_datetime(provider.created_at),
        }

    def _deserialize_provider(self, data: dict) -> UserAuthProvider:
        return UserAuthProvider(
            id=int(data["id"]),
            user_id=str(data["user_id"]),
            provider=data["provider"],
            provider_uid=data["provider_uid"],
            created_at=self._deserialize_datetime(
                data.get("created_at", datetime.utcnow().isoformat())
            ),
        )

    def _serialize_session(self, session: Session) -> dict:
        return {
            "id": session.id,
            "user_id": session.user_id,
            "created_at": self._serialize_datetime(session.created_at),
            "expires_at": self._serialize_datetime(session.expires_at),
            "user_agent": session.user_agent,
            "ip_addr": str(session.ip_addr) if session.ip_addr is not None else None,
            "mfa_required": session.mfa_required,
            "mfa_verified": session.mfa_verified,
            "tenant_id": session.tenant_id,
            "meta": session.meta,
        }

    def _deserialize_session(self, data: dict) -> Session:
        raw_ip = data.get("ip_addr")
        ip_val = None
        if isinstance(raw_ip, str):
            if raw_ip.strip():
                ip_val = ip_address(raw_ip)
        else:
            ip_val = raw_ip
        return Session(
            id=data["id"],
            user_id=data["user_id"],
            created_at=self._deserialize_datetime(data["created_at"]),
            expires_at=self._deserialize_datetime(data["expires_at"]),
            user_agent=data.get("user_agent"),
            ip_addr=ip_val,
            mfa_required=data.get("mfa_required", False),
            mfa_verified=data.get("mfa_verified", False),
            tenant_id=data.get("tenant_id", "public"),
            meta=data.get("meta"),
            allow_expired=True,
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
            "content_struct": message.content_struct,
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
            content_struct=data.get("content_struct"),
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
            "base_model": artifact.base_model,
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
            base_model=data.get("base_model"),
            meta=data.get("meta"),
        )

    def _serialize_artifact_version(self, version: ArtifactVersion) -> dict:
        return {
            "id": version.id,
            "artifact_id": version.artifact_id,
            "version": version.version,
            "schema": version.schema,
            "created_by": version.created_by,
            "change_note": version.change_note,
            "created_at": self._serialize_datetime(version.created_at),
            "fs_path": version.fs_path,
            "base_model": version.base_model,
            "meta": version.meta,
        }

    def _deserialize_artifact_version(self, data: dict) -> ArtifactVersion:
        return ArtifactVersion(
            id=data["id"],
            artifact_id=data["artifact_id"],
            version=data["version"],
            schema=data.get("schema", {}),
            created_by=data.get("created_by", "system_llm"),
            change_note=data.get("change_note"),
            created_at=self._deserialize_datetime(data["created_at"]),
            fs_path=data.get("fs_path"),
            base_model=data.get("base_model"),
            meta=data.get("meta"),
        )

    def _serialize_config_patch(self, patch: ConfigPatchAudit) -> dict:
        return {
            "id": patch.id,
            "artifact_id": patch.artifact_id,
            "proposer": patch.proposer,
            "patch": patch.patch,
            "justification": patch.justification,
            "status": patch.status,
            "created_at": self._serialize_datetime(patch.created_at),
            "decided_at": (
                self._serialize_datetime(patch.decided_at) if patch.decided_at else None
            ),
            "applied_at": (
                self._serialize_datetime(patch.applied_at) if patch.applied_at else None
            ),
            "meta": patch.meta,
        }

    def _deserialize_config_patch(self, data: dict) -> ConfigPatchAudit:
        return ConfigPatchAudit(
            id=data["id"],
            artifact_id=data["artifact_id"],
            proposer=data.get("proposer", "user"),
            patch=data.get("patch", {}),
            justification=data.get("justification"),
            status=data.get("status", "pending"),
            created_at=self._deserialize_datetime(data["created_at"]),
            decided_at=(
                self._deserialize_datetime(data["decided_at"])
                if data.get("decided_at")
                else None
            ),
            applied_at=(
                self._deserialize_datetime(data["applied_at"])
                if data.get("applied_at")
                else None
            ),
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
            owner_user_id=data["owner_user_id"],
            name=data["name"],
            description=data.get("description", ""),
            created_at=self._deserialize_datetime(data["created_at"]),
            updated_at=self._deserialize_datetime(data["updated_at"]),
            fs_path=data.get("fs_path"),
            meta=data.get("meta"),
        )

    def _serialize_context_source(self, src: ContextSource) -> dict:
        return {
            "id": src.id,
            "context_id": src.context_id,
            "fs_path": src.fs_path,
            "recursive": src.recursive,
            "meta": src.meta,
        }

    def _deserialize_context_source(self, data: dict) -> ContextSource:
        return ContextSource(
            id=data["id"],
            context_id=data["context_id"],
            fs_path=data.get("fs_path", ""),
            recursive=bool(data.get("recursive", True)),
            meta=data.get("meta"),
        )

    def _serialize_chunk(self, chunk: KnowledgeChunk) -> dict:
        return {
            "id": chunk.id,
            "context_id": chunk.context_id,
            "fs_path": chunk.fs_path,
            "content": chunk.content,
            "embedding": chunk.embedding,
            "chunk_index": chunk.chunk_index,
            "created_at": self._serialize_datetime(chunk.created_at),
            "meta": chunk.meta,
        }

    def _deserialize_chunk(self, data: dict) -> KnowledgeChunk:
        return KnowledgeChunk(
            id=int(data["id"]) if data.get("id") is not None else None,
            context_id=data["context_id"],
            fs_path=data.get("fs_path", ""),
            content=data.get("content", ""),
            embedding=data.get("embedding", []),
            chunk_index=data.get("chunk_index", 0),
            created_at=self._deserialize_datetime(data["created_at"]),
            meta=data.get("meta"),
        )

    def _serialize_preference_event(self, event: PreferenceEvent) -> dict:
        return {
            "id": event.id,
            "user_id": event.user_id,
            "conversation_id": event.conversation_id,
            "message_id": event.message_id,
            "feedback": event.feedback,
            "score": event.score,
            "explicit_signal": event.explicit_signal,
            "context_embedding": event.context_embedding,
            "cluster_id": event.cluster_id,
            "context_text": event.context_text,
            "corrected_text": event.corrected_text,
            "created_at": self._serialize_datetime(event.created_at),
            "weight": event.weight,
            "meta": event.meta,
        }

    def _deserialize_preference_event(self, data: dict) -> PreferenceEvent:
        return PreferenceEvent(
            id=data["id"],
            user_id=data["user_id"],
            conversation_id=data["conversation_id"],
            message_id=data["message_id"],
            feedback=data["feedback"],
            score=data.get("score"),
            explicit_signal=data.get("explicit_signal"),
            context_embedding=data.get("context_embedding", []),
            cluster_id=data.get("cluster_id"),
            context_text=data.get("context_text"),
            corrected_text=data.get("corrected_text"),
            created_at=self._deserialize_datetime(data["created_at"]),
            weight=float(data.get("weight", 1.0)),
            meta=data.get("meta"),
        )

    def _serialize_training_job(self, job: TrainingJob) -> dict:
        return {
            "id": job.id,
            "user_id": job.user_id,
            "adapter_id": job.adapter_id,
            "status": job.status,
            "num_events": job.num_events,
            "created_at": self._serialize_datetime(job.created_at),
            "updated_at": self._serialize_datetime(job.updated_at),
            "loss": job.loss,
            "preference_event_ids": job.preference_event_ids,
            "dataset_path": job.dataset_path,
            "new_version": job.new_version,
            "meta": job.meta,
        }

    def _deserialize_training_job(self, data: dict) -> TrainingJob:
        return TrainingJob(
            id=data["id"],
            user_id=data["user_id"],
            adapter_id=data["adapter_id"],
            status=data.get("status", "queued"),
            num_events=data.get("num_events"),
            created_at=self._deserialize_datetime(data["created_at"]),
            updated_at=self._deserialize_datetime(
                data.get("updated_at", data["created_at"])
            ),
            loss=data.get("loss"),
            preference_event_ids=list(data.get("preference_event_ids", [])),
            dataset_path=data.get("dataset_path"),
            new_version=data.get("new_version"),
            meta=data.get("meta"),
        )

    def _serialize_semantic_cluster(self, cluster: SemanticCluster) -> dict:
        return {
            "id": cluster.id,
            "user_id": cluster.user_id,
            "centroid": cluster.centroid,
            "size": cluster.size,
            "label": cluster.label,
            "description": cluster.description,
            "sample_message_ids": cluster.sample_message_ids,
            "created_at": self._serialize_datetime(cluster.created_at),
            "updated_at": self._serialize_datetime(cluster.updated_at),
            "meta": cluster.meta,
        }

    def _deserialize_semantic_cluster(self, data: dict) -> SemanticCluster:
        return SemanticCluster(
            id=data["id"],
            user_id=data.get("user_id"),
            centroid=data.get("centroid", []),
            size=data.get("size", 0),
            label=data.get("label"),
            description=data.get("description"),
            sample_message_ids=list(data.get("sample_message_ids", [])),
            created_at=self._deserialize_datetime(
                data.get("created_at", datetime.utcnow().isoformat())
            ),
            updated_at=self._deserialize_datetime(
                data.get("updated_at", datetime.utcnow().isoformat())
            ),
            meta=data.get("meta"),
        )

    def _serialize_mfa_config(self, cfg: UserMFAConfig) -> dict:
        return {
            "user_id": cfg.user_id,
            "secret": cfg.secret,
            "enabled": cfg.enabled,
            "created_at": self._serialize_datetime(cfg.created_at),
            "meta": cfg.meta,
        }

    def _deserialize_mfa_config(self, data: dict) -> UserMFAConfig:
        return UserMFAConfig(
            user_id=data["user_id"],
            secret=data["secret"],
            enabled=bool(data.get("enabled", False)),
            created_at=self._deserialize_datetime(
                data.get("created_at", datetime.utcnow().isoformat())
            ),
            meta=data.get("meta"),
        )
