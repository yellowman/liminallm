from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, List, Optional

import psycopg
from psycopg import errors
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from liminallm.logging import get_logger
from liminallm.service.artifact_validation import ArtifactValidationError, validate_artifact
from liminallm.storage.errors import ConstraintViolation
from liminallm.storage.models import (
    Artifact,
    ConfigPatchAudit,
    Conversation,
    KnowledgeChunk,
    KnowledgeContext,
    Message,
    PreferenceEvent,
    SemanticCluster,
    Session,
    TrainingJob,
    User,
    UserMFAConfig,
)


class PostgresStore:
    """Thin Postgres-backed store to persist kernel primitives."""

    def __init__(self, dsn: str, fs_root: str) -> None:
        self.dsn = dsn
        self.fs_root = Path(fs_root)
        self.fs_root.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(__name__)
        self.pool = ConnectionPool(self.dsn, min_size=2, max_size=10, kwargs={"row_factory": dict_row, "autocommit": True})
        self.mfa_secrets: dict[str, UserMFAConfig] = {}
        self.sessions: dict[str, Session] = {}
        self._load_training_state()
        self._ensure_default_artifacts()

    def _connect(self):
        return self.pool.connection()

    def _ensure_default_artifacts(self) -> None:
        """Seed the default workflow artifact if the database is empty."""

        existing = self.list_artifacts()
        if any(artifact.name == "default_chat_workflow" for artifact in existing):
            return

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
        self.create_artifact(
            "workflow",
            "default_chat_workflow",
            default_schema,
            "LLM-only chat workflow defined as data.",
            created_by="system_llm",
            change_note="Seeded default workflow",
        )

    # preference events
    def record_preference_event(
        self,
        user_id: str,
        conversation_id: str,
        message_id: str,
        feedback: str,
        *,
        score: float | None = None,
        explicit_signal: str | None = None,
        corrected_text: str | None = None,
        weight: float | None = None,
        context_embedding: list[float] | None = None,
        cluster_id: str | None = None,
        context_text: str | None = None,
        meta: dict | None = None,
    ) -> PreferenceEvent:
        normalized_weight = weight if weight is not None else (score if score is not None else 1.0)
        event_id = str(uuid.uuid4())
        with self._connect() as conn:
            row = conn.execute(
                """
                INSERT INTO preference_event (
                    id, user_id, conversation_id, message_id, feedback, score, explicit_signal,
                    context_embedding, cluster_id, context_text, corrected_text, weight, meta
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING *
                """,
                (
                    event_id,
                    user_id,
                    conversation_id,
                    message_id,
                    feedback,
                    score,
                    explicit_signal,
                    context_embedding,
                    cluster_id,
                    context_text,
                    corrected_text,
                    normalized_weight,
                    meta,
                ),
            ).fetchone()
        return PreferenceEvent(
            id=event_id,
            user_id=user_id,
            conversation_id=conversation_id,
            message_id=message_id,
            feedback=feedback,
            score=score,
            explicit_signal=explicit_signal,
            context_embedding=context_embedding or [],
            cluster_id=cluster_id,
            context_text=context_text,
            corrected_text=corrected_text,
            created_at=row.get("created_at", datetime.utcnow()) if row else datetime.utcnow(),
            weight=normalized_weight,
            meta=meta,
        )

    def list_preference_events(
        self, user_id: str | None = None, feedback: str | None = None, cluster_id: str | None = None
    ) -> list[PreferenceEvent]:
        clauses = []
        params: list[Any] = []
        if user_id:
            clauses.append("user_id = %s")
            params.append(user_id)
        if feedback:
            clauses.append("feedback = %s")
            params.append(feedback)
        if cluster_id:
            clauses.append("cluster_id = %s")
            params.append(cluster_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM preference_event {where} ORDER BY created_at", params
            ).fetchall()
        return [
            PreferenceEvent(
                id=str(row["id"]),
                user_id=str(row["user_id"]),
                conversation_id=str(row["conversation_id"]),
                message_id=str(row["message_id"]),
                feedback=row["feedback"],
                score=row.get("score"),
                explicit_signal=row.get("explicit_signal"),
                context_embedding=row.get("context_embedding") or [],
                cluster_id=row.get("cluster_id"),
                context_text=row.get("context_text"),
                corrected_text=row.get("corrected_text"),
                created_at=row.get("created_at", datetime.utcnow()),
                weight=float(row.get("weight", 1.0)),
                meta=row.get("meta"),
            )
            for row in rows
        ]

    def update_preference_event(self, event_id: str, *, cluster_id: str | None = None) -> PreferenceEvent | None:
        if cluster_id is None:
            return self.get_preference_event(event_id)
        with self._connect() as conn:
            row = conn.execute(
                "UPDATE preference_event SET cluster_id = %s WHERE id = %s RETURNING *",
                (cluster_id, event_id),
            ).fetchone()
        if not row:
            return None
        return PreferenceEvent(
            id=str(row["id"]),
            user_id=str(row["user_id"]),
            conversation_id=str(row["conversation_id"]),
            message_id=str(row["message_id"]),
            feedback=row["feedback"],
            score=row.get("score"),
            explicit_signal=row.get("explicit_signal"),
            context_embedding=row.get("context_embedding") or [],
            cluster_id=row.get("cluster_id"),
            context_text=row.get("context_text"),
            corrected_text=row.get("corrected_text"),
            created_at=row.get("created_at", datetime.utcnow()),
            weight=float(row.get("weight", 1.0)),
            meta=row.get("meta"),
        )

    def get_preference_event(self, event_id: str) -> PreferenceEvent | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM preference_event WHERE id = %s", (event_id,)).fetchone()
        if not row:
            return None
        return PreferenceEvent(
            id=str(row["id"]),
            user_id=str(row["user_id"]),
            conversation_id=str(row["conversation_id"]),
            message_id=str(row["message_id"]),
            feedback=row["feedback"],
            score=row.get("score"),
            explicit_signal=row.get("explicit_signal"),
            context_embedding=row.get("context_embedding") or [],
            cluster_id=row.get("cluster_id"),
            context_text=row.get("context_text"),
            corrected_text=row.get("corrected_text"),
            created_at=row.get("created_at", datetime.utcnow()),
            weight=float(row.get("weight", 1.0)),
            meta=row.get("meta"),
        )

    # semantic clusters
    def upsert_semantic_cluster(
        self,
        *,
        cluster_id: str | None = None,
        user_id: str | None,
        centroid: list[float],
        size: int,
        label: str | None = None,
        description: str | None = None,
        sample_message_ids: list[str] | None = None,
        meta: dict | None = None,
    ) -> SemanticCluster:
        cid = cluster_id or str(uuid.uuid4())
        now = datetime.utcnow()
        existing = self.get_semantic_cluster(cid)
        created_at = existing.created_at if existing else now
        cluster = SemanticCluster(
            id=cid,
            user_id=user_id,
            centroid=list(centroid),
            size=size,
            label=label if label is not None else (existing.label if existing else None),
            description=description if description is not None else (existing.description if existing else None),
            sample_message_ids=sample_message_ids or (existing.sample_message_ids if existing else []),
            created_at=created_at,
            updated_at=now,
            meta=meta or (existing.meta if existing else None),
        )
        with self._connect() as conn:
            row = conn.execute(
                """
                INSERT INTO semantic_cluster (id, user_id, centroid, size, label, description, sample_message_ids, meta, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE
                SET centroid = EXCLUDED.centroid,
                    size = EXCLUDED.size,
                    label = COALESCE(EXCLUDED.label, semantic_cluster.label),
                    description = COALESCE(EXCLUDED.description, semantic_cluster.description),
                    sample_message_ids = EXCLUDED.sample_message_ids,
                    meta = EXCLUDED.meta,
                    updated_at = now()
                RETURNING *
                """,
                (
                    cid,
                    user_id,
                    cluster.centroid,
                    size,
                    cluster.label,
                    cluster.description,
                    cluster.sample_message_ids,
                    cluster.meta,
                    created_at,
                    now,
                ),
            ).fetchone()
        return SemanticCluster(
            id=cid,
            user_id=user_id,
            centroid=cluster.centroid,
            size=size,
            label=row.get("label") if row else cluster.label,
            description=row.get("description") if row else cluster.description,
            sample_message_ids=row.get("sample_message_ids") if row else cluster.sample_message_ids,
            created_at=row.get("created_at", created_at) if row else created_at,
            updated_at=row.get("updated_at", now) if row else now,
            meta=row.get("meta") if row else cluster.meta,
        )

    def update_semantic_cluster(
        self,
        cluster_id: str,
        *,
        label: str | None = None,
        description: str | None = None,
        centroid: list[float] | None = None,
        size: int | None = None,
        meta: dict | None = None,
    ) -> SemanticCluster | None:
        existing = self.get_semantic_cluster(cluster_id)
        if not existing:
            return None
        new_centroid = list(centroid) if centroid is not None else existing.centroid
        new_size = size if size is not None else existing.size
        with self._connect() as conn:
            row = conn.execute(
                """
                UPDATE semantic_cluster
                SET label = %s,
                    description = %s,
                    centroid = %s,
                    size = %s,
                    meta = %s,
                    updated_at = now()
                WHERE id = %s
                RETURNING *
                """,
                (
                    label if label is not None else existing.label,
                    description if description is not None else existing.description,
                    new_centroid,
                    new_size,
                    meta if meta is not None else existing.meta,
                    cluster_id,
                ),
            ).fetchone()
        if not row:
            return None
        return SemanticCluster(
            id=str(row["id"]),
            user_id=row.get("user_id"),
            centroid=row.get("centroid", new_centroid) or [],
            size=row.get("size", new_size),
            label=row.get("label"),
            description=row.get("description"),
            sample_message_ids=row.get("sample_message_ids") or [],
            created_at=row.get("created_at", existing.created_at),
            updated_at=row.get("updated_at", datetime.utcnow()),
            meta=row.get("meta"),
        )

    def list_semantic_clusters(self, user_id: str | None = None) -> list[SemanticCluster]:
        with self._connect() as conn:
            if user_id:
                rows = conn.execute(
                    "SELECT * FROM semantic_cluster WHERE user_id = %s ORDER BY updated_at DESC", (user_id,)
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM semantic_cluster ORDER BY updated_at DESC", ()).fetchall()
        return [
            SemanticCluster(
                id=str(row["id"]),
                user_id=row.get("user_id"),
                centroid=row.get("centroid") or [],
                size=row.get("size", 0),
                label=row.get("label"),
                description=row.get("description"),
                sample_message_ids=row.get("sample_message_ids") or [],
                created_at=row.get("created_at", datetime.utcnow()),
                updated_at=row.get("updated_at", datetime.utcnow()),
                meta=row.get("meta"),
            )
            for row in rows
        ]

    def get_semantic_cluster(self, cluster_id: str) -> SemanticCluster | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM semantic_cluster WHERE id = %s", (cluster_id,)).fetchone()
        if not row:
            return None
        return SemanticCluster(
            id=str(row["id"]),
            user_id=row.get("user_id"),
            centroid=row.get("centroid") or [],
            size=row.get("size", 0),
            label=row.get("label"),
            description=row.get("description"),
            sample_message_ids=row.get("sample_message_ids") or [],
            created_at=row.get("created_at", datetime.utcnow()),
            updated_at=row.get("updated_at", datetime.utcnow()),
            meta=row.get("meta"),
        )

    def create_training_job(
        self,
        user_id: str,
        adapter_id: str,
        preference_event_ids: list[str] | None = None,
        dataset_path: str | None = None,
        meta: dict | None = None,
    ) -> TrainingJob:
        job_id = str(uuid.uuid4())
        now = datetime.utcnow()
        pref_ids = preference_event_ids or []
        num_events = len(pref_ids) if pref_ids else None
        with self._connect() as conn:
            row = conn.execute(
                """
                INSERT INTO training_job (
                    id, adapter_artifact_id, user_id, created_at, updated_at, status, num_events, loss,
                    dataset_path, new_version, preference_event_ids, meta
                )
                VALUES (%s, %s, %s, %s, %s, 'queued', %s, NULL, %s, NULL, %s, %s)
                RETURNING *
                """,
                (
                    job_id,
                    adapter_id,
                    user_id,
                    now,
                    now,
                    num_events,
                    dataset_path,
                    pref_ids if pref_ids else None,
                    meta,
                ),
            ).fetchone()
        return TrainingJob(
            id=job_id,
            user_id=user_id,
            adapter_id=adapter_id,
            status=row.get("status", "queued") if row else "queued",
            num_events=row.get("num_events", num_events) if row else num_events,
            created_at=row.get("created_at", now) if row else now,
            updated_at=row.get("updated_at", now) if row else now,
            loss=row.get("loss"),
            preference_event_ids=row.get("preference_event_ids") or pref_ids,
            dataset_path=row.get("dataset_path", dataset_path),
            new_version=row.get("new_version"),
            meta=row.get("meta") if row else meta,
        )

    def update_training_job(
        self,
        job_id: str,
        *,
        status: str | None = None,
        loss: float | None = None,
        new_version: int | None = None,
        dataset_path: str | None = None,
        meta: dict | None = None,
    ) -> TrainingJob | None:
        existing = self.get_training_job(job_id)
        if not existing:
            return None
        new_updated_at = datetime.utcnow()
        with self._connect() as conn:
            row = conn.execute(
                """
                UPDATE training_job
                SET status = %s,
                    loss = %s,
                    new_version = %s,
                    dataset_path = %s,
                    meta = %s,
                    updated_at = %s
                WHERE id = %s
                RETURNING *
                """,
                (
                    status if status is not None else existing.status,
                    loss if loss is not None else existing.loss,
                    new_version if new_version is not None else existing.new_version,
                    dataset_path if dataset_path is not None else existing.dataset_path,
                    meta if meta is not None else existing.meta,
                    new_updated_at,
                    job_id,
                ),
            ).fetchone()
        if not row:
            return None
        return TrainingJob(
            id=str(row["id"]),
            user_id=str(row["user_id"]),
            adapter_id=str(row["adapter_artifact_id"]),
            status=row.get("status", existing.status),
            num_events=row.get("num_events", existing.num_events),
            created_at=row.get("created_at", existing.created_at),
            updated_at=row.get("updated_at", new_updated_at),
            loss=row.get("loss"),
            preference_event_ids=row.get("preference_event_ids") or [],
            dataset_path=row.get("dataset_path"),
            new_version=row.get("new_version"),
            meta=row.get("meta"),
        )

    def get_training_job(self, job_id: str) -> TrainingJob | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM training_job WHERE id = %s", (job_id,)).fetchone()
        if not row:
            return None
        return TrainingJob(
            id=str(row["id"]),
            user_id=str(row["user_id"]),
            adapter_id=str(row["adapter_artifact_id"]),
            status=row.get("status", "queued"),
            num_events=row.get("num_events"),
            created_at=row.get("created_at", datetime.utcnow()),
            updated_at=row.get("updated_at", datetime.utcnow()),
            loss=row.get("loss"),
            preference_event_ids=row.get("preference_event_ids") or [],
            dataset_path=row.get("dataset_path"),
            new_version=row.get("new_version"),
            meta=row.get("meta"),
        )

    # users
    def create_user(self, email: str, handle: Optional[str] = None) -> User:
        user_id = str(uuid.uuid4())
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO app_user (id, email, handle) VALUES (%s, %s, %s)",
                    (user_id, email, handle),
                )
        except errors.UniqueViolation:
            raise ConstraintViolation("email already exists", {"field": "email"})
        return User(id=user_id, email=email, handle=handle)

    def save_password(self, user_id: str, password_hash: str, password_algo: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO user_auth_credential (user_id, password_hash, password_algo, last_updated_at)
                VALUES (%s, %s, %s, now())
                ON CONFLICT (user_id) DO UPDATE
                SET password_hash = EXCLUDED.password_hash,
                    password_algo = EXCLUDED.password_algo,
                    last_updated_at = now()
                """,
                (user_id, password_hash, password_algo),
            )

    def get_password_record(self, user_id: str) -> Optional[tuple[str, str]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT password_hash, password_algo FROM user_auth_credential WHERE user_id = %s", (user_id,)
            ).fetchone()
        if not row:
            return None
        return str(row["password_hash"]), str(row["password_algo"])

    def set_user_mfa_secret(self, user_id: str, secret: str, enabled: bool = False) -> UserMFAConfig:
        record = UserMFAConfig(user_id=user_id, secret=secret, enabled=enabled)
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO user_mfa_secret (user_id, secret, enabled, created_at)
                    VALUES (%s, %s, %s, now())
                    ON CONFLICT (user_id) DO UPDATE SET secret = EXCLUDED.secret, enabled = EXCLUDED.enabled
                    """,
                    (user_id, secret, enabled),
                )
        except Exception as exc:
            self.logger.warning("set_user_mfa_secret_failed", error=str(exc))
        self.mfa_secrets[user_id] = record
        self._persist_training_state()
        return record

    def get_user_mfa_secret(self, user_id: str) -> Optional[UserMFAConfig]:
        if user_id in self.mfa_secrets:
            return self.mfa_secrets[user_id]
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT * FROM user_mfa_secret WHERE user_id = %s", (user_id,)
                ).fetchone()
            if row:
                cfg = UserMFAConfig(
                    user_id=row["user_id"],
                    secret=row["secret"],
                    enabled=bool(row.get("enabled", False)),
                    created_at=row.get("created_at", datetime.utcnow()),
                    meta=row.get("meta"),
                )
                self.mfa_secrets[user_id] = cfg
                return cfg
        except Exception as exc:
            self.logger.warning("get_user_mfa_secret_failed", error=str(exc))
            return None
        return None

    def get_user_by_email(self, email: str) -> Optional[User]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM app_user WHERE email = %s", (email,)).fetchone()
        if not row:
            return None
        return User(
            id=str(row["id"]),
            email=row["email"],
            handle=row.get("handle"),
            created_at=row.get("created_at", datetime.utcnow()),
            is_active=row.get("is_active", True),
            plan_tier=row.get("plan_tier", "free"),
            meta=row.get("meta"),
        )

    # sessions
    def create_session(
        self, user_id: str, ttl_minutes: int = 60 * 24, user_agent: str | None = None, ip_addr: str | None = None, *, mfa_required: bool = False
    ) -> Session:
        sess = Session.new(user_id=user_id, ttl_minutes=ttl_minutes, user_agent=user_agent, ip_addr=ip_addr, mfa_required=mfa_required)
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO auth_session (id, user_id, created_at, expires_at, user_agent, ip_addr, mfa_required, mfa_verified)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (sess.id, sess.user_id, sess.created_at, sess.expires_at, user_agent, ip_addr, mfa_required, sess.mfa_verified),
                )
        except errors.ForeignKeyViolation:
            raise ConstraintViolation("session user missing", {"user_id": user_id})
        return sess

    def revoke_session(self, session_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM auth_session WHERE id = %s", (session_id,))

    def mark_session_verified(self, session_id: str) -> None:
        try:
            with self._connect() as conn:
                conn.execute(
                    "UPDATE auth_session SET mfa_verified = TRUE, mfa_required = TRUE WHERE id = %s",
                    (session_id,),
                )
        except Exception as exc:
            self.logger.warning("mark_session_verified_failed", error=str(exc))
        sess = self.sessions.get(session_id)
        if sess:
            sess.mfa_verified = True
            sess.mfa_required = True

    def get_session(self, session_id: str) -> Optional[Session]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM auth_session WHERE id = %s", (session_id,)).fetchone()
        if not row:
            return None
        sess = Session(
            id=str(row["id"]),
            user_id=str(row["user_id"]),
            created_at=row.get("created_at", datetime.utcnow()),
            expires_at=row.get("expires_at", datetime.utcnow()),
            user_agent=row.get("user_agent"),
            ip_addr=row.get("ip_addr"),
            mfa_required=row.get("mfa_required", False),
            mfa_verified=row.get("mfa_verified", False),
            meta=row.get("meta"),
        )
        self.sessions[sess.id] = sess
        return sess

    # conversations
    def create_conversation(self, user_id: str, title: Optional[str] = None, active_context_id: Optional[str] = None) -> Conversation:
        conv_id = str(uuid.uuid4())
        now = datetime.utcnow()
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO conversation (id, user_id, title, created_at, updated_at, active_context_id) VALUES (%s, %s, %s, %s, %s, %s)",
                    (conv_id, user_id, title, now, now, active_context_id),
                )
        except errors.ForeignKeyViolation:
            raise ConstraintViolation("conversation owner or context missing", {"user_id": user_id, "context_id": active_context_id})
        return Conversation(id=conv_id, user_id=user_id, title=title, created_at=now, updated_at=now, active_context_id=active_context_id)

    def append_message(self, conversation_id: str, sender: str, role: str, content: str, meta: Optional[dict] = None) -> Message:
        try:
            with self._connect() as conn:
                seq_row = conn.execute("SELECT COUNT(*) AS c FROM message WHERE conversation_id = %s", (conversation_id,)).fetchone()
                seq = seq_row["c"] if seq_row else 0
                msg_id = str(uuid.uuid4())
                now = datetime.utcnow()
                conn.execute(
                    "INSERT INTO message (id, conversation_id, sender, role, content, seq, created_at, meta) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                    (msg_id, conversation_id, sender, role, content, seq, now, json.dumps(meta) if meta else None),
                )
                conn.execute("UPDATE conversation SET updated_at = %s WHERE id = %s", (now, conversation_id))
        except errors.ForeignKeyViolation:
            raise ConstraintViolation("conversation not found", {"conversation_id": conversation_id})
        return Message(id=msg_id, conversation_id=conversation_id, sender=sender, role=role, content=content, seq=seq, created_at=now, meta=meta)

    def list_messages(self, conversation_id: str, limit: int = 10) -> List[Message]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM message WHERE conversation_id = %s ORDER BY seq DESC LIMIT %s", (conversation_id, limit)
            ).fetchall()
        messages: List[Message] = []
        for row in reversed(rows):
            messages.append(
                Message(
                    id=str(row["id"]),
                    conversation_id=str(row["conversation_id"]),
                    sender=row["sender"],
                    role=row["role"],
                    content=row["content"],
                    seq=row["seq"],
                    created_at=row.get("created_at", datetime.utcnow()),
                    meta=row.get("meta"),
                )
            )
        return messages

    def list_conversations(self, user_id: str, limit: int = 20) -> List[Conversation]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM conversation
                WHERE user_id = %s
                ORDER BY updated_at DESC
                LIMIT %s
                """,
                (user_id, limit),
            ).fetchall()
        conversations: List[Conversation] = []
        for row in rows:
            conversations.append(
                Conversation(
                    id=str(row["id"]),
                    user_id=str(row["user_id"]),
                    created_at=row.get("created_at", datetime.utcnow()),
                    updated_at=row.get("updated_at", datetime.utcnow()),
                    title=row.get("title"),
                    status=row.get("status", "open"),
                    active_context_id=row.get("active_context_id"),
                    meta=row.get("meta"),
                )
            )
        return conversations

    # artifacts
    def list_artifacts(
        self, type_filter: Optional[str] = None, kind_filter: Optional[str] = None
    ) -> List[Artifact]:
        with self._connect() as conn:
            if type_filter and kind_filter:
                rows = conn.execute(
                    "SELECT * FROM artifact WHERE type = %s AND schema->>'kind' = %s",
                    (type_filter, kind_filter),
                ).fetchall()
            elif kind_filter:
                rows = conn.execute(
                    "SELECT * FROM artifact WHERE schema->>'kind' = %s", (kind_filter,)
                ).fetchall()
            elif type_filter:
                rows = conn.execute("SELECT * FROM artifact WHERE type = %s", (type_filter,)).fetchall()
            else:
                rows = conn.execute("SELECT * FROM artifact", ()).fetchall()
        artifacts: List[Artifact] = []
        for row in rows:
            artifacts.append(
                Artifact(
                    id=str(row["id"]),
                    type=row["type"],
                    name=row["name"],
                    description=row.get("description") or "",
                    schema=row.get("schema") or {},
                    owner_user_id=(str(row["owner_user_id"]) if row.get("owner_user_id") else None),
                    visibility=row.get("visibility", "private"),
                    created_at=row.get("created_at", datetime.utcnow()),
                    updated_at=row.get("updated_at", datetime.utcnow()),
                    fs_path=row.get("fs_path"),
                    meta=row.get("meta"),
                )
            )
        return artifacts

    def get_artifact(self, artifact_id: str) -> Optional[Artifact]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM artifact WHERE id = %s", (artifact_id,)).fetchone()
        if not row:
            return None
        schema = row.get("schema") if isinstance(row, dict) else row["schema"]
        if isinstance(schema, str):
            try:
                schema = json.loads(schema)
            except Exception as exc:
                self.logger.warning("artifact_schema_parse_failed", error=str(exc))
                schema = {}
        return Artifact(
            id=str(row["id"]),
            type=row["type"],
            name=row["name"],
            description=row.get("description") or "",
            schema=schema or {},
            owner_user_id=(str(row["owner_user_id"]) if row.get("owner_user_id") else None),
            visibility=row.get("visibility", "private"),
            created_at=row.get("created_at", datetime.utcnow()),
            updated_at=row.get("updated_at", datetime.utcnow()),
            fs_path=row.get("fs_path"),
            meta=row.get("meta"),
        )

    def create_artifact(
        self,
        type_: str,
        name: str,
        schema: dict,
        description: str = "",
        owner_user_id: Optional[str] = None,
        *,
        created_by: Optional[str] = None,
        change_note: Optional[str] = None,
    ) -> Artifact:
        try:
            validate_artifact(type_, schema)
        except ArtifactValidationError as exc:
            self.logger.warning("artifact_validation_failed", errors=exc.errors)
            raise
        artifact_id = str(uuid.uuid4())
        fs_path = self._persist_payload(artifact_id, 1, schema)
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO artifact (id, owner_user_id, type, name, description, schema, fs_path) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (artifact_id, owner_user_id, type_, name, description, json.dumps(schema), fs_path),
                )
                conn.execute(
                    "INSERT INTO artifact_version (artifact_id, version, schema, fs_path, created_by, change_note) VALUES (%s, %s, %s, %s, %s, %s)",
                    (artifact_id, 1, json.dumps(schema), fs_path, created_by or owner_user_id or "system_llm", change_note),
                )
        except errors.ForeignKeyViolation:
            raise ConstraintViolation("artifact owner missing", {"owner_user_id": owner_user_id})
        return Artifact(
            id=artifact_id,
            type=type_,
            name=name,
            description=description,
            schema=schema,
            owner_user_id=owner_user_id,
            fs_path=fs_path,
        )

    def update_artifact(
        self,
        artifact_id: str,
        schema: dict,
        description: Optional[str] = None,
        *,
        created_by: Optional[str] = None,
        change_note: Optional[str] = None,
    ) -> Optional[Artifact]:
        try:
            validate_artifact("workflow" if schema.get("kind") == "workflow.chat" else "tool" if schema.get("kind") == "tool.spec" else "artifact", schema)  # type: ignore[arg-type]
        except ArtifactValidationError as exc:
            self.logger.warning("artifact_validation_failed", errors=exc.errors)
            raise
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM artifact WHERE id = %s", (artifact_id,)).fetchone()
            if not row:
                return None
            versions = conn.execute("SELECT COALESCE(MAX(version), 0) AS v FROM artifact_version WHERE artifact_id = %s", (artifact_id,)).fetchone()
            next_version = (versions["v"] or 0) + 1
            fs_path = self._persist_payload(artifact_id, next_version, schema)
            conn.execute(
                "UPDATE artifact SET schema = %s, description = COALESCE(%s, description), updated_at = now(), fs_path = %s WHERE id = %s",
                (json.dumps(schema), description, fs_path, artifact_id),
            )
            conn.execute(
                "INSERT INTO artifact_version (artifact_id, version, schema, fs_path, created_by, change_note) VALUES (%s, %s, %s, %s, %s, %s)",
                (
                    artifact_id,
                    next_version,
                    json.dumps(schema),
                    fs_path,
                    created_by or (str(row["owner_user_id"]) if row.get("owner_user_id") else None) or "system_llm",
                    change_note,
                ),
            )
        return Artifact(
            id=str(row["id"]),
            type=row["type"],
            name=row["name"],
            description=description or row.get("description") or "",
            schema=schema,
            owner_user_id=(str(row["owner_user_id"]) if row.get("owner_user_id") else None),
            fs_path=fs_path,
            visibility=row.get("visibility", "private"),
        )

    def list_artifact_versions(self, artifact_id: str) -> List[ArtifactVersion]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM artifact_version WHERE artifact_id = %s ORDER BY version DESC",
                (artifact_id,),
            ).fetchall()
        versions: List[ArtifactVersion] = []
        for row in rows:
            schema = row.get("schema") if isinstance(row, dict) else row["schema"]
            if isinstance(schema, str):
                try:
                    schema = json.loads(schema)
                except Exception as exc:
                    self.logger.warning("artifact_version_schema_parse_failed", error=str(exc))
                    schema = {}
            versions.append(
                ArtifactVersion(
                    id=row["id"],
                    artifact_id=str(row["artifact_id"]),
                    version=row["version"],
                    schema=schema or {},
                    created_by=row.get("created_by", "system_llm"),
                    change_note=row.get("change_note"),
                    created_at=row.get("created_at", datetime.utcnow()),
                    fs_path=row.get("fs_path"),
                    meta=row.get("meta"),
                )
            )
        return versions

    def get_latest_workflow(self, workflow_id: str) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT schema FROM artifact_version WHERE artifact_id = %s ORDER BY version DESC LIMIT 1",
                (workflow_id,),
            ).fetchone()
        return row["schema"] if row else None

    def record_config_patch(self, artifact_id: str, proposer_user_id: Optional[str], patch: dict, justification: Optional[str]) -> ConfigPatchAudit:
        audit_id = str(uuid.uuid4())
        now = datetime.utcnow()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO config_patch (id, artifact_id, proposer_user_id, patch, justification, status, created_at, updated_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                (audit_id, artifact_id, proposer_user_id, json.dumps(patch), justification, "pending", now, now),
            )
        return ConfigPatchAudit(id=audit_id, artifact_id=artifact_id, proposer_user_id=proposer_user_id, patch=patch, justification=justification, created_at=now, updated_at=now)

    def get_config_patch(self, patch_id: str) -> Optional[ConfigPatchAudit]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM config_patch WHERE id = %s", (patch_id,)).fetchone()
        return self._config_patch_from_row(row) if row else None

    def list_config_patches(self, status: Optional[str] = None) -> List[ConfigPatchAudit]:
        query = "SELECT * FROM config_patch"
        params: tuple = ()
        if status:
            query += " WHERE status = %s"
            params = (status,)
        query += " ORDER BY created_at DESC"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._config_patch_from_row(row) for row in rows]

    def update_config_patch_status(
        self, patch_id: str, status: str, *, meta: Optional[Dict] = None, mark_decided: bool = False, mark_applied: bool = False
    ) -> Optional[ConfigPatchAudit]:
        with self._connect() as conn:
            existing = conn.execute("SELECT * FROM config_patch WHERE id = %s", (patch_id,)).fetchone()
            if not existing:
                return None
            now = datetime.utcnow()
            existing_meta = existing.get("meta") or {}
            if isinstance(existing_meta, str):
                try:
                    existing_meta = json.loads(existing_meta)
                except Exception as exc:
                    self.logger.warning("config_patch_meta_parse_failed", error=str(exc))
                    existing_meta = {}
            merged_meta: Dict = dict(existing_meta)
            if meta:
                merged_meta.update(meta)
            if mark_decided and "decided_at" not in merged_meta:
                merged_meta["decided_at"] = now.isoformat()
            if mark_applied:
                merged_meta["applied_at"] = now.isoformat()
            conn.execute(
                "UPDATE config_patch SET status = %s, updated_at = %s, meta = %s WHERE id = %s",
                (status, now, json.dumps(merged_meta), patch_id),
            )
            row = conn.execute("SELECT * FROM config_patch WHERE id = %s", (patch_id,)).fetchone()
        return self._config_patch_from_row(row) if row else None

    def _config_patch_from_row(self, row) -> ConfigPatchAudit:
        raw_patch = row.get("patch") if isinstance(row, dict) else row["patch"]
        patch_data = raw_patch if isinstance(raw_patch, dict) else json.loads(raw_patch or "{}")
        meta = row.get("meta") if isinstance(row, dict) else row.get("meta")
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception as exc:
                self.logger.warning("config_patch_meta_parse_failed", error=str(exc))
                meta = {}
        decided_at = None
        applied_at = None
        if isinstance(meta, dict):
            decided_at = self._parse_ts(meta.get("decided_at"))
            applied_at = self._parse_ts(meta.get("applied_at"))
        created = row.get("created_at") if isinstance(row, dict) else row["created_at"]
        updated = row.get("updated_at") if isinstance(row, dict) else row["updated_at"]
        return ConfigPatchAudit(
            id=str(row["id"]),
            artifact_id=str(row["artifact_id"]),
            proposer_user_id=(str(row["proposer_user_id"]) if row.get("proposer_user_id") else None) if isinstance(row, dict) else (str(row["proposer_user_id"]) if row["proposer_user_id"] else None),
            patch=patch_data,
            justification=row.get("justification") if isinstance(row, dict) else row["justification"],
            status=row.get("status", "pending") if isinstance(row, dict) else row["status"],
            created_at=created if isinstance(created, datetime) else datetime.fromisoformat(str(created)),
            updated_at=updated if isinstance(updated, datetime) else datetime.fromisoformat(str(updated)),
            decided_at=decided_at,
            applied_at=applied_at,
            meta=meta if isinstance(meta, dict) else {},
        )

    @staticmethod
    def _parse_ts(value: Optional[Any]) -> Optional[datetime]:
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                return None
        return None

    def get_runtime_config(self) -> dict:
        """Return deployment config sourced from SQL (placeholder until admin UI writes it).

        Replace this stub with typed reads from an ``instance_config`` table
        once config patches are persisted server-side, and enforce versioning
        plus source attribution (UI vs. drift detection) per SPEC §10.
        """

        return {}

    # knowledge
    def upsert_context(self, owner_user_id: Optional[str], name: str, description: str, fs_path: Optional[str] = None) -> KnowledgeContext:
        ctx_id = str(uuid.uuid4())
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO knowledge_context (id, owner_user_id, name, description, fs_path) VALUES (%s, %s, %s, %s, %s)",
                    (ctx_id, owner_user_id, name, description, fs_path),
                )
        except errors.ForeignKeyViolation:
            raise ConstraintViolation("context owner missing", {"owner_user_id": owner_user_id})
        return KnowledgeContext(id=ctx_id, owner_user_id=owner_user_id, name=name, description=description, fs_path=fs_path)

    def list_contexts(self) -> List[KnowledgeContext]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM knowledge_context ORDER BY created_at DESC", ()).fetchall()
        contexts: List[KnowledgeContext] = []
        for row in rows:
            contexts.append(
                KnowledgeContext(
                    id=str(row["id"]),
                    owner_user_id=str(row["owner_user_id"]) if row.get("owner_user_id") else None,
                    name=row["name"],
                    description=row["description"],
                    created_at=row.get("created_at", datetime.utcnow()),
                    updated_at=row.get("updated_at", datetime.utcnow()),
                    fs_path=row.get("fs_path"),
                    meta=row.get("meta"),
                )
            )
        return contexts

    def add_chunks(self, context_id: str, chunks: Iterable[KnowledgeChunk]) -> None:
        try:
            with self._connect() as conn:
                for chunk in chunks:
                    conn.execute(
                        "INSERT INTO knowledge_chunk (id, context_id, text, embedding, seq, created_at, meta) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                        (chunk.id, context_id, chunk.text, chunk.embedding, chunk.seq, chunk.created_at, json.dumps(chunk.meta) if chunk.meta else None),
                    )
        except errors.ForeignKeyViolation:
            raise ConstraintViolation("context not found", {"context_id": context_id})

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

        with self._connect() as conn:
            if context_id:
                rows = conn.execute("SELECT * FROM knowledge_chunk WHERE context_id = %s", (context_id,)).fetchall()
            else:
                sample_size = max(limit * 5, 20)
                rows = conn.execute("SELECT * FROM knowledge_chunk ORDER BY created_at DESC LIMIT %s", (sample_size,)).fetchall()
        results: List[KnowledgeChunk] = []
        for row in rows:
            results.append(
                KnowledgeChunk(
                    id=str(row["id"]),
                    context_id=str(row["context_id"]),
                    text=row["text"],
                    embedding=row.get("embedding") or [],
                    seq=row.get("seq", 0),
                    created_at=row.get("created_at", datetime.utcnow()),
                    meta=row.get("meta"),
                )
            )
        if query_embedding:
            results.sort(key=lambda ch: _cosine(query_embedding, ch.embedding), reverse=True)
        elif context_id:
            results.sort(key=lambda ch: ch.seq)
        else:
            results.sort(key=lambda ch: ch.created_at, reverse=True)
        return results[:limit]

    def _persist_payload(self, artifact_id: str, version: int, schema: dict) -> str:
        artifact_dir = self.fs_root / "artifacts" / artifact_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        path = artifact_dir / f"v{version}.json"
        path.write_text(json.dumps(schema, indent=2))
        return str(path)

    def _training_state_path(self) -> Path:
        state_dir = self.fs_root / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        return state_dir / "training_pg.json"

    def _persist_training_state(self) -> None:
        data = {
            "mfa_secrets": [
                {
                    "user_id": cfg.user_id,
                    "secret": cfg.secret,
                    "enabled": cfg.enabled,
                    "created_at": cfg.created_at.isoformat(),
                    "meta": cfg.meta,
                }
                for cfg in self.mfa_secrets.values()
            ],
        }
        self._training_state_path().write_text(json.dumps(data, indent=2))

    def _load_training_state(self) -> None:
        path = self._training_state_path()
        if not path.exists():
            return
        raw = json.loads(path.read_text())
        self.mfa_secrets = {
            cfg["user_id"]: UserMFAConfig(
                user_id=cfg["user_id"],
                secret=cfg["secret"],
                enabled=bool(cfg.get("enabled", False)),
                created_at=datetime.fromisoformat(cfg.get("created_at", datetime.utcnow().isoformat())),
                meta=cfg.get("meta"),
            )
            for cfg in raw.get("mfa_secrets", [])
        }
