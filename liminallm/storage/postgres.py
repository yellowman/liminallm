from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

import psycopg
from psycopg import errors
from psycopg.rows import dict_row

from liminallm.storage.errors import ConstraintViolation
from liminallm.storage.models import (
    Artifact,
    ConfigPatchAudit,
    Conversation,
    KnowledgeChunk,
    KnowledgeContext,
    Message,
    PreferenceEvent,
    Session,
    TrainingJob,
    User,
)


class PostgresStore:
    """Thin Postgres-backed store to persist kernel primitives."""

    def __init__(self, dsn: str, fs_root: str) -> None:
        self.dsn = dsn
        self.fs_root = Path(fs_root)
        self.fs_root.mkdir(parents=True, exist_ok=True)
        self.preference_events: dict[str, PreferenceEvent] = {}
        self.training_jobs: dict[str, TrainingJob] = {}
        self._load_training_state()

    def _connect(self):
        return psycopg.connect(self.dsn, autocommit=True, row_factory=dict_row)

    # preference events (filesystem-backed placeholder until full tables exist)
    def record_preference_event(
        self,
        user_id: str,
        conversation_id: str,
        message_id: str,
        feedback: str,
        *,
        score: float | None = None,
        corrected_text: str | None = None,
        weight: float | None = None,
        context_embedding: list[float] | None = None,
        cluster_id: str | None = None,
        context_text: str | None = None,
        meta: dict | None = None,
    ) -> PreferenceEvent:
        normalized_weight = weight if weight is not None else (score if score is not None else 1.0)
        event = PreferenceEvent(
            id=str(uuid.uuid4()),
            user_id=user_id,
            conversation_id=conversation_id,
            message_id=message_id,
            feedback=feedback,
            score=score,
            context_embedding=context_embedding or [],
            cluster_id=cluster_id,
            context_text=context_text,
            corrected_text=corrected_text,
            weight=normalized_weight,
            meta=meta,
        )
        self.preference_events[event.id] = event
        self._persist_training_state()
        return event

    def list_preference_events(self, user_id: str | None = None, feedback: str | None = None) -> list[PreferenceEvent]:
        events = list(self.preference_events.values())
        if user_id:
            events = [e for e in events if e.user_id == user_id]
        if feedback:
            events = [e for e in events if e.feedback == feedback]
        return sorted(events, key=lambda e: e.created_at)

    def create_training_job(
        self,
        user_id: str,
        adapter_id: str,
        preference_event_ids: list[str] | None = None,
        dataset_path: str | None = None,
    ) -> TrainingJob:
        job = TrainingJob(
            id=str(uuid.uuid4()),
            user_id=user_id,
            adapter_id=adapter_id,
            preference_event_ids=preference_event_ids or [],
            dataset_path=dataset_path,
        )
        self.training_jobs[job.id] = job
        self._persist_training_state()
        return job

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
        self._persist_training_state()
        return job

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
    def create_session(self, user_id: str, ttl_minutes: int = 60 * 24, user_agent: str | None = None, ip_addr: str | None = None) -> Session:
        sess = Session.new(user_id=user_id, ttl_minutes=ttl_minutes, user_agent=user_agent, ip_addr=ip_addr)
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO auth_session (id, user_id, created_at, expires_at, user_agent, ip_addr) VALUES (%s, %s, %s, %s, %s, %s)",
                    (sess.id, sess.user_id, sess.created_at, sess.expires_at, user_agent, ip_addr),
                )
        except errors.ForeignKeyViolation:
            raise ConstraintViolation("session user missing", {"user_id": user_id})
        return sess

    def revoke_session(self, session_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM auth_session WHERE id = %s", (session_id,))

    def get_session(self, session_id: str) -> Optional[Session]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM auth_session WHERE id = %s", (session_id,)).fetchone()
        if not row:
            return None
        return Session(
            id=str(row["id"]),
            user_id=str(row["user_id"]),
            created_at=row.get("created_at", datetime.utcnow()),
            expires_at=row.get("expires_at", datetime.utcnow()),
            user_agent=row.get("user_agent"),
            ip_addr=row.get("ip_addr"),
            meta=row.get("meta"),
        )

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
    def list_artifacts(self, type_filter: Optional[str] = None) -> List[Artifact]:
        with self._connect() as conn:
            if type_filter:
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

    def create_artifact(self, type_: str, name: str, schema: dict, description: str = "", owner_user_id: Optional[str] = None) -> Artifact:
        artifact_id = str(uuid.uuid4())
        fs_path = self._persist_payload(artifact_id, 1, schema)
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO artifact (id, owner_user_id, type, name, description, schema, fs_path) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (artifact_id, owner_user_id, type_, name, description, json.dumps(schema), fs_path),
                )
                conn.execute(
                    "INSERT INTO artifact_version (artifact_id, version, schema, fs_path) VALUES (%s, %s, %s, %s)",
                    (artifact_id, 1, json.dumps(schema), fs_path),
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

    def update_artifact(self, artifact_id: str, schema: dict, description: Optional[str] = None) -> Optional[Artifact]:
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
                "INSERT INTO artifact_version (artifact_id, version, schema, fs_path) VALUES (%s, %s, %s, %s)",
                (artifact_id, next_version, json.dumps(schema), fs_path),
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

    def get_runtime_config(self) -> dict:
        """Return deployment config sourced from SQL (placeholder until admin UI writes it)."""

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
            "preference_events": [
                {
                    "id": e.id,
                    "user_id": e.user_id,
                    "conversation_id": e.conversation_id,
                    "message_id": e.message_id,
                    "feedback": e.feedback,
                    "score": e.score,
                    "context_embedding": e.context_embedding,
                    "cluster_id": e.cluster_id,
                    "context_text": e.context_text,
                    "corrected_text": e.corrected_text,
                    "created_at": e.created_at.isoformat(),
                    "weight": e.weight,
                    "meta": e.meta,
                }
                for e in self.preference_events.values()
            ],
            "training_jobs": [
                {
                    "id": j.id,
                    "user_id": j.user_id,
                    "adapter_id": j.adapter_id,
                    "status": j.status,
                    "created_at": j.created_at.isoformat(),
                    "updated_at": j.updated_at.isoformat(),
                    "loss": j.loss,
                    "preference_event_ids": j.preference_event_ids,
                    "dataset_path": j.dataset_path,
                    "new_version": j.new_version,
                    "meta": j.meta,
                }
                for j in self.training_jobs.values()
            ],
        }
        self._training_state_path().write_text(json.dumps(data, indent=2))

    def _load_training_state(self) -> None:
        path = self._training_state_path()
        if not path.exists():
            return
        raw = json.loads(path.read_text())
        self.preference_events = {
            e["id"]: PreferenceEvent(
                id=e["id"],
                user_id=e["user_id"],
                conversation_id=e["conversation_id"],
                message_id=e["message_id"],
                feedback=e["feedback"],
                score=e.get("score"),
                context_embedding=e.get("context_embedding", []),
                cluster_id=e.get("cluster_id"),
                context_text=e.get("context_text"),
                corrected_text=e.get("corrected_text"),
                created_at=datetime.fromisoformat(e["created_at"]),
                weight=float(e.get("weight", 1.0)),
                meta=e.get("meta"),
            )
            for e in raw.get("preference_events", [])
        }
        self.training_jobs = {
            j["id"]: TrainingJob(
                id=j["id"],
                user_id=j["user_id"],
                adapter_id=j["adapter_id"],
                status=j.get("status", "pending"),
                created_at=datetime.fromisoformat(j["created_at"]),
                updated_at=datetime.fromisoformat(j.get("updated_at", j["created_at"])),
                loss=j.get("loss"),
                preference_event_ids=list(j.get("preference_event_ids", [])),
                dataset_path=j.get("dataset_path"),
                new_version=j.get("new_version"),
                meta=j.get("meta"),
            )
            for j in raw.get("training_jobs", [])
        }
