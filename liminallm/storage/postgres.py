from __future__ import annotations

import hashlib
import json
import threading
import uuid
from datetime import datetime
from pathlib import Path
from ipaddress import ip_address
from typing import Any, Dict, Iterable, List, Optional, Sequence

from psycopg import errors
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

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
from liminallm.storage.errors import ConstraintViolation
from liminallm.storage.models import (
    Artifact,
    ArtifactVersion,
    ConfigPatchAudit,
    ContextSource,
    Conversation,
    AdapterRouterState,
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


_MAX_SESSION_CACHE_SIZE = 10000


class PostgresStore:
    """Thin Postgres-backed store to persist kernel primitives."""

    def __init__(self, dsn: str, fs_root: str) -> None:
        self.dsn = dsn
        self.fs_root = Path(fs_root)
        self.fs_root.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(__name__)
        self.pool = ConnectionPool(
            self.dsn,
            min_size=2,
            max_size=10,
            kwargs={"row_factory": dict_row, "autocommit": False},
        )
        self.sessions: dict[str, Session] = {}
        self._session_lock = threading.Lock()
        self._ensure_runtime_config_table()
        self._verify_required_schema()
        self._load_training_state()
        self._ensure_default_artifacts()

    def _cache_session(self, session: Session) -> Session:
        """Store session in the in-memory cache and return it.

        Thread-safe per SPEC §18 inference/adapter cache discipline.
        """
        with self._session_lock:
            # Evict soonest-to-expire entries if cache is at capacity
            if len(self.sessions) >= _MAX_SESSION_CACHE_SIZE:
                # Remove ~10% of entries closest to expiration
                sorted_sessions = sorted(
                    self.sessions.values(),
                    key=lambda s: s.expires_at if s.expires_at else datetime.min,
                )
                evict_count = max(1, _MAX_SESSION_CACHE_SIZE // 10)
                for old_session in sorted_sessions[:evict_count]:
                    self.sessions.pop(old_session.id, None)

            self.sessions[session.id] = session
            return session

    def _evict_session(self, session_id: str) -> None:
        """Remove a session from the in-memory cache if present.

        Thread-safe per SPEC §18.
        """
        with self._session_lock:
            self.sessions.pop(session_id, None)

    def _update_cached_session(self, session_id: str, **updates: Any) -> None:
        """Apply field updates to a cached session if it exists.

        Thread-safe per SPEC §18.
        """
        with self._session_lock:
            sess = self.sessions.get(session_id)
            if not sess:
                return
            for field, value in updates.items():
                setattr(sess, field, value)
            self.sessions[session_id] = sess

    def _connect(self):
        return self.pool.connection()

    def _ensure_runtime_config_table(self) -> None:
        """Create the ``instance_config`` table if it is missing."""

        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS instance_config (
                    name TEXT PRIMARY KEY,
                    config JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )

    def _verify_required_schema(self) -> None:
        """Ensure core tables and pgvector expectations exist before serving requests."""

        required_tables = [
            "app_user",
            "user_auth_credential",
            "user_auth_provider",
            "user_settings",
            "auth_session",
            "conversation",
            "message",
            "artifact",
            "artifact_version",
            "config_patch",
            "knowledge_context",
            "context_source",
            "knowledge_chunk",
            "preference_event",
            "semantic_cluster",
            "adapter_router_state",
            "training_job",
            "user_mfa_secret",
            "instance_config",
        ]

        with self._connect() as conn:
            missing_tables = []
            for table in required_tables:
                row = conn.execute(
                    "SELECT to_regclass(%s) AS oid", (f"public.{table}",)
                ).fetchone()
                if not row or not row.get("oid"):
                    missing_tables.append(table)

            if missing_tables:
                raise RuntimeError(
                    "Missing required Postgres tables: {}. Run scripts/migrate.sh to install the SPEC §2 schema.".format(
                        ", ".join(sorted(missing_tables))
                    )
                )

            vector_ext = conn.execute(
                "SELECT extname FROM pg_extension WHERE extname = 'vector'"
            ).fetchone()
            if not vector_ext:
                raise RuntimeError(
                    "pgvector extension is missing. Install it and rerun scripts/migrate.sh to satisfy SPEC §3 RAG requirements."
                )

            citext_ext = conn.execute(
                "SELECT extname FROM pg_extension WHERE extname = 'citext'"
            ).fetchone()
            if not citext_ext:
                raise RuntimeError(
                    "citext extension is missing. Install it and rerun scripts/migrate.sh to satisfy SPEC §2 auth expectations."
                )

            embedding_col = conn.execute(
                """
                SELECT udt_name, data_type
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = 'knowledge_chunk' AND column_name = 'embedding'
                """
            ).fetchone()
            if not embedding_col or embedding_col.get("udt_name") != "vector":
                raise RuntimeError(
                    "knowledge_chunk.embedding must be a pgvector column; run migrations to align with SPEC §§2–3."
                )

            embedding_index = conn.execute(
                """
                SELECT i.relname AS index_name, am.amname AS access_method
                FROM pg_index idx
                JOIN pg_class i ON i.oid = idx.indexrelid
                JOIN pg_class t ON t.oid = idx.indrelid
                JOIN pg_am am ON i.relam = am.oid
                WHERE t.relname = 'knowledge_chunk' AND i.relname = 'knowledge_chunk_embedding_idx'
                """
            ).fetchone()
            if not embedding_index or embedding_index.get("access_method") != "ivfflat":
                raise RuntimeError(
                    "knowledge_chunk_embedding_idx (ivfflat) is missing. Run scripts/migrate.sh to install pgvector indices."
                )

            context_index = conn.execute(
                """
                SELECT i.relname AS index_name
                FROM pg_index idx
                JOIN pg_class i ON i.oid = idx.indexrelid
                JOIN pg_class t ON t.oid = idx.indrelid
                WHERE t.relname = 'knowledge_chunk' AND i.relname = 'knowledge_chunk_context_idx'
                """
            ).fetchone()
            if not context_index:
                raise RuntimeError(
                    "knowledge_chunk_context_idx is missing. Run scripts/migrate.sh to align with SPEC §2 context lookups."
                )

    def _set_runtime_config(self, config: dict) -> None:
        """Persist the runtime configuration for admin-driven overrides."""

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO instance_config (name, config, created_at, updated_at)
                VALUES (%s, %s, now(), now())
                ON CONFLICT (name) DO UPDATE SET config = EXCLUDED.config, updated_at = EXCLUDED.updated_at
                """,
                ("default", json.dumps(config)),
            )

    def _ensure_default_artifacts(self) -> None:
        """Seed the default workflow artifact if the database is empty."""

        existing = self.list_artifacts()
        if any(artifact.name == "default_chat_workflow" for artifact in existing):
            seeded_workflow = True
        else:
            seeded_workflow = False

        if not seeded_workflow:
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
                version_author="system_llm",
                change_note="Seeded default workflow",
            )

        seeded_tools = {
            art.schema.get("name")
            for art in existing
            if isinstance(art.schema, dict) and art.schema.get("kind") == "tool.spec"
        }
        default_tools = [
            {
                "kind": "tool.spec",
                "name": "llm.generic",
                "description": "Plain chat response from the base model.",
                "inputs": {"message": {"type": "string"}},
                "handler": "llm.generic",
            },
            {
                "kind": "tool.spec",
                "name": "rag.answer_with_context_v1",
                "description": "Retrieval augmented answer with pgvector context.",
                "inputs": {
                    "message": {"type": "string"},
                    "context_id": {"type": "string", "optional": True},
                },
                "handler": "rag.answer_with_context_v1",
            },
        ]
        for spec in default_tools:
            if spec["name"] in seeded_tools:
                continue
            self.create_artifact(
                "tool",
                spec["name"],
                spec,
                spec.get("description", ""),
                version_author="system_llm",
                change_note="Seeded default tool spec",
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
        normalized_weight = weight if weight is not None else 1.0
        event_id = str(uuid.uuid4())
        with self._connect() as conn:
            msg_row = conn.execute(
                "SELECT conversation_id, content FROM message WHERE id = %s",
                (message_id,),
            ).fetchone()
            if not msg_row:
                raise ConstraintViolation(
                    "preference message missing", {"message_id": message_id}
                )
            if msg_row.get("conversation_id") != conversation_id:
                raise ConstraintViolation(
                    "preference message conversation mismatch",
                    {"message_id": message_id, "conversation_id": conversation_id},
                )
            embedding = context_embedding or self._text_embedding(
                context_text or msg_row.get("content")
            )
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
                    self._format_vector(embedding),
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
            context_embedding=embedding,
            cluster_id=cluster_id,
            context_text=context_text,
            corrected_text=corrected_text,
            created_at=(
                row.get("created_at", datetime.utcnow()) if row else datetime.utcnow()
            ),
            weight=normalized_weight,
            meta=meta,
        )

    def _text_embedding(self, text: Optional[str]) -> list[float]:
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

    def list_preference_events(
        self,
        user_id: str | None = None,
        feedback: Iterable[str] | str | None = None,
        cluster_id: str | None = None,
        *,
        tenant_id: str | None = None,
    ) -> list[PreferenceEvent]:
        clauses = []
        params: list[Any] = []
        if user_id:
            clauses.append("user_id = %s")
            params.append(user_id)
        if feedback:
            feedback_values = (
                [feedback] if isinstance(feedback, str) else list(feedback)
            )
            if feedback_values:
                placeholders = ", ".join(["%s"] * len(feedback_values))
                clauses.append(f"feedback IN ({placeholders})")
                params.extend(feedback_values)
        if cluster_id:
            clauses.append("cluster_id = %s")
            params.append(cluster_id)
        if tenant_id:
            clauses.append("user_id IN (SELECT id FROM app_user WHERE tenant_id = %s)")
            params.append(tenant_id)
        query = "SELECT * FROM preference_event"
        if clauses:
            query = " ".join([query, "WHERE", " AND ".join(clauses)])
        query = " ".join([query, "ORDER BY created_at"])
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
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

    def update_preference_event(
        self, event_id: str, *, cluster_id: str | None = None
    ) -> PreferenceEvent | None:
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
            row = conn.execute(
                "SELECT * FROM preference_event WHERE id = %s", (event_id,)
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
            sample_message_ids=(
                row.get("sample_message_ids") if row else cluster.sample_message_ids
            ),
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

    def list_semantic_clusters(
        self, user_id: str | None = None
    ) -> list[SemanticCluster]:
        with self._connect() as conn:
            if user_id:
                rows = conn.execute(
                    "SELECT * FROM semantic_cluster WHERE user_id = %s ORDER BY updated_at DESC",
                    (user_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM semantic_cluster ORDER BY updated_at DESC", ()
                ).fetchall()
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
            row = conn.execute(
                "SELECT * FROM semantic_cluster WHERE id = %s", (cluster_id,)
            ).fetchone()
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
        columns = (
            "id, adapter_id, user_id, created_at, updated_at, status, num_events, loss, "
            "dataset_path, new_version, preference_event_ids, meta"
        )
        placeholders = "%s, %s, %s, %s, %s, 'queued', %s, NULL, %s, NULL, %s, %s"
        with self._connect() as conn:
            row = conn.execute(
                f"""
                INSERT INTO training_job ({columns})
                VALUES ({placeholders})
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
        adapter_id_value = row.get("adapter_id") or existing.adapter_id
        return TrainingJob(
            id=str(row["id"]),
            user_id=str(row["user_id"]),
            adapter_id=(
                str(adapter_id_value) if adapter_id_value else existing.adapter_id
            ),
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
            row = conn.execute(
                "SELECT * FROM training_job WHERE id = %s", (job_id,)
            ).fetchone()
        if not row:
            return None
        return TrainingJob(
            id=str(row["id"]),
            user_id=str(row["user_id"]),
            adapter_id=self._require_training_adapter_id(
                row.get("adapter_id"), row.get("id")
            ),
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

    def list_training_jobs(
        self,
        user_id: str | None = None,
        status: str | None = None,
        *,
        limit: int | None = None,
        tenant_id: str | None = None,
    ) -> List[TrainingJob]:
        query = "SELECT * FROM training_job WHERE 1=1"
        params: list[Any] = []
        if user_id:
            params.append(user_id)
            query += " AND user_id = %s"
        if status:
            params.append(status)
            query += " AND status = %s"
        if tenant_id:
            params.append(tenant_id)
            query += " AND user_id IN (SELECT id FROM app_user WHERE tenant_id = %s)"
        query += " ORDER BY COALESCE(updated_at, created_at) DESC"
        if limit:
            params.append(limit)
            query += " LIMIT %s"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [
            TrainingJob(
                id=str(row["id"]),
                user_id=str(row["user_id"]),
                adapter_id=self._require_training_adapter_id(
                    row.get("adapter_id"), row.get("id")
                ),
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
            for row in rows
        ]

    @staticmethod
    def _require_training_adapter_id(adapter_id: Any, job_id: Any) -> str:
        if adapter_id is None:
            raise ValueError(f"training_job {job_id} is missing adapter_id")
        return str(adapter_id)

    # users
    def create_user(
        self,
        email: str,
        handle: Optional[str] = None,
        *,
        tenant_id: str = "public",
        role: str = "user",
        plan_tier: str = "free",
        is_active: bool = True,
        meta: Optional[dict] = None,
    ) -> User:
        user_id = str(uuid.uuid4())
        normalized_meta = meta.copy() if meta else {}
        normalized_meta.setdefault("email_verified", False)
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO app_user (id, email, handle, tenant_id, role, plan_tier, is_active, meta)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        user_id,
                        email,
                        handle,
                        tenant_id,
                        role,
                        plan_tier,
                        is_active,
                        json.dumps(normalized_meta) if normalized_meta else None,
                    ),
                )
        except errors.UniqueViolation:
            raise ConstraintViolation("email already exists", {"field": "email"})
        return User(
            id=user_id,
            email=email,
            handle=handle,
            tenant_id=tenant_id,
            role=role,
            plan_tier=plan_tier,
            is_active=is_active,
            meta=normalized_meta,
        )

    def link_user_auth_provider(
        self, user_id: str, provider: str, provider_uid: str
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO user_auth_provider (user_id, provider, provider_uid)
                VALUES (%s, %s, %s)
                ON CONFLICT (provider, provider_uid) DO NOTHING
                """,
                (user_id, provider, provider_uid),
            )

    def get_user_by_provider(self, provider: str, provider_uid: str) -> Optional[User]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT u.* FROM user_auth_provider p JOIN app_user u ON u.id = p.user_id WHERE p.provider = %s AND p.provider_uid = %s",
                (provider, provider_uid),
            ).fetchone()
        if not row:
            return None
        return User(
            id=str(row["id"]),
            email=row["email"],
            handle=row.get("handle"),
            created_at=row.get("created_at", datetime.utcnow()),
            is_active=row.get("is_active", True),
            plan_tier=row.get("plan_tier", "free"),
            role=row.get("role", "user"),
            tenant_id=row.get("tenant_id", "public"),
            meta=row.get("meta"),
        )

    def save_password(
        self, user_id: str, password_hash: str, password_algo: str
    ) -> None:
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
                "SELECT password_hash, password_algo FROM user_auth_credential WHERE user_id = %s",
                (user_id,),
            ).fetchone()
        if not row:
            return None
        return str(row["password_hash"]), str(row["password_algo"])

    def set_user_mfa_secret(
        self, user_id: str, secret: str, enabled: bool = False
    ) -> UserMFAConfig:
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
            raise
        return record

    def get_user_mfa_secret(self, user_id: str) -> Optional[UserMFAConfig]:
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
                return cfg
        except Exception as exc:
            self.logger.warning("get_user_mfa_secret_failed", error=str(exc))
            return None
        return None

    def get_user_by_email(self, email: str) -> Optional[User]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM app_user WHERE email = %s", (email,)
            ).fetchone()
        if not row:
            return None
        return User(
            id=str(row["id"]),
            email=row["email"],
            handle=row.get("handle"),
            created_at=row.get("created_at", datetime.utcnow()),
            is_active=row.get("is_active", True),
            plan_tier=row.get("plan_tier", "free"),
            role=row.get("role", "user"),
            tenant_id=row.get("tenant_id", "public"),
            meta=row.get("meta"),
        )

    def get_user(self, user_id: str) -> Optional[User]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM app_user WHERE id = %s", (user_id,)
            ).fetchone()
        if not row:
            return None
        return User(
            id=str(row["id"]),
            email=row["email"],
            handle=row.get("handle"),
            created_at=row.get("created_at", datetime.utcnow()),
            is_active=row.get("is_active", True),
            plan_tier=row.get("plan_tier", "free"),
            role=row.get("role", "user"),
            tenant_id=row.get("tenant_id", "public"),
            meta=row.get("meta"),
        )

    def list_users(
        self, tenant_id: Optional[str] = None, limit: int = 100
    ) -> List[User]:
        with self._connect() as conn:
            if tenant_id:
                rows = conn.execute(
                    "SELECT * FROM app_user WHERE tenant_id = %s ORDER BY created_at DESC LIMIT %s",
                    (tenant_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM app_user ORDER BY created_at DESC LIMIT %s", (limit,)
                ).fetchall()
        users: List[User] = []
        for row in rows:
            users.append(
                User(
                    id=str(row["id"]),
                    email=row["email"],
                    handle=row.get("handle"),
                    created_at=row.get("created_at", datetime.utcnow()),
                    is_active=row.get("is_active", True),
                    plan_tier=row.get("plan_tier", "free"),
                    role=row.get("role", "user"),
                    tenant_id=row.get("tenant_id", "public"),
                    meta=row.get("meta"),
                )
            )
        return users

    def update_user_role(self, user_id: str, role: str) -> Optional[User]:
        with self._connect() as conn:
            row = conn.execute(
                "UPDATE app_user SET role = %s, updated_at = now() WHERE id = %s RETURNING *",
                (role, user_id),
            ).fetchone()
        if not row:
            return None
        return User(
            id=str(row["id"]),
            email=row["email"],
            handle=row.get("handle"),
            created_at=row.get("created_at", datetime.utcnow()),
            is_active=row.get("is_active", True),
            plan_tier=row.get("plan_tier", "free"),
            role=row.get("role", "user"),
            tenant_id=row.get("tenant_id", "public"),
            meta=row.get("meta"),
        )

    def mark_email_verified(self, user_id: str) -> Optional[User]:
        with self._connect() as conn:
            row = conn.execute(
                """
                UPDATE app_user
                SET meta = jsonb_set(COALESCE(meta, '{}'::jsonb), '{email_verified}', 'true', true),
                    updated_at = now()
                WHERE id = %s
                RETURNING *
                """,
                (user_id,),
            ).fetchone()
        if not row:
            return None
        return User(
            id=str(row["id"]),
            email=row["email"],
            handle=row.get("handle"),
            created_at=row.get("created_at", datetime.utcnow()),
            is_active=row.get("is_active", True),
            plan_tier=row.get("plan_tier", "free"),
            role=row.get("role", "user"),
            tenant_id=row.get("tenant_id", "public"),
            meta=row.get("meta"),
        )

    def delete_user(self, user_id: str) -> bool:
        with self._connect() as conn:
            result = conn.execute("DELETE FROM app_user WHERE id = %s", (user_id,))
            return result.rowcount > 0

    # sessions
    def create_session(
        self,
        user_id: str,
        ttl_minutes: int = 60 * 24,
        user_agent: str | None = None,
        ip_addr: str | None = None,
        *,
        mfa_required: bool = False,
        tenant_id: str = "public",
        meta: Optional[dict] = None,
    ) -> Session:
        sess = Session.new(
            user_id=user_id,
            ttl_minutes=ttl_minutes,
            user_agent=user_agent,
            ip_addr=ip_addr,
            mfa_required=mfa_required,
            tenant_id=tenant_id,
            meta=meta,
        )
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO auth_session (id, user_id, tenant_id, created_at, expires_at, user_agent, ip_addr, mfa_required, mfa_verified, meta)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        sess.id,
                        sess.user_id,
                        tenant_id,
                        sess.created_at,
                        sess.expires_at,
                        user_agent,
                        str(sess.ip_addr) if sess.ip_addr is not None else None,
                        mfa_required,
                        sess.mfa_verified,
                        json.dumps(meta) if meta else None,
                    ),
                )
        except errors.ForeignKeyViolation:
            raise ConstraintViolation("session user missing", {"user_id": user_id})
        return self._cache_session(sess)

    def revoke_session(self, session_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM auth_session WHERE id = %s", (session_id,))
        self._evict_session(session_id)

    def revoke_user_sessions(self, user_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM auth_session WHERE user_id = %s", (user_id,))
        # Thread-safe iteration over session cache per SPEC §18
        with self._session_lock:
            stale_ids = [
                sid for sid, sess in self.sessions.items() if sess.user_id == user_id
            ]
            for sid in stale_ids:
                self.sessions.pop(sid, None)

    def mark_session_verified(self, session_id: str) -> None:
        try:
            with self._connect() as conn:
                conn.execute(
                    "UPDATE auth_session SET mfa_verified = TRUE WHERE id = %s",
                    (session_id,),
                )
        except Exception as exc:
            self.logger.warning("mark_session_verified_failed", error=str(exc))
        self._update_cached_session(session_id, mfa_verified=True)

    def get_session(self, session_id: str) -> Optional[Session]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM auth_session WHERE id = %s", (session_id,)
            ).fetchone()
        if not row:
            return None
        meta = row.get("meta")
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = None
        raw_ip = row.get("ip_addr")
        ip_val = None
        if isinstance(raw_ip, str):
            if raw_ip.strip():
                ip_val = ip_address(raw_ip)
        else:
            ip_val = raw_ip
        sess = Session(
            id=str(row["id"]),
            user_id=str(row["user_id"]),
            created_at=row.get("created_at", datetime.utcnow()),
            expires_at=row.get("expires_at", datetime.utcnow()),
            user_agent=row.get("user_agent"),
            ip_addr=ip_val,
            mfa_required=row.get("mfa_required", False),
            mfa_verified=row.get("mfa_verified", False),
            tenant_id=row.get("tenant_id", "public"),
            meta=meta,
            allow_expired=True,
        )
        return self._cache_session(sess)

    def set_session_meta(self, session_id: str, meta: dict) -> None:
        if not isinstance(meta, dict):
            raise ValueError("session meta must be a dictionary")
        try:
            serialized_meta = json.dumps(meta)
        except TypeError as exc:
            raise ValueError("session meta must be JSON serializable") from exc
        with self._connect() as conn:
            conn.execute(
                "UPDATE auth_session SET meta = %s WHERE id = %s",
                (serialized_meta, session_id),
            )
        self._update_cached_session(session_id, meta=meta)

    # conversations
    def create_conversation(
        self,
        user_id: str,
        title: Optional[str] = None,
        active_context_id: Optional[str] = None,
    ) -> Conversation:
        conv_id = str(uuid.uuid4())
        now = datetime.utcnow()
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO conversation (id, user_id, title, created_at, updated_at, active_context_id) VALUES (%s, %s, %s, %s, %s, %s)",
                    (conv_id, user_id, title, now, now, active_context_id),
                )
        except errors.ForeignKeyViolation:
            raise ConstraintViolation(
                "conversation owner or context missing",
                {"user_id": user_id, "context_id": active_context_id},
            )
        return Conversation(
            id=conv_id,
            user_id=user_id,
            title=title,
            created_at=now,
            updated_at=now,
            active_context_id=active_context_id,
        )

    def get_conversation(
        self, conversation_id: str, *, user_id: Optional[str] = None
    ) -> Optional[Conversation]:
        with self._connect() as conn:
            params: tuple[Any, ...] = (conversation_id,)
            query = "SELECT * FROM conversation WHERE id = %s"
            if user_id:
                query += " AND user_id = %s"
                params = (conversation_id, user_id)
            row = conn.execute(query, params).fetchone()
        if not row:
            return None

        def _row_value(key: str, default: Optional[Any] = None) -> Optional[Any]:
            if hasattr(row, "get"):
                return row.get(key, default)
            try:
                return row[key]
            except Exception:
                return default

        raw_meta = _row_value("meta")
        if isinstance(raw_meta, str):
            try:
                raw_meta = json.loads(raw_meta)
            except Exception:
                raw_meta = None
        return Conversation(
            id=str(_row_value("id")),
            user_id=str(_row_value("user_id")),
            created_at=_row_value("created_at", datetime.utcnow()),
            updated_at=_row_value("updated_at", datetime.utcnow()),
            title=_row_value("title"),
            status=_row_value("status") or "open",
            active_context_id=_row_value("active_context_id"),
            meta=raw_meta,
        )

    def append_message(
        self,
        conversation_id: str,
        sender: str,
        role: str,
        content: str,
        meta: Optional[dict] = None,
        content_struct: Optional[dict] = None,
    ) -> Message:
        try:
            normalized_content_struct = normalize_content_struct(
                content_struct, content
            )
            with self._connect() as conn:
                with conn.transaction():
                    conn.execute(
                        "SELECT 1 FROM conversation WHERE id = %s FOR UPDATE",
                        (conversation_id,),
                    )
                    seq_row = conn.execute(
                        "SELECT COUNT(*) AS c FROM message WHERE conversation_id = %s",
                        (conversation_id,),
                    ).fetchone()
                    seq = seq_row["c"] if seq_row else 0
                    msg_id = str(uuid.uuid4())
                    now = datetime.utcnow()
                    conn.execute(
                        "INSERT INTO message (id, conversation_id, sender, role, content, content_struct, seq, created_at, meta) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                        (
                            msg_id,
                            conversation_id,
                            sender,
                            role,
                            content,
                            (
                                json.dumps(normalized_content_struct)
                                if normalized_content_struct is not None
                                else None
                            ),
                            seq,
                            now,
                            json.dumps(meta) if meta else None,
                        ),
                    )
                    conn.execute(
                        "UPDATE conversation SET updated_at = %s WHERE id = %s",
                        (now, conversation_id),
                    )
        except errors.ForeignKeyViolation:
            raise ConstraintViolation(
                "conversation not found", {"conversation_id": conversation_id}
            )
        return Message(
            id=msg_id,
            conversation_id=conversation_id,
            sender=sender,
            role=role,
            content=content,
            content_struct=normalized_content_struct,
            seq=seq,
            created_at=now,
            meta=meta,
        )

    def list_messages(
        self,
        conversation_id: str,
        limit: Optional[int] = None,
        *,
        user_id: Optional[str] = None,
    ) -> List[Message]:
        with self._connect() as conn:
            params: list[Any] = []
            query = "SELECT m.* FROM message m"
            if user_id:
                query += " JOIN conversation c ON c.id = m.conversation_id AND c.user_id = %s"
                params.append(user_id)
            query += " WHERE m.conversation_id = %s ORDER BY m.seq DESC"
            params.append(conversation_id)
            if limit is not None:
                query += " LIMIT %s"
                params.append(limit)
            rows = conn.execute(query, tuple(params)).fetchall()
        messages: List[Message] = []
        for row in reversed(rows):
            if not isinstance(row, dict):
                raise TypeError("list_messages expects mapping rows")
            content_struct = row.get("content_struct")
            if isinstance(content_struct, str):
                try:
                    content_struct = json.loads(content_struct)
                except Exception:
                    content_struct = None
            content_struct = normalize_content_struct(
                content_struct, row.get("content")
            )
            meta = row.get("meta")
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except Exception:
                    meta = None
            messages.append(
                Message(
                    id=str(row["id"]),
                    conversation_id=str(row["conversation_id"]),
                    sender=row["sender"],
                    role=row["role"],
                    content=row["content"],
                    content_struct=content_struct,
                    seq=row["seq"],
                    created_at=row.get("created_at", datetime.utcnow()),
                    meta=meta,
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
        self,
        type_filter: Optional[str] = None,
        kind_filter: Optional[str] = None,
        *,
        page: int = 1,
        page_size: int = 100,
        owner_user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> List[Artifact]:
        offset = max(page - 1, 0) * max(page_size, 1)
        limit = max(page_size, 1)
        with self._connect() as conn:
            clauses = []
            params: list[Any] = []
            if type_filter:
                clauses.append("type = %s")
                params.append(type_filter)
            if kind_filter:
                clauses.append("schema->>'kind' = %s")
                params.append(kind_filter)
            if owner_user_id:
                clauses.append("owner_user_id = %s")
                params.append(owner_user_id)
            if tenant_id:
                clauses.append(
                    "owner_user_id IN (SELECT id FROM app_user WHERE tenant_id = %s)"
                )
                params.append(tenant_id)
            where = ""
            if clauses:
                where = " WHERE " + " AND ".join(clauses)
            query = (
                "SELECT * FROM artifact"
                + where
                + " ORDER BY created_at DESC LIMIT %s OFFSET %s"
            )
            params.extend([limit, offset])
            rows = conn.execute(query, tuple(params)).fetchall()
        artifacts: List[Artifact] = []
        for row in rows:
            artifacts.append(
                Artifact(
                    id=str(row["id"]),
                    type=row["type"],
                    name=row["name"],
                    description=row.get("description") or "",
                    schema=row.get("schema") or {},
                    owner_user_id=(
                        str(row["owner_user_id"]) if row.get("owner_user_id") else None
                    ),
                    visibility=row.get("visibility", "private"),
                    created_at=row.get("created_at", datetime.utcnow()),
                    updated_at=row.get("updated_at", datetime.utcnow()),
                    fs_path=row.get("fs_path"),
                    base_model=row.get("base_model")
                    or (row.get("schema") or {}).get("base_model"),
                    meta=row.get("meta"),
                )
            )
        return artifacts

    def get_artifact(self, artifact_id: str) -> Optional[Artifact]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM artifact WHERE id = %s", (artifact_id,)
            ).fetchone()
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
            owner_user_id=(
                str(row["owner_user_id"]) if row.get("owner_user_id") else None
            ),
            visibility=row.get("visibility", "private"),
            created_at=row.get("created_at", datetime.utcnow()),
            updated_at=row.get("updated_at", datetime.utcnow()),
            fs_path=row.get("fs_path"),
            base_model=row.get("base_model") or (schema or {}).get("base_model"),
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
        version_author: Optional[str] = None,
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
            with self._connect() as conn, conn.transaction():
                conn.execute(
                    "INSERT INTO artifact (id, owner_user_id, type, name, description, schema, fs_path, base_model) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                    (
                        artifact_id,
                        owner_user_id,
                        type_,
                        name,
                        description,
                        json.dumps(schema),
                        fs_path,
                        schema.get("base_model"),
                    ),
                )
                conn.execute(
                    "INSERT INTO artifact_version (artifact_id, version, schema, fs_path, base_model, created_by, change_note) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (
                        artifact_id,
                        1,
                        json.dumps(schema),
                        fs_path,
                        schema.get("base_model"),
                        version_author or owner_user_id or "system_llm",
                        change_note,
                    ),
                )
        except errors.ForeignKeyViolation:
            raise ConstraintViolation(
                "artifact owner missing", {"owner_user_id": owner_user_id}
            )
        return Artifact(
            id=artifact_id,
            type=type_,
            name=name,
            description=description,
            schema=schema,
            owner_user_id=owner_user_id,
            fs_path=fs_path,
            base_model=schema.get("base_model"),
        )

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
        with self._connect() as conn, conn.transaction():
            row = conn.execute(
                "SELECT * FROM artifact WHERE id = %s", (artifact_id,)
            ).fetchone()
            if not row:
                return None
            versions = conn.execute(
                "SELECT COALESCE(MAX(version), 0) AS v FROM artifact_version WHERE artifact_id = %s",
                (artifact_id,),
            ).fetchone()
            next_version = (versions["v"] or 0) + 1
            fs_path = self._persist_payload(artifact_id, next_version, schema)
            base_model = (
                schema.get("base_model")
                if "base_model" in schema
                else row.get("base_model")
            )
            conn.execute(
                "UPDATE artifact SET schema = %s, description = COALESCE(%s, description), updated_at = now(), fs_path = %s, base_model = %s WHERE id = %s",
                (json.dumps(schema), description, fs_path, base_model, artifact_id),
            )
            conn.execute(
                "INSERT INTO artifact_version (artifact_id, version, schema, fs_path, base_model, created_by, change_note) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (
                    artifact_id,
                    next_version,
                    json.dumps(schema),
                    fs_path,
                    base_model,
                    version_author
                    or (str(row["owner_user_id"]) if row.get("owner_user_id") else None)
                    or "system_llm",
                    change_note,
                ),
            )
        new_base_model = base_model
        return Artifact(
            id=str(row["id"]),
            type=row["type"],
            name=row["name"],
            description=description or row.get("description") or "",
            schema=schema,
            owner_user_id=(
                str(row["owner_user_id"]) if row.get("owner_user_id") else None
            ),
            fs_path=fs_path,
            visibility=row.get("visibility", "private"),
            base_model=new_base_model,
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
                    self.logger.warning(
                        "artifact_version_schema_parse_failed", error=str(exc)
                    )
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
                    base_model=row.get("base_model")
                    or (schema or {}).get("base_model"),
                    meta=row.get("meta"),
                )
            )
        return versions

    def persist_artifact_payload(self, artifact_id: str, schema: dict) -> str:
        """Public wrapper kept for parity with MemoryStore."""

        existing = self.list_artifact_versions(artifact_id)
        next_version = (existing[0].version + 1) if existing else 1
        return self._persist_payload(artifact_id, next_version, schema)

    def get_latest_workflow(self, workflow_id: str) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT schema FROM artifact_version WHERE artifact_id = %s ORDER BY version DESC LIMIT 1",
                (workflow_id,),
            ).fetchone()
        if not row:
            return None
        schema = row.get("schema") if isinstance(row, dict) else row["schema"]
        if isinstance(schema, str):
            try:
                schema = json.loads(schema)
            except Exception as exc:
                self.logger.warning("workflow_schema_parse_failed", error=str(exc))
                schema = {}
        return schema

    def list_adapter_router_state(
        self, user_id: Optional[str] = None
    ) -> list[AdapterRouterState]:
        """Return adapter router state rows scoped by user ownership when provided."""

        query = (
            "SELECT ars.*, a.base_model FROM adapter_router_state ars "
            "JOIN artifact a ON ars.artifact_id = a.id"
        )
        params: tuple[Any, ...] = ()
        if user_id:
            query += " WHERE a.owner_user_id = %s"
            params = (user_id,)
        query += " ORDER BY ars.last_used_at DESC NULLS LAST, ars.usage_count DESC, ars.last_trained_at DESC NULLS LAST"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        states: list[AdapterRouterState] = []
        for row in rows:
            meta = row.get("meta") if isinstance(row, dict) else None
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except Exception:
                    meta = None
            centroid_vec = row.get("centroid_vec") if isinstance(row, dict) else None
            states.append(
                AdapterRouterState(
                    artifact_id=str(row["artifact_id"]),
                    base_model=row.get("base_model") if isinstance(row, dict) else None,
                    centroid_vec=centroid_vec if centroid_vec is not None else [],
                    usage_count=(
                        row.get("usage_count", 0) if isinstance(row, dict) else 0
                    ),
                    success_score=(
                        row.get("success_score", 0.0) if isinstance(row, dict) else 0.0
                    ),
                    last_used_at=(
                        row.get("last_used_at") if isinstance(row, dict) else None
                    ),
                    last_trained_at=(
                        row.get("last_trained_at") if isinstance(row, dict) else None
                    ),
                    meta=meta,
                )
            )
        return states

    def record_config_patch(
        self, artifact_id: str, proposer: str, patch: dict, justification: Optional[str]
    ) -> ConfigPatchAudit:
        with self._connect() as conn:
            row = conn.execute(
                "INSERT INTO config_patch (artifact_id, proposer, patch, justification, status) VALUES (%s, %s, %s, %s, %s) RETURNING id, created_at, decided_at, applied_at, status, meta",
                (artifact_id, proposer, json.dumps(patch), justification, "pending"),
            ).fetchone()
        return ConfigPatchAudit(
            id=row["id"],
            artifact_id=artifact_id,
            proposer=proposer,
            patch=patch,
            justification=justification,
            status=(
                row.get("status", "pending") if isinstance(row, dict) else row["status"]
            ),
            created_at=(
                row.get("created_at", datetime.utcnow())
                if isinstance(row, dict)
                else row["created_at"]
            ),
            decided_at=(
                row.get("decided_at")
                if isinstance(row, dict)
                else row.get("decided_at")
            ),
            applied_at=(
                row.get("applied_at")
                if isinstance(row, dict)
                else row.get("applied_at")
            ),
            meta=row.get("meta") if isinstance(row, dict) else row.get("meta"),
        )

    def get_config_patch(self, patch_id: int) -> Optional[ConfigPatchAudit]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM config_patch WHERE id = %s", (patch_id,)
            ).fetchone()
        return self._config_patch_from_row(row) if row else None

    def list_config_patches(
        self, status: Optional[str] = None
    ) -> List[ConfigPatchAudit]:
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
        self,
        patch_id: int,
        status: str,
        *,
        meta: Optional[Dict] = None,
        mark_decided: bool = False,
        mark_applied: bool = False,
    ) -> Optional[ConfigPatchAudit]:
        with self._connect() as conn, conn.transaction():
            existing = conn.execute(
                "SELECT * FROM config_patch WHERE id = %s", (patch_id,)
            ).fetchone()
            if not existing:
                return None
            now = datetime.utcnow()
            existing_meta = existing.get("meta") or {}
            if isinstance(existing_meta, str):
                try:
                    existing_meta = json.loads(existing_meta)
                except Exception as exc:
                    self.logger.warning(
                        "config_patch_meta_parse_failed", error=str(exc)
                    )
                    existing_meta = {}
            merged_meta: Dict = dict(existing_meta)
            if meta:
                merged_meta.update(meta)
            decided_at = (
                existing.get("decided_at")
                if isinstance(existing, dict)
                else existing["decided_at"]
            )
            applied_at = (
                existing.get("applied_at")
                if isinstance(existing, dict)
                else existing["applied_at"]
            )
            if mark_decided and not decided_at:
                decided_at = now
            if mark_applied:
                applied_at = now
            conn.execute(
                "UPDATE config_patch SET status = %s, decided_at = %s, applied_at = %s, meta = %s WHERE id = %s",
                (status, decided_at, applied_at, json.dumps(merged_meta), patch_id),
            )
            row = conn.execute(
                "SELECT * FROM config_patch WHERE id = %s", (patch_id,)
            ).fetchone()
        return self._config_patch_from_row(row) if row else None

    def _config_patch_from_row(self, row) -> ConfigPatchAudit:
        raw_patch = row.get("patch") if isinstance(row, dict) else row["patch"]
        patch_data = (
            raw_patch if isinstance(raw_patch, dict) else json.loads(raw_patch or "{}")
        )
        meta = row.get("meta") if isinstance(row, dict) else row.get("meta")
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception as exc:
                self.logger.warning("config_patch_meta_parse_failed", error=str(exc))
                meta = {}
        decided_at = (
            row.get("decided_at") if isinstance(row, dict) else row.get("decided_at")
        )
        applied_at = (
            row.get("applied_at") if isinstance(row, dict) else row.get("applied_at")
        )
        created = row.get("created_at") if isinstance(row, dict) else row["created_at"]
        return ConfigPatchAudit(
            id=int(row["id"]),
            artifact_id=str(row["artifact_id"]),
            proposer=row.get("proposer") if isinstance(row, dict) else row["proposer"],
            patch=patch_data,
            justification=(
                row.get("justification")
                if isinstance(row, dict)
                else row["justification"]
            ),
            status=(
                row.get("status", "pending") if isinstance(row, dict) else row["status"]
            ),
            created_at=(
                created
                if isinstance(created, datetime)
                else datetime.fromisoformat(str(created))
            ),
            decided_at=(
                decided_at
                if isinstance(decided_at, datetime) or decided_at is None
                else datetime.fromisoformat(str(decided_at))
            ),
            applied_at=(
                applied_at
                if isinstance(applied_at, datetime) or applied_at is None
                else datetime.fromisoformat(str(applied_at))
            ),
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

        with self._connect() as conn:
            row = conn.execute(
                "SELECT config FROM instance_config WHERE name = %s", ("default",)
            ).fetchone()
        raw_config = row.get("config") if row else {}
        if isinstance(raw_config, str):
            try:
                return json.loads(raw_config)
            except Exception as exc:
                self.logger.warning("runtime_config_parse_failed", error=str(exc))
                return {}
        if isinstance(raw_config, dict):
            return raw_config
        try:
            return dict(raw_config or {})
        except Exception:
            return {}

    # knowledge
    def upsert_context(
        self,
        owner_user_id: Optional[str],
        name: str,
        description: str,
        fs_path: Optional[str] = None,
        meta: Optional[dict] = None,
    ) -> KnowledgeContext:
        ctx_id = str(uuid.uuid4())
        if not owner_user_id:
            raise ConstraintViolation(
                "context owner required", {"owner_user_id": owner_user_id}
            )
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO knowledge_context (id, owner_user_id, name, description, fs_path, meta) VALUES (%s, %s, %s, %s, %s, %s)",
                    (ctx_id, owner_user_id, name, description, fs_path, meta),
                )
        except errors.ForeignKeyViolation:
            raise ConstraintViolation(
                "context owner missing", {"owner_user_id": owner_user_id}
            )
        except errors.NotNullViolation as exc:
            missing_field = getattr(getattr(exc, "diag", None), "column_name", None)
            error_fields = {"owner_user_id": owner_user_id}
            if missing_field:
                error_fields[missing_field] = None
            raise ConstraintViolation("context fields required", error_fields) from exc
        return KnowledgeContext(
            id=ctx_id,
            owner_user_id=owner_user_id,
            name=name,
            description=description,
            fs_path=fs_path,
            meta=meta,
        )

    def get_context(self, context_id: str) -> Optional[KnowledgeContext]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM knowledge_context WHERE id = %s", (context_id,)
            ).fetchone()
        if not row:
            return None

        def _row_value(key: str, default: Optional[Any] = None) -> Optional[Any]:
            if hasattr(row, "get"):
                return row.get(key, default)
            try:
                return row[key]
            except Exception:
                return default

        raw_meta = _row_value("meta")
        if isinstance(raw_meta, str):
            try:
                raw_meta = json.loads(raw_meta)
            except Exception:
                raw_meta = None
        return KnowledgeContext(
            id=str(_row_value("id")),
            owner_user_id=str(_row_value("owner_user_id")),
            name=_row_value("name", ""),
            description=_row_value("description", ""),
            created_at=_row_value("created_at", datetime.utcnow()),
            updated_at=_row_value("updated_at", datetime.utcnow()),
            fs_path=_row_value("fs_path"),
            meta=raw_meta,
        )

    def list_contexts(
        self, owner_user_id: Optional[str] = None
    ) -> List[KnowledgeContext]:
        if not owner_user_id:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM knowledge_context WHERE owner_user_id = %s ORDER BY created_at DESC",
                (owner_user_id,),
            ).fetchall()
        contexts: List[KnowledgeContext] = []
        for row in rows:
            contexts.append(
                KnowledgeContext(
                    id=str(row["id"]),
                    owner_user_id=str(row["owner_user_id"]),
                    name=row["name"],
                    description=row["description"],
                    created_at=row.get("created_at", datetime.utcnow()),
                    updated_at=row.get("updated_at", datetime.utcnow()),
                    fs_path=row.get("fs_path"),
                    meta=row.get("meta"),
                )
            )
        return contexts

    def add_context_source(
        self,
        context_id: str,
        fs_path: str,
        recursive: bool = True,
        meta: Optional[dict] = None,
    ) -> ContextSource:
        if not fs_path or not fs_path.strip():
            raise ConstraintViolation(
                "fs_path required for context_source", {"fs_path": fs_path}
            )
        src_id = str(uuid.uuid4())
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO context_source (id, context_id, fs_path, recursive, meta) VALUES (%s, %s, %s, %s, %s)",
                    (
                        src_id,
                        context_id,
                        fs_path,
                        recursive,
                        json.dumps(meta) if meta else None,
                    ),
                )
        except errors.ForeignKeyViolation:
            raise ConstraintViolation("context not found", {"context_id": context_id})
        return ContextSource(
            id=src_id,
            context_id=context_id,
            fs_path=fs_path,
            recursive=recursive,
            meta=meta,
        )

    def list_context_sources(
        self, context_id: Optional[str] = None
    ) -> List[ContextSource]:
        with self._connect() as conn:
            if context_id:
                rows = conn.execute(
                    "SELECT * FROM context_source WHERE context_id = %s ORDER BY fs_path ASC",
                    (context_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM context_source ORDER BY context_id, fs_path ASC", ()
                ).fetchall()
        return [
            ContextSource(
                id=str(row["id"]),
                context_id=str(row["context_id"]),
                fs_path=row["fs_path"],
                recursive=bool(row.get("recursive", True)),
                meta=row.get("meta"),
            )
            for row in rows
        ]

    def add_chunks(self, context_id: str, chunks: Iterable[KnowledgeChunk]) -> None:
        try:
            with self._connect() as conn:
                for chunk in chunks:
                    if not chunk.fs_path or not str(chunk.fs_path).strip():
                        raise ConstraintViolation(
                            "fs_path required for knowledge_chunk",
                            {"fs_path": chunk.fs_path},
                        )
                    conn.execute(
                        "INSERT INTO knowledge_chunk (context_id, fs_path, chunk_index, content, embedding, created_at, meta) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                        (
                            context_id,
                            chunk.fs_path,
                            chunk.chunk_index,
                            chunk.content,
                            chunk.embedding,
                            chunk.created_at,
                            json.dumps(chunk.meta) if chunk.meta else None,
                        ),
                    )
        except errors.ForeignKeyViolation:
            raise ConstraintViolation("context not found", {"context_id": context_id})

    def list_chunks(
        self, context_id: Optional[str] = None, *, owner_user_id: Optional[str] = None
    ) -> List[KnowledgeChunk]:
        with self._connect() as conn:
            if context_id:
                params: list[Any] = []
                query = "SELECT kc.* FROM knowledge_chunk kc"
                if owner_user_id:
                    query += " JOIN knowledge_context ctx ON ctx.id = kc.context_id AND ctx.owner_user_id = %s"
                    params.append(owner_user_id)
                query += " WHERE kc.context_id = %s ORDER BY kc.chunk_index ASC"
                params.append(context_id)
                rows = conn.execute(query, tuple(params)).fetchall()
            else:
                if not owner_user_id:
                    return []
                rows = conn.execute(
                    "SELECT kc.* FROM knowledge_chunk kc JOIN knowledge_context ctx ON ctx.id = kc.context_id WHERE ctx.owner_user_id = %s ORDER BY kc.created_at DESC",
                    (owner_user_id,),
                ).fetchall()
        chunks: List[KnowledgeChunk] = []
        for row in rows:
            chunks.append(
                KnowledgeChunk(
                    id=int(row["id"]),
                    context_id=str(row["context_id"]),
                    fs_path=row["fs_path"],
                    content=row["content"],
                    embedding=row.get("embedding") or [],
                    chunk_index=row.get("chunk_index", 0),
                    created_at=row.get("created_at", datetime.utcnow()),
                    meta=row.get("meta"),
                )
            )
        return chunks

    def _format_vector(self, embedding: Sequence[float]) -> str:
        return "[" + ",".join(f"{float(val):.6f}" for val in embedding) + "]"

    def search_chunks_pgvector(
        self,
        context_ids: Optional[Sequence[str]],
        query: str,
        query_embedding: List[float],
        limit: int = 4,
        filters: Optional[dict[str, Any]] = None,
        *,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> List[KnowledgeChunk]:
        """Primary pgvector-backed retrieval over knowledge chunks."""

        if not query_embedding or not context_ids:
            return []
        where_clauses: list[str] = []
        params: list[Any] = []
        where_clauses.append("kc.context_id = ANY(%s)")
        params.append(list(context_ids))
        if user_id:
            where_clauses.append("ctx.owner_user_id = %s")
            params.append(user_id)
        if tenant_id:
            where_clauses.append("u.tenant_id = %s")
            params.append(tenant_id)
        if filters and filters.get("fs_path"):
            where_clauses.append("kc.fs_path = %s")
            params.append(filters["fs_path"])
        if filters and filters.get("embedding_model_id"):
            where_clauses.append("kc.meta->>'embedding_model_id' = %s")
            params.append(filters["embedding_model_id"])
        where = ""
        if where_clauses:
            where = " WHERE " + " AND ".join(where_clauses)
        with self._connect() as conn:
            query = (
                " "
                """
                SELECT kc.id, kc.context_id, kc.fs_path, kc.content, kc.embedding, kc.chunk_index, kc.created_at, kc.meta
                FROM knowledge_chunk kc
                JOIN knowledge_context ctx ON kc.context_id = ctx.id
                LEFT JOIN app_user u ON ctx.owner_user_id = u.id
                """
            )
            query += where
            query += " ORDER BY kc.embedding <-> %s::vector LIMIT %s"
            rows = conn.execute(
                query, (*params, self._format_vector(query_embedding), limit)
            ).fetchall()
        # pgvector mode: results already ordered by vector similarity via SQL
        # No BM25 re-ranking per SPEC §3 - pure vector search for pgvector mode
        return [
            KnowledgeChunk(
                id=int(row["id"]),
                context_id=str(row["context_id"]),
                fs_path=row["fs_path"],
                content=row["content"],
                embedding=row.get("embedding") or [],
                chunk_index=row.get("chunk_index", 0),
                created_at=row.get("created_at", datetime.utcnow()),
                meta=row.get("meta"),
            )
            for row in rows
        ]

    def search_chunks(
        self,
        context_id: Optional[str],
        query: str,
        query_embedding: Optional[List[float]],
        limit: int = 4,
    ) -> List[KnowledgeChunk]:
        """Non-pgvector hybrid search; suitable for tests and tiny corpora only."""

        def _cosine(a: List[float], b: List[float]) -> float:
            if not a or not b:
                return 0.0
            length = min(len(a), len(b))
            dot = sum(a[i] * b[i] for i in range(length))
            norm_a = sum(x * x for x in a) ** 0.5 or 1.0
            norm_b = sum(x * x for x in b) ** 0.5 or 1.0
            return dot / (norm_a * norm_b)

        candidates = self.list_chunks(context_id)
        if not candidates:
            return []
        query_tokens = _tokenize_text(query)
        documents = [_tokenize_text(ch.content) for ch in candidates]
        bm25_scores = _compute_bm25_scores(query_tokens, documents)
        semantic_scores = [
            (_cosine(query_embedding, ch.embedding) if query_embedding else 0.0)
            for ch in candidates
        ]
        max_bm25 = max(bm25_scores) or 1.0
        combined: dict[str, tuple[KnowledgeChunk, float]] = {}
        for chunk, lex, sem in zip(candidates, bm25_scores, semantic_scores):
            hybrid = 0.45 * (lex / max_bm25) + 0.55 * sem
            key = " ".join(chunk.content.split()).lower() or str(chunk.id or "")
            existing = combined.get(key)
            if not existing or hybrid > existing[1]:
                combined[key] = (chunk, hybrid)
        ranked = sorted(combined.values(), key=lambda pair: pair[1], reverse=True)
        return [pair[0] for pair in ranked[:limit]]

    def inspect_state(
        self,
        *,
        tenant_id: Optional[str] = None,
        kind: Optional[str] = None,
        limit: int = 50,
    ) -> dict:
        def _serialize(row: dict) -> dict:
            data = dict(row)
            for key, value in list(data.items()):
                if isinstance(value, datetime):
                    data[key] = value.isoformat()
            return data

        sections: dict[str, list] = {}
        with self._connect() as conn:
            if kind in (None, "users"):
                if tenant_id:
                    rows = conn.execute(
                        "SELECT * FROM app_user WHERE tenant_id = %s ORDER BY created_at DESC LIMIT %s",
                        (tenant_id, limit),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM app_user ORDER BY created_at DESC LIMIT %s",
                        (limit,),
                    ).fetchall()
                sections["users"] = [_serialize(row) for row in rows]
            if kind in (None, "sessions"):
                rows = conn.execute(
                    "SELECT * FROM auth_session WHERE (%s IS NULL OR tenant_id = %s) ORDER BY created_at DESC LIMIT %s",
                    (tenant_id, tenant_id, limit),
                ).fetchall()
                sections["sessions"] = [_serialize(row) for row in rows]
            if kind in (None, "conversations"):
                rows = conn.execute(
                    """
                    SELECT c.*
                    FROM conversation c
                    JOIN app_user u ON c.user_id = u.id
                    WHERE (%s IS NULL OR u.tenant_id = %s)
                    ORDER BY c.updated_at DESC
                    LIMIT %s
                    """,
                    (tenant_id, tenant_id, limit),
                ).fetchall()
                sections["conversations"] = [_serialize(row) for row in rows]
            if kind in (None, "messages"):
                rows = conn.execute(
                    """
                    SELECT m.*
                    FROM message m
                    JOIN conversation c ON m.conversation_id = c.id
                    JOIN app_user u ON c.user_id = u.id
                    WHERE (%s IS NULL OR u.tenant_id = %s)
                    ORDER BY m.created_at DESC
                    LIMIT %s
                    """,
                    (tenant_id, tenant_id, limit),
                ).fetchall()
                sections["messages"] = [_serialize(row) for row in rows]
            if kind in (None, "artifacts"):
                rows = conn.execute(
                    "SELECT * FROM artifact ORDER BY created_at DESC LIMIT %s", (limit,)
                ).fetchall()
                sections["artifacts"] = [_serialize(row) for row in rows]
            if kind in (None, "contexts"):
                rows = conn.execute(
                    """
                    SELECT kc.*
                    FROM knowledge_context kc
                    LEFT JOIN app_user u ON kc.owner_user_id = u.id
                    WHERE (%s IS NULL OR u.tenant_id = %s)
                    ORDER BY kc.created_at DESC
                    LIMIT %s
                    """,
                    (tenant_id, tenant_id, limit),
                ).fetchall()
                sections["contexts"] = [_serialize(row) for row in rows]
            if kind in (None, "chunks"):
                rows = conn.execute(
                    """
                    SELECT kc.*
                    FROM knowledge_chunk kc
                    LEFT JOIN knowledge_context ctx ON kc.context_id = ctx.id
                    LEFT JOIN app_user u ON ctx.owner_user_id = u.id
                    WHERE (%s IS NULL OR u.tenant_id = %s)
                    ORDER BY kc.created_at DESC
                    LIMIT %s
                    """,
                    (tenant_id, tenant_id, limit),
                ).fetchall()
                sections["chunks"] = [_serialize(row) for row in rows]
            if kind in (None, "training_jobs"):
                rows = conn.execute(
                    """
                    SELECT tj.*
                    FROM training_job tj
                    JOIN app_user u ON tj.user_id = u.id
                    WHERE (%s IS NULL OR u.tenant_id = %s)
                    ORDER BY tj.created_at DESC
                    LIMIT %s
                    """,
                    (tenant_id, tenant_id, limit),
                ).fetchall()
                sections["training_jobs"] = [_serialize(row) for row in rows]
            if kind in (None, "config_patches"):
                rows = conn.execute(
                    "SELECT * FROM config_patch ORDER BY created_at DESC LIMIT %s",
                    (limit,),
                ).fetchall()
                sections["config_patches"] = [_serialize(row) for row in rows]
        return sections

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

    def _load_training_state(self) -> None:
        path = self._training_state_path()
        if not path.exists():
            return
        try:
            path.unlink()
            self.logger.info("removed_legacy_training_state_file", path=str(path))
        except OSError as exc:
            self.logger.warning(
                "remove_legacy_training_state_file_failed",
                path=str(path),
                error=str(exc),
            )
