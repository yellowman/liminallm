from __future__ import annotations

import json
import math
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence

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
        self._ensure_runtime_config_table()
        self._verify_required_schema()
        self._load_training_state()
        self._ensure_default_artifacts()

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
                row = conn.execute("SELECT to_regclass(%s) AS oid", (f"public.{table}",)).fetchone()
                if not row or not row.get("oid"):
                    missing_tables.append(table)

            if missing_tables:
                raise RuntimeError(
                    "Missing required Postgres tables: {}. Run scripts/migrate.sh to install the SPEC §2 schema.".format(
                        ", ".join(sorted(missing_tables))
                    )
                )

            vector_ext = conn.execute("SELECT extname FROM pg_extension WHERE extname = 'vector'").fetchone()
            if not vector_ext:
                raise RuntimeError(
                    "pgvector extension is missing. Install it and rerun scripts/migrate.sh to satisfy SPEC §3 RAG requirements."
                )

            citext_ext = conn.execute("SELECT extname FROM pg_extension WHERE extname = 'citext'").fetchone()
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
                created_by="system_llm",
                change_note="Seeded default workflow",
            )

        seeded_tools = {art.schema.get("name") for art in existing if isinstance(art.schema, dict) and art.schema.get("kind") == "tool.spec"}
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
                "inputs": {"message": {"type": "string"}, "context_id": {"type": "string", "optional": True}},
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
                created_by="system_llm",
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
        columns = (
            "id, adapter_artifact_id, user_id, created_at, updated_at, status, num_events, loss, "
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
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO app_user (id, email, handle, tenant_id, role, plan_tier, is_active, meta)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (user_id, email, handle, tenant_id, role, plan_tier, is_active, json.dumps(meta) if meta else None),
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
            meta=meta,
        )

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
            raise
        self.mfa_secrets[user_id] = record
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
            role=row.get("role", "user"),
            tenant_id=row.get("tenant_id", "public"),
            meta=row.get("meta"),
        )

    def get_user(self, user_id: str) -> Optional[User]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM app_user WHERE id = %s", (user_id,)).fetchone()
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

    def list_users(self, tenant_id: Optional[str] = None, limit: int = 100) -> List[User]:
        with self._connect() as conn:
            if tenant_id:
                rows = conn.execute(
                    "SELECT * FROM app_user WHERE tenant_id = %s ORDER BY created_at DESC LIMIT %s",
                    (tenant_id, limit),
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM app_user ORDER BY created_at DESC LIMIT %s", (limit,)).fetchall()
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
                "UPDATE app_user SET role = %s, updated_at = now() WHERE id = %s RETURNING *", (role, user_id)
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
                        ip_addr,
                        mfa_required,
                        sess.mfa_verified,
                        json.dumps(meta) if meta else None,
                    ),
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
        meta = row.get("meta")
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = None
        sess = Session(
            id=str(row["id"]),
            user_id=str(row["user_id"]),
            created_at=row.get("created_at", datetime.utcnow()),
            expires_at=row.get("expires_at", datetime.utcnow()),
            user_agent=row.get("user_agent"),
            ip_addr=row.get("ip_addr"),
            mfa_required=row.get("mfa_required", False),
            mfa_verified=row.get("mfa_verified", False),
            tenant_id=row.get("tenant_id", "public"),
            meta=meta,
        )
        self.sessions[sess.id] = sess
        return sess

    def set_session_meta(self, session_id: str, meta: dict) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE auth_session SET meta = %s WHERE id = %s", (json.dumps(meta), session_id))
        sess = self.sessions.get(session_id)
        if sess:
            sess.meta = meta
            self.sessions[session_id] = sess

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
        self,
        type_filter: Optional[str] = None,
        kind_filter: Optional[str] = None,
        *,
        page: int = 1,
        page_size: int = 100,
    ) -> List[Artifact]:
        offset = max(page - 1, 0) * max(page_size, 1)
        limit = max(page_size, 1)
        with self._connect() as conn:
            if type_filter and kind_filter:
                rows = conn.execute(
                    "SELECT * FROM artifact WHERE type = %s AND schema->>'kind' = %s ORDER BY created_at DESC LIMIT %s OFFSET %s",
                    (type_filter, kind_filter, limit, offset),
                ).fetchall()
            elif kind_filter:
                rows = conn.execute(
                    "SELECT * FROM artifact WHERE schema->>'kind' = %s ORDER BY created_at DESC LIMIT %s OFFSET %s",
                    (kind_filter, limit, offset),
                ).fetchall()
            elif type_filter:
                rows = conn.execute(
                    "SELECT * FROM artifact WHERE type = %s ORDER BY created_at DESC LIMIT %s OFFSET %s",
                    (type_filter, limit, offset),
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM artifact ORDER BY created_at DESC LIMIT %s OFFSET %s", (limit, offset)).fetchall()
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
            base_model=(schema or {}).get("base_model"),
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
            base_model=schema.get("base_model"),
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
            base_model=schema.get("base_model"),
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
                    base_model=(schema or {}).get("base_model"),
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

    def record_config_patch(self, artifact_id: str, proposer: str, patch: dict, justification: Optional[str]) -> ConfigPatchAudit:
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
            status=row.get("status", "pending") if isinstance(row, dict) else row["status"],
            created_at=row.get("created_at", datetime.utcnow()) if isinstance(row, dict) else row["created_at"],
            decided_at=row.get("decided_at") if isinstance(row, dict) else row.get("decided_at"),
            applied_at=row.get("applied_at") if isinstance(row, dict) else row.get("applied_at"),
            meta=row.get("meta") if isinstance(row, dict) else row.get("meta"),
        )

    def get_config_patch(self, patch_id: int) -> Optional[ConfigPatchAudit]:
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
        self, patch_id: int, status: str, *, meta: Optional[Dict] = None, mark_decided: bool = False, mark_applied: bool = False
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
            decided_at = existing.get("decided_at") if isinstance(existing, dict) else existing["decided_at"]
            applied_at = existing.get("applied_at") if isinstance(existing, dict) else existing["applied_at"]
            if mark_decided and not decided_at:
                decided_at = now
            if mark_applied:
                applied_at = now
            conn.execute(
                "UPDATE config_patch SET status = %s, decided_at = %s, applied_at = %s, meta = %s WHERE id = %s",
                (status, decided_at, applied_at, json.dumps(merged_meta), patch_id),
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
        decided_at = row.get("decided_at") if isinstance(row, dict) else row.get("decided_at")
        applied_at = row.get("applied_at") if isinstance(row, dict) else row.get("applied_at")
        created = row.get("created_at") if isinstance(row, dict) else row["created_at"]
        return ConfigPatchAudit(
            id=int(row["id"]),
            artifact_id=str(row["artifact_id"]),
            proposer=row.get("proposer") if isinstance(row, dict) else row["proposer"],
            patch=patch_data,
            justification=row.get("justification") if isinstance(row, dict) else row["justification"],
            status=row.get("status", "pending") if isinstance(row, dict) else row["status"],
            created_at=created if isinstance(created, datetime) else datetime.fromisoformat(str(created)),
            decided_at=decided_at if isinstance(decided_at, datetime) or decided_at is None else datetime.fromisoformat(str(decided_at)),
            applied_at=applied_at if isinstance(applied_at, datetime) or applied_at is None else datetime.fromisoformat(str(applied_at)),
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
        self, owner_user_id: Optional[str], name: str, description: str, fs_path: Optional[str] = None, meta: Optional[dict] = None
    ) -> KnowledgeContext:
        ctx_id = str(uuid.uuid4())
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO knowledge_context (id, owner_user_id, name, description, fs_path, meta) VALUES (%s, %s, %s, %s, %s, %s)",
                    (ctx_id, owner_user_id, name, description, fs_path, meta),
                )
        except errors.ForeignKeyViolation:
            raise ConstraintViolation("context owner missing", {"owner_user_id": owner_user_id})
        return KnowledgeContext(
            id=ctx_id, owner_user_id=owner_user_id, name=name, description=description, fs_path=fs_path, meta=meta
        )

    def list_contexts(self, owner_user_id: Optional[str] = None) -> List[KnowledgeContext]:
        with self._connect() as conn:
            if owner_user_id:
                rows = conn.execute(
                    "SELECT * FROM knowledge_context WHERE owner_user_id = %s ORDER BY created_at DESC",
                    (owner_user_id,),
                ).fetchall()
            else:
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
                    if not chunk.fs_path:
                        raise ConstraintViolation("fs_path required for knowledge_chunk", {"fs_path": chunk.fs_path})
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

    def list_chunks(self, context_id: Optional[str] = None) -> List[KnowledgeChunk]:
        with self._connect() as conn:
            if context_id:
                rows = conn.execute(
                    "SELECT * FROM knowledge_chunk WHERE context_id = %s ORDER BY chunk_index ASC",
                    (context_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM knowledge_chunk ORDER BY created_at DESC",
                    (),
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
        where = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT kc.id, kc.context_id, kc.fs_path, kc.content, kc.embedding, kc.chunk_index, kc.created_at, kc.meta
                FROM knowledge_chunk kc
                JOIN knowledge_context ctx ON kc.context_id = ctx.id
                LEFT JOIN app_user u ON ctx.owner_user_id = u.id
                {where}
                ORDER BY kc.embedding <-> %s::vector
                LIMIT %s
                """,
                (*params, self._format_vector(query_embedding), limit),
            ).fetchall()
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

        def _tokenize(text: str) -> List[str]:
            return re.findall(r"\w+", text.lower())

        def _cosine(a: List[float], b: List[float]) -> float:
            if not a or not b:
                return 0.0
            length = min(len(a), len(b))
            dot = sum(a[i] * b[i] for i in range(length))
            norm_a = sum(x * x for x in a) ** 0.5 or 1.0
            norm_b = sum(x * x for x in b) ** 0.5 or 1.0
            return dot / (norm_a * norm_b)

        def _bm25_scores(query_tokens: Sequence[str], documents: List[List[str]]) -> List[float]:
            if not query_tokens or not documents:
                return [0.0 for _ in documents]
            N = len(documents)
            avgdl = sum(len(doc) for doc in documents) / float(N)
            doc_freq: dict[str, int] = {}
            for doc in documents:
                seen = set(doc)
                for tok in seen:
                    doc_freq[tok] = doc_freq.get(tok, 0) + 1
            k1 = 1.5
            b = 0.75
            scores: List[float] = []
            for doc in documents:
                tf: dict[str, int] = {}
                for tok in doc:
                    tf[tok] = tf.get(tok, 0) + 1
                score = 0.0
                for tok in query_tokens:
                    df = doc_freq.get(tok, 0)
                    if df == 0:
                        continue
                    idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
                    freq = tf.get(tok, 0)
                    denom = freq + k1 * (1 - b + b * (len(doc) / (avgdl or 1.0)))
                    score += idf * (freq * (k1 + 1)) / denom if denom else 0.0
                scores.append(score)
            return scores

        candidates = self.list_chunks(context_id)
        if not candidates:
            return []
        query_tokens = _tokenize(query)
        documents = [_tokenize(ch.content) for ch in candidates]
        bm25_scores = _bm25_scores(query_tokens, documents)
        semantic_scores = [(_cosine(query_embedding, ch.embedding) if query_embedding else 0.0) for ch in candidates]
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

    def inspect_state(self, *, tenant_id: Optional[str] = None, kind: Optional[str] = None, limit: int = 50) -> dict:
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
                        "SELECT * FROM app_user WHERE tenant_id = %s ORDER BY created_at DESC LIMIT %s", (tenant_id, limit)
                    ).fetchall()
                else:
                    rows = conn.execute("SELECT * FROM app_user ORDER BY created_at DESC LIMIT %s", (limit,)).fetchall()
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
                rows = conn.execute("SELECT * FROM artifact ORDER BY created_at DESC LIMIT %s", (limit,)).fetchall()
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
                    "SELECT * FROM config_patch ORDER BY created_at DESC LIMIT %s", (limit,)
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
            self.logger.warning("remove_legacy_training_state_file_failed", path=str(path), error=str(exc))
