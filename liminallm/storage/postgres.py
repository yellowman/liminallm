from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from datetime import datetime, timezone
from ipaddress import ip_address
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from psycopg import errors
from psycopg.abc import Buffer
from psycopg.adapt import Loader
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from liminallm.config import SYSTEM_SETTINGS_DEFAULTS, redact_secrets
from liminallm.content_struct import normalize_content_struct
from liminallm.logging import get_logger
from liminallm.service.artifact_validation import (
    ArtifactValidationError,
    validate_artifact,
)
from liminallm.service.bm25 import (
    compute_bm25_scores as _compute_bm25_scores,
)
from liminallm.service.bm25 import (
    tokenize_text as _tokenize_text,
)
from liminallm.service.embeddings import (
    EMBEDDING_DIM,
    validated_embedding,
)
from liminallm.service.errors import NotFoundError
from liminallm.service.ranking import (
    LEXICAL_WEIGHT as _LEXICAL_WEIGHT,
)
from liminallm.service.ranking import (
    SEMANTIC_WEIGHT as _SEMANTIC_WEIGHT,
)
from liminallm.service.ranking import (
    fuse_ranks as _fuse_ranks,
)
from liminallm.service.ranking import (
    ranked_positive as _ranked_positive,
)
from liminallm.storage.common import (
    blend_centroid,
    clamp_success_score,
    compute_text_embedding,
    ensure_policy_compliant_texts,
    get_default_chat_workflow_schema,
    get_default_tool_specs,
    normalize_optional_text,
)
from liminallm.storage.cursors import (
    decode_artifact_cursor,
    decode_index_cursor,
    decode_time_id_cursor,
)
from liminallm.storage.errors import ConstraintViolation
from liminallm.storage.models import (
    AdapterRouterState,
    ApiKey,
    Artifact,
    ArtifactVersion,
    ConfigPatchAudit,
    ContextSource,
    Conversation,
    KnowledgeChunk,
    KnowledgeContext,
    Message,
    Note,
    PreferenceEvent,
    SemanticCluster,
    Session,
    TrainingJob,
    User,
    UserMFAConfig,
    UserSettings,
)

_MAX_SESSION_CACHE_SIZE = 10000


class _UUIDAsText(Loader):
    """Hand UUID columns back as strings.

    Every id in the models is typed ``str``, and the ids flowing in from JWTs,
    URLs and JSON are strings. Left as ``uuid.UUID``, a column value compares
    unequal to the very id it was looked up by and is not JSON serializable —
    both of which have shipped as bugs here. Convert once, at the boundary.
    """

    def load(self, data: Buffer) -> str:
        return bytes(data).decode()


def _configure_connection(conn) -> None:
    conn.adapters.register_loader("uuid", _UUIDAsText)
    # Without this, TIMESTAMPTZ comes back in whatever timezone the server was
    # initialised with; the instant is right but the tzinfo is not UTC, and
    # code that expects UTC (cookie expiry formatting, for one) rejects it.
    conn.execute("SET TIME ZONE 'UTC'")
    conn.commit()


def _is_uuid(value: Any) -> bool:
    """Every id in this schema is a UUID.

    A malformed id cannot match a row, but Postgres raises on it rather than
    returning nothing — so callers get a 500 where they should get a miss.
    Check here and let the lookup return None.
    """
    try:
        uuid.UUID(str(value))
    except (AttributeError, TypeError, ValueError):
        return False
    return True


# A term longer than this is a checksum or a mangled blob, not a word anyone
# searched for; tsquery also refuses tokens past 2KB.
_MAX_TSQUERY_TERM = 100
_MAX_TSQUERY_TERMS = 32


def _tsquery_terms(query: str, *, max_terms: int = _MAX_TSQUERY_TERMS) -> str:
    """OR'd ``to_tsquery`` input built from a free-text query.

    Terms come from the BM25 tokenizer, which yields ``\\w+`` and nothing else,
    so no tsquery operator can reach the parser and a user query cannot become
    a syntax error. Duplicates drop out and the term count is capped, because
    the cost of the scan grows with the number of terms.
    """
    seen: set[str] = set()
    terms: list[str] = []
    for token in _tokenize_text(query):
        if len(token) > _MAX_TSQUERY_TERM or token in seen:
            continue
        seen.add(token)
        terms.append(token)
        if len(terms) >= max_terms:
            break
    return " | ".join(terms)


class PostgresStore:
    """Thin Postgres-backed store to persist kernel primitives."""

    def __init__(self, dsn: str, fs_root: str) -> None:
        self.dsn = dsn
        self.fs_root = Path(fs_root)
        self.logger = get_logger(__name__)
        self._connect_max_retries = 3
        self._connect_retry_backoff = 0.25
        self._last_pool_metrics_log = 0.0

        try:
            self.fs_root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self.logger.error(
                "postgres_store_fs_root_error",
                fs_root=fs_root,
                error=str(exc),
            )
            raise

        try:
            # Issue 48.2: Add connection pool timeout configuration
            # timeout: max seconds to wait for a connection from the pool
            # max_waiting: max requests that can wait for a connection
            # reconnect_timeout: seconds to wait before reconnecting failed connection
            self.pool = ConnectionPool(
                self.dsn,
                min_size=2,
                max_size=10,
                timeout=30.0,  # Don't wait more than 30s for a connection
                max_waiting=100,  # Limit waiting queue to prevent unbounded growth
                reconnect_timeout=5.0,  # Quick reconnection on failure
                configure=_configure_connection,
                kwargs={"row_factory": dict_row, "autocommit": False},
            )
            self.logger.info(
                "postgres_pool_created",
                min_size=2,
                max_size=10,
                timeout=30.0,
                max_waiting=100,
            )
        except Exception as exc:
            self.logger.error(
                "postgres_pool_creation_failed",
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise

        self.sessions: dict[str, Session] = {}
        self._session_lock = threading.Lock()

        try:
            self._ensure_runtime_config_table()
            self._verify_required_schema()
            self._load_training_state()
            self._ensure_default_artifacts()
            self.logger.info("postgres_store_initialized")
        except Exception as exc:
            self.logger.error(
                "postgres_store_init_failed",
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise

    def verify_connection(self) -> None:
        """Round-trip a query. Raises if the database is unreachable."""
        with self._connect() as conn:
            conn.execute("SELECT 1").fetchone()

    def close_pool(self) -> None:
        """Synchronously close the pool (test resets, shutdown without a loop)."""
        try:
            self.pool.close()
            self.pool.wait_closed()
        except Exception as exc:  # noqa: BLE001 - already closed is fine
            self.logger.debug("postgres_pool_close_failed", error=str(exc))

    async def close(self) -> None:
        """Close the connection pool for graceful shutdown (Issues 57.7, 59.1)."""

        try:
            self.pool.close()
            # wait_closed is synchronous; run in thread to avoid blocking event loop callers
            await asyncio.to_thread(self.pool.wait_closed)
            self.logger.info("postgres_pool_closed")
        except Exception as exc:
            self.logger.warning("postgres_pool_close_failed", error=str(exc))

    def _cache_session(self, session: Session) -> Session:
        """Store session in the in-memory cache and return it.

        Thread-safe per SPEC §18 inference/adapter cache discipline.
        """
        with self._session_lock:
            now = datetime.now(timezone.utc)

            def _is_expired(session: Session) -> bool:
                if session.expires_at is None:
                    return False
                expires_at = (
                    session.expires_at
                    if session.expires_at.tzinfo
                    else session.expires_at.replace(tzinfo=timezone.utc)
                )
                return expires_at <= now

            # First prune any expired sessions to avoid evicting valid ones (Issue 53.10)
            expired_ids = [sid for sid, sess in self.sessions.items() if _is_expired(sess)]
            for sid in expired_ids:
                self.sessions.pop(sid, None)
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
        attempt = 0
        last_exc: Exception | None = None
        while attempt < self._connect_max_retries:
            try:
                start = time.monotonic()
                conn = self.pool.connection()
                elapsed_ms = (time.monotonic() - start) * 1000
                self._maybe_log_pool_metrics(elapsed_ms)
                return conn
            except Exception as exc:
                last_exc = exc
                attempt += 1
                self.logger.warning(
                    "postgres_connect_retry",
                    attempt=attempt,
                    max_attempts=self._connect_max_retries,
                    error=str(exc),
                )
                time.sleep(self._connect_retry_backoff * attempt)
        if last_exc:
            raise last_exc
        return self.pool.connection()

    def _maybe_log_pool_metrics(self, wait_ms: float) -> None:
        """Emit periodic pool health metrics to surface saturation early."""

        now = time.monotonic()
        if (now - self._last_pool_metrics_log) < 60 and wait_ms < 500:
            return

        stats_fn = getattr(self.pool, "get_stats", None)
        waiting = used = free = open_conns = None
        max_size = getattr(self.pool, "max_size", None)
        if callable(stats_fn):
            try:
                stats = stats_fn()
                waiting = getattr(stats, "waiting", None)
                used = getattr(stats, "used", None)
                free = getattr(stats, "free", None)
                open_conns = getattr(stats, "open", None)
            except Exception as exc:  # pragma: no cover - defensive logging
                self.logger.debug("postgres_pool_stats_failed", error=str(exc))

        if wait_ms >= 500 or (waiting and waiting > 0) or (used and max_size and used >= max_size):
            self.logger.warning(
                "postgres_pool_pressure",
                wait_ms=round(wait_ms, 2),
                waiting=waiting,
                used=used,
                free=free,
                open_connections=open_conns,
                max_size=max_size,
            )
            self._last_pool_metrics_log = now

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

    @staticmethod
    def _configured_embedding_dim() -> Optional[int]:
        """Vector width the configured encoder produces, if knowable."""
        try:
            from liminallm.config import get_settings

            dim = int(getattr(get_settings(), "embedding_vector_dim", 0) or 0)
            return dim or None
        except Exception:  # noqa: BLE001 - never block startup on config read
            return None

    def _verify_required_schema(self) -> None:
        """Ensure core tables and pgvector expectations exist before serving requests."""

        required_tables = [
            "app_user",
            "user_auth_credential",
            "user_auth_provider",
            "user_api_key",
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
            "knowledge_chunk_vector",
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

            # Retrieval's lexical channel reads this generated column by
            # name. The table list above cannot catch its absence — the table
            # is old, the column is not — so an install that pulled new code
            # without re-running migrations booted clean and then answered
            # every grounded chat turn with a 500. Fail here instead, where
            # the message can name the fix.
            fts_column = conn.execute(
                """
                SELECT 1 FROM pg_attribute
                WHERE attrelid = 'knowledge_chunk'::regclass
                  AND attname = 'content_tsv' AND NOT attisdropped
                """
            ).fetchone()
            if not fts_column:
                raise RuntimeError(
                    "knowledge_chunk.content_tsv is missing, so hybrid retrieval's "
                    "keyword channel cannot run. Rerun scripts/migrate.sh to apply "
                    "the SPEC §2.5 schema."
                )

            # `get_or_create_conversation_attachment_context` is correct only
            # while this index exists: its `ON CONFLICT DO NOTHING` needs a
            # constraint to collide with, and without one two concurrent first
            # attachments each insert an index for the same conversation —
            # leaving an acknowledged attachment somewhere no lookup returns.
            # Same reasoning as `content_tsv` above: code can be newer than
            # the database, so a load-bearing schema feature is checked where
            # the message can name the fix.
            #
            # Checked by shape rather than by name, so an index that happens
            # to carry the name without the predicate does not satisfy it.
            uniqueness = conn.execute(
                """
                SELECT 1 FROM pg_index i
                JOIN pg_class c ON c.oid = i.indexrelid
                WHERE i.indrelid = 'knowledge_context'::regclass
                  AND i.indisunique
                  AND pg_get_expr(i.indpred, i.indrelid) IS NOT NULL
                  AND pg_get_indexdef(i.indexrelid) LIKE %s
                  AND pg_get_expr(i.indpred, i.indrelid) LIKE %s
                """,
                ("%conversation_id%", "%auto%"),
            ).fetchone()
            if not uniqueness:
                raise RuntimeError(
                    "the unique index making one conversation have one implicit "
                    "attachment context is missing, so concurrent attachments can "
                    "create two and one of them becomes unreachable. Rerun "
                    "scripts/migrate.sh to apply the SPEC §19.5 schema."
                )

            vector_ext = conn.execute(
                "SELECT extname FROM pg_extension WHERE extname = 'vector'"
            ).fetchone()
            if not vector_ext:
                raise RuntimeError(
                    "pgvector extension is missing. Install it and rerun scripts/migrate.sh to satisfy SPEC §3 RAG requirements."
                )

            # The embedding column's dimension is fixed at schema time and the
            # encoder's is fixed at runtime; if they disagree, every chunk
            # insert fails later with an opaque pgvector error. Say so now,
            # with the two numbers and the fix.
            column_dim = conn.execute(
                """
                SELECT atttypmod FROM pg_attribute
                WHERE attrelid = 'knowledge_chunk'::regclass AND attname = 'embedding'
                """
            ).fetchone()
            configured = self._configured_embedding_dim()
            if column_dim and configured:
                actual = int(column_dim["atttypmod"])
                if actual > 0 and actual != configured:
                    raise RuntimeError(
                        f"embedding dimension mismatch: knowledge_chunk.embedding "
                        f"is vector({actual}) but the configured encoder produces "
                        f"{configured}-d vectors. Set EMBEDDING_VECTOR_DIM={configured} "
                        f"and re-apply sql/schema.sql, or configure an encoder "
                        f"matching vector({actual})."
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

    def _ensure_default_artifacts(self) -> None:
        """Seed default artifacts using common schema definitions."""
        existing = self.list_artifacts()

        # Seed default chat workflow if not present
        if not any(artifact.name == "default_chat_workflow" for artifact in existing):
            default_schema = get_default_chat_workflow_schema()
            self.create_artifact(
                "workflow",
                "default_chat_workflow",
                default_schema,
                "LLM-only chat workflow defined as data.",
                visibility="global",
                version_author="system_llm",
                change_note="Seeded default workflow",
            )

        # Seed default tool specs if not present
        seeded_tools = {
            art.schema.get("name")
            for art in existing
            if isinstance(art.schema, dict) and art.schema.get("kind") == "tool.spec"
        }
        for spec in get_default_tool_specs():
            if spec["name"] in seeded_tools:
                continue
            self.create_artifact(
                "tool",
                spec["name"],
                spec,
                spec.get("description", ""),
                visibility="global",
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
        normalized_weight = self._safe_float(
            weight if weight is not None else 1.0,
            default=1.0,
            context="record_preference_event_weight",
        )
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
            ensure_policy_compliant_texts(
                (msg_row.get("content"), context_text, corrected_text),
                violation_context={
                    "message_id": message_id,
                    "conversation_id": conversation_id,
                    "user_id": user_id,
                },
            )
            embedding_source = context_embedding or compute_text_embedding(
                context_text or msg_row.get("content")
            )
            try:
                embedding = validated_embedding(
                    embedding_source,
                    expected_dim=EMBEDDING_DIM,
                    name="context_embedding",
                )
            except ValueError as exc:
                raise ConstraintViolation(
                    "invalid context_embedding",
                    {
                        "message_id": message_id,
                        "conversation_id": conversation_id,
                        "error": str(exc),
                    },
                ) from exc
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
                    self._json_param(meta),
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
                row.get("created_at", datetime.now(timezone.utc)) if row else datetime.now(timezone.utc)
            ),
            weight=normalized_weight,
            meta=meta,
        )

    # _text_embedding moved to common.compute_text_embedding

    def list_preference_events(
        self,
        user_id: str | None = None,
        feedback: Iterable[str] | str | None = None,
        cluster_id: str | None = None,
        *,
        tenant_id: str | None = None,
        limit: int = 1000,
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
        # SPEC compliance: Always apply LIMIT to prevent unbounded queries
        query = " ".join([query, "ORDER BY created_at LIMIT %s"])
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_preference_event(row) for row in rows]

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
        return self._row_to_preference_event(row)

    def get_preference_event(self, event_id: str) -> PreferenceEvent | None:
        if not _is_uuid(event_id):
            return None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM preference_event WHERE id = %s", (event_id,)
            ).fetchone()
        if not row:
            return None
        return self._row_to_preference_event(row)

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
        now = datetime.now(timezone.utc)
        existing = self.get_semantic_cluster(cid)
        created_at = existing.created_at if existing else now
        normalized_label = normalize_optional_text(
            label if label is not None else (existing.label if existing else None)
        )
        normalized_description = normalize_optional_text(
            description
            if description is not None
            else (existing.description if existing else None)
        )
        cluster = SemanticCluster(
            id=cid,
            user_id=user_id,
            centroid=list(centroid),
            size=size,
            label=normalized_label,
            description=normalized_description,
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
                    self._format_vector(cluster.centroid) if cluster.centroid else None,
                    size,
                    cluster.label,
                    cluster.description,
                    cluster.sample_message_ids,
                    self._json_param(cluster.meta),
                    created_at,
                    now,
                ),
            ).fetchone()
        return SemanticCluster(
            id=cid,
            user_id=user_id,
            centroid=cluster.centroid,
            size=size,
            label=normalize_optional_text(row.get("label")) if row else cluster.label,
            description=(
                normalize_optional_text(row.get("description"))
                if row
                else cluster.description
            ),
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
        new_label = normalize_optional_text(
            label if label is not None else existing.label
        )
        new_description = normalize_optional_text(
            description if description is not None else existing.description
        )
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
                    new_label,
                    new_description,
                    self._format_vector(new_centroid) if new_centroid else None,
                    new_size,
                    self._json_param(meta if meta is not None else existing.meta),
                    cluster_id,
                ),
            ).fetchone()
        if not row:
            return None
        return self._row_to_semantic_cluster(row)

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
        return [self._row_to_semantic_cluster(row) for row in rows]

    def get_semantic_cluster(self, cluster_id: str) -> SemanticCluster | None:
        if not _is_uuid(cluster_id):
            return None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM semantic_cluster WHERE id = %s", (cluster_id,)
            ).fetchone()
        if not row:
            return None
        return self._row_to_semantic_cluster(row)

    def create_training_job(
        self,
        user_id: str,
        adapter_id: str,
        preference_event_ids: list[str] | None = None,
        dataset_path: str | None = None,
        meta: dict | None = None,
    ) -> TrainingJob:
        job_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
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
                    self._json_param(meta),
                ),
            ).fetchone()
        return self._row_to_training_job(row)

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
        new_updated_at = datetime.now(timezone.utc)
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
                    self._json_param(meta if meta is not None else existing.meta),
                    new_updated_at,
                    job_id,
                ),
            ).fetchone()
        if not row:
            return None
        return self._row_to_training_job(row)

    def claim_training_job(self, job_id: str) -> TrainingJob | None:
        """Atomically claim a training job for processing (Issue 26.2).

        Only claims the job if its status is 'queued'. This prevents race
        conditions where multiple workers could claim the same job.

        Args:
            job_id: The job to claim

        Returns:
            The claimed TrainingJob with status='running' if successful, None if
            the job doesn't exist or was already claimed by another worker.
        """
        now = datetime.now(timezone.utc)
        with self._connect() as conn:
            # Atomic conditional update - only succeeds if status is still 'queued'
            row = conn.execute(
                """
                UPDATE training_job
                SET status = 'running', updated_at = %s
                WHERE id = %s AND status = 'queued'
                RETURNING *
                """,
                (now, job_id),
            ).fetchone()
        if not row:
            # Either job doesn't exist or already claimed
            return None
        return self._row_to_training_job(row)

    def get_training_job(self, job_id: str) -> TrainingJob | None:
        if not _is_uuid(job_id):
            return None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM training_job WHERE id = %s", (job_id,)
            ).fetchone()
        if not row:
            return None
        return self._row_to_training_job(row)

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
        return [self._row_to_training_job(row) for row in rows]

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
        normalized_handle = normalize_optional_text(handle)
        try:
            with self._connect() as conn, conn.transaction():
                conn.execute(
                    """
                    INSERT INTO app_user (id, email, handle, tenant_id, role, plan_tier, is_active, meta)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        user_id,
                        email,
                        normalized_handle,
                        tenant_id,
                        role,
                        plan_tier,
                        is_active,
                        json.dumps(normalized_meta) if normalized_meta else None,
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO user_settings (user_id, locale, timezone, default_voice, default_style, flags)
                    VALUES (%s, NULL, NULL, NULL, NULL, NULL)
                    ON CONFLICT (user_id) DO NOTHING
                    """,
                    (user_id,),
                )
        except errors.UniqueViolation:
            raise ConstraintViolation("email already exists", {"field": "email"})
        return User(
            id=user_id,
            email=email,
            handle=normalized_handle,
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
        return self._row_to_user(row)

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
        if not _is_uuid(user_id):
            return None
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
        if not _is_uuid(user_id):
            return None
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
                    created_at=row.get("created_at", datetime.now(timezone.utc)),
                    meta=row.get("meta"),
                )
                return cfg
        except Exception as exc:
            self.logger.warning("get_user_mfa_secret_failed", error=str(exc))
            return None
        return None

    def _api_key_from_row(self, row) -> ApiKey:
        return ApiKey(
            id=row["id"],
            user_id=row["user_id"],
            name=row["name"],
            prefix=row["prefix"],
            created_at=row["created_at"],
            last_used_at=row.get("last_used_at"),
            revoked_at=row.get("revoked_at"),
        )

    def create_api_key(
        self, user_id: str, *, name: str, key_hash: str, prefix: str
    ) -> ApiKey:
        with self._connect() as conn:
            row = conn.execute(
                """
                INSERT INTO user_api_key (user_id, name, key_hash, prefix)
                VALUES (%s, %s, %s, %s)
                RETURNING id, user_id, name, prefix, created_at, last_used_at, revoked_at
                """,
                (user_id, name, key_hash, prefix),
            ).fetchone()
        return self._api_key_from_row(row)

    def get_api_key_by_hash(self, key_hash: str) -> Optional[ApiKey]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, user_id, name, prefix, created_at, last_used_at, revoked_at "
                "FROM user_api_key WHERE key_hash = %s",
                (key_hash,),
            ).fetchone()
        return self._api_key_from_row(row) if row else None

    def list_api_keys(self, user_id: str) -> List[ApiKey]:
        if not _is_uuid(user_id):
            return []
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, user_id, name, prefix, created_at, last_used_at, revoked_at "
                "FROM user_api_key WHERE user_id = %s ORDER BY created_at DESC",
                (user_id,),
            ).fetchall()
        return [self._api_key_from_row(row) for row in rows]

    def count_active_api_keys(self, user_id: str) -> int:
        if not _is_uuid(user_id):
            return 0
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM user_api_key "
                "WHERE user_id = %s AND revoked_at IS NULL",
                (user_id,),
            ).fetchone()
        return int(row["n"]) if row else 0

    def revoke_api_key(self, key_id: str, *, user_id: str) -> bool:
        """Tombstone the key. Owner-scoped: someone else's id is a miss."""
        if not _is_uuid(key_id) or not _is_uuid(user_id):
            return False
        with self._connect() as conn:
            row = conn.execute(
                "UPDATE user_api_key SET revoked_at = now() "
                "WHERE id = %s AND user_id = %s AND revoked_at IS NULL "
                "RETURNING id",
                (key_id, user_id),
            ).fetchone()
        return row is not None

    def touch_api_key(self, key_id: str) -> None:
        """Best-effort last-used stamp; auth must not fail on it."""
        try:
            with self._connect() as conn:
                conn.execute(
                    "UPDATE user_api_key SET last_used_at = now() WHERE id = %s",
                    (key_id,),
                )
        except Exception as exc:
            self.logger.warning("touch_api_key_failed", error=str(exc))

    def get_user_settings(self, user_id: str) -> Optional[UserSettings]:
        if not _is_uuid(user_id):
            return None
        """Get user settings/preferences."""
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT * FROM user_settings WHERE user_id = %s", (user_id,)
                ).fetchone()
                if not row:
                    return None
                return UserSettings(
                    user_id=str(row["user_id"]),
                    locale=row.get("locale"),
                    timezone=row.get("timezone"),
                    default_voice=row.get("default_voice"),
                    default_style=row.get("default_style"),
                    flags=row.get("flags"),
                )
        except Exception as exc:
            self.logger.warning("get_user_settings_failed", error=str(exc))
            return None

    def set_user_settings(
        self,
        user_id: str,
        *,
        locale: Optional[str] = None,
        timezone: Optional[str] = None,
        default_voice: Optional[str] = None,
        default_style: Optional[dict] = None,
        flags: Optional[dict] = None,
    ) -> UserSettings:
        """Create or update user settings."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO user_settings (user_id, locale, timezone, default_voice, default_style, flags)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id) DO UPDATE SET
                    locale = COALESCE(EXCLUDED.locale, user_settings.locale),
                    timezone = COALESCE(EXCLUDED.timezone, user_settings.timezone),
                    default_voice = COALESCE(EXCLUDED.default_voice, user_settings.default_voice),
                    default_style = COALESCE(EXCLUDED.default_style, user_settings.default_style),
                    flags = COALESCE(EXCLUDED.flags, user_settings.flags)
                """,
                (
                    user_id,
                    locale,
                    timezone,
                    default_voice,
                    json.dumps(default_style) if default_style else None,
                    json.dumps(flags) if flags else None,
                ),
            )
        return UserSettings(
            user_id=user_id,
            locale=locale,
            timezone=timezone,
            default_voice=default_voice,
            default_style=default_style,
            flags=flags,
        )

    def get_user_by_email(self, email: str) -> Optional[User]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM app_user WHERE email = %s", (email,)
            ).fetchone()
        if not row:
            return None
        return self._row_to_user(row)

    def get_user(self, user_id: str) -> Optional[User]:
        if not _is_uuid(user_id):
            return None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM app_user WHERE id = %s", (user_id,)
            ).fetchone()
        if not row:
            return None
        return self._row_to_user(row)

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
        return [self._row_to_user(row) for row in rows]

    def update_user_role(self, user_id: str, role: str) -> Optional[User]:
        with self._connect() as conn:
            row = conn.execute(
                "UPDATE app_user SET role = %s, updated_at = now() WHERE id = %s RETURNING *",
                (role, user_id),
            ).fetchone()
        if not row:
            return None
        return self._row_to_user(row)

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
        return self._row_to_user(row)

    def delete_user(self, user_id: str) -> bool:
        """Delete user and cascade to all related records.

        Per SPEC §12, user deletion must clean up all associated data to prevent
        orphaned records and ensure complete data removal for privacy compliance.

        Deletes (in order to respect foreign key constraints):
        - User MFA config
        - User auth credentials
        - User auth providers
        - Sessions
        - Messages (via conversation)
        - Conversations
        - Knowledge chunks (via context)
        - Context sources (via context)
        - Knowledge contexts
        - Config patches (via artifact)
        - Artifact versions (via artifact)
        - Artifacts
        - Preference events
        - Training jobs
        - Semantic clusters
        - Adapter router state
        - The user record itself
        """
        with self._connect() as conn:
            # Check if user exists first
            exists = conn.execute(
                "SELECT 1 FROM app_user WHERE id = %s", (user_id,)
            ).fetchone()
            if not exists:
                return False

            # Get user's artifacts for cascade (needed for config patches, versions, router state)
            artifact_rows = conn.execute(
                "SELECT id FROM artifact WHERE owner_user_id = %s", (user_id,)
            ).fetchall()
            artifact_ids = [str(row["id"]) for row in artifact_rows]

            # Get user's contexts for cascade (needed for chunks, sources)
            context_rows = conn.execute(
                "SELECT id FROM knowledge_context WHERE owner_user_id = %s", (user_id,)
            ).fetchall()
            context_ids = [str(row["id"]) for row in context_rows]

            # Get user's conversations for cascade (needed for messages)
            conv_rows = conn.execute(
                "SELECT id FROM conversation WHERE user_id = %s", (user_id,)
            ).fetchall()
            conv_ids = [str(row["id"]) for row in conv_rows]

            # Delete in reverse dependency order

            # 1. Delete messages for user's conversations
            if conv_ids:
                conn.execute(
                    "DELETE FROM message WHERE conversation_id = ANY(%s)", (conv_ids,)
                )

            # 2. Delete conversations
            conn.execute("DELETE FROM conversation WHERE user_id = %s", (user_id,))

            # 3. Delete knowledge chunks for user's contexts
            if context_ids:
                conn.execute(
                    "DELETE FROM knowledge_chunk WHERE context_id = ANY(%s)", (context_ids,)
                )

            # 4. Delete context sources for user's contexts
            if context_ids:
                conn.execute(
                    "DELETE FROM context_source WHERE context_id = ANY(%s)", (context_ids,)
                )

            # 5. Delete knowledge contexts
            conn.execute("DELETE FROM knowledge_context WHERE owner_user_id = %s", (user_id,))

            # 6. Delete config patches for user's artifacts
            if artifact_ids:
                conn.execute(
                    "DELETE FROM config_patch WHERE artifact_id = ANY(%s)", (artifact_ids,)
                )

            # 7. Delete artifact versions for user's artifacts
            if artifact_ids:
                conn.execute(
                    "DELETE FROM artifact_version WHERE artifact_id = ANY(%s)", (artifact_ids,)
                )

            # 8. Delete adapter router state for user's artifacts
            if artifact_ids:
                conn.execute(
                    "DELETE FROM adapter_router_state WHERE artifact_id = ANY(%s)", (artifact_ids,)
                )

            # 9. Delete artifacts
            conn.execute("DELETE FROM artifact WHERE owner_user_id = %s", (user_id,))

            # 10. Delete preference events
            conn.execute("DELETE FROM preference_event WHERE user_id = %s", (user_id,))

            # 11. Delete training jobs
            conn.execute("DELETE FROM training_job WHERE user_id = %s", (user_id,))

            # 12. Delete semantic clusters
            conn.execute("DELETE FROM semantic_cluster WHERE user_id = %s", (user_id,))

            # 13. Delete sessions
            conn.execute("DELETE FROM auth_session WHERE user_id = %s", (user_id,))

            # 14. Delete MFA config
            conn.execute("DELETE FROM user_mfa_secret WHERE user_id = %s", (user_id,))

            # 15. Delete auth credentials
            conn.execute("DELETE FROM user_auth_credential WHERE user_id = %s", (user_id,))

            # 16. Delete auth providers
            conn.execute("DELETE FROM user_auth_provider WHERE user_id = %s", (user_id,))

            # 17. Finally delete the user
            result = conn.execute("DELETE FROM app_user WHERE id = %s", (user_id,))

            # Clean up session cache
            with self._session_lock:
                stale_ids = [
                    sid for sid, sess in self.sessions.items() if sess.user_id == user_id
                ]
                for sid in stale_ids:
                    self.sessions.pop(sid, None)

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

    def revoke_user_sessions(
        self, user_id: str, except_session_id: Optional[str] = None
    ) -> None:
        with self._connect() as conn:
            if except_session_id:
                conn.execute(
                    "DELETE FROM auth_session WHERE user_id = %s AND id != %s",
                    (user_id, except_session_id),
                )
            else:
                conn.execute("DELETE FROM auth_session WHERE user_id = %s", (user_id,))
        # Thread-safe iteration over session cache per SPEC §18
        with self._session_lock:
            stale_ids = [
                sid
                for sid, sess in self.sessions.items()
                if sess.user_id == user_id and sid != except_session_id
            ]
            for sid in stale_ids:
                self.sessions.pop(sid, None)

    def mark_session_verified(self, session_id: str) -> None:
        # Issue 53.1: Cache update MUST only happen if DB update succeeds
        # to prevent MFA bypass via transient database failures
        try:
            with self._connect() as conn:
                result = conn.execute(
                    "UPDATE auth_session SET mfa_verified = TRUE WHERE id = %s",
                    (session_id,),
                )
                # If no row was updated, treat as failure and do not mutate cache
                if getattr(result, "rowcount", 0) == 0:
                    raise RuntimeError("session_update_failed")
            # Only update cache after successful DB commit
            self._update_cached_session(session_id, mfa_verified=True)
        except Exception as exc:
            self.logger.error("mark_session_verified_failed", session_id=session_id, error=str(exc))
            raise  # Re-raise to signal failure to caller

    def get_session(self, session_id: str) -> Optional[Session]:
        if not _is_uuid(session_id):
            return None
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
            created_at=row.get("created_at", datetime.now(timezone.utc)),
            expires_at=row.get("expires_at", datetime.now(timezone.utc)),
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

    # notes vault
    @staticmethod
    def _row_to_note(row: dict) -> Note:
        embedding = row.get("embedding")
        if isinstance(embedding, str):
            embedding = json.loads(embedding)
        meta = row.get("meta")
        if isinstance(meta, str):
            meta = json.loads(meta)
        return Note(
            id=str(row["id"]),
            user_id=str(row["user_id"]),
            title=row["title"],
            content=row.get("content") or "",
            embedding=embedding,
            created_at=row.get("created_at"),
            updated_at=row.get("updated_at"),
            meta=meta or {},
        )

    def create_note(
        self,
        user_id: str,
        title: str,
        content: str = "",
        embedding: Optional[List[float]] = None,
        meta: Optional[dict] = None,
    ) -> Note:
        note_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO note (id, user_id, title, content, embedding, created_at, updated_at, meta)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        note_id,
                        user_id,
                        title,
                        content,
                        json.dumps(embedding) if embedding else None,
                        now,
                        now,
                        json.dumps(meta) if meta else None,
                    ),
                )
        except errors.UniqueViolation:
            raise ConstraintViolation("note title already exists", {"field": "title"})
        except errors.ForeignKeyViolation:
            raise ConstraintViolation("note owner missing", {"user_id": user_id})
        return Note(
            id=note_id,
            user_id=user_id,
            title=title,
            content=content,
            embedding=list(embedding) if embedding else None,
            created_at=now,
            updated_at=now,
            meta=meta or {},
        )

    def update_note(
        self,
        note_id: str,
        *,
        title: Optional[str] = None,
        content: Optional[str] = None,
        embedding: Optional[List[float]] = None,
    ) -> Optional[Note]:
        sets = ["updated_at = %s"]
        params: list = [datetime.now(timezone.utc)]
        if title is not None:
            sets.append("title = %s")
            params.append(title)
        if content is not None:
            sets.append("content = %s")
            params.append(content)
        if embedding is not None:
            sets.append("embedding = %s")
            params.append(json.dumps(embedding))
        params.append(note_id)
        try:
            with self._connect() as conn:
                row = conn.execute(
                    f"UPDATE note SET {', '.join(sets)} WHERE id = %s RETURNING *",
                    params,
                ).fetchone()
        except errors.UniqueViolation:
            raise ConstraintViolation("note title already exists", {"field": "title"})
        return self._row_to_note(row) if row else None

    def update_note_meta(self, note_id: str, meta_patch: dict) -> Optional[Note]:
        with self._connect() as conn:
            row = conn.execute(
                """
                UPDATE note
                SET meta = COALESCE(meta, '{}'::jsonb) || %s::jsonb
                WHERE id = %s
                RETURNING *
                """,
                (json.dumps(meta_patch), note_id),
            ).fetchone()
        return self._row_to_note(row) if row else None

    def delete_note(self, note_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "DELETE FROM note WHERE id = %s RETURNING id", (note_id,)
            ).fetchone()
        return row is not None

    def get_note(self, note_id: str, user_id: Optional[str] = None) -> Optional[Note]:
        if not _is_uuid(note_id):
            return None
        with self._connect() as conn:
            if user_id:
                row = conn.execute(
                    "SELECT * FROM note WHERE id = %s AND user_id = %s",
                    (note_id, user_id),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT * FROM note WHERE id = %s", (note_id,)
                ).fetchone()
        return self._row_to_note(row) if row else None

    def get_note_by_title(self, user_id: str, title: str) -> Optional[Note]:
        key = " ".join(str(title or "").split())
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM note WHERE user_id = %s AND lower(title) = lower(%s)",
                (user_id, key),
            ).fetchone()
        return self._row_to_note(row) if row else None

    def list_notes(
        self, user_id: str, limit: int = 200, offset: int = 0
    ) -> List[Note]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM note WHERE user_id = %s
                ORDER BY updated_at DESC LIMIT %s OFFSET %s
                """,
                (user_id, limit, offset),
            ).fetchall()
        return [self._row_to_note(r) for r in rows]

    def count_notes(self, user_id: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT count(*) AS n FROM note WHERE user_id = %s", (user_id,)
            ).fetchone()
        return int(row["n"]) if row else 0

    def set_note_links(self, src_note_id: str, dst_note_ids: List[str]) -> None:
        deduped: List[str] = []
        for dst in dst_note_ids:
            if dst != src_note_id and dst not in deduped:
                deduped.append(dst)
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM note_link WHERE src_note_id = %s", (src_note_id,)
            )
            if deduped:
                # Insert-where-exists instead of catching FK violations: a
                # caught violation still aborts the transaction, which would
                # silently drop every link after the first bad target.
                conn.execute(
                    """
                    INSERT INTO note_link (src_note_id, dst_note_id)
                    SELECT %s, n.id FROM note n WHERE n.id = ANY(%s::uuid[])
                    ON CONFLICT DO NOTHING
                    """,
                    (src_note_id, deduped),
                )

    def list_note_links_from(self, note_id: str) -> List[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT dst_note_id FROM note_link WHERE src_note_id = %s",
                (note_id,),
            ).fetchall()
        return [str(r["dst_note_id"]) for r in rows]

    def list_backlinks(self, note_id: str) -> List[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT src_note_id FROM note_link WHERE dst_note_id = %s",
                (note_id,),
            ).fetchall()
        return [str(r["src_note_id"]) for r in rows]

    def list_note_edges(self, user_id: str) -> List[tuple[str, str]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT l.src_note_id, l.dst_note_id
                FROM note_link l JOIN note n ON n.id = l.src_note_id
                WHERE n.user_id = %s
                """,
                (user_id,),
            ).fetchall()
        return [(str(r["src_note_id"]), str(r["dst_note_id"])) for r in rows]

    def find_notes_with_dangling_link(
        self, user_id: str, title_key: str
    ) -> List[Note]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM note
                WHERE user_id = %s AND meta->'dangling' ? %s
                """,
                (user_id, title_key),
            ).fetchall()
        return [self._row_to_note(r) for r in rows]

    def save_sweep_report(self, user_id: str, report: dict) -> dict:
        with self._connect() as conn:
            row = conn.execute(
                """
                INSERT INTO sweep_report (user_id, report)
                VALUES (%s, %s) RETURNING id, user_id, created_at, report
                """,
                (user_id, json.dumps(report)),
            ).fetchone()
        return {
            "id": str(row["id"]),
            "user_id": str(row["user_id"]),
            "created_at": row["created_at"],
            "report": row["report"] if isinstance(row["report"], dict)
            else json.loads(row["report"]),
        }

    def list_sweep_reports(self, user_id: str, limit: int = 10) -> List[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, user_id, created_at, report FROM sweep_report
                WHERE user_id = %s ORDER BY created_at DESC LIMIT %s
                """,
                (user_id, limit),
            ).fetchall()
        out = []
        for row in rows:
            report = row["report"]
            out.append({
                "id": str(row["id"]),
                "user_id": str(row["user_id"]),
                "created_at": row["created_at"],
                "report": report if isinstance(report, dict) else json.loads(report),
            })
        return out

    # conversations
    def create_conversation(
        self,
        user_id: str,
        title: Optional[str] = None,
        active_context_id: Optional[str] = None,
        meta: Optional[dict] = None,
    ) -> Conversation:
        conv_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO conversation (id, user_id, title, created_at, updated_at, active_context_id, meta) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (
                        conv_id,
                        user_id,
                        title,
                        now,
                        now,
                        active_context_id,
                        json.dumps(meta) if meta else None,
                    ),
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
            meta=meta,
        )

    def get_conversation(
        self, conversation_id: str, *, user_id: Optional[str] = None
    ) -> Optional[Conversation]:
        if not _is_uuid(conversation_id):
            return None
        with self._connect() as conn:
            params: tuple[Any, ...] = (conversation_id,)
            query = "SELECT * FROM conversation WHERE id = %s"
            if user_id:
                query += " AND user_id = %s"
                params = (conversation_id, user_id)
            row = conn.execute(query, params).fetchone()
        if not row:
            return None

        return Conversation(
            id=str(row["id"]),
            user_id=str(row["user_id"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            title=row.get("title"),
            status=row.get("status") or "open",
            active_context_id=row.get("active_context_id"),
            meta=row.get("meta"),
        )

    def set_conversation_title(
        self, conversation_id: str, *, user_id: str, title: str
    ) -> Optional[Conversation]:
        """Rename a conversation; owner-only."""
        with self._connect() as conn:
            row = conn.execute(
                "UPDATE conversation SET title = %s, updated_at = %s "
                "WHERE id = %s AND user_id = %s RETURNING id",
                (title, datetime.now(timezone.utc), conversation_id, user_id),
            ).fetchone()
        return self.get_conversation(conversation_id) if row else None

    def update_message_meta(
        self, message_id: str, *, user_id: str, patch: dict
    ) -> Optional[Any]:
        """Shallow-merge keys into a message's meta; owner-only."""
        with self._connect() as conn:
            row = conn.execute(
                "UPDATE message m SET meta = COALESCE(m.meta, '{}'::jsonb) || %s::jsonb "
                "WHERE m.id = %s AND EXISTS ("
                "  SELECT 1 FROM conversation c WHERE c.id = m.conversation_id AND c.user_id = %s"
                ") RETURNING m.id",
                (json.dumps(patch), message_id, user_id),
            ).fetchone()
        return row or None

    def merge_conversation_meta(
        self, conversation_id: str, *, user_id: str, patch: dict
    ) -> Optional[Conversation]:
        """Shallow-merge keys into a conversation's meta; owner-only."""
        now = datetime.now(timezone.utc)
        with self._connect() as conn:
            row = conn.execute(
                "UPDATE conversation SET meta = COALESCE(meta, '{}'::jsonb) || %s::jsonb, "
                "updated_at = %s WHERE id = %s AND user_id = %s RETURNING id",
                (json.dumps(patch), now, conversation_id, user_id),
            ).fetchone()
        if not row:
            return None
        return self.get_conversation(conversation_id)

    def upsert_conversation_attachment(
        self,
        conversation_id: str,
        *,
        user_id: str,
        record: dict,
        prune_context_id: Optional[str] = None,
        paths_for: Optional[Any] = None,
        generation_prefix: Optional[str] = None,
    ) -> Optional[list]:
        """Add or replace one attachment record, atomically. Owner-only.

        The list is one JSON value holding every attachment, so editing it
        outside a transaction is a read-modify-write on shared state: two
        uploads that both read before either wrote each store their own copy,
        and the later write erases the earlier addition. Measured with two
        filenames uploaded at once, one record disappeared entirely.

        `SELECT ... FOR UPDATE` takes the conversation row, so the second
        writer reads what the first one committed rather than what it saw
        before. A file lock could not have done this — the state is in
        Postgres, and §22 has several replicas sharing exactly that.

        Retiring what this record displaces happens here too, in the same
        transaction and under the same row lock. Attaching a name a second
        time produces a *different* generation, so the new ingestion replaces
        no rows and the previous one stays searchable — but pruning the index
        to an absolute set afterwards is a read-modify-act on shared state.
        Measured with two filenames uploaded at once, the first one's prune
        ran from a snapshot that did not name the second and deleted its
        chunks, with both uploads returning 200.

        So only what *this* record displaces is retired. A generation whose
        record has not been written yet is not unauthorized, it is
        unfinished. `paths_for` maps records to the objects they name, which
        keeps that layout in the service that owns it; the displaced object
        survives if another record still names it, which is what makes two
        names sharing identical bytes work.

        `generation_prefix` additionally retires rows that can never become
        authorized — anything in this context that is not an attachment
        generation at all, which is what these contexts held before the
        store existed.

        Returns the resulting list, or None when the conversation is not this
        user's.
        """
        now = datetime.now(timezone.utc)
        name = record.get("name")
        with self._connect() as conn:
            row = conn.execute(
                "SELECT meta FROM conversation WHERE id = %s AND user_id = %s FOR UPDATE",
                (conversation_id, user_id),
            ).fetchone()
            if not row:
                return None
            meta = dict(row["meta"] or {})
            current = [
                a for a in (meta.get("attachments") or []) if isinstance(a, dict)
            ]
            displaced = [a for a in current if a.get("name") == name]
            attachments = [a for a in current if a.get("name") != name]
            attachments.append(record)
            meta["attachments"] = attachments
            conn.execute(
                "UPDATE conversation SET meta = %s::jsonb, updated_at = %s "
                "WHERE id = %s AND user_id = %s",
                (json.dumps(meta), now, conversation_id, user_id),
            )
            if prune_context_id and paths_for is not None:
                keep = set(paths_for(attachments))
                retired = sorted(set(paths_for(displaced)) - keep)
                if retired:
                    conn.execute(
                        "DELETE FROM knowledge_chunk WHERE context_id = %s "
                        "AND fs_path = ANY(%s)",
                        (prune_context_id, retired),
                    )
                if generation_prefix:
                    conn.execute(
                        "DELETE FROM knowledge_chunk WHERE context_id = %s "
                        "AND (fs_path IS NULL "
                        "     OR left(fs_path, %s) <> %s)",
                        (
                            prune_context_id,
                            len(generation_prefix),
                            generation_prefix,
                        ),
                    )
        return attachments

    def set_conversation_public(
        self, conversation_id: str, *, user_id: str, public: bool
    ) -> Optional[Conversation]:
        """Toggle a conversation's public sharing flag; owner-only."""
        return self.merge_conversation_meta(
            conversation_id, user_id=user_id, patch={"public": bool(public)}
        )

    def get_public_conversation(self, conversation_id: str) -> Optional[Conversation]:
        if not _is_uuid(conversation_id):
            return None
        """Fetch a conversation only if it has been explicitly made public."""
        conv = self.get_conversation(conversation_id)
        if conv and (conv.meta or {}).get("public"):
            return conv
        return None

    def list_public_conversations(self, limit: int = 50) -> List[Conversation]:
        """Public conversations, newest first, for the share directory."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id FROM conversation WHERE (meta->>'public') = 'true' "
                "ORDER BY updated_at DESC LIMIT %s",
                (limit,),
            ).fetchall()
        ids = [str(r["id"]) for r in rows]
        found = (self.get_conversation(cid) for cid in ids)
        return [c for c in found if c]

    def delete_conversation(
        self, conversation_id: str, *, user_id: Optional[str] = None
    ) -> bool:
        """Delete a conversation and its messages atomically."""

        with self._connect() as conn, conn.transaction():
            params: list[Any] = [conversation_id]
            where_clause = "id = %s"
            if user_id:
                where_clause += " AND user_id = %s"
                params.append(user_id)

            deleted = conn.execute(
                f"DELETE FROM conversation WHERE {where_clause} RETURNING id", tuple(params)
            ).fetchone()
            if not deleted:
                return False

            conn.execute(
                "DELETE FROM message WHERE conversation_id = %s", (conversation_id,)
            )
        return True

    def append_message(
        self,
        conversation_id: str,
        sender: str,
        role: str,
        content: str,
        meta: Optional[dict] = None,
        content_struct: Optional[dict] = None,
        message_id: Optional[str] = None,
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
                    # Issue 37.6: Use MAX(seq) instead of COUNT(*) to handle gaps in sequence
                    seq_row = conn.execute(
                        "SELECT COALESCE(MAX(seq), -1) + 1 AS next_seq FROM message WHERE conversation_id = %s",
                        (conversation_id,),
                    ).fetchone()
                    seq = seq_row["next_seq"] if seq_row else 0
                    # A caller-minted id lets the streaming Responses surface
                    # announce the id before the row exists; anything invalid
                    # falls back to a fresh one rather than a failed INSERT.
                    msg_id = (
                        message_id
                        if message_id and _is_uuid(message_id)
                        else str(uuid.uuid4())
                    )
                    now = datetime.now(timezone.utc)
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

    def get_message(self, message_id: str) -> Optional[Message]:
        """One message by id, or None.

        Training needs the target's sequence *and* its content (SPEC §5.4.2),
        and can get neither by scanning a fetch window: an older target simply
        is not in the newest N messages, so the bound silently does nothing
        and the fallback target text silently disappears.
        """
        if not _is_uuid(message_id):
            return None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM message WHERE id = %s", (message_id,)
            ).fetchone()
        return self._message_from_row(row) if row else None

    def list_messages_before(
        self, conversation_id: str, seq: int, *, limit: int = 200
    ) -> List[Message]:
        """The messages preceding ``seq``, oldest-first, bounded to the newest
        ``limit`` of them — the conversation as it stood when that turn was
        written."""
        if not _is_uuid(conversation_id):
            return []
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM message WHERE conversation_id = %s AND seq < %s "
                "ORDER BY seq DESC LIMIT %s",
                (conversation_id, seq, limit),
            ).fetchall()
        return [self._message_from_row(row) for row in reversed(rows)]

    def get_message_conversation(self, message_id: str) -> Optional[str]:
        """The conversation a message belongs to, or None.

        Purpose-built for the served Responses API, which resolves a
        ``previous_response_id`` back to the conversation it continues. Only
        the id comes back: continuity needs nothing else, and ownership is
        the caller's check to make against the conversation itself.
        """
        if not _is_uuid(message_id):
            return None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT conversation_id FROM message WHERE id = %s",
                (message_id,),
            ).fetchone()
        return str(row["conversation_id"]) if row else None

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
        return [self._message_from_row(row) for row in reversed(rows)]

    def _message_from_row(self, row) -> Message:
        content_struct = row.get("content_struct")
        if isinstance(content_struct, str):
            try:
                content_struct = json.loads(content_struct)
            except Exception:
                content_struct = None
        content_struct = normalize_content_struct(content_struct, row.get("content"))
        meta = row.get("meta")
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = None
        return Message(
            id=str(row["id"]),
            conversation_id=str(row["conversation_id"]),
            sender=row["sender"],
            role=row["role"],
            content=row["content"],
            content_struct=content_struct,
            seq=row["seq"],
            created_at=row.get("created_at", datetime.now(timezone.utc)),
            meta=meta,
        )

    def list_conversations(
        self, user_id: str, limit: int = 20, offset: int = 0
    ) -> List[Conversation]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM conversation
                WHERE user_id = %s
                ORDER BY updated_at DESC
                LIMIT %s OFFSET %s
                """,
                (user_id, limit, offset),
            ).fetchall()
        conversations: List[Conversation] = []
        for row in rows:
            conversations.append(
                Conversation(
                    id=str(row["id"]),
                    user_id=str(row["user_id"]),
                    created_at=row.get("created_at", datetime.now(timezone.utc)),
                    updated_at=row.get("updated_at", datetime.now(timezone.utc)),
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
        cursor: Optional[str] = None,
        include_sentinel: bool = False,
        owner_user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        visibility: Optional[str] = None,
    ) -> List[Artifact]:
        """List artifacts with proper visibility filtering.

        Visibility logic:
        - If visibility='private': only user's own private artifacts
        - If visibility='global': all global artifacts
        - If visibility='shared': shared artifacts within user's tenant
        - If no visibility filter: user's private + all global + shared within tenant
        """
        # SPEC pagination: default 100, cap 500 to avoid unbounded scans
        max_page_size = 500
        requested_page_size = max(page_size, 1)
        capped_page_size = min(requested_page_size, max_page_size)
        # Optionally fetch one extra record for has_next detection
        limit = capped_page_size + (1 if include_sentinel else 0)
        offset = 0 if cursor else max(page - 1, 0) * capped_page_size
        cursor_filter: list[str] = []
        cursor_params: list[Any] = []
        if cursor:
            try:
                created_at_cursor, artifact_cursor_id = decode_artifact_cursor(cursor)
                cursor_filter.append("(created_at, id) < (%s, %s)")
                cursor_params.extend([created_at_cursor, artifact_cursor_id])
            except Exception as exc:
                self.logger.warning("artifact_cursor_decode_failed", error=str(exc))
        # An explicit visibility filter needs the identity that scopes it.
        # Without one, the branches below dropped the scoping clause and
        # returned *every* user's private rows, or every tenant's shared ones.
        # Unreachable from `/v1/artifacts` — `app_user.tenant_id` is NOT NULL
        # and the route always passes the caller — but it is the same
        # fail-open default that `get_latest_workflow` shipped with, so it
        # narrows here rather than waiting for a caller to find it.
        if (visibility == "private" and not owner_user_id) or (
            visibility == "shared" and not tenant_id
        ):
            self.logger.warning(
                "artifact_list_unscoped",
                visibility=visibility,
                owner_user_id=owner_user_id,
                tenant_id=tenant_id,
            )
            return []
        with self._connect() as conn:
            clauses = []
            params: list[Any] = []

            # Type/kind filters apply regardless of visibility
            if type_filter:
                clauses.append("type = %s")
                params.append(type_filter)
            if kind_filter:
                clauses.append("schema->>'kind' = %s")
                params.append(kind_filter)

            # Build visibility access control clause
            if visibility:
                # Specific visibility filter requested
                if visibility == "private":
                    # Only user's own private artifacts
                    if owner_user_id:
                        clauses.append("(visibility = 'private' AND owner_user_id = %s)")
                        params.append(owner_user_id)
                    else:
                        clauses.append("visibility = 'private'")
                elif visibility == "global":
                    # All global artifacts (visible to everyone)
                    clauses.append("visibility = 'global'")
                elif visibility == "shared":
                    # Shared artifacts within tenant
                    if tenant_id:
                        clauses.append(
                            "(visibility = 'shared' AND owner_user_id IN "
                            "(SELECT id FROM app_user WHERE tenant_id = %s))"
                        )
                        params.append(tenant_id)
                    else:
                        clauses.append("visibility = 'shared'")
            else:
                # No visibility filter: show accessible artifacts
                # User sees: their private + all global + shared within tenant
                visibility_parts = []
                if owner_user_id:
                    visibility_parts.append("(visibility = 'private' AND owner_user_id = %s)")
                    params.append(owner_user_id)
                visibility_parts.append("visibility = 'global'")
                if tenant_id:
                    visibility_parts.append(
                        "(visibility = 'shared' AND owner_user_id IN "
                        "(SELECT id FROM app_user WHERE tenant_id = %s))"
                    )
                    params.append(tenant_id)
                if visibility_parts:
                    clauses.append("(" + " OR ".join(visibility_parts) + ")")

            if cursor_filter:
                clauses.extend(cursor_filter)

            where = " WHERE " + " AND ".join(clauses) if clauses else ""
            query = (
                "SELECT * FROM artifact"
                + where
                + " ORDER BY created_at DESC, id DESC LIMIT %s OFFSET %s"
            )
            params.extend(cursor_params)
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
                    created_at=row.get("created_at", datetime.now(timezone.utc)),
                    updated_at=row.get("updated_at", datetime.now(timezone.utc)),
                    fs_path=row.get("fs_path"),
                    base_model=row.get("base_model")
                    or (row.get("schema") or {}).get("base_model"),
                    meta=row.get("meta"),
                )
            )
        return artifacts

    def get_artifact(self, artifact_id: str) -> Optional[Artifact]:
        if not _is_uuid(artifact_id):
            return None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM artifact WHERE id = %s", (artifact_id,)
            ).fetchone()
        if not row:
            return None
        schema = row.get("schema")
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
            created_at=row.get("created_at", datetime.now(timezone.utc)),
            updated_at=row.get("updated_at", datetime.now(timezone.utc)),
            fs_path=row.get("fs_path"),
            base_model=row.get("base_model") or (schema or {}).get("base_model"),
            meta=row.get("meta"),
        )

    def artifacts_for_paths(self, paths: Sequence[str]) -> List[Artifact]:
        """Artifacts whose `fs_path` is exactly one of `paths`.

        The caller passes a filesystem path and its ancestors, so an artifact
        naming a corpus directory answers for the files inside it while an
        artifact naming a sibling directory does not. Exact matches rather than
        a `LIKE` prefix: `/shared/corpus` must not match `/shared/corpus-2`,
        and a prefix comparison on strings says it does.
        """
        wanted = [p for p in paths if p]
        if not wanted:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id FROM artifact WHERE fs_path = ANY(%s)", (list(wanted),)
            ).fetchall()
        found = [self.get_artifact(str(row["id"])) for row in rows]
        return [artifact for artifact in found if artifact is not None]

    def create_artifact(
        self,
        type_: str,
        name: str,
        schema: dict,
        description: str = "",
        owner_user_id: Optional[str] = None,
        visibility: str = "private",
        *,
        version_author: Optional[str] = None,
        change_note: Optional[str] = None,
    ) -> Artifact:
        normalized_description = normalize_optional_text(description)
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
                    "INSERT INTO artifact (id, owner_user_id, type, name, description, schema, fs_path, base_model, visibility) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    (
                        artifact_id,
                        owner_user_id,
                        type_,
                        name,
                        normalized_description,
                        json.dumps(schema),
                        fs_path,
                        schema.get("base_model"),
                        visibility,
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
            description=normalized_description or "",
            schema=schema,
            owner_user_id=owner_user_id,
            visibility=visibility,
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
            # Issue 19.5: Use SELECT ... FOR UPDATE to prevent race condition
            # This locks the artifact row until the transaction completes,
            # preventing concurrent version inserts from calculating the same next_version
            row = conn.execute(
                "SELECT * FROM artifact WHERE id = %s FOR UPDATE", (artifact_id,)
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

    def list_artifact_versions(
        self, artifact_id: str, *, limit: Optional[int] = None
    ) -> List[ArtifactVersion]:
        with self._connect() as conn:
            query = "SELECT * FROM artifact_version WHERE artifact_id = %s ORDER BY version DESC"
            params: list[Any] = [artifact_id]
            if limit is not None:
                query += " LIMIT %s"
                params.append(limit)
            rows = conn.execute(query, tuple(params)).fetchall()
        versions: List[ArtifactVersion] = []
        for row in rows:
            schema = row.get("schema")
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
                    created_at=row.get("created_at", datetime.now(timezone.utc)),
                    fs_path=row.get("fs_path"),
                    base_model=row.get("base_model")
                    or (schema or {}).get("base_model"),
                    meta=row.get("meta"),
                )
            )
        return versions

    def get_artifact_current_version(self, artifact_id: str) -> int:
        """Get the current (highest) version number for an artifact."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT MAX(version) as max_version FROM artifact_version WHERE artifact_id = %s",
                (artifact_id,),
            ).fetchone()
        if row and row.get("max_version"):
            return row["max_version"]
        return 1

    def get_artifact_current_versions(self, artifact_ids: List[str]) -> Dict[str, int]:
        """Get current versions for multiple artifacts efficiently."""
        if not artifact_ids:
            return {}
        with self._connect() as conn:
            # Use a single query with GROUP BY for efficiency
            placeholders = ", ".join(["%s"] * len(artifact_ids))
            rows = conn.execute(
                f"SELECT artifact_id, MAX(version) as max_version FROM artifact_version "
                f"WHERE artifact_id IN ({placeholders}) GROUP BY artifact_id",
                tuple(artifact_ids),
            ).fetchall()
        result: Dict[str, int] = {aid: 1 for aid in artifact_ids}  # Default to 1
        for row in rows:
            result[str(row["artifact_id"])] = row["max_version"]
        return result

    def persist_artifact_payload(self, artifact_id: str, schema: dict) -> str:
        with self._connect() as conn, conn.transaction():
            artifact_row = conn.execute(
                "SELECT id, schema, base_model FROM artifact WHERE id = %s FOR UPDATE",
                (artifact_id,),
            ).fetchone()
            if not artifact_row:
                raise ConstraintViolation(
                    "artifact missing", {"artifact_id": artifact_id}
                )
            version_row = conn.execute(
                "SELECT COALESCE(MAX(version), 0) AS v FROM artifact_version WHERE artifact_id = %s FOR UPDATE",
                (artifact_id,),
            ).fetchone()
            next_version = (version_row["v"] or 0) + 1
            fs_path = self._persist_payload(artifact_id, next_version, schema)

            base_model = schema.get("base_model")
            existing_schema = artifact_row.get("schema")
            if not base_model and existing_schema:
                if isinstance(existing_schema, str):
                    try:
                        existing_schema = json.loads(existing_schema)
                    except Exception:
                        existing_schema = {}
                base_model = (existing_schema or {}).get("base_model")

            conn.execute(
                "UPDATE artifact SET schema = %s, fs_path = %s, base_model = %s, updated_at = now() WHERE id = %s",
                (json.dumps(schema), fs_path, base_model, artifact_id),
            )
            conn.execute(
                "INSERT INTO artifact_version (artifact_id, version, schema, fs_path, base_model, created_by, change_note) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (
                    artifact_id,
                    next_version,
                    json.dumps(schema),
                    fs_path,
                    base_model,
                    "system_llm",
                    None,
                ),
            )
        return fs_path

    def _deny_workflow(self, workflow_id, user_id, owner_user_id, visibility) -> None:
        self.logger.warning(
            "workflow_access_denied",
            workflow_id=workflow_id,
            user_id=user_id,
            owner_user_id=owner_user_id,
            visibility=visibility,
        )

    def get_latest_workflow(
        self,
        workflow_id: str,
        *,
        user_id: Optional[str],
        tenant_id: Optional[str] = None,
    ) -> Optional[dict]:
        """The newest version of a workflow this caller may run, or None.

        `user_id` is required rather than optional: `workflow_id` arrives in a
        request body, and this method used to select on `artifact_id` alone,
        so naming another user's private workflow ran it. A keyword with no
        default means a caller cannot omit the question by accident — the two
        callers that existed both had the identity to hand and neither passed
        it.
        """
        artifact = self.get_artifact(workflow_id)
        if artifact is None:
            return None
        visibility = getattr(artifact, "visibility", "private")
        owner_id = getattr(artifact, "owner_user_id", None)
        if visibility == "private":
            # Ownerless too: an artifact nobody owns cannot be shown to be
            # this caller's, and the previous form only refused when an owner
            # was present, so a null owner served everyone.
            if not owner_id or owner_id != user_id:
                self._deny_workflow(workflow_id, user_id, owner_id, visibility)
                return None
        elif visibility == "shared":
            # `shared` means within a tenant, and the tenant is the owner's —
            # `Artifact` has no tenant column, so the previous
            # `getattr(artifact, "tenant_id", None)` was always None and the
            # `None in (...)` acceptance served shared workflows across
            # tenants. Read it from the owner, the way list_artifacts does.
            owner = self.get_user(owner_id) if owner_id else None
            if not owner or not tenant_id or owner.tenant_id != tenant_id:
                self._deny_workflow(workflow_id, user_id, owner_id, visibility)
                return None
        elif visibility != "global":
            # An unrecognized visibility is not a licence.
            self._deny_workflow(workflow_id, user_id, owner_id, visibility)
            return None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT schema FROM artifact_version WHERE artifact_id = %s ORDER BY version DESC LIMIT 1",
                (workflow_id,),
            ).fetchone()
        if not row:
            return None
        schema = row.get("schema")
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
            meta = row.get("meta")
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except Exception:
                    meta = None
            centroid_vec = (
                self._parse_vector(row.get("centroid_vec"))
            )
            states.append(
                AdapterRouterState(
                    artifact_id=str(row["artifact_id"]),
                    base_model=row.get("base_model"),
                    centroid_vec=centroid_vec if centroid_vec is not None else [],
                    usage_count=(
                        row.get("usage_count", 0)
                    ),
                    success_score=(
                        row.get("success_score", 0.0)
                    ),
                    last_used_at=(
                        row.get("last_used_at")
                    ),
                    last_trained_at=(
                        row.get("last_trained_at")
                    ),
                    meta=meta,
                )
            )
        return states

    def update_adapter_router_state(
        self,
        adapter_id: str,
        *,
        centroid_vec: Optional[list[float]] = None,
        success_score: Optional[float] = None,
        last_used_at: Optional[datetime] = None,
        last_trained_at: Optional[datetime] = None,
    ) -> AdapterRouterState:
        """Upsert adapter router state with EMA centroids and bounded scores."""

        with self._connect() as conn, conn.transaction():
            artifact_row = conn.execute(
                "SELECT id, base_model FROM artifact WHERE id = %s FOR UPDATE",
                (adapter_id,),
            ).fetchone()
            if not artifact_row:
                raise ConstraintViolation(
                    "adapter missing for router state", {"adapter_id": adapter_id}
                )

            existing = conn.execute(
                "SELECT * FROM adapter_router_state WHERE artifact_id = %s FOR UPDATE",
                (adapter_id,),
            ).fetchone()

            merged_centroid = blend_centroid(
                self._parse_vector(existing.get("centroid_vec")) if existing else None,
                centroid_vec,
            )
            merged_success = clamp_success_score(
                success_score
                if success_score is not None
                else (existing.get("success_score") if existing else 0.0)
            )
            usage_count = existing.get("usage_count", 0) if existing else 0
            merged_last_used = last_used_at or (existing.get("last_used_at") if existing else None)
            merged_last_trained = last_trained_at or (
                existing.get("last_trained_at") if existing else None
            )
            meta = existing.get("meta") if existing else None
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except Exception:
                    meta = None

            if existing:
                conn.execute(
                    "UPDATE adapter_router_state SET centroid_vec = %s, usage_count = %s, "
                    "success_score = %s, last_used_at = %s, last_trained_at = %s, meta = %s "
                    "WHERE artifact_id = %s",
                    (
                        self._format_vector(merged_centroid) if merged_centroid else None,
                        usage_count,
                        merged_success,
                        merged_last_used,
                        merged_last_trained,
                        json.dumps(meta) if isinstance(meta, dict) else meta,
                        adapter_id,
                    ),
                )
            else:
                conn.execute(
                    "INSERT INTO adapter_router_state (artifact_id, centroid_vec, usage_count, success_score, last_used_at, last_trained_at, meta) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (
                        adapter_id,
                        self._format_vector(merged_centroid) if merged_centroid else None,
                        usage_count,
                        merged_success,
                        merged_last_used,
                        merged_last_trained,
                        json.dumps(meta) if isinstance(meta, dict) else meta,
                    ),
                )

        return AdapterRouterState(
            artifact_id=adapter_id,
            base_model=artifact_row.get("base_model") if artifact_row else None,
            centroid_vec=merged_centroid or None,
            usage_count=usage_count,
            success_score=merged_success,
            last_used_at=merged_last_used,
            last_trained_at=merged_last_trained,
            meta=meta,
        )

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
                row.get("status", "pending")
            ),
            created_at=(
                row.get("created_at", datetime.now(timezone.utc))
            ),
            decided_at=(
                row.get("decided_at")
            ),
            applied_at=(
                row.get("applied_at")
            ),
            meta=row.get("meta"),
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
            now = datetime.now(timezone.utc)
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

    def apply_config_patch(
        self,
        patch: ConfigPatchAudit,
        new_schema: dict,
        *,
        artifact_description: Optional[str] = None,
        approver_user_id: Optional[str] = None,
    ) -> tuple[Artifact, ConfigPatchAudit]:
        """Atomically persist a config patch application and mark it applied."""

        with self._connect() as conn, conn.transaction():
            artifact_row = conn.execute(
                "SELECT * FROM artifact WHERE id = %s FOR UPDATE", (patch.artifact_id,)
            ).fetchone()
            if not artifact_row:
                raise NotFoundError("artifact missing", detail={"artifact_id": patch.artifact_id})

            versions = conn.execute(
                "SELECT COALESCE(MAX(version), 0) AS v FROM artifact_version WHERE artifact_id = %s",
                (patch.artifact_id,),
            ).fetchone()
            next_version = (versions["v"] or 0) + 1
            fs_path = self._persist_payload(patch.artifact_id, next_version, new_schema)
            base_model = new_schema.get("base_model") or artifact_row.get("base_model")

            conn.execute(
                "UPDATE artifact SET schema = %s, description = COALESCE(%s, description), updated_at = now(), fs_path = %s, base_model = %s WHERE id = %s",
                (json.dumps(new_schema), artifact_description, fs_path, base_model, patch.artifact_id),
            )
            conn.execute(
                "INSERT INTO artifact_version (artifact_id, version, schema, fs_path, base_model, created_by, change_note) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (
                    patch.artifact_id,
                    next_version,
                    json.dumps(new_schema),
                    fs_path,
                    base_model,
                    approver_user_id or patch.proposer,
                    patch.justification,
                ),
            )

            merged_meta: Dict[str, Any] = {}
            if patch.meta:
                if isinstance(patch.meta, dict):
                    merged_meta.update(patch.meta)
                else:
                    try:
                        parsed = json.loads(patch.meta)
                        if isinstance(parsed, dict):
                            merged_meta.update(parsed)
                    except Exception:
                        merged_meta = {}
            if approver_user_id:
                merged_meta["applied_by"] = approver_user_id

            conn.execute(
                "UPDATE config_patch SET status = %s, applied_at = now(), meta = %s WHERE id = %s",
                ("applied", json.dumps(merged_meta) if merged_meta else json.dumps({}), patch.id),
            )
            refreshed = conn.execute(
                "SELECT * FROM config_patch WHERE id = %s", (patch.id,)
            ).fetchone()

        updated_artifact = Artifact(
            id=str(artifact_row["id"]),
            type=artifact_row["type"],
            name=artifact_row["name"],
            description=artifact_description or artifact_row.get("description") or "",
            schema=new_schema,
            owner_user_id=(
                str(artifact_row["owner_user_id"]) if artifact_row.get("owner_user_id") else None
            ),
            visibility=artifact_row.get("visibility", "private"),
            fs_path=fs_path,
            base_model=base_model,
        )
        return updated_artifact, self._config_patch_from_row(refreshed)

    def _config_patch_from_row(self, row) -> ConfigPatchAudit:
        raw_patch = row.get("patch")
        patch_data = (
            raw_patch if isinstance(raw_patch, dict) else json.loads(raw_patch or "{}")
        )
        meta = row.get("meta")
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception as exc:
                self.logger.warning("config_patch_meta_parse_failed", error=str(exc))
                meta = {}
        decided_at = (
            row.get("decided_at")
        )
        applied_at = (
            row.get("applied_at")
        )
        created = row.get("created_at")
        return ConfigPatchAudit(
            id=int(row["id"]),
            artifact_id=str(row["artifact_id"]),
            proposer=row.get("proposer"),
            patch=patch_data,
            justification=(
                row.get("justification")
            ),
            status=(
                row.get("status", "pending")
            ),
            created_at=self._parse_ts(created) or datetime.now(timezone.utc),
            decided_at=self._parse_ts(decided_at),
            applied_at=self._parse_ts(applied_at),
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

    def _safe_float(self, value: Any, default: float = 1.0, *, context: str = "") -> float:
        """Parse floats defensively to avoid crashes on malformed data (Issue 39.3)."""

        try:
            return float(value)
        except (TypeError, ValueError):
            self.logger.warning("postgres_float_parse_failed", context=context, value=value)
            return default

    def get_system_settings(self) -> dict:
        """Get admin-managed system settings from database.

        Returns settings for session rotation, concurrency caps, and rate limit
        multipliers. These are managed via the admin UI instead of env vars.
        Uses SYSTEM_SETTINGS_DEFAULTS from config.py as single source of truth.
        """
        # Always merge stored overrides over defaults so callers get a complete,
        # current settings dict even when the row is partial, absent, or corrupt.
        # Secrets are blanked: this feeds the admin API, and a value that is
        # echoed back to every admin and into every log is not a secret.
        return redact_secrets(
            {**SYSTEM_SETTINGS_DEFAULTS, **self._get_stored_system_settings()}
        )

    def get_system_settings_raw(self) -> dict:
        """Merged settings including secrets. For the runtime, not the API."""
        return {**SYSTEM_SETTINGS_DEFAULTS, **self._get_stored_system_settings()}

    def get_system_settings_overrides(self) -> dict:
        """Explicitly stored admin settings only, no defaults merged in.

        Lets the runtime give env vars precedence over code defaults for
        settings the admin never actually overrode.
        """
        return dict(self._get_stored_system_settings())

    def get_system_settings_version(self) -> Optional[str]:
        """Return a token that changes whenever system settings are written.

        Uses instance_config.updated_at (bumped on every set) so every Uvicorn
        worker, which holds its own in-process Runtime, can detect a settings
        change made by another worker and reload its model services.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT updated_at FROM instance_config WHERE name = %s",
                ("system_settings",),
            ).fetchone()
        if not row:
            return None
        updated_at = row.get("updated_at")
        return updated_at.isoformat() if updated_at else None

    def _get_stored_system_settings(self) -> dict:
        """Return only the explicitly-persisted settings (no defaults merged).

        Defaults are never baked into storage, so a future change to a default
        propagates to keys the admin never overrode.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT config FROM instance_config WHERE name = %s",
                ("system_settings",),
            ).fetchone()
        return self._coerce_stored_settings(row)

    @staticmethod
    def _coerce_stored_settings(row: Any) -> dict:
        if not row:
            return {}
        raw_config = row.get("config")
        if isinstance(raw_config, str):
            try:
                parsed = json.loads(raw_config)
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                return {}
        return raw_config if isinstance(raw_config, dict) else {}

    def get_instance_config(self, name: str) -> dict:
        """Read a named JSONB blob from instance_config ({} when absent)."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT config FROM instance_config WHERE name = %s", (name,)
            ).fetchone()
        if not row:
            return {}
        config = row.get("config")
        if isinstance(config, str):
            try:
                config = json.loads(config)
            except Exception:  # noqa: BLE001
                return {}
        return config if isinstance(config, dict) else {}

    def merge_instance_config(self, name: str, patch: dict) -> dict:
        """Merge keys into a named blob atomically; returns the merged dict."""
        with self._connect() as conn, conn.transaction():
            row = conn.execute(
                "SELECT config FROM instance_config WHERE name = %s FOR UPDATE",
                (name,),
            ).fetchone()
            current = row.get("config") if row else {}
            if isinstance(current, str):
                try:
                    current = json.loads(current)
                except Exception:  # noqa: BLE001
                    current = {}
            merged = {**(current if isinstance(current, dict) else {}), **patch}
            conn.execute(
                """
                INSERT INTO instance_config (name, config, created_at, updated_at)
                VALUES (%s, %s, now(), now())
                ON CONFLICT (name) DO UPDATE
                SET config = EXCLUDED.config, updated_at = now()
                """,
                (name, json.dumps(merged)),
            )
        return merged

    def set_system_settings(self, settings: dict) -> dict:
        """Update admin-managed system settings.

        Reads and writes the stored overrides in one transaction so concurrent
        admin updates don't clobber each other, and never persists defaults.
        Returns the full effective settings (defaults + overrides).
        """
        with self._connect() as conn, conn.transaction():
            row = conn.execute(
                "SELECT config FROM instance_config WHERE name = %s FOR UPDATE",
                ("system_settings",),
            ).fetchone()
            merged = {**self._coerce_stored_settings(row), **settings}
            conn.execute(
                """
                INSERT INTO instance_config (name, config, created_at, updated_at)
                VALUES (%s, %s, now(), now())
                ON CONFLICT (name) DO UPDATE SET
                    config = EXCLUDED.config,
                    updated_at = now()
                """,
                ("system_settings", json.dumps(merged)),
            )
        return {**SYSTEM_SETTINGS_DEFAULTS, **merged}

    # knowledge
    def get_conversation_attachment_context(
        self, owner_user_id: str, conversation_id: str
    ) -> Optional[KnowledgeContext]:
        """The conversation's implicit index, by identity rather than by page.

        This used to be "the first row a 500-context listing matched", which
        is not an identity lookup: an account with more recent contexts than
        the page holds lost an older conversation's index entirely, and the
        conversation's own attachments stopped being searchable while their
        records and objects were still perfectly intact.
        """
        if not _is_uuid(owner_user_id):
            return None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM knowledge_context "
                "WHERE owner_user_id = %s "
                "AND COALESCE((meta ->> 'auto')::boolean, false) "
                "AND meta ->> 'conversation_id' = %s "
                "ORDER BY created_at ASC, id ASC LIMIT 1",
                (owner_user_id, str(conversation_id)),
            ).fetchone()
        return self._context_from_row(row) if row else None

    def get_contexts_for_scope(
        self, owner_user_id: str, context_ids: Sequence[str]
    ) -> List[KnowledgeContext]:
        """The named contexts this user owns, by identity rather than by page.

        Authorization is a question about particular ids, and answering it
        with `list_contexts` answered a different one: whether those ids are
        near the top of a listing. That listing pages at 100 rows in SQL, so
        a context the request had already validated by direct lookup dropped
        out of retrieval once the account had a hundred newer ones — the turn
        succeeded and the model was given no grounding.

        Implicit conversation indexes are excluded here as everywhere else:
        they are reachable only through the conversation that owns them.
        """
        wanted = [ctx_id for ctx_id in context_ids or [] if _is_uuid(ctx_id)]
        if not wanted or not _is_uuid(owner_user_id):
            return []
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM knowledge_context "
                "WHERE owner_user_id = %s AND id = ANY(%s::uuid[]) "
                "AND COALESCE((meta ->> 'auto')::boolean, false) IS FALSE",
                (owner_user_id, wanted),
            ).fetchall()
        return [self._context_from_row(row) for row in rows]

    def get_or_create_conversation_attachment_context(
        self, owner_user_id: str, conversation_id: str, name: str, description: str
    ) -> KnowledgeContext:
        """That context, creating it once however many callers ask at once.

        Lookup-then-insert in one process is not a guard when Postgres is
        shared across replicas (§22), and it was not one within a process
        either: two first attachments both looked, both found nothing, and
        both inserted. Measured, the conversation ended up with two hidden
        contexts and one of the two acknowledged attachments was searchable
        from neither, because the later lookup returns one row.

        The database decides. `ON CONFLICT DO NOTHING` against the partial
        unique index means the loser inserts nothing and then reads the
        winner, so every caller comes back with the same context.
        """
        existing = self.get_conversation_attachment_context(
            owner_user_id, conversation_id
        )
        if existing is not None:
            return existing
        meta = {"auto": True, "conversation_id": str(conversation_id)}
        ctx_id = str(uuid.uuid4())
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO knowledge_context "
                    "(id, owner_user_id, name, description, meta) "
                    "VALUES (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                    (
                        ctx_id,
                        owner_user_id,
                        name,
                        description,
                        self._json_param(meta),
                    ),
                )
        except errors.ForeignKeyViolation:
            raise ConstraintViolation(
                "context owner missing", {"owner_user_id": owner_user_id}
            )
        created = self.get_conversation_attachment_context(
            owner_user_id, conversation_id
        )
        if created is None:
            raise ConstraintViolation(
                "conversation context could not be created",
                {"conversation_id": conversation_id},
            )
        return created

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
                    (ctx_id, owner_user_id, name, description, fs_path, self._json_param(meta)),
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

    @staticmethod
    def _context_from_row(row) -> KnowledgeContext:
        return KnowledgeContext(
            id=str(row["id"]),
            owner_user_id=str(row["owner_user_id"]),
            name=row["name"],
            description=row.get("description") or "",
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            fs_path=row.get("fs_path"),
            meta=row.get("meta"),
        )

    def get_context(self, context_id: str) -> Optional[KnowledgeContext]:
        if not _is_uuid(context_id):
            return None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM knowledge_context WHERE id = %s", (context_id,)
            ).fetchone()
        return self._context_from_row(row) if row else None

    def list_contexts(
        self,
        owner_user_id: Optional[str] = None,
        *,
        page: int = 1,
        page_size: int = 100,
        cursor: Optional[str] = None,
        include_sentinel: bool = False,
        limit: Optional[int] = None,
        include_auto: bool = True,
    ) -> List[KnowledgeContext]:
        """This user's contexts, ordered newest first.

        `include_auto` decides whether conversations' implicit indexes are
        part of the domain. Dropping them from the *result* instead made
        pagination lie: the ordering and the LIMIT had already happened, so a
        page whose sentinel row was an implicit context reported no next page
        with ordinary contexts still unreached.
        """
        if not owner_user_id:
            return []

        effective_page_size = max(1, limit or page_size)
        capped_page_size = min(effective_page_size, 500)
        fetch_limit = capped_page_size + (1 if include_sentinel else 0)
        offset = 0 if cursor else max(page - 1, 0) * capped_page_size

        cursor_filter = ""
        cursor_params: list[Any] = []
        if cursor:
            try:
                cursor_ts, cursor_id = decode_time_id_cursor(cursor)
                cursor_filter = " AND (created_at, id) < (%s, %s)"
                cursor_params.extend([cursor_ts, cursor_id])
            except Exception as exc:  # pragma: no cover - defensive
                self.logger.warning("context_cursor_decode_failed", error=str(exc))

        auto_filter = (
            ""
            if include_auto
            else " AND COALESCE((meta ->> 'auto')::boolean, false) IS FALSE"
        )
        with self._connect() as conn:
            query = (
                "SELECT * FROM knowledge_context WHERE owner_user_id = %s"
                + auto_filter
                + cursor_filter
                + " ORDER BY created_at DESC, id DESC LIMIT %s OFFSET %s"
            )
            params: list[Any] = [owner_user_id]
            params.extend(cursor_params)
            params.extend([fetch_limit, offset])
            rows = conn.execute(query, tuple(params)).fetchall()

        contexts: List[KnowledgeContext] = []
        for row in rows:
            contexts.append(
                KnowledgeContext(
                    id=str(row["id"]),
                    owner_user_id=str(row["owner_user_id"]),
                    name=row["name"],
                    description=row["description"],
                    created_at=row.get("created_at", datetime.now(timezone.utc)),
                    updated_at=row.get("updated_at", datetime.now(timezone.utc)),
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

    def delete_context_source(self, source_id: str) -> bool:
        """Delete a context source by ID. Returns True if deleted."""
        with self._connect() as conn:
            result = conn.execute(
                "DELETE FROM context_source WHERE id = %s", (source_id,)
            )
            return result.rowcount > 0

    def add_chunks(
        self, context_id: str, chunks: Iterable[KnowledgeChunk]
    ) -> List[int]:
        """Insert chunks and return their ids, in order.

        The id is returned (and set on the passed chunk) because late
        interaction has to attach segment vectors to the row it just wrote,
        and reading them back by content would be both slower and ambiguous.
        """
        inserted: List[int] = []
        try:
            with self._connect() as conn:
                for chunk in chunks:
                    if not chunk.fs_path or not str(chunk.fs_path).strip():
                        raise ConstraintViolation(
                            "fs_path required for knowledge_chunk",
                            {"fs_path": chunk.fs_path},
                        )
                    row = conn.execute(
                        "INSERT INTO knowledge_chunk (context_id, fs_path, chunk_index, content, embedding, created_at, meta) VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id",
                        (
                            context_id,
                            chunk.fs_path,
                            chunk.chunk_index,
                            chunk.content,
                            chunk.embedding,
                            chunk.created_at,
                            json.dumps(chunk.meta) if chunk.meta else None,
                        ),
                    ).fetchone()
                    if row:
                        chunk.id = int(row["id"])
                        inserted.append(int(row["id"]))
        except errors.ForeignKeyViolation:
            raise ConstraintViolation("context not found", {"context_id": context_id})
        return inserted

    def replace_chunks_for_path(
        self, context_id: str, fs_path: str, chunks: Iterable[KnowledgeChunk]
    ) -> List[int]:
        """Make `chunks` the whole of what this context says about `fs_path`.

        Ingestion used to append, so re-uploading a file left the previous
        generation's chunks in place beside the new ones and a search could
        return, as the contents of that path, text it had not held since an
        earlier upload. SPEC §2.5 dedupes by checksum *and path* and refreshes
        a changed path by ingesting it, which only describes one generation.

        Delete and insert in one transaction, so a reader never sees the path
        with no chunks at all — an interrupted refresh that emptied a path
        would be a worse answer than a stale one.
        """
        deleted_generation: List[int] = []
        try:
            with self._connect() as conn:
                conn.execute(
                    "DELETE FROM knowledge_chunk WHERE context_id = %s AND fs_path = %s",
                    (context_id, fs_path),
                )
                for chunk in chunks:
                    row = conn.execute(
                        "INSERT INTO knowledge_chunk (context_id, fs_path, chunk_index, content, embedding, created_at, meta) VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id",
                        (
                            context_id,
                            fs_path,
                            chunk.chunk_index,
                            chunk.content,
                            chunk.embedding,
                            chunk.created_at,
                            json.dumps(chunk.meta) if chunk.meta else None,
                        ),
                    ).fetchone()
                    if row:
                        chunk.id = int(row["id"])
                        deleted_generation.append(int(row["id"]))
        except errors.ForeignKeyViolation:
            raise ConstraintViolation("context not found", {"context_id": context_id})
        return deleted_generation

    def prune_context_to_paths(self, context_id: str, keep_paths: Sequence[str]) -> int:
        """Drop everything this context says about anything not in `keep_paths`.

        For a conversation's implicit index, where the paths are attachment
        generations and the conversation's records say which ones it holds.
        Re-attaching a filename produces a different generation, so the new
        ingestion replaces nothing and the retired one stays searchable
        until this removes it.

        A row with no path cannot be matched against a generation, so it goes
        too: everything in one of these contexts is there to describe an
        attachment.
        """
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM knowledge_chunk WHERE context_id = %s "
                "AND (fs_path IS NULL OR NOT (fs_path = ANY(%s)))",
                (context_id, list(keep_paths)),
            )
            return cursor.rowcount or 0

    def referenced_attachment_checksums(self, owner_user_id: str) -> set[str]:
        """Every attachment generation this user's conversations still name.

        The marks for the generation store's sweep. They already exist —
        each attachment record names its generation — so a reference count
        would be a second record of the same fact, to be kept correct across
        every way a conversation is created, edited and deleted.

        Raises rather than returning an empty set when the query fails: the
        caller deletes what is not in here, and "unknown" must not be
        mistaken for "nothing".
        """
        if not _is_uuid(owner_user_id):
            raise ValueError("owner_user_id is not a user identifier")
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT att ->> 'checksum' AS checksum "
                "FROM conversation c, LATERAL jsonb_array_elements("
                "  CASE WHEN jsonb_typeof(c.meta -> 'attachments') = 'array' "
                "       THEN c.meta -> 'attachments' ELSE '[]'::jsonb END"
                ") AS att "
                "WHERE c.user_id = %s AND att ->> 'checksum' IS NOT NULL",
                (owner_user_id,),
            ).fetchall()
        return {str(row["checksum"]) for row in rows}

    def attachment_checksum_referenced(
        self, owner_user_id: str, checksum: str
    ) -> bool:
        """Whether any of this user's conversations still names `checksum`.

        The same question `referenced_attachment_checksums` answers for a
        whole account, asked about one object so the sweep can re-ask it
        while holding that object's lock.
        """
        if not _is_uuid(owner_user_id):
            raise ValueError("owner_user_id is not a user identifier")
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM conversation c, LATERAL jsonb_array_elements("
                "  CASE WHEN jsonb_typeof(c.meta -> 'attachments') = 'array' "
                "       THEN c.meta -> 'attachments' ELSE '[]'::jsonb END"
                ") AS att "
                "WHERE c.user_id = %s AND att ->> 'checksum' = %s LIMIT 1",
                (owner_user_id, checksum),
            ).fetchone()
        return row is not None

    def invalidate_path_in_other_contexts(
        self,
        owner_user_id: str,
        fs_path: str,
        *,
        keep_context_id: Optional[str] = None,
    ) -> int:
        """Empty what this user's path-following contexts say about `fs_path`.

        Asked of the database rather than of the upload manifest, because the
        manifest records only the contexts an upload named. A context that
        acquired the path through ``POST /contexts/{id}/sources`` is not in it
        and never becomes so, so a sweep driven by the manifest walked past it
        and left the previous generation's chunks answering for the new bytes.
        The rows themselves are the reverse index: they are what claims to be
        the contents of this path, so they are what the question is put to.

        `keep_context_id` is the context about to receive the new generation.
        Everything else the caller owns is emptied for this path.

        Contexts marked ``meta.auto`` are conversations' implicit indexes and
        are left alone. §19.5 scopes an attachment to the chat that received
        it, so another chat's upload of the same filename must not reach into
        one — removing its chunks would be one chat changing another chat's
        state as much as replacing them would.
        """
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM knowledge_chunk kc USING knowledge_context ctx "
                "WHERE kc.context_id = ctx.id AND ctx.owner_user_id = %s "
                "AND kc.fs_path = %s "
                "AND (%s::uuid IS NULL OR ctx.id <> %s::uuid) "
                "AND COALESCE((ctx.meta ->> 'auto')::boolean, false) IS FALSE",
                (owner_user_id, fs_path, keep_context_id, keep_context_id),
            )
            return cursor.rowcount or 0

    def delete_chunks_under_path(self, owner_user_id: str, fs_path: str) -> int:
        """Drop everything this user's contexts say about `fs_path` or its tree.

        A chunk's ``fs_path`` claims to be the contents of that path, and the
        claim is about the path's current bytes — nothing in the row records
        which generation it came from. So when the path stops existing the
        claim has to stop with it, or a deleted file stays retrievable through
        any conversation grounded in a context that indexed it.

        Scoped by owner rather than by context, because neither of the two
        ways a path gets indexed leaves the route a list to work from: the
        same file uploaded to a second context is ingested again, and an
        extracted tree's members are recorded nowhere. Ownership covers both,
        and covers nothing else.

        The prefix match ends at a separator, so deleting ``bundle`` does not
        take ``bundle2.md``. ``LIKE`` is avoided rather than escaped, since
        ``_`` and ``%`` are wildcards a filename may legitimately contain.
        Segment vectors go with their chunks by cascade.
        """
        prefix = fs_path.rstrip("/") + "/"
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM knowledge_chunk kc USING knowledge_context ctx "
                "WHERE kc.context_id = ctx.id AND ctx.owner_user_id = %s "
                "AND (kc.fs_path = %s OR left(kc.fs_path, %s) = %s)",
                (owner_user_id, fs_path, len(prefix), prefix),
            )
            return cursor.rowcount or 0

    def add_chunk_vectors(
        self,
        chunk_id: int,
        segments: Sequence[Tuple[str, List[float]]],
        *,
        meta: Optional[dict[str, Any]] = None,
    ) -> int:
        """Persist a chunk's segment vectors, replacing any it already had.

        The replace covers re-indexing one chunk — a backfill, a repair. It is
        not what makes re-ingestion idempotent: ``add_chunks`` still inserts
        fresh rows with new ids, so a caller that appends leaves two
        generations of both chunks and segments. What a named path does
        instead is ``replace_chunks_for_path``, which drops the path's previous
        rows in the same transaction that writes its new ones; the segments go
        with them, because a chunk's segments are keyed on the chunk id.
        """
        if not segments:
            return 0
        payload = self._json_param(meta)
        rows = [
            (chunk_id, index, content, embedding, payload)
            for index, (content, embedding) in enumerate(segments)
        ]
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM knowledge_chunk_vector WHERE chunk_id = %s", (chunk_id,)
            )
            # One round trip for the batch. Ingestion calls this per chunk, so
            # a 500-chunk file at eight segments each is 4000 inserts; sending
            # them one at a time made late interaction the slowest thing in
            # the pipeline by an order of magnitude.
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO knowledge_chunk_vector (chunk_id, segment_index, content, embedding, meta) VALUES (%s, %s, %s, %s, %s)",
                    rows,
                )
        return len(segments)

    def late_candidate_ids(
        self,
        context_ids: Optional[Sequence[str]],
        query_embedding: List[float],
        limit: int = 4,
        filters: Optional[dict[str, Any]] = None,
        *,
        user_id: str,  # REQUIRED per SPEC §12.2 - user isolation is mandatory
        tenant_id: Optional[str] = None,
        path_scope: Optional[dict[str, Sequence[str]]] = None,
    ) -> List[int]:
        """Chunks with a segment near this query vector, nearest first.

        Candidate generation only. A chunk qualifies on its best segment, not
        on its average — which is the whole reason the segments are stored
        separately — and the exact MaxSim score is computed by the caller over
        every segment of the chunks this returns.

        SECURITY: user_id is required to enforce data isolation per SPEC §12.2.
        All queries MUST be filtered by user_id to prevent cross-user data leakage.
        """
        if not query_embedding or not context_ids:
            return []
        if not user_id:
            self.logger.error("late_candidate_ids_missing_user_id")
            return []
        where, params = self._chunk_scope(
            context_ids,
            user_id=user_id,
            tenant_id=tenant_id,
            filters=filters,
            path_scope=path_scope,
        )
        with self._connect() as conn:
            sql = """
                SELECT kcv.chunk_id
                FROM knowledge_chunk_vector kcv
                JOIN knowledge_chunk kc ON kcv.chunk_id = kc.id
                JOIN knowledge_context ctx ON kc.context_id = ctx.id
                LEFT JOIN app_user u ON ctx.owner_user_id = u.id
                """
            sql += where
            sql += " ORDER BY kcv.embedding <-> %s::vector LIMIT %s"
            # Over-fetch: several segments of one chunk can occupy the nearest
            # rows, and each of those is one candidate, not many.
            rows = conn.execute(
                sql, (*params, self._format_vector(query_embedding), limit * 4)
            ).fetchall()
        seen: List[int] = []
        for row in rows:
            chunk_id = int(row["chunk_id"])
            if chunk_id not in seen:
                seen.append(chunk_id)
            if len(seen) >= limit:
                break
        return seen

    def chunks_with_vectors(
        self, chunk_ids: Sequence[int]
    ) -> List[Tuple[KnowledgeChunk, List[List[float]]]]:
        """Candidate chunks and every segment vector each one owns.

        Access was already enforced when the ids were generated; this reads
        rows by primary key and adds no scope of its own.
        """
        if not chunk_ids:
            return []
        ids = list(chunk_ids)
        with self._connect() as conn:
            chunk_rows = conn.execute(
                """
                SELECT kc.id, kc.context_id, kc.fs_path, kc.content, kc.embedding, kc.chunk_index, kc.created_at, kc.meta
                FROM knowledge_chunk kc WHERE kc.id = ANY(%s)
                """,
                (ids,),
            ).fetchall()
            vector_rows = conn.execute(
                "SELECT chunk_id, embedding FROM knowledge_chunk_vector"
                " WHERE chunk_id = ANY(%s) ORDER BY chunk_id, segment_index",
                (ids,),
            ).fetchall()

        vectors: dict[int, List[List[float]]] = {}
        for row in vector_rows:
            vectors.setdefault(int(row["chunk_id"]), []).append(
                self._parse_vector(row["embedding"])
            )

        by_id = {
            int(row["id"]): self._row_to_knowledge_chunk(row) for row in chunk_rows
        }
        # Caller's order is the candidate order; preserve it.
        return [
            (by_id[chunk_id], vectors.get(chunk_id, []))
            for chunk_id in ids
            if chunk_id in by_id
        ]

    def list_chunks(
        self,
        context_id: Optional[str] = None,
        *,
        owner_user_id: Optional[str] = None,
        allowed_paths: Optional[Sequence[str]] = None,
        page: int = 1,
        page_size: int = 100,
        cursor: Optional[str] = None,
        include_sentinel: bool = False,
        limit: Optional[int] = None,
    ) -> List[KnowledgeChunk]:
        effective_page_size = max(1, limit or page_size)
        capped_page_size = min(effective_page_size, 500)
        fetch_limit = capped_page_size + (1 if include_sentinel else 0)
        offset = 0 if cursor else max(page - 1, 0) * capped_page_size

        with self._connect() as conn:
            if context_id:
                params: list[Any] = []
                cursor_filter = ""
                cursor_params: list[Any] = []
                if cursor:
                    try:
                        cursor_idx, cursor_id = decode_index_cursor(cursor)
                        cursor_filter = " AND (kc.chunk_index, kc.id) > (%s, %s)"
                        cursor_params.extend([cursor_idx, cursor_id])
                    except Exception as exc:  # pragma: no cover - defensive
                        self.logger.warning("chunk_cursor_decode_failed", error=str(exc))

                query = "SELECT kc.* FROM knowledge_chunk kc"
                if owner_user_id:
                    query += " JOIN knowledge_context ctx ON ctx.id = kc.context_id AND ctx.owner_user_id = %s"
                    params.append(owner_user_id)
                query += " WHERE kc.context_id = %s"
                params.append(context_id)
                if allowed_paths is not None:
                    # Part of what the bounded read selects from, not a
                    # filter over what it returned.
                    query += " AND kc.fs_path = ANY(%s)"
                    params.append(list(allowed_paths))
                if cursor_filter:
                    query += cursor_filter
                    params.extend(cursor_params)
                query += " ORDER BY kc.chunk_index ASC, kc.id ASC LIMIT %s OFFSET %s"
                params.extend([fetch_limit, offset])
                rows = conn.execute(query, tuple(params)).fetchall()
            else:
                if not owner_user_id:
                    return []
                params = [owner_user_id]
                cursor_filter = ""
                if cursor:
                    try:
                        cursor_ts, cursor_id = decode_time_id_cursor(cursor)
                        cursor_filter = " AND (kc.created_at, kc.id) < (%s, %s)"
                        params.extend([cursor_ts, cursor_id])
                    except Exception as exc:  # pragma: no cover - defensive
                        self.logger.warning("chunk_cursor_decode_failed", error=str(exc))

                query = (
                    "SELECT kc.* FROM knowledge_chunk kc JOIN knowledge_context ctx "
                    "ON ctx.id = kc.context_id WHERE ctx.owner_user_id = %s"
                    + cursor_filter
                    + " ORDER BY kc.created_at DESC, kc.id DESC LIMIT %s OFFSET %s"
                )
                params.extend([fetch_limit, offset])
                rows = conn.execute(query, tuple(params)).fetchall()
        return [self._row_to_knowledge_chunk(row) for row in rows]

    # ------------------------------------------------------------------
    # Row -> model
    #
    # One reader per table. Written out at each call site, they drifted:
    # `get_semantic_cluster` was the only one not normalizing label and
    # description, so the same row came back with a different label
    # depending on which method fetched it.
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_user(row: Dict[str, Any]) -> User:
        return User(
            id=str(row["id"]),
            email=row["email"],
            handle=row.get("handle"),
            created_at=row.get("created_at", datetime.now(timezone.utc)),
            is_active=row.get("is_active", True),
            plan_tier=row.get("plan_tier", "free"),
            role=row.get("role", "user"),
            tenant_id=row.get("tenant_id", "public"),
            meta=row.get("meta"),
        )

    def _row_to_preference_event(self, row: Dict[str, Any]) -> PreferenceEvent:
        return PreferenceEvent(
            id=str(row["id"]),
            user_id=str(row["user_id"]),
            conversation_id=str(row["conversation_id"]),
            message_id=str(row["message_id"]),
            feedback=row["feedback"],
            score=row.get("score"),
            explicit_signal=row.get("explicit_signal"),
            context_embedding=self._parse_vector(row.get("context_embedding")),
            cluster_id=row.get("cluster_id"),
            context_text=row.get("context_text"),
            corrected_text=row.get("corrected_text"),
            created_at=row.get("created_at", datetime.now(timezone.utc)),
            weight=self._safe_float(row.get("weight", 1.0), context="preference_event"),
            meta=row.get("meta"),
        )

    def _row_to_training_job(self, row: Dict[str, Any]) -> TrainingJob:
        return TrainingJob(
            id=str(row["id"]),
            user_id=str(row["user_id"]),
            adapter_id=self._require_training_adapter_id(
                row.get("adapter_id"), row.get("id")
            ),
            status=row.get("status", "queued"),
            num_events=row.get("num_events"),
            created_at=row.get("created_at", datetime.now(timezone.utc)),
            updated_at=row.get("updated_at", datetime.now(timezone.utc)),
            loss=row.get("loss"),
            preference_event_ids=row.get("preference_event_ids") or [],
            dataset_path=row.get("dataset_path"),
            new_version=row.get("new_version"),
            meta=row.get("meta"),
        )

    def _row_to_knowledge_chunk(self, row: Dict[str, Any]) -> KnowledgeChunk:
        return KnowledgeChunk(
            id=int(row["id"]),
            context_id=str(row["context_id"]),
            fs_path=row["fs_path"],
            content=row["content"],
            embedding=self._parse_vector(row.get("embedding")),
            chunk_index=row.get("chunk_index", 0),
            created_at=row.get("created_at", datetime.now(timezone.utc)),
            meta=row.get("meta"),
        )

    def _row_to_semantic_cluster(self, row: Dict[str, Any]) -> SemanticCluster:
        return SemanticCluster(
            id=str(row["id"]),
            user_id=row.get("user_id"),
            centroid=self._parse_vector(row.get("centroid")),
            size=row.get("size", 0),
            label=normalize_optional_text(row.get("label")),
            description=normalize_optional_text(row.get("description")),
            sample_message_ids=row.get("sample_message_ids") or [],
            created_at=row.get("created_at", datetime.now(timezone.utc)),
            updated_at=row.get("updated_at", datetime.now(timezone.utc)),
            meta=row.get("meta"),
        )

    def _format_vector(self, embedding: Sequence[float]) -> str:
        safe_vals = (
            self._safe_float(val, default=0.0, context="format_vector") for val in embedding
        )
        return "[" + ",".join(f"{val:.6f}" for val in safe_vals) + "]"

    def _parse_vector(self, value: Any) -> List[float]:
        """Coerce a pgvector column value to list[float].

        Without a registered pgvector type adapter, VECTOR columns come back as
        their text representation ("[0.1,0.2]"). Normalize that and any
        already-parsed sequence to a plain float list so downstream vector math
        never operates on characters.
        """
        if value is None:
            return []
        if isinstance(value, str):
            stripped = value.strip().strip("[]")
            if not stripped:
                return []
            return [
                self._safe_float(part, default=0.0, context="parse_vector")
                for part in stripped.split(",")
            ]
        try:
            return [float(v) for v in value]
        except (TypeError, ValueError):
            return []

    @staticmethod
    def _json_param(value: Any) -> Optional[str]:
        """Serialize a dict/list to JSON text for a JSONB column parameter.

        psycopg cannot adapt a bare dict/list to JSONB, so mappings are dumped
        to text (Postgres applies the text->jsonb assignment cast). Strings pass
        through unchanged and None maps to SQL NULL.
        """
        if value is None:
            return None
        if isinstance(value, str):
            return value
        return json.dumps(value)

    # Both chunk searches read the same rows through the same access rules;
    # only the ORDER BY differs. Kept in one place so a filter added to one
    # channel cannot go missing from the other — and the missing filter that
    # matters here is the user isolation one.
    _CHUNK_SELECT = """
                SELECT kc.id, kc.context_id, kc.fs_path, kc.content, kc.embedding, kc.chunk_index, kc.created_at, kc.meta
                FROM knowledge_chunk kc
                JOIN knowledge_context ctx ON kc.context_id = ctx.id
                LEFT JOIN app_user u ON ctx.owner_user_id = u.id
                """

    @staticmethod
    def _chunk_scope(
        context_ids: Sequence[str],
        *,
        user_id: str,
        tenant_id: Optional[str],
        filters: Optional[dict[str, Any]],
        path_scope: Optional[dict[str, Sequence[str]]] = None,
    ) -> tuple[str, list[Any]]:
        """Access and metadata predicate shared by the chunk searches.

        `path_scope` restricts named contexts to a set of paths and leaves
        every other context alone. A conversation's implicit index is scoped
        this way, to the generations its records still authorize.

        It belongs here rather than after retrieval because it has to reach
        candidate selection. Discarding unauthorized rows from the result
        keeps them out of the prompt but not out of the ranking: measured,
        eight unauthorized rows took every slot and `file_search` reported
        that nothing matched, while the file the conversation actually held
        sat just outside the cut. Over-fetching is not an answer either —
        any fixed over-fetch is consumed by enough rows.
        """
        where_clauses: list[str] = ["kc.context_id = ANY(%s)"]
        params: list[Any] = [list(context_ids)]
        # Always enforce user isolation - this is not optional
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
        if path_scope:
            scoped = sorted(path_scope)
            # Unscoped contexts are unrestricted; a scoped one contributes
            # only its authorized paths. An empty set is a real answer: a
            # conversation holding nothing retrieves nothing from its index.
            branches = ["kc.context_id <> ALL(%s)"]
            params.append(scoped)
            for ctx_id in scoped:
                branches.append("(kc.context_id = %s AND kc.fs_path = ANY(%s))")
                params.extend([ctx_id, list(path_scope[ctx_id])])
            where_clauses.append("(" + " OR ".join(branches) + ")")
        return " WHERE " + " AND ".join(where_clauses), params

    def search_chunks_pgvector(
        self,
        context_ids: Optional[Sequence[str]],
        query: str,
        query_embedding: List[float],
        limit: int = 4,
        filters: Optional[dict[str, Any]] = None,
        *,
        user_id: str,  # REQUIRED per SPEC §12.2 - user isolation is mandatory
        tenant_id: Optional[str] = None,
        path_scope: Optional[dict[str, Sequence[str]]] = None,
    ) -> List[KnowledgeChunk]:
        """Dense candidate generation over knowledge chunks.

        Ordered by vector distance and nothing else. This is a first-stage
        candidate pool, not a final ranking: ``RAGService`` reranks the pool
        against the lexical channel per SPEC §2.5, because a single vector
        cannot express every top-k set a query might want.

        SECURITY: user_id is required to enforce data isolation per SPEC §12.2.
        All queries MUST be filtered by user_id to prevent cross-user data leakage.
        """

        if not query_embedding or not context_ids:
            return []
        if not user_id:
            # Defense in depth: reject if user_id somehow bypasses type checking
            self.logger.error("search_chunks_pgvector_missing_user_id")
            return []
        where, params = self._chunk_scope(
            context_ids,
            user_id=user_id,
            tenant_id=tenant_id,
            filters=filters,
            path_scope=path_scope,
        )
        with self._connect() as conn:
            sql = self._CHUNK_SELECT + where
            sql += " ORDER BY kc.embedding <-> %s::vector LIMIT %s"
            rows = conn.execute(
                sql, (*params, self._format_vector(query_embedding), limit)
            ).fetchall()
        return [self._row_to_knowledge_chunk(row) for row in rows]

    def search_chunks_lexical(
        self,
        context_ids: Optional[Sequence[str]],
        query: str,
        limit: int = 4,
        filters: Optional[dict[str, Any]] = None,
        *,
        user_id: str,  # REQUIRED per SPEC §12.2 - user isolation is mandatory
        tenant_id: Optional[str] = None,
        path_scope: Optional[dict[str, Sequence[str]]] = None,
    ) -> List[KnowledgeChunk]:
        """Lexical candidate generation over knowledge chunks.

        The keyword half of the hybrid. It is the only channel that works at
        all when the encoder is the hash fallback, and it is what keeps exact
        identifiers, error codes and numbers findable when it is not.

        Terms are OR'd, not AND'd: one absent rare word must not empty the
        pool. ``ts_rank`` only has to be a decent recall filter here — the
        real ranking is BM25 over the returned pool.

        SECURITY: user_id is required to enforce data isolation per SPEC §12.2.
        All queries MUST be filtered by user_id to prevent cross-user data leakage.
        """

        if not context_ids:
            return []
        if not user_id:
            # Defense in depth: reject if user_id somehow bypasses type checking
            self.logger.error("search_chunks_lexical_missing_user_id")
            return []
        terms = _tsquery_terms(query)
        if not terms:
            return []
        where, params = self._chunk_scope(
            context_ids,
            user_id=user_id,
            tenant_id=tenant_id,
            filters=filters,
            path_scope=path_scope,
        )
        # 'simple' rather than 'english': no stemming, so an identifier stays
        # itself, and no language is assumed of the user's own files.
        with self._connect() as conn:
            sql = self._CHUNK_SELECT + where
            # content_tsv is a stored generated column, so neither the match
            # nor the rank tokenizes anything at query time.
            sql += " AND kc.content_tsv @@ to_tsquery('simple', %s)"
            sql += (
                " ORDER BY ts_rank(kc.content_tsv, to_tsquery('simple', %s))"
                " DESC, kc.id LIMIT %s"
            )
            rows = conn.execute(sql, (*params, terms, terms, limit)).fetchall()
        return [self._row_to_knowledge_chunk(row) for row in rows]

    def search_chunks(
        self,
        context_id: Optional[str],
        query: str,
        query_embedding: Optional[List[float]],
        limit: int = 4,
        *,
        semantic: bool = False,
        allowed_paths: Optional[Sequence[str]] = None,
    ) -> List[KnowledgeChunk]:
        """Non-pgvector hybrid search; suitable for tests and tiny corpora only.

        `allowed_paths` restricts this context to a set of paths, applied
        before the candidate cut for the same reason the pgvector path
        applies it in SQL.

        ``semantic`` is the caller's assertion that ``query_embedding`` came
        from a real encoder. It defaults to False because the kernel's default
        encoder is the hash fallback, and cosine over those vectors is noise
        that must never enter a score (SPEC §2.5).
        """

        def _cosine(a: List[float], b: List[float]) -> float:
            # Belt and braces: knowledge_chunk.embedding is VECTOR(dim) NOT
            # NULL, so widths cannot differ in practice. If one ever did,
            # scoring the overlapping prefix would produce a number that looks
            # like a similarity and is not — contribute nothing instead.
            if not a or not b or len(a) != len(b):
                return 0.0
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = sum(x * x for x in a) ** 0.5 or 1.0
            norm_b = sum(y * y for y in b) ** 0.5 or 1.0
            return dot / (norm_a * norm_b)

        candidate_limit = limit or 4
        # Issue 25.3: prevent unbounded candidate loading by limiting DB reads
        max_candidates = min(candidate_limit * 5, 500)
        # The restriction goes into the query that produces the bounded set,
        # not over what it returned. `list_chunks` orders by chunk index and
        # id, and every generation starts at index 0, so unauthorized rows
        # inserted earlier hold the lower ids and filled the whole window —
        # measured, forty retired rows consumed a twenty-row read and the
        # authorized generation was never loaded, so retrieval answered with
        # nothing. Raising the cap does not fix that; any finite pre-filter
        # window has the same counterexample.
        candidates = self.list_chunks(
            context_id, limit=max_candidates, allowed_paths=allowed_paths
        )
        if not candidates:
            return []
        query_tokens = _tokenize_text(query)
        documents = [_tokenize_text(ch.content) for ch in candidates]
        bm25_scores = _compute_bm25_scores(query_tokens, documents)
        # Each channel ranks what it matched, and the two orders are fused by
        # position rather than by score — the same rule the pgvector path uses
        # (SPEC §2.5). Without a real encoder the semantic channel does not
        # speak at all, because cosine over hash vectors is noise.
        channels: list[tuple[float, list[int]]] = [
            (_LEXICAL_WEIGHT, _ranked_positive(bm25_scores))
        ]
        if semantic and query_embedding:
            channels.append(
                (
                    _SEMANTIC_WEIGHT,
                    _ranked_positive(
                        [_cosine(query_embedding, ch.embedding) for ch in candidates]
                    ),
                )
            )
        fused = _fuse_ranks(channels)

        combined: dict[str, tuple[KnowledgeChunk, float]] = {}
        for index, chunk in enumerate(candidates):
            score = fused.get(index)
            if score is None:
                continue
            key = " ".join(chunk.content.split()).lower() or str(chunk.id or "")
            existing = combined.get(key)
            if not existing or score > existing[1]:
                combined[key] = (chunk, score)
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
                    "SELECT * FROM auth_session WHERE (%s::text IS NULL OR tenant_id = %s) ORDER BY created_at DESC LIMIT %s",
                    (tenant_id, tenant_id, limit),
                ).fetchall()
                sections["sessions"] = [_serialize(row) for row in rows]
            if kind in (None, "conversations"):
                rows = conn.execute(
                    """
                    SELECT c.*
                    FROM conversation c
                    JOIN app_user u ON c.user_id = u.id
                    WHERE (%s::text IS NULL OR u.tenant_id = %s)
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
                    WHERE (%s::text IS NULL OR u.tenant_id = %s)
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
                    WHERE (%s::text IS NULL OR u.tenant_id = %s)
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
                    WHERE (%s::text IS NULL OR u.tenant_id = %s)
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
                    WHERE (%s::text IS NULL OR u.tenant_id = %s)
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
