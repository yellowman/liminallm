from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

import psycopg
from psycopg.rows import dict_row

from liminallm.storage.models import (
    Artifact,
    ConfigPatchAudit,
    Conversation,
    KnowledgeChunk,
    KnowledgeContext,
    Message,
    Session,
    User,
)


class PostgresStore:
    """Thin Postgres-backed store to persist kernel primitives."""

    def __init__(self, dsn: str, fs_root: str) -> None:
        self.dsn = dsn
        self.fs_root = Path(fs_root)
        self.fs_root.mkdir(parents=True, exist_ok=True)

    def _connect(self):
        return psycopg.connect(self.dsn, autocommit=True, row_factory=dict_row)

    # users
    def create_user(self, email: str, handle: Optional[str] = None) -> User:
        user_id = str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO app_user (id, email, handle) VALUES (%s, %s, %s)",
                (user_id, email, handle),
            )
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
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO auth_session (id, user_id, created_at, expires_at, user_agent, ip_addr) VALUES (%s, %s, %s, %s, %s, %s)",
                (sess.id, sess.user_id, sess.created_at, sess.expires_at, user_agent, ip_addr),
            )
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
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO conversation (id, user_id, title, created_at, updated_at, active_context_id) VALUES (%s, %s, %s, %s, %s, %s)",
                (conv_id, user_id, title, now, now, active_context_id),
            )
        return Conversation(id=conv_id, user_id=user_id, title=title, created_at=now, updated_at=now, active_context_id=active_context_id)

    def append_message(self, conversation_id: str, sender: str, role: str, content: str, meta: Optional[dict] = None) -> Message:
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
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO artifact (id, owner_user_id, type, name, description, schema, fs_path) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (artifact_id, owner_user_id, type_, name, description, json.dumps(schema), fs_path),
            )
            conn.execute(
                "INSERT INTO artifact_version (artifact_id, version, schema, fs_path) VALUES (%s, %s, %s, %s)",
                (artifact_id, 1, json.dumps(schema), fs_path),
            )
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

    # knowledge
    def upsert_context(self, owner_user_id: Optional[str], name: str, description: str, fs_path: Optional[str] = None) -> KnowledgeContext:
        ctx_id = str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO knowledge_context (id, owner_user_id, name, description, fs_path) VALUES (%s, %s, %s, %s, %s)",
                (ctx_id, owner_user_id, name, description, fs_path),
            )
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
        with self._connect() as conn:
            for chunk in chunks:
                conn.execute(
                    "INSERT INTO knowledge_chunk (id, context_id, text, embedding, seq, created_at, meta) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (chunk.id, context_id, chunk.text, chunk.embedding, chunk.seq, chunk.created_at, json.dumps(chunk.meta) if chunk.meta else None),
                )

    def search_chunks(self, context_id: Optional[str], limit: int = 4) -> List[KnowledgeChunk]:
        with self._connect() as conn:
            if context_id:
                rows = conn.execute(
                    "SELECT * FROM knowledge_chunk WHERE context_id = %s ORDER BY seq ASC LIMIT %s",
                    (context_id, limit),
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM knowledge_chunk ORDER BY created_at DESC LIMIT %s", (limit,)).fetchall()
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
        return results

    def _persist_payload(self, artifact_id: str, version: int, schema: dict) -> str:
        artifact_dir = self.fs_root / "artifacts" / artifact_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        path = artifact_dir / f"v{version}.json"
        path.write_text(json.dumps(schema, indent=2))
        return str(path)
