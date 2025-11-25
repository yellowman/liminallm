from __future__ import annotations

import math
import os
import uuid
from pathlib import Path
from typing import Callable, List, Optional, Sequence

from liminallm.service.embeddings import deterministic_embedding
from liminallm.storage.memory import MemoryStore
from liminallm.storage.postgres import PostgresStore
from liminallm.storage.models import KnowledgeChunk


class RAGService:
    """Hybrid retriever against knowledge chunks."""

    def __init__(
        self,
        store: PostgresStore | MemoryStore,
        default_chunk_size: int = 400,
        *,
        rag_mode: str | None = None,
        embed: Callable[[str], List[float]] = deterministic_embedding,
        embedding_model_id: str = "text-embedding",
    ) -> None:
        self.store = store
        self.default_chunk_size = max(default_chunk_size, 64)
        self.rag_mode = (rag_mode or os.getenv("RAG_MODE") or "pgvector").lower()
        self.embed = embed
        self.embedding_model_id = embedding_model_id

        if self._uses_pgvector():
            if not hasattr(store, "search_chunks_pgvector"):
                raise ValueError("pgvector-backed store required for RAGService")
            self._retriever = self._retrieve_pgvector
        else:
            if not hasattr(store, "search_chunks_legacy") and not hasattr(store, "search_chunks"):
                raise ValueError("legacy-backed store required for hybrid RAG mode")
            self._retriever = self._retrieve_local_hybrid

    def retrieve(
        self,
        context_ids: Optional[Sequence[str]],
        query: Optional[str],
        limit: int = 4,
        *,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> List[KnowledgeChunk]:
        if not query or not context_ids:
            return []

        return self._retriever(context_ids, query, limit, user_id=user_id, tenant_id=tenant_id)

    def _uses_pgvector(self) -> bool:
        return self.rag_mode in {"pgvector", "pg", "vector"}

    def _allowed_context_ids(
        self, context_ids: Sequence[str], *, user_id: Optional[str], tenant_id: Optional[str]
    ) -> List[str]:
        if not hasattr(self.store, "contexts"):
            return list(context_ids)

        contexts = getattr(self.store, "contexts")
        users = getattr(self.store, "users", {})
        allowed: List[str] = []
        for ctx_id in context_ids:
            ctx = contexts.get(ctx_id) if isinstance(contexts, dict) else None
            if not ctx:
                continue
            if user_id and ctx.owner_user_id != user_id:
                continue
            if tenant_id:
                owner = users.get(ctx.owner_user_id) if isinstance(users, dict) else None
                if not owner or owner.tenant_id != tenant_id:
                    continue
            allowed.append(ctx_id)
        return allowed

    def _retrieve_pgvector(
        self,
        context_ids: Sequence[str],
        query: str,
        limit: int,
        *,
        user_id: Optional[str],
        tenant_id: Optional[str],
    ) -> List[KnowledgeChunk]:
        query_embedding = self.embed(query)
        filters = {"embedding_model_id": self.embedding_model_id}

        return self.store.search_chunks_pgvector(  # type: ignore[attr-defined]
            context_ids, query_embedding, limit, filters=filters, user_id=user_id, tenant_id=tenant_id
        )

    def _retrieve_local_hybrid(
        self,
        context_ids: Sequence[str],
        query: str,
        limit: int,
        *,
        user_id: Optional[str],
        tenant_id: Optional[str],
    ) -> List[KnowledgeChunk]:
        allowed_ids = self._allowed_context_ids(context_ids, user_id=user_id, tenant_id=tenant_id)
        if not allowed_ids:
            return []

        query_embedding = self.embed(query)
        legacy_search = getattr(self.store, "search_chunks_legacy", None) or getattr(self.store, "search_chunks", None)
        if not legacy_search:
            return []

        results: List[KnowledgeChunk] = []
        for ctx_id in allowed_ids:
            results.extend(legacy_search(ctx_id, query, query_embedding, limit))

        filtered = [
            chunk for chunk in results if (chunk.meta or {}).get("embedding_model_id") == self.embedding_model_id
        ]
        return filtered[:limit]

    def ingest_text(
        self, context_id: str, text: str, chunk_size: Optional[int] = None, source_path: Optional[str] = None
    ) -> int:
        if not hasattr(self.store, "add_chunks"):
            return 0
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        blob = " ".join(lines)
        if not blob:
            return 0
        chosen_chunk = max(chunk_size or self.default_chunk_size, 64)
        chunks: List[KnowledgeChunk] = []
        default_path = source_path or "inline"
        for idx in range(0, len(blob), chosen_chunk):
            segment = blob[idx : idx + chosen_chunk]
            chunks.append(
                KnowledgeChunk(
                    id=None,
                    context_id=context_id,
                    fs_path=default_path,
                    content=segment,
                    embedding=self.embed(segment),
                    chunk_index=math.floor(idx / chosen_chunk),
                    meta={"embedding_model_id": self.embedding_model_id},
                )
            )
        if chunks:
            self.store.add_chunks(context_id, chunks)  # type: ignore[attr-defined]
        return len(chunks)

    def ingest_file(self, context_id: str, path: str, chunk_size: Optional[int] = None) -> int:
        data = Path(path).read_text(encoding="utf-8", errors="ignore")
        return self.ingest_text(context_id, data, chunk_size=chunk_size, source_path=path)

    def embed_text(self, text: str) -> List[float]:
        """Backward-compatible hook: use the deterministic embedding pipeline."""

        return self.embed(text)
