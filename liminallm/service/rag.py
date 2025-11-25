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
        if not hasattr(store, "search_chunks_pgvector"):
            raise ValueError("pgvector-backed store required for RAGService")
        self.embed = embed
        self.embedding_model_id = embedding_model_id

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

        query_embedding = self.embed(query)
        filters = {"embedding_model_id": self.embedding_model_id}

        return self.store.search_chunks_pgvector(  # type: ignore[attr-defined]
            context_ids, query_embedding, limit, filters=filters, user_id=user_id, tenant_id=tenant_id
        )

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
        for idx in range(0, len(blob), chosen_chunk):
            segment = blob[idx : idx + chosen_chunk]
            chunks.append(
                KnowledgeChunk(
                    id=str(uuid.uuid4()),
                    context_id=context_id,
                    text=segment,
                    embedding=self.embed(segment),
                    seq=math.floor(idx / chosen_chunk),
                    meta={
                        "fs_path": source_path,
                        "embedding_model_id": self.embedding_model_id,
                    }
                    if source_path
                    else {"embedding_model_id": self.embedding_model_id},
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
