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
        self.rag_mode = (rag_mode or os.getenv("RAG_MODE") or "pgvector").lower().replace("-", "_")
        self.use_pgvector = self.rag_mode not in {"local_hybrid", "legacy", "hybrid_local"}
        if self.use_pgvector and not hasattr(store, "search_chunks_pgvector"):
            raise ValueError("pgvector-backed store required for RAGService")
        if not self.use_pgvector and not (
            hasattr(store, "search_chunks_legacy") or hasattr(store, "search_chunks")
        ):
            raise ValueError("legacy chunk search required for local_hybrid RAG mode")
        self.embed = embed
        self.embedding_model_id = embedding_model_id

    def retrieve(self, context_id: Optional[str], query: Optional[str], limit: int = 4) -> List[KnowledgeChunk]:
        if not query:
            return []

        query_embedding = self.embed(query)
        contexts: Sequence[str] | None = [context_id] if context_id else None
        filters = {"embedding_model_id": self.embedding_model_id}

        if self.use_pgvector and hasattr(self.store, "search_chunks_pgvector"):
            return self.store.search_chunks_pgvector(  # type: ignore[attr-defined]
                contexts, query_embedding, limit, filters=filters
            )

        legacy_search = getattr(self.store, "search_chunks_legacy", None) or getattr(self.store, "search_chunks", None)
        if not legacy_search:
            return []

        results = legacy_search(context_id, query, query_embedding, limit)
        if filters and filters.get("embedding_model_id"):
            results = [
                chunk
                for chunk in results
                if (chunk.meta or {}).get("embedding_model_id") == filters.get("embedding_model_id")
            ]
        return results

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
