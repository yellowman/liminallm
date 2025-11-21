from __future__ import annotations

import math
import uuid
from typing import List, Optional

from liminallm.storage.memory import MemoryStore
from liminallm.storage.postgres import PostgresStore
from liminallm.storage.models import KnowledgeChunk


class RAGService:
    """Lightweight retriever against knowledge chunks."""

    def __init__(self, store: PostgresStore | MemoryStore) -> None:
        self.store = store

    def retrieve(self, context_id: Optional[str], limit: int = 4) -> List[KnowledgeChunk]:
        if hasattr(self.store, "search_chunks"):
            return self.store.search_chunks(context_id, limit)  # type: ignore[attr-defined]
        return []

    def ingest_text(self, context_id: str, text: str, chunk_size: int = 400) -> None:
        if not hasattr(self.store, "add_chunks"):
            return
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        blob = " ".join(lines)
        chunks: List[KnowledgeChunk] = []
        for idx in range(0, len(blob), chunk_size):
            segment = blob[idx : idx + chunk_size]
            chunks.append(
                KnowledgeChunk(
                    id=str(uuid.uuid4()),
                    context_id=context_id,
                    text=segment,
                    embedding=[],
                    seq=math.floor(idx / chunk_size),
                )
            )
        if chunks:
            self.store.add_chunks(context_id, chunks)  # type: ignore[attr-defined]
