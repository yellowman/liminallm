from __future__ import annotations

import hashlib
import math
import re
import uuid
from pathlib import Path
from typing import List, Optional

from liminallm.storage.memory import MemoryStore
from liminallm.storage.postgres import PostgresStore
from liminallm.storage.models import KnowledgeChunk


class RAGService:
    """Lightweight retriever against knowledge chunks."""

    def __init__(self, store: PostgresStore | MemoryStore) -> None:
        self.store = store

    def retrieve(self, context_id: Optional[str], query: Optional[str], limit: int = 4) -> List[KnowledgeChunk]:
        if hasattr(self.store, "search_chunks"):
            query_embedding = self.embed_text(query or "") if query else None
            return self.store.search_chunks(context_id, query_embedding, limit)  # type: ignore[attr-defined]
        return []

    def ingest_text(self, context_id: str, text: str, chunk_size: int = 400, source_path: Optional[str] = None) -> int:
        if not hasattr(self.store, "add_chunks"):
            return 0
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        blob = " ".join(lines)
        if not blob:
            return 0
        chunks: List[KnowledgeChunk] = []
        for idx in range(0, len(blob), chunk_size):
            segment = blob[idx : idx + chunk_size]
            chunks.append(
                KnowledgeChunk(
                    id=str(uuid.uuid4()),
                    context_id=context_id,
                    text=segment,
                    embedding=self.embed_text(segment),
                    seq=math.floor(idx / chunk_size),
                    meta={"fs_path": source_path} if source_path else None,
                )
            )
        if chunks:
            self.store.add_chunks(context_id, chunks)  # type: ignore[attr-defined]
        return len(chunks)

    def ingest_file(self, context_id: str, path: str) -> int:
        data = Path(path).read_text(encoding="utf-8", errors="ignore")
        return self.ingest_text(context_id, data, source_path=path)

    def embed_text(self, text: str, dim: int = 64) -> List[float]:
        tokens = re.findall(r"\w+", text.lower())
        if not tokens:
            return [0.0 for _ in range(dim)]
        vec = [0.0 for _ in range(dim)]
        for token in tokens:
            h = int(hashlib.sha256(token.encode()).hexdigest(), 16)
            vec[h % dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]
