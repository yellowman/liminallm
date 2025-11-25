from __future__ import annotations

import uuid

from liminallm.service.embeddings import deterministic_embedding
from liminallm.service.rag import RAGService
from liminallm.storage.models import KnowledgeChunk


class LegacyOnlyStore:
    def __init__(self):
        self.chunks: list[KnowledgeChunk] = []
        self.used_legacy = False

    def add_chunks(self, context_id: str, chunks: list[KnowledgeChunk]) -> None:
        self.chunks.extend(chunks)

    def search_chunks_legacy(self, context_id: str | None, query: str, query_embedding, limit: int = 4):
        self.used_legacy = True
        candidates = [c for c in self.chunks if c.context_id == context_id] if context_id else list(self.chunks)
        return candidates[:limit]


def test_rag_local_hybrid_initializes_without_pgvector_and_filters_embedding_id():
    store = LegacyOnlyStore()
    rag = RAGService(
        store,
        rag_mode="local_hybrid",
        embed=deterministic_embedding,
        embedding_model_id="demo-embed",
    )

    ctx_id = str(uuid.uuid4())
    rag.ingest_text(ctx_id, "alpha beta gamma")
    store.chunks.append(
        KnowledgeChunk(
            id=str(uuid.uuid4()),
            context_id=ctx_id,
            text="should be filtered out",
            embedding=[0.0],
            seq=99,
            meta={"embedding_model_id": "other-model"},
        )
    )

    results = rag.retrieve(ctx_id, "alpha", limit=4)

    assert store.used_legacy is True
    assert all((chunk.meta or {}).get("embedding_model_id") == "demo-embed" for chunk in results)
