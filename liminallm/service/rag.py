from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from liminallm.service.embeddings import deterministic_embedding
from liminallm.storage.memory import MemoryStore
from liminallm.storage.postgres import PostgresStore
from liminallm.storage.models import KnowledgeChunk


# Default overlap per SPEC §2.5: "50 token overlap"
DEFAULT_OVERLAP_TOKENS = 50


def _simple_tokenize(text: str) -> List[str]:
    """Simple word-based tokenizer for chunking.

    Per SPEC §2.5: Uses token-based chunking. This is a simple whitespace/punctuation
    tokenizer that approximates token boundaries for chunking purposes.
    """
    # Split on whitespace and punctuation while preserving meaningful tokens
    return re.findall(r"\b\w+\b|[^\w\s]", text)


def _detokenize(tokens: List[str]) -> str:
    """Reconstruct text from tokens, handling spacing."""
    if not tokens:
        return ""
    result = []
    for i, token in enumerate(tokens):
        # Add space before non-punctuation tokens (except first)
        if i > 0 and re.match(r"\w", token):
            result.append(" ")
        result.append(token)
    return "".join(result)


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
            if hasattr(store, "search_chunks"):
                self._retriever = self._retrieve_local_hybrid
            elif hasattr(store, "search_chunks_legacy"):
                self._retriever = self._retrieve_local_hybrid
            else:
                raise ValueError("legacy-backed store required for hybrid RAG mode")

    def retrieve(
        self,
        context_ids: Optional[Sequence[str]],
        query: Optional[str],
        limit: int = 4,
        *,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> List[KnowledgeChunk]:
        if not context_ids:
            return []

        normalized_query = query or ""
        return self._retriever(
            context_ids, normalized_query, limit, user_id=user_id, tenant_id=tenant_id
        )

    def _uses_pgvector(self) -> bool:
        return self.rag_mode in {"pgvector", "pg", "vector"}

    def _allowed_context_ids(
        self,
        context_ids: Sequence[str],
        *,
        user_id: Optional[str],
        tenant_id: Optional[str],
    ) -> List[str]:
        """Filter context IDs to only those accessible by the user.

        Per SPEC §12.2, user isolation is mandatory for RAG retrieval.
        This method logs warnings when contexts are filtered out to aid debugging.

        Args:
            context_ids: Requested context IDs
            user_id: Requesting user ID (required for access)
            tenant_id: Optional tenant ID for multi-tenant filtering

        Returns:
            List of accessible context IDs (may be empty if none accessible)
        """
        if not user_id:
            logger.warning(
                "rag_retrieval_no_user_id",
                context_ids=list(context_ids),
                message="RAG retrieval requires user_id for access control; returning empty results",
            )
            return []

        allowed: List[str] = []
        filtered_reasons: Dict[str, str] = {}

        if hasattr(self.store, "contexts"):
            contexts = getattr(self.store, "contexts")
            users = getattr(self.store, "users", {})
            for ctx_id in context_ids:
                ctx = contexts.get(ctx_id) if isinstance(contexts, dict) else None
                if not ctx:
                    filtered_reasons[ctx_id] = "not_found"
                    continue
                if user_id and ctx.owner_user_id != user_id:
                    filtered_reasons[ctx_id] = "owner_mismatch"
                    continue
                if tenant_id:
                    owner = (
                        users.get(ctx.owner_user_id)
                        if isinstance(users, dict)
                        else None
                    )
                    if not owner or owner.tenant_id != tenant_id:
                        filtered_reasons[ctx_id] = "tenant_mismatch"
                        continue
                allowed.append(ctx_id)
        else:
            for ctx_id in context_ids:
                context = getattr(self.store, "get_context", lambda *_: None)(ctx_id)
                if not context:
                    filtered_reasons[ctx_id] = "not_found"
                    continue
                if user_id and context.owner_user_id != user_id:
                    filtered_reasons[ctx_id] = "owner_mismatch"
                    continue
                if tenant_id:
                    owner = getattr(self.store, "get_user", lambda *_: None)(
                        context.owner_user_id
                    )
                    if not owner or owner.tenant_id != tenant_id:
                        filtered_reasons[ctx_id] = "tenant_mismatch"
                        continue
                allowed.append(ctx_id)

        # Log if any contexts were filtered for debugging
        if filtered_reasons:
            logger.info(
                "rag_contexts_filtered",
                user_id=user_id,
                tenant_id=tenant_id,
                requested_count=len(context_ids),
                allowed_count=len(allowed),
                filtered=filtered_reasons,
                message="Some requested contexts were filtered due to access control",
            )

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
        allowed_ids = self._allowed_context_ids(
            context_ids, user_id=user_id, tenant_id=tenant_id
        )
        if not allowed_ids:
            return []

        query_embedding = self.embed(query)
        filters = {"embedding_model_id": self.embedding_model_id}

        return self.store.search_chunks_pgvector(  # type: ignore[attr-defined]
            allowed_ids,
            query,
            query_embedding,
            limit,
            filters=filters,
            user_id=user_id,
            tenant_id=tenant_id,
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
        allowed_ids = self._allowed_context_ids(
            context_ids, user_id=user_id, tenant_id=tenant_id
        )
        if not allowed_ids:
            return []

        query_embedding = self.embed(query)
        legacy_search = getattr(self.store, "search_chunks", None) or getattr(
            self.store, "search_chunks_legacy", None
        )
        if not callable(legacy_search):
            return []

        results: List[KnowledgeChunk] = []
        per_context_limit = max(1, math.ceil(limit / len(allowed_ids)))
        for ctx_id in allowed_ids:
            results.extend(
                legacy_search(ctx_id, query, query_embedding, per_context_limit)
            )

        filtered = [
            chunk
            for chunk in results
            if (chunk.meta or {}).get("embedding_model_id") == self.embedding_model_id
        ]
        return filtered[:limit]

    def ingest_text(
        self,
        context_id: str,
        text: str,
        chunk_size: Optional[int] = None,
        source_path: Optional[str] = None,
        overlap_tokens: Optional[int] = None,
    ) -> int:
        """Ingest text into chunks using token-based sliding window with overlap.

        Per SPEC §2.5: Uses token-based splitter (300-500 tokens with 50 token overlap).
        This implementation:
        - Tokenizes the input text
        - Creates chunks with specified token count
        - Applies overlap between consecutive chunks for context continuity
        """
        if not hasattr(self.store, "add_chunks"):
            return 0
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        blob = " ".join(lines)
        if not blob:
            return 0

        # Tokenize the text per SPEC §2.5 requirement for token-based chunking
        tokens = _simple_tokenize(blob)
        if not tokens:
            return 0

        chosen_chunk_tokens = max(chunk_size or self.default_chunk_size, 64)
        # Use default overlap if not specified (SPEC §2.5: 50 token overlap)
        effective_overlap = overlap_tokens if overlap_tokens is not None else DEFAULT_OVERLAP_TOKENS
        # Ensure overlap doesn't exceed chunk size
        effective_overlap = min(effective_overlap, chosen_chunk_tokens // 2)
        # Step size accounts for overlap
        step_size = max(1, chosen_chunk_tokens - effective_overlap)

        chunks: List[KnowledgeChunk] = []
        default_path = source_path or "inline"
        chunk_index = 0

        for start in range(0, len(tokens), step_size):
            end = min(start + chosen_chunk_tokens, len(tokens))
            chunk_tokens = tokens[start:end]

            # Skip if we've reached the end and this would be a tiny fragment
            if chunk_index > 0 and len(chunk_tokens) < effective_overlap:
                break

            segment = _detokenize(chunk_tokens)
            if not segment.strip():
                continue

            chunks.append(
                KnowledgeChunk(
                    id=None,
                    context_id=context_id,
                    fs_path=default_path,
                    content=segment,
                    embedding=self.embed(segment),
                    chunk_index=chunk_index,
                    meta={
                        "embedding_model_id": self.embedding_model_id,
                        "token_count": len(chunk_tokens),
                        "start_token": start,
                        "end_token": end,
                        "overlap_tokens": effective_overlap if chunk_index > 0 else 0,
                    },
                )
            )
            chunk_index += 1

            # Break if we've processed all tokens
            if end >= len(tokens):
                break

        if chunks:
            self.store.add_chunks(context_id, chunks)  # type: ignore[attr-defined]
        return len(chunks)

    def ingest_file(
        self, context_id: str, path: str, chunk_size: Optional[int] = None
    ) -> int:
        data = Path(path).read_text(encoding="utf-8", errors="ignore")
        return self.ingest_text(
            context_id, data, chunk_size=chunk_size, source_path=path
        )

    def embed_text(self, text: str) -> List[float]:
        """Backward-compatible hook: use the deterministic embedding pipeline."""

        return self.embed(text)
