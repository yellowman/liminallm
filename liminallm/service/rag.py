from __future__ import annotations

import math
import os
import re
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Union

from liminallm.logging import get_logger
from liminallm.service.embeddings import deterministic_embedding
from liminallm.service.fs import PathTraversalError, safe_join
from liminallm.storage.memory import MemoryStore
from liminallm.storage.models import KnowledgeChunk
from liminallm.storage.postgres import PostgresStore

logger = get_logger(__name__)

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
        rag_mode: str | Enum | None = None,
        embed: Callable[[str], List[float]] = deterministic_embedding,
        embedding_model_id: str = "text-embedding",
    ) -> None:
        self.store = store
        self.default_chunk_size = max(default_chunk_size, 64)
        mode_value = rag_mode.value if isinstance(rag_mode, Enum) else rag_mode
        self.rag_mode = str(mode_value or os.getenv("RAG_MODE") or "pgvector").lower()
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
        max_tokens: Optional[int] = None,
        min_token_count: int = 10,
    ) -> List[KnowledgeChunk]:
        """Retrieve relevant chunks for a query.

        Args:
            context_ids: Context IDs to search within
            query: Search query
            limit: Maximum number of chunks to return
            user_id: User ID for access control
            tenant_id: Tenant ID for multi-tenant filtering
            max_tokens: Optional maximum total tokens across all returned chunks.
                       Uses token_count from chunk metadata if available.
            min_token_count: Minimum tokens per chunk (filters out very short chunks)

        Returns:
            List of relevant chunks, optionally limited by total token budget
        """
        if not context_ids:
            return []

        normalized_query = query or ""
        results = self._retriever(
            context_ids, normalized_query, limit * 2 if max_tokens else limit,
            user_id=user_id, tenant_id=tenant_id
        )

        # Filter out very short chunks (likely noise)
        if min_token_count > 0:
            results = [
                chunk for chunk in results
                if self._get_chunk_token_count(chunk) >= min_token_count
            ]

        # Apply token budget if specified
        if max_tokens is not None and max_tokens > 0:
            results = self._apply_token_budget(results, max_tokens, limit)

        return results[:limit]

    def _get_chunk_token_count(self, chunk: KnowledgeChunk) -> int:
        """Get token count from chunk metadata, or estimate from content."""
        if chunk.meta and isinstance(chunk.meta.get("token_count"), int):
            return chunk.meta["token_count"]
        # Estimate ~4 chars per token as fallback
        return len(chunk.content) // 4

    def _apply_token_budget(
        self,
        chunks: List[KnowledgeChunk],
        max_tokens: int,
        limit: int,
    ) -> List[KnowledgeChunk]:
        """Select chunks that fit within the token budget.

        Prioritizes chunks in their existing order (by relevance score)
        while respecting the total token budget.
        """
        selected: List[KnowledgeChunk] = []
        total_tokens = 0

        for chunk in chunks:
            if len(selected) >= limit:
                break

            chunk_tokens = self._get_chunk_token_count(chunk)

            # Check if adding this chunk would exceed budget
            if total_tokens + chunk_tokens > max_tokens:
                # If we have no chunks yet, include at least one
                if not selected:
                    selected.append(chunk)
                    total_tokens += chunk_tokens
                continue

            selected.append(chunk)
            total_tokens += chunk_tokens

        logger.debug(
            "rag_token_budget_applied",
            max_tokens=max_tokens,
            total_tokens=total_tokens,
            chunk_count=len(selected),
        )
        return selected

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

        # Minimum tokens for a standalone final chunk (avoid losing meaningful content)
        min_final_chunk_tokens = max(10, effective_overlap // 2)

        for start in range(0, len(tokens), step_size):
            end = min(start + chosen_chunk_tokens, len(tokens))
            chunk_tokens = tokens[start:end]

            # Skip only if this is a truly tiny trailing fragment
            # Use a smaller threshold to avoid losing meaningful final content
            if chunk_index > 0 and len(chunk_tokens) < min_final_chunk_tokens:
                # Append remaining tokens to the previous chunk instead of dropping
                if chunks and chunk_tokens:
                    prev_chunk = chunks[-1]
                    prev_content = prev_chunk.content
                    extra_segment = _detokenize(chunk_tokens)
                    if extra_segment.strip():
                        # Update the previous chunk to include the trailing content
                        combined_content = prev_content + " " + extra_segment.strip()
                        prev_meta = dict(prev_chunk.meta or {})
                        prev_meta["end_token"] = end
                        prev_meta["token_count"] = prev_meta.get("token_count", 0) + len(chunk_tokens)
                        prev_meta["includes_trailing"] = True
                        chunks[-1] = KnowledgeChunk(
                            id=prev_chunk.id,
                            context_id=prev_chunk.context_id,
                            fs_path=prev_chunk.fs_path,
                            content=combined_content,
                            embedding=self.embed(combined_content),
                            chunk_index=prev_chunk.chunk_index,
                            meta=prev_meta,
                        )
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

    # Issue 38.3: Default limits for recursive ingestion to prevent resource exhaustion
    MAX_INGEST_FILES = 10000  # Maximum files to process in one ingest operation
    MAX_INGEST_DEPTH = 20  # Maximum directory depth for recursive ingestion

    def ingest_path(
        self,
        context_id: str,
        fs_path: str,
        *,
        recursive: bool = True,
        chunk_size: Optional[int] = None,
        extensions: Optional[List[str]] = None,
        allowed_base: Optional[Union[str, Path]] = None,
        max_files: Optional[int] = None,
        max_depth: Optional[int] = None,
    ) -> int:
        """Ingest content from a filesystem path (file or directory).

        Args:
            context_id: Knowledge context to add chunks to
            fs_path: Path to file or directory
            recursive: Whether to recursively process subdirectories
            chunk_size: Optional chunk size override
            extensions: File extensions to include (e.g., ['.txt', '.md', '.py'])
                       If None, defaults to common text file extensions.
            allowed_base: If provided, validates that fs_path is within this base
                         directory. Raises PathTraversalError if path escapes.
                         Per SPEC §18, path traversal prevention is mandatory.
            max_files: Maximum number of files to process (default: 10000)
            max_depth: Maximum directory depth for recursive mode (default: 20)

        Returns:
            Total number of chunks created

        Raises:
            PathTraversalError: If allowed_base is set and fs_path escapes it
        """
        # SECURITY: Validate path against allowed base if specified (Issue 14.1)
        if allowed_base is not None:
            base = Path(allowed_base)
            # For absolute paths, verify they're within allowed base
            path_obj = Path(fs_path)
            if path_obj.is_absolute():
                resolved = path_obj.resolve()
                base_resolved = base.resolve()
                if resolved != base_resolved and base_resolved not in resolved.parents:
                    logger.warning(
                        "ingest_path_traversal_blocked",
                        fs_path=fs_path,
                        allowed_base=str(allowed_base),
                    )
                    raise PathTraversalError(
                        f"path must be within allowed base directory: {allowed_base}"
                    )
            else:
                # For relative paths, use safe_join which validates traversal
                try:
                    path_obj = safe_join(base, fs_path)
                    fs_path = str(path_obj)  # Use validated absolute path
                except PathTraversalError:
                    logger.warning(
                        "ingest_path_traversal_blocked",
                        fs_path=fs_path,
                        allowed_base=str(allowed_base),
                    )
                    raise

        path = Path(fs_path)
        # Issue 38.3: Apply default limits to prevent resource exhaustion
        file_limit = max_files or self.MAX_INGEST_FILES
        depth_limit = max_depth or self.MAX_INGEST_DEPTH
        base_depth = len(path.resolve().parts)

        # Default extensions for text-like files
        if extensions is None:
            extensions = [
                ".txt", ".md", ".rst", ".py", ".js", ".ts", ".jsx", ".tsx",
                ".html", ".css", ".json", ".yaml", ".yml", ".xml", ".csv",
                ".sql", ".sh", ".bash", ".go", ".rs", ".java", ".c", ".cpp",
                ".h", ".hpp", ".rb", ".php", ".swift", ".kt", ".scala",
            ]

        total_chunks = 0
        files_processed = 0

        if path.is_file():
            # Single file
            if not extensions or path.suffix.lower() in extensions:
                try:
                    total_chunks += self.ingest_file(context_id, str(path), chunk_size)
                except Exception as exc:
                    logger.warning(
                        "ingest_path_file_failed",
                        path=str(path),
                        error=str(exc),
                    )
            return total_chunks

        if not path.is_dir():
            logger.warning("ingest_path_not_found", path=str(path))
            return 0

        # Directory - iterate through files with limits
        pattern = "**/*" if recursive else "*"
        for file_path in path.glob(pattern):
            if not file_path.is_file():
                continue
            if extensions and file_path.suffix.lower() not in extensions:
                continue

            # Issue 38.3: Check depth limit for recursive ingestion
            if recursive:
                file_depth = len(file_path.resolve().parts) - base_depth
                if file_depth > depth_limit:
                    continue

            # Issue 38.3: Check file count limit
            if files_processed >= file_limit:
                logger.warning(
                    "ingest_path_file_limit_reached",
                    context_id=context_id,
                    fs_path=fs_path,
                    limit=file_limit,
                )
                break

            try:
                total_chunks += self.ingest_file(context_id, str(file_path), chunk_size)
                files_processed += 1
            except Exception as exc:
                logger.warning(
                    "ingest_path_file_failed",
                    path=str(file_path),
                    error=str(exc),
                )

        logger.info(
            "ingest_path_completed",
            context_id=context_id,
            fs_path=fs_path,
            recursive=recursive,
            files_processed=files_processed,
            total_chunks=total_chunks,
        )
        return total_chunks

