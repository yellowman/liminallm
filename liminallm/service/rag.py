from __future__ import annotations

import math
import os
import re
import unicodedata
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Union

from liminallm.logging import get_logger
from liminallm.service.bm25 import compute_bm25_scores, tokenize_text
from liminallm.service.embeddings import cosine_similarity, deterministic_embedding
from liminallm.service.fs import PathTraversalError, safe_join
from liminallm.service.late import (
    QUERY_MIN_SEGMENT_WORDS,
    maxsim,
    segment_text,
)
from liminallm.service.ranking import (
    LATE_WEIGHT,
    LEXICAL_WEIGHT,
    POOLED_WITH_LATE_WEIGHT,
    SEMANTIC_WEIGHT,
    fuse_ranks,
    ranked_positive,
)
from liminallm.storage.models import KnowledgeChunk
from liminallm.storage.postgres import PostgresStore

logger = get_logger(__name__)

# Default overlap per SPEC §2.5: "50 token overlap"
DEFAULT_OVERLAP_TOKENS = 50

# How many candidates each channel offers before the rerank. Wide enough that
# a chunk one channel buries but the other would rank first still reaches the
# rerank; capped so the rerank stays cheap on a large context.
CANDIDATE_POOL_FACTOR = 5
MAX_CANDIDATE_POOL = 100


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
    """Hybrid retriever against knowledge chunks.

    Retrieval runs two channels and fuses them: dense for meaning, lexical for
    the exact word. Neither is sound alone. A single vector of dimension d
    cannot express every top-k set of documents a query might ask for — that
    ceiling is geometric, so no encoder or training set removes it, and a
    dense-only ranking has no second opinion when it hits one. Keywords fail
    the opposite way, on anything the user phrased differently.

    An optional reranker takes the fused shortlist last. It is the only stage
    not bound by either ceiling, and the only one that can return nothing.
    """

    def __init__(
        self,
        store: PostgresStore,
        default_chunk_size: int = 400,
        *,
        rag_mode: str | Enum | None = None,
        embed: Callable[[str], List[float]] = deterministic_embedding,
        embed_many: Optional[Callable[[Sequence[str]], List[List[float]]]] = None,
        embedding_model_id: str = "text-embedding",
        semantic: bool = False,
        rerank: Optional[Callable[[str, Sequence[KnowledgeChunk]], List[KnowledgeChunk]]] = None,
        late_interaction: bool = False,
        late_segments: int = 0,
    ) -> None:
        self.store = store
        self.default_chunk_size = max(default_chunk_size, 64)
        mode_value = rag_mode.value if isinstance(rag_mode, Enum) else rag_mode
        self.rag_mode = str(mode_value or os.getenv("RAG_MODE") or "pgvector").lower()
        self.embed = embed
        # One round trip per chunk instead of one per segment. Falls back to
        # the single encoder so a caller that supplies only ``embed`` still
        # works — just slowly, which is the pre-existing behaviour.
        self.embed_many = embed_many or (
            lambda texts: [embed(text) for text in texts]
        )
        self.embedding_model_id = embedding_model_id
        # EmbeddingsService.is_semantic, carried in. Defaults to False to match
        # the kernel's default encoder, which is the hash fallback: cosine over
        # those vectors is noise and must never reach a score (SPEC §2.5).
        self.semantic = semantic
        # Injected like ``embed`` so the kernel keeps no opinion about which
        # model does the reranking, or whether one does at all. The runtime's
        # reranker decides per call and reports a budget of zero when the
        # operator has it off, so nothing here has to be rebuilt to turn it
        # on — and nothing here has to know that.
        self.rerank = rerank
        # Late interaction needs a real encoder for the same reason the dense
        # channel does: MaxSim over hash vectors is noise with extra steps.
        self.late_interaction = bool(late_interaction and semantic)
        # Floor of two, not one: a single segment is the pooled vector by
        # another name, so a caller that enabled the feature without naming a
        # segment count would index nothing at all and never be told.
        self.late_segments = max(2, late_segments)
        # Set when segment indexing fails structurally — a missing table, a
        # width mismatch. Nothing clears it, because nothing that would fix
        # those leaves this object standing.
        self._segment_index_broken = False

        self._retriever = (
            self._retrieve_pgvector
            if self._uses_pgvector()
            else self._retrieve_local_hybrid
        )

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
        # Retrievers return a shortlist, not the final answer: recall is their
        # job, and the stages below decide precision. A reranker that only ever
        # saw ``limit`` chunks could reorder them but never rescue the one that
        # placed just outside the cut.
        results = self._retriever(
            context_ids, normalized_query, limit * 2 if max_tokens else limit,
            user_id=user_id, tenant_id=tenant_id
        )

        # Filter out very short chunks (likely noise). Before reranking, so no
        # rerank slot is spent on a chunk that is about to be dropped anyway.
        if min_token_count > 0:
            results = [
                chunk for chunk in results
                if self._get_chunk_token_count(chunk) >= min_token_count
            ]

        # Both retrieval paths get the rerank stage, and both get it here
        # rather than inside themselves, so the stage sees the whole shortlist.
        if self.rerank is not None and results:
            results = list(self.rerank(normalized_query, results))

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

        for ctx_id in context_ids:
            context = self.store.get_context(ctx_id)
            if not context:
                filtered_reasons[ctx_id] = "not_found"
                continue
            visibility = (
                (context.meta or {}).get("visibility") if context.meta else None
            )
            if user_id and context.owner_user_id != user_id:
                if visibility not in {"shared", "global"}:
                    filtered_reasons[ctx_id] = "owner_mismatch"
                    continue
            # "global" is cross-tenant by design (see the visibility contract
            # in the store); only "shared" is tenant-scoped.
            if tenant_id and visibility == "shared":
                owner = self.store.get_user(context.owner_user_id)
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
        """Hybrid retrieval: two candidate pools, one rerank (SPEC §2.5)."""
        allowed_ids = self._allowed_context_ids(
            context_ids, user_id=user_id, tenant_id=tenant_id
        )
        if not allowed_ids:
            return []

        # The encoder filter belongs to the vector channels and only to them:
        # it exists so a query vector is never compared against a chunk from a
        # different encoder. Keyword search compares no vectors, and gating it
        # on encoder identity meant that changing embedding_model_id — a
        # managed setting an admin can flip — made every stored chunk invisible
        # to BM25 as well, so retrieval returned nothing at all for an exact
        # filename or error code until the whole corpus was re-ingested by
        # hand. There is no backfill job (SPEC §2.5), so "until" was forever.
        vector_filters = {"embedding_model_id": self.embedding_model_id}
        pool_size = self._pool_size(limit)

        try:
            lexical = list(
                self.store.search_chunks_lexical(  # type: ignore[attr-defined]
                    allowed_ids,
                    query,
                    pool_size,
                    filters=None,
                    user_id=user_id,
                    tenant_id=tenant_id,
                )
            )
        except Exception as exc:  # noqa: BLE001 - the other channels stand
            # Belt and braces behind the startup check: a channel failing is
            # a channel's worth of ranking lost, never the whole turn. The
            # answer degrades to the vectors rather than 500-ing.
            logger.warning("rag_lexical_channel_failed", error=str(exc))
            lexical = []

        # Without a real encoder the dense pool is not a weaker channel, it is
        # a random sample: hash-vector distance carries no meaning to sort by.
        # Asking for it would only dilute the keyword pool.
        dense: List[KnowledgeChunk] = []
        late: List[KnowledgeChunk] = []
        if self.semantic:
            query_vectors = self._query_vectors(query)
            dense = list(
                self.store.search_chunks_pgvector(  # type: ignore[attr-defined]
                    allowed_ids,
                    query,
                    query_vectors[0],
                    pool_size,
                    filters=vector_filters,
                    user_id=user_id,
                    tenant_id=tenant_id,
                )
            )
            if self.late_interaction:
                try:
                    late = self._retrieve_late(
                        allowed_ids,
                        query_vectors,
                        pool_size,
                        filters=vector_filters,
                        user_id=user_id,
                        tenant_id=tenant_id,
                    )
                except Exception as exc:  # noqa: BLE001 - the other channels stand
                    # This channel is an addition, so its failure must cost
                    # its own contribution and nothing else. The setting is
                    # hot-reloadable, so the first person to enable it on a
                    # database that never had sql/schema.sql re-applied would
                    # otherwise break every chat turn that touches RAG, not
                    # just the part of ranking that is new.
                    logger.warning("rag_late_channel_failed", error=str(exc))
                    late = []

        ranked = self._fuse(query, lexical, dense, late)
        if not ranked:
            # Silence here is a result, not a fault: with no encoder and no
            # keyword overlap there is nothing honest to ground on, and four
            # arbitrary chunks would read to the model as evidence.
            logger.info(
                "rag_no_candidates",
                semantic=self.semantic,
                context_count=len(allowed_ids),
            )
            return []

        logger.debug(
            "rag_hybrid_pool",
            semantic=self.semantic,
            lexical=len(lexical),
            dense=len(dense),
            late=len(late),
            fused=len(ranked),
        )
        return ranked

    def _query_vectors(self, query: str) -> List[List[float]]:
        """The query as its parts, for MaxSim; the whole query stays first.

        The first vector is the whole query and is what the pooled dense
        channel uses, so a multi-clause question still gets one honest
        whole-question vector even when its clauses are also embedded.
        """
        vectors = [self.embed(query)]
        if not self.late_interaction:
            return vectors
        parts = segment_text(
            query,
            max_segments=self.late_segments,
            min_words=QUERY_MIN_SEGMENT_WORDS,
        )
        if len(parts) > 1:
            vectors.extend(self.embed(part) for part in parts)
        return vectors

    def _retrieve_late(
        self,
        allowed_ids: Sequence[str],
        query_vectors: Sequence[List[float]],
        pool_size: int,
        *,
        filters: Dict[str, str],
        user_id: Optional[str],
        tenant_id: Optional[str],
    ) -> List[KnowledgeChunk]:
        """MaxSim ranking over chunks that keep their segments separately.

        Two stages, as every multi-vector retriever does it: each part of the
        query gathers candidates by nearest segment, then the candidates are
        scored exactly against *all* of their segments. Approximate search
        decides who is considered; it never decides the order.
        """
        # A share each, not first-come. The pool has to be bounded — MaxSim
        # below is pure Python over every segment of every candidate — but a
        # single overall cap is spent by the first vector before any other is
        # consulted, and the first vector is the whole query. That collapses
        # candidate generation back to single-vector recall, which is the one
        # thing this channel exists not to be: MaxSim could then only reorder
        # what a pooled vector had already found.
        share = max(1, math.ceil(pool_size / len(query_vectors)))
        candidate_ids: List[int] = []
        seen: set[int] = set()
        for vector in query_vectors:
            taken = 0
            for chunk_id in self.store.late_candidate_ids(  # type: ignore[attr-defined]
                allowed_ids,
                vector,
                share,
                filters=filters,
                user_id=user_id,
                tenant_id=tenant_id,
            ):
                if chunk_id not in seen:
                    seen.add(chunk_id)
                    candidate_ids.append(chunk_id)
                    taken += 1
                if taken >= share:
                    break
        if not candidate_ids:
            return []

        scored: List[tuple[float, KnowledgeChunk]] = []
        for chunk, segments in self.store.chunks_with_vectors(candidate_ids):  # type: ignore[attr-defined]
            score = maxsim(query_vectors, segments)
            if score > 0:
                scored.append((score, chunk))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [chunk for _score, chunk in scored[:pool_size]]

    def _pool_size(self, limit: int) -> int:
        """How many candidates each channel offers the stages above.

        Wide enough that a chunk one channel buries but the other would rank
        first still survives to fusion, and never narrower than what the
        reranker is willing to read — a reranker handed exactly the chunks
        that were going to be returned anyway can only reorder them.
        """
        appetite = int(getattr(self.rerank, "max_candidates", 0) or 0)
        wanted = max(limit * CANDIDATE_POOL_FACTOR, limit, appetite)
        return min(wanted, MAX_CANDIDATE_POOL)

    @staticmethod
    def _chunk_key(chunk: KnowledgeChunk) -> object:
        """Identity for fusion. Falls back for chunks not yet given an id."""
        if chunk.id is not None:
            return chunk.id
        return (chunk.context_id, chunk.fs_path, chunk.chunk_index)

    def _fuse(
        self,
        query: str,
        lexical: Sequence[KnowledgeChunk],
        dense: Sequence[KnowledgeChunk],
        late: Sequence[KnowledgeChunk] = (),
        *,
        lexical_is_matched: bool = True,
    ) -> List[KnowledgeChunk]:
        """Fuse the channels by rank, per SPEC §2.5.

        The lexical pool arrives ordered by ``ts_rank``, which was only ever a
        recall filter; it is reordered here by real BM25 before fusion. That
        BM25 scores against the pool rather than the whole corpus, so its IDF
        is an approximation — sound for ordering a shortlist, which is all it
        does, since the corpus-wide decision was already made by the SQL.

        The dense and late pools keep the order they arrived in: nearest and
        MaxSim are what those channels mean, and re-scoring here would say
        nothing new. When late interaction has something to say, the pooled
        vector steps back to a lower weight — it is the same signal read less
        precisely, so it should not vote twice at full strength.
        """
        chunks: Dict[object, KnowledgeChunk] = {}
        for chunk in list(lexical) + list(dense) + list(late):
            chunks.setdefault(self._chunk_key(chunk), chunk)
        if not chunks:
            return []

        channels: List[tuple[float, List[object]]] = []
        if lexical:
            # ``lexical_is_matched`` says whether membership of this pool is
            # itself the match signal. It is for the pgvector path, where the
            # store's own full-text query selected every member — so BM25 may
            # order that pool but must not empty it. Dropping its zeros
            # deleted answers the store had found: Postgres indexes "user_id"
            # as 'user' + 'id' while this tokenizer keeps it whole, so a query
            # of "user id" scored a matching chunk 0.0, and with the hash
            # encoder — where lexical is the only live channel — retrieval
            # returned nothing at all for a question the corpus answers. Any
            # pre-filtered pool re-scored by a different scorer can be emptied
            # that way; ordering is safe, discarding is not.
            #
            # The local path's pool is not pre-filtered — it is a top-N by
            # another score, and a zero there really is a non-match — so it
            # keeps the silence rule.
            scores = compute_bm25_scores(
                tokenize_text(query),
                [tokenize_text(chunk.content) for chunk in lexical],
            )
            if lexical_is_matched:
                order = sorted(
                    range(len(lexical)), key=lambda i: scores[i], reverse=True
                )
            else:
                order = ranked_positive(scores)
            channels.append(
                (LEXICAL_WEIGHT, [self._chunk_key(lexical[i]) for i in order])
            )
        if late:
            channels.append(
                (LATE_WEIGHT, [self._chunk_key(chunk) for chunk in late])
            )
        if dense:
            # The pooled vector steps back only for chunks late interaction
            # actually ranked. Weighting the whole channel down because *some
            # other* chunk had segments demoted a chunk from 0.55 to 0.25 on
            # its neighbours' behalf, with no late contribution to make up the
            # difference — buried by the arrival of a feature that had nothing
            # to say about it.
            late_keys = {self._chunk_key(chunk) for chunk in late}
            covered = [
                self._chunk_key(chunk)
                for chunk in dense
                if self._chunk_key(chunk) in late_keys
            ]
            uncovered = [
                self._chunk_key(chunk)
                for chunk in dense
                if self._chunk_key(chunk) not in late_keys
            ]
            if covered:
                channels.append((POOLED_WITH_LATE_WEIGHT, covered))
            if uncovered:
                channels.append((SEMANTIC_WEIGHT, uncovered))

        fused = fuse_ranks(channels)
        order = sorted(fused, key=lambda key: fused[key], reverse=True)
        return [chunks[key] for key in order]

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

        # Only when it will be used: with the hash fallback the store ignores
        # the vector entirely, and encoding it is a round trip bought for
        # nothing — the same reason the pgvector path stopped asking.
        query_embedding = self.embed(query) if self.semantic else None
        # A shortlist per context, not a share of the final answer: the stages
        # above still cut this to ``limit``, and a reranker needs more than the
        # answer to improve on it.
        per_context_limit = max(1, math.ceil(self._pool_size(limit) / len(allowed_ids)))
        # No encoder gate here either. This path scores keywords as well as
        # vectors, and dropping a chunk whose vector came from a previous
        # encoder takes its *text* out of reach too — so flipping
        # embedding_model_id answered nothing at all until the whole corpus
        # was re-ingested, which the pgvector path was changed to stop doing.
        # A stale vector still contributes nothing: cosine against a query
        # from another encoder is not a match, and scores nothing.
        per_context: List[List[KnowledgeChunk]] = []
        for ctx_id in allowed_ids:
            per_context.append(list(self.store.search_chunks(
                ctx_id,
                query,
                query_embedding,
                per_context_limit,
                semantic=self.semantic,
            )))

        # Rank across contexts, and interleave only to break ties.
        # Concatenating hands every slot to whichever context was listed
        # first. Interleaving alone fixes that but guarantees an irrelevant
        # context half the answer. So the union is scored again — the same
        # fusion the pgvector path uses, the only thing here that can compare
        # a chunk in one context against a chunk in another — and it is built
        # in interleaved order so that chunks the scoring cannot separate fall
        # back to a fair share rather than to whoever was listed first.
        union: List[KnowledgeChunk] = []
        for rank in range(max((len(chunks) for chunks in per_context), default=0)):
            for chunks in per_context:
                if rank < len(chunks):
                    union.append(chunks[rank])
        if not union:
            return []
        dense: List[KnowledgeChunk] = []
        if self.semantic:
            scores = [
                cosine_similarity(query_embedding, chunk.embedding) for chunk in union
            ]
            dense = [union[index] for index in ranked_positive(scores)]
        return self._fuse(query, union, dense, lexical_is_matched=False)

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
        # Issue 24.5: Normalize Unicode input so equivalent text shares canonical form
        text = unicodedata.normalize("NFC", text)
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        blob = " ".join(lines)
        if not blob:
            return self._commit_generation(context_id, source_path, [])

        # Tokenize the text per SPEC §2.5 requirement for token-based chunking
        tokens = _simple_tokenize(blob)
        if not tokens:
            return self._commit_generation(context_id, source_path, [])

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

        return self._commit_generation(context_id, source_path, chunks)

    def _commit_generation(
        self,
        context_id: str,
        source_path: Optional[str],
        chunks: List[KnowledgeChunk],
    ) -> int:
        """Make `chunks` the whole of what this context says about a path.

        A named path is replaced rather than appended to, **including by
        nothing**. These chunks claim to *be* the contents of `source_path`,
        so once new bytes are committed the previous generation's chunks make
        that claim about a file that is gone; "this generation produced no
        text" is an answer about the current bytes, not permission to keep the
        last ones. Empty input and an extractor refusal both arrive here.

        The cost is worth stating: a *re-scan* of unchanged bytes whose
        extraction fails transiently — a sandbox timeout, say — drops that
        path from retrieval until the next ingest. That is recoverable and it
        is logged, where the alternative is an index that answers with text
        the file has not held since an earlier generation, which is not.

        `inline` text has no path to be a generation of, so it is added.
        """
        replace = getattr(self.store, "replace_chunks_for_path", None)
        if source_path and callable(replace):
            replace(context_id, source_path, chunks)
        elif chunks:
            self.store.add_chunks(context_id, chunks)  # type: ignore[attr-defined]
        if chunks:
            self._index_segments(chunks)
        return len(chunks)

    def _index_segments(self, chunks: Sequence[KnowledgeChunk]) -> None:
        """Store each chunk's segment vectors for late interaction.

        Best effort on purpose: this is an extra index over content that is
        already ingested and already retrievable. If the encoder fails here,
        the chunk keeps its pooled vector and its text, and the late channel
        simply has nothing to say about it — which the fusion already treats
        as silence rather than as a bad score.
        """
        if not self.late_interaction or self._segment_index_broken:
            return
        for position, chunk in enumerate(chunks):
            if chunk.id is None:
                continue
            parts = segment_text(chunk.content, max_segments=self.late_segments)
            if len(parts) < 2:
                # One segment is the pooled vector by another name, and it
                # would earn the chunk a second full-weight vote for it.
                continue
            try:
                segments = list(zip(parts, self.embed_many(parts)))
                self.store.add_chunk_vectors(  # type: ignore[attr-defined]
                    chunk.id,
                    segments,
                    meta={"embedding_model_id": self.embedding_model_id},
                )
            except Exception as exc:  # noqa: BLE001 - the pooled vector stands
                # The write is inside the guard, not just the embed: the chunk
                # rows are already committed by here, so letting a missing
                # table or a dimension mismatch escape would fail an ingest
                # that in fact succeeded, and the retry would duplicate it.
                #
                # And stop. These failures are structural — a missing table, a
                # width mismatch — so carrying on would pay the provider for
                # `segments x remaining chunks` embeddings, throw every one
                # away, and log the same warning several thousand times.
                # Latched for the rest of this service's life, not just this
                # file: ingest_path walks a tree one file at a time, so a
                # per-call stop still paid `segments x chunks` embeddings and
                # logged an identical warning for every one of ten thousand
                # files. Changing the setting rebuilds the service, which is
                # also how an operator clears it after fixing the schema.
                self._segment_index_broken = True
                logger.warning(
                    "rag_segment_index_failed",
                    chunk_id=chunk.id,
                    error=str(exc),
                    skipped=len(chunks) - position - 1,
                )
                return

    def ingest_file(
        self, context_id: str, path: str, chunk_size: Optional[int] = None
    ) -> int:
        # Route through the shared extractor: read_text() on a PDF "succeeds"
        # and fills the index with stripped-binary garbage that then wins
        # similarity searches. Better to skip a file than to poison retrieval.
        from liminallm.service.extract import ExtractError, extract_text

        try:
            data = extract_text(Path(path))["text"]
        except ExtractError as exc:
            # A refusal is still an answer about the current bytes, so it
            # commits an empty generation rather than leaving the last
            # readable one standing as this path's contents.
            logger.warning(
                "ingest_file_skipped", path=str(path), reason=exc.reason
            )
            return self._commit_generation(context_id, path, [])
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
                    # The base is in the log line above, not in the message:
                    # this reaches the client as a 400 body, and naming the
                    # server's root is free reconnaissance for whoever is
                    # probing with traversal paths.
                    raise PathTraversalError(
                        "path is outside the allowed base directory"
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

