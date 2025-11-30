"""Common storage utilities shared between memory and postgres implementations.

This module extracts duplicated logic to reduce code redundancy and ensure
consistent behavior across storage backends.
"""

from __future__ import annotations

import json
import uuid
from ipaddress import ip_address
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

# Import from canonical location to avoid duplication
from liminallm.service.embeddings import (
    cosine_similarity,
    deterministic_embedding,
)
from liminallm.storage.errors import ConstraintViolation

if TYPE_CHECKING:
    from liminallm.storage.models import KnowledgeChunk, SemanticCluster


# ============================================================================
# TEXT EMBEDDING - Re-export from embeddings service
# ============================================================================

# Alias for backwards compatibility with storage code
def compute_text_embedding(text: Optional[str], dim: int = 64) -> List[float]:
    """Generate deterministic hash-based embedding from text.

    This is a thin wrapper around deterministic_embedding for backwards
    compatibility with storage code. Returns empty list for None/empty text.
    """
    if not text:
        return []
    return deterministic_embedding(text, dim=dim)


# ============================================================================
# HYBRID SEARCH - BM25 + Semantic
# ============================================================================

def hybrid_search_chunks(
    candidates: List["KnowledgeChunk"],
    query: str,
    query_embedding: Optional[List[float]],
    limit: int = 4,
    bm25_weight: float = 0.45,
    semantic_weight: float = 0.55,
    *,
    tokenize_fn: Callable[[str], List[str]],
    bm25_scores_fn: Callable[[List[str], List[List[str]]], List[float]],
    cosine_fn: Callable[[List[float], List[float]], float] = cosine_similarity,
) -> List["KnowledgeChunk"]:
    """Perform hybrid BM25 + semantic search over chunks.

    Combines lexical BM25 scoring with semantic similarity to find
    the most relevant chunks for a query.

    Args:
        candidates: List of chunks to search
        query: Search query text
        query_embedding: Optional query embedding vector for semantic search
        limit: Maximum results to return
        bm25_weight: Weight for lexical BM25 score (default 0.45)
        semantic_weight: Weight for semantic similarity (default 0.55)
        tokenize_fn: Function to tokenize text into tokens
        bm25_scores_fn: Function to compute BM25 scores
        cosine_fn: Function to compute cosine similarity

    Returns:
        Ranked list of chunks, deduplicated by content
    """
    if not candidates:
        return []

    # Tokenize query and documents
    query_tokens = tokenize_fn(query)
    doc_tokens = [tokenize_fn(ch.content) for ch in candidates]
    bm25_scores = bm25_scores_fn(query_tokens, doc_tokens)

    # Compute semantic scores
    semantic_scores = []
    for ch in candidates:
        if not query_embedding or not ch.embedding:
            semantic_scores.append(0.0)
            continue
        dim = min(len(query_embedding), len(ch.embedding))
        semantic_scores.append(cosine_fn(query_embedding[:dim], ch.embedding[:dim]))

    # Normalize BM25 scores and combine
    max_bm25 = max(bm25_scores) if bm25_scores else 1.0
    if max_bm25 == 0:
        max_bm25 = 1.0

    # Deduplicate by normalized content and keep highest score
    combined: Dict[str, tuple["KnowledgeChunk", float]] = {}
    for chunk, lex, sem in zip(candidates, bm25_scores, semantic_scores):
        hybrid = bm25_weight * (lex / max_bm25) + semantic_weight * sem
        # Deduplicate by normalized content
        key = " ".join(chunk.content.split()).lower() or str(chunk.id or "")
        existing = combined.get(key)
        if not existing or hybrid > existing[1]:
            combined[key] = (chunk, hybrid)

    # Sort by score and return top results
    ranked = sorted(combined.values(), key=lambda pair: pair[1], reverse=True)
    return [pair[0] for pair in ranked[:limit]]


# ============================================================================
# VALIDATION HELPERS
# ============================================================================

def validate_context_owner(
    owner_user_id: Optional[str],
    user_exists_fn: Callable[[str], bool],
) -> str:
    """Validate that context owner exists.

    Args:
        owner_user_id: User ID to validate
        user_exists_fn: Callable that returns True if user exists

    Returns:
        Validated owner_user_id

    Raises:
        ConstraintViolation: If owner is missing or doesn't exist
    """
    if not owner_user_id:
        raise ConstraintViolation(
            "context owner required", {"owner_user_id": owner_user_id}
        )
    if not user_exists_fn(owner_user_id):
        raise ConstraintViolation(
            "context owner missing", {"owner_user_id": owner_user_id}
        )
    return owner_user_id


def validate_chunk_fs_path(chunk: "KnowledgeChunk") -> None:
    """Validate that chunk has a non-empty fs_path.

    Args:
        chunk: Chunk to validate

    Raises:
        ConstraintViolation: If fs_path is missing or empty
    """
    if not chunk.fs_path or not str(chunk.fs_path).strip():
        raise ConstraintViolation(
            "fs_path required for knowledge_chunk",
            {"fs_path": chunk.fs_path}
        )


def validate_context_source_fs_path(fs_path: str) -> None:
    """Validate context source fs_path is non-empty.

    Args:
        fs_path: Path to validate

    Raises:
        ConstraintViolation: If path is empty
    """
    if not fs_path or not fs_path.strip():
        raise ConstraintViolation(
            "fs_path required for context_source",
            {"fs_path": fs_path}
        )


# ============================================================================
# ARTIFACT HELPERS
# ============================================================================

def get_artifact_validator_type(schema: dict) -> str:
    """Map schema kind to artifact validator type.

    Args:
        schema: Artifact schema dict

    Returns:
        Validator type string for use with validate_artifact()
    """
    schema_kind = schema.get("kind")
    if schema_kind == "workflow.chat":
        return "workflow"
    elif schema_kind == "tool.spec":
        return "tool"
    elif schema_kind == "adapter.lora":
        return "adapter"
    else:
        return "artifact"


def resolve_artifact_version_author(
    version_author: Optional[str],
    owner_user_id: Optional[str],
    default: str = "system_llm"
) -> str:
    """Resolve artifact version author with fallback chain.

    Args:
        version_author: Explicit version author
        owner_user_id: Artifact owner user ID
        default: Default author if neither is provided

    Returns:
        Resolved author string
    """
    return version_author or owner_user_id or default


# ============================================================================
# DATA TRANSFORMATION HELPERS
# ============================================================================

def normalize_user_meta(meta: Optional[Dict]) -> Dict:
    """Normalize user metadata with defaults.

    Args:
        meta: Input metadata dict or None

    Returns:
        Normalized metadata with email_verified = False default
    """
    normalized = meta.copy() if meta else {}
    normalized.setdefault("email_verified", False)
    return normalized


def parse_ip_address(raw_ip: Any) -> Optional[Any]:
    """Parse IP address from various formats.

    Args:
        raw_ip: Raw IP address value (string, object, or None)

    Returns:
        Parsed IP address object or None
    """
    if isinstance(raw_ip, str):
        stripped = raw_ip.strip()
        if stripped:
            return ip_address(stripped)
        return None
    return raw_ip


def parse_json_meta(raw_meta: Any) -> Optional[Dict]:
    """Parse metadata field from JSON string or dict.

    Args:
        raw_meta: Raw metadata value (string, dict, or None)

    Returns:
        Parsed dict or None
    """
    if isinstance(raw_meta, str):
        try:
            return json.loads(raw_meta)
        except Exception:
            return None
    if isinstance(raw_meta, dict):
        return raw_meta
    return None


def normalize_preference_weight(weight: Optional[float]) -> float:
    """Normalize preference event weight to default of 1.0.

    Args:
        weight: Optional weight value

    Returns:
        Weight value, defaulting to 1.0 if None
    """
    return weight if weight is not None else 1.0


# ============================================================================
# SEMANTIC CLUSTER HELPERS
# ============================================================================

def coalesce_semantic_cluster_fields(
    new_label: Optional[str],
    new_description: Optional[str],
    new_sample_ids: Optional[List[str]],
    existing: Optional["SemanticCluster"],
) -> tuple[Optional[str], Optional[str], List[str]]:
    """Coalesce semantic cluster fields with existing values.

    When updating a cluster, None values mean "keep existing", while
    explicit values override.

    Args:
        new_label: New label value (None means keep existing)
        new_description: New description value (None means keep existing)
        new_sample_ids: New sample IDs (None means keep existing)
        existing: Existing cluster or None

    Returns:
        Tuple of (label, description, sample_message_ids)
    """
    label = (
        new_label if new_label is not None
        else (existing.label if existing else None)
    )
    description = (
        new_description if new_description is not None
        else (existing.description if existing else None)
    )
    sample_message_ids = (
        new_sample_ids if new_sample_ids is not None
        else (existing.sample_message_ids if existing else [])
    )
    return label, description, sample_message_ids


# ============================================================================
# DEFAULT ARTIFACTS
# ============================================================================

def get_default_chat_workflow_schema() -> dict:
    """Generate default chat workflow schema.

    This is the standard workflow that routes messages through
    intent classification and then to appropriate handlers.

    Returns:
        Default workflow schema dict conforming to SPEC §9
    """
    return {
        "kind": "workflow.chat",
        "entrypoint": "classify",
        "nodes": [
            {
                "id": "classify",
                "type": "tool_call",
                "tool": "llm.intent_classifier_v1",
                "inputs": {"message": "${input.message}"},
                "outputs": ["intent"],
                "next": "route",
            },
            {
                "id": "route",
                "type": "switch",
                "branches": [
                    {"when": "vars.intent == 'qa_with_docs'", "next": "rag"},
                    {"when": "vars.intent == 'code_edit'", "next": "code"},
                    {"when": "true", "next": "plain_chat"},
                ],
            },
            {
                "id": "rag",
                "type": "tool_call",
                "tool": "rag.answer_with_context_v1",
                "inputs": {"message": "${input.message}"},
                "next": "end",
            },
            {
                "id": "code",
                "type": "tool_call",
                "tool": "agent.code_v1",
                "inputs": {"message": "${input.message}"},
                "next": "end",
            },
            {
                "id": "plain_chat",
                "type": "tool_call",
                "tool": "llm.generic",
                "inputs": {"message": "${input.message}"},
                "next": "end",
            },
            {"id": "end", "type": "end"},
        ],
    }


def get_default_tool_specs() -> List[dict]:
    """Generate default tool specifications.

    Returns:
        List of default tool spec dicts conforming to SPEC §10
    """
    return [
        {
            "kind": "tool.spec",
            "name": "llm.generic",
            "description": "Plain chat response from the base model.",
            "inputs": {"message": {"type": "string"}},
            "handler": "llm.generic",
        },
        {
            "kind": "tool.spec",
            "name": "rag.answer_with_context_v1",
            "description": "Retrieval augmented answer with pgvector context.",
            "inputs": {
                "message": {"type": "string"},
                "context_id": {"type": "string", "optional": True},
            },
            "handler": "rag.answer_with_context_v1",
        },
        {
            "kind": "tool.spec",
            "name": "llm.intent_classifier_v1",
            "description": "Classify user intent for routing.",
            "inputs": {"message": {"type": "string"}},
            "handler": "llm.intent_classifier_v1",
        },
        {
            "kind": "tool.spec",
            "name": "agent.code_v1",
            "description": "Code editing and generation agent.",
            "inputs": {"message": {"type": "string"}},
            "handler": "agent.code_v1",
        },
    ]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def safe_row_value(row: Any, key: str, default: Optional[Any] = None) -> Optional[Any]:
    """Safely extract value from row dict or object.

    Works with both dict-like objects and objects with attribute access.

    Args:
        row: Row data (dict-like or object)
        key: Key/attribute name
        default: Default value if not found

    Returns:
        Extracted value or default
    """
    if hasattr(row, "get"):
        return row.get(key, default)
    try:
        return row[key]
    except (KeyError, TypeError, AttributeError):
        return default


def generate_uuid() -> str:
    """Generate a new UUID string.

    Returns:
        String representation of a new UUID4
    """
    return str(uuid.uuid4())
