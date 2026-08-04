"""Common storage utilities shared between memory and postgres implementations.

This module extracts duplicated logic to reduce code redundancy and ensure
consistent behavior across storage backends.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional

# Import from canonical location to avoid duplication
from liminallm.service.embeddings import (
    deterministic_embedding,
)
from liminallm.storage.errors import ConstraintViolation

if TYPE_CHECKING:
    pass


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
# POLICY COMPLIANCE FILTERING (SPEC §15.1)
# ============================================================================

_DEFAULT_POLICY_BLOCKLIST = (
    re.compile(r"<script", re.IGNORECASE),
    re.compile(r"(?i)child\s+sexual|csam"),
    re.compile(r"(?i)terror|bomb|explosive|weapon"),
)


def ensure_policy_compliant_texts(
    texts: Iterable[str | None], *, violation_context: Optional[Dict[str, Any]] = None
) -> None:
    """Raise ConstraintViolation if any text violates basic policy filters.

    The blocklist is intentionally conservative to keep training data free of
    obviously disallowed content. SPEC §15.1 requires preference events to be
    policy-compliant before they are persisted or used for training.
    """

    for text in texts:
        if not text:
            continue
        for pattern in _DEFAULT_POLICY_BLOCKLIST:
            if pattern.search(text):
                raise ConstraintViolation(
                    "preference policy violation",
                    {
                        "pattern": pattern.pattern,
                        **(violation_context or {}),
                    },
                )


# ============================================================================
# HYBRID SEARCH - BM25 + Semantic
# ============================================================================


# ============================================================================
# VALIDATION HELPERS
# ============================================================================


# ============================================================================
# ARTIFACT HELPERS
# ============================================================================


# ============================================================================
# DATA TRANSFORMATION HELPERS
# ============================================================================


def normalize_optional_text(value: Optional[str]) -> Optional[str]:
    """Standardize optional text fields across storage backends.

    Empty strings are treated as missing values so both in-memory and Postgres
    stores persist `None` rather than backend-specific empty strings. This
    avoids subtle discrepancies when round-tripping optional text attributes
    such as user handles or cluster labels. (docs/ISSUES.md §24.4)
    """

    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


# ============================================================================
# SEMANTIC CLUSTER HELPERS
# ============================================================================


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


def get_default_attachment_workflow_schema() -> dict:
    """Workflow used when the conversation has attachments.

    A single agent node: the model sees the attachment list (with small text
    files inlined) and decides for itself whether to search them or run code,
    so no intent classification is needed.
    """
    return {
        "kind": "workflow.chat",
        "entrypoint": "files",
        "nodes": [
            {
                "id": "files",
                "type": "tool_call",
                "tool": "agent.files_v1",
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
        {
            "kind": "tool.spec",
            "name": "agent.files_v1",
            "description": (
                "Answers using the conversation's attachments. The model calls "
                "file_search and run_python itself, as many times as it needs."
            ),
            "inputs": {"message": {"type": "string"}},
            "handler": "agent.files_v1",
            # Multiple model round-trips plus sandboxed code; the engine caps
            # this at MAX_NODE_TIMEOUT_SECONDS anyway.
            "timeout_seconds": 60,
        },
        {
            "kind": "tool.spec",
            "name": "file.search_v1",
            "description": "Search the conversation's attached files for relevant excerpts.",
            "inputs": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "optional": True},
            },
            "handler": "file.search_v1",
        },
        {
            "kind": "tool.spec",
            "name": "web.search_v1",
            "description": (
                "Search the public web. Results are sanitized and scanned for "
                "prompt injection before the model sees them."
            ),
            "inputs": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "optional": True},
            },
            "handler": "web.search_v1",
            "timeout_seconds": 30,
        },
        {
            "kind": "tool.spec",
            "name": "web.fetch_v1",
            "description": (
                "Read a web page as untrusted data. Blocks private/reserved "
                "addresses, strips hidden content, and redacts injections."
            ),
            "inputs": {"url": {"type": "string"}},
            "handler": "web.fetch_v1",
            "timeout_seconds": 30,
        },
        {
            "kind": "tool.spec",
            "name": "notes.search_v1",
            "description": (
                "Search the user's notes vault; returns titles, dates, and "
                "excerpts as data to cite."
            ),
            "inputs": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "optional": True},
            },
            "handler": "notes.search_v1",
        },
        {
            "kind": "tool.spec",
            "name": "code.python_v1",
            "description": (
                "Run Python in a resource-limited sandbox whose working "
                "directory holds copies of the conversation's attachments."
            ),
            "inputs": {"code": {"type": "string"}},
            "handler": "code.python_v1",
            "timeout_seconds": 30,
        },
    ]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def blend_centroid(
    previous: Optional[List[float]], update: Optional[List[float]], alpha: float = 0.5
) -> List[float]:
    """Compute an exponential moving average of centroid vectors."""

    if not update:
        return list(previous or [])
    if not previous:
        return list(update)
    weight = max(0.0, min(1.0, alpha))
    size = max(len(previous), len(update))
    padded_prev = list(previous) + [0.0] * (size - len(previous))
    padded_update = list(update) + [0.0] * (size - len(update))
    return [
        (1.0 - weight) * p + weight * u for p, u in zip(padded_prev, padded_update)
    ]


def clamp_success_score(score: Optional[float]) -> float:
    """Clamp success_score into [-1, 1] for router state tracking."""

    if score is None:
        return 0.0
    return max(-1.0, min(1.0, float(score)))


