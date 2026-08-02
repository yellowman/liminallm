from __future__ import annotations

import hashlib
import math
from typing import Any, Callable, Iterable, List

from liminallm.logging import get_logger

logger = get_logger(__name__)

# Fixed embedding size shared across routing/RAG/clustering
EMBEDDING_DIM = 64


def validate_embedding(vec: Iterable[float], *, name: str = "embedding") -> List[float]:
    """Validate embedding vector for NaN/Infinity values (Issue 45.1).

    Args:
        vec: Embedding vector to validate
        name: Name for error messages

    Returns:
        Validated list of floats

    Raises:
        ValueError: If vector contains NaN or Infinity values
    """
    result = list(vec)
    for i, val in enumerate(result):
        if math.isnan(val):
            raise ValueError(f"{name}[{i}] contains NaN")
        if math.isinf(val):
            raise ValueError(f"{name}[{i}] contains Infinity")
    return result


def sanitize_embedding(vec: Iterable[float], *, replace_value: float = 0.0) -> List[float]:
    """Sanitize embedding by replacing NaN/Infinity with safe values (Issue 45.1).

    Args:
        vec: Embedding vector to sanitize
        replace_value: Value to use for NaN/Infinity replacement

    Returns:
        Sanitized list of floats
    """
    result = []
    for val in vec:
        if math.isnan(val) or math.isinf(val):
            result.append(replace_value)
        else:
            result.append(val)
    return result


def validate_embedding_dimension(
    vec: Iterable[float], expected_dim: int, *, name: str = "embedding"
) -> List[float]:
    """Validate embedding has expected dimension (Issue 45.2).

    Args:
        vec: Embedding vector to validate
        expected_dim: Expected dimension
        name: Name for error messages

    Returns:
        Materialized embedding list for reuse after validation

    Raises:
        ValueError: If dimension doesn't match
    """
    materialized = list(vec)
    actual = len(materialized)
    if actual != expected_dim:
        raise ValueError(f"{name} has dimension {actual}, expected {expected_dim}")
    return materialized


def deterministic_embedding(text: str, dim: int = EMBEDDING_DIM) -> List[float]:
    """Generate a small deterministic embedding without external models.

    This keeps the kernel self-contained for routing and clustering when a real
    encoder is unavailable.
    """

    if not text:
        return ensure_embedding_dim([], dim=dim)
    tokens = text.lower().split()
    vec = [0.0] * dim
    for tok in tokens:
        h = int(hashlib.sha256(tok.encode()).hexdigest(), 16)
        idx = h % dim
        vec[idx] += 1.0
    norm = sum(v * v for v in vec) ** 0.5 or 1.0
    return ensure_embedding_dim([v / norm for v in vec], dim=dim)


def cosine_similarity(
    a: Iterable[float], b: Iterable[float], *, validate: bool = True
) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        a: First vector
        b: Second vector
        validate: If True, validate for NaN/Infinity (Issue 45.1, 45.5)

    Returns:
        Cosine similarity score between -1 and 1, or 0.0 on error
    """
    list_a = list(a)
    list_b = list(b)
    if not list_a or not list_b or len(list_a) != len(list_b):
        return 0.0

    # Issue 45.1/45.5: Validate for NaN/Infinity to prevent score corruption
    if validate:
        for i, val in enumerate(list_a):
            if math.isnan(val) or math.isinf(val):
                return 0.0  # Return neutral similarity for invalid vectors
        for i, val in enumerate(list_b):
            if math.isnan(val) or math.isinf(val):
                return 0.0  # Return neutral similarity for invalid vectors

    num = sum(x * y for x, y in zip(list_a, list_b))
    denom = (sum(x * x for x in list_a) ** 0.5) * (sum(y * y for y in list_b) ** 0.5)

    if not denom:
        return 0.0

    result = num / denom

    # Issue 45.5: Validate result is bounded
    if math.isnan(result) or math.isinf(result):
        return 0.0

    # Clamp to valid cosine similarity range
    return max(-1.0, min(1.0, result))


def ensure_embedding_dim(
    vec: Iterable[float] | None,
    *,
    dim: int = EMBEDDING_DIM,
    sanitize: bool = True,
    validate_length: bool = False,
    name: str = "embedding",
) -> List[float]:
    """Ensure embedding has the correct dimension by padding or truncating.

    Args:
        vec: Input embedding vector
        dim: Target dimension
        sanitize: If True, replace NaN/Infinity with 0.0 (Issue 45.2)

    Returns:
        Embedding with exactly dim dimensions
    """
    if not vec:
        return [0.0] * dim

    materialized = list(vec)
    if validate_length and len(materialized) != dim:
        raise ValueError(f"{name} has dimension {len(materialized)}, expected {dim}")

    trimmed = materialized[:dim]

    # Issue 45.2: Sanitize NaN/Infinity values during dimension adjustment
    if sanitize:
        trimmed = [0.0 if (math.isnan(v) or math.isinf(v)) else v for v in trimmed]

    if len(trimmed) < dim:
        trimmed += [0.0] * (dim - len(trimmed))
    return trimmed


def validated_embedding(
    vec: Iterable[float] | None,
    *,
    expected_dim: int = EMBEDDING_DIM,
    name: str = "embedding",
) -> List[float]:
    """Validate and size an embedding before downstream use (Issues 45.2, 45.4)."""

    if vec is None:
        raise ValueError(f"{name} is required")
    sanitized = validate_embedding(vec, name=name)
    _ = validate_embedding_dimension(sanitized, expected_dim, name=name)
    return ensure_embedding_dim(
        sanitized, dim=expected_dim, sanitize=True, validate_length=True, name=name
    )


def pad_vectors(
    vectors: list[list[float]], *, dim: int = EMBEDDING_DIM, sanitize: bool = True
) -> list[list[float]]:
    """Pad/truncate vectors to uniform dimension.

    Args:
        vectors: List of embedding vectors
        dim: Target dimension
        sanitize: If True, sanitize NaN/Infinity values

    Returns:
        List of embeddings with uniform dimension
    """
    if not vectors:
        return []
    return [ensure_embedding_dim(v, dim=dim, sanitize=sanitize) for v in vectors]


def normalize_vector(vec: List[float]) -> List[float]:
    """Normalize a vector to unit length (Issue 45.10).

    Args:
        vec: Vector to normalize

    Returns:
        Unit-length vector, or zero vector if input has zero magnitude
    """
    if not vec:
        return vec

    # Sanitize NaN/Infinity first
    clean = [0.0 if (math.isnan(v) or math.isinf(v)) else v for v in vec]

    magnitude = math.sqrt(sum(v * v for v in clean))
    if magnitude == 0 or math.isnan(magnitude) or math.isinf(magnitude):
        return [0.0] * len(clean)

    return [v / magnitude for v in clean]


def validate_centroid(
    centroid: List[float], *, name: str = "centroid", expected_dim: int | None = EMBEDDING_DIM
) -> List[float]:
    """Validate and normalize a centroid vector (Issue 45.3).

    Args:
        centroid: Centroid vector to validate
        name: Name for error messages
        expected_dim: Optional dimension check

    Returns:
        Validated and normalized centroid

    Raises:
        ValueError: If centroid is invalid
    """
    if not centroid:
        raise ValueError(f"{name} is empty")

    if expected_dim:
        _ = validate_embedding_dimension(centroid, expected_dim, name=name)

    # Check for NaN/Infinity
    for i, val in enumerate(centroid):
        if math.isnan(val):
            raise ValueError(f"{name}[{i}] contains NaN")
        if math.isinf(val):
            raise ValueError(f"{name}[{i}] contains Infinity")

    # Normalize to prevent magnitude drift
    return normalize_vector(centroid)


_EMBED_INPUT_MAX_CHARS = 8000


def make_provider_encoder(client: Any, model_id: str) -> Callable[[str], List[float]]:
    """Real embeddings through an OpenAI-compatible client.

    Works for OpenAI, the Gemini compat endpoint, and self-hosted servers —
    whichever client the model backend already holds. A provider failure
    falls back to the deterministic hash vector, whose different dimension
    makes cosine against real vectors return a neutral 0.0 instead of noise:
    a failed embed quietly costs recall, never corrupts ranking.
    """

    def encode(text: str) -> List[float]:
        try:
            response = client.embeddings.create(
                model=model_id, input=(text or "")[:_EMBED_INPUT_MAX_CHARS]
            )
            vector = list(response.data[0].embedding)
            if vector:
                return vector
        except Exception as exc:  # noqa: BLE001 - degrade, never block
            logger.warning(
                "provider_embedding_failed", model=model_id, error=str(exc)[:200]
            )
        return deterministic_embedding(text)

    return encode


class EmbeddingsService:
    """Wrapper for embedding providers with a stable model identifier.

    ``is_semantic`` is the honesty flag: the default encoder is a
    deterministic hash whose cosine similarities are noise. Every consumer
    that blends cosine into a ranking must check it — hash cosine polluting
    a BM25 score is worse than BM25 alone.
    """

    def __init__(
        self,
        model_id: str,
        *,
        encoder: Callable[[str], List[float]] = deterministic_embedding,
        semantic: bool = False,
    ):
        self.model_id = model_id
        self._encoder = encoder
        self.is_semantic = bool(semantic)

    def embed(self, text: str) -> List[float]:
        return self._encoder(text)
