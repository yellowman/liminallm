from __future__ import annotations

import hashlib
from typing import Callable, Iterable, List


def deterministic_embedding(text: str, dim: int = 64) -> List[float]:
    """Generate a small deterministic embedding without external models.

    This keeps the kernel self-contained for routing and clustering when a real
    encoder is unavailable.
    """

    if not text:
        return [0.0] * dim
    tokens = text.lower().split()
    vec = [0.0] * dim
    for tok in tokens:
        h = int(hashlib.sha256(tok.encode()).hexdigest(), 16)
        idx = h % dim
        vec[idx] += 1.0
    norm = sum(v * v for v in vec) ** 0.5 or 1.0
    return [v / norm for v in vec]


def cosine_similarity(a: Iterable[float], b: Iterable[float]) -> float:
    list_a = list(a)
    list_b = list(b)
    if not list_a or not list_b or len(list_a) != len(list_b):
        return 0.0
    num = sum(x * y for x, y in zip(list_a, list_b))
    denom = (sum(x * x for x in list_a) ** 0.5) * (sum(y * y for y in list_b) ** 0.5)
    return num / denom if denom else 0.0


def pad_vectors(vectors: list[list[float]]) -> list[list[float]]:
    if not vectors:
        return []
    dim = max(len(v) for v in vectors)
    return [list(v) + [0.0] * (dim - len(v)) for v in vectors]


class EmbeddingsService:
    """Wrapper for embedding providers with a stable model identifier."""

    def __init__(self, model_id: str, *, encoder: Callable[[str], List[float]] = deterministic_embedding):
        self.model_id = model_id
        self._encoder = encoder

    def embed(self, text: str) -> List[float]:
        return self._encoder(text)
