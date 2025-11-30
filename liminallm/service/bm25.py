"""BM25 scoring utilities for hybrid RAG retrieval.

Shared implementation per SPEC §2.5 retrieval strategy to avoid duplication
between PostgresStore and MemoryStore.
"""

from __future__ import annotations

import math
import re
from typing import List, Sequence

# BM25 hyperparameters (standard defaults)
BM25_K1 = 1.5
BM25_B = 0.75


def tokenize_text(text: str) -> List[str]:
    """Tokenize text for BM25 scoring.

    Simple word tokenization using regex word boundaries.
    """
    return re.findall(r"\w+", text.lower())


def compute_bm25_scores(
    query_tokens: Sequence[str],
    documents: List[List[str]],
    k1: float = BM25_K1,
    b: float = BM25_B,
) -> List[float]:
    """Compute BM25 relevance scores for documents against query tokens.

    Args:
        query_tokens: Tokenized query terms
        documents: List of tokenized documents
        k1: Term frequency saturation parameter (default 1.5)
        b: Document length normalization parameter (default 0.75)

    Returns:
        List of BM25 scores corresponding to each document
    """
    if not query_tokens or not documents:
        return [0.0 for _ in documents]

    N = len(documents)
    avgdl = sum(len(doc) for doc in documents) / float(N)

    # Compute document frequencies for IDF
    doc_freq: dict[str, int] = {}
    for doc in documents:
        seen = set(doc)
        for tok in seen:
            doc_freq[tok] = doc_freq.get(tok, 0) + 1

    # Compute BM25 score for each document
    scores: List[float] = []
    for doc in documents:
        # Term frequencies for this document
        tf: dict[str, int] = {}
        for tok in doc:
            tf[tok] = tf.get(tok, 0) + 1

        score = 0.0
        for tok in query_tokens:
            df = doc_freq.get(tok, 0)
            if df == 0:
                continue
            # IDF component with smoothing
            idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
            freq = tf.get(tok, 0)
            # BM25 term weight
            denom = freq + k1 * (1 - b + b * (len(doc) / (avgdl or 1.0)))
            score += idf * (freq * (k1 + 1)) / denom if denom else 0.0

        scores.append(score)

    return scores
