"""Late interaction: many vectors per chunk, compared at query time.

A pooled chunk embedding has to answer for the whole chunk at once. Everything
the chunk says is averaged into one point, and the ranking is then one inner
product — which is precisely the object the dimension bound in SPEC §2.5
applies to. A chunk that covers two subjects lands between them and is the
best match for neither.

Late interaction keeps the parts separate and defers the comparison to query
time: the chunk is stored as several vectors, and the score is MaxSim — for
each part of the query, the best-matching part of the chunk, summed. The score
is no longer a single inner product, so the single-vector bound does not
describe it. That is why multi-vector models are the one architecture in the
literature that materially beats single-vector retrieval on the hard cases.

**What this is, and is not.** Segments here are sentence-sized, embedded by
the same encoder as everything else, because that encoder is reached through
an OpenAI-compatible ``/embeddings`` endpoint and such an endpoint returns one
vector per input. It cannot return per-token vectors, so this is not ColBERT
and must not be read as carrying ColBERT's published numbers. What it does
carry is the mechanism — several vectors per document, MaxSim at query time —
at a coarser granularity, and roughly an order of magnitude less storage than
per-token would cost. The seam is the encoder: a real late-interaction model
can replace ``segment_text`` + the embed call without touching the storage,
the candidate generation, or the scoring below.
"""

from __future__ import annotations

import math
import re
from typing import List, Sequence

from liminallm.service.embeddings import cosine_similarity

# Sentence-ish boundaries. Deliberately dumb: a real sentence splitter is a
# language model's worth of dependency for a gain no ranking would notice.
_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")

# A segment shorter than this carries too little to embed usefully, so it is
# merged forward rather than stored as its own vector.
MIN_SEGMENT_WORDS = 12


def segment_text(text: str, *, max_segments: int) -> List[str]:
    """Split a chunk into the parts that get their own vector.

    Short fragments merge forward, and the result is packed down to
    ``max_segments`` by joining neighbours — never by dropping them, because a
    dropped segment is content that becomes unretrievable.
    """
    if max_segments < 1 or not (text or "").strip():
        return []

    pieces = [piece.strip() for piece in _BOUNDARY.split(text) if piece.strip()]
    if not pieces:
        return []

    merged: List[str] = []
    for piece in pieces:
        if merged and len(merged[-1].split()) < MIN_SEGMENT_WORDS:
            merged[-1] = f"{merged[-1]} {piece}"
        else:
            merged.append(piece)

    # The loop can only merge a fragment forward into a segment that is still
    # short, so a trailing fragment after a long segment survives on its own.
    # Fold it backwards: at query time MaxSim takes the best segment, and a
    # one-word segment's vector is near enough to noise to win on nothing.
    if len(merged) > 1 and len(merged[-1].split()) < MIN_SEGMENT_WORDS:
        tail = merged.pop()
        merged[-1] = f"{merged[-1]} {tail}"

    if len(merged) <= max_segments:
        return merged

    # Pack into exactly max_segments buckets of near-equal size.
    per_bucket = math.ceil(len(merged) / max_segments)
    return [
        " ".join(merged[start:start + per_bucket])
        for start in range(0, len(merged), per_bucket)
    ]


def maxsim(
    query_vectors: Sequence[Sequence[float]],
    doc_vectors: Sequence[Sequence[float]],
) -> float:
    """MaxSim: for each query part, its best-matching document part, summed.

    Each term clamps at zero so a query part the document simply does not
    address contributes nothing, rather than subtracting from the parts it
    does address. A negative cosine is an absence of match, not evidence
    against the chunk.
    """
    if not query_vectors or not doc_vectors:
        return 0.0
    return sum(
        max(0.0, max(cosine_similarity(query, doc) for doc in doc_vectors))
        for query in query_vectors
    )
