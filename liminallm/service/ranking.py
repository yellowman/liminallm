"""Rank fusion for hybrid retrieval.

Retrieval runs more than one channel — dense for meaning, lexical for the
exact word — and the two produce numbers that share no scale. Cosine is
bounded to [-1, 1]. BM25 is unbounded, and its magnitude depends on the pool
it was scored against, so normalizing it means dividing by a maximum that
moves with every query: the same chunk scores differently depending on what
it happened to be ranked beside.

Reciprocal rank fusion reads only the position a channel put a chunk in, and
never the number the channel attached to it. That removes the normalization
problem entirely, and it lets a chunk both channels rank well beat one that
only a single channel loves — which a weighted sum of scores cannot do once
the weights are fixed.

Weights say which channel is trusted more, per SPEC §2.5.
"""

from __future__ import annotations

from typing import Dict, Hashable, Iterable, List, Sequence, Tuple

# SPEC §2.5 ranking precedence: semantic leads, lexical is close behind and
# can still win when it is the channel that actually matched.
SEMANTIC_WEIGHT = 0.55
LEXICAL_WEIGHT = 0.45

# Late interaction is the stronger reading of the same semantic signal: it
# compares the query against each part of a chunk instead of against one
# averaged point. So it leads when it is available, and the pooled vector
# steps back rather than out — a pooled embedding still says something about
# the chunk as a whole that no single best-matching part does.
LATE_WEIGHT = 0.55
POOLED_WITH_LATE_WEIGHT = 0.25

# Standard RRF damping constant. It flattens the head of the 1/(k+rank) curve
# so rank 1 does not swamp ranks 2 and 3, which is exactly what lets a chunk
# ranked well by both channels overtake one ranked first by only one.
RRF_K = 60


def fuse_ranks(
    channels: Iterable[Tuple[float, Sequence[Hashable]]],
    *,
    k: int = RRF_K,
) -> Dict[Hashable, float]:
    """Weighted reciprocal rank fusion over ranked keys.

    Args:
        channels: ``(weight, keys)`` per channel, keys in rank order, best
            first. A channel that matched nothing contributes nothing, so an
            empty list is the correct way to say "no opinion".
        k: damping constant.

    Returns:
        Fused score per key. Higher is better. Keys absent from a channel
        simply take no credit from it.
    """
    scores: Dict[Hashable, float] = {}
    for weight, keys in channels:
        for rank, key in enumerate(keys, start=1):
            scores[key] = scores.get(key, 0.0) + weight / (k + rank)
    return scores


def fusion_ceiling(
    channels: Sequence[Tuple[float, Sequence[Hashable]]],
    *,
    k: int = RRF_K,
) -> float:
    """The score a key would get by placing first in every channel that spoke.

    Fused scores are tiny by construction (a two-channel first place is about
    0.016), so anything shown to a person or compared against a threshold has
    to be divided by this. Dividing by the best score actually seen would make
    the top result 1.0 every time, whatever it was; dividing by the ceiling
    keeps 1.0 meaning "ranked first everywhere it could be".

    A channel that ranked nothing is left out of the total, for the same
    reason ``fuse_ranks`` gives it no say: an empty channel is silence. Count
    its weight and a note that placed first in the only channel with an
    opinion publishes as half a bar.

    ``Sequence``, not ``Iterable``: the caller passes the same channels to
    ``fuse_ranks``, and a generator would arrive here already exhausted.
    """
    total = sum(weight for weight, keys in channels if keys)
    return total / (k + 1) if total else 0.0


def ranked_positive(scores: Sequence[float]) -> List[int]:
    """Indices a channel actually matched, best first.

    Zero is silence, not a weak opinion. A channel that scored nothing must
    not still impose an order on the documents it failed to match — under
    rank fusion that arbitrary order would carry the channel's full weight.
    """
    positive = [index for index, score in enumerate(scores) if score > 0]
    positive.sort(key=lambda index: scores[index], reverse=True)
    return positive
