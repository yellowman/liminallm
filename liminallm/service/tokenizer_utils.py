from __future__ import annotations

import math
import re
from typing import Any

from liminallm.logging import get_logger

logger = get_logger(__name__)

DEFAULT_VOCAB_SIZE = 32000
MAX_GENERATION_TOKENS = 4096


def estimate_token_count(text: str) -> int:
    """Lightweight token estimate for enforcing model window limits.

    Avoids heavyweight tokenizer dependencies while providing a conservative
    approximation that scales with both whitespace-delimited words and overall
    character length. This is used to enforce SPEC token budgets on inputs and
    assembled prompts (Issue 15.1/15.2).
    """

    if not text:
        return 0
    normalized = text.strip()
    # Count non-space spans and fall back to a character-based heuristic to
    # avoid undercounting text with long tokens or lack of whitespace.
    wordish = len(re.findall(r"\S+", normalized))
    char_estimate = math.ceil(len(normalized) / 4)
    return max(wordish, char_estimate)


def vocab_size_from_tokenizer(
    tokenizer: Any, *, fallback: int = DEFAULT_VOCAB_SIZE
) -> int:
    """Return the vocabulary size for a tokenizer with safe fallbacks.

    The helper mirrors the usage across backends and training flows while
    logging failures instead of silently swallowing them. A configurable
    fallback allows callers to tune defaults per environment.
    """

    if tokenizer is None:
        return fallback
    if hasattr(tokenizer, "vocab_size"):
        try:
            return int(tokenizer.vocab_size)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.debug("tokenizer_vocab_size_unavailable", error=str(exc))
    if hasattr(tokenizer, "get_vocab"):
        try:
            return int(len(tokenizer.get_vocab()))
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.debug("tokenizer_get_vocab_unavailable", error=str(exc))
    return fallback
