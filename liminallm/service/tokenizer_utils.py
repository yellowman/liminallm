from __future__ import annotations

from typing import Any

from liminallm.logging import get_logger

logger = get_logger(__name__)

DEFAULT_VOCAB_SIZE = 32000


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
