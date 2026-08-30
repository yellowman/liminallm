from __future__ import annotations

import math
import re
from typing import Any

from liminallm.logging import get_logger

logger = get_logger(__name__)

DEFAULT_VOCAB_SIZE = 32000
MAX_GENERATION_TOKENS = 4096
# Sanity ceiling for a single inbound message. Not a model budget - that is
# derived per model in the workflow, which can prune; validation can only
# reject, so this is generous enough to accept a pasted document on a
# large-window model and still stop absurd payloads.
MAX_SINGLE_MESSAGE_TOKENS = 200_000


# CJK bills near one token per character; an estimator without this split
# undercounts it roughly fourfold, which matters everywhere this number
# gates something: message-size validation, compaction budgets, and the
# no-tokenizer fallback in TokenCounter.
_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")


def estimate_token_count(text: str) -> int:
    """Tokenizer-free estimate, deliberately conservative (over-counts).

    Splits by script: CJK characters bill near 1:1, everything else at
    4 chars/token with a word-count floor. The single estimator - the
    calibrated TokenCounter multiplies this by a learned factor; validation
    and compaction use it raw.
    """
    if not text:
        return 0
    normalized = text.strip()
    if not normalized:
        return 0
    cjk = len(_CJK_RE.findall(normalized))
    rest = len(normalized) - cjk
    wordish = len(re.findall(r"\S+", normalized))
    return max(wordish, cjk + math.ceil(rest / 4))


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
