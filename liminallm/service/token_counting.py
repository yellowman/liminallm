"""Count tokens the way the serving model counts them.

The old heuristic (max of word count and chars/4) was fine guarding a 4096
cap with huge headroom. Against a million-token budget the absolute error
grows into thousands of tokens, and it is wrong in different directions for
different content: CJK and code undercount badly, ordinary prose overcounts.

The fix deliberately avoids depending on a vendor tokenizer library:

- **exact when we own the tokenizer.** local_gpu_lora already has the
  checkpoint's HF tokenizer resident — exact, offline, free.
- **calibrated from ground truth otherwise.** every provider returns
  ``usage.prompt_tokens`` for the prompt we just sent. Feeding that back
  (``observe``) gives a per-model correction factor that converges on the
  real tokenizer's behavior for the traffic this deployment actually sends —
  and it works for Gemini, Claude, GLM, and anything else, none of which
  tiktoken can count anyway.
- **tiktoken only if it is already installed and its data is local.** It is
  an optional extra, never required: it downloads BPE files on first use,
  which locked-down and air-gapped deployments block.

Every path degrades instead of failing, and the uncalibrated fallback
over-counts on purpose: over-counting prunes a turn early, under-counting
overruns the model.
"""

from __future__ import annotations

import math
import re
import threading
from typing import Any, Callable, Optional

from liminallm.logging import get_logger

logger = get_logger(__name__)

# Encodings for OpenAI families, used only when tiktoken is already present
# with its data cached locally.
_OPENAI_ENCODINGS = (
    ("gpt-4o", "o200k_base"),
    ("gpt-4.1", "o200k_base"),
    ("gpt-5", "o200k_base"),
    ("chatgpt-4o", "o200k_base"),
    ("o1", "o200k_base"),
    ("o3", "o200k_base"),
    ("o4", "o200k_base"),
    ("gpt-4", "cl100k_base"),
    ("gpt-3.5", "cl100k_base"),
)

# Per-message overhead in the chat wire format (role, delimiters).
MESSAGE_OVERHEAD_TOKENS = 4

# CJK bills near one token per character; the old estimator undercounted it
# roughly fourfold.
_CJK_RE = re.compile(r"[぀-ヿ㐀-鿿가-힯]")

# Calibration bounds: a correction factor outside this range means the
# observation was mismatched (tool results, images, provider-side system
# prompts), not that our estimate is that wrong.
_MIN_FACTOR = 0.5
_MAX_FACTOR = 3.0
_EMA_ALPHA = 0.25
# Ignore tiny prompts: fixed per-request overhead dominates and would skew
# the factor upward forever.
_MIN_OBSERVE_TOKENS = 200


def heuristic_token_count(text: str) -> int:
    """Tokenizer-free estimate, deliberately conservative (over-counts).

    Splits by script: CJK characters bill near 1:1, everything else at
    4 chars/token with a word-count floor.
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


def _tiktoken_encoding_name(model: str) -> Optional[str]:
    lowered = (model or "").lower()
    best: Optional[tuple[str, str]] = None
    for prefix, encoding in _OPENAI_ENCODINGS:
        if lowered.startswith(prefix) and (best is None or len(prefix) > len(best[0])):
            best = (prefix, encoding)
    return best[1] if best else None


class TokenCounter:
    """Counts tokens for one model, resolving the best available method once.

    ``method`` is "hf:<model>", "tiktoken:<encoding>", or "heuristic" —
    worth logging, since a deployment on the heuristic is one whose budget
    math is calibrated rather than exact.
    """

    def __init__(self, *, model: str = "", tokenizer: Any = None) -> None:
        self.model = model or ""
        self._lock = threading.Lock()
        self._encode: Optional[Callable[[str], Any]] = None
        self.method = "heuristic"
        self.factor = 1.0
        self.observations = 0
        self._resolve(tokenizer)

    def _resolve(self, tokenizer: Any) -> None:
        # A live tokenizer object (local checkpoint) is exact and offline —
        # strictly better than any vendor library guess.
        if tokenizer is not None and callable(getattr(tokenizer, "encode", None)):
            try:
                tokenizer.encode("probe")
                self._encode = tokenizer.encode
                self.method = f"hf:{self.model or 'local'}"
                return
            except Exception as exc:  # noqa: BLE001 - fall through
                logger.debug("hf_tokenizer_unusable", error=str(exc))

        encoding_name = _tiktoken_encoding_name(self.model)
        if encoding_name:
            try:
                import tiktoken  # optional extra; absent by default

                encoding = tiktoken.get_encoding(encoding_name)
                encoding.encode("probe")  # forces the BPE load (may download)
                self._encode = encoding.encode
                self.method = f"tiktoken:{encoding_name}"
                return
            except Exception as exc:  # noqa: BLE001 - absent or air-gapped
                logger.info(
                    "tiktoken_unavailable_using_calibrated_heuristic",
                    encoding=encoding_name,
                    error=str(exc)[:120],
                )

    @property
    def exact(self) -> bool:
        return self._encode is not None

    def count(self, text: str) -> int:
        """Tokens in ``text``: exact when possible, else calibrated estimate."""
        if not text:
            return 0
        if self._encode is not None:
            try:
                with self._lock:  # encoders are not thread-safe
                    return len(self._encode(text))
            except Exception as exc:  # noqa: BLE001 - never block a turn
                logger.warning("token_count_failed", error=str(exc))
        return math.ceil(heuristic_token_count(text) * self.factor)

    def count_messages(self, messages: list[dict]) -> int:
        """Chat-format total: content plus per-message wire overhead."""
        total = 0
        for message in messages or []:
            content = message.get("content")
            if isinstance(content, str):
                total += self.count(content)
            elif isinstance(content, list):
                # Multimodal parts: text counts; images are priced by the
                # provider and are not estimable here.
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        total += self.count(part["text"])
            total += MESSAGE_OVERHEAD_TOKENS
        return total

    def observe(self, estimated: int, actual: int) -> None:
        """Learn from a provider's reported prompt_tokens.

        The model itself is the authority on its own tokenizer; this converges
        the heuristic on real behavior for the traffic this deployment sends.
        No-op when counting is already exact.
        """
        if self.exact or estimated < _MIN_OBSERVE_TOKENS or actual <= 0:
            return
        observed = actual / max(estimated, 1)
        if not (_MIN_FACTOR <= observed <= _MAX_FACTOR):
            logger.debug(
                "token_calibration_outlier_ignored",
                model=self.model, estimated=estimated, actual=actual,
            )
            return
        with self._lock:
            self.factor = (1 - _EMA_ALPHA) * self.factor + _EMA_ALPHA * observed
            self.observations += 1
            observations = self.observations
            factor = self.factor
        if observations in (1, 5, 25) or observations % 100 == 0:
            logger.info(
                "token_calibration_updated",
                model=self.model, factor=round(factor, 3), observations=observations,
            )


_COUNTERS: dict[str, TokenCounter] = {}
_COUNTERS_LOCK = threading.Lock()


def counter_for(model: str = "", tokenizer: Any = None) -> TokenCounter:
    """Cached counter for a model id (resolution does real work: cache it)."""
    key = model or "default"
    with _COUNTERS_LOCK:
        counter = _COUNTERS.get(key)
        if counter is None:
            counter = TokenCounter(model=model, tokenizer=tokenizer)
            _COUNTERS[key] = counter
        return counter


def reset_counters_for_tests() -> None:
    with _COUNTERS_LOCK:
        _COUNTERS.clear()
