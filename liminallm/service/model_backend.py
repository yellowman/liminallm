from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Protocol, Tuple

from liminallm.config import (
    AdapterMode,
    RemoteStyle,
    get_provider_capabilities,
)
from liminallm.logging import get_logger
from liminallm.service import responses_compat
from liminallm.service.fs import safe_join
from liminallm.service.prompt_utils import extract_prompt_instructions
from liminallm.service.tokenizer_utils import (
    DEFAULT_VOCAB_SIZE,
    estimate_token_count,
    vocab_size_from_tokenizer,
)

logger = get_logger(__name__)


def get_adapter_mode(adapter: dict) -> str:
    """Extract adapter mode from schema, inferring from legacy fields if needed.

    Per SPEC §5.0.1, adapter modes determine how adapters are applied during inference:

    - LOCAL: Adapter has trained LoRA weights stored locally (fs_dir).
      Requires local JAX/transformer backend. Best for fine-tuned behavior.

    - REMOTE: Adapter is hosted by external provider (Together, LoRAX, etc.).
      Uses remote_model_id or remote_adapter_id. Supports provider scaling.

    - PROMPT: Adapter contributes only prompt/system instructions.
      No LoRA weights needed. Useful for behavior modification via prompting.

    - HYBRID: Combines LOCAL weights with PROMPT fallback.
      Uses LoRA when available, prompt instructions otherwise.
      DEFAULT for backwards compatibility: existing adapters without explicit
      mode may have both weights and prompts, so HYBRID ensures both are used.

    Mode selection priority:
    1. Explicit 'mode' field in adapter or schema
    2. Inference from 'backend' or 'provider' fields
    3. Default to HYBRID (safest for legacy adapters)

    Args:
        adapter: Adapter dict with mode, backend, provider fields

    Returns:
        AdapterMode string (local, remote, prompt, hybrid)
    """
    if not adapter:
        return AdapterMode.PROMPT

    # Check explicit mode field first
    mode = adapter.get("mode") or adapter.get("schema", {}).get("mode")
    if mode:
        return mode

    # Infer from legacy backend/provider fields
    backend = (adapter.get("backend") or "").lower()
    provider = (adapter.get("provider") or "").lower()

    if backend in {"prompt", "prompt_distill"}:
        return AdapterMode.PROMPT
    if backend in {"local", "local_lora"} or provider == "local":
        if adapter.get("prompt_instructions") or adapter.get("behavior_prompt"):
            return AdapterMode.HYBRID
        return AdapterMode.LOCAL
    if backend in {"api", "remote"} or adapter.get("remote_model_id"):
        return AdapterMode.REMOTE
    if backend == "hybrid":
        return AdapterMode.HYBRID

    # Default to HYBRID for backwards compatibility:
    # Legacy adapters may have both LoRA weights and prompt instructions,
    # so HYBRID ensures both mechanisms are available during inference.
    return AdapterMode.HYBRID


def filter_adapters_by_mode(adapters: List[dict], compatible_modes: set) -> List[dict]:
    """Filter adapters to only those compatible with the current backend mode."""
    result = []
    for adapter in adapters:
        mode = get_adapter_mode(adapter)
        if mode in compatible_modes:
            result.append(adapter)
        else:
            logger.debug(
                "adapter_mode_incompatible",
                adapter_id=adapter.get("id"),
                mode=mode,
                compatible_modes=list(compatible_modes),
            )
    return result


def validate_adapter_base_model(
    adapter: dict,
    backend_base_model: str,
    *,
    strict: bool = False,
) -> Tuple[bool, Optional[str]]:
    """Validate that adapter was trained on a compatible base model.

    Per SPEC §5.1, LoRA adapters are tied to specific base models. Using an adapter
    with an incompatible base model can produce incorrect or degraded outputs.

    Args:
        adapter: Adapter dict with optional base_model field
        backend_base_model: The base model the inference backend is using
        strict: If True, reject adapters with missing base_model field

    Returns:
        Tuple of (is_valid, warning_message)
        - is_valid: True if adapter is compatible or validation can't be determined
        - warning_message: Human-readable warning if validation failed or uncertain
    """
    if not adapter:
        return True, None

    adapter_id = adapter.get("id") or adapter.get("name") or "unknown"
    schema = adapter.get("schema", {})

    # Extract adapter's base model
    adapter_base = (
        adapter.get("base_model")
        or schema.get("base_model")
        or adapter.get("model")
        or schema.get("model")
    )

    if not adapter_base:
        if strict:
            return (
                False,
                f"Adapter '{adapter_id}' missing base_model field (strict mode)",
            )
        logger.warning(
            "adapter_base_model_missing",
            adapter_id=adapter_id,
            backend_base_model=backend_base_model,
            message="Adapter has no base_model field; compatibility cannot be verified",
        )
        return (
            True,
            f"Adapter '{adapter_id}' has no base_model; compatibility unverified",
        )

    # Normalize model names for comparison
    def normalize_model_name(name: str) -> str:
        """Normalize model name for fuzzy matching."""
        name = name.lower().strip()
        # Remove common prefixes/suffixes
        for prefix in ("models/", "model/", "hf://", "huggingface/"):
            if name.startswith(prefix):
                name = name[len(prefix) :]
        # Remove version suffixes for family comparison
        # e.g., "llama-7b-v1.0" -> "llama-7b"
        for suffix in ("-v1", "-v2", "-v1.0", "-v2.0", ".0", ".1"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
        return name

    adapter_normalized = normalize_model_name(adapter_base)
    backend_normalized = normalize_model_name(backend_base_model)

    # Check for exact match (after normalization)
    if adapter_normalized == backend_normalized:
        return True, None

    # Check for model family compatibility (e.g., llama-7b adapter on llama-7b-chat)
    # Extract base family name (first part before version/variant)
    def extract_family(name: str) -> str:
        # Split on common separators and take base
        for sep in ("-chat", "-instruct", "-base", "-hf", "-gguf"):
            if sep in name:
                name = name.split(sep)[0]
        return name

    adapter_family = extract_family(adapter_normalized)
    backend_family = extract_family(backend_normalized)

    if adapter_family == backend_family:
        # Same family but different variant - likely compatible but warn
        logger.info(
            "adapter_base_model_variant_match",
            adapter_id=adapter_id,
            adapter_base_model=adapter_base,
            backend_base_model=backend_base_model,
            message="Adapter base model is variant of backend model; proceeding with caution",
        )
        return (
            True,
            f"Adapter '{adapter_id}' trained on '{adapter_base}' (variant of '{backend_base_model}')",
        )

    # Incompatible base models
    logger.warning(
        "adapter_base_model_mismatch",
        adapter_id=adapter_id,
        adapter_base_model=adapter_base,
        backend_base_model=backend_base_model,
        message="Adapter was trained on different base model; outputs may be degraded",
    )
    if strict:
        return (
            False,
            f"Adapter '{adapter_id}' incompatible: trained on '{adapter_base}', backend uses '{backend_base_model}'",
        )
    return (
        True,
        f"Adapter '{adapter_id}' trained on '{adapter_base}' but backend uses '{backend_base_model}'",
    )


_OPENAI_SPEC = importlib.util.find_spec("openai")
if _OPENAI_SPEC:
    from openai import OpenAI as _OpenAIClient  # pragma: no cover
else:  # pragma: no cover - optional dependency absent
    _OpenAIClient = None  # type: ignore


def _safe_weight(value: Any, default: float = 1.0, *, context: str = "") -> float:
    """Coerce adapter weights to float with defensive fallback.

    Router artifacts may carry user-authored weights that fail `float()`
    coercion. Issue 39.3 requires gracefully handling these cases to avoid
    request crashes. Shared by every backend (the JAX blending path used to
    call a method that only existed on ApiAdapterBackend - dead code until
    jax was actually installed).
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        logger.warning("adapter_weight_parse_failed", context=context, value=value)
        return default


class ModelBackend(Protocol):
    """Interface for pluggable generation backends."""

    mode: str

    def generate(
        self,
        messages: List[dict],
        adapters: List[dict],
        *,
        user_id: Optional[str] = None,
    ) -> dict: ...

    def generate_stream(
        self,
        messages: List[dict],
        adapters: List[dict],
        *,
        user_id: Optional[str] = None,
    ) -> Iterator[dict]:
        """Stream tokens from the model.

        Yields dicts with:
        - {"event": "token", "data": "token_text"}
        - {"event": "message_done", "data": {"content": "full_text", "usage": {...}}}
        - {"event": "error", "data": {"code": "...", "message": "..."}}
        """
        ...


class StubBackend:
    """Stub backend for testing - returns deterministic canned responses.

    Used with MODEL_BACKEND=stub to enable smoke tests without a real LLM.
    """

    mode = "stub"
    context_window = 8192

    STUB_RESPONSE = "This is a stub response for testing purposes."

    def generate(
        self,
        messages: List[dict],
        adapters: List[dict],
        *,
        user_id: Optional[str] = None,
    ) -> dict:
        return {
            "content": self.STUB_RESPONSE,
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        }

    def generate_with_tools(
        self,
        messages: List[dict],
        tools: List[dict],
        adapters: List[dict],
        *,
        user_id: Optional[str] = None,
    ) -> dict:
        """Deterministic tool-calling stand-in for tests.

        Calls each offered tool exactly once (in order) before answering, so
        the agent loop is exercised end to end without a live model.
        """
        called = {
            m.get("name")
            for m in messages
            if m.get("role") == "tool" and m.get("name")
        }
        for tool in tools or []:
            fn = (tool or {}).get("function") or {}
            name = fn.get("name")
            if not name or name in called:
                continue
            last_user = next(
                (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"),
                "",
            )
            args = {"query": last_user} if name == "file_search" else {}
            if name == "run_python":
                args = {"code": "import os\nprint(sorted(os.listdir('.')))"}
            return {
                "content": "",
                "tool_calls": [{"id": f"stub-{name}", "name": name, "arguments": json.dumps(args)}],
                "assistant_message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"stub-{name}",
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(args)},
                        }
                    ],
                },
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
        tool_outputs = [m.get("content", "") for m in messages if m.get("role") == "tool"]
        summary = " | ".join(o[:120] for o in tool_outputs if o)
        return {
            "content": f"{self.STUB_RESPONSE} Tool results: {summary}" if summary else self.STUB_RESPONSE,
            "tool_calls": [],
            "assistant_message": {"role": "assistant", "content": self.STUB_RESPONSE},
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        }

    def generate_stream(
        self,
        messages: List[dict],
        adapters: List[dict],
        *,
        user_id: Optional[str] = None,
    ) -> Iterator[dict]:
        # Yield tokens one word at a time, with space between words (not after last)
        words = self.STUB_RESPONSE.split()
        for i, word in enumerate(words):
            # Add space after all words except the last
            token = word + " " if i < len(words) - 1 else word
            yield {"event": "token", "data": token}
        yield {
            "event": "message_done",
            "data": {
                "content": self.STUB_RESPONSE,
                "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            },
        }


# ---------------------------------------------------------------------------
# Context-window discovery
#
# The prompt budget must come from the model actually serving requests, not a
# constant. Resolution, most authoritative first:
#   1. an explicit override (admin setting / MODEL_CONTEXT_WINDOW) — handled
#      by the caller, not here;
#   2. asking the provider (Gemini's models endpoint states inputTokenLimit;
#      self-hosted OpenAI-compatible servers like vLLM/LoRAX put
#      max_model_len/context_length in /models);
#   3. a table of well-known model families (prefix-matched);
#   4. a conservative default.

DEFAULT_CONTEXT_WINDOW = 8192

# Models that reject a caller-supplied temperature (400, whole request fails).
# Omitting the parameter is always safe — every model has a sane default — so
# this list errs toward including a family rather than risking the 400.
_REASONING_PREFIXES = (
    "o1", "o3", "o4", "gpt-5",
    "gemini-3",
    # Anthropic removed sampling parameters from Opus 4.7 onward; 4.6 and
    # earlier still accept temperature.
    "claude-opus-4-7", "claude-opus-4-8", "claude-opus-5",
    "claude-sonnet-5", "claude-fable-5", "claude-mythos-5",
    # DeepSeek's thinking endpoint ignores/rejects sampling parameters.
    "deepseek-reasoner",
)


def is_reasoning_model(model: str) -> bool:
    lowered = (model or "").lower()
    return any(lowered.startswith(prefix) for prefix in _REASONING_PREFIXES)

# Longest-prefix wins. This is the *fallback* — the provider probe
# (GeminiBackend's models/{id}, an adapter server's config) is consulted first,
# and model_context_window overrides everything. So each value is the safe
# published number, never a beta or extended tier: under-guessing costs a
# little prompt budget, over-guessing overflows the window and fails the turn.
KNOWN_CONTEXT_WINDOWS: List[Tuple[str, int]] = [
    # Google. Verified against a live ListModels call: every current Gemini
    # text model reports inputTokenLimit 1048576. The image, TTS, and
    # computer-use variants are much smaller and would otherwise inherit 1M.
    ("gemini", 1_048_576),
    ("gemini-2.5-flash-image", 32_768),
    ("gemini-3.1-flash-image", 65_536),
    ("gemini-3-pro-image", 131_072),
    ("gemini-2.5-computer-use", 131_072),
    ("gemini-omni", 131_072),
    ("gemma", 262_144),
    # OpenAI. The 5.x line splits by tier, not by version: 5.6/5.5/5.4 and
    # their Pro variants are 1,050,000, but 5.4 mini/nano and 5.3-codex are
    # 400,000, and the legacy chat-latest aliases are 128,000. Each exception
    # needs its own entry or the family prefix over-guesses it.
    ("gpt-5", 400_000),
    ("gpt-5.4", 1_050_000),
    ("gpt-5.4-mini", 400_000),
    ("gpt-5.4-nano", 400_000),
    ("gpt-5.5", 1_050_000),
    ("gpt-5.6", 1_050_000),
    ("gpt-5.3-codex", 400_000),
    ("gpt-5-chat-latest", 128_000),
    ("gpt-5.1-chat-latest", 128_000),
    ("gpt-5.2-chat-latest", 128_000),
    ("gpt-5.3-chat-latest", 128_000),
    ("gpt-4.1", 1_000_000),
    ("gpt-4o", 128_000),
    ("chatgpt-4o", 128_000),
    ("gpt-4-turbo", 128_000),
    ("gpt-4", 8_192),
    ("gpt-3.5-turbo", 16_385),
    ("o1", 200_000),
    ("o3", 200_000),
    ("o4", 200_000),
    # Anthropic. The 4.6-and-later families moved to 1M; Haiku and everything
    # older stay at 200K, which is why "claude" keeps the smaller floor.
    ("claude", 200_000),
    ("claude-opus-4-6", 1_000_000),
    ("claude-opus-4-7", 1_000_000),
    ("claude-opus-4-8", 1_000_000),
    ("claude-opus-5", 1_000_000),
    ("claude-sonnet-4-6", 1_000_000),
    ("claude-sonnet-5", 1_000_000),
    ("claude-fable-5", 1_000_000),
    ("claude-mythos-5", 1_000_000),
    # xAI. Newer is not larger here: 4.5 is the current flagship at 500K while
    # 4.3 and the 4.20 deployments carry 1M. The slugs retired on 2026-05-15
    # (grok-4-fast, grok-4-0709, grok-3, grok-code-fast-1) still resolve — xAI
    # routes them to newer models — so the 256K floor under-guesses rather than
    # overflowing. grok-build-latest aliases 4.5, but an alias target can move,
    # so it takes the same conservative floor as grok-build-0.1.
    ("grok", 131_072),
    ("grok-4", 256_000),
    ("grok-4.3", 1_000_000),
    ("grok-4.5", 500_000),
    ("grok-4.20", 1_000_000),
    ("grok-build", 256_000),
    # DeepSeek. The chat/reasoner aliases track V3.2 at 128K; the V4 line
    # ships 1M.
    ("deepseek", 128_000),
    ("deepseek-v4", 1_000_000),
    # Zhipu / GLM.
    ("glm", 128_000),
    ("glm-4.7", 200_000),
    ("glm-5", 200_000),
    ("glm-5.2", 1_000_000),
    # Moonshot / Kimi.
    ("moonshot", 131_072),
    ("moonshot-v1-8k", 8_192),
    ("moonshot-v1-32k", 32_768),
    ("kimi", 256_000),
    ("kimi-k3", 1_000_000),
    # Alibaba Model Studio. The 3.5-and-later tiers and the long-lived
    # plus/flash families are 1M; qwen3-max and qwen3.6-max-preview stay at
    # 262,144 despite the newer-looking names, and qwen-long is a 10M
    # document model. The bare "qwen" floor covers self-hosted open weights,
    # whose window is set by the deployment rather than by Alibaba.
    ("qwen", 131_072),
    ("qwen-plus", 1_000_000),
    ("qwen-flash", 1_000_000),
    ("qwen-long", 10_000_000),
    ("qwen3-coder", 1_000_000),
    ("qwen3-max", 262_144),
    ("qwen3.5", 1_000_000),
    ("qwen3.6", 1_000_000),
    ("qwen3.6-max", 262_144),
    ("qwen3.7", 1_000_000),
    ("qwen3.8", 1_000_000),
    # Baichuan documents every current model at 32k; only the explicitly
    # named 128k variant is larger.
    ("baichuan", 32_768),
    ("baichuan3-turbo-128k", 128_000),
    # MiniMax publishes exact integers rather than a rounded "200k".
    ("minimax-m2", 204_800),
    ("minimax-m3", 1_000_000),
    # Mistral. Small's moving alias takes the conservative 128K reading: if
    # the alias has moved to Small 4 we merely under-budget, and if a compat
    # layer still resolves it to Small 3.2 we are right.
    ("mistral", 32_768),
    ("mistral-medium", 256_000),
    ("mistral-large", 256_000),
    ("mistral-small", 256_000),
    ("mistral-small-latest", 128_000),
    ("codestral", 128_000),
    # Cohere.
    ("command-a", 256_000),
    ("command-a-plus", 128_000),
    ("command-a-vision", 128_000),
    ("command-r", 128_000),
    # Meta Model API (Muse Spark). Llama weights served by someone else
    # belong under that host, not here.
    ("muse-spark", 1_048_576),
    # Open weights, commonly self-hosted.
    ("llama-3.1", 131_072),
    ("llama-3.2", 131_072),
    ("llama-3.3", 131_072),
    ("llama-4", 131_072),
]


# Resellers serve a smaller window than the model's native ceiling, so the
# host has to answer before the model family does: MiniMax M3 is 1M native but
# 524,288 on Together and 512K on Fireworks, and DeepSeek V4 Pro is 1M native
# but 512K on Together. Cerebras additionally varies by account tier, so these
# take the free-tier figure and let discovery or the admin setting raise it.
HOSTED_CONTEXT_WINDOWS: dict[str, List[Tuple[str, int]]] = {
    "together": [
        ("minimaxai/minimax-m3", 524_288),
        ("qwen/qwen3.6-plus", 1_000_000),
        ("qwen/qwen3.7-plus", 1_000_000),
        ("qwen/qwen3.5-9b", 262_144),
        ("moonshotai/kimi-k3", 1_000_000),
        ("moonshotai/kimi-k2.7-code", 262_144),
        ("moonshotai/kimi-k2.6", 262_144),
        ("zai-org/glm-5.2", 262_144),
        ("openai/gpt-oss-120b", 128_000),
        ("openai/gpt-oss-20b", 128_000),
        ("deepseek-ai/deepseek-v4-pro", 512_000),
        ("deepseek-ai/deepseek-v4-flash", 1_000_000),
        ("nvidia/nemotron-3-ultra", 512_300),
        ("meta-llama/llama-3.3-70b", 131_072),
        ("qwen/qwen2.5-7b", 32_768),
        ("google/gemma-4-31b", 262_144),
    ],
    "fireworks": [
        # Fireworks labels these 1040k/262k/196k and describes them as 1M in
        # prose; the round floor is the defensible reading for a fallback.
        ("accounts/fireworks/models/kimi-k3", 1_000_000),
        ("accounts/fireworks/models/glm-5p2", 1_000_000),
        ("accounts/fireworks/models/deepseek-v4-pro", 1_000_000),
        ("accounts/fireworks/models/deepseek-v4-flash", 1_000_000),
        ("accounts/fireworks/models/minimax-m3", 512_000),
        ("accounts/fireworks/models/minimax-m2p7", 196_000),
        ("accounts/fireworks/models/kimi-k2p7-code", 262_000),
        ("accounts/fireworks/models/kimi-k2p6", 262_000),
        ("accounts/fireworks/models/qwen3p7-plus", 262_000),
    ],
    "cerebras": [
        ("gpt-oss-120b", 65_000),
        ("gemma-4-31b", 65_000),
    ],
    "groq": [
        ("llama-3.1-8b-instant", 131_072),
        ("llama-3.3-70b-versatile", 131_072),
        ("openai/gpt-oss-120b", 131_072),
        ("openai/gpt-oss-20b", 131_072),
        ("groq/compound", 131_072),
        ("minimaxai/minimax-m2.7", 196_608),
        ("qwen/qwen3.6-27b", 131_072),
    ],
}


def _longest_prefix(lowered: str, table: List[Tuple[str, int]]) -> Optional[int]:
    best: Optional[Tuple[str, int]] = None
    for prefix, window in table:
        if lowered.startswith(prefix) and (best is None or len(prefix) > len(best[0])):
            best = (prefix, window)
    return best[1] if best else None


def context_window_from_table(
    model_id: str, provider: Optional[str] = None
) -> Optional[int]:
    """Longest matching prefix for this host, else for the model family.

    The host is consulted first because a reseller's serving limit overrides
    the model's native ceiling. Returns None for an unknown model — the caller
    then falls back to DEFAULT_CONTEXT_WINDOW rather than to a guess.
    """
    lowered = (model_id or "").lower()
    hosted = HOSTED_CONTEXT_WINDOWS.get((provider or "").lower())
    if hosted:
        window = _longest_prefix(lowered, hosted)
        if window:
            return window
    return _longest_prefix(lowered, KNOWN_CONTEXT_WINDOWS)


# Keys self-hosted OpenAI-compatible servers use for the model's window.
_WINDOW_KEYS = (
    "max_model_len", "context_length", "max_context_length",
    "context_window", "n_ctx", "inputTokenLimit", "input_token_limit",
)


# A listing can be long — Together publishes several hundred models — but not
# unbounded; a payload past this is treated as not naming the model.
_MAX_LISTING_ENTRIES = 2048


def _entry_names(entry: dict, model: str) -> bool:
    """Does this listing entry describe the model we asked about?

    Hosts qualify ids in their own way — Together's `moonshotai/Kimi-K3`,
    Gemini's `models/gemini-3.6-flash` — so the trailing segment counts too.
    """
    for key in ("id", "name", "model"):
        value = entry.get(key)
        if isinstance(value, str):
            lowered = value.lower()
            if lowered == model or lowered.rsplit("/", 1)[-1] == model:
                return True
    return False


def _window_from_json(payload: Any, model: Optional[str] = None) -> Optional[int]:
    """Depth-limited scan of a /models-style payload for a window field.

    With a model name, a listing is resolved by id first: scanning a
    multi-model listing freely returns whichever window appears earliest,
    which on a reseller's catalogue is some unrelated model's. That is an
    over-guess, the one direction this whole fallback chain avoids. A listing
    that does not name the model says nothing about it, so the answer is None.
    """
    def scan(node: Any, depth: int) -> Optional[int]:
        if depth > 3 or not isinstance(node, (dict, list)):
            return None
        if isinstance(node, dict):
            for key in _WINDOW_KEYS:
                value = node.get(key)
                if isinstance(value, int) and value > 0:
                    return value
                if isinstance(value, str) and value.isdigit():
                    return int(value)
            for value in node.values():
                found = scan(value, depth + 1)
                if found:
                    return found
        else:
            for item in node[:5]:
                found = scan(item, depth + 1)
                if found:
                    return found
        return None

    if model:
        entries = payload.get("data") if isinstance(payload, dict) else payload
        if isinstance(entries, list):
            wanted = model.lower()
            for entry in entries[:_MAX_LISTING_ENTRIES]:
                if isinstance(entry, dict) and _entry_names(entry, wanted):
                    return scan(entry, 0)
            return None
    return scan(payload, 0)


def probe_context_window(
    *, provider: str, model: str, base_url: Optional[str], api_key: Optional[str]
) -> Optional[int]:
    """Ask the serving endpoint for the model's window; None when it won't say.

    Best-effort by design: 5s timeout, any failure returns None and the
    caller falls back to the table. Never raises.
    """
    import httpx

    try:
        if provider in {"gemini", "google", "vertex"} and api_key:
            # The OpenAI-compat base_url nests under /v1beta/openai; the
            # native models endpoint (which states inputTokenLimit) is a
            # sibling of that prefix.
            root = "https://generativelanguage.googleapis.com/v1beta"
            if base_url and "/v1beta" in base_url:
                root = base_url.split("/v1beta")[0] + "/v1beta"
            resp = httpx.get(
                f"{root}/models/{model}",
                headers={"x-goog-api-key": api_key},
                timeout=5.0,
            )
            if resp.status_code == 200:
                return _window_from_json(resp.json())
            return None
        if base_url:
            # Self-hosted OpenAI-compatible servers (vLLM, LoRAX, LM Studio)
            # often expose the window in their models listing.
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
            for url in (f"{base_url.rstrip('/')}/models/{model}",
                        f"{base_url.rstrip('/')}/models"):
                resp = httpx.get(url, headers=headers, timeout=5.0)
                if resp.status_code == 200:
                    window = _window_from_json(resp.json(), model=model)
                    if window:
                        return window
    except Exception as exc:  # noqa: BLE001 - probing must never break serving
        logger.debug("context_window_probe_failed", error=str(exc))
    return None


def context_window_from_model_dir(model_dir: str | Path) -> Optional[int]:
    """Local HF-style checkout: the window lives in config.json."""
    try:
        config = json.loads((Path(model_dir) / "config.json").read_text())
    except Exception:  # noqa: BLE001 - missing/unparseable config
        return None
    for key in ("max_position_embeddings", "n_positions", "seq_length",
                "max_seq_len", "model_max_length"):
        value = config.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return None


class ApiAdapterBackend:
    """Backend that targets external APIs with capability-aware adapter handling.

    Supports SPEC §5.0.2 provider-specific adapter handling:
    - MODEL_ID style (OpenAI, Azure, Vertex): One fine-tuned model per request
    - ADAPTER_PARAM style (Together, LoRAX): adapter_id parameter with multi-adapter
    - PROMPT style: Inject behavior via system prompt (universal fallback)

    The backend inspects provider capabilities to determine how to format
    adapter requests, respecting multi-adapter limits and gate weight support.
    """

    # Modes compatible with this backend
    COMPATIBLE_MODES = {AdapterMode.REMOTE, AdapterMode.PROMPT, AdapterMode.HYBRID}

    def __init__(
        self,
        base_model: str,
        *,
        adapter_mode: str = "api_adapters",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        adapter_server_model: Optional[str] = None,
        provider: Optional[str] = None,
        api_key_env: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
    ) -> None:
        self.base_model = base_model
        self.adapter_server_model = adapter_server_model
        self.adapter_mode = adapter_mode
        self.mode = adapter_mode
        self._api_key = api_key
        self._base_url = base_url
        # Thinking control for reasoning models (OpenAI o-series, Gemini 2.5/3
        # via the OpenAI-compatible endpoint): "low" | "medium" | "high", or
        # "none" to disable where the provider allows it. Sent via extra_body
        # so older openai SDKs work; omitted entirely when unset.
        self._reasoning_effort = (reasoning_effort or "").strip().lower() or None
        # Env var consulted for credential rotation; provider-specific so that,
        # e.g., a Zhipu backend reads ZHIPU_API_KEY rather than OPENAI_API_KEY.
        self._api_key_env = api_key_env or "OPENAI_API_KEY"
        self._client_timeout = 30.0
        self._active_api_key: Optional[str] = None
        # Responses is the primary endpoint; None = not yet probed, False =
        # this provider only ships chat/completions (sticky for the process).
        self._responses_ok: Optional[bool] = None
        self.client = None
        self._ensure_client()
        # Infer provider from adapter_mode if not specified
        self.provider = provider or self._infer_provider(adapter_mode)
        self.capabilities = get_provider_capabilities(self.provider)
        self._context_window: Optional[int] = None

    @property
    def context_window(self) -> int:
        """The serving model's input window: probed, else table, else default.

        Resolved once and cached; a wrong guess here misprices every budget
        decision, so the provider's own answer outranks the table.
        """
        if self._context_window is None:
            model = self.adapter_server_model or self.base_model
            window = probe_context_window(
                provider=self.provider,
                model=model,
                base_url=self._base_url,
                api_key=self._active_api_key or self._api_key,
            )
            source = "probe"
            if not window:
                window = context_window_from_table(model, provider=self.provider)
                source = "table"
            if not window:
                window, source = DEFAULT_CONTEXT_WINDOW, "default"
            self._context_window = window
            # "default" means neither the provider nor the table knew this
            # model, so every turn is budgeted against 8192 — for a
            # million-token model that is under one percent of its window,
            # and nothing else says so. Worth an operator's attention.
            log = logger.warning if source == "default" else logger.info
            log(
                "model_context_window_resolved",
                model=model, provider=self.provider, window=window, source=source,
            )
        return self._context_window

    def _safe_float(self, value: Any, default: float = 1.0, *, context: str = "") -> float:
        return _safe_weight(value, default, context=context)

    def _ensure_client(self) -> None:
        """Ensure the OpenAI-compatible client reflects the latest credentials."""

        if not _OpenAIClient:
            self.client = None
            return

        # Allow runtime rotation by picking up environment updates. The env var
        # is provider-specific (self._api_key_env), so each backend reads its
        # own credentials rather than defaulting to OPENAI_API_KEY.
        env_key = os.getenv(self._api_key_env)
        active_key = env_key or self._api_key

        if not active_key:
            self.client = None
            return

        # Rebuild the client if credentials changed
        if not self.client or self._active_api_key != active_key:
            self.client = _OpenAIClient(
                api_key=active_key, base_url=self._base_url, timeout=self._client_timeout
            )
            self._active_api_key = active_key

    def _infer_provider(self, adapter_mode: str) -> str:
        """Infer provider from adapter_mode string."""
        mode_lower = (adapter_mode or "").lower()
        if mode_lower in {
            "openai",
            "anthropic",
            "azure",
            "azure_openai",
            "vertex",
            "gemini",
            "bedrock",
        }:
            return mode_lower
        if mode_lower in {"zhipu", "zhipu.ai", "glm"}:
            return "zhipu"
        if mode_lower in {"together", "together.ai"}:
            return "together"
        if mode_lower in {"lorax", "adapter_server"}:
            return mode_lower
        if mode_lower in {"sagemaker", "aws_sagemaker"}:
            return "sagemaker"
        # Default to openai-style for unknown modes
        return "openai"

    def _with_reasoning_effort(self, extra_body: Optional[dict]) -> Optional[dict]:
        """Merge the configured reasoning effort into the request extra_body."""
        if not self._reasoning_effort:
            return extra_body
        return {**(extra_body or {}), "reasoning_effort": self._reasoning_effort}

    def _sampling_params(self, model: str) -> dict:
        """Sampling args this model will accept.

        Reasoning models (OpenAI o-series, gpt-5, and Gemini 3 through the
        compat endpoint) reject any temperature other than the default and
        fail the whole request with a 400. Omitting the parameter is the
        portable choice: every model has a sane default, and only these
        models treat setting it as an error.
        """
        if is_reasoning_model(model):
            return {}
        return {"temperature": 0.2}

    def generate(
        self,
        messages: List[dict],
        adapters: List[dict],
        *,
        user_id: Optional[str] = None,
    ) -> dict:
        self._ensure_client()

        adapter_list = adapters or []
        # Process adapters based on provider capabilities
        processed = self._process_adapters_for_provider(adapter_list)
        target_model = processed["model"]
        extra_body = processed["extra_body"]
        prompt_injections = processed["prompt_injections"]

        # Inject adapter prompts if any hybrid/prompt adapters
        augmented_messages = self._inject_adapter_prompts(messages, prompt_injections)
        extra_body = self._with_reasoning_effort(extra_body)

        if self.client and self._responses_available():
            # Convert outside the try: is_unsupported() reads an AttributeError
            # or TypeError as "this SDK has no /responses", so a bug in our own
            # conversion would be blamed on the provider and turn the endpoint
            # off for the whole process.
            items = responses_compat.to_input_items(augmented_messages)
            kwargs = self._responses_kwargs(target_model, processed["extra_body"])
            response = self._try_responses(lambda: self.client.responses.create(
                input=items, **kwargs
            ))
            if response is not None:
                return {
                    "content": responses_compat.output_text(response),
                    "usage": responses_compat.usage_dict(response),
                    "adapters_applied": processed["applied"],
                }

        if self.client:
            completion = self.client.chat.completions.create(
                model=target_model,
                messages=augmented_messages,
                extra_body=extra_body,
                **self._sampling_params(target_model),
            )
            choices = getattr(completion, "choices", None) or []
            first_choice = next(iter(choices), None)
            if not first_choice:
                logger.warning(
                    "API completion returned no choices; returning empty content"
                )
                content = ""
            else:
                content = first_choice.message.content or ""
            usage = {
                "prompt_tokens": getattr(completion.usage, "prompt_tokens", 0),
                "completion_tokens": getattr(completion.usage, "completion_tokens", 0),
                "total_tokens": getattr(completion.usage, "total_tokens", 0),
            }
            return {
                "content": content,
                "usage": usage,
                "adapters_applied": processed["applied"],
            }
        fallback = augmented_messages[-1]["content"] if augmented_messages else ""
        return {
            "content": f"[api-backend model={target_model} provider={self.provider} adapters={processed['applied']}] {fallback}",
            "usage": {
                "prompt_tokens": len(fallback.split()),
                "completion_tokens": max(5, min(20, len(fallback.split()))),
            },
            "adapters_applied": processed["applied"],
        }

    @property
    def supports_tools(self) -> bool:
        """True only when a real client is configured to receive tool calls."""
        self._ensure_client()
        return self.client is not None

    def _responses_available(self) -> bool:
        if self._responses_ok is False or self.client is None:
            return False
        if not hasattr(self.client, "responses"):
            self._responses_ok = False
            return False
        return True

    def _mark_responses_unsupported(self, exc: Exception) -> None:
        logger.info(
            "responses_endpoint_unsupported",
            provider=self.provider,
            error=str(exc)[:200],
        )
        self._responses_ok = False

    def _try_responses(self, call):
        """Run one /responses call. Returns the response, or None when the
        provider turns out to have no such endpoint — the one case where
        falling back to chat/completions is correct. Any other failure is the
        provider's real answer and propagates.

        Only the call itself belongs in here. Conversion and parsing raise the
        same exception types is_unsupported() treats as "endpoint missing".
        """
        try:
            response = call()
        except Exception as exc:
            if self._responses_ok is not True and responses_compat.is_unsupported(exc):
                self._mark_responses_unsupported(exc)
                return None
            raise
        self._responses_ok = True
        return response

    def _responses_kwargs(self, model: str, extra_body: Optional[dict]) -> dict:
        """Request kwargs for /responses. Reasoning effort travels as the
        first-class `reasoning` parameter here, not extra_body."""
        kwargs: Dict[str, Any] = {"model": model, **self._sampling_params(model)}
        reasoning = responses_compat.reasoning_param(self._reasoning_effort)
        if reasoning:
            kwargs["reasoning"] = reasoning
        if extra_body:
            kwargs["extra_body"] = extra_body
        return kwargs

    def generate_with_tools(
        self,
        messages: List[dict],
        tools: List[dict],
        adapters: List[dict],
        *,
        user_id: Optional[str] = None,
    ) -> dict:
        """One turn of an OpenAI-style tool-calling exchange.

        Returns the assistant's content, any tool calls it requested, and the
        raw assistant message to append before sending tool results back — the
        caller drives the loop.
        """
        self._ensure_client()
        if not self.client:
            raise RuntimeError("tool calling requires a configured API client")

        processed = self._process_adapters_for_provider(adapters or [])
        augmented = self._inject_adapter_prompts(
            messages, processed["prompt_injections"]
        )
        if self._responses_available():
            kwargs = self._responses_kwargs(processed["model"], processed["extra_body"])
            if tools:
                kwargs["tools"] = responses_compat.to_tools(tools)
                kwargs["tool_choice"] = "auto"
            items = responses_compat.to_input_items(augmented)
            response = self._try_responses(lambda: self.client.responses.create(
                input=items, **kwargs
            ))
            if response is not None:
                content = responses_compat.output_text(response)
                calls = responses_compat.tool_calls_of(response)
                return {
                    "content": content,
                    "tool_calls": calls,
                    "assistant_message": responses_compat.assistant_message(content, calls),
                    "usage": responses_compat.usage_dict(response),
                }

        extra_body = self._with_reasoning_effort(processed["extra_body"])
        # The loop's final round offers no tools; OpenAI rejects an empty
        # tools list outright, so omit the parameters entirely.
        tool_kwargs = {"tools": tools, "tool_choice": "auto"} if tools else {}
        completion = self.client.chat.completions.create(
            model=processed["model"],
            messages=augmented,
            **tool_kwargs,
            **self._sampling_params(processed["model"]),
            extra_body=extra_body,
        )
        choices = getattr(completion, "choices", None) or []
        first = next(iter(choices), None)
        message = getattr(first, "message", None) if first else None
        raw_calls = list(getattr(message, "tool_calls", None) or []) if message else []
        tool_calls = [
            {
                "id": getattr(tc, "id", "") or "",
                "name": getattr(getattr(tc, "function", None), "name", "") or "",
                "arguments": getattr(getattr(tc, "function", None), "arguments", "") or "{}",
            }
            for tc in raw_calls
        ]
        # Round-trip the provider's own serialization so vendor extras survive.
        # Gemini attaches a `thought_signature` to each tool call inside
        # extra_content and rejects the follow-up request (400) if it is missing,
        # so a hand-rebuilt assistant message breaks multi-turn tool calling.
        assistant_message: Dict[str, Any] = {}
        if message is not None and hasattr(message, "model_dump"):
            try:
                assistant_message = message.model_dump(exclude_none=True)
            except Exception:  # pragma: no cover - defensive
                assistant_message = {}
        if not assistant_message:
            assistant_message = {
                "role": "assistant",
                "content": getattr(message, "content", None) if message else None,
            }
            if tool_calls:
                assistant_message["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": tc["arguments"]},
                    }
                    for tc in tool_calls
                ]
        assistant_message.setdefault("role", "assistant")
        usage_obj = getattr(completion, "usage", None)
        return {
            "content": (getattr(message, "content", None) if message else None) or "",
            "tool_calls": tool_calls,
            "assistant_message": assistant_message,
            "usage": {
                "prompt_tokens": getattr(usage_obj, "prompt_tokens", 0),
                "completion_tokens": getattr(usage_obj, "completion_tokens", 0),
                "total_tokens": getattr(usage_obj, "total_tokens", 0),
            },
        }

    def _stream_via_responses(
        self, messages: List[dict], model: str, processed: dict
    ):
        """Stream via /responses. Returns True if any event was emitted (the
        caller must not fall through to chat), False to fall back — which is
        only safe when nothing has been yielded yet."""
        full_content = ""
        usage: Dict[str, Any] = {}
        # Converted before the try for the same reason as the blocking paths:
        # our own AttributeError must not read as "provider has no /responses".
        items = responses_compat.to_input_items(messages)
        kwargs = self._responses_kwargs(model, processed["extra_body"])
        try:
            stream = self.client.responses.create(input=items, stream=True, **kwargs)
            for event in stream:
                etype = getattr(event, "type", "") or ""
                if etype == "response.output_text.delta":
                    delta = getattr(event, "delta", "") or ""
                    if delta:
                        full_content += delta
                        yield {"event": "token", "data": delta}
                elif etype == "response.completed":
                    usage = responses_compat.usage_dict(getattr(event, "response", None))
                elif etype in ("response.failed", "error"):
                    raise RuntimeError(str(getattr(event, "error", None) or "response failed"))
        except Exception as exc:
            # A provider without /responses fails at request time or on the
            # first read, before any token; only then is falling back safe.
            if (
                not full_content
                and self._responses_ok is not True
                and responses_compat.is_unsupported(exc)
            ):
                self._mark_responses_unsupported(exc)
                return False
            logger.error("streaming_error", error=str(exc))
            yield {"event": "error", "data": {"code": "server_error", "message": str(exc)}}
            return True
        self._responses_ok = True
        if not usage:
            # No response.completed carrying usage. The shared estimator, not a
            # word count: word counts undercount CJK roughly fourfold.
            estimated = estimate_token_count(full_content)
            usage = {
                "prompt_tokens": 0,
                "completion_tokens": estimated,
                "total_tokens": estimated,
            }
        yield {
            "event": "message_done",
            "data": {
                "content": full_content,
                "usage": usage,
                "adapters_applied": processed["applied"],
            },
        }
        return True

    def generate_stream(
        self,
        messages: List[dict],
        adapters: List[dict],
        *,
        user_id: Optional[str] = None,
    ) -> Iterator[dict]:
        """Stream tokens from the model per SPEC §18.

        Yields events:
        - {"event": "token", "data": "token_text"}
        - {"event": "message_done", "data": {"content": "full_text", "usage": {...}}}
        - {"event": "error", "data": {"code": "...", "message": "..."}}
        """
        self._ensure_client()

        adapter_list = adapters or []
        processed = self._process_adapters_for_provider(adapter_list)
        target_model = processed["model"]
        extra_body = processed["extra_body"]
        prompt_injections = processed["prompt_injections"]
        augmented_messages = self._inject_adapter_prompts(messages, prompt_injections)
        extra_body = self._with_reasoning_effort(extra_body)

        if self.client and self._responses_available():
            emitted = yield from self._stream_via_responses(
                augmented_messages, target_model, processed
            )
            if emitted:
                return

        if self.client:
            try:
                stream = self.client.chat.completions.create(
                    model=target_model,
                    messages=augmented_messages,
                    **self._sampling_params(target_model),
                    extra_body=extra_body,
                    stream=True,
                )
                full_content = ""
                prompt_tokens = 0
                completion_tokens = 0

                for chunk in stream:
                    choices = getattr(chunk, "choices", None) or []
                    if not choices:
                        continue
                    delta = choices[0].delta
                    if delta and delta.content:
                        token_text = delta.content
                        full_content += token_text
                        completion_tokens += 1
                        yield {"event": "token", "data": token_text}

                    # Check for usage in final chunk
                    if hasattr(chunk, "usage") and chunk.usage:
                        prompt_tokens = getattr(chunk.usage, "prompt_tokens", 0)
                        completion_tokens = getattr(chunk.usage, "completion_tokens", completion_tokens)

                yield {
                    "event": "message_done",
                    "data": {
                        "content": full_content,
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": prompt_tokens + completion_tokens,
                        },
                        "adapters_applied": processed["applied"],
                    },
                }
            except Exception as exc:
                logger.error("streaming_error", error=str(exc))
                yield {
                    "event": "error",
                    "data": {"code": "server_error", "message": str(exc)},
                }
        else:
            # Fallback: simulate streaming for non-API mode
            fallback = augmented_messages[-1]["content"] if augmented_messages else ""
            response = f"[api-backend model={target_model}] {fallback}"
            # Simulate token-by-token streaming. Preserve original whitespace so
            # the concatenation of streamed tokens equals message_done.content.
            for token in re.findall(r"\S+\s*", response):
                yield {"event": "token", "data": token}
            yield {
                "event": "message_done",
                "data": {
                    "content": response,
                    "usage": {
                        "prompt_tokens": len(fallback.split()),
                        "completion_tokens": len(response.split()),
                        "total_tokens": len(fallback.split()) + len(response.split()),
                    },
                    "adapters_applied": processed["applied"],
                },
            }

    def _process_adapters_for_provider(self, adapters: List[dict]) -> dict:
        """Process adapters based on provider capabilities.

        Returns dict with:
        - model: Target model ID
        - extra_body: Additional request body parameters
        - prompt_injections: List of prompt strings to inject
        - applied: List of adapter IDs that were applied
        - dropped: List of adapter IDs that were dropped
        """
        prompt_injections: List[str] = []
        remote_adapters: List[dict] = []
        applied: List[str] = []
        dropped: List[str] = []

        for adapter in adapters:
            mode = get_adapter_mode(adapter)
            adapter_id = adapter.get("id") or adapter.get("name") or "unknown"

            if mode == AdapterMode.LOCAL:
                # Local-only adapter - can't use in API mode
                logger.warning(
                    "adapter_mode_incompatible_api",
                    adapter_id=adapter_id,
                    mode=mode,
                    provider=self.provider,
                )
                dropped.append(adapter_id)
                continue

            if mode == AdapterMode.PROMPT:
                # Pure prompt adapter
                prompt = self._extract_prompt_instructions(adapter)
                if prompt:
                    prompt_injections.append(prompt)
                    applied.append(f"{adapter_id}:prompt")
                continue

            if mode == AdapterMode.HYBRID:
                # Hybrid: always extract prompt, optionally add to remote
                prompt = self._extract_prompt_instructions(adapter)
                if prompt:
                    prompt_injections.append(prompt)
                # Check if has remote component
                if adapter.get("remote_model_id") or adapter.get("remote_adapter_id"):
                    remote_adapters.append(adapter)
                    applied.append(f"{adapter_id}:hybrid")
                else:
                    applied.append(f"{adapter_id}:prompt")
                continue

            if mode == AdapterMode.REMOTE:
                remote_adapters.append(adapter)

        # Process remote adapters based on provider remote_style
        model, extra_body, remote_applied, remote_dropped = (
            self._format_remote_adapters(remote_adapters)
        )

        applied.extend(remote_applied)
        dropped.extend(remote_dropped)

        return {
            "model": model,
            "extra_body": extra_body,
            "prompt_injections": prompt_injections,
            "applied": applied,
            "dropped": dropped,
        }

    def _format_remote_adapters(
        self, adapters: List[dict]
    ) -> Tuple[str, Optional[dict], List[str], List[str]]:
        """Format remote adapters based on provider capabilities.

        Returns:
            Tuple of (model_id, extra_body, applied_ids, dropped_ids)
        """
        if not adapters:
            return self.base_model, None, [], []

        applied: List[str] = []
        dropped: List[str] = []
        caps = self.capabilities

        if caps.remote_style == RemoteStyle.MODEL_ID:
            # Provider uses fine-tuned model as endpoint (OpenAI style)
            # Can only use ONE adapter - pick highest weight or first
            selected = (
                self._select_best_adapter(adapters, max_count=1)[0]
                if adapters
                else None
            )
            if selected:
                model_id = selected.get("remote_model_id") or selected.get("model_id")
                if model_id:
                    applied.append(f"{selected.get('id', 'unknown')}:model_id")
                    # Drop other adapters
                    for a in adapters:
                        if a is not selected:
                            dropped.append(a.get("id") or "unknown")
                            logger.debug(
                                "adapter_dropped_single_model",
                                adapter_id=a.get("id"),
                                reason="provider only supports one model_id",
                                provider=self.provider,
                            )
                    return model_id, None, applied, dropped
            # No valid remote_model_id found, fall back to base
            return self.base_model, None, [], [a.get("id", "unknown") for a in adapters]

        elif caps.remote_style == RemoteStyle.ADAPTER_PARAM:
            # Provider uses adapter_id parameter (Together, LoRAX style)
            # Can use multiple adapters up to max_adapters
            selected = self._select_best_adapter(adapters, max_count=caps.max_adapters)
            adapter_ids: List[str] = []
            gate_weights: List[float] = []

            for adapter in selected:
                aid = (
                    adapter.get("remote_adapter_id")
                    or adapter.get("adapter_id")
                    or adapter.get("id")
                )
                if aid:
                    adapter_ids.append(aid)
                    applied.append(f"{adapter.get('id', 'unknown')}:adapter_param")
                    if caps.gate_weights:
                        # Use explicit None checks to handle weight=0.0 correctly
                        # (0.0 is falsy in Python but is a valid weight for disabling adapters)
                        weight = adapter.get("weight")
                        if weight is None:
                            weight = adapter.get("gate_weight")
                        if weight is None:
                            weight = 1.0
                        gate_weights.append(
                            self._safe_float(weight, context="adapter_param_gate_weight")
                        )

            # Mark dropped adapters
            for a in adapters:
                if a not in selected:
                    dropped.append(a.get("id") or "unknown")
                    logger.debug(
                        "adapter_dropped_max_exceeded",
                        adapter_id=a.get("id"),
                        max_adapters=caps.max_adapters,
                        provider=self.provider,
                    )

            if not adapter_ids:
                return self.adapter_server_model or self.base_model, None, [], []

            # Build extra_body based on provider
            extra_body: dict = {
                caps.adapter_param_name: (
                    adapter_ids if len(adapter_ids) > 1 else adapter_ids[0]
                )
            }
            if caps.gate_weights and gate_weights:
                extra_body["adapter_weights"] = (
                    gate_weights if len(gate_weights) > 1 else gate_weights[0]
                )

            return (
                self.adapter_server_model or self.base_model,
                extra_body,
                applied,
                dropped,
            )

        else:
            # RemoteStyle.NONE - shouldn't have remote adapters
            for a in adapters:
                dropped.append(a.get("id") or "unknown")
            return self.base_model, None, [], dropped

    def _select_best_adapter(self, adapters: List[dict], max_count: int) -> List[dict]:
        """Select best adapters up to max_count, sorted by weight descending."""
        if not adapters:
            return []

        # Sort by weight/gate_weight descending
        # Use explicit None checks to handle weight=0.0 correctly
        def get_weight(a: dict) -> float:
            weight = a.get("weight")
            if weight is None:
                weight = a.get("gate_weight")
            if weight is None:
                weight = 1.0
            return float(weight)

        sorted_adapters = sorted(adapters, key=get_weight, reverse=True)
        return sorted_adapters[:max_count]

    def _extract_prompt_instructions(self, adapter: dict) -> Optional[str]:
        """Extract prompt instructions from adapter for system prompt injection.

        Delegates to shared prompt_utils.extract_prompt_instructions for consistent
        behavior across all backends per SPEC §5.0.1.

        Args:
            adapter: Adapter dict with potential prompt fields

        Returns:
            Prompt instructions string or None if no suitable prompt found
        """
        adapter_id = adapter.get("id") or adapter.get("name") or "unknown"
        result = extract_prompt_instructions(adapter, log_source=adapter_id)

        if result is None:
            # Log when no prompt found for debugging
            logger.debug(
                "adapter_no_prompt_instructions",
                adapter_id=adapter_id,
                mode=get_adapter_mode(adapter),
                message="Adapter has no prompt_instructions; behavior may be undefined in prompt mode",
            )

        return result

    def _inject_adapter_prompts(
        self, messages: List[dict], prompts: List[str]
    ) -> List[dict]:
        """Inject adapter prompt instructions into message list."""
        if not prompts:
            return messages

        prompt_text = "\n".join(f"- {p}" for p in prompts)
        system_addition = f"\n\nAdapter guidance:\n{prompt_text}"

        # Find and augment system message, or prepend new one
        augmented = [dict(m) for m in messages]
        for i, msg in enumerate(augmented):
            if msg.get("role") == "system":
                augmented[i] = {
                    **msg,
                    "content": msg.get("content", "") + system_addition,
                }
                return augmented

        # No system message found, prepend one
        augmented.insert(
            0, {"role": "system", "content": f"Adapter guidance:\n{prompt_text}"}
        )
        return augmented


class LocalJaxLoRABackend:
    """Backend for local JAX generation with filesystem-backed LoRA adapters.

    Supports SPEC §5 dual-mode operation:
    - LOCAL adapters: Load weights from filesystem, apply LoRA math
    - HYBRID adapters: Load local weights, with prompt fallback
    - PROMPT adapters: Inject behavior via system prompt (no weights)

    The backend keeps a tokenizer and (optional) Flax model resident, reads
    LoRA matrices from ``fs_root`` paths, and runs a lightweight JAX forward
    pass that mirrors the training sketch in ``TrainingService``. It performs
    fixed-shape padding, enforces conservative limits, and emits usage stats
    so callers can track prompt/completion token counts.
    """

    # Modes compatible with this backend
    COMPATIBLE_MODES = {AdapterMode.LOCAL, AdapterMode.HYBRID, AdapterMode.PROMPT}

    def __init__(
        self,
        base_model: str,
        fs_root: str,
        *,
        max_seq_len: int = 512,
        max_batch_size: int = 4,
    ) -> None:
        self.base_model = base_model
        self.fs_root = Path(fs_root)
        self.mode = "local_lora"
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        # The checkpoint's config states its trained positions; max_seq_len is
        # the serving cap. The window is whichever is smaller and known.
        discovered = context_window_from_model_dir(base_model)
        self.context_window = (
            min(discovered, max_seq_len) if discovered else max_seq_len
        ) or DEFAULT_CONTEXT_WINDOW
        self.default_vocab_size = DEFAULT_VOCAB_SIZE
        self._base_vocab_size = DEFAULT_VOCAB_SIZE
        self._adapter_vocab_size: Optional[int] = None
        self._adapter_cache: Dict[str, Tuple[float, dict]] = {}
        self._tokenizer = None
        self._tokenizer_error: Optional[str] = None
        self._jax = None
        self._jnp = None
        self._rng = None
        self._device = None

    def _ensure_jax(self):
        if self._jax is not None and self._jnp is not None and self._device is not None:
            return
        import jax
        import jax.numpy as jnp

        devices = jax.devices()
        self._device = devices[0] if devices else jax.devices("cpu")[0]
        self._jax = jax
        self._jnp = jnp
        self._rng = jax.random.PRNGKey(0)

    def get_tokenizer(self):
        """The checkpoint's own tokenizer, loading it if needed.

        This is the same tokenizer used to encode prompts for generation, so
        anything counting with it counts exactly what the model will see —
        no vendor library, no network, no estimate.
        """
        self._ensure_tokenizer()
        return self._tokenizer

    def _ensure_tokenizer(self):
        if self._tokenizer is not None or self._tokenizer_error is not None:
            return
        try:  # pragma: no cover - optional dependency
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.base_model)
            self._base_vocab_size = vocab_size_from_tokenizer(
                self._tokenizer, fallback=self.default_vocab_size
            )
        except Exception as exc:  # pragma: no cover - optional dependency
            self._tokenizer = None
            self._base_vocab_size = self.default_vocab_size
            self._tokenizer_error = str(exc)
            logger.warning(
                "tokenizer_load_failed", base_model=self.base_model, error=str(exc)
            )

    def _vocab_size(self) -> int:
        if isinstance(self._adapter_vocab_size, int) and self._adapter_vocab_size > 0:
            return self._adapter_vocab_size
        self._ensure_tokenizer()
        return self._base_vocab_size

    def _normalize_messages(self, messages: List[dict]) -> str:
        if not messages:
            return ""
        return "\n".join(
            [f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages]
        )

    def _apply_adapter_vocab_size(self, adapter: dict) -> None:
        self._adapter_vocab_size = None
        if not isinstance(adapter, dict):
            return
        schema = adapter.get("schema") or {}
        vocab_size = schema.get("vocab_size")
        if isinstance(vocab_size, int) and vocab_size > 0:
            self._adapter_vocab_size = vocab_size

    def _deterministic_token_hash(self, token: str, vocab_size: int) -> int:
        """Compute deterministic hash for a token.

        Unlike Python's built-in hash(), this produces consistent results across
        Python processes and versions by using a fixed-seed algorithm based on
        character codes. This ensures reproducibility for tokenizer fallback mode.

        The algorithm uses FNV-1a hash which is fast, simple, and deterministic.
        """
        # FNV-1a parameters for 32-bit
        FNV_PRIME = 0x01000193
        FNV_OFFSET = 0x811C9DC5

        h = FNV_OFFSET
        for char in token.encode("utf-8"):
            h ^= char
            h = (h * FNV_PRIME) & 0xFFFFFFFF  # Keep 32-bit

        return h % vocab_size

    def _tokenize(self, text: str) -> Tuple[List[int], List[int]]:
        """Tokenize text into token IDs and attention mask.

        Uses the configured tokenizer if available, otherwise falls back to
        a deterministic whitespace-based tokenizer with FNV-1a hashing.

        Returns:
            Tuple of (token_ids, attention_mask)
        """
        self._ensure_tokenizer()
        if self._tokenizer:
            encoded = self._tokenizer(
                text,
                truncation=True,
                max_length=self.max_seq_len,
                return_tensors="np",
            )
            ids = encoded["input_ids"][0].tolist()
            attention = encoded["attention_mask"][0].tolist()
            return ids, attention

        # Fallback: deterministic whitespace tokenization with FNV-1a hash
        # This produces consistent results across Python versions and processes
        vocab_size = self._vocab_size()
        tokens = text.split()
        ids = [
            self._deterministic_token_hash(tok, vocab_size)
            for tok in tokens[: self.max_seq_len]
        ]
        attention = [1] * len(ids)
        return ids, attention

    def _pad_batch(
        self, ids: List[int], attention: List[int]
    ) -> Tuple[List[int], List[int]]:
        length = min(len(ids), self.max_seq_len)
        ids = ids[:length]
        attention = attention[:length]
        if not ids:
            ids = [0]
            attention = [0]
        return ids, attention

    def _load_adapter_weights(
        self,
        adapter: dict,
        *,
        user_id: Optional[str] = None,
        strict_base_model: bool = False,
    ) -> dict:
        """Load adapter weights from filesystem with checksum and base model verification.

        Per SPEC §18, checksum of params is verified against schema.checksum before activation.
        Per SPEC §5.1, base model compatibility is validated to prevent degraded outputs.

        Args:
            adapter: Adapter dict with weights path and metadata
            user_id: User context for ownership validation
            strict_base_model: If True, reject adapters with incompatible base model

        Returns:
            Weight dict with LoRA matrices as JAX arrays

        Raises:
            ValueError: If checksum mismatch or base model incompatible (in strict mode)
        """
        if not adapter:
            return {}
        adapter_id = adapter.get("id", "unknown")

        # Validate base model compatibility before loading weights
        is_compatible, warning = validate_adapter_base_model(
            adapter, self.base_model, strict=strict_base_model
        )
        if not is_compatible:
            raise ValueError(
                warning or f"Adapter '{adapter_id}' incompatible with base model"
            )
        if warning:
            # Log warning but continue - adapter may still work
            logger.info(
                "adapter_base_model_warning",
                adapter_id=adapter_id,
                warning=warning,
            )

        path = Path(self._adapter_path(adapter, requested_user_id=user_id))
        # SPEC §5.4.6: only the promoted version may be served. current_version
        # is authoritative - without it, resolution would fall back to "newest
        # directory on disk", which serves weights the eval gate rejected.
        schema = adapter.get("schema") if isinstance(adapter.get("schema"), dict) else {}
        current_version = adapter.get("current_version")
        if current_version is None:
            current_version = (schema or {}).get("current_version")
        params_path = self._resolve_params_path(path, current_version=current_version)
        if not params_path:
            return {}
        mtime = params_path.stat().st_mtime
        cached = self._adapter_cache.get(adapter_id)
        if cached and cached[0] == mtime:
            return cached[1]
        payload = params_path.read_bytes()
        checksum = adapter.get("checksum") or adapter.get("schema", {}).get("checksum")
        if checksum:
            # SPEC §18: checksum verified against schema.checksum before activation
            digest = hashlib.sha256(payload).hexdigest()
            if digest != checksum:
                logger.error(
                    "adapter_checksum_mismatch",
                    adapter_id=adapter_id,
                    path=str(params_path),
                    expected=checksum,
                    actual=digest,
                )
                raise ValueError(
                    "adapter checksum mismatch - refusing to load potentially tampered weights"
                )
        else:
            # SPEC §18 requires checksum verification; missing checksums are a security concern
            logger.warning(
                "adapter_checksum_missing",
                adapter_id=adapter_id,
                path=str(params_path),
                message="Adapter loaded without checksum verification - add schema.checksum for production use",
            )
        # Issue 39.1: Add error handling for JSON deserialization
        try:
            weights_raw = json.loads(payload.decode())
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            logger.error(
                "adapter_json_decode_error",
                adapter_id=adapter_id,
                path=str(params_path),
                error=str(exc),
            )
            raise ValueError(f"Invalid adapter params.json: {exc}") from exc
        self._ensure_jax()
        weights = {
            k: self._jnp.array(v, dtype=self._jnp.float32)
            for k, v in weights_raw.items()
        }
        self._adapter_cache[adapter_id] = (mtime, weights)
        return weights

    def _resolve_params_path(
        self, path: Path, *, current_version: Optional[int] = None
    ) -> Optional[Path]:
        if path.is_file() and path.name == "params.json":
            return path
        # When the artifact records a promoted version, serve exactly that
        # version (or the `latest` pointer maintained alongside it). Never fall
        # back to scanning for the newest directory: an un-promoted version
        # left on disk by a gate-rejected training run would win that scan.
        if current_version:
            try:
                pinned = int(current_version)
            except (TypeError, ValueError):
                pinned = 0
            if pinned > 0:
                exact = path / f"v{pinned:04d}" / "params.json"
                if exact.exists():
                    return exact
                latest_pinned = path / "latest" / "params.json"
                if latest_pinned.exists():
                    return latest_pinned
                logger.warning(
                    "adapter_promoted_version_missing",
                    adapter_path=str(path),
                    current_version=pinned,
                )
                return None
        candidates: list[Path] = []
        direct = path / "params.json"
        if direct.exists():
            candidates.append(direct)
        latest = path / "latest" / "params.json"
        if latest.exists():
            candidates.append(latest)
        versioned = [p for p in path.glob("v*/params.json") if p.parent.is_dir()]
        versioned.sort(key=lambda p: self._version_sort_key(p.parent.name))
        candidates.extend(versioned)
        wildcard = [p for p in path.glob("*/params.json") if p.parent.is_dir()]
        wildcard.sort(key=lambda p: p.stat().st_mtime)
        candidates.extend(wildcard)
        for candidate in reversed(candidates):
            if candidate.exists():
                return candidate
        return None

    def _version_sort_key(self, name: str) -> Tuple[int, str]:
        try:
            if name.startswith("v"):
                return int(name[1:]), name
            return int(name), name
        except ValueError:
            return 0, name

    def _align_last_dim(self, arr, width: int):
        current = arr.shape[-1]
        if current > width:
            slices = (slice(None),) * (arr.ndim - 1) + (slice(0, width),)
            return arr[slices]
        if current < width:
            pad = [(0, 0)] * (arr.ndim - 1) + [(0, width - current)]
            return self._jnp.pad(arr, pad)
        return arr

    def _lora_forward(self, params: dict, inputs):
        hidden_dim = max(
            (mat.shape[1] for name, mat in params.items() if name.endswith(".A")),
            default=16,
        )
        vocab_size = max(self._vocab_size(), 1)

        if inputs.ndim == 2:
            if inputs.size:
                max_token = int(self._jnp.max(inputs))
                vocab_size = max(vocab_size, max_token + 1)
            emb_table = self._jnp.sin(
                self._jnp.arange(
                    vocab_size * hidden_dim, dtype=self._jnp.float32
                ).reshape(vocab_size, hidden_dim)
                / float(hidden_dim)
            )
            clipped = self._jnp.clip(inputs, 0, vocab_size - 1)
            embeds = emb_table[clipped]
        elif inputs.ndim == 3:
            embeds = self._jnp.asarray(inputs, dtype=self._jnp.float32)
        else:
            raise ValueError("inputs must be token IDs (2D) or embeddings (3D)")

        embeds = self._align_last_dim(embeds, hidden_dim)
        acc = self._jnp.zeros_like(embeds, dtype=self._jnp.float32)
        for name, mat in params.items():
            if not name.endswith(".A"):
                continue
            b_key = name.replace(".A", ".B")
            if b_key not in params:
                continue
            inputs_aligned = self._align_last_dim(embeds, mat.shape[1])
            base = inputs_aligned @ mat.T
            update = base @ params[b_key].T
            update = self._align_last_dim(update, acc.shape[-1])
            acc = acc + update
        return embeds + acc

    def _decode(self, token_ids: List[int]) -> str:
        self._ensure_tokenizer()
        if self._tokenizer:
            try:  # pragma: no cover - optional dependency
                return self._tokenizer.decode(token_ids, skip_special_tokens=True)
            except Exception as exc:
                logger.warning(
                    "tokenizer_decode_failed",
                    base_model=self.base_model,
                    error=str(exc),
                )
        return " ".join([f"tok-{tid}" for tid in token_ids])

    def _sample_tokens(self, lora_scores, seed_token: int) -> List[int]:
        vocab = self._vocab_size()
        score = float(self._jnp.mean(lora_scores)) if lora_scores.size else 0.0
        token = int(seed_token)
        generated: List[int] = []
        for _ in range(32):
            token = int(abs(token + score)) % vocab
            generated.append(token)
            score = score * 0.9 + 0.1 * token
        return generated

    def generate(
        self,
        messages: List[dict],
        adapters: List[dict],
        *,
        user_id: Optional[str] = None,
    ) -> dict:
        prompt = self._normalize_messages(messages)
        adapter = adapters[0] if adapters else {}
        self._apply_adapter_vocab_size(adapter)
        ids, attention = self._tokenize(prompt)

        # Handle empty prompts gracefully
        if not ids:
            return {
                "content": "",
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "model": self.base_model,
                    "adapter_id": None,
                    "latency_ms": 0.0,
                },
            }

        ids, attention = self._pad_batch(ids, attention)
        if len(ids) > self.max_seq_len:
            raise ValueError(f"prompt exceeds max length ({self.max_seq_len})")
        if len(attention) > self.max_seq_len:
            raise ValueError(f"attention mask exceeds max length ({self.max_seq_len})")
        if self.max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")

        self._ensure_jax()
        token_array = self._jnp.array([ids], dtype=self._jnp.int32)
        attn_array = self._jnp.array([attention], dtype=self._jnp.int32)
        token_array = self._jax.device_put(token_array, self._device)
        attn_array = self._jax.device_put(attn_array, self._device)

        weights = (
            self._blend_adapter_weights(adapters, user_id=user_id) if adapters else {}
        )
        start = time.perf_counter()
        lora_scores = (
            self._lora_forward(weights, token_array)
            if weights
            else self._jnp.zeros_like(token_array)
        )
        lora_scores = lora_scores * attn_array
        # Use the last token as seed; array is guaranteed non-empty due to check above
        generated_ids = self._sample_tokens(lora_scores, seed_token=token_array[0][-1])
        completion = self._decode(generated_ids)
        duration = time.perf_counter() - start

        return {
            "content": completion,
            "usage": {
                "prompt_tokens": len(ids),
                "completion_tokens": len(generated_ids),
                "model": self.base_model,
                "adapter_id": (
                    ",".join(str(a.get("id")) for a in adapters if a.get("id"))
                    if adapters
                    else None
                ),
                "latency_ms": round(duration * 1000, 2),
            },
        }

    def generate_stream(
        self,
        messages: List[dict],
        adapters: List[dict],
        *,
        user_id: Optional[str] = None,
    ) -> Iterator[dict]:
        """Stream tokens from local LoRA model per SPEC §18.

        For local models, we simulate streaming by yielding tokens one at a time
        from the generated response.
        """
        try:
            result = self.generate(messages, adapters, user_id=user_id)
            content = result.get("content", "")
            # Simulate token streaming by yielding characters/words
            for char in content:
                yield {"event": "token", "data": char}
            yield {
                "event": "message_done",
                "data": {
                    "content": content,
                    "usage": result.get("usage", {}),
                },
            }
        except Exception as exc:
            logger.error("local_streaming_error", error=str(exc))
            yield {
                "event": "error",
                "data": {"code": "server_error", "message": str(exc)},
            }

    def _blend_adapter_weights(
        self, adapters: List[dict], user_id: Optional[str]
    ) -> dict:
        """Blend multiple adapter weights using router-assigned gate weights.

        Per SPEC §5.2, effective weight composition is:
            W_eff = W_base + Σ_j (g_j * α_j * B_j @ A_j)

        Where g_j is the gate weight from the router. This implementation
        respects per-adapter weights rather than simple averaging.

        Args:
            adapters: List of adapter dicts, each may have 'weight' or 'gate_weight'
            user_id: User context for path resolution and ownership checks

        Returns:
            Combined weight dict with properly weighted LoRA matrices
        """
        if not adapters:
            return {}

        combined: dict[str, Any] = {}
        total_weight: dict[str, float] = {}

        for adapter in adapters:
            weights = self._load_adapter_weights(adapter, user_id=user_id)
            if not weights:
                continue

            # Extract gate weight from adapter (router-assigned or default 1.0)
            # Note: Can't use `or` chain because 0.0 is falsy in Python
            gate_weight = adapter.get("weight")
            if gate_weight is None:
                gate_weight = adapter.get("gate_weight")
            if gate_weight is None:
                gate_weight = adapter.get("schema", {}).get("weight")
            if gate_weight is None:
                gate_weight = 1.0
            gate_weight = _safe_weight(
                gate_weight, default=1.0, context="blend_gate_weight"
            )

            # Clamp gate weight to [0, 1] per SPEC §8.1 guardrails
            gate_weight = max(0.0, min(1.0, gate_weight))

            if gate_weight == 0.0:
                logger.debug(
                    "adapter_zero_weight_skipped",
                    adapter_id=adapter.get("id"),
                )
                continue

            for name, tensor in weights.items():
                if name in combined:
                    if combined[name].shape != tensor.shape:
                        logger.warning(
                            "adapter_shape_mismatch",
                            adapter_id=adapter.get("id"),
                            name=name,
                            expected_shape=combined[name].shape,
                            actual_shape=tensor.shape,
                        )
                        continue
                    # Weighted accumulation: W += g_j * W_j
                    combined[name] = combined[name] + (gate_weight * tensor)
                    total_weight[name] += gate_weight
                else:
                    combined[name] = gate_weight * tensor
                    total_weight[name] = gate_weight

        # Normalize by total weight to maintain scale
        # If weights sum to 1.0, this is a no-op; otherwise it prevents
        # over-amplification when sum > 1 or under-representation when sum < 1
        for name, tensor in combined.items():
            w = total_weight.get(name, 1.0)
            if w > 0.0 and w != 1.0:
                combined[name] = tensor / w

        return combined

    def _adapter_path(self, adapter: dict, *, requested_user_id: Optional[str]) -> str:
        if not adapter:
            return str(self.fs_root / "adapters")
        explicit = adapter.get("cephfs_dir") or adapter.get("fs_dir")
        if explicit:
            if not requested_user_id:
                raise ValueError(
                    "adapter path resolution requires requesting user context"
                )
            owner = adapter.get("owner_user_id") or adapter.get("schema", {}).get(
                "owner_user_id"
            )
            visibility = adapter.get("visibility") or adapter.get("schema", {}).get(
                "visibility"
            )
            if (
                owner
                and owner != requested_user_id
                and visibility not in {"shared", "global"}
            ):
                raise ValueError("adapter owner mismatch")
            base = self.fs_root.resolve()
            candidate = (
                Path(str(explicit)) if isinstance(explicit, (str, Path)) else Path("")
            )
            resolved = (
                candidate if candidate.is_absolute() else base / candidate
            ).resolve()
            # Path must be within fs_root: base must be a parent of resolved, or they must be equal
            if not (base in resolved.parents or resolved == base):
                raise ValueError("adapter path must reside within fs_root")
            return str(resolved)
        adapter_id = adapter.get("id", "unknown")
        candidate = safe_join(self.fs_root, f"adapters/{adapter_id}")
        latest = candidate / "latest"
        if latest.exists():
            return str(latest)
        return str(candidate)
