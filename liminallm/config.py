from __future__ import annotations

import json
import os
import secrets as secrets_module
import string
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from types import UnionType
from typing import Annotated, Any, Literal, Optional, Union, get_args, get_origin

from dotenv import dotenv_values
from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from liminallm.logging import get_logger

logger = get_logger(__name__)


class ModelBackend(str, Enum):
    """Accepted model backend modes as defined in SPEC §5."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE = "azure"
    AZURE_OPENAI = "azure_openai"
    AZURE_OPENAI_ALT = AZURE_OPENAI
    VERTEX = "vertex"
    GEMINI = "gemini"
    GOOGLE = "google"
    # Native generativelanguage API (SSE, thoughts/cached token counts);
    # "gemini" stays the OpenAI-compat shim.
    GEMINI_NATIVE = "gemini_native"
    BEDROCK = "bedrock"
    TOGETHER = "together"
    TOGETHER_AI = "together.ai"
    LORAX = "lorax"
    ADAPTER_SERVER = "adapter_server"
    SAGEMAKER = "sagemaker"
    AWS_SAGEMAKER = "aws_sagemaker"
    # Zhipu AI GLM models (glm-4, glm-4-plus, glm-5, etc.)
    ZHIPU = "zhipu"
    ZHIPU_AI = "zhipu.ai"
    GLM = "glm"
    # Further OpenAI-compatible providers. Each resolves through
    # PROVIDER_ENDPOINTS, so adding one is a table entry, not a new backend.
    XAI = "xai"
    GROK = "grok"
    DEEPSEEK = "deepseek"
    MOONSHOT = "moonshot"
    KIMI = "kimi"
    QWEN = "qwen"
    DASHSCOPE = "dashscope"
    BAICHUAN = "baichuan"
    MINIMAX = "minimax"
    MISTRAL = "mistral"
    COHERE = "cohere"
    # Resellers: same models, smaller serving windows than native.
    FIREWORKS = "fireworks"
    GROQ = "groq"
    CEREBRAS = "cerebras"
    # Local JAX + LoRA serving
    LOCAL_LORA = "local_lora"
    LOCAL_GPU_LORA = "local_gpu_lora"
    # Stub backend for testing - returns canned responses
    STUB = "stub"


class RagMode(str, Enum):
    """RAG retrieval implementations supported by the kernel."""

    PGVECTOR = "pgvector"
    MEMORY = "memory"
    LOCAL_HYBRID = "local_hybrid"


class AdapterMode(str, Enum):
    """Adapter execution modes for dual local/API support.

    SPEC §5 clarification: Adapters can operate in different modes depending
    on deployment. This enum explicitly tracks where adapter weights live
    and how they're applied during inference.

    - LOCAL: Weights stored on filesystem, loaded by LocalJaxLoRABackend
    - REMOTE: Weights hosted by external service (Together, LoRAX, etc.)
    - PROMPT: No weights; adapter behavior injected via system prompt
    - HYBRID: Local weights with prompt fallback for API mode
    """

    LOCAL = "local"
    REMOTE = "remote"
    PROMPT = "prompt"
    HYBRID = "hybrid"


# Mapping of ModelBackend to compatible AdapterModes
BACKEND_ADAPTER_COMPATIBILITY: dict[str, set[str]] = {
    # API backends can use remote adapters or prompt-based
    "openai": {AdapterMode.REMOTE, AdapterMode.PROMPT, AdapterMode.HYBRID},
    "anthropic": {AdapterMode.REMOTE, AdapterMode.PROMPT, AdapterMode.HYBRID},
    "azure": {AdapterMode.REMOTE, AdapterMode.PROMPT, AdapterMode.HYBRID},
    "azure_openai": {AdapterMode.REMOTE, AdapterMode.PROMPT, AdapterMode.HYBRID},
    "vertex": {AdapterMode.REMOTE, AdapterMode.PROMPT, AdapterMode.HYBRID},
    "gemini": {AdapterMode.REMOTE, AdapterMode.PROMPT, AdapterMode.HYBRID},
    "google": {AdapterMode.REMOTE, AdapterMode.PROMPT, AdapterMode.HYBRID},
    "bedrock": {AdapterMode.REMOTE, AdapterMode.PROMPT, AdapterMode.HYBRID},
    # Zhipu AI GLM models
    "zhipu": {AdapterMode.REMOTE, AdapterMode.PROMPT, AdapterMode.HYBRID},
    "zhipu.ai": {AdapterMode.REMOTE, AdapterMode.PROMPT, AdapterMode.HYBRID},
    "glm": {AdapterMode.REMOTE, AdapterMode.PROMPT, AdapterMode.HYBRID},
    # Adapter-aware API backends support remote adapter IDs
    "together": {AdapterMode.REMOTE, AdapterMode.PROMPT, AdapterMode.HYBRID},
    "together.ai": {AdapterMode.REMOTE, AdapterMode.PROMPT, AdapterMode.HYBRID},
    "lorax": {AdapterMode.REMOTE, AdapterMode.PROMPT, AdapterMode.HYBRID},
    "adapter_server": {AdapterMode.REMOTE, AdapterMode.PROMPT, AdapterMode.HYBRID},
    "sagemaker": {AdapterMode.REMOTE, AdapterMode.PROMPT, AdapterMode.HYBRID},
    "aws_sagemaker": {AdapterMode.REMOTE, AdapterMode.PROMPT, AdapterMode.HYBRID},
    # Local backend supports local weights
    "local_lora": {AdapterMode.LOCAL, AdapterMode.PROMPT, AdapterMode.HYBRID},
    "local_gpu_lora": {AdapterMode.LOCAL, AdapterMode.PROMPT, AdapterMode.HYBRID},
    # Stub backend for testing
    "stub": {AdapterMode.PROMPT},
}


def get_compatible_adapter_modes(backend: str) -> set[str]:
    """Return adapter modes compatible with the given backend."""
    return BACKEND_ADAPTER_COMPATIBILITY.get(backend.lower(), {AdapterMode.PROMPT})


class RemoteStyle(str, Enum):
    """How remote adapters are passed to API providers.

    SPEC §5.0.2: Different providers accept adapters in different ways:
    - MODEL_ID: Fine-tuned model as endpoint (OpenAI ft:..., Azure, Vertex)
    - ADAPTER_PARAM: Adapter ID passed as request parameter (Together, LoRAX)
    - NONE: Provider doesn't support remote adapters (prompt-only)
    """

    MODEL_ID = "model_id"  # Adapter = separate model endpoint
    ADAPTER_PARAM = "adapter_param"  # Adapter ID in request body/params
    NONE = "none"  # No remote adapter support


@dataclass
class ProviderCapabilities:
    """Capabilities of an LLM API provider for adapter handling.

    Defines how the provider handles fine-tuned models and LoRA adapters,
    enabling proper routing and request formatting per provider.
    """

    remote_style: RemoteStyle  # How remote adapters are specified
    multi_adapter: bool  # Can compose multiple adapters per request
    gate_weights: bool  # Supports per-adapter gate weights
    max_adapters: int  # Maximum concurrent adapters (1 for model_id style)
    adapter_param_name: str = "adapter_id"  # Parameter name for adapter_param style
    supports_streaming: bool = True
    model_id_prefix: str = ""  # e.g., "ft:" for OpenAI fine-tunes


# Provider capability registry - defines how each backend handles adapters
PROVIDER_CAPABILITIES: dict[str, ProviderCapabilities] = {
    # Fine-tuned model as endpoint providers (one adapter = one model)
    "openai": ProviderCapabilities(
        remote_style=RemoteStyle.MODEL_ID,
        multi_adapter=False,
        gate_weights=False,
        max_adapters=1,
        model_id_prefix="ft:",
    ),
    "azure": ProviderCapabilities(
        remote_style=RemoteStyle.MODEL_ID,
        multi_adapter=False,
        gate_weights=False,
        max_adapters=1,
    ),
    "azure_openai": ProviderCapabilities(
        remote_style=RemoteStyle.MODEL_ID,
        multi_adapter=False,
        gate_weights=False,
        max_adapters=1,
    ),
    "vertex": ProviderCapabilities(
        remote_style=RemoteStyle.MODEL_ID,
        multi_adapter=False,
        gate_weights=False,
        max_adapters=1,
    ),
    "gemini": ProviderCapabilities(
        remote_style=RemoteStyle.MODEL_ID,
        multi_adapter=False,
        gate_weights=False,
        max_adapters=1,
    ),
    "google": ProviderCapabilities(
        remote_style=RemoteStyle.MODEL_ID,
        multi_adapter=False,
        gate_weights=False,
        max_adapters=1,
    ),
    "bedrock": ProviderCapabilities(
        remote_style=RemoteStyle.MODEL_ID,
        multi_adapter=False,
        gate_weights=False,
        max_adapters=1,
    ),
    # Anthropic Claude models (claude-3, claude-sonnet-4, claude-opus-4-5, etc.)
    "anthropic": ProviderCapabilities(
        remote_style=RemoteStyle.MODEL_ID,
        multi_adapter=False,
        gate_weights=False,
        max_adapters=1,
    ),
    # Zhipu AI GLM models (glm-4, glm-4-plus, glm-5, etc.)
    "zhipu": ProviderCapabilities(
        remote_style=RemoteStyle.MODEL_ID,
        multi_adapter=False,
        gate_weights=False,
        max_adapters=1,
    ),
    "zhipu.ai": ProviderCapabilities(
        remote_style=RemoteStyle.MODEL_ID,
        multi_adapter=False,
        gate_weights=False,
        max_adapters=1,
    ),
    "glm": ProviderCapabilities(
        remote_style=RemoteStyle.MODEL_ID,
        multi_adapter=False,
        gate_weights=False,
        max_adapters=1,
    ),
    # Adapter-parameter style providers (support multi-adapter)
    "together": ProviderCapabilities(
        remote_style=RemoteStyle.ADAPTER_PARAM,
        multi_adapter=True,
        gate_weights=True,
        max_adapters=3,
        adapter_param_name="adapter_id",
    ),
    "together.ai": ProviderCapabilities(
        remote_style=RemoteStyle.ADAPTER_PARAM,
        multi_adapter=True,
        gate_weights=True,
        max_adapters=3,
        adapter_param_name="adapter_id",
    ),
    "lorax": ProviderCapabilities(
        remote_style=RemoteStyle.ADAPTER_PARAM,
        multi_adapter=True,
        gate_weights=True,
        max_adapters=5,
        adapter_param_name="adapter_id",
    ),
    "adapter_server": ProviderCapabilities(
        remote_style=RemoteStyle.ADAPTER_PARAM,
        multi_adapter=True,
        gate_weights=True,
        max_adapters=3,
        adapter_param_name="adapter_id",
    ),
    "sagemaker": ProviderCapabilities(
        remote_style=RemoteStyle.ADAPTER_PARAM,
        multi_adapter=False,  # Depends on setup
        gate_weights=False,
        max_adapters=1,
        adapter_param_name="adapter_id",
    ),
    "aws_sagemaker": ProviderCapabilities(
        remote_style=RemoteStyle.ADAPTER_PARAM,
        multi_adapter=False,
        gate_weights=False,
        max_adapters=1,
        adapter_param_name="adapter_id",
    ),
    # Local backends
    "local_lora": ProviderCapabilities(
        remote_style=RemoteStyle.NONE,
        multi_adapter=True,
        gate_weights=True,
        max_adapters=3,
    ),
    "local_gpu_lora": ProviderCapabilities(
        remote_style=RemoteStyle.NONE,
        multi_adapter=True,
        gate_weights=True,
        max_adapters=3,
    ),
    # Stub backend for testing (no adapter support)
    "stub": ProviderCapabilities(
        remote_style=RemoteStyle.NONE,
        multi_adapter=False,
        gate_weights=False,
        max_adapters=0,
    ),
}

# Default capabilities for unknown providers (conservative)
DEFAULT_PROVIDER_CAPABILITIES = ProviderCapabilities(
    remote_style=RemoteStyle.MODEL_ID,
    multi_adapter=False,
    gate_weights=False,
    max_adapters=1,
)


def get_provider_capabilities(provider: str) -> ProviderCapabilities:
    """Get capabilities for a provider, with sensible defaults for unknown providers."""
    return PROVIDER_CAPABILITIES.get(provider.lower(), DEFAULT_PROVIDER_CAPABILITIES)


# OpenAI-compatible API endpoints for first-party providers. Each provider is
# reachable through the OpenAI chat-completions client by pointing base_url at
# the provider and supplying its API key, so a single ApiAdapterBackend serves
# OpenAI, Anthropic, Zhipu/GLM, Together, Gemini, etc. `provider` is the key
# into PROVIDER_CAPABILITIES; `api_key_env` is the env var read for credentials.
PROVIDER_ENDPOINTS: dict[str, dict[str, Optional[str]]] = {
    "openai": {"provider": "openai", "api_key_env": "OPENAI_API_KEY", "base_url": None},
    "anthropic": {"provider": "anthropic", "api_key_env": "ANTHROPIC_API_KEY", "base_url": "https://api.anthropic.com/v1"},
    "azure": {"provider": "azure", "api_key_env": "AZURE_OPENAI_API_KEY", "base_url": None},
    "azure_openai": {"provider": "azure_openai", "api_key_env": "AZURE_OPENAI_API_KEY", "base_url": None},
    "gemini": {"provider": "gemini", "api_key_env": "GEMINI_API_KEY", "base_url": "https://generativelanguage.googleapis.com/v1beta/openai"},
    "google": {"provider": "gemini", "api_key_env": "GEMINI_API_KEY", "base_url": "https://generativelanguage.googleapis.com/v1beta/openai"},
    "zhipu": {"provider": "zhipu", "api_key_env": "ZHIPU_API_KEY", "base_url": "https://open.bigmodel.cn/api/paas/v4"},
    "zhipu.ai": {"provider": "zhipu", "api_key_env": "ZHIPU_API_KEY", "base_url": "https://open.bigmodel.cn/api/paas/v4"},
    "glm": {"provider": "zhipu", "api_key_env": "ZHIPU_API_KEY", "base_url": "https://open.bigmodel.cn/api/paas/v4"},
    "together": {"provider": "together", "api_key_env": "TOGETHER_API_KEY", "base_url": "https://api.together.xyz/v1"},
    "together.ai": {"provider": "together", "api_key_env": "TOGETHER_API_KEY", "base_url": "https://api.together.xyz/v1"},
    "xai": {"provider": "xai", "api_key_env": "XAI_API_KEY", "base_url": "https://api.x.ai/v1"},
    "grok": {"provider": "xai", "api_key_env": "XAI_API_KEY", "base_url": "https://api.x.ai/v1"},
    "deepseek": {"provider": "deepseek", "api_key_env": "DEEPSEEK_API_KEY", "base_url": "https://api.deepseek.com/v1"},
    "moonshot": {"provider": "moonshot", "api_key_env": "MOONSHOT_API_KEY", "base_url": "https://api.moonshot.ai/v1"},
    "kimi": {"provider": "moonshot", "api_key_env": "MOONSHOT_API_KEY", "base_url": "https://api.moonshot.ai/v1"},
    # Alibaba Model Studio. The international endpoint; mainland deployments
    # set adapter_openai_base_url to dashscope.aliyuncs.com instead.
    "qwen": {"provider": "qwen", "api_key_env": "DASHSCOPE_API_KEY", "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"},
    "dashscope": {"provider": "qwen", "api_key_env": "DASHSCOPE_API_KEY", "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"},
    "baichuan": {"provider": "baichuan", "api_key_env": "BAICHUAN_API_KEY", "base_url": "https://api.baichuan-ai.com/v1"},
    "minimax": {"provider": "minimax", "api_key_env": "MINIMAX_API_KEY", "base_url": "https://api.minimax.io/v1"},
    "mistral": {"provider": "mistral", "api_key_env": "MISTRAL_API_KEY", "base_url": "https://api.mistral.ai/v1"},
    "cohere": {"provider": "cohere", "api_key_env": "COHERE_API_KEY", "base_url": "https://api.cohere.ai/compatibility/v1"},
    # Resellers. The provider name is load-bearing beyond credentials: it
    # selects the host's own serving limits in HOSTED_CONTEXT_WINDOWS, which
    # are smaller than the models' native ceilings.
    "fireworks": {"provider": "fireworks", "api_key_env": "FIREWORKS_API_KEY", "base_url": "https://api.fireworks.ai/inference/v1"},
    "groq": {"provider": "groq", "api_key_env": "GROQ_API_KEY", "base_url": "https://api.groq.com/openai/v1"},
    "cerebras": {"provider": "cerebras", "api_key_env": "CEREBRAS_API_KEY", "base_url": "https://api.cerebras.ai/v1"},
}


def resolve_provider_endpoint(mode: str) -> Optional[dict[str, Optional[str]]]:
    """Return endpoint wiring for an OpenAI-compatible API provider backend mode.

    Returns None for modes that are not API providers (stub, local_lora,
    local_gpu_lora, adapter_server), which are constructed by other paths.
    """
    return PROVIDER_ENDPOINTS.get((mode or "").lower())


def env_field(default: Any, env: str, **kwargs):
    """A setting that can only come from the environment.

    Reserved for two cases, and no others:

    * secrets — API keys, client secrets, the JWT signing key;
    * bootstrap — anything needed before the database is reachable, or that
      describes the machine rather than the install (where the data lives, how
      wide the vector column is, whether this is a test process).

    Security controls also live here on purpose: CORS, HSTS, the SSRF
    allowlist, the SMTP downgrade flag. Those decide who may reach the instance
    and who it will talk to; an admin session should not be able to widen them
    from a web form.

    Everything else is a managed_field.
    """
    extra = kwargs.pop("json_schema_extra", {}) or {}
    extra = {**extra, "env": env, "admin": False}
    return Field(default, json_schema_extra=extra, **kwargs)


def secret_field(default: Any = "", **kwargs):
    """A managed setting whose value is never read back out.

    Stored in the database like any other managed setting — an operator can
    rotate an SMTP password without a redeploy — but it is redacted from every
    read path: GET /admin/settings returns it as an empty string, and the
    console renders a write-only control that submits only when the operator
    types something.

    That redaction is the whole reason this is a separate kind. The settings
    endpoint returns the merged dict verbatim, so a secret stored as an
    ordinary managed_field would be echoed back to every admin, into logs, and
    into anything that captures a response body.

    Not for bootstrap secrets. JWT_SECRET and DATABASE_URL are needed before
    the database can be read at all, so they stay env_field.
    """
    extra = kwargs.pop("json_schema_extra", {}) or {}
    extra = {**extra, "admin": True, "secret": True}
    return Field(default, json_schema_extra=extra, **kwargs)


def managed_field(default: Any, **kwargs):
    """A setting that lives in the database, editable from the admin UI.

    No environment variable. Configuration that *can* live in the database
    belongs there: it is auditable, it changes without a restart, and every
    replica sees the same value. An env var for it means a setting that can
    differ per container and cannot be inspected or changed from inside the
    running system.

    The default written here is the shipped default, and it is written exactly
    once — SYSTEM_SETTINGS_DEFAULTS is derived from these fields. For
    declarative deploys, INSTANCE_SETTINGS_JSON seeds them on first boot.
    """
    extra = kwargs.pop("json_schema_extra", {}) or {}
    extra = {**extra, "admin": True}
    return Field(default, json_schema_extra=extra, **kwargs)


class Settings(BaseModel):
    """Runtime settings aligned with the SPEC kernel contracts."""

    database_url: str = env_field(
        "postgresql://localhost:5432/liminallm", "DATABASE_URL"
    )
    redis_url: str = managed_field(
        "redis://localhost:6379/0",
        description=(
            "Redis DSN. Optional: without it, rate limits, idempotency "
            "durability and caches fall back to in-process state."
        ),
    )
    shared_fs_root: str = managed_field("/srv/liminallm")
    tmp_cleanup_interval_seconds: int = managed_field(
        86400,
        description="How often to sweep per-user tmp scratch directories (seconds)",
    )
    tmp_max_age_hours: int = managed_field(
        24,
        description="Delete tmp scratch files older than this many hours",
    )
    model_path: str = managed_field(
        "gpt-4o-mini", description="Model path",
    )
    model_backend: ModelBackend | None = managed_field(
        ModelBackend.OPENAI, description="Model backend",
    )
    adapter_openai_api_key: str = secret_field(
        description="API key for the model provider. Write-only.",
    )
    adapter_openai_base_url: str | None = managed_field(
        None,
        description="Base URL override for an OpenAI-compatible endpoint. Blank uses the provider default.",
    )
    adapter_server_model: str | None = managed_field(
        None,
        description="Model name to send when pointing at an adapter server",
    )
    gemini_api_key: str = secret_field(
        description=(
            "API key for the Gemini backends (model_backend=gemini_native "
            "or gemini). Falls back to the provider key above, then the "
            "GEMINI_API_KEY env var. Write-only."
        ),
    )
    # Voice service settings
    voice_api_key: str = secret_field(
        description="API key for transcription and synthesis. Write-only.",
    )
    # Literal, not str: the allowed values are part of the setting, and
    # declaring them here means the admin API validates against the same list
    # the admin UI renders as a dropdown.
    voice_transcription_model: Literal["whisper-1"] = managed_field(
        "whisper-1",
        description="Transcription model",
    )
    voice_synthesis_model: Literal["tts-1", "tts-1-hd"] = managed_field(
        "tts-1",
        description="Synthesis model",
    )
    voice_default_voice: Literal[
        "alloy", "echo", "fable", "onyx", "nova", "shimmer"
    ] = managed_field("alloy", description="Default voice")
    # OAuth settings
    oauth_google_client_id: str = managed_field(
        "", description="OAuth client ID for google (not a secret)",
    )
    oauth_google_client_secret: str = secret_field(
        description="OAuth client secret for google. Write-only.",
    )
    oauth_github_client_id: str = managed_field(
        "", description="OAuth client ID for github (not a secret)",
    )
    oauth_github_client_secret: str = secret_field(
        description="OAuth client secret for github. Write-only.",
    )
    oauth_microsoft_client_id: str = managed_field(
        "", description="OAuth client ID for microsoft (not a secret)",
    )
    oauth_microsoft_client_secret: str = secret_field(
        description="OAuth client secret for microsoft. Write-only.",
    )
    oauth_redirect_uri: str | None = managed_field(
        None, description="OAuth redirect URI",
    )
    # Email service settings (env vars are fallbacks - prefer admin UI)
    smtp_host: str | None = managed_field(
        None, description="SMTP server host",
    )
    smtp_port: int = managed_field(
        587, description="SMTP server port",
    )
    smtp_user: str | None = managed_field(
        None, description="SMTP username",
    )
    smtp_password: str = secret_field(
        description="SMTP password. Write-only: saved, never shown again.",
    )
    smtp_security: Literal["starttls", "ssl", "none"] = managed_field(
        "starttls",
        description=(
            "How the SMTP connection is encrypted. starttls: connect in the "
            "clear and upgrade (usually port 587). ssl: encrypted from the "
            "first byte (usually port 465). none: no encryption — for a relay "
            "on this machine only, and refused if a username is set, because "
            "the password would cross the wire in the clear."
        ),
    )
    email_from_address: str | None = managed_field(
        None, description="Email from address",
    )
    email_from_name: str = managed_field(
        "LiminalLM", description="Email from name",
    )
    app_base_url: str = managed_field(
        "http://localhost:8000",
        description="Application base URL",
    )
    default_adapter_mode: AdapterMode = managed_field(
        AdapterMode.HYBRID,
        description="Default mode for new adapters: local, remote, prompt, or hybrid",
    )
    allow_signup: bool = managed_field(
        True,
        description="Allow new user signups",
    )
    build_sha: str = env_field("dev", "BUILD_SHA")
    cors_allow_origins: list[str] = managed_field(
        [
            "http://localhost",
            "http://localhost:3000",
            "http://localhost:5173",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:5173",
        ],
        description="Browser origins allowed to call this API",
    )
    cors_allow_credentials: bool = managed_field(
        False, description="Allow cookies on cross-origin requests",
    )
    enable_hsts: bool = managed_field(
        False, description="Send Strict-Transport-Security. Turn on once TLS is in front.",
    )
    allow_redis_fallback_dev: bool = managed_field(False)
    test_mode: bool = env_field(
        False,
        "TEST_MODE",
        description="Toggle deterministic testing behaviors; required for CI pathways described in SPEC §14.",
    )
    tool_network_allowlist: list[str] = managed_field(
        ["api.openai.com"],
        description="Allowlisted hostnames/CIDRs for tool egress (SPEC §18)",
    )
    tool_network_proxy_url: str | None = managed_field(
        None,
        description="Proxy URL tools must use for outbound HTTP(S) fetches",
    )
    tool_fetch_connect_timeout: float = managed_field(
        10.0,
        description="Connect timeout (seconds) for tool HTTP fetches",
    )
    tool_fetch_timeout: float = managed_field(
        30.0,
        description="Total timeout (seconds) for tool HTTP fetches",
    )
    interpreter_scratch_dir: str | None = managed_field(
        None,
        description=(
            "Node-local directory for code-interpreter session dirs. Defaults "
            "to the system temp dir. Must NOT be on shared storage: these hold "
            "throwaway copies of attachments for one tool call."
        ),
    )
    history_budget_fraction: float = managed_field(
        0.5,
        ge=0.1,
        le=0.9,
        description=(
            "Share of the prompt budget kept as verbatim recent turns "
            "(0.1-0.9). The rest is left for system blocks, RAG, attachments, "
            "and the new message."
        ),
    )
    history_recall_fraction: float = managed_field(
        0.25,
        ge=0.0,
        le=0.9,
        description=(
            "Share of the history budget spent recalling older turns picked "
            "by relevance to the current message — the window is assembled, "
            "not just a recency prefix. 0 disables recall."
        ),
    )
    embedding_vector_dim: int = env_field(
        1536,
        "EMBEDDING_VECTOR_DIM",
        description=(
            "Dimension of the pgvector embedding column. MUST match the "
            "configured encoder (1536 for text-embedding-3-small, 64 for the "
            "hash fallback) — pgvector cannot index a dimensionless column. "
            "Read by scripts/migrate.sh; changing it requires re-running "
            "migrations and re-embedding."
        ),
    )
    model_reasoning_effort: str = managed_field(
        "",
        description=(
            "Thinking budget for reasoning models: low | medium | high, or "
            "none to disable. Empty means omit the parameter entirely."
        ),
    )
    model_context_window: int = managed_field(
        0,
        ge=0,
        description=(
            "Input window of the serving model, in tokens. 0 = discover: ask "
            "the provider, else a known-family table, else 8192. Set this "
            "when discovery guesses wrong for your deployment."
        ),
    )
    extract_reader_plugins: str = env_field(
        "",
        "EXTRACT_READER_PLUGINS",
        description=(
            "Comma-separated Python modules to import for extra extract "
            "readers. Env-only on purpose: this imports code, so an admin who "
            "could set it from the UI would have remote code execution."
        ),
    )
    extract_readers: str = managed_field(
        "ocr,vision",
        description=(
            "Ordered roster of image readers for uploads/scanned pdfs. "
            "Built-ins: ocr (tesseract), vision (the model backend). New "
            "readers register via extract.register_reader."
        ),
    )
    notes_enabled: bool = managed_field(
        True,
        description=(
            "Notes vault + witness. Admin-overridable via system settings; "
            "when off, notes routes and the note_search tool disappear."
        ),
    )
    cluster_bus_backend: str = managed_field(
        "auto",
        description=(
            "Transport for cross-replica coordination (cancelling a stream "
            "owned by another worker): auto | redis | postgres | local. 'auto' "
            "prefers Redis and falls back to Postgres LISTEN/NOTIFY, so Redis "
            "stays optional; 'local' disables peer coordination entirely."
        ),
    )
    # Web tools (SPEC §18). Browsing is enabled by default but constrained:
    # SSRF protection is always on, and search stays inert until a provider and
    # its key are configured.
    web_tools_enabled: bool = managed_field(
        True,
        description="Allow the model to call web_search / web_fetch",
    )
    web_search_provider: str = managed_field(
        "none",
        description="Search backend: none | brave | tavily | google_cse | duckduckgo",
    )
    web_search_api_key: str = secret_field(
        description="API key for the search provider. Write-only.",
    )
    web_search_engine_id: str | None = managed_field(
        None,
        description="Search engine ID (google_cse only)",
    )
    web_fetch_timeout: float = managed_field(
        15.0,
        gt=0,
        le=120.0,
        description="Total timeout (seconds) for a web page fetch",
    )
    web_fetch_max_bytes: int = managed_field(
        2 * 1024 * 1024,
        gt=0,
        le=64 * 1024 * 1024,
        description="Maximum bytes read from a fetched page",
    )
    web_fetch_allow_private: bool = managed_field(
        False,
        description=(
            "TEST ONLY: permit fetching private/loopback addresses. Disables "
            "SSRF protection — never enable in production."
        ),
    )
    enable_mfa: bool = managed_field(
        True,
        description="Enable multi-factor authentication",
    )
    jwt_secret: str = secret_field(
        "",
        description=(
            "Token signing key. Generated on first boot if empty. Rotating it "
            "signs out every session."
        ),
    )
    jwt_issuer: str = managed_field(
        "liminallm",
        description="JWT issuer",
    )
    jwt_audience: str = managed_field(
        "liminal-clients",
        description="JWT audience",
    )
    access_token_ttl_minutes: int = managed_field(
        30,
        description="Access token TTL in minutes",
    )
    refresh_token_ttl_minutes: int = managed_field(
        24 * 60,
        description="Refresh token TTL in minutes",
    )
    default_tenant_id: str = managed_field(
        "public",
        description="Tenant for an install that serves one site. Also the tenant for any host not listed in tenant_domains.",
    )
    tenant_domains: dict[str, str] = managed_field(
        {},
        description=(
            "Hostname to tenant id, e.g. {\"acme.example.com\": \"acme\"}. "
            "Empty means one tenant for the whole install. Once any mapping "
            "exists, a request arriving on an unlisted host is refused rather "
            "than served the default tenant."
        ),
    )
    trust_forwarded_host: bool = managed_field(
        False,
        description=(
            "Read the visited hostname from X-Forwarded-Host instead of Host. "
            "Turn this on only when a reverse proxy you control sets it — "
            "otherwise a client can name its own tenant."
        ),
    )
    rag_mode: RagMode = managed_field(
        RagMode.PGVECTOR,
        description="RAG mode: pgvector or memory",
    )
    embedding_model_id: Literal[
        "text-embedding",
        "text-embedding-3-small",
        "text-embedding-3-large",
        "text-embedding-ada-002",
    ] = managed_field(
        "text-embedding",
        description="Embedding model. 'text-embedding' is the hash fallback.",
    )

    # NOTE: The following operational settings have been moved to database-managed
    # system settings (accessible via admin UI at /admin.html and API at /v1/admin/settings).
    # Env var values serve as fallbacks when database settings are not present.
    #
    # Session & Concurrency:
    # - session_rotation_hours, session_rotation_grace_seconds
    # - max_concurrent_workflows, max_concurrent_inference
    #
    # Rate Limits:
    # - chat_rate_limit_per_minute, chat_rate_limit_window_seconds
    # - login_rate_limit_per_minute, signup_rate_limit_per_minute
    # - reset_rate_limit_per_minute, mfa_rate_limit_per_minute
    # - admin_rate_limit_per_minute, admin_rate_limit_window_seconds
    # - files_upload_rate_limit_per_minute, configops_rate_limit_per_hour
    # - read_rate_limit_per_minute, write_rate_limit_per_minute
    # - rate_limit_multiplier_free/paid/enterprise
    #
    # Pagination & Files:
    # - default_page_size, max_page_size, default_conversations_limit
    # - max_upload_bytes, rag_chunk_size
    #
    # Token TTL:
    # - access_token_ttl_minutes, refresh_token_ttl_minutes
    #
    # Feature Flags:
    # - enable_mfa, allow_signup
    #
    # Training Worker:
    # - training_worker_enabled, training_worker_poll_interval
    #
    # SMTP / Email (all settings including secrets):
    # - smtp_host, smtp_port, smtp_user, smtp_password, smtp_security
    # - email_from_address, email_from_name
    #
    # URL Settings:
    # - oauth_redirect_uri, app_base_url
    #
    # Voice Settings:
    # - voice_transcription_model, voice_synthesis_model, voice_default_voice
    #
    # Model Settings:
    # - model_path, model_backend, default_adapter_mode, rag_mode, embedding_model_id
    #
    # Tenant & JWT Settings:
    # - default_tenant_id, jwt_issuer, jwt_audience

    # Training worker settings (env vars are fallbacks - prefer admin UI)
    training_worker_enabled: bool = managed_field(
        True,
        description="Enable background training job worker",
    )
    training_worker_poll_interval: int = managed_field(
        60,
        description="Training worker poll interval in seconds",
    )
    settings_watch_interval_seconds: int = managed_field(
        10,
        description=(
            "How often each worker checks for admin settings changes and "
            "reloads its model services (cross-process consistency across "
            "multiple Uvicorn workers)"
        ),
    )
    max_active_training_jobs: int = managed_field(
        10,
        description="Global cap on simultaneously active training jobs",
    )
    training_distillation_enabled: bool = managed_field(
        False,
        description=(
            "Distill preference-event targets through the configured LLM "
            "(teacher) before adapter training (SPEC §7.5)"
        ),
    )

    model_config = ConfigDict(extra="ignore")


    # ------------------------------------------------------------------
    # Operational limits. Managed like everything else — an operator tunes
    # these from the admin UI while the instance is running, which is exactly
    # when you find out a rate limit is wrong.
    # ------------------------------------------------------------------
    # Session & concurrency
    # Device-specific windows (SPEC §18: 7d web, 1d mobile). Read by auth.py,
    # which used to carry its own copy of these defaults — a fourth declaration
    # site, for four keys the admin API would then reject as unknown.
    # Session windows. A one-minute floor: a zero would log everyone out on
    # arrival, and there is no reading of "0 minutes" that anyone wants.
    session_ttl_minutes_web: int = managed_field(
        60 * 24 * 7, ge=1,
        description="How long a browser session stays valid (SPEC §18: 7 days)",
    )
    session_ttl_minutes_mobile: int = managed_field(
        60 * 24, ge=1,
        description="How long a mobile session stays valid (SPEC §18: 1 day)",
    )
    refresh_token_ttl_minutes_web: int = managed_field(
        60 * 24 * 7, ge=1,
        description="Refresh window for browser sessions; match the session TTL",
    )
    refresh_token_ttl_minutes_mobile: int = managed_field(
        60 * 24, ge=1,
        description="Refresh window for mobile sessions; match the session TTL",
    )
    session_rotation_hours: int = managed_field(
        24, ge=1,
        description="Re-issue a session's tokens after this long of continuous use",
    )
    session_rotation_grace_seconds: int = managed_field(
        300, ge=0,
        description="How long the previous token still works after a rotation, so an in-flight request does not 401",
    )
    # Concurrency: at least one, or the instance accepts work it will never run.
    max_concurrent_workflows: int = managed_field(
        3, ge=1,
        description="Workflows running at once per instance. Raising it costs memory, not just CPU.",
    )
    max_concurrent_inference: int = managed_field(
        2, ge=1,
        description="Model calls in flight at once. Keep at or below what the provider will accept.",
    )
    # Rate limit multipliers. Positive: a zero multiplier would silently make
    # every limit "0 requests" for that tier rather than disabling anything.
    rate_limit_multiplier_free: float = managed_field(
        1.0, gt=0,
        description="Rate limits are multiplied by this for free-tier users",
    )
    rate_limit_multiplier_paid: float = managed_field(
        2.0, gt=0,
        description="Rate limits are multiplied by this for paid users",
    )
    rate_limit_multiplier_enterprise: float = managed_field(
        5.0, gt=0,
        description="Rate limits are multiplied by this for enterprise users",
    )
    # Rate limits. 0 means unlimited, which is why these floor at 0 rather
    # than 1 — an operator turning one off is a legitimate choice.
    chat_rate_limit_per_minute: int = managed_field(
        60, ge=0,
        description="Chat requests allowed per window. 0 disables the limit.",
    )
    chat_rate_limit_window_seconds: int = managed_field(
        60, ge=1,
        description="Length of the chat rate-limit window",
    )
    login_rate_limit_per_minute: int = managed_field(
        10, ge=0,
        description="Login attempts per minute per client. Low values slow credential stuffing.",
    )
    refresh_rate_limit_per_minute: int = managed_field(
        20, ge=0,
        description="Token refreshes allowed per window",
    )
    refresh_rate_limit_window_seconds: int = managed_field(
        60, ge=1,
        description="Length of the token-refresh window",
    )
    signup_rate_limit_per_minute: int = managed_field(
        5, ge=0,
        description="New accounts per minute per client",
    )
    reset_rate_limit_per_minute: int = managed_field(
        5, ge=0,
        description="Password-reset requests per minute per client",
    )
    mfa_rate_limit_per_minute: int = managed_field(
        5, ge=0,
        description="MFA attempts per minute. Low values slow code guessing.",
    )
    admin_rate_limit_per_minute: int = managed_field(
        30, ge=0,
        description="Admin API calls per window",
    )
    admin_rate_limit_window_seconds: int = managed_field(
        60, ge=1,
        description="Length of the admin rate-limit window",
    )
    files_upload_rate_limit_per_minute: int = managed_field(
        10, ge=0,
        description="Uploads per minute per user",
    )
    websocket_connect_rate_limit_per_minute: int = managed_field(
        30, ge=0,
        description="Websocket connection attempts per minute per user",
    )
    configops_rate_limit_per_hour: int = managed_field(
        30, ge=0,
        description="ConfigOps patch proposals per hour",
    )
    read_rate_limit_per_minute: int = managed_field(
        120, ge=0,
        description="Read requests per minute per user. 0 disables the limit.",
    )
    write_rate_limit_per_minute: int = managed_field(
        60, ge=0,
        description="Write requests per minute per user. 0 disables the limit.",
    )
    max_websocket_connections_per_user: int = managed_field(
        5, ge=1,
        description="Simultaneous websockets one user may hold",
    )
    # Pagination. Capped so a page size cannot be set to something that reads
    # the whole table into memory.
    default_page_size: int = managed_field(
        100, ge=1, le=1000,
        description="Items per page when a request does not ask for a size",
    )
    max_page_size: int = managed_field(
        500, ge=1, le=1000,
        description="Largest page a caller may request",
    )
    default_conversations_limit: int = managed_field(
        50, ge=1, le=1000,
        description="Conversations returned when no limit is given",
    )
    # Files
    max_upload_bytes: int = managed_field(
        10485760, ge=1024,
        description="Largest single upload accepted, in bytes",
    )
    rag_chunk_size: int = managed_field(
        400, ge=64, le=4000,
        description="Tokens per knowledge chunk. Changing this rebuilds the model services and only affects newly ingested content.",
    )

    @field_validator("tenant_domains", mode="before")
    @classmethod
    def _parse_tenant_domains(cls, value: Any) -> dict[str, str]:
        """Normalize hosts on the way in, so lookup is a plain dict hit.

        Hosts are compared lowercase and without a port, because that is what
        varies between how an operator types it and how a browser sends it.
        """
        if value in (None, ""):
            return {}
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except (TypeError, ValueError) as exc:
                raise ValueError("tenant_domains must be a JSON object") from exc
        if not isinstance(value, dict):
            raise ValueError("tenant_domains must be a JSON object")
        normalized: dict[str, str] = {}
        for host, tenant in value.items():
            host = str(host).strip().lower().rstrip(".").split(":")[0]
            tenant = str(tenant).strip()
            if not host or not tenant:
                raise ValueError("tenant_domains entries need a host and a tenant")
            normalized[host] = tenant
        return normalized

    @field_validator("cors_allow_origins", mode="before")
    @classmethod
    def _parse_cors_origins(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [v.strip() for v in value.split(",") if v.strip()]
        if isinstance(value, list):
            return value
        return []

    @field_validator(
        "smtp_port", "training_worker_poll_interval", "tmp_cleanup_interval_seconds", "tmp_max_age_hours", "max_active_training_jobs", "settings_watch_interval_seconds"
    )
    @classmethod
    def _validate_positive_int(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("must be positive")
        return value

    @field_validator("smtp_port")
    @classmethod
    def _validate_smtp_port(cls, value: int) -> int:
        if not 1 <= value <= 65535:
            raise ValueError("smtp_port must be between 1 and 65535")
        return value

    @model_validator(mode="after")
    def _validate_required_pairs(self):
        if self.oauth_google_client_id and not self.oauth_google_client_secret:
            raise ValueError("oauth_google_client_secret required when client_id is set")
        if self.oauth_github_client_id and not self.oauth_github_client_secret:
            raise ValueError("oauth_github_client_secret required when client_id is set")
        if self.oauth_microsoft_client_id and not self.oauth_microsoft_client_secret:
            raise ValueError("oauth_microsoft_client_secret required when client_id is set")
        if (self.smtp_host or self.smtp_user) and not self.smtp_password:
            raise ValueError("smtp_password required when smtp_host or smtp_user is set")
        return self

    @classmethod
    def from_env(cls) -> "Settings":
        """Read the env-only settings. Managed settings keep their defaults.

        Only fields declared with env_field are read from the environment. A
        managed_field has no env var by design, and must not acquire one by
        accident through a name-matching fallback — that would resurrect the
        per-container configuration this split exists to remove. Managed values
        come from the database; see Runtime.effective_settings.
        """
        env_file_values = dotenv_values(".env")
        merged: dict[str, str] = {}
        for name, field in cls.model_fields.items():
            extra = field.json_schema_extra or {}
            env_name = extra.get("env") if isinstance(extra, dict) else None
            if not env_name:
                continue
            if env_name in os.environ:
                merged[name] = os.environ[env_name]
            elif env_name in env_file_values:
                merged[name] = env_file_values[env_name]
        return cls(**merged)

    @field_validator("model_backend")
    @classmethod
    def _validate_backend(cls, value: ModelBackend | None) -> ModelBackend | None:
        if value is None:
            return None
        return ModelBackend(value)

    @field_validator("rag_mode")
    @classmethod
    def _validate_rag_mode(cls, value: RagMode) -> RagMode:
        return RagMode(value)

    @field_validator("default_adapter_mode")
    @classmethod
    def _validate_adapter_mode(cls, value: AdapterMode) -> AdapterMode:
        return AdapterMode(value)

    @field_validator("tool_network_allowlist", mode="before")
    @classmethod
    def _parse_tool_allowlist(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return [str(v) for v in parsed if str(v).strip()]
            except Exception:
                pass
            return [v.strip() for v in value.split(",") if v.strip()]
        return list(value)

    @field_validator("jwt_secret")
    @classmethod
    def _check_jwt_secret(cls, value: str) -> str:
        """Reject a weak key. An empty one means "generate me" (see runtime)."""
        if not value:
            return value
        secret = value.strip()
        if len(secret) < 32:
            raise ValueError("jwt_secret must be at least 32 characters long")
        classes = [
            any(ch.islower() for ch in secret),
            any(ch.isupper() for ch in secret),
            any(ch.isdigit() for ch in secret),
            any(ch in string.punctuation for ch in secret),
        ]
        if sum(classes) < 3 or len(set(secret)) < 10:
            raise ValueError(
                "jwt_secret must mix character classes and contain enough "
                "unique characters"
            )
        return secret


def generate_jwt_secret() -> str:
    """A signing key that satisfies _check_jwt_secret."""
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*-_=+"
    return "".join(secrets_module.choice(alphabet) for _ in range(64))



def _admin_defaults_from_settings() -> dict:
    """Shipped defaults for every managed_field, as the admin API sends them.

    Enums become their values and an unset value becomes "" rather than None,
    because the admin form posts an empty box, not a null.
    """
    derived: dict = {}
    for name, field in Settings.model_fields.items():
        extra = field.json_schema_extra or {}
        if not (isinstance(extra, dict) and extra.get("admin")):
            continue
        value = field.default
        if isinstance(value, Enum):
            value = value.value
        elif value is None:
            value = ""
        derived[name] = value
    return derived


# Every admin-managed setting and its shipped default, derived from the one
# place each is declared. The store merges this under whatever an admin
# actually saved; nothing seeds it into the database, so a shipped default
# never masquerades as an explicit override.
SYSTEM_SETTINGS_DEFAULTS: dict = _admin_defaults_from_settings()

def secret_setting_names() -> set[str]:
    """Managed settings that must never be returned by a read."""
    return {
        name
        for name, field in Settings.model_fields.items()
        if isinstance(field.json_schema_extra, dict)
        and field.json_schema_extra.get("secret")
    }


def redact_secrets(settings: dict) -> dict:
    """Blank every secret in a settings dict before it leaves the process."""
    hidden = secret_setting_names()
    return {k: ("" if k in hidden else v) for k, v in settings.items()}


# Admin console grouping. A table rather than a `group=` on all 78 fields:
# the same information, in one place you can read top to bottom, and a new
# setting that matches no rule still shows up (under "Other") instead of
# vanishing from the console.
_SETTING_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Model", ("model_path", "model_backend", "model_context_window",
               "model_reasoning_effort", "default_adapter_mode",
               "adapter_openai_base_url", "adapter_openai_api_key",
               "adapter_server_model", "gemini_api_key")),
    ("Retrieval", ("rag_mode", "rag_chunk_size", "embedding_model_id",
                   "history_budget_fraction", "history_recall_fraction")),
    ("Features", ("notes_enabled", "allow_signup", "enable_mfa",
                  "web_tools_enabled", "extract_readers")),
    ("Web tools", ("web_search_provider", "web_search_engine_id",
                   "web_search_api_key", "web_fetch_timeout",
                   "web_fetch_max_bytes", "tool_fetch_timeout",
                   "tool_fetch_connect_timeout", "tool_network_proxy_url")),
    ("Sign-in with a provider", ("oauth_",)),
    ("Security", ("cors_", "enable_hsts", "tool_network_allowlist",
                  "web_fetch_allow_private")),
    ("Sessions & tokens", ("session_", "access_token_", "refresh_token_")),
    ("Rate limits", ("rate_limit_", "_rate_limit_")),
    ("Concurrency", ("max_concurrent_", "max_websocket_")),
    ("Pagination", ("default_page_size", "max_page_size",
                    "default_conversations_limit")),
    ("Files", ("max_upload_bytes", "tmp_")),
    ("Training", ("training_", "max_active_training_jobs")),
    ("Email", ("smtp_", "email_from_")),
    ("Voice", ("voice_",)),
    ("Tenancy", ("default_tenant_id", "tenant_domains", "trust_forwarded_host")),
    ("URLs & identity", ("app_base_url", "oauth_redirect_uri", "jwt_")),
    ("Operations", ("settings_watch_interval_seconds",)),
    ("Infrastructure", ("redis_url", "allow_redis_fallback_dev",
                        "cluster_bus_backend", "shared_fs_root",
                        "interpreter_scratch_dir")),
)

# Changing one of these rebuilds the model service stack, which takes a moment
# and briefly interrupts in-flight work. The console says so before you save.
MODEL_AFFECTING_SETTINGS = frozenset({
    "model_backend",
    "model_path",
    "default_adapter_mode",
    "rag_mode",
    "embedding_model_id",
    "rag_chunk_size",
})


def _group_for(name: str) -> str:
    for label, patterns in _SETTING_GROUPS:
        for pattern in patterns:
            if name == pattern or (pattern.endswith("_") and pattern in name):
                return label
    return "Other"


def managed_settings_schema() -> list[dict]:
    """Describe every managed setting well enough to render a form for it.

    The admin console builds itself from this. It used to hand-mirror the
    field list — every setting written out twice in JavaScript plus a block of
    HTML — which is why thirty settings had no control at all and one that had
    since become env-only was still being posted, failing every save.
    """
    described: list[dict] = []
    for name, field in Settings.model_fields.items():
        extra = field.json_schema_extra or {}
        if not (isinstance(extra, dict) and extra.get("admin")):
            continue
        entry: dict[str, Any] = {
            "name": name,
            "secret": bool(extra.get("secret")),
            "group": _group_for(name),
            "description": field.description or "",
            "default": SYSTEM_SETTINGS_DEFAULTS[name],
            "reloads_model": name in MODEL_AFFECTING_SETTINGS,
        }
        annotation = field.annotation
        # Unwrap Optional[...]: several of these are `Enum | None`, and an
        # annotation that is a union hides the enum from a naive issubclass.
        if get_origin(annotation) is UnionType or get_origin(annotation) is Union:
            candidates = [a for a in get_args(annotation) if a is not type(None)]
            if len(candidates) == 1:
                annotation = candidates[0]
        choices = get_args(annotation) if get_origin(annotation) is Literal else ()
        if isinstance(annotation, type) and issubclass(annotation, Enum):
            choices = tuple(member.value for member in annotation)
        if choices:
            entry["type"] = "choice"
            entry["choices"] = [str(choice) for choice in choices]
        elif entry["default"] is True or entry["default"] is False:
            entry["type"] = "bool"
        elif isinstance(entry["default"], int):
            entry["type"] = "int"
        elif isinstance(entry["default"], float):
            entry["type"] = "float"
        else:
            entry["type"] = "text"
        for constraint in field.metadata:
            for attr, key in (
                ("ge", "min"), ("gt", "exclusive_min"),
                ("le", "max"), ("lt", "exclusive_max"),
            ):
                value = getattr(constraint, attr, None)
                if value is not None:
                    entry[key] = value
        described.append(entry)
    order = [label for label, _ in _SETTING_GROUPS] + ["Other"]
    described.sort(key=lambda item: (order.index(item["group"]), item["name"]))
    return described


@lru_cache(maxsize=None)
def _field_adapter(name: str) -> TypeAdapter:
    """Validate one field's type and bounds, without the model's own rules.

    Constructing a whole Settings to check a single field runs the model-level
    validators too, which then complain about fields the admin never touched
    (setting an SMTP host would report a missing password before the merged
    view was even considered). Those rules belong to the second pass.
    """
    field = Settings.model_fields[name]
    parts: list[Any] = [field.annotation, *field.metadata]
    # The field's own validators, which the raw annotation does not carry.
    # Without them this pass judged a field by its declared type alone, so a
    # setting whose validator accepts a friendlier form — tenant_domains as
    # typed JSON, cors_allow_origins and tool_network_allowlist as a
    # comma-separated list — was rejected before the friendlier form was ever
    # parsed. Those three were simply not settable from the console.
    for dec in Settings.__pydantic_decorators__.field_validators.values():
        if name not in dec.info.fields:
            continue
        fn = getattr(dec.func, "__func__", dec.func)
        wrapper = BeforeValidator if dec.info.mode == "before" else AfterValidator
        parts.append(wrapper(lambda v, _fn=fn: _fn(Settings, v)))
    if len(parts) == 1:
        return TypeAdapter(field.annotation)
    return TypeAdapter(Annotated[tuple(parts)])


def validate_managed_settings(patch: dict, current: dict | None = None) -> dict[str, str]:
    """Check an admin's patch against the Settings model itself.

    Returns {field: message} for whatever failed, empty when the patch is
    good. Types, bounds and allowed values are declared once, on the fields;
    re-stating them in the API is how the two end up disagreeing, and the one
    that would be wrong is the API — the model is what the code runs against.

    Two passes. Per-field first, so the response names every bad field rather
    than only the first. Then the patch applied to the current settings as a
    whole, because some rules span fields (SMTP needs a password once a host
    is set) and cannot be judged one field at a time.
    """
    errors: dict[str, str] = {}
    for key, value in patch.items():
        if key not in SYSTEM_SETTINGS_DEFAULTS:
            errors[key] = "not a managed setting"
            continue
        try:
            _field_adapter(key).validate_python(value)
        except ValidationError as exc:
            errors[key] = exc.errors()[0].get("msg", "invalid value")
    if errors:
        return errors

    merged = {**(current or {}), **patch}
    merged = {k: v for k, v in merged.items() if k in SYSTEM_SETTINGS_DEFAULTS}
    try:
        Settings(**merged)
    except ValidationError as exc:
        for detail in exc.errors():
            # A cross-field rule names no single field; attribute it to one the
            # admin actually touched so the console can point at a control.
            field = str(detail["loc"][0]) if detail.get("loc") else ""
            if field not in patch:
                field = next(iter(patch), "settings")
            errors[field] = detail.get("msg", "invalid value").replace(
                "Value error, ", ""
            )
    return errors


def apply_managed_settings(base: Settings, stored: dict) -> Settings:
    """Overlay the admin-managed values a store handed back onto `base`.

    The result is what the rest of the code should read: one object where every
    setting already answers correctly, instead of sixty call sites each doing
    `stored.get("x", <default written out a third time>)`.

    Values that fail validation are dropped with a warning rather than taking
    the process down — a bad row in instance_config must not make the instance
    unbootable, since the admin UI that would fix it is served by this process.
    """
    if not stored:
        return base
    managed = {
        name
        for name, field in Settings.model_fields.items()
        if isinstance(field.json_schema_extra, dict)
        and field.json_schema_extra.get("admin")
    }
    updates = {k: v for k, v in stored.items() if k in managed}
    if not updates:
        return base
    try:
        return base.model_copy(update=Settings(**updates).model_dump(include=set(updates)))
    except ValidationError as exc:
        logger.warning("managed_settings_invalid", error=str(exc))
    # One bad value should not discard the good ones: apply them one at a time.
    resolved = base
    for key, value in updates.items():
        try:
            resolved = resolved.model_copy(
                update=Settings(**{key: value}).model_dump(include={key})
            )
        except ValidationError:
            logger.warning("managed_setting_invalid", setting=key)
    return resolved


def get_settings() -> Settings:
    global _settings_cache
    if _settings_cache is None:
        _settings_cache = Settings.from_env()
    return _settings_cache


_settings_cache: Settings | None = None


def reset_settings_cache() -> None:
    """Clear cached settings so future calls re-read the environment."""

    global _settings_cache
    _settings_cache = None
