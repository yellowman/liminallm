from __future__ import annotations

import json
import os
import secrets
import string
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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
    BEDROCK = "bedrock"
    TOGETHER = "together"
    TOGETHER_AI = "together.ai"
    LORAX = "lorax"
    ADAPTER_SERVER = "adapter_server"
    SAGEMAKER = "sagemaker"
    AWS_SAGEMAKER = "aws_sagemaker"
    # Zhipu AI GLM models (glm-4, glm-4-plus, etc.)
    ZHIPU = "zhipu"
    ZHIPU_AI = "zhipu.ai"
    GLM = "glm"
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
}


def resolve_provider_endpoint(mode: str) -> Optional[dict[str, Optional[str]]]:
    """Return endpoint wiring for an OpenAI-compatible API provider backend mode.

    Returns None for modes that are not API providers (stub, local_lora,
    local_gpu_lora, adapter_server), which are constructed by other paths.
    """
    return PROVIDER_ENDPOINTS.get((mode or "").lower())


def env_field(default: Any, env: str, **kwargs):
    extra = kwargs.pop("json_schema_extra", {}) or {}
    extra = {**extra, "env": env}
    return Field(default, json_schema_extra=extra, **kwargs)


class Settings(BaseModel):
    """Runtime settings aligned with the SPEC kernel contracts."""

    database_url: str = env_field(
        "postgresql://localhost:5432/liminallm", "DATABASE_URL"
    )
    redis_url: str = env_field("redis://localhost:6379/0", "REDIS_URL")
    shared_fs_root: str = env_field("/srv/liminallm", "SHARED_FS_ROOT")
    tmp_cleanup_interval_seconds: int = env_field(
        86400,
        "TMP_CLEANUP_INTERVAL_SECONDS",
        description="How often to sweep per-user tmp scratch directories (seconds)",
    )
    tmp_max_age_hours: int = env_field(
        24,
        "TMP_MAX_AGE_HOURS",
        description="Delete tmp scratch files older than this many hours",
    )
    model_path: str = env_field(
        "gpt-4o-mini", "MODEL_PATH", description="Model path (overridable via admin UI)"
    )
    model_backend: ModelBackend | None = env_field(
        ModelBackend.OPENAI, "MODEL_BACKEND", description="Model backend (overridable via admin UI)"
    )
    adapter_openai_api_key: str | None = env_field(None, "OPENAI_ADAPTER_API_KEY")
    adapter_openai_base_url: str | None = env_field(None, "OPENAI_ADAPTER_BASE_URL")
    adapter_server_model: str | None = env_field(None, "ADAPTER_SERVER_MODEL")
    # Voice service settings
    voice_api_key: str | None = env_field(None, "VOICE_API_KEY")
    voice_transcription_model: str = env_field(
        "whisper-1", "VOICE_TRANSCRIPTION_MODEL",
        description="Transcription model (overridable via admin UI)"
    )
    voice_synthesis_model: str = env_field(
        "tts-1", "VOICE_SYNTHESIS_MODEL",
        description="Synthesis model (overridable via admin UI)"
    )
    voice_default_voice: str = env_field(
        "alloy", "VOICE_DEFAULT_VOICE",
        description="Default voice (overridable via admin UI)"
    )
    # OAuth settings
    oauth_google_client_id: str | None = env_field(None, "OAUTH_GOOGLE_CLIENT_ID")
    oauth_google_client_secret: str | None = env_field(None, "OAUTH_GOOGLE_CLIENT_SECRET")
    oauth_github_client_id: str | None = env_field(None, "OAUTH_GITHUB_CLIENT_ID")
    oauth_github_client_secret: str | None = env_field(None, "OAUTH_GITHUB_CLIENT_SECRET")
    oauth_microsoft_client_id: str | None = env_field(None, "OAUTH_MICROSOFT_CLIENT_ID")
    oauth_microsoft_client_secret: str | None = env_field(None, "OAUTH_MICROSOFT_CLIENT_SECRET")
    oauth_redirect_uri: str | None = env_field(
        None, "OAUTH_REDIRECT_URI", description="OAuth redirect URI (overridable via admin UI)"
    )
    # Email service settings (env vars are fallbacks - prefer admin UI)
    smtp_host: str | None = env_field(
        None, "SMTP_HOST", description="SMTP server host (overridable via admin UI)"
    )
    smtp_port: int = env_field(
        587, "SMTP_PORT", description="SMTP server port (overridable via admin UI)"
    )
    smtp_user: str | None = env_field(
        None, "SMTP_USER", description="SMTP username (overridable via admin UI)"
    )
    smtp_password: str | None = env_field(
        None, "SMTP_PASSWORD", description="SMTP password (overridable via admin UI)"
    )
    smtp_use_tls: bool = env_field(
        True, "SMTP_USE_TLS", description="Use TLS for SMTP (overridable via admin UI)"
    )
    smtp_allow_insecure: bool = env_field(
        False,
        "SMTP_ALLOW_INSECURE",
        description="Allow plaintext SMTP when explicitly enabled (overridable via admin UI)",
    )
    email_from_address: str | None = env_field(
        None, "EMAIL_FROM_ADDRESS", description="Email from address (overridable via admin UI)"
    )
    email_from_name: str = env_field(
        "LiminalLM", "EMAIL_FROM_NAME", description="Email from name (overridable via admin UI)"
    )
    app_base_url: str = env_field(
        "http://localhost:8000", "APP_BASE_URL",
        description="Application base URL (overridable via admin UI)"
    )
    default_adapter_mode: AdapterMode = env_field(
        AdapterMode.HYBRID,
        "DEFAULT_ADAPTER_MODE",
        description="Default mode for new adapters: local, remote, prompt, or hybrid (overridable via admin UI)",
    )
    allow_signup: bool = env_field(
        True,
        "ALLOW_SIGNUP",
        description="Allow new user signups (overridable via admin UI)",
    )
    log_level: str = env_field("INFO", "LOG_LEVEL")
    log_json: bool = env_field(True, "LOG_JSON")
    log_dev_mode: bool = env_field(False, "LOG_DEV_MODE")
    build_sha: str = env_field("dev", "BUILD_SHA")
    cors_allow_origins: list[str] = env_field(
        [
            "http://localhost",
            "http://localhost:3000",
            "http://localhost:5173",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:5173",
        ],
        "CORS_ALLOW_ORIGINS",
        description="Comma-separated CORS origins (overridable via admin UI)",
    )
    cors_allow_credentials: bool = env_field(False, "CORS_ALLOW_CREDENTIALS")
    enable_hsts: bool = env_field(False, "ENABLE_HSTS")
    mfa_secret_key: str | None = env_field(None, "MFA_SECRET_KEY")
    use_memory_store: bool = env_field(False, "USE_MEMORY_STORE")
    allow_redis_fallback_dev: bool = env_field(False, "ALLOW_REDIS_FALLBACK_DEV")
    test_mode: bool = env_field(
        False,
        "TEST_MODE",
        description="Toggle deterministic testing behaviors; required for CI pathways described in SPEC §14.",
    )
    tool_network_allowlist: list[str] = env_field(
        ["api.openai.com"],
        "TOOL_NETWORK_ALLOWLIST",
        description="Allowlisted hostnames/CIDRs for tool egress (SPEC §18)",
    )
    tool_network_proxy_url: str | None = env_field(
        None,
        "TOOL_NETWORK_PROXY_URL",
        description="Proxy URL tools must use for outbound HTTP(S) fetches",
    )
    tool_fetch_connect_timeout: float = env_field(
        10.0,
        "TOOL_FETCH_CONNECT_TIMEOUT",
        description="Connect timeout (seconds) for tool HTTP fetches",
    )
    tool_fetch_timeout: float = env_field(
        30.0,
        "TOOL_FETCH_TIMEOUT",
        description="Total timeout (seconds) for tool HTTP fetches",
    )
    enable_mfa: bool = env_field(
        True,
        "ENABLE_MFA",
        description="Enable multi-factor authentication (overridable via admin UI)",
    )
    jwt_secret: str = env_field(None, "JWT_SECRET", validate_default=True)
    jwt_issuer: str = env_field(
        "liminallm", "JWT_ISSUER",
        description="JWT issuer (overridable via admin UI)"
    )
    jwt_audience: str = env_field(
        "liminal-clients", "JWT_AUDIENCE",
        description="JWT audience (overridable via admin UI)"
    )
    access_token_ttl_minutes: int = env_field(
        30,
        "ACCESS_TOKEN_TTL_MINUTES",
        description="Access token TTL in minutes (overridable via admin UI)",
    )
    refresh_token_ttl_minutes: int = env_field(
        24 * 60,
        "REFRESH_TOKEN_TTL_MINUTES",
        description="Refresh token TTL in minutes (overridable via admin UI)",
    )
    default_tenant_id: str = env_field(
        "public", "DEFAULT_TENANT_ID",
        description="Default tenant ID (overridable via admin UI)"
    )
    rag_mode: RagMode = env_field(
        RagMode.PGVECTOR, "RAG_MODE",
        description="RAG mode: pgvector or memory (overridable via admin UI)"
    )
    embedding_model_id: str = env_field(
        "text-embedding", "EMBEDDING_MODEL_ID",
        description="Embedding model ID (overridable via admin UI)"
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
    # - smtp_host, smtp_port, smtp_user, smtp_password, smtp_use_tls
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
    training_worker_enabled: bool = env_field(
        True,
        "TRAINING_WORKER_ENABLED",
        description="Enable background training job worker (overridable via admin UI)",
    )
    training_worker_poll_interval: int = env_field(
        60,
        "TRAINING_WORKER_POLL_INTERVAL",
        description="Training worker poll interval in seconds (overridable via admin UI)",
    )
    max_active_training_jobs: int = env_field(
        10,
        "MAX_ACTIVE_TRAINING_JOBS",
        description="Global cap on simultaneously active training jobs",
    )

    model_config = ConfigDict(extra="ignore")

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
        "smtp_port", "training_worker_poll_interval", "tmp_cleanup_interval_seconds", "tmp_max_age_hours", "max_active_training_jobs"
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

    @field_validator("log_level")
    @classmethod
    def _normalize_log_level(cls, value: str) -> str:
        return (value or "INFO").upper()

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
        env_file_values = dotenv_values(".env")
        merged: dict[str, str] = {}
        for name, field in cls.model_fields.items():
            extra = field.json_schema_extra or {}
            env_key = extra.get("env") if isinstance(extra, dict) else None
            env_name = env_key or name.upper()
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

    @field_validator("jwt_secret", mode="before")
    @classmethod
    def _ensure_jwt_secret(cls, value: str | None) -> str:
        def _validate_secret(secret: str) -> str:
            secret = secret.strip()
            if len(secret) < 32:
                raise ValueError("JWT_SECRET must be at least 32 characters long")

            character_classes = [
                any(ch.islower() for ch in secret),
                any(ch.isupper() for ch in secret),
                any(ch.isdigit() for ch in secret),
                any(ch in string.punctuation for ch in secret),
            ]
            if sum(character_classes) < 3 or len(set(secret)) < 10:
                raise ValueError(
                    "JWT_SECRET must mix character classes and contain sufficient unique characters",
                )
            return secret

        if value:
            return _validate_secret(value)
        # Persist a generated JWT secret so tokens remain valid across restarts
        fs_root = Path(os.getenv("SHARED_FS_ROOT", "/srv/liminallm"))
        secret_path = fs_root / ".jwt_secret"

        # Create directory with restrictive permissions
        try:
            fs_root.mkdir(parents=True, exist_ok=True)
            # Ensure directory has restrictive permissions
            os.chmod(fs_root, 0o700)
        except PermissionError:
            # Directory may already exist with different permissions (e.g., in container)
            pass
        except Exception as exc:
            logger.warning(
                "jwt_secret_dir_setup",
                error=str(exc),
                path=str(fs_root),
                message="Could not set directory permissions",
            )

        if secret_path.exists() and not secret_path.is_symlink():
            try:
                persisted = secret_path.read_text().strip()
                if persisted:
                    return _validate_secret(persisted)
            except Exception as exc:
                logger.error(
                    "jwt_secret_read_failed", error=str(exc), path=str(secret_path)
                )

        generated = _validate_secret(secrets.token_urlsafe(64))
        try:
            # Use atomic write pattern: write to temp file then rename
            import tempfile
            fd, tmp_path = tempfile.mkstemp(
                dir=str(fs_root), prefix=".jwt_secret_", suffix=".tmp"
            )
            try:
                os.write(fd, generated.encode())
                os.fchmod(fd, 0o600)  # Set permissions before closing
            finally:
                os.close(fd)
            os.rename(tmp_path, str(secret_path))
        except Exception as exc:
            # Clean up temp file if it exists
            try:
                if "tmp_path" in locals():
                    os.unlink(tmp_path)
            except Exception:
                pass
            logger.error(
                "jwt_secret_persist_failed", error=str(exc), path=str(secret_path)
            )
            raise RuntimeError(
                "Unable to persist JWT secret; set JWT_SECRET env var or make SHARED_FS_ROOT writable"
            ) from exc
        return generated


# Centralized system settings defaults - single source of truth
# Used by routes.py, memory.py, postgres.py, and sql seeds
SYSTEM_SETTINGS_DEFAULTS: dict = {
    # Session & concurrency
    "session_rotation_hours": 24,
    "session_rotation_grace_seconds": 300,
    "max_concurrent_workflows": 3,
    "max_concurrent_inference": 2,
    # Rate limit multipliers
    "rate_limit_multiplier_free": 1.0,
    "rate_limit_multiplier_paid": 2.0,
    "rate_limit_multiplier_enterprise": 5.0,
    # Rate limits (0 = disabled/unlimited)
    "chat_rate_limit_per_minute": 60,
    "chat_rate_limit_window_seconds": 60,
    "login_rate_limit_per_minute": 10,
    "refresh_rate_limit_per_minute": 20,
    "refresh_rate_limit_window_seconds": 60,
    "signup_rate_limit_per_minute": 5,
    "reset_rate_limit_per_minute": 5,
    "mfa_rate_limit_per_minute": 5,
    "admin_rate_limit_per_minute": 30,
    "admin_rate_limit_window_seconds": 60,
    "files_upload_rate_limit_per_minute": 10,
    "websocket_connect_rate_limit_per_minute": 30,
    "configops_rate_limit_per_hour": 30,
    "read_rate_limit_per_minute": 120,
    "write_rate_limit_per_minute": 60,
    "max_websocket_connections_per_user": 5,
    # Pagination
    "default_page_size": 100,
    "max_page_size": 500,
    "default_conversations_limit": 50,
    # Files
    "max_upload_bytes": 10485760,
    "rag_chunk_size": 400,
    # Token TTLs
    "access_token_ttl_minutes": 30,
    "refresh_token_ttl_minutes": 1440,
    # Feature flags
    "enable_mfa": True,
    "allow_signup": True,
    "training_worker_enabled": True,
    "training_worker_poll_interval": 60,
    # SMTP
    "smtp_host": "",
    "smtp_port": 587,
    "smtp_user": "",
    "smtp_password": "",
    "smtp_use_tls": True,
    "smtp_allow_insecure": False,
    "email_from_address": "",
    "email_from_name": "LiminalLM",
    # URLs
    "oauth_redirect_uri": "",
    "app_base_url": "http://localhost:8000",
    # Voice
    "voice_transcription_model": "whisper-1",
    "voice_synthesis_model": "tts-1",
    "voice_default_voice": "alloy",
    # Model/RAG
    "rag_mode": "pgvector",
    "embedding_model_id": "text-embedding",
    "model_path": "gpt-4o-mini",
    "model_backend": "openai",
    "default_adapter_mode": "hybrid",
    # Tenant/JWT
    "default_tenant_id": "public",
    "jwt_issuer": "liminallm",
    "jwt_audience": "liminal-clients",
}

# Derived validation sets for admin settings API
SYSTEM_SETTINGS_INT_KEYS = {
    k for k, v in SYSTEM_SETTINGS_DEFAULTS.items()
    if isinstance(v, int) and not isinstance(v, bool)
}

SYSTEM_SETTINGS_FLOAT_KEYS = {
    k for k, v in SYSTEM_SETTINGS_DEFAULTS.items()
    if isinstance(v, float)
}

SYSTEM_SETTINGS_BOOL_KEYS = {
    k for k, v in SYSTEM_SETTINGS_DEFAULTS.items()
    if isinstance(v, bool)
}

SYSTEM_SETTINGS_STRING_KEYS = {
    k for k, v in SYSTEM_SETTINGS_DEFAULTS.items()
    if isinstance(v, str)
}

# Rate limit keys that allow 0 (disabled/unlimited)
SYSTEM_SETTINGS_RATE_LIMIT_KEYS = {
    k for k in SYSTEM_SETTINGS_DEFAULTS.keys()
    if k.endswith("_per_minute") or k.endswith("_per_hour")
}


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
