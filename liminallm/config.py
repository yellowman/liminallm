from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, field_validator

from liminallm.logging import get_logger

logger = get_logger(__name__)


class ModelBackend(str, Enum):
    """Accepted model backend modes as defined in SPEC §5."""

    OPENAI = "openai"
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


class RagMode(str, Enum):
    """RAG retrieval implementations supported by the kernel."""

    PGVECTOR = "pgvector"
    MEMORY = "memory"


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
    "azure": {AdapterMode.REMOTE, AdapterMode.PROMPT, AdapterMode.HYBRID},
    "azure_openai": {AdapterMode.REMOTE, AdapterMode.PROMPT, AdapterMode.HYBRID},
    "vertex": {AdapterMode.REMOTE, AdapterMode.PROMPT, AdapterMode.HYBRID},
    "gemini": {AdapterMode.REMOTE, AdapterMode.PROMPT, AdapterMode.HYBRID},
    "google": {AdapterMode.REMOTE, AdapterMode.PROMPT, AdapterMode.HYBRID},
    "bedrock": {AdapterMode.REMOTE, AdapterMode.PROMPT, AdapterMode.HYBRID},
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
    model_path: str = env_field("gpt-4o-mini", "MODEL_PATH")
    model_backend: ModelBackend | None = env_field(ModelBackend.OPENAI, "MODEL_BACKEND")
    adapter_openai_api_key: str | None = env_field(None, "OPENAI_ADAPTER_API_KEY")
    adapter_openai_base_url: str | None = env_field(None, "OPENAI_ADAPTER_BASE_URL")
    adapter_server_model: str | None = env_field(None, "ADAPTER_SERVER_MODEL")
    # Voice service settings
    voice_api_key: str | None = env_field(None, "VOICE_API_KEY")
    voice_transcription_model: str = env_field("whisper-1", "VOICE_TRANSCRIPTION_MODEL")
    voice_synthesis_model: str = env_field("tts-1", "VOICE_SYNTHESIS_MODEL")
    voice_default_voice: str = env_field("alloy", "VOICE_DEFAULT_VOICE")
    # OAuth settings
    oauth_google_client_id: str | None = env_field(None, "OAUTH_GOOGLE_CLIENT_ID")
    oauth_google_client_secret: str | None = env_field(None, "OAUTH_GOOGLE_CLIENT_SECRET")
    oauth_github_client_id: str | None = env_field(None, "OAUTH_GITHUB_CLIENT_ID")
    oauth_github_client_secret: str | None = env_field(None, "OAUTH_GITHUB_CLIENT_SECRET")
    oauth_microsoft_client_id: str | None = env_field(None, "OAUTH_MICROSOFT_CLIENT_ID")
    oauth_microsoft_client_secret: str | None = env_field(None, "OAUTH_MICROSOFT_CLIENT_SECRET")
    oauth_redirect_uri: str | None = env_field(None, "OAUTH_REDIRECT_URI")
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
    email_from_address: str | None = env_field(
        None, "EMAIL_FROM_ADDRESS", description="Email from address (overridable via admin UI)"
    )
    email_from_name: str = env_field(
        "LiminalLM", "EMAIL_FROM_NAME", description="Email from name (overridable via admin UI)"
    )
    app_base_url: str = env_field("http://localhost:8000", "APP_BASE_URL")
    default_adapter_mode: AdapterMode = env_field(
        AdapterMode.HYBRID,
        "DEFAULT_ADAPTER_MODE",
        description="Default mode for new adapters: local, remote, prompt, or hybrid",
    )
    allow_signup: bool = env_field(
        True,
        "ALLOW_SIGNUP",
        description="Allow new user signups (overridable via admin UI)",
    )
    use_memory_store: bool = env_field(False, "USE_MEMORY_STORE")
    allow_redis_fallback_dev: bool = env_field(False, "ALLOW_REDIS_FALLBACK_DEV")
    test_mode: bool = env_field(
        False,
        "TEST_MODE",
        description="Toggle deterministic testing behaviors; required for CI pathways described in SPEC §14.",
    )
    enable_mfa: bool = env_field(
        True,
        "ENABLE_MFA",
        description="Enable multi-factor authentication (overridable via admin UI)",
    )
    jwt_secret: str = env_field(None, "JWT_SECRET")
    jwt_issuer: str = env_field("liminallm", "JWT_ISSUER")
    jwt_audience: str = env_field("liminal-clients", "JWT_AUDIENCE")
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
    default_tenant_id: str = env_field("public", "DEFAULT_TENANT_ID")
    rag_mode: RagMode = env_field(RagMode.PGVECTOR, "RAG_MODE")
    embedding_model_id: str = env_field("text-embedding", "EMBEDDING_MODEL_ID")

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
    # - read_rate_limit_per_minute
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

    model_config = ConfigDict(extra="ignore")

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

    @field_validator("jwt_secret")
    @classmethod
    def _ensure_jwt_secret(cls, value: str | None) -> str:
        if value:
            return value
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
                if persisted and len(persisted) >= 32:  # Minimum reasonable secret length
                    return persisted
            except Exception as exc:
                logger.error(
                    "jwt_secret_read_failed", error=str(exc), path=str(secret_path)
                )

        generated = secrets.token_urlsafe(64)
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
