from __future__ import annotations

import os
import secrets
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


def env_field(default: Any, env: str, **kwargs):
    extra = kwargs.pop("json_schema_extra", {}) or {}
    extra = {**extra, "env": env}
    return Field(default, json_schema_extra=extra, **kwargs)


class Settings(BaseModel):
    """Runtime settings aligned with the SPEC kernel contracts."""

    database_url: str = env_field("postgresql://localhost:5432/liminallm", "DATABASE_URL")
    redis_url: str = env_field("redis://localhost:6379/0", "REDIS_URL")
    shared_fs_root: str = env_field("/srv/liminallm", "SHARED_FS_ROOT")
    model_path: str = env_field("gpt-4o-mini", "MODEL_PATH")
    model_backend: ModelBackend | None = env_field(ModelBackend.OPENAI, "MODEL_BACKEND")
    adapter_openai_api_key: str | None = env_field(None, "OPENAI_ADAPTER_API_KEY")
    adapter_openai_base_url: str | None = env_field(None, "OPENAI_ADAPTER_BASE_URL")
    adapter_server_model: str | None = env_field(None, "ADAPTER_SERVER_MODEL")
    allow_signup: bool = env_field(True, "ALLOW_SIGNUP")
    use_memory_store: bool = env_field(False, "USE_MEMORY_STORE")
    allow_redis_fallback_dev: bool = env_field(False, "ALLOW_REDIS_FALLBACK_DEV")
    test_mode: bool = env_field(
        False,
        "TEST_MODE",
        description="Toggle deterministic testing behaviors; required for CI pathways described in SPEC §14.",
    )
    chat_rate_limit_per_minute: int = env_field(60, "CHAT_RATE_LIMIT_PER_MINUTE")
    chat_rate_limit_window_seconds: int = env_field(60, "CHAT_RATE_LIMIT_WINDOW_SECONDS")
    enable_mfa: bool = env_field(True, "ENABLE_MFA")
    jwt_secret: str = env_field(None, "JWT_SECRET")
    jwt_issuer: str = env_field("liminallm", "JWT_ISSUER")
    jwt_audience: str = env_field("liminal-clients", "JWT_AUDIENCE")
    access_token_ttl_minutes: int = env_field(30, "ACCESS_TOKEN_TTL_MINUTES")
    refresh_token_ttl_minutes: int = env_field(24 * 60, "REFRESH_TOKEN_TTL_MINUTES")
    default_tenant_id: str = env_field("public", "DEFAULT_TENANT_ID")
    rag_chunk_size: int = env_field(400, "RAG_CHUNK_SIZE")
    rag_mode: RagMode = env_field(RagMode.PGVECTOR, "RAG_MODE")
    embedding_model_id: str = env_field("text-embedding", "EMBEDDING_MODEL_ID")
    max_upload_bytes: int = env_field(10 * 1024 * 1024, "MAX_UPLOAD_BYTES")
    login_rate_limit_per_minute: int = env_field(10, "LOGIN_RATE_LIMIT_PER_MINUTE")
    signup_rate_limit_per_minute: int = env_field(5, "SIGNUP_RATE_LIMIT_PER_MINUTE")
    reset_rate_limit_per_minute: int = env_field(5, "RESET_RATE_LIMIT_PER_MINUTE")
    admin_rate_limit_per_minute: int = env_field(30, "ADMIN_RATE_LIMIT_PER_MINUTE")
    admin_rate_limit_window_seconds: int = env_field(60, "ADMIN_RATE_LIMIT_WINDOW_SECONDS")
    files_upload_rate_limit_per_minute: int = env_field(10, "FILES_UPLOAD_RATE_LIMIT_PER_MINUTE")
    configops_rate_limit_per_hour: int = env_field(30, "CONFIGOPS_RATE_LIMIT_PER_HOUR")

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

    @field_validator("jwt_secret")
    @classmethod
    def _ensure_jwt_secret(cls, value: str | None) -> str:
        if value:
            return value
        # Persist a generated JWT secret so tokens remain valid across restarts
        fs_root = Path(os.getenv("SHARED_FS_ROOT", "/srv/liminallm"))
        secret_path = fs_root / ".jwt_secret"
        secret_path.parent.mkdir(parents=True, exist_ok=True)

        if secret_path.exists():
            try:
                persisted = secret_path.read_text().strip()
                if persisted:
                    return persisted
            except Exception as exc:
                logger.error("jwt_secret_read_failed", error=str(exc), path=str(secret_path))

        generated = secrets.token_urlsafe(64)
        try:
            secret_path.write_text(generated)
            os.chmod(secret_path, 0o600)
        except Exception as exc:
            logger.error("jwt_secret_persist_failed", error=str(exc), path=str(secret_path))
            raise RuntimeError(
                "Unable to persist JWT secret; set JWT_SECRET or make SHARED_FS_ROOT writable"
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
