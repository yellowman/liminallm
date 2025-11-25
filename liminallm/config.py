from __future__ import annotations

from enum import Enum

from pydantic import BaseSettings, Field, field_validator


class ModelBackend(str, Enum):
    """Accepted model backend modes as defined in SPEC §5."""

    OPENAI = "openai"
    AZURE = "azure"
    AZURE_OPENAI = "azure_openai"
    AZURE_OPENAI_ALT = "azure-openai"
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


class Settings(BaseSettings):
    """Runtime settings aligned with the SPEC kernel contracts."""

    database_url: str = Field("postgresql://localhost:5432/liminallm", env="DATABASE_URL")
    redis_url: str = Field("redis://localhost:6379/0", env="REDIS_URL")
    shared_fs_root: str = Field("/srv/liminallm", env="SHARED_FS_ROOT")
    model_path: str = Field("gpt-4o-mini", env="MODEL_PATH")
    model_backend: ModelBackend | None = Field(ModelBackend.OPENAI, env="MODEL_BACKEND")
    adapter_openai_api_key: str | None = Field(None, env="OPENAI_ADAPTER_API_KEY")
    adapter_openai_base_url: str | None = Field(None, env="OPENAI_ADAPTER_BASE_URL")
    adapter_server_model: str | None = Field(None, env="ADAPTER_SERVER_MODEL")
    allow_signup: bool = Field(True, env="ALLOW_SIGNUP")
    use_memory_store: bool = Field(False, env="USE_MEMORY_STORE")
    allow_redis_fallback_dev: bool = Field(False, env="ALLOW_REDIS_FALLBACK_DEV")
    test_mode: bool = Field(
        False,
        env="TEST_MODE",
        description="Toggle deterministic testing behaviors; required for CI pathways described in SPEC §14.",
    )
    chat_rate_limit_per_minute: int = Field(60, env="CHAT_RATE_LIMIT_PER_MINUTE")
    chat_rate_limit_window_seconds: int = Field(60, env="CHAT_RATE_LIMIT_WINDOW_SECONDS")
    enable_mfa: bool = Field(True, env="ENABLE_MFA")
    jwt_secret: str = Field("dev-secret", env="JWT_SECRET")
    jwt_issuer: str = Field("liminallm", env="JWT_ISSUER")
    jwt_audience: str = Field("liminal-clients", env="JWT_AUDIENCE")
    access_token_ttl_minutes: int = Field(30, env="ACCESS_TOKEN_TTL_MINUTES")
    refresh_token_ttl_minutes: int = Field(24 * 60, env="REFRESH_TOKEN_TTL_MINUTES")
    default_tenant_id: str = Field("public", env="DEFAULT_TENANT_ID")
    rag_chunk_size: int = Field(400, env="RAG_CHUNK_SIZE")
    rag_mode: RagMode = Field(RagMode.PGVECTOR, env="RAG_MODE")
    embedding_model_id: str = Field("text-embedding", env="EMBEDDING_MODEL_ID")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

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


def get_settings() -> Settings:
    return Settings()  # relies on pydantic caching
