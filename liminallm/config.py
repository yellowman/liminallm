from __future__ import annotations

from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Runtime settings aligned with the SPEC kernel contracts."""

    database_url: str = Field("postgresql://localhost:5432/liminallm", env="DATABASE_URL")
    redis_url: str = Field("redis://localhost:6379/0", env="REDIS_URL")
    shared_fs_root: str = Field("/srv/liminallm", env="SHARED_FS_ROOT")
    model_path: str = Field("gpt-4o-mini", env="MODEL_PATH")
    model_backend: str | None = Field("openai", env="MODEL_BACKEND")
    adapter_openai_api_key: str | None = Field(None, env="OPENAI_ADAPTER_API_KEY")
    adapter_openai_base_url: str | None = Field(None, env="OPENAI_ADAPTER_BASE_URL")
    adapter_server_model: str | None = Field(None, env="ADAPTER_SERVER_MODEL")
    allow_signup: bool = Field(True, env="ALLOW_SIGNUP")
    use_memory_store: bool = Field(False, env="USE_MEMORY_STORE")
    chat_rate_limit_per_minute: int = Field(60, env="CHAT_RATE_LIMIT_PER_MINUTE")
    chat_rate_limit_window_seconds: int = Field(60, env="CHAT_RATE_LIMIT_WINDOW_SECONDS")
    enable_mfa: bool = Field(True, env="ENABLE_MFA")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


def get_settings() -> Settings:
    return Settings()  # relies on pydantic caching
