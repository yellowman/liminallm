from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

from liminallm.config import get_settings, reset_settings_cache
from liminallm.service.config_ops import ConfigOpsService
from liminallm.service.auth import AuthService
from liminallm.service.clustering import SemanticClusterer
from liminallm.service.llm import LLMService
from liminallm.service.rag import RAGService
from liminallm.service.router import RouterEngine
from liminallm.service.training import TrainingService
from liminallm.service.training_worker import TrainingWorker
from liminallm.service.voice import VoiceService
from liminallm.service.workflow import WorkflowEngine
from liminallm.service.email import EmailService
from liminallm.storage.memory import MemoryStore
from liminallm.storage.postgres import PostgresStore
from liminallm.storage.redis_cache import RedisCache
from liminallm.service.embeddings import EmbeddingsService

from liminallm.logging import get_logger

logger = get_logger(__name__)


class Runtime:
    """Holds singleton service instances for the FastAPI app."""

    def __init__(self):
        self.settings = get_settings()
        self.store = (
            MemoryStore(fs_root=self.settings.shared_fs_root)
            if self.settings.use_memory_store
            else PostgresStore(
                self.settings.database_url, fs_root=self.settings.shared_fs_root
            )
        )
        self.cache = None
        redis_error: Exception | None = None
        if self.settings.redis_url:
            try:
                cache = RedisCache(self.settings.redis_url)
                cache.verify_connection()
                self.cache = cache
            except Exception as exc:
                redis_error = exc
                self.cache = None

        if not self.cache:
            if (
                not self.settings.test_mode
                and not self.settings.allow_redis_fallback_dev
            ):
                raise RuntimeError(
                    "Redis is required for sessions, rate limits, idempotency, and workflow caches; "
                    "start Redis or set TEST_MODE=true/ALLOW_REDIS_FALLBACK_DEV=true for local fallback."
                ) from redis_error

            fallback_mode = (
                "TEST_MODE" if self.settings.test_mode else "ALLOW_REDIS_FALLBACK_DEV"
            )

            logger.warning(
                "redis_disabled_fallback",
                redis_url=self.settings.redis_url,
                error=str(redis_error) if redis_error else "redis_url_missing",
                message=(
                    f"Running without Redis under {fallback_mode}; rate limits, idempotency durability, and "
                    "workflow/router caches are in-memory only."
                ),
                mode=fallback_mode,
            )
        # Compute backend_mode before creating services that need it
        runtime_config = {}
        db_backend_mode = None
        if hasattr(self.store, "get_runtime_config"):
            runtime_config = self.store.get_runtime_config() or {}
            db_backend_mode = runtime_config.get("model_backend")
        resolved_base_model = (
            runtime_config.get("model_path") or self.settings.model_path
        )
        backend_mode = db_backend_mode or self.settings.model_backend
        self.router = RouterEngine(cache=self.cache, backend_mode=backend_mode)
        adapter_configs = {
            "openai": {
                "api_key": self.settings.adapter_openai_api_key,
                "base_url": self.settings.adapter_openai_base_url,
                "adapter_server_model": self.settings.adapter_server_model,
            }
        }
        self.embeddings = EmbeddingsService(self.settings.embedding_model_id)
        self.llm = LLMService(
            base_model=resolved_base_model,
            backend_mode=backend_mode,
            adapter_configs=adapter_configs,
            api_key=self.settings.adapter_openai_api_key,
            base_url=self.settings.adapter_openai_base_url,
            adapter_server_model=self.settings.adapter_server_model,
            fs_root=self.settings.shared_fs_root,
        )
        self.rag = RAGService(
            self.store,
            default_chunk_size=self.settings.rag_chunk_size,
            rag_mode=self.settings.rag_mode,
            embed=self.embeddings.embed,
            embedding_model_id=self.settings.embedding_model_id,
        )
        self.training = TrainingService(
            self.store,
            self.settings.shared_fs_root,
            runtime_base_model=resolved_base_model,
            default_adapter_mode=self.settings.default_adapter_mode,
            backend_mode=backend_mode,
        )
        self.clusterer = SemanticClusterer(self.store, self.llm, self.training)
        self.workflow = WorkflowEngine(
            self.store, self.llm, self.router, self.rag, cache=self.cache
        )
        self.voice = VoiceService(
            self.settings.shared_fs_root,
            api_key=self.settings.voice_api_key,
            transcription_model=self.settings.voice_transcription_model,
            synthesis_model=self.settings.voice_synthesis_model,
            default_voice=self.settings.voice_default_voice,
        )
        self.config_ops = ConfigOpsService(
            self.store, self.llm, self.router, self.training
        )
        self.auth = AuthService(
            self.store,
            self.cache,
            self.settings,
            mfa_enabled=self.settings.enable_mfa,
        )
        self.email = EmailService(
            smtp_host=self.settings.smtp_host,
            smtp_port=self.settings.smtp_port,
            smtp_user=self.settings.smtp_user,
            smtp_password=self.settings.smtp_password,
            smtp_use_tls=self.settings.smtp_use_tls,
            from_email=self.settings.email_from_address,
            from_name=self.settings.email_from_name,
            base_url=self.settings.app_base_url,
        )
        # Training worker for background job processing
        self.training_worker = TrainingWorker(
            store=self.store,
            training_service=self.training,
            clusterer=self.clusterer,
            poll_interval=self.settings.training_worker_poll_interval,
        )
        self._local_idempotency: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        self._local_idempotency_lock = asyncio.Lock()
        self._local_rate_limits: Dict[str, Tuple[datetime, int]] = {}
        self._local_rate_limit_lock = asyncio.Lock()


runtime: Runtime | None = None


def get_runtime() -> Runtime:
    global runtime
    if runtime is None:
        runtime = Runtime()
    return runtime


def reset_runtime_for_tests() -> Runtime:
    """Reinitialize the runtime singleton for isolated test runs."""

    global runtime
    reset_settings_cache()
    settings = get_settings()
    if not settings.test_mode:
        raise RuntimeError("runtime reset is only allowed in TEST_MODE")
    runtime = Runtime()
    return runtime


IDEMPOTENCY_TTL_SECONDS = 60 * 60 * 24


async def _get_cached_idempotency_record(
    runtime: Runtime, route: str, user_id: str, key: str
) -> Optional[dict]:
    now = datetime.utcnow()
    if runtime.cache:
        return await runtime.cache.get_idempotency_record(route, user_id, key)
    async with runtime._local_idempotency_lock:
        record = runtime._local_idempotency.get((route, user_id, key))
        if not record:
            return None
        if record.get("expires_at") and record["expires_at"] < now:
            runtime._local_idempotency.pop((route, user_id, key), None)
            return None
        return record


async def _set_cached_idempotency_record(
    runtime: Runtime,
    route: str,
    user_id: str,
    key: str,
    record: dict,
    *,
    ttl_seconds: int = IDEMPOTENCY_TTL_SECONDS,
) -> None:
    expires_at = datetime.utcnow() + timedelta(seconds=ttl_seconds)
    if runtime.cache:
        await runtime.cache.set_idempotency_record(
            route, user_id, key, record, ttl_seconds=ttl_seconds
        )
        return
    async with runtime._local_idempotency_lock:
        runtime._local_idempotency[(route, user_id, key)] = {
            **record,
            "expires_at": expires_at,
        }


from typing import Union, Tuple as TypingTuple


async def check_rate_limit(
    runtime: Runtime, key: str, limit: int, window_seconds: int, *, return_remaining: bool = False
) -> Union[bool, TypingTuple[bool, int]]:
    """Enforce rate limits even when Redis is unavailable.

    Per SPEC §18, rate limits use Redis token bucket with configurable defaults.

    Args:
        runtime: Runtime instance with cache
        key: Rate limit key
        limit: Maximum requests per window
        window_seconds: Window duration in seconds
        return_remaining: If True, return tuple of (allowed, remaining)

    Returns:
        bool if return_remaining is False, else (bool, int) tuple
    """
    if limit <= 0:
        return (True, limit) if return_remaining else True
    if window_seconds <= 0:
        logger.warning(
            "rate_limit_invalid_window",
            key=key,
            window_seconds=window_seconds,
            message="Invalid rate limit window_seconds; defaulting to 60 seconds",
        )
        window_seconds = 60  # Default to 1 minute if invalid window per SPEC §18
    now = datetime.utcnow()
    if runtime.cache:
        result = await runtime.cache.check_rate_limit(key, limit, window_seconds, return_remaining=return_remaining)
        return result
    window = timedelta(seconds=window_seconds)
    async with runtime._local_rate_limit_lock:
        window_start, count = runtime._local_rate_limits.get(key, (now, 0))
        if now - window_start >= window:
            window_start, count = now, 0
        new_count = count + 1
        allowed = new_count <= limit
        if allowed:
            runtime._local_rate_limits[key] = (window_start, new_count)
        remaining = max(0, limit - new_count)
    if return_remaining:
        return (allowed, remaining)
    return allowed
