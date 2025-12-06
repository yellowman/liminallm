from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple, Union

from liminallm.config import get_settings, reset_settings_cache
from liminallm.logging import get_logger
from liminallm.service.auth import AuthService
from liminallm.service.clustering import SemanticClusterer
from liminallm.service.config_ops import ConfigOpsService
from liminallm.service.email import EmailService
from liminallm.service.embeddings import EmbeddingsService
from liminallm.service.llm import LLMService
from liminallm.service.rag import RAGService
from liminallm.service.router import RouterEngine
from liminallm.service.training import TrainingService
from liminallm.service.training_worker import TrainingWorker
from liminallm.service.voice import VoiceService
from liminallm.service.workflow import WorkflowEngine
from liminallm.storage.memory import MemoryStore
from liminallm.storage.postgres import PostgresStore
from liminallm.storage.redis_cache import RedisCache, SyncRedisCache

logger = get_logger(__name__)


class Runtime:
    """Holds singleton service instances for the FastAPI app."""

    def __init__(self):
        self.settings = get_settings()
        logger.info(
            "runtime_init_started",
            use_memory_store=self.settings.use_memory_store,
            test_mode=self.settings.test_mode,
        )

        try:
            self.store = (
                MemoryStore(fs_root=self.settings.shared_fs_root)
                if self.settings.use_memory_store
                else PostgresStore(
                    self.settings.database_url, fs_root=self.settings.shared_fs_root
                )
            )
            logger.info(
                "runtime_store_initialized",
                store_type="memory" if self.settings.use_memory_store else "postgres",
            )
        except Exception as exc:
            logger.error(
                "runtime_store_init_failed",
                store_type="memory" if self.settings.use_memory_store else "postgres",
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise

        self.cache = None
        redis_error: Exception | None = None
        if self.settings.redis_url:
            try:
                # Use sync Redis client in test mode to avoid event loop issues
                if self.settings.test_mode:
                    cache = SyncRedisCache(self.settings.redis_url)
                else:
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
        # Get system settings from DB early (falls back to env vars if not in DB)
        sys_settings = {}
        if hasattr(self.store, "get_system_settings"):
            sys_settings = self.store.get_system_settings() or {}

        # Resolve model settings from DB with env var fallback
        resolved_base_model = (
            sys_settings.get("model_path") or self.settings.model_path
        )
        backend_mode = sys_settings.get("model_backend") or self.settings.model_backend
        default_adapter_mode = (
            sys_settings.get("default_adapter_mode") or self.settings.default_adapter_mode
        )

        # Resolve other settings from DB with env var fallback
        rag_mode = sys_settings.get("rag_mode") or self.settings.rag_mode
        embedding_model_id = (
            sys_settings.get("embedding_model_id") or self.settings.embedding_model_id
        )
        voice_transcription_model = (
            sys_settings.get("voice_transcription_model")
            or self.settings.voice_transcription_model
        )
        voice_synthesis_model = (
            sys_settings.get("voice_synthesis_model")
            or self.settings.voice_synthesis_model
        )
        voice_default_voice = (
            sys_settings.get("voice_default_voice") or self.settings.voice_default_voice
        )
        app_base_url = sys_settings.get("app_base_url") or self.settings.app_base_url

        self.router = RouterEngine(cache=self.cache, backend_mode=backend_mode)
        adapter_configs = {
            "openai": {
                "api_key": self.settings.adapter_openai_api_key,
                "base_url": self.settings.adapter_openai_base_url,
                "adapter_server_model": self.settings.adapter_server_model,
            }
        }
        self.embeddings = EmbeddingsService(embedding_model_id)
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
            rag_mode=rag_mode,
            embed=self.embeddings.embed,
            embedding_model_id=embedding_model_id,
        )
        self.training = TrainingService(
            self.store,
            self.settings.shared_fs_root,
            runtime_base_model=resolved_base_model,
            default_adapter_mode=default_adapter_mode,
            backend_mode=backend_mode,
        )
        self.clusterer = SemanticClusterer(self.store, self.llm, self.training)
        self.workflow = WorkflowEngine(
            self.store, self.llm, self.router, self.rag, cache=self.cache
        )
        self.voice = VoiceService(
            self.settings.shared_fs_root,
            api_key=self.settings.voice_api_key,
            transcription_model=voice_transcription_model,
            synthesis_model=voice_synthesis_model,
            default_voice=voice_default_voice,
        )
        self.config_ops = ConfigOpsService(
            self.store, self.llm, self.router, self.training
        )
        # Use MFA setting from sys_settings (already fetched above)
        mfa_enabled = sys_settings.get("enable_mfa", self.settings.enable_mfa)
        self.auth = AuthService(
            self.store,
            self.cache,
            self.settings,
            mfa_enabled=mfa_enabled,
        )
        # Get SMTP settings from sys_settings (falls back to env vars if not in DB)
        smtp_host = sys_settings.get("smtp_host") or self.settings.smtp_host
        smtp_port = sys_settings.get("smtp_port") or self.settings.smtp_port
        smtp_user = sys_settings.get("smtp_user") or self.settings.smtp_user
        smtp_password = sys_settings.get("smtp_password") or self.settings.smtp_password
        smtp_use_tls = sys_settings.get("smtp_use_tls", self.settings.smtp_use_tls)
        email_from_address = (
            sys_settings.get("email_from_address") or self.settings.email_from_address
        )
        email_from_name = (
            sys_settings.get("email_from_name") or self.settings.email_from_name
        )
        self.email = EmailService(
            smtp_host=smtp_host,
            smtp_port=smtp_port,
            smtp_user=smtp_user,
            smtp_password=smtp_password,
            smtp_use_tls=smtp_use_tls,
            from_email=email_from_address,
            from_name=email_from_name,
            base_url=app_base_url,
        )
        # Training worker for background job processing
        poll_interval = sys_settings.get(
            "training_worker_poll_interval", self.settings.training_worker_poll_interval
        )
        self.training_worker = TrainingWorker(
            store=self.store,
            training_service=self.training,
            clusterer=self.clusterer,
            poll_interval=poll_interval,
        )
        self._local_idempotency: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        self._local_idempotency_lock = asyncio.Lock()
        self._local_rate_limits: Dict[str, Tuple[datetime, int]] = {}
        self._local_rate_limit_lock = asyncio.Lock()

        # Log successful initialization with summary
        logger.info(
            "runtime_initialized",
            model_path=resolved_base_model,
            model_backend=str(backend_mode),
            rag_mode=str(rag_mode),
            adapter_mode=str(default_adapter_mode),
            redis_enabled=self.cache is not None,
            email_configured=self.email.is_configured,
            voice_configured=self.voice.is_configured,
            mfa_enabled=mfa_enabled,
        )


runtime: Runtime | None = None


def get_runtime() -> Runtime:
    global runtime
    if runtime is None:
        runtime = Runtime()
    return runtime


def reset_runtime_for_tests() -> Runtime:
    """Reinitialize the runtime singleton for isolated test runs."""
    import asyncio

    global runtime

    # Close existing Redis connections to avoid event loop issues
    if runtime is not None and runtime.cache is not None:
        try:
            # SyncRedisCache uses a sync client internally, close it directly
            if isinstance(runtime.cache, SyncRedisCache):
                runtime.cache.client.close()
            else:
                # Async RedisCache - try to close properly
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(runtime.cache.close())
                except RuntimeError:
                    asyncio.run(runtime.cache.close())
        except Exception:
            # Ignore errors during cleanup - connection may already be closed
            pass

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


async def check_rate_limit(
    runtime: Runtime, key: str, limit: int, window_seconds: int, *, return_remaining: bool = False
) -> Union[bool, Tuple[bool, int]]:
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
