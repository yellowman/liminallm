from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

from liminallm.config import get_settings
from liminallm.service.config_ops import ConfigOpsService
from liminallm.service.auth import AuthService
from liminallm.service.clustering import SemanticClusterer
from liminallm.service.llm import LLMService
from liminallm.service.rag import RAGService
from liminallm.service.router import RouterEngine
from liminallm.service.training import TrainingService
from liminallm.service.voice import VoiceService
from liminallm.service.workflow import WorkflowEngine
from liminallm.storage.memory import MemoryStore
from liminallm.storage.postgres import PostgresStore
from liminallm.storage.redis_cache import RedisCache

from liminallm.logging import get_logger

logger = get_logger(__name__)


class Runtime:
    """Holds singleton service instances for the FastAPI app."""

    def __init__(self):
        self.settings = get_settings()
        self.store = MemoryStore(fs_root=self.settings.shared_fs_root) if self.settings.use_memory_store else PostgresStore(
            self.settings.database_url, fs_root=self.settings.shared_fs_root
        )
        self.cache = None
        if self.settings.redis_url:
            try:
                self.cache = RedisCache(self.settings.redis_url)
            except Exception as exc:
                self.cache = None
                logger.warning(
                    "redis_cache_init_failed", redis_url=self.settings.redis_url, error=str(exc)
                )
        self.router = RouterEngine(cache=self.cache)
        runtime_config = {}
        db_backend_mode = None
        if hasattr(self.store, "get_runtime_config"):
            runtime_config = self.store.get_runtime_config() or {}
            db_backend_mode = runtime_config.get("model_backend")
        resolved_base_model = runtime_config.get("model_path") or self.settings.model_path
        backend_mode = db_backend_mode or self.settings.model_backend
        adapter_configs = {
            "openai": {
                "api_key": self.settings.adapter_openai_api_key,
                "base_url": self.settings.adapter_openai_base_url,
                "adapter_server_model": self.settings.adapter_server_model,
            }
        }
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
        )
        self.training = TrainingService(
            self.store, self.settings.shared_fs_root, runtime_base_model=resolved_base_model
        )
        self.clusterer = SemanticClusterer(self.store, self.llm, self.training)
        self.workflow = WorkflowEngine(self.store, self.llm, self.router, self.rag, cache=self.cache)
        self.voice = VoiceService(self.settings.shared_fs_root)
        self.config_ops = ConfigOpsService(self.store, self.llm, self.router, self.training)
        self.auth = AuthService(
            self.store,
            self.cache,
            self.settings,
            mfa_enabled=self.settings.enable_mfa,
        )
        self._local_idempotency: Dict[Tuple[str, str, str], Dict[str, Any]] = {}


runtime = Runtime()


def get_runtime() -> Runtime:
    return runtime


IDEMPOTENCY_TTL_SECONDS = 60 * 60 * 24


async def _get_cached_idempotency_record(runtime: Runtime, route: str, user_id: str, key: str) -> Optional[dict]:
    now = datetime.utcnow()
    if runtime.cache:
        return await runtime.cache.get_idempotency_record(route, user_id, key)
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
        await runtime.cache.set_idempotency_record(route, user_id, key, record, ttl_seconds=ttl_seconds)
        return
    runtime._local_idempotency[(route, user_id, key)] = {**record, "expires_at": expires_at}
