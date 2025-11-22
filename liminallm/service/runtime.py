from __future__ import annotations

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
            except Exception:
                self.cache = None
        self.router = RouterEngine()
        db_backend_mode = None
        if hasattr(self.store, "get_runtime_config"):
            runtime_config = self.store.get_runtime_config() or {}
            db_backend_mode = runtime_config.get("model_backend")
        backend_mode = db_backend_mode or self.settings.model_backend
        adapter_configs = {
            "openai": {
                "api_key": self.settings.adapter_openai_api_key,
                "base_url": self.settings.adapter_openai_base_url,
                "adapter_server_model": self.settings.adapter_server_model,
            }
        }
        self.llm = LLMService(
            base_model=self.settings.model_path,
            backend_mode=backend_mode,
            adapter_configs=adapter_configs,
            api_key=self.settings.adapter_openai_api_key,
            base_url=self.settings.adapter_openai_base_url,
            adapter_server_model=self.settings.adapter_server_model,
            fs_root=self.settings.shared_fs_root,
        )
        self.rag = RAGService(self.store)
        self.training = TrainingService(self.store, self.settings.shared_fs_root)
        self.clusterer = SemanticClusterer(self.store, self.llm, self.training)
        self.workflow = WorkflowEngine(self.store, self.llm, self.router, self.rag)
        self.voice = VoiceService(self.settings.shared_fs_root)
        self.config_ops = ConfigOpsService(self.store, self.llm, self.router, self.training)
        self.auth = AuthService(self.store, self.cache, mfa_enabled=self.settings.enable_mfa)


runtime = Runtime()


def get_runtime() -> Runtime:
    return runtime
