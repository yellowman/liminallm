from __future__ import annotations

from liminallm.config import get_settings
from liminallm.service.auth import AuthService
from liminallm.service.llm import LLMService
from liminallm.service.rag import RAGService
from liminallm.service.router import RouterEngine
from liminallm.service.training import TrainingService
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
        backend_mode = self.settings.model_backend or self.settings.llm_mode
        self.llm = LLMService(
            base_model=self.settings.model_path,
            backend_mode=backend_mode,
            api_key=self.settings.openai_api_key,
            base_url=self.settings.openai_base_url,
            adapter_server_model=self.settings.adapter_server_model,
            fs_root=self.settings.shared_fs_root,
        )
        self.rag = RAGService(self.store)
        self.workflow = WorkflowEngine(self.store, self.llm, self.router, self.rag)
        self.auth = AuthService(self.store, self.cache)
        self.training = TrainingService(self.store, self.settings.shared_fs_root)


runtime = Runtime()


def get_runtime() -> Runtime:
    return runtime
