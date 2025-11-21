from __future__ import annotations

from typing import Dict, List, Optional

from liminallm.service.llm import LLMService
from liminallm.service.rag import RAGService
from liminallm.service.router import RouterEngine
from liminallm.storage.memory import MemoryStore
from liminallm.storage.postgres import PostgresStore


class WorkflowEngine:
    """Executes workflow.chat graphs using a small tool registry."""

    def __init__(self, store: PostgresStore | MemoryStore, llm: LLMService, router: RouterEngine, rag: RAGService) -> None:
        self.store = store
        self.llm = llm
        self.router = router
        self.rag = rag

    def run(self, workflow_id: Optional[str], conversation_id: Optional[str], user_message: str, context_id: Optional[str]) -> dict:
        workflow_schema = None
        if workflow_id:
            workflow_schema = self.store.get_latest_workflow(workflow_id) if hasattr(self.store, "get_latest_workflow") else None
        if not workflow_schema:
            workflow_schema = self._default_workflow()

        adapters = self._select_adapters(context_id)
        ctx_chunks = self.rag.retrieve(context_id)
        context_snippets = [c.text for c in ctx_chunks]
        history = []
        if conversation_id and hasattr(self.store, "list_messages"):
            history = self.store.list_messages(conversation_id)  # type: ignore[attr-defined]
        llm_resp = self.llm.generate(user_message, adapters=adapters, context_snippets=context_snippets, history=history)
        return {
            "content": llm_resp["content"],
            "usage": llm_resp["usage"],
            "adapters": adapters,
            "context_snippets": context_snippets,
        }

    def _default_workflow(self) -> dict:
        return {
            "kind": "workflow.chat",
            "entrypoint": "plain_chat",
            "nodes": [{"id": "plain_chat", "tool": "llm.generic"}],
        }

    def _select_adapters(self, context_id: Optional[str]) -> List[str]:
        adapter_artifacts = [a for a in self.store.list_artifacts(type_filter="adapter")]  # type: ignore[arg-type]
        policy = None
        for art in self.store.list_artifacts(type_filter="policy"):  # type: ignore[arg-type]
            if art.name == "default_routing":
                policy = art.schema
                break
        context_embedding = None
        actions = self.router.route(policy or {}, context_embedding, [a.schema for a in adapter_artifacts])
        activated: List[str] = []
        for action in actions:
            if action.get("type") == "activate_adapter":
                if adapter_artifacts:
                    activated.append(adapter_artifacts[0].id if action.get("adapter_id") == "closest" else action.get("adapter_id", ""))
        return [a for a in activated if a]
