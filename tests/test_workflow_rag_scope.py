import pytest
from types import SimpleNamespace

from liminallm.service.workflow import WorkflowEngine


class RecordingRAG:
    def __init__(self):
        self.calls = []

    def retrieve(
        self,
        context_ids,
        query,
        limit=4,
        *,
        user_id=None,
        tenant_id=None,
        path_scope=None,
    ):
        self.calls.append(
            {
                "context_ids": context_ids,
                "query": query,
                "user_id": user_id,
                "tenant_id": tenant_id,
                "path_scope": path_scope,
            }
        )
        return []


class StubLLM:
    # A real LLMService always answers this; the engine asks before offering
    # tools. Without it the agent path errors and retries into a failure.
    supports_tools = False

    def generate(self, message, adapters=None, context_snippets=None, history=None):
        return {"content": message, "usage": {"tokens": 1}}


class StubRouter:
    async def route(
        self, policy, context_embedding, candidates, *, ctx_cluster=None, user_id=None
    ):
        return {"adapters": [], "trace": []}


class StubStore:
    def list_artifacts(self, type_filter=None, **kwargs):
        return []

    def list_semantic_clusters(self, user_id=None):
        return []

    def list_contexts(self, owner_user_id=None, **kwargs):
        return [self._context("ctx-1", owner_user_id)]

    @staticmethod
    def _context(ctx_id, owner_user_id):
        return SimpleNamespace(id=ctx_id, owner_user_id=owner_user_id, meta=None)

    def get_contexts_for_scope(self, owner_user_id, context_ids):
        """The named contexts this owner has, mirroring the real store.

        Authorization asks about particular ids rather than reading a page,
        so the double has to answer that question too — a stub that only
        knows the old method silently authorizes nothing, and the test then
        fails somewhere else entirely.
        """
        return [
            self._context(ctx_id, owner_user_id)
            for ctx_id in context_ids or []
            if ctx_id == "ctx-1"
        ]

    def get_user(self, user_id):
        return SimpleNamespace(id=user_id, tenant_id="tenant-1")


@pytest.mark.asyncio
async def test_workflow_rag_tools_receive_identity():
    rag = RecordingRAG()
    engine = WorkflowEngine(StubStore(), StubLLM(), StubRouter(), rag)

    await engine.invoke_tool(
        {"name": "llm.generic"},
        {"context_id": "ctx-1", "message": "hello"},
        user_id="user-1",
        tenant_id="tenant-1",
    )
    await engine.run(
        None, None, "question?", "ctx-2", user_id="user-1", tenant_id="tenant-1"
    )

    assert rag.calls[0]["user_id"] == "user-1"
    assert rag.calls[0]["tenant_id"] == "tenant-1"
    assert rag.calls[1]["user_id"] == "user-1"
    assert rag.calls[1]["tenant_id"] == "tenant-1"
