import pytest

from liminallm.service.workflow import WorkflowEngine


class RecordingRAG:
    def __init__(self):
        self.calls = []

    def retrieve(self, context_ids, query, limit=4, *, user_id=None, tenant_id=None):
        self.calls.append({
            "context_ids": context_ids,
            "query": query,
            "user_id": user_id,
            "tenant_id": tenant_id,
        })
        return []


class StubLLM:
    def generate(self, message, adapters=None, context_snippets=None, history=None):
        return {"content": message, "usage": {"tokens": 1}}


class StubRouter:
    async def route(self, policy, context_embedding, candidates, *, ctx_cluster=None, user_id=None):
        return {"adapters": [], "trace": []}


class StubStore:
    def list_artifacts(self, type_filter=None):
        return []

    def list_semantic_clusters(self, user_id=None):
        return []


@pytest.mark.asyncio
async def test_workflow_rag_tools_receive_identity():
    rag = RecordingRAG()
    engine = WorkflowEngine(StubStore(), StubLLM(), StubRouter(), rag)

    engine.invoke_tool(
        {"name": "llm.generic"},
        {"context_id": "ctx-1", "message": "hello"},
        user_id="user-1",
        tenant_id="tenant-1",
    )
    await engine.run(None, None, "question?", "ctx-2", user_id="user-1", tenant_id="tenant-1")

    assert rag.calls[0]["user_id"] == "user-1"
    assert rag.calls[0]["tenant_id"] == "tenant-1"
    assert rag.calls[1]["user_id"] == "user-1"
    assert rag.calls[1]["tenant_id"] == "tenant-1"
