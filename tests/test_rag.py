import uuid

from liminallm.service.rag import RAGService
from liminallm.storage.memory import MemoryStore


def _setup_store() -> tuple[RAGService, str, str, str, str]:
    store = MemoryStore(fs_root=f"/tmp/liminallm-test-rag-{uuid.uuid4()}")
    user_a = store.create_user("a@example.com", tenant_id="tenant_a")
    user_b = store.create_user("b@example.com", tenant_id="tenant_b")
    ctx_a = store.upsert_context(owner_user_id=user_a.id, name="tenant_a_ctx", description="ctx")
    ctx_b = store.upsert_context(owner_user_id=user_b.id, name="tenant_b_ctx", description="ctx")
    service = RAGService(store)
    service.ingest_text(ctx_a.id, "tenant a data")
    service.ingest_text(ctx_b.id, "tenant b data")
    return service, ctx_a.id, ctx_b.id, user_a.id, user_b.id


def test_retrieve_requires_context_scope():
    service, _, _, _, _ = _setup_store()

    results = service.retrieve(None, "tenant a data")

    assert results == []


def test_retrieve_filters_by_user_and_tenant():
    service, ctx_a, ctx_b, user_a, _ = _setup_store()

    allowed = service.retrieve([ctx_a], "tenant a data", user_id=user_a, tenant_id="tenant_a")
    assert allowed
    assert all(chunk.context_id == ctx_a for chunk in allowed)

    blocked = service.retrieve([ctx_b], "tenant b data", user_id=user_a, tenant_id="tenant_a")
    assert blocked == []
