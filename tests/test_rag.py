import uuid

from liminallm.service.rag import RAGService
from liminallm.storage.memory import MemoryStore
from liminallm.storage.models import KnowledgeChunk, KnowledgeContext, User


def _setup_store() -> tuple[RAGService, str, str, str, str]:
    store = MemoryStore(fs_root=f"/tmp/liminallm-test-rag-{uuid.uuid4()}")
    user_a = store.create_user("a@example.com", tenant_id="tenant_a")
    user_b = store.create_user("b@example.com", tenant_id="tenant_b")
    ctx_a = store.upsert_context(
        owner_user_id=user_a.id, name="tenant_a_ctx", description="ctx"
    )
    ctx_b = store.upsert_context(
        owner_user_id=user_b.id, name="tenant_b_ctx", description="ctx"
    )
    service = RAGService(store)
    # Use longer content to ensure chunks have >= 10 tokens (min_token_count filter)
    service.ingest_text(ctx_a.id, "This is tenant A data with enough content to pass the minimum token count filter for retrieval")
    service.ingest_text(ctx_b.id, "This is tenant B data with enough content to pass the minimum token count filter for retrieval")
    return service, ctx_a.id, ctx_b.id, user_a.id, user_b.id


def test_retrieve_requires_context_scope():
    service, _, _, _, _ = _setup_store()

    results = service.retrieve(None, "tenant a data")

    assert results == []


def test_retrieve_filters_by_user_and_tenant():
    service, ctx_a, ctx_b, user_a, _ = _setup_store()

    allowed = service.retrieve(
        [ctx_a], "tenant a data", user_id=user_a, tenant_id="tenant_a"
    )
    assert allowed
    assert all(chunk.context_id == ctx_a for chunk in allowed)

    blocked = service.retrieve(
        [ctx_b], "tenant b data", user_id=user_a, tenant_id="tenant_a"
    )
    assert blocked == []


def test_pgvector_retrieve_requires_auth_scope():
    service, _, ctx_b, _, _ = _setup_store()

    # Without a user context, pgvector retrieval should not surface chunks from any context.
    blocked = service.retrieve([ctx_b], "tenant b data", user_id=None, tenant_id=None)

    assert blocked == []


class LegacyOnlyStore:
    def __init__(self):
        self.contexts = {}
        self.users = {}
        self.chunks = {}
        self._chunk_id_seq = 1

    def add_user(self, tenant_id: str) -> User:
        user = User(
            id=str(uuid.uuid4()),
            email=f"user-{tenant_id}@example.com",
            tenant_id=tenant_id,
        )
        self.users[user.id] = user
        return user

    def upsert_context(
        self, owner_user_id: str, name: str, description: str
    ) -> KnowledgeContext:
        ctx = KnowledgeContext(
            id=str(uuid.uuid4()),
            owner_user_id=owner_user_id,
            name=name,
            description=description,
        )
        self.contexts[ctx.id] = ctx
        return ctx

    def add_chunks(self, context_id: str, chunks: list[KnowledgeChunk]) -> None:
        bucket = self.chunks.setdefault(context_id, [])
        for chunk in chunks:
            if not chunk.id:
                chunk.id = self._chunk_id_seq
                self._chunk_id_seq += 1
            bucket.append(chunk)

    def search_chunks_legacy(
        self,
        context_id: str | None,
        query: str,
        query_embedding: list[float] | None,
        limit: int = 4,
    ) -> list[KnowledgeChunk]:
        return list(self.chunks.get(context_id or "", []))[:limit]


def test_local_hybrid_without_pgvector():
    store = LegacyOnlyStore()
    owner = store.add_user("tenant_legacy")
    ctx = store.upsert_context(owner.id, "legacy", "local hybrid")

    rag = RAGService(
        store, rag_mode="local_hybrid", embedding_model_id="legacy-embedding"
    )
    # Use longer content to ensure chunks have >= 10 tokens (min_token_count filter)
    rag.ingest_text(ctx.id, "This is legacy search path content with enough tokens to pass the minimum token count filter")
    existing_chunks = store.chunks.get(ctx.id, [])
    store.add_chunks(
        ctx.id,
        [
            KnowledgeChunk(
                id=None,
                context_id=ctx.id,
                fs_path="inline",
                content="This is other model content with enough tokens to pass the minimum token count filter",
                embedding=[],
                chunk_index=len(existing_chunks),
                meta={"embedding_model_id": "other"},
            )
        ],
    )

    allowed = rag.retrieve(
        [ctx.id], "legacy", user_id=owner.id, tenant_id="tenant_legacy"
    )
    assert allowed
    assert all(
        (chunk.meta or {}).get("embedding_model_id") == "legacy-embedding"
        for chunk in allowed
    )

    blocked_user = store.add_user("other")
    denied = rag.retrieve(
        [ctx.id], "legacy", user_id=blocked_user.id, tenant_id="other"
    )
    assert denied == []


def test_memory_store_pgvector_filters_fs_path(tmp_path):
    store = MemoryStore(fs_root=tmp_path)
    user = store.create_user("fs@example.com", tenant_id="tenant_fs")
    ctx = store.upsert_context(owner_user_id=user.id, name="fs ctx", description="desc")

    store.add_chunks(
        ctx.id,
        [
            KnowledgeChunk(
                id=None,
                context_id=ctx.id,
                fs_path="keep_me",
                content="keep",
                embedding=[1.0, 0.0],
                chunk_index=0,
            ),
            KnowledgeChunk(
                id=None,
                context_id=ctx.id,
                fs_path="skip_me",
                content="skip",
                embedding=[0.0, 1.0],
                chunk_index=1,
            ),
        ],
    )

    results = store.search_chunks_pgvector(
        [ctx.id],
        "query",
        [1.0, 0.0],
        filters={"fs_path": "keep_me"},
        user_id=user.id,
        tenant_id="tenant_fs",
    )

    assert len(results) == 1
    assert results[0].fs_path == "keep_me"
