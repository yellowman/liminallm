import uuid

from liminallm.service.rag import RAGService
from liminallm.storage.models import KnowledgeChunk, KnowledgeContext, User
from tests.harness import get_test_store


def _setup_store() -> tuple[RAGService, str, str, str, str]:
    store = get_test_store()
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

    def get_context(self, context_id: str) -> KnowledgeContext | None:
        return self.contexts.get(context_id)

    def get_user(self, user_id: str) -> User | None:
        return self.users.get(user_id)

    def add_chunks(self, context_id: str, chunks: list[KnowledgeChunk]) -> list[int]:
        # Returns ids and stamps chunk.id, as PostgresStore does — late
        # interaction attaches segment vectors to the row that was just
        # written, and a double that returns None makes any test of it pass
        # by skipping the code entirely.
        bucket = self.chunks.setdefault(context_id, [])
        written: list[int] = []
        for chunk in chunks:
            if not chunk.id:
                chunk.id = self._chunk_id_seq
                self._chunk_id_seq += 1
            bucket.append(chunk)
            written.append(chunk.id)
        return written

    def search_chunks(
        self,
        context_id: str | None,
        query: str,
        query_embedding: list[float] | None,
        limit: int = 4,
        *,
        semantic: bool = False,
        allowed_paths: list[str] | None = None,
    ) -> list[KnowledgeChunk]:
        """The local path's candidate generation.

        `allowed_paths` is part of the interface rather than optional
        politeness: a store that cannot scope a context to a set of paths
        cannot serve a conversation's implicit index, and skipping the
        argument when a store does not accept it would authorize by
        omission. Applied before the cut, as the real store does.
        """
        found = list(self.chunks.get(context_id or "", []))
        if allowed_paths is not None:
            permitted = set(allowed_paths)
            found = [c for c in found if c.fs_path in permitted]
        return found[:limit]


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


def _hybrid_fixture(store, *, encoder="hybrid-encoder"):
    """Two chunks: one matches the query's words, one matches its vector.

    Vectors are orthogonal unit vectors so the semantic channel is decisive
    when it is allowed to speak and provably absent when it is not.
    """
    from liminallm.service.embeddings import EMBEDDING_DIM

    user = store.create_user(email=f"hy_{uuid.uuid4().hex[:8]}@example.com")
    ctx = store.upsert_context(user.id, f"hy-{uuid.uuid4().hex[:6]}", "fixture")

    near = [0.0] * EMBEDDING_DIM
    near[3] = 1.0
    far = [0.0] * EMBEDDING_DIM
    far[9] = 1.0
    store.add_chunks(ctx.id, [
        KnowledgeChunk(
            context_id=ctx.id, fs_path="/vector", chunk_index=0, embedding=near,
            content="marsupials of western australia and their grazing habits",
            meta={"embedding_model_id": encoder},
        ),
        KnowledgeChunk(
            context_id=ctx.id, fs_path="/keyword", chunk_index=1, embedding=far,
            content="the quokka population census for rottnest island",
            meta={"embedding_model_id": encoder},
        ),
    ])
    return user, ctx, near


def test_pgvector_retrieval_finds_the_keyword_match_without_an_encoder(store):
    """The default encoder is the hash fallback, so keywords must carry it.

    Before hybrid retrieval this path was ORDER BY embedding <-> query and
    nothing else, so the ranking of a user's own files was decided by hash
    distance — the SPEC's own definition of noise.
    """
    user, ctx, near = _hybrid_fixture(store)
    rag = RAGService(
        store, embed=lambda _text: near, embedding_model_id="hybrid-encoder"
    )

    hits = rag.retrieve(
        [ctx.id], "quokka census", limit=2, user_id=user.id, min_token_count=0
    )

    assert hits and hits[0].fs_path == "/keyword"


def test_pgvector_retrieval_returns_nothing_rather_than_noise(store):
    """No encoder and no keyword overlap is a miss, not four arbitrary chunks.

    Returning the nearest hash vectors would hand the model text it has no
    reason to trust and every reason to cite.
    """
    user, ctx, near = _hybrid_fixture(store)
    rag = RAGService(
        store, embed=lambda _text: near, embedding_model_id="hybrid-encoder"
    )

    hits = rag.retrieve(
        [ctx.id], "unrelated zzzqqq", limit=2, user_id=user.id, min_token_count=0
    )

    assert hits == []


def test_a_real_encoder_finds_what_shares_no_words(store):
    """The semantic channel earns its place: no lexical overlap, still found."""
    user, ctx, near = _hybrid_fixture(store)
    rag = RAGService(
        store,
        embed=lambda _text: near,
        embedding_model_id="hybrid-encoder",
        semantic=True,
    )

    hits = rag.retrieve(
        [ctx.id], "unrelated zzzqqq", limit=2, user_id=user.id, min_token_count=0
    )

    assert hits and hits[0].fs_path == "/vector"


def test_a_real_encoder_still_keeps_exact_terms_in_play(store):
    """bm25 stays in the score so an exact term is not lost to a near vector.

    The keyword chunk is orthogonal to the query vector, so a dense-only
    ranking puts it last. It should win anyway: it is the one that says the
    word the user typed.
    """
    user, ctx, near = _hybrid_fixture(store)
    rag = RAGService(
        store,
        embed=lambda _text: near,
        embedding_model_id="hybrid-encoder",
        semantic=True,
    )

    hits = rag.retrieve(
        [ctx.id], "quokka census", limit=2, user_id=user.id, min_token_count=0
    )

    assert [hit.fs_path for hit in hits] == ["/keyword", "/vector"]


def test_lexical_search_enforces_user_isolation(store):
    """SPEC §12.2 applies to the new channel exactly as it does to the old."""
    user, ctx, _ = _hybrid_fixture(store)
    intruder = store.create_user(email=f"ix_{uuid.uuid4().hex[:8]}@example.com")

    mine = store.search_chunks_lexical([ctx.id], "quokka", 4, user_id=user.id)
    theirs = store.search_chunks_lexical([ctx.id], "quokka", 4, user_id=intruder.id)

    assert mine and theirs == []


def test_lexical_search_survives_a_query_of_pure_punctuation(store):
    """to_tsquery would raise on an empty term list; the caller sees a miss."""
    user, ctx, _ = _hybrid_fixture(store)

    assert store.search_chunks_lexical([ctx.id], "?? -- ??", 4, user_id=user.id) == []


def test_pgvector_filters_fs_path(tmp_path):
    store = get_test_store()
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
                embedding=[1.0] + [0.0] * 63,
                chunk_index=0,
            ),
            KnowledgeChunk(
                id=None,
                context_id=ctx.id,
                fs_path="skip_me",
                content="skip",
                embedding=[0.0] + [1.0] + [0.0] * 62,
                chunk_index=1,
            ),
        ],
    )

    results = store.search_chunks_pgvector(
        [ctx.id],
        "query",
        [1.0] + [0.0] * 63,
        filters={"fs_path": "keep_me"},
        user_id=user.id,
        tenant_id="tenant_fs",
    )

    assert len(results) == 1
    assert results[0].fs_path == "keep_me"


def test_local_hybrid_does_not_hand_the_whole_answer_to_one_context():
    """Per-context lists are ranked within a context and not across them.

    Concatenating them and letting the caller truncate gives every slot to
    whichever context was listed first, however well the second matched.
    """
    store = LegacyOnlyStore()
    owner = store.add_user("tenant_multi")
    first = store.upsert_context(owner.id, "first", "ctx")
    second = store.upsert_context(owner.id, "second", "ctx")

    rag = RAGService(store, rag_mode="local_hybrid", embedding_model_id="multi")
    for ctx in (first, second):
        for index in range(6):
            store.add_chunks(ctx.id, [
                KnowledgeChunk(
                    id=None, context_id=ctx.id, fs_path=f"/{ctx.name}-{index}",
                    content=f"shared subject line number {index} with enough words to keep it",
                    embedding=[], chunk_index=index,
                    meta={"embedding_model_id": "multi"},
                )
            ])

    hits = rag.retrieve(
        [first.id, second.id], "shared subject", limit=4,
        user_id=owner.id, tenant_id="tenant_multi", min_token_count=0,
    )

    assert {hit.context_id for hit in hits} == {first.id, second.id}


def test_local_hybrid_ranks_across_contexts_by_relevance():
    """An irrelevant context must not get a fixed share of the answer.

    Interleaving alone guaranteed it half the slots. The union is scored
    again so relevance decides, and the interleave survives only as the
    tie-break. This also exercises the semantic branch, which nothing else
    reached — it carried an undefined name until ruff found it.
    """
    from liminallm.service.embeddings import EMBEDDING_DIM

    store = LegacyOnlyStore()
    owner = store.add_user("tenant_rank")
    good = store.upsert_context(owner.id, "good", "ctx")
    poor = store.upsert_context(owner.id, "poor", "ctx")

    near = [0.0] * EMBEDDING_DIM
    near[5] = 1.0
    far = [0.0] * EMBEDDING_DIM
    far[9] = 1.0

    rag = RAGService(
        store, rag_mode="local_hybrid", embedding_model_id="rank",
        embed=lambda _text: near, semantic=True,
    )
    for ctx, body, vec in (
        (good, "quokka census figures for rottnest island this year", near),
        (poor, "unrelated pottery glazing notes from the studio archive", far),
    ):
        for index in range(6):
            store.add_chunks(ctx.id, [
                KnowledgeChunk(
                    id=None, context_id=ctx.id, fs_path=f"/{ctx.name}-{index}",
                    content=body, embedding=vec, chunk_index=index,
                    meta={"embedding_model_id": "rank"},
                )
            ])

    hits = rag.retrieve(
        [good.id, poor.id], "quokka census", limit=4,
        user_id=owner.id, tenant_id="tenant_rank", min_token_count=0,
    )

    assert hits
    assert all(hit.context_id == good.id for hit in hits)


def test_a_chunk_the_store_matched_is_never_dropped_by_the_rescore(store):
    """Postgres and the BM25 tokenizer disagree, and the store wins.

    to_tsvector('simple', ...) splits "user_id" into 'user' + 'id'; bm25's
    \\w+ keeps it whole. So SQL matched the chunk on "user id" and the
    re-score gave it 0.0. With the hash encoder lexical is the only live
    channel, so the whole turn came back empty for a question the corpus
    answers — the common shape for the source files ingest_path defaults to.
    """
    user = store.create_user(email=f"tk_{uuid.uuid4().hex[:8]}@example.com")
    ctx = store.upsert_context(user.id, f"tk-{uuid.uuid4().hex[:6]}", "fixture")

    rag = RAGService(store, embedding_model_id="tok-encoder")
    rag.ingest_text(
        ctx.id,
        "def resolve_user_id(request): the tenant_id and the user_id are both "
        "derived from the authenticated jwt token and never from user input",
        source_path="/auth.py",
    )

    # The store finds it either way; retrieval must not then throw it away.
    assert store.search_chunks_lexical([ctx.id], "user id", 8, user_id=user.id)
    hits = rag.retrieve(
        [ctx.id], "user id", limit=4, user_id=user.id, min_token_count=0
    )

    assert [hit.fs_path for hit in hits] == ["/auth.py"]
