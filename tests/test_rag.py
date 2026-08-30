import uuid

import pytest

from liminallm.service.embeddings import EMBEDDING_DIM
from liminallm.service.rag import RAGService
from liminallm.storage.models import KnowledgeChunk
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
    distance - the SPEC's own definition of noise.
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


@pytest.mark.parametrize("absent", ["", None], ids=["empty", "none"])
def test_a_retrieval_channel_with_no_user_refuses_rather_than_widens(store, absent):
    """Both chunk channels refuse an absent principal rather than widening.

    `user_id` is keyword-only and annotated as required, so an absent one can
    only arrive from a caller that bypassed the annotation - which is the case
    the check exists for. The failure mode is not an error: `_chunk_scope`
    builds a WHERE clause with no owner term, so the query runs and returns
    every user's chunks in the named contexts.

    Measured before this existed: removing the check from
    `search_chunks_pgvector` left the whole fast lane green. The positive
    control is in the same test because a refusal that returns nothing is
    indistinguishable from a query that would have found nothing anyway.
    `late_candidate_ids` carries the same check and is covered beside the
    corpus that can exercise it, in `test_late_interaction.py`.
    """
    user, ctx, near = _hybrid_fixture(store)

    assert store.search_chunks_lexical([ctx.id], "quokka", 4, user_id=user.id), (
        "the fixture matches nothing, so refusing it would prove nothing"
    )
    assert store.search_chunks_pgvector([ctx.id], "quokka", near, 4, user_id=user.id)

    assert store.search_chunks_lexical([ctx.id], "quokka", 4, user_id=absent) == []
    assert store.search_chunks_pgvector(
        [ctx.id], "quokka", near, 4, user_id=absent
    ) == []


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


def test_retrieval_ranks_across_contexts_by_relevance(store):
    """An irrelevant context must not get a fixed share of the answer.

    Ported from the deleted second engine: relevance decides across contexts,
    so a context that matches neither the words nor the vector contributes
    nothing however early it was listed.
    """
    user = store.create_user(email=f"rk_{uuid.uuid4().hex[:8]}@example.com")
    poor = store.upsert_context(user.id, f"rk-poor-{uuid.uuid4().hex[:6]}", "ctx")
    good = store.upsert_context(user.id, f"rk-good-{uuid.uuid4().hex[:6]}", "ctx")

    near = [0.0] * EMBEDDING_DIM
    near[5] = 1.0
    far = [0.0] * EMBEDDING_DIM
    far[9] = 1.0
    for ctx, body, vec in (
        (good, "quokka census figures for rottnest island this year", near),
        (poor, "unrelated pottery glazing notes from the studio archive", far),
    ):
        store.add_chunks(ctx.id, [
            KnowledgeChunk(
                context_id=ctx.id, fs_path=f"/{ctx.id}-{index}", chunk_index=index,
                embedding=vec, content=body,
                meta={"embedding_model_id": "rank"},
            )
            for index in range(6)
        ])

    rag = RAGService(
        store, embedding_model_id="rank", embed=lambda _text: near, semantic=True,
    )
    # The irrelevant context listed first, which is exactly the case that
    # used to hand it the answer.
    hits = rag.retrieve(
        [poor.id, good.id], "quokka census", limit=4,
        user_id=user.id, min_token_count=0,
    )

    assert hits
    assert all(hit.context_id == good.id for hit in hits)


def test_a_chunk_the_store_matched_is_never_dropped_by_the_rescore(store):
    """Postgres and the BM25 tokenizer disagree, and the store wins.

    to_tsvector('simple', ...) splits "user_id" into 'user' + 'id'; bm25's
    \\w+ keeps it whole. So SQL matched the chunk on "user id" and the
    re-score gave it 0.0. With the hash encoder lexical is the only live
    channel, so the whole turn came back empty for a question the corpus
    answers - the common shape for the source files ingest_path defaults to.
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


def test_bm25_orders_the_lexical_pool_not_arrival_order(store):
    """The SQL's ts_rank is a recall filter; BM25 decides the lexical order.

    Pinned at the fusion seam with a pool whose arrival order disagrees with
    its BM25 order, because a mutation that fused the pool as it arrived
    passed every retrieval test - the two scorers agree too often on small
    fixtures for an end-to-end red to catch the difference deterministically.
    """
    rag = RAGService(store)
    off_topic = KnowledgeChunk(
        id=1, context_id="ctx", fs_path="/off", chunk_index=0, embedding=[],
        content="unrelated pottery glazing notes from the studio archive",
    )
    on_topic = KnowledgeChunk(
        id=2, context_id="ctx", fs_path="/on", chunk_index=1, embedding=[],
        content="quokka census figures and quokka census methods",
    )

    fused = rag._fuse("quokka census", [off_topic, on_topic], dense=[])

    assert [chunk.fs_path for chunk in fused] == ["/on", "/off"], (
        "the lexical pool kept its arrival order; BM25 never spoke"
    )
