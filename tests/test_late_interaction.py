"""Late interaction: several vectors per chunk, compared at query time.

The property under test is the one a pooled embedding cannot have. A chunk
that covers two subjects averages them into a single point and is the best
match for neither; keeping the parts separate and taking the best-matching
part at query time finds it.
"""

from __future__ import annotations

import uuid

from liminallm.service.embeddings import EMBEDDING_DIM
from liminallm.service.late import MIN_SEGMENT_WORDS, maxsim, segment_text
from liminallm.service.rag import RAGService
from liminallm.service.ranking import (
    LATE_WEIGHT,
    POOLED_WITH_LATE_WEIGHT,
    SEMANTIC_WEIGHT,
)

# --------------------------------------------------------------------------
# segmentation
# --------------------------------------------------------------------------


def _long(word: str, count: int = MIN_SEGMENT_WORDS + 2) -> str:
    return " ".join([word] * count)


def test_sentences_become_their_own_segments():
    text = f"{_long('alpha')}. {_long('beta')}. {_long('gamma')}."

    assert len(segment_text(text, max_segments=8)) == 3


def test_a_fragment_merges_forward_instead_of_becoming_a_vector():
    """A three-word sentence has too little in it to embed on its own."""
    text = f"Short one. {_long('beta')}."

    segments = segment_text(text, max_segments=8)

    assert len(segments) == 1
    assert "Short one" in segments[0]


def test_packing_to_the_cap_joins_segments_and_never_drops_them():
    """A dropped segment is content that becomes unretrievable."""
    text = ". ".join(_long(f"w{i}") for i in range(9)) + "."

    segments = segment_text(text, max_segments=3)

    assert len(segments) <= 3
    for i in range(9):
        assert f"w{i}" in " ".join(segments)


def test_empty_input_has_no_segments():
    assert segment_text("   ", max_segments=8) == []
    assert segment_text("anything", max_segments=0) == []


# --------------------------------------------------------------------------
# MaxSim
# --------------------------------------------------------------------------


def _unit(index: int) -> list[float]:
    vec = [0.0] * EMBEDDING_DIM
    vec[index] = 1.0
    return vec


def test_maxsim_scores_the_best_part_not_the_average():
    """The whole point: one strong part is not diluted by unrelated parts."""
    query = [_unit(1)]
    focused = [_unit(1)]
    mixed = [_unit(1), _unit(2), _unit(3)]

    assert maxsim(query, mixed) == maxsim(query, focused) == 1.0


def test_each_query_part_is_scored_and_the_parts_are_summed():
    query = [_unit(1), _unit(2)]

    assert maxsim(query, [_unit(1), _unit(2)]) == 2.0
    assert maxsim(query, [_unit(1)]) == 1.0


def test_an_unaddressed_query_part_contributes_nothing_rather_than_subtracting():
    """A negative cosine is an absence of match, not evidence against."""
    opposite = [-v for v in _unit(1)]

    assert maxsim([_unit(1)], [opposite]) == 0.0


def test_maxsim_of_nothing_is_zero():
    assert maxsim([], [_unit(1)]) == 0.0
    assert maxsim([_unit(1)], []) == 0.0


# --------------------------------------------------------------------------
# the retrieval channel
# --------------------------------------------------------------------------


def test_the_pooled_vector_steps_back_when_late_interaction_speaks():
    """Two readings of one signal must not both vote at full strength."""
    assert LATE_WEIGHT >= SEMANTIC_WEIGHT > POOLED_WITH_LATE_WEIGHT


# Subjects on their own axes. "marsupial" sits close to "quokka" without
# being it, which is what makes the decoy chunk a near miss rather than an
# unrelated one — the case where averaging actually costs you the answer.
_NEAR = 0.9
_SUBJECTS = {
    "quokka": [(1, 1.0)],
    "zzzq": [(1, 1.0)],  # the query's word, deliberately in no document
    "diesel": [(2, 1.0)],
    "marsupial": [(1, _NEAR), (4, (1 - _NEAR ** 2) ** 0.5)],
}


def _subject_encoder(text: str) -> list[float]:
    """A vector per subject named, averaged — as a real encoder would pool."""
    lowered = text.lower()
    hits = [axes for word, axes in _SUBJECTS.items() if word in lowered]
    vec = [0.0] * EMBEDDING_DIM
    if not hits:
        return _unit(0)
    for axes in hits:
        for index, value in axes:
            vec[index] += value / len(hits)
    return vec


def _near_miss_corpus(store):
    """A chunk that says the answer among other things, and a near miss.

    ``/mixed`` answers the query but also discusses something else, so its
    pooled vector is the average of the two and sits 0.71 from the query.
    ``/decoy`` is about a neighbouring subject only, and its pooled vector
    sits 0.9 from the query — closer. Pooled similarity picks the wrong one.
    """
    user = store.create_user(email=f"li_{uuid.uuid4().hex[:8]}@example.com")
    ctx = store.upsert_context(user.id, f"li-{uuid.uuid4().hex[:6]}", "fixture")

    writer = RAGService(
        store,
        embed=_subject_encoder,
        embedding_model_id="late-encoder",
        semantic=True,
        late_interaction=True,
        late_segments=8,
    )
    writer.ingest_text(
        ctx.id, f"{_long('quokka')}. {_long('diesel')}.", source_path="/mixed"
    )
    writer.ingest_text(
        ctx.id, f"{_long('marsupial')}. {_long('marsupial')}.", source_path="/decoy"
    )
    return user, ctx


def _retriever(store, *, late: bool):
    return RAGService(
        store,
        embed=_subject_encoder,
        embedding_model_id="late-encoder",
        semantic=True,
        late_interaction=late,
        late_segments=8,
    )


def test_pooled_similarity_alone_picks_the_near_miss(store):
    """The baseline this feature exists to fix — asserted, not assumed.

    The query word appears in no document, so BM25 is silent and only the
    vector channels decide. Averaged over its whole chunk, the right answer
    loses to a chunk that is merely nearby.
    """
    user, ctx = _near_miss_corpus(store)

    hits = _retriever(store, late=False).retrieve(
        [ctx.id], "zzzq", limit=2, user_id=user.id, min_token_count=0
    )

    assert [hit.fs_path for hit in hits] == ["/decoy", "/mixed"]


def test_late_interaction_finds_the_chunk_on_its_best_part(store):
    """Same corpus, same encoder, same query — segments kept separate.

    ``/mixed`` owns a segment that is exactly the query, so MaxSim scores it
    1.0 against the decoy's 0.9, and the answer comes back first.
    """
    user, ctx = _near_miss_corpus(store)

    hits = _retriever(store, late=True).retrieve(
        [ctx.id], "zzzq", limit=2, user_id=user.id, min_token_count=0
    )

    assert hits[0].fs_path == "/mixed"


def test_a_chunk_with_no_segments_is_not_penalised_only_unranked(store):
    """Coverage grows with ingestion, so a mixed corpus must still work.

    Content ingested before late interaction was turned on has no segment
    vectors. The late channel simply has nothing to say about it — silence,
    which the fusion already distinguishes from a bad score.
    """
    def encoder(_text: str) -> list[float]:
        return _unit(1)

    user = store.create_user(email=f"lc_{uuid.uuid4().hex[:8]}@example.com")
    ctx = store.upsert_context(user.id, f"lc-{uuid.uuid4().hex[:6]}", "fixture")

    before = RAGService(
        store, embed=encoder, embedding_model_id="late-encoder", semantic=True
    )
    before.ingest_text(ctx.id, f"{_long('legacy')}.", source_path="/legacy")

    after = RAGService(
        store,
        embed=encoder,
        embedding_model_id="late-encoder",
        semantic=True,
        late_interaction=True,
        late_segments=8,
    )

    hits = after.retrieve(
        [ctx.id], "legacy", limit=4, user_id=user.id, min_token_count=0
    )

    assert [hit.fs_path for hit in hits] == ["/legacy"]


def test_reindexing_a_chunk_replaces_its_segments(store):
    """A chunk must never be scored against two generations of itself."""
    from liminallm.storage.models import KnowledgeChunk

    user = store.create_user(email=f"lr_{uuid.uuid4().hex[:8]}@example.com")
    ctx = store.upsert_context(user.id, f"lr-{uuid.uuid4().hex[:6]}", "fixture")
    chunk = KnowledgeChunk(
        context_id=ctx.id, fs_path="/f", chunk_index=0,
        content="body", embedding=_unit(1),
    )
    [chunk_id] = store.add_chunks(ctx.id, [chunk])

    store.add_chunk_vectors(chunk_id, [("a", _unit(1)), ("b", _unit(2))])
    store.add_chunk_vectors(chunk_id, [("c", _unit(3))])

    [(_chunk, vectors)] = store.chunks_with_vectors([chunk_id])
    assert vectors == [_unit(3)]


def test_add_chunks_hands_back_the_ids_it_wrote(store):
    """Late interaction attaches vectors to the row it just inserted."""
    assert store.add_chunks("00000000-0000-0000-0000-000000000000", []) == []


def test_late_interaction_needs_a_real_encoder(store):
    """MaxSim over hash vectors is noise with extra steps."""
    rag = RAGService(store, late_interaction=True, late_segments=8)

    assert rag.late_interaction is False
