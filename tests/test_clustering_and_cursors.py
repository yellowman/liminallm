"""Keyset cursors, and the clustering the router's skills emerge from.

Both were around 40% covered. The clustering tests here stay on the parts that
are pure functions of their input - k-means, reservoir sampling, parsing a
model's label reply - because those decide which adapters exist, and a model
reply is not a format anyone controls.
"""

from __future__ import annotations

import random
from datetime import datetime, timezone

import pytest

from liminallm.service.clustering import SemanticClusterer
from liminallm.service.embeddings import EMBEDDING_DIM
from liminallm.storage.cursors import (
    decode_artifact_cursor,
    decode_index_cursor,
    decode_time_id_cursor,
    encode_artifact_cursor,
    encode_index_cursor,
    encode_time_id_cursor,
)
from liminallm.storage.models import PreferenceEvent

# ---------------------------------------------------------------------------
# Cursors
# ---------------------------------------------------------------------------


def test_a_time_cursor_round_trips():
    when = datetime(2026, 3, 4, 5, 6, 7, 890123, tzinfo=timezone.utc)
    ts, ident = decode_time_id_cursor(encode_time_id_cursor(when, "abc-123"))
    assert ident == "abc-123"
    assert ts == when.replace(tzinfo=None)


def test_a_naive_timestamp_is_read_as_utc():
    """Naive stamps are UTC by convention here; assuming local time would move
    the page boundary by hours."""
    naive = datetime(2026, 3, 4, 5, 6, 7)
    aware = naive.replace(tzinfo=timezone.utc)
    assert encode_time_id_cursor(naive, "x") == encode_time_id_cursor(aware, "x")


def test_a_non_utc_timestamp_is_converted_not_truncated():
    from datetime import timedelta

    plus_two = timezone(timedelta(hours=2))
    local = datetime(2026, 3, 4, 12, 0, 0, tzinfo=plus_two)
    ts, _ = decode_time_id_cursor(encode_time_id_cursor(local, "x"))
    assert ts == datetime(2026, 3, 4, 10, 0, 0)


def test_a_decoded_cursor_is_naive():
    """The store compares it against timestamptz with the session pinned to
    UTC (see _configure_connection), so this must not silently become aware."""
    ts, _ = decode_time_id_cursor(encode_time_id_cursor(datetime.now(timezone.utc), "x"))
    assert ts.tzinfo is None


def test_an_identifier_containing_a_pipe_survives():
    """Only the first separator is significant; ids are opaque to the cursor."""
    _, ident = decode_time_id_cursor(
        encode_time_id_cursor(datetime.now(timezone.utc), "a|b|c")
    )
    assert ident == "a|b|c"


def test_an_index_cursor_round_trips():
    assert decode_index_cursor(encode_index_cursor(42, "chunk-9")) == (42, "chunk-9")


def test_the_artifact_cursor_is_the_time_cursor():
    when = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert encode_artifact_cursor(when, "a1") == encode_time_id_cursor(when, "a1")
    assert decode_artifact_cursor(encode_artifact_cursor(when, "a1")) == (
        decode_time_id_cursor(encode_time_id_cursor(when, "a1"))
    )


@pytest.mark.parametrize(
    "bad", ["", "no-separator", "notadate|id", "2026-01-01T00:00:00+00:00"]
)
def test_a_malformed_time_cursor_raises_rather_than_paging_wrongly(bad):
    """The store logs and falls back to page one; silently decoding garbage
    into a boundary would skip or repeat rows."""
    with pytest.raises(ValueError):
        decode_time_id_cursor(bad)


@pytest.mark.parametrize("bad", ["", "no-separator", "abc|id"])
def test_a_malformed_index_cursor_raises(bad):
    with pytest.raises(ValueError):
        decode_index_cursor(bad)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


@pytest.fixture
def clusterer():
    return SemanticClusterer.__new__(SemanticClusterer)


def _vec(*leading):
    """A unit-ish embedding of the right width, distinguished by its head."""
    v = [0.0] * EMBEDDING_DIM
    for i, x in enumerate(leading):
        v[i] = float(x)
    return v


def test_kmeans_separates_two_obvious_groups(clusterer):
    far_apart = [_vec(10, 0), _vec(10, 0.1), _vec(-10, 0), _vec(-10, 0.1)]
    centroids, assignments = clusterer._mini_batch_kmeans(far_apart, k=2, iters=20)

    assert len(centroids) == 2
    assert assignments[0] == assignments[1]
    assert assignments[2] == assignments[3]
    assert assignments[0] != assignments[2]


def test_kmeans_returns_a_centroid_of_the_right_width(clusterer):
    centroids, _ = clusterer._mini_batch_kmeans([_vec(1), _vec(2)], k=1)
    assert len(centroids[0]) == EMBEDDING_DIM


def test_kmeans_cannot_ask_for_more_clusters_than_points(clusterer):
    centroids, assignments = clusterer._mini_batch_kmeans([_vec(1), _vec(2)], k=50)
    assert len(centroids) <= 2
    assert all(0 <= a < len(centroids) for a in assignments)


@pytest.mark.parametrize("embeddings, k", [([], 3), ([_vec(1)], 0), ([_vec(1)], -1)])
def test_kmeans_on_nothing_returns_nothing(clusterer, embeddings, k):
    assert clusterer._mini_batch_kmeans(embeddings, k=k) == ([], [])


def test_kmeans_every_point_is_assigned(clusterer):
    points = [_vec(i, i % 3) for i in range(20)]
    centroids, assignments = clusterer._mini_batch_kmeans(points, k=4, iters=15)
    assert len(assignments) == len(points)
    assert -1 not in assignments


def test_a_malformed_warm_start_centroid_does_not_stop_clustering(clusterer):
    """Seeds come off stored SemanticCluster rows, which an earlier encoder or
    a hand edit can leave the wrong width."""
    centroids, assignments = clusterer._mini_batch_kmeans(
        [_vec(1), _vec(-1)],
        k=2,
        initial_centroids=[[0.1, 0.2], _vec(1)],  # first is the wrong width
        iters=5,
    )
    assert len(centroids) == 2
    assert all(len(c) == EMBEDDING_DIM for c in centroids)


def test_reservoir_sampling_keeps_exactly_k(clusterer):
    events = list(range(1000))
    assert len(clusterer._reservoir_sample(events, 50)) == 50


def test_reservoir_sampling_of_a_short_run_keeps_everything(clusterer):
    events = list(range(7))
    assert clusterer._reservoir_sample(events, 50) == events


def test_reservoir_sampling_is_uniform_enough_to_be_a_sample(clusterer):
    """Not a statistics test - just that it does not always pick the head,
    which is what a broken reservoir degenerates to."""
    random.seed(1234)
    tail_hits = sum(
        1 for _ in range(200) if any(e > 900 for e in clusterer._reservoir_sample(range(1000), 10))
    )
    assert tail_hits > 100


# ---------------------------------------------------------------------------
# Reading the model's label reply
# ---------------------------------------------------------------------------


def test_a_clean_json_label_is_used(clusterer):
    label, desc = clusterer._parse_label_response(
        '{"label": "Debugging", "description": "Fixing broken code"}'
    )
    assert label == "Debugging"
    assert desc == "Fixing broken code"


def test_a_label_alone_fills_the_description(clusterer):
    assert clusterer._parse_label_response('{"label": "Poetry"}') == ("Poetry", "Poetry")


def test_a_description_alone_yields_a_label(clusterer):
    label, desc = clusterer._parse_label_response('{"description": "Writing verse"}')
    assert label == "Writing verse"
    assert desc == "Writing verse"


def test_prose_instead_of_json_still_yields_a_label(clusterer):
    """Small models are the point of this app, and they answer in prose."""
    label, desc = clusterer._parse_label_response("Debugging\nFixing broken code")
    assert label == "Debugging"
    assert desc == "Fixing broken code"


def test_one_line_of_prose_is_both(clusterer):
    assert clusterer._parse_label_response("Debugging") == ("Debugging", "Debugging")


@pytest.mark.parametrize("reply", ["", "   \n  \n ", None])
def test_an_empty_reply_is_no_label_rather_than_a_blank_one(clusterer, reply):
    assert clusterer._parse_label_response(reply) is None


def test_a_runaway_label_is_cut(clusterer):
    """The label lands in a UI list and a prompt; a model returning an essay
    must not widen either."""
    long_label = clusterer._parse_label_response("x" * 500)[0]
    assert len(long_label) <= 64


def test_a_json_array_falls_back_to_prose(clusterer):
    """Valid JSON of the wrong shape is not an error, just not a label."""
    label, _ = clusterer._parse_label_response('["Debugging", "Poetry"]')
    assert label.startswith("[")


# ---------------------------------------------------------------------------
# Skill prompt instructions
# ---------------------------------------------------------------------------


class _Cluster:
    def __init__(self, label, description):
        self.label = label
        self.description = description


def _event(text):
    return PreferenceEvent(
        id="e", user_id="u", conversation_id="c", message_id="m",
        feedback="positive", context_text=text,
    )


def test_the_skill_prompt_names_the_skill(clusterer):
    out = clusterer._skill_prompt_instructions(
        _Cluster("Debugging", "Fixing broken code"), []
    )
    assert "Debugging" in out and "Fixing broken code" in out


def test_the_skill_prompt_carries_at_most_three_exemplars(clusterer):
    """SPEC §5.6: born as behaviour-as-prompt, and deliberately short -
    small models pay for every token."""
    out = clusterer._skill_prompt_instructions(
        _Cluster("Debugging", ""), [_event(f"example {i}") for i in range(10)]
    )
    assert out.count("- example") == 3


def test_a_long_exemplar_is_cut(clusterer):
    out = clusterer._skill_prompt_instructions(
        _Cluster("Debugging", ""), [_event("y" * 900)]
    )
    assert "y" * 201 not in out


def test_an_unnamed_cluster_still_produces_instructions(clusterer):
    out = clusterer._skill_prompt_instructions(_Cluster(None, None), [])
    assert "unnamed skill" in out


def test_blank_exemplars_are_skipped(clusterer):
    out = clusterer._skill_prompt_instructions(
        _Cluster("Debugging", ""), [_event("   "), _event("")]
    )
    assert "Examples" not in out
