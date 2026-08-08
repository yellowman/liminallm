"""Adapter judgements belong to TrainingService, not to the thing on a timer.

Before this split, deciding an adapter had stopped earning its keep, turning a
training loss into a router score, and re-embedding stale note vectors all
lived in TrainingWorker — reachable only by constructing a worker and waiting
for a tick. These exercise them directly.
"""

from liminallm.service import notes
from liminallm.service.training import TrainingService


class TestRunSummary:
    """describe_run: the directory layout is the service's own knowledge."""

    def test_none_result_summarises_to_none(self):
        assert TrainingService.describe_run(None) is None

    def test_version_is_recovered_from_the_job_directory(self):
        out = TrainingService.describe_run({"version_dir": "/a/b/v7", "loss": 0.5})
        assert out["version"] == 7
        assert out["loss"] == 0.5

    def test_explicit_version_wins_over_the_directory_name(self):
        out = TrainingService.describe_run({"version_dir": "/a/v7", "new_version": 9})
        assert out["version"] == 9

    def test_unparseable_directory_yields_no_version_rather_than_raising(self):
        out = TrainingService.describe_run({"version_dir": "/a/b/vNOPE"})
        assert out["version"] is None

    def test_carries_through_the_fields_the_router_needs(self):
        out = TrainingService.describe_run(
            {"version_dir": "/a/v1", "clusters": [{"count": 1}], "jax_trace": "t"}
        )
        assert out["clusters"] == [{"count": 1}]
        assert out["jax_trace"] == "t"


class TestScoreFromLoss:
    def test_lower_loss_scores_higher(self):
        better = TrainingService._score_from_loss(0.1)
        worse = TrainingService._score_from_loss(2.0)
        assert better > worse

    def test_score_stays_in_range(self):
        for loss in (0.0, 0.5, 10.0, 1e9):
            assert 0.0 <= TrainingService._score_from_loss(loss) <= 1.0

    def test_missing_or_nonsense_loss_scores_zero(self):
        """A run that reported no loss must not look like a good one."""
        for loss in (None, "nope", -1.0):
            assert TrainingService._score_from_loss(loss) == 0.0


class TestCentroidAggregation:
    def test_no_clusters_gives_no_centroid(self):
        assert TrainingService._aggregate_cluster_centroid(None) == []
        assert TrainingService._aggregate_cluster_centroid([]) == []

    def test_weighted_by_cluster_size(self):
        """A cluster of 90 should pull the centroid further than one of 10."""
        out = TrainingService._aggregate_cluster_centroid(
            [
                {"centroid": [0.0, 0.0], "count": 90},
                {"centroid": [1.0, 1.0], "count": 10},
            ]
        )
        assert out == [0.1, 0.1]

    def test_ragged_centroids_are_padded_not_dropped(self):
        out = TrainingService._aggregate_cluster_centroid(
            [{"centroid": [1.0, 1.0, 1.0], "count": 1}, {"centroid": [1.0], "count": 1}]
        )
        assert out == [1.0, 0.5, 0.5]

    def test_clusters_with_no_members_are_ignored(self):
        out = TrainingService._aggregate_cluster_centroid(
            [{"centroid": [1.0], "count": 0}, {"centroid": [3.0], "count": 1}]
        )
        assert out == [3.0]


class TestReembedSweep:
    """The sweep that stops unread notes falling out of semantic search."""

    class _Encoder:
        model_id = "encoder-v2"
        is_semantic = True

        def embed(self, text):
            return [0.5, 0.5]

    def test_only_stale_and_missing_vectors_are_rewritten(self, store):
        user = store.create_user("sweep-owner@example.com")
        fresh = store.create_note(user.id, "Fresh", "x", embedding=[1.0] + [0.0] * 63)
        store.update_note_meta(fresh.id, {"embedding_model": "encoder-v2"})
        stale = store.create_note(user.id, "Stale", "y", embedding=[1.0] + [0.0] * 63)
        store.update_note_meta(stale.id, {"embedding_model": "encoder-v1"})
        store.create_note(user.id, "Never", "z")

        assert notes.reembed_stale(store, self._Encoder(), user_limit=100) == 2
        assert store.get_note(stale.id).meta["embedding_model"] == "encoder-v2"

    def test_the_sweep_converges(self, store):
        user = store.create_user("sweep-converge@example.com")
        store.create_note(user.id, "One", "x")
        encoder = self._Encoder()
        assert notes.reembed_stale(store, encoder, user_limit=100) == 1
        assert notes.reembed_stale(store, encoder, user_limit=100) == 0

    def test_a_batch_bounds_one_pass(self, store):
        user = store.create_user("sweep-batch@example.com")
        for i in range(5):
            store.create_note(user.id, f"Note {i}", "text")
        done = notes.reembed_stale(store, self._Encoder(), user_limit=100, batch=3)
        assert done == 3

    def test_a_provider_failure_keeps_what_already_landed(self, store):
        """Progress is persisted per note, so a 500 costs the pass, not the work."""
        user = store.create_user("sweep-flaky@example.com")
        for i in range(5):
            store.create_note(user.id, f"Note {i}", "text")

        class Flaky(self._Encoder):
            calls = 0

            def embed(self, text):
                Flaky.calls += 1
                if Flaky.calls > 2:
                    raise RuntimeError("provider 500")
                return [0.5, 0.5]

        assert notes.reembed_stale(store, Flaky(), user_limit=100) == 2

    def test_a_store_that_cannot_list_users_is_survived(self):
        class Broken:
            def list_users(self, limit=None):
                raise RuntimeError("database down")

        assert notes.reembed_stale(Broken(), self._Encoder(), user_limit=10) == 0


def test_the_worker_only_schedules():
    """The worker's surface should be the loop and its gates, nothing else.

    500 of its ~700 lines were domain logic reachable only by waiting for a
    tick. If a judgement about adapters, clustering or embeddings lands back
    here, this is the reminder to put it with the thing it is about.
    """
    import re

    from liminallm.service import training_worker

    with open(training_worker.__file__) as handle:
        source = handle.read()
    methods = set(re.findall(r"^    (?:async )?def (\w+)", source, re.M))
    assert methods == {
        "__init__",
        "start",
        "stop",
        "_run_loop",
        # each gate is "is it time, and am I the leader"
        "_maybe_run_periodic_clustering",
        "_maybe_recommend_adapter_pruning",
        "_maybe_reembed_stale_vectors",
        # queue consumption and the job state machine
        "_process_queued_jobs",
        "_get_queued_jobs",
        "_process_job",
        "_execute_training",
        "_get_cluster_id",
    }
