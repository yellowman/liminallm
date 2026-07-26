"""Tests for the adapter ladder: prompt-first skills, pooled training data,
teacher distillation, and the holdout eval gate (SPEC §5.4/§5.6/§7)."""

import uuid

import pytest

from liminallm.service.clustering import SemanticClusterer
from liminallm.service.training import TrainingService
from liminallm.storage.memory import MemoryStore


def _seed_user_with_events(store, email, cluster_id, n_events, corrected="use tabs"):
    user = store.create_user(email=f"{uuid.uuid4().hex[:8]}_{email}")
    convo = store.create_conversation(user.id, title="t")
    events = []
    for i in range(n_events):
        prompt_msg = store.append_message(convo.id, "user", "user", f"question {i}")
        reply = store.append_message(convo.id, "assistant", "assistant", f"answer {i}")
        _ = prompt_msg
        events.append(
            store.record_preference_event(
                user.id,
                convo.id,
                reply.id,
                "positive",
                corrected_text=corrected,
                context_embedding=[0.1] * 64,
                cluster_id=cluster_id,
                context_text=f"context {i}",
            )
        )
    return user, events


def _make_cluster(store, *, user_id=None, size=10):
    return store.upsert_semantic_cluster(
        user_id=user_id,
        centroid=[0.1] * 64,
        size=size,
        label="tab formatting",
        description="user prefers tab-indented answers",
    )


class TestPromptFirstLadder:
    def test_skill_adapter_born_in_prompt_mode(self, tmp_path):
        store = MemoryStore(fs_root=str(tmp_path))
        training = TrainingService(store, str(tmp_path))
        clusterer = SemanticClusterer(store, llm=None, training=training)
        cluster = _make_cluster(store)
        _seed_user_with_events(store, "a@t.local", cluster.id, 6)

        promoted = clusterer.promote_skill_adapters(min_size=5, weights_min_events=20)

        assert len(promoted) == 1
        adapter = store.get_artifact(promoted[0])
        schema = adapter.schema
        assert schema["mode"] == "prompt"
        assert "tab formatting" in schema["prompt_instructions"]
        assert "use tabs" in schema["prompt_instructions"]
        assert schema["lifecycle"]["stage"] == "prompt"
        # Below the weights threshold: no training job enqueued.
        jobs = [j for j in store.training_jobs.values() if j.adapter_id == adapter.id]
        assert jobs == []

    def test_weights_job_enqueued_at_threshold(self, tmp_path):
        store = MemoryStore(fs_root=str(tmp_path))
        training = TrainingService(store, str(tmp_path))
        clusterer = SemanticClusterer(store, llm=None, training=training)
        cluster = _make_cluster(store)
        _seed_user_with_events(store, "a@t.local", cluster.id, 8)

        promoted = clusterer.promote_skill_adapters(min_size=5, weights_min_events=8)

        adapter_id = promoted[0]
        jobs = [j for j in store.training_jobs.values() if j.adapter_id == adapter_id]
        assert len(jobs) == 1

    def test_global_cluster_gets_nominal_job_owner(self, tmp_path):
        store = MemoryStore(fs_root=str(tmp_path))
        training = TrainingService(store, str(tmp_path))
        clusterer = SemanticClusterer(store, llm=None, training=training)
        cluster = _make_cluster(store, user_id=None)
        _seed_user_with_events(store, "a@t.local", cluster.id, 3)
        heavy_user, _ = _seed_user_with_events(store, "b@t.local", cluster.id, 5)

        promoted = clusterer.promote_skill_adapters(min_size=5, weights_min_events=8)

        adapter = store.get_artifact(promoted[0])
        assert adapter.owner_user_id is None  # global skill
        jobs = [j for j in store.training_jobs.values() if j.adapter_id == adapter.id]
        assert len(jobs) == 1
        assert jobs[0].user_id == heavy_user.id  # most frequent contributor


class TestPooledTrainingData:
    def test_global_skill_adapter_pools_events_across_users(self, tmp_path):
        store = MemoryStore(fs_root=str(tmp_path))
        training = TrainingService(store, str(tmp_path))
        clusterer = SemanticClusterer(store, llm=None, training=training)
        cluster = _make_cluster(store, user_id=None)
        user_a, _ = _seed_user_with_events(store, "a@t.local", cluster.id, 4)
        _seed_user_with_events(store, "b@t.local", cluster.id, 4)
        promoted = clusterer.promote_skill_adapters(min_size=5, weights_min_events=8)
        adapter_id = promoted[0]

        result = training.train_from_preferences(user_a.id, adapter_id=adapter_id)

        assert result is not None
        job = store.get_training_job(result["job_id"])
        # Both users' events flowed into the dataset (deduped per message).
        assert result["eval_gate"]["holdout_examples"] >= 1
        dataset = open(job.dataset_path).read().strip().splitlines()
        assert len(dataset) == 8 - result["eval_gate"]["holdout_examples"] or len(
            dataset
        ) == 8  # dataset file holds all entries pre-split or post-split count

    def test_persona_adapter_stays_per_user(self, tmp_path):
        store = MemoryStore(fs_root=str(tmp_path))
        training = TrainingService(store, str(tmp_path))
        cluster = _make_cluster(store, user_id=None)
        user_a, _ = _seed_user_with_events(store, "a@t.local", cluster.id, 2)
        _seed_user_with_events(store, "b@t.local", cluster.id, 2)

        result = training.train_from_preferences(user_a.id)

        assert result is not None
        job = store.get_training_job(result["job_id"])
        dataset = open(job.dataset_path).read().strip().splitlines()
        assert len(dataset) == 2  # only user_a's events


class TestEvalGate:
    def test_training_skip_blocks_promotion(self, tmp_path, monkeypatch):
        store = MemoryStore(fs_root=str(tmp_path))
        training = TrainingService(store, str(tmp_path))
        cluster = _make_cluster(store)
        user, _ = _seed_user_with_events(store, "a@t.local", cluster.id, 3)
        adapter = training.ensure_user_adapter(user.id)
        before_version = adapter.schema.get("current_version", 0)

        monkeypatch.setattr(
            training,
            "_run_jax_optax_training",
            lambda *a, **k: {"status": "skipped", "reason": "test"},
        )
        result = training.train_from_preferences(user.id)

        assert result["eval_gate"]["promoted"] is False
        refreshed = store.get_artifact(adapter.id)
        assert refreshed.schema.get("current_version", 0) == before_version

    def test_real_jax_training_runs_eval_and_promotes(self, tmp_path):
        pytest.importorskip("jax")
        pytest.importorskip("optax")
        store = MemoryStore(fs_root=str(tmp_path))
        training = TrainingService(store, str(tmp_path))
        cluster = _make_cluster(store)
        user, _ = _seed_user_with_events(store, "a@t.local", cluster.id, 10)

        result = training.train_from_preferences(user.id)

        trace = result["jax_trace"]
        assert trace["status"] == "ok"
        gate = result["eval_gate"]
        assert gate["holdout_examples"] == 2  # every 5th of 10
        assert isinstance(gate["eval_before"], float)
        assert isinstance(gate["eval_after"], float)
        # Training on a repetitive corpus should improve holdout loss and pass.
        assert gate["promoted"] is True
        adapter = store.get_artifact(result["adapter_id"])
        assert adapter.schema["current_version"] == 1

    def test_prompt_adapter_graduates_to_hybrid_on_promotion(self, tmp_path):
        pytest.importorskip("jax")
        store = MemoryStore(fs_root=str(tmp_path))
        training = TrainingService(store, str(tmp_path))
        clusterer = SemanticClusterer(store, llm=None, training=training)
        cluster = _make_cluster(store, user_id=None)
        user, _ = _seed_user_with_events(store, "a@t.local", cluster.id, 10)
        promoted = clusterer.promote_skill_adapters(min_size=5, weights_min_events=10)
        adapter_id = promoted[0]

        result = training.train_from_preferences(user.id, adapter_id=adapter_id)

        adapter = store.get_artifact(adapter_id)
        if result["eval_gate"]["promoted"]:
            assert adapter.schema["mode"] == "hybrid"
            assert adapter.schema["lifecycle"]["stage"] == "weights"
            assert adapter.schema["prompt_instructions"]  # kept as fallback
        else:
            assert adapter.schema["mode"] == "prompt"


class _FakeTeacher:
    def __init__(self):
        self.calls = 0

    def generate(self, message, adapters=None, context_snippets=None, history=None):
        self.calls += 1
        return {"content": "distilled exemplar response"}


class TestDistillation:
    def test_targets_rewritten_when_enabled(self, tmp_path):
        store = MemoryStore(fs_root=str(tmp_path))
        teacher = _FakeTeacher()
        training = TrainingService(
            store, str(tmp_path), teacher=teacher, distillation_enabled=True
        )
        cluster = _make_cluster(store)
        user, _ = _seed_user_with_events(store, "a@t.local", cluster.id, 3)

        result = training.train_from_preferences(user.id)

        assert teacher.calls == 3
        job = store.get_training_job(result["job_id"])
        assert job.meta["distilled"] == 3
        dataset = open(job.dataset_path).read()
        assert "distilled exemplar response" in dataset

    def test_disabled_by_default(self, tmp_path):
        store = MemoryStore(fs_root=str(tmp_path))
        teacher = _FakeTeacher()
        training = TrainingService(store, str(tmp_path), teacher=teacher)
        cluster = _make_cluster(store)
        user, _ = _seed_user_with_events(store, "a@t.local", cluster.id, 2)

        training.train_from_preferences(user.id)

        assert teacher.calls == 0
