"""Tests for the adapter ladder: prompt-first skills, pooled training data,
teacher distillation, and the holdout eval gate (SPEC §5.4/§5.6/§7)."""

import uuid

import pytest

from liminallm.service.clustering import SemanticClusterer
from liminallm.service.training import TrainingService
from tests.harness import get_test_store


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
        store = get_test_store()
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
        jobs = [j for j in store.list_training_jobs() if j.adapter_id == adapter.id]
        assert jobs == []

    def test_weights_job_enqueued_at_threshold(self, tmp_path):
        store = get_test_store()
        training = TrainingService(store, str(tmp_path))
        clusterer = SemanticClusterer(store, llm=None, training=training)
        cluster = _make_cluster(store)
        _seed_user_with_events(store, "a@t.local", cluster.id, 8)

        promoted = clusterer.promote_skill_adapters(min_size=5, weights_min_events=8)

        adapter_id = promoted[0]
        jobs = [j for j in store.list_training_jobs() if j.adapter_id == adapter_id]
        assert len(jobs) == 1

    def test_cross_user_cluster_is_tenant_scoped_not_global(self, tmp_path):
        store = get_test_store()
        training = TrainingService(store, str(tmp_path))
        clusterer = SemanticClusterer(store, llm=None, training=training)
        cluster = _make_cluster(store, user_id=None)
        _seed_user_with_events(store, "a@t.local", cluster.id, 3)
        heavy_user, _ = _seed_user_with_events(store, "b@t.local", cluster.id, 5)

        promoted = clusterer.promote_skill_adapters(min_size=5, weights_min_events=8)

        adapter = store.get_artifact(promoted[0])
        # A cluster spanning several users is owned by its top contributor and
        # shared within that tenant - never "global", which every tenant sees.
        assert adapter.visibility == "shared"
        assert adapter.owner_user_id == heavy_user.id
        assert adapter.schema["scope"] == "tenant"
        assert adapter.schema["tenant_id"] == "public"
        jobs = [j for j in store.list_training_jobs() if j.adapter_id == adapter.id]
        assert len(jobs) == 1
        assert jobs[0].user_id == heavy_user.id  # most frequent contributor


class TestPooledTrainingData:
    def test_global_skill_adapter_pools_events_across_users(self, tmp_path):
        store = get_test_store()
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
        store = get_test_store()
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
        store = get_test_store()
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
        store = get_test_store()
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
        store = get_test_store()
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
        store = get_test_store()
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
        store = get_test_store()
        teacher = _FakeTeacher()
        training = TrainingService(store, str(tmp_path), teacher=teacher)
        cluster = _make_cluster(store)
        user, _ = _seed_user_with_events(store, "a@t.local", cluster.id, 2)

        training.train_from_preferences(user.id)

        assert teacher.calls == 0


class TestTenantIsolation:
    """Regression tests: pooled skill training must never cross tenants."""

    def _seed_tenant(self, store, email, tenant, cluster_id, n, secret):
        user = store.create_user(email=f"{uuid.uuid4().hex[:8]}_{email}", tenant_id=tenant)
        convo = store.create_conversation(user.id, title="t")
        for i in range(n):
            store.append_message(convo.id, "user", "user", f"q{i}")
            reply = store.append_message(convo.id, "assistant", "assistant", secret)
            store.record_preference_event(
                user.id, convo.id, reply.id, "positive",
                corrected_text=secret, context_embedding=[0.1] * 64,
                cluster_id=cluster_id, context_text=secret,
            )
        return user

    def test_pooled_training_excludes_other_tenants(self, tmp_path):
        store = get_test_store()
        training = TrainingService(store, str(tmp_path))
        clusterer = SemanticClusterer(store, llm=None, training=training)
        cluster = _make_cluster(store, user_id=None)
        acme = self._seed_tenant(store, "a@acme", "acme", cluster.id, 6, "ACME_SECRET")
        self._seed_tenant(store, "b@initech", "initech", cluster.id, 4, "INITECH_SECRET")

        adapter_id = clusterer.promote_skill_adapters(
            min_size=5, weights_min_events=10
        )[0]
        result = training.train_from_preferences(acme.id, adapter_id=adapter_id)

        dataset = open(store.get_training_job(result["job_id"]).dataset_path).read()
        assert "ACME_SECRET" in dataset
        assert "INITECH_SECRET" not in dataset

    def test_tenant_adapter_not_visible_to_other_tenant(self, tmp_path):
        store = get_test_store()
        training = TrainingService(store, str(tmp_path))
        clusterer = SemanticClusterer(store, llm=None, training=training)
        cluster = _make_cluster(store, user_id=None)
        self._seed_tenant(store, "a@acme", "acme", cluster.id, 6, "ACME_SECRET")
        outsider = self._seed_tenant(
            store, "b@initech", "initech", cluster.id, 4, "INITECH_SECRET"
        )

        adapter_id = clusterer.promote_skill_adapters(
            min_size=5, weights_min_events=10
        )[0]

        visible = [
            a.id
            for a in store.list_artifacts(
                type_filter="adapter", owner_user_id=outsider.id, tenant_id="initech"
            )
        ]
        assert adapter_id not in visible


class TestRejectedWeightsNotServed:
    """Regression test: gate-rejected weights must be unreachable at inference."""

    def test_rejected_version_is_quarantined_and_unresolvable(self, tmp_path):
        pytest.importorskip("jax")
        from pathlib import Path

        from liminallm.service.model_backend import LocalJaxLoRABackend

        store = get_test_store()
        training = TrainingService(store, str(tmp_path))
        cluster = _make_cluster(store)
        user, _ = _seed_user_with_events(store, "a@t.local", cluster.id, 10)
        adapter = training.ensure_user_adapter(user.id)

        original = training._run_jax_optax_training

        def regressing(*args, **kwargs):
            trace = original(*args, **kwargs)
            if trace.get("status") == "ok":
                trace["eval_before"], trace["eval_after"] = 1.0, 2.0  # got worse
            return trace

        training._run_jax_optax_training = regressing
        result = training.train_from_preferences(user.id)

        assert result["eval_gate"]["promoted"] is False
        adapter_dir = Path(training._adapter_dir(user.id, adapter.id, adapter.schema))
        assert list(adapter_dir.glob("v*")) == []  # nothing live
        assert (adapter_dir / "rejected").exists()  # kept for debugging

        refreshed = store.get_artifact(adapter.id)
        backend = LocalJaxLoRABackend.__new__(LocalJaxLoRABackend)
        resolved = backend._resolve_params_path(
            adapter_dir, current_version=refreshed.schema.get("current_version")
        )
        assert resolved is None

    def test_promoted_version_is_pinned_not_newest_on_disk(self, tmp_path):
        from pathlib import Path

        from liminallm.service.model_backend import LocalJaxLoRABackend

        adapter_dir = tmp_path / "adapter"
        for version in (1, 2):
            vdir = adapter_dir / f"v{version:04d}"
            vdir.mkdir(parents=True)
            (vdir / "params.json").write_text("{}")

        backend = LocalJaxLoRABackend.__new__(LocalJaxLoRABackend)
        # v0002 is newest on disk, but v0001 is the promoted version.
        resolved = backend._resolve_params_path(adapter_dir, current_version=1)
        assert resolved == adapter_dir / "v0001" / "params.json"
