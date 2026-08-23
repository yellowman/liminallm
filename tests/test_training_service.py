import pytest

from liminallm.service.training import TrainingService
from liminallm.storage.postgres import PostgresStore
from tests.harness import get_test_store


def _create_user_and_conversation(store: PostgresStore, suffix: str = ""):
    user = store.create_user(f"test{suffix}@example.com")
    conversation = store.create_conversation(user.id, title="test conversation")
    return user, conversation


def _append_assistant_message(store: PostgresStore, conversation_id: str, sender_id: str):
    return store.append_message(
        conversation_id, sender=sender_id, role="assistant", content="hello"
    )


def test_feedback_enqueues_single_training_job_with_cooldown(tmp_path):
    store = get_test_store()
    training = TrainingService(store, fs_root=str(tmp_path))
    user, conversation = _create_user_and_conversation(store)
    message = _append_assistant_message(store, conversation.id, user.id)

    training.record_feedback_event(
        user_id=user.id,
        conversation_id=conversation.id,
        message_id=message.id,
        feedback="positive",
    )

    assert len(store.list_training_jobs(user_id=user.id)) == 1

    next_message = _append_assistant_message(store, conversation.id, user.id)
    training.record_feedback_event(
        user_id=user.id,
        conversation_id=conversation.id,
        message_id=next_message.id,
        feedback="like",
    )

    assert len(store.list_training_jobs(user_id=user.id)) == 1

    job = store.list_training_jobs(user_id=user.id)[0]
    store.update_training_job(job.id, status="succeeded")
    training.training_job_cooldown_seconds = 1000

    third_message = _append_assistant_message(store, conversation.id, user.id)
    training.record_feedback_event(
        user_id=user.id,
        conversation_id=conversation.id,
        message_id=third_message.id,
        feedback="positive",
    )

    assert len(store.list_training_jobs(user_id=user.id)) == 1

    training.training_job_cooldown_seconds = 0
    fourth_message = _append_assistant_message(store, conversation.id, user.id)
    training.record_feedback_event(
        user_id=user.id,
        conversation_id=conversation.id,
        message_id=fourth_message.id,
        feedback="positive",
    )

    assert len(store.list_training_jobs(user_id=user.id)) == 2


class TestAJobSaysWhatActuallyHappened:
    """`skipped` is not `succeeded`, and neither of them is `gate_rejected`.

    `_run_jax_optax_training` returns `status="skipped"` when there is no
    JAX, no base checkpoint, no real tokenizer, or no usable LoRA weights, and
    its own comments say such a run did not train and cannot promote.
    `_promotion_gate` agrees: any non-`ok` trace is `promoted=False` with
    "training did not run".

    The job was then written `succeeded` anyway, carrying
    `1.0 / (1 + len(dataset_entries))` — a number no training produced. The
    worker overwrote it afterwards to `gate_rejected`, which its own comment
    defines as "a run that trained but failed the eval gate". So a run that
    never trained passed through a `succeeded` another replica could read,
    and settled on a status blaming model quality for a missing checkpoint.

    One owner: `TrainingService` decides the terminal status, and the worker
    records it rather than re-deriving it. Exceptions stay the worker's
    retry and dead-letter path.
    """

    def _adapter(self, store, user_id):
        import uuid as _uuid

        return store.create_artifact(
            "adapter",
            f"a_{_uuid.uuid4().hex[:6]}",
            {
                "kind": "adapter.lora",
                "mode": "local",
                "base_model": "b",
                "current_version": 0,
            },
            owner_user_id=user_id,
        )

    def _job_with_one_positive_event(self, store, training, suffix):
        user, conversation = _create_user_and_conversation(store, suffix)
        adapter = self._adapter(store, user.id)
        message = _append_assistant_message(store, conversation.id, user.id)
        training.record_feedback_event(
            user_id=user.id,
            conversation_id=conversation.id,
            message_id=message.id,
            feedback="positive",
            context_text="the correction this run is meant to learn",
            corrected_text="the corrected answer",
        )
        return user, adapter

    def _trace(self, training, trace):
        training._run_jax_optax_training = lambda *a, **k: dict(trace)

    def test_a_skipped_run_is_not_recorded_as_a_success(self, tmp_path):
        store = get_test_store()
        training = TrainingService(store, fs_root=str(tmp_path))
        user, adapter = self._job_with_one_positive_event(store, training, "_skip")
        self._trace(training, {"status": "skipped", "reason": "no base checkpoint"})

        result = training.train_from_preferences(user.id, adapter.id)

        job = store.get_training_job(result["job_id"])
        assert job.status == "skipped", (
            f"a run that never trained was recorded as {job.status!r}"
        )
        assert job.loss is None, (
            f"a run that never trained reported a loss of {job.loss}"
        )
        assert job.new_version is None
        assert job.meta["jax_trace"]["reason"] == "no base checkpoint"

    def test_a_skipped_run_earns_no_router_credit_and_keeps_its_status(self, tmp_path):
        """The worker records the service's decision; it does not re-derive it."""
        import asyncio

        from liminallm.service.training_worker import TrainingWorker

        store = get_test_store()
        training = TrainingService(store, fs_root=str(tmp_path))
        user, adapter = self._job_with_one_positive_event(store, training, "_wskip")
        self._trace(training, {"status": "skipped", "reason": "jax/optax not installed"})
        credited = []
        training.record_training_outcome = lambda **kw: credited.append(kw)

        job = store.create_training_job(user.id, adapter.id)
        asyncio.run(TrainingWorker(store, training)._process_job(job))

        refreshed = store.get_training_job(job.id)
        assert refreshed.status == "skipped", (
            "the worker relabelled a run that never trained as "
            f"{refreshed.status!r}, which blames the eval gate for a "
            "missing checkpoint"
        )
        assert credited == [], "a run that never trained was credited to the router"

    def test_a_trained_run_that_fails_the_gate_keeps_its_training_loss(self, tmp_path):
        store = get_test_store()
        training = TrainingService(store, fs_root=str(tmp_path))
        user, adapter = self._job_with_one_positive_event(store, training, "_gate")
        # `ok`, so it trained; no holdout, so the gate refuses to promote.
        self._trace(training, {
            "status": "ok",
            "steps": [{"step": 0, "loss": 2.5}, {"step": 1, "loss": 0.9}],
        })

        result = training.train_from_preferences(user.id, adapter.id)

        job = store.get_training_job(result["job_id"])
        assert job.status == "gate_rejected", job.status
        assert job.loss == pytest.approx(0.9), (
            "the recorded loss is not the one the training loop produced"
        )
        assert job.new_version is None
        # Named, so this keeps testing the branch it was written for: the gate
        # also refuses a run it cannot evaluate, and `gate_rejected` covers
        # both that and a measured regression.
        assert "no holdout" in job.meta["eval_gate"]["reason"], (
            job.meta["eval_gate"]
        )

    def test_no_train_batches_is_a_skipped_run_not_an_empty_success(self, tmp_path):
        """Zero optimizer updates is not a training run that went well.

        The loop is `for batch in batches`, so an empty list ran nothing and
        still returned `ok` with `steps: []` — which the gate then judged on
        an eval it never moved.
        """
        store = get_test_store()
        training = TrainingService(store, fs_root=str(tmp_path))

        trace = training._run_jax_optax_training(
            {"layer0.attn_q.lora_a": [[0.0]]},
            [],
            params_path=tmp_path / "params.json",
        )

        assert trace["status"] == "skipped", trace
        assert "batch" in trace["reason"], trace

    def test_a_skipped_run_clears_the_previous_attempt_terminal_fields(
        self, tmp_path
    ):
        """`None` has to mean NULL here, not "leave what is already there".

        The worker retries the same claimed `job_id` after an exception, and
        the service writes the terminal result before the worker re-reads and
        finalizes it. So a transient failure in that later database work
        leaves a second attempt running against a job that already carries a
        previous attempt's `loss` and `new_version`. If the second attempt is
        skipped, saying so is not enough: the earlier numbers have to go, or
        the job reads as a run that never trained and yet produced version 7
        at loss 0.42.
        """
        store = get_test_store()
        training = TrainingService(store, fs_root=str(tmp_path))
        user, adapter = self._job_with_one_positive_event(store, training, "_stale")
        job = store.create_training_job(user.id, adapter.id)
        store.update_training_job(job.id, loss=0.42, new_version=7)
        assert store.get_training_job(job.id).loss == pytest.approx(0.42)

        self._trace(training, {"status": "skipped", "reason": "no base checkpoint"})
        training.train_from_preferences(user.id, adapter.id, job_id=job.id)

        refreshed = store.get_training_job(job.id)
        assert refreshed.status == "skipped"
        assert refreshed.loss is None, (
            f"a run that never trained kept an earlier attempt's loss "
            f"({refreshed.loss})"
        )
        assert refreshed.new_version is None, (
            f"a run that never trained kept an earlier attempt's version "
            f"({refreshed.new_version})"
        )

    def test_a_promoted_run_keeps_the_version_the_service_recorded(self, tmp_path):
        """The worker finalizes the job; it must not undo the promotion.

        `TrainingService` writes `new_version` when the gate promotes, and the
        result it returns names the directory rather than the number. The
        worker therefore leaves that column alone — which under
        "None preserves" it did by passing `None`, and under "None is NULL" it
        does by not passing it at all. Measured: passing `None` here erased a
        promoted version and nothing in the suite noticed.
        """
        import asyncio

        from liminallm.service.training_worker import TrainingWorker

        store = get_test_store()
        training = TrainingService(store, fs_root=str(tmp_path))
        user, conversation = _create_user_and_conversation(store, "_promoted")
        adapter = self._adapter(store, user.id)
        job = store.create_training_job(user.id, adapter.id)
        # What the service records on promotion, before the worker finalizes.
        store.update_training_job(job.id, new_version=3, loss=0.5)
        credited = []
        training.record_training_outcome = lambda **kw: credited.append(kw)
        training.train_from_preferences = lambda *a, **k: {
            "job_id": job.id,
            "adapter_id": adapter.id,
            "version_dir": str(tmp_path / "v0003"),
            "status": "succeeded",
            "loss": 0.5,
            "jax_trace": {"status": "ok", "steps": [{"loss": 0.5}]},
            "eval_gate": {"promoted": True, "reason": "improved"},
        }

        asyncio.run(TrainingWorker(store, training)._process_job(job))

        refreshed = store.get_training_job(job.id)
        assert refreshed.status == "succeeded", refreshed.status
        assert refreshed.new_version == 3, (
            "the worker erased the version the promotion recorded: "
            f"{refreshed.new_version}"
        )
        assert credited, "a promoted run was not credited to the router"
