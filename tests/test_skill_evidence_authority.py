"""The evidence that earned a job is the evidence that trains, or nothing is.

A training run is authorized by a specific set of events: the ones whose
counts crossed the bar. Training a subset of them is not a smaller version
of that decision, it is a different one that nobody made. Nineteen of the
twenty messages, or two of the three qualifying contributors, can still
pass a holdout and promote weights - and the evidence record will say the
run was justified by numbers it never actually trained on.

So resolution is all-or-nothing. Every named event must load and must sit
inside the run's authorization boundary; anything else refuses the run.

Which boundary is a deliberate choice. `cluster_id` is mutable
classification state - reclustering moves events between clusters as a
matter of course - so the job's ids are the snapshot of membership and
current cluster membership is *not* re-required. Tenant (for a shared
skill) and ownership (for a personal one) are the boundaries that must
still hold, because those are the ones that would leak.
"""

from __future__ import annotations

import uuid

import pytest

from liminallm.service.clustering import SemanticClusterer
from liminallm.service.training import TrainingService
from liminallm.storage.errors import ConstraintViolation
from tests.harness import get_test_store


def _user(store, label: str):
    return store.create_user(email=f"{uuid.uuid4().hex[:8]}_{label}@t.local")


def _seed(store, user, cluster_id, *, conversations, per_conversation, age_hours=0.0):
    recorded = []
    for _ in range(conversations):
        convo = store.create_conversation(user.id, title="t")
        for i in range(per_conversation):
            store.append_message(convo.id, "user", "user", f"q {i}")
            reply = store.append_message(convo.id, "assistant", "assistant", f"a {i}")
            recorded.append(
                store.record_preference_event(
                    user.id, convo.id, reply.id, "positive",
                    corrected_text="use tabs", context_embedding=[0.1] * 64,
                    cluster_id=cluster_id, context_text="context",
                ).id
            )
    if age_hours:
        import psycopg

        with psycopg.connect(store.dsn, autocommit=True) as conn:
            conn.execute(
                "UPDATE preference_event SET created_at = created_at - "
                "make_interval(hours => %s) WHERE id = ANY(%s)",
                (age_hours, recorded),
            )
    return recorded


def _cluster(store, *, user_id=None):
    return store.upsert_semantic_cluster(
        user_id=user_id, centroid=[0.1] * 64, size=10,
        label="routing debug", description="start from the routing table",
    )


def _earned_personal_job(store, tmp_path):
    """A personal skill that has earned weights, and its queued job."""
    owner = _user(store, "solo")
    cluster = _cluster(store, user_id=owner.id)
    _seed(store, owner, cluster.id, conversations=4, per_conversation=3,
          age_hours=96)
    _seed(store, owner, cluster.id, conversations=4, per_conversation=3)
    training = TrainingService(store, str(tmp_path))
    clusterer = SemanticClusterer(store, llm=None, training=training)
    adapter_id = clusterer.promote_skill_adapters(min_size=5)[0]
    jobs = [j for j in store.list_training_jobs() if j.adapter_id == adapter_id]
    assert len(jobs) == 1, "the fixture did not earn weights"
    return owner, adapter_id, jobs[0], training


class TestAPartialEvidenceSetRefusesTheRun:
    def test_a_vanished_event_refuses_rather_than_training_the_rest(
        self, tmp_path
    ):
        """The failure the log line used to hide.

        Dropping the missing ones and training the remainder keeps the
        run alive on evidence that no longer meets the bar it cleared.
        """
        store = get_test_store()
        owner, adapter_id, job, training = _earned_personal_job(store, tmp_path)

        import psycopg

        with psycopg.connect(store.dsn, autocommit=True) as conn:
            conn.execute(
                "DELETE FROM preference_event WHERE id = %s",
                (job.preference_event_ids[0],),
            )

        with pytest.raises(ConstraintViolation):
            training.train_from_preferences(
                owner.id, adapter_id=adapter_id, job_id=job.id
            )

    def test_an_event_outside_the_boundary_refuses_the_run(self, tmp_path):
        """Reassigning an event to another account takes it out of scope."""
        store = get_test_store()
        owner, adapter_id, job, training = _earned_personal_job(store, tmp_path)
        stranger = _user(store, "stranger")

        import psycopg

        with psycopg.connect(store.dsn, autocommit=True) as conn:
            conn.execute(
                "UPDATE preference_event SET user_id = %s WHERE id = %s",
                (stranger.id, job.preference_event_ids[0]),
            )

        with pytest.raises(ConstraintViolation):
            training.train_from_preferences(
                owner.id, adapter_id=adapter_id, job_id=job.id
            )

    def test_reclustering_alone_does_not_refuse_the_run(self, tmp_path):
        """The boundary is tenancy and ownership, not classification.

        Reclustering moves events between clusters as ordinary behaviour.
        The job's ids are the snapshot, so a moved event is still its
        evidence.
        """
        store = get_test_store()
        owner, adapter_id, job, training = _earned_personal_job(store, tmp_path)

        import psycopg

        with psycopg.connect(store.dsn, autocommit=True) as conn:
            conn.execute(
                "UPDATE preference_event SET cluster_id = NULL WHERE id = %s",
                (job.preference_event_ids[0],),
            )

        selected = training.resolve_job_evidence(job, user_id=owner.id)
        assert len(selected) == len(job.preference_event_ids), (
            "a reclustered event was treated as out of scope; classification "
            "is not the authorization boundary"
        )

    def test_a_job_with_no_pinned_evidence_trains_nothing(self, tmp_path):
        """An empty list must not widen back into a fresh query."""
        store = get_test_store()
        owner, adapter_id, job, training = _earned_personal_job(store, tmp_path)

        import psycopg

        with psycopg.connect(store.dsn, autocommit=True) as conn:
            conn.execute(
                "UPDATE training_job SET preference_event_ids = '{}' WHERE id = %s",
                (job.id,),
            )

        with pytest.raises(ConstraintViolation):
            training.train_from_preferences(
                owner.id, adapter_id=adapter_id, job_id=job.id
            )


    def test_a_persona_job_is_not_held_to_the_pinned_rule(self, tmp_path):
        """The limit of "no pinned evidence trains nothing".

        Only a skill job has a promotion gate that named its evidence. A
        persona adapter is not cluster-bound and never pinned anything, so
        holding it to the rule would not make it safe, it would make
        persona training unrunnable. The full suite found this: two
        existing worker tests queue exactly such a job.
        """
        store = get_test_store()
        owner = _user(store, "persona")
        convo = store.create_conversation(owner.id, title="t")
        store.append_message(convo.id, "user", "user", "q")
        reply = store.append_message(convo.id, "assistant", "assistant", "a")
        store.record_preference_event(
            owner.id, convo.id, reply.id, "positive",
            corrected_text="use tabs", context_embedding=[0.1] * 64,
            context_text="context",
        )
        adapter = store.create_artifact(
            "adapter", f"persona_{uuid.uuid4().hex[:6]}",
            {"kind": "adapter.lora", "mode": "local", "base_model": "b",
             "current_version": 0},
            owner_user_id=owner.id,
        )
        job = store.create_training_job(owner.id, adapter.id)
        assert job.preference_event_ids in ([], None)

        training = TrainingService(store, str(tmp_path))
        # No base checkpoint here, so this skips rather than trains - but it
        # must reach that decision instead of being refused for having
        # pinned nothing.
        training._run_jax_optax_training = lambda *a, **k: {
            "status": "skipped", "reason": "no base checkpoint"
        }
        result = training.train_from_preferences(
            owner.id, adapter_id=adapter.id, job_id=job.id
        )
        assert result is not None, "the persona run was refused, not skipped"


class TestPinnedExactnessSurvivesPreparation:
    """Resolution is not the last place the set can shrink.

    Twenty events resolving and nineteen examples building is the same
    failure one stage later: the run trains on less than the evidence that
    authorized it, and a holdout on nineteen still promotes.
    """

    def test_an_unusable_target_after_enqueue_refuses_the_run(self, tmp_path):
        """Emptying one answer's text is the reachable version of this.

        `preference_event.message_id` is `NOT NULL REFERENCES message(id)
        ON DELETE CASCADE`, so an event cannot outlive or lose its target:
        deleting the message deletes the event, which the resolver catches
        one stage earlier. What *can* happen is an answer whose text is
        gone, which builds an example carrying no supervision. The strict
        branch for a genuinely missing target is therefore defensive - it
        guards a store that is not this schema - and is not witnessed here.
        """
        store = get_test_store()
        owner, adapter_id, job, training = _earned_personal_job(store, tmp_path)
        assert job.preference_event_ids is not None, "the fixture job is not pinned"

        victim = store.get_preference_event(job.preference_event_ids[0])

        import psycopg

        with psycopg.connect(store.dsn, autocommit=True) as conn:
            conn.execute(
                "UPDATE preference_event SET corrected_text = '' WHERE id = %s",
                (victim.id,),
            )
            conn.execute(
                "UPDATE message SET content = '' WHERE id = %s",
                (victim.message_id,),
            )

        # The event still resolves and still passes ownership: this is about
        # the stage after resolution.
        resolved = training.resolve_job_evidence(job, user_id=owner.id)
        assert len(resolved) == len(job.preference_event_ids)

        with pytest.raises(ConstraintViolation, match="pinned_evidence_not_tokenizable"):
            training.train_from_preferences(
                owner.id, adapter_id=adapter_id, job_id=job.id
            )


class TestTemporalSpanCannotBeManufactured:
    def test_re_rating_an_old_answer_does_not_create_a_span(self, tmp_path):
        """Repeat ratings are one piece of evidence, including in time.

        Twenty answers in one sitting, then one of them rated again two
        days later. Nothing new succeeded, so nothing new was proved.
        """
        store = get_test_store()
        owner = _user(store, "solo")
        cluster = _cluster(store, user_id=owner.id)
        # One sitting, dated three days back.
        first_ids = _seed(
            store, owner, cluster.id, conversations=7, per_conversation=3,
            age_hours=72,
        )
        # Today: rate one of those same answers again.
        old = store.get_preference_event(first_ids[0])
        store.record_preference_event(
            owner.id, old.conversation_id, old.message_id, "positive",
            corrected_text="use tabs", context_embedding=[0.1] * 64,
            cluster_id=cluster.id, context_text="context",
        )

        training = TrainingService(store, str(tmp_path))
        clusterer = SemanticClusterer(store, llm=None, training=training)
        adapter_id = clusterer.promote_skill_adapters(min_size=5)[0]

        evidence = store.get_artifact(adapter_id).schema["lifecycle"]["evidence"]
        assert evidence["span_hours"] < 48, (
            f"span_hours is {evidence['span_hours']}: re-rating one old answer "
            "manufactured temporal independence out of a single sitting"
        )
        jobs = [j for j in store.list_training_jobs() if j.adapter_id == adapter_id]
        assert jobs == [], "a one-sitting skill graduated on a repeat rating"

    def test_a_genuinely_new_answer_does_create_a_span(self, tmp_path):
        """The rule must not block real evidence arriving later."""
        store = get_test_store()
        owner = _user(store, "solo")
        cluster = _cluster(store, user_id=owner.id)
        _seed(store, owner, cluster.id, conversations=7, per_conversation=3,
              age_hours=72)
        # A new thread with new answers, today.
        _seed(store, owner, cluster.id, conversations=1, per_conversation=1)

        training = TrainingService(store, str(tmp_path))
        clusterer = SemanticClusterer(store, llm=None, training=training)
        adapter_id = clusterer.promote_skill_adapters(min_size=5)[0]

        evidence = store.get_artifact(adapter_id).schema["lifecycle"]["evidence"]
        assert evidence["span_hours"] >= 48
        jobs = [j for j in store.list_training_jobs() if j.adapter_id == adapter_id]
        assert len(jobs) == 1


class TestDuplicateAdaptersAreAmbiguousNotResolved:
    def test_two_adapters_for_one_cluster_fail_closed(self, tmp_path):
        """The old implementation really did create these.

        Picking the oldest would make historical ambiguity into authority
        and leave the other rows live and routable.
        """
        store = get_test_store()
        owner = _user(store, "solo")
        cluster = _cluster(store, user_id=owner.id)
        schema = {
            "kind": "adapter.lora", "base_model": "jax-base", "mode": "prompt",
            "scope": "per-user", "current_version": 0, "rank": 4,
            "layers": [0], "matrices": ["attn_q"], "cluster_id": cluster.id,
        }
        for name in ("dup_a", "dup_b"):
            store.create_artifact(
                type_="adapter", name=name, schema=dict(schema),
                description="d", owner_user_id=owner.id, visibility="private",
            )

        with pytest.raises(ConstraintViolation):
            store.adapter_for_cluster(cluster.id)
