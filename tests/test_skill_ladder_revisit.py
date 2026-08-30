"""The ladder has a second rung, and a cluster has to be able to reach it.

A skill is normally discovered long before it has earned weights: the
prompt rung wants a handful of events and the weights rung wants twenty
independent ones. So the usual life of a skill is to sit on the prompt
rung for a while and graduate later, which only works if a later pass
looks at it again.

These also pin what the training job is allowed to train on. The job
records the exact events whose evidence crossed the threshold, and those
are the events that must be tokenized, split and evaluated. Re-deriving
the set at training time lets the two ends disagree about what counts as
positive, and then the evidence that justified the job is not the
evidence the weights were fitted to.
"""

from __future__ import annotations

import uuid

from liminallm.service.clustering import SemanticClusterer
from liminallm.service.training import TrainingService
from tests.harness import get_test_store


def _user(store, label: str):
    return store.create_user(email=f"{uuid.uuid4().hex[:8]}_{label}@t.local")


def _backdate(store, event_ids, hours: float):
    """Move events into the past.

    A personal skill has to show its evidence did not all arrive in one
    sitting (SPEC §5.5), and `record_preference_event` writes `now`. The
    timestamp is moved here rather than through a parameter that would
    exist only for tests.
    """
    import psycopg

    with psycopg.connect(store.dsn, autocommit=True) as conn:
        conn.execute(
            "UPDATE preference_event SET created_at = created_at - "
            "make_interval(hours => %s) WHERE id = ANY(%s)",
            (hours, list(event_ids)),
        )


def _add_evidence(
    store, user, cluster_id, *, conversations: int, per_conversation: int, age_hours=0.0
):
    """Rated answers spread over fresh threads, appended to what exists."""
    recorded = []
    for _ in range(conversations):
        convo = store.create_conversation(user.id, title="t")
        for i in range(per_conversation):
            store.append_message(convo.id, "user", "user", f"q {i}")
            reply = store.append_message(convo.id, "assistant", "assistant", f"a {i}")
            recorded.append(
                store.record_preference_event(
                    user.id,
                    convo.id,
                    reply.id,
                    "positive",
                    corrected_text="use tabs",
                    context_embedding=[0.1] * 64,
                    cluster_id=cluster_id,
                    context_text="context",
                ).id
            )
    if age_hours:
        _backdate(store, recorded, age_hours)
    return recorded


def _cluster(store, *, user_id=None):
    return store.upsert_semantic_cluster(
        user_id=user_id,
        centroid=[0.1] * 64,
        size=10,
        label="routing debug",
        description="start from the routing table",
    )


def _clusterer(store, tmp_path):
    return SemanticClusterer(
        store, llm=None, training=TrainingService(store, str(tmp_path))
    )


def _jobs_for(store, adapter_id):
    return [j for j in store.list_training_jobs() if j.adapter_id == adapter_id]


class TestASkillCanGraduateOnALaterPass:
    """The reported shape: discovered small, grown later, must graduate."""

    def test_evidence_added_after_the_prompt_rung_still_earns_weights(
        self, tmp_path
    ):
        store = get_test_store()
        owner = _user(store, "solo")
        cluster = _cluster(store, user_id=owner.id)
        clusterer = _clusterer(store, tmp_path)

        # Pass 1: enough to be noticed, not enough to be trained. Dated a
        # while back, so the span requirement is satisfied by the time the
        # later evidence arrives.
        _add_evidence(
            store, owner, cluster.id, conversations=5, per_conversation=1,
            age_hours=72,
        )
        first = clusterer.promote_skill_adapters(min_size=5)
        assert len(first) == 1, "the skill was never noticed"
        adapter_id = first[0]
        assert _jobs_for(store, adapter_id) == [], "5 events should not earn weights"

        # The skill keeps working, and the evidence accumulates.
        _add_evidence(store, owner, cluster.id, conversations=6, per_conversation=3)

        # Pass 2: the same adapter, now trained.
        clusterer.promote_skill_adapters(min_size=5)
        jobs = _jobs_for(store, adapter_id)
        assert len(jobs) == 1, (
            "the cluster already had an adapter, so the pass skipped it and the "
            "skill can never leave the prompt rung however much evidence arrives"
        )

        # Pass 3: still one. Every clustering pass must not queue another.
        clusterer.promote_skill_adapters(min_size=5)
        assert len(_jobs_for(store, adapter_id)) == 1, (
            "a second training job was queued for the same adapter"
        )

    def test_revisiting_refreshes_the_recorded_evidence(self, tmp_path):
        """The record has to track the cluster, not the moment it was born."""
        store = get_test_store()
        owner = _user(store, "solo")
        cluster = _cluster(store, user_id=owner.id)
        clusterer = _clusterer(store, tmp_path)

        _add_evidence(store, owner, cluster.id, conversations=5, per_conversation=1)
        adapter_id = clusterer.promote_skill_adapters(min_size=5)[0]
        born = store.get_artifact(adapter_id).schema["lifecycle"]["evidence"]
        assert born["positive_events"] == 5

        _add_evidence(store, owner, cluster.id, conversations=6, per_conversation=3)
        clusterer.promote_skill_adapters(min_size=5)

        now = store.get_artifact(adapter_id).schema["lifecycle"]["evidence"]
        assert now["positive_events"] == 23, (
            f"the adapter still reports {now['positive_events']} events; the "
            "recorded evidence was frozen at the moment the skill was born"
        )
        assert now["conversations"] == 11
        # Recorded so the thresholds can later be chosen from real
        # distributions rather than guessed a second time.
        for key in ("first_event_at", "last_event_at", "span_hours"):
            assert key in now, f"lifecycle.evidence does not record {key}"


class TestPersonalEvidenceMustSpanTime:
    """A personal skill's only independence axis besides the conversation.

    One user cannot supply independent contributors, so the substitute is
    that the evidence did not all arrive in one sitting.
    """

    def test_one_sitting_does_not_earn_personal_weights(self, tmp_path):
        store = get_test_store()
        owner = _user(store, "solo")
        cluster = _cluster(store, user_id=owner.id)
        # Plenty of answers across plenty of threads, all created now.
        _add_evidence(store, owner, cluster.id, conversations=7, per_conversation=3)

        promoted = _clusterer(store, tmp_path).promote_skill_adapters(min_size=5)

        adapter = store.get_artifact(promoted[0])
        assert _jobs_for(store, adapter.id) == [], (
            "21 answers across 7 threads all created within minutes earned "
            "weights; that is one binge, not repeated independent success"
        )
        assert adapter.schema["lifecycle"]["evidence"]["span_hours"] < 1

    def test_the_same_evidence_spread_over_days_earns_weights(self, tmp_path):
        store = get_test_store()
        owner = _user(store, "solo")
        cluster = _cluster(store, user_id=owner.id)
        _add_evidence(
            store, owner, cluster.id, conversations=4, per_conversation=3,
            age_hours=96,
        )
        _add_evidence(store, owner, cluster.id, conversations=3, per_conversation=3)

        promoted = _clusterer(store, tmp_path).promote_skill_adapters(min_size=5)

        adapter = store.get_artifact(promoted[0])
        assert adapter.schema["lifecycle"]["evidence"]["span_hours"] >= 48
        assert len(_jobs_for(store, adapter.id)) == 1, (
            "the same evidence spread over four days did not earn weights"
        )


class TestASharedContributorHasToActuallyContribute:
    """`users >= 3` counting bare appearances lets 18/1/1 pass."""

    def test_two_token_contributors_do_not_qualify(self, tmp_path):
        store = get_test_store()
        cluster = _cluster(store)
        # Alice carries the cluster; Bob and Carol appear once each.
        _add_evidence(store, _user(store, "alice"), cluster.id,
                      conversations=6, per_conversation=3)
        _add_evidence(store, _user(store, "bob"), cluster.id,
                      conversations=1, per_conversation=1)
        _add_evidence(store, _user(store, "carol"), cluster.id,
                      conversations=1, per_conversation=1)

        promoted = _clusterer(store, tmp_path).promote_skill_adapters(min_size=5)

        adapter = store.get_artifact(promoted[0])
        evidence = adapter.schema["lifecycle"]["evidence"]
        assert evidence["users"] == 3, "three people did contribute something"
        assert evidence["qualifying_contributors"] == 1, (
            "a single answer in a single thread counted as an independent "
            "demonstration of the behaviour"
        )
        assert _jobs_for(store, adapter.id) == [], (
            "a tenant-wide adapter was queued on evidence that is "
            "essentially one person's"
        )

    def test_three_real_contributors_qualify(self, tmp_path):
        store = get_test_store()
        cluster = _cluster(store)
        for label in ("alice", "bob", "carol"):
            _add_evidence(store, _user(store, label), cluster.id,
                          conversations=3, per_conversation=3)

        promoted = _clusterer(store, tmp_path).promote_skill_adapters(min_size=5)

        adapter = store.get_artifact(promoted[0])
        assert adapter.schema["lifecycle"]["evidence"]["qualifying_contributors"] == 3
        assert len(_jobs_for(store, adapter.id)) == 1

    def test_a_shared_skill_needs_no_time_span(self, tmp_path):
        """Contributors are the shared scope's independence axis.

        Requiring a span as well would make a shared skill strictly harder
        than a personal one on every axis, which is not the intent.
        """
        store = get_test_store()
        cluster = _cluster(store)
        for label in ("alice", "bob", "carol"):
            _add_evidence(store, _user(store, label), cluster.id,
                          conversations=3, per_conversation=3)

        promoted = _clusterer(store, tmp_path).promote_skill_adapters(min_size=5)

        adapter = store.get_artifact(promoted[0])
        assert adapter.schema["lifecycle"]["evidence"]["span_hours"] < 1
        assert len(_jobs_for(store, adapter.id)) == 1


class TestTheSharedPromptRungAlsoNeedsMoreThanOnePerson:
    """One person's evidence must not change other people's behaviour.

    Nothing here crosses a tenant boundary, so this is about scope rather
    than isolation: a shared prompt skill is delivered to users who never
    contributed to it, and one contributor is not evidence that anyone
    else benefits.
    """

    def test_one_contributor_does_not_create_a_shared_prompt_skill(self, tmp_path):
        store = get_test_store()
        cluster = _cluster(store)  # no user_id: the shared kind
        _add_evidence(store, _user(store, "a"), cluster.id,
                      conversations=6, per_conversation=1)

        promoted = _clusterer(store, tmp_path).promote_skill_adapters(min_size=5)

        assert promoted == [], (
            "one user's events created a tenant-wide prompt skill that every "
            "other user in the tenant now receives"
        )

    def test_two_contributors_do_create_a_shared_prompt_skill(self, tmp_path):
        store = get_test_store()
        cluster = _cluster(store)
        for label in ("a", "b"):
            _add_evidence(store, _user(store, label), cluster.id,
                          conversations=3, per_conversation=1)

        promoted = _clusterer(store, tmp_path).promote_skill_adapters(min_size=5)

        assert len(promoted) == 1, "two contributors should earn the prompt rung"
        adapter = store.get_artifact(promoted[0])
        assert adapter.schema["scope"] == "tenant"
        assert _jobs_for(store, adapter.id) == [], "the weights rung needs three"

    def test_a_solo_install_still_gets_its_own_private_prompt_skill(self, tmp_path):
        """The rule above must not cost a one-person install anything."""
        store = get_test_store()
        owner = _user(store, "solo")
        cluster = _cluster(store, user_id=owner.id)
        _add_evidence(store, owner, cluster.id, conversations=6, per_conversation=1)

        promoted = _clusterer(store, tmp_path).promote_skill_adapters(min_size=5)

        assert len(promoted) == 1
        adapter = store.get_artifact(promoted[0])
        assert adapter.visibility == "private"
        assert adapter.schema["scope"] == "per-user"


class TestTheJobsEvidenceIsWhatGetsTrained:
    def test_training_uses_the_events_recorded_on_the_job(self, tmp_path):
        """Reproducibility, and one definition of positive rather than two.

        The promotion gate counts an event as positive on feedback *or* a
        positive score; the trainer used to re-query on feedback alone. The
        two can disagree, and then the run trains on a different set than
        the one that earned it.
        """
        store = get_test_store()
        owner = _user(store, "solo")
        cluster = _cluster(store, user_id=owner.id)
        clusterer = _clusterer(store, tmp_path)
        _add_evidence(
            store, owner, cluster.id, conversations=7, per_conversation=3,
            age_hours=96,
        )

        # One more that only a score marks as positive: the gate counts it,
        # a feedback-only re-query would not see it.
        convo = store.create_conversation(owner.id, title="scored")
        store.append_message(convo.id, "user", "user", "q")
        reply = store.append_message(convo.id, "assistant", "assistant", "a")
        scored = store.record_preference_event(
            owner.id, convo.id, reply.id, "neutral", score=1.0,
            context_embedding=[0.1] * 64, cluster_id=cluster.id,
            context_text="context",
        )

        adapter_id = clusterer.promote_skill_adapters(min_size=5)[0]
        job = _jobs_for(store, adapter_id)[0]

        assert scored.id in job.preference_event_ids, (
            "the score-positive event counted toward the gate but was not "
            "recorded on the job, so the trainer cannot reproduce the set"
        )
        # Resolved through the path training actually takes. An earlier
        # version of this test called a helper that loaded the ids directly,
        # so it asserted nothing about what a run would train on.
        selected = TrainingService(store, str(tmp_path)).resolve_job_evidence(
            job, user_id=owner.id
        )
        assert {e.id for e in selected} == set(job.preference_event_ids), (
            "training resolved a different set of events than the job "
            "recorded; the evidence that earned the weights is not the "
            "evidence they were fitted to"
        )
        assert scored.id in {e.id for e in selected}, (
            "the score-positive event was dropped by a feedback-only "
            "re-query, which is the predicate disagreement itself"
        )
