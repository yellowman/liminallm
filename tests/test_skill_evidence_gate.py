"""Weights are earned by independent evidence, not by a raw event count.

Twenty thumbs from one long debugging session and twenty spread across
twenty unrelated conversations are the same number and not the same
evidence. The first is one episode rated repeatedly; the second is a
behaviour that kept working. Only the second should move a skill off the
prompt rung, because the prompt rung is cheap and reversible while weights
are neither.

The gate is deliberately not "more than one user". A single-user install
must still be able to learn a skill, or it can never learn one at all.
What separates the two kinds of skill is scope, not capability:

* a personal skill trains from one user's repeated evidence and stays
  private to them;
* a shared skill pools several users' evidence and is visible across the
  tenant, so it additionally has to show that several users independently
  benefited.
"""

from __future__ import annotations

import uuid

from liminallm.service.clustering import SemanticClusterer
from liminallm.service.training import TrainingService
from tests.harness import get_test_store


def _user(store, label: str):
    return store.create_user(email=f"{uuid.uuid4().hex[:8]}_{label}@t.local")


def _rate(store, user, convo_id, message_id, cluster_id):
    return store.record_preference_event(
        user.id,
        convo_id,
        message_id,
        "positive",
        corrected_text="use tabs",
        context_embedding=[0.1] * 64,
        cluster_id=cluster_id,
        context_text="context",
    )


def _seed(store, user, cluster_id, *, conversations: int, per_conversation: int):
    """`conversations` threads, each with `per_conversation` rated answers."""
    for _ in range(conversations):
        convo = store.create_conversation(user.id, title="t")
        for i in range(per_conversation):
            store.append_message(convo.id, "user", "user", f"question {i}")
            reply = store.append_message(convo.id, "assistant", "assistant", f"a {i}")
            _rate(store, user, convo.id, reply.id, cluster_id)


def _cluster(store, *, user_id=None):
    return store.upsert_semantic_cluster(
        user_id=user_id,
        centroid=[0.1] * 64,
        size=10,
        label="routing debug",
        description="start from the routing table, not the logs",
    )


def _clusterer(store, tmp_path):
    return SemanticClusterer(store, llm=None, training=TrainingService(store, str(tmp_path)))


def _jobs_for(store, adapter_id):
    return [j for j in store.list_training_jobs() if j.adapter_id == adapter_id]


class TestAPersonalSkillNeedsIndependentEvidence:
    """One user is enough. One session is not."""

    def test_one_long_session_does_not_earn_weights(self, tmp_path):
        """Twenty ratings, one conversation: a single episode, rated a lot."""
        store = get_test_store()
        owner = _user(store, "solo")
        cluster = _cluster(store, user_id=owner.id)
        _seed(store, owner, cluster.id, conversations=1, per_conversation=20)

        promoted = _clusterer(store, tmp_path).promote_skill_adapters(min_size=5)

        assert len(promoted) == 1, "the prompt rung is cheap and should still happen"
        adapter = store.get_artifact(promoted[0])
        assert adapter.schema["mode"] == "prompt"
        assert _jobs_for(store, adapter.id) == [], (
            "20 ratings inside one conversation enqueued weights training; that "
            "is one episode rated repeatedly, not a skill that kept working"
        )

    def test_evidence_spread_across_conversations_earns_weights(self, tmp_path):
        """The single-user install has to be able to learn."""
        store = get_test_store()
        owner = _user(store, "solo")
        cluster = _cluster(store, user_id=owner.id)
        _seed(store, owner, cluster.id, conversations=7, per_conversation=3)

        promoted = _clusterer(store, tmp_path).promote_skill_adapters(min_size=5)

        adapter = store.get_artifact(promoted[0])
        assert adapter.schema["scope"] == "per-user"
        assert adapter.visibility == "private"
        assert len(_jobs_for(store, adapter.id)) == 1, (
            "21 ratings across 7 conversations did not earn weights; a "
            "single-user install can then never train a skill at all"
        )

    def test_repeat_ratings_on_one_answer_do_not_count_twenty_times(self, tmp_path):
        """Distinct targets, not distinct rows.

        Rating the same answer over and over is one piece of evidence
        however many rows it writes.
        """
        store = get_test_store()
        owner = _user(store, "solo")
        cluster = _cluster(store, user_id=owner.id)
        # Six conversations clears the conversation bar, but every rating in
        # each one lands on that conversation's single answer.
        for _ in range(6):
            convo = store.create_conversation(owner.id, title="t")
            store.append_message(convo.id, "user", "user", "question")
            reply = store.append_message(convo.id, "assistant", "assistant", "answer")
            for _ in range(4):
                _rate(store, owner, convo.id, reply.id, cluster.id)

        promoted = _clusterer(store, tmp_path).promote_skill_adapters(min_size=5)

        adapter = store.get_artifact(promoted[0])
        assert _jobs_for(store, adapter.id) == [], (
            "24 ratings over 6 answers earned weights; the count has to be of "
            "distinct answers, or re-rating one reply reaches the bar alone"
        )


class TestASharedSkillAlsoNeedsSeveralUsers:
    """Scope is what the extra requirement buys, not capability."""

    def test_two_users_do_not_earn_a_tenant_wide_skill(self, tmp_path):
        store = get_test_store()
        cluster = _cluster(store)  # no user_id: a cross-user cluster
        for label in ("a", "b"):
            _seed(store, _user(store, label), cluster.id,
                  conversations=6, per_conversation=3)

        promoted = _clusterer(store, tmp_path).promote_skill_adapters(min_size=5)

        adapter = store.get_artifact(promoted[0])
        assert adapter.schema["scope"] == "tenant"
        assert _jobs_for(store, adapter.id) == [], (
            "two users earned a tenant-wide skill; shared weights are served "
            "to people who never contributed evidence for them"
        )

    def test_three_users_earn_a_tenant_wide_skill(self, tmp_path):
        store = get_test_store()
        cluster = _cluster(store)
        for label in ("a", "b", "c"):
            _seed(store, _user(store, label), cluster.id,
                  conversations=3, per_conversation=3)

        promoted = _clusterer(store, tmp_path).promote_skill_adapters(min_size=5)

        adapter = store.get_artifact(promoted[0])
        assert adapter.schema["scope"] == "tenant"
        assert len(_jobs_for(store, adapter.id)) == 1, (
            "27 ratings from 3 users across 9 conversations did not earn a "
            "shared skill"
        )


class TestTheEvidenceIsRecorded:
    def test_the_adapter_records_what_it_was_measured_on(self, tmp_path):
        """A skill that stayed on the prompt rung has to say why.

        Without this the only way to find out is to re-run the query that
        rejected it.
        """
        store = get_test_store()
        owner = _user(store, "solo")
        cluster = _cluster(store, user_id=owner.id)
        _seed(store, owner, cluster.id, conversations=1, per_conversation=20)

        promoted = _clusterer(store, tmp_path).promote_skill_adapters(min_size=5)

        lifecycle = store.get_artifact(promoted[0]).schema["lifecycle"]
        evidence = lifecycle.get("evidence")
        assert evidence, "the adapter does not record the evidence it was judged on"
        assert evidence["positive_events"] == 20
        assert evidence["conversations"] == 1
        assert evidence["messages"] == 20
        assert evidence["users"] == 1
