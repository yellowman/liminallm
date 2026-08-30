"""Two replicas deciding one cluster's rung must reach one answer.

SPEC §22 puts several replicas on one Postgres, and §5.5.2 promises that
enqueueing weights is idempotent per adapter. Neither promise survives a
process-local "read, then write": both replicas read no adapter, both
create one; both read no job, both queue one. The only arbiter they share
is the database.

These drive two genuinely independent `PostgresStore` instances - separate
connection pools, the way two replicas are separate - against the same
database, and start them together on a barrier so the interleaving is real
rather than described.
"""

from __future__ import annotations

import threading
import uuid

from liminallm.service.clustering import SemanticClusterer
from liminallm.service.training import TrainingService
from liminallm.storage.postgres import PostgresStore
from tests.harness import get_test_store


def _second_store(store) -> PostgresStore:
    """A replica: same database, its own pool."""
    return PostgresStore(store.dsn, fs_root=str(store.fs_root))


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


def _cluster(store, user_id):
    return store.upsert_semantic_cluster(
        user_id=user_id, centroid=[0.1] * 64, size=10,
        label="routing debug", description="start from the routing table",
    )


def _run_both(first, second, call):
    barrier = threading.Barrier(2)
    errors: list = []

    def run(store):
        try:
            barrier.wait(timeout=30)
            call(store)
        except Exception as exc:  # noqa: BLE001 - re-raised after the join
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(s,)) for s in (first, second)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)
    return errors


def _adapters_for(store, cluster_id):
    import psycopg

    with psycopg.connect(store.dsn, autocommit=True) as conn:
        rows = conn.execute(
            "SELECT id FROM artifact WHERE type = 'adapter' "
            "AND schema->>'cluster_id' = %s",
            (cluster_id,),
        ).fetchall()
    return [r[0] for r in rows]


class TestOnlyOneReplicaCreatesThePromptRung:
    def test_two_replicas_produce_one_cluster_adapter(self, tmp_path):
        store = get_test_store()
        owner = _user(store, "solo")
        cluster = _cluster(store, owner.id)
        _seed(store, owner, cluster.id, conversations=6, per_conversation=1)

        other = _second_store(store)
        try:
            def promote(s):
                SemanticClusterer(
                    s, llm=None, training=TrainingService(s, str(tmp_path))
                ).promote_skill_adapters(min_size=5)

            errors = _run_both(store, other, promote)
            assert not errors, f"a replica raised: {errors}"

            bound = _adapters_for(store, cluster.id)
            assert len(bound) == 1, (
                f"{len(bound)} adapters bound to one cluster; two replicas "
                "both read 'no adapter' and both created one"
            )
        finally:
            other.close_pool()


class TestOnlyOneReplicaQueuesTheWeightsJob:
    def test_two_replicas_produce_one_training_job(self, tmp_path):
        store = get_test_store()
        owner = _user(store, "solo")
        cluster = _cluster(store, owner.id)
        # Enough evidence that both replicas will judge the bar cleared.
        evidence = _seed(store, owner, cluster.id, conversations=4,
                         per_conversation=3, age_hours=96)
        evidence += _seed(store, owner, cluster.id, conversations=4,
                          per_conversation=3)

        # Born on the prompt rung first, with no trainer, so the race below
        # is purely about the enqueue.
        adapter_id = SemanticClusterer(
            store, llm=None, training=None
        ).promote_skill_adapters(min_size=5)[0]
        assert [j for j in store.list_training_jobs()
                if j.adapter_id == adapter_id] == [], (
            "the fixture queued a job before the race began"
        )

        # The enqueue primitive itself, raced directly. Going through
        # promote_skill_adapters would not exercise it: update_artifact takes
        # the adapter's row lock on the way past, which serializes the two
        # callers by accident and hides whether the insert is atomic. That
        # accident is not a guarantee - the row lock is released before the
        # enqueue - so the primitive has to be witnessed on its own.
        other = _second_store(store)
        try:
            created: list = []

            def enqueue(s):
                created.append(
                    s.create_training_job_if_absent(
                        user_id=owner.id,
                        adapter_id=adapter_id,
                        preference_event_ids=evidence,
                    )
                )

            errors = _run_both(store, other, enqueue)
            assert not errors, f"a replica raised: {errors}"

            winners = [job for job in created if job is not None]
            assert len(winners) == 1, (
                f"{len(winners)} replicas believed they queued the job"
            )
            jobs = [j for j in store.list_training_jobs()
                    if j.adapter_id == adapter_id]
            assert len(jobs) == 1, (
                f"{len(jobs)} training jobs for one adapter; check-then-insert "
                "is a race both replicas win"
            )
        finally:
            other.close_pool()
