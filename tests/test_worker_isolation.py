"""Under xdist, every destructive resource belongs to exactly one worker.

The suite wipes its database before every test. That is what makes tests
independent of each other, and it is only true while one process owns the
database. Point four workers at one, and `TRUNCATE every table` stops being
isolation and becomes every test deleting every other test's rows — not
flakiness, a guarantee inverted.

Most of the isolation is free, and measured rather than assumed: under xdist
each worker is its own process, so the module-level temp root, the scratch
Postgres and the scratch Redis are already per-worker. What is not free is the
case where the services are supplied from outside — `TEST_DATABASE_URL`,
`TEST_REDIS_URL` — because then every worker is handed the same one.

These tests run pytest inside pytest, against services stood up for the
occasion. Asserting that the derivation *functions* return different strings
would only prove the code meant well; what has to hold is that a real parallel
run leaves the base database and the base Redis exactly as it found them.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

#: Set by the outer test when it runs this file inside a nested pytest. The
#: probe below is not a test of the product — it is a worker reporting what it
#: was given — so it does nothing in an ordinary run.
_PROBE_OUT = os.environ.get("LIMINALLM_HARNESS_PROBE")


@pytest.mark.skipif(not _PROBE_OUT, reason="only runs inside the nested harness run")
def test_probe_records_what_this_worker_was_given():
    """Report this process's resources, and use them the way a test would.

    The reporting is half of it. The other half is the fixed identifiers:
    every worker registers the *same* email address and writes the *same*
    cache key. The address is unique-constrained, so on a shared database
    the second worker to arrive fails — this succeeding in every worker is
    end-to-end evidence of isolation, rather than evidence that some
    derived strings differed.
    """
    from liminallm.config import get_settings
    from liminallm.service.runtime import get_runtime

    runtime = get_runtime()
    runtime.store.create_user(email="fixed@example.com", tenant_id="public")
    if runtime.cache is not None:
        import redis as redis_

        client = redis_.Redis.from_url(get_settings().redis_url, decode_responses=True)
        try:
            client.set("harness:fixed-key", os.environ.get("PYTEST_XDIST_WORKER", ""))
        finally:
            client.close()

    with runtime.store._connect() as conn:
        cloned = conn.execute(
            "SELECT to_regclass('public.harness_sentinel') AS t"
        ).fetchone()["t"]

    with open(_PROBE_OUT, "a") as fh:
        fh.write(
            json.dumps(
                {
                    "worker": os.environ.get("PYTEST_XDIST_WORKER", ""),
                    "database_url": os.environ["DATABASE_URL"],
                    "fs_root": get_settings().shared_fs_root,
                    "redis_url": get_settings().redis_url or "",
                    "inherited_sentinel_table": bool(cloned),
                }
            )
            + "\n"
        )


class _External:
    """A Postgres and a Redis standing in for services supplied from outside.

    With sentinels in both, so "the base was left alone" is a question with an
    answer rather than an absence of evidence.
    """

    def __init__(self):
        from tests.harness import ScratchPostgres, ScratchRedis

        self.pg = ScratchPostgres()
        self.redis = ScratchRedis()
        self.url = None
        self.redis_url = None

    @property
    def available(self) -> bool:
        return self.pg.available and self.redis.available

    def __enter__(self):
        import psycopg
        import redis

        from tests.harness import apply_schema

        self.url = self.pg.start()
        self.redis_url = self.redis.start()
        # What `scripts/migrate.sh` would have left behind, plus a fact that
        # only a clone can inherit.
        apply_schema(self.url, embedding_dim=64)
        with psycopg.connect(self.url, autocommit=True) as conn:
            conn.execute("CREATE TABLE harness_sentinel (note text)")
            conn.execute("INSERT INTO harness_sentinel VALUES ('base survived')")
        client = redis.Redis.from_url(self.redis_url, decode_responses=True)
        try:
            client.set("harness:sentinel", "base db0 survived")
        finally:
            client.close()
        return self

    def __exit__(self, *exc):
        self.pg.stop()
        self.redis.stop()

    def run_pytest(self, *args, probe_out=None, prepared=True, env_extra=None):
        env = {
            k: v
            for k, v in os.environ.items()
            if k
            not in {
                "DATABASE_URL",
                "TEST_PG_PORT",
                "TEST_REDIS_PORT",
                "SHARED_FS_ROOT",
                "LIMINALLM_TEST_RUN",
                "TEST_SCHEMA_PREPARED",
            }
        }
        env.update(
            TEST_DATABASE_URL=self.url,
            TEST_REDIS_URL=self.redis_url,
            EMBEDDING_VECTOR_DIM="64",
            ALLOW_REDIS_FALLBACK_DEV="true",
        )
        if prepared:
            env["TEST_SCHEMA_PREPARED"] = "true"
        if probe_out:
            env["LIMINALLM_HARNESS_PROBE"] = probe_out
        env.update(env_extra or {})
        return subprocess.run(
            [
                sys.executable, "-m", "pytest", "-q", "--no-header",
                "-p", "no:cacheprovider", "-p", "no:randomly", *args,
            ],
            cwd=ROOT, env=env, capture_output=True, text=True, timeout=900,
        )

    def databases(self) -> list[str]:
        import psycopg

        with psycopg.connect(self.url, autocommit=True) as conn:
            # A bare psycopg connection yields tuples; the store's dict rows
            # come from a row factory this does not set.
            return [
                r[0]
                for r in conn.execute("SELECT datname FROM pg_database").fetchall()
            ]

    def base_sentinel(self):
        import psycopg

        with psycopg.connect(self.url, autocommit=True) as conn:
            row = conn.execute("SELECT note FROM harness_sentinel").fetchone()
        return row and row[0]

    def redis_db0(self):
        import redis

        client = redis.Redis.from_url(self.redis_url, decode_responses=True)
        try:
            return client.get("harness:sentinel"), sorted(client.keys("*"))
        finally:
            client.close()


def _external_or_skip():
    ext = _External()
    if not ext.available:
        pytest.skip("needs initdb and redis-server to stand up external services")
    return ext


PROBE = "tests/test_worker_isolation.py::test_probe_records_what_this_worker_was_given"


@pytest.mark.slow  # stands up a Postgres and a Redis, then runs pytest in them
class TestAWorkerOwnsItsResources:
    def test_every_destructive_resource_is_derived_from_the_worker(self, tmp_path):
        """Database, filesystem root and Redis database, all three.

        `--dist each` rather than the default scheduler, so both workers run
        the probe. With `load` the two reports could land on one worker and the
        test would pass having compared a worker with itself.
        """
        out = tmp_path / "probe.jsonl"
        with _external_or_skip() as ext:
            done = ext.run_pytest(
                "-n", "2", "--dist", "each", PROBE, probe_out=str(out)
            )
            assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-2000:]
            reports = [json.loads(line) for line in out.read_text().splitlines()]
            base_db = ext.url.rsplit("/", 1)[-1]

            assert len(reports) == 2, f"expected one report per worker: {reports}"
            workers = {r["worker"] for r in reports}
            assert workers == {"gw0", "gw1"}, workers

            databases = {r["database_url"].rsplit("/", 1)[-1] for r in reports}
            assert len(databases) == 2, f"the workers shared a database: {databases}"
            assert base_db not in databases, (
                f"a worker was given the database this run was handed: {databases}"
            )
            for name in databases:
                assert name.startswith(base_db), name

            roots = {r["fs_root"] for r in reports}
            assert len(roots) == 2, f"the workers shared a filesystem root: {roots}"
            for report in reports:
                assert report["worker"] in report["fs_root"], report

            redis_dbs = {r["redis_url"].rsplit("/", 1)[-1] for r in reports}
            assert len(redis_dbs) == 2, f"the workers shared a Redis database: {redis_dbs}"
            assert "0" not in redis_dbs, (
                f"a worker was given the base Redis database: {redis_dbs}"
            )

    def test_the_base_database_is_left_exactly_as_it_was(self, tmp_path):
        """It is truncated before every test — but not this one."""
        with _external_or_skip() as ext:
            done = ext.run_pytest(
                "-n", "2", "--dist", "each", PROBE,
                probe_out=str(tmp_path / "probe.jsonl"),
            )
            assert done.returncode == 0, done.stdout[-3000:]
            assert ext.base_sentinel() == "base survived", (
                "a worker truncated the database this run was handed, which is "
                "every other worker's rows as well as this sentinel"
            )
            leftovers = [d for d in ext.databases() if d != "postgres" and "_xd_" in d]
            assert leftovers == [], f"workers left their databases behind: {leftovers}"

    def test_the_base_redis_database_is_left_exactly_as_it_was(self, tmp_path):
        with _external_or_skip() as ext:
            done = ext.run_pytest(
                "-n", "2", "--dist", "each", PROBE,
                probe_out=str(tmp_path / "probe.jsonl"),
            )
            assert done.returncode == 0, done.stdout[-3000:]
            sentinel, keys = ext.redis_db0()
            assert sentinel == "base db0 survived", (
                "a worker flushed the Redis database this run was handed"
            )
            assert keys == ["harness:sentinel"], (
                f"a worker wrote into the base Redis database: {keys}"
            )

    def test_a_prepared_database_is_cloned_rather_than_rebuilt(self, tmp_path):
        """The invariant CI's `TEST_SCHEMA_PREPARED` exists to protect.

        CI runs `scripts/migrate.sh` and then sets the flag so conftest cannot
        quietly repair a deploy command that does nothing. A worker that built
        its own schema instead of cloning would restore exactly the hole the
        flag was added to close — every worker rebuilding from `schema.sql`,
        and the suite green over a `migrate.sh` that never ran.

        The sentinel table is not in `schema.sql`. Only a clone has it.
        """
        out = tmp_path / "probe.jsonl"
        with _external_or_skip() as ext:
            done = ext.run_pytest(
                "-n", "2", "--dist", "each", PROBE, probe_out=str(out)
            )
            assert done.returncode == 0, done.stdout[-3000:]
            reports = [json.loads(line) for line in out.read_text().splitlines()]
            assert reports and all(r["inherited_sentinel_table"] for r in reports), (
                "a worker's database does not have the prepared database's "
                "sentinel, so it was built from schema.sql rather than cloned "
                "— and TEST_SCHEMA_PREPARED no longer means anything"
            )

    def test_serial_runs_are_untouched(self, tmp_path):
        """No worker, no derivation: the database it was handed, directly."""
        out = tmp_path / "probe.jsonl"
        with _external_or_skip() as ext:
            done = ext.run_pytest(PROBE, probe_out=str(out))
            assert done.returncode == 0, done.stdout[-3000:]
            reports = [json.loads(line) for line in out.read_text().splitlines()]
            assert len(reports) == 1 and reports[0]["worker"] == ""
            assert reports[0]["database_url"] == ext.url, (
                "a serial run was given a derived database instead of the one "
                "it was configured with"
            )
            assert reports[0]["redis_url"] == ext.redis_url
            assert [d for d in ext.databases() if "_xd_" in d] == [], (
                "a serial run created a worker database"
            )

    def test_every_test_has_the_same_name_twice(self):
        """Collect the suite twice and compare, name for name.

        xdist requires every worker to collect an identical set, and each
        worker collects independently — so a test whose id is not a function of
        the source refuses to run in parallel at all. Found that way: a
        parametrization built two of its cases with `uuid.uuid4()` at
        collection time, and four workers produced four different suites.

        The parallel lane would have failed loudly, so this is not what catches
        it first. It is here because the instability is worth naming on its
        own: a test whose id changes every run cannot be re-run from a failure
        report, and `pytest ...::test_x[309601fa-...]` is a command that works
        exactly once.
        """
        def collect() -> list[str]:
            done = subprocess.run(
                [sys.executable, "-m", "pytest", "--collect-only", "-q",
                 "-p", "no:cacheprovider", "-p", "no:randomly", "tests/"],
                cwd=ROOT, capture_output=True, text=True, timeout=900,
            )
            assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-2000:]
            return [ln for ln in done.stdout.splitlines() if "::" in ln]

        first, second = collect(), collect()
        assert first, "collected nothing"
        drifted = sorted(set(first) ^ set(second))
        assert not drifted, (
            "these tests were named differently on two collections of the same "
            "source, so they cannot be re-run from a report and cannot be "
            "distributed across workers:\n" + "\n".join(drifted[:10])
        )

    def test_cross_replica_proofs_still_hold_inside_a_worker(self):
        """Worker isolation must not weaken what these tests prove.

        The two simulated replicas in an advisory-lock test share their
        worker's Postgres, and the actors in a path-race test share its
        filesystem root, so both still contend exactly as before. Isolation
        keeps unrelated tests out; it does not stand between a test and
        itself. Nothing here is serial-marked, and it should not need to be.
        """
        with _external_or_skip() as ext:
            done = ext.run_pytest(
                "-n", "2",
                "tests/test_account_erasure.py::TestSubordinateSweepsDoNotUndercutTheGrace",
                "tests/test_account_erasure.py::TestAnInFlightRequestCannotUndoTheErasure",
            )
            assert done.returncode == 0, (
                "advisory-lock and file-lock proofs stopped holding under "
                "xdist:\n" + done.stdout[-4000:]
            )
