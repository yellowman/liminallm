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
import time
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


#: Set by the outer test for the run that must survive another run's flush.
#: Three paths: where to write the sentinel's name, when it is in place, and
#: when it may check.
_HOLD = os.environ.get("LIMINALLM_HARNESS_HOLD")


@pytest.mark.skipif(not _HOLD, reason="only runs inside the nested harness run")
def test_probe_holds_redis_state_while_another_run_flushes():
    """Write into this run's Redis database, wait, and look again.

    The waiting is the test. Another pytest invocation starts while this one
    is paused and empties what it believes is its own database before every
    test — so if the two runs were handed the same number, this comes back to
    nothing.
    """
    import time as time_

    from liminallm.config import get_settings
    from liminallm.service.runtime import get_runtime

    ready, release = Path(_HOLD + ".ready"), Path(_HOLD + ".release")
    runtime = get_runtime()
    assert runtime.cache is not None, "this probe needs a real Redis"

    import redis as redis_

    client = redis_.Redis.from_url(get_settings().redis_url, decode_responses=True)
    try:
        client.set("harness:held", "this run's own state")
        Path(_HOLD).write_text(get_settings().redis_url)
        ready.write_text("ready")
        deadline = time_.monotonic() + 120
        while not release.exists() and time_.monotonic() < deadline:
            time_.sleep(0.05)
        assert release.exists(), "the other run never finished"
        assert client.get("harness:held") == "this run's own state", (
            "another pytest invocation flushed this run's Redis database. Two "
            "runs were handed the same number, and each one empties it before "
            "every test believing it owns it."
        )
    finally:
        client.close()


#: Set for the run that must notice it no longer owns its database.
_LOSE = os.environ.get("LIMINALLM_HARNESS_LOSE")


@pytest.mark.skipif(not _LOSE, reason="only runs inside the nested harness run")
def test_probe_gives_its_lease_away():
    """Hand this run's claim to somebody else, then let it try another test.

    Standing in for the schedule where a lease expires and another run takes
    the number: the outcome is the same, and this one can be forced. The next
    test's reset is what has to refuse — this one does nothing destructive
    itself.
    """
    import redis as redis_

    from liminallm.config import get_settings
    from tests.harness import REDIS_LEASE_PREFIX, redis_database_index

    url = get_settings().redis_url
    index = redis_database_index(url)
    assert index != 0, "this probe expects a leased database"

    ledger = redis_.Redis.from_url(
        url.rsplit("/", 1)[0] + "/0", decode_responses=True
    )
    mine = redis_.Redis.from_url(url, decode_responses=True)
    try:
        ledger.set(f"{REDIS_LEASE_PREFIX}:{index}", "another-run:gw0", ex=900)
        mine.set("harness:taken-over", "the new holder's state")
    finally:
        ledger.close()
        mine.close()


_OUTLIVE = os.environ.get("LIMINALLM_HARNESS_OUTLIVE")


@pytest.mark.skipif(not _OUTLIVE, reason="only runs inside the nested harness run")
@pytest.mark.parametrize("step", [0, 1, 2, 3, 4])
def test_probe_runs_longer_than_a_short_lease(step):
    """Five tests spanning more than the shortened TTL.

    Several rather than one long one, because the refresh happens in the
    per-test reset: a single test that sleeps would never reach it, and would
    prove the opposite of what this is for.

    Five seconds against the caller's three-second TTL. The gap between two
    refreshes is a third of the TTL, so a working refresh has margin; the run
    as a whole outlasts it comfortably, so a reservation written once and
    never refreshed is definitely gone by the end. Both margins measured.
    """
    time.sleep(1.0)


class _External:
    """A Postgres and a Redis standing in for services supplied from outside.

    With sentinels in both, so "the base was left alone" is a question with an
    answer rather than an absence of evidence.
    """

    def __init__(self, redis_db: int = 0, db_in_query: bool = False):
        from tests.harness import ScratchPostgres, ScratchRedis

        self.pg = ScratchPostgres()
        self.redis = ScratchRedis()
        self.redis_db = redis_db
        self.db_in_query = db_in_query
        self.url = None
        self.redis_url = None

    @property
    def available(self) -> bool:
        return self.pg.available and self.redis.available

    @property
    def unavailable_reason(self) -> str:
        """Which of the two is missing, and why.

        A fixed "needs initdb" was wrong for every host that has initdb and
        lacks pgvector — the case this availability check was extended to
        catch, reported with the one explanation that could not apply to it.
        """
        if not self.pg.available:
            return self.pg.unavailable_reason
        if not self.redis.available:
            return "redis-server not available"
        return ""

    def __enter__(self):
        import psycopg
        import redis

        from tests.harness import apply_schema

        self.url = self.pg.start()
        # `TEST_REDIS_URL` is documented as "point at an existing service".
        # Nothing says that service's database must be 0, and using a numbered
        # one for tests is ordinary — so the fixture can name one.
        host = self.redis.start().rsplit("/", 1)[0]
        # Both spellings redis-py accepts. The query form is the one where the
        # path lies about which database the URL reaches.
        self.redis_url = (
            f"{host}/0?db={self.redis_db}"
            if self.db_in_query
            else f"{host}/{self.redis_db}"
        )
        # What `scripts/migrate.sh` would have left behind, plus a fact that
        # only a clone can inherit.
        apply_schema(self.url, embedding_dim=64)
        with psycopg.connect(self.url, autocommit=True) as conn:
            conn.execute("CREATE TABLE harness_sentinel (note text)")
            conn.execute("INSERT INTO harness_sentinel VALUES ('base survived')")
        client = redis.Redis.from_url(self.redis_url, decode_responses=True)
        try:
            client.set("harness:sentinel", "the base survived")
        finally:
            client.close()
        return self

    def __exit__(self, *exc):
        self.pg.stop()
        self.redis.stop()

    def env(self, *, probe_out=None, hold_out=None, prepared=True, extra=None):
        """The environment a nested run gets.

        `LIMINALLM_TEST_RUN` is dropped so each invocation mints its own — two
        runs sharing one would derive the same Postgres database name.
        `PYTEST_XDIST_WORKER` is dropped because an inherited one is a claim
        to belong to a run this process is not part of; conftest reads the
        worker id from `config` rather than the environment, so this is belt
        as well as braces.
        """
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
                "PYTEST_XDIST_WORKER",
                "LIMINALLM_HARNESS_PROBE",
                "LIMINALLM_HARNESS_OUTLIVE",
                "LIMINALLM_HARNESS_HOLD",
                "LIMINALLM_HARNESS_LOSE",
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
        if hold_out:
            env["LIMINALLM_HARNESS_HOLD"] = hold_out
        env.update(extra or {})
        return env

    def run_pytest(self, *args, probe_out=None, prepared=True, env_extra=None):
        env = self.env(probe_out=probe_out, prepared=prepared, extra=env_extra)
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

    def assert_control_keys_expire(self):
        """A control key with no expiry would be litter, not bookkeeping."""
        import redis

        from tests.harness import REDIS_BASE_PREFIX, REDIS_LEASE_PREFIX

        client = redis.Redis.from_url(self.redis_url, decode_responses=True)
        try:
            for prefix in (REDIS_LEASE_PREFIX, REDIS_BASE_PREFIX):
                for key in client.keys(f"{prefix}:*"):
                    assert client.ttl(key) > 0, f"{key} will never expire"
        finally:
            client.close()

    def base_redis(self):
        import redis

        client = redis.Redis.from_url(self.redis_url, decode_responses=True)
        try:
            return client.get("harness:sentinel"), sorted(client.keys("*"))
        finally:
            client.close()


def _data_keys(keys) -> list[str]:
    """Everything except the harness's own control keys.

    The lease ledger lives in database 0, which is also the caller's database
    when they named that one, so its bookkeeping shows up in a listing there.
    Those keys carry a TTL — `assert_control_keys_expire` checks it — so they
    are not data the run left behind. They are also not the sentinel.
    """
    from tests.harness import REDIS_BASE_PREFIX, REDIS_LEASE_PREFIX

    return sorted(
        key
        for key in keys
        if not key.startswith((REDIS_LEASE_PREFIX, REDIS_BASE_PREFIX))
    )


def _external_or_skip(redis_db: int = 0, db_in_query: bool = False):
    ext = _External(redis_db, db_in_query)
    if not ext.available:
        pytest.skip(f"cannot stand up external services: {ext.unavailable_reason}")
    return ext


class TestTheRedisLease:
    """The claim itself, at the level where each rule is one question.

    The end-to-end red above proves the property that matters and takes a
    minute and a half to do it. These are the rules it rests on, asked
    directly, so a broken one says which.
    """

    @pytest.fixture
    def server(self):
        from tests.harness import ScratchRedis

        redis_server = ScratchRedis()
        if not redis_server.available:
            pytest.skip("needs redis-server")
        url = redis_server.start()
        try:
            yield url
        finally:
            redis_server.stop()

    def test_a_claim_is_exclusive_and_the_ledger_says_who_holds_it(self, server):
        import redis

        from tests.harness import REDIS_LEASE_PREFIX, claim_redis_database

        first, first_index = claim_redis_database(server, "run-a:gw0")
        second, second_index = claim_redis_database(server, "run-b:gw0")
        assert first_index != second_index, "one database was claimed twice"
        assert first.endswith(f"/{first_index}") and second.endswith(f"/{second_index}")
        assert 0 not in (first_index, second_index), "database 0 holds the ledger"

        client = redis.Redis.from_url(server, decode_responses=True)
        try:
            assert client.get(f"{REDIS_LEASE_PREFIX}:{first_index}") == "run-a:gw0"
            assert client.ttl(f"{REDIS_LEASE_PREFIX}:{first_index}") > 0, (
                "a claim with no expiry outlives the run that made it, and the "
                "database never comes back"
            )
        finally:
            client.close()

    def test_a_release_frees_the_number_for_the_next_run(self, server):
        from tests.harness import claim_redis_database, release_redis_database

        _, index = claim_redis_database(server, "run-a:gw0")
        release_redis_database(server, index, "run-a:gw0")
        _, again = claim_redis_database(server, "run-b:gw0")
        assert again == index, (
            "a released database was not offered to the next run, so runs "
            "would stop working after fifteen of them"
        )

    def test_a_release_cannot_take_somebody_else_s_claim(self, server):
        """After a lease expires the number may already belong to somebody.

        A teardown that deletes by number alone would then hand a live run's
        database to a third one.
        """
        import redis

        from tests.harness import REDIS_LEASE_PREFIX, release_redis_database

        client = redis.Redis.from_url(server, decode_responses=True)
        try:
            client.set(f"{REDIS_LEASE_PREFIX}:1", "somebody-else:gw0")
            release_redis_database(server, 1, "run-a:gw0")
            assert client.get(f"{REDIS_LEASE_PREFIX}:1") == "somebody-else:gw0", (
                "a run released a database it did not hold"
            )
        finally:
            client.close()

    def test_a_renewal_pushes_the_expiry_out_again(self, server):
        """Renewed from the per-test reset, so a live run keeps its claim."""
        import redis

        from tests.harness import (
            REDIS_LEASE_PREFIX,
            claim_redis_database,
            renew_redis_database,
        )

        _, index = claim_redis_database(server, "run-a:gw0")
        client = redis.Redis.from_url(server, decode_responses=True)
        try:
            client.expire(f"{REDIS_LEASE_PREFIX}:{index}", 5)
            assert client.ttl(f"{REDIS_LEASE_PREFIX}:{index}") <= 5
            renew_redis_database(server, index, "run-a:gw0")
            assert client.ttl(f"{REDIS_LEASE_PREFIX}:{index}") > 60, (
                "a run that is still running let its claim run down"
            )
            renew_redis_database(server, index, "somebody-else:gw0")
            client.expire(f"{REDIS_LEASE_PREFIX}:{index}", 5)
            renew_redis_database(server, index, "somebody-else:gw0")
            assert client.ttl(f"{REDIS_LEASE_PREFIX}:{index}") <= 5, (
                "a run renewed a claim that was not its own"
            )
        finally:
            client.close()

    def test_running_out_of_databases_says_so(self, server):
        from tests.harness import REDIS_LEASE_SLOTS, claim_redis_database

        for n in REDIS_LEASE_SLOTS:
            claim_redis_database(server, f"run-{n}:gw0")
        with pytest.raises(RuntimeError) as caught:
            claim_redis_database(server, "one-too-many:gw0")
        assert "claimed by a test run" in str(caught.value)


def test_a_fixed_scratch_port_is_refused_under_xdist(monkeypatch):
    """Every worker would send its own cluster to the same port.

    The second one fails inside `pg_ctl`, which is loud but says nothing about
    why. This says why.
    """
    from tests.harness import _free_port

    # Cleared first: this test may itself be running inside a worker, and the
    # serial half has to be asked as a serial process would ask it.
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    monkeypatch.setenv("TEST_PG_PORT", "5439")
    assert _free_port("TEST_PG_PORT") == 5439, "a serial run may still pin a port"

    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw1")
    with pytest.raises(RuntimeError) as caught:
        _free_port("TEST_PG_PORT")
    assert "TEST_PG_PORT" in str(caught.value) and "worker" in str(caught.value)


PROBE = "tests/test_worker_isolation.py::test_probe_records_what_this_worker_was_given"
HOLD_PROBE = (
    "tests/test_worker_isolation.py"
    "::test_probe_holds_redis_state_while_another_run_flushes"
)
LOSE_PROBE = "tests/test_worker_isolation.py::test_probe_gives_its_lease_away"
SLOW_PROBE = "tests/test_worker_isolation.py::test_probe_runs_longer_than_a_short_lease"


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
            sentinel, keys = ext.base_redis()
            assert sentinel == "the base survived", (
                "a worker flushed the Redis database this run was handed"
            )
            assert _data_keys(keys) == ["harness:sentinel"], (
                f"a worker wrote into the base Redis database: {keys}"
            )
            ext.assert_control_keys_expire()

    def test_the_database_the_caller_named_is_never_leased(self):
        """`TEST_REDIS_URL` may name any database, not only 0.

        Offering `1..15` regardless of which one the URL named meant a caller
        who pointed the harness at `redis://host/1` had that exact database
        leased to the first worker — and then flushed before every test,
        because the lease said it was owned.

        Both spellings, because redis-py accepts two and they disagree: the
        path, and a `db=` query argument that outranks it. Asserted here on
        the list rather than only through a run, because a run with one worker
        is handed the first free number and that is 1 whichever way the
        exclusion was computed — the end-to-end red cannot see this half.
        """
        from tests.harness import lease_candidates, redis_database_index

        spellings = [
            (0, "redis://127.0.0.1:6379/0"),
            (1, "redis://127.0.0.1:6379/1"),
            (7, "redis://127.0.0.1:6379/7"),
            (15, "redis://127.0.0.1:6379/15"),
            (7, "redis://127.0.0.1:6379/0?db=7"),
            (7, "redis://127.0.0.1:6379/3?db=7"),
            (15, "redis://127.0.0.1:6379?db=15"),
        ]
        for named, url in spellings:
            assert redis_database_index(url) == named, (
                f"{url} reaches a different database than {named}"
            )
            assert named not in lease_candidates(url), (
                f"the database {url} names ({named}) was offered to a worker"
            )
        assert redis_database_index("redis://127.0.0.1:6379") == 0

    def test_a_base_database_that_is_not_zero_survives_a_run(self, tmp_path):
        """The same rule, through a real run rather than through a list."""
        with _external_or_skip(redis_db=1) as ext:
            done = ext.run_pytest(
                "-n", "1", PROBE, probe_out=str(tmp_path / "probe.jsonl")
            )
            assert done.returncode == 0, done.stdout[-3000:]
            sentinel, keys = ext.base_redis()
            assert sentinel == "the base survived", (
                "the harness leased the database TEST_REDIS_URL named, and "
                "then flushed it before every test"
            )
            assert _data_keys(keys) == ["harness:sentinel"], (
                f"a lease was left in the caller's database: {keys}"
            )
            worker_db = json.loads(
                (tmp_path / "probe.jsonl").read_text().splitlines()[0]
            )["redis_url"].rsplit("/", 1)[-1]
            assert worker_db != "1", "the worker was given the caller's database"

    def test_the_database_a_query_argument_names_is_never_leased(self, tmp_path):
        """`db=` in the query wins over the path, and redis-py says so.

        `redis://host:6379/0?db=7` connects to database seven. A base
        exclusion that reads only the path protects database zero, leases
        seven to the first worker, and flushes the caller's data before every
        test — the same defect as the path case, through the other spelling
        redis-py accepts.
        """
        with _external_or_skip(redis_db=7, db_in_query=True) as ext:
            done = ext.run_pytest(
                "-n", "1", PROBE, probe_out=str(tmp_path / "probe.jsonl")
            )
            assert done.returncode == 0, done.stdout[-3000:]
            sentinel, keys = ext.base_redis()
            assert sentinel == "the base survived", (
                "the harness leased the database the URL's `db=` argument "
                "named, and then flushed it before every test"
            )
            assert _data_keys(keys) == ["harness:sentinel"], (
                f"a worker wrote into the caller's database: {keys}"
            )
            from tests.harness import redis_database_index

            worker_url = json.loads(
                (tmp_path / "probe.jsonl").read_text().splitlines()[0]
            )["redis_url"]
            assert redis_database_index(worker_url) != 7, (
                f"the worker's URL still reaches database 7: {worker_url}"
            )

    def test_a_dbname_argument_does_not_point_every_worker_at_the_base(self):
        """The same defect on the other service, found by looking for it.

        libpq takes connection keywords from a URL's query string, and
        `dbname` there outranks the path — measured, the same way redis-py's
        `db=` was. So `postgresql://host:5432/?dbname=liminallm` names no
        database in its path and connects to `liminallm`, and a worker URL
        built by replacing the path keeps the argument that outranks it:

            postgresql://host:5432/base_xd_ab12_gw0?dbname=liminallm

        That URL says one database and reaches another. Every worker then ran
        against the caller's database and truncated it before every test,
        which is the destructive half of the Redis defect with none of the
        Redis part. Reproduced before it was fixed.
        """
        import psycopg

        from tests.harness import (
            ScratchPostgres,
            apply_schema,
            create_worker_database,
            drop_worker_database,
            postgres_database_name,
        )

        pg = ScratchPostgres()
        if not pg.available:
            pytest.skip(pg.unavailable_reason)
        base = pg.start()
        try:
            name = postgres_database_name(base)
            apply_schema(base, embedding_dim=64)
            with psycopg.connect(base, autocommit=True) as conn:
                conn.execute("CREATE TABLE harness_sentinel (note text)")
                conn.execute("INSERT INTO harness_sentinel VALUES ('the base survived')")

            # The same server and the same database, spelled the other way
            # libpq accepts. Nothing in the documentation says a caller may
            # not.
            quirky = f"{base.rsplit('/', 1)[0]}/?dbname={name}"
            assert postgres_database_name(quirky) == name

            worker = create_worker_database(quirky, "gw0", "abc123", prepared=False)
            try:
                assert postgres_database_name(worker) != name, (
                    f"the worker's URL still reaches the caller's database: {worker}"
                )
                # What the per-test reset does, through the URL the worker was
                # actually given.
                with psycopg.connect(worker, autocommit=True) as conn:
                    conn.execute(
                        "TRUNCATE app_user, conversation, message RESTART IDENTITY CASCADE"
                    )
                with psycopg.connect(base, autocommit=True) as conn:
                    row = conn.execute("SELECT note FROM harness_sentinel").fetchone()
                assert row and row[0] == "the base survived", (
                    "a worker truncated the database the caller was using"
                )
            finally:
                drop_worker_database(quirky, worker)

            # And the guard that refuses to drop the base has to read the URL
            # the same way, or it compares two paths and lets the base
            # through.
            with pytest.raises(RuntimeError, match="refusing to drop"):
                drop_worker_database(quirky, f"{base.rsplit('/', 1)[0]}/?dbname={name}")
        finally:
            pg.stop()

    def test_a_database_under_a_live_lease_cannot_become_a_base(self):
        """The transition the other way, which was not serialized.

        `_CLAIM_IF_FREE` refuses to lease a database somebody has reserved as
        their base. Reservation did not refuse the reverse: it was a bare
        `SET`, so a run configured with a database another run had already
        leased marked it as a base and used it, while the holder went on
        renewing and flushing it before every test:

            RUN A, base /1
            gw0 leases /2, writes state
                                    RUN B, TEST_REDIS_URL=.../2
                                    reserves /2 as its base, uses it
            next test: renews /2, FLUSHDB
                                    B's data is gone

        DB0 already held the fact needed to decide, so the residual the
        previous commit called unavoidable was not. Both transitions test the
        other key in the same Lua step, so exactly one of them wins and the
        loser is told.
        """
        from tests.harness import (
            ScratchRedis,
            claim_redis_database,
            reserve_base_database,
        )

        server = ScratchRedis()
        if not server.available:
            pytest.skip("needs redis-server")
        host = server.start().rsplit("/", 1)[0]
        try:
            _, index = claim_redis_database(f"{host}/1", "run-a:gw0")

            assert reserve_base_database(f"{host}/{index}") is False, (
                f"database {index} carries a live lease and was still accepted "
                "as another run's base"
            )
            # And a free one is still accepted, so the refusal is about the
            # lease and not about refusing everything.
            assert reserve_base_database(f"{host}/15") is True
        finally:
            server.stop()

    def test_a_run_told_to_use_a_leased_database_says_so_and_stops(self):
        """The refusal has to reach the caller, not just the return value."""
        from tests.harness import (
            ScratchRedis,
            claim_redis_database,
        )

        server = ScratchRedis()
        if not server.available:
            pytest.skip("needs redis-server")
        host = server.start().rsplit("/", 1)[0]
        try:
            _, index = claim_redis_database(f"{host}/1", "run-a:gw0")
            with pytest.raises(RuntimeError, match="already leased"):
                claim_redis_database(f"{host}/{index}", "run-b:gw0")
        finally:
            server.stop()

    def test_a_serial_run_reserves_the_database_it_was_pointed_at(self, tmp_path):
        """A serial run leases nothing, and used to record nothing either.

        Reservation happened only through a worker's claim and renewal, so
        `make test` against somebody's Redis left its database looking free.
        The parallel lane in the next terminal leased that very number and
        flushed it before every test. One serial run and one parallel run at
        once is an ordinary pair, not an exotic schedule.
        """
        with _external_or_skip(redis_db=4) as ext:
            done = ext.run_pytest(PROBE, probe_out=str(tmp_path / "probe.jsonl"))
            assert done.returncode == 0, done.stdout[-3000:]

            import redis

            from tests.harness import (
                REDIS_BASE_PREFIX,
                claim_redis_database,
                lease_candidates,
            )

            # The serial run has finished, but its reservation outlives it by
            # the TTL — which is the point. Another run starting now must not
            # be handed database 4.
            ledger = redis.Redis.from_url(
                ext.redis_url.rsplit("/", 1)[0] + "/0", decode_responses=True
            )
            try:
                assert ledger.get(f"{REDIS_BASE_PREFIX}:4") is not None, (
                    "a serial run left no record that it was using database 4"
                )
            finally:
                ledger.close()

            base = f"{ext.redis_url.rsplit('/', 1)[0]}/1"
            taken = {
                claim_redis_database(base, f"other:{n}")[1]
                for n in range(len(lease_candidates(base)) - 1)
            }
            assert 4 not in taken, (
                "another run leased the database a serial run was using: "
                f"{sorted(taken)}"
            )

    def test_a_serial_run_keeps_its_reservation_alive_while_it_runs(self):
        """Reserving once at provisioning is not enough for a long run.

        The reservation carries the lease TTL, and the serial lane measures
        881s against 900. Written once and never refreshed, a run on a machine
        slightly slower than this one loses its reservation partway through
        and the database it is still using goes back into circulation.

        Forced rather than waited for: the run gets a 3-second TTL and
        takes five, so the expiry the 19-second margin is about happens while
        the suite is watching.
        """
        with _external_or_skip(redis_db=4) as ext:
            done = ext.run_pytest(
                SLOW_PROBE,
                env_extra={
                    "LIMINALLM_TEST_LEASE_TTL": "3",
                    "LIMINALLM_HARNESS_OUTLIVE": "1",
                },
            )
            assert done.returncode == 0, done.stdout[-3000:]
            assert "5 passed" in done.stdout, (
                "the probe did not actually run, so nothing outlived "
                f"anything:\n{done.stdout[-2000:]}"
            )

            import redis

            from tests.harness import REDIS_BASE_PREFIX

            ledger = redis.Redis.from_url(
                ext.redis_url.rsplit("/", 1)[0] + "/0", decode_responses=True
            )
            try:
                assert ledger.get(f"{REDIS_BASE_PREFIX}:4") is not None, (
                    "the reservation expired while the run was still using "
                    "the database — it is written once and never refreshed"
                )
            finally:
                ledger.close()

    def test_two_base_databases_on_one_server_do_not_lease_the_same_number(self):
        """One server, one ledger — whatever database each caller was given.

        A ledger kept in each caller's own database cannot see the other's
        claims, so two runs configured with different base databases hand out
        the same numbers and then flush each other. The exclusivity a lease
        exists for is server-wide or it is nothing.
        """
        from tests.harness import (
            ScratchRedis,
            claim_redis_database,
            reserve_base_database,
        )

        server = ScratchRedis()
        if not server.available:
            pytest.skip("needs redis-server")
        host = server.start().rsplit("/", 1)[0]
        try:
            # Each run says which database it was given before it takes any.
            # That is the order provisioning uses, and it is the whole
            # guarantee: a run cannot protect its base from one that finished
            # claiming before it ever started, because nothing on the server
            # knew about it yet. It is protected from every run after.
            reserve_base_database(f"{host}/1")
            reserve_base_database(f"{host}/2")

            first = {claim_redis_database(f"{host}/1", f"a{n}:gw0")[1] for n in (1, 2)}
            second = {claim_redis_database(f"{host}/2", f"b{n}:gw0")[1] for n in (1, 2)}
            assert len(first) == 2 and len(second) == 2
            assert first.isdisjoint(second), (
                "two runs with different base databases were handed the same "
                f"numbers: {sorted(first)} and {sorted(second)}"
            )
            assert not {1, 2} & (first | second), (
                "a run leased the database another run was configured with: "
                f"{sorted(first)} and {sorted(second)}"
            )
        finally:
            server.stop()

    def test_a_worker_that_lost_its_lease_stops_before_it_flushes(self, tmp_path):
        """Losing the claim is not something to carry on best-effort through.

        Once a lease expires the number is very likely already somebody's, and
        the next thing the per-test reset does is empty that database. So the
        run stops instead. Forced by having the run overwrite its own claim
        and then asking it to start another test.
        """
        import redis

        with _external_or_skip() as ext:
            done = ext.run_pytest(
                "-n", "1",
                LOSE_PROBE, PROBE,
                probe_out=str(tmp_path / "probe.jsonl"),
                env_extra={"LIMINALLM_HARNESS_LOSE": "1"},
            )
            assert done.returncode != 0, (
                "a run whose lease had been taken carried on and flushed the "
                "database anyway:\n" + done.stdout[-3000:]
            )
            assert "no longer holds Redis database" in done.stdout, done.stdout[-3000:]
            # Which database the probe was using, found the way anything
            # else would: the ledger still names its new holder, because the
            # run's own release compares before it deletes.
            from tests.harness import (
                REDIS_LEASE_PREFIX,
                REDIS_LEDGER_DB,
                redis_url_for_database,
            )

            # Addressed the way the harness addresses it, not as "the URL the
            # fixture was given". Those are the same database only while this
            # fixture's base is 0, and a later edit to that argument would
            # otherwise leave this reading an empty database and reporting a
            # confusing failure.
            ledger = redis.Redis.from_url(
                redis_url_for_database(ext.redis_url, REDIS_LEDGER_DB),
                decode_responses=True,
            )
            try:
                taken_over = [
                    key
                    for key in ledger.keys(f"{REDIS_LEASE_PREFIX}:*")
                    if ledger.get(key) == "another-run:gw0"
                ]
                assert len(taken_over) == 1, (
                    f"expected the handed-over claim to still stand: {taken_over}"
                )
                index = taken_over[0].rsplit(":", 1)[-1]
            finally:
                ledger.close()

            client = redis.Redis.from_url(
                ext.redis_url.rsplit("/", 1)[0] + f"/{index}", decode_responses=True
            )
            try:
                taken = client.get("harness:taken-over")
            finally:
                client.close()
            assert taken == "the new holder's state", (
                "the run emptied a database that had been claimed by another"
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

    def test_two_invocations_at_once_do_not_share_a_redis_database(self, tmp_path):
        """Ownership across runs, not only across the workers of one run.

        The Postgres name carries a run id exactly so two invocations cannot
        both take `gw0`. The Redis number could not carry one — there are
        fifteen numbers, not an alphabet — and deriving it from the worker id
        alone made every invocation pick the same one, while each flushed it
        before every test believing it owned it. Two runs at once is one
        terminal and one editor.

        So run A pauses holding state, run B starts and flushes, and A looks
        again. Nothing here inspects a URL: what has to hold is that A's state
        is still there.
        """
        hold = tmp_path / "hold"
        with _external_or_skip() as ext:
            first = subprocess.Popen(
                [
                    sys.executable, "-m", "pytest", "-q", "--no-header",
                    "-p", "no:cacheprovider", "-p", "no:randomly", "-n", "1",
                    HOLD_PROBE,
                ],
                cwd=ROOT,
                env=ext.env(hold_out=str(hold)),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            try:
                ready = Path(str(hold) + ".ready")
                deadline = time.monotonic() + 180
                while not ready.exists() and time.monotonic() < deadline:
                    if first.poll() is not None:
                        raise AssertionError(
                            "the holding run exited before it was ready:\n"
                            + (first.stdout.read() if first.stdout else "")
                        )
                    time.sleep(0.05)
                assert ready.exists(), "the holding run never signalled ready"

                second = ext.run_pytest("-n", "1", PROBE,
                                        probe_out=str(tmp_path / "probe.jsonl"))
                assert second.returncode == 0, second.stdout[-3000:]
            finally:
                Path(str(hold) + ".release").write_text("go")
                out = first.communicate(timeout=180)[0]
            assert first.returncode == 0, (
                "a second pytest invocation destroyed the first one's Redis "
                "state:\n" + (out or "")
            )
            # And the numbers really were different, which is why it survived.
            held = Path(hold).read_text().rsplit("/", 1)[-1]
            other = json.loads(
                (tmp_path / "probe.jsonl").read_text().splitlines()[0]
            )["redis_url"].rsplit("/", 1)[-1]
            assert held != other, f"both runs used Redis database {held}"

    def test_a_released_database_can_be_claimed_again(self, tmp_path):
        """A lease is a loan. Runs would otherwise stop after fifteen."""
        import redis

        from tests.harness import REDIS_LEASE_PREFIX

        out = tmp_path / "probe.jsonl"
        with _external_or_skip() as ext:
            seen = []
            for _ in range(3):
                out.write_text("")
                done = ext.run_pytest("-n", "1", PROBE, probe_out=str(out))
                assert done.returncode == 0, done.stdout[-2000:]
                seen.append(
                    json.loads(out.read_text().splitlines()[0])["redis_url"]
                )
            assert len(set(seen)) == 1, (
                f"consecutive runs did not reuse the released database: {seen}"
            )
            client = redis.Redis.from_url(ext.redis_url, decode_responses=True)
            try:
                leases = client.keys(f"{REDIS_LEASE_PREFIX}:*")
            finally:
                client.close()
            assert leases == [], f"a finished run left its claim behind: {leases}"

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
