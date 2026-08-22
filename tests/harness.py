"""Throwaway Postgres and Redis for the test suite.

Tests run against the real thing, not a stand-in, for both. The reasoning is
the same each time and the suite has been bitten by it twice:

* the in-memory store doubled the storage layer, so every storage feature was
  written twice and verified once — and the untested half was the one
  production runs. Removing it surfaced three Postgres-only bugs, including
  preference recording that had never worked;
* a synchronous Redis client existed for the tests alone. It drifted eight
  methods behind the real one and broke the attachment agent the moment Redis
  was present. It was only caught by starting a real ``redis-server``.

So the suite starts both. Redis is what rate limits, idempotency, the session
cache and the concurrency slots actually run on; without it the fallbacks were
exercised and the production path was not (24% covered).

Set ``TEST_DATABASE_URL`` or ``TEST_REDIS_URL`` to point at existing services
instead — CI with service containers, or a developer's local pair.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

_PG_BIN_CANDIDATES = ("/usr/lib/postgresql/16/bin", "/usr/lib/postgresql/15/bin")

#: Postgres truncates identifiers at 63 bytes, and a silently truncated name is
#: two workers sharing a database.
_MAX_IDENTIFIER = 63


def worker_id() -> str:
    """Which xdist worker this process is, or "" when there is no xdist.

    Set by xdist in the worker's environment before conftest is imported —
    measured, not assumed — so everything a worker must own can be derived at
    module scope.
    """
    return os.environ.get("PYTEST_XDIST_WORKER", "")


def run_id() -> str:
    """One id per pytest invocation, shared by every worker in it.

    The controller imports conftest first and `setdefault` stores the value in
    its environment; execnet spawns workers from that environment, so they all
    read the controller's. Measured the same way.

    It exists so two pytest invocations running at once cannot both derive
    `..._gw0` and share a database. Worker id alone is not unique across runs.
    """
    return os.environ.setdefault("LIMINALLM_TEST_RUN", uuid4().hex[:6])


def worker_database_name(base: str, worker: str, run: str) -> str:
    """The database this worker owns, derived from the one it was given.

    Databases rather than schemas: the schema, its triggers and a good deal of
    the store address `public` by name and cast with `::regclass`, so a
    per-worker schema would be a different production model, tested. A
    per-worker database is the same one, twice.
    """
    suffix = f"_xd_{run}_{worker}"
    return base[: _MAX_IDENTIFIER - len(suffix)] + suffix


#: Where a claim on one numbered Redis database is recorded, in database 0.
#: Database 0 is never a worker's, so nothing a worker does can remove these.
REDIS_LEASE_PREFIX = "liminallm:test:redis-db-lease"

#: Long enough that no single test outruns it — the slowest is about a minute
#: — and renewed before every test, so a run that dies stops renewing and its
#: databases come back on their own.
REDIS_LEASE_TTL = 900

#: Redis numbers its databases 0-15 and 0 is the ledger, so fifteen at once.
REDIS_LEASE_SLOTS = range(1, 16)

_RELEASE_IF_OURS = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('DEL', KEYS[1])
end
return 0
"""


def claim_redis_database(base_url: str, holder: str) -> tuple[str, int]:
    """Take a numbered Redis database nobody else is using, and say so.

    Deriving the number from the worker id alone looked sufficient and was
    not: it is a function of `gw0`, so every simultaneous pytest invocation
    picks the same one — and each of them flushes it before every test,
    believing it owns it. Two runs at once is not exotic; it is one terminal
    and one editor.

    The database name cannot carry a run id the way the Postgres one does,
    because there are fifteen numbers rather than an alphabet, so possession
    is recorded instead of encoded. `SET NX` is the claim, and it is atomic,
    so two runs reaching for the same number cannot both get it.

    Returns the URL and the number, the latter so the holder can renew and
    eventually release it.
    """
    import redis

    parts = urlsplit(base_url)
    ledger = redis.Redis.from_url(
        urlunsplit(parts._replace(path="/0")), decode_responses=True
    )
    try:
        for index in REDIS_LEASE_SLOTS:
            if ledger.set(
                f"{REDIS_LEASE_PREFIX}:{index}", holder, nx=True, ex=REDIS_LEASE_TTL
            ):
                return urlunsplit(parts._replace(path=f"/{index}")), index
    finally:
        ledger.close()
    raise RuntimeError(
        f"every one of Redis's {len(REDIS_LEASE_SLOTS)} usable databases is "
        "claimed by a test run. Wait for one to finish, run fewer workers, or "
        "unset TEST_REDIS_URL so each worker starts a server of its own."
    )


def renew_redis_database(base_url: str, index: int, holder: str) -> None:
    """Push the lease out again, for as long as this run is still running.

    Called from the per-test reset, which already talks to Redis, so a live
    run renews continuously and a dead one does not renew at all.
    """
    import redis

    parts = urlsplit(base_url)
    ledger = redis.Redis.from_url(
        urlunsplit(parts._replace(path="/0")), decode_responses=True
    )
    try:
        if ledger.get(f"{REDIS_LEASE_PREFIX}:{index}") == holder:
            ledger.expire(f"{REDIS_LEASE_PREFIX}:{index}", REDIS_LEASE_TTL)
    except Exception:  # pragma: no cover - a Redis that went away mid-run
        pass
    finally:
        ledger.close()


def release_redis_database(base_url: str, index: int, holder: str) -> None:
    """Give the database back, if it is still ours to give.

    Compare-and-delete: after a lease has expired the number may already
    belong to somebody else, and releasing it then would hand their database
    to a third run.
    """
    import redis

    parts = urlsplit(base_url)
    ledger = redis.Redis.from_url(
        urlunsplit(parts._replace(path="/0")), decode_responses=True
    )
    try:
        ledger.eval(_RELEASE_IF_OURS, 1, f"{REDIS_LEASE_PREFIX}:{index}", holder)
    except Exception:  # pragma: no cover - a Redis that went away at teardown
        pass
    finally:
        ledger.close()


def _free_port(env_override: str | None = None) -> int:
    """A port the kernel just handed out is one no other run is holding.

    A fixed override is the opposite of that, and under xdist it would send
    every worker's scratch service to one port. The second worker then fails
    somewhere inside `pg_ctl`, which is a loud failure but not a legible one —
    so refuse here, where the reason can be stated.
    """
    if env_override and os.environ.get(env_override):
        if worker_id():
            raise RuntimeError(
                f"{env_override} pins every scratch service to one port, and "
                f"this run has more than one worker. Unset {env_override}, or "
                "point the workers at services of your own with "
                "TEST_DATABASE_URL / TEST_REDIS_URL."
            )
        return int(os.environ[env_override])
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _pg_bin() -> str | None:
    for candidate in _PG_BIN_CANDIDATES:
        if Path(candidate, "initdb").exists():
            return candidate
    if shutil.which("initdb"):
        return str(Path(shutil.which("initdb")).parent)
    return None


class ScratchPostgres:
    """An initdb'd cluster in a temp dir, torn down with the session."""

    def __init__(self) -> None:
        self.datadir: str | None = None
        self.url: str | None = None
        self.port = _free_port("TEST_PG_PORT")
        self._bin = _pg_bin()

    @property
    def available(self) -> bool:
        return self._bin is not None

    def start(self) -> str:
        assert self._bin
        self.datadir = tempfile.mkdtemp(prefix="liminallm_pg_")
        # initdb refuses to run as root, so hand the directory to the postgres
        # user when we are root (containers) and run everything through su.
        as_postgres = os.geteuid() == 0
        if as_postgres:
            shutil.chown(self.datadir, "postgres", "postgres")
            os.chmod(self.datadir, 0o700)
        self._run(f"{self._bin}/initdb -D {self.datadir} -U postgres --auth=trust")
        self._run(
            f"{self._bin}/pg_ctl -D {self.datadir} "
            f"-o '-p {self.port} -c listen_addresses=127.0.0.1 -c fsync=off' "
            f"-l {self.datadir}/log -w start"
        )
        self._run(f"{self._bin}/createdb -h 127.0.0.1 -p {self.port} -U postgres liminallm_test")
        self.url = f"postgresql://postgres@127.0.0.1:{self.port}/liminallm_test"
        return self.url

    def stop(self) -> None:
        if self.datadir and self._bin:
            self._run(f"{self._bin}/pg_ctl -D {self.datadir} -m immediate -w stop", check=False)
            shutil.rmtree(self.datadir, ignore_errors=True)

    def _run(self, command: str, *, check: bool = True) -> None:
        if os.geteuid() == 0:
            command = f"su postgres -s /bin/sh -c \"{command}\""
        subprocess.run(
            command, shell=True, check=check,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120,
        )


class ScratchRedis:
    """A redis-server on a free port, torn down with the session.

    ``--save ''`` because nothing here outlives the run, and a background
    rewrite during a test is noise. If redis-server is missing the suite still
    runs: ``available`` is False and the code takes its documented fallback,
    which is the same thing a Redis outage does in production.
    """

    def __init__(self) -> None:
        self.port = _free_port("TEST_REDIS_PORT")
        self.url = f"redis://127.0.0.1:{self.port}/0"
        self._proc: subprocess.Popen | None = None
        self._bin = shutil.which("redis-server")

    @property
    def available(self) -> bool:
        return self._bin is not None

    def start(self) -> str:
        assert self._bin
        self._proc = subprocess.Popen(
            [self._bin, "--port", str(self.port), "--bind", "127.0.0.1",
             "--save", "", "--appendonly", "no"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            with socket.socket() as probe:
                probe.settimeout(0.2)
                if probe.connect_ex(("127.0.0.1", self.port)) == 0:
                    return self.url
            time.sleep(0.05)
        raise RuntimeError(f"redis-server did not accept connections on {self.port}")

    def stop(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None


_STORE = None
_STORE_ROOT: str | None = None


def get_test_store():
    """The one store the suite talks to — the same class production runs.

    Shared across tests so the suite holds a single connection pool instead of
    one per test; isolation comes from truncating between tests (see conftest),
    not from handing each test its own database. Lives here rather than in
    conftest so test modules can import it without re-executing conftest as a
    second module and starting a second cluster.
    """
    global _STORE, _STORE_ROOT
    if _STORE is None:
        from liminallm.storage.postgres import PostgresStore

        # The root the runtime resolves everything else against. A store built
        # by `Runtime` is handed `settings.shared_fs_root`, so the two agree by
        # construction; this one is built here, and minting a temporary
        # directory of its own left artifact payloads under one root while
        # filesystem authority, adapters, archive staging and the interpreter
        # resolved paths under another.
        #
        # From the settings, which read SHARED_FS_ROOT — the throwaway root
        # conftest exports before any import.
        from liminallm.config import get_settings

        _STORE_ROOT = get_settings().shared_fs_root
        _STORE = PostgresStore(os.environ["DATABASE_URL"], fs_root=_STORE_ROOT)
    return _STORE


def reset_shared_store(store) -> None:
    """Put the session-wide store back into the state a fresh boot leaves.

    `PostgresStore.__init__` seeds the default chat workflow and tool specs,
    and it used to run twice per test, so the per-test TRUNCATE was undone by
    the next construction. One store for the session means that construction
    happens once — so the first TRUNCATE removed the defaults and every test
    after it ran in a boot state production never has, exercising fallbacks
    where the application runs on seeded rows.

    Re-seeding a handful of rows is a fraction of what rebuilding the store
    cost, so the isolation is restored without giving the time back.

    `sessions` is cleared here for the same reason: it is an in-memory cache
    that TRUNCATE cannot reach, and with a session-wide store it accumulated
    for the length of the run.
    """
    with store._session_lock:
        store.sessions.clear()
    store._ensure_default_artifacts()


def close_test_store() -> None:
    global _STORE
    if _STORE is not None:
        _STORE.close_pool()
        _STORE = None
    # _STORE_ROOT is SHARED_FS_ROOT, which conftest owns and which holds
    # everything the run wrote. It is not this function's to remove.


def apply_schema(url: str, *, embedding_dim: int = 64) -> None:
    """Apply sql/schema.sql. 64-d matches the test encoder (hash fallback)."""
    root = Path(__file__).resolve().parent.parent
    subprocess.run(
        ["psql", url, "-v", "ON_ERROR_STOP=1", "-v", f"embedding_dim={embedding_dim}",
         "-q", "-f", str(root / "sql" / "schema.sql")],
        check=True, stdout=subprocess.DEVNULL, timeout=180,
    )


def _maintenance_url(base_url: str) -> str:
    """A connection that is not to the database we are about to clone.

    `CREATE DATABASE ... TEMPLATE t` refuses while any session is connected to
    `t`, and a session connected to `t` in order to issue the statement counts.
    So the statement is issued against `postgres`, which every server has.
    """
    parts = urlsplit(base_url)
    return urlunsplit(parts._replace(path="/postgres"))


def create_worker_database(base_url: str, worker: str, run: str, *, prepared: bool) -> str:
    """Give one xdist worker a database of its own, and return its URL.

    Two provisioning histories, and the difference between them is the whole
    reason this is not one line:

    * `TEST_SCHEMA_PREPARED` means something outside this suite already built
      the schema — CI runs `scripts/migrate.sh` and then sets it, precisely so
      that conftest cannot quietly repair a deploy command that does nothing.
      A worker must therefore *clone* that database rather than build its own,
      or the invariant is lost the moment the suite runs in parallel: gut
      migrate.sh, and every worker would rebuild the schema from scratch and
      go green over it.
    * Without it, the database this suite was handed is one it prepared
      itself, so an empty database plus `apply_schema` is the same thing.

    The worker owns what it creates and drops it at the end. Nothing here ever
    writes to the base database.
    """
    import psycopg

    parts = urlsplit(base_url)
    base_name = parts.path.lstrip("/")
    name = worker_database_name(base_name, worker, run)
    with psycopg.connect(_maintenance_url(base_url), autocommit=True) as conn:
        conn.execute(f'DROP DATABASE IF EXISTS "{name}"')
        if prepared:
            conn.execute(f'CREATE DATABASE "{name}" TEMPLATE "{base_name}"')
        else:
            conn.execute(f'CREATE DATABASE "{name}"')
    url = urlunsplit(parts._replace(path=f"/{name}"))
    if not prepared:
        apply_schema(url, embedding_dim=64)
    return url


def drop_worker_database(base_url: str, url: str) -> None:
    """Remove what this worker created, and only that."""
    import psycopg

    name = urlsplit(url).path.lstrip("/")
    if name == urlsplit(base_url).path.lstrip("/"):
        raise RuntimeError("refusing to drop the database this run was given")
    try:
        with psycopg.connect(_maintenance_url(base_url), autocommit=True) as conn:
            conn.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
    except Exception:  # pragma: no cover - a server going away at teardown
        pass
