"""Throwaway Postgres and Redis for the test suite.

Tests run against the real thing, not a stand-in, for both. The reasoning is
the same each time and the suite has been bitten by it twice:

* the in-memory store doubled the storage layer, so every storage feature was
  written twice and verified once - and the untested half was the one
  production runs. Removing it surfaced three Postgres-only bugs, including
  preference recording that had never worked;
* a synchronous Redis client existed for the tests alone. It drifted eight
  methods behind the real one and broke the attachment agent the moment Redis
  was present. It was only caught by starting a real ``redis-server``.

So the suite starts both. Redis is what rate limits, idempotency, the session
cache and the concurrency slots actually run on; without it the fallbacks were
exercised and the production path was not (24% covered).

Set ``TEST_DATABASE_URL`` or ``TEST_REDIS_URL`` to point at existing services
instead - CI with service containers, or a developer's local pair.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

_PG_BIN_CANDIDATES = ("/usr/lib/postgresql/16/bin", "/usr/lib/postgresql/15/bin")

#: Postgres truncates identifiers at 63 bytes, and a silently truncated name is
#: two workers sharing a database.
_MAX_IDENTIFIER = 63


def worker_id() -> str:
    """Which xdist worker this process is, or "" when there is no xdist.

    Set by xdist in the worker's environment before conftest is imported -
    measured, not assumed - so everything a worker must own can be derived at
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


#: Where a claim on one numbered Redis database is recorded.
#:
#: Database 0, one ledger for the whole server, and never a candidate. It was
#: briefly moved into whichever database `TEST_REDIS_URL` named, to keep every
#: write inside the database the caller pointed at - but that fragments the
#: only thing a lease is for. Two runs given different base databases on one
#: server then keep separate ledgers, cannot see each other's claims, and hand
#: out the same number twice:
#:
#:     RUN A, base /1                 RUN B, base /2
#:     claim /3  [ledger in DB1]      claim /3  [ledger in DB2]
#:                    both believe they own database 3
#:
#: Exclusivity across callers needs one place to record it. The cost is that
#: the harness writes into database 0 even when told to use another - short
#: lived keys under the two prefixes below, compare-deleted at teardown and
#: expiring on their own. The database the caller named is still never leased
#: and never flushed, which was the actual defect.
REDIS_LEDGER_DB = 0
REDIS_LEASE_PREFIX = "liminallm:test:redis-db-lease"

#: A run also records which database it was *given*, so other callers do not
#: lease it. Excluding our own base protects us from ourselves; it cannot
#: protect us from a run configured with a different one, because nothing else
#: on the server knows that database is spoken for:
#:
#:     RUN A, base /1                 RUN B, base /2
#:     candidates skip /1             candidates skip /2
#:                                    claims /1  <- A's base, flushed per test
#:
#: Reserved rather than leased: several workers of one run share a base, so it
#: is not one holder's to release. It expires instead, which errs towards
#: leaving a database alone.
REDIS_BASE_PREFIX = "liminallm:test:redis-db-base"

#: Long enough that no single test outruns it - the slowest is about a minute
#: - and renewed before every test, so a run that dies stops renewing and its
#: databases come back on their own.
#:
#: It is the renewal that makes 900 safe, not the number: the serial lane
#: measures 881s, which would otherwise be a 19-second margin on a machine no
#: slower than this one. `LIMINALLM_TEST_LEASE_TTL` shortens it, which lets a
#: test force the expiry the margin is about instead of waiting a quarter of
#: an hour for one.
def _lease_ttl() -> int:
    """The lease TTL, from the override when there is one.

    A value below one second is refused rather than clamped. `SET ... EX 0`
    is an error in Redis and a negative TTL deletes on write, so the run
    would fail somewhere inside the ledger with a message about the wrong
    thing - while the mistake is right here, in a number somebody typed.
    """
    raw = os.environ.get("LIMINALLM_TEST_LEASE_TTL")
    if not raw:
        return 900
    try:
        seconds = int(raw)
    except ValueError:
        raise RuntimeError(
            f"LIMINALLM_TEST_LEASE_TTL must be a whole number of seconds, not {raw!r}"
        ) from None
    if seconds < 1:
        raise RuntimeError(
            f"LIMINALLM_TEST_LEASE_TTL must be at least 1 second, not {seconds}. "
            "A lease that expires on write is not a shorter lease, it is no lease."
        )
    return seconds


REDIS_LEASE_TTL = _lease_ttl()

#: Redis numbers its databases 0-15. Database 0 is left out even when it is
#: not the ledger: it is the conventional default and the likeliest to hold
#: somebody's data.
REDIS_LEASE_SLOTS = range(1, 16)

_RELEASE_IF_OURS = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('DEL', KEYS[1])
end
return 0
"""

#: Claim a database only if nobody holds it and nobody has it as their base.
#: One step, because a check followed by a claim is a window in which another
#: run reserves the base we just looked at.
_CLAIM_IF_FREE = """
if redis.call('EXISTS', KEYS[2]) == 1 then
    return 0
end
if redis.call('SET', KEYS[1], ARGV[1], 'NX', 'EX', ARGV[2]) then
    return 1
end
return 0
"""

#: Mark a database as somebody's base, unless a worker already holds it.
#:
#: The mirror of `_CLAIM_IF_FREE`, and it has to exist for the same reason.
#: Each transition tests the other's key in the same step it writes its own,
#: so of two runs reaching for one number in either order exactly one wins:
#:
#:     lease first   -> the later run is refused this database as a base
#:     reserve first -> the later worker is refused this database as a lease
#:
#: A bare `SET` here was a real hole, not a theoretical one. A run handed
#: `TEST_REDIS_URL=redis://host/2` while another run's worker held a lease on
#: database 2 used it anyway, and the holder went on flushing it before every
#: test.
_RESERVE_IF_UNLEASED = """
if redis.call('EXISTS', KEYS[2]) == 1 then
    return 0
end
redis.call('SET', KEYS[1], ARGV[1], 'EX', ARGV[2])
return 1
"""

_RENEW_IF_OURS = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('EXPIRE', KEYS[1], ARGV[2])
end
return 0
"""


def redis_database_index(url: str) -> int:
    """Which database a Redis URL actually connects to.

    Asked of redis-py rather than read off the path, because the path is not
    the whole answer: a `db=` query argument wins over it, so
    `redis://host:6379/3?db=7` is database seven. Measured, not assumed.

    Re-deriving that precedence by hand is how an exclusion rule ends up
    protecting a database nobody is using while the one it meant to protect
    gets leased and flushed.
    """
    import redis

    client = redis.Redis.from_url(url)
    try:
        return int(client.connection_pool.connection_kwargs.get("db") or 0)
    finally:
        client.close()


def redis_url_for_database(base_url: str, index: int) -> str:
    """`base_url`, pointed at a different numbered database.

    The path is replaced *and* any `db=` argument dropped. Leaving it would
    win over the path, so the URL would name one database and connect to
    another - which is the same defect one layer along.
    """
    parts = urlsplit(base_url)
    query = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if key != "db"
    ]
    return urlunsplit(
        parts._replace(path=f"/{index}", query=urlencode(query))
    )


def _ledger(base_url: str):
    """A client on the one database where every claim on this server is kept."""
    import redis

    return redis.Redis.from_url(
        redis_url_for_database(base_url, REDIS_LEDGER_DB), decode_responses=True
    )


def lease_candidates(base_url: str) -> list[int]:
    """The numbers a worker may be given.

    Two are excluded. Never the one `TEST_REDIS_URL` reaches: that database is
    the caller's, and a worker leasing it would flush it before every test,
    which is the base-preservation rule inverted rather than bent. Pointing
    the harness at `redis://host/1` is an ordinary thing to do. Never
    database 0 either, which holds the ledger for the whole server - a
    worker's `FLUSHDB` must not be able to erase the record of who owns what.

    Which database the URL *reaches*, not which one its path spells: see
    `redis_database_index`.
    """
    base = redis_database_index(base_url)
    return [
        index
        for index in REDIS_LEASE_SLOTS
        if index != base and index != REDIS_LEDGER_DB
    ]


def claim_redis_database(base_url: str, holder: str) -> tuple[str, int]:
    """Take a numbered Redis database nobody else is using, and say so.

    Deriving the number from the worker id alone looked sufficient and was
    not: it is a function of `gw0`, so every simultaneous pytest invocation
    picks the same one - and each of them flushes it before every test,
    believing it owns it. Two runs at once is not exotic; it is one terminal
    and one editor.

    The database name cannot carry a run id the way the Postgres one does,
    because there are fifteen numbers rather than an alphabet, so possession
    is recorded instead of encoded. `SET NX` is the claim, and it is atomic,
    so two runs reaching for the same number cannot both get it.

    Returns the URL and the number, the latter so the holder can renew and
    eventually release it.
    """
    candidates = lease_candidates(base_url)
    ledger = _ledger(base_url)
    try:
        if not reserve_base_database(base_url, ledger=ledger):
            raise _base_in_use_error(redis_database_index(base_url))
        for index in candidates:
            if ledger.eval(
                _CLAIM_IF_FREE, 2,
                f"{REDIS_LEASE_PREFIX}:{index}", f"{REDIS_BASE_PREFIX}:{index}",
                holder, REDIS_LEASE_TTL,
            ):
                return redis_url_for_database(base_url, index), index
    finally:
        ledger.close()
    raise RuntimeError(
        f"every one of the {len(candidates)} Redis databases this harness may "
        "use is claimed by a test run. Wait for one to finish, run fewer "
        "workers, or unset TEST_REDIS_URL so each worker starts a server of "
        "its own."
    )


def reserve_base_database(base_url: str, *, ledger=None) -> bool:
    """Say, where every caller can see it, that this database is spoken for.

    Answers False when a worker of another run already holds a lease on that
    database. The two transitions are symmetric - see `_RESERVE_IF_UNLEASED` -
    so whichever run gets there first keeps the database and the other is
    told, rather than both proceeding and one flushing the other.

    Refreshed rather than claimed: workers of one run all reserve the same
    base, and none of them owns it alone. Nothing releases it - it expires,
    which leaves a database alone for a while longer rather than handing it
    out early. Refreshing re-tests the lease key, so a reservation that lapsed
    and was leased away is not silently re-taken.

    Failures are not swallowed here. Each caller already answers for one, and
    they answer differently: a claim that could not say which database it was
    given has silently dropped the protection it was about to rely on and
    should fail, while renewal reports False and its caller stops. A catch in
    between could only hide the case the two do not share - a Redis that
    permits `EVAL` and refuses `SET`, which is a permissions fault worth
    seeing rather than a database quietly left unprotected.
    """
    own = ledger is None
    ledger = ledger if ledger is not None else _ledger(base_url)
    index = redis_database_index(base_url)
    try:
        return bool(
            ledger.eval(
                _RESERVE_IF_UNLEASED, 2,
                f"{REDIS_BASE_PREFIX}:{index}", f"{REDIS_LEASE_PREFIX}:{index}",
                "in use as a base", REDIS_LEASE_TTL,
            )
        )
    finally:
        if own:
            ledger.close()


def _base_in_use_error(index: int) -> RuntimeError:
    return RuntimeError(
        f"Redis database {index} is already leased to another test run, so "
        "this run cannot use it as its base. Wait for that run to finish, or "
        "point TEST_REDIS_URL at a different database."
    )


def renew_redis_database(base_url: str, index: int, holder: str) -> bool:
    """Push the lease out again, and say whether it is still ours.

    Called from the per-test reset, which already talks to Redis, so a live
    run renews continuously and a dead one does not renew at all.

    Compare-and-expire in one step, for the reason release compares before it
    deletes: once a lease has expired the number may already belong to
    somebody else, and a read followed by an `EXPIRE` would extend their
    claim. The answer is returned rather than logged because the caller is
    about to flush that database - a run that has lost its lease must stop,
    not continue best-effort.

    A Redis that cannot be reached answers False for the same reason: unknown
    is not owned.
    """
    try:
        ledger = _ledger(base_url)
    except Exception:  # pragma: no cover - a Redis that went away mid-run
        return False
    try:
        # The base reservation is refreshed on the same schedule and answers
        # the same way. If it lapsed and another run took that database as a
        # lease, this run has stopped being isolated and must not carry on.
        if not reserve_base_database(base_url, ledger=ledger):
            return False
        return bool(
            ledger.eval(
                _RENEW_IF_OURS, 1,
                f"{REDIS_LEASE_PREFIX}:{index}", holder, REDIS_LEASE_TTL,
            )
        )
    except Exception:  # pragma: no cover - a Redis that went away mid-run
        return False
    finally:
        ledger.close()


def release_redis_database(base_url: str, index: int, holder: str) -> None:
    """Give the database back, if it is still ours to give.

    Compare-and-delete: after a lease has expired the number may already
    belong to somebody else, and releasing it then would hand their database
    to a third run.
    """
    ledger = _ledger(base_url)
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
    somewhere inside `pg_ctl`, which is a loud failure but not a legible one -
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


#: The extensions `sql/schema.sql` creates. `vector` is the one that travels
#: separately - a stock PostgreSQL install does not have it, and the CI runner
#: reaches pgvector only through a service *container*, while a scratch cluster
#: is built from the host's own binaries. Without this the shortfall surfaced
#: as `psql` exiting 3 with its stderr discarded, which names nothing.
_SCHEMA_EXTENSIONS = ("vector", "citext", "uuid-ossp")


def _missing_extensions(bin_dir: str) -> list[str]:
    """Which of the schema's extensions this installation cannot supply.

    Read from the control files beside the binaries rather than from a running
    cluster, because the answer decides whether to bother starting one.
    """
    try:
        done = subprocess.run(
            [f"{bin_dir}/pg_config", "--sharedir"],
            capture_output=True, text=True, timeout=30,
        )
    except OSError:  # pragma: no cover - pg_config ships with the binaries
        return []
    share = done.stdout.strip()
    if not share:  # pragma: no cover - cannot tell; let the schema speak
        return []
    return [
        name
        for name in _SCHEMA_EXTENSIONS
        if not Path(share, "extension", f"{name}.control").exists()
    ]


class ScratchPostgres:
    """An initdb'd cluster in a temp dir, torn down with the session."""

    def __init__(self) -> None:
        self.datadir: str | None = None
        self.url: str | None = None
        self.port = _free_port("TEST_PG_PORT")
        self._bin = _pg_bin()

    @property
    def available(self) -> bool:
        return self._bin is not None and not self.missing_extensions

    @property
    def missing_extensions(self) -> list[str]:
        """Extensions the schema needs that this installation cannot supply."""
        return _missing_extensions(self._bin) if self._bin else []

    @property
    def unavailable_reason(self) -> str:
        if self._bin is None:
            return "initdb not available"
        missing = self.missing_extensions
        if missing:
            return (
                "this PostgreSQL installation has no "
                + ", ".join(missing)
                + " extension, so sql/schema.sql cannot be applied to a "
                "scratch cluster built from it. A pgvector service container "
                "does not help; that is a different server."
            )
        return ""

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
        # The unix socket goes in the data directory, which is the one place
        # this cluster's own user is guaranteed to own. Debian and Ubuntu
        # compile the default as /var/run/postgresql, owned by `postgres`,
        # which is writable when we are root and `su` to that user and is not
        # writable by anybody else - so on a CI runner that runs the suite as
        # an ordinary user the postmaster died with "could not create lock
        # file /var/run/postgresql/.s.PGSQL.<port>.lock: Permission denied".
        # A scratch cluster should not reach outside its scratch directory in
        # the first place; `createdb` and the tests connect over TCP anyway.
        self._run(
            f"{self._bin}/pg_ctl -D {self.datadir} "
            f"-o '-p {self.port} -c listen_addresses=127.0.0.1 -c fsync=off "
            f"-c unix_socket_directories={self.datadir}' "
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
        done = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=120,
        )
        if check and done.returncode != 0:
            raise RuntimeError(self._why_it_failed(command, done))

    def _why_it_failed(self, command: str, done: subprocess.CompletedProcess) -> str:
        """`pg_ctl` says "Examine the log output". This examines it.

        Both streams used to go to DEVNULL, so a cluster that would not start
        surfaced as a bare `CalledProcessError` naming a command and no cause
        - and `pg_ctl` itself only prints "could not start server", because
        the reason is in the server log it was handed with `-l`. That cost a
        full CI round trip to discover a one-line permissions problem that the
        log had stated plainly the whole time.
        """
        parts = [f"{command}\nexit status {done.returncode}"]
        for label, text in (("stdout", done.stdout), ("stderr", done.stderr)):
            if text and text.strip():
                parts.append(f"{label}:\n  " + "\n  ".join(text.strip().splitlines()))
        log = Path(self.datadir or "", "log")
        if log.exists():
            tail = log.read_text(errors="replace").strip().splitlines()[-8:]
            if tail:
                parts.append("server log:\n  " + "\n  ".join(tail))
        return "\n".join(parts)


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
    """The one store the suite talks to - the same class production runs.

    Shared across tests so the suite holds a single connection pool instead of
    one per test; isolation comes from truncating between tests (see conftest),
    not from handing each test its own database. Lives here rather than in
    conftest so test modules can import it without re-executing conftest as a
    second module and starting a second cluster.
    """
    global _STORE, _STORE_ROOT
    if _STORE is None:
        # The root the runtime resolves everything else against. A store built
        # by `Runtime` is handed `settings.shared_fs_root`, so the two agree by
        # construction; this one is built here, and minting a temporary
        # directory of its own left artifact payloads under one root while
        # filesystem authority, adapters, archive staging and the interpreter
        # resolved paths under another.
        #
        # From the settings, which read SHARED_FS_ROOT - the throwaway root
        # conftest exports before any import.
        from liminallm.config import get_settings
        from liminallm.storage.postgres import PostgresStore

        _STORE_ROOT = get_settings().shared_fs_root
        _STORE = PostgresStore(os.environ["DATABASE_URL"], fs_root=_STORE_ROOT)
    return _STORE


def reset_shared_store(store) -> None:
    """Put the session-wide store back into the state a fresh boot leaves.

    `PostgresStore.__init__` seeds the default chat workflow and tool specs,
    and it used to run twice per test, so the per-test TRUNCATE was undone by
    the next construction. One store for the session means that construction
    happens once - so the first TRUNCATE removed the defaults and every test
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
    """Apply sql/schema.sql. 64-d matches the test encoder (hash fallback).

    `ON_ERROR_STOP=1` makes psql exit 3 on the first failing statement, and
    stderr is where it says which one. That used to be discarded, so a schema
    that would not apply arrived as `CalledProcessError ... exit status 3` -
    a number, with the explanation thrown away one line earlier.
    """
    root = Path(__file__).resolve().parent.parent
    done = subprocess.run(
        ["psql", url, "-v", "ON_ERROR_STOP=1", "-v", f"embedding_dim={embedding_dim}",
         "-q", "-f", str(root / "sql" / "schema.sql")],
        capture_output=True, text=True, timeout=180,
    )
    if done.returncode != 0:
        detail = (done.stderr or done.stdout or "").strip()
        raise RuntimeError(
            f"applying sql/schema.sql to {postgres_database_name(url)!r} failed "
            f"(psql exit {done.returncode})"
            + (":\n  " + "\n  ".join(detail.splitlines()[-10:]) if detail else "")
        )


def postgres_database_name(url: str) -> str:
    """Which database a Postgres URL actually connects to.

    Asked of psycopg rather than read off the path, for the reason
    `redis_database_index` asks redis-py: libpq takes connection keywords from
    the query string as well, and `dbname` there outranks the path. Measured -

        postgresql://host:5432/mydb?dbname=other  ->  libpq connects to other

    Only `dbname` is normalized. A `host` or `port` argument redirects the
    maintenance connection and the worker's together, which is the caller
    naming a server; `dbname` is the one that makes a URL say one database and
    reach another, and that asymmetry is what destroys data.
    """
    from psycopg.conninfo import conninfo_to_dict

    return conninfo_to_dict(url).get("dbname") or ""


def postgres_url_for_database(base_url: str, name: str) -> str:
    """`base_url`, pointed at a different database.

    The path is replaced *and* any `dbname` argument dropped, because leaving
    it would outrank the path - the URL would name the worker's database and
    connect to the caller's, which every worker would then truncate before
    every test.
    """
    parts = urlsplit(base_url)
    query = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if key != "dbname"
    ]
    return urlunsplit(parts._replace(path=f"/{name}", query=urlencode(query)))


def _maintenance_url(base_url: str) -> str:
    """A connection that is not to the database we are about to clone.

    `CREATE DATABASE ... TEMPLATE t` refuses while any session is connected to
    `t`, and a session connected to `t` in order to issue the statement counts.
    So the statement is issued against `postgres`, which every server has.
    """
    return postgres_url_for_database(base_url, "postgres")


def create_worker_database(base_url: str, worker: str, run: str, *, prepared: bool) -> str:
    """Give one xdist worker a database of its own, and return its URL.

    Two provisioning histories, and the difference between them is the whole
    reason this is not one line:

    * `TEST_SCHEMA_PREPARED` means something outside this suite already built
      the schema - CI runs `scripts/migrate.sh` and then sets it, precisely so
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

    base_name = postgres_database_name(base_url)
    name = worker_database_name(base_name, worker, run)
    with psycopg.connect(_maintenance_url(base_url), autocommit=True) as conn:
        conn.execute(f'DROP DATABASE IF EXISTS "{name}"')
        if prepared:
            conn.execute(f'CREATE DATABASE "{name}" TEMPLATE "{base_name}"')
        else:
            conn.execute(f'CREATE DATABASE "{name}"')
    url = postgres_url_for_database(base_url, name)
    if not prepared:
        apply_schema(url, embedding_dim=64)
    return url


def drop_worker_database(base_url: str, url: str) -> None:
    """Remove what this worker created, and only that."""
    import psycopg

    name = postgres_database_name(url)
    if name == postgres_database_name(base_url):
        raise RuntimeError("refusing to drop the database this run was given")
    try:
        with psycopg.connect(_maintenance_url(base_url), autocommit=True) as conn:
            conn.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
    except Exception:  # pragma: no cover - a server going away at teardown
        pass
