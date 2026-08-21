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

_PG_BIN_CANDIDATES = ("/usr/lib/postgresql/16/bin", "/usr/lib/postgresql/15/bin")


def _free_port(env_override: str | None = None) -> int:
    """A port the kernel just handed out is one no other run is holding."""
    if env_override and os.environ.get(env_override):
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
