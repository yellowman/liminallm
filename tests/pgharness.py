"""Start a throwaway Postgres for the test suite.

Tests run against the real store, not a stand-in. The in-memory store used to
double the storage layer so the suite could avoid a database; that meant every
storage feature was written twice and verified once, and the version that
production actually runs was the untested one. A scratch cluster costs a few
seconds at session start and removes the whole class of "passes in tests,
breaks in Postgres".

Set ``TEST_DATABASE_URL`` to point at an existing database instead (CI with a
service container, or a developer's local Postgres).
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import tempfile
from pathlib import Path

_PG_BIN_CANDIDATES = ("/usr/lib/postgresql/16/bin", "/usr/lib/postgresql/15/bin")


def _free_port() -> int:
    """A port the kernel just handed out is one no other run is holding."""
    if os.environ.get("TEST_PG_PORT"):
        return int(os.environ["TEST_PG_PORT"])
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
        self.port = _free_port()
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

        _STORE_ROOT = tempfile.mkdtemp(prefix="liminallm_store_")
        _STORE = PostgresStore(os.environ["DATABASE_URL"], fs_root=_STORE_ROOT)
    return _STORE


def close_test_store() -> None:
    global _STORE
    if _STORE is not None:
        _STORE.close_pool()
        _STORE = None
    if _STORE_ROOT:
        shutil.rmtree(_STORE_ROOT, ignore_errors=True)


def apply_schema(url: str, *, embedding_dim: int = 64) -> None:
    """Apply sql/schema.sql. 64-d matches the test encoder (hash fallback)."""
    root = Path(__file__).resolve().parent.parent
    subprocess.run(
        ["psql", url, "-v", "ON_ERROR_STOP=1", "-v", f"embedding_dim={embedding_dim}",
         "-q", "-f", str(root / "sql" / "schema.sql")],
        check=True, stdout=subprocess.DEVNULL, timeout=180,
    )
