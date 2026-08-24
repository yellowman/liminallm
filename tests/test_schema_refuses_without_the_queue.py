"""An application that needs a table must refuse to start without it.

`ingest_job` is where "this context owes this path a re-read" is recorded.
Every replacement writes to it and every worker poll reads it, so a database
without it is not a degraded deployment — it is one where the first file
replacement raises at request time and the queue that would have repaired the
index cannot be read at all.

`_verify_required_schema` exists to turn that into a refusal at startup, with
the command that fixes it. This is the witness that `ingest_job` is on its
list, because it was on that list in the tranche that introduced the table and
did not survive being merged into another branch — a conflict resolution can
silently un-require a table, and nothing else would have noticed.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import psycopg
import pytest

from liminallm.storage.postgres import PostgresStore

PROBE_DB = "liminallm_missing_queue_probe"


def _apply_schema(dsn: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["psql", dsn, "-v", "ON_ERROR_STOP=1", "-v", "embedding_dim=64", "-q", "-f", "-"],
        input=Path("sql/schema.sql").read_text(),
        capture_output=True,
        text=True,
    )


def test_a_database_without_the_ingest_queue_refuses_to_start(client):
    """A throwaway database, because the claim is about starting against one.

    `client` is requested only so the harness has a server running; nothing
    here touches the store the other tests share.
    """
    base, _, _name = os.environ["DATABASE_URL"].rpartition("/")
    admin = f"{base}/postgres"
    probe = f"{base}/{PROBE_DB}"

    def _drop(conn) -> None:
        # WITH (FORCE): the store opens its pool before it verifies the
        # schema, so the connections outlive the refusal and an ordinary
        # DROP is refused for sessions belonging to a store that has
        # already failed.
        conn.execute(f"DROP DATABASE IF EXISTS {PROBE_DB} WITH (FORCE)")

    with psycopg.connect(admin, autocommit=True) as conn:
        _drop(conn)
        conn.execute(f"CREATE DATABASE {PROBE_DB}")
    try:
        applied = _apply_schema(probe)
        assert applied.returncode == 0, applied.stderr

        # A database from before the queue existed.
        with psycopg.connect(probe, autocommit=True) as conn:
            conn.execute("DROP TABLE ingest_job")

        with pytest.raises(RuntimeError) as raised:
            PostgresStore(probe, fs_root="/tmp")
        message = str(raised.value)
        assert "ingest_job" in message, (
            "startup accepted a database with no ingest queue, so the first "
            f"replacement fails at request time instead: {message}"
        )
        assert "migrate.sh" in message, (
            f"the refusal does not say how to fix it: {message}"
        )
    finally:
        with psycopg.connect(admin, autocommit=True) as conn:
            _drop(conn)
