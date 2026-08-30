"""`sql/schema.sql` has to converge, not just create.

SPEC removed the numbered migrations in favour of one idempotent file, on the
grounds that this project has never been deployed. That argument holds for
*stored* data. It says nothing about the databases developers and reviewers
already have, and it is not an exemption from the word "idempotent": applying
the file to a database built from an earlier revision of itself must leave the
shape the file describes.

`CREATE TABLE IF NOT EXISTS` does not do that. It skips the whole statement
when the table exists, so a column added later never appears, and every query
naming that column fails against a database that predates it. `CREATE INDEX IF
NOT EXISTS` has the same shape of problem for a changed definition: the old
index survives under the new name and the planner keeps using it.

The test builds the old shape, applies the file, and asks what happened -
rather than reading it and reasoning, which is what let the gap through.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

PROBE_DB = "liminallm_schema_convergence_probe"

# `ingest_job` as the first revision of this tranche created it: no
# `next_attempt_at`, and the ready index on `created_at`.
EARLIER_REVISION = """
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE TABLE ingest_job (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  context_id UUID NOT NULL,
  fs_path TEXT NOT NULL,
  generation TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'queued',
  attempts INT NOT NULL DEFAULT 0,
  detail TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX ingest_job_ready_idx ON ingest_job (status, created_at);
"""


def _shape(dsn: str) -> tuple:
    """Whether the column exists, and what the ready index is actually on."""
    with psycopg.connect(dsn, row_factory=dict_row) as conn:
        column = conn.execute(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name = 'ingest_job' AND column_name = 'next_attempt_at'"
        ).fetchall()
        index = conn.execute(
            "SELECT indexdef FROM pg_indexes WHERE indexname = 'ingest_job_ready_idx'"
        ).fetchall()
    return bool(column), (index[0]["indexdef"] if index else None)


def _apply_schema(dsn: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["psql", dsn, "-v", "ON_ERROR_STOP=1", "-v", "embedding_dim=64", "-q", "-f", "-"],
        input=Path("sql/schema.sql").read_text(),
        capture_output=True,
        text=True,
    )


def test_applying_the_schema_to_an_older_database_updates_its_shape(client):
    """A throwaway database, because the claim is about creating one badly.

    `client` is requested only to guarantee the harness has a server running;
    nothing here touches the store the other tests share.
    """
    base, _, _name = os.environ["DATABASE_URL"].rpartition("/")
    admin = f"{base}/postgres"
    probe = f"{base}/{PROBE_DB}"

    with psycopg.connect(admin, autocommit=True) as conn:
        conn.execute(f"DROP DATABASE IF EXISTS {PROBE_DB}")
        conn.execute(f"CREATE DATABASE {PROBE_DB}")
    try:
        with psycopg.connect(probe, autocommit=True) as conn:
            conn.execute(EARLIER_REVISION)
        assert _shape(probe) == (
            False,
            "CREATE INDEX ingest_job_ready_idx ON public.ingest_job "
            "USING btree (status, created_at)",
        ), "the older shape was not built, so this test proves nothing"

        first = _apply_schema(probe)
        assert first.returncode == 0, first.stderr
        has_column, index = _shape(probe)
        assert has_column, (
            "a column added after the table existed never arrived, so every "
            "query naming it fails against this database"
        )
        assert "next_attempt_at" in (index or ""), (
            f"the ready index still covers the old columns: {index}"
        )

        # And again, because idempotent means it can be applied any number of
        # times - the second run is the one that would fail on a bare ALTER.
        second = _apply_schema(probe)
        assert second.returncode == 0, second.stderr
        assert _shape(probe) == (has_column, index)
    finally:
        with psycopg.connect(admin, autocommit=True) as conn:
            conn.execute(f"DROP DATABASE IF EXISTS {PROBE_DB}")
