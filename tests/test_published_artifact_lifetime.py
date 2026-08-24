"""Published configuration outlives the account that published it.

SPEC §12.3 says an artifact that is `shared` or `global` leaves its owner's
sole control and changes only through config ops. The physical lifecycle said
otherwise: `delete_user` removed every row with that `owner_user_id` whatever
its visibility, and the foreign key cascaded independently, so deleting the
admin who published a tool server also deleted the server, its versions and
its patch history — with no review and no record that it had ever existed.

That is not an escape: it needs an admin and it fails closed. It is a
contradiction between two rules the installation states about itself, and it
made installation-wide configuration share a personnel account's lifetime.

The model these fix on: publishing detaches. A private artifact still dies
with its account. A published one keeps its row, its versions and its audit
trail, and loses its owner — which for an MCP server means it goes inert,
because the admin attestation is what made it a capability, and it stays that
way until an admin re-publishes it.
"""

from __future__ import annotations

import subprocess
import uuid
from pathlib import Path

import pytest

from liminallm.service import mcp_client
from liminallm.service.runtime import get_runtime
from tests.mcpfixture import MCPFixture

ROOT = Path(__file__).resolve().parent.parent


def _account(client, prefix, *, admin=False):
    email = f"{prefix}_{uuid.uuid4().hex[:8]}@example.com"
    password = "TestPassword123!"
    resp = client.post("/v1/auth/signup", json={"email": email, "password": password})
    assert resp.status_code == 201, resp.text
    data = resp.json()["data"]
    user_id = data["user_id"]
    if admin:
        get_runtime().store.update_user_role(user_id, role="admin")
        resp = client.post(
            "/v1/auth/login", json={"email": email, "password": password}
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()["data"]
    return user_id, {"Authorization": f"Bearer {data['access_token']}"}


def _count(sql, params) -> int:
    with get_runtime().store._connect() as conn:
        return int(conn.execute(sql, params).fetchone()["n"])


def _publish(client, headers, fixture, *, visibility="global"):
    resp = client.post(
        "/v1/artifacts",
        json={
            "type": "mcp",
            "name": fixture.name,
            "visibility": visibility,
            "schema": fixture.as_artifact_schema(),
        },
        headers=headers,
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["data"]["id"]


class TestDeletingThePublisherLeavesTheConfiguration:
    def test_a_published_server_survives_its_publishers_account(self, client):
        """The reviewer's red, end to end through the routes that do it.

        One admin publishes, another deletes the first. The row, its version
        history and its patch history all have to still be there afterwards —
        the audit trail especially, because cascading it away removes the
        record of what the installation used to be configured to do.
        """
        _publisher_id, publisher = _account(client, "pub", admin=True)
        _other_id, other = _account(client, "other", admin=True)
        with MCPFixture(f"live{uuid.uuid4().hex[:6]}") as fixture:
            artifact_id = _publish(client, publisher, fixture)
            versions_before = _count(
                "SELECT count(*) AS n FROM artifact_version WHERE artifact_id = %s",
                (artifact_id,),
            )
            assert versions_before, "nothing was versioned, so this proves nothing"

            deleted = client.delete(
                f"/v1/admin/users/{_publisher_id}", headers=other
            )

            assert deleted.status_code == 200, deleted.text
            assert _count(
                "SELECT count(*) AS n FROM artifact WHERE id = %s", (artifact_id,)
            ) == 1, "the published server went with the account"
            assert _count(
                "SELECT count(*) AS n FROM artifact_version WHERE artifact_id = %s",
                (artifact_id,),
            ) == versions_before, "the version history cascaded away"

    def test_the_detached_server_is_inert_until_republished(self, client):
        """Preserved is not the same as still trusted.

        The admin attestation is what made this a capability, and the admin is
        gone. So the row stays, the history stays, and no turn gets its tools
        until an admin publishes it again — which is the same answer
        `servers_for_turn` already gives any artifact with no owner.
        """
        publisher_id, publisher = _account(client, "inert", admin=True)
        _other_id, other = _account(client, "other", admin=True)
        with MCPFixture(f"gone{uuid.uuid4().hex[:6]}") as fixture:
            _publish(client, publisher, fixture)
            store = get_runtime().store
            assert any(
                s["url"] == fixture.url for s in mcp_client.servers_for_turn(store)
            ), "it was never a capability, so losing it proves nothing"

            assert (
                client.delete(
                    f"/v1/admin/users/{publisher_id}", headers=other
                ).status_code
                == 200
            )

            assert not any(
                s["url"] == fixture.url for s in mcp_client.servers_for_turn(store)
            ), "a server with no admin behind it is still being offered"

    def test_a_private_artifact_still_dies_with_the_account(self, client):
        """The erasure guarantee is not weakened, only narrowed.

        A private artifact is the account's own, and account deletion has to
        keep meaning that its data is gone. Only publishing — which is an
        admin act that binds the artifact into everyone else's work — changes
        the answer.
        """
        owner_id, owner = _account(client, "priv", admin=True)
        _other_id, other = _account(client, "other", admin=True)
        with MCPFixture(f"mine{uuid.uuid4().hex[:6]}") as fixture:
            artifact_id = _publish(client, owner, fixture, visibility="private")

            assert (
                client.delete(
                    f"/v1/admin/users/{owner_id}", headers=other
                ).status_code
                == 200
            )

            assert _count(
                "SELECT count(*) AS n FROM artifact WHERE id = %s", (artifact_id,)
            ) == 0, "a private artifact outlived the account that owned it"

    def test_the_foreign_key_refuses_rather_than_guessing(self, client):
        """The database's own answer, not the delete path's.

        `delete_user` is one caller. The constraint decides what happens on
        every other path there will ever be — a maintenance statement, a
        future admin flow, a restore — and it cannot see visibility, so every
        answer it could give on its own destroys something. CASCADE removes
        published configuration; SET NULL leaves a private artifact, and its
        payload, behind an account that was deleted.

        So it refuses, and the operation that skipped the lifecycle stops.
        """
        import psycopg

        publisher_id, publisher = _account(client, "fk", admin=True)
        with MCPFixture(f"fk{uuid.uuid4().hex[:6]}") as fixture:
            artifact_id = _publish(client, publisher, fixture)

            # Straight at the table, so no application code is in the way.
            with pytest.raises(psycopg.errors.ForeignKeyViolation):
                with get_runtime().store._connect() as conn:
                    conn.execute(
                        "DELETE FROM app_user WHERE id = %s", (publisher_id,)
                    )

            assert _count(
                "SELECT count(*) AS n FROM artifact WHERE id = %s", (artifact_id,)
            ) == 1, "the refusal did not leave the artifact alone"
            assert _count(
                "SELECT count(*) AS n FROM app_user WHERE id = %s", (publisher_id,)
            ) == 1, "the account went anyway"

    def test_a_private_artifact_is_never_left_behind_by_the_key(self, client):
        """The failure the first correction introduced.

        `SET NULL` was wrong in the direction nobody was watching: a private
        artifact whose owner is deleted survives, unattributed, with its
        payload still under the shared filesystem root. §2.1 says an account's
        private artifacts go with it, so a key that keeps them is a key that
        breaks the erasure guarantee to protect published configuration.
        """
        import psycopg

        owner_id, owner = _account(client, "keep", admin=True)
        with MCPFixture(f"keep{uuid.uuid4().hex[:6]}") as fixture:
            artifact_id = _publish(client, owner, fixture, visibility="private")

            with pytest.raises(psycopg.errors.ForeignKeyViolation):
                with get_runtime().store._connect() as conn:
                    conn.execute("DELETE FROM app_user WHERE id = %s", (owner_id,))

            with get_runtime().store._connect() as conn:
                row = conn.execute(
                    "SELECT owner_user_id FROM artifact WHERE id = %s", (artifact_id,)
                ).fetchone()

            assert row is not None
            assert str(row["owner_user_id"]) == owner_id, (
                "a private artifact was detached from an account that is being "
                "deleted, which outlives the erasure it belongs to"
            )


class TestTheDeletePathDefendsItselfWithoutTheKey:
    """Two mechanisms, and only one of them is in this repository's control.

    The foreign key does the detaching on every path, including ones no code
    here will ever read. But a database provisioned before the migration still
    carries `ON DELETE CASCADE`, and on that database the key is the thing
    destroying published rows. So `delete_user` detaches them itself, first,
    and this proves it does so without help.

    Recorded rather than hidden: reverting the constraint in `sql/schema.sql`
    does not fail any test on an already-provisioned database, because the
    migration is `IF confdeltype = 'c'` and re-applying the file to a database
    that has already been corrected is a no-op. The mutation is invisible to
    the harness, not survivable in production. The two tests below cover the
    behaviour from both sides instead: what the installed key does, and what
    the delete path does when the key is wrong.
    """

    def _constraint_delete_rule(self) -> str:
        with get_runtime().store._connect() as conn:
            row = conn.execute(
                "SELECT c.confdeltype FROM pg_constraint c "
                "WHERE c.conrelid = 'artifact'::regclass AND c.contype = 'f' "
                "AND c.confrelid = 'app_user'::regclass"
            ).fetchone()
        return row["confdeltype"] if row else ""

    def test_the_installed_key_refuses(self):
        """`r` is RESTRICT. `c` was the cascade; `n` the SET NULL that
        replaced it and guessed wrong for private rows."""
        assert self._constraint_delete_rule() == "r"

    def test_erasure_detaches_on_a_database_that_never_migrated(self, client):
        """The key put back the way it was, then a real deletion over it."""
        publisher_id, publisher = _account(client, "unmig", admin=True)
        store = get_runtime().store
        with MCPFixture(f"unmig{uuid.uuid4().hex[:6]}") as fixture:
            artifact_id = _publish(client, publisher, fixture)
            with store._connect() as conn:
                conn.execute(
                    "ALTER TABLE artifact DROP CONSTRAINT artifact_owner_user_id_fkey"
                )
                conn.execute(
                    "ALTER TABLE artifact ADD CONSTRAINT artifact_owner_user_id_fkey "
                    "FOREIGN KEY (owner_user_id) REFERENCES app_user(id) "
                    "ON DELETE CASCADE"
                )
            assert self._constraint_delete_rule() == "c", "the revert did not take"

            try:
                store.delete_user(publisher_id)

                assert _count(
                    "SELECT count(*) AS n FROM artifact WHERE id = %s", (artifact_id,)
                ) == 1, (
                    "with a cascading key, the delete path has to detach the "
                    "row itself and did not"
                )
            finally:
                with store._connect() as conn:
                    conn.execute(
                        "ALTER TABLE artifact "
                        "DROP CONSTRAINT artifact_owner_user_id_fkey"
                    )
                    conn.execute(
                        "ALTER TABLE artifact "
                        "ADD CONSTRAINT artifact_owner_user_id_fkey "
                        "FOREIGN KEY (owner_user_id) REFERENCES app_user(id) "
                        "ON DELETE RESTRICT"
                    )


#: Slow-marked: each of these creates and drops a database. What they check is
#: a migration rather than a request path, so they belong to the release gate
#: rather than to every commit.
@pytest.mark.slow
class TestAFreshDatabaseGetsTheRightKey:
    """What `sql/schema.sql` installs, rather than what this database has.

    The two are different questions and only one of them was covered. An
    already-provisioned database keeps the corrected constraint no matter what
    the file says — the migration is `IF confdeltype = 'c'`, so re-applying it
    to a database that has already been fixed does nothing. So reverting the
    file failed no test, which is a hole for exactly the case that matters:
    a new installation.

    Both directions are checked on a scratch database: the table definition a
    fresh install gets, and the migration an old one gets.
    """

    def _delete_rule(self, dsn: str) -> str:
        import psycopg

        with psycopg.connect(dsn, autocommit=True) as conn:
            row = conn.execute(
                "SELECT c.confdeltype FROM pg_constraint c "
                "WHERE c.conrelid = 'artifact'::regclass AND c.contype = 'f' "
                "AND c.confrelid = 'app_user'::regclass"
            ).fetchone()
        return row[0] if row else ""

    def _scratch(self):
        """A database of its own, so nothing here touches the suite's."""
        import psycopg

        from tests.harness import get_test_store

        base = get_test_store().dsn
        name = f"lifetime_{uuid.uuid4().hex[:10]}"
        admin = base.rsplit("/", 1)[0] + "/postgres"
        with psycopg.connect(admin, autocommit=True) as conn:
            conn.execute(f'CREATE DATABASE "{name}"')
        return name, base.rsplit("/", 1)[0] + f"/{name}", admin

    def _drop(self, admin: str, name: str) -> None:
        import psycopg

        with psycopg.connect(admin, autocommit=True) as conn:
            conn.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')

    def _apply(self, dsn: str) -> None:
        done = subprocess.run(
            ["psql", dsn, "-v", "ON_ERROR_STOP=1", "-v", "embedding_dim=64",
             "-q", "-f", "sql/schema.sql"],
            cwd=ROOT, capture_output=True, text=True, timeout=300,
        )
        assert done.returncode == 0, done.stderr[-2000:]

    def test_a_new_installation_never_had_the_cascade(self):
        name, dsn, admin = self._scratch()
        try:
            self._apply(dsn)

            assert self._delete_rule(dsn) == "r", (
                "a fresh install gets a key that decides an artifact's "
                "lifetime on its own, and it cannot see visibility"
            )
        finally:
            self._drop(admin, name)

    @pytest.mark.parametrize("was", ["CASCADE", "SET NULL"], ids=["cascade", "set_null"])
    def test_an_old_installation_is_migrated(self, was):
        """Both earlier answers, because two databases exist in the wild.

        One never ran the first correction and still cascades. One ran it and
        carries `SET NULL`. The migration has to reach both, which is why its
        condition is `confdeltype <> 'r'` rather than a test for the cascade.
        """
        import psycopg

        name, dsn, admin = self._scratch()
        try:
            self._apply(dsn)
            with psycopg.connect(dsn, autocommit=True) as conn:
                conn.execute(
                    "ALTER TABLE artifact "
                    "DROP CONSTRAINT artifact_owner_user_id_fkey"
                )
                conn.execute(
                    "ALTER TABLE artifact ADD CONSTRAINT artifact_owner_user_id_fkey "
                    f"FOREIGN KEY (owner_user_id) REFERENCES app_user(id) "
                    f"ON DELETE {was}"
                )
            assert self._delete_rule(dsn) == {"CASCADE": "c", "SET NULL": "n"}[was], (
                "the setup did not take"
            )

            self._apply(dsn)

            assert self._delete_rule(dsn) == "r", "the migration did not run"
        finally:
            self._drop(admin, name)
