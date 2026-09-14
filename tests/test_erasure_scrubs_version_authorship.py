"""An erased account's id survives nowhere, including its published history.

A published artifact outlives its owner on purpose (SPEC §12.3): taking one
away because a personnel account was removed is a change nobody reviewed, and
it would destroy the record of what the installation used to do. `delete_user`
therefore detaches `artifact.owner_user_id` rather than deleting the row.

`artifact_version.created_by` was not detached with it. The column is
`TEXT NOT NULL` with no foreign key, so the cascade never reached it, and the
versions endpoint hands the value straight to any caller who can read the
artifact - which for a global one is everybody. Measured before this: after
erasing A, an unrelated account read A's raw UUID back from
`GET /v1/artifacts/{id}/versions`.

Not content exposure and not an authority bypass. It is durable attribution to
an account that asked to be erased, reachable through an ordinary API.

The version is kept and its author replaced, because the history is what has
to survive and the identity is what must not. The column already carries
non-identifying authors - `system_llm`, `human_admin` - so the sentinel needs
no schema change and no new shape for a reader to handle.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

from liminallm.service.runtime import get_runtime
from liminallm.storage.postgres import ERASED_AUTHOR

WORKFLOW = {"kind": "workflow.chat", "nodes": []}


def _account(admin=False):
    runtime = get_runtime()
    user = runtime.store.create_user(
        email=f"cb_{uuid.uuid4().hex[:8]}@t.local",
        role="admin" if admin else "user",
    )
    session = runtime.store.create_session(user.id, tenant_id=user.tenant_id)
    _u, _s, tokens = runtime.auth.issue_tokens_for_session(session.id)
    return user, {"Authorization": f"Bearer {tokens['access_token']}"}


def _publish(client, headers, visibility="global"):
    resp = client.post(
        "/v1/artifacts",
        headers={**headers, "Idempotency-Key": uuid.uuid4().hex},
        json={"type": "workflow", "name": f"pub-{uuid.uuid4().hex[:6]}",
              "description": "published", "visibility": visibility,
              "schema": WORKFLOW},
    )
    assert resp.status_code in (200, 201), resp.text
    return resp.json()["data"]["id"]


def _authors(artifact_id) -> list[str]:
    with get_runtime().store._connect() as conn:
        rows = conn.execute(
            "SELECT created_by FROM artifact_version WHERE artifact_id = %s "
            "ORDER BY version",
            (artifact_id,),
        ).fetchall()
    return [row["created_by"] for row in rows]


def _rows_naming(user_id: str) -> int:
    with get_runtime().store._connect() as conn:
        return conn.execute(
            "SELECT count(*) c FROM artifact_version WHERE created_by = %s",
            (user_id,),
        ).fetchone()["c"]


def _listed_authors(client, headers, artifact_id) -> list[str]:
    resp = client.get(f"/v1/artifacts/{artifact_id}/versions", headers=headers)
    assert resp.status_code == 200, resp.text
    return [row.get("created_by") for row in resp.json()["data"]["items"]]


class TestTheHistorySurvivesAndTheIdentityDoesNot:
    def test_a_published_versions_author_is_replaced_not_removed(self, client):
        runtime = get_runtime()
        author, headers = _account(admin=True)
        _reader, reader_headers = _account()
        artifact_id = _publish(client, headers)
        assert _authors(artifact_id) == [author.id], "the fixture never attributed"

        assert asyncio.run(runtime.auth.delete_user(author.id))

        assert _authors(artifact_id) == [ERASED_AUTHOR]
        assert _rows_naming(author.id) == 0
        assert _listed_authors(client, reader_headers, artifact_id) == [
            ERASED_AUTHOR
        ]

    def test_the_artifact_and_its_history_still_exist(self, client):
        """The half that must not be fixed by deleting the row.

        An erasure that took the published artifact away would pass every
        assertion about the identity being gone.
        """
        runtime = get_runtime()
        author, headers = _account(admin=True)
        _reader, reader_headers = _account()
        artifact_id = _publish(client, headers)
        before = len(_authors(artifact_id))

        assert asyncio.run(runtime.auth.delete_user(author.id))

        assert len(_authors(artifact_id)) == before, "a version was destroyed"
        read = client.get(f"/v1/artifacts/{artifact_id}", headers=reader_headers)
        assert read.status_code == 200, read.text
        assert read.json()["data"]["schema"] == WORKFLOW

    def test_a_version_on_someone_elses_artifact_is_scrubbed_too(self, client):
        """The class, not the instance.

        `apply_config_patch` writes `created_by` from `approver_user_id`, so
        an admin's id lands on versions of artifacts they do not own. Scrubbing
        only the erased account's own artifacts would leave those behind - and
        they are the ones most likely to survive, because somebody else owns
        them.
        """
        runtime = get_runtime()
        owner, owner_headers = _account(admin=True)
        approver, _approver_headers = _account(admin=True)
        _reader, reader_headers = _account()
        artifact_id = _publish(client, owner_headers)

        patch = runtime.store.record_config_patch(
            artifact_id=artifact_id, proposer="human_admin",
            patch={"ops": [{"op": "add", "path": "/reviewed", "value": True}]},
            justification="a second version, authored by the approver",
        )
        runtime.store.update_config_patch_status(patch.id, "approved")
        runtime.config_ops.apply_patch(
            patch.id, approver_user_id=approver.id,
            tenant_id=approver.tenant_id,
        )
        assert approver.id in _authors(artifact_id), "the fixture never attributed"

        assert asyncio.run(runtime.auth.delete_user(approver.id))

        assert _rows_naming(approver.id) == 0
        assert ERASED_AUTHOR in _authors(artifact_id)
        assert approver.id not in _listed_authors(
            client, reader_headers, artifact_id
        )


class TestTheSameIdInTheSameTransactionsOtherRecord:
    """`created_by` was one sighting of a shape, not the shape.

    `apply_config_patch` writes the approver twice: into the version's
    `created_by`, and into `config_patch.meta["applied_by"]`. The patch row
    cascades from the artifact, so it survives exactly when the artifact does
    - and `meta` is a field of `ConfigPatchAuditResponse`, so the listing
    hands it to any admin who may administer that target.

    Found by scanning every text, varchar, uuid and jsonb column in the schema
    for the erased id rather than by reasoning about which ones mattered.
    """

    def _applied_patch(self, client, runtime, artifact_id, approver):
        patch = runtime.store.record_config_patch(
            artifact_id=artifact_id, proposer="human_admin",
            patch={"ops": [{"op": "add", "path": "/checked", "value": True}]},
            justification="a patch somebody applied",
        )
        runtime.store.update_config_patch_status(patch.id, "approved")
        runtime.config_ops.apply_patch(
            patch.id, approver_user_id=approver.id, tenant_id=approver.tenant_id
        )
        return patch.id

    def test_the_applier_of_a_surviving_patch_is_replaced(self, client):
        runtime = get_runtime()
        owner, owner_headers = _account(admin=True)
        approver, _h = _account(admin=True)
        artifact_id = _publish(client, owner_headers)
        patch_id = self._applied_patch(client, runtime, artifact_id, approver)
        before = runtime.store.get_config_patch(patch_id)
        assert (before.meta or {}).get("applied_by") == approver.id, (
            "the fixture never recorded an applier"
        )

        assert asyncio.run(runtime.auth.delete_user(approver.id))

        after = runtime.store.get_config_patch(patch_id)
        assert after is not None, "the patch record was destroyed"
        assert (after.meta or {}).get("applied_by") == ERASED_AUTHOR
        assert after.status == "applied", "the record lost what it was for"

    def test_the_listing_never_hands_out_the_erased_id(self, client):
        runtime = get_runtime()
        owner, owner_headers = _account(admin=True)
        approver, _h = _account(admin=True)
        reader, reader_headers = _account(admin=True)
        artifact_id = _publish(client, owner_headers)
        self._applied_patch(client, runtime, artifact_id, approver)

        assert asyncio.run(runtime.auth.delete_user(approver.id))

        listed = client.get("/v1/config/patches", headers=reader_headers)
        assert listed.status_code == 200, listed.text
        assert approver.id not in listed.text

    def test_another_appliers_record_is_untouched(self, client):
        runtime = get_runtime()
        owner, owner_headers = _account(admin=True)
        kept, _k = _account(admin=True)
        doomed, _d = _account(admin=True)
        artifact_id = _publish(client, owner_headers)
        kept_patch = self._applied_patch(client, runtime, artifact_id, kept)
        self._applied_patch(client, runtime, artifact_id, doomed)

        assert asyncio.run(runtime.auth.delete_user(doomed.id))

        still = runtime.store.get_config_patch(kept_patch)
        assert (still.meta or {}).get("applied_by") == kept.id


class TestNothingElseKeepsTheId:
    """The check that found `config_patch.meta`, kept so it finds the next one.

    Reasoning about which columns matter is how `created_by` was missed twice
    - once when the erasure was written, once when this sweep first called
    account deletion clean. Asking the schema is not reasoning.
    """

    #: The one place an erased id is supposed to remain, and why: the
    #: retirement is what reclaims the account's filesystem namespace, and it
    #: is defined by outliving the account (`hold_user_lifetime` asks for a
    #: retirement whose `app_user` row is gone). It reaches no API, and
    #: `clear_user_namespace_retirement` removes it once the namespace is
    #: collected.
    ALLOWED = {("user_namespace_retirement", "user_id")}

    def test_no_column_keeps_an_erased_accounts_id(self, client):
        runtime = get_runtime()
        doomed, headers = _account(admin=True)
        other, other_headers = _account(admin=True)

        # Spread the id as widely as the app's own write paths allow.
        client.post("/v1/conversations", headers=headers, json={"title": "t"})
        client.post("/v1/notes", headers=headers,
                    json={"title": "n", "content": "c"})
        client.post("/v1/contexts", headers=headers,
                    json={"name": f"c-{uuid.uuid4().hex[:6]}", "description": "d"})
        for visibility in ("private", "global", "shared"):
            _publish(client, headers, visibility)
        client.post("/v1/auth/api-keys", headers=headers, json={"name": "k"})
        theirs = _publish(client, other_headers)
        patch = runtime.store.record_config_patch(
            artifact_id=theirs, proposer="human_admin",
            patch={"ops": [{"op": "add", "path": "/x", "value": 1}]},
            justification="j")
        runtime.store.update_config_patch_status(patch.id, "approved")
        runtime.config_ops.apply_patch(
            patch.id, approver_user_id=doomed.id, tenant_id=doomed.tenant_id
        )

        before = self._columns_naming(doomed.id)
        assert len(before) > 5, f"the fixture barely spread the id: {before}"

        assert asyncio.run(runtime.auth.delete_user(doomed.id))

        remaining = set(self._columns_naming(doomed.id))
        assert remaining <= self.ALLOWED, (
            f"an erased account's id survives in {sorted(remaining - self.ALLOWED)}"
        )

    @staticmethod
    def _columns_naming(user_id: str) -> list:
        """Every column in the schema whose text form equals this id."""
        store = get_runtime().store
        found = []
        with store._connect() as conn:
            columns = conn.execute(
                "SELECT table_name, column_name, data_type "
                "FROM information_schema.columns WHERE table_schema = 'public' "
                "AND data_type IN ('text','character varying','uuid','jsonb')"
            ).fetchall()
            for row in columns:
                table, column = row["table_name"], row["column_name"]
                if row["data_type"] == "jsonb":
                    sql = (f'SELECT count(*) c FROM "{table}" '
                           f'WHERE "{column}"::text LIKE %s')
                    param = f"%{user_id}%"
                else:
                    sql = (f'SELECT count(*) c FROM "{table}" '
                           f'WHERE "{column}"::text = %s')
                    param = user_id
                try:
                    if conn.execute(sql, (param,)).fetchone()["c"]:
                        found.append((table, column))
                except Exception:
                    conn.rollback()
        return found


class TestItScrubsOnlyTheErasedAccount:
    def test_another_accounts_authorship_is_untouched(self, client):
        """The door must not be a wall.

        Replacing every author would satisfy every assertion above, and would
        erase the attribution of accounts that still exist.
        """
        runtime = get_runtime()
        owner, owner_headers = _account(admin=True)
        approver, _h = _account(admin=True)
        doomed, _d = _account(admin=True)
        artifact_id = _publish(client, owner_headers)

        for admin in (approver, doomed):
            patch = runtime.store.record_config_patch(
                artifact_id=artifact_id, proposer="human_admin",
                patch={"ops": [{"op": "add", "path": f"/by_{admin.id[:8]}",
                                "value": True}]},
                justification="a version",
            )
            runtime.store.update_config_patch_status(patch.id, "approved")
            runtime.config_ops.apply_patch(
                patch.id, approver_user_id=admin.id, tenant_id=admin.tenant_id
            )

        assert asyncio.run(runtime.auth.delete_user(doomed.id))

        authors = _authors(artifact_id)
        assert doomed.id not in authors
        assert approver.id in authors, "an account that still exists lost its name"
        assert owner.id in authors, "the original author lost their name"

    @pytest.mark.parametrize("sentinel", ["system_llm", "human_admin"])
    def test_an_existing_non_identifying_author_is_left_alone(
        self, client, sentinel
    ):
        """The column's other authors are not account ids and are not a
        deletion's business."""
        runtime = get_runtime()
        owner, owner_headers = _account(admin=True)
        artifact_id = _publish(client, owner_headers)
        with runtime.store._connect() as conn:
            conn.execute(
                "UPDATE artifact_version SET created_by = %s WHERE artifact_id = %s",
                (sentinel, artifact_id),
            )
        doomed, _d = _account(admin=True)

        assert asyncio.run(runtime.auth.delete_user(doomed.id))

        assert _authors(artifact_id) == [sentinel]
