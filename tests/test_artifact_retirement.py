"""Retiring a private artifact, and everything that hangs off one.

SPEC §12.3 gives users CRUD over their *private* artifacts. There was no
DELETE at all, and the PATCH that existed used a read-capability helper as its
mutation rule — so an admin could edit a global system workflow through the
ordinary artifact endpoint, which is the change ConfigOps exists to review.

Deletion is harder than the context case for two reasons. An artifact's
versions are files on disk as well as rows, so the order matters: revoke the
database capability, commit, then clean the payload. And an adapter owns
weights that a training worker may be writing right now, so deleting one has
to be serialized against training rather than racing it.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from liminallm.service.runtime import get_runtime


def _unique(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _account(client, *, admin=False):
    email = f"{_unique('art')}@example.com"
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


def _artifact(client, headers, *, name=None, kind="workflow.chat") -> str:
    resp = client.post(
        "/v1/artifacts",
        headers={**headers, "Idempotency-Key": _unique("k")},
        json={
            "type": "workflow",
            "name": name or _unique("wf"),
            "description": "a private workflow",
            "schema": {"kind": kind, "nodes": []},
        },
    )
    assert resp.status_code in (200, 201), resp.text
    return resp.json()["data"]["id"]


def _global_artifact(owner_user_id=None) -> str:
    """A system artifact: no owner, global visibility. ConfigOps territory."""
    store = get_runtime().store
    artifact = store.create_artifact(
        "workflow",
        _unique("system_wf"),
        {"kind": "workflow.chat", "nodes": []},
        "a system workflow",
        visibility="global",
        version_author="system_llm",
    )
    if owner_user_id is None:
        with store._connect() as conn:
            conn.execute(
                "UPDATE artifact SET owner_user_id = NULL WHERE id = %s",
                (artifact.id,),
            )
    return artifact.id


def _count(sql, params) -> int:
    with get_runtime().store._connect() as conn:
        return int(conn.execute(sql, params).fetchone()["n"])


def _payload_dir(artifact_id: str) -> Path:
    return Path(get_runtime().settings.shared_fs_root) / "artifacts" / artifact_id


class TestRetiringAPrivateArtifact:
    def test_the_owner_deletes_it_with_its_versions(self, client):
        user_id, headers = _account(client)
        artifact_id = _artifact(client, headers)
        # A second version, so there is history to lose.
        assert client.patch(
            f"/v1/artifacts/{artifact_id}",
            headers=headers,
            json={"description": "edited"},
        ).status_code == 200
        assert _count(
            "SELECT COUNT(*) AS n FROM artifact_version WHERE artifact_id = %s",
            (artifact_id,),
        ) >= 2

        resp = client.delete(f"/v1/artifacts/{artifact_id}", headers=headers)
        assert resp.status_code == 200, resp.text

        assert _count(
            "SELECT COUNT(*) AS n FROM artifact WHERE id = %s", (artifact_id,)
        ) == 0
        assert _count(
            "SELECT COUNT(*) AS n FROM artifact_version WHERE artifact_id = %s",
            (artifact_id,),
        ) == 0
        assert _count(
            "SELECT COUNT(*) AS n FROM config_patch WHERE artifact_id = %s",
            (artifact_id,),
        ) == 0
        assert client.get(
            f"/v1/artifacts/{artifact_id}", headers=headers
        ).status_code == 404

    def test_the_server_owned_payload_is_retired_by_the_sweep(self, client):
        """Versions live on disk as well as in rows.

        The request revokes the capability and returns; the bytes go later.
        Unlinking them inside the request would let a caller that had already
        resolved the artifact read a filesystem where it no longer exists.
        """
        from liminallm.service.artifacts import sweep_artifact_payloads

        _, headers = _account(client)
        artifact_id = _artifact(client, headers)
        payload = _payload_dir(artifact_id)
        assert payload.is_dir(), f"nothing was written under {payload}"

        runtime = get_runtime()
        assert client.delete(
            f"/v1/artifacts/{artifact_id}", headers=headers
        ).status_code == 200
        assert payload.is_dir(), "the request unlinked the payload itself"

        sweep_artifact_payloads(
            runtime.store, runtime.settings.shared_fs_root, grace_seconds=0
        )
        assert not payload.exists(), (
            "the artifact's version payloads outlived the artifact"
        )

    def test_another_user_gets_nothing(self, client):
        _, owner_headers = _account(client)
        _, other_headers = _account(client)
        artifact_id = _artifact(client, owner_headers, name="private-workflow")

        resp = client.delete(f"/v1/artifacts/{artifact_id}", headers=other_headers)
        assert resp.status_code in (403, 404), resp.text
        assert "private-workflow" not in resp.text
        assert _count(
            "SELECT COUNT(*) AS n FROM artifact WHERE id = %s", (artifact_id,)
        ) == 1


class TestPrivateMeansPrivate:
    """The mutation rule is ownership plus private visibility, for both verbs.

    `_get_owned_artifact` is a *read* helper: it lets an admin through to
    another user's artifact and to ownerless system artifacts, which is right
    for viewing. Using it as the rule for PATCH made `/artifacts/{id}` a way
    to edit a global system workflow directly, bypassing the ConfigOps review
    that governs those. §12.3 scopes user CRUD to private artifacts; system
    artifacts change through ConfigOps.
    """

    def test_an_admin_cannot_patch_a_system_artifact_through_this_route(self, client):
        _, admin_headers = _account(client, admin=True)
        artifact_id = _global_artifact()

        resp = client.patch(
            f"/v1/artifacts/{artifact_id}",
            headers=admin_headers,
            json={"description": "edited around ConfigOps"},
        )
        assert resp.status_code in (403, 404), resp.text
        artifact = get_runtime().store.get_artifact(artifact_id)
        assert artifact.description == "a system workflow"

    def test_an_admin_cannot_delete_a_system_artifact_through_this_route(self, client):
        _, admin_headers = _account(client, admin=True)
        artifact_id = _global_artifact()

        resp = client.delete(f"/v1/artifacts/{artifact_id}", headers=admin_headers)
        assert resp.status_code in (403, 404), resp.text
        assert _count(
            "SELECT COUNT(*) AS n FROM artifact WHERE id = %s", (artifact_id,)
        ) == 1

    def test_an_admin_cannot_patch_another_users_private_artifact(self, client):
        _, owner_headers = _account(client)
        _, admin_headers = _account(client, admin=True)
        artifact_id = _artifact(client, owner_headers)

        resp = client.patch(
            f"/v1/artifacts/{artifact_id}",
            headers=admin_headers,
            json={"description": "not yours"},
        )
        assert resp.status_code in (403, 404), resp.text
        assert get_runtime().store.get_artifact(artifact_id).description == (
            "a private workflow"
        )

    def test_the_owner_cannot_delete_their_own_shared_artifact(self, client):
        """Publishing changes who the artifact belongs to.

        A shared or global artifact may be bound into other users' work, so
        retiring one is not a private decision any more.
        """
        user_id, headers = _account(client)
        artifact_id = _artifact(client, headers)
        with get_runtime().store._connect() as conn:
            conn.execute(
                "UPDATE artifact SET visibility = 'shared' WHERE id = %s",
                (artifact_id,),
            )

        resp = client.delete(f"/v1/artifacts/{artifact_id}", headers=headers)
        assert resp.status_code in (403, 409), resp.text
        assert _count(
            "SELECT COUNT(*) AS n FROM artifact WHERE id = %s", (artifact_id,)
        ) == 1


def _adapter(client, headers, user_id, *, fs_dir=None) -> str:
    """A private adapter artifact with weights on disk."""
    store = get_runtime().store
    schema = {
        "kind": "adapter.lora",
        "base_model": "test-base",
        "current_version": 1,
        "rank": 4,
    }
    if fs_dir is not None:
        schema["fs_dir"] = str(fs_dir)
    artifact = store.create_artifact(
        "adapter",
        _unique("ad"),
        schema,
        "an adapter",
        owner_user_id=user_id,
        visibility="private",
        version_author=user_id,
    )
    root = Path(get_runtime().settings.shared_fs_root) / "adapters" / artifact.id
    (root / "v0001").mkdir(parents=True, exist_ok=True)
    (root / "v0001" / "params.json").write_text(json.dumps({"w": [1, 2]}))
    (root / "latest").write_text("v0001")
    return artifact.id


class TestRetiringAnAdapterAgainstTraining:
    def test_a_running_job_refuses_the_delete(self, client):
        """Weights are being written right now; the row cannot go.

        `training_job.adapter_id` cascades, so deleting the artifact would
        take the job record with it while the worker carried on writing the
        adapter tree and then tried to promote a version onto an artifact
        that no longer exists.
        """
        user_id, headers = _account(client)
        adapter_id = _adapter(client, headers, user_id)
        store = get_runtime().store
        job = store.create_training_job(user_id=user_id, adapter_id=adapter_id)
        assert store.claim_training_job(job.id) is not None, "could not start the job"

        resp = client.delete(f"/v1/artifacts/{adapter_id}", headers=headers)
        assert resp.status_code == 409, resp.text
        assert _count(
            "SELECT COUNT(*) AS n FROM artifact WHERE id = %s", (adapter_id,)
        ) == 1
        assert (
            Path(get_runtime().settings.shared_fs_root) / "adapters" / adapter_id
        ).is_dir(), "the weights were removed while training was running"

    def test_a_queued_job_cannot_be_claimed_after_the_delete_wins(self, client):
        """Deletion takes the queued job with it, and the worker sees that."""
        user_id, headers = _account(client)
        adapter_id = _adapter(client, headers, user_id)
        store = get_runtime().store
        job = store.create_training_job(user_id=user_id, adapter_id=adapter_id)

        assert client.delete(
            f"/v1/artifacts/{adapter_id}", headers=headers
        ).status_code == 200
        assert store.claim_training_job(job.id) is None, (
            "a worker claimed a job whose adapter had been deleted"
        )
        assert _count(
            "SELECT COUNT(*) AS n FROM training_job WHERE id = %s", (job.id,)
        ) == 0

    def test_only_this_adapters_tree_is_removed(self, client):
        from liminallm.service.artifacts import sweep_artifact_payloads

        user_id, headers = _account(client)
        doomed = _adapter(client, headers, user_id)
        sibling = _adapter(client, headers, user_id)
        runtime = get_runtime()
        root = Path(runtime.settings.shared_fs_root) / "adapters"

        assert client.delete(
            f"/v1/artifacts/{doomed}", headers=headers
        ).status_code == 200
        sweep_artifact_payloads(
            runtime.store, runtime.settings.shared_fs_root, grace_seconds=0
        )
        assert not (root / doomed).exists()
        assert (root / sibling / "v0001" / "params.json").exists(), (
            "deleting one adapter took another adapter's weights"
        )

    def test_a_schema_named_directory_is_never_the_deletion_target(self, client):
        """`schema.fs_dir` is user-editable, so it cannot authorize `rmtree`.

        Path resolution accepts an explicit directory whose final component
        matches the adapter id, which is enough to stop adapter A *serving* B's
        weights. It is not authority to destroy: a schema naming
        `<shared>/something-important/<own-artifact-id>` satisfies that rule
        while pointing at somebody else's data. Cleanup therefore derives its
        target from the artifact id alone.
        """
        user_id, headers = _account(client)
        root = Path(get_runtime().settings.shared_fs_root)

        # Stand up a victim directory, then name it from the adapter's schema.
        placeholder = _adapter(client, headers, user_id)
        victim_parent = root / "irreplaceable"
        victim = victim_parent / placeholder
        victim.mkdir(parents=True, exist_ok=True)
        (victim / "keep.json").write_text("important")
        with get_runtime().store._connect() as conn:
            conn.execute(
                "UPDATE artifact SET schema = schema || %s::jsonb WHERE id = %s",
                (json.dumps({"fs_dir": str(victim)}), placeholder),
            )

        assert client.delete(
            f"/v1/artifacts/{placeholder}", headers=headers
        ).status_code == 200
        from liminallm.service.artifacts import sweep_artifact_payloads

        sweep_artifact_payloads(
            get_runtime().store,
            get_runtime().settings.shared_fs_root,
            grace_seconds=0,
        )
        assert (victim / "keep.json").exists(), (
            "deletion used schema.fs_dir as its target and destroyed a path "
            "the artifact merely named"
        )


class TestPatchAgainstDelete:
    def test_one_serial_outcome_and_no_row_pointing_at_nothing(self, client):
        """Whoever wins, the durable state is consistent.

        The failure to avoid is a surviving artifact row whose version
        payloads were cleaned up underneath it.
        """
        from liminallm.api import routes as routes_module

        _, headers = _account(client)
        artifact_id = _artifact(client, headers)
        deleted: dict = {}

        real_update = get_runtime().store.update_artifact

        def update_then_delete(*a, **kw):
            if not deleted:
                resp = client.delete(
                    f"/v1/artifacts/{artifact_id}", headers=headers
                )
                deleted["status"] = resp.status_code
            return real_update(*a, **kw)

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(get_runtime().store, "update_artifact", update_then_delete)
        try:
            patched = client.patch(
                f"/v1/artifacts/{artifact_id}",
                headers=headers,
                json={"description": "edited during deletion"},
            )
        finally:
            monkeypatch.undo()

        assert deleted.get("status") == 200, "the deletion under test did not happen"
        rows = _count(
            "SELECT COUNT(*) AS n FROM artifact WHERE id = %s", (artifact_id,)
        )
        if rows:
            # The PATCH recreated or kept the row: its payloads must be there.
            assert _payload_dir(artifact_id).is_dir(), (
                "an artifact row survived while its version payloads were "
                "cleaned up underneath it"
            )
        else:
            assert patched.status_code >= 400, (
                "the PATCH reported success for an artifact that is gone: "
                f"{patched.status_code} {patched.text[:300]}"
            )


class TestDeletionDoesNotYankFilesFromUnderAReader:
    """The writer was serialized. The reader was not.

    A turn resolves a promoted adapter from Postgres and only then touches
    disk: `params_path.stat()` comes after the capability has been acquired,
    and the in-memory cache is consulted after that stat. DELETE committed the
    row removal and immediately `rmtree`'d the adapter tree, so a turn that
    had legitimately acquired the adapter could reach the filesystem after it
    was gone.

    That state has no serial explanation. If the turn ran first it held the
    adapter and should be able to finish; if the delete ran first the turn
    should never have acquired it. Reclamation therefore stops being part of
    the request: DELETE revokes the capability and returns, and the payloads
    are collected later by a sweep whose grace period outlives any in-flight
    request.
    """

    def test_a_turn_that_already_acquired_the_adapter_can_still_read_it(
        self, client
    ):
        user_id, headers = _account(client)
        adapter_id = _adapter(client, headers, user_id)
        root = Path(get_runtime().settings.shared_fs_root) / "adapters" / adapter_id
        params = root / "v0001" / "params.json"
        assert params.exists()

        # The turn has resolved the adapter and is about to touch disk.
        deleted = client.delete(f"/v1/artifacts/{adapter_id}", headers=headers)
        assert deleted.status_code == 200, deleted.text

        # It resumes. The bytes it already had a capability for are still here.
        assert params.exists(), (
            "the adapter's weights were removed inside the DELETE request, so "
            "a turn holding the pre-delete capability reads a post-delete "
            "filesystem"
        )
        assert params.read_bytes(), "the weights are empty"

    def test_the_sweep_collects_the_payloads_once_they_are_orphans(self, client):
        """Delayed, not skipped. The storage still goes."""
        from liminallm.service.artifacts import sweep_artifact_payloads

        user_id, headers = _account(client)
        adapter_id = _adapter(client, headers, user_id)
        workflow_id = _artifact(client, headers)
        runtime = get_runtime()
        root = Path(runtime.settings.shared_fs_root)

        for artifact_id in (adapter_id, workflow_id):
            assert client.delete(
                f"/v1/artifacts/{artifact_id}", headers=headers
            ).status_code == 200

        removed = sweep_artifact_payloads(
            runtime.store, runtime.settings.shared_fs_root, grace_seconds=0
        )
        assert removed >= 2, f"the sweep collected {removed} directories"
        assert not (root / "adapters" / adapter_id).exists()
        assert not (root / "artifacts" / workflow_id).exists()

    def test_the_sweep_leaves_a_live_artifact_alone(self, client):
        """Its rule is "no artifact row names this id", nothing looser."""
        from liminallm.service.artifacts import sweep_artifact_payloads

        user_id, headers = _account(client)
        keeper = _adapter(client, headers, user_id)
        runtime = get_runtime()
        root = Path(runtime.settings.shared_fs_root)

        sweep_artifact_payloads(
            runtime.store, runtime.settings.shared_fs_root, grace_seconds=0
        )
        assert (root / "adapters" / keeper / "v0001" / "params.json").exists(), (
            "the sweep removed the payloads of an artifact that still exists"
        )

    def test_the_grace_period_protects_a_recent_orphan(self, client):
        """Long enough to outlive any request that already holds the id."""
        from liminallm.service.artifacts import sweep_artifact_payloads

        user_id, headers = _account(client)
        adapter_id = _adapter(client, headers, user_id)
        runtime = get_runtime()
        root = Path(runtime.settings.shared_fs_root)
        assert client.delete(
            f"/v1/artifacts/{adapter_id}", headers=headers
        ).status_code == 200

        removed = sweep_artifact_payloads(
            runtime.store, runtime.settings.shared_fs_root, grace_seconds=3600
        )
        assert removed == 0
        assert (root / "adapters" / adapter_id).is_dir(), (
            "an orphan younger than the grace period was collected, which is "
            "exactly the window an in-flight request needs"
        )


class TestPatchCarriesItsOwnPredicate:
    def test_publishing_between_the_check_and_the_write_stops_the_patch(
        self, client
    ):
        """The authorization has to be in the statement that mutates.

        PATCH validated `private` and then called a generic update that locked
        and wrote by id alone. Anything that publishes the artifact in between
        — config ops, an admin action, a future share endpoint — lands after
        the check and before the write, and the edit goes through on an
        artifact that is no longer the caller's alone.
        """
        user_id, headers = _account(client)
        artifact_id = _artifact(client, headers, name="about-to-be-published")
        published: dict = {}

        store = get_runtime().store
        real_get = store.get_private_artifact

        def get_then_publish(*a, **kw):
            artifact = real_get(*a, **kw)
            if artifact is not None and not published:
                with store._connect() as conn:
                    conn.execute(
                        "UPDATE artifact SET visibility = 'shared' WHERE id = %s",
                        (artifact_id,),
                    )
                published["done"] = True
            return artifact

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(store, "get_private_artifact", get_then_publish)
        try:
            resp = client.patch(
                f"/v1/artifacts/{artifact_id}",
                headers=headers,
                json={"description": "edited after publication"},
            )
        finally:
            monkeypatch.undo()

        assert published.get("done"), "the publication under test did not happen"
        assert resp.status_code >= 400, (
            "the patch was applied to an artifact that had been published "
            f"between the check and the write: {resp.status_code}"
        )
        assert store.get_artifact(artifact_id).description == (
            "a private workflow"
        ), "the description changed on a published artifact"


def test_list_artifacts_uses_the_same_mapping_as_every_other_reader():
    """One mapper, or the copies drift.

    `list_contexts` hand-built its rows and silently dropped a column the
    model had gained. The artifact listing was the last hand-written copy.
    """
    import inspect

    from liminallm.storage.postgres import PostgresStore

    source = inspect.getsource(PostgresStore.list_artifacts)
    assert "_artifact_from_row" in source, (
        "list_artifacts still builds Artifact(...) by hand"
    )
    assert "Artifact(" not in source, (
        "list_artifacts still has a hand-written mapping beside the shared one"
    )
