"""A tenant admin's inspection shows that tenant's objects, artifacts included.

`GET /v1/admin/objects` hands `inspect_state` the admin's own tenant, and
every section honoured it but one. Users, sessions, conversations, messages,
contexts, chunks and training jobs join through `app_user` to the tenant;
artifacts were `SELECT * FROM artifact`. So an admin of one tenant inspecting
their install saw every other tenant's artifacts - owner id, description, the
schema itself and the path on disk - beside a summary that counted only their
own users. Measured by the release-qualification sweep.

`Artifact` has no tenant column. Its tenant is its owner's, which is how
`list_artifacts` and `get_latest_workflow` already resolve it; this branch
now does the same as its siblings.
"""

from __future__ import annotations

import uuid

from liminallm.service.runtime import get_runtime

ACME, GLOBEX = "acme.test", "globex.test"
MARKER = "vermilion-cabinet-launch-code"


def _admin(tenant_id, host):
    runtime = get_runtime()
    user = runtime.store.create_user(
        email=f"adm_{uuid.uuid4().hex[:8]}@t.local", tenant_id=tenant_id
    )
    runtime.store.update_user_role(user.id, role="admin")
    session = runtime.store.create_session(user.id, tenant_id=tenant_id)
    _u, _s, tokens = runtime.auth.issue_tokens_for_session(session.id)
    return user.id, {"Authorization": f"Bearer {tokens['access_token']}",
                     "host": host}


def _artifact(client, headers) -> str:
    resp = client.post(
        "/v1/artifacts",
        headers={**headers, "Idempotency-Key": uuid.uuid4().hex},
        json={"type": "workflow", "name": f"wf-{uuid.uuid4().hex[:6]}",
              "description": MARKER,
              "schema": {"kind": "workflow.chat", "nodes": []}},
    )
    assert resp.status_code in (200, 201), resp.text
    return resp.json()["data"]["id"]


class TestInspectionStaysInTenant:
    def test_another_tenants_artifacts_are_not_listed_or_counted(
        self, client, monkeypatch
    ):
        runtime = get_runtime()
        monkeypatch.setattr(
            runtime.settings, "tenant_domains", {ACME: "acme", GLOBEX: "globex"}
        )
        acme_id, acme_admin = _admin("acme", ACME)
        _gid, globex_admin = _admin("globex", GLOBEX)
        theirs = _artifact(client, acme_admin)

        data = client.get("/v1/admin/objects", headers=globex_admin).json()["data"]

        listed = data["details"]["artifacts"]
        assert not any(row["id"] == theirs for row in listed), "another tenant's artifact listed"
        assert not any(row.get("owner_user_id") == acme_id for row in listed)
        assert MARKER not in str(listed)
        assert data["summary"]["artifacts"] == len(listed)

    def test_the_owning_tenants_admin_still_sees_it(self, client, monkeypatch):
        runtime = get_runtime()
        monkeypatch.setattr(
            runtime.settings, "tenant_domains", {ACME: "acme", GLOBEX: "globex"}
        )
        _aid, acme_admin = _admin("acme", ACME)
        mine = _artifact(client, acme_admin)

        data = client.get("/v1/admin/objects", headers=acme_admin,
                          params={"kind": "artifacts"}).json()["data"]

        assert any(row["id"] == mine for row in data["details"]["artifacts"])
