"""An admin's read of another user's artifact stops at the tenant edge.

`_get_owned_artifact` is the read capability behind `GET /artifacts/{id}`,
its version history, and a tool's spec and invocation. Its admin bypass -
an admin may read what another user owns - carried no tenant condition, while
every sibling admin surface does: the user listing refuses another tenant,
erasure refuses another tenant's user, and inspection is scoped to the
admin's own. So an admin of one tenant could read any tenant's private
artifact by id, schema and version history included. Measured by the
release-qualification sweep.

An artifact's tenant is its owner's, which is how `list_artifacts` and
`get_latest_workflow` already resolve it. An ownerless artifact - a published
one whose publisher was erased - has no tenant to compare and stays readable
to any admin, as before.
"""

from __future__ import annotations

import uuid

from liminallm.service.runtime import get_runtime

ACME, GLOBEX = "acme.test", "globex.test"
MARKER = "vermilion-cabinet-launch-code"


def _account(tenant_id, host, *, admin=False):
    runtime = get_runtime()
    user = runtime.store.create_user(
        email=f"r_{uuid.uuid4().hex[:8]}@t.local", tenant_id=tenant_id
    )
    if admin:
        runtime.store.update_user_role(user.id, role="admin")
    session = runtime.store.create_session(user.id, tenant_id=tenant_id)
    _u, _s, tokens = runtime.auth.issue_tokens_for_session(session.id)
    return user.id, {"Authorization": f"Bearer {tokens['access_token']}",
                     "host": host}


def _artifact(client, headers, visibility="private") -> str:
    resp = client.post(
        "/v1/artifacts",
        headers={**headers, "Idempotency-Key": uuid.uuid4().hex},
        json={"type": "workflow", "name": f"wf-{uuid.uuid4().hex[:6]}",
              "description": MARKER, "visibility": visibility,
              "schema": {"kind": "workflow.chat", "nodes": []}},
    )
    assert resp.status_code in (200, 201), resp.text
    return resp.json()["data"]["id"]


class TestAnAdminReadsWithinTheirTenant:
    def test_another_tenants_private_artifact_is_refused_by_id(
        self, client, monkeypatch
    ):
        runtime = get_runtime()
        monkeypatch.setattr(
            runtime.settings, "tenant_domains", {ACME: "acme", GLOBEX: "globex"}
        )
        _uid, user = _account("acme", ACME)
        _sid, same_tenant_admin = _account("acme", ACME, admin=True)
        _oid, other_tenant_admin = _account("globex", GLOBEX, admin=True)
        theirs = _artifact(client, user)

        for path in (f"/v1/artifacts/{theirs}", f"/v1/artifacts/{theirs}/versions"):
            refused = client.get(path, headers=other_tenant_admin)
            assert refused.status_code == 403, refused.text
            assert MARKER not in refused.text
            allowed = client.get(path, headers=same_tenant_admin)
            assert allowed.status_code == 200, allowed.text

    def test_an_ownerless_published_artifact_stays_readable_to_any_admin(
        self, client, monkeypatch
    ):
        """The unchanged half: with no owner there is no tenant to compare."""
        runtime = get_runtime()
        monkeypatch.setattr(
            runtime.settings, "tenant_domains", {ACME: "acme", GLOBEX: "globex"}
        )
        publisher_id, publisher = _account("acme", ACME, admin=True)
        _aid, second_admin = _account("acme", ACME, admin=True)
        _oid, other_tenant_admin = _account("globex", GLOBEX, admin=True)
        published = _artifact(client, publisher, visibility="global")

        erased = client.delete(f"/v1/admin/users/{publisher_id}", headers=second_admin)
        assert erased.status_code == 200, erased.text

        for who in (second_admin, other_tenant_admin):
            assert client.get(f"/v1/artifacts/{published}", headers=who).status_code == 200
