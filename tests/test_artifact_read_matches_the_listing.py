"""What the listing shows you, you can read by id - and run.

Three surfaces resolve one artifact for one caller, and they disagreed.
`list_artifacts` pages a user's own private artifacts, every global one, and
shared ones inside the owner's tenant - returning the whole schema. The engine
runs exactly those same tiers, so a chat turn naming a global workflow ran it.
Reading by id was narrower than both: `_get_owned_artifact` asked only who
owned the row, so a user whose listing had already handed them a global
artifact's schema, and who could run it, was refused 403 when they asked for
it by id. The refusal protected nothing and surprised everyone.

The rule now lives once, in the store, and all three ask it.
"""

from __future__ import annotations

import uuid

import pytest

from liminallm.service.runtime import get_runtime

ACME, GLOBEX = "acme.test", "globex.test"
MARKER = "vermilion-cabinet-launch-code"

TOOL_SCHEMA = {
    "kind": "tool.spec",
    "name": "marker_echo",
    "handler": "llm.generic",
    "description": MARKER,
}


def _account(tenant_id, host, *, admin=False):
    runtime = get_runtime()
    user = runtime.store.create_user(
        email=f"v_{uuid.uuid4().hex[:8]}@t.local", tenant_id=tenant_id
    )
    if admin:
        runtime.store.update_user_role(user.id, role="admin")
    session = runtime.store.create_session(user.id, tenant_id=tenant_id)
    _u, _s, tokens = runtime.auth.issue_tokens_for_session(session.id)
    return user.id, {"Authorization": f"Bearer {tokens['access_token']}",
                     "host": host}


def _artifact(client, headers, visibility, *, schema=None, type_="workflow"):
    resp = client.post(
        "/v1/artifacts",
        headers={**headers, "Idempotency-Key": uuid.uuid4().hex},
        json={"type": type_, "name": f"a-{uuid.uuid4().hex[:6]}",
              "description": MARKER, "visibility": visibility,
              "schema": schema or {"kind": "workflow.chat", "nodes": []}},
    )
    assert resp.status_code in (200, 201), resp.text
    return resp.json()["data"]["id"]


def _listed(client, headers) -> set[str]:
    resp = client.get("/v1/artifacts", headers=headers)
    assert resp.status_code == 200, resp.text
    return {row["id"] for row in resp.json()["data"]["items"]}


@pytest.fixture
def tenants(monkeypatch):
    monkeypatch.setattr(
        get_runtime().settings, "tenant_domains", {ACME: "acme", GLOBEX: "globex"}
    )


class TestReadingByIdMatchesTheListing:
    def test_a_global_artifact_is_readable_by_anyone_who_can_list_it(
        self, client, tenants
    ):
        _pid, publisher = _account("acme", ACME, admin=True)
        _uid, user = _account("acme", ACME)
        _oid, outsider = _account("globex", GLOBEX)
        published = _artifact(client, publisher, "global")

        for label, who in (("same tenant", user), ("other tenant", outsider)):
            assert published in _listed(client, who), label
            assert client.get(f"/v1/artifacts/{published}", headers=who).status_code == 200
            assert client.get(
                f"/v1/artifacts/{published}/versions", headers=who
            ).status_code == 200

    def test_a_shared_artifact_reads_inside_its_tenant_and_not_outside(
        self, client, tenants
    ):
        _pid, publisher = _account("acme", ACME, admin=True)
        _uid, same_tenant = _account("acme", ACME)
        _oid, outsider = _account("globex", GLOBEX)
        shared = _artifact(client, publisher, "shared")

        assert shared in _listed(client, same_tenant)
        assert client.get(f"/v1/artifacts/{shared}", headers=same_tenant).status_code == 200

        assert shared not in _listed(client, outsider)
        refused = client.get(f"/v1/artifacts/{shared}", headers=outsider)
        assert refused.status_code == 403, refused.text
        assert MARKER not in refused.text

    def test_a_private_artifact_is_still_only_its_owners(self, client, tenants):
        _oid, owner = _account("acme", ACME)
        _sid, stranger = _account("acme", ACME)
        private = _artifact(client, owner, "private")

        assert private not in _listed(client, stranger)
        refused = client.get(f"/v1/artifacts/{private}", headers=stranger)
        assert refused.status_code == 403, refused.text
        assert MARKER not in refused.text
        assert client.get(f"/v1/artifacts/{private}", headers=owner).status_code == 200

    def test_the_three_surfaces_now_agree_on_one_artifact(self, client, tenants):
        """List, read and run, asked of the same global workflow by the same
        caller: the disagreement this closes was among these three answers."""
        _pid, publisher = _account("acme", ACME, admin=True)
        _uid, user = _account("acme", ACME)
        published = _artifact(client, publisher, "global", schema={
            "kind": "workflow.chat", "entrypoint": "plain_chat", "nodes": [
                {"id": "plain_chat", "type": "tool_call", "tool": "llm.generic",
                 "inputs": {"message": "${input.message}"}, "next": "end"},
                {"id": "end", "type": "end"}]})

        listed = published in _listed(client, user)
        read = client.get(f"/v1/artifacts/{published}", headers=user).status_code
        ran = client.post(
            "/v1/chat", headers=user,
            json={"message": {"content": "hi", "mode": "text"}, "stream": False,
                  "workflow_id": published},
        ).status_code

        assert (listed, read, ran) == (True, 200, 200)


class TestInvokingFollowsTheSameRule:
    def test_a_global_tool_is_invocable_by_anyone_it_is_published_to(
        self, client, tenants
    ):
        """Publishing globally takes an admin, and means everyone. The engine
        already ran a global *workflow* for any caller; a global tool spec was
        refused at `invoke` by the same helper that refused the read."""
        _pid, publisher = _account("acme", ACME, admin=True)
        _uid, user = _account("acme", ACME)
        tool = _artifact(client, publisher, "global", schema=TOOL_SCHEMA,
                         type_="tool")

        spec = client.get(f"/v1/tools/specs/{tool}", headers=user)
        invoked = client.post(f"/v1/tools/{tool}/invoke", headers=user,
                              json={"inputs": {}})

        assert spec.status_code == 200, spec.text
        assert invoked.status_code != 403, invoked.text

    def test_a_private_tool_is_not_invocable_by_a_stranger(self, client, tenants):
        _oid, owner = _account("acme", ACME)
        _sid, stranger = _account("acme", ACME)
        tool = _artifact(client, owner, "private", schema=TOOL_SCHEMA, type_="tool")

        invoked = client.post(f"/v1/tools/{tool}/invoke", headers=stranger,
                              json={"inputs": {}})

        assert invoked.status_code == 403, invoked.text
        assert MARKER not in invoked.text


class TestAnUnknownVisibilityIsNotALicence:
    def test_a_visibility_nobody_recognises_reaches_nobody(self, client, tenants):
        """The tiers are a closed set, and the default is refusal.

        The API only ever writes private, shared or global, so this row is
        made in the store - but a migration, a hand-edit or a future tier
        added in one place and not the other can produce one, and "not a
        recognised tier" must mean no, not yes. The rule enforced this before
        it was lifted out of `get_latest_workflow`; nothing witnessed it.
        """
        runtime = get_runtime()
        _oid, owner = _account("acme", ACME)
        _sid, stranger = _account("acme", ACME)
        odd = _artifact(client, owner, "private")
        with runtime.store._connect() as conn:
            conn.execute(
                "UPDATE artifact SET visibility = %s WHERE id = %s",
                ("everyone-ish", odd),
            )

        refused = client.get(f"/v1/artifacts/{odd}", headers=stranger)

        assert refused.status_code == 403, refused.text
        assert MARKER not in refused.text
        assert odd not in _listed(client, stranger)


class TestTheAdminPathsAreUnchanged:
    def test_an_ownerless_system_artifact_stays_admin_only(self, client, tenants):
        """No tier reaches an ownerless private artifact, so the admin branch
        is the only way in - which is what system artifacts need."""
        runtime = get_runtime()
        _pid, publisher = _account("acme", ACME, admin=True)
        _uid, user = _account("acme", ACME)
        _aid, admin = _account("acme", ACME, admin=True)
        orphan = _artifact(client, publisher, "private")
        with runtime.store._connect() as conn:
            conn.execute(
                "UPDATE artifact SET owner_user_id = NULL WHERE id = %s", (orphan,)
            )

        refused = client.get(f"/v1/artifacts/{orphan}", headers=user)

        assert refused.status_code == 403
        assert "admin" in refused.text
        assert client.get(f"/v1/artifacts/{orphan}", headers=admin).status_code == 200
