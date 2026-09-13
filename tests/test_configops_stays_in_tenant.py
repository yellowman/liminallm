"""A ConfigOps patch is authorized by its target, not by being an admin.

Measured on the code before this: an admin in one tenant listed, approved and
applied a patch against another tenant's *private* artifact, and the owner's
schema changed. A second route was worse - proposal recorded whatever
`artifact_id` an admin supplied, unchecked, not even for existence - so the
same admin could reach into another tenant by id in one call.

`config_patch` has no tenant column and needs none. The scope is derived from
`patch.artifact_id -> artifact -> owner`, re-derived on every call, at
proposal, listing, decision and application alike.
"""

from __future__ import annotations

import uuid

import pytest

from liminallm.service.runtime import get_runtime

ACME, GLOBEX = "acme.test", "globex.test"
HIJACK = "hijacked-by-the-other-tenant"

ADD_HIJACK = {"ops": [{"op": "add", "path": "/hijacked", "value": HIJACK}]}


def _admin(tenant_id, host):
    runtime = get_runtime()
    user = runtime.store.create_user(
        email=f"c_{uuid.uuid4().hex[:8]}@t.local", tenant_id=tenant_id, role="admin"
    )
    session = runtime.store.create_session(user.id, tenant_id=tenant_id)
    _u, _s, tokens = runtime.auth.issue_tokens_for_session(session.id)
    return user.id, {"Authorization": f"Bearer {tokens['access_token']}",
                     "host": host}


def _artifact(client, headers, visibility="private"):
    resp = client.post(
        "/v1/artifacts",
        headers={**headers, "Idempotency-Key": uuid.uuid4().hex},
        json={"type": "workflow", "name": f"w-{uuid.uuid4().hex[:6]}",
              "description": "a workflow under patch", "visibility": visibility,
              "schema": {"kind": "workflow.chat", "nodes": []}},
    )
    assert resp.status_code in (200, 201), resp.text
    return resp.json()["data"]["id"]


def _schema(client, headers, artifact_id) -> dict:
    resp = client.get(f"/v1/artifacts/{artifact_id}", headers=headers)
    assert resp.status_code == 200, resp.text
    return resp.json()["data"]["schema"]


def _propose(client, headers, artifact_id, patch=None):
    return client.post(
        "/v1/config/propose_patch",
        headers=headers,
        json={"artifact_id": artifact_id, "patch": patch or ADD_HIJACK,
              "justification": "a proposal under test"},
    )


def _patch_ids(client, headers) -> list[int]:
    resp = client.get("/v1/config/patches", headers=headers)
    assert resp.status_code == 200, resp.text
    return [row["id"] for row in resp.json()["data"]["items"]]


@pytest.fixture
def tenants(monkeypatch):
    monkeypatch.setattr(
        get_runtime().settings, "tenant_domains", {ACME: "acme", GLOBEX: "globex"}
    )


class TestTheTargetDecides:
    def test_another_tenants_admin_cannot_apply_a_patch(self, client, tenants):
        """The reproduced defect, at the surface it was reproduced on.

        Globex proposes against its own artifact, which is allowed. Acme then
        approves and applies it. Before the fix both calls returned 200 and
        Globex's schema gained the key.
        """
        _a, acme = _admin("acme", ACME)
        _g, globex = _admin("globex", GLOBEX)
        target = _artifact(client, globex, "private")

        proposed = _propose(client, globex, target)
        assert proposed.status_code == 200, proposed.text
        patch_id = proposed.json()["data"]["id"]

        decided = client.post(f"/v1/config/patches/{patch_id}/decide",
                              headers=acme, json={"decision": "approve"})
        applied = client.post(f"/v1/config/patches/{patch_id}/apply",
                              headers=acme, json={})

        assert decided.status_code == 404, decided.text
        assert applied.status_code == 404, applied.text
        assert "hijacked" not in _schema(client, globex, target)

    def test_another_tenants_admin_cannot_propose_by_id(self, client, tenants):
        """Proposal is the entry point, so it is where the id must stop.

        Recording an unauthorized proposal and refusing it later would leave
        the target's id, and the proposer's interest in it, in another
        tenant's audit trail.
        """
        _a, acme = _admin("acme", ACME)
        _g, globex = _admin("globex", GLOBEX)
        target = _artifact(client, globex, "private")

        refused = _propose(client, acme, target)

        assert refused.status_code == 404, refused.text
        assert _patch_ids(client, globex) == []

    def test_another_tenants_admin_cannot_auto_generate_by_id(
        self, client, tenants
    ):
        """`auto_patch` takes the same `artifact_id` and writes the same row.

        The proposal comes from the model rather than the caller, which
        changes who wrote the patch and nothing about who may target it.
        """
        _a, acme = _admin("acme", ACME)
        _g, globex = _admin("globex", GLOBEX)
        target = _artifact(client, globex, "private")

        refused = client.post("/v1/config/auto_patch", headers=acme,
                              json={"artifact_id": target, "goal": "tune it"})

        assert refused.status_code == 404, refused.text
        assert _patch_ids(client, globex) == []

    def test_the_listing_shows_only_administrable_patches(self, client, tenants):
        """Listing is scoped by the same rule, and scoped is not empty.

        An over-narrow filter would hide the owner's own patch too, which is
        the failure this half has to rule out.
        """
        _a, acme = _admin("acme", ACME)
        _g, globex = _admin("globex", GLOBEX)
        theirs = _propose(client, globex, _artifact(client, globex, "private"))
        mine = _propose(client, acme, _artifact(client, acme, "private"))
        theirs_id = theirs.json()["data"]["id"]
        mine_id = mine.json()["data"]["id"]

        seen_by_acme = _patch_ids(client, acme)

        assert mine_id in seen_by_acme
        assert theirs_id not in seen_by_acme
        assert theirs_id in _patch_ids(client, globex)

    def test_a_shared_artifact_is_its_owners_tenants(self, client, tenants):
        """Publishing to a tenant widens who reads it, not who patches it.

        Shared is the tier most likely to be read as "administrable by
        anyone", because it is the one whose whole purpose is being visible
        to other people.
        """
        _a, acme = _admin("acme", ACME)
        _g, globex = _admin("globex", GLOBEX)
        target = _artifact(client, globex, "shared")

        assert _propose(client, acme, target).status_code == 404
        assert _propose(client, globex, target).status_code == 200


class TestTheRuleIsNotOverNarrow:
    """Refusing everything would also pass the tests above."""

    def test_an_admin_administers_a_colleagues_private_artifact(
        self, client, tenants
    ):
        """Private means private from other *users*, not from the tenant.

        ConfigOps is the admin surface for an installation's configuration,
        and an admin who cannot patch anything they do not personally own
        cannot do the job.
        """
        _admin_id, admin = _admin("acme", ACME)
        runtime = get_runtime()
        colleague = runtime.store.create_user(
            email=f"c_{uuid.uuid4().hex[:8]}@t.local", tenant_id="acme"
        )
        target = runtime.store.create_artifact(
            type_="workflow",
            name=f"w-{uuid.uuid4().hex[:6]}",
            schema={"kind": "workflow.chat", "nodes": []},
            owner_user_id=colleague.id,
            visibility="private",
        )

        proposed = _propose(client, admin, target.id)
        patch_id = proposed.json()["data"]["id"]
        client.post(f"/v1/config/patches/{patch_id}/decide", headers=admin,
                    json={"decision": "approve"})
        applied = client.post(f"/v1/config/patches/{patch_id}/apply",
                              headers=admin, json={})

        assert proposed.status_code == 200, proposed.text
        assert applied.status_code == 200, applied.text
        assert runtime.store.get_artifact(target.id).schema["hijacked"] == HIJACK

    def test_a_global_artifact_is_administrable_by_any_admin(
        self, client, tenants
    ):
        """Global is installation-wide, and so is its configuration."""
        _a, acme = _admin("acme", ACME)
        _g, globex = _admin("globex", GLOBEX)
        target = _artifact(client, globex, "global")

        proposed = _propose(client, acme, target)
        patch_id = proposed.json()["data"]["id"]
        client.post(f"/v1/config/patches/{patch_id}/decide", headers=acme,
                    json={"decision": "approve"})
        applied = client.post(f"/v1/config/patches/{patch_id}/apply",
                              headers=acme, json={})

        assert proposed.status_code == 200, proposed.text
        assert applied.status_code == 200, applied.text
        assert _schema(client, acme, target)["hijacked"] == HIJACK

    def test_an_ownerless_artifact_is_administrable_by_any_admin(
        self, client, tenants
    ):
        """A system artifact has no owner, so it has no tenant to belong to.

        Deriving the scope from the owner has to answer this case
        deliberately: nobody owns these, and somebody has to be able to
        configure them.
        """
        _a, acme = _admin("acme", ACME)
        runtime = get_runtime()
        target = runtime.store.create_artifact(
            type_="workflow",
            name=f"sys-{uuid.uuid4().hex[:6]}",
            schema={"kind": "workflow.chat", "nodes": []},
            owner_user_id=None,
            visibility="private",
        )

        proposed = _propose(client, acme, target.id)
        patch_id = proposed.json()["data"]["id"]
        client.post(f"/v1/config/patches/{patch_id}/decide", headers=acme,
                    json={"decision": "approve"})
        applied = client.post(f"/v1/config/patches/{patch_id}/apply",
                              headers=acme, json={})

        assert proposed.status_code == 200, proposed.text
        assert applied.status_code == 200, applied.text
        assert runtime.store.get_artifact(target.id).schema["hijacked"] == HIJACK


class TestAuthorityIsNotCarriedOver:
    def test_an_unrecognised_visibility_is_administrable_by_nobody(
        self, client, tenants
    ):
        """The tiers are a closed set here too, and the default is refusal.

        The API writes only the three, so this row is made in the store - a
        migration or a future tier added in one place and not the other can
        produce one.
        """
        _a, acme = _admin("acme", ACME)
        runtime = get_runtime()
        target = _artifact(client, acme, "private")
        with runtime.store._connect() as conn:
            conn.execute(
                "UPDATE artifact SET visibility = %s WHERE id = %s",
                ("everyone-ish", target),
            )

        refused = _propose(client, acme, target)

        assert refused.status_code == 404, refused.text

    def test_a_patch_listed_earlier_is_re_checked_at_apply(
        self, client, tenants
    ):
        """Authority is a property of the target now, not of the listing then.

        The patch here is proposed and approved while the target is global,
        so an Acme admin legitimately holds an approved patch. The target is
        then transferred to a Globex owner as private, and apply - the call
        that writes - must ask again rather than honour the approval.
        """
        _a, acme = _admin("acme", ACME)
        globex_id, _globex = _admin("globex", GLOBEX)
        runtime = get_runtime()
        target = _artifact(client, acme, "global")
        proposed = _propose(client, acme, target)
        patch_id = proposed.json()["data"]["id"]
        decided = client.post(f"/v1/config/patches/{patch_id}/decide",
                              headers=acme, json={"decision": "approve"})
        assert decided.status_code == 200, decided.text

        with runtime.store._connect() as conn:
            conn.execute(
                "UPDATE artifact SET visibility = %s, owner_user_id = %s "
                "WHERE id = %s",
                ("private", globex_id, target),
            )
        applied = client.post(f"/v1/config/patches/{patch_id}/apply",
                              headers=acme, json={})

        assert applied.status_code == 404, applied.text
        assert "hijacked" not in runtime.store.get_artifact(target).schema
