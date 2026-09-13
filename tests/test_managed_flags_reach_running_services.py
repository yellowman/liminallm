"""A managed flag an admin changes reaches the service that captured it.

SPEC §18.6: operational settings take effect without restart.
`refresh_settings` hands every service the new settings object, which is
enough for a service that reads per use - and its comment said that was
everything but citation offers. Two services capture a flag as a plain
attribute at construction and never look again: `AuthService.mfa_enabled`
and `TrainingService.distillation_enabled`. Measured through the real admin
route: `enable_mfa: false` left MFA challenges issuing. Both are now told, the
way citation offers are.
"""

from __future__ import annotations

import uuid

import pytest

from liminallm.service.runtime import get_runtime


def _admin(client):
    runtime = get_runtime()
    tenant = runtime.settings.default_tenant_id
    user = runtime.store.create_user(
        email=f"adm_{uuid.uuid4().hex[:8]}@t.local", tenant_id=tenant
    )
    runtime.store.update_user_role(user.id, role="admin")
    session = runtime.store.create_session(user.id, tenant_id=tenant)
    _u, _s, tokens = runtime.auth.issue_tokens_for_session(session.id)
    return user.id, {"Authorization": f"Bearer {tokens['access_token']}"}


@pytest.fixture
def restore(client):
    """Whatever the test flips through the admin route, flip back."""
    runtime = get_runtime()
    before = {k: getattr(runtime.settings, k)
              for k in ("enable_mfa", "training_distillation_enabled")}
    _uid, admin = _admin(client)
    yield admin
    client.put("/v1/admin/settings", headers=admin, json=before)


class TestATellingIsNotAHint:
    @pytest.mark.asyncio
    async def test_disabling_mfa_reaches_the_running_auth_service(self, client, restore):
        runtime = get_runtime()
        uid, admin = _admin(client)
        assert runtime.auth.mfa_enabled is True

        put = client.put("/v1/admin/settings", headers=admin, json={"enable_mfa": False})
        assert put.status_code == 200, put.text

        assert runtime.auth.mfa_enabled is False
        assert (await runtime.auth.issue_mfa_challenge(uid)) == {"status": "disabled"}

    @pytest.mark.asyncio
    async def test_enabling_it_again_reaches_it_too(self, client, restore):
        runtime = get_runtime()
        uid, admin = _admin(client)
        client.put("/v1/admin/settings", headers=admin, json={"enable_mfa": False})
        assert runtime.auth.mfa_enabled is False

        client.put("/v1/admin/settings", headers=admin, json={"enable_mfa": True})

        assert runtime.auth.mfa_enabled is True
        assert (await runtime.auth.issue_mfa_challenge(uid)).get("status") != "disabled"

    def test_distillation_follows_the_console(self, client, restore):
        runtime = get_runtime()
        _uid, admin = _admin(client)
        before = runtime.training.distillation_enabled

        client.put("/v1/admin/settings", headers=admin,
                   json={"training_distillation_enabled": not before})

        assert runtime.training.distillation_enabled is (not before)
