"""The periodic clustering pass, which never ran.

`cluster_everyone` is the pass the training worker schedules. It called
`self.clusterer.cluster_user_preferences(...)`, but `SemanticClusterer` has no
`clusterer` attribute - so every user and every tenant raised AttributeError
into the per-iteration `except Exception`, was logged as a warning, and the
whole pass did nothing. Nothing failed loudly; the feature was simply absent.

The scheduler's own test uses a fake clusterer, so it could never see this.
These tests drive the real object.
"""

from __future__ import annotations

import uuid

import pytest

from liminallm.service.clustering import SemanticClusterer
from liminallm.service.runtime import get_runtime


class _Recorder(SemanticClusterer):
    """The real class, with only the two leaf calls stubbed out."""

    def __init__(self, store):
        super().__init__(store, llm=None, training=None)
        self.per_user = []
        self.per_tenant = []

    async def cluster_user_preferences(self, user_id, **kwargs):
        self.per_user.append((user_id, kwargs))
        return []

    async def cluster_global_preferences(self, **kwargs):
        self.per_tenant.append(kwargs)
        return []


@pytest.fixture
def store():
    return get_runtime().store


def _user(store, tenant_id):
    return store.create_user(
        email=f"cl_{uuid.uuid4().hex[:8]}@example.com", tenant_id=tenant_id
    )


@pytest.mark.asyncio
async def test_the_periodic_pass_reaches_every_user(store):
    users = [_user(store, "public") for _ in range(3)]
    rec = _Recorder(store)

    await rec.cluster_everyone(user_limit=500, max_events=50)

    reached = {uid for uid, _ in rec.per_user}
    assert {u.id for u in users} <= reached


@pytest.mark.asyncio
async def test_the_periodic_pass_forwards_its_budget(store):
    _user(store, "public")
    rec = _Recorder(store)

    await rec.cluster_everyone(user_limit=500, max_events=17)

    assert rec.per_user, "no user was clustered"
    _, kwargs = rec.per_user[0]
    assert kwargs["max_events"] == 17
    assert kwargs["streaming"] is True
    assert kwargs["approximate"] is True


@pytest.mark.asyncio
async def test_the_global_pass_runs_once_per_tenant_not_once_overall(store):
    """Clustering across tenants would put another tenant's content into a
    skill adapter, so the global pass is per-tenant (CLAUDE.md isolation)."""
    tenant = f"t-{uuid.uuid4().hex[:6]}"
    _user(store, tenant)
    _user(store, tenant)
    rec = _Recorder(store)

    await rec.cluster_everyone(user_limit=500, max_events=50)

    seen = [kw["tenant_id"] for kw in rec.per_tenant]
    assert seen.count(tenant) == 1
    assert None not in seen


@pytest.mark.asyncio
async def test_one_users_failure_does_not_stop_the_rest(store):
    """The point of the per-iteration except - but it must be catching real
    clustering failures, not an AttributeError on every single user."""
    good = [_user(store, "public") for _ in range(3)]
    rec = _Recorder(store)
    original = rec.cluster_user_preferences
    first = {"seen": False}

    async def explode_once(user_id, **kwargs):
        if not first["seen"]:
            first["seen"] = True
            raise RuntimeError("this user will not cluster")
        return await original(user_id, **kwargs)

    rec.cluster_user_preferences = explode_once

    await rec.cluster_everyone(user_limit=500, max_events=50)

    reached = {uid for uid, _ in rec.per_user}
    assert len(reached & {u.id for u in good}) >= 2


@pytest.mark.asyncio
async def test_an_install_with_no_users_is_not_an_error(store):
    rec = _Recorder(store)
    rec.store = type("Empty", (), {"list_users": staticmethod(lambda **kw: [])})()

    await rec.cluster_everyone(user_limit=500, max_events=50)

    assert rec.per_user == []
    assert rec.per_tenant == []
