"""The two training-worker settings take effect on restart, and say so.

`training_worker_enabled` is read once, in the application lifespan, to decide
whether to start the worker at all. `training_worker_poll_interval` is
captured in `TrainingWorker.__init__`, and the worker is built in
`Runtime.__init__` rather than in `_build_model_services`, so neither
`refresh_settings` nor a model-stack rebuild reaches it.

Starting and stopping a background worker from a settings write is real
lifecycle machinery - a running loop to cancel mid-job, a leader lock to
release - and this records the honest shape instead of building it: the
console tells the operator a restart is needed. These tests pin both halves
together, so making either setting live fails here and the description gets
corrected with it rather than drifting into a false promise.
"""

from __future__ import annotations

import uuid

import pytest

from liminallm.config import managed_settings_schema
from liminallm.service.runtime import get_runtime

RESTART_PHRASE = "takes effect on restart"


@pytest.fixture
def admin_headers():
    runtime = get_runtime()
    user = runtime.store.create_user(
        email=f"t_{uuid.uuid4().hex[:8]}@t.local", role="admin"
    )
    session = runtime.store.create_session(user.id, tenant_id=user.tenant_id)
    _u, _s, tokens = runtime.auth.issue_tokens_for_session(session.id)
    return {"Authorization": f"Bearer {tokens['access_token']}"}


@pytest.fixture
def restore_worker_settings():
    runtime = get_runtime()
    before = {
        "training_worker_enabled": runtime.settings.training_worker_enabled,
        "training_worker_poll_interval": (
            runtime.settings.training_worker_poll_interval
        ),
    }
    yield
    runtime.store.set_system_settings(before)
    runtime.refresh_settings()


def _described(name: str) -> str:
    for entry in managed_settings_schema():
        if entry["name"] == name:
            return entry["description"]
    raise AssertionError(f"{name} is not in the admin schema")


def _put(client, headers, body):
    resp = client.put("/v1/admin/settings", headers=headers, json=body)
    assert resp.status_code == 200, resp.text
    return resp


class TestTheSettingDoesNotReachTheRunningWorker:
    def test_a_new_poll_interval_does_not_reach_it(
        self, client, admin_headers, restore_worker_settings
    ):
        runtime = get_runtime()
        before = runtime.training_worker.poll_interval

        _put(client, admin_headers,
             {"training_worker_poll_interval": before + 7})

        assert get_runtime().settings.training_worker_poll_interval == before + 7
        assert get_runtime().training_worker.poll_interval == before

    def test_neither_value_starts_or_stops_the_worker(
        self, client, admin_headers, restore_worker_settings, monkeypatch
    ):
        """No settings write reaches the worker's lifecycle, in either
        direction.

        The worker is idle in a test process, so "a running worker keeps
        running" cannot be observed here. What can be observed is the
        stronger statement behind it: the write calls neither `start` nor
        `stop`, so there is no live transition to observe in the first place.
        The spies are on the real worker's own methods.
        """
        worker = get_runtime().training_worker
        calls: list[str] = []

        async def _spy_start():
            calls.append("start")

        async def _spy_stop():
            calls.append("stop")

        monkeypatch.setattr(worker, "start", _spy_start)
        monkeypatch.setattr(worker, "stop", _spy_stop)

        _put(client, admin_headers, {"training_worker_enabled": False})
        _put(client, admin_headers, {"training_worker_enabled": True})

        assert get_runtime().settings.training_worker_enabled is True
        assert calls == []


class TestTheConsoleSaysSo:
    @pytest.mark.parametrize(
        "name", ["training_worker_enabled", "training_worker_poll_interval"]
    )
    def test_the_description_asks_for_a_restart(self, name):
        assert RESTART_PHRASE in _described(name).lower(), (
            f"{name} does not reach a running worker, and its description "
            "does not say so"
        )

    @pytest.mark.parametrize(
        "name", ["training_worker_enabled", "training_worker_poll_interval"]
    )
    def test_a_rebuild_would_not_deliver_them_either(self, name):
        """`reloads_model` must stay false: the worker is not in the stack a
        rebuild replaces, so claiming a reload delivers these would be a
        second false promise in the same console row."""
        entry = next(
            e for e in managed_settings_schema() if e["name"] == name
        )
        assert entry["reloads_model"] is False
