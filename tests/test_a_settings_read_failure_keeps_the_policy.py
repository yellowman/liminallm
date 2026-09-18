"""A worker that cannot read the settings keeps the ones it has.

`refresh_settings` rebuilds `runtime.settings` from the shipped defaults plus
what an admin stored. `_system_settings_overrides` turned any read failure into
`{}`, and `{}` does not mean "the admin chose nothing" to the layer above - it
means "apply every default". So one transient database error replaced the whole
managed policy with the shipped one.

For most settings that is a surprise. For `citation_offers_enabled` it is a
fail-open: the default is `True`, so a worker whose read failed hands citation
authority back to the executions an operator had withdrawn it from, and
`refresh_settings` pushes that straight into the live registry.

The second half is what made it stick. `maybe_reload_model_services` recorded
the version it had just tried to apply even when the apply had failed, and every
later poll compares against that record and returns early. One failed read at
the moment a worker picks up a change left that worker on shipped defaults until
the next settings write or a restart - not until the next poll.

Scope: the policy a worker already holds. A worker that has never read anything
has no policy to keep, so a failure during boot still yields the defaults.
"""

from __future__ import annotations

import pytest

from liminallm.service.runtime import Runtime, get_runtime


def _boom(*args, **kwargs):
    raise RuntimeError("settings table unavailable")


@pytest.fixture
def worker(store):
    """A second runtime over the same store, as a peer worker is."""
    store.set_system_settings({"citation_offers_enabled": False})
    peer = Runtime()
    assert peer.settings.citation_offers_enabled is False, (
        "the control failed: the peer never read the stored policy, so a "
        "later reading of `True` would prove nothing"
    )
    assert peer.workflow.invocations.citation_offers is False
    return peer


def test_the_stored_policy_reaches_a_worker(store):
    """The control, stated on its own.

    Every assertion below is that some value is still `False`. That is only
    evidence if `False` is a value this probe can produce at all.
    """
    runtime = get_runtime()
    assert runtime.settings.citation_offers_enabled is True
    store.set_system_settings({"citation_offers_enabled": False})
    runtime.refresh_settings()
    assert runtime.settings.citation_offers_enabled is False


def test_a_failed_read_does_not_restore_the_default(worker, monkeypatch):
    monkeypatch.setattr(worker.store, "get_system_settings_overrides", _boom)

    worker.refresh_settings()

    assert worker.settings.citation_offers_enabled is False, (
        "a failed settings read re-enabled citation offers, which the "
        "operator had turned off"
    )


def test_a_failed_read_does_not_re_grant_authority_to_live_executions(
    worker, monkeypatch
):
    """The registry is the part that hands the authority out."""
    import uuid

    live = worker.workflow.invocations.open(uuid.uuid4().hex, tool="t")
    assert live.citation_offers_intact is False

    monkeypatch.setattr(worker.store, "get_system_settings_overrides", _boom)
    worker.refresh_settings()

    assert worker.workflow.invocations.citation_offers is False
    assert worker.workflow.invocations.open(
        uuid.uuid4().hex, tool="t"
    ).citation_offers_intact is False, (
        "an execution opened after the failed read was granted citation "
        "authority the operator had withdrawn"
    )


def test_a_failed_read_is_retried_on_the_next_poll(worker, store, monkeypatch):
    """The stickiness, which is what turns one error into an outage.

    The poll records the version it applied so an unrelated write is not
    rechecked. Recording a version it did *not* manage to apply is what left
    the worker on defaults indefinitely.
    """
    store.set_system_settings({
        "citation_offers_enabled": False, "default_page_size": 120
    })

    monkeypatch.setattr(worker.store, "get_system_settings_overrides", _boom)
    worker.maybe_reload_model_services()
    assert worker.settings.citation_offers_enabled is False

    monkeypatch.undo()
    worker.maybe_reload_model_services()

    assert worker.settings.default_page_size == 120, (
        "the worker recorded a settings version it never applied, so the "
        "admin's change is not picked up until the next write or a restart"
    )
    assert worker.settings.citation_offers_enabled is False


def test_a_recovered_read_replaces_the_kept_policy(worker, store, monkeypatch):
    """Keeping the last policy must not become ignoring the current one."""
    monkeypatch.setattr(worker.store, "get_system_settings_overrides", _boom)
    worker.refresh_settings()
    monkeypatch.undo()

    store.set_system_settings({"citation_offers_enabled": True})
    worker.refresh_settings()

    assert worker.settings.citation_offers_enabled is True


def test_a_model_rebuild_with_a_failed_read_is_retried_too(
    worker, store, monkeypatch
):
    """The same record, on the other path that writes it.

    `reload_model_services` rebuilds through `refresh_settings`, so a read
    that fails there builds the new stack from the policy this worker already
    had - and must not then claim the version was applied.
    """
    store.set_system_settings({
        "citation_offers_enabled": False, "default_page_size": 140
    })

    monkeypatch.setattr(worker.store, "get_system_settings_overrides", _boom)
    worker.reload_model_services()
    assert worker.settings.citation_offers_enabled is False

    monkeypatch.undo()
    worker.maybe_reload_model_services()

    assert worker.settings.default_page_size == 140
