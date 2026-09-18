"""A rollback a peer never observed still reaches that peer's executions.

Workers learn about settings changes by polling `instance_config.updated_at`
and then reading the *current* value. That is enough for a transition a peer
happens to poll across, and not enough for one that begins and ends between two
polls: if citation offers go off and back on inside one interval, the peer reads
`True`, calls `configure_citation_offers(True)`, and the executions that were
live throughout keep the authority an operator withdrew.

`citation_offers_enabled` promises the opposite in its own description: "a turn
already running loses citation authority immediately and permanently". The
version token cannot carry that, because it says only that *something* changed.

So the withdrawal is counted. `citation_rollback_generation` increments when the
canonical value transitions `True -> False`, atomically with the settings write.
A worker keeps the last generation it observed; if it advanced, every live
execution loses authority before the current value is applied prospectively -
whatever that current value now is.

The first test is the one that matters. A peer that merely observes
`True -> False` proves only that the old mechanism still works.
"""

from __future__ import annotations

import uuid

import pytest

from liminallm.service.runtime import Runtime


@pytest.fixture
def peer(store):
    """A second worker over the same store, already caught up."""
    worker = Runtime()
    assert worker.settings.citation_offers_enabled is True
    assert worker.workflow.invocations.citation_offers is True
    return worker


def _live(worker):
    return worker.workflow.invocations.open(uuid.uuid4().hex, tool="t")


def test_a_rollback_and_restore_between_polls_still_disables(peer, store):
    """The skipped transition: off and on again inside one interval."""
    caught = _live(peer)
    assert caught.citation_offers_intact is True

    store.set_system_settings({"citation_offers_enabled": False})
    store.set_system_settings({"citation_offers_enabled": True})

    peer.maybe_reload_model_services()

    assert caught.citation_offers_intact is False, (
        "an execution that was live across a rollback kept citation "
        "authority because its worker never polled while the setting was off"
    )
    assert peer.settings.citation_offers_enabled is True
    assert _live(peer).citation_offers_intact is True, (
        "the restore did not take effect prospectively"
    )


def test_an_observed_rollback_still_disables(peer, store):
    """The control: the case the version token already handled."""
    caught = _live(peer)

    store.set_system_settings({"citation_offers_enabled": False})
    peer.maybe_reload_model_services()

    assert caught.citation_offers_intact is False
    assert peer.settings.citation_offers_enabled is False
    assert _live(peer).citation_offers_intact is False


def test_an_unrelated_settings_write_does_not_withdraw_authority(peer, store):
    """The counter is a withdrawal count, not a change count.

    Disabling on any unobserved version change would also work for the test
    above, and would strip authority from every in-flight turn whenever an
    admin saved anything at all.
    """
    caught = _live(peer)

    store.set_system_settings({"default_page_size": 75})
    peer.maybe_reload_model_services()

    assert peer.settings.default_page_size == 75, "the write never reached the peer"
    assert caught.citation_offers_intact is True, (
        "an unrelated settings change withdrew citation authority"
    )


def test_repeated_writes_of_false_count_once(peer, store):
    """Idempotent writes are not new withdrawals."""
    store.set_system_settings({"citation_offers_enabled": False})
    peer.maybe_reload_model_services()
    store.set_system_settings({"citation_offers_enabled": True})
    peer.maybe_reload_model_services()

    after_restore = _live(peer)
    assert after_restore.citation_offers_intact is True

    store.set_system_settings({"citation_offers_enabled": True})
    peer.maybe_reload_model_services()

    assert after_restore.citation_offers_intact is True, (
        "a write that withdrew nothing still counted as a withdrawal"
    )


def test_a_new_worker_adopts_the_current_generation(store):
    """It has no executions predating the generation, so nothing to withdraw."""
    store.set_system_settings({"citation_offers_enabled": False})
    store.set_system_settings({"citation_offers_enabled": True})

    fresh = Runtime()
    born = fresh.workflow.invocations.open(uuid.uuid4().hex, tool="t")

    assert fresh.settings.citation_offers_enabled is True
    assert born.citation_offers_intact is True, (
        "a worker that started after the rollback withdrew authority from an "
        "execution that never existed during it"
    )


def test_the_generation_survives_a_worker_restart(peer, store):
    """Durable, not in-process: a restarted worker must not re-withdraw."""
    store.set_system_settings({"citation_offers_enabled": False})
    store.set_system_settings({"citation_offers_enabled": True})
    peer.maybe_reload_model_services()

    restarted = Runtime()
    born = restarted.workflow.invocations.open(uuid.uuid4().hex, tool="t")
    restarted.maybe_reload_model_services()

    assert born.citation_offers_intact is True
