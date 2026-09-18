"""The settings boolean and the rollback generation come from one moment.

A worker decides two things from that pair: whether a citation withdrawal
happened that it never observed, and what the policy is now. They have to
describe the same instant, or the two decisions disagree.

A transaction is not enough to get that. PostgreSQL defaults to READ
COMMITTED, where every statement takes its own snapshot, so two SELECTs inside
one transaction still straddle a write committing between them. Measured
before the fix: with a disable committing in that gap the pair came back as the
boolean from *before* it and the generation from *after*.

That is the harmful direction. `refresh_settings` sees the generation advance
and correctly withdraws authority from every live execution - then applies the
stale `True` prospectively, so executions opened afterwards are born with the
authority the operator has just removed, until the next poll.

One statement takes one snapshot, so the pair cannot straddle anything.
"""

from __future__ import annotations

import pytest


def _connect_that_writes_after_the_first_statement(store, between):
    """Replace `_connect` so `between()` runs once, after one statement.

    Before the fix this lands between the two SELECTs. After it, there is only
    one SELECT and the write lands after the read - which is the point: the
    interleaving the old shape allowed no longer exists.
    """
    real = store._connect
    fired: list = []

    class Proxy:
        def __init__(self, cm):
            self._cm = cm
            self._c = None

        def __enter__(self):
            self._c = self._cm.__enter__()
            return self

        def __exit__(self, *a):
            return self._cm.__exit__(*a)

        def transaction(self, *a, **k):
            return self._c.transaction(*a, **k)

        def execute(self, *a, **k):
            cur = self._c.execute(*a, **k)
            if not fired:
                fired.append(True)
                between()
            return cur

        def __getattr__(self, name):
            return getattr(self._c, name)

    return real, (lambda: Proxy(real())), fired


@pytest.fixture
def enabled(store):
    store.set_system_settings({"citation_offers_enabled": True})
    return store.get_system_settings_state()


def test_a_disable_committing_mid_read_cannot_split_the_pair(store, enabled):
    _base_overrides, base_generation = enabled

    def _disable():
        store.set_system_settings({"citation_offers_enabled": False})

    real, proxied, fired = _connect_that_writes_after_the_first_statement(
        store, _disable
    )
    store._connect = proxied
    try:
        overrides, generation = store.get_system_settings_state()
    finally:
        store._connect = real

    assert fired, "the interleaved write never ran, so this proves nothing"
    settled_overrides, settled_generation = store.get_system_settings_state()
    assert settled_overrides.get("citation_offers_enabled") is False
    assert settled_generation == base_generation + 1, (
        "the control failed: the interleaved disable did not land"
    )

    if generation > base_generation:
        assert overrides.get("citation_offers_enabled") is False, (
            "the pair straddled the write: the generation says a withdrawal "
            "happened and the boolean says offers are still on, so the worker "
            "withdraws from live executions and then re-grants to new ones"
        )
    else:
        assert overrides.get("citation_offers_enabled") is True, (
            "the pair straddled the write the other way"
        )


def test_the_pair_still_reports_a_settled_withdrawal(store, enabled):
    """The control for the test above: with no race, the pair moves."""
    _base_overrides, base_generation = enabled
    store.set_system_settings({"citation_offers_enabled": False})

    overrides, generation = store.get_system_settings_state()

    assert overrides.get("citation_offers_enabled") is False
    assert generation == base_generation + 1
