"""Cross-replica coordination: cancel across workers, run periodic work once.

The transports (Redis pub/sub, Postgres LISTEN/NOTIFY) need live services, so
these tests wire two buses to each other through the same framing/dispatch code
the real transports feed, and separately assert the local/degraded behavior an
operator actually gets when Redis is absent.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from liminallm import app as app_module
from liminallm.api import routes
from liminallm.service.cluster import (
    MAX_PAYLOAD_BYTES,
    AdvisoryLock,
    ClusterBus,
    _lock_key,
)


class PairedBus(ClusterBus):
    """Bus whose frames are delivered straight to its declared peers."""

    def __init__(self, name: str) -> None:
        super().__init__(node_id=name)
        self.backend = "paired"
        self._started = True
        self.peers: list["PairedBus"] = []
        self.sent = 0

    async def _transmit(self, payload: str) -> bool:
        self.sent += 1
        for peer in self.peers:
            await peer._dispatch(payload)
        return True


def pair(*buses: PairedBus) -> None:
    for bus in buses:
        bus.peers = [other for other in buses if other is not bus]
        bus._loop = asyncio.get_event_loop()


# --------------------------------------------------------------------------
# ClusterBus


async def test_local_backend_has_no_peers_to_ask():
    """With no Redis and no Postgres the bus is inert, not broken."""
    bus = ClusterBus(redis_url=None, database_url=None)
    await bus.start()
    try:
        assert bus.backend == "local"
        assert await bus.publish("chat.cancel", {"request_id": "x"}) is False
        assert await bus.request("chat.cancel", {"request_id": "x"}) is None
    finally:
        await bus.stop()


async def test_backend_local_forced_by_setting():
    bus = ClusterBus(
        redis_url="redis://127.0.0.1:6379/0",
        database_url="postgresql://127.0.0.1/none",
        backend="local",
    )
    await bus.start()
    try:
        assert bus.backend == "local"
    finally:
        await bus.stop()


async def test_request_returns_owning_peer_verdict():
    asker, owner, bystander = PairedBus("a"), PairedBus("b"), PairedBus("c")
    pair(asker, owner, bystander)

    async def owner_handler(data):
        assert data["request_id"] == "req-1"
        return {"cancelled": True, "reason": "cancelled"}

    seen = []

    async def bystander_handler(data):
        seen.append(data)
        return None  # not mine: stay silent

    owner.subscribe("chat.cancel", owner_handler)
    bystander.subscribe("chat.cancel", bystander_handler)

    ack = await asker.request("chat.cancel", {"request_id": "req-1"}, timeout=2.0)
    assert ack == {"cancelled": True, "reason": "cancelled"}
    assert seen == [{"request_id": "req-1"}]  # bystander saw it, said nothing


async def test_request_times_out_when_nobody_owns_it():
    asker, peer = PairedBus("a"), PairedBus("b")
    pair(asker, peer)

    async def silent(data):
        return None

    peer.subscribe("chat.cancel", silent)
    assert await asker.request("chat.cancel", {"request_id": "gone"}, timeout=0.2) is None
    assert not asker._pending  # no leaked futures


async def test_handler_exception_does_not_ack():
    asker, peer = PairedBus("a"), PairedBus("b")
    pair(asker, peer)

    async def boom(data):
        raise RuntimeError("handler blew up")

    peer.subscribe("chat.cancel", boom)
    assert await asker.request("chat.cancel", {"request_id": "x"}, timeout=0.2) is None


async def test_own_broadcast_is_not_handled_twice():
    """The local path already ran; the echo must not re-run the handler."""
    bus = PairedBus("solo")
    pair(bus)  # no peers, but frames still round-trip through _dispatch
    calls = []

    async def handler(data):
        calls.append(data)
        return {"cancelled": True, "reason": "cancelled"}

    bus.subscribe("chat.cancel", handler)
    await bus._dispatch(
        json.dumps({"ch": "chat.cancel", "node": bus.node_id, "data": {"x": 1}})
    )
    assert calls == []


async def test_concurrent_requests_get_their_own_acks():
    asker, owner = PairedBus("a"), PairedBus("b")
    pair(asker, owner)

    async def handler(data):
        await asyncio.sleep(0.05 if data["request_id"] == "slow" else 0)
        return {"reason": data["request_id"]}

    owner.subscribe("chat.cancel", handler)
    slow, fast = await asyncio.gather(
        asker.request("chat.cancel", {"request_id": "slow"}, timeout=2.0),
        asker.request("chat.cancel", {"request_id": "fast"}, timeout=2.0),
    )
    assert slow == {"reason": "slow"}
    assert fast == {"reason": "fast"}


async def test_oversized_frame_is_refused_before_transmit():
    """NOTIFY caps payloads at 8000 bytes; refuse rather than error at the DB."""
    bus, peer = PairedBus("a"), PairedBus("b")
    pair(bus, peer)
    assert await bus.publish("chat.cancel", {"blob": "x" * MAX_PAYLOAD_BYTES}) is False
    assert bus.sent == 0


async def test_malformed_frame_is_ignored():
    bus = PairedBus("a")
    pair(bus)
    calls = []

    async def handler(data):
        calls.append(data)
        return {}

    bus.subscribe("chat.cancel", handler)
    await bus._dispatch("not json")
    await bus._dispatch("[1, 2, 3]")
    assert calls == []


async def test_unreachable_postgres_leaves_bus_local():
    """A dead coordination backend degrades to local; it never raises."""
    bus = ClusterBus(
        redis_url=None,
        database_url="postgresql://127.0.0.1:1/nonexistent?connect_timeout=1",
    )
    await bus.start()
    try:
        assert bus.backend == "local"
    finally:
        await bus.stop()


# --------------------------------------------------------------------------
# AdvisoryLock


async def test_local_lock_serializes_one_holder():
    lock = AdvisoryLock(None)
    async with lock.try_hold("job") as first:
        assert first is True
        async with lock.try_hold("job") as second:
            assert second is False
    # released again afterwards
    async with lock.try_hold("job") as third:
        assert third is True


async def test_local_lock_is_per_name():
    lock = AdvisoryLock(None)
    async with lock.try_hold("a") as a:
        async with lock.try_hold("b") as b:
            assert a is True and b is True


async def test_lock_fails_open_when_database_unreachable():
    """Maintenance running twice beats maintenance never running."""
    lock = AdvisoryLock("postgresql://127.0.0.1:1/nonexistent?connect_timeout=1")
    async with lock.try_hold("job") as acquired:
        assert acquired is True


def test_lock_key_is_stable_and_fits_bigint():
    key = _lock_key("training_worker:clustering")
    assert key == _lock_key("training_worker:clustering")
    assert key != _lock_key("training_worker:adapter_prune")
    assert -(2**63) <= key < 2**63


# --------------------------------------------------------------------------
# Cancel routing


async def test_cancel_reaches_the_replica_holding_the_stream():
    """The cancel event lives in one process; the other must still stop it."""
    worker_a, worker_b = PairedBus("a"), PairedBus("b")
    pair(worker_a, worker_b)
    worker_b.subscribe(routes.CANCEL_CHANNEL, routes.handle_remote_cancel)

    event = asyncio.Event()
    await routes._register_cancel_event("req-remote", event, "user-1")
    try:
        # worker_a doesn't know the request; ask the cluster.
        ack = await worker_a.request(
            routes.CANCEL_CHANNEL,
            {"request_id": "req-remote", "user_id": "user-1"},
            timeout=2.0,
        )
        assert ack == {"cancelled": True, "reason": "cancelled"}
        assert event.is_set()
    finally:
        await routes._unregister_cancel_event("req-remote")


async def test_remote_cancel_still_enforces_ownership():
    event = asyncio.Event()
    await routes._register_cancel_event("req-owned", event, "owner-user")
    try:
        reply = await routes.handle_remote_cancel(
            {"request_id": "req-owned", "user_id": "attacker-user"}
        )
        assert reply == {"cancelled": False, "reason": "not_owner"}
        assert not event.is_set()
    finally:
        await routes._unregister_cancel_event("req-owned")


async def test_remote_cancel_stays_silent_for_unknown_request():
    assert await routes.handle_remote_cancel(
        {"request_id": "never-existed", "user_id": "u"}
    ) is None
    assert await routes.handle_remote_cancel({}) is None


async def test_cancel_request_prefers_local_and_skips_the_bus():
    class ExplodingBus:
        async def request(self, *a, **kw):  # pragma: no cover - must not run
            raise AssertionError("local hit should not consult the bus")

    event = asyncio.Event()
    await routes._register_cancel_event("req-local", event, "user-1")
    runtime = routes.get_runtime()
    original = runtime.bus
    runtime.bus = ExplodingBus()
    try:
        cancelled, reason = await routes._cancel_request("req-local", "user-1")
        assert (cancelled, reason) == (True, "cancelled")
    finally:
        runtime.bus = original
        await routes._unregister_cancel_event("req-local")


async def test_cancel_endpoint_reports_the_remote_verdict(monkeypatch):
    """A 403 for someone else's request must survive the extra hop."""

    class OwnerElsewhere:
        async def request(self, channel, data, **kw):
            assert channel == routes.CANCEL_CHANNEL
            return {"cancelled": False, "reason": "not_owner"}

    runtime = routes.get_runtime()
    original = runtime.bus
    runtime.bus = OwnerElsewhere()
    try:
        cancelled, reason = await routes._cancel_request("req-elsewhere", "user-1")
        assert (cancelled, reason) == (False, "not_owner")
    finally:
        runtime.bus = original


async def test_bus_failure_does_not_break_cancel():
    class BrokenBus:
        async def request(self, *a, **kw):
            raise RuntimeError("bus down")

    runtime = routes.get_runtime()
    original = runtime.bus
    runtime.bus = BrokenBus()
    try:
        cancelled, reason = await routes._cancel_request("req-x", "user-1")
        assert (cancelled, reason) == (False, "request_not_found")
    finally:
        runtime.bus = original


def test_cancel_endpoint_still_answers_when_no_replica_owns_it():
    client = TestClient(app_module.app)
    email = "cancelbus@example.com"
    signup = client.post(
        "/v1/auth/signup", json={"email": email, "password": "TestPassword123!"}
    )
    assert signup.status_code == 201, signup.text
    headers = {"Authorization": f"Bearer {signup.json()['data']['access_token']}"}
    response = client.post(
        "/v1/chat/cancel", headers=headers, json={"request_id": "nobody-owns-this"}
    )
    assert response.status_code == 200
    assert response.json()["data"]["cancelled"] is False


# --------------------------------------------------------------------------
# Periodic work runs once per cluster, not once per replica


class _FakeClusterer:
    def __init__(self) -> None:
        self.global_runs = 0

    async def cluster_user_preferences(self, *a, **kw):
        return None

    async def cluster_global_preferences(self, *a, **kw):
        self.global_runs += 1


class _NoUsersStore:
    """One user in one tenant.

    Named for what it originally was; it now needs a user because "global"
    clustering is scoped per tenant (a cluster spanning tenants would train a
    skill adapter on mixed-tenant content). Tenants are derived from users, so
    with no users there is nothing to cluster and the lock has nothing to
    guard.
    """

    def list_users(self, limit=None):
        return [SimpleNamespace(id="u1", tenant_id="t1")]


def _worker(clusterer, lock):
    from liminallm.service.training_worker import TrainingWorker

    return TrainingWorker(
        store=_NoUsersStore(),
        training_service=object(),
        clusterer=clusterer,
        leader_lock=lock,
    )


async def test_only_the_lock_holder_runs_periodic_clustering():
    lock = AdvisoryLock(None)
    clusterer = _FakeClusterer()
    leader = _worker(clusterer, lock)
    follower = _worker(clusterer, lock)

    async def hold():
        # Occupy the cluster-wide lock while the follower tries to run.
        async with lock.try_hold("training_worker:clustering") as held:
            assert held is True
            await follower._maybe_run_periodic_clustering()

    await hold()
    assert clusterer.global_runs == 0  # follower stood down

    await leader._maybe_run_periodic_clustering()
    assert clusterer.global_runs == 1  # lock free again


async def test_prune_scan_is_leader_gated():
    lock = AdvisoryLock(None)
    worker = _worker(_FakeClusterer(), lock)
    calls = []
    worker._run_adapter_prune_scan = lambda: calls.append(1) or asyncio.sleep(0)

    async with lock.try_hold("training_worker:adapter_prune") as held:
        assert held is True
        await worker._maybe_recommend_adapter_pruning()
    assert calls == []

    worker._last_prune_run = 0.0
    await worker._maybe_recommend_adapter_pruning()
    assert calls == [1]


# --------------------------------------------------------------------------
# Review fixes: transports must recover, forced backends must not substitute


async def test_forced_redis_never_substitutes_postgres():
    """backend=redis is a promise: dead Redis degrades to local, not postgres."""
    bus = ClusterBus(
        redis_url="redis://127.0.0.1:1/0",
        database_url="postgresql://postgres@127.0.0.1:5432/liminallm",
        backend="redis",
    )
    called = []
    bus._start_postgres = lambda: called.append(1) or True
    await bus.start()
    try:
        assert bus.backend == "local"
        assert called == []
    finally:
        await bus.stop()


class _FlakyListenConn:
    """First connection dies mid-listen; the replacement delivers a notify."""

    class _Notify:
        payload = json.dumps({"ch": "chat.cancel", "node": "peer", "data": {"n": 1}})

    def __init__(self, fail_once_holder, stop_event):
        self._fail = fail_once_holder
        self._stop = stop_event
        self.closed = False

    def notifies(self, timeout=None):
        if self._fail["remaining"] > 0:
            self._fail["remaining"] -= 1
            raise ConnectionError("server closed the connection unexpectedly")
        if not self._fail["delivered"]:
            self._fail["delivered"] = True
            yield self._Notify()
        else:
            self._stop.set()  # test is done; let the loop exit

    def close(self):
        self.closed = True


async def test_pg_listen_loop_reconnects_after_connection_death():
    bus = ClusterBus(database_url="postgresql://unused/db")
    bus._loop = asyncio.get_running_loop()
    state = {"remaining": 1, "delivered": False}
    connects = []

    def fake_connect():
        connects.append(1)
        return _FlakyListenConn(state, bus._pg_stop)

    bus._pg_connect_listen = fake_connect
    received = asyncio.Event()

    async def handler(data):
        received.set()
        return None

    bus.subscribe("chat.cancel", handler)
    first_conn = fake_connect()
    thread = __import__("threading").Thread(
        target=bus._pg_listen_loop, args=(first_conn,), daemon=True
    )
    thread.start()
    await asyncio.wait_for(received.wait(), timeout=10.0)
    bus._pg_stop.set()
    await asyncio.to_thread(thread.join, 5.0)
    # Died once, reconnected at least once, and the notify got through.
    assert len(connects) >= 2
    assert bus._connected is True


async def test_redis_reader_survives_a_dropped_connection():
    bus = ClusterBus(redis_url="redis://unused/0")
    bus.backend = "redis"
    dispatched = asyncio.Event()

    async def handler(data):
        dispatched.set()
        return None

    bus.subscribe("chat.cancel", handler)
    frame = json.dumps({"ch": "chat.cancel", "node": "peer", "data": {}})

    class FlakyPubSub:
        def __init__(self):
            self.calls = 0

        async def listen(self):
            self.calls += 1
            if self.calls == 1:
                raise ConnectionError("connection lost")
            yield {"type": "message", "data": frame}
            await asyncio.sleep(3600)  # hold the stream open until cancelled

    bus._pubsub = FlakyPubSub()
    reader = asyncio.create_task(bus._redis_reader())
    try:
        await asyncio.wait_for(dispatched.wait(), timeout=10.0)
        assert bus._pubsub.calls == 2
        assert bus._connected is True
    finally:
        reader.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await reader


async def test_concurrent_fs_probes_do_not_starve_each_other():
    """Shared-filename health probes 503'd healthy replicas; names are unique now."""
    from liminallm import app as app_module

    results = await asyncio.gather(
        *[app_module._dependency_checks() for _ in range(8)]
    )
    assert all(serving_ok for _, serving_ok, _ in results)


def test_connected_property_tracks_transport():
    bus = ClusterBus()
    assert bus.connected is True  # local: trivially connected
    bus.backend = "postgres"
    assert bus.connected is False
    bus._connected = True
    assert bus.connected is True


# --------------------------------------------------------------------------
# Live transports. Opt in by pointing these at real services, e.g.
#   CLUSTER_TEST_DATABASE_URL=postgresql://postgres@127.0.0.1:5432/liminallm \
#   CLUSTER_TEST_REDIS_URL=redis://127.0.0.1:6379/9 pytest tests/test_cluster.py

LIVE_PG = os.environ.get("CLUSTER_TEST_DATABASE_URL")
LIVE_REDIS = os.environ.get("CLUSTER_TEST_REDIS_URL")


async def _round_trip_over(**kw) -> None:
    """Two replicas, one live transport, one real cancel event."""
    asker = ClusterBus(node_id="live-asker", **kw)
    owner = ClusterBus(node_id="live-owner", **kw)
    owner.subscribe(routes.CANCEL_CHANNEL, routes.handle_remote_cancel)
    await asker.start()
    await owner.start()
    assert asker.backend != "local" and owner.backend != "local"
    await asyncio.sleep(0.3)  # let LISTEN / SUBSCRIBE settle

    event = asyncio.Event()
    await routes._register_cancel_event("live-req", event, "user-1")
    try:
        ack = await asker.request(
            routes.CANCEL_CHANNEL,
            {"request_id": "live-req", "user_id": "user-1"},
            timeout=5.0,
        )
        assert ack == {"cancelled": True, "reason": "cancelled"}
        assert event.is_set()

        unowned = await asker.request(
            routes.CANCEL_CHANNEL,
            {"request_id": "not-here", "user_id": "user-1"},
            timeout=1.0,
        )
        assert unowned is None
    finally:
        await routes._unregister_cancel_event("live-req")
        await asker.stop()
        await owner.stop()


@pytest.mark.skipif(not LIVE_PG, reason="CLUSTER_TEST_DATABASE_URL not set")
async def test_live_postgres_listen_notify_round_trip():
    await _round_trip_over(database_url=LIVE_PG, backend="postgres")


@pytest.mark.skipif(not LIVE_REDIS, reason="CLUSTER_TEST_REDIS_URL not set")
async def test_live_redis_pubsub_round_trip():
    await _round_trip_over(redis_url=LIVE_REDIS, backend="redis")


@pytest.mark.skipif(not LIVE_PG, reason="CLUSTER_TEST_DATABASE_URL not set")
async def test_live_advisory_lock_excludes_other_replicas():
    replica_a, replica_b = AdvisoryLock(LIVE_PG), AdvisoryLock(LIVE_PG)
    async with replica_a.try_hold("training_worker:clustering") as holder:
        assert holder is True
        async with replica_b.try_hold("training_worker:clustering") as other:
            assert other is False
    async with replica_b.try_hold("training_worker:clustering") as after_release:
        assert after_release is True


def test_health_reports_bus_backend_without_gating_readiness():
    """Operators need to see which transport a replica actually picked up."""
    with TestClient(app_module.app) as client:
        health = client.get("/healthz")
        ready = client.get("/readyz")
    assert health.status_code == 200
    bus_check = health.json()["checks"]["cluster_bus"]
    assert bus_check["backend"] in {"local", "redis", "postgres"}
    assert bus_check["status"] == "healthy"
    # A local/degraded bus must never drain the replica.
    assert ready.status_code == 200


def test_runtime_exposes_coordination_primitives():
    runtime = routes.get_runtime()
    assert isinstance(runtime.bus, ClusterBus)
    assert isinstance(runtime.leader_lock, AdvisoryLock)
    # Memory store means no Postgres to coordinate through.
    assert runtime.training_worker.leader_lock is runtime.leader_lock
