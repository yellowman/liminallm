"""Cross-replica coordination that degrades instead of requiring Redis.

Two primitives, both best-effort by design:

``ClusterBus``
    A tiny request/reply bus so one replica can ask the others to do something
    only the owner of a live in-process object can do (today: cancel a
    streaming request whose WebSocket lives on another worker). Redis pub/sub
    is the fast path; Postgres ``LISTEN``/``NOTIFY`` is the fallback so a Redis
    outage degrades latency rather than correctness; a single-process
    deployment gets a no-op backend because there are no peers to ask.

``AdvisoryLock``
    Cluster-wide mutual exclusion for periodic background work that should run
    once per interval, not once per replica. Postgres session advisory locks
    when a database is configured, a process-local lock otherwise.

Nothing here is required for serving traffic: every operation fails soft and
logs, because the alternative is making Redis (or the bus) load-bearing.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import threading
import uuid
from typing import Any, Awaitable, Callable, Dict, Optional

from liminallm.logging import get_logger

logger = get_logger(__name__)

# One physical channel carries every logical channel so the Postgres backend
# only has to hold a single LISTEN.
BUS_CHANNEL = "liminallm_bus"
# NOTIFY payloads are capped at 8000 bytes by Postgres; stay well under it.
MAX_PAYLOAD_BYTES = 6000
# Bound every fresh connection: an unreachable coordination host must degrade
# a request, not hang it for the TCP stack's idea of forever.
CONNECT_TIMEOUT_SECONDS = 5
_RECONNECT_MAX_BACKOFF = 30.0
_ACK = "__ack__"

Handler = Callable[[Dict[str, Any]], Awaitable[Optional[Dict[str, Any]]]]


def _lock_key(name: str) -> int:
    """Stable 64-bit signed key for pg_advisory_lock(bigint)."""
    digest = hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=True)


class ClusterBus:
    """Best-effort request/reply across replicas.

    Handlers registered with :meth:`subscribe` run on every replica that
    receives the message. A handler returning ``None`` stays silent (it does
    not own the subject); returning a dict sends an ack back to the asker,
    which is what makes :meth:`request` able to report a truthful outcome
    instead of an optimistic one.
    """

    def __init__(
        self,
        *,
        redis_url: Optional[str] = None,
        database_url: Optional[str] = None,
        backend: str = "auto",
        node_id: Optional[str] = None,
    ) -> None:
        self.node_id = node_id or f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self._redis_url = redis_url or None
        self._database_url = database_url or None
        self._requested_backend = (backend or "auto").lower()
        self.backend = "local"
        self._handlers: Dict[str, Handler] = {}
        self._pending: Dict[str, asyncio.Future] = {}
        self._started = False
        self._connected = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        # Redis backend state
        self._redis = None
        self._pubsub = None
        self._reader: Optional[asyncio.Task] = None
        # Postgres backend state
        self._pg_thread: Optional[threading.Thread] = None
        self._pg_stop = threading.Event()
        self._pg_notify_conn = None
        self._pg_notify_lock = threading.Lock()

    # -- lifecycle ---------------------------------------------------------

    def subscribe(self, channel: str, handler: Handler) -> None:
        """Register the handler for a logical channel (last one wins)."""
        self._handlers[channel] = handler

    async def start(self) -> None:
        if self._started:
            return
        self._loop = asyncio.get_running_loop()
        wanted = self._requested_backend
        if wanted in {"local", "off", "none", "disabled"}:
            self.backend = "local"
        else:
            if wanted in {"auto", "redis"} and self._redis_url:
                if await self._start_redis():
                    self.backend = "redis"
            # A forced backend is a promise, not a preference: "redis" that
            # can't connect degrades to local rather than silently becoming
            # postgres. Only "auto" falls through.
            if self.backend == "local" and wanted in {"auto", "postgres"}:
                if self._database_url and await asyncio.to_thread(self._start_postgres):
                    self.backend = "postgres"
        self._started = True
        logger.info("cluster_bus_started", backend=self.backend, node_id=self.node_id)

    @property
    def connected(self) -> bool:
        """True when the transport is live (trivially true with no peers)."""
        return self.backend == "local" or self._connected

    async def stop(self) -> None:
        self._started = False
        if self._reader:
            self._reader.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._reader
            self._reader = None
        for closer in (self._pubsub, self._redis):
            if closer is not None:
                with contextlib.suppress(Exception):
                    await closer.aclose()
        self._pubsub = None
        self._redis = None
        if self._pg_thread:
            self._pg_stop.set()
            await asyncio.to_thread(self._pg_thread.join, 5.0)
            self._pg_thread = None
        with self._pg_notify_lock:
            if self._pg_notify_conn is not None:
                with contextlib.suppress(Exception):
                    self._pg_notify_conn.close()
                self._pg_notify_conn = None
        for future in list(self._pending.values()):
            if not future.done():
                future.cancel()
        self._pending.clear()
        self.backend = "local"
        self._connected = False

    # -- public API --------------------------------------------------------

    async def publish(self, channel: str, data: Dict[str, Any]) -> bool:
        """Fire-and-forget broadcast. Returns False when nothing was sent."""
        return await self._send({"ch": channel, "node": self.node_id, "data": data})

    async def request(
        self,
        channel: str,
        data: Dict[str, Any],
        *,
        timeout: float = 1.5,
    ) -> Optional[Dict[str, Any]]:
        """Ask peers and return the first ack, or None if nobody answered.

        A None result is indistinguishable from "no replica owns this", which
        is exactly how callers should treat it.
        """
        if self.backend == "local":
            return None
        corr = uuid.uuid4().hex
        loop = self._loop or asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self._pending[corr] = future
        try:
            sent = await self._send(
                {"ch": channel, "node": self.node_id, "corr": corr, "data": data}
            )
            if not sent:
                return None
            return await asyncio.wait_for(future, timeout=timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            return None
        finally:
            self._pending.pop(corr, None)

    # -- dispatch ----------------------------------------------------------

    async def _dispatch(self, raw: str) -> None:
        try:
            message = json.loads(raw)
        except Exception:
            logger.warning("cluster_bus_bad_payload")
            return
        if not isinstance(message, dict):
            return
        channel = message.get("ch")
        corr = message.get("corr")

        if channel == _ACK:
            future = self._pending.get(corr or "")
            if future is not None and not future.done():
                future.set_result(message.get("data") or {})
            return

        # Our own broadcasts come back to us; the local path already ran.
        if message.get("node") == self.node_id:
            return
        handler = self._handlers.get(channel or "")
        if handler is None:
            return
        try:
            reply = await handler(message.get("data") or {})
        except Exception as exc:
            logger.warning(
                "cluster_bus_handler_failed", channel=channel, error=str(exc)
            )
            return
        if reply is not None and corr:
            await self._send({"ch": _ACK, "node": self.node_id, "corr": corr, "data": reply})

    async def _send(self, message: Dict[str, Any]) -> bool:
        if self.backend == "local":
            return False
        payload = json.dumps(message, separators=(",", ":"))
        if len(payload.encode("utf-8")) > MAX_PAYLOAD_BYTES:
            logger.warning("cluster_bus_payload_too_large", channel=message.get("ch"))
            return False
        try:
            return await self._transmit(payload)
        except Exception as exc:
            logger.warning(
                "cluster_bus_publish_failed", backend=self.backend, error=str(exc)
            )
        return False

    async def _transmit(self, payload: str) -> bool:
        """Hand a serialized frame to the active transport.

        Split out from :meth:`_send` so the framing, size limit, ack
        correlation and handler contract can be exercised without a live Redis
        or Postgres; subclasses override this to deliver frames elsewhere.
        """
        if self.backend == "redis" and self._redis is not None:
            await self._redis.publish(BUS_CHANNEL, payload)
            return True
        if self.backend == "postgres":
            return await asyncio.to_thread(self._pg_notify, payload)
        return False

    # -- redis backend -----------------------------------------------------

    async def _start_redis(self) -> bool:
        try:
            import redis.asyncio as aioredis

            redis = aioredis.from_url(
                self._redis_url,
                decode_responses=True,
                socket_connect_timeout=CONNECT_TIMEOUT_SECONDS,
            )
            pubsub = redis.pubsub(ignore_subscribe_messages=True)
            await pubsub.subscribe(BUS_CHANNEL)
            self._redis = redis
            self._pubsub = pubsub
            self._connected = True
            self._reader = asyncio.create_task(self._redis_reader())
            return True
        except Exception as exc:
            logger.warning("cluster_bus_redis_unavailable", error=str(exc))
            with contextlib.suppress(Exception):
                if self._redis is not None:
                    await self._redis.aclose()
            self._redis = None
            self._pubsub = None
            return False

    async def _redis_reader(self) -> None:
        """Consume the pubsub stream, resubscribing after transport drops.

        redis-py re-registers this pubsub's channels when the connection comes
        back, so recovery is just calling listen() again; without this loop a
        single dropped connection silently ended cross-replica coordination
        for the life of the process.
        """
        assert self._pubsub is not None
        backoff = 1.0
        while True:
            try:
                self._connected = True
                async for message in self._pubsub.listen():
                    backoff = 1.0
                    if message and message.get("type") == "message":
                        await self._dispatch(message.get("data") or "")
                return  # stream closed cleanly (stop())
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._connected = False
                logger.warning("cluster_bus_redis_reader_error", error=str(exc))
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, _RECONNECT_MAX_BACKOFF)

    # -- postgres backend --------------------------------------------------

    def _pg_connect_listen(self):
        """Fresh LISTENing connection (separate method so tests can fake it)."""
        import psycopg

        conn = psycopg.connect(
            self._database_url,
            autocommit=True,
            connect_timeout=CONNECT_TIMEOUT_SECONDS,
        )
        conn.execute(f"LISTEN {BUS_CHANNEL}")
        return conn

    def _start_postgres(self) -> bool:
        try:
            conn = self._pg_connect_listen()
        except Exception as exc:
            logger.warning("cluster_bus_postgres_unavailable", error=str(exc))
            return False
        self._connected = True
        self._pg_stop.clear()
        self._pg_thread = threading.Thread(
            target=self._pg_listen_loop,
            args=(conn,),
            name="cluster-bus-listen",
            daemon=True,
        )
        self._pg_thread.start()
        return True

    def _pg_listen_loop(self, conn) -> None:
        """Consume NOTIFYs, replacing the connection when it dies.

        A Postgres restart kills the LISTEN session; retrying notifies() on
        the dead connection would just raise every second forever, so on error
        the connection is rebuilt (with backoff) and LISTEN re-issued.
        """
        loop = self._loop
        backoff = 1.0
        try:
            while not self._pg_stop.is_set():
                if conn is None:
                    try:
                        conn = self._pg_connect_listen()
                        self._connected = True
                        backoff = 1.0
                        logger.info("cluster_bus_listen_reconnected")
                    except Exception as exc:
                        logger.warning(
                            "cluster_bus_listen_reconnect_failed", error=str(exc)
                        )
                        self._pg_stop.wait(backoff)
                        backoff = min(backoff * 2, _RECONNECT_MAX_BACKOFF)
                        continue
                try:
                    for notify in conn.notifies(timeout=1.0):
                        if loop is not None and not loop.is_closed():
                            asyncio.run_coroutine_threadsafe(
                                self._dispatch(notify.payload), loop
                            )
                        if self._pg_stop.is_set():
                            break
                    backoff = 1.0
                except Exception as exc:
                    if self._pg_stop.is_set():
                        break
                    self._connected = False
                    logger.warning("cluster_bus_listen_error", error=str(exc))
                    with contextlib.suppress(Exception):
                        conn.close()
                    conn = None
                    self._pg_stop.wait(1.0)
        finally:
            if conn is not None:
                with contextlib.suppress(Exception):
                    conn.close()

    def _pg_notify(self, payload: str) -> bool:
        import psycopg

        with self._pg_notify_lock:
            for attempt in (0, 1):
                try:
                    if self._pg_notify_conn is None or self._pg_notify_conn.closed:
                        self._pg_notify_conn = psycopg.connect(
                            self._database_url,
                            autocommit=True,
                            connect_timeout=CONNECT_TIMEOUT_SECONDS,
                        )
                    self._pg_notify_conn.execute(
                        "SELECT pg_notify(%s, %s)", (BUS_CHANNEL, payload)
                    )
                    return True
                except Exception:
                    with contextlib.suppress(Exception):
                        if self._pg_notify_conn is not None:
                            self._pg_notify_conn.close()
                    self._pg_notify_conn = None
                    if attempt:
                        raise
        return False


class AdvisoryLock:
    """Run-once-per-cluster guard for periodic work.

    Postgres session advisory locks when a database is configured; otherwise a
    process-local lock, which is the correct answer for a single-process
    deployment. Failure to reach Postgres yields the lock rather than blocking
    the work forever — background maintenance running twice is cheaper than
    never running.
    """

    def __init__(self, database_url: Optional[str] = None) -> None:
        self._database_url = database_url or None
        self._local: Dict[str, asyncio.Lock] = {}

    @contextlib.asynccontextmanager
    async def try_hold(self, name: str):
        """Yield True if this replica may run ``name`` right now."""
        if not self._database_url:
            lock = self._local.setdefault(name, asyncio.Lock())
            if lock.locked():
                yield False
                return
            async with lock:
                yield True
            return

        conn = None
        acquired = False
        try:
            conn, acquired = await asyncio.to_thread(self._pg_try_lock, name)
        except Exception as exc:
            logger.warning("advisory_lock_unavailable", name=name, error=str(exc))
            yield True  # fail open: better to run twice than never
            return
        try:
            yield acquired
        finally:
            if conn is not None:
                await asyncio.to_thread(self._pg_release, conn)

    def _pg_try_lock(self, name: str):
        import psycopg

        conn = psycopg.connect(
            self._database_url,
            autocommit=True,
            connect_timeout=CONNECT_TIMEOUT_SECONDS,
        )
        try:
            row = conn.execute(
                "SELECT pg_try_advisory_lock(%s)", (_lock_key(name),)
            ).fetchone()
        except Exception:
            with contextlib.suppress(Exception):
                conn.close()
            raise
        acquired = bool(row and row[0])
        if not acquired:
            with contextlib.suppress(Exception):
                conn.close()
            return None, False
        return conn, True

    @staticmethod
    def _pg_release(conn) -> None:
        # Closing the session drops every advisory lock it holds.
        with contextlib.suppress(Exception):
            conn.close()
