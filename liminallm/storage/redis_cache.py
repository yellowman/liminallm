from __future__ import annotations

import asyncio
import hashlib
import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple, Union
from uuid import uuid4

import redis.asyncio as aioredis

# Atomically acquire an idempotency slot: claim it if absent, or reclaim it if
# the existing record is a prior "failed" attempt (so a retry can proceed
# without a separate, racy overwrite). Otherwise return the existing record.
# Returns {acquired(1/0), existing_json_or_empty}.
_ACQUIRE_OR_RECLAIM_IDEMPOTENCY = """
local existing = redis.call('GET', KEYS[1])
if not existing then
    redis.call('SET', KEYS[1], ARGV[1], 'EX', ARGV[2])
    return {1, ''}
end
local ok, decoded = pcall(cjson.decode, existing)
if ok and type(decoded) == 'table' and decoded['status'] == 'failed' then
    redis.call('SET', KEYS[1], ARGV[1], 'EX', ARGV[2])
    return {1, ''}
end
return {0, existing}
"""

# Read a key and remove it in one step, for a redis-py client with no `getdel`
# method. `EVAL` is atomic, so the two calls inside it cannot be interleaved
# with another client's. Not a fallback for an old *server*: see
# `consume_identity_token`.
_GETDEL = """
local value = redis.call('GET', KEYS[1])
if value then
    redis.call('DEL', KEYS[1])
end
return value
"""


class RedisCache:
    """Thin Redis wrapper for sessions and rate limits."""

    # Issue 48.3: Default operation timeout for Redis commands
    DEFAULT_OPERATION_TIMEOUT = 5.0  # 5 seconds

    # The breaker failure history is a sorted set keyed by this suffix. It
    # carried a plain counter under `:failures` before the rolling window,
    # and a sorted set is a different Redis type at the same key: a rolling
    # deploy where old replicas `INCR` a string while new ones `ZADD` a set
    # makes every cross-version command fail `WRONGTYPE`. The version suffix
    # gives the new type its own key; the legacy counter is left to expire.
    # The `:open` key is a plain string on both versions and stays shared.
    _FAILURES_SUFFIX = "failures:v2"

    # Lua token bucket script (Issue 77.2/77.10/77.12): atomic refill + consume
    _TOKEN_BUCKET_SCRIPT = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local refill_rate = tonumber(ARGV[2])
local capacity = tonumber(ARGV[3])
local cost = tonumber(ARGV[4])

local data = redis.call('HMGET', key, 'tokens', 'ts')
local tokens = tonumber(data[1])
local last = tonumber(data[2])

if tokens == nil or last == nil then
  tokens = capacity
  last = now
end

local delta = math.max(0, now - last)
tokens = math.min(capacity, tokens + delta * refill_rate)

if tokens < cost then
  redis.call('HMSET', key, 'tokens', tokens, 'ts', now)
  local reset_after = math.ceil((cost - tokens) / refill_rate)
  redis.call('EXPIRE', key, math.max(reset_after, 1))
  return {0, tokens, reset_after}
end

tokens = tokens - cost
redis.call('HMSET', key, 'tokens', tokens, 'ts', now)
local ttl = math.ceil(capacity / refill_rate)
redis.call('EXPIRE', key, math.max(ttl, 1))
return {1, tokens, 0}
"""

    def __init__(self, redis_url: str, *, socket_timeout: float = 5.0):
        self.redis_url = redis_url
        self._socket_timeout = socket_timeout
        # A redis-py asyncio client pools connections bound to the loop that
        # opened them; reusing one across loops raises "attached to a different
        # loop". Uvicorn workers, startup probes and the test client all run on
        # loops of their own, so hold one client per loop rather than one for
        # the process. (A second, synchronous cache class used to exist to dodge
        # this; it drifted out of sync and broke features instead.)
        self._clients: Dict[int, Any] = {}
        self._scripts: Dict[int, Any] = {}

    def _loop_key(self) -> int:
        try:
            return id(asyncio.get_running_loop())
        except RuntimeError:
            return 0

    @property
    def client(self):
        """The Redis client bound to the running event loop."""
        key = self._loop_key()
        client = self._clients.get(key)
        if client is None:
            # Issue 48.3: Configure connection with explicit timeouts
            client = aioredis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_timeout=self._socket_timeout,
                socket_connect_timeout=self._socket_timeout,
            )
            self._clients[key] = client
            # Register token bucket script for atomic rate limiting (Issue 77.10)
            self._scripts[key] = client.register_script(self._TOKEN_BUCKET_SCRIPT)
        return client

    @property
    def _token_bucket(self):
        self.client  # ensure the script is registered for this loop
        return self._scripts[self._loop_key()]

    @staticmethod
    def _normalize_utc(dt: datetime) -> datetime:
        """Return a UTC-aware datetime for safe arithmetic.

        Redis TTL calculations must avoid mixing naive and aware timestamps. New
        session records are created with timezone-aware UTC expirations, but
        older callers may still provide naive datetimes. Normalize everything to
        UTC before computing relative durations. (Issue 2.6)
        """

        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    @staticmethod
    def _ttl_seconds(expires_at: datetime, *, now: Optional[datetime] = None) -> int:
        """Compute a safe TTL from an absolute expiry timestamp.

        Sessions now use timezone-aware UTC timestamps; older records may be naive.
        Normalize to UTC and clamp to at least 1 second to avoid Redis rejecting
        negative or zero TTL values. (Issue 2.6)
        """

        expires_at = RedisCache._normalize_utc(expires_at)
        now = now or datetime.now(timezone.utc)
        return max(1, int((expires_at - now).total_seconds()))

    async def _redis_now(self) -> datetime:
        """Return Redis server time to align TTLs with Redis clock (Issue 76.8)."""

        try:
            seconds, microseconds = await self.client.time()
            return datetime.fromtimestamp(
                seconds + microseconds / 1_000_000, tz=timezone.utc
            )
        except Exception:
            # Fall back to application clock if Redis TIME is unavailable.
            return datetime.now(timezone.utc)

    @staticmethod
    def _prepare_oauth_state(
        provider: str,
        expires_at: datetime,
        tenant_id: Optional[str],
        *,
        now: Optional[datetime] = None,
    ) -> tuple[dict, int]:
        """Normalize OAuth expirations and compute a safe TTL.

        OAuth callers now pass timezone-aware timestamps; older callers may pass
        naive datetimes. Normalize everything to UTC before calculating TTL to
        avoid naive/aware subtraction errors. Returns the serialized payload and
        TTL seconds so both async and sync caches share identical logic.
        """

        normalized_expiry = RedisCache._normalize_utc(expires_at)
        ttl = RedisCache._ttl_seconds(normalized_expiry, now=now)
        payload = {
            "provider": provider,
            "expires_at": normalized_expiry.isoformat(),
            "tenant_id": tenant_id,
        }
        return payload, ttl

    def verify_connection(self) -> None:
        """Assert Redis connectivity before enabling dependent features."""
        from redis import Redis

        # Use a short-lived synchronous client to avoid binding the async client to a
        # temporary event loop during startup checks.
        sync_client = Redis.from_url(self.redis_url, decode_responses=True)
        try:
            sync_client.ping()
        finally:
            sync_client.close()

    async def cache_session(
        self, session_id: str, user_id: str, expires_at: datetime
    ) -> None:
        now = await self._redis_now()
        ttl = self._ttl_seconds(expires_at, now=now)
        async with self.client.pipeline() as pipe:
            pipe.set(f"auth:session:{session_id}", user_id, ex=ttl)
            # Track session in user's session set for bulk revocation (Issue 22.3)
            pipe.sadd(f"auth:user_sessions:{user_id}", session_id)
            pipe.expire(f"auth:user_sessions:{user_id}", ttl)
            await pipe.execute()

    @staticmethod
    def _normalize_rate_key(key: str, tenant_id: Optional[str]) -> str:
        """Generate collision-resistant rate keys (Issue 77.1).

        Components are hashed to avoid delimiter injection while still
        providing stable keys per logical rate limit subject.
        """

        digest = hashlib.sha256(key.encode()).hexdigest()
        tenant_prefix = f"{tenant_id}:" if tenant_id else ""
        return f"rate:{tenant_prefix}{digest}"

    async def get_session_user(self, session_id: str) -> Tuple[bool, Optional[str]]:
        key = f"auth:session:{session_id}"
        value = await self.client.get(key)
        if value is not None:
            return True, value
        exists = bool(await self.client.exists(key))
        return exists, None

    async def revoke_session(self, session_id: str) -> None:
        await self.client.delete(f"auth:session:{session_id}")

    async def revoke_user_sessions(
        self, user_id: str, except_session_id: Optional[str] = None
    ) -> int:
        """Revoke all cached sessions for a user.

        Args:
            user_id: User whose sessions to revoke
            except_session_id: Optional session ID to keep active

        Returns:
            Number of sessions revoked from cache
        """
        user_sessions_key = f"auth:user_sessions:{user_id}"
        session_ids = await self.client.smembers(user_sessions_key)
        if not session_ids:
            return 0

        revoked = 0
        async with self.client.pipeline() as pipe:
            for session_id in session_ids:
                if except_session_id and session_id == except_session_id:
                    continue
                pipe.delete(f"auth:session:{session_id}")
                pipe.srem(user_sessions_key, session_id)
                revoked += 1
            await pipe.execute()
        return revoked

    async def check_rate_limit(
        self,
        key: str,
        limit: int,
        window_seconds: int,
        *,
        return_remaining: bool = False,
        tenant_id: Optional[str] = None,
        cost: int = 1,
    ) -> Union[bool, Tuple[bool, int, int]]:
        if limit <= 0:
            return (True, limit, 0) if return_remaining else True

        if window_seconds <= 0:
            window_seconds = 60

        """Check rate limit using Redis-backed token bucket.

        Uses a Lua script for atomic refill and consumption to avoid race
        conditions (Issues 77.2, 77.10, 77.12) and hashes the key to prevent
        delimiter collisions (Issue 77.1).
        """

        safe_key = self._normalize_rate_key(key, tenant_id)
        refill_rate = float(limit) / float(window_seconds)
        # Execute Lua script atomically
        allowed, tokens, reset_after = await self._token_bucket(
            keys=[safe_key],
            args=[time.time(), refill_rate, limit, max(1, cost)],
        )

        allowed_bool = bool(int(allowed))
        remaining = max(0, int(tokens))
        reset_seconds = int(reset_after) if reset_after else 0
        if return_remaining:
            return (allowed_bool, remaining, reset_seconds)
        return allowed_bool

    async def mark_refresh_revoked(self, jti: str, ttl_seconds: int) -> None:
        await self.client.set(f"auth:refresh:revoked:{jti}", "1", ex=ttl_seconds)

    async def is_refresh_revoked(self, jti: str) -> bool:
        return bool(await self.client.exists(f"auth:refresh:revoked:{jti}"))

    # SPEC §12.1: "logout: add JWT to short-lived denylist if JWTs used"
    async def denylist_access_token(self, jti: str, ttl_seconds: int) -> None:
        """Add access token JTI to denylist with TTL matching token expiry.

        Per SPEC §4, token blacklists are stored in Redis for hot ephemeral state.
        """
        if ttl_seconds > 0:
            await self.client.set(f"auth:access:denylist:{jti}", "1", ex=ttl_seconds)

    async def is_access_token_denylisted(self, jti: str) -> bool:
        """Check if access token JTI is in denylist."""
        return bool(await self.client.exists(f"auth:access:denylist:{jti}"))

    async def get_router_cache(
        self, user_id: str, ctx_hash: str, *, tenant_id: Optional[str] = None
    ) -> Optional[dict]:
        # Issue 44.5: Include tenant_id in cache key for tenant isolation
        tenant_prefix = f"{tenant_id}:" if tenant_id else ""
        cached = await self.client.get(f"router:last:{tenant_prefix}{user_id}:{ctx_hash}")
        if not cached:
            return None
        try:
            return json.loads(cached)
        except (json.JSONDecodeError, TypeError):
            # Corrupted cache entry - treat as cache miss
            return None

    async def set_router_cache(
        self, user_id: str, ctx_hash: str, payload: dict, ttl_seconds: int = 300,
        *, tenant_id: Optional[str] = None
    ) -> None:
        # Issue 44.5: Include tenant_id in cache key for tenant isolation
        tenant_prefix = f"{tenant_id}:" if tenant_id else ""
        await self.client.set(
            f"router:last:{tenant_prefix}{user_id}:{ctx_hash}", json.dumps(payload), ex=ttl_seconds
        )

    async def get_workflow_state(
        self, state_key: str, *, tenant_id: Optional[str] = None
    ) -> Optional[dict]:
        # Issue 44.4: Include tenant_id in cache key for tenant isolation
        tenant_prefix = f"{tenant_id}:" if tenant_id else ""
        cached = await self.client.get(f"workflow:state:{tenant_prefix}{state_key}")
        if not cached:
            return None
        try:
            return json.loads(cached)
        except (json.JSONDecodeError, TypeError):
            # Corrupted cache entry - treat as cache miss
            return None

    async def set_workflow_state(
        self, state_key: str, state: dict, ttl_seconds: int = 1800,
        *, tenant_id: Optional[str] = None
    ) -> None:
        # Issue 44.4: Include tenant_id in cache key for tenant isolation
        tenant_prefix = f"{tenant_id}:" if tenant_id else ""
        await self.client.set(
            f"workflow:state:{tenant_prefix}{state_key}", json.dumps(state), ex=ttl_seconds
        )

    async def get_conversation_summary(self, conversation_id: str) -> Optional[dict]:
        cached = await self.client.get(f"chat:summary:{conversation_id}")
        if not cached:
            return None
        try:
            return json.loads(cached)
        except (json.JSONDecodeError, TypeError):
            # Corrupted cache entry - treat as cache miss
            return None

    async def set_conversation_summary(
        self, conversation_id: str, summary: Dict[str, Any], ttl_seconds: int = 3600
    ) -> None:
        await self.client.set(
            f"chat:summary:{conversation_id}", json.dumps(summary), ex=ttl_seconds
        )

    async def delete_conversation_summary(self, conversation_id: str) -> None:
        """Retire a deleted conversation's cached messages.

        The TTL is an optimization, not a lifetime. Without this, deleting a
        chat left its recent messages readable here for up to an hour after
        every trace of it had gone from Postgres.
        """
        await self.client.delete(f"chat:summary:{conversation_id}")

    async def set_oauth_state(
        self, state: str, provider: str, expires_at: datetime, tenant_id: Optional[str]
    ) -> None:
        now = await self._redis_now()
        payload, ttl = self._prepare_oauth_state(
            provider, expires_at, tenant_id, now=now
        )
        await self.client.set(f"auth:oauth:{state}", json.dumps(payload), ex=ttl)

    async def consume_identity_token(self, prefix: str, token: str) -> Optional[str]:
        """Hand out a one-time token's subject, and only once.

        The whole point is that the read and the removal are one step. Reading
        first and deleting after the work is done leaves the token readable for
        the length of that work, so two requests holding it both get a subject
        and both proceed — and for a token that arrives by email, that window
        is reachable by anyone who has read the message, and by an ordinary
        double-click.

        `GETDEL`, with an `EVAL` for a redis-py old enough not to have the
        method — `AttributeError` is a missing client method, not a server
        that refuses the command. The server side needs Redis 6.2 or newer to
        answer `GETDEL` at all, and `docker-compose.test.yml` pins Redis 7, so
        there is nothing here that reaches an older one. Supporting a server
        that predates `GETDEL` would mean catching the unknown-command
        `ResponseError` specifically, and blanket-catching `ResponseError`
        instead would turn an ACL denial into a silent `EVAL` attempt.

        One helper for every token of this shape: OAuth state, password reset,
        email verification. The version that mattered was written three times,
        and only one of the three was written this way.

        Returns the stored subject, or None if the token was not there — which
        includes the case where somebody else has just taken it.
        """
        key = f"{prefix}:{token}"
        try:
            value = await self.client.getdel(key)
        except AttributeError:
            # A redis-py without the method, not a server without the command.
            value = await self.client.eval(_GETDEL, 1, key)
        if isinstance(value, bytes):
            value = value.decode()
        return value

    async def pop_oauth_state(
        self, state: str
    ) -> Optional[tuple[str, datetime, Optional[str]]]:
        """Atomically get and delete OAuth state to prevent replay attacks.

        This prevents race conditions where two concurrent requests could both
        consume the same OAuth state. See `consume_identity_token`, which is
        where that guarantee lives for every token of this shape.

        Args:
            state: The OAuth state token to consume

        Returns:
            Tuple of (provider, expires_at, tenant_id) or None if not found
        """
        cached = await self.consume_identity_token("auth:oauth", state)

        if cached is None:
            return None

        try:
            data = json.loads(cached)
        except (json.JSONDecodeError, TypeError):
            # Corrupted data - already deleted, return None
            return None

        expires_raw = data.get("expires_at")
        # Issue 39.2: Add error handling for datetime parsing
        expires_at = datetime.now(timezone.utc)
        if isinstance(expires_raw, str):
            try:
                expires_at = datetime.fromisoformat(expires_raw)
                if expires_at.tzinfo is None:
                    expires_at = expires_at.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                pass  # Use default current UTC time
        return data.get("provider"), expires_at, data.get("tenant_id")

    async def set_idempotency_record(
        self,
        route: str,
        user_id: str,
        key: str,
        record: dict,
        ttl_seconds: int = 60 * 60 * 24,
        *,
        tenant_id: Optional[str] = None,
    ) -> None:
        # Issue 22.2: Include tenant_id in cache key for multi-tenant isolation
        tenant_prefix = f"{tenant_id}:" if tenant_id else ""
        await self.client.set(
            f"idemp:{tenant_prefix}{route}:{user_id}:{key}", json.dumps(record), ex=ttl_seconds
        )

    async def acquire_idempotency_slot(
        self,
        route: str,
        user_id: str,
        key: str,
        record: dict,
        ttl_seconds: int = 60 * 60 * 24,
        *,
        tenant_id: Optional[str] = None,
    ) -> tuple[bool, Optional[dict]]:
        """Atomically acquire an idempotency slot using SETNX pattern (Issue 19.4).

        Args:
            route: Route/operation name
            user_id: User ID
            key: Idempotency key
            record: Record to set if slot acquired (typically status=in_progress)
            ttl_seconds: TTL for the record
            tenant_id: Optional tenant ID for multi-tenant isolation (Issue 22.2)

        Returns:
            Tuple of (acquired: bool, existing_record: Optional[dict])
            - If acquired=True, the slot was successfully claimed
            - If acquired=False, existing_record contains the current record
        """
        # Issue 22.2: Include tenant_id in cache key for multi-tenant isolation
        tenant_prefix = f"{tenant_id}:" if tenant_id else ""
        cache_key = f"idemp:{tenant_prefix}{route}:{user_id}:{key}"
        # Atomic acquire-or-reclaim-failed via Lua so concurrent retries after a
        # failure can't both claim the slot (a plain SET would bypass the gate).
        result = await self.client.eval(
            _ACQUIRE_OR_RECLAIM_IDEMPOTENCY,
            1,
            cache_key,
            json.dumps(record),
            str(ttl_seconds),
        )
        if result and result[0]:
            return (True, None)
        existing = result[1] if result and len(result) > 1 else None
        if existing:
            try:
                return (False, json.loads(existing))
            except (json.JSONDecodeError, TypeError):
                pass
        return (False, None)

    async def _delete_scanned(self, pattern: str, keeps) -> int:
        """Remove every key matching `pattern` that `keeps` accepts.

        `SCAN`, never `KEYS`: this walks the whole keyspace in bounded slices
        instead of blocking the server for the length of it. Account deletion
        is rare, so paying a scan for the key families that carry no index is
        the right trade — the alternative is maintaining a per-user index for
        each of them and getting the erasure wrong whenever one expires.

        `keeps` re-checks each key, because a glob cannot express "this exact
        field equals the user id" and a pattern alone would trust the position
        of a colon in a route name somebody chooses.
        """
        removed = 0
        async for key in self.client.scan_iter(match=pattern, count=500):
            if not keeps(key):
                continue
            removed += int(await self.client.delete(key) or 0)
        return removed

    async def _delete_tokens_naming(self, prefix: str, user_id: str) -> int:
        """Remove short-lived identity tokens whose subject is this account.

        A password reset token and an email verification token each name one
        account and outlive nothing else about it. They are bounded by their
        own TTL rather than by anything the erasure controls, so leaving them
        keeps a usable reference to an account that no longer exists.
        """
        removed = 0
        async for key in self.client.scan_iter(match=f"{prefix}:*", count=500):
            if await self.client.get(key) != user_id:
                continue
            removed += int(await self.client.delete(key) or 0)
        return removed

    async def purge_user_state(self, erasure) -> Dict[str, int]:
        """Remove every Redis key this kernel can identify as one account's.

        Called after the deleting transaction commits, and after it on
        purpose: Postgres is canonical, so a cache that cannot be reached must
        not be able to prevent an erasure. What it must not do is give up
        early. Each family is its own attempt, because one unreachable key
        pattern is not a reason to leave the rest of an erased account's
        content readable — the first version of this ran every category inside
        one `try`, so a failure revoking sessions meant no conversation
        summary was even attempted.

        Deliberately *not* purged: `rate:*` is keyed by a salted digest, so it
        cannot be addressed and carries no content; `auth:access:denylist:*`
        and `auth:refresh:revoked:*` are revocations, and removing them would
        bring the erased account's outstanding tokens back to life.

        Returns how many keys each family gave up, for the log.
        """
        user_id = erasure.user_id
        sessions = list(erasure.session_ids)
        conversations = list(erasure.conversation_ids)
        exact = f":{user_id}:"

        families = {
            "sessions": [f"auth:session:{s}" for s in sessions]
            + [f"auth:user_sessions:{user_id}"],
            "session_activity": [f"session:activity:{s}" for s in sessions],
            "session_rotation": [f"session:rotation:{s}" for s in sessions],
            "conversation_summaries": [f"chat:summary:{c}" for c in conversations],
            "mfa": [f"mfa:attempts:{user_id}", f"mfa:lockout:{user_id}"],
        }
        purged: Dict[str, int] = {}
        for name, keys in families.items():
            if not keys:
                purged[name] = 0
                continue
            try:
                purged[name] = int(await self.client.delete(*keys) or 0)
            except Exception:
                purged[name] = -1

        scans = (
            # The idempotency record holds a completed API response, which for
            # a chat turn is the assistant's message. It is the most
            # content-bearing thing in this cache and it lives for a day.
            ("idempotency", f"idemp:*{exact}*", lambda k: exact in k),
            ("router_cache", f"router:last:*{user_id}:*", lambda k: user_id in k),
            (
                "concurrency",
                f"concurrency:*:{user_id}",
                lambda k: k.endswith(f":{user_id}"),
            ),
        )
        for name, pattern, keeps in scans:
            try:
                purged[name] = await self._delete_scanned(pattern, keeps)
            except Exception:
                purged[name] = -1

        for name, prefix in (("reset_tokens", "reset"), ("verify_tokens", "verify")):
            try:
                purged[name] = await self._delete_tokens_naming(prefix, user_id)
            except Exception:
                purged[name] = -1
        return purged

    async def close(self) -> None:
        """Close every per-loop client. Call on shutdown or runtime reset."""
        clients, self._clients, self._scripts = self._clients, {}, {}
        for client in clients.values():
            try:
                await client.close()
                await client.connection_pool.disconnect()
            except Exception:  # noqa: BLE001 - a client from a dead loop
                pass

    async def delete_workflow_state(
        self, state_key: str, *, tenant_id: Optional[str] = None
    ) -> None:
        """Delete workflow state from cache during rollback or cleanup."""
        # Issue 44.4: Include tenant_id in cache key for tenant isolation
        tenant_prefix = f"{tenant_id}:" if tenant_id else ""
        await self.client.delete(f"workflow:state:{tenant_prefix}{state_key}")

    # =========================================================================
    # Concurrency Caps (SPEC §18)
    # =========================================================================

    async def acquire_concurrency_slot(
        self, slot_type: str, user_id: str, max_slots: int, ttl_seconds: int = 3600,
        *, tenant_id: Optional[str] = None
    ) -> tuple[bool, int]:
        """Atomically acquire a concurrency slot for a user.

        Args:
            slot_type: Type of slot (e.g., "workflow", "inference")
            user_id: User ID
            max_slots: Maximum concurrent slots allowed
            ttl_seconds: TTL for slot keys (safety cleanup)
            tenant_id: Optional tenant ID for isolation (Issue 44.3)

        Returns:
            Tuple of (acquired: bool, current_count: int)
        """
        # Issue 44.3: Include tenant_id in concurrency key for tenant isolation
        tenant_prefix = f"{tenant_id}:" if tenant_id else ""
        key = f"concurrency:{tenant_prefix}{slot_type}:{user_id}"
        # Use Lua script for atomic check-and-increment
        lua_script = """
        local current = tonumber(redis.call('GET', KEYS[1]) or '0')
        local max_allowed = tonumber(ARGV[1])
        local ttl = tonumber(ARGV[2])
        if current < max_allowed then
            redis.call('INCR', KEYS[1])
            redis.call('EXPIRE', KEYS[1], ttl)
            return {1, current + 1}
        end
        return {0, current}
        """
        result = await self.client.eval(lua_script, 1, key, max_slots, ttl_seconds)
        return (bool(result[0]), int(result[1]))

    async def release_concurrency_slot(
        self, slot_type: str, user_id: str, *, tenant_id: Optional[str] = None
    ) -> int:
        """Release a concurrency slot for a user.

        Args:
            slot_type: Type of slot (e.g., "workflow", "inference")
            user_id: User ID
            tenant_id: Optional tenant ID for isolation (Issue 44.3)

        Returns:
            Current count after release
        """
        # Issue 44.3: Include tenant_id in concurrency key for tenant isolation
        tenant_prefix = f"{tenant_id}:" if tenant_id else ""
        key = f"concurrency:{tenant_prefix}{slot_type}:{user_id}"
        # Use Lua script to ensure we don't go below 0
        lua_script = """
        local current = tonumber(redis.call('GET', KEYS[1]) or '0')
        if current > 0 then
            return redis.call('DECR', KEYS[1])
        end
        return 0
        """
        result = await self.client.eval(lua_script, 1, key)
        return int(result)

    # =========================================================================
    # Session Activity Tracking (SPEC §12.1)
    # =========================================================================

    async def update_session_activity(self, session_id: str, ttl_seconds: int = 86400) -> None:
        """Update session last activity timestamp."""
        key = f"session:activity:{session_id}"
        now = datetime.now(timezone.utc).isoformat()
        await self.client.set(key, now, ex=ttl_seconds)

    async def get_session_activity(self, session_id: str) -> Optional[datetime]:
        """Get session last activity timestamp."""
        key = f"session:activity:{session_id}"
        value = await self.client.get(key)
        if value:
            try:
                parsed = datetime.fromisoformat(value)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return parsed
            except (ValueError, TypeError):
                return None
        return None

    async def set_session_rotation_grace(
        self, old_session_id: str, new_session_id: str, grace_seconds: int = 300
    ) -> None:
        """Store mapping from old to new session ID during grace period."""
        key = f"session:rotation:{old_session_id}"
        await self.client.set(key, new_session_id, ex=grace_seconds)

    async def get_rotated_session(self, old_session_id: str) -> Optional[str]:
        """Get new session ID if old session was rotated."""
        key = f"session:rotation:{old_session_id}"
        return await self.client.get(key)

    # =========================================================================
    # MFA Lockout Tracking (Issue 19.3 - Atomic operations)
    # =========================================================================

    async def check_mfa_lockout(self, user_id: str) -> bool:
        """Check if user is locked out from MFA attempts.

        Returns:
            True if user is locked out, False otherwise
        """
        key = f"mfa:lockout:{user_id}"
        return bool(await self.client.exists(key))

    async def atomic_mfa_attempt(
        self, user_id: str, max_attempts: int = 5, lockout_seconds: int = 300
    ) -> tuple[bool, int]:
        """Atomically record a failed MFA attempt and check/trigger lockout.

        This uses a Lua script to ensure atomicity and prevent the race condition
        where multiple concurrent failed attempts could each pass the lockout
        check before any increment the counter.

        Args:
            user_id: User ID to track
            max_attempts: Maximum failed attempts before lockout
            lockout_seconds: Duration of lockout in seconds

        Returns:
            Tuple of (is_now_locked_out: bool, current_attempts: int)
        """
        lockout_key = f"mfa:lockout:{user_id}"
        attempts_key = f"mfa:attempts:{user_id}"

        # Lua script for atomic check-and-increment with lockout trigger
        lua_script = """
        -- Check if already locked out
        if redis.call('EXISTS', KEYS[1]) == 1 then
            return {1, -1}  -- Already locked out
        end

        -- Increment attempt counter
        local attempts = redis.call('INCR', KEYS[2])
        redis.call('EXPIRE', KEYS[2], ARGV[2])

        -- Check if we should trigger lockout
        local max_attempts = tonumber(ARGV[1])
        if attempts >= max_attempts then
            redis.call('SET', KEYS[1], '1', 'EX', ARGV[2])
            redis.call('DEL', KEYS[2])
            return {1, attempts}  -- Now locked out
        end

        return {0, attempts}  -- Not locked out
        """
        result = await self.client.eval(
            lua_script, 2, lockout_key, attempts_key, max_attempts, lockout_seconds
        )
        return (bool(result[0]), int(result[1]))

    async def clear_mfa_attempts(self, user_id: str) -> None:
        """Clear MFA attempt counter on successful verification."""
        attempts_key = f"mfa:attempts:{user_id}"
        await self.client.delete(attempts_key)

    # =========================================================================
    # Circuit Breaker (SPEC §18)
    # =========================================================================

    async def check_circuit_breaker(
        self,
        tool_id: str,
        *,
        failure_threshold: int = 5,
        window_seconds: int = 60,
        cooldown_seconds: int = 60,
        tenant_id: Optional[str] = None,
    ) -> tuple[bool, int]:
        """Check if circuit breaker is open for a tool.

        SPEC §18: Circuit breaker opens for a tool after 5 failures in 1 minute.

        Args:
            tool_id: Tool identifier
            failure_threshold: Number of failures to trip breaker (default: 5)
            window_seconds: Time window for failure counting (default: 60)
            cooldown_seconds: How long breaker stays open (default: 60)
            tenant_id: Optional tenant ID for isolation

        Returns:
            Tuple of (is_open: bool, failure_count: int)
            - is_open=True means the circuit is open and tool should not be called
            - failure_count is the current failure count in the window
        """
        tenant_prefix = f"{tenant_id}:" if tenant_id else ""
        open_key = f"circuit:{tenant_prefix}{tool_id}:open"
        failures_key = f"circuit:{tenant_prefix}{tool_id}:{self._FAILURES_SUFFIX}"

        # The cutoff comes from Redis's own clock, not this host's: the write
        # side timestamps with the same clock, so a skewed replica cannot
        # push a score into the future and outlast the window, nor have its
        # failures pruned early (SPEC §18). Returns -1 for an open breaker,
        # else the failures inside the window ending now.
        lua_script = """
        if redis.call('EXISTS', KEYS[1]) == 1 then
            return -1
        end
        local t = redis.call('TIME')
        local now = tonumber(t[1]) + tonumber(t[2]) / 1000000
        local window = tonumber(ARGV[1])
        return redis.call('ZCOUNT', KEYS[2], now - window, '+inf')
        """
        count = int(
            await self.client.eval(
                lua_script, 2, open_key, failures_key, window_seconds
            )
        )
        if count < 0:
            return (True, -1)
        return (False, count)

    async def record_tool_failure(
        self,
        tool_id: str,
        *,
        failure_threshold: int = 5,
        window_seconds: int = 60,
        cooldown_seconds: int = 60,
        tenant_id: Optional[str] = None,
    ) -> tuple[bool, int]:
        """Record a tool failure and potentially trip the circuit breaker.

        SPEC §18: Circuit breaker opens for a tool after 5 failures in 1 minute.

        Uses atomic Lua script to prevent race conditions.

        Args:
            tool_id: Tool identifier
            failure_threshold: Number of failures to trip breaker (default: 5)
            window_seconds: Time window for failure counting (default: 60)
            cooldown_seconds: How long breaker stays open (default: 60)
            tenant_id: Optional tenant ID for isolation

        Returns:
            Tuple of (circuit_tripped: bool, failure_count: int)
        """
        tenant_prefix = f"{tenant_id}:" if tenant_id else ""
        open_key = f"circuit:{tenant_prefix}{tool_id}:open"
        failures_key = f"circuit:{tenant_prefix}{tool_id}:{self._FAILURES_SUFFIX}"

        # A timestamped set, not a counter with a refreshed TTL. The TTL
        # refresh made the rule "a chain of failures with no gap over the
        # window": one failure every fifty seconds eventually tripped a
        # breaker whose sixty-second window never held five. Entries older
        # than the window are pruned before counting, so the count is the
        # failures actually inside one window ending now.
        #
        # The timestamp is Redis's own clock, read inside the script, not
        # this host's `time.time()`: with several replicas a skewed one
        # would otherwise score a failure in the future and keep the breaker
        # tripped past the window, or score it in the past and be pruned
        # early. One clock, the ledger's, so the window means the same thing
        # to every replica (SPEC §18). The member is a caller-supplied unique
        # id — data, not a clock — so two failures at the same instant both
        # count.
        lua_script = """
        -- Check if circuit is already open
        if redis.call('EXISTS', KEYS[1]) == 1 then
            return {1, -1}  -- Already open
        end

        local t = redis.call('TIME')
        local now = tonumber(t[1]) + tonumber(t[2]) / 1000000
        local window = tonumber(ARGV[1])
        redis.call('ZREMRANGEBYSCORE', KEYS[2], '-inf', now - window)
        redis.call('ZADD', KEYS[2], now, ARGV[4])
        redis.call('EXPIRE', KEYS[2], math.ceil(window))
        local failures = redis.call('ZCARD', KEYS[2])

        -- Check if we should trip the circuit
        local threshold = tonumber(ARGV[2])
        if failures >= threshold then
            redis.call('SET', KEYS[1], '1', 'EX', ARGV[3])
            redis.call('DEL', KEYS[2])  -- Clear the window
            return {1, failures}  -- Circuit tripped
        end

        return {0, failures}  -- Not tripped
        """
        result = await self.client.eval(
            lua_script, 2, open_key, failures_key,
            window_seconds, failure_threshold, cooldown_seconds,
            uuid4().hex,
        )
        return (bool(result[0]), int(result[1]))

    async def record_tool_success(
        self,
        tool_id: str,
        *,
        tenant_id: Optional[str] = None,
    ) -> None:
        """Record a successful tool execution, resetting failure count.

        This implements the "half-open" behavior where a success resets the
        failure counter, allowing the circuit to eventually close.
        """
        tenant_prefix = f"{tenant_id}:" if tenant_id else ""
        failures_key = f"circuit:{tenant_prefix}{tool_id}:{self._FAILURES_SUFFIX}"
        await self.client.delete(failures_key)
