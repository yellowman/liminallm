from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple, Union

import redis.asyncio as aioredis
from redis import Redis


class RedisCache:
    """Thin Redis wrapper for sessions and rate limits."""

    # Issue 48.3: Default operation timeout for Redis commands
    DEFAULT_OPERATION_TIMEOUT = 5.0  # 5 seconds

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
        # Issue 48.3: Configure connection with explicit timeouts
        self.client = aioredis.from_url(
            redis_url,
            decode_responses=True,
            socket_timeout=socket_timeout,
            socket_connect_timeout=socket_timeout,
        )
        # Register token bucket script for atomic rate limiting (Issue 77.10)
        self._token_bucket = self.client.register_script(self._TOKEN_BUCKET_SCRIPT)

    @staticmethod
    def _ttl_seconds(expires_at: datetime) -> int:
        """Compute a safe TTL from an absolute expiry timestamp.

        Sessions now use timezone-aware UTC timestamps; older records may be naive.
        Normalize to UTC and clamp to at least 1 second to avoid Redis rejecting
        negative or zero TTL values. (Issue 2.6)
        """

        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        else:
            expires_at = expires_at.astimezone(timezone.utc)
        return max(1, int((expires_at - datetime.now(timezone.utc)).total_seconds()))

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
        ttl = self._ttl_seconds(expires_at)
        pipe = self.client.pipeline()
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

    async def get_session_user(self, session_id: str) -> Optional[str]:
        return await self.client.get(f"auth:session:{session_id}")

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
        pipe = self.client.pipeline()
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

    async def set_oauth_state(
        self, state: str, provider: str, expires_at: datetime, tenant_id: Optional[str]
    ) -> None:
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        else:
            expires_at = expires_at.astimezone(timezone.utc)

        ttl = self._ttl_seconds(expires_at)
        payload = {
            "provider": provider,
            "expires_at": expires_at.isoformat(),
            "tenant_id": tenant_id,
        }
        await self.client.set(f"auth:oauth:{state}", json.dumps(payload), ex=ttl)

    async def pop_oauth_state(
        self, state: str
    ) -> Optional[tuple[str, datetime, Optional[str]]]:
        """Atomically get and delete OAuth state to prevent replay attacks.

        Uses Redis GETDEL command (Redis 6.2+) or Lua script fallback to ensure
        atomicity. This prevents race conditions where two concurrent requests
        could both consume the same OAuth state.

        Args:
            state: The OAuth state token to consume

        Returns:
            Tuple of (provider, expires_at, tenant_id) or None if not found
        """
        key = f"auth:oauth:{state}"

        # Try GETDEL first (Redis 6.2+) for atomic get-and-delete
        try:
            cached = await self.client.getdel(key)
        except AttributeError:
            # Fallback for older redis-py versions: use Lua script for atomicity
            lua_script = """
            local value = redis.call('GET', KEYS[1])
            if value then
                redis.call('DEL', KEYS[1])
            end
            return value
            """
            cached = await self.client.eval(lua_script, 1, key)

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
            except (ValueError, TypeError):
                pass  # Use default current UTC time
        return data.get("provider"), expires_at, data.get("tenant_id")

    async def get_idempotency_record(
        self, route: str, user_id: str, key: str, *, tenant_id: Optional[str] = None
    ) -> Optional[dict]:
        # Issue 22.2: Include tenant_id in cache key for multi-tenant isolation
        tenant_prefix = f"{tenant_id}:" if tenant_id else ""
        cached = await self.client.get(f"idemp:{tenant_prefix}{route}:{user_id}:{key}")
        if not cached:
            return None
        try:
            return json.loads(cached)
        except (json.JSONDecodeError, TypeError):
            # Corrupted idempotency record - treat as not found
            return None

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
        # Use SET NX (set if not exists) for atomic acquisition
        acquired = await self.client.set(
            cache_key, json.dumps(record), ex=ttl_seconds, nx=True
        )
        if acquired:
            return (True, None)
        # Slot was not acquired, fetch existing record
        existing = await self.client.get(cache_key)
        if existing:
            try:
                return (False, json.loads(existing))
            except (json.JSONDecodeError, TypeError):
                pass
        return (False, None)

    async def close(self) -> None:
        """Close Redis connection pool. Call when shutting down or resetting runtime."""
        await self.client.close()
        await self.client.connection_pool.disconnect()

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

    async def get_concurrency_count(
        self, slot_type: str, user_id: str, *, tenant_id: Optional[str] = None
    ) -> int:
        """Get current concurrency count for a user."""
        # Issue 44.3: Include tenant_id in concurrency key for tenant isolation
        tenant_prefix = f"{tenant_id}:" if tenant_id else ""
        key = f"concurrency:{tenant_prefix}{slot_type}:{user_id}"
        count = await self.client.get(key)
        return int(count) if count else 0

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
                return datetime.fromisoformat(value)
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
        failures_key = f"circuit:{tenant_prefix}{tool_id}:failures"

        # Check if circuit is open (tripped)
        is_open = await self.client.exists(open_key)
        if is_open:
            return (True, -1)

        # Get current failure count
        failures_raw = await self.client.get(failures_key)
        failure_count = int(failures_raw) if failures_raw else 0

        return (False, failure_count)

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
        failures_key = f"circuit:{tenant_prefix}{tool_id}:failures"

        # Lua script for atomic failure recording and circuit tripping
        lua_script = """
        -- Check if circuit is already open
        if redis.call('EXISTS', KEYS[1]) == 1 then
            return {1, -1}  -- Already open
        end

        -- Increment failure counter
        local failures = redis.call('INCR', KEYS[2])
        redis.call('EXPIRE', KEYS[2], ARGV[1])

        -- Check if we should trip the circuit
        local threshold = tonumber(ARGV[2])
        if failures >= threshold then
            redis.call('SET', KEYS[1], '1', 'EX', ARGV[3])
            redis.call('DEL', KEYS[2])  -- Clear failures counter
            return {1, failures}  -- Circuit tripped
        end

        return {0, failures}  -- Not tripped
        """
        result = await self.client.eval(
            lua_script, 2, open_key, failures_key,
            window_seconds, failure_threshold, cooldown_seconds
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
        failures_key = f"circuit:{tenant_prefix}{tool_id}:failures"
        await self.client.delete(failures_key)

    async def reset_circuit_breaker(
        self,
        tool_id: str,
        *,
        tenant_id: Optional[str] = None,
    ) -> None:
        """Manually reset a circuit breaker (admin action)."""
        tenant_prefix = f"{tenant_id}:" if tenant_id else ""
        open_key = f"circuit:{tenant_prefix}{tool_id}:open"
        failures_key = f"circuit:{tenant_prefix}{tool_id}:failures"
        pipe = self.client.pipeline()
        pipe.delete(open_key)
        pipe.delete(failures_key)
        await pipe.execute()


class _SyncClientAdapter:
    """Adapter that wraps a sync Redis client with async method signatures.

    This allows code that uses `await self.cache.client.method()` to work
    with either async or sync Redis clients uniformly.
    """

    def __init__(self, sync_client: Redis):
        self._sync = sync_client

    async def get(self, key: str) -> Optional[str]:
        return self._sync.get(key)

    async def set(self, key: str, value: str, ex: Optional[int] = None) -> bool:
        return self._sync.set(key, value, ex=ex)

    async def delete(self, key: str) -> int:
        return self._sync.delete(key)

    async def incr(self, key: str) -> int:
        return self._sync.incr(key)

    async def expire(self, key: str, ttl: int) -> bool:
        return self._sync.expire(key, ttl)

    async def exists(self, key: str) -> int:
        return self._sync.exists(key)

    def pipeline(self):
        """Return the underlying sync pipeline for batch operations."""
        return self._sync.pipeline()


class SyncRedisCache:
    """Synchronous Redis wrapper for use in tests.

    Uses a synchronous Redis client internally to avoid event loop binding
    issues in pytest, but exposes async methods so they can be awaited
    uniformly like RedisCache. The async methods simply call sync Redis
    operations and return the results.
    """

    # Issue 48.3: Default operation timeout for Redis commands
    DEFAULT_OPERATION_TIMEOUT = 5.0  # 5 seconds

    def __init__(self, redis_url: str, *, socket_timeout: float = 5.0):
        self.redis_url = redis_url
        # Issue 48.3: Configure connection with explicit timeouts
        self._sync_client = Redis.from_url(
            redis_url,
            decode_responses=True,
            socket_timeout=socket_timeout,
            socket_connect_timeout=socket_timeout,
        )
        # Wrap sync client with async-compatible adapter for direct client access
        self.client = _SyncClientAdapter(self._sync_client)
        # Register token bucket script for atomic rate limiting (Issue 77.10)
        self._token_bucket = self._sync_client.register_script(
            RedisCache._TOKEN_BUCKET_SCRIPT
        )

    def verify_connection(self) -> None:
        """Assert Redis connectivity."""
        self._sync_client.ping()

    async def cache_session(
        self, session_id: str, user_id: str, expires_at: datetime
    ) -> None:
        ttl = RedisCache._ttl_seconds(expires_at)
        pipe = self._sync_client.pipeline()
        pipe.set(f"auth:session:{session_id}", user_id, ex=ttl)
        # Track session in user's session set for bulk revocation (Issue 22.3)
        pipe.sadd(f"auth:user_sessions:{user_id}", session_id)
        pipe.expire(f"auth:user_sessions:{user_id}", ttl)
        pipe.execute()

    async def get_session_user(self, session_id: str) -> Optional[str]:
        return self._sync_client.get(f"auth:session:{session_id}")

    async def revoke_session(self, session_id: str) -> None:
        self._sync_client.delete(f"auth:session:{session_id}")

    async def revoke_user_sessions(
        self, user_id: str, except_session_id: Optional[str] = None
    ) -> int:
        """Revoke all cached sessions for a user (sync version)."""
        user_sessions_key = f"auth:user_sessions:{user_id}"
        session_ids = self._sync_client.smembers(user_sessions_key)
        if not session_ids:
            return 0

        revoked = 0
        pipe = self._sync_client.pipeline()
        for session_id in session_ids:
            if except_session_id and session_id == except_session_id:
                continue
            pipe.delete(f"auth:session:{session_id}")
            pipe.srem(user_sessions_key, session_id)
            revoked += 1
        pipe.execute()
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
        now = time.time()
        safe_key = RedisCache._normalize_rate_key(key, tenant_id)
        refill_rate = float(limit) / float(window_seconds)
        allowed, tokens, reset_after = self._token_bucket(
            keys=[safe_key], args=[now, refill_rate, limit, max(1, cost)]
        )

        allowed_bool = bool(int(allowed))
        remaining = max(0, int(tokens))
        reset_seconds = int(reset_after) if reset_after else 0
        if return_remaining:
            return (allowed_bool, remaining, reset_seconds)
        return allowed_bool

    async def mark_refresh_revoked(self, jti: str, ttl_seconds: int) -> None:
        self._sync_client.set(f"auth:refresh:revoked:{jti}", "1", ex=ttl_seconds)

    async def is_refresh_revoked(self, jti: str) -> bool:
        return bool(self._sync_client.exists(f"auth:refresh:revoked:{jti}"))

    # SPEC §12.1: "logout: add JWT to short-lived denylist if JWTs used"
    async def denylist_access_token(self, jti: str, ttl_seconds: int) -> None:
        """Add access token JTI to denylist with TTL matching token expiry."""
        if ttl_seconds > 0:
            self._sync_client.set(f"auth:access:denylist:{jti}", "1", ex=ttl_seconds)

    async def is_access_token_denylisted(self, jti: str) -> bool:
        """Check if access token JTI is in denylist."""
        return bool(self._sync_client.exists(f"auth:access:denylist:{jti}"))

    async def get_router_cache(
        self, user_id: str, ctx_hash: str, *, tenant_id: Optional[str] = None
    ) -> Optional[dict]:
        # Issue 44.5: Include tenant_id in cache key for tenant isolation
        tenant_prefix = f"{tenant_id}:" if tenant_id else ""
        cached = self._sync_client.get(f"router:last:{tenant_prefix}{user_id}:{ctx_hash}")
        if not cached:
            return None
        try:
            return json.loads(cached)
        except (json.JSONDecodeError, TypeError):
            return None

    async def set_router_cache(
        self, user_id: str, ctx_hash: str, payload: dict, ttl_seconds: int = 300,
        *, tenant_id: Optional[str] = None
    ) -> None:
        # Issue 44.5: Include tenant_id in cache key for tenant isolation
        tenant_prefix = f"{tenant_id}:" if tenant_id else ""
        self._sync_client.set(f"router:last:{tenant_prefix}{user_id}:{ctx_hash}", json.dumps(payload), ex=ttl_seconds)

    async def get_workflow_state(
        self, state_key: str, *, tenant_id: Optional[str] = None
    ) -> Optional[dict]:
        # Issue 44.4: Include tenant_id in cache key for tenant isolation
        tenant_prefix = f"{tenant_id}:" if tenant_id else ""
        cached = self._sync_client.get(f"workflow:state:{tenant_prefix}{state_key}")
        if not cached:
            return None
        try:
            return json.loads(cached)
        except (json.JSONDecodeError, TypeError):
            return None

    async def set_workflow_state(
        self, state_key: str, state: dict, ttl_seconds: int = 1800,
        *, tenant_id: Optional[str] = None
    ) -> None:
        # Issue 44.4: Include tenant_id in cache key for tenant isolation
        tenant_prefix = f"{tenant_id}:" if tenant_id else ""
        self._sync_client.set(f"workflow:state:{tenant_prefix}{state_key}", json.dumps(state), ex=ttl_seconds)

    async def get_conversation_summary(self, conversation_id: str) -> Optional[dict]:
        cached = self._sync_client.get(f"chat:summary:{conversation_id}")
        if not cached:
            return None
        try:
            return json.loads(cached)
        except (json.JSONDecodeError, TypeError):
            return None

    async def set_conversation_summary(
        self, conversation_id: str, summary: Dict[str, Any], ttl_seconds: int = 3600
    ) -> None:
        self._sync_client.set(f"chat:summary:{conversation_id}", json.dumps(summary), ex=ttl_seconds)

    async def set_oauth_state(
        self, state: str, provider: str, expires_at: datetime, tenant_id: Optional[str]
    ) -> None:
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        else:
            expires_at = expires_at.astimezone(timezone.utc)

        ttl = RedisCache._ttl_seconds(expires_at)
        payload = {
            "provider": provider,
            "expires_at": expires_at.isoformat(),
            "tenant_id": tenant_id,
        }
        self._sync_client.set(f"auth:oauth:{state}", json.dumps(payload), ex=ttl)

    async def pop_oauth_state(
        self, state: str
    ) -> Optional[tuple[str, datetime, Optional[str]]]:
        """Atomically get and delete OAuth state."""
        key = f"auth:oauth:{state}"
        try:
            cached = self._sync_client.getdel(key)
        except AttributeError:
            # Fallback for older redis-py versions
            cached = self._sync_client.get(key)
            if cached:
                self._sync_client.delete(key)

        if cached is None:
            return None

        try:
            data = json.loads(cached)
        except (json.JSONDecodeError, TypeError):
            return None

        expires_raw = data.get("expires_at")
        # Issue 39.2: Add error handling for datetime parsing
        expires_at = datetime.now(timezone.utc)
        if isinstance(expires_raw, str):
            try:
                expires_at = datetime.fromisoformat(expires_raw)
            except (ValueError, TypeError):
                pass  # Use default current UTC time
        return data.get("provider"), expires_at, data.get("tenant_id")

    async def get_idempotency_record(
        self, route: str, user_id: str, key: str, *, tenant_id: Optional[str] = None
    ) -> Optional[dict]:
        # Issue 22.2: Include tenant_id in cache key for multi-tenant isolation
        tenant_prefix = f"{tenant_id}:" if tenant_id else ""
        cached = self._sync_client.get(f"idemp:{tenant_prefix}{route}:{user_id}:{key}")
        if not cached:
            return None
        try:
            return json.loads(cached)
        except (json.JSONDecodeError, TypeError):
            return None

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
        self._sync_client.set(f"idemp:{tenant_prefix}{route}:{user_id}:{key}", json.dumps(record), ex=ttl_seconds)

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
        """Atomically acquire an idempotency slot using SETNX (sync version)."""
        # Issue 22.2: Include tenant_id in cache key for multi-tenant isolation
        tenant_prefix = f"{tenant_id}:" if tenant_id else ""
        cache_key = f"idemp:{tenant_prefix}{route}:{user_id}:{key}"
        acquired = self._sync_client.set(cache_key, json.dumps(record), ex=ttl_seconds, nx=True)
        if acquired:
            return (True, None)
        existing = self._sync_client.get(cache_key)
        if existing:
            try:
                return (False, json.loads(existing))
            except (json.JSONDecodeError, TypeError):
                pass
        return (False, None)

    async def close(self) -> None:
        """Close Redis connection."""
        self._sync_client.close()

    async def delete_workflow_state(
        self, state_key: str, *, tenant_id: Optional[str] = None
    ) -> None:
        """Delete workflow state from cache."""
        # Issue 44.4: Include tenant_id in cache key for tenant isolation
        tenant_prefix = f"{tenant_id}:" if tenant_id else ""
        self._sync_client.delete(f"workflow:state:{tenant_prefix}{state_key}")

    # =========================================================================
    # Concurrency Caps (SPEC §18)
    # =========================================================================

    async def acquire_concurrency_slot(
        self, slot_type: str, user_id: str, max_slots: int, ttl_seconds: int = 3600,
        *, tenant_id: Optional[str] = None
    ) -> tuple[bool, int]:
        """Atomically acquire a concurrency slot for a user."""
        # Issue 44.3: Include tenant_id in concurrency key for tenant isolation
        tenant_prefix = f"{tenant_id}:" if tenant_id else ""
        key = f"concurrency:{tenant_prefix}{slot_type}:{user_id}"
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
        result = self._sync_client.eval(lua_script, 1, key, max_slots, ttl_seconds)
        return (bool(result[0]), int(result[1]))

    async def release_concurrency_slot(
        self, slot_type: str, user_id: str, *, tenant_id: Optional[str] = None
    ) -> int:
        """Release a concurrency slot for a user."""
        # Issue 44.3: Include tenant_id in concurrency key for tenant isolation
        tenant_prefix = f"{tenant_id}:" if tenant_id else ""
        key = f"concurrency:{tenant_prefix}{slot_type}:{user_id}"
        lua_script = """
        local current = tonumber(redis.call('GET', KEYS[1]) or '0')
        if current > 0 then
            return redis.call('DECR', KEYS[1])
        end
        return 0
        """
        result = self._sync_client.eval(lua_script, 1, key)
        return int(result)

    async def get_concurrency_count(
        self, slot_type: str, user_id: str, *, tenant_id: Optional[str] = None
    ) -> int:
        """Get current concurrency count for a user."""
        # Issue 44.3: Include tenant_id in concurrency key for tenant isolation
        tenant_prefix = f"{tenant_id}:" if tenant_id else ""
        key = f"concurrency:{tenant_prefix}{slot_type}:{user_id}"
        count = self._sync_client.get(key)
        return int(count) if count else 0

    # =========================================================================
    # Session Activity Tracking (SPEC §12.1)
    # =========================================================================

    async def update_session_activity(self, session_id: str, ttl_seconds: int = 86400) -> None:
        """Update session last activity timestamp."""
        key = f"session:activity:{session_id}"
        now = datetime.now(timezone.utc).isoformat()
        self._sync_client.set(key, now, ex=ttl_seconds)

    async def get_session_activity(self, session_id: str) -> Optional[datetime]:
        """Get session last activity timestamp."""
        key = f"session:activity:{session_id}"
        value = self._sync_client.get(key)
        if value:
            try:
                return datetime.fromisoformat(value)
            except (ValueError, TypeError):
                return None
        return None

    async def set_session_rotation_grace(
        self, old_session_id: str, new_session_id: str, grace_seconds: int = 300
    ) -> None:
        """Store mapping from old to new session ID during grace period."""
        key = f"session:rotation:{old_session_id}"
        self._sync_client.set(key, new_session_id, ex=grace_seconds)

    async def get_rotated_session(self, old_session_id: str) -> Optional[str]:
        """Get new session ID if old session was rotated."""
        key = f"session:rotation:{old_session_id}"
        return self._sync_client.get(key)

    # =========================================================================
    # MFA Lockout Tracking (Issue 19.3 - Atomic operations)
    # =========================================================================

    async def check_mfa_lockout(self, user_id: str) -> bool:
        """Check if user is locked out from MFA attempts."""
        key = f"mfa:lockout:{user_id}"
        return bool(self._sync_client.exists(key))

    async def atomic_mfa_attempt(
        self, user_id: str, max_attempts: int = 5, lockout_seconds: int = 300
    ) -> tuple[bool, int]:
        """Atomically record a failed MFA attempt and check/trigger lockout (sync version)."""
        lockout_key = f"mfa:lockout:{user_id}"
        attempts_key = f"mfa:attempts:{user_id}"

        lua_script = """
        if redis.call('EXISTS', KEYS[1]) == 1 then
            return {1, -1}
        end
        local attempts = redis.call('INCR', KEYS[2])
        redis.call('EXPIRE', KEYS[2], ARGV[2])
        local max_attempts = tonumber(ARGV[1])
        if attempts >= max_attempts then
            redis.call('SET', KEYS[1], '1', 'EX', ARGV[2])
            redis.call('DEL', KEYS[2])
            return {1, attempts}
        end
        return {0, attempts}
        """
        result = self._sync_client.eval(lua_script, 2, lockout_key, attempts_key, max_attempts, lockout_seconds)
        return (bool(result[0]), int(result[1]))

    async def clear_mfa_attempts(self, user_id: str) -> None:
        """Clear MFA attempt counter on successful verification."""
        attempts_key = f"mfa:attempts:{user_id}"
        self._sync_client.delete(attempts_key)
