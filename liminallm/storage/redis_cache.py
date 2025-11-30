from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, Union

import redis.asyncio as redis


class RedisCache:
    """Thin Redis wrapper for sessions and rate limits."""

    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self.client = redis.from_url(redis_url, decode_responses=True)

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
        ttl = max(1, int((expires_at - datetime.utcnow()).total_seconds()))
        await self.client.set(f"auth:session:{session_id}", user_id, ex=ttl)

    async def get_session_user(self, session_id: str) -> Optional[str]:
        return await self.client.get(f"auth:session:{session_id}")

    async def revoke_session(self, session_id: str) -> None:
        await self.client.delete(f"auth:session:{session_id}")

    async def check_rate_limit(
        self, key: str, limit: int, window_seconds: int, *, return_remaining: bool = False
    ) -> Union[bool, Tuple[bool, int]]:
        """Check rate limit using Redis token bucket with atomic operations.

        Args:
            key: Rate limit key
            limit: Maximum requests per window
            window_seconds: Window duration in seconds
            return_remaining: If True, return (allowed, remaining) tuple

        Returns:
            bool if return_remaining is False, else (allowed, remaining) tuple
        """
        now = datetime.utcnow()
        now_bucket = int(now.timestamp() // window_seconds)
        redis_key = f"rate:{key}:{now_bucket}"
        bucket_end = (now_bucket + 1) * window_seconds
        ttl = max(1, int(bucket_end - now.timestamp()))

        # Use pipeline for atomic INCR + EXPIRE to prevent race conditions
        pipe = self.client.pipeline()
        pipe.incr(redis_key)
        pipe.expire(redis_key, ttl)
        results = await pipe.execute()
        current = results[0]

        allowed = current <= limit
        if return_remaining:
            remaining = max(0, limit - current)
            return (allowed, remaining)
        return allowed

    async def mark_refresh_revoked(self, jti: str, ttl_seconds: int) -> None:
        await self.client.set(f"auth:refresh:revoked:{jti}", "1", ex=ttl_seconds)

    async def is_refresh_revoked(self, jti: str) -> bool:
        return bool(await self.client.exists(f"auth:refresh:revoked:{jti}"))

    async def get_router_cache(self, user_id: str, ctx_hash: str) -> Optional[dict]:
        cached = await self.client.get(f"router:last:{user_id}:{ctx_hash}")
        if not cached:
            return None
        try:
            return json.loads(cached)
        except (json.JSONDecodeError, TypeError):
            # Corrupted cache entry - treat as cache miss
            return None

    async def set_router_cache(
        self, user_id: str, ctx_hash: str, payload: dict, ttl_seconds: int = 300
    ) -> None:
        await self.client.set(
            f"router:last:{user_id}:{ctx_hash}", json.dumps(payload), ex=ttl_seconds
        )

    async def get_workflow_state(self, state_key: str) -> Optional[dict]:
        cached = await self.client.get(f"workflow:state:{state_key}")
        if not cached:
            return None
        try:
            return json.loads(cached)
        except (json.JSONDecodeError, TypeError):
            # Corrupted cache entry - treat as cache miss
            return None

    async def set_workflow_state(
        self, state_key: str, state: dict, ttl_seconds: int = 1800
    ) -> None:
        await self.client.set(
            f"workflow:state:{state_key}", json.dumps(state), ex=ttl_seconds
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
        ttl = max(1, int((expires_at - datetime.utcnow()).total_seconds()))
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
        expires_at = (
            datetime.fromisoformat(expires_raw)
            if isinstance(expires_raw, str)
            else datetime.utcnow()
        )
        return data.get("provider"), expires_at, data.get("tenant_id")

    async def get_idempotency_record(
        self, route: str, user_id: str, key: str
    ) -> Optional[dict]:
        cached = await self.client.get(f"idemp:{route}:{user_id}:{key}")
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
    ) -> None:
        await self.client.set(
            f"idemp:{route}:{user_id}:{key}", json.dumps(record), ex=ttl_seconds
        )

    async def close(self) -> None:
        """Close Redis connection pool. Call when shutting down or resetting runtime."""
        await self.client.close()
        await self.client.connection_pool.disconnect()

    async def delete_workflow_state(self, state_key: str) -> None:
        """Delete workflow state from cache during rollback or cleanup."""
        await self.client.delete(f"workflow:state:{state_key}")
