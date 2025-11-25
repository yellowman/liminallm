from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, Optional

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

    async def cache_session(self, session_id: str, user_id: str, expires_at: datetime) -> None:
        ttl = int((expires_at - datetime.utcnow()).total_seconds())
        await self.client.set(f"auth:session:{session_id}", user_id, ex=ttl)

    async def get_session_user(self, session_id: str) -> Optional[str]:
        return await self.client.get(f"auth:session:{session_id}")

    async def revoke_session(self, session_id: str) -> None:
        await self.client.delete(f"auth:session:{session_id}")

    async def check_rate_limit(self, key: str, limit: int, window_seconds: int) -> bool:
        now_bucket = int(datetime.utcnow().timestamp() // window_seconds)
        redis_key = f"rate:{key}:{now_bucket}"
        current = await self.client.incr(redis_key)
        if current == 1:
            await self.client.expire(redis_key, window_seconds)
        return current <= limit

    async def mark_refresh_revoked(self, jti: str, ttl_seconds: int) -> None:
        await self.client.set(f"auth:refresh:revoked:{jti}", "1", ex=ttl_seconds)

    async def is_refresh_revoked(self, jti: str) -> bool:
        return bool(await self.client.exists(f"auth:refresh:revoked:{jti}"))

    async def get_router_cache(self, user_id: str, ctx_hash: str) -> Optional[dict]:
        cached = await self.client.get(f"router:last:{user_id}:{ctx_hash}")
        return json.loads(cached) if cached else None

    async def set_router_cache(self, user_id: str, ctx_hash: str, payload: dict, ttl_seconds: int = 300) -> None:
        await self.client.set(
            f"router:last:{user_id}:{ctx_hash}", json.dumps(payload), ex=ttl_seconds
        )

    async def get_workflow_state(self, state_key: str) -> Optional[dict]:
        cached = await self.client.get(f"workflow:state:{state_key}")
        return json.loads(cached) if cached else None

    async def set_workflow_state(self, state_key: str, state: dict, ttl_seconds: int = 1800) -> None:
        await self.client.set(
            f"workflow:state:{state_key}", json.dumps(state), ex=ttl_seconds
        )

    async def get_conversation_summary(self, conversation_id: str) -> Optional[dict]:
        cached = await self.client.get(f"chat:summary:{conversation_id}")
        return json.loads(cached) if cached else None

    async def set_conversation_summary(
        self, conversation_id: str, summary: Dict[str, Any], ttl_seconds: int = 3600
    ) -> None:
        await self.client.set(
            f"chat:summary:{conversation_id}", json.dumps(summary), ex=ttl_seconds
        )

    async def get_idempotency_record(self, route: str, user_id: str, key: str) -> Optional[dict]:
        cached = await self.client.get(f"idemp:{route}:{user_id}:{key}")
        return json.loads(cached) if cached else None

    async def set_idempotency_record(
        self, route: str, user_id: str, key: str, record: dict, ttl_seconds: int = 60 * 60 * 24
    ) -> None:
        await self.client.set(
            f"idemp:{route}:{user_id}:{key}", json.dumps(record), ex=ttl_seconds
        )
