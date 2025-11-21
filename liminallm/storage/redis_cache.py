from __future__ import annotations

from datetime import datetime
from typing import Optional

import redis.asyncio as redis


class RedisCache:
    """Thin Redis wrapper for sessions and rate limits."""

    def __init__(self, redis_url: str):
        self.client = redis.from_url(redis_url, decode_responses=True)

    async def cache_session(self, session_id: str, user_id: str, expires_at: datetime) -> None:
        ttl = int((expires_at - datetime.utcnow()).total_seconds())
        await self.client.set(f"session:{session_id}", user_id, ex=ttl)

    async def get_session_user(self, session_id: str) -> Optional[str]:
        return await self.client.get(f"session:{session_id}")

    async def revoke_session(self, session_id: str) -> None:
        await self.client.delete(f"session:{session_id}")

    async def check_rate_limit(self, key: str, limit: int, window_seconds: int) -> bool:
        now_bucket = int(datetime.utcnow().timestamp() // window_seconds)
        redis_key = f"rl:{key}:{now_bucket}"
        current = await self.client.incr(redis_key)
        if current == 1:
            await self.client.expire(redis_key, window_seconds)
        return current <= limit
