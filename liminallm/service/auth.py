from __future__ import annotations

import hashlib
import hmac
import os
from datetime import datetime, timedelta
from typing import Optional, Tuple

from liminallm.storage.models import Session, User
from liminallm.storage.postgres import PostgresStore
from liminallm.storage.redis_cache import RedisCache
from liminallm.storage.memory import MemoryStore


class AuthService:
    """Session and MFA handling for both persistent and in-memory modes."""

    def __init__(self, store: PostgresStore | MemoryStore, cache: Optional[RedisCache] = None) -> None:
        self.store = store
        self.cache = cache
        self._mfa_challenges: dict[str, tuple[str, datetime]] = {}

    async def signup(self, email: str, password: str, handle: Optional[str] = None) -> tuple[User, Session]:
        user = self.store.create_user(email=email, handle=handle)
        pwd_hash, algo = self._hash_password(password)
        if hasattr(self.store, "save_password"):
            self.store.save_password(user.id, pwd_hash, algo)  # type: ignore[attr-defined]
        session = self.store.create_session(user.id)
        if self.cache:
            await self.cache.cache_session(session.id, user.id, session.expires_at)
        return user, session

    async def login(self, email: str, password: str) -> tuple[Optional[User], Optional[Session]]:
        user = self.store.get_user_by_email(email) if hasattr(self.store, "get_user_by_email") else None  # type: ignore[attr-defined]
        if not user or not self._verify_password(user.id, password):
            return None, None
        session = self.store.create_session(user.id)
        if self.cache:
            await self.cache.cache_session(session.id, user.id, session.expires_at)
        return user, session

    async def revoke(self, session_id: str) -> None:
        self.store.revoke_session(session_id)
        if self.cache:
            await self.cache.revoke_session(session_id)

    async def resolve_session(self, session_id: Optional[str]) -> Optional[str]:
        if not session_id:
            return None
        if self.cache:
            cached = await self.cache.get_session_user(session_id)
            if cached:
                return cached
        sess = None
        if hasattr(self.store, "get_session"):
            sess = self.store.get_session(session_id)  # type: ignore[attr-defined]
        if sess and sess.expires_at > datetime.utcnow():
            if self.cache:
                await self.cache.cache_session(sess.id, sess.user_id, sess.expires_at)
            return sess.user_id
        return None

    async def issue_mfa_challenge(self, user_id: str) -> str:
        code = hashlib.sha256(f"{user_id}-{os.urandom(6)}".encode()).hexdigest()[:6]
        expires_at = datetime.utcnow() + timedelta(minutes=5)
        ttl = int((expires_at - datetime.utcnow()).total_seconds())
        if self.cache:
            await self.cache.client.set(f"mfa:{user_id}", code, ex=ttl)
        else:
            self._mfa_challenges[user_id] = (code, expires_at)
        # A real implementation would deliver this via SMS/email; here we just return it
        return code

    async def verify_mfa_challenge(self, user_id: str, code: str) -> bool:
        expected: Optional[str | bytes] = None
        expires_at: Optional[datetime] = None
        if self.cache:
            expected = await self.cache.client.get(f"mfa:{user_id}")
        else:
            if user_id in self._mfa_challenges:
                expected, expires_at = self._mfa_challenges[user_id]
        if not expected:
            return False
        if expires_at and expires_at < datetime.utcnow():
            self._mfa_challenges.pop(user_id, None)
            return False
        if isinstance(expected, bytes):
            expected = expected.decode()
        if not hmac.compare_digest(code, expected):
            return False
        if self.cache:
            await self.cache.client.delete(f"mfa:{user_id}")
        else:
            self._mfa_challenges.pop(user_id, None)
        return True

    async def initiate_password_reset(self, email: str) -> str:
        token = hashlib.sha256(f"reset-{email}-{os.urandom(6)}".encode()).hexdigest()
        # Persist a short-lived reset token with TTL in Redis if available
        if self.cache:
            expires_at = datetime.utcnow() + timedelta(minutes=15)
            await self.cache.client.set(
                f"reset:{token}", email, ex=int((expires_at - datetime.utcnow()).total_seconds())
            )
        return token

    async def complete_password_reset(self, token: str, new_password: str) -> bool:
        email = None
        if self.cache:
            email = await self.cache.client.get(f"reset:{token}")
        if not email:
            return False
        if isinstance(email, bytes):
            email = email.decode()
        user = self.store.get_user_by_email(email) if hasattr(self.store, "get_user_by_email") else None  # type: ignore[attr-defined]
        if not user:
            return False
        pwd_hash, algo = self._hash_password(new_password)
        if hasattr(self.store, "save_password"):
            self.store.save_password(user.id, pwd_hash, algo)  # type: ignore[attr-defined]
        return True

    def _hash_password(self, password: str) -> Tuple[str, str]:
        salt = os.urandom(8).hex()
        algo = "sha256"
        digest = hashlib.sha256(f"{salt}:{password}".encode()).hexdigest()
        return f"{salt}${digest}", algo

    def _verify_password(self, user_id: str, password: str) -> bool:
        if not hasattr(self.store, "get_password_record"):
            return False
        record = self.store.get_password_record(user_id)  # type: ignore[attr-defined]
        if not record:
            return False
        stored_hash, algo = record
        if algo != "sha256" or "$" not in stored_hash:
            return False
        salt, expected = stored_hash.split("$", 1)
        digest = hashlib.sha256(f"{salt}:{password}".encode()).hexdigest()
        return hmac.compare_digest(expected, digest)
