from __future__ import annotations

import base64
import hashlib
import hmac
import os
import time
from datetime import datetime, timedelta
from typing import Optional, Tuple

from liminallm.storage.models import Session, User
from liminallm.storage.postgres import PostgresStore
from liminallm.storage.redis_cache import RedisCache
from liminallm.storage.memory import MemoryStore


class AuthService:
    """Session and MFA handling for both persistent and in-memory modes."""

    def __init__(self, store: PostgresStore | MemoryStore, cache: Optional[RedisCache] = None, *, mfa_enabled: bool = True) -> None:
        self.store = store
        self.cache = cache
        self._mfa_challenges: dict[str, tuple[str, datetime]] = {}
        self.mfa_enabled = mfa_enabled

    async def signup(self, email: str, password: str, handle: Optional[str] = None) -> tuple[User, Session]:
        user = self.store.create_user(email=email, handle=handle)
        pwd_hash, algo = self._hash_password(password)
        if hasattr(self.store, "save_password"):
            self.store.save_password(user.id, pwd_hash, algo)  # type: ignore[attr-defined]
        session = self.store.create_session(user.id)
        if self.cache:
            await self.cache.cache_session(session.id, user.id, session.expires_at)
        return user, session

    async def login(self, email: str, password: str, mfa_code: Optional[str] = None) -> tuple[Optional[User], Optional[Session]]:
        user = self.store.get_user_by_email(email) if hasattr(self.store, "get_user_by_email") else None  # type: ignore[attr-defined]
        if not user or not self._verify_password(user.id, password):
            return None, None
        mfa_cfg = self.store.get_user_mfa_secret(user.id) if self.mfa_enabled and hasattr(self.store, "get_user_mfa_secret") else None  # type: ignore[attr-defined]
        require_mfa = bool(self.mfa_enabled and mfa_cfg and mfa_cfg.enabled)
        session = self.store.create_session(user.id, mfa_required=require_mfa)
        if require_mfa and mfa_cfg:
            if mfa_code and self._verify_totp(mfa_cfg.secret, mfa_code):
                self._mark_session_verified(session.id)
                session.mfa_verified = True
            else:
                session.mfa_verified = False
        if self.cache and (not require_mfa or session.mfa_verified):
            await self.cache.cache_session(session.id, user.id, session.expires_at)
        return user, session

    async def revoke(self, session_id: str) -> None:
        self.store.revoke_session(session_id)
        if self.cache:
            await self.cache.revoke_session(session_id)

    async def resolve_session(self, session_id: Optional[str]) -> Optional[str]:
        if not session_id:
            return None
        sess = None
        if hasattr(self.store, "get_session"):
            sess = self.store.get_session(session_id)  # type: ignore[attr-defined]
        if not sess and self.cache:
            cached_user = await self.cache.get_session_user(session_id)
            if cached_user and hasattr(self.store, "get_session"):
                sess = self.store.get_session(session_id)  # type: ignore[attr-defined]
        if sess and sess.expires_at > datetime.utcnow():
            if self.cache and (not sess.mfa_required or sess.mfa_verified or not self.mfa_enabled):
                await self.cache.cache_session(sess.id, sess.user_id, sess.expires_at)
            if sess.mfa_required and not sess.mfa_verified and self.mfa_enabled:
                return None
            return sess.user_id
        return None

    async def issue_mfa_challenge(self, user_id: str) -> dict:
        if not self.mfa_enabled:
            return {"status": "disabled"}
        existing = self.store.get_user_mfa_secret(user_id) if hasattr(self.store, "get_user_mfa_secret") else None  # type: ignore[attr-defined]
        secret = existing.secret if existing else base64.b32encode(os.urandom(10)).decode("utf-8").rstrip("=")
        self.store.set_user_mfa_secret(user_id, secret, enabled=False)
        uri = f"otpauth://totp/liminallm:{user_id}?secret={secret}&issuer=LiminalLM"
        return {"otpauth_uri": uri, "secret": secret}

    async def verify_mfa_challenge(self, user_id: str, code: str, session_id: Optional[str] = None) -> bool:
        if not self.mfa_enabled:
            return True
        cfg = self.store.get_user_mfa_secret(user_id) if hasattr(self.store, "get_user_mfa_secret") else None  # type: ignore[attr-defined]
        if not cfg:
            return False
        if not self._verify_totp(cfg.secret, code):
            return False
        self.store.set_user_mfa_secret(user_id, cfg.secret, enabled=True)
        if session_id:
            self._mark_session_verified(session_id)
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

    def _verify_totp(self, secret: str, code: str, *, window: int = 1, interval: int = 30) -> bool:
        for offset in range(-window, window + 1):
            if self._generate_totp(secret, time.time() + offset * interval, interval=interval) == code:
                return True
        return False

    def _generate_totp(self, secret: str, timestamp: float, *, interval: int = 30, digits: int = 6) -> str:
        padded = secret + "=" * ((8 - len(secret) % 8) % 8)
        key = base64.b32decode(padded, True)
        counter = int(timestamp // interval).to_bytes(8, "big")
        digest = hmac.new(key, counter, hashlib.sha1).digest()
        offset = digest[-1] & 0x0F
        code_int = (int.from_bytes(digest[offset : offset + 4], "big") & 0x7FFFFFFF) % (10**digits)
        return str(code_int).zfill(digits)

    def _mark_session_verified(self, session_id: str) -> None:
        if hasattr(self.store, "mark_session_verified"):
            self.store.mark_session_verified(session_id)  # type: ignore[attr-defined]
