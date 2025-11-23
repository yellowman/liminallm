from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional, Tuple

from liminallm.config import Settings
from liminallm.storage.models import Session, User
from liminallm.storage.postgres import PostgresStore
from liminallm.storage.redis_cache import RedisCache
from liminallm.storage.memory import MemoryStore


@dataclass
class AuthContext:
    user_id: str
    role: str
    tenant_id: str
    session_id: Optional[str] = None


class AuthService:
    """Session, JWT, and MFA handling for both persistent and in-memory modes."""

    def __init__(
        self,
        store: PostgresStore | MemoryStore,
        cache: Optional[RedisCache],
        settings: Settings,
        *,
        mfa_enabled: bool = True,
    ) -> None:
        self.store = store
        self.cache = cache
        self.settings = settings
        self._mfa_challenges: dict[str, tuple[str, datetime]] = {}
        self.mfa_enabled = mfa_enabled
        self.revoked_refresh_tokens: set[str] = set()

    def _generate_password(self) -> str:
        return base64.urlsafe_b64encode(os.urandom(12)).decode().rstrip("=")

    async def signup(
        self,
        email: str,
        password: str,
        handle: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> tuple[User, Session, dict[str, str]]:
        user = self.store.create_user(
            email=email,
            handle=handle,
            tenant_id=tenant_id or self.settings.default_tenant_id,
        )
        pwd_hash, algo = self._hash_password(password)
        if hasattr(self.store, "save_password"):
            self.store.save_password(user.id, pwd_hash, algo)  # type: ignore[attr-defined]
        session = self.store.create_session(user.id, tenant_id=user.tenant_id)
        tokens = self._issue_tokens(user, session)
        if self.cache:
            await self.cache.cache_session(session.id, user.id, session.expires_at)
        return user, session, tokens

    async def admin_create_user(
        self,
        *,
        email: str,
        password: Optional[str] = None,
        handle: Optional[str] = None,
        tenant_id: Optional[str] = None,
        role: Optional[str] = None,
        plan_tier: Optional[str] = None,
        is_active: Optional[bool] = None,
        meta: Optional[dict] = None,
    ) -> tuple[User, str]:
        pwd = password or self._generate_password()
        user = self.store.create_user(
            email=email,
            handle=handle,
            tenant_id=tenant_id or self.settings.default_tenant_id,
            role=role or "user",
            plan_tier=plan_tier or "free",
            is_active=is_active if is_active is not None else True,
            meta=meta,
        )
        pwd_hash, algo = self._hash_password(pwd)
        if hasattr(self.store, "save_password"):
            self.store.save_password(user.id, pwd_hash, algo)  # type: ignore[attr-defined]
        return user, pwd

    def list_users(self, tenant_id: Optional[str] = None, limit: int = 100) -> list[User]:
        if hasattr(self.store, "list_users"):
            return self.store.list_users(tenant_id=tenant_id, limit=limit)  # type: ignore[attr-defined]
        return []

    def set_user_role(self, user_id: str, role: str) -> Optional[User]:
        if hasattr(self.store, "update_user_role"):
            return self.store.update_user_role(user_id, role)  # type: ignore[attr-defined]
        return None

    def delete_user(self, user_id: str) -> bool:
        if hasattr(self.store, "delete_user"):
            return bool(self.store.delete_user(user_id))  # type: ignore[attr-defined]
        return False

    async def login(
        self,
        email: str,
        password: str,
        mfa_code: Optional[str] = None,
        *,
        tenant_id: Optional[str] = None,
    ) -> tuple[Optional[User], Optional[Session], dict[str, str]]:
        user = self.store.get_user_by_email(email) if hasattr(self.store, "get_user_by_email") else None  # type: ignore[attr-defined]
        if not user or not self._verify_password(user.id, password):
            return None, None, {}
        if tenant_id and tenant_id != user.tenant_id:
            return None, None, {}
        mfa_cfg = self.store.get_user_mfa_secret(user.id) if self.mfa_enabled and hasattr(self.store, "get_user_mfa_secret") else None  # type: ignore[attr-defined]
        require_mfa = bool(self.mfa_enabled and mfa_cfg and mfa_cfg.enabled)
        session = self.store.create_session(user.id, mfa_required=require_mfa, tenant_id=user.tenant_id)
        tokens: dict[str, str] = {}
        if require_mfa and mfa_cfg:
            if mfa_code and self._verify_totp(mfa_cfg.secret, mfa_code):
                self._mark_session_verified(session.id)
                session.mfa_verified = True
                tokens = self._issue_tokens(user, session)
            else:
                session.mfa_verified = False
        else:
            tokens = self._issue_tokens(user, session)
        if self.cache and (not require_mfa or session.mfa_verified):
            await self.cache.cache_session(session.id, user.id, session.expires_at)
        return user, session, tokens

    async def refresh_tokens(self, refresh_token: str, tenant_hint: Optional[str] = None) -> tuple[Optional[User], Optional[Session], dict[str, str]]:
        payload = self._decode_jwt(refresh_token)
        if not payload or payload.get("token_type") != "refresh":
            return None, None, {}
        jti = payload.get("jti")
        if not jti or await self._is_refresh_revoked(jti):
            return None, None, {}
        session_id = payload.get("sid")
        session = self.store.get_session(session_id) if session_id else None  # type: ignore[attr-defined]
        if not session or session.expires_at <= datetime.utcnow():
            return None, None, {}
        user = self.store.get_user(session.user_id) if hasattr(self.store, "get_user") else None  # type: ignore[attr-defined]
        if not user:
            return None, None, {}
        if payload.get("sub") != user.id or payload.get("tenant_id") != user.tenant_id:
            return None, None, {}
        if tenant_hint and tenant_hint != user.tenant_id:
            return None, None, {}
        if not self._refresh_token_matches(session, jti):
            return None, None, {}
        tokens = self._issue_tokens(user, session)
        await self._revoke_refresh_token(jti, payload.get("exp"))
        return user, session, tokens

    async def revoke(self, session_id: str) -> None:
        sess = self.store.get_session(session_id) if hasattr(self.store, "get_session") else None  # type: ignore[attr-defined]
        if sess:
            refresh_jti = (sess.meta or {}).get("refresh_jti") if isinstance(sess.meta, dict) else None
            if refresh_jti:
                await self._revoke_refresh_token(refresh_jti, (sess.meta or {}).get("refresh_exp"))
        self.store.revoke_session(session_id)
        if self.cache:
            await self.cache.revoke_session(session_id)

    async def resolve_session(
        self,
        session_id: Optional[str],
        *,
        allow_pending_mfa: bool = False,
        tenant_hint: Optional[str] = None,
        required_role: Optional[str] = None,
    ) -> Optional[AuthContext]:
        if not session_id:
            return None
        sess = None
        if hasattr(self.store, "get_session"):
            sess = self.store.get_session(session_id)  # type: ignore[attr-defined]
        if not sess and self.cache:
            cached_user = await self.cache.get_session_user(session_id)
            if cached_user and hasattr(self.store, "get_session"):
                sess = self.store.get_session(session_id)  # type: ignore[attr-defined]
        if not sess or sess.expires_at <= datetime.utcnow():
            return None
        user = self.store.get_user(sess.user_id) if hasattr(self.store, "get_user") else None  # type: ignore[attr-defined]
        if not user:
            return None
        if tenant_hint and tenant_hint != user.tenant_id:
            return None
        if required_role and not self._role_allows(user.role, required_role):
            return None
        if self.cache and (not sess.mfa_required or sess.mfa_verified or not self.mfa_enabled):
            await self.cache.cache_session(sess.id, sess.user_id, sess.expires_at)
        if sess.mfa_required and not sess.mfa_verified and self.mfa_enabled and not allow_pending_mfa:
            return None
        return AuthContext(user_id=user.id, role=user.role, tenant_id=user.tenant_id, session_id=sess.id)

    async def authenticate(
        self,
        authorization: Optional[str],
        session_id: Optional[str],
        *,
        allow_pending_mfa: bool = False,
        tenant_hint: Optional[str] = None,
        required_role: Optional[str] = None,
    ) -> Optional[AuthContext]:
        token = self._extract_bearer(authorization)
        if token:
            token_ctx = self._authenticate_access_token(
                token,
                allow_pending_mfa=allow_pending_mfa,
                tenant_hint=tenant_hint,
                required_role=required_role,
            )
            if token_ctx:
                return token_ctx
        return await self.resolve_session(
            session_id,
            allow_pending_mfa=allow_pending_mfa,
            tenant_hint=tenant_hint,
            required_role=required_role,
        )

    def issue_tokens_for_session(self, session_id: str) -> tuple[Optional[User], Optional[Session], dict[str, str]]:
        sess = self.store.get_session(session_id) if hasattr(self.store, "get_session") else None  # type: ignore[attr-defined]
        if not sess or sess.expires_at <= datetime.utcnow():
            return None, None, {}
        if sess.mfa_required and self.mfa_enabled and not sess.mfa_verified:
            return None, None, {}
        user = self.store.get_user(sess.user_id) if hasattr(self.store, "get_user") else None  # type: ignore[attr-defined]
        if not user:
            return None, None, {}
        tokens = self._issue_tokens(user, sess)
        return user, sess, tokens

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

    def _encode_segment(self, data: bytes) -> str:
        return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")

    def _decode_segment(self, segment: str) -> bytes:
        padding = "=" * ((4 - len(segment) % 4) % 4)
        return base64.urlsafe_b64decode(segment + padding)

    def _encode_jwt(self, payload: dict[str, Any]) -> str:
        header = {"alg": "HS256", "typ": "JWT"}
        header_enc = self._encode_segment(json.dumps(header, separators=(",", ":")).encode())
        payload_enc = self._encode_segment(json.dumps(payload, separators=(",", ":")).encode())
        signing_input = f"{header_enc}.{payload_enc}"
        signature = hmac.new(self.settings.jwt_secret.encode(), signing_input.encode(), hashlib.sha256).digest()
        return f"{signing_input}.{self._encode_segment(signature)}"

    def _decode_jwt(self, token: str) -> Optional[dict[str, Any]]:
        try:
            header_b64, payload_b64, sig_b64 = token.split(".")
        except ValueError:
            return None
        signing_input = f"{header_b64}.{payload_b64}"
        expected_sig = self._encode_segment(
            hmac.new(self.settings.jwt_secret.encode(), signing_input.encode(), hashlib.sha256).digest()
        )
        if not hmac.compare_digest(expected_sig, sig_b64):
            return None
        try:
            payload = json.loads(self._decode_segment(payload_b64))
        except Exception:
            return None
        if payload.get("iss") != self.settings.jwt_issuer:
            return None
        aud = payload.get("aud")
        valid_aud = False
        if isinstance(aud, str):
            valid_aud = aud == self.settings.jwt_audience
        elif isinstance(aud, list):
            valid_aud = self.settings.jwt_audience in aud
        if not valid_aud:
            return None
        exp = payload.get("exp")
        if exp and datetime.utcfromtimestamp(int(exp)) <= datetime.utcnow():
            return None
        return payload

    def _issue_tokens(self, user: User, session: Session) -> dict[str, str]:
        now = datetime.utcnow()
        access_exp = int((now + timedelta(minutes=self.settings.access_token_ttl_minutes)).timestamp())
        refresh_exp = int((now + timedelta(minutes=self.settings.refresh_token_ttl_minutes)).timestamp())
        refresh_jti = str(uuid.uuid4())
        access_payload = {
            "iss": self.settings.jwt_issuer,
            "aud": self.settings.jwt_audience,
            "sub": user.id,
            "sid": session.id,
            "tenant_id": user.tenant_id,
            "role": user.role,
            "token_type": "access",
            "exp": access_exp,
        }
        refresh_payload = {
            "iss": self.settings.jwt_issuer,
            "aud": self.settings.jwt_audience,
            "sub": user.id,
            "sid": session.id,
            "tenant_id": user.tenant_id,
            "role": user.role,
            "token_type": "refresh",
            "jti": refresh_jti,
            "exp": refresh_exp,
        }
        access_token = self._encode_jwt(access_payload)
        refresh_token = self._encode_jwt(refresh_payload)
        self._persist_session_meta(session, refresh_jti, refresh_exp)
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_at": datetime.utcfromtimestamp(access_exp).isoformat(),
        }

    def _persist_session_meta(self, session: Session, refresh_jti: str, refresh_exp: int) -> None:
        meta = dict(session.meta or {})
        meta.update({"refresh_jti": refresh_jti, "refresh_exp": refresh_exp})
        session.meta = meta
        if hasattr(self.store, "set_session_meta"):
            self.store.set_session_meta(session.id, meta)  # type: ignore[attr-defined]

    def _refresh_token_matches(self, session: Session, jti: str) -> bool:
        meta = session.meta or {}
        if not isinstance(meta, dict):
            return False
        exp_raw = meta.get("refresh_exp")
        exp_ts: Optional[int] = None
        if isinstance(exp_raw, str) and exp_raw.isdigit():
            exp_ts = int(exp_raw)
        elif isinstance(exp_raw, (int, float)):
            exp_ts = int(exp_raw)
        if exp_ts and datetime.utcfromtimestamp(exp_ts) <= datetime.utcnow():
            return False
        return meta.get("refresh_jti") == jti

    async def _revoke_refresh_token(self, jti: str, exp: Any = None) -> None:
        self.revoked_refresh_tokens.add(jti)
        ttl = None
        if isinstance(exp, (int, float)):
            ttl = max(int(exp - datetime.utcnow().timestamp()), 0)
        if self.cache and ttl is not None:
            try:
                await self.cache.mark_refresh_revoked(jti, ttl)
            except Exception:
                return

    async def _is_refresh_revoked(self, jti: str) -> bool:
        if jti in self.revoked_refresh_tokens:
            return True
        if self.cache:
            try:
                return await self.cache.is_refresh_revoked(jti)
            except Exception:
                return False
        return False

    def _role_allows(self, role: str, required: str) -> bool:
        if role == required:
            return True
        if role == "admin" and required in {"admin", "user"}:
            return True
        return False

    def _extract_bearer(self, header: Optional[str]) -> Optional[str]:
        if not header:
            return None
        lower = header.lower()
        if not lower.startswith("bearer "):
            return None
        return header.split(" ", 1)[1]

    def _authenticate_access_token(
        self,
        token: str,
        *,
        allow_pending_mfa: bool,
        tenant_hint: Optional[str],
        required_role: Optional[str],
    ) -> Optional[AuthContext]:
        payload = self._decode_jwt(token)
        if not payload or payload.get("token_type") != "access":
            return None
        session_id = payload.get("sid")
        sess = self.store.get_session(session_id) if session_id and hasattr(self.store, "get_session") else None  # type: ignore[attr-defined]
        if not sess or sess.expires_at <= datetime.utcnow():
            return None
        user = self.store.get_user(payload.get("sub")) if hasattr(self.store, "get_user") else None  # type: ignore[attr-defined]
        if not user:
            return None
        if payload.get("tenant_id") != user.tenant_id or payload.get("role") != user.role:
            return None
        if tenant_hint and tenant_hint != user.tenant_id:
            return None
        if required_role and not self._role_allows(user.role, required_role):
            return None
        if sess.mfa_required and not sess.mfa_verified and self.mfa_enabled and not allow_pending_mfa:
            return None
        return AuthContext(user_id=user.id, role=user.role, tenant_id=user.tenant_id, session_id=sess.id)
