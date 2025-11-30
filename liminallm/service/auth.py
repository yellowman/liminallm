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
from typing import Any, List, Optional, Protocol, Tuple

import httpx
from argon2 import PasswordHasher, Type
from argon2.exceptions import InvalidHash, VerifyMismatchError

from liminallm.config import Settings
from liminallm.logging import get_logger
from liminallm.storage.models import Session, User, UserMFAConfig
from liminallm.storage.redis_cache import RedisCache

# OAuth provider configurations
OAUTH_PROVIDERS = {
    "google": {
        "auth_url": "https://accounts.google.com/o/oauth2/v2/auth",
        "token_url": "https://oauth2.googleapis.com/token",
        "userinfo_url": "https://www.googleapis.com/oauth2/v2/userinfo",
        "scope": "openid email profile",
    },
    "github": {
        "auth_url": "https://github.com/login/oauth/authorize",
        "token_url": "https://github.com/login/oauth/access_token",
        "userinfo_url": "https://api.github.com/user",
        "scope": "read:user user:email",
    },
    "microsoft": {
        "auth_url": "https://login.microsoftonline.com/common/oauth2/v2.0/authorize",
        "token_url": "https://login.microsoftonline.com/common/oauth2/v2.0/token",
        "userinfo_url": "https://graph.microsoft.com/v1.0/me",
        "scope": "openid email profile User.Read",
    },
}

logger = get_logger(__name__)


class AuthStore(Protocol):
    def create_user(
        self,
        email: str,
        handle: Optional[str] = None,
        *,
        tenant_id: str,
        role: str = "user",
        plan_tier: str = "free",
        is_active: bool = True,
        meta: Optional[dict] = None,
    ) -> User: ...

    def save_password(
        self, user_id: str, password_hash: str, password_algo: str
    ) -> None: ...

    def get_password_record(self, user_id: str) -> Optional[tuple[str, str]]: ...

    def set_user_mfa_secret(
        self, user_id: str, secret: str, enabled: bool = False
    ) -> UserMFAConfig: ...

    def get_user_mfa_secret(self, user_id: str) -> Optional[UserMFAConfig]: ...

    def create_session(
        self,
        user_id: str,
        ttl_minutes: int = 60 * 24,
        user_agent: str | None = None,
        ip_addr: str | None = None,
        *,
        mfa_required: bool = False,
        tenant_id: str = "public",
        meta: Optional[dict] = None,
    ) -> Session: ...

    def get_session(self, session_id: str) -> Optional[Session]: ...

    def set_session_meta(self, session_id: str, meta: dict) -> None: ...

    def revoke_session(self, session_id: str) -> None: ...

    def mark_session_verified(self, session_id: str) -> None: ...

    def get_user_by_email(self, email: str) -> Optional[User]: ...

    def get_user(self, user_id: str) -> Optional[User]: ...

    def update_user_role(self, user_id: str, role: str) -> Optional[User]: ...

    def delete_user(self, user_id: str) -> bool: ...

    def list_users(
        self, tenant_id: Optional[str] = None, limit: int = 100
    ) -> List[User]: ...


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
        store: AuthStore,
        cache: Optional[RedisCache],
        settings: Settings,
        *,
        mfa_enabled: bool = True,
    ) -> None:
        self.store: AuthStore = store
        self.cache = cache
        self.settings = settings
        self._mfa_challenges: dict[str, tuple[str, datetime]] = {}
        self.mfa_enabled = mfa_enabled
        self.revoked_refresh_tokens: set[str] = set()
        self._oauth_states: dict[str, tuple[str, datetime, Optional[str]]] = {}
        self._oauth_code_registry: dict[tuple[str, str], dict] = {}
        self._email_verification_tokens: dict[str, tuple[str, datetime]] = {}
        self._pwd_hasher = PasswordHasher(type=Type.ID)
        self.logger = logger

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
        self.store.save_password(user.id, pwd_hash, algo)
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
        self.store.save_password(user.id, pwd_hash, algo)
        return user, pwd

    def _get_oauth_credentials(self, provider: str) -> tuple[Optional[str], Optional[str]]:
        """Get OAuth client credentials for a provider."""
        if provider == "google":
            return self.settings.oauth_google_client_id, self.settings.oauth_google_client_secret
        elif provider == "github":
            return self.settings.oauth_github_client_id, self.settings.oauth_github_client_secret
        elif provider == "microsoft":
            return self.settings.oauth_microsoft_client_id, self.settings.oauth_microsoft_client_secret
        return None, None

    async def start_oauth(
        self,
        provider: str,
        redirect_uri: Optional[str] = None,
        *,
        tenant_id: Optional[str] = None,
    ) -> dict:
        if provider not in OAUTH_PROVIDERS:
            raise ValueError(f"Unsupported OAuth provider: {provider}")

        client_id, client_secret = self._get_oauth_credentials(provider)
        if not client_id:
            self.logger.warning("oauth_not_configured", provider=provider)
            raise ValueError(f"OAuth provider {provider} is not configured")

        state = uuid.uuid4().hex
        expires_at = datetime.utcnow() + timedelta(minutes=10)
        self._oauth_states[state] = (provider, expires_at, tenant_id)
        if self.cache:
            await self.cache.set_oauth_state(state, provider, expires_at, tenant_id)

        # Use configured redirect URI or fall back to settings
        callback_uri = redirect_uri or self.settings.oauth_redirect_uri
        if not callback_uri:
            self.logger.error("oauth_no_redirect_uri_configured", provider=provider)
            raise ValueError("No OAuth redirect URI configured")

        # Build proper OAuth authorization URL
        provider_config = OAUTH_PROVIDERS[provider]
        params = {
            "client_id": client_id,
            "redirect_uri": callback_uri,
            "response_type": "code",
            "scope": provider_config["scope"],
            "state": state,
        }
        # Add provider-specific parameters
        if provider == "google":
            params["access_type"] = "offline"
            params["prompt"] = "consent"

        from urllib.parse import urlencode
        query_string = urlencode(params)
        authorization_url = f"{provider_config['auth_url']}?{query_string}"

        return {
            "authorization_url": authorization_url,
            "state": state,
            "provider": provider,
        }

    def register_oauth_code(self, provider: str, code: str, payload: dict) -> None:
        """Record an exchanged OAuth payload for testing or offline flows."""

        self._oauth_code_registry[(provider, code)] = payload

    async def _exchange_oauth_code(self, provider: str, code: str) -> Optional[dict]:
        """Exchange OAuth authorization code for user identity.

        First checks cache/registry (for testing), then calls real OAuth providers.
        """
        # Check cache first (for testing or pre-registered codes)
        cached_payload: Optional[dict] = None
        if self.cache:
            raw = await self.cache.client.get(f"auth:oauth:code:{provider}:{code}")
            if raw:
                try:
                    cached_payload = json.loads(raw)
                except Exception as exc:
                    self.logger.warning("oauth_code_parse_failed", error=str(exc))
                else:
                    await self.cache.client.delete(f"auth:oauth:code:{provider}:{code}")
        if not cached_payload:
            cached_payload = self._oauth_code_registry.pop((provider, code), None)
        if cached_payload:
            return cached_payload

        # No cached payload - exchange code with real OAuth provider
        if provider not in OAUTH_PROVIDERS:
            self.logger.error("oauth_unknown_provider", provider=provider)
            return None

        client_id, client_secret = self._get_oauth_credentials(provider)
        if not client_id or not client_secret:
            self.logger.error("oauth_credentials_missing", provider=provider)
            return None

        redirect_uri = self.settings.oauth_redirect_uri
        if not redirect_uri:
            self.logger.error("oauth_redirect_uri_missing", provider=provider)
            return None

        provider_config = OAUTH_PROVIDERS[provider]

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Exchange code for access token
                token_data = {
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "code": code,
                    "redirect_uri": redirect_uri,
                    "grant_type": "authorization_code",
                }

                headers = {"Accept": "application/json"}
                token_response = await client.post(
                    provider_config["token_url"],
                    data=token_data,
                    headers=headers,
                )
                token_response.raise_for_status()
                token_result = token_response.json()

                access_token = token_result.get("access_token")
                if not access_token:
                    self.logger.error("oauth_no_access_token", provider=provider)
                    return None

                # Fetch user info
                userinfo_headers = {"Authorization": f"Bearer {access_token}"}
                # GitHub requires a special header
                if provider == "github":
                    userinfo_headers["Accept"] = "application/vnd.github+json"

                userinfo_response = await client.get(
                    provider_config["userinfo_url"],
                    headers=userinfo_headers,
                )
                userinfo_response.raise_for_status()
                userinfo = userinfo_response.json()

                # Extract user identity based on provider
                identity = self._parse_oauth_userinfo(provider, userinfo)

                # For GitHub, we may need to fetch email separately
                if provider == "github" and not identity.get("email"):
                    emails_response = await client.get(
                        "https://api.github.com/user/emails",
                        headers=userinfo_headers,
                    )
                    if emails_response.status_code == 200:
                        emails = emails_response.json()
                        primary_email = next(
                            (e["email"] for e in emails if e.get("primary") and e.get("verified")),
                            None,
                        )
                        if primary_email:
                            identity["email"] = primary_email

                self.logger.info(
                    "oauth_exchange_success",
                    provider=provider,
                    provider_uid=identity.get("provider_uid"),
                )
                return identity

        except httpx.HTTPStatusError as e:
            self.logger.error(
                "oauth_exchange_http_error",
                provider=provider,
                status_code=e.response.status_code,
                error=str(e),
            )
            return None
        except Exception as e:
            self.logger.error("oauth_exchange_error", provider=provider, error=str(e))
            return None

    def _parse_oauth_userinfo(self, provider: str, userinfo: dict) -> dict:
        """Parse user info from OAuth provider into standardized format."""
        if provider == "google":
            return {
                "provider_uid": userinfo.get("id"),
                "email": userinfo.get("email"),
                "handle": userinfo.get("name") or userinfo.get("email", "").split("@")[0],
                "name": userinfo.get("name"),
                "picture": userinfo.get("picture"),
            }
        elif provider == "github":
            return {
                "provider_uid": str(userinfo.get("id")),
                "email": userinfo.get("email"),
                "handle": userinfo.get("login"),
                "name": userinfo.get("name"),
                "picture": userinfo.get("avatar_url"),
            }
        elif provider == "microsoft":
            return {
                "provider_uid": userinfo.get("id"),
                "email": userinfo.get("mail") or userinfo.get("userPrincipalName"),
                "handle": userinfo.get("displayName") or userinfo.get("userPrincipalName", "").split("@")[0],
                "name": userinfo.get("displayName"),
                "picture": None,  # Microsoft requires a separate Graph API call for photos
            }
        return {"provider_uid": userinfo.get("id") or userinfo.get("sub")}

    async def complete_oauth(
        self, provider: str, code: str, state: str, *, tenant_id: Optional[str] = None
    ) -> tuple[Optional[User], Optional[Session], dict[str, str]]:
        cache_state_used = False
        stored = None
        if self.cache:
            try:
                stored = await self.cache.pop_oauth_state(state)
                cache_state_used = stored is not None
            except Exception as exc:
                self.logger.warning("pop_oauth_state_failed", error=str(exc))
        if stored is None:
            stored = self._oauth_states.pop(state, None)
        else:
            self._oauth_states.pop(state, None)
        now = datetime.utcnow()

        async def _clear_oauth_state() -> None:
            self._oauth_states.pop(state, None)
            if self.cache and not cache_state_used:
                await self.cache.pop_oauth_state(state)

        if not stored or stored[1] < now or stored[0] != provider:
            await _clear_oauth_state()
            return None, None, {}
        _, _, tenant_hint = stored
        if tenant_id and tenant_hint and tenant_id != tenant_hint:
            await _clear_oauth_state()
            return None, None, {}
        identity = await self._exchange_oauth_code(provider, code)
        if not identity:
            await _clear_oauth_state()
            return None, None, {}
        await _clear_oauth_state()
        provider_uid = identity.get("provider_uid")
        if not provider_uid:
            return None, None, {}
        normalized_tenant = tenant_id or tenant_hint or self.settings.default_tenant_id
        existing = None
        if hasattr(self.store, "get_user_by_provider"):
            existing = self.store.get_user_by_provider(provider, provider_uid)
        email = identity.get("email")
        user = existing or (self.store.get_user_by_email(email) if email else None)
        if not user:
            user_email = identity.get("email") or f"{provider_uid}@{provider}.oauth"
            handle = identity.get("handle") or provider_uid
            user = self.store.create_user(
                email=user_email, handle=handle, tenant_id=normalized_tenant
            )
            # Store an unusable password marker so password-based auth cannot succeed for OAuth users
            unusable_secret = base64.urlsafe_b64encode(os.urandom(24)).decode()
            self.store.save_password(user.id, unusable_secret, "oauth")
        if hasattr(self.store, "link_user_auth_provider"):
            try:
                self.store.link_user_auth_provider(user.id, provider, provider_uid)
            except Exception as exc:
                self.logger.error("link_oauth_provider_failed", error=str(exc))
                raise
        session = self.store.create_session(user.id, tenant_id=user.tenant_id)
        tokens = self._issue_tokens(user, session)
        if self.cache:
            await self.cache.cache_session(session.id, user.id, session.expires_at)
        return user, session, tokens

    def list_users(
        self, tenant_id: Optional[str] = None, limit: int = 100
    ) -> list[User]:
        return self.store.list_users(tenant_id=tenant_id, limit=limit)

    def set_user_role(self, user_id: str, role: str) -> Optional[User]:
        return self.store.update_user_role(user_id, role)

    def delete_user(self, user_id: str) -> bool:
        return bool(self.store.delete_user(user_id))

    async def login(
        self,
        email: str,
        password: str,
        mfa_code: Optional[str] = None,
        *,
        tenant_id: Optional[str] = None,
    ) -> tuple[Optional[User], Optional[Session], dict[str, str]]:
        user = self.store.get_user_by_email(email)
        if not user or not self.verify_password(user.id, password):
            return None, None, {}
        if tenant_id and tenant_id != user.tenant_id:
            return None, None, {}
        mfa_cfg = self.store.get_user_mfa_secret(user.id) if self.mfa_enabled else None
        require_mfa = bool(self.mfa_enabled and mfa_cfg and mfa_cfg.enabled)
        session = self.store.create_session(
            user.id, mfa_required=require_mfa, tenant_id=user.tenant_id
        )
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

    async def refresh_tokens(
        self, refresh_token: str, tenant_hint: Optional[str] = None
    ) -> tuple[Optional[User], Optional[Session], dict[str, str]]:
        payload = self._decode_jwt(refresh_token)
        if not payload or payload.get("token_type") != "refresh":
            return None, None, {}
        jti = payload.get("jti")
        if not jti or await self._is_refresh_revoked(jti):
            return None, None, {}
        session_id = payload.get("sid")
        session = self.store.get_session(session_id) if session_id else None
        if not session or session.expires_at <= datetime.utcnow():
            return None, None, {}
        user = self.store.get_user(session.user_id)
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
        sess = self.store.get_session(session_id)
        if sess:
            refresh_jti = (
                (sess.meta or {}).get("refresh_jti")
                if isinstance(sess.meta, dict)
                else None
            )
            if refresh_jti:
                await self._revoke_refresh_token(
                    refresh_jti, (sess.meta or {}).get("refresh_exp")
                )
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
        sess = self.store.get_session(session_id)
        if not sess and self.cache:
            cached_user = await self.cache.get_session_user(session_id)
            if not cached_user:
                return None
            sess = self.store.get_session(session_id)
        if not sess:
            return None
        if sess.expires_at <= datetime.utcnow():
            return None
        user = self.store.get_user(sess.user_id)
        if not user:
            return None
        if tenant_hint and tenant_hint != user.tenant_id:
            return None
        if required_role and not self._role_allows(user.role, required_role):
            return None
        if self.cache and (
            not sess.mfa_required or sess.mfa_verified or not self.mfa_enabled
        ):
            await self.cache.cache_session(sess.id, sess.user_id, sess.expires_at)
        if (
            sess.mfa_required
            and not sess.mfa_verified
            and self.mfa_enabled
            and not allow_pending_mfa
        ):
            return None
        return AuthContext(
            user_id=user.id,
            role=user.role,
            tenant_id=user.tenant_id,
            session_id=sess.id,
        )

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

    def issue_tokens_for_session(
        self, session_id: str
    ) -> tuple[Optional[User], Optional[Session], dict[str, str]]:
        sess = self.store.get_session(session_id)
        if not sess or sess.expires_at <= datetime.utcnow():
            return None, None, {}
        if sess.mfa_required and self.mfa_enabled and not sess.mfa_verified:
            return None, None, {}
        user = self.store.get_user(sess.user_id)
        if not user:
            return None, None, {}
        tokens = self._issue_tokens(user, sess)
        return user, sess, tokens

    async def issue_mfa_challenge(self, user_id: str) -> dict:
        if not self.mfa_enabled:
            return {"status": "disabled"}
        existing = self.store.get_user_mfa_secret(user_id)
        secret = (
            existing.secret
            if existing
            else base64.b32encode(os.urandom(10)).decode("utf-8").rstrip("=")
        )
        self.store.set_user_mfa_secret(user_id, secret, enabled=False)
        uri = f"otpauth://totp/liminallm:{user_id}?secret={secret}&issuer=LiminalLM"
        return {"otpauth_uri": uri}

    async def verify_mfa_challenge(
        self, user_id: str, code: str, session_id: Optional[str] = None
    ) -> bool:
        if not self.mfa_enabled:
            return True
        cfg = self.store.get_user_mfa_secret(user_id)
        if not cfg:
            return False

        # Check MFA lockout (5 failed attempts = 5 minute lockout per SPEC §18)
        lockout_key = f"mfa:lockout:{user_id}"
        attempts_key = f"mfa:attempts:{user_id}"

        if self.cache:
            locked = await self.cache.client.get(lockout_key)
            if locked:
                self.logger.warning("mfa_locked_out", user_id=user_id)
                return False

        if not self._verify_totp(cfg.secret, code):
            # Track failed attempt
            if self.cache:
                attempts = await self.cache.client.incr(attempts_key)
                await self.cache.client.expire(attempts_key, 300)  # 5 minute window
                if attempts >= 5:
                    # Lock out for 5 minutes
                    await self.cache.client.set(lockout_key, "1", ex=300)
                    await self.cache.client.delete(attempts_key)
                    self.logger.warning("mfa_lockout_triggered", user_id=user_id, attempts=attempts)
            return False

        # Success - clear any failed attempts
        if self.cache:
            await self.cache.client.delete(attempts_key)

        self.store.set_user_mfa_secret(user_id, cfg.secret, enabled=True)
        if session_id:
            self._mark_session_verified(session_id)
        return True

    async def initiate_password_reset(self, email: str) -> str:
        # Use raw bytes for proper entropy (not string representation)
        token = hashlib.sha256(b"reset-" + email.encode() + os.urandom(32)).hexdigest()
        # Persist a short-lived reset token with TTL in Redis if available
        if self.cache:
            expires_at = datetime.utcnow() + timedelta(minutes=15)
            await self.cache.client.set(
                f"reset:{token}",
                email,
                ex=int((expires_at - datetime.utcnow()).total_seconds()),
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
        user = self.store.get_user_by_email(email)
        if not user:
            return False
        pwd_hash, algo = self._hash_password(new_password)
        self.store.save_password(user.id, pwd_hash, algo)
        if hasattr(self.store, "revoke_user_sessions"):
            try:
                self.store.revoke_user_sessions(user.id)  # type: ignore[attr-defined]
            except Exception as exc:
                self.logger.warning(
                    "revoke_sessions_failed", user_id=user.id, error=str(exc)
                )
        if self.cache:
            await self.cache.client.delete(f"reset:{token}")
        return True

    async def request_email_verification(self, user: User) -> str:
        # Use raw bytes for proper entropy (not string representation)
        token = hashlib.sha256(
            b"verify-" + user.email.encode() + os.urandom(32)
        ).hexdigest()
        expires_at = datetime.utcnow() + timedelta(hours=24)
        if self.cache:
            await self.cache.client.set(
                f"verify:{token}",
                user.id,
                ex=int((expires_at - datetime.utcnow()).total_seconds()),
            )
        else:
            self._email_verification_tokens[token] = (user.id, expires_at)
        return token

    async def complete_email_verification(self, token: str) -> bool:
        user_id = None
        if self.cache:
            user_id = await self.cache.client.get(f"verify:{token}")
        else:
            stored = self._email_verification_tokens.get(token)
            if stored:
                user_id, expires_at = stored
                if expires_at <= datetime.utcnow():
                    # Remove expired token to prevent memory leak
                    self._email_verification_tokens.pop(token, None)
                    user_id = None
                else:
                    self._email_verification_tokens.pop(token, None)
        if isinstance(user_id, bytes):
            user_id = user_id.decode()
        if not user_id:
            return False
        user = self.store.get_user(user_id)
        if not user:
            return False
        if hasattr(self.store, "mark_email_verified"):
            self.store.mark_email_verified(user.id)
        if self.cache:
            await self.cache.client.delete(f"verify:{token}")
        else:
            self._email_verification_tokens.pop(token, None)
        return True

    def _hash_password(self, password: str) -> Tuple[str, str]:
        algo = "argon2id"
        digest = self._pwd_hasher.hash(password)
        return digest, algo

    def verify_password(self, user_id: str, password: str) -> bool:
        """Verify a user's password against stored hash."""
        record = self.store.get_password_record(user_id)
        if not record:
            return False
        stored_hash, algo = record
        if algo != "argon2id":
            return False
        try:
            return self._pwd_hasher.verify(stored_hash, password)
        except (InvalidHash, VerifyMismatchError):
            return False

    def save_password(self, user_id: str, password: str) -> None:
        """Hash and save a new password for a user."""
        pwd_hash, algo = self._hash_password(password)
        self.store.save_password(user_id, pwd_hash, algo)

    def _verify_totp(
        self, secret: str, code: str, *, window: int = 1, interval: int = 30
    ) -> bool:
        for offset in range(-window, window + 1):
            generated = self._generate_totp(
                secret, time.time() + offset * interval, interval=interval
            )
            # SECURITY: Use constant-time comparison to prevent timing attacks
            if generated and hmac.compare_digest(generated, code):
                return True
        return False

    def _generate_totp(
        self, secret: str, timestamp: float, *, interval: int = 30, digits: int = 6
    ) -> str:
        padded = secret + "=" * ((8 - len(secret) % 8) % 8)
        try:
            key = base64.b32decode(padded, True)
        except Exception:
            self.logger.warning("totp_secret_invalid")
            return ""
        counter = int(timestamp // interval).to_bytes(8, "big")
        # TOTP standard uses SHA1 - do not change without migration
        digest = hmac.new(key, counter, hashlib.sha1).digest()
        offset = digest[-1] & 0x0F
        code_int = (int.from_bytes(digest[offset : offset + 4], "big") & 0x7FFFFFFF) % (
            10**digits
        )
        return str(code_int).zfill(digits)

    def _mark_session_verified(self, session_id: str) -> None:
        self.store.mark_session_verified(session_id)

    def _encode_segment(self, data: bytes) -> str:
        return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")

    def _decode_segment(self, segment: str) -> bytes:
        padding = "=" * ((4 - len(segment) % 4) % 4)
        return base64.urlsafe_b64decode(segment + padding)

    def _encode_jwt(self, payload: dict[str, Any]) -> str:
        header = {"alg": "HS256", "typ": "JWT"}
        header_enc = self._encode_segment(
            json.dumps(header, separators=(",", ":")).encode()
        )
        payload_enc = self._encode_segment(
            json.dumps(payload, separators=(",", ":")).encode()
        )
        signing_input = f"{header_enc}.{payload_enc}"
        signature = hmac.new(
            self.settings.jwt_secret.encode(), signing_input.encode(), hashlib.sha256
        ).digest()
        return f"{signing_input}.{self._encode_segment(signature)}"

    def _decode_jwt(self, token: str) -> Optional[dict[str, Any]]:
        try:
            header_b64, payload_b64, sig_b64 = token.split(".")
        except ValueError:
            return None
        signing_input = f"{header_b64}.{payload_b64}"
        expected_sig = self._encode_segment(
            hmac.new(
                self.settings.jwt_secret.encode(),
                signing_input.encode(),
                hashlib.sha256,
            ).digest()
        )
        if not hmac.compare_digest(expected_sig, sig_b64):
            return None
        try:
            payload = json.loads(self._decode_segment(payload_b64))
        except Exception as exc:
            logger.warning("jwt_payload_decode_failed", error=str(exc))
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
        if not exp:
            return None
        try:
            exp_ts = float(exp)
        except (TypeError, ValueError):
            return None
        if exp_ts <= time.time():
            return None
        return payload

    def _issue_tokens(self, user: User, session: Session) -> dict[str, str]:
        now = datetime.utcnow()
        access_exp = int(
            (
                now + timedelta(minutes=self.settings.access_token_ttl_minutes)
            ).timestamp()
        )
        refresh_exp = int(
            (
                now + timedelta(minutes=self.settings.refresh_token_ttl_minutes)
            ).timestamp()
        )
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

    def _persist_session_meta(
        self, session: Session, refresh_jti: str, refresh_exp: int
    ) -> None:
        meta = dict(session.meta or {})
        meta.update({"refresh_jti": refresh_jti, "refresh_exp": refresh_exp})
        session.meta = meta
        self.store.set_session_meta(session.id, meta)

    def _refresh_token_matches(self, session: Session, jti: str) -> bool:
        meta = session.meta or {}
        if not isinstance(meta, dict):
            return False
        exp_raw = meta.get("refresh_exp")
        exp_ts: Optional[float] = None
        if isinstance(exp_raw, str) and exp_raw.isdigit():
            exp_ts = float(exp_raw)
        elif isinstance(exp_raw, (int, float)):
            exp_ts = float(exp_raw)
        if exp_ts is not None and exp_ts <= time.time():
            return False
        return meta.get("refresh_jti") == jti

    async def _revoke_refresh_token(self, jti: str, exp: Any = None) -> None:
        self.revoked_refresh_tokens.add(jti)
        ttl = None
        if isinstance(exp, (int, float)):
            ttl = max(int(exp - datetime.utcnow().timestamp()), 0)
        if ttl is None:
            ttl = max(int(self.settings.refresh_token_ttl_minutes * 60), 0)
        if self.cache:
            try:
                await self.cache.mark_refresh_revoked(jti, ttl)
            except Exception as exc:
                logger.warning(
                    "cache_revoked_refresh_token_failed", jti=jti, error=str(exc)
                )
                return

    async def _is_refresh_revoked(self, jti: str) -> bool:
        if jti in self.revoked_refresh_tokens:
            return True
        if self.cache:
            try:
                return await self.cache.is_refresh_revoked(jti)
            except Exception as exc:
                # SECURITY: Default to revoked when cache is unavailable to prevent
                # accepting potentially revoked tokens during Redis outages.
                # This forces re-authentication rather than risking privilege escalation.
                logger.warning(
                    "check_revoked_refresh_token_failed_defaulting_to_revoked",
                    jti=jti,
                    error=str(exc),
                )
                return True
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
        sess = self.store.get_session(session_id) if session_id else None
        if not sess or sess.expires_at <= datetime.utcnow():
            return None
        user = self.store.get_user(payload.get("sub"))
        if not user:
            return None
        if (
            payload.get("tenant_id") != user.tenant_id
            or payload.get("role") != user.role
        ):
            return None
        if tenant_hint and tenant_hint != user.tenant_id:
            return None
        if required_role and not self._role_allows(user.role, required_role):
            return None
        if (
            sess.mfa_required
            and not sess.mfa_verified
            and self.mfa_enabled
            and not allow_pending_mfa
        ):
            return None
        return AuthContext(
            user_id=user.id,
            role=user.role,
            tenant_id=user.tenant_id,
            session_id=sess.id,
        )
