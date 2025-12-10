from __future__ import annotations

import base64
import contextlib
import hashlib
import hmac
import json
import os
import secrets
import time
import uuid
from urllib.parse import urlparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
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
        # Issue 28.4: Thread-safe lock for mutable state dictionaries
        # Protects all in-memory fallback state from concurrent access
        import threading

        self._state_lock = threading.Lock()
        self._mfa_challenges: dict[str, tuple[str, datetime]] = {}
        # Issue 11.1: In-memory fallback for MFA lockout when Redis unavailable
        self._mfa_attempts: dict[str, tuple[int, datetime]] = {}  # user_id -> (count, window_start)
        self._mfa_lockouts: dict[str, datetime] = {}  # user_id -> locked_until
        self.mfa_enabled = mfa_enabled
        self.revoked_refresh_tokens: set[str] = set()
        self._oauth_states: dict[str, tuple[str, datetime, Optional[str]]] = {}
        self._oauth_code_registry: dict[tuple[str, str], dict] = {}
        self._email_verification_tokens: dict[str, tuple[str, datetime]] = {}
        # Issue 11.2: In-memory fallback for password reset tokens when Redis unavailable
        self._password_reset_tokens: dict[str, tuple[str, datetime]] = {}  # token -> (email, expires_at)
        self._pwd_hasher = PasswordHasher(type=Type.ID)
        self.logger = logger
        self._last_cleanup = datetime.now(timezone.utc)
        # Allowance for small clock skew across nodes (Issue 76.1/76.2)
        self._clock_skew_leeway = timedelta(seconds=120)

    def _now(self) -> datetime:
        """Timezone-aware UTC helper to avoid naive datetime usage."""

        return datetime.now(timezone.utc)

    @contextlib.contextmanager
    def _with_state_lock(self):
        """Context manager for thread-safe state dictionary access (Issue 28.4)."""

        with self._state_lock:
            yield

    def cleanup_expired_states(self) -> int:
        """Clean up expired OAuth states, MFA challenges, and email verification tokens.

        Should be called periodically (e.g., every few minutes) to prevent memory leaks.

        Returns:
            Number of expired entries cleaned up
        """
        now = self._now()
        cleaned = 0

        # Issue 28.4: Thread-safe access to mutable state dictionaries
        with self._state_lock:
            # Clean expired OAuth states
            expired_oauth = [
                state for state, (_, expires_at, _) in self._oauth_states.items()
                if expires_at <= now
            ]
            for state in expired_oauth:
                self._oauth_states.pop(state, None)
                cleaned += 1

            # Clean expired MFA challenges
            expired_mfa = [
                user_id for user_id, (_, expires_at) in self._mfa_challenges.items()
                if expires_at <= now
            ]
            for user_id in expired_mfa:
                self._mfa_challenges.pop(user_id, None)
                cleaned += 1

            # Clean expired email verification tokens
            expired_email = [
                token for token, (_, expires_at) in self._email_verification_tokens.items()
                if expires_at <= now
            ]
            for token in expired_email:
                self._email_verification_tokens.pop(token, None)
                cleaned += 1

            # Clean expired password reset tokens (Issue 11.2)
            expired_reset = [
                token for token, (_, expires_at) in self._password_reset_tokens.items()
                if expires_at <= now
            ]
            for token in expired_reset:
                self._password_reset_tokens.pop(token, None)
                cleaned += 1

            # Clean expired MFA lockouts (Issue 11.1)
            expired_lockouts = [
                user_id for user_id, locked_until in self._mfa_lockouts.items()
                if locked_until <= now
            ]
            for user_id in expired_lockouts:
                self._mfa_lockouts.pop(user_id, None)
                cleaned += 1

            # Clean expired MFA attempts (5-minute window)
            window_threshold = now - timedelta(minutes=5)
            expired_attempts = [
                user_id for user_id, (_, window_start) in self._mfa_attempts.items()
                if window_start <= window_threshold
            ]
            for user_id in expired_attempts:
                self._mfa_attempts.pop(user_id, None)
                cleaned += 1

        if cleaned > 0:
            self.logger.debug(
                "auth_state_cleanup", cleaned=cleaned, oauth=len(expired_oauth),
                mfa=len(expired_mfa), email=len(expired_email), reset=len(expired_reset),
                mfa_lockouts=len(expired_lockouts), mfa_attempts=len(expired_attempts)
            )

        self._last_cleanup = now
        return cleaned

    def maybe_cleanup(self, interval_minutes: int = 5) -> int:
        """Run cleanup if interval has elapsed since last cleanup.

        Args:
            interval_minutes: Minimum time between cleanups

        Returns:
            Number of entries cleaned, or 0 if cleanup was skipped
        """
        now = self._now()
        if (now - self._last_cleanup).total_seconds() >= interval_minutes * 60:
            return self.cleanup_expired_states()
        return 0

    def _generate_password(self) -> str:
        return base64.urlsafe_b64encode(os.urandom(12)).decode().rstrip("=")

    def _get_system_settings(self) -> dict:
        """Get admin-managed system settings from database.

        Returns merged settings with defaults for any missing values.
        """
        if hasattr(self.store, "get_system_settings"):
            db_settings = self.store.get_system_settings()
        else:
            db_settings = {}
        defaults = {
            "session_rotation_hours": 24,
            "session_rotation_grace_seconds": 300,
            "access_token_ttl_minutes": 30,
            "refresh_token_ttl_minutes": 1440,
            # Device-specific session TTLs (SPEC §18: 7d web, 1d mobile)
            "session_ttl_minutes_web": 60 * 24 * 7,
            "session_ttl_minutes_mobile": 60 * 24,
            # Device-specific refresh TTLs aligned with session windows
            "refresh_token_ttl_minutes_web": 60 * 24 * 7,
            "refresh_token_ttl_minutes_mobile": 60 * 24,
        }
        return {**defaults, **db_settings}

    def _get_session_ttl(self, device_type: str) -> int:
        sys_settings = self._get_system_settings()
        normalized = (device_type or "web").lower()
        if normalized == "mobile":
            return int(sys_settings.get("session_ttl_minutes_mobile", 60 * 24))
        return int(sys_settings.get("session_ttl_minutes_web", 60 * 24 * 7))

    def _get_refresh_ttl(self, device_type: str, sys_settings: dict[str, Any]) -> int:
        normalized = (device_type or "web").lower()
        if normalized == "mobile":
            return int(
                sys_settings.get("refresh_token_ttl_minutes_mobile", 60 * 24)
            )
        return int(
            sys_settings.get("refresh_token_ttl_minutes_web", 60 * 24 * 7)
        )

    def _get_session_device(self, session: Session) -> str:
        meta = session.meta or {}
        if isinstance(meta, dict):
            raw = meta.get("device_type")
            if isinstance(raw, str) and raw:
                normalized = raw.lower()
                if normalized in {"web", "mobile"}:
                    return normalized
        return "web"

    def _ensure_csrf_token(self, session: Session) -> str:
        meta = session.meta if isinstance(session.meta, dict) else {}
        token = None
        if isinstance(meta, dict):
            raw = meta.get("csrf_token")
            if isinstance(raw, str) and raw:
                token = raw
        if not token:
            token = secrets.token_urlsafe(32)
            meta = dict(meta or {})
            meta["csrf_token"] = token
            session.meta = meta
            self.store.set_session_meta(session.id, meta)
        return token

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
        session = self.store.create_session(
            user.id,
            tenant_id=user.tenant_id,
            ttl_minutes=self._get_session_ttl("web"),
            meta={"device_type": "web"},
        )
        tokens = self._issue_tokens(user, session, device_type="web")
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

    def _validate_redirect_uri(self, redirect_uri: str) -> str:
        parsed = urlparse(redirect_uri)
        if parsed.scheme not in {"https", "http"}:
            raise ValueError("OAuth redirect URI must be http(s)")
        if parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1"}:
            raise ValueError("Insecure redirect URI not allowed outside localhost")
        if not parsed.netloc:
            raise ValueError("OAuth redirect URI must include host")
        return redirect_uri

    async def start_oauth(
        self,
        provider: str,
        redirect_uri: Optional[str] = None,
        *,
        tenant_id: Optional[str] = None,
    ) -> dict:
        # Opportunistically clean up expired states to prevent memory leaks
        self.maybe_cleanup()

        if provider not in OAUTH_PROVIDERS:
            raise ValueError(f"Unsupported OAuth provider: {provider}")

        client_id, client_secret = self._get_oauth_credentials(provider)
        if not client_id:
            self.logger.warning("oauth_not_configured", provider=provider)
            raise ValueError(f"OAuth provider {provider} is not configured")

        state = uuid.uuid4().hex
        expires_at = self._now() + timedelta(minutes=10)
        # Issue 28.4: Thread-safe state mutation
        with self._state_lock:
            with self._with_state_lock():
                self._oauth_states[state] = (provider, expires_at, tenant_id)
        if self.cache:
            await self.cache.set_oauth_state(state, provider, expires_at, tenant_id)

        # Use configured redirect URI or fall back to settings
        callback_uri = redirect_uri or self.settings.oauth_redirect_uri
        if not callback_uri:
            self.logger.error("oauth_no_redirect_uri_configured", provider=provider)
            raise ValueError("No OAuth redirect URI configured")
        callback_uri = self._validate_redirect_uri(callback_uri)

        if not self.cache and not self.settings.test_mode:
            raise RuntimeError("OAuth state cache is required for multi-process safety")

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
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=False) as client:
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
                try:
                    token_result = token_response.json()
                except Exception as exc:
                    self.logger.error(
                        "oauth_token_parse_error", provider=provider, error=str(exc)
                    )
                    return None

                access_token = token_result.get("access_token") if isinstance(token_result, dict) else None
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
                try:
                    userinfo = userinfo_response.json()
                except Exception as exc:
                    self.logger.error(
                        "oauth_userinfo_parse_error",
                        provider=provider,
                        error=str(exc),
                    )
                    return None

                if not isinstance(userinfo, dict):
                    self.logger.error(
                        "oauth_userinfo_invalid_format", provider=provider, type=str(type(userinfo))
                    )
                    return None

                # Extract user identity based on provider
                identity = self._parse_oauth_userinfo(provider, userinfo)

                if not identity or not isinstance(identity, dict):
                    self.logger.error("oauth_identity_empty", provider=provider)
                    return None
                if not identity.get("provider_uid"):
                    self.logger.error("oauth_identity_missing_uid", provider=provider)
                    return None

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

                if not identity.get("email"):
                    self.logger.error("oauth_identity_missing_email", provider=provider)
                    return None

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
        self, provider: str, code: str, state: str
    ) -> tuple[Optional[User], Optional[Session], dict[str, str]]:
        """Complete OAuth authentication flow.

        SECURITY: tenant_id is derived exclusively from the validated OAuth state
        token, never from external parameters. This prevents tenant spoofing attacks
        per CLAUDE.md security guidelines.
        """
        cache_state_used = False
        stored = None
        if self.cache:
            try:
                stored = await self.cache.pop_oauth_state(state)
                cache_state_used = stored is not None
            except Exception as exc:
                # Issue 53.9: Fail closed if cache state cannot be retrieved to
                # avoid stale OAuth state reuse
                self.logger.error("pop_oauth_state_failed", error=str(exc))
                return None, None, {}
        # Issue 28.4: Thread-safe state mutation
        with self._state_lock:
            if stored is None:
                with self._with_state_lock():
                    stored = self._oauth_states.pop(state, None)
            else:
                with self._with_state_lock():
                    self._oauth_states.pop(state, None)
        now = self._now()

        async def _clear_oauth_state() -> None:
            with self._state_lock:
                with self._with_state_lock():
                    self._oauth_states.pop(state, None)
            if self.cache and not cache_state_used:
                await self.cache.pop_oauth_state(state)

        if not stored or stored[1] < now or stored[0] != provider:
            await _clear_oauth_state()
            return None, None, {}
        # SECURITY: tenant_id comes from validated state, not from user input
        _, _, state_tenant_id = stored
        identity = await self._exchange_oauth_code(provider, code)
        if not identity:
            await _clear_oauth_state()
            return None, None, {}
        await _clear_oauth_state()
        provider_uid = identity.get("provider_uid")
        if not provider_uid:
            return None, None, {}
        # Use tenant from state, falling back to default only if state had none
        normalized_tenant = state_tenant_id or self.settings.default_tenant_id
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
        session = self.store.create_session(
            user.id,
            tenant_id=user.tenant_id,
            ttl_minutes=self._get_session_ttl("web"),
            meta={"device_type": "web"},
        )
        tokens = self._issue_tokens(user, session, device_type="web")
        if self.cache:
            await self.cache.cache_session(session.id, user.id, session.expires_at)
        return user, session, tokens

    def list_users(
        self, tenant_id: Optional[str] = None, limit: int = 100
    ) -> list[User]:
        return self.store.list_users(tenant_id=tenant_id, limit=limit)

    async def set_user_role(self, user_id: str, role: str) -> Optional[User]:
        """Update user role and invalidate all existing sessions.

        Issue 22.1: When role changes, existing sessions must be invalidated
        to prevent users from retaining previous privilege levels.
        """
        user = self.store.update_user_role(user_id, role)
        if user:
            # Revoke all sessions for this user to enforce new role
            try:
                await self.revoke_all_user_sessions(user_id)
                self.logger.info(
                    "user_role_updated_sessions_revoked",
                    user_id=user_id,
                    new_role=role,
                )
            except Exception as exc:
                # Log but don't fail role update if session revocation fails
                self.logger.warning(
                    "user_role_session_revocation_failed",
                    user_id=user_id,
                    error=str(exc),
                )
        return user

    def delete_user(self, user_id: str) -> bool:
        return bool(self.store.delete_user(user_id))

    async def login(
        self,
        email: str,
        password: str,
        mfa_code: Optional[str] = None,
        *,
        tenant_id: Optional[str] = None,
        user_agent: Optional[str] = None,
        ip_addr: Optional[str] = None,
        device_type: str = "web",
    ) -> tuple[Optional[User], Optional[Session], dict[str, str]]:
        user = self.store.get_user_by_email(email)
        if not user or not self.verify_password(user.id, password):
            return None, None, {}
        if tenant_id and tenant_id != user.tenant_id:
            return None, None, {}

        # SPEC §18: Single-session mode - invalidate prior sessions if enabled
        user_meta = user.meta or {}
        if user_meta.get("single_session"):
            self.logger.info(
                "single_session_mode_active",
                user_id=user.id,
                action="revoking_prior_sessions",
            )
            await self.revoke_all_user_sessions(user.id)

        mfa_cfg = self.store.get_user_mfa_secret(user.id) if self.mfa_enabled else None
        require_mfa = bool(self.mfa_enabled and mfa_cfg and mfa_cfg.enabled)
        device = (device_type or "web").lower()
        session_ttl = self._get_session_ttl(device)
        session = self.store.create_session(
            user.id,
            mfa_required=require_mfa,
            tenant_id=user.tenant_id,
            user_agent=user_agent,
            ip_addr=ip_addr,
            ttl_minutes=session_ttl,
            meta={"device_type": device},
        )
        tokens: dict[str, str] = {}
        if require_mfa and mfa_cfg:
            if mfa_code and self._verify_totp(mfa_cfg.secret, mfa_code):
                self._mark_session_verified(session.id)
                session.mfa_verified = True
                tokens = self._issue_tokens(user, session, device_type=device)
            else:
                session.mfa_verified = False
        else:
            tokens = self._issue_tokens(user, session, device_type=device)
        if self.cache and (not require_mfa or session.mfa_verified):
            await self.cache.cache_session(session.id, user.id, session.expires_at)
            # Initialize session activity tracking
            await self.cache.update_session_activity(session.id)
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
        now = self._now()
        if not session or session.expires_at <= now - self._clock_skew_leeway:
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
        device = self._get_session_device(session)
        tokens = self._issue_tokens(user, session, device_type=device)
        await self._revoke_refresh_token(jti, payload.get("exp"))
        return user, session, tokens

    async def revoke(self, session_id: str) -> None:
        """Revoke a session and denylist associated tokens per SPEC §12.1."""
        sess = self.store.get_session(session_id)
        if sess and isinstance(sess.meta, dict):
            meta = sess.meta
            # Denylist refresh token
            refresh_jti = meta.get("refresh_jti")
            if refresh_jti:
                await self._revoke_refresh_token(refresh_jti, meta.get("refresh_exp"))
            # SPEC §12.1: "add JWT to short-lived denylist if JWTs used"
            # Denylist access token to prevent use after logout
            access_jti = meta.get("access_jti")
            access_exp = meta.get("access_exp")
            if access_jti and access_exp and self.cache:
                ttl = max(0, int(access_exp - time.time()))
                if ttl > 0:
                    try:
                        await self.cache.denylist_access_token(access_jti, ttl)
                    except Exception as exc:
                        # Log but don't fail - session revocation should still proceed
                        self.logger.warning(
                            "access_token_denylist_failed",
                            session_id=session_id,
                            error=str(exc),
                        )
        self.store.revoke_session(session_id)
        if self.cache:
            await self.cache.revoke_session(session_id)

    async def revoke_all_user_sessions(
        self, user_id: str, except_session_id: Optional[str] = None
    ) -> int:
        """Revoke all sessions for a user, optionally keeping one session active.

        Args:
            user_id: The user whose sessions to revoke
            except_session_id: Optional session ID to keep active (e.g., current session)

        Returns:
            Number of sessions revoked
        """
        revoked_count = 0
        if hasattr(self.store, "revoke_user_sessions"):
            # Use store method if available for better performance
            try:
                self.store.revoke_user_sessions(user_id, except_session_id)  # type: ignore[attr-defined]
                revoked_count = -1  # Unknown count when using bulk method
            except Exception as exc:
                self.logger.warning(
                    "revoke_user_sessions_failed", user_id=user_id, error=str(exc)
                )
        # Also clear from cache using proper method (Issue 22.3)
        if self.cache:
            try:
                await self.cache.revoke_user_sessions(user_id, except_session_id)
            except Exception as exc:
                self.logger.warning(
                    "revoke_user_sessions_cache_clear_failed",
                    user_id=user_id,
                    error=str(exc),
                )
        return revoked_count

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

        # SPEC §12.1: Check if this is a rotated session (grace period)
        actual_session_id = session_id
        if self.cache:
            rotated_to = await self.cache.get_rotated_session(session_id)
            if rotated_to:
                actual_session_id = rotated_to
                self.logger.debug(
                    "session_rotation_grace",
                    old_session=session_id,
                    new_session=rotated_to,
                )

        sess = self.store.get_session(actual_session_id)
        if not sess and self.cache:
            found, cached_user = await self.cache.get_session_user(actual_session_id)
            if not found or not cached_user:
                return None
            sess = self.store.get_session(actual_session_id)
        if not sess:
            return None
        now = self._now()
        if sess.expires_at <= now - self._clock_skew_leeway:
            return None
        user = self.store.get_user(sess.user_id)
        if not user:
            return None
        if tenant_hint and tenant_hint != user.tenant_id:
            return None
        if required_role and not self._role_allows(user.role, required_role):
            return None

        # SPEC §12.1: Session rotation after 24h of activity
        final_session = sess
        if self.cache:
            rotated_session = await self._maybe_rotate_session(sess, user)
            if rotated_session:
                final_session = rotated_session

        if self.cache and (
            not final_session.mfa_required or final_session.mfa_verified or not self.mfa_enabled
        ):
            await self.cache.cache_session(final_session.id, final_session.user_id, final_session.expires_at)
            # Update activity timestamp on each session use
            await self.cache.update_session_activity(final_session.id)

        if (
            final_session.mfa_required
            and not final_session.mfa_verified
            and self.mfa_enabled
            and not allow_pending_mfa
        ):
            return None
        return AuthContext(
            user_id=user.id,
            role=user.role,
            tenant_id=user.tenant_id,
            session_id=final_session.id,
        )

    async def _maybe_rotate_session(
        self, sess: Session, user: User
    ) -> Optional[Session]:
        """Rotate session if 24h of activity has passed (SPEC §12.1).

        Returns the new session if rotated, None otherwise.

        Bug fix: Uses Redis SETNX lock to prevent race condition where concurrent
        requests with the same session could both trigger rotation.
        """
        if not self.cache:
            return None

        last_activity = await self.cache.get_session_activity(sess.id)
        if not last_activity:
            # No activity record - initialize it
            await self.cache.update_session_activity(sess.id)
            return None

        sys_settings = self._get_system_settings()
        rotation_hours = sys_settings.get("session_rotation_hours", 24)
        rotation_threshold = timedelta(hours=rotation_hours)
        if self._now() - last_activity < rotation_threshold:
            return None

        # Bug fix: Acquire rotation lock to prevent duplicate rotations
        lock_key = f"session:rotation_lock:{sess.id}"
        acquired = await self.cache.client.set(lock_key, "1", nx=True, ex=30)
        if not acquired:
            # Another request is already rotating this session
            return None

        try:
            # Double-check the session hasn't been rotated while waiting
            rotated_to = await self.cache.get_rotated_session(sess.id)
            if rotated_to:
                # Session was already rotated by another request
                return None

            # Time to rotate - create new session
            self.logger.info(
                "session_rotation",
                old_session=sess.id,
                user_id=sess.user_id,
                last_activity=last_activity.isoformat(),
            )

            # Bug fix: Don't copy refresh_jti/refresh_exp from old session meta
            # to prevent the new session from referencing the wrong refresh token
            new_meta = None
            if sess.meta:
                new_meta = {k: v for k, v in sess.meta.items()
                           if k not in ("refresh_jti", "refresh_exp")}

            device_type = self._get_session_device(sess)
            new_session = self.store.create_session(
                sess.user_id,
                mfa_required=sess.mfa_required,
                tenant_id=sess.tenant_id,
                user_agent=sess.user_agent,
                ip_addr=str(sess.ip_addr) if sess.ip_addr else None,
                ttl_minutes=self._get_session_ttl(device_type),
                meta=new_meta,
            )

            # Mark new session as MFA verified if old one was
            if sess.mfa_verified:
                self._mark_session_verified(new_session.id)
                new_session.mfa_verified = True

            # Set up grace period mapping
            grace_seconds = sys_settings.get("session_rotation_grace_seconds", 300)
            await self.cache.set_session_rotation_grace(
                sess.id,
                new_session.id,
                grace_seconds,
            )

            # Revoke old session
            self.store.revoke_session(sess.id)
            await self.cache.revoke_session(sess.id)

            # Initialize activity for new session
            await self.cache.update_session_activity(new_session.id)

            return new_session
        finally:
            # Release the lock
            await self.cache.client.delete(lock_key)

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
            # SPEC §12.1: Check if access token is denylisted before validating
            if self.cache:
                payload = self._decode_jwt(token)
                if payload and payload.get("token_type") == "access":
                    jti = payload.get("jti")
                    if jti:
                        try:
                            if await self.cache.is_access_token_denylisted(jti):
                                self.logger.info("access_token_denylisted", jti=jti)
                                return None
                        except Exception as exc:
                            # Fail-open: log warning but allow auth to proceed
                            # This prevents Redis failures from blocking all auth
                            self.logger.warning(
                                "denylist_check_failed",
                                jti=jti,
                                error=str(exc),
                            )
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
        now = self._now()
        if not sess or sess.expires_at <= now - self._clock_skew_leeway:
            return None, None, {}
        if sess.mfa_required and self.mfa_enabled and not sess.mfa_verified:
            return None, None, {}
        user = self.store.get_user(sess.user_id)
        if not user:
            return None, None, {}
        device = self._get_session_device(sess)
        tokens = self._issue_tokens(user, sess, device_type=device)
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
        now = self._now()

        # Issue 19.3: Use atomic MFA lockout to prevent check-then-act race condition
        # First check if already locked out (before TOTP verification)
        if self.cache:
            if await self.cache.check_mfa_lockout(user_id):
                self.logger.warning("mfa_locked_out", user_id=user_id)
                return False
        else:
            # In-memory fallback for lockout check
            # Issue 28.4: Thread-safe state access
            with self._state_lock:
                locked_until = self._mfa_lockouts.get(user_id)
                if locked_until and locked_until > now:
                    self.logger.warning("mfa_locked_out", user_id=user_id)
                    return False
                elif locked_until:
                    # Expired lockout, clean up
                    self._mfa_lockouts.pop(user_id, None)

        if not self._verify_totp(cfg.secret, code):
            # Issue 19.3: Use atomic operation to track failed attempt and check lockout
            if self.cache:
                is_locked, attempts = await self.cache.atomic_mfa_attempt(
                    user_id, max_attempts=5, lockout_seconds=300
                )
                if is_locked and attempts >= 0:
                    # Just triggered lockout (attempts >= 0 means we just incremented)
                    self.logger.warning("mfa_lockout_triggered", user_id=user_id, attempts=attempts)
            else:
                # In-memory fallback for attempt tracking (Issue 11.1)
                # Issue 28.4: Thread-safe state access
                with self._state_lock:
                    current = self._mfa_attempts.get(user_id)
                    window_start = now
                    attempts = 1
                    if current:
                        count, prev_window_start = current
                        # If within 5-minute window, increment; otherwise reset
                        if now - prev_window_start < timedelta(minutes=5):
                            attempts = count + 1
                            window_start = prev_window_start
                    self._mfa_attempts[user_id] = (attempts, window_start)
                    if attempts >= 5:
                        # Lock out for 5 minutes
                        self._mfa_lockouts[user_id] = now + timedelta(minutes=5)
                        self._mfa_attempts.pop(user_id, None)
                        self.logger.warning("mfa_lockout_triggered", user_id=user_id, attempts=attempts)
            return False

        # Success - clear any failed attempts
        if self.cache:
            await self.cache.clear_mfa_attempts(user_id)
        else:
            # In-memory fallback - clear attempts on success
            with self._state_lock:
                self._mfa_attempts.pop(user_id, None)

        self.store.set_user_mfa_secret(user_id, cfg.secret, enabled=True)
        if session_id:
            self._mark_session_verified(session_id)
        return True

    async def initiate_password_reset(self, email: str) -> str:
        # Use raw bytes for proper entropy (not string representation)
        token = hashlib.sha256(b"reset-" + email.encode() + os.urandom(32)).hexdigest()
        expires_at = self._now() + timedelta(minutes=15)
        # Persist a short-lived reset token with TTL in Redis if available
        if self.cache:
            await self.cache.client.set(
                f"reset:{token}",
                email,
                ex=int((expires_at - self._now()).total_seconds()),
            )
        else:
            # Issue 11.2: In-memory fallback for password reset tokens
            with self._state_lock:
                with self._with_state_lock():
                    self._password_reset_tokens[token] = (email, expires_at)
        self.logger.info(
            "password_reset_requested",
            email_hash=hashlib.sha256(email.encode()).hexdigest(),
        )
        return token

    async def complete_password_reset(self, token: str, new_password: str) -> bool:
        email = None
        if self.cache:
            email = await self.cache.client.get(f"reset:{token}")
        else:
            # Issue 11.2: In-memory fallback for password reset tokens
            with self._state_lock:
                with self._with_state_lock():
                    stored = self._password_reset_tokens.get(token)
                if stored:
                    stored_email, expires_at = stored
                    if expires_at <= self._now() - self._clock_skew_leeway:
                        # Remove expired token to prevent memory leak
                        with self._with_state_lock():
                            self._password_reset_tokens.pop(token, None)
                    else:
                        email = stored_email
                        with self._with_state_lock():
                            self._password_reset_tokens.pop(token, None)
        if not email:
            self.logger.warning("password_reset_invalid_token", token_prefix=token[:8])
            return False
        if isinstance(email, bytes):
            email = email.decode()
        user = self.store.get_user_by_email(email)
        if not user:
            self.logger.warning(
                "password_reset_user_missing",
                email_hash=hashlib.sha256(email.encode()).hexdigest(),
            )
            return False
        pwd_hash, algo = self._hash_password(new_password)
        self.store.save_password(user.id, pwd_hash, algo)
        try:
            await self.revoke_all_user_sessions(user.id)
        except Exception as exc:
            self.logger.warning(
                "revoke_sessions_failed", user_id=user.id, error=str(exc)
            )
        if self.cache:
            await self.cache.client.delete(f"reset:{token}")
        self.logger.info("password_reset_completed", user_id=user.id)
        return True

    async def request_email_verification(self, user: User) -> str:
        # Use raw bytes for proper entropy (not string representation)
        token = hashlib.sha256(
            b"verify-" + user.email.encode() + os.urandom(32)
        ).hexdigest()
        expires_at = self._now() + timedelta(hours=24)
        if self.cache:
            await self.cache.client.set(
                f"verify:{token}",
                user.id,
                ex=int((expires_at - self._now()).total_seconds()),
            )
        else:
            # Issue 28.4: Thread-safe state mutation
            with self._state_lock:
                self._email_verification_tokens[token] = (user.id, expires_at)
        self.logger.info("email_verification_requested", user_id=user.id)
        return token

    async def complete_email_verification(self, token: str) -> bool:
        user_id = None
        if self.cache:
            user_id = await self.cache.client.get(f"verify:{token}")
        else:
            # Issue 28.4: Thread-safe state access
            with self._state_lock:
                stored = self._email_verification_tokens.get(token)
                if stored:
                    user_id, expires_at = stored
                    if expires_at <= self._now() - self._clock_skew_leeway:
                        # Remove expired token to prevent memory leak
                        self._email_verification_tokens.pop(token, None)
                        user_id = None
                    else:
                        self._email_verification_tokens.pop(token, None)
        if isinstance(user_id, bytes):
            user_id = user_id.decode()
        if not user_id:
            self.logger.warning("email_verification_invalid_token", token_prefix=token[:8])
            return False
        user = self.store.get_user(user_id)
        if not user:
            self.logger.warning("email_verification_missing_user", user_id=user_id)
            return False
        if hasattr(self.store, "mark_email_verified"):
            self.store.mark_email_verified(user.id)
        if self.cache:
            await self.cache.client.delete(f"verify:{token}")
        else:
            with self._state_lock:
                self._email_verification_tokens.pop(token, None)
        self.logger.info("email_verified", user_id=user.id)
        return True

    def _hash_password(self, password: str) -> Tuple[str, str]:
        algo = "argon2id"
        digest = self._pwd_hasher.hash(password)
        return digest, algo

    def verify_password(self, user_id: str, password: str) -> bool:
        """Verify a user's password against stored hash."""
        record = self.store.get_password_record(user_id)
        if not record:
            self.logger.warning("password_record_missing", user_id=user_id)
            return False
        stored_hash, algo = record
        if algo != "argon2id":
            self.logger.warning("password_algo_mismatch", user_id=user_id, algo=algo)
            return False
        try:
            return self._pwd_hasher.verify(stored_hash, password)
        except (InvalidHash, VerifyMismatchError):
            self.logger.warning("password_verification_failed", user_id=user_id)
            return False

    def save_password(self, user_id: str, password: str) -> None:
        """Hash and save a new password for a user."""
        pwd_hash, algo = self._hash_password(password)
        self.store.save_password(user_id, pwd_hash, algo)

    def _verify_totp(
        self, secret: str, code: str, *, window: int = 0, interval: int = 30
    ) -> bool:
        # Issue 76.3: Narrow TOTP validation window and only allow a single
        # adjacent step for minor clock skew.
        grace_steps = min(1, int(self._clock_skew_leeway.total_seconds() // interval))
        allowed_window = max(window, grace_steps)
        for offset in range(-allowed_window, allowed_window + 1):
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
        digest = hmac.new(key, counter, hashlib.sha256).digest()
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

        # Issue 35.1: Validate header algorithm to prevent algorithm confusion attacks
        try:
            header = json.loads(self._decode_segment(header_b64))
            if header.get("alg") != "HS256":
                logger.warning("jwt_invalid_algorithm", alg=header.get("alg"))
                return None
        except Exception:
            logger.warning("jwt_header_decode_failed")
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
        if exp_ts <= time.time() - self._clock_skew_leeway.total_seconds():
            return None
        return payload

    def _issue_tokens(
        self, user: User, session: Session, *, device_type: Optional[str] = None
    ) -> dict[str, str]:
        now = self._now()
        sys_settings = self._get_system_settings()
        access_exp = int(
            (
                now + timedelta(minutes=sys_settings.get("access_token_ttl_minutes", 30))
            ).timestamp()
        )
        device = (device_type or self._get_session_device(session)).lower()
        refresh_ttl = self._get_refresh_ttl(device, sys_settings)
        refresh_exp = int((now + timedelta(minutes=refresh_ttl)).timestamp())
        # SPEC §12.1: Generate JTIs for both tokens to support denylist on logout
        access_jti = str(uuid.uuid4())
        refresh_jti = str(uuid.uuid4())
        access_payload = {
            "iss": self.settings.jwt_issuer,
            "aud": self.settings.jwt_audience,
            "sub": user.id,
            "sid": session.id,
            "tenant_id": user.tenant_id,
            "role": user.role,
            "token_type": "access",
            "jti": access_jti,  # Added for denylist support
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
        csrf_token = self._ensure_csrf_token(session)
        access_token = self._encode_jwt(access_payload)
        refresh_token = self._encode_jwt(refresh_payload)
        self._persist_session_meta(session, access_jti, access_exp, refresh_jti, refresh_exp)
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_at": datetime.utcfromtimestamp(access_exp).isoformat(),
            "csrf_token": csrf_token,
        }

    def _persist_session_meta(
        self, session: Session, access_jti: str, access_exp: int, refresh_jti: str, refresh_exp: int
    ) -> None:
        meta = dict(session.meta or {})
        meta.update({
            "access_jti": access_jti,
            "access_exp": access_exp,
            "refresh_jti": refresh_jti,
            "refresh_exp": refresh_exp,
        })
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
            ttl = max(int(exp - self._now().timestamp()), 0)
        if ttl is None:
            sys_settings = self._get_system_settings()
            ttl = max(int(sys_settings.get("refresh_token_ttl_minutes", 1440) * 60), 0)
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
        now = self._now()
        if not sess or sess.expires_at <= now - self._clock_skew_leeway:
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
