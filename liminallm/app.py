from __future__ import annotations

import asyncio
import contextlib
import os
import posixpath
import re
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from liminallm.api.error_handling import register_exception_handlers
from liminallm.api.routes import router
from liminallm.config import Settings
from liminallm.logging import get_logger, set_correlation_id

logger = get_logger(__name__)

_settings = Settings.from_env()

# Version info per SPEC §18
__version__ = "0.1.0"
__build__ = _settings.build_sha


_cleanup_task: asyncio.Task | None = None
_settings_watch_task: asyncio.Task | None = None


async def _run_settings_watcher(runtime, interval_seconds: int) -> None:
    """Per-worker loop that reloads model services when admin settings change.

    Uvicorn runs multiple workers, each with its own in-process Runtime, so a
    settings change persisted by one worker must be picked up by the others.
    This polls the persisted settings version and reloads only when a
    model-affecting value actually differs (see Runtime.maybe_reload_model_services).
    """
    interval = max(int(interval_seconds), 1)
    try:
        while True:
            await asyncio.sleep(interval)
            try:
                await asyncio.to_thread(runtime.maybe_reload_model_services)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # best-effort; keep the worker serving
                logger.warning("settings_watch_reload_failed", error=str(exc))
    except asyncio.CancelledError:
        logger.info("settings_watch_task_cancelled")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    global _cleanup_task, _settings_watch_task
    # Startup
    from liminallm.service.runtime import get_runtime

    try:
        runtime = get_runtime()
        # Cross-replica coordination: lets POST /chat/cancel reach the worker
        # holding the stream's WebSocket instead of only stopping local ones.
        try:
            from liminallm.service import cancellation, token_counting

            runtime.bus.subscribe(
                cancellation.CANCEL_CHANNEL, cancellation.handle_remote_cancel
            )

            async def _apply_calibration(data: dict):
                """A peer learned a better token factor; adopt it."""
                model = str(data.get("model") or "")
                try:
                    factor = float(data.get("factor") or 0)
                except (TypeError, ValueError):
                    return None
                if factor > 0:
                    token_counting.apply_shared_factor(
                        model, factor, int(data.get("observations") or 0)
                    )
                return None  # broadcast, not a request: never ack

            runtime.bus.subscribe(
                token_counting.CALIBRATION_CHANNEL, _apply_calibration
            )
            await runtime.bus.start()
        except Exception as exc:  # coordination is optional; keep serving
            logger.warning("cluster_bus_start_failed", error=str(exc))
        # Check training_worker_enabled from DB settings (falls back to env var)
        training_worker_enabled = runtime.settings.training_worker_enabled
        sys_settings = runtime.store.get_system_settings() or {}
        training_worker_enabled = sys_settings.get(
            "training_worker_enabled", training_worker_enabled
        )
        if training_worker_enabled:
            await runtime.training_worker.start()
            logger.info("training_worker_started_on_startup")
        # Issue 4.6: Schedule cleanup of per-user tmp scratch directories
        _cleanup_task = asyncio.create_task(
            _run_tmp_cleanup(
                Path(runtime.settings.shared_fs_root),
                runtime.settings.tmp_cleanup_interval_seconds,
                runtime.settings.tmp_max_age_hours,
            )
        )
        # Watch for admin settings changes made by other workers and reload.
        _settings_watch_task = asyncio.create_task(
            _run_settings_watcher(
                runtime, runtime.settings.settings_watch_interval_seconds
            )
        )
    except Exception as exc:
        logger.error("startup_training_worker_failed", error=str(exc))

    yield

    # Shutdown
    try:
        runtime = get_runtime()
        if _cleanup_task:
            _cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await _cleanup_task
        if _settings_watch_task:
            _settings_watch_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await _settings_watch_task
        await runtime.close()
        logger.info("runtime_cleanup_complete")
    except Exception as exc:
        logger.error("shutdown_failed", error=str(exc))


app = FastAPI(title="LiminalLM Kernel", version=__version__, lifespan=lifespan)


_CSRF_SAFE_METHODS = {"GET", "HEAD", "OPTIONS", "TRACE"}


#: Local dev hosts, used when nothing is configured. Never a wildcard: with
#: credentials allowed, a wildcard is an open door.
_DEFAULT_ORIGINS = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
]


def _cors_policy() -> tuple:
    """The origins and credentials flag currently in force.

    Prefers the running instance's settings — which carry what an admin saved —
    and falls back to the environment before the runtime exists (import time,
    and tests that construct the app directly).
    """
    try:
        from liminallm.service import runtime as runtime_module

        settings = getattr(runtime_module.runtime, "settings", None)
    except Exception:  # noqa: BLE001 - no runtime yet is the normal early case
        settings = None
    if settings is None:
        settings = Settings.from_env()
    return (
        tuple(settings.cors_allow_origins or _DEFAULT_ORIGINS),
        bool(settings.cors_allow_credentials),
    )


class DynamicCORSMiddleware:
    """Starlette's CORS implementation, rebuilt when the policy changes.

    CORSMiddleware fixes its origins at construction, which used to mean the
    allowlist was whatever the environment said when the process started.
    Origins are an admin-managed setting now, so this re-reads the policy per
    request and rebuilds the inner middleware only when it actually differs —
    reusing Starlette's implementation rather than reimplementing CORS, which
    is not a thing to get subtly wrong.
    """

    def __init__(self, app, **static):
        self.app = app
        self._static = static
        self._policy = None
        self._inner = None

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        policy = _cors_policy()
        if policy != self._policy or self._inner is None:
            self._policy = policy
            self._inner = CORSMiddleware(
                self.app,
                allow_origins=list(policy[0]),
                allow_credentials=policy[1],
                **self._static,
            )
        return await self._inner(scope, receive, send)


app.add_middleware(
    DynamicCORSMiddleware,
    # Restrict to only required HTTP methods
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    # Restrict to only required headers
    allow_headers=[
        "Content-Type",
        "Authorization",
        "session_id",
        "Idempotency-Key",
        "X-CSRF-Token",
        "X-Request-ID",  # For correlation ID tracing per SPEC §15.2
    ],
    # Issue 52.9: Expose X-Request-ID header to frontend JavaScript
    expose_headers=["X-Request-ID", "API-Version"],
    # Issue 52.8: Cache preflight requests for 1 hour
    max_age=3600,
)


@app.middleware("http")
async def add_correlation_id(request, call_next):
    """Add correlation ID to each request for tracing per SPEC §15.2.

    Correlation IDs enable request tracing across logs. The ID is:
    1. Extracted from X-Request-ID header if provided by client
    2. Otherwise generated as a new UUID

    The correlation ID is:
    - Set in context for structured logging
    - Returned in X-Request-ID response header for client tracing
    """
    client_request_id = request.headers.get("X-Request-ID")
    correlation_id = set_correlation_id(client_request_id)
    response = await call_next(request)
    response.headers["X-Request-ID"] = correlation_id
    return response


@app.middleware("http")
async def enforce_csrf_token(request: Request, call_next):
    # Only enforce CSRF for state-changing requests that include session cookies
    if request.method.upper() in _CSRF_SAFE_METHODS:
        return await call_next(request)
    if request.headers.get("Authorization"):
        return await call_next(request)
    session_cookie = request.cookies.get("session_id")
    if not session_cookie:
        return await call_next(request)
    try:
        from liminallm.service.runtime import get_runtime

        runtime = get_runtime()
        session = runtime.store.get_session(session_cookie)
    except Exception as exc:
        logger.warning("csrf_session_lookup_failed", error=str(exc))
        session = None
    if session is None:
        # Stale cookie referencing a session the server no longer knows about
        # (expired, revoked, or lost across a restart). A dead session cannot
        # authenticate anything, so there is nothing for CSRF to protect;
        # blocking here would lock browsers out of /auth/login until cookies
        # are cleared manually. Handlers still treat the request as
        # unauthenticated.
        return await call_next(request)
    header_token = request.headers.get("X-CSRF-Token")
    cookie_token = request.cookies.get("csrf_token")
    if not header_token or not cookie_token or header_token != cookie_token:
        return JSONResponse(
            status_code=403,
            content={
                "status": "error",
                "error": {"code": "forbidden", "message": "missing or invalid CSRF token"},
            },
        )
    expected = session.meta.get("csrf_token") if isinstance(session.meta, dict) else None
    if not expected or expected != header_token:
        return JSONResponse(
            status_code=403,
            content={
                "status": "error",
                "error": {"code": "forbidden", "message": "missing or invalid CSRF token"},
            },
        )
    return await call_next(request)


@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-XSS-Protection", "1; mode=block")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    # Issue 52.3: Add Cache-Control header to prevent caching of API responses
    # API responses should not be cached by proxies/CDNs to prevent data leakage
    if request.url.path.startswith("/v1/") or request.url.path in ("/healthz", "/readyz", "/metrics"):
        response.headers.setdefault("Cache-Control", "no-store, no-cache, must-revalidate, private")
    response.headers.setdefault(
        "Permissions-Policy", "camera=(), microphone=(), geolocation=(), payment=()"
    )
    if request.url.scheme == "https" and _settings.enable_hsts:
        response.headers.setdefault(
            "Strict-Transport-Security", "max-age=63072000; includeSubDomains"
        )
    response.headers.setdefault(
        "Content-Security-Policy",
        "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; font-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'",
    )
    return response


# Issue 49.5, 49.7: API versioning middleware
# Deprecated endpoints that will be removed in future versions
_DEPRECATED_ENDPOINTS = {
    # "/v1/old-endpoint": {"sunset": "2025-06-01", "replacement": "/v1/new-endpoint"},
}


@app.middleware("http")
async def add_api_version_headers(request, call_next):
    """Add API version and deprecation headers (Issue 49.5, 49.7).

    Adds:
    - API-Version: Current API version
    - Deprecation: If endpoint is deprecated
    - Sunset: Date when deprecated endpoint will be removed
    - Link: Link to documentation for deprecated endpoints
    """
    response = await call_next(request)

    # Issue 49.7: Always include API version header
    response.headers.setdefault("API-Version", __version__)

    # Issue 49.5: Check for deprecated endpoints
    path = request.url.path
    if path in _DEPRECATED_ENDPOINTS:
        deprecation_info = _DEPRECATED_ENDPOINTS[path]
        response.headers["Deprecation"] = "true"
        if "sunset" in deprecation_info:
            response.headers["Sunset"] = deprecation_info["sunset"]
        if "replacement" in deprecation_info:
            response.headers["Link"] = (
                f'<{deprecation_info["replacement"]}>; rel="successor-version"'
            )

    return response


register_exception_handlers(app)
app.include_router(router)


STATIC_DIR = Path(__file__).resolve().parent.parent / "frontend"


@app.middleware("http")
async def block_direct_admin_access(request: Request, call_next):
    """Block direct access to /static/admin.html - use /admin route instead."""
    # Canonicalize the path so dot-segments and redundant slashes
    # (/static/./admin.html, /static//admin.html, /static/../static/admin.html)
    # can't slip past the guard while StaticFiles still resolves the file.
    raw_path = request.url.path.replace("\\", "/")
    normalized = posixpath.normpath(raw_path).rstrip("/").lower()
    if normalized in {"/static/admin.html", "/static/admin"}:
        return JSONResponse(
            status_code=403,
            content={"detail": "Use /admin route for admin UI"},
        )
    return await call_next(request)


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR, html=False), name="static")
else:
    logger.warning("frontend_assets_missing", path=str(STATIC_DIR))


@app.get("/", response_class=FileResponse)
async def serve_chat() -> FileResponse:
    if not STATIC_DIR.exists():
        logger.warning("frontend_not_built", path=str(STATIC_DIR))
        raise HTTPException(status_code=404, detail="frontend not built")
    index = STATIC_DIR / "index.html"
    if not index.exists():
        logger.warning("frontend_missing_entrypoint", index=str(index))
        raise HTTPException(status_code=404, detail="chat ui missing")
    return FileResponse(index)


# Public share pages (SPEC §18): read-only views of conversations the owner
# explicitly published. The directory and pages are noindex by default — both
# via X-Robots-Tag here and robots.txt below — so shared chats don't end up
# in search engines unless that policy is deliberately changed.
_SHARE_NOINDEX = {"X-Robots-Tag": "noindex, nofollow"}


def _serve_share_page() -> FileResponse:
    share = STATIC_DIR / "share.html"
    if not STATIC_DIR.exists() or not share.exists():
        logger.warning("frontend_missing_share", share=str(share))
        raise HTTPException(status_code=404, detail="share ui missing")
    return FileResponse(share, headers=_SHARE_NOINDEX)


@app.get("/share", response_class=FileResponse)
async def serve_share_directory() -> FileResponse:
    return _serve_share_page()


@app.get("/share/{conversation_id}", response_class=FileResponse)
async def serve_share_conversation(conversation_id: str) -> FileResponse:
    # The id is resolved client-side from the URL; this route only serves the
    # static page (and exists so deep links work), so the parameter is unused.
    return _serve_share_page()


@app.get("/robots.txt")
async def serve_robots() -> Response:
    return Response(
        "User-agent: *\nDisallow: /share/\nDisallow: /share\n",
        media_type="text/plain",
    )


# The admin page carries its own sign-in form and every API it calls enforces
# the admin role server-side. Requiring admin auth to serve the static HTML
# made the page unreachable: browser navigation sends no Authorization header,
# so this route always returned 403 — including to admins.
@app.get("/admin", response_class=FileResponse)
async def serve_admin() -> FileResponse:
    if not STATIC_DIR.exists():
        logger.warning("frontend_not_built", path=str(STATIC_DIR))
        raise HTTPException(status_code=404, detail="frontend not built")
    admin = STATIC_DIR / "admin.html"
    if not admin.exists():
        logger.warning("frontend_missing_admin", admin=str(admin))
        raise HTTPException(status_code=404, detail="admin ui missing")
    return FileResponse(admin)


_VOICE_FILE_RE = re.compile(r"^[0-9a-fA-F-]{36}\.(mp3|txt)$")


@app.get("/voice/{user_id}/{filename}")
async def serve_voice_file(user_id: str, filename: str, request: Request) -> FileResponse:
    """Serve synthesized voice files referenced by /v1/voice/synthesize.

    The synthesize endpoint returns audio_url values under /voice/, which
    browsers load via <audio> elements. Those requests carry session cookies
    but no Authorization header, so this route authenticates via the session
    cookie and only serves a user their own files (or the shared pool).
    """
    if not _VOICE_FILE_RE.fullmatch(filename):
        raise HTTPException(status_code=404, detail="not found")
    session_cookie = request.cookies.get("session_id")
    if not session_cookie:
        raise HTTPException(status_code=401, detail="authentication required")
    from liminallm.service.runtime import get_runtime

    runtime = get_runtime()
    try:
        session = runtime.store.get_session(session_cookie)
    except Exception:
        session = None
    if session is None:
        raise HTTPException(status_code=401, detail="authentication required")
    if user_id != "shared" and session.user_id != user_id:
        raise HTTPException(status_code=403, detail="forbidden")
    base = (Path(runtime.settings.shared_fs_root) / "voice" / user_id).resolve()
    path = (base / filename).resolve()
    if base not in path.parents or not path.is_file():
        raise HTTPException(status_code=404, detail="not found")
    media_type = "audio/mpeg" if filename.endswith(".mp3") else "text/plain"
    return FileResponse(path, media_type=media_type)


HEALTH_CHECK_TIMEOUT_SECONDS = 3


async def _dependency_checks() -> tuple[Dict[str, Dict[str, Any]], bool, bool]:
    """Probe dependencies once.

    Returns (checks, serving_ok, fully_healthy).

    ``serving_ok`` covers only what a replica needs in order to answer
    requests — the database and the filesystem. Redis is deliberately excluded:
    it is an optional accelerator (rate limits, idempotency, caches all fall
    back), so a Redis outage must not make every replica fail its readiness
    probe and drain the entire fleet. It still shows up as degraded in the body
    and in ``fully_healthy`` for monitoring.
    """
    from liminallm.service.runtime import get_runtime

    checks: Dict[str, Dict[str, Any]] = {}

    async def _run_bounded(label: str, func) -> bool:
        try:
            await asyncio.wait_for(asyncio.to_thread(func), HEALTH_CHECK_TIMEOUT_SECONDS)
            return True
        except asyncio.TimeoutError:
            logger.error(
                "health_check_timeout", component=label, timeout=HEALTH_CHECK_TIMEOUT_SECONDS
            )
        except Exception as exc:  # pragma: no cover - defensive logging path
            logger.error(f"health_check_{label}_failed", error=str(exc))
        return False

    # Database check
    runtime = get_runtime()
    db_ok = await _run_bounded("database", runtime.store.verify_connection)

    if not checks.get("database"):
        checks["database"] = {"status": "healthy" if db_ok else "unhealthy"}

    # Redis check (if configured). Never gates serving — see the docstring.
    redis_ok = True
    if runtime.cache is not None:
        redis_ok = await _run_bounded("redis", runtime.cache.verify_connection)
        checks["redis"] = {"status": "healthy" if redis_ok else "unhealthy", "degraded": not redis_ok}
    else:
        checks["redis"] = {"status": "not_configured"}

    # Filesystem check
    fs_path = Path(runtime.store.fs_root)

    def _fs_probe() -> None:
        if not fs_path.exists() or not fs_path.is_dir():
            raise FileNotFoundError(fs_path)
        # Unique name per probe: SHARED_FS_ROOT is shared across replicas,
        # and concurrent probes racing on one fixed filename made read_text
        # hit another probe's unlink — a spurious 503 that drained healthy
        # replicas.
        health_file = fs_path / f".health_check-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        try:
            health_file.write_text(datetime.now(timezone.utc).isoformat())
            health_file.read_text()
        finally:
            health_file.unlink(missing_ok=True)

    fs_ok = await _run_bounded("filesystem", _fs_probe)
    checks["filesystem"] = {"status": "healthy" if fs_ok else "unhealthy"}

    # Reported, never gating: the cluster bus only affects cross-replica
    # cancellation, and "local" is the right answer for a single process.
    bus = getattr(runtime, "bus", None)
    checks["cluster_bus"] = {
        "status": "healthy" if getattr(bus, "connected", True) else "degraded",
        "backend": getattr(bus, "backend", "local"),
    }

    serving_ok = db_ok and fs_ok
    return checks, serving_ok, serving_ok and redis_ok


def _health_body(checks: Dict[str, Dict[str, Any]], healthy: bool) -> Dict[str, Any]:
    return {
        "status": "healthy" if healthy else "unhealthy",
        "checks": checks,
        "version": __version__,
        "build": __build__,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/healthz")
async def health() -> Dict[str, Any]:
    """Human/monitoring health report. Always HTTP 200.

    Reports database, Redis, and filesystem state plus build info. Kept at 200
    regardless of status so scrapers and the settings UI can always read the
    body; load balancers should check /readyz instead.
    """
    checks, _serving_ok, fully_healthy = await _dependency_checks()
    return _health_body(checks, fully_healthy)


@app.get("/readyz")
async def readiness() -> Response:
    """Load-balancer readiness probe: 200 when this replica can serve, else 503.

    /healthz always returns 200 by design, which means an LB checking it can
    never drain a broken replica. This one answers with a status code, and only
    fails on the dependencies that actually stop the replica from working —
    a Redis outage reports as degraded but stays ready, since every Redis-backed
    feature has a fallback.
    """
    checks, serving_ok, fully_healthy = await _dependency_checks()
    return JSONResponse(
        _health_body(checks, fully_healthy),
        status_code=200 if serving_ok else 503,
    )


@app.get("/metrics", response_class=Response)
async def metrics() -> Response:
    """Prometheus-compatible metrics endpoint.

    Exposes application metrics in Prometheus text format for monitoring.
    """
    from liminallm.service.runtime import get_runtime

    lines = []

    # Application info
    lines.append('# HELP liminallm_info Application version info')
    lines.append('# TYPE liminallm_info gauge')
    lines.append(f'liminallm_info{{version="{__version__}",build="{__build__}"}} 1')

    try:
        runtime = get_runtime()

        # User count
        try:
            user_count = len(runtime.store.list_users(limit=10000))
            lines.append('# HELP liminallm_users_total Total number of users')
            lines.append('# TYPE liminallm_users_total gauge')
            lines.append(f'liminallm_users_total {user_count}')
        except Exception as exc:
            logger.warning("metrics_user_count_failed", error=str(exc))

        # Active sessions (if Redis available)
        if runtime.cache is not None:
            lines.append('# HELP liminallm_cache_available Redis cache availability')
            lines.append('# TYPE liminallm_cache_available gauge')
            lines.append('liminallm_cache_available 1')
        else:
            lines.append('# HELP liminallm_cache_available Redis cache availability')
            lines.append('# TYPE liminallm_cache_available gauge')
            lines.append('liminallm_cache_available 0')

        # Database status
        db_healthy = 0
        try:
            runtime.store.verify_connection()
            db_healthy = 1
        except Exception as exc:
            logger.warning("metrics_database_health_failed", error=str(exc))
        lines.append('# HELP liminallm_database_healthy Database connection health')
        lines.append('# TYPE liminallm_database_healthy gauge')
        lines.append(f'liminallm_database_healthy {db_healthy}')

        # Training job activity
        try:
            jobs = runtime.store.list_training_jobs()
            active = len([j for j in jobs if j.status in {"queued", "running"}])
            lines.append('# HELP liminallm_training_jobs_active Active training jobs')
            lines.append('# TYPE liminallm_training_jobs_active gauge')
            lines.append(f'liminallm_training_jobs_active {active}')
        except Exception as exc:
            logger.warning("metrics_training_jobs_failed", error=str(exc))

        # Preference event ingestion rate proxy
        try:
            events = runtime.store.list_preference_events(user_id=None)
            lines.append('# HELP liminallm_preference_events_total Total recorded preference events')
            lines.append('# TYPE liminallm_preference_events_total counter')
            lines.append(f'liminallm_preference_events_total {len(events)}')
        except Exception as exc:
            logger.warning("metrics_preference_events_failed", error=str(exc))

        # Adapter usage counts
        try:
            adapters = runtime.store.list_artifacts(type_filter="adapter")
            lines.append('# HELP liminallm_adapters_total Adapters stored in system')
            lines.append('# TYPE liminallm_adapters_total gauge')
            lines.append(f'liminallm_adapters_total {len(adapters)}')
        except Exception as exc:
            logger.warning("metrics_adapters_failed", error=str(exc))

    except Exception as exc:
        logger.error("metrics_collection_failed", error=str(exc))

    return Response(content="\n".join(lines) + "\n", media_type="text/plain")


def _sweep_tmp_dirs(shared_root: Path, max_age_hours: int) -> None:
    """Remove stale files from per-user tmp scratch directories.

    SPEC §18 requires per-user scratch cleanup on a daily cadence. This helper
    runs in a thread to avoid blocking the event loop.
    """

    cutoff_ts = (datetime.now(timezone.utc) - timedelta(hours=max_age_hours)).timestamp()
    user_root = shared_root / "users"
    if not user_root.exists():
        return

    for user_dir in user_root.iterdir():
        tmp_dir = user_dir / "tmp"
        if not tmp_dir.exists():
            continue
        # Delete stale files and then prune empty directories depth-first
        for path in sorted(tmp_dir.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue
            if path.is_file() and stat.st_mtime < cutoff_ts:
                path.unlink(missing_ok=True)
        # Remove empty directories after file cleanup
        for path in sorted(tmp_dir.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            if path.is_dir():
                try:
                    next(path.iterdir())
                except (OSError, StopIteration):
                    with contextlib.suppress(OSError):
                        path.rmdir()
        try:
            next(tmp_dir.iterdir())
        except (OSError, StopIteration):
            with contextlib.suppress(OSError):
                tmp_dir.rmdir()


async def _run_tmp_cleanup(
    shared_root: Path, interval_seconds: int, max_age_hours: int
) -> None:
    """Background loop to periodically clean tmp scratch directories."""

    interval = max(interval_seconds, 300)
    try:
        while True:
            try:
                await asyncio.to_thread(_sweep_tmp_dirs, shared_root, max_age_hours)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover - best-effort cleanup
                logger.warning("tmp_cleanup_failed", error=str(exc))
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        logger.info("tmp_cleanup_task_cancelled")

