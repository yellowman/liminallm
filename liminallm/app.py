from __future__ import annotations

import asyncio
import contextlib
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from liminallm.api.error_handling import register_exception_handlers
from liminallm.api.routes import get_admin_user, router
from liminallm.config import Settings
from liminallm.logging import get_logger, set_correlation_id

logger = get_logger(__name__)

_settings = Settings.from_env()

# Version info per SPEC §18
__version__ = "0.1.0"
__build__ = _settings.build_sha


_cleanup_task: asyncio.Task | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    global _cleanup_task
    # Startup
    from liminallm.service.runtime import get_runtime

    try:
        runtime = get_runtime()
        # Check training_worker_enabled from DB settings (falls back to env var)
        training_worker_enabled = runtime.settings.training_worker_enabled
        if hasattr(runtime.store, "get_system_settings"):
            sys_settings = runtime.store.get_system_settings() or {}
            training_worker_enabled = sys_settings.get("training_worker_enabled", training_worker_enabled)
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
        await runtime.close()
        logger.info("runtime_cleanup_complete")
    except Exception as exc:
        logger.error("shutdown_failed", error=str(exc))


app = FastAPI(title="LiminalLM Kernel", version=__version__, lifespan=lifespan)


_CSRF_SAFE_METHODS = {"GET", "HEAD", "OPTIONS", "TRACE"}


def _allowed_origins() -> List[str]:
    if _settings.cors_allow_origins:
        return _settings.cors_allow_origins
    # Default to common local dev hosts; avoid wildcard when credentials are enabled.
    return [
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]


def _allow_credentials() -> bool:
    return _settings.cors_allow_credentials


app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_credentials=_allow_credentials(),
    # Restrict to only required HTTP methods
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    # Restrict to only required headers
    allow_headers=[
        "Content-Type",
        "Authorization",
        "X-Tenant-ID",
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
    try:
        from liminallm.service.runtime import get_runtime

        runtime = get_runtime()
        session = runtime.store.get_session(session_cookie)
    except Exception as exc:
        logger.warning("csrf_validation_failed", error=str(exc))
        return JSONResponse(
            status_code=403,
            content={
                "status": "error",
                "error": {"code": "forbidden", "message": "invalid session for CSRF check"},
            },
        )
    expected = session.meta.get("csrf_token") if session and isinstance(session.meta, dict) else None
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
    if request.url.path.startswith("/v1/") or request.url.path in ("/healthz", "/metrics"):
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


@app.get("/admin", response_class=FileResponse, dependencies=[Depends(get_admin_user)])
async def serve_admin() -> FileResponse:
    if not STATIC_DIR.exists():
        logger.warning("frontend_not_built", path=str(STATIC_DIR))
        raise HTTPException(status_code=404, detail="frontend not built")
    admin = STATIC_DIR / "admin.html"
    if not admin.exists():
        logger.warning("frontend_missing_admin", admin=str(admin))
        raise HTTPException(status_code=404, detail="admin ui missing")
    return FileResponse(admin)


HEALTH_CHECK_TIMEOUT_SECONDS = 3


@app.get("/healthz")
async def health() -> Dict[str, Any]:
    """Health check endpoint per SPEC §18.

    Performs dependency checks for:
    - Database connectivity
    - Redis availability (if configured)
    - Filesystem mount status

    Reports build/version info.
    """
    from liminallm.service.runtime import get_runtime

    checks: Dict[str, Dict[str, Any]] = {}
    overall_healthy = True

    # Version/build info per SPEC §18
    version_info = {
        "version": __version__,
        "build": __build__,
    }

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
    if hasattr(runtime.store, "verify_connection"):
        db_ok = await _run_bounded("database", runtime.store.verify_connection)
    elif hasattr(runtime.store, "_connect"):
        def _db_probe() -> None:
            with runtime.store._connect() as conn:
                conn.execute("SELECT 1").fetchone()

        db_ok = await _run_bounded("database", _db_probe)
    else:
        db_ok = True
        checks["database"] = {"status": "healthy", "type": "memory"}

    if not checks.get("database"):
        checks["database"] = {"status": "healthy" if db_ok else "unhealthy"}
    overall_healthy = overall_healthy and db_ok

    # Redis check (if configured)
    if hasattr(runtime, "cache") and runtime.cache is not None:
        redis_ok = await _run_bounded("redis", runtime.cache.verify_connection)
        checks["redis"] = {"status": "healthy" if redis_ok else "unhealthy", "degraded": not redis_ok}
        overall_healthy = overall_healthy and redis_ok
    else:
        checks["redis"] = {"status": "not_configured"}

    # Filesystem check
    fs_root = getattr(runtime.store, "fs_root", None)
    if fs_root:
        fs_path = Path(fs_root)

        def _fs_probe() -> None:
            if not fs_path.exists() or not fs_path.is_dir():
                raise FileNotFoundError(fs_path)
            health_file = fs_path / ".health_check"
            health_file.write_text(datetime.utcnow().isoformat())
            health_file.read_text()
            health_file.unlink(missing_ok=True)

        fs_ok = await _run_bounded("filesystem", _fs_probe)
        checks["filesystem"] = {"status": "healthy" if fs_ok else "unhealthy"}
        overall_healthy = overall_healthy and fs_ok
    else:
        checks["filesystem"] = {"status": "not_configured"}

    status = "healthy" if overall_healthy else "unhealthy"
    return {
        "status": status,
        "checks": checks,
        **version_info,
        "timestamp": datetime.utcnow().isoformat(),
    }


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
        if hasattr(runtime.store, "list_users"):
            try:
                all_users = runtime.store.list_users(limit=10000)
                user_count = len(all_users)
                lines.append('# HELP liminallm_users_total Total number of users')
                lines.append('# TYPE liminallm_users_total gauge')
                lines.append(f'liminallm_users_total {user_count}')
            except Exception as exc:
                logger.warning("metrics_user_count_failed", error=str(exc))

        # Active sessions (if Redis available)
        if hasattr(runtime, "cache") and runtime.cache is not None:
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
            if hasattr(runtime.store, "_connect"):
                with runtime.store._connect() as conn:
                    conn.execute("SELECT 1").fetchone()
                db_healthy = 1
            else:
                db_healthy = 1  # Memory store
        except Exception as exc:
            logger.warning("metrics_database_health_failed", error=str(exc))
        lines.append('# HELP liminallm_database_healthy Database connection health')
        lines.append('# TYPE liminallm_database_healthy gauge')
        lines.append(f'liminallm_database_healthy {db_healthy}')

        # Training job activity
        list_jobs = getattr(runtime.store, "list_training_jobs", None)
        if callable(list_jobs):
            try:
                jobs = list_jobs()
                active = len([j for j in jobs if j.status in {"queued", "running"}])
                lines.append('# HELP liminallm_training_jobs_active Active training jobs')
                lines.append('# TYPE liminallm_training_jobs_active gauge')
                lines.append(f'liminallm_training_jobs_active {active}')
            except Exception as exc:
                logger.warning("metrics_training_jobs_failed", error=str(exc))

        # Preference event ingestion rate proxy
        if hasattr(runtime.store, "list_preference_events"):
            try:
                events = runtime.store.list_preference_events(user_id=None)  # type: ignore[arg-type]
                lines.append('# HELP liminallm_preference_events_total Total recorded preference events')
                lines.append('# TYPE liminallm_preference_events_total counter')
                lines.append(f'liminallm_preference_events_total {len(events)}')
            except Exception as exc:
                logger.warning("metrics_preference_events_failed", error=str(exc))

        # Adapter usage counts
        if hasattr(runtime.store, "list_artifacts"):
            try:
                adapters = runtime.store.list_artifacts(kind="adapter", owner_user_id=None)  # type: ignore[arg-type]
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


def create_app() -> FastAPI:
    return app
