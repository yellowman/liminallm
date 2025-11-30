from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from fastapi import Depends, FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from liminallm.api.error_handling import register_exception_handlers
from liminallm.api.routes import get_admin_user, router
from liminallm.logging import get_logger, set_correlation_id

logger = get_logger(__name__)

# Version info per SPEC §18
__version__ = "0.1.0"
__build__ = os.getenv("BUILD_SHA", "dev")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    from liminallm.service.runtime import get_runtime

    try:
        runtime = get_runtime()
        if runtime.settings.training_worker_enabled:
            await runtime.training_worker.start()
            logger.info("training_worker_started_on_startup")
    except Exception as exc:
        logger.error("startup_training_worker_failed", error=str(exc))

    yield

    # Shutdown
    try:
        runtime = get_runtime()
        if runtime.training_worker:
            await runtime.training_worker.stop()
            logger.info("training_worker_stopped_on_shutdown")
    except Exception as exc:
        logger.error("shutdown_training_worker_failed", error=str(exc))


app = FastAPI(title="LiminalLM Kernel", version=__version__, lifespan=lifespan)


def _allowed_origins() -> List[str]:
    env_value = os.getenv("CORS_ALLOW_ORIGINS")
    if env_value:
        return [origin.strip() for origin in env_value.split(",") if origin.strip()]
    # Default to common local dev hosts; avoid wildcard when credentials are enabled.
    return [
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]


def _allow_credentials() -> bool:
    flag = os.getenv("CORS_ALLOW_CREDENTIALS")
    if flag is None:
        return False
    return flag.lower() in {"1", "true", "yes", "on"}


app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_credentials=_allow_credentials(),
    # Restrict to only required HTTP methods
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
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
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-XSS-Protection", "1; mode=block")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault(
        "Permissions-Policy", "camera=(), microphone=(), geolocation=(), payment=()"
    )
    if request.url.scheme == "https" and os.getenv("ENABLE_HSTS", "false").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        response.headers.setdefault(
            "Strict-Transport-Security", "max-age=63072000; includeSubDomains"
        )
    response.headers.setdefault(
        "Content-Security-Policy",
        "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; font-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'",
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

    # Database check
    try:
        runtime = get_runtime()
        if hasattr(runtime.store, "verify_connection"):
            runtime.store.verify_connection()
            checks["database"] = {"status": "healthy"}
        elif hasattr(runtime.store, "_connect"):
            # Postgres store - try a simple query
            with runtime.store._connect() as conn:
                conn.execute("SELECT 1").fetchone()
            checks["database"] = {"status": "healthy"}
        else:
            # Memory store - always healthy if runtime exists
            checks["database"] = {"status": "healthy", "type": "memory"}
    except Exception as exc:
        logger.error("health_check_database_failed", error=str(exc))
        # SECURITY: Don't expose internal error details in response
        checks["database"] = {"status": "unhealthy"}
        overall_healthy = False

    # Redis check (if configured)
    try:
        runtime = get_runtime()
        if hasattr(runtime, "cache") and runtime.cache is not None:
            runtime.cache.verify_connection()
            checks["redis"] = {"status": "healthy"}
        else:
            checks["redis"] = {"status": "not_configured"}
    except Exception as exc:
        logger.error("health_check_redis_failed", error=str(exc))
        # SECURITY: Don't expose internal error details in response
        checks["redis"] = {"status": "unhealthy", "degraded": True}

    # Filesystem check
    try:
        runtime = get_runtime()
        fs_root = getattr(runtime.store, "fs_root", None)
        if fs_root:
            fs_path = Path(fs_root)
            if fs_path.exists() and fs_path.is_dir():
                # Try to write/read a health check file
                health_file = fs_path / ".health_check"
                health_file.write_text(datetime.utcnow().isoformat())
                health_file.read_text()
                health_file.unlink(missing_ok=True)
                # SECURITY: Don't expose filesystem paths in response
                checks["filesystem"] = {"status": "healthy"}
            else:
                checks["filesystem"] = {"status": "unhealthy"}
                overall_healthy = False
        else:
            checks["filesystem"] = {"status": "not_configured"}
    except Exception as exc:
        logger.error("health_check_filesystem_failed", error=str(exc))
        # SECURITY: Don't expose internal error details in response
        checks["filesystem"] = {"status": "unhealthy"}
        overall_healthy = False

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
            except Exception:
                pass

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
        except Exception:
            pass
        lines.append('# HELP liminallm_database_healthy Database connection health')
        lines.append('# TYPE liminallm_database_healthy gauge')
        lines.append(f'liminallm_database_healthy {db_healthy}')

    except Exception as exc:
        logger.error("metrics_collection_failed", error=str(exc))

    return Response(content="\n".join(lines) + "\n", media_type="text/plain")


def create_app() -> FastAPI:
    return app
