from __future__ import annotations

import os
from pathlib import Path
from typing import List

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from liminallm.api.routes import get_admin_user, router
from liminallm.api.error_handling import register_exception_handlers
from liminallm.logging import get_logger

logger = get_logger(__name__)

app = FastAPI(title="LiminalLM Kernel", version="0.1.0")


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
    allow_headers=["Content-Type", "Authorization", "X-Tenant-ID", "session_id", "Idempotency-Key", "X-CSRF-Token"],
)


@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-XSS-Protection", "1; mode=block")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=(), payment=()")
    if request.url.scheme == "https" and os.getenv("ENABLE_HSTS", "false").lower() in {"1", "true", "yes", "on"}:
        response.headers.setdefault("Strict-Transport-Security", "max-age=63072000; includeSubDomains")
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
async def health() -> dict:
    return {"status": "ok"}


def create_app() -> FastAPI:
    return app
