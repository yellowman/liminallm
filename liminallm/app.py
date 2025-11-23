from __future__ import annotations

from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from liminallm.api.routes import get_admin_user, router

app = FastAPI(title="LiminalLM Kernel", version="0.1.0")
app.include_router(router)

STATIC_DIR = Path(__file__).resolve().parent.parent / "frontend"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR, html=False), name="static")


@app.get("/", response_class=FileResponse)
async def serve_chat() -> FileResponse:
    if not STATIC_DIR.exists():
        raise HTTPException(status_code=404, detail="frontend not built")
    index = STATIC_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="chat ui missing")
    return FileResponse(index)


@app.get("/admin", response_class=FileResponse, dependencies=[Depends(get_admin_user)])
async def serve_admin() -> FileResponse:
    if not STATIC_DIR.exists():
        raise HTTPException(status_code=404, detail="frontend not built")
    admin = STATIC_DIR / "admin.html"
    if not admin.exists():
        raise HTTPException(status_code=404, detail="admin ui missing")
    return FileResponse(admin)


@app.get("/healthz")
async def health() -> dict:
    return {"status": "ok"}


def create_app() -> FastAPI:
    return app
