from __future__ import annotations

from fastapi import FastAPI

from liminallm.api.routes import router

app = FastAPI(title="LiminalLM Kernel", version="0.1.0")
app.include_router(router)


@app.get("/healthz")
async def health() -> dict:
    return {"status": "ok"}


def create_app() -> FastAPI:
    return app
