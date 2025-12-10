# LiminalLM Deployment Guide

This document describes how to install and run LiminalLM with Docker Compose or directly on a host. It focuses on the core services (FastAPI app, PostgreSQL with pgvector, and Redis) and the filesystem layout the kernel expects.

## Prerequisites

- **Runtime:** Python 3.11+.
- **Datastores:** PostgreSQL 16 with the `vector` and `citext` extensions, and Redis 7 with authentication enabled.
- **Filesystem:** A writable shared path mounted at `SHARED_FS_ROOT` (default: `/srv/liminallm`) for adapters, artifacts, and user files.
- **GPU/TPU (optional):** Required only when using the local JAX LoRA backend (`MODEL_BACKEND=local_gpu_lora`).
- **TLS / reverse proxy:** Optional but recommended for production; an nginx template is provided.

## Configuration

Set the following environment variables before starting the app (whether containerized or bare-metal):

- **Database:** `DATABASE_URL` (e.g., `postgres://liminallm:<password>@postgres:5432/liminallm`).
- **Redis:** `REDIS_URL` (e.g., `redis://:<password>@redis:6379/0`).
- **Secrets:** `JWT_SECRET` (required), `JWT_ISSUER`/`JWT_AUDIENCE` (defaults provided).
- **Model backend:** `MODEL_BACKEND` (`openai` by default), `MODEL_PATH` (for local models), adapter provider secrets such as `ADAPTER_OPENAI_API_KEY`/`ADAPTER_OPENAI_BASE_URL`, and optional speech keys (`VOICE_*`).
- **Routing / rate limits:** `CHAT_RATE_LIMIT_PER_MINUTE`, `RESET_RATE_LIMIT_PER_MINUTE`.
- **SMTP/OAuth (optional):** `SMTP_*` for outbound mail and `OAUTH_*` for social login.
- **Ports:** `HOST_PORT` (host-facing port when using Docker Compose) and `PORT` (app listening port).

### Data directories

The app writes persistent payloads under `SHARED_FS_ROOT`, and the default Docker Compose stack also persists PostgreSQL and Redis volumes. Ensure the service user (or container) can read/write the following:

- `/srv/liminallm/adapters`
- `/srv/liminallm/artifacts`
- `/srv/liminallm/models` (if using local JAX LoRA)
- `/srv/liminallm/users/<user_id>/files`

## Option A: Docker Compose (recommended quickstart)

1. **Set secrets and passwords** in your shell (or an `.env` file): `POSTGRES_PASSWORD`, `REDIS_PASSWORD`, `JWT_SECRET`, and any provider keys you need.
2. **Start the stack**: `docker compose up -d` (add `--profile production nginx` to run the bundled nginx reverse proxy). The compose file builds the app image, provisions PostgreSQL with pgvector, and enables Redis with password auth.
3. **Verify health**: `curl http://localhost:${HOST_PORT:-8000}/healthz` should return `"ok"` once the app is ready.
4. **Persistence**: application data lives in the `liminallm-data` volume, PostgreSQL in `postgres-data`, and Redis in `redis-data`. The SQL migrations in `sql/` are applied automatically by Postgres on first start via Docker volume mounts.

## Option B: Manual host installation

1. **Install system packages** for Python 3.11, PostgreSQL client libraries, pgvector, Redis, and build tooling (e.g., `gcc` and `libpq-dev`).
2. **Create a virtual environment** and install LiminalLM: `python -m venv .venv && source .venv/bin/activate && pip install -e .`.
3. **Prepare data directories**: `mkdir -p /srv/liminallm` and ensure the service user owns the path.
4. **Configure databases**:
   - Create the `liminallm` database and user with the `vector` and `citext` extensions enabled.
   - Apply migrations: `DATABASE_URL=postgres://liminallm:<password>@localhost:5432/liminallm ./scripts/migrate.sh`.
5. **Run Redis** with `--requirepass` and export `REDIS_URL` accordingly.
6. **Start the API**: `python -m uvicorn liminallm.api.routes:app --host 0.0.0.0 --port 8000 --workers 4`. Static assets under `/frontend` are served by the app, and `/healthz` reports readiness.
7. **Front-end access**: browse to `http://<host>:8000/` for chat and `http://<host>:8000/admin` for the admin console (requires an admin role). Place TLS in front of the app with nginx or your preferred proxy.

## Reverse proxy and OS-specific notes

- An nginx configuration is provided at `nginx.conf`; enable the `nginx` service in Docker Compose with the `production` profile or adapt the config for your host TLS setup.
- OpenBSD-specific instructions (service user, rc scripts, relayd/httpd configs, and TLS with `acme-client`) live under `deploy/openbsd/` for operators targeting that platform.

## Troubleshooting

- **Health checks:** `curl http://localhost:8000/healthz` should return OK.
- **Database migrations:** rerun `scripts/migrate.sh` if schema drift is suspected.
- **Permissions:** confirm the app user can write to `SHARED_FS_ROOT` and that Redis/PostgreSQL accept authenticated connections with the configured URLs.
- **Logs:** check the container logs (`docker compose logs -f app`) or your process supervisor logs for startup errors.
