# liminallm deployment

quick, spunky notes for landing liminallm in the wild. everything here is terse, lowercase, and assumes you already skimmed the spec.

## prerequisites
- runtime: python 3.11+.
- datastores: postgres 16 with `vector` + `citext`; redis 7 with auth.
- filesystem: writable `SHARED_FS_ROOT` (defaults to `/srv/liminallm`) for adapters, artifacts, and user files.
- gpu/tpu: only if `MODEL_BACKEND=local_gpu_lora`.
- tls/reverse proxy: optional but recommended; nginx template ships in-repo.

## config cheatsheet
set env vars before boot:
- database: `DATABASE_URL` (example: `postgres://liminallm:<password>@postgres:5432/liminallm`).
- redis: `REDIS_URL` (example: `redis://:<password>@redis:6379/0`).
- secrets: `JWT_SECRET` (required), `JWT_ISSUER`/`JWT_AUDIENCE` (optional defaults exist).
- model backend: `MODEL_BACKEND` (defaults to `openai`), `MODEL_PATH` for local, adapter keys like `ADAPTER_OPENAI_API_KEY`/`ADAPTER_OPENAI_BASE_URL`, optional `VOICE_*`.
- routing/limits: `CHAT_RATE_LIMIT_PER_MINUTE`, `RESET_RATE_LIMIT_PER_MINUTE`.
- smtp/oauth: `SMTP_*`, `OAUTH_*` if needed.
- ports: `HOST_PORT` (compose host), `PORT` (app listen).

data directories under `SHARED_FS_ROOT` must be writable:
- `/srv/liminallm/adapters`
- `/srv/liminallm/artifacts`
- `/srv/liminallm/models` (for local jax lora)
- `/srv/liminallm/users/<user_id>/files`

## option a: docker compose (fast path)
1. export secrets/passwords (`POSTGRES_PASSWORD`, `REDIS_PASSWORD`, `JWT_SECRET`, provider keys).
2. `docker compose up -d` (add `--profile production nginx` to enable the bundled reverse proxy). builds the app, brings postgres + pgvector, and secures redis with password auth.
3. health: `curl http://localhost:${HOST_PORT:-8000}/healthz` should yield `"ok"`.
4. persistence: app data in `liminallm-data`, postgres in `postgres-data`, redis in `redis-data`; migrations from `sql/` apply on first boot.

## option b: manual host (bare metal)
1. install python 3.11, postgres client libs + pgvector, redis, build deps (`gcc`, `libpq-dev`).
2. create venv + install: `python -m venv .venv && source .venv/bin/activate && pip install -e .`.
3. prep storage: `mkdir -p /srv/liminallm` and give the service user ownership.
4. databases:
   - create the `liminallm` db/user; enable `vector` and `citext`.
   - run migrations: `DATABASE_URL=postgres://liminallm:<password>@localhost:5432/liminallm ./scripts/migrate.sh`.
5. run redis with `--requirepass` and set `REDIS_URL`.
6. start api: `python -m uvicorn liminallm.api.routes:app --host 0.0.0.0 --port 8000 --workers 4`. static assets live under `/frontend`; readiness at `/healthz`.
7. front-end: browse `http://<host>:8000/` (chat) and `/admin` (admin role needed). put tls in front via nginx or your proxy of choice.

## reverse proxy + os notes
- nginx config sits at `nginx.conf`; enable the compose `nginx` service with the `production` profile or adapt for your host tls.
- openbsd operators: see `deploy/openbsd/` for service user setup, rc scripts, relayd/httpd, and `acme-client` tls.

## troubleshooting quick hits
- health: `curl http://localhost:8000/healthz`.
- migrations: rerun `scripts/migrate.sh` if schema drift bites.
- permissions: ensure write access to `SHARED_FS_ROOT` and authenticated connections to redis/postgres.
- logs: `docker compose logs -f app` or your process supervisor for startup clues.
