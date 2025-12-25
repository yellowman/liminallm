# liminallm deployment

a self-contained guide for taking liminallm from zero to one on linux. it fills the gaps that the readme leaves out and does
not assume you have read the spec. two execution lanes are available: local gpu lora (adapters over a frozen base model) and
api backends (remote inference). you can run them side by side.

## prerequisites
- runtime: python 3.11+.
- datastores: postgres 16 with `vector` + `citext`; redis 7 with auth.
- filesystem: writable `SHARED_FS_ROOT` (defaults to `/srv/liminallm`) for adapters, artifacts, and user files.
- gpu/tpu: only if `MODEL_BACKEND=local_gpu_lora` (nvidia cuda/cuDNN for jax gpu builds; amd/rocm if you build your own wheel).
- tls/reverse proxy: optional but recommended; an nginx template ships in-repo.

## config cheatsheet
set env vars before boot:
- database: `DATABASE_URL` (example: `postgres://liminallm:<password>@postgres:5432/liminallm`).
- redis: `REDIS_URL` (example: `redis://:<password>@redis:6379/0`).
- secrets: `JWT_SECRET` (required), `JWT_ISSUER`/`JWT_AUDIENCE` (optional defaults exist).
- model backend: `MODEL_BACKEND` (defaults to `openai`), `MODEL_PATH` for local base models, adapter keys like `ADAPTER_OPENAI_API_KEY`/`ADAPTER_OPENAI_BASE_URL`, optional `VOICE_*`.
- routing/limits: `CHAT_RATE_LIMIT_PER_MINUTE`, `RESET_RATE_LIMIT_PER_MINUTE`.
- smtp/oauth: `SMTP_*`, `OAUTH_*` if needed.
- ports: `HOST_PORT` (compose host), `PORT` (app listen).

data directories under `SHARED_FS_ROOT` must be writable:
- `/srv/liminallm/adapters`
- `/srv/liminallm/artifacts`
- `/srv/liminallm/models` (for local jax lora base models)
- `/srv/liminallm/users/<user_id>/files`

## zero-to-one linux checklist
1. create a service user, set `umask 027`, and `mkdir -p /srv/liminallm/{adapters,artifacts,models,users}`.
2. install python 3.11, gcc/build-essentials, `libpq-dev`, postgres 16 + pgvector, redis 7 with a password, and nginx if you want tls.
3. bootstrap db: create the `liminallm` role/db, enable `vector` + `citext`, then run `./scripts/migrate.sh` with `DATABASE_URL` set.
4. lock redis down with `requirepass`; point `REDIS_URL` at it.
5. pick your backend lane (see below), set env vars, and ensure `/srv/liminallm` is owned by the service user.
6. enable lingering for the service user or wire up systemd/supervisor to run `uvicorn` (see api launch below).
7. smoke test: `curl http://localhost:8000/healthz` returns `"ok"`; hit `/` for chat ui, `/admin` for admin-only controls.

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
6. start api: `python -m uvicorn liminallm.app:app --host 0.0.0.0 --port 8000 --workers 4`. static assets live under `/frontend`; readiness at `/healthz`.
7. front-end: browse `http://<host>:8000/` (chat) and `/admin` (admin role needed). put tls in front via nginx or your proxy of choice.

## backend lanes and scenarios
### local gpu lora (adapters only; base stays frozen)
- set `MODEL_BACKEND=local_gpu_lora`, `MODEL_PATH=/srv/liminallm/models/<base-model>` (hugging face-style dir), optional `MODEL_TOKENIZER` if tokenizer differs.
- copy base weights into `/srv/liminallm/models`; adapters live under `/srv/liminallm/adapters/<adapter_id>/adapter.lora`.
- gpu prep: install the matching jax gpu wheel (cuda/rocm), verify `nvidia-smi` sees the card, and keep drivers + cuda in `$LD_LIBRARY_PATH`.
- run: `python -m uvicorn liminallm.app:app --host 0.0.0.0 --port 8000 --workers 1` (jax likes fewer workers). requests specify `adapter_id` and optionally `adapter_mode` (local/hybrid/prompt); the backend overlays adapters over the frozen base and serves tokens locally.
- the base model remains immutable; training writes only adapter weights.

### api backend (remote inference)
- set `MODEL_BACKEND=openai` (default) or another provider hook; set `ADAPTER_OPENAI_API_KEY`/`ADAPTER_OPENAI_BASE_URL` or peer keys for your provider.
- calls go out to the remote model id you pass as `base_model`; adapters travel as ids or prompt patches when the provider supports multi-lora/prompt layering.
- scenarios:
  - **managed foundation only**: set `base_model` to the provider model, omit adapters for pure hosted inference.
  - **hosted foundation + local adapters**: keep adapters on disk and send adapter metadata with the request so the provider overlays your deltas over its model.
  - **prompt-only adapters**: for providers without lora, use `adapter_mode=prompt` to inject adapter prompts instead of weights.
- switching providers is a restart-level change: adjust `MODEL_BACKEND`, set the new api keys, and restart the process or container.

### hybrid deployments
- keep `MODEL_BACKEND=local_gpu_lora` for on-prem traffic and point select routes or tenants at an api backend via adapter modes or routing policies stored in artifacts.
- filesystem artifacts stay authoritative even with api backends; adapter payloads live under `/srv/liminallm/adapters`.

## model handling at a glance
- base models are frozen. local deployments place them under `/srv/liminallm/models` and the local jax backend keeps them resident; api backends treat `base_model` as a provider-owned model id and never upload local weights.
- adapters live under `/srv/liminallm/adapters/<adapter_id>/adapter.lora` and are loaded or streamed as metadata depending on backend capabilities.
- training and clustering write only adapter weights. the base model on disk or at the provider is untouched.
- when sending requests, set `base_model` to the foundation you want and `adapter_id`/`adapter_mode` to pick the adapter path (local weights, provider-hosted adapters, or prompt patching).

## ops & safety defaults (from the spec)
- observability: metrics and traces should include chat latency/error rates, adapter usage, preference/training counts, and workflow node timings; health checks live at `/healthz` and probe postgres, redis, and filesystem mounts.
- retention: metrics 7–14d, logs 30–90d with payload sampling and pii minimization.
- alerts: latency slo breaches, adapter cache miss spikes (>20%), training failure bursts, ingestion lag over an hour.
- backups: nightly postgres logical backup kept 7d; weekly filesystem snapshot pointers kept 4 weeks; redis is treated as ephemeral (state is recreated from postgres + filesystem artifacts).
- safety rails: content safety classifier on user/assistant text; preference events and training skip disallowed content.

## configuration management expectations
- principle: most runtime knobs live in the database and are editable via the admin ui (`/admin`) instead of env vars.
- database-managed settings include session rotation, concurrency caps, rate limits, pagination defaults, token ttls, feature flags (mfa/signup), training worker toggles, smtp/oauth and url settings, voice defaults, model backend/path, rag mode, embedding model id, and tenant/jwt claims.
- environment-only settings are reserved for infra/bootstrap secrets: `DATABASE_URL`, `REDIS_URL`, `JWT_SECRET`, provider api keys, and the minimal env overrides noted in the config cheatsheet above.

## reverse proxy + os notes
- nginx config sits at `nginx.conf`; enable the compose `nginx` service with the `production` profile or adapt for your host tls.
- openbsd operators: see `deploy/openbsd/` for service user setup, rc scripts, relayd/httpd, and `acme-client` tls.

## troubleshooting quick hits
- health: `curl http://localhost:8000/healthz`.
- migrations: rerun `scripts/migrate.sh` if schema drift bites.
- permissions: ensure write access to `SHARED_FS_ROOT` and authenticated connections to redis/postgres.
- logs: `docker compose logs -f app` or your process supervisor for startup clues.
