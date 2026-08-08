# liminallm operations

Installation is in [INSTALL.md](../INSTALL.md). This covers what comes
after: backend lanes, model handling, replicas, and ops defaults.

## prerequisites
- runtime: python 3.10+.
- ocr: `tesseract-ocr` (apt/brew) plus `pip install 'liminallm[ocr]'`. technically optional, practically required: it is what lets uploaded images and scanned pdfs be read locally — deterministic, free per call, and it quotes documents instead of paraphrasing them. without it every image read costs a model vision call (and a text-only backend can't read images at all). install it unless you have a reason not to.
- extraction security: uploads are parsed (pillow/pypdf/tesseract/poppler) inside a disposable rlimited child process with a wall-clock kill — a malicious file that exploits a parser lands in a short-lived capped process, not the api server. the child shares the service user's uid, so run the app in a container/vm as the outer wall.
- pdf rasterization: `poppler-utils` (apt/brew), auto-detected. scanned pdfs are rendered page-by-page through `pdftoppm` before ocr, which reads anything a viewer could show — jbig2 and ccitt fax compression included; without poppler, only pdfs whose embedded page images pypdf can decode are readable. pillow is the converter for images themselves: png/jpg (cmyk included)/webp/gif/tiff (multi-page)/bmp all normalize to what tesseract expects. `.docx`/`.odt` extract natively (stdlib zip+xml, no ocr involved); legacy `.doc` is refused with a save-as suggestion.
- image reader order is configurable via the `extract_readers` admin setting (default `ocr,vision`); new readers — another ocr engine, a dedicated ocr model, a model on new hardware — register via `extract.register_reader` without touching the ladder.
- datastores: postgres 16 with `vector` + `citext`; redis 7 with auth.
- filesystem: a writable path for adapters, artifacts, and user files. set `shared_fs_root` in the admin console; defaults to `/srv/liminallm`.
- gpu/tpu: only if `model_backend` is `local_gpu_lora` (nvidia cuda/cuDNN for jax gpu builds; amd/rocm if you build your own wheel).
- tls/reverse proxy: optional but recommended; an nginx template ships in-repo.

## backend lanes and scenarios
### local gpu lora (adapters only; base stays frozen)
- set `model_backend` to `local_gpu_lora` and `model_path` to `/srv/liminallm/models/<base-model>` (hugging face-style dir) in the admin console.
- copy base weights into `/srv/liminallm/models`; adapters live under `/srv/liminallm/adapters/<adapter_id>/adapter.lora`.
- gpu prep: install the matching jax gpu wheel (cuda/rocm), verify `nvidia-smi` sees the card, and keep drivers + cuda in `$LD_LIBRARY_PATH`.
- run: `python -m uvicorn liminallm.app:app --host 0.0.0.0 --port 8000 --workers 1` (jax likes fewer workers). requests specify `adapter_id` and optionally `adapter_mode` (local/hybrid/prompt); the backend overlays adapters over the frozen base and serves tokens locally.
- the base model remains immutable; training writes only adapter weights.

### api backend (remote inference)
- set `model_backend` (default `openai`) and `adapter_openai_base_url` in the admin console; the provider key goes in its own admin setting, or the matching `<PROVIDER>_API_KEY` environment variable as a fallback.
- calls go out to the remote model id you pass as `base_model`; adapters travel as ids or prompt patches when the provider supports multi-lora/prompt layering.
- scenarios:
  - **managed foundation only**: set `base_model` to the provider model, omit adapters for pure hosted inference.
  - **hosted foundation + local adapters**: keep adapters on disk and send adapter metadata with the request so the provider overlays your deltas over its model.
  - **prompt-only adapters**: for providers without lora, use `adapter_mode=prompt` to inject adapter prompts instead of weights.
- switching providers is an admin-console change: set `model_backend` and the provider key; model services rebuild without a restart.

### hybrid deployments
- keep `model_backend` on `local_gpu_lora` for on-prem traffic and point select routes or tenants at an api backend via adapter modes or routing policies stored in artifacts.
- filesystem artifacts stay authoritative even with api backends; adapter payloads live under `/srv/liminallm/adapters`.

## model handling at a glance
- base models are frozen. local deployments place them under `/srv/liminallm/models` and the local jax backend keeps them resident; api backends treat `base_model` as a provider-owned model id and never upload local weights.
- adapters live under `/srv/liminallm/adapters/<adapter_id>/adapter.lora` and are loaded or streamed as metadata depending on backend capabilities.
- training and clustering write only adapter weights. the base model on disk or at the provider is untouched.
- when sending requests, set `base_model` to the foundation you want and `adapter_id`/`adapter_mode` to pick the adapter path (local weights, provider-hosted adapters, or prompt patching).

## ops & safety defaults (from the spec)
- observability: metrics and traces should include chat latency/error rates, adapter usage, preference/training counts, and workflow node timings; `/healthz` always answers 200 with a per-dependency breakdown (postgres, redis, filesystem) for humans and dashboards, while `/readyz` is the load balancer probe and returns 503 when postgres or the filesystem is unusable.
- retention: metrics 7–14d, logs 30–90d with payload sampling and pii minimization.
- alerts: latency slo breaches, adapter cache miss spikes (>20%), training failure bursts, ingestion lag over an hour.
- backups: nightly postgres logical backup kept 7d; weekly filesystem snapshot pointers kept 4 weeks; redis is treated as ephemeral (state is recreated from postgres + filesystem artifacts).
- safety rails: content safety classifier on user/assistant text; preference events and training skip disallowed content.

## running multiple replicas
scale out by running the same image behind a load balancer. postgres and `shared_fs_root` are the shared state; nothing in the app assumes it is the only process.

- **probes**: point the balancer at `/readyz`, not `/healthz`. `/readyz` fails a node whose database or filesystem is gone; redis is deliberately excluded so a redis outage degrades every node instead of draining the whole fleet.
- **shared filesystem**: every replica must mount the *same* `shared_fs_root` (nfs/efs or equivalent). adapters, artifacts, and user uploads are written by whichever node handled the request and read by any other.
- **node-local scratch**: `interpreter_scratch_dir` must stay off shared storage — it holds throwaway per-tool-call copies and defaults to the system temp dir.
- **sessions/routing**: sticky sessions are not required. websockets are per-connection, and `POST /chat/cancel` reaches the replica holding the stream over the cluster bus, so a stop button works no matter which node the request lands on.
- **cluster bus**: `cluster_bus_backend` (default `auto`) uses redis pub/sub when redis is reachable and otherwise falls back to postgres `LISTEN`/`NOTIFY`, which is why redis stays optional. force one with `redis`/`postgres`, or set `local` for a single-process deployment to skip peer coordination entirely. the bus is best-effort: if it is down, cancellation falls back to local-only behavior and nothing else changes.
- **background work**: periodic clustering and adapter-prune proposals take a postgres advisory lock, so they run once per interval cluster-wide rather than once per replica. training jobs need no lock — claiming a job is an atomic conditional update, so exactly one replica wins each one.
- **workers per node**: with `model_backend` on `local_gpu_lora` keep `--workers 1` per gpu and scale by adding nodes; api backends scale fine with several workers per node.

## configuration management expectations
- principle: most runtime knobs live in the database and are editable via the admin ui (`/admin`) instead of env vars.
- database-managed settings include session rotation, concurrency caps, rate limits, pagination defaults, token ttls, feature flags (mfa/signup), training worker toggles, smtp/oauth and url settings, voice defaults, model backend/path, rag mode, embedding model id, and tenant/jwt claims.
- environment holds `DATABASE_URL` and four things that are not settings (`BUILD_SHA`, `TEST_MODE`, `EMBEDDING_VECTOR_DIM`, `EXTRACT_READER_PLUGINS`). everything else is in the database, so replicas cannot disagree and nothing needs a redeploy to change. see `docs/CONFIGURATION.md`.

## reverse proxy + os notes
- nginx config sits at `nginx.conf`; enable the compose `nginx` service with the `production` profile or adapt for your host tls.
- openbsd operators: see `deploy/openbsd/` for service user setup, rc scripts, relayd/httpd, and `acme-client` tls.

## troubleshooting quick hits
- health: `curl http://localhost:8000/healthz`.
- migrations: rerun `scripts/migrate.sh` if schema drift bites.
- permissions: ensure write access to `shared_fs_root` and authenticated connections to redis/postgres.
- logs: `docker compose logs -f app` or your process supervisor for startup clues.

