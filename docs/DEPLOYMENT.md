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
- copy base weights into `/srv/liminallm/models` (`config.json` + `*.safetensors` + tokenizer files; a checkpoint that exists but will not load fails requests rather than degrading to the synthetic stand-in).
- adapters live under `/srv/liminallm/adapters/<adapter_id>/vNNNN/params.json`, one directory per trained version. serving reads **only** the version the artifact's `current_version` names — a loose `params.json`, a `latest` pointer, or the newest directory on disk are not servable state, so hand-placing weights does nothing until an artifact records the version. the directory is named for the adapter that owns it; an explicit `fs_dir` may move it, never rename it to another adapter's.
- gpu prep: install the matching jax gpu wheel (cuda/rocm), verify `nvidia-smi` sees the card, and keep drivers + cuda in `$LD_LIBRARY_PATH`.
- run: `python -m uvicorn liminallm.app:app --host 0.0.0.0 --port 8000 --workers 1` (jax likes fewer workers). callers do not choose adapters: a chat request carries no `adapter_id` or `adapter_mode`, and the router selects and gates adapters internally from policies and clusters. this backend serves `local` and `hybrid` adapters as weights over the frozen base, carries `prompt` adapters through their instructions, and refuses `remote` — a provider-hosted adapter reaching it means the routing hand-off is broken, not that it should improvise.
- the base model remains immutable; training writes only adapter weights.

### api backend (remote inference)
- set `model_backend` (default `openai`) and `adapter_openai_base_url` in the admin console; the provider key goes in its own admin setting, or the matching `<PROVIDER>_API_KEY` environment variable as a fallback.
- calls go out to the remote model id you pass as `base_model`; adapters travel as provider adapter ids, as a fine-tuned model id, or as prompt text, depending on what the provider supports.
- scenarios:
  - **managed foundation only**: set `base_model` to the provider model; with no adapters routed this is pure hosted inference.
  - **provider-hosted adapters** (`mode: remote`): the provider holds the weights. the adapter records `remote_adapter_id` or `remote_model_id`, and the backend sends it — as an `adapter_id` parameter with its gate for a multi-lora provider, or as the model id where one fine-tune serves the request. locally trained weights on disk are **not** usable this way: an api backend cannot apply your `params.json`, and §5.0.1's matrix marks `local` incompatible with it.
  - **prompt-only adapters**: for providers without lora, route an adapter whose artifact records `mode: prompt` — or a hybrid one, which falls back to its prompt wherever its weights cannot apply. the instructions are materialized once, by `LLMService`, before any backend runs; backends transport prepared messages and never add a second copy.
- switching providers is an admin-console change: set `model_backend` and the provider key; model services rebuild without a restart.

### hybrid deployments
- keep `model_backend` on `local_gpu_lora` for on-prem traffic and point select routes or tenants at an api backend via adapter modes or routing policies stored in artifacts.
- adapter metadata and routing policy stay authoritative across a backend change — that is what makes the switch a config edit. the weight *payloads* do not travel: local `params.json` under `shared_fs_root` serves the local backend only, and an api backend carries the same adapter as a provider-hosted adapter/model id (`mode: remote`) or as prompt fallback, never by reading those files.

## model handling at a glance
- base models are frozen. local deployments place them under `/srv/liminallm/models` and the local jax backend keeps them resident; api backends treat `base_model` as a provider-owned model id and never upload local weights.
- adapters live under `/srv/liminallm/adapters/<adapter_id>/vNNNN/`, and a version becomes servable only when an artifact's `current_version` names it. the local backend reads that version's `params.json`; api backends never receive it, and carry the adapter as a provider adapter id or as prompt text instead.
- training and clustering write only adapter weights. the base model on disk or at the provider is untouched.
- requests do not pick adapters. `base_model` and `model_backend` are admin settings, and the router chooses and gates adapters per turn from policies, clusters and the user's own personas — which is what makes the same conversation portable across backends without the caller knowing which one answered.

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
- **`shared_fs_root/shared` is not a drop box**: per SPEC §18 a path there is reachable only when an artifact whose visibility is `shared` or `global` covers it, never because a user can spell the pathname. no route currently mints such an artifact, so files placed there by hand are not ingestible by anyone — put shared material in the owning user's area and share it as an artifact instead.
- **kernel confinement is a hard requirement**: every tool call runs in a spawned worker that confines itself before it does anything (SPEC §18), and there is no unconfined fallback — a host that cannot provide the boundary runs no tools at all, chat included. on linux that means user namespaces: `/proc/self/uid_map` present and `kernel.unprivileged_userns_clone` not set to `0`. under docker the default seccomp profile is enough; `--security-opt seccomp=unconfined` is **not** needed and neither is `--privileged`. openbsd uses `unveil`/`pledge` and needs nothing extra. it costs about 6ms of a roughly 75ms worker start, so it is not a tuning knob. a host without it reports `worker_unconfined` on every tool call rather than degrading quietly — which is the intended failure, because the alternative is a process holding `DATABASE_URL` and an open network while being called contained.
- **sessions/routing**: sticky sessions are not required. websockets are per-connection, and `POST /chat/cancel` reaches the replica holding the stream over the cluster bus, so a stop button works no matter which node the request lands on.
- **cluster bus**: `cluster_bus_backend` (default `auto`) uses redis pub/sub when redis is reachable and otherwise falls back to postgres `LISTEN`/`NOTIFY`, which is why redis stays optional. force one with `redis`/`postgres`, or set `local` for a single-process deployment to skip peer coordination entirely. the bus is best-effort: if it is down, cancellation falls back to local-only behavior and nothing else changes.
- **background work**: periodic clustering and adapter-prune proposals take a postgres advisory lock, so they run once per interval cluster-wide rather than once per replica. training jobs need no lock — claiming a job is an atomic conditional update, so exactly one replica wins each one.
- **workers per node**: with `model_backend` on `local_gpu_lora` keep `--workers 1` per gpu and scale by adding nodes; api backends scale fine with several workers per node.

## upgrades: rolling by default, one exception
ordinary upgrades roll: replace replicas one at a time behind the balancer, and old and new images serve side by side while the fleet turns over. nothing in the app requires a flag day for a normal deploy.

the one exception is a change to the **circuit-breaker failure-history representation** (SPEC §18.3). the breaker's failure history is ephemeral, per-window redis state, and it has two on-disk shapes: the legacy plain counter at `circuit:*:failures` and the rolling-window sorted set at `circuit:*:failures:v2`. they are **independent ledgers** — a key is one shape or the other, never both — so if old and new replicas serve at once, a success on one shape does not clear the other, and five real failures can split across the two and trip neither. a representation change must therefore be a coordinated reset, not a rolling deploy:

1. stop admitting traffic to the replicas on the old representation (drain them from the balancer);
2. wait for their in-flight requests to finish;
3. purge the **superseded** representation's failure history: `python scripts/reset_breaker_history.py --representation legacy` when cutting over to the rolling window (`--representation v2` when rolling back). add `--dry-run` first to see the count. this retires the old keys rather than trusting their ≤60s ttl — a rollback inside that window would otherwise find the old counter still live and resume counting from it, opening a breaker the reset was meant to have cleared;
4. start the replicas on the new representation;
5. do **not** overlap the two representations at any point;
6. the shared `circuit:*:open` cooldown keys are intentionally left alone — an already-open breaker stays open across the reset, because a representation change does not make a proven-unhealthy tool healthy;
7. a rollback is the same procedure in reverse: drain the new replicas, purge `--representation v2`, start the old ones.

losing at most one 60s window of failure history at the boundary is acceptable — redis breaker state is ephemeral by definition, and one empty minute is cheaper than two contradictory ledgers.

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

