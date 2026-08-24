# Stub and skeletal implementations

This project ships several placeholder components intended to keep the kernel lightweight in constrained environments. Replace them with production-grade implementations before exposing the stack broadly.

## Local JAX + LoRA backend (`liminallm/service/model_backend.py`)
- Status: **no longer a stub.** `service/transformer.py` is a real decoder-only transformer in plain JAX (RMSNorm, RoPE, grouped-query attention with a KV cache, SwiGLU) loading `config.json` + `*.safetensors` from the model directory, and LoRA matrices apply inside its attention projections. Training uses the same forward pass. See `docs/jax_backend.md`.
- The sinusoidal table this entry used to describe survives in exactly one place: the `absent` checkpoint state, where nothing is on disk. It logs `local_checkpoint_absent`, moves tokens for CI and dev boxes, and does not answer questions. A checkpoint that exists but cannot be served fails requests instead of falling back to it.
- Path forward: batching (generation builds a single example), per-request RNG seeding for sampling other than greedy, and an LRU with a byte budget on the adapter cache.

## Jsonschema shim (`jsonschema/__init__.py`)
- Status: replaced with the upstream `jsonschema` package so artifact schemas run through full Draft 2020-12 validation.
- Path forward: keep schemas synchronized with runtime expectations and surface validation errors in the admin UI.

## Auth dependency (`liminallm/api/routes.py`)
- Status: Routes delegate to `AuthService.authenticate`, which verifies bearer access tokens or cached sessions, enforces admin gating where requested, and respects tenant hints. The service now issues/revokes refresh tokens, supports signup/login with SHA-256 password checks, offers MFA setup/verification, and handles password resets via short-lived Redis tokens. Perimeter controls (device binding, per-route scopes, rate limits, audit propagation) are still thin compared to SPEC §13 expectations.
- Path forward: tighten scope enforcement and device-binding/MFA coverage, add rate limiting on auth endpoints, and thread audit metadata through the handlers.

## Voice service stub (`liminallm/service/voice.py`)
- Status: no real ASR/TTS; text is echoed from bytes and synthesis writes plain text files.
- Path forward: wrap a streaming ASR/TTS provider or local model, add per-user quotas, and store generated audio in durable object storage with audit metadata.

## Preference event persistence (`liminallm/storage/postgres.py`)
- Status: feedback events are stored in the `preference_event` table with embeddings, weights, and optional cluster IDs; writes include deterministic embeddings for clustering and kick off adapter training jobs. Reads/writes flow through the Postgres store rather than filesystem shims.
- Path forward: add vector indexes and retention/aggregation policies, surface the data in the admin UI, and enforce tenant scoping.

## Runtime config loader (`liminallm/storage/postgres.py`)
- Status: reads from the `instance_config` table and returns the `config` column for the `default` row, parsing JSON strings when needed; returns `{}` on parse errors. The loader remains a thin placeholder without typed fields or versioning.
- Path forward: enforce versioned reads and source attribution (UI vs. drift detection) to support safe rollout and rollback.

## Frontend static exposure (`liminallm/app.py`)
- Status: the `/static` mount serves the entire `frontend` directory without authentication, so `/static/admin.html` remains reachable even though the `/admin` route now depends on `get_admin_user`. The admin HTML bundle is still exposed to unauthenticated users via the static path.
- Path forward: remove the admin bundle from the public static mount or wrap static file serving in an auth gate; alternatively, serve admin assets from a separate, protected path.
