# Stub and skeletal implementations

This project ships several placeholder components intended to keep the kernel lightweight in constrained environments. Replace them with production-grade implementations before exposing the stack broadly.

## Local JAX + LoRA backend (`liminallm/service/model_backend.py`)
- Status: lightweight JAX path that tokenizes requests (or hashes tokens when `transformers` is absent), loads cached adapter weights keyed by `params.json` mtime, and applies paired `.A`/`.B` matrices with width alignment. Generation always builds a single-item batch, multiplies LoRA scores by the attention mask, and samples deterministically from a sinusoidal embedding table seeded by the last prompt token.
- Path forward: add proper base model logits instead of hashed token sampling, enforce batch-size/length safety for multi-example calls, and plumb per-request RNG seeding to avoid process-wide determinism when concurrency arrives.

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
