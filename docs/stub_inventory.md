# Stub and skeletal implementations

This project ships several placeholder components intended to keep the kernel lightweight in constrained environments. Replace them with production-grade implementations before exposing the stack broadly.

## Local JAX + LoRA backend (`liminallm/service/model_backend.py`)
- Status: lightweight JAX path that tokenizes requests, loads cached adapter weights, and applies paired `.A`/`.B` matrices. The LoRA forward pass now accepts either token IDs or precomputed `(batch, seq, hidden)` embeddings and aligns widths to adapter expectations so training outputs can be reused during inference without shape errors.
- Path forward: add proper base model logits instead of hashed token sampling, and reintroduce batching/safety limits now that the training/inference embedding layouts are reconciled.

## Jsonschema shim (`jsonschema/__init__.py`)
- Status: tiny validator covering only required-field checks used by tests.
- Path forward: depend on the upstream `jsonschema` package (or another full validator) once the runtime image can install optional dependencies; re-enable full schema validation in artifact handling.

## Auth dependency (`liminallm/api/routes.py`)
- Status: Routes now delegate to `AuthService.authenticate`, which validates signed JWT access/refresh tokens, enforces roles (e.g., `/admin`), supports MFA, and falls back to cached sessions. The dependency is functional rather than a stub, but perimeter controls (e.g., device binding and per-route scopes) remain thin compared to SPEC §13 expectations.
- Path forward: tighten scope enforcement and device-binding/MFA coverage, add rate limiting on auth endpoints, and thread audit metadata through the handlers.

## Voice service stub (`liminallm/service/voice.py`)
- Status: no real ASR/TTS; text is echoed from bytes and synthesis writes plain text files.
- Path forward: wrap a streaming ASR/TTS provider or local model, add per-user quotas, and store generated audio in durable object storage with audit metadata.

## Preference event persistence (`liminallm/storage/postgres.py`)
- Status: feedback events are stored in the `preference_event` table with embeddings, weights, and optional cluster IDs; reads/writes flow through the Postgres store rather than filesystem shims.
- Path forward: add vector indexes and retention/aggregation policies, surface the data in the admin UI, and enforce tenant scoping.

## Runtime config loader (`liminallm/storage/postgres.py`)
- Status: reads from the `instance_config` table so database overrides (e.g., admin-set model backend) apply at startup.
- Path forward: enforce versioned reads and source attribution (UI vs. drift detection) to support safe rollout and rollback.

## Frontend static exposure (`liminallm/app.py`)
- Status: the `/static` mount serves the entire `frontend` directory without authentication, so `/static/admin.html` is reachable even though `/admin` depends on `get_admin_user`. This exposes the admin console UI to unauthenticated users.
- Path forward: remove the admin bundle from the public static mount or wrap static file serving in an auth gate; alternatively, serve admin assets from a separate, protected path.
