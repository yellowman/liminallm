# Stub and skeletal implementations

This project ships several placeholder components intended to keep the kernel lightweight in constrained environments. Replace them with production-grade implementations before exposing the stack broadly.

## Local JAX + LoRA backend (`liminallm/service/model_backend.py`)
- Status: echo-only backend that never loads adapters or models; described in-depth in `docs/jax_backend.md`.
- Path forward: implement checkpoint loading, adapter merging, prompt/tokenization parity, batching, safety limits, and persistence as outlined in the backend doc.

## Jsonschema shim (`jsonschema/__init__.py`)
- Status: tiny validator covering only required-field checks used by tests.
- Path forward: depend on the upstream `jsonschema` package (or another full validator) once the runtime image can install optional dependencies; re-enable full schema validation in artifact handling.

## Auth dependency placeholder (`liminallm/api/routes.py`)
- Status: FastAPI dependency that trusts a session header and lacks token signing, refresh, and role enforcement.
- Path forward: switch to signed access/refresh tokens, enforce per-route scopes, and integrate device-binding/MFA checks per SPEC §13 before multi-tenant use.

## Voice service stub (`liminallm/service/voice.py`)
- Status: no real ASR/TTS; text is echoed from bytes and synthesis writes plain text files.
- Path forward: wrap a streaming ASR/TTS provider or local model, add per-user quotas, and store generated audio in durable object storage with audit metadata.

## Preference event persistence (`liminallm/storage/postgres.py`)
- Status: feedback events are cached on the filesystem instead of a Postgres table and lack vector indexes/retention.
- Path forward: add a `preference_event` table with embeddings, conflict handling, and retention policies; wire it through migrations and the admin UI.

## Runtime config loader (`liminallm/storage/postgres.py`)
- Status: returns an empty dict because instance-level config patches are not yet persisted.
- Path forward: introduce an `instance_config` table, enforce versioned reads, and attribute sources (UI vs. drift detection) to support safe rollout and rollback.
