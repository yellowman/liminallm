# Agent guidance for liminallm

This file gives the shortest possible orientation; see **SPEC.md** for authoritative detail.

## Core principles (SPEC §§0–1)
- "Small kernel, big data": code stays minimal; behavior lives in artifacts (`adapter.lora`, `workflow.chat`, `policy.routing`, `tool.spec`, `context.knowledge`).
- No hard-coded domains; clusters and labels are emergent (SPEC §7).
- Keep the kernel safe and stable; intelligence is in editable artifacts (SPEC §6, §18).

## Storage & persistence (SPEC §§2–3, §4)
- Primary stores: Postgres (pgvector), Redis, shared filesystem. Artifact and adapter payloads must be filesystem-backed.
- Table outlines for users/auth, conversations/messages, artifacts + versions, semantic clusters, knowledge contexts/chunks, preference/training, and config patches are in SPEC §2.
- Filesystem layout and adapter payload expectations: SPEC §3; Redis usage for sessions/limits/summaries: SPEC §4.

## LLM, routing, and workflows (SPEC §§5, §8–§11)
- Two inference modes: cloud fine-tune endpoints (model IDs) vs adapter servers with `adapter_id`/multi-LoRA (SPEC §5).
- Routing policies are data-driven artifacts with a tiny DSL (SPEC §8); workflows are editable graphs executed by the kernel (SPEC §9).
- ConfigOps / LLM-as-architect expectations live in SPEC §10; memory model and summaries in SPEC §11.

## Auth and APIs (SPEC §§12–13)
- Auth covers session lifecycle, MFA/reset, and WS auth/streaming requirements; follow the envelope/idempotency rules in SPEC §13 when touching APIs.

## Safety, ops, and front-end (SPEC §§14–17)
- Implementation phases and locked kernel behaviors: SPEC §14–§18.
- Safety/resource limits, observability/retention defaults, and backup expectations: SPEC §15.
- Front-end is intentionally thin; see SPEC §17 for LLM-visible client expectations.

## Repo structure & tips
- Entry: `liminallm/app.py` (FastAPI app factory) with routes/schemas under `liminallm/api/`.
- Services (`liminallm/service/`) cover runtime, workflow/router, LLM client, auth, and RAG helpers.
- Storage backends (`liminallm/storage/`) include in-memory, Postgres, and Redis cache; SQL scaffolding in `sql/`, migration script in `scripts/migrate.sh`.
- Default test/validation is `python -m compileall liminallm`.

When in doubt, align changes with SPEC sections above and keep the kernel minimal and data-driven.

## Documentation reminders
- When a task is fully completed, update `README.md` to reflect the finished status.
- If a change affects the spec (requested by the user or clarifying an ambiguous area), edit `SPEC.md` accordingly.
