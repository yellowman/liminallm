# Issue audit: first-turn report coverage

This note tracks the status of the issues listed in the first user report.

- **Admin artifact access override**: `_get_owned_artifact` now allows admins to bypass user ownership checks and supplies role-specific error messages. 【liminallm/api/routes.py†L225-L238】
- **Chunk ID handling**: `/contexts/{context_id}/chunks` raises a server error when a chunk ID is missing instead of silently coercing `None` to `0`, and returns integer IDs. 【liminallm/api/routes.py†L1270-L1284】
- **OAuth state cleanup**: `complete_oauth` clears cached and in-memory state for all failure paths before returning. 【liminallm/service/auth.py†L196-L222】
- **SQL WHERE construction**: Postgres artifact and chunk queries now build WHERE clauses safely without f-strings. 【liminallm/storage/postgres.py†L1332-L1359】【liminallm/storage/postgres.py†L1917-L1945】
- **Admin user request validation**: Admin user creation enforces `EmailStr`; role updates are limited to literal roles. 【liminallm/api/schemas.py†L326-L347】
- **Pagination constraints**: Artifact listing uses validated `page`/`page_size` parameters with bounds. 【liminallm/api/routes.py†L718-L724】
- **Config decision validation**: Config patch decisions accept only `approve` or `reject`. 【liminallm/api/schemas.py†L281-L286】
- **Voice transcription safety**: Chat voice handling tolerates `None` transcripts before accessing fields. 【liminallm/api/routes.py†L640-L654】
- **OAuth link failures**: Linking provider failures log errors and propagate exceptions. 【liminallm/service/auth.py†L233-L241】
- **WebSocket chat hardening**: WebSocket chat adds idempotency guards, rate limits, and broader exception handling to avoid duplicate responses and crashes. 【liminallm/api/routes.py†L1331-L1345】【liminallm/api/routes.py†L1398-L1418】
- **Admin rate limits**: Admin create/set-role/delete operations enforce rate limits. 【liminallm/api/routes.py†L420-L470】
- **Error message clarity**: Ownership helpers return specific forbidden messages for conversations, contexts, and artifacts. 【liminallm/api/routes.py†L219-L238】
- **Session meta validation**: Session metadata must be a JSON-serializable dictionary before persistence. 【liminallm/storage/postgres.py†L1145-L1158】
- **Tokenizer load errors surfaced**: Training service raises tokenizer load failures instead of swallowing them. 【liminallm/service/training.py†L49-L66】
- **Artifact schema validation**: Artifact create/patch validate schema structure, key types, JSON serializability, and type/kind consistency. 【liminallm/api/routes.py†L975-L1011】
- **File upload path safety**: File uploads enforce safe joins and resolved path checks within the user directory. 【liminallm/api/routes.py†L1153-L1169】

All issues in the initial report are currently addressed in code; no outstanding items remain.
