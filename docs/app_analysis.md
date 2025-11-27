# Application Analysis (post-fix review)

## API Layer (FastAPI entry + routes)
- **CORS now aligns with cookie flows.** The app enables credentials and defaults to a set of localhost origins (overridable via `CORS_ALLOW_ORIGINS`), avoiding the earlier wildcard/credential mismatch while keeping dev usability. Security headers remain applied uniformly. 【F:liminallm/app.py†L6-L48】
- **Static UI failures are logged for operators.** When frontend assets or entrypoints are missing, the server now emits explicit warnings (including missing paths) instead of silent 404s, improving deploy debugging. 【F:liminallm/app.py†L50-L79】

## Runtime Initialization
- **Rate limits respect “disabled” configuration.** `check_rate_limit` now treats `limit <= 0` as unlimited rather than blocking, preventing misconfiguration from bricking endpoints. 【F:liminallm/service/runtime.py†L171-L205】
- **Redis fallback flag works for local/dev.** The runtime honors `ALLOW_REDIS_FALLBACK_DEV` (and `TEST_MODE`) to permit in-memory fallbacks with warnings instead of hard failures, while still enforcing Redis in production by default. 【F:liminallm/service/runtime.py†L17-L71】

## Workflow & LLM Execution
- **Current user prompts are preserved.** `LLMService.generate` always appends the latest user message before injecting context, so multi-turn chats include the active prompt. 【F:liminallm/service/llm.py†L40-L58】
- **Context traces are bounded and deduplicated.** Workflow execution now keeps a unique, capped list of context snippets to avoid runaway growth during loops. 【F:liminallm/service/workflow.py†L62-L118】
- **Tool execution reuses a shared executor.** A process-level thread pool backs tool handlers instead of spawning per-call executors, reducing thread churn and leakage risk; it shuts down on engine cleanup. 【F:liminallm/service/workflow.py†L20-L38】【F:liminallm/service/workflow.py†L420-L457】【F:liminallm/service/workflow.py†L485-L496】

## Retrieval-Augmented Generation (RAG)
- **Context access requires an authenticated user.** `_allowed_context_ids` now returns nothing without a `user_id`, closing the prior anonymous access path and keeping tenant/user checks intact. 【F:liminallm/service/rag.py†L63-L97】
- **Hybrid retrieval shares limits across contexts.** Local hybrid mode distributes limits per context before truncation so no single context starves others, improving fairness without changing scores. 【F:liminallm/service/rag.py†L112-L142】

## Overall Consistency
- Redis outages or absent limits no longer halt traffic unintentionally: dev/test fallbacks run in-memory with warnings, and zero/negative limits truly disable throttling. 【F:liminallm/service/runtime.py†L17-L71】【F:liminallm/service/runtime.py†L171-L205】
- Multi-turn chat fidelity improves through reliable prompt inclusion and controlled context payloads, reducing both missed intent and oversized responses. 【F:liminallm/service/llm.py†L40-L58】【F:liminallm/service/workflow.py†L62-L118】
- Operational visibility is stronger via logged frontend asset issues and credential-aware CORS defaults that align with cookie-based auth. 【F:liminallm/app.py†L6-L79】
