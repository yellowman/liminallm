# adaptive chat system spec  
## (“small kernel, big data” with lora adapters, postgres, redis, filesystem, python, jax)

---

## 0. goals & principles

### 0.1 goals

- **chatgpt-like web interface** with:
  - multi-user, multi-session chat
  - text + (optional) voice interface
- **small models, deeply adapted** (the core bet):
  - the system targets small/self-hosted base models, where LoRA genuinely
    outperforms prompting: behavior baked into weights survives context
    pressure, frees the window for user content, and costs nothing per token.
  - JAX is the primary training and local-serving framework.
  - a frontier model may be used as a **teacher** (distillation, labeling)
    but is never required at inference time.
- **deep, evolving user-specific behavior** via:
  - per-user persona adapters (small, low-stakes: tone and format)
  - per-skill LoRA adapters trained on **pooled** cluster data (§7.3) - a
    single user's feedback is too sparse to train weights on
  - the adapter ladder (§5.5): every skill is born as a prompt, and only
    earns weights when the data justifies it and the eval gate passes
  - natural, emergent domains & skills from usage
- **natural, notebookLM-style grounding** via:
  - user files on a shared filesystem
  - RAG over embedded chunks in Postgres
- **minimal “kernel” code**:
  - core system only knows about generic primitives:
    - artifacts
    - workflows (graphs)
    - routing policies
  - everything else (skills, domains, behaviors) lives as **data** the LLM can read / write
- **storage stack**:
  - PostgreSQL (primary store + pgvector)
  - Redis (sessions / hot cache / rate limits)
  - filesystem (files, adapters, artifacts)

### 0.2 design principles

1. **small kernel, big data**
   - core is boring, tiny, and stable.
   - behavior is expressed in self-describing artifacts (JSON + text), versioned and editable by LLM under guardrails.

2. **no hard-coded domains**
   - no enums like `DEBUGGING`, `WRITING`.
   - use embeddings + clustering + natural-language labels to create emergent domains and skills.

3. **LLM as architect (under guardrails)**
   - LLM can propose edits to:
     - routing policies
     - workflows
     - skill definitions
   - changes go through a ConfigOps pipeline with validation + (optionally) human approval.

4. **continuous personalization**
   - preference events → adapter training jobs → LoRA weight updates → router state updates.
   - minimal disruption, incremental learning only.

---

## 1. high-level architecture

### 1.1 components

- **clients**
  - Web SPA (React/Vue/Svelte — not critical here).
  - Mobile app (native or cross-platform wrapper) using same HTTP/WebSocket API.
- **edge**
  - API gateway:
    - TLS termination
    - authentication (JWT or session)
    - rate limiting (uses Redis)
- **backend services (can be one monolith initially)**
  - Auth Service
  - Chat Orchestrator
  - Artifact Service
  - Router Service
  - Workflow Engine
  - LLM Inference Service (JAX, LoRA)
  - Knowledge/RAG Service
  - File Service (filesystem abstraction)
  - Preference & Training Service
  - Clusterer & Skill Discovery
  - ConfigOps Service
- **data stores**
  - PostgreSQL
  - Redis
  - filesystem

for a minimal v1, all “services” can be modules inside a single Python app with clear boundaries.

---

## 2. data model (postgres schemas)

### 2.1 users & auth

```sql
CREATE TABLE app_user (
  id              UUID PRIMARY KEY,
  email           CITEXT UNIQUE NOT NULL,
  handle          TEXT,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  plan_tier       TEXT NOT NULL DEFAULT 'free',
  is_active       BOOLEAN NOT NULL DEFAULT TRUE,
  meta            JSONB
);

CREATE TABLE user_auth_credential (
  user_id         UUID PRIMARY KEY REFERENCES app_user(id) ON DELETE CASCADE,
  password_hash   TEXT,          -- null if external oauth only
  password_algo   TEXT,          -- 'argon2id', etc.
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  last_updated_at TIMESTAMPTZ
);

CREATE TABLE user_auth_provider (
  id              BIGSERIAL PRIMARY KEY,
  user_id         UUID NOT NULL REFERENCES app_user(id) ON DELETE CASCADE,
  provider        TEXT NOT NULL, -- 'google', 'github', 'oidc:foo'
  provider_uid    TEXT NOT NULL,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (provider, provider_uid)
);

CREATE TABLE user_settings (
  user_id         UUID PRIMARY KEY REFERENCES app_user(id) ON DELETE CASCADE,
  locale          TEXT,
  timezone        TEXT,
  default_voice   TEXT,
  default_style   JSONB,         -- tone, verbosity, etc.
  flags           JSONB          -- experimental toggles, etc.
);

CREATE TABLE auth_session (
  id              UUID PRIMARY KEY,
  user_id         UUID NOT NULL REFERENCES app_user(id) ON DELETE CASCADE,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  expires_at      TIMESTAMPTZ NOT NULL,
  user_agent      TEXT,
  ip_addr         INET,
  meta            JSONB
);

-- long-lived bearer keys for the served responses api (§13.1). only the
-- sha-256 lands here; the plaintext is shown once at mint time. revocation
-- is a tombstone (revoked_at), keeping the audit trail.
CREATE TABLE user_api_key (
  id            UUID PRIMARY KEY,
  user_id       UUID NOT NULL REFERENCES app_user(id) ON DELETE CASCADE,
  name          TEXT NOT NULL DEFAULT '',
  key_hash      TEXT NOT NULL UNIQUE,
  prefix        TEXT NOT NULL,     -- enough to recognize, never to use
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  last_used_at  TIMESTAMPTZ,
  revoked_at    TIMESTAMPTZ
);
```

### 2.2 conversations & messages

```sql
CREATE TABLE conversation (
  id              UUID PRIMARY KEY,
  user_id         UUID NOT NULL REFERENCES app_user(id) ON DELETE CASCADE,
  title           TEXT,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  status          TEXT NOT NULL DEFAULT 'open', -- 'open','archived'
  active_context_id UUID, -- references knowledge_context(id)
  meta            JSONB
);

CREATE TABLE message (
  id              UUID PRIMARY KEY,
  conversation_id UUID NOT NULL REFERENCES conversation(id) ON DELETE CASCADE,
  sender          TEXT NOT NULL,             -- 'user','assistant','system','tool'
  role            TEXT NOT NULL,             -- LLM role
  content         TEXT NOT NULL,             -- linearized
  content_struct  JSONB,                     -- structured segments (code blocks, citations)
  seq             INT NOT NULL,              -- per-conversation order
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  token_count_in  INT,
  token_count_out INT,
  meta            JSONB
);

CREATE UNIQUE INDEX ON message (conversation_id, seq);
```

special “summary” messages can be `sender='system', role='system', meta.summary=true`.

**`content_struct` schema (structured message payload)**

- Stored alongside `content` to avoid reparsing plain text; kept lightweight so renderers and downstream agents can rely on a consistent shape.
- Expected shape:

```json
{
  "segments": [
    {"type": "text", "text": "...", "start": 0, "end": 42, "tags": ["markdown"]},
    {"type": "code", "text": "print('hi')", "language": "python"},
    {"type": "citation", "text": "...", "source_id": "doc-123", "chunk_id": "chunk-5", "score": 0.87},
    {"type": "tool_call", "name": "lookup_customer", "arguments": {"id": "42"}, "result": {"status": "ok"}, "duration_ms": 123},
    {"type": "attachment", "kind": "image", "uri": "s3://...", "mime": "image/png", "description": "rendered chart"},
    {"type": "redaction", "text": "[redacted]", "reason": "policy", "policy": "p0"}
  ],
  "summary": {"highlights": "optional per-turn summary"}
}
```

- Segment intents:
  - **text/code/citation**: renderable spans with optional source and similarity scores for RAG provenance.
  - **tool_call**: capture name/args/result/status/timing to support replay and audit.
  - **attachment**: structured references to non-text payloads (images, audio, tables).
  - **redaction**: mark filtered spans and the policies that applied to them for safety reviews.
- Callers may attach custom annotations under `meta` inside each segment; storage normalizes to the keys above and drops invalid structures.

### 2.3 artifacts (generic primitives)

single generic table for everything that is “configuration-like”:

```sql
CREATE TABLE artifact (
  id              UUID PRIMARY KEY,
  owner_user_id   UUID REFERENCES app_user(id),  -- null for global/shared
  type            TEXT NOT NULL,                 -- e.g. 'adapter','workflow','policy','tool','memory'
  name            TEXT NOT NULL,
  description     TEXT,
  schema          JSONB NOT NULL,                -- typed metadata
  fs_path         TEXT,                          -- optional link to files on filesystem
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  visibility      TEXT NOT NULL DEFAULT 'private', -- 'private','shared','global'
  meta            JSONB
);
```

payloads for artifacts (JSON schemas, adapter weights) are additionally written under the shared filesystem root so they can be
mounted by inference/training jobs without round-trips through the database.

**artifact versions** for history & rollbacks:

```sql
CREATE TABLE artifact_version (
  id              BIGSERIAL PRIMARY KEY,
  artifact_id     UUID NOT NULL REFERENCES artifact(id) ON DELETE CASCADE,
  version         INT NOT NULL,
  schema          JSONB NOT NULL,
  fs_path         TEXT,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  created_by      TEXT NOT NULL, -- 'system','user','llm'
  change_note     TEXT
);

CREATE UNIQUE INDEX ON artifact_version (artifact_id, version);
```

#### artifact “kinds” (in `schema.kind`)

- `adapter.lora` (LoRA adapter metadata)
- `workflow.chat` (graph-based workflow)
- `policy.routing` (routing policy; artifact type `policy`, validated like the other kinds — it had no schema entry, so `POST /v1/artifacts {type: "policy"}` answered "unknown artifact type" and routing-as-data had no way to get its data in)
- `tool.spec` (declarative tool definitions)
- `memory.summary` (long-term memory summaries)
- `context.knowledge` (knowledge/RAG context definitions)
- others later.

### 2.4 semantic clusters (emergent domains/skills)

clusters are *data*-driven, not enums.

```sql
CREATE TABLE semantic_cluster (
  id              UUID PRIMARY KEY,
  user_id         UUID,     -- null for global cluster
  centroid        VECTOR,   -- pgvector
  size            INT NOT NULL,
  label           TEXT,     -- LLM-generated short label
  description     TEXT,     -- longer natural language explanation
  sample_message_ids UUID[], -- optional
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  meta            JSONB
);
```

clusters can be referenced from artifacts via `schema.cluster_id`.

### 2.5 knowledge & RAG

```sql
CREATE TABLE knowledge_context (
  id              UUID PRIMARY KEY,
  owner_user_id   UUID NOT NULL REFERENCES app_user(id) ON DELETE CASCADE,
  name            TEXT NOT NULL,
  description     TEXT,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  meta            JSONB
);

CREATE TABLE context_source (
  id              UUID PRIMARY KEY,
  context_id      UUID NOT NULL REFERENCES knowledge_context(id) ON DELETE CASCADE,
  fs_path         TEXT NOT NULL,  -- directory or file
  recursive       BOOLEAN NOT NULL DEFAULT TRUE,
  meta            JSONB
);

CREATE TABLE knowledge_chunk (
  id              BIGSERIAL PRIMARY KEY,
  context_id      UUID NOT NULL REFERENCES knowledge_context(id) ON DELETE CASCADE,
  fs_path         TEXT NOT NULL,
  chunk_index     INT NOT NULL,
  content         TEXT NOT NULL,
  embedding       VECTOR NOT NULL,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  meta            JSONB
);

CREATE INDEX knowledge_chunk_context_idx ON knowledge_chunk (context_id);
CREATE INDEX knowledge_chunk_embedding_idx ON knowledge_chunk
USING ivfflat (embedding) WITH (lists = 100);
-- the lexical half of hybrid retrieval; 'simple' takes no stemming, and the
-- two-arg to_tsvector is IMMUTABLE, which is what makes it indexable.
CREATE INDEX knowledge_chunk_content_fts_idx ON knowledge_chunk
USING gin (to_tsvector('simple', content));

-- late interaction: several vectors per chunk, compared at query time by
-- MaxSim, so a chunk is found on its best part rather than its average.
-- written only when rag_late_interaction is on.
CREATE TABLE knowledge_chunk_vector (
  id              BIGSERIAL PRIMARY KEY,
  chunk_id        BIGINT NOT NULL REFERENCES knowledge_chunk(id) ON DELETE CASCADE,
  segment_index   INT NOT NULL,
  content         TEXT NOT NULL,
  embedding       VECTOR NOT NULL,   -- same encoder, so same width as above
  meta            JSONB
);
CREATE INDEX knowledge_chunk_vector_chunk_idx ON knowledge_chunk_vector (chunk_id);
CREATE INDEX knowledge_chunk_vector_embedding_idx ON knowledge_chunk_vector
USING ivfflat (embedding) WITH (lists = 100);
CREATE UNIQUE INDEX knowledge_chunk_vector_segment_idx
ON knowledge_chunk_vector (chunk_id, segment_index);
```

#### ingestion pipeline (knowledge → chunks)

- **parsers**: text, markdown, PDF (pdftotext), HTML (readability). Additional parsers can be registered via `artifact` type `tool.spec`.
- **chunking**: sliding window token-based splitter (e.g., 300–500 tokens with 50 token overlap) tuned per file type; store `chunk_index` and offsets.
- **hygiene**: dedupe by file checksum + path; skip binary blobs unless parser registered; enforce max file size per plan tier; optional PII-scrub per context.
- **embedding model** *(revised — the original text assumed the hash fallback was the only encoder)*: the encoder is resolved from the model backend, not pinned to a named local model. when the backend exposes an openai-compatible `/embeddings` client (openai, gemini-compat, vllm/lorax self-hosted), embeddings go through it at the provider's **native** dimensionality; otherwise the kernel's deterministic hash embedding applies. the encoder id is recorded with every vector (`knowledge_chunk.meta.embedding_model_id`, `note.meta`, `message.meta.embedding_model`).
- **`EmbeddingsService.is_semantic`**: the load-bearing honesty flag. hash-embedding cosine is *noise*, not weak signal — so every consumer that would let cosine into a ranking checks this flag and falls back to bm25 alone when it is false. blending noise at any weight is worse than keywords alone. **the flag is passed in, never inferred**: `RAGService` receives it from the runtime alongside the encoder, because a retriever handed only an `embed` callable cannot tell a real encoder from the hash fallback — and for a while the primary rag path could not, and ranked the user's files by hash distance.
- **two spaces, deliberately**:
  - *retrieval space* (rag chunks, notes, message recall): native dimensionality, provider encoder, compared only against vectors carrying the same encoder id.
  - *routing/clustering space* (`preference_event.context_embedding`, `adapter_router_state.centroid_vec`): always the 64-d hash embedding via `deterministic_embedding`. this is intentional — clustering compares vectors across users and months, so it needs a space that is stable and free, and that does not shift when an admin swaps embedding providers.
- **dimension handling is dynamic, never pinned**: retrieval validates that query and chunk share a dimension rather than asserting 64. pinning it to `EMBEDDING_DIM` made every real-encoder query fail validation and silently score 0 — collapsing semantic search to bm25 while appearing to work. a chunk from a different encoder scores 0 rather than being garbage-compared.
- **embedding dimensionality**: 64-d (`EMBEDDING_DIM`) is the *hash-fallback* size and remains mandatory for routing and clustering, where vectors from many contexts are compared in one space.
  **amended:** external providers persist their **native** dimensionality (e.g. 1536) for rag chunks, notes, and message recall. truncating a real 1536-d embedding to 64-d discards most of the signal the encoder exists to provide — obeying the original rule would defeat semantic retrieval. the invariant that actually matters is *never compare vectors from different encoders*: every consumer records the encoder id alongside the vector (`knowledge_chunk.meta.embedding_model_id`, `note.meta`, `message.meta.embedding_model`) and filters on it; a mismatch is treated as "not embedded", so the backfill re-embeds rather than comparing across spaces.
  **verified and fixed:** `knowledge_chunk.embedding` was declared bare `VECTOR` and indexed `USING ivfflat`. reproduced against real pgvector: `ERROR: column does not have dimensions` — so with `ON_ERROR_STOP` (which `scripts/migrate.sh` uses) migrations aborted at 002, and without it the index silently never existed and every similarity search was a sequential scan. the column is now pinned to `EMBEDDING_VECTOR_DIM` (default 1536, use 64 for the hash fallback), passed to psql by `migrate.sh`. there is no upgrade path to get wrong: the numbered migrations are gone in favour of one idempotent `sql/schema.sql` (see §2), because this project has never been deployed and a migration history that reconciles states no database was ever in is fiction with a data-loss hazard attached. a wrong `EMBEDDING_VECTOR_DIM` can no longer corrupt anything quietly either — startup compares the column's dimension against the encoder's and refuses with both numbers and the fix. verified end to end on postgres 16 + pgvector at 1536 and 64.
- **refresh cadence**:
  - watch filesystem path events; enqueue ingestion job on file change.
  - encoder change is handled by *invalidation*, not a sweep: a vector whose recorded encoder id differs from the current one reads as "not embedded", so the normal backfill re-embeds it lazily. no daily job exists — a scheduled re-embed is still open work, and until it lands, old vectors are re-embedded only when something reads them.
- **retrieval strategy** *(rewritten — the original made a single dense query both the whole pipeline and the primary path)*:

  **why more than one channel.** a single vector of dimension `d` bounds how many distinct top-k sets of documents any query can ever return. the bound is geometric, not statistical, so no amount of training data or model size removes it: for `n` documents, `k` relevant, and a score margin `γ`, realizing every `k`-subset needs `d ≥ log C(n,k) / log(1 + 1/γ)` (Weller, Boratko, Naim & Lee, ICLR 2026, *On the Theoretical Limitations of Embedding-Based Retrieval*). that is a floor, and a loose one. optimizing the vectors directly against the test set — no language model, no generalization, the best case that can exist — the same work measures a *critical-n* per dimension: 10 documents at `d=4`, 99 at `d=18`, extrapolating to ~500k at 512, ~4m at 1024, ~250m at 4096. real encoders land far below their own floor: the 46-document probe below is solvable in 12 free dimensions, and real models at 64 dimensions still cannot solve it.

  **what that looks like in practice.** their LIMIT probe is deliberately trivial — documents like "Jon likes quokkas and apples", queries like "who likes quokkas?" — and it breaks state-of-the-art embedders:

  | | recall@2, 46 docs | recall@2, 50k docs |
  |---|---|---|
  | BM25 | 97.8 | 85.7 |
  | GTE-ModernColBERT (multi-vector) | 83.5 | 23.1 |
  | Promptriever 8B @4096 (best single-vector) | 54.3 | 3.0 |
  | Qwen3 Embed @4096 | 19.0 | 0.8 |

  two results matter more than the ranking. first, **it is not domain shift**: fine-tuning on in-domain training data moves recall@10 from ~0 to 2.8, while training on the test set solves it — the task is representationally hard, not unfamiliar. second, **lexical is not the answer either**: rewriting the same corpus with synonyms drops BM25 by ~89% (97.8 → 10.6) while the dense models hold, leaving BM25 *below* most of them. the two channels fail on disjoint inputs. neither is safe alone, which is the entire argument for running both.

  a third result is worth recording because it is about evaluation, not retrieval: LIMIT scores do not correlate with BEIR. an encoder's benchmark position predicts nothing about this failure, so "we use a good embedding model" is not a mitigation.

  **candidate generation — up to three channels in parallel**, each scoped by `context_id` and by the access rules in §12.2 (they share one predicate builder, so a filter cannot go missing from one of them — and the filter that matters is user isolation):
  - *dense (pooled)*: pgvector `ORDER BY embedding <-> $query LIMIT n`, ivfflat, one vector per chunk.
  - *late interaction (multi-vector)*: several vectors per chunk in `knowledge_chunk_vector`, scored by MaxSim. off by default (`rag_late_interaction`); see below.
  - *lexical*: guarded like the late channel, and **checked at startup** — `content_tsv` is a new column on an old table, so the table list the schema verifier walks could not catch its absence, and an install that pulled new code without re-running migrations booted clean and then 500'd every grounded turn. startup now names the column and the fix; retrieval degrades to the vectors if it fails anyway. postgres FTS over `content_tsv`, a **stored generated column** (`to_tsvector('simple', content)`) with a GIN index, ranked by `ts_rank`. stored rather than computed per query: the index serves the match either way, but `ts_rank` in the `ORDER BY` re-tokenizes every matching row, which on a large context was the dominant cost of the channel and was paid on every grounded turn. **the encoder filter does not apply here** — it exists so a query vector is never compared against a chunk from a different encoder, and keyword search compares no vectors. gating it on encoder identity meant flipping `embedding_model_id` made every stored chunk invisible to BM25 too, so retrieval answered nothing at all, for an exact filename as much as for a paraphrase, until the whole corpus was re-ingested by hand. terms are **OR'd** — one absent rare word must not empty the pool. `'simple'` takes no stemming and assumes no language, so an identifier or error code indexes as itself. terms come from the BM25 tokenizer (`\w+` only), so a user query cannot reach the tsquery parser as syntax. measured on 50k chunks: **28.7 ms/query with the GIN index, 239.7 ms without**.
  - pool width is `max(limit × 5, reranker appetite)`, capped at 100. the reranker publishes how much it will read, because a reranker handed exactly the chunks that were going to be returned anyway can reorder them but never reach the one that placed just outside the cut.

  **fusion — weighted reciprocal rank fusion**, `Σ wᵢ / (k + rankᵢ)`, `k=60`, semantic 0.55 / lexical 0.45:
  - **rank, never score.** cosine is bounded and BM25 is not, and BM25's magnitude depends on the pool it was scored against — so any weighted sum needs a normalizer, and every normalizer moves with the pool. the same chunk would score differently depending on what it was ranked beside.
  - rank fusion also expresses something a weighted sum cannot: a chunk **both** channels rank well beats one that only a single channel loves. under a fixed-weight sum, a perfect cosine always beats a perfect BM25 and the lexical channel can never win a head-to-head — which the table above says is exactly backwards.
  - **a channel ranks only what it matched.** zero is silence, not a weak opinion: an arbitrary order over non-matches would otherwise carry the channel's full weight.
  - the lexical pool is reordered by real BM25 before fusion (`ts_rank` was only ever the recall filter). that BM25 scores against the pool, not the corpus, so its IDF is an approximation — sound for ordering a shortlist, which is all it does.

  **pipeline order.** retrievers do recall and return a shortlist; the stages above decide precision — short-chunk filter → rerank → token budget → `limit`. the filter runs first so no rerank slot is spent on a chunk that is about to be dropped; the truncation runs last so the rerank sees more than the answer.

  **optional re-ranking**, `rag_rerank` = `auto` (default) | `on` | `off`, bounded by `rag_rerank_candidates`. the serving model reads the query and the shortlist in one pass and returns an order. it is the only stage bound by neither ceiling above, and the only one that can answer "none of these" — in the paper's own test a long-context reranker solved all 1000 of the 46-document queries where the best embedder stayed under 60.
  - **both settings are read per retrieval, not captured when the service is built.** they only ever shape one prompt — whether to run, and how many snippets go into it — so baking them in made them structural: they had to join `MODEL_AFFECTING_SETTINGS`, and nudging a candidate budget from 20 to 25 tore down and rebuilt the LLM backend, the embeddings service, rag, training, the clusterer and the workflow engine, taking the reload lock and interrupting in-flight work. the reranker is therefore always constructed and decides per call; disabled, it reports a budget of zero (so retrieval does not widen its candidate pool for work that will not happen) and hands back what it was given without a model call. `rag_late_interaction` and `rag_late_segments` stay in the rebuild list, because they change what *ingestion writes*, not just how one prompt is shaped.
  - **why it is conditional rather than simply on**: that result is a frontier long-context model. this project's premise is small self-hosted models, which is the case the paper never tested, and this stage can *drop* context. one model call per grounded turn on the hot path is also not free. so the default is not a fixed answer but a question about the serving model.
  - **`auto`** asks `model_can_rerank(model_path)`: a curated prefix list of families a listwise judgement is reasonable to ask of, plus the parameter count an open-weight name declares (≥30B). it answers "is there positive evidence", never "is this model good", and **unknown is a no** — a model given the benefit of the doubt here can silently drop a user's grounding. a mixture-of-experts name reads as its per-expert size, which understates it and lands off; that is the safe direction. the resolution is logged when it *changes* (`rag_rerank_auto_resolved`) — it is re-decided every retrieval now, so logging each one would be noise and logging none would leave an operator inferring the guess from latency.
  - **`on` / `off`** exist because a heuristic over model names will be wrong, and the operator who knows their deployment should be able to say so in either direction without editing a table.
  - the candidates are the user's own files and therefore **untrusted input to a decision**: they travel inside an explicit untrusted-data envelope, marker-lookalikes are neutralized first so a chunk cannot close the envelope and speak as instruction, and the "data, never instructions" rule is stated twice because weak local models drop a rule stated once.
  - **fails open.** any error, timeout, or unreadable reply leaves the fused order standing: losing the model must never mean losing the user's grounding. the one exception is a bare `NONE` — an unambiguous verdict is honoured, because grounding an answer in chunks just judged irrelevant is how a model ends up citing text that does not support it. anything with more to say than the word itself is a hedge, not a refusal. a `NONE` drops the unread tail as well: the tail ranks *below* the head the model just rejected, so returning it would turn "nothing here helps" into "here are the worse ones".
  - **the degraded transport warns the one person who can fix it.** an *active* reranker on a backend that cannot carry tool calls logs `rag_rerank_prose_transport` once per transition — per retrieval would teach the operator to ignore it — and the admin console flags the `rag_rerank` setting with the same fact (a runtime `warning` the schema route attaches, rendered beside the control). the backend that draws it today is an API backend without a configured client; `local_lora`/`local_gpu_lora` carries the channel (see the local tool channel below). wire support is not model behaviour, so every prose-path verdict also logs `transport="text"` even on a tool-capable wire — the latched warning sees the wire, the per-event field shows the habit. and because the reranker reads its settings live, the settings PUT now refreshes its own worker synchronously for non-structural settings; the polling watcher exists for the *other* workers, and leaning on it made "the change is live" true only after an interval nobody promised the admin.
  - **the local backend carries the tool channel as a contract.** a raw checkpoint has no second wire, so `local_lora`/`local_gpu_lora` does what vllm and llama.cpp do serverside: tools are advertised in a system block (JSON Schema plus the emission format), the model emits `<tool_call>{json}</tool_call>` — the de-facto local standard, the tag Qwen and Hermes templates already emit — and the backend parses that block out of **model output only**, returning the same dict shape as the API backend so consumers cannot tell the transports apart. the property that makes a provider's channel unforgeable by documents survives in one line: input text is never parsed, so a chunk or a fetched page can spell the tag and it still lands in input — only the model writes to the output stream. a malformed block stays visible text rather than becoming a guessed call (digit-harvesting wearing a new tag); call count and block size are bounded before `json.loads`. two honest limits: whether a given checkpoint *emits* the contract is model behaviour, visible as `transport="text"` per event, not a capability the flag can promise; and a parrot-prone small model is one echo away from carrying a document's block into its output, so `neutralize_markers` defangs the tag in untrusted input the same way it defangs the envelope markers. `supports_tools` going true also opens the agent tool loop for local deployments — same channel, same contract, deliberate.
  - **out-of-band verdicts are a rule, not a reranker feature.** any model verdict that gates, deletes, or reorders data must prefer a structured channel over prose parsing whenever the backend provides one — a tool call's arguments arrive in a wire field document text cannot write to, so the parser problem and the forgery problem disappear together. where prose parsing survives (backends without tool support, and today the witness verdict in `judge_pair` and the digest's NARRATIVE/ANCHORS sections), the parser must be **total** — bounded input, no reachable exception, no opinion as the safe result — and the component must fail open to the state before the verdict. the witness and digest parsers are lower stakes (they annotate; they do not delete grounding) but they are the named candidates for this transport when they are next touched.
  - **the verdict arrives out-of-band wherever the backend allows it.** when the serving backend speaks tool calls, the reranker offers one tool (`submit_ranking`, ranking array, empty = none help) and reads the verdict from the `tool_calls` wire field — a channel beside `content` that document text physically cannot write to: a passage (or a whole reply) that spells out a perfect ranking call is still just characters in the content channel. that kills the parser problem and the forgery problem with one structure. one model call either way — a tool-capable model that answers in text anyway falls through to the prose parser on the *same* response, never to a second call. both transports land in one validator, so a verdict means the same thing however it arrived (booleans rejected before int admits them — `true` is not passage 1; picks deduped, range-checked, bounded).
  - **only the answer is parsed, never the working.** several allowlisted families emit a visible reasoning block, and "passage 3 mentions 2024 revenue" is full of digits that are not a ranking. harvesting them counts as a successful parse, so the fail-open path never runs and the user's context is silently reordered by prose. reasoning blocks are stripped — **including an unclosed one**, which is what a reply truncated mid-thought leaves behind — and the answer is then picked by shape rather than position: an ordered list has its markers stripped (the marker is the rank, the number after it is the passage, and **one such line is still a list** — requiring two read `1. Passage 3` as both 1 and 3), else the last line that is only numbers, else the last line with a digit. the `NONE` test runs against the same stripped text, because every reasoning family on the allowlist wraps its verdict in a block and matching the raw reply meant the verdict never registered for exactly those models. "last line with a digit" alone read `2. Passage 1` as the answer `2`, inverting the list.
  - **only what the reranker kept comes back.** the unread tail is not appended after a partial rerank: it ranks below every chunk in the head, so returning it would let fusion ranks 21+ take grounding slots from head chunks the model just read and rejected. that is the same "here are the worse ones" the `NONE` branch refuses, on the far more common path.
  - the candidate list is **one line per passage, whitespace collapsed**. the numbering is what the model replies with, which makes it a forgery target: a chunk containing its own `[1] …` on a line of its own would otherwise add a candidate and make the returned index point somewhere else.
  - **the query is the other seam, and it is outside the envelope.** the model has to read the query as the question, so it cannot live inside the untrusted block — and on the agent path the query is model-authored, which after a tainted web fetch means attacker-influenced. it gets the passage treatment anyway: collapsed to one line (cannot mint a numbered entry, a role marker, or an instruction block), markers neutralized (cannot open or close the envelope), bounded (cannot bury the instructions that follow it).
  - **the prose parser is total.** it is the fallback transport, and its job is to survive whatever a model sends back: reply bounded before any regex touches it; digit runs longer than any valid index skipped before `int()` — CPython refuses str→int past ~4300 digits, and that ValueError sat outside the fail-open guard, so an unreadable reply crashed the very turn fail-open exists to save; ascending spans (`1-3`) expand so the middle passage is ranked rather than deleted, while a descending pair is ambiguous and fails open; a bare `</think>` with no opener (R1-style templates put the opener in the prompt) marks everything before it as reasoning; an explicit trailing answer (`Final: 3, 1`) outranks numbered narration, whose prose digits otherwise pollute the order.
  - **`auto` judges the model that will actually answer** — `LLMService.serving_model`, which resolves `adapter_server_model or base_model` off the *backend*, where they live. reading those attributes off the service found neither and silently fell back to the configured base, so a 7B adapter server inherited `gpt-4o-mini`'s reputation. one property now, because the same pair is resolved by the tokenizer and the context window and had already been copied twice.
  - the capability test matches small-variant names (`mini`, `nano`, `lite`, …) as **whole name parts, never substrings** — `mini` lives inside `gemini` — and a size the name declares beats family membership in both directions, so `gemini-2.0-flash-8b` is refused on its stated size rather than admitted on its prefix.

  **late interaction**, `rag_late_interaction`, **off by default**, bounded by `rag_late_segments` (default 8). multi-vector retrieval is the one architecture in the paper that beats single-vector on both splits — 83.5 against 54.3 recall@2 on the 46-document set, 23.1 against 3.0 on the 50k one — and it is the only entry here that attacks the bound itself rather than working around it.

  - **why it escapes the bound.** the dimension bound is a statement about a score that is *one inner product* between *one* query vector and *one* document vector. MaxSim is not that: the chunk is stored as several vectors and the score is, for each part of the query, its best-matching part of the chunk, summed. the paper says as much in its limitations — the theory "do[es] not hold necessarily for other architectures, such as multi-vector models".
  - **what it fixes, concretely.** a pooled embedding has to answer for the whole chunk at once, so a chunk covering two subjects lands between them and is the best match for neither. `tests/test_late_interaction.py` asserts both halves of that: pooled-only similarity returns the near-miss chunk first, and the same corpus with segments kept separate returns the right one.
  - **two stages, as every multi-vector retriever does it.** each part of the query gathers candidates by nearest *segment* — so a chunk qualifies on its best part, not its average — and the candidates are then scored exactly against *all* of their segments. approximate search decides who is considered; it never decides the order.
  - **this is not ColBERT, and must not be read as carrying its numbers.** segments here are sentence-sized, embedded by the same encoder as everything else, because that encoder is reached through an OpenAI-compatible `/embeddings` endpoint and such an endpoint returns one vector per input — it cannot return per-token vectors. what carries over is the mechanism, not the granularity, at roughly an order of magnitude less storage than per-token would cost. **the seam is the encoder**: a real late-interaction model replaces `segment_text` and the embed call without touching the storage, the candidate generation, or the scoring.
  - **weights.** late leads at 0.55 and the pooled vector *steps back to 0.25* rather than out — it is the same signal read less precisely, so it should not vote twice at full strength, but a whole-chunk vector still says something no single best part does. lexical is unchanged at 0.45.
  - **coverage is not retroactive.** segment vectors are written at ingestion, so turning this on covers new content only; existing corpora need re-ingesting. a chunk without segments is *unranked* by this channel, never penalised by it — the same silence rule as everywhere else. that also means a partly-covered corpus tilts toward the covered part, which is the honest cost of enabling it without a backfill, and a backfill job is open work.
  - **it requires a real encoder**, for the same reason the dense channel does: MaxSim over hash vectors is noise with extra steps. `late_interaction` is silently false when `is_semantic` is false.
  - **MaxSim normalizes once per vector and compares by dot product.** a general cosine per pair re-derived both norms, copied both vectors and rescanned them for NaN on every (query x segment x candidate) comparison — 0.44 s per retrieval at the shipped defaults and 2 s at the candidate cap, on a request thread, before the answering model was called. same arithmetic, 6x less of it.
  - **segments are embedded one batch per chunk**, not one call per segment: `/embeddings` takes an array, and a 500-chunk file at eight segments each is 4000 sequential round trips otherwise.
  - **the channel fails to itself.** the read path is guarded like the write path: this is an addition to ranking, so its failure costs its own contribution and nothing else. the setting is hot-reloadable, so without the guard the first operator to enable it on a database that never had `sql/schema.sql` re-applied would break every chat turn that touches rag rather than losing the part of ranking that is new. indexing stops after the first failure for the same reason it is guarded — a missing table or a width mismatch is structural, and carrying on would buy `segments x remaining chunks` embeddings and throw every one away.
  - **the query has its own segmentation floor.** the chunk threshold is tuned for hundreds of words, and applying it to a query folded every real question back into one segment — so the multi-vector query side never ran, `share` was the whole pool, and the channel quietly became the single-vector recall its own design rejects.
  - each query part gets a **share** of the candidate pool, not first-come. a single overall cap is spent by the first vector, and the first vector is the whole query — which collapses candidate generation back to single-vector recall and leaves MaxSim reordering what a pooled vector had already found.
  - **cost.** one embedding call per segment at ingestion (so up to `rag_late_segments`× the ingestion cost) and one row per segment in an ivfflat index. that is the reason it is off by default, and the reason the segment count is capped rather than per-token.

  **dimensionality.** retrieval vectors persist at the provider's **native** width (`EMBEDDING_VECTOR_DIM`, default 1536). do not truncate them: the same work shows recall falling monotonically with dimension for every model tested, and truncation without matryoshka training is worse still. the 64-d hash space is for routing and clustering only and never ranks a retrieval.

  **where this system is most exposed.** the paper's own difficulty metric is qrel graph density — how often one document is relevant to many queries, and how much queries share documents. LIMIT scores 0.085 density / 28.5 average query strength against ≤0.026 / ≤0.6 for NQ, HotpotQA, SciFact and FollowIR. the parts of this system that look like the hard end of that scale are the notes vault and the witness (§ notes), where the task *is* relating documents to each other and a hub note is relevant to many questions — not chat RAG over a handful of uploaded files, which sits at the easy end.

  - return chunk text + `fs_path` for citation; orchestrator can ask LLM to cite paths.
  - optional dev fallback: in-process BM25 + cosine over a bounded candidate window (`RAG_MODE=local_hybrid`), fused by the same rule, intended for tests or tiny corpora when pgvector is absent.
  - **ranking precedence (applies to rag, notes, and conversation recall alike):** *(amended — the rule was "bm25 is the fallback and the tie-breaker, never the peer", which no weighting of a rank fusion can express, and which the evidence above contradicts)* semantic leads and lexical is weighted just below it, but lexical is a **peer that can win**: when it is the channel that actually matched, it takes the top slot. **without** a real encoder the semantic channel does not speak at all — it is not a weaker vote, it is noise. hash-embedding cosine must never enter a ranking; `EmbeddingsService.is_semantic` is the flag every consumer checks, and every retriever is now passed it rather than left to guess.
  - **a fused score is an ordering, not a measurement.** rank fusion packs its scores together by construction: a two-channel first place is about 0.016, and scaling to the theoretical ceiling only moves the whole set into 0.90–1.00. so it is never published as a similarity. `/v1/notes/search` returns a `rank`, and the witness report's `similarity` is a real cosine — the same thing `vault_sweep` means by that name, which it previously contradicted from the same module.
  - **one mechanism, three callers.** rag, notes search (`search_notes`) and conversation recall (`rank_turns`) all fuse by rank through the same `service/ranking.py`, and all three honour the `is_semantic` guard. the weighted sums they used before are gone: notes weighted *lexical* 0.6 against the rule above, and recall had to score an un-embedded turn as a literal zero — which a weighted sum then held against it, where rank fusion simply reads it as a turn that channel never ranked. recall keeps its `semantic_weight` setting; it is now the semantic channel's fusion weight rather than a blend coefficient.
  - **a miss is a result.** with no real encoder and no lexical match there is nothing honest to ground on, and retrieval returns empty rather than the nearest hash vectors. arbitrary chunks read to the model as evidence, and it will cite them.
  - baseline kernel ships with a deterministic hashing-based embedding fallback (no external model dependency) shared across RAG/routing/clustering so chunks always have non-empty vectors. those vectors are for routing and clustering; they never rank a retrieval.

### 2.6 preferences & training

```sql
CREATE TABLE preference_event (
  id                 BIGSERIAL PRIMARY KEY,
  user_id            UUID NOT NULL REFERENCES app_user(id) ON DELETE CASCADE,
  conversation_id    UUID NOT NULL REFERENCES conversation(id) ON DELETE CASCADE,
  message_id         UUID NOT NULL REFERENCES message(id) ON DELETE CASCADE,
  created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
  explicit_signal    TEXT,         -- 'like','dislike','always','never', etc.
  score              DOUBLE PRECISION, -- normalized [-1,1]
  context_embedding  VECTOR NOT NULL,  -- embedding of situation
  context_text       TEXT,         -- optional raw snippet of the surrounding exchange
  cluster_id         UUID,         -- link to semantic_cluster
  meta               JSONB
);

CREATE TABLE adapter_router_state (
  artifact_id       UUID PRIMARY KEY REFERENCES artifact(id) ON DELETE CASCADE, -- adapter artifact
  centroid_vec      VECTOR,        -- EMA of context embeddings that trained this adapter
  usage_count       BIGINT NOT NULL DEFAULT 0,
  success_score     DOUBLE PRECISION DEFAULT 0.0, -- e.g. running avg of feedback
  last_used_at      TIMESTAMPTZ,
  last_trained_at   TIMESTAMPTZ,
  meta              JSONB
);

CREATE TABLE training_job (
  id                 UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  adapter_id         UUID NOT NULL REFERENCES artifact(id) ON DELETE CASCADE,
  user_id            UUID NOT NULL REFERENCES app_user(id) ON DELETE CASCADE,
  created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
  status             TEXT NOT NULL DEFAULT 'queued', -- 'queued','running','succeeded','failed'
  num_events         INT,
  loss               DOUBLE PRECISION,
  dataset_path       TEXT,
  new_version        INT,
  preference_event_ids UUID[],
  meta               JSONB
);
```

**preference_event → dataset → tokenized batches (single-adapter pipeline)**

- fetch positive `preference_event` rows by `user_id` (optionally filtered by `adapter_id`).
- reconstruct prompts from recent `message` rows in the linked `conversation` (limit ~200, keep last 50 turns).
- target text = `preference_event.corrected_text` when provided, otherwise the original `message.content`, with optional `context_text` appended for grounding.
- write JSONL dataset rows `{prompt, target, weight, context}` to `${SHARED_FS_ROOT}/users/{user}/adapters/{adapter}/jobs/{job}/dataset.jsonl`.
- tokenize with the configured tokenizer (fallback: whitespace hash IDs) into padded batches of `input_ids`, `labels`, and `attention_mask` (track `{batch, prompt_len, target_len}` in metadata for allocation).
- cluster context embeddings per-user (and optionally globally) to surface emergent themes; persist cluster summaries alongside token batch shapes for routing/training diagnostics.
- feed batches into a JAX/Optax loop that only updates LoRA matrices for the adapter; base model weights are frozen.

### 2.7 config ops (LLM as architect)

```sql
CREATE TABLE config_patch (
  id              BIGSERIAL PRIMARY KEY,
  artifact_id     UUID NOT NULL REFERENCES artifact(id) ON DELETE CASCADE,
  proposer        TEXT NOT NULL,            -- 'system_llm','human_admin','user'
  patch           JSONB NOT NULL,           -- JSON Patch / JSONPath-like ops
  justification   TEXT,
  status          TEXT NOT NULL DEFAULT 'pending', -- 'pending','approved','rejected','applied'
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  decided_at      TIMESTAMPTZ,
  applied_at      TIMESTAMPTZ,
  meta            JSONB
);
```

---

## 3. filesystem layout

### 3.1 directory structure

logical layout (any POSIX-like shared filesystem):

```text
/                             # filesystem root
  shared/
    models/
      base_lm_v1/
        config.json
        shard_00.npz
        ...
    tools/
      ...
  users/
    {user_id}/
      files/                 # uploads / synced documents
        ...
      artifacts/             # artifact-backed files (e.g. notebooks)
        ...
      adapters/
        {adapter_id}/
          v0001/
             params_layer_00_q.npz
             params_layer_00_v.npz
             ...
          v0002/
             ...
      tmp/                   # temporary scratch (ephemeral)
```

### 3.2 adapter files

for each LoRA adapter artifact:

- `schema` in artifact holds:
  - `kind: "adapter.lora"`
  - `rank`, `layers`, `matrices`, etc.
  - `current_version`, `fs_dir`.

on the shared filesystem in `/users/{user}/adapters/{artifact_id}/vNNNN/`:

- `metadata.json` – redundancy with DB, for direct JAX loader.
- weight npz files, e.g.:

```text
params_layer_00_attn_q_A.npy
params_layer_00_attn_q_B.npy
params_layer_00_attn_v_A.npy
params_layer_00_attn_v_B.npy
...
```

or a single `params.npz` keyed by `"layer_00.attn_q.A"`, etc.

### 3.3 storage

postgres is the only store. there is no in-memory fallback: a second
implementation of the storage layer means every feature is written twice and
verified once, and the copy production runs is the untested one.

- `sql/schema.sql` is the whole schema, applied idempotently by
  `scripts/migrate.sh`. the embedding column's width is pinned at apply time
  (`-v embedding_dim=...`) because pgvector's ivfflat index requires a fixed
  dimension.
- artifact payloads (e.g., workflow JSON) live under
  `{shared_fs_root}/artifacts/{artifact_id}/vNNNN.json`; the database holds the
  metadata and version pointers.

---

## 4. redis usage

redis is for hot, ephemeral state:

- **auth / rate limiting**
  - token blacklists, login attempts, per-ip counters.
- **session cache**
  - mapping `session_id → user_id`, short TTL.
- **conversation hot state**
  - recent summary, last N messages, to avoid frequent DB queries.
- **router cache**
  - for `(user_id, ctx_embedding_hash)` store most recent adapter/gate set.
- **workflow state**
  - for long-running workflows: ephemeral node state, partial results.

all redis keys should be namespaced, e.g.:

- `auth:session:{session_id}`
- `chat:summary:{conversation_id}`
- `router:last:{user_id}:{ctx_hash}`

---

## 5. llm & lora adapter stack (python + jax)

### 5.0 deployment modes (kernel treats both as adapter endpoints)

- **cloud API mode: fine-tuned model = endpoint**
  - external providers expose each fine-tune as a first-class `model` id.
  - the kernel maps `artifact` entries of kind `adapter.lora` to these model ids 1:1; activating an adapter means choosing the matching model id.
  - no dynamic multi-adapter composition; switching behavior = switching model id; router can still choose among models based on policy.
  - examples: OpenAI/Azure fine-tuned deployments (`model=ft:...`), Vertex AI Gemini tuned model resource names, Bedrock custom models.

- **self-hosted adapter servers (open source)**
  - base model served once; hundreds–thousands of LoRA fragments mounted behind an OpenAI-compatible API (e.g., LoRAX/Predibase-style) that accepts `adapter_id`/multi-LoRA parameters.
  - kernel passes `adapter_id` + optional gate weights; server composes multiple adapters per request when supported.
  - providers with adapter-id style APIs include Together AI Serverless Multi-LoRA (`adapter_id`), SageMaker adapter inference components, or custom LoRAX deployments behind OpenAI-compatible routes.
  - both modes share the same artifact metadata; only the transport differs, so workflows/policies remain data-driven.

### 5.0.1 adapter mode schema field (clarification)

to support seamless switching between deployment modes, each adapter artifact includes an explicit `mode` field in its schema:

```json
{
  "kind": "adapter.lora",
  "mode": "hybrid",  // local | remote | prompt | hybrid
  "backend": "hybrid",
  "provider": "local",
  "base_model": "llama-7b",
  "rank": 4,
  "fs_dir": "/users/{user_id}/adapters/{adapter_id}",
  "remote_model_id": null,
  "prompt_instructions": "You are a helpful coding assistant..."
}
```

**adapter modes:**

| Mode | Weights | Execution | Use Case |
|------|---------|-----------|----------|
| `local` | Filesystem (`params.json`) | LocalJaxLoRABackend | Self-hosted GPU inference |
| `remote` | External service | API passthrough (`adapter_id`) | Cloud fine-tuned models |
| `prompt` | None | System prompt injection | Behavior without weights |
| `hybrid` | Filesystem + prompt | Local when available, prompt fallback | Portable adapters |

**mode compatibility matrix:**

| Backend | local | remote | prompt | hybrid |
|---------|-------|--------|--------|--------|
| local_lora | ✓ | ✗ | ✓ | ✓ |
| openai | ✗ | ✓ | ✓ | ✓ |
| together | ✗ | ✓ | ✓ | ✓ |
| lorax | ✗ | ✓ | ✓ | ✓ |

**router filtering:**

the router filters adapters before policy evaluation, only considering those compatible with the active backend mode. incompatible adapters are logged and excluded from routing decisions.

**hybrid mode behavior:**

for `hybrid` adapters:
- if running local backend: load weights from `fs_dir`
- if running API backend: extract `prompt_instructions` and inject into system prompt
- if adapter has `remote_model_id`: also pass to API for backends that support it

this allows the same adapter artifact to work across deployment modes without modification.

**adapter gate semantics (normative).**

every routed adapter carries an effective gate

\[
g = \operatorname{clamp}(g_{\text{router}}, 0, 1)
\]

after router selection and filtering. the gate has two meanings, **in this
order**:

1. **activation.** `g == 0` means the adapter is absent from the effective
   request.
2. **intensity.** when the active execution mechanism has a mathematically
   defined continuous weight, `g > 0` supplies that weight.

a zero-gated adapter therefore:

- contributes no local LoRA delta;
- injects no `prompt_instructions`;
- is not sent as a remote adapter or model selection;
- is omitted from the effective adapter stack and the KV-cache signature;
- is omitted from the set of adapters reported as actually applied.

it may still appear in the routing trace, because "the router considered this
adapter and assigned it a zero gate" is a different fact from "this adapter
affected inference", and only the second one is a claim about the answer.

**what a turn reports as applied names mechanisms, never modes.** an adapter
contributes through prompt instructions, through local weights, through a
remote adapter or model selection — and each one that actually ran gets its
own entry. an adapter whose mode permits a mechanism it does not carry has
applied nothing: `hybrid` requires neither `prompt_instructions` nor a remote
id, so an artifact with neither is valid, materializes nothing, sends
nothing, changes no model, and must be reported as **dropped**. reporting it
as applied on the strength of its mode named an adapter the answer never
saw.

**continuous gates apply only where the mechanism supports continuous
composition.**

- local LoRA, and remote multi-LoRA backends that accept adapter weights,
  apply the number exactly: `g · αBA`.
- **prompt execution is binary.** for a prompt or prompt-fallback adapter,
  `g == 0` injects nothing and `g > 0` injects the instructions **once and
  unchanged**. a fractional gate does not shorten, repeat, paraphrase,
  probabilistically inject or otherwise scale natural-language text: there is
  no defined analogue of multiplying a sentence by `g`.
- remote mechanisms with no continuous adapter weight likewise read `g > 0`
  as activation; the number may still inform router ranking and capping
  *before* execution.
- **no threshold downstream.** `g = 0.01` means the router activated the
  adapter. rounding a small positive gate to "off" after the fact would be a
  second routing policy, hidden downstream of the one that owns the decision.
  the router's own `weight_floor` and `max_active_adapters` (§8.1) are not
  that: they are the routing decision, taken by policy, before execution —
  an adapter they exclude is one the router did not activate.

for `hybrid` adapters the rule applies to whichever representation the active
backend uses: on a local backend with a promoted version, `g > 0` applies
weights scaled by `g` and injects no fallback prompt; on an API or
prompt-fallback backend, `g > 0` injects the fallback once; an unpromoted
hybrid has no weights, so `g > 0` activates the prompt fallback once.

**one effective-adapter set drives everything downstream, and it carries the
magnitude too.** after clamping, adapters with `g == 0` are removed *once*,
before backend-specific weight loading, prompt injection, remote passthrough,
effective-stack hashing and inference accounting — and every adapter that
survives carries its canonical `g`, so a consumer reading the adapter's
weight reads the number composition scales by. membership alone is not
enough: a mechanism that re-derives the magnitude from the raw artifact will
disagree about range, about precedence, and about what an unparseable value
means. deciding either half separately in each mechanism is how they came to
disagree — composition dropped the zero-gated term while prompt injection did
not read the gate at all, the KV signature hashed an adapter contributing
nothing, and the remote formatter sent a provider `5.0` for an adapter this
kernel had already clamped to `1.0`.

a backend must hold this line at its own entry, not only downstream of the
service that calls it: the local backend reported a zero-gated adapter as the
turn's `adapter_id`, and sized its tokenizer from it, while correctly
excluding it from both the LoRA sum and the cache key.

**prompt materialization happens once, in the service, before any backend
runs.** `LLMService` places `prompt_instructions` into the messages according
to the (mode, backend) rules above; backends receive prepared messages and
materialize only what is theirs — LoRA weights locally, an adapter or model
selection remotely. the choice of representation is a rule about the pair,
not something one backend can decide alone, and a single materializer is what
keeps an adapter's instructions from appearing twice. a backend that also
injects is a second materializer by another name.

**every entry point into a backend passes through that one primitive** —
`generate`, `generate_stream`, and the caller-built-message paths
`generate_with_tools` and `stream_messages` alike. when only the first two
materialized, the API backends materialized as well "to be safe", so those
two paths sent the text twice while the other two sent it once; removing the
backend copy on its own would have taken the caller-built pair to zero. one
primitive, used everywhere, is what makes "once" true rather than average.

an adapter's mode may be **stated or inferred**, and the rules above apply to
both. an artifact that states no `mode` is `hybrid` by default (§5.0.1), and
a materializer that recognizes only a stated mode silently drops exactly the
legacy adapters the default exists for.

### 5.0.2 provider capabilities (implementation detail)

different API providers handle adapters in fundamentally different ways. the kernel maintains a capability registry to format requests correctly:

**remote styles:**

| Style | Description | Example Providers |
|-------|-------------|-------------------|
| `model_id` | Fine-tuned model as endpoint; one adapter per request | OpenAI, Azure, Vertex, Bedrock |
| `adapter_param` | Adapter ID in request body; multi-adapter supported | Together, LoRAX, adapter_server |
| `none` | No remote adapter support; local/prompt only | local_lora, local_gpu_lora |

**provider capability matrix:**

| Provider | Remote Style | Multi-Adapter | Gate Weights | Max Adapters |
|----------|-------------|---------------|--------------|--------------|
| `openai` | model_id | ✗ | ✗ | 1 |
| `anthropic` | model_id | ✗ | ✗ | 1 |
| `azure` | model_id | ✗ | ✗ | 1 |
| `vertex` | model_id | ✗ | ✗ | 1 |
| `bedrock` | model_id | ✗ | ✗ | 1 |
| `zhipu` | model_id | ✗ | ✗ | 1 |
| `together` | adapter_param | ✓ | ✓ | 3 |
| `lorax` | adapter_param | ✓ | ✓ | 5 |
| `adapter_server` | adapter_param | ✓ | ✓ | 3 |
| `sagemaker` | adapter_param | ✗ | ✗ | 1 |
| `local_lora` | none | ✓ | ✓ | 3 |
| `stub` | none | ✗ | ✗ | 0 |

The `stub` backend returns deterministic canned responses without calling any LLM. It is intended for testing and CI pipelines where real inference is not required.

**adapter schema fields by provider type:**

for `model_id` providers (OpenAI, Azure, etc.):
```json
{
  "mode": "remote",
  "remote_model_id": "ft:gpt-4o-mini-2024-07-18:org:custom:abc123"
}
```

for `adapter_param` providers (Together, LoRAX, etc.):
```json
{
  "mode": "remote",
  "remote_adapter_id": "user-123/my-lora-adapter",
  "weight": 0.8
}
```

**request formatting:**

- `model_id` style: adapter's `remote_model_id` becomes the `model` parameter
- `adapter_param` style: adapter IDs passed as `extra_body.adapter_id` (or provider-specific param)
- when multiple adapters exceed `max_adapters`, lowest-weight adapters are dropped and logged

**hybrid mode with remote fallback:**

hybrid adapters can include both `prompt_instructions` (for prompt injection) and `remote_model_id`/`remote_adapter_id` (for API passthrough):

```json
{
  "mode": "hybrid",
  "prompt_instructions": "You are a coding assistant...",
  "remote_adapter_id": "user-123/code-lora",
  "weight": 0.9
}
```

when using API backend:
1. prompt instructions are always injected into system message
2. if adapter has remote ID and provider supports it, also passed to API
3. if no remote ID or provider doesn't support, only prompt injection used

### 5.1 base model

- plain-JAX implementation of a decoder-only transformer
  (`liminallm/service/transformer.py`): RMSNorm, RoPE, grouped-query
  attention with a KV cache, SwiGLU MLP — the llama/qwen family shape, which
  is what an HF-layout checkout on disk actually contains.
- config + params loaded from the `model_path` directory: `config.json` plus
  `*.safetensors` shards, read framework-neutrally (no torch, no flax). a
  missing tensor **raises**: a half-loaded model answers confidently and
  wrongly, which is worse than not starting.
- base model **frozen**: no gradient / updates on base weights.
- **serving invariants, pinned by tests** — incremental decode with the KV
  cache reproduces a full recompute; attention is causal; a LoRA adapter with
  `B = 0` (how every adapter initializes) changes not one logit; and a warm
  prefix cache produces byte-identical output to a cold one.
- **three checkpoint states, and the middle one is not a state.** `absent`
  (nothing on disk — a dev box or CI) falls back to the synthetic stand-in it
  used to run always, a sinusoidal embedding table with no attention, and
  logs `local_checkpoint_absent`: that path exercises the plumbing, does not
  answer questions, and the log exists so a production box cannot serve it
  quietly. `valid` serves the real model. `broken` — a checkpoint that
  exists but cannot be served (its tokenizer will not load, the weights will
  not read, the tokenizer disagrees with its vocabulary) — **fails every
  request closed**. collapsing the last two into "no real model" meant a
  refused request was followed by one silently answered from the stand-in,
  which is the opposite of refusing.

#### the local text format is one function

the role labels are tokens to a raw decoder, so `USER:` and `user:` are
different inputs, not two styles. training and serving therefore share one
serializer (`service/local_format.py`) for turn labels, for the injected
context marker, and for truncation — which **keeps the newest tokens**,
because a tokenizer's own `truncation` keeps the oldest and a chat's newest
turn is the one the answer responds to. an adapter fitted to one format and
asked to serve another is fitted to a model that does not exist, however
identical the checkpoint and tokenizer are.
- **training uses this same forward pass** (§5.4): the loss is computed over
  the real model with the LoRA matrices applied inside its attention
  projections, so an adapter is fitted to the model that will serve it. the
  base parameters are closed over and never differentiated, which makes
  "only on adapters, never on the base model" structural rather than a
  promise — and it is asserted by a test that the base weights come out of
  training bit-identical.

#### weights serve one base, and only that base (normative)

before any LoRA weights load, the adapter's declared base must be the base
the backend serves. identity is compared on the final path component,
case-insensitively, so `/models/qwen3-4b` and `qwen3-4b` are the same
checkpoint named two ways — and nothing looser. **an undeclared base refuses
too**: an adapter that does not say what it was fitted against cannot show it
was fitted against this one.

**one implementation answers it for both ends** of the ladder
(`transformer.same_base_model`): training asks it before fitting an adapter,
serving before applying one. two spellings of the same rule drift, and the
looser one decides — training compared the raw strings while serving compared
path components, so which spelling a deployment happened to store determined
whether an adapter could be trained at all.

the rule is a consequence of §5.2, not a policy choice. `B·A` was optimized
against one particular frozen `W`, so a gate (§5.4.6) passed on that `W` says
nothing about a different one. training already refuses a base mismatch; a
serving path that accepted one with a warning applied weights no gate ever
scored. a family-similarity comparison — treating `-chat`, `-base` and
version suffixes as the same model — is expressly not sufficient here: those
are different frozen weights and therefore different models.

the rule guards weights, not adapters. it is checked at the point weights
would be applied — after version resolution (§5.5) and before the adapter
cache — so a prompt-rung adapter, an adapter with nothing promoted yet, and
one whose router gate is closed are all unaffected: they contribute no
tensors either way. checking it at selection time instead would turn renaming
a checkpoint directory into an outage on every routed turn rather than on the
weight-bearing ones.

### 5.2 lora parameterization

for each hooked weight matrix `W ∈ ℝ^{d_out × d_in}`:

- LoRA params for adapter `j`:
  - `A_j ∈ ℝ^{r × d_in}`
  - `B_j ∈ ℝ^{d_out × r}`
  - scale `α_j` (scalar or per-matrix)
- **naming, because serving matches on it**: matrices are keyed
  `layers.{i}.{target}.{A|B}` with `target ∈ {attn_q, attn_k, attn_v,
  attn_o}` and an optional `layers.{i}.{target}.scale`. names outside that
  shape are counted and logged, never partially applied — an adapter trained
  for a different architecture must fail visibly rather than land on half its
  projections.
- effective weight for given adapter gate weight `g_j`:

\[
W_{\text{eff}} = W + \sum_j g_j \cdot \alpha_j B_j A_j
\]

- **the gate decides before the weights are read (normative).** a term with
  `g_j = 0` is not in the sum, so that adapter is not part of the effective
  model and nothing about its files can matter — not the base they declare
  (§5.1), not their checksum, not whether they parse. composition therefore
  reads `g_j` first and skips the adapter entirely; a zero-gated adapter with
  a promoted version on disk must be a no-op, exactly as one with no file at
  all is. router weights are clamped to `[0, 1]` (§8.1), so a negative gate
  is a closed one. this is per adapter, not per stack: an open-gated adapter
  beside a closed one still composes, and still has to be valid.

in JAX:

- represent `params_base` and `params_lora[adapter_id]` as nested PyTrees.
- composition function:

```python
def compose_params(params_base, lora_params_list, gate_weights):
    # lora_params_list: list of LoRA pytrees for each active adapter
    # gate_weights: list of floats (same order)
    def combine(base_leaf, *lora_leaves):
        # base_leaf: base weight
        # each lora_leaf: dict { 'A':..., 'B':..., 'alpha':... } or None
        W = base_leaf
        delta = 0
        for gate, lp in zip(gate_weights, lora_leaves):
            if lp is None or gate == 0.0:
                continue
            A, B, alpha = lp["A"], lp["B"], lp["alpha"]
            # precompute BA offline if rank/static, or compute on the fly
            delta = delta + gate * alpha * (B @ A)
        return W + delta
    # use jax.tree_map to map combine over all matrices.
```

**composition is by rank concatenation, never by averaging matrices.** the
implementation composes several adapters into one equivalent pair:

```
A* = [A_1 ; A_2 ; …]                 stacked on the rank axis
B* = [g_1α_1B_1 , g_2α_2B_2 , …]     stacked on the rank axis
⇒  B*A* = Σ_j g_j α_j B_j A_j        exactly
```

this matters because the obvious alternative is wrong in two ways, and both
shipped: gate-weighting `A` and `B` separately and dividing by the total
weight computes `(gA)/g = A` for a lone adapter — the router's gate cancels
itself, so 0.2 and 1.0 behave identically — and for two adapters it forms
`B̄Ā`, whose expansion contains `B_1A_2` and `B_2A_1`, products of one
adapter's up-projection with another's down-projection that appear in no term
of the sum. ranks may differ between adapters; concatenation needs no padding.
a gate of 0 contributes nothing rather than being normalized back into
existence.

**composition refuses rather than partially applies.** an `A` without its
matching `B`, a `B` without its `A`, or adapters that disagree on a
projection's dimensions, raise and refuse the whole stack. dropping the odd
contribution and applying the rest serves the request with a stack the router
never chose, which is the partial application this section forbids one
paragraph above — logging it and continuing is still doing it.

**one validator, checked per adapter, before composition.**
`validate_lora_weights(config, weights)` verifies every key against this
shape — name, target, layer index, rank agreement, the projection's real
`(d_out, d_in)`, and pairing for **every** projection a key mentions,
`scale` included — and raises on the first violation. a `scale` is a scalar
attached to a hooked weight, so a projection named only by a `scale` has no
matrices and is refused: that adapter is non-empty on the way in and
contributes nothing on the way out, which is how it slipped past the rule
below. the config argument is optional, so this is the only validator: what
is knowable from the weights alone is checked either way, and layer bounds
and projection widths are added when a model is loaded. it
runs on each adapter's **raw** matrices as they load, then again on the
composed pair as a defensive check. the order matters twice over: composition
carries only A/B pairs forward, so a foreign key never reaches a validator
that runs afterwards; and concatenation *adds ranks up*, so two adapters that
each disagree with themselves (A of rank 2 with B of rank 1, and A of rank 1
with B of rank 2) compose into a pair whose totals agree — 3 and 3 — while
every row pairs with the wrong column. training validates before indexing
(and skips, since a skipped run cannot promote).

**a selected adapter never silently leaves the stack.** weightless is
legitimate exactly where §5.5 says so — the prompt rung, or nothing promoted
yet, or a closed gate. a promoted local/hybrid adapter with an open gate
whose weights will not load refuses the stack instead, because serving
without it is serving a stack the router did not select.
this is what makes "never partially applied" true rather than aspirational:
assembly that merely *skips* names it does not recognize still applies the
ones it does, so an adapter carrying a single foreign matrix changed the
model through its recognized half while the rest went to a log line.

for performance:

- restrict LoRA to:
  - attention projections: Q, K, V, O
  - optionally MLP projections: W_in, W_out
- rank `r` small (4–8) for per-user adapters.

### 5.3 inference service

- keep base params resident on GPU/TPU.
- per-request:

  1. determine active adapters & gate weights (`adapter_ids`, `gate_weights`).
  2. load corresponding LoRA parameter PyTrees from the shared filesystem (cache hot ones in RAM).
     - cache policy: LRU by `(adapter_id, version)` — keyed by both, because
       two versions of one adapter are different weights and an id-only key
       leaves file mtime as the only thing standing between a promotion and
       its predecessor's tensors; pin persona adapters for logged-in user; max resident bytes guarded by config with periodic eviction.
     - **the router's gate travels on the adapter it gates.** `_select_adapters`
       attaches each gate weight to the activated adapter dict; the backend
       reads it there and nowhere else. returning gates alongside the adapters
       for tracing only — which is what used to happen — means composition
       runs every adapter at 1.0 no matter what the policy decided, and the
       §5.2 equation is exactly right about a number it never receives.
     - lazy load: if adapter missing from cache, fetch `metadata.json` + `params.npz`; validate checksum + version; keep small adapters in RAM, map large ones with memmap if supported.
     - per-request adapter cap (e.g., top 3) to bound composition cost; reject requests exceeding cap.
  3. compose an effective view of weights:
     - for small K (top 2–3 adapters) this is cheap.
     - composition happens in JIT-compiled function to avoid Python overhead.
  4. run generation with sampling parameters (top-p, temperature, max tokens).
     - batching policy: group requests by base model + active adapter set hash; cap batch size to avoid latency spikes.
     - timeouts: cancel generation if wall clock > `max_decode_ms` (configurable per plan tier); return partial tokens with `truncated=true` flag.
     - cancellation: orchestrator can send `cancel` by `request_id`; worker releases adapter references and frees KV cache slots.
  5. stream tokens back to orchestrator.
     - protocol: Server-Sent Events (text/event-stream) or WebSocket frames `{ "event": "token", "data": "..." }`.
     - final frame contains usage stats and adapter gates actually applied.

#### KV prefix cache (local lane)

a chat turn re-sends the whole conversation, so turn *N*'s prompt is a strict
prefix of turn *N+1*'s. the local backend exploits exactly that, and nothing
looser:

- **content-addressed, not conversation-keyed.** entries are
  `(adapter signature, token tuple, kv state)`; a lookup takes the longest
  stored entry that is a **strict token prefix** of the incoming prompt and
  truncates its KV to that length. no conversation id is plumbed anywhere,
  so the cache cannot mistake one thread for another — only identical tokens
  match.
- **the signature identifies the effective stack, gates included**: each
  active adapter contributes `(id, version, gate)`. gates are per-request
  (§5.3), so the same adapter at 0.2 and at 0.8 is a different model and
  every cached tensor was computed under one of them. keying on id+version
  alone — which this section used to specify — would let a 0.2 request
  continue a prefix built at 0.8, which is the cheapest imaginable way to
  serve a model nobody asked for.
- **why strict.** reusing keys computed for different tokens would answer
  from a history the user never wrote. the shared-prefix count is the only
  thing reused; the divergent tail is always recomputed.
- **adapter-keyed twice over.** the signature is adapter id + version (version
  dirs are immutable), and any actual reload of adapter weights from disk
  clears the cache outright — which closes the case of an in-place edit that
  never bumped a version.
- **bounded.** total cached tokens are capped (`max_cached_tokens`), evicted
  LRU; an entry superseded by a longer one that extends it is dropped rather
  than kept twice.
- **reported, not estimated.** the reused prefix length is `cached_tokens` in
  usage, which surfaces as `input_tokens_details.cached_tokens` on the served
  Responses api (§13.1) — the same field a provider's own prefix cache fills.
- a fully cached prompt still runs its final token, because logits to sample
  from have to come from somewhere.

initial minimal version:

- support **only persona adapter** or **no adapters**.
- later, add domain/skill adapters.

### 5.4 training service

training updates only LoRA params of a single adapter.

loop for a `training_job`:

1. fetch job + related `preference_event`s.
2. reconstruct training examples:

   - for each event:
     - assemble `prompt` = preceding user + assistant messages up to event.
       "up to" is bounded by the target message's **sequence number**, not by
       its position in the fetched list: dropping the target row alone leaves
       every later turn in the prompt, so an event trained after the
       conversation continued teaches its answer conditioned on things that
       had not happened when the answer was written. the sequence is resolved
       from the store *by message id* and the history is queried as
       `seq < target_seq` — searching for the target inside a fetch window
       silently disables the bound for any event older than the window, which
       is exactly the event most likely to have later turns after it. a target
       that cannot be resolved drops the example rather than falling back to
       whatever messages happened to be fetched.
     - target `y` = preferred assistant answer:
       - either the answer that got “like”
       - or user’s corrected text.

3. build batched dataset.

4. define JAX loss function:

   - SFT (supervised fine-tuning):

     ```python
     def loss_fn(lora_params, batch):
         logits = model_apply(params_base, lora_params, batch.inputs)
         logprobs = log_softmax(logits, axis=-1)
         # standard token-level CE loss
         loss = -jnp.mean(jnp.sum(batch.target_mask * jnp.take_along_axis(
             logprobs, batch.targets[...,None], axis=-1
         ), axis=-1))
         # regularization
         loss += lambda_l2 * l2_norm(lora_params)
         return loss
     ```

   - optionally DPO if we have good/bad pairs.

5. dataset format + hygiene:

    - write JSONL to the shared filesystem per job: `{ "prompt", "target", "weight", "context" }`.
   - dedupe by `(conversation_id, message_seq)` to avoid replaying the same correction.
   - cap per-example tokens (e.g., 2048) and per-job total tokens (plan-tier bound) to control spend.
   - batch layout is causal-LM SFT: one `prompt+target` sequence per example,
     next-token labels, loss masked to the target span only.
   - the two spans are tokenized under one convention: the prompt carries the
     model's special tokens, the target is encoded as a **continuation** so no
     second BOS is spliced into the middle of a sequence the model would never
     see that way at serving time.
   - **truncation reserves the target first** and trims the oldest prompt
     context. slicing the head of `prompt + target` instead can drop the whole
     supervised span, leaving an all-zero loss mask — an example that reports
     a loss of zero, which reads as one the model already answers perfectly.
     an example with no supervised token is dropped rather than emitted.
   - optional teacher distillation pass rewrites targets first (§7.5).

6. evaluation + rollout (**normative - the eval gate**):

   - once a dataset has ≥5 examples, every 5th example is held out; the job
     trains on the remainder for several epochs and evaluates holdout loss
     with the initial weights and again with the trained weights.
   - the holdout number is **cross-entropy only**, without the L2 term of the
     training objective: the gate asks whether predictions improved, and
     since `B` starts at zero and can only grow, charging the regularizer to
     the eval would count honest learning as a penalty against promotion.
   - a new adapter version is promoted (bumps `current_version` — which is
     what promotion *is* — and graduates a prompt-mode adapter to `hybrid`
     per §5.5) **only** when holdout loss improves by ≥1% relative. the
     `latest` pointer is refreshed for humans and tooling as a side effect,
     and is **best-effort**: it is not consulted by serving (§5.5), so a
     failure to write it is logged and the promotion stands. re-raising there
     aborted a run *after* the version was bumped, which left the gate
     decision below unrecorded and let the worker retry against weights that
     were already authoritative.
   - a skipped run (JAX unavailable) or a regression **never** promotes:
     the artifact is left untouched and the gate decision is recorded in
     `training_job.meta.eval_gate` for audit. "training ran without raising"
     is not a promotion criterion.
   - **the decision travels with the run summary, and its absence is not
     approval.** the background worker records job status and credits router
     state from that summary; a summary that dropped `eval_gate` combined
     with a default of "promoted" marked gate-rejected runs succeeded and
     credited an adapter for a rollout that never happened. missing means
     unknown, and unknown is not promoted.
   - **a dataset too small to hold anything out never promotes either.**
     "promoted only when holdout loss improves" refuses what it cannot
     measure. weights change the model now, so shipping an unevaluated
     version is exactly the regression §5.5 forbids; the adapter waits on
     the prompt rung until it has the data to prove itself.
   - **what "skipped" covers**, each leaving the adapter on the prompt rung:
     JAX/optax missing; **no base checkpoint to train against** (there is no
     `model_apply(params_base, …)` without `params_base`, and training some
     other function would report success for a model nobody serves); an
     adapter carrying no LoRA matrices; an adapter whose matrices match no
     projection in the model (weights trained for another architecture must
     fail visibly rather than land on half their targets); **a checkpoint
     whose own tokenizer will not load**; and **token ids outside the
     checkpoint's vocabulary**.
   - the last two are the same invariant as the first: "train against the
     model that will serve it" includes its tokenizer. gradients through the
     right weights teach nothing transferable if the text reached them
     through an invented token space — and the holdout, tokenized the same
     wrong way, would agree that it worked. the id-hash fallback exists for
     plumbing tests against the synthetic stand-in; it must never produce
     promotable weights for a real checkpoint, and serving refuses such a
     checkpoint for the same reason. an out-of-range id is refused rather
     than clipped, because clipping trains on a token nobody wrote.

7. scheduling:

   - per-user throttle (max 1 concurrent job, cooldown between jobs) to avoid GPU starvation.
   - queue respects priority (admin > paying > free) with fairness to prevent starvation.

5. run optimizer (Optax) for a few steps:

   - small learning rate, few epochs.
   - early stopping based on batch loss.

6. write new LoRA params to the shared filesystem in a new version directory.

7. update:

   - `adapter_router_state.centroid_vec` via EMA of event embeddings.
   - `adapter_router_state.last_trained_at`, `success_score`.

8. mark training job `status='succeeded'` with `loss`.

**scheduling & prioritization:**

- queue ordering: prioritize `(user_id, cluster_id)` pairs with highest recent positive feedback density and no recent training.
- per-user fairness: limit concurrent jobs per user to 1; global cap to avoid GPU exhaustion.
- retry policy: exponential backoff on transient failures (I/O, OOM); max 3 attempts; mark failed with reason.
- dataset materialization: store tokenized batches (packed with attention masks) in `/users/{u}/adapters/{id}/vNNNN/batches/` for reproducibility; include manifest JSON summarizing sources.
- evaluation: the held-out batch of §5.4.6 is required whenever the dataset supports it; gate decisions are recorded in `training_job.meta.eval_gate`.

### 5.5 adapter ladder (prompt → weights lifecycle)

every skill adapter climbs the same ladder; the rungs are data thresholds and
eval gates, not human ceremony:

```
cluster qualifies          pooled events ≥ threshold      eval gate passes
      │                              │                          │
      ▼                              ▼                          ▼
 mode: prompt      ──────▶   training job enqueued   ──────▶  mode: hybrid
 (instructions from          (data pooled across              (trained weights,
  cluster label +             the whole cluster)               prompt kept as
  positive exemplars)                                          portable fallback)
```

1. **born as a prompt.** when a cluster qualifies (§7.3), its skill adapter is
   created with `mode: "prompt"` and `prompt_instructions` composed from the
   cluster label, description, and up to 3 highly-rated exemplars. it is
   immediately useful on every backend and costs nothing to create.
   `lifecycle: { "stage": "prompt", "weights_min_events": N }` records the
   next rung.
2. **weights when the data earns them.** once the cluster has pooled at least
   `weights_min_events` positive events (default 20), a training job is
   enqueued. skill training data is pooled **across all contributors to the
   cluster** - persona adapters remain strictly per-user.
3. **graduation is gated.** if the job passes the §5.4.6 eval gate, the
   adapter flips to `mode: "hybrid"` (`lifecycle.stage: "weights"`): trained
   weights where the backend supports them, with `prompt_instructions` kept
   as the portable fallback. a failed or skipped gate leaves the adapter on
   the prompt rung; nothing regresses.
   - **two independent locks make "before graduation" unservable**, because a
     training job writes its `vNNNN/params.json` *before* the gate runs:
     `current_version <= 0` pins nothing and resolves to no weights (never a
     directory scan, which would find exactly that un-gated version), and
     `mode: "prompt"` contributes no LoRA weights whatever files exist on
     disk. one lock would be a race; a crash between writing the version and
     quarantining it would have made the race permanent.
   - **version authority outranks path shape, absolutely.** a positive
     `current_version` of N resolves to **this adapter's**
     `vNNNN/params.json` and nothing else. the `latest` pointer takes no part
     in authoritative resolution: it remains a convenience for humans and
     tooling, but serving does not consult it. checking that its target was
     merely *named* `vNNNN` proved a basename, not an identity, so
     `A/latest → B/v0001` served another adapter's weights as A's version 1
     — and it enabled nothing, because a pointer legitimately aimed at
     `A/vNNNN` means that directory exists and the exact path has already
     answered. a path pointing straight at a `params.json` cannot demonstrate
     which version it is, so it cannot satisfy a versioned artifact at all
     (only an artifact that has never been versioned may use one).
     resolution therefore starts at the adapter **root** (`adapters/<id>`),
     never at `adapters/<id>/latest`: handing it the pointer made it look for
     `latest/vNNNN`, so a correctly promoted adapter became unservable merely
     because the convenience pointer existed beside its versions.
   - **the version is pinned, and so is the adapter.** weights belong to the
     adapter whose directory holds them, checked two ways where weights are
     about to be read. **by layout**: the directory containing a
     `params.json` is named for its owner, so `adapters/A/vNNNN/params.json`
     is A's and nothing else's. an explicit `fs_dir`/`cephfs_dir` may say
     *where* an adapter's directory lives — a per-user root, another mount —
     never *whose* it is; validating only that it sits under `fs_root` proved
     nothing, since every adapter's directory does, and an artifact naming
     `adapters/B` had B's weights served as A's version 1. **by provenance**:
     training records `adapter_id` and `version` inside each version's
     `metadata.json`, and a recorded id or version that disagrees refuses —
     that is what catches a directory renamed to A holding B's run, which
     layout alone cannot see. provenance is verified when present rather than
     required, so a hand-written version fails on disagreement rather than on
     absence. the same identity binds the write side: a training job may not
     place a new version in another adapter's tree, where that adapter's
     promotion would authorize it.
   - **a versionless artifact cannot authorize a version.** an artifact with
     no `current_version` at all — the never-versioned legacy shape, which
     the adapter schema no longer permits to be created — may serve its own
     `params.json` and nothing else. it must not scan for `latest`, for
     `vNNNN`, or for any subdirectory, because both holes closed above reopen
     immediately in a lane that skips the version check: a `latest` aimed
     elsewhere serves another adapter's weights, and a bare `vNNNN` serves
     exactly what a gate-rejected run leaves on disk.
   - **after graduation the prompt is the fallback, not a second voice.** on a
     backend that applies LoRA weights, a promoted hybrid adapter is carried
     by its weights and its `prompt_instructions` are NOT injected (§5.0.1);
     on an API backend the prompt carries it. injecting both gave the local
     model the weights *and* the instructions they were distilled from — an
     input the eval gate never scored. a hybrid adapter with nothing promoted
     yet keeps its prompt locally, because no weights will load for it.
4. **demotion mirrors promotion.** pruning (§7.4) can push an adapter back
   down the ladder (disable weights, keep prompt) via the same ConfigOps
   pipeline.

### 5.6 remote multi-lora serving (scale-out option)

JAX local serving is the primary path. at scale, the same artifacts can be
served by a dedicated multi-LoRA server (LoRAX-style, vLLM multi-LoRA,
Together adapter APIs) behind the existing OpenAI-compatible transport:

- **native Gemini** (`model_backend: gemini_native`): speaks
  generativelanguage.googleapis.com directly — generateContent /
  streamGenerateContent SSE — rather than the OpenAI-compat shim (which
  remains `gemini`). usageMetadata's thoughtsTokenCount and
  cachedContentTokenCount map to the same reasoning_tokens / cached_tokens
  keys as the Responses path, so the rich usage is provider-uniform. The
  chat-shaped internal history (system prompt, assistant tool_calls,
  role:"tool" results) converts losslessly to native `contents`
  (service/gemini_backend.py), so a conversation resumes mid-history on any
  provider.
- **endpoint selection**: the Responses API (`/responses`) is the primary
  endpoint for OpenAI-compatible backends — richer usage (reasoning and
  cached-token counts flow into turn usage), typed output items, first-class
  reasoning control. The backend probes once per process and falls back to
  `/chat/completions` permanently for providers that answer 404/405; the
  internal message shape stays chat-format, translated at the wire
  (service/responses_compat.py).

- the kernel already models this as `remote`/`adapter_param` providers
  (§5.0.2); adapters trained by the JAX pipeline are exported per-version to
  the shared filesystem and mounted by the server.
- prompt-rung adapters work unchanged on every remote backend (instructions
  are injected into the system prompt), so the ladder is portable across
  deployment modes by construction.
- switching serving modes is a config change, not a migration: artifacts,
  versions, and router policies are identical in both.

---

## 6. generic primitives in practice

### 6.1 artifact.schemas (examples)

**adapter.lora**:

```json
{
  "kind": "adapter.lora",
  "backend": "local",
  "provider": "local",
  "scope": "per-user",
  "user_id": "…",
  "base_model": "jax-base",
  "rank": 8,
  "layers": [0,1,2,3,4,5],
  "matrices": ["attn_q", "attn_v"],
  "current_version": 3,
  "fs_dir": "/users/.../adapters/{id}",
  "cluster_id": "…",  // semantic cluster this adapter is tied to
  "remote_model_id": null, // populated when backend == "api"
  "applicability": {
    "natural_language": "Helps this user debug kernel panics via reproduce→bisect→log-analysis.",
    "embedding_centroid": null  // also in adapter_router_state; optional redundancy.
  }
}
```

Router policies remain agnostic: they pick adapters by id/metadata and hand them to the inference backend. An adapter with `backend="api"` implies switching the request model ID to `remote_model_id` (e.g., Zhipu BigModel or Alibaba DashScope); `backend="local"` means applying filesystem-backed LoRA weights on the base model. `backend="prompt"` distills adapter behavior into a prompt/system-message overlay for API-only providers.

`mode` — not `backend` — is authoritative wherever behaviour depends on it (§5.0.1); `backend` and `provider` are legacy fields that mode is *inferred from* when it is absent, and code that branches on them directly can disagree with the field the spec calls controlling. In particular `hybrid` is defined by §5.0.1: **local weights on a backend that applies them, `prompt_instructions` as the portable fallback everywhere else** — one adapter served one way per backend, never a two-step controller/executor plan. (An earlier revision of this paragraph described `backend="hybrid"` that way; §5.0.1 controls, and this paragraph is corrected to match rather than left to produce a second valid answer.)

**workflow.chat**:

```json
{
  "kind": "workflow.chat",
  "entrypoint": "node_classify",
  "nodes": [
    {
      "id": "node_classify",
      "type": "tool_call",
      "tool": "llm.intent_classifier_v1",
      "inputs": { "message": "${input.message}" },
      "outputs": ["intent"]
    },
    {
      "id": "node_route",
      "type": "switch",
      "branches": [
        { "when": "intent in ['qa_with_docs','analysis']", "next": "node_rag" },
        { "when": "intent == 'code_edit'", "next": "node_code_agent" },
        { "when": "true", "next": "node_plain" }
      ]
    },
    {
      "id": "node_rag",
      "type": "tool_call",
      "tool": "rag.answer_with_context_v1",
      "inputs": { "message": "${input.message}" }
    },
    {
      "id": "node_code_agent",
      "type": "tool_call",
      "tool": "agent.code_v1",
      "inputs": { "message": "${input.message}" }
    },
    {
      "id": "node_plain",
      "type": "tool_call",
      "tool": "llm.generic_chat_v1",
      "inputs": { "message": "${input.message}" }
    }
  ]
}
```

**workflow.chat schema / contracts** (JSON Schema sketch):

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["kind", "entrypoint", "nodes"],
  "properties": {
    "kind": {"const": "workflow.chat"},
    "entrypoint": {"type": "string"},
    "timeout_ms": {"type": "integer", "minimum": 1000},
    "max_retries": {"type": "integer", "minimum": 0, "default": 1},
    "nodes": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["id", "type"],
        "properties": {
          "id": {"type": "string"},
          "type": {"enum": ["tool_call", "switch", "parallel", "end"]},
          "description": {"type": "string"},
          "tool": {"type": "string"},
          "inputs": {"type": "object"},
          "outputs": {"type": "array", "items": {"type": "string"}},
          "branches": {
            "type": "array",
            "items": {
              "type": "object",
              "required": ["when", "next"],
              "properties": {
                "when": {"type": "string"},
                "next": {"type": "string"}
              }
            }
          },
          "next": {"type": "string"}
        }
      }
    }
  }
}
```

**workflow engine contracts**:

- `vars` is a `dict[str, Any]` scoped to a workflow execution; tool outputs merge into `vars` by key.
- tool inputs are resolved by templating from `input` + `vars` (e.g., `${vars.intent}`); missing keys cause a node failure.
- **error handling:**
  - node failure triggers retry up to `max_retries`; exponential backoff capped at `timeout_ms`.
  - if retries exhausted, engine emits an `error` event and returns structured error to orchestrator; optional fallback node `on_error` can be specified in node metadata.
- **timeouts:**
  - per-node timeout default 15s unless overridden in node metadata; workflow-level `timeout_ms` caps total wall clock.
  - on timeout, mark node as failed and follow retry rules.
- **idempotency:**
  - workflow runs identified by `(conversation_id, request_id)`; repeated request_id aborts duplicates.

**policy.routing**:

```json
{
  "kind": "policy.routing",
  "name": "default_user_routing",
  "description": "Select adapters & tools based on context cluster + safety.",
  "rules": [
    {
      "id": "always_persona",
      "when": "true",
      "action": {
        "type": "activate_adapter_by_type",
        "adapter_type": "persona",
        "weight": 0.5
      }
    },
    {
      "id": "cluster_near_skill",
      "when": "cosine_similarity(ctx_embedding, adapter_centroid) > 0.6",
      "action": {
        "type": "activate_adapter_by_cluster",
        "cluster_id": "closest",
        "weight": "similarity"
      }
    },
    {
      "id": "safety_never",
      "when": "safety_risk == 'high'",
      "action": {
        "type": "deactivate_all_adapters"
      }
    }
  ]
}
```

the kernel doesn’t know what “debug” or “kernel” is; it just evaluates `when` expressions and actions using a restricted expression interpreter.

---

## 7. emergent domains & skills via clusters

### 7.1 clusterer job

periodic batch job (Python):

- input: `preference_event.context_embedding` (per user & globally).
- algorithm:
  - per user: run incremental clustering (e.g., streaming kmeans / HDBSCAN).
  - for large datasets, approximate incremental clustering.

outputs:

- upsert `semantic_cluster` rows:
  - `centroid`, `size`.
- maintain mapping from events → `cluster_id`.

### 7.2 LLM labeling

for each sizeable cluster:

1. sample some `message.content` around events in that cluster.
2. call LLM with prompt:

> here are N examples of tasks this user asked about. please give a short label and a one-sentence description.

3. upsert `semantic_cluster.label` & `description`.

### 7.3 skill adapter creation (prompt-first, pooled data)

when:

- `semantic_cluster.size >= min_skill_size`, AND
- positive ratio among the cluster's `preference_event`s ≥ threshold, AND
- no existing adapter bound to this cluster:

then a skill adapter is created **on the prompt rung of the ladder (§5.5)**:

```json
{
  "kind": "adapter.lora",
  "mode": "prompt",
  "scope": "global",              // or "per-user" for user-scoped clusters
  "prompt_instructions": "Skill: <label>.\n<description>\nExamples of responses users rated highly:\n- ...",
  "lifecycle": { "stage": "prompt", "weights_min_events": 20 },
  "rank": 4,
  "layers": [0,1,2],
  "matrices": ["attn_q"],
  "cluster_id": "<cluster_id>",
  "applicability": {
    "natural_language": "Skill: " + semantic_cluster.label + " – " + semantic_cluster.description
  }
}
```

- the adapter is useful immediately (instructions injected on any backend);
  no zero-weight artifact ever becomes `latest`.
- a **weights** training job is enqueued only once the cluster has pooled
  `weights_min_events` positive events. training data for skill adapters is
  pooled **across every contributor to the cluster** (tenant-scoped); the
  job's nominal owner is the cluster's user, or for global clusters the most
  frequent contributor.
- graduation to `hybrid` happens only through the §5.4.6 eval gate.
- persona adapters are exempt from pooling: they train strictly on their
  owner's events.

### 7.4 adapter pruning / merging

monitor `adapter_router_state` over time:

- if:
  - `usage_count` low,
  - `success_score` poor,
  - no recent preference_events,
then:

- propose via ConfigOps:
  - disable adapter (`status=disabled`),
  - or merge into another adapter:
    - training job that distills it into a more successful sibling adapter.

### 7.5 teacher distillation (optional)

small students train better on clean exemplars than on raw chat transcripts.
when `training_distillation_enabled` is set, the configured serving LLM acts
as a **teacher** during dataset assembly:

- each example's target is rewritten by the teacher into a concise, ideal
  exemplar that preserves meaning and facts; the raw target is kept on any
  failure, so distillation can never lose data.
- calls are capped per job (default 32) to bound teacher cost.
- the count of distilled examples is recorded in `training_job.meta`.
- the teacher can be the local model itself, or a frontier API model used
  offline - inference never depends on it.

---

## 8. router as data (policy-driven)

### 8.1 router engine

router engine is a small, deterministic piece of code that:

1. loads `policy.routing` artifact for the user (fallback to global default).
2. builds evaluation context:

   - `ctx_embedding` (vector)
   - `cluster_candidates` (nearest semantic clusters)
   - proposed `adapter_candidates` (persona, domain, skill)
   - `safety_risk` (low/med/high)
   - `usage_stats` etc.

3. interprets rules:

   - restricted language: boolean conditions with:
     - comparisons (`>`, `<`, `==`, `in`)
     - simple functions: `cosine_similarity`, `cluster_label_contains`, etc.
   - actions:
     - `activate_adapter_by_id`
     - `activate_adapter_by_type`
     - `activate_adapter_by_cluster`
     - `scale_adapter_weight`
     - `deactivate_adapter`
     - `deactivate_all_adapters`

4. produces:

   - `adapter_ids` and `gate_weights`.

no explicit “if debugging then do X” in code; that lives in the data-driven policy.

**execution semantics:**

- evaluate rules in order; later rules can override earlier weights if `action.overwrite=true` (default false).
- expression interpreter only supports whitelisted functions (`cosine_similarity`, `contains`, `len`, numeric ops) and literals; no arbitrary Python.
- provide `trace` object capturing which rules fired, resulting gate weights, safety overrides; stored in logs for LLM auditors.
- guardrails: clamp resulting gate weights to `[0, 1]`, normalize if sum > 1; enforce max active adapters (default 3) and per-adapter weight floor (default 0.05).

**prototype implementation notes:** sandboxed evaluation is implemented with adapter activation/deactivation, weight scaling, cosine-similarity-based "closest" selection, per-rule traces, and normalized adapter gate outputs returned on chat responses.

### 8.2 llm editing routing policies

LLM can propose patches like:

```json
{
  "op": "add",
  "path": "/rules/-",
  "value": {
    "id": "avoid_creative_adapter_for_debug",
    "when": "cluster_label_contains(ctx_cluster, 'debug') and adapter_type == 'creative_writing'",
    "action": { "type": "deactivate_adapter", "adapter_id": "current" }
  }
}
```

guardrails:

- JSON schema validation.
- safe expression checking (no arbitrary code).
- simulation against past logs before applying.

---

## 9. workflows as editable graphs

### 9.1 workflow engine semantics

workflow engine interprets `workflow.chat` artifacts:

- node types:
  - `tool_call` — call a named tool (LLM, RAG, code agent, STT/TTS).
  - `switch` — branching based on condition expressions.
  - `parallel` — fan-out to multiple nodes, then join.
  - `end` — produce a final response.

- execution context:
  - `input`: user message, conversation context, etc.
  - `vars`: dictionary storing intermediate results (intent, retrieved docs, etc).

kernel only implements:

- `run_workflow(artifact_id, input, vars_initial={})`
- tool registry with signatures.

### 9.2 tools registry

tools themselves are described as artifacts `tool.spec`:

```json
{
  "kind": "tool.spec",
  "name": "rag.answer_with_context_v1",
  "description": "Use RAG to answer based on bound knowledge contexts.",
  "input_schema": {
    "type": "object",
    "properties": {
      "question": { "type": "string" },
      "contexts": { "type": "array", "items": { "type": "string" } }
    }
  },
  "output_schema": { "type": "object", "properties": { "answer": { "type": "string" } } }
}
```

python code registers functions implementing these tools, checks I/O against schema. LLM can inspect `tool.spec` artifacts to decide how to wire workflows.

execution guardrails:

- tools run in constrained worker pool with CPU/memory limits; network egress allowlisted.
- no shell execution unless tool is marked `privileged:true` and restricted to admins; sandbox defaults to pure Python/HTTP.
- per-node `max_retries` and `backoff_ms` defaults (1 retry, 200ms backoff) are overridable in workflow nodes.
- per-node `timeout_ms` (default 15000) after which the node fails; workflow either retries or aborts per policy.

---

## 10. llm as architect: config ops api

### 10.1 api endpoints

- `POST /v1/config/propose_patch`
  - body: `{ artifact_id, patch, justification }`
  - auth: restricted (system-LLM, admins, or power users).
  - creates `config_patch` row with status `pending`.

- `GET /v1/config/patches?status=pending`
  - for admin review.

- `POST /v1/config/patches/{id}/decide`
  - approve/reject with reason.

- `POST /v1/config/patches/{id}/apply`
  - apply patch:
    - load current `artifact.schema`
    - apply JSON patch (RFC 6902 style)
    - validate against artifact kind schema
    - write new `artifact_version`
    - update `artifact` row
  - mark patch `applied`.

### 10.2 guardrails

- **validation**:
  - JSON schema per artifact.kind.
  - check all references (adapter ids, cluster ids, tool names) exist.
- **sandbox simulation**:
  - run router/workflow in dry-run mode on a small sample of past conversations.
  - compute metrics; optionally block patch if regression is obvious.
- **eval gates before promotion** (implemented for adapters):
  - adapter weight promotion is gated on measured holdout improvement
    (§5.4.6); the same principle applies to any auto-applied change - no
    artifact version becomes active on "it ran without raising" alone.
- **rate limiting**:
  - limit how often automatic patches can be applied.
- **rollback**:
  - ability to revert to prior `artifact_version`.

---

## 11. memory model end-to-end

### 11.1 memory types

1. **working memory**
   - recent conversation history + summary messages in `message` table.
   - small derived summary cached in Redis.

2. **factual memory**
  - files on the shared filesystem under `/users/{user}/files`.
  - embedded chunks in `knowledge_chunk` tied to `knowledge_context`s.

3. **behavioral memory**
  - preference_events in DB.
  - LoRA adapters (weights on the shared filesystem).
  - router state (adapter centroids & stats).

4. **config memory**
   - artifacts (persona summaries, workflows, policies, tools).
   - user settings.

### 11.2 read path per request

1. **auth** → resolve user.
2. **load conversation state**:
   - fetch last N messages or summary from DB/Redis.
3. **embed context**:
   - compute `ctx_embedding` from last user message (+ context).
4. **RAG retrieval (if contexts)**:
   - use `knowledge_context` bound to conversation.
   - select chunks from `knowledge_chunk` over both channels — pgvector and
     postgres FTS — fused by rank, then optionally reranked (§2.5).
5. **router**:
   - find nearest clusters → candidate skills.
   - load routing policy artifact.
   - evaluate rules → `adapter_ids`, `gate_weights`.
6. **workflow**:
   - load chat workflow artifact.
   - execute graph:
     - calls LLM Inference with RAG context + adapter config.
7. **LLM generation**:
   - InferenceService composes LoRA + base; runs decode.
8. **response** streamed back.

### 11.3 write path after response

1. **store assistant message** in DB.
2. **preference extraction**:
   - watch explicit feedback (thumbs, text like “i like this approach”).
   - if found, create `preference_event` with `context_embedding` and `cluster_id`.
3. **cluster update**:
   - clusterer runs periodically to refine `semantic_cluster` and event mappings.
4. **training scheduling**:
  - group new preference_events per `(user, cluster)` → adapter_id.
   - create `training_job`s.
5. **adapter training** (offline):
  - TrainingService updates LoRA weights; writes new version to the shared filesystem.
   - update router state (centroid, metrics).
6. **config evolution**:
   - separate offline “architect” runs LLM to inspect metrics + artifacts.
   - uses ConfigOps to propose routing/workflow changes.

---

## 12. auth & multi-user isolation

### 12.1 auth flows

- **password**:
  - sign up: email + password → store `password_hash`.
  - login: verify hash, create `auth_session`, set secure cookie/JWT.
- **oauth/oidc**:
  - standard provider flows; on callback:
    - map `provider_uid` to existing user or create new.
    - create `auth_session`.
- **session management**:
  - sessions stored in DB + mirrored in Redis for quick lookup.
  - rotation: refresh `id`/`expires_at` every 24h of activity; invalidate old session id after grace period.
  - logout: delete session row + Redis key; add JWT to short-lived denylist if JWTs used.
  - expiry defaults: 7 days web, 1 day mobile; configurable per plan.
  - password reset: `POST /v1/auth/request_reset { email }` issues signed, single-use token stored in Redis with 30m TTL; `POST /v1/auth/complete_reset { token, new_password }` rotates all sessions and refresh tokens.
  - email verification: signed link stored in Redis; user blocked or rate-limited until verified or grace period expires.
  - optional TOTP MFA: `POST /v1/auth/mfa/enable` issues secret + QR; `POST /v1/auth/mfa/verify { code }` required for login/refresh once enabled.
  - WebSockets require `X-Session: <session id>` header or `Authorization: Bearer`; reject mixed transports without fresh session.

### 12.2 isolation

- **tenant**: a tenant *is* a site. `tenant_domains` maps hostname to tenant id;
  the request's hostname decides, and nothing a caller sends can override it —
  no request field, no header. An empty map means the install serves one tenant
  (`default_tenant_id`), which is every deployment until a second site exists.
  Once any mapping exists, a request arriving on an unlisted host is refused
  (`not_found`) rather than served the default tenant, because otherwise any DNS
  name pointed at the box would reach that tenant's login page.
  - one normalizer, shared by the request path and the `tenant_domains`
    validator, so a host an operator can type is a host that can match. They
    were two and disagreed on bracketed IPv6 — settings split at the first
    colon, requests kept the brackets — which made `::1` impossible to map and,
    once bare addresses stopped being exempt, impossible to reach. A bare IPv6
    literal is canonicalized to the bracketed spelling the wire uses.
  - the hostname is read from `Host`, or from `X-Forwarded-Host` when
    `trust_forwarded_host` is on. That flag is the entire trust boundary: turn it
    on only when a reverse proxy you control sets the header from the real
    request and refuses hosts it does not serve. `Host` is a client-supplied
    header like any other.
  - **no host is exempt.** *(fixed — `localhost`, `127.0.0.1`, `::1` and the
    test client's `testserver` used to resolve to `default_tenant_id` even with
    a mapping configured, on the theory that a probe arrives by address rather
    than by site name.)* that exemption was a hole, not a convenience: `Host` is
    chosen by whoever can reach the port, so anyone reaching the service
    directly named the default tenant — and with `allow_signup` on, registered
    an account there. probes do not authenticate and never resolve a tenant, so
    nothing legitimate depended on it. an operator who wants a bare hostname
    served lists it like any other.
  - **a tenanted request has two halves, and both must agree.** the *site* comes
    from the host, resolved through `tenant_domains`. the *account* comes from
    the authenticated session, never from the request. neither is sufficient
    alone, which is why the check is a comparison rather than a lookup: the host
    is attacker-chosen on the unproxied path, and a session is a bearer
    credential that stays valid against whatever site it is replayed at.
    requiring a match means a stolen acme session is useless at globex, and a
    forged `Host` reaches nothing the caller could not already reach.
    `tenancy.user_belongs_to_site` is that rule and `AuthService._site_matches`
    is its single caller-facing form — one method, not one copy per entry
    point, because the copy that gets missed on the next edit is an
    authorization hole. Every way in goes through it: password login, OAuth
    completion, refresh, and every authenticated request. **A blank on either
    side is a mismatch, not a pass**, because the caller with nothing to
    compare is the one that resolved no site. `None` is different from blank —
    it means the caller is not making a tenanted decision at all (logout
    revoking your own session), not that it tried and failed.
  - **OAuth is the same rule, and needed saying separately.** *(fixed — it had
    no check at all.)* `app_user.email` is globally unique, so resolving an
    account by provider id or email finds it whatever site the flow began at.
    The provider proves who someone is, not where they belong: signing in with
    Google at globex used to mint acme's tokens, while the password path
    refused exactly that. Both ways in now agree.
  - **`default_tenant_id` cannot be blank** (`min_length=1`). A blank site
    tenant matches no account under the rule above, so clearing it in the
    console would 401 every user — including the admin who would have to set
    it back, since that route authenticates too. The field refuses instead.
  - signup joins the tenant serving the site it arrived at.
  - `POST /v1/auth/signup` and `POST /v1/auth/oauth/{provider}/start` reject a
    `tenant_id` in the body with `validation_error` rather than ignoring it.
    An admin creates users in their own tenant only; reaching another tenant
    means visiting its site.

- **postgres**:
  - all queries must be filtered by `user_id` where appropriate.
  - Optionally: PostgreSQL Row-Level Security (RLS) to enforce `user_id = current_user_id()`.

- **filesystem**:
  - every access goes through FileService:
    - resolves `user_id` → root path `/users/{user_id}`.
    - rejects any path escape attempts (`..`).
    - enforces visibility of shared/global artifacts separately.
  - signed download URLs for browser fetch; upload size limits per tier enforced at gateway; server joins/normalizes paths to avoid traversal.
  - per-user concurrent workflow caps and rate limits to avoid noisy neighbors; circuit breakers for tools that error repeatedly.

- **artifacts / contexts**:
  - `owner_user_id` + `visibility` field:
    - `private`: only owner.
    - `shared`: selected users/groups (future).
    - `global`: system.

### 12.3 permission model

- minimal initial roles:

  - user:
    - can CRUD their conversations, files, contexts, private artifacts.
    - can see some global artifacts (default routing, workflows).
  - admin:
    - can view system artifacts, approve config patches.

---

## 13. protocols & apis (kernel surface)

principles:

- HTTP+JSON for control planes, WebSocket/SSE for streaming chat; stable versioned paths `/v1/...`.
- every endpoint enforces auth via session cookie or bearer token; `X-User-Id` is ignored/forbidden.
- request/response schemas stored as `artifact` of type `tool.spec` for LLM discoverability.
- responses use envelope `{ "status": "ok|error", "data": ..., "error": { "code", "message", "details" } }`. compatibility surfaces that exist to speak someone else's dialect (`POST /v1/responses` and `POST /v1/mcp`, §13.1) are the exception: they keep that dialect's shape on success **and** on error, because an SDK parses by it.
- pagination uses `page`/`page_size` or opaque `next_cursor`; errors map to HTTP (400 validation, 401/403 auth, 404 missing, 409 conflict, 429 rate limit, 500 server).
- idempotency via `Idempotency-Key` header on POST chat/tool calls; server replays prior response if key repeats within TTL.

### 13.1 chat protocol

- `POST /v1/chat` (start chat turn)

request:

```json
{
  "conversation_id": "optional",
  "message": {
    "content": "string",
    "mode": "text"
  },
  "context_id": "optional knowledge_context id",
  "workflow_id": "optional artifact id override",
  "stream": true,
  "client_request_id": "uuid for idempotency"
}
```

response:

- if `stream=true`: SSE (`event: token`) or WebSocket frames `{event,data}` until `event=done` with `{message_id, usage, adapters, workflow_trace}`.
- if `stream=false`: blocking JSON `{message_id, content, usage, adapters}`.

#### served responses api (`POST /v1/responses`)

the same chat turn in OpenAI's Responses API shape, so any agent framework
that speaks that dialect gets the kernel's enrichment — personas, skill
adapters, RAG, notes, memory — behind a base-model-shaped endpoint. that is
the point: a weak model plus this kernel presents as a much richer model, and
the caller changes nothing but the base URL.

- **wire shapes are OpenAI's, both ways.** success bodies are the bare
  Responses object (never the kernel envelope); error bodies are
  `{"error": {message, type, param, code}}` (never the envelope). the route
  reads the body raw and validates by hand, so malformed JSON and non-object
  bodies get the same 400 shape instead of FastAPI's 422 — and every
  mid-turn failure class is reshaped before it leaves: envelope-styled
  HTTPExceptions (ownership, rate limit, admission), service errors
  (provider failures keep their status, e.g. 502), storage conflicts (409),
  and crashes (a generic 500 — internals never reach the wire); the kernel
  `code` rides in `error.code`. one documented seam: a 401 from the auth
  dependency is still envelope-shaped, because the app-wide auth path is not
  rewritten for one route.
- **stateful by design.** `id` is `resp_<assistant_message_id>`;
  `previous_response_id` resolves through that message to its conversation
  and continues it. ownership is the same owned-conversation check `/v1/chat`
  runs; a foreign or unknown id is 404 either way, so existence is not
  confirmed across users.
- **`context_id` (liminallm extension).** binds a knowledge context on the
  first turn so retrieval grounds the whole thread; continuations inherit the
  conversation's binding, exactly as on `/v1/chat`.
- **streaming.** `stream: true` answers `text/event-stream` speaking the
  OpenAI event dialect: `response.created` → `response.in_progress` →
  server-side tool items as they run (`response.output_item.added` /
  `.done` for `file_search_call` / `web_search_call`, closed before the
  text opens) → the message item and part → `response.output_text.delta`* →
  the `.done` trio → `response.completed` (with full usage and the
  `liminallm` extension), monotonic `sequence_number` throughout, output
  indexes assigned in arrival order. the reply's id is minted before the
  first event, so `created` and `completed` carry the same id and the
  assistant message persists under it. everything that can refuse
  the request refuses before the stream starts, as a proper HTTP error;
  after that, failures are a `response.failed` event (generic on crashes —
  internals never reach the wire), a client disconnect cancels generation,
  and admission slots release however the stream ends.
- **v1 scope line, each rejection named.** caller `tools` (the kernel runs
  its own tool loop server-side), `instructions` (the system prompt belongs
  to per-user personas and adapters — the reason this server exists),
  `store=false` (turns persist to the conversation; that persistence is what
  `previous_response_id` continues). input items accept user text only;
  system/developer items are refused by position, and input is bounded to
  the same 100k-character DoS cap `/v1/chat` enforces — checked as it
  accumulates, not after the join.
- **auth: api keys or session.** `Authorization: Bearer sk-liminal-…` — keys
  minted at `POST /v1/auth/api-keys` (§13.2), stored as sha-256 in
  `user_api_key` (§2.1), plaintext shown once. keys authenticate **only the
  agent surfaces** (`/v1/responses` and `/v1/mcp`): the native routes'
  dependency cannot read them, so a leaked key can drive chat turns and
  retrieval and nothing else — it cannot list conversations, mint another
  key, or revoke one. keys skip session/mfa machinery (minting one already
  required a fully authenticated session) but never the tenant check.
  session jwts also work here, so the web ui can drive the surface.
- **the thread is a native conversation.** turns land in the same
  conversation store; the web ui lists them beside the rest, badged via
  `source: "responses"` (from `conversation.meta`, exposed on the
  conversation endpoints), and title generation, sharing, compaction and
  retention behave exactly as on `/v1/chat`.
- **kernel tool use keeps its transport.** the loop behind this surface is
  the /chat agentic loop: internal tool calls — retrieval, notes, the
  reranker's out-of-band verdict (§2.5) — ride the provider tool channel
  wherever one exists, including `local_lora` / `local_gpu_lora` through the
  advertised `<tool_call>` channel (§5.0.2). callers see only the final
  text; caller-supplied tools stay rejected (the scope line above).
- **same budget, same gate.** the `/v1/chat` rate bucket and admission slots
  are shared deliberately: this is a chat turn, and a second bucket would be
  a second limit to misconfigure.
- `model` echoes the serving model; `metadata` is bounded (16 keys, 64/512
  chars) and echoed back. `usage` serves everything the turn learned: the
  three totals, plus `input_tokens_details.cached_tokens` and
  `output_tokens_details.reasoning_tokens` when the upstream reported them —
  on any ingestion transport: the Responses api, chat.completions
  (vLLM-style prefix caching reports cached_tokens there), or gemini's
  usageMetadata. the compat layers carry both through the agent loop, and
  the details objects are always present, zeros when unknown, because typed
  SDKs require the fields. on `local_lora`/`local_gpu_lora` the counts come
  from our own tokenizer, and `cached_tokens` is the KV prefix the local
  backend genuinely reused this turn (§5.3) — earned, not estimated.
- **server-side tool runs are served, not hidden.** tool activity appears in
  `output` as the dialect's own items — `file_search_call` (with queries)
  and `web_search_call` — and only those: dialect-native types keep typed
  SDK parsers away from unknown discriminators, so note_search,
  history_search, run_python and web_fetch are never dressed up as
  something they are not. the FULL trace, the grounding snippets and the
  active adapters ride under one namespaced top-level key, `liminallm`
  (`{context_snippets, tool_trace, adapters}`) — extra keys survive the
  openai sdks and stay invisible to strict readers. citations are NOT faked
  into `annotations`: an annotation needs a character anchor and a file
  identity this surface cannot honestly provide, so `annotations` stays
  empty until the model actually cites and provenance rides the extension.

#### mcp server (`POST /v1/mcp`)

the kernel's retrieval, spoken in the Model Context Protocol so any
mcp-speaking agent can ground itself in a user's vault and knowledge
contexts without adopting the responses api. same credentials as
`/v1/responses` (api key or session), same tenant check, envelope-free wire
(json-rpc is the dialect here, and the §13 envelope exception covers it).

- **protocol subset, honestly drawn.** streamable http, one POST endpoint,
  json responses only; protocol revision 2025-06-18 (2025-03-26 accepted on
  initialize). implemented: `initialize`, `ping`, `tools/list`,
  `tools/call`; notifications answer 202 with no body. not implemented: no
  sessions (stateless — `Mcp-Session-Id` ignored), no server-initiated
  stream (GET answers 405), no resources or prompts yet. json-rpc batching
  was removed from the protocol in 2025-06-18 and is rejected by name.
- **two tools, both read-only, both the kernel's own.** `note_search` ranks
  the notes vault (the same bm25+semantic fusion `note_search` uses inside
  the agent loop); `knowledge_search` retrieves from knowledge contexts
  through the full hybrid pipeline (§2.5), scoped to one owned context via
  `context_id` or across everything the user owns. ownership verdicts match
  the http surface: absent is absent, foreign is refused — as tool errors,
  not protocol errors.
- **read-only is the security posture, not a v1 shortcut.** these tools
  reach nothing outside the install, so an injected document has no egress
  to abuse here, and every result opens by naming its own text as document
  content, never instructions — the same untrusted-data doctrine the kernel
  repeats at every boundary.
- **roadmap, so growth is a decision and not drift:** resources (notes and
  chunks addressable by uri, with subscriptions as a change feed); prompts
  (personas and prompt-mode skills offered as mcp prompts);
  `tools/list_changed` notifications when artifacts change the tool set;
  oauth 2.1 + protected-resource metadata (rfc 9728) so standard mcp
  clients onboard without pasting keys; structured tool output
  (`structuredContent`) beside the text; and — the consequential one — an
  mcp **client** in the kernel loop, consuming external servers as tools
  under the taint discipline: each server assigned a taint class, egress
  withdrawal extended to third-party tools, so outside capability never
  outruns the injection defenses.

### 13.2 auth/session api (minimal definitions)

- `POST /v1/auth/signup { email, password }` → create user.
- `POST /v1/auth/login { email, password }` → set session cookie / bearer token.
- `POST /v1/auth/oauth/{provider}/start` + `GET /v1/auth/oauth/{provider}/callback` (standard OAuth).
- `POST /v1/auth/logout` → revoke session.
- `POST /v1/auth/refresh` → rotate session/refresh token.
- responses include `session_expires_at`; headers `Set-Cookie: session_id=...; HttpOnly; Secure` when cookies are used.
- `POST /v1/auth/mfa/verify` when MFA enabled; returns new session + requires one-time recovery code flow if user is locked out.
- `POST /v1/auth/api-keys { name }` → mint a key for the agent surfaces (`/v1/responses`, `/v1/mcp`); the plaintext appears only in this response. `GET /v1/auth/api-keys` lists them (prefix only, revoked included — that is the audit view); `DELETE /v1/auth/api-keys/{key_id}` revokes immediately. session auth only, envelope-shaped, at most 20 active keys per user; a key can never manage keys.

### 13.3 files & contexts

- `POST /v1/files/upload` — multipart; stores under `/users/{u}/files`; returns `fs_path`; optional `context_id` form field triggers chunking + embedding ingestion into that knowledge context.
- `GET /v1/files` — list user files (paginated); returns `{ files: [...], total, has_next }`.
- `GET /v1/files/{filename}/url` — get signed download URL; returns `{ download_url, expires_at }`; URL valid for 10 minutes.
- `GET /v1/files/download?path=...&expires=...&sig=...` — download file with validated HMAC signature; returns binary file with `Content-Disposition: attachment`.
- `DELETE /v1/files/{filename}` — delete user file; returns `{ deleted: true }`.
- `POST /v1/contexts` — create `knowledge_context`, attach file paths.
- `GET /v1/contexts?limit=N` — list contexts + stats; supports `?owner=me|global`.
- `GET /v1/contexts/{id}/chunks?limit=N` — list chunks for a context; default limit 100, max 500.

### 13.4 artifacts

- `GET /v1/artifacts?type=workflow|policy|adapter|tool&visibility=private|shared|global&limit=N&page=N&page_size=N` — list accessible artifacts; `limit` is accepted as alias for `page_size`.
- `GET /v1/artifacts/{id}` — fetch current version + metadata.
- `POST /v1/artifacts` — create; validates `schema.kind` using per-kind schema.
- `PATCH /v1/artifacts/{id}` — update via JSON Patch; writes new `artifact_version`.
- `GET /v1/artifacts/{id}/versions?limit=N` — list versions; default limit 100, max 500.

### 13.5 config ops

- same endpoints as §10; PATCH application triggers validation + dry-run.

### 13.6 migrations (basic shell tool)

- repository includes `scripts/migrate.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
psql "$DATABASE_URL" -v embedding_dim="${EMBEDDING_VECTOR_DIM:-1536}" -f sql/schema.sql
# add future numbered files in order
```

- no special tooling; developers add ordered `sql/*.sql` files; CI runs script; idempotency via `CREATE TABLE IF NOT EXISTS` inside SQL.
- optional seeding happens inside numbered SQL (idempotent upserts) to create default workflow, routing policy, and tool specs as artifacts; keep seeds versioned so reruns are safe.

---

## 14. implementation phases (minimal-first)

### phase 0: vanilla chat + files

- implement:
  - users, auth, conversations, messages.
  - FileService + filesystem.
  - a single global `workflow.chat` that just calls `llm.generic`.
  - no LoRA, no preferences, no clusters.

### phase 1: RAG + artifacts

- add:
  - `knowledge_context`, `context_source`, `knowledge_chunk`.
  - ingestion + embedding jobs.
  - RAG tool (`rag.answer_with_context_v1`).
  - workflows that branch into RAG vs plain chat.
  - artifact table for workflows + tools.

### phase 2: preferences + persona adapter

- add:
  - `preference_event`, `training_job`.
  - a single per-user `adapter.lora` for persona.
  - minimal TrainingService to update persona adapter from positive events.
  - RouterService: always apply persona adapter with fixed gate.

### phase 3: clusters + skill adapters

- implement:
  - `semantic_cluster` and clustering job.
  - `adapter_router_state`.
  - skill adapter creation based on clusters + preference events.
  - Router policy as data (basic form).
  - Router engine that uses similarity to activate skill adapters.

### phase 4: LLM as architect

- implement:
  - `config_patch` table.
  - ConfigOps API + admin UI.
  - LLM “architect” job that:
    - reads metrics/summary.
    - proposes patches to routing/workflows.
  - validation + sandboxing.

always keep the kernel small:

- no new hard-coded “modes”; always introduce new behaviors as artifacts.

---

## 15. safety, monitoring, operations

### 15.1 safety layer

- content safety classifier (small model) applied:
  - on user inputs (for logging / abuse detection).
  - on assistant outputs (to filter / edit before sending).
- training filters:
  - only create `preference_event` if the interaction is policy-compliant.
  - never train adapters on disallowed content.

### 15.2 observability

metrics (per service):

- request latency, error rates.
- tokens in/out per call.
- adapter usage counts & success_score.
- preference_event rates.
- training job counts and average loss.
- workflow traces: per-node latency, retries, timeout counts.

logs:

- structured logs with correlation IDs for each chat request.
- include routing trace (rules fired, adapters activated) and workflow trace (nodes executed, errors).
- redact PII where possible; configurable log sampling for payloads.
- retention defaults: metrics 7–14 days (Prometheus), logs 30–90 days with payload sampling; alerts on ingestion lag, adapter cache miss spikes, training failure bursts.

traces:

- optional OpenTelemetry traces:
  - gateway → orchestrator → router → workflow → inference → training.
- dashboards/alerts:
  - SLOs on chat latency and token error rates.
  - alerts on adapter cache misses > threshold, training job failure spikes, ingestion lag.

---

## 16. core code boundaries (“small kernel”)

**the code should only “know” how to:**

- authenticate users & enforce isolation.
- CRUD generic artifacts, conversations, files.
- run a workflow graph.
- run a routing policy.
- call LLM with LoRA adapters.
- do basic cluster updates.
- enforce safety, validation, and resource limits.

**it should *not* know**:

- what “debugging” vs “creative writing” is.
- which adapters are “good” for which tasks.
- which workflows should be used for which conversations.

those live as data (artifacts, clusters, policies) that the LLM can inspect and evolve via ConfigOps.

that’s the whole point: minimal glue, maximal evolution.

---

## 17. front-end expectations (LLM-visible, thin client)

- single-page app speaking the public APIs; no domain knowledge baked in.

### 17.1 layout architecture

- **sidebar-main layout**: persistent conversation list sidebar (280px) with main content area.
- **tab navigation**: primary tabs organize functionality:
  - **Chat**: conversation interface with message streaming
  - **Notes**: the vault — editor, link graph, witness (§19; hidden when `notes_enabled` is off)
  - **Contexts**: knowledge context management
  - **Files**: knowledge upload and the user's file browser
  - **Artifacts**: system artifact browser
  - **Tools**: tool specs and workflows
  - **Insights**: preference clusters
  - **Settings**: user preferences and session info
- tab data loads lazily on first activation; login preloads only what the chat needs.
- responsive breakpoints: sidebar hidden on mobile (<1080px), single-column tabs on small screens (<640px).

### 17.2 conversation sidebar

- **conversation list**: paginated list of user conversations sorted by `updated_at`.
- **search**: client-side filter by title or conversation ID.
- **active indicator**: highlight currently loaded conversation.
- **source tag**: conversations with `source: "responses"` carry a small “api” tag — agent-created threads sit beside native ones, visibly distinct.
- **new conversation**: button to reset chat state and start fresh thread.
- API endpoints: `GET /v1/conversations`, `GET /v1/conversations/{id}`, `GET /v1/conversations/{id}/messages`.

### 17.3 chat view (Chat tab)

- **message stream**: scrollable container with message bubbles differentiated by role (user/assistant/system).
- **token streaming**: WebSocket primary with HTTP fallback; display blinking cursor during streaming; accumulates tokens into message bubble in real-time; supports cancel via connection close.
- **citation rendering**: inline clickable links for citations from `content_struct.citations`; each citation shows source filename/path as tooltip.
- **context binding**: dropdown to select active `knowledge_context` for RAG-grounded responses.
- **workflow override**: optional text input for `workflow_id` to steer execution.
- **optimistic UI**: user messages displayed immediately before server confirmation.
- **collapsible sections**:
  - **Preferences**: thumbs up/down feedback with optional notes, displays routing metadata and trace.
  - (Knowledge upload and the file browser live in the Files tab — they are
    persistent panels, not conversation state, and pinning them above the
    message stream cost the chat column ~300px that never scrolled away.)
- **typography**: assistant prose is set in a serif column with a github-grade
  markdown renderer (escape-first: html-escape, then rewrite to a fixed safe
  tag set; nested/task lists, aligned tables, blockquotes, autolinks,
  backslash escapes, and lightweight syntax highlighting across nine language
  families). streaming batches dom writes with `requestAnimationFrame`,
  auto-closes a dangling code fence mid-stream, and only auto-scrolls when the
  reader is already near the bottom.
- **per-message controls**: a copy button on every user and model message.
- **tool activity**: while the agent loop runs, the typing indicator names the
  tool in flight; injection findings surface as a warning on the message.
- **attachments**: drag-and-drop or attach-button chips in the composer;
  uploads bind to the conversation automatically (§19.5 tier 1).
- **sharing**: conversations are private by default. a "Share It" control
  publishes a read-only page; the public directory and shared pages carry
  `noindex, nofollow` and a matching `robots.txt`, so sharing never means
  indexing.
- **turn rail**: a right-hand rail of tick marks, one per turn, labeled with a
  model-written description of that turn. hovering (or focusing) expands the
  bars into a selector for jumping between turns; moving away collapses them
  back. conversation titles are model-written too — never raw uuids.

### 17.4 context manager (Contexts tab)

- **context creation form**: name (required) and description fields; `POST /v1/contexts` on submit.
- **contexts list**: card layout showing context name, description, ID prefix, and creation date.
- **context selection**: click to load details; selected context highlighted.
- **context details panel**:
  - full ID, description, visibility badge, creation timestamp.
  - chunk count and preview of recent chunks via `GET /v1/contexts/{id}/chunks`.
- **context selects**: chat and upload dropdowns populated from `state.contexts` and updated on context CRUD.

### 17.5 artifact browser (Artifacts tab)

- **filter controls**:
  - type dropdown: all, workflow, policy, adapter, tool.
  - visibility dropdown: all, private, shared, global.
- **artifacts table**: sortable columns for type, name, visibility, version, updated date.
- **type badges**: color-coded labels (workflow=blue, policy=pink, adapter=green, tool=amber).
- **visibility badges**: color-coded (private=red, shared=amber, global=green).
- **artifact details panel**:
  - header with name and type badge.
  - detail rows: ID, description, version, owner.
  - **schema viewer**: syntax-highlighted JSON display of `artifact.schema`.
- **version history table**: list of `artifact_version` entries with version number, timestamp, and change summary.
- API endpoints: `GET /v1/artifacts`, `GET /v1/artifacts/{id}`, `GET /v1/artifacts/{id}/versions`.

### 17.6 settings panel (Settings tab)

- **session information**: display user ID, role, tenant, truncated session ID.
- **local storage management**:
  - draft count indicator.
  - clear drafts button (removes all from localStorage).
  - export drafts button (downloads JSON file).
- **upload limits**: display max file size and allowed extensions from `GET /v1/files/limits`.
- **api keys**: mint/list/revoke keys for the agent surfaces (§13.1) against `/v1/auth/api-keys`; the plaintext renders once at mint time, the list shows prefix + created/last-used/revoked, revoke confirms before firing.
- **about section**: version and build info from `/healthz`.

### 17.7 draft persistence (offline-safe)

- drafts stored in localStorage under key `liminal.drafts` as `{ [conversationId]: { text, savedAt } }`.
- auto-save: 1-second debounce on message input changes.
- draft restoration: on conversation load, restore any saved draft to input field.
- draft indicator: displays count of saved drafts in chat input area.
- new conversation drafts stored under key `_new`.

### 17.8 file upload

- **upload section**: collapsible panel within Chat tab.
- **context selection**: dropdown to choose target context (or private/no context).
- **chunk size**: optional numeric input (64–4000 range validation).
- **file validation**: client-side checks for size limit and allowed extensions before upload.
- **upload status**: progress and result feedback inline.
- API endpoint: `POST /v1/files/upload` with multipart form data.

### 17.9 feedback controls

- **thumbs up/down buttons**: visible in preferences section; disabled until assistant message exists.
- **notes field**: optional free-text input for additional feedback context.
- **target display**: shows conversation ID and message ID of feedback target.
- **metadata display**: JSON preview of adapters, context snippets, and adapter gates.
- **routing trace display**: JSON preview of routing and workflow traces when available.
- API endpoint: `POST /v1/preferences`.

### 17.10 auth flow

- **auth panel**: shown when not authenticated; hidden after successful login.
- **login form**: email, password, optional MFA code. No tenant field — the tenant is the site you visited (§12.2), never something a user types.
- **MFA handling**: if `mfa_required` returned without token, prompt user to enter code.
- **token management**: access token, refresh token, session ID stored in sessionStorage.
- **auto-refresh**: on 401 response, attempt token refresh before failing.
- **logout**: calls `POST /v1/auth/logout`, clears storage, reloads page.

### 17.11 API integration patterns

- **request headers**: `Authorization: Bearer`, `X-Tenant-ID`, `session_id`, `Idempotency-Key` (auto-generated UUID).
- **envelope handling**: parse `{ status, data, error }` responses; extract error messages from `error.message` or `detail`.
- **retry logic**: exponential backoff (400ms base, 3 retries) for 5xx errors; no retry on 4xx.
- **WebSocket protocol**: connect to `/v1/chat/stream`; send auth + message in initial frame `{ access_token, session_id, tenant_id, message, conversation_id?, context_id?, workflow_id?, stream?: bool }`; when `stream: true` (default), receive streaming events `{ event: "token"|"trace"|"message_done"|"error"|"cancel_ack", data: ... }`; when `stream: false`, receive single envelope `{ status, data: ChatResponse }`. Frontend displays blinking cursor during streaming and accumulates tokens into message bubble.

### 17.12 styling system

- CSS custom properties for theming: `--accent`, `--text`, `--panel`, `--border`, etc.
- component classes: `.panel`, `.badge`, `.table`, `.code-block`, `.context-card`, `.type-badge`, `.visibility-badge`.
- utility classes: `.hidden`, `.flex-row`, `.pill-row`, `.divider`, `.mb-14`, `.monospace`.
- responsive: media queries at 1080px (hide sidebar) and 640px (single-column layout).

---

## 18. implementation details (locked, kernel-safe)

the following are treated as constants the kernel must honor; LLM edits happen only to data artifacts, not to these guardrails.

- **API envelopes & transports**
  - success: `{ "status": "ok", "data": <payload>, "request_id": "uuid" }`; error: `{ "status": "error", "error": { "code": "string", "message": "string", "details": <object|array|null> }, "request_id": "uuid" }`.
  - pagination: either `{ data: [...], next_cursor: "opaque" }` or `{ page, page_size, total }`; choose per-endpoint but keep stable once published. For simple bounded queries, `limit` is accepted as an alias for `page_size` (defaults to 100, max 500).
  - idempotency: POST endpoints that create side effects (`/v1/chat`, `/v1/tools/run`, `/v1/artifacts`) accept `Idempotency-Key`; server replays prior response within a 24h TTL and returns `409` if the prior attempt is still running.
  - auth header is `Authorization: Bearer <token>` in REST; WebSockets accept inline auth in the initial message frame: `{ "access_token": "...", "message": "...", ... }` **or** `{ "session_id": "...", ... }` — exactly one, never both (§12.1: mixed transports are rejected without a fresh session). No `tenant_id`: the socket's tenant comes from the host it was opened against, like every HTTP route. Unauthenticated sockets close with code `4401`.
  - streaming events: `token`, `message_done`, `error`, `cancel_ack`, `trace` (router/workflow trace snapshot). SSE uses `event:` labels; WebSockets wrap as `{ "event": "token", "data": "..." }`.
  - minimal REST surface (kernel-stable):
    - `POST /v1/auth/login { email, password, mfa_code? } → { access_token, refresh_token, user }`.
    - `POST /v1/auth/refresh { refresh_token } → { access_token, refresh_token }`.
    - `POST /v1/chat { conversation_id?, message, context_ids?, artifact_ids?, stream: bool } → { conversation_id, message_id, stream_id? }`; stream events carry `{ event, data, request_id }` with `trace` payloads showing router/workflow steps.
    - `POST /v1/chat/cancel { request_id }`.
    - `GET /v1/conversations?limit=N` returns paginated conversation list; `GET /v1/conversations/{id}` returns single conversation; `GET /v1/conversations/{id}/messages?limit=N` returns messages.
    - `POST /v1/artifacts { type, name, schema, visibility?, fs_path? }` and `PATCH /v1/artifacts/{id}`; both emit a new `artifact_version` row and validate JSON Schema against `type` registry.
    - `POST /v1/config/patches { artifact_id, patch, justification }` queues a ConfigOps proposal; `POST /v1/config/apply { patch_id }` (admin-only) applies a validated patch.
    - `POST /v1/tools/run { tool_id, input }` executes a tool node outside a workflow (for testing) with the same retry/timeout caps.
  - errors MUST use stable `error.code` values: `unauthorized`, `forbidden`, `not_found`, `rate_limited`, `validation_error`, `conflict`, `server_error`; HTTP codes mirror the error (`401/403/404/429/400/409/500`).
  - constraint violations (FK/unique) return `conflict` with a short `details` map identifying the offending field/id; kernel surfaces storage errors instead of leaking database-specific messages.

- **auth/session flows (minimal, deterministic)**
  - password reset: `POST /v1/auth/request_reset { email }` stores a one-time token in Redis (15m TTL) and emails it; `POST /v1/auth/complete_reset { token, new_password }` rotates credentials and revokes sessions.
  - email verification: `POST /v1/auth/verify_email { token }` marks `user.meta.email_verified=true`; unverified accounts are limited to 24h and low rate limits.
  - MFA: `POST /v1/auth/mfa/enable` returns TOTP secret + QR; `POST /v1/auth/mfa/verify { code }` gates login/refresh when `user.meta.mfa_enabled=true`; 5 failed codes locks MFA for 5 minutes.
    - TOTP is **HMAC-SHA-1, 6 digits, 30s**, with a 160-bit secret (RFC 6238 / RFC 4226 §4 R6). These are the Key Uri Format defaults, so an authenticator app assumes them whatever the `otpauth://` URI omits — the server must verify the same thing its own QR code promises. It verified SHA-256 while every app computed SHA-1, so enrolment could never complete and nothing said why. The URI now states `algorithm`, `digits` and `period` explicitly rather than relying on the defaults holding.
  - session model: short-lived access token (15–60m configurable) + refresh token (7–30d) stored HttpOnly; refresh rotation on each use; logout revokes both; login from a new device invalidates prior refresh tokens if `meta.single_session=true`.

- **multi-tenant isolation & filesystem guards**
  - all filesystem paths resolved via `safe_join(base=/users/{user_id}, relative)` unless `artifact.visibility in ('shared','global')` points into `/shared`; path traversal or `..` segments are rejected.
  - uploads enforce per-plan size caps (e.g., free: 25MB/file, paid: 200MB/file) at gateway; downloads use signed URLs with 10m expiry and content-disposition set to prevent inline execution.
  - per-user scratch `/users/{id}/tmp` auto-cleans daily; no cross-user hardlinks.

- **safety & resource limits**
  - rate limits (Redis token bucket): defaults `chat: 60 req/min`, `files.upload: 10 req/min`, `configops: 30 req/hour`, adjustable per plan; 429 response uses standard error envelope.
  - concurrency caps: max 3 concurrent workflows and 2 concurrent inference decodes per user; requests beyond cap return `409 busy`.
  - external fetches from tools use a allowlisted proxy with 10s connect + 30s total timeout; circuit breaker opens for a tool after 5 failures in 1 minute.

- **workflow/tool sandboxing**
  - tool workers run in a spawned child process under POSIX rlimits (memory hard cap `RLIMIT_AS`, CPU seconds, max file size, no core dumps), backstopped by a wall-clock kill, and have no filesystem access except a tmp scratch; `privileged:true` tools require admin-owned artifacts and are never called by default workflows.
  - JSON Schema validation enforced on tool inputs/outputs; outputs flagged `content_type: "html_untrusted"` must be sanitized by client before render.
  - retries: default 2 retries with exponential backoff (1s, 4s); per-node override allowed but capped at 3; node timeout default 15s, hard cap 60s.

- **inference/adapter cache discipline**
  - per-GPU adapter cache budget configured in bytes (e.g., 6GB); eviction LRU with pinning for persona adapter of active user; checksum of `params.json` verified against `schema.checksum` before activation.
  - per-request adapter cap = 3; if router selects more, lowest-weight adapters are dropped and the trace records the drop.
  - cancellation: orchestrator issues `{event:"cancel", request_id}`; worker aborts decode, frees KV cache and adapter refs, and emits `cancel_ack` with partial tokens if any.

- **adapter mode configuration**
  - `default_adapter_mode` admin setting (default: `hybrid`): controls mode for newly created adapters.
  - valid values: `local`, `remote`, `prompt`, `hybrid` (see §5.0.1 for mode definitions).
  - `model_backend` determines which adapter modes are compatible:
    - `local_lora`/`local_gpu_lora`: supports `local`, `prompt`, `hybrid`
    - API backends (`openai`, `together`, `lorax`, etc.): support `remote`, `prompt`, `hybrid`
  - router automatically filters incompatible adapters before policy evaluation; filtered adapters logged with `adapter_filtered_by_mode` event.
  - existing adapters without `mode` field are migrated on first access: `backend=local` → `local` or `hybrid` (if has prompt_instructions); `backend=api/remote` → `remote`.

- **training pipeline knobs**
  - dataset: JSONL on the shared filesystem `/users/{u}/adapters/{id}/train_jobs/{job}/dataset.jsonl` with fields `{prompt, target, weight, context, conversation_id, message_id}`; max 2k tokens per sample; dedupe by `(conversation_id, message_id)`.
  - evaluation: hold-out 10% of most recent events; metrics: loss and preference alignment rate; apply new adapter version only if both improve or if human approves via ConfigOps; otherwise keep previous version.
  - scheduling: one running job per user; cooldown 1h between jobs; global queue fair-shares across users to avoid single-tenant starvation.

- **knowledge ingestion hygiene**
  - dedupe by `(fs_path_checksum, path)`; skip files over plan cap or unknown mime type unless a `tool.spec` parser declares support; optional PII scrub set per context (`context.meta.pii_scrub=true`).
  - re-embed on encoder bump with rolling replacement: write new chunks with `meta.embedding_version`, switch pointer when >=95% ready, then delete old chunks; ingestion lag surfaced in metrics.

- **observability & ops defaults**
  - metrics retention 14d (Prometheus) with alerts on latency SLO breaches, adapter cache miss rate > 20%, training failure rate spikes, ingestion lag > 1h; logs 30–90d with payload sampling and PII minimization.
  - backups: nightly Postgres logical backup retained 7d; weekly filesystem snapshot pointers retained 4 weeks; Redis not backed up (ephemeral) but seeded data survives via Postgres + filesystem artifacts.
  - health checks: `/healthz` per service does dependency checks (DB, Redis, filesystem mount) and reports build/version; readiness gates traffic in orchestrator/gateway.

- **migrations & seeding**
  - `scripts/migrate.sh` is the only required tool; it applies ordered `sql/*.sql` files and optional `sql/seed/*.sql` that upsert default artifacts (workflow, routing policy, base tool specs); rerunning is safe due to `IF NOT EXISTS` and deterministic upserts.
  - CI runs migrations on a fresh DB to validate schema; production runs migrations during maintenance windows with `DATABASE_URL` from environment and fails fast on checksum mismatch.

- **configuration management (database-driven)**
  - **principle**: most operational settings MUST be database-managed and editable via admin/instance-admin UI (`/admin.html`, `GET/PUT /v1/admin/settings`); environment variables serve only as bootstrap defaults or for infrastructure/secrets that cannot safely reside in the database.
  - **database-managed settings** (modifiable at runtime without restart):
    - session & concurrency: `session_rotation_hours`, `session_rotation_grace_seconds`, `max_concurrent_workflows`, `max_concurrent_inference`
    - rate limits: `chat_rate_limit_per_minute`, `login_rate_limit_per_minute`, `signup_rate_limit_per_minute`, `reset_rate_limit_per_minute`, `mfa_rate_limit_per_minute`, `admin_rate_limit_per_minute`, `files_upload_rate_limit_per_minute`, `configops_rate_limit_per_hour`, `read_rate_limit_per_minute` and their window/multiplier variants
    - pagination & files: `default_page_size`, `max_page_size`, `default_conversations_limit`, `max_upload_bytes`, `rag_chunk_size`
    - token TTL: `access_token_ttl_minutes`, `refresh_token_ttl_minutes`
    - feature flags: `enable_mfa`, `allow_signup`
    - training worker: `training_worker_enabled`, `training_worker_poll_interval`
    - notes vault: `notes_enabled` (see §19)
    - SMTP (all settings including secrets): `smtp_host`, `smtp_port`, `smtp_user`, `smtp_password`, `smtp_security`, `email_from_address`, `email_from_name`
      - `smtp_security` is `starttls` (default, usually port 587), `ssl` (encrypted from the first byte, usually 465) or `none`. `none` exists for a relay on the same machine and is refused when a username is set, since the password would cross the wire in the clear. It replaces a `smtp_use_tls`/`smtp_allow_insecure` pair in which the second flag never enabled plaintext at all — it only removed a port guard in front of an SSL connection, so the ordinary self-hosted arrangement could not send.
    - URL settings: `oauth_redirect_uri`, `app_base_url`
    - voice settings: `voice_transcription_model` (enum: whisper-1), `voice_synthesis_model` (enum: tts-1, tts-1-hd), `voice_default_voice` (enum: alloy, echo, fable, onyx, nova, shimmer)
    - model settings: `model_path` (with common suggestions: gpt-4o, gpt-4o-mini, gpt-5.2, claude-opus-4-5, claude-sonnet-4, glm-4-plus), `model_backend` (enum: openai, anthropic, azure, azure_openai, vertex, gemini, google, bedrock, together, together.ai, lorax, adapter_server, sagemaker, aws_sagemaker, zhipu, zhipu.ai, glm, stub), `default_adapter_mode` (enum: local, remote, prompt, hybrid), `rag_mode` (enum: pgvector, memory), `embedding_model_id` (enum: text-embedding, text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002), `rag_rerank` (enum: auto, on, off), `rag_rerank_candidates` (int, 2–100), `rag_late_interaction` (bool, default off), `rag_late_segments` (int, 2–32)
    - tenancy: `default_tenant_id`, `tenant_domains` (host → tenant id),
      `trust_forwarded_host`; JWT: `jwt_issuer`, `jwt_audience`
  - **environment-only settings** — everything that must be known *before* the
    database is readable, or that describes the machine rather than the install.
    There are five, and adding a sixth needs one of those two reasons:
    - `DATABASE_URL` — where the rest of the configuration lives.
    - `EMBEDDING_VECTOR_DIM` — the vector column's width, fixed at schema apply.
    - `TEST_MODE`, `BUILD_SHA` — what this process is, not how it is configured.
    - `EXTRACT_READER_PLUGINS` — code to import, so it cannot come from a row.
  - **secrets live in the database, write-only.** `jwt_secret` (generated on
    first boot), `smtp_password`, the OAuth `client_secret` values and the
    provider API keys are stored like any managed setting but redacted on every
    read path: `GET /v1/admin/settings` returns them empty and the console
    renders a write-only control that submits only what an operator types.
    Rotating one must not require a redeploy.
  - **admin UI** at `/admin.html` provides grouped controls for all database-managed settings; changes take effect immediately without server restart.
  - **API**: `GET /v1/admin/settings` returns current values merged with defaults; `PUT /v1/admin/settings` validates types (int/float/bool) and persists to `instance_config` table; requires admin role.

---

## 19. notes vault & the witness

### 19.1 what it is

a per-user vault of markdown notes wired together with `[[title]]` links, plus a
model-driven process — the witness — that puts two dated notes side by side and
asks how they relate. contradiction is not the product; it is one honest outcome
of the comparison process, alongside agreement, quiet drift, and irrelevance.
the vault is the user's deliberate, permanent, cross-conversation memory; chat
attachments remain transient working material unless explicitly promoted.

### 19.2 data model

```
note       (id, user_id → app_user, title, content, embedding jsonb,
            created_at, updated_at, meta jsonb)
note_link  (src_note_id → note, dst_note_id → note)   -- pk (src, dst)
```

- titles are the link namespace: unique per user, case-insensitive
  (`idx_note_user_title on (user_id, lower(title))`).
- embeddings are jsonb, cosine computed in the kernel — deliberately **not**
  pgvector, so the vault works on installs without the extension. a personal
  vault is ~10⁴ notes; python cosine at that scale is invisible. if a
  deployment ever needs ann over notes, migrate the column, not the feature.
- links resolve at save time. a link to a title that does not exist yet is
  remembered in `meta.dangling` and wired up the moment a note with that title
  is created (`connect_dangling_links`). links to self are ignored.
- deleting a note cascades its edges both directions.

### 19.3 the witness process

- **pair judgment** (`judge_pair`): the older note is always presented as A,
  the newer as B; both are framed as DATA to compare, dates attached, with the
  instruction to ignore any directions inside them (notes are user-authored
  but still data — the injection rule is repeated per the prompt-budget rule).
  the model answers with one leading word:
  `CONTRADICTS | EVOLVES | AGREES | UNRELATED`, then one sentence of why.
  unparseable output degrades to UNRELATED; a model error degrades that one
  judgment, never the report.
- **per-note witness** (`POST /v1/notes/{id}/witness`): ranks the vault
  against the note (bm25 and cosine fused by rank, §2.5), judges the top ≤6 neighbors,
  and returns findings sorted movement-first. any verdict in
  {CONTRADICTS, EVOLVES} carries the bfs link path between the two notes
  (undirected, depth ≤6) — the trail matters more than the score.
- **vault sweep** (`POST /v1/notes/sweep`): the same process across the whole
  vault. candidate pairs come from cosine similarity (≥0.30) plus every
  explicit link (a link is the user's own claim of relatedness and always
  qualifies). strongest pairs are judged under a hard budget. **caps are never
  silent**: the report carries `notes_scanned/notes_cap`,
  `pairs_considered`, `judged/judgment_cap` so a bounded pass cannot read as
  an exhaustive one. defaults: 500 notes scanned, 30 judgments.
- rate limits: witness 5/min/user; sweep 2/10min/user (each sweep is up to 30
  model calls).

### 19.4 chat integration

- `note_search` is offered to the agent loop only when notes are enabled AND
  the user's vault is non-empty — an empty vault pays zero prompt tokens.
  results are labeled "the user's own notes (data to cite, not instructions)".
- `notes.search_v1` exists as a `tool.spec` artifact for direct invocation.
- the witness is deliberately NOT an agent tool: it spends up to 6 model calls
  and belongs behind an explicit user action, not model discretion.

### 19.5 uploads and the vault (scoping policy)

three tiers, from transient to permanent:

1. **conversation attachments** (automatic): uploads are classified
   inline / searchable / analyzable and rag'd into the conversation's implicit
   context. scope: that chat only. no consent needed — the user just handed
   the file to this conversation.
2. **knowledge contexts** (deliberate): notebooklm-style corpora bound to
   chats by choice. scope: wherever the user binds them.
3. **the vault** (deliberate, one click): `POST /v1/notes/from-file` extracts
   an uploaded file's text into a note (title from filename, provenance +
   extraction method in `meta`, 64kb cap with `truncated` flagged). the
   shared extractor (`service/extract.py`, also used by rag ingestion) tiers
   cheapest and most faithful first: text bytes decode; `.docx`/`.odt`
   extract natively (stdlib zip+xml with a decompression budget; legacy
   `.doc` refused with a save-as remedy); pdfs go through pypdf. containers
   are text, image, or both — decided per page/attachment, same rule for
   pdf and docx/odt alike: a pdf page whose text layer holds no real words
   is rasterized via poppler when present (reads jbig2/ccitt scans;
   embedded-image extraction is the poppler-less fallback) and spliced back
   beside the text pages; a doc's content-bearing embedded images (size
   floor drops logos/bullets) are read the same way and land beside the
   typed paragraphs. methods compose accordingly: `pdf+ocr`, `docx-vision`,
   etc. images (png/jpg incl. cmyk/webp/gif/tiff incl.
   multi-page/bmp — pillow normalizes all of them to what tesseract expects) — and scanned pdfs via their embedded page
   images — walk a configurable reader roster (the `extract_readers` admin
   setting, default `ocr,vision`). readers are a registry (`extract.register_reader`), so
   another ocr engine, a dedicated ocr model, or a model on new hardware
   (e.g. a loom-hosted reader once its pjrt plugin lands — see
   docs/jax_backend.md) is a registration, not a rewrite. built-ins: `ocr` =
   tesseract (auto-detected, `liminallm[ocr]` extra; technically optional,
   practically required — deterministic, free per call, quotes rather than
   paraphrases) and `vision` = the configured model (one bounded call, image
   framed as DATA to read; capability probed per backend, never assumed —
   api backends use openai-compatible content parts, a local multimodal
   model implements `transcribe_image`). "ocr"-kind readers yield to the
   next reader when they find less than a document's worth of text; "vision"
   readers are deliberate readings, accepted as-is. files nothing can read
   are refused with the reason and the remedy, never stored as garbage.
   from then on it is ordinary vault material: searchable mid-chat, swept by
   the witness.

   **extraction is sandboxed, parsers assumed compromisable.** uploads are
   attacker-controlled bytes and every parser in the ladder — pillow's c
   decoders, pypdf, expat, tesseract+leptonica, poppler — has a cve history.
   all parsing runs in a disposable rlimited child (service/sandbox.py:
   memory/cpu/file-size caps inherited by tesseract/pdftoppm grandchildren,
   wall-clock kill, hard pixel ceiling so decompression bombs raise instead
   of allocating). the model's vision pass never runs in that child — it
   needs the network, but it never parses: the child hands extracted image
   bytes back over the pipe as pending slots (private-use-area markers,
   stripped from all extracted content so a file can't forge a slot) and the
   parent fills them. honest limit: the child shares the server's uid — this
   converts api-process compromise into compromise of a short-lived capped
   process, not into nothing; the container/vm recommendation from the
   interpreter section is the outer wall.

the rule: **per-chat grounding is automatic; permanent cross-chat memory is a
decision.** silently promoting every upload into a global corpus would make
old files bleed into unrelated conversations and turn a one-off "summarize
this" into standing memory the user never asked for. the vault IS the central
cross-conversation repo — there is deliberately no second one.

### 19.6 future: sweep report archive (not yet built)

sweep and witness reports are currently ephemeral — returned to the caller,
rendered in the ui, gone on reload; the red/amber markings live for the
session only, and re-running a sweep re-spends its model calls. we may want to
archive sweep reports: a small table (`sweep_report(id, user_id, created_at,
report jsonb)`) would give a "what moved this year" ledger, let the ui replay
the last sweep for free, and let a future scheduled sweep (leader-locked like
other periodic work) diff against the previous run instead of re-judging
unchanged pairs. nothing in the current shape blocks this — reports are
already self-contained json.

### 19.7 activation

`notes_enabled` — code default on, overridable from the admin console
(database-managed feature flag). when off: all `/v1/notes/*`
routes return 403 `notes_disabled`, the `note_search` tool is never offered,
and the front-end hides the notes tab on first contact. precedence follows the
platform rule: admin override > env var > code default.

---

## 20. context window, budget, and compaction

### 20.1 the window is discovered, not assumed

the prompt budget must come from the model actually serving requests. a
constant (the old `MAX_GENERATION_TOKENS = 4096` used as a whole-prompt cap)
is wrong in both directions: it wastes 99% of a million-token gemini window
and overruns a small local checkpoint. resolution order, most authoritative
first:

1. **admin override**: the `model_context_window` setting. set this when
   discovery guesses wrong.
2. **provider probe** (5s, best-effort, never raises): gemini's native
   `models/{id}` states `inputTokenLimit`; self-hosted openai-compatible
   servers (vllm, lorax, lm studio) expose `max_model_len` /
   `context_length` in `/models`. a probe result outranks the table because
   a local server may serve a small window under a big-model name.
3. **known-family table** (`KNOWN_CONTEXT_WINDOWS`, longest prefix wins).
4. **`DEFAULT_CONTEXT_WINDOW = 8192`** — conservative, so an unknown model
   degrades to "less context", never to overrun.

local jax takes `min(config.json max_position_embeddings, max_seq_len)`: the
checkpoint's trained positions and the serving cap, whichever binds.

### 20.2 budget

`prompt_budget = window − MAX_GENERATION_TOKENS`, floored at 2048 so the
reply always has room. resolved per turn, cached 60s so admin changes apply
without a restart. every prompt-assembling path enforces it — including the
attachment agent, whose 32kb inlined preamble previously bypassed budgeting
entirely.

pruning order when over budget: retrieved context from the least-relevant
end, then oldest history. the digest snippet is inserted **first** so it
survives pruning longest — losing the summary of everything older is worse
than losing one retrieved chunk.

### 20.3 compaction (rolling digest)

recent turns are sent verbatim (`RECENT_MESSAGES = 20`); older turns are
folded into a digest stored on `conversation.meta.digest` and prepended as a
labeled record. this is what makes long conversations degrade to "remembers
less precisely" instead of "forgets entirely".

- the digest is built off the hot path (same discipline as turn labels),
  merges the previous digest with only messages newer than its
  `through_seq`, and never re-summarizes covered turns.
- digest input is prior conversation text — including anything a user
  pasted — so it is framed as DATA to summarize and the injected block is
  labeled a record, not instructions.
- failure leaves the previous digest in place; a missing digest costs
  precision, never correctness, because the recent window is always sent.
- the history window is identical warm or cold. loading the *entire*
  conversation on a cache miss (the old behavior) made the model's memory
  depend on redis being up, which made "why did it forget that"
  unreproducible.

### 20.4 compaction is lossy — so it is not the only mechanism

a rolling digest re-summarizes its own previous output, which decays: each
fold paraphrases the paraphrase, and specifics (chosen values, hard
constraints, identifiers) go first. three mitigations, in order of
importance:

1. **nothing is ever actually lost.** every message stays in postgres
   verbatim. the digest is a *view*, not a replacement — "losing detail"
   only ever means losing it from the model's current view.
2. **verbatim anchors.** the digest call returns two sections: a NARRATIVE
   (re-summarized each fold) and ANCHORS — one specific per line, quoted
   exactly. anchors are carried forward **byte-identical** on every fold,
   never re-summarized, so they cannot drift through generations of
   paraphrase. bounded at 40 (oldest dropped, and logged, never silently).
3. **retrieval beats summary.** `history_search` returns earlier turns of
   this conversation verbatim (bm25 over the conversation's own messages,
   so no embeddings required). it is offered exactly when turns have fallen
   outside the verbatim window, and the digest block itself tells the model
   the summary is lossy and to call the tool for exact wording.

the resulting division of labour: **narrative for continuity, anchors for
what must not drift, retrieval for everything else.**

**the window is assembled, not a recency prefix.** chatgpt/claude-style
compaction (summarize what fell off the recency window) is an efficiency
mechanism and the fallback shape here — not the model. each turn's context
is assembled from three sources, all budget-derived from the discovered
model window:

1. **verbatim tail** — the longest suffix of recent turns that fits
   `history_budget` (= `history_budget_fraction`, default 0.5, of the
   prompt budget; floor of 4 turns). on a large-window model turns stay
   verbatim until the window actually pressures; on a small one digestion
   starts early. the boundary is tokens, never a message count.
2. **recall** — older turns chosen by relevance to the message being
   answered, restored verbatim from the permanent transcript, in
   chronological order, within `history_recall_fraction` (default 0.25) of
   the history budget. ranking is **the same rank fusion rag uses**
   (§2.5) when a real embedding encoder is configured, bm25 alone otherwise.
   semantic wins the cases keywords miss: "which database did we pick" finds
   "let's go with postgres" though they share no words; bm25 keeps exact
   terms (ids, numbers) in play. a turn the embedding budget never reached
   is absent from the semantic channel rather than scored zero by it. cost is bounded — cheap bm25 ranks everything, and
   only the top ~20 candidates get the embedding rerank; per-turn
   embeddings are persisted by a background backfill so the hot path reads
   vectors rather than computing them. recency is one relevance signal, not
   the whole policy: a decision from turn 3 competes for window space on
   merit when the current question touches it. 0 disables.

   the encoder is real when the model backend has an openai-compatible
   `/embeddings` client (openai, gemini-compat, self-hosted); otherwise the
   kernel's deterministic hash embedding is used and `is_semantic` is false.
   that flag is load-bearing: hash-embedding cosine is noise, so every
   consumer that blends cosine into a ranking checks it and falls back to
   bm25 rather than letting noise pollute a real score.
3. **digest + anchors** — connective tissue for everything neither tail
   nor recall carries.

pruning order under pressure: recall drops before the digest, the digest
before the verbatim tail — optional context yields to essential context.

### 20.5 token counting

budget math is only as good as the count. resolution per backend:

- **exact where we own the tokenizer.** `local_gpu_lora` already loads the
  checkpoint's own HF tokenizer for generation; the counter uses that same
  object, so counting is exact, offline, and free. it is forced eagerly —
  the tokenizer loads lazily, and reading it before first generate would
  cache a "heuristic" decision forever.
- **calibrated from ground truth otherwise.** every provider returns
  `usage.prompt_tokens` for the prompt just sent. feeding that back
  (`TokenCounter.observe`) maintains a per-model correction factor (ema,
  outliers and sub-200-token prompts ignored) that converges on the real
  tokenizer for the traffic this deployment sends. this works for gemini,
  claude, glm — none of which a vendor bpe library can count.
- **tiktoken is an optional extra, never a dependency.** it downloads bpe
  files on first use, which locked-down deployments block; it is used only
  when already installed with data cached locally, and only for openai-family
  ids where it is actually correct.
- the uncalibrated heuristic splits by script (cjk bills ~1 token/char, the
  old estimator undercounted it ~4x) and over-counts on purpose:
  over-counting prunes a turn early, under-counting overruns the model.
- **calibration is shared across replicas.** learned factors persist in
  `instance_config` under `token_calibration` (durable across restarts,
  works with redis absent) and are broadcast on the cluster bus so peers
  adopt them immediately instead of each re-learning the same number.
  publishing is debounced (every 10 observations), adopted factors are
  clamped like any observation, a peer's observation count is merged with
  `max()` so a fresh replica cannot publish over a well-calibrated one, and
  exact counters ignore shared factors entirely. entirely best-effort:
  without the bus the store write still lands, and without either,
  calibration is per-process — correct, just slower to converge.

### 20.6 other model-specific hazards

- **temperature**: reasoning models (o-series, gpt-5, gemini 3) reject a
  caller-supplied temperature with a 400 that fails the whole request, and
  others prescribe one fixed value. `temperature_policy` classifies each
  family as tunable, tunable-only-with-reasoning-off, or omit; nothing is
  sent unless an operator sets `model_temperature`, because a default of
  ours would override whatever the provider tuned its model around.
- **single-message validation** is a dos ceiling
  (`MAX_SINGLE_MESSAGE_TOKENS`), not a model budget. it previously reused
  the 4096 generation constant, which rejected a pasted document that a
  large-window model handles trivially. validation can only reject; the
  model budget is enforced in the workflow, which can prune.
- **embedding spaces**: rag chunks were already keyed by
  `embedding_model_id`; message recall was not, so a model switch would have
  ranked on vectors from a dead space. every consumer now records the encoder
  id with the vector and treats a mismatch as "not embedded" (see §3).

---

## 21. tools the model can call for itself

beyond `llm.generic` and `rag.answer_with_context_v1`, the agent loop offers
tools conditionally — a schema is only spent when the capability can actually
be used, so an empty vault or a disabled feature costs zero prompt tokens.

| tool | offered when | returns |
|---|---|---|
| `file_search` | conversation has searchable attachments | excerpts + file names |
| `run_python` | conversation has analyzable attachments | stdout of a sandboxed run |
| `web_search` | web tools on **and** a provider+key configured | titles/urls/snippets |
| `web_fetch` | web tools on | one page's visible text |
| `note_search` | notes enabled **and** the user's vault is non-empty | vault excerpts |
| `history_search` | turns have fallen outside the verbatim window (§20.3) | earlier turns, verbatim |

### 21.1 untrusted content and the injection rule

web pages, search results, attachments, notes, and recalled turns are all
**data**, never instructions. containment is layered because this app targets
weak local models, which drop a rule stated once:

- **sanitize at source**: the html extractor drops what a human cannot see
  (script/style/comments, `hidden`, `aria-hidden`, `display:none`,
  `visibility:hidden`); zero-width and format characters are stripped; page
  `<title>` is sanitized like body text — it escaped the envelope before it
  was.
- **structural containment**: fetched text is wrapped in
  `<<<UNTRUSTED_WEB_CONTENT>>>` markers with marker-lookalikes neutralized
  inside, so content cannot forge an envelope boundary. the `source` label is
  defanged for the same reason.
- **heuristic detection**: 14 injection patterns are scanned; matches are
  redacted, counted, and reported both in the trace and as a warning banner
  next to the payload the model reads.
- **capability limits**: fetches refuse private, loopback, link-local, and
  cloud-metadata addresses, re-checked on **every** redirect hop, with a byte
  cap enforced by streaming rather than trusting `content-length`.
- **the rule is repeated deliberately** in the system prompt, the tool
  descriptions, and the payload envelope. tighten the phrasing, never the
  repetition (see CLAUDE.md's prompt budget rule).

**known gap:** findings inform but do not yet restrict. after a poisoned
fetch the model may still call `run_python` in the same turn; only
instructions stand between fetched text and the interpreter. taint that
degrades capability (an `injection_tainted` session refusing further
code execution) is the next step, and enforcement beating instruction is
this codebase's own doctrine.

### 21.2 sandboxing untrusted work

two kinds of untrusted work run outside the api process:

- **code interpreter** (`run_python`): spawned child with rlimits
  (memory/cpu/file-size/no core dumps), wall-clock kill, network policy with
  an empty allowlist, and import-level blocking of networking/process
  modules. artifacts it publishes go through the same upload extension
  allowlist as user uploads.
- **file extraction** (§19.5): pillow, pypdf, expat, tesseract, and poppler
  all parse attacker-controlled bytes, so all of it runs in a disposable
  rlimited child with a hard pixel ceiling against decompression bombs.

both share the honest limit: the child runs as the same uid as the server, so
this converts api-process compromise into compromise of a short-lived capped
process — not into nothing. a container or vm is the outer wall.

### 21.3 archives

un-archiving is streamed and budgeted, never trusting headers: entry count,
per-member size, total size, and compression-ratio caps are enforced as bytes
are read (zip bombs), and every member path is sanitized component-wise and
re-joined through `safe_join` (zip slip). member type is checked with
`stat.S_IFMT` because many writers store permissions with no type bits.

---

## 22. running more than one replica

postgres and `shared_fs_root` are the shared state; nothing in the app
assumes it is the only process.

- **probes**: `/readyz` is the load-balancer probe and returns 503 when
  postgres or the filesystem is unusable. `/healthz` always returns 200 with
  a per-dependency breakdown, so it can never drain a replica. **redis is
  deliberately excluded from readiness** — every redis-backed feature has a
  fallback, so a redis outage must degrade the fleet, not drain it.
- **cluster bus** (`cluster_bus_backend`, default `auto`): redis pub/sub when
  reachable, else postgres `LISTEN`/`NOTIFY` — which is why redis stays
  optional. a single-process deployment gets a no-op backend. carries
  cross-replica cancellation (`POST /chat/cancel` reaching the worker that
  holds the stream) and token-calibration sharing (§20.5). best-effort: if it
  is down, cancel degrades to local-only and nothing else changes.
- **leader-locked periodic work**: clustering and adapter-prune proposals
  take a postgres advisory lock so they run once per interval cluster-wide,
  not once per replica. training jobs need no lock — claiming one is an
  atomic conditional update. the lock **fails open** when postgres is
  unreachable: maintenance running twice beats never running.
- **shared vs node-local storage**: every replica mounts the same
  `shared_fs_root` (adapters, artifacts, uploads). `interpreter_scratch_dir`
  must **not** be on it — throwaway per-call copies belong on local disk.
- **sticky sessions are not required.** websockets are per-connection and
  cancellation crosses the bus.
