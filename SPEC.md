# adaptive chat system spec  
## (“small kernel, big data” with lora adapters, postgres, redis, filesystem, python, jax)

---

## 0. goals & principles

### 0.1 goals

- **chatgpt-like web interface** with:
  - multi-user, multi-session chat
  - text + (optional) voice interface
- **deep, evolving user-specific behavior** via:
  - per-user / per-skill LoRA adapters
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
- `policy.routing` (routing policy)
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
```

#### ingestion pipeline (knowledge → chunks)

- **parsers**: text, markdown, PDF (pdftotext), HTML (readability). Additional parsers can be registered via `artifact` type `tool.spec`.
- **chunking**: sliding window token-based splitter (e.g., 300–500 tokens with 50 token overlap) tuned per file type; store `chunk_index` and offsets.
- **hygiene**: dedupe by file checksum + path; skip binary blobs unless parser registered; enforce max file size per plan tier; optional PII-scrub per context.
- **embedding model**: fixed small encoder (e.g., `all-MiniLM` equivalent) referenced in config; keep version in `knowledge_context.meta.embedding_model`.
- **embedding dimensionality**: embeddings are normalized/padded to a fixed 64-d vector (`EMBEDDING_DIM`) across routing, RAG, and clustering; external providers must truncate/pad to this size before persistence.
- **refresh cadence**:
  - watch filesystem path events; enqueue ingestion job on file change.
  - periodic sweep (daily) to re-embed if encoder version changes.
- **retrieval strategy**:
  - primary path: pgvector `ORDER BY embedding <-> $query LIMIT k` filtered by `context_id`.
  - optional re-ranking via lightweight cross-encoder tool if available.
  - return chunk text + `fs_path` for citation; orchestrator can ask LLM to cite paths.
  - optional dev fallback: in-process hybrid BM25 + cosine search (controlled by `RAG_MODE=local_hybrid`), intended for tests or tiny corpora when pgvector is absent.
  - baseline kernel ships with a deterministic hashing-based embedding fallback (no external model dependency) shared across RAG/routing/clustering so chunks always have non-empty vectors for cosine search in both Postgres and in-memory stores.

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

### 3.3 memory-store snapshots (dev / single-node mode)

when the kernel runs in the in-memory fallback (no Postgres), it must persist state onto the shared filesystem so restarts do not wipe user data:

- write a JSON snapshot under `{shared_fs_root}/state/memory_store.json` after each mutation covering users, auth sessions, credentials, conversations/messages, artifacts + versions, config patches, knowledge contexts, and chunks.
- on startup, reload this snapshot before seeding default artifacts; only seed when no persisted state is present.
- artifact payloads (e.g., workflow JSON) still live under `{shared_fs_root}/artifacts/{artifact_id}/vNNNN.json` so snapshot + files can fully reconstruct state.

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
| `azure` | model_id | ✗ | ✗ | 1 |
| `vertex` | model_id | ✗ | ✗ | 1 |
| `bedrock` | model_id | ✗ | ✗ | 1 |
| `together` | adapter_param | ✓ | ✓ | 3 |
| `lorax` | adapter_param | ✓ | ✓ | 5 |
| `adapter_server` | adapter_param | ✓ | ✓ | 3 |
| `sagemaker` | adapter_param | ✗ | ✗ | 1 |
| `local_lora` | none | ✓ | ✓ | 3 |

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

- JAX/Flax implementation of a decoder-only transformer:
  - config + params loaded from `/shared/models/base_lm_v1`.
- base model **frozen**:
  - no gradient / updates on base weights.

### 5.2 lora parameterization

for each hooked weight matrix `W ∈ ℝ^{d_out × d_in}`:

- LoRA params for adapter `j`:
  - `A_j ∈ ℝ^{r × d_in}`
  - `B_j ∈ ℝ^{d_out × r}`
  - scale `α_j` (scalar or per-matrix)
- effective weight for given adapter gate weight `g_j`:

\[
W_{\text{eff}} = W + \sum_j g_j \cdot \alpha_j B_j A_j
\]

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
     - cache policy: LRU by `(adapter_id, version)`; pin persona adapters for logged-in user; max resident bytes guarded by config with periodic eviction.
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

6. evaluation + rollout:

   - hold-out slice from recent preference events; metrics = loss + alignment rate to positive feedback.
   - auto-apply new adapter version only if eval improves or human review approves; otherwise keep previous version pinned.

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
- evaluation: optional held-out batch; record perplexity / accuracy proxies in `training_job.meta`.

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

Router policies remain agnostic: they pick adapters by id/metadata and hand them to the inference backend. An adapter with `backend="api"` implies switching the request model ID to `remote_model_id` (e.g., Zhipu BigModel or Alibaba DashScope); `backend="local"` means applying filesystem-backed LoRA weights on the base model. `backend="prompt"` distills adapter behavior into a prompt/system-message overlay for API-only providers, and `backend="hybrid"` indicates a two-step plan where a local adapter-enabled controller plans and an external API model executes.

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

### 7.3 skill adapter creation

when:

- `semantic_cluster.size >= min_skill_size`, AND
- many positive `preference_event`s in that cluster, AND
- no existing adapter bound to this cluster:

then:

1. ConfigOps proposes a new adapter artifact:

```json
{
  "kind": "adapter.lora",
  "scope": "per-user",
  "user_id": "...",
  "rank": 4,
  "layers": [0,1,2],
  "matrices": ["attn_q"],
  "cluster_id": "<cluster_id>",
  "applicability": {
    "natural_language": "Skill: " + semantic_cluster.label + " – " + semantic_cluster.description
  }
}
```

2. human/admin or automated rule approves creation.
3. artifact created with initial LoRA params = zeros.
4. training jobs for that cluster events are enqueued to train new adapter.

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
   - select chunks from `knowledge_chunk` via pgvector.
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
- responses use envelope `{ "status": "ok|error", "data": ..., "error": { "code", "message", "details" } }`.
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

### 13.2 auth/session api (minimal definitions)

- `POST /v1/auth/signup { email, password }` → create user.
- `POST /v1/auth/login { email, password }` → set session cookie / bearer token.
- `POST /v1/auth/oauth/{provider}/start` + `GET /v1/auth/oauth/{provider}/callback` (standard OAuth).
- `POST /v1/auth/logout` → revoke session.
- `POST /v1/auth/refresh` → rotate session/refresh token.
- responses include `session_expires_at`; headers `Set-Cookie: session_id=...; HttpOnly; Secure` when cookies are used.
- `POST /v1/auth/mfa/verify` when MFA enabled; returns new session + requires one-time recovery code flow if user is locked out.

### 13.3 files & contexts

- `POST /v1/files/upload` — multipart; stores under `/users/{u}/files`; returns `fs_path`; optional `context_id` form field triggers chunking + embedding ingestion into that knowledge context.
- `GET /v1/files` — list user files (paginated).
- `POST /v1/contexts` — create `knowledge_context`, attach file paths.
- `GET /v1/contexts` — list contexts + stats; supports `?owner=me|global`.

### 13.4 artifacts

- `GET /v1/artifacts?type=workflow|policy|adapter|tool` — list accessible artifacts.
- `GET /v1/artifacts/{id}` — fetch current version + metadata.
- `POST /v1/artifacts` — create; validates `schema.kind` using per-kind schema.
- `PATCH /v1/artifacts/{id}` — update via JSON Patch; writes new `artifact_version`.
- `GET /v1/artifacts/{id}/versions` — list versions.

### 13.5 config ops

- same endpoints as §10; PATCH application triggers validation + dry-run.

### 13.6 migrations (basic shell tool)

- repository includes `scripts/migrate.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
psql "$DATABASE_URL" -f sql/000_base.sql
psql "$DATABASE_URL" -f sql/001_artifacts.sql
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
- **tab navigation**: four primary tabs organize functionality:
  - **Chat**: conversation interface with message streaming
  - **Contexts**: knowledge context management
  - **Artifacts**: system artifact browser
  - **Settings**: user preferences and session info
- responsive breakpoints: sidebar hidden on mobile (<1080px), single-column tabs on small screens (<640px).

### 17.2 conversation sidebar

- **conversation list**: paginated list of user conversations sorted by `updated_at`.
- **search**: client-side filter by title or conversation ID.
- **active indicator**: highlight currently loaded conversation.
- **new conversation**: button to reset chat state and start fresh thread.
- API endpoints: `GET /v1/conversations`, `GET /v1/conversations/{id}`, `GET /v1/conversations/{id}/messages`.

### 17.3 chat view (Chat tab)

- **message stream**: scrollable container with message bubbles differentiated by role (user/assistant/system).
- **token streaming**: WebSocket primary with HTTP fallback; display streaming indicator during generation.
- **citation rendering**: inline clickable links for citations from `content_struct.citations`; each citation shows source filename/path as tooltip.
- **context binding**: dropdown to select active `knowledge_context` for RAG-grounded responses.
- **workflow override**: optional text input for `workflow_id` to steer execution.
- **optimistic UI**: user messages displayed immediately before server confirmation.
- **collapsible sections**:
  - **Upload knowledge**: file upload with context selection and chunk size configuration.
  - **Preferences**: thumbs up/down feedback with optional notes, displays routing metadata and trace.

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
- **login form**: email, password, optional MFA code, optional tenant ID.
- **MFA handling**: if `mfa_required` returned without token, prompt user to enter code.
- **token management**: access token, refresh token, session ID stored in sessionStorage.
- **auto-refresh**: on 401 response, attempt token refresh before failing.
- **logout**: calls `POST /v1/auth/logout`, clears storage, reloads page.

### 17.11 API integration patterns

- **request headers**: `Authorization: Bearer`, `X-Tenant-ID`, `session_id`, `Idempotency-Key` (auto-generated UUID).
- **envelope handling**: parse `{ status, data, error }` responses; extract error messages from `error.message` or `detail`.
- **retry logic**: exponential backoff (400ms base, 3 retries) for 5xx errors; no retry on 4xx.
- **WebSocket protocol**: connect to `/v1/chat/stream`; send auth + message in initial frame; handle `token`, `message_done`, `error` events.

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
  - pagination: either `{ data: [...], next_cursor: "opaque" }` or `{ page, page_size, total }`; choose per-endpoint but keep stable once published.
  - idempotency: POST endpoints that create side effects (`/v1/chat`, `/v1/tools/run`, `/v1/artifacts`) accept `Idempotency-Key`; server replays prior response within a 24h TTL and returns `409` if the prior attempt is still running.
  - auth header is `Authorization: Bearer <token>` in REST; WebSockets require an initial `{ "type": "auth", "session": "<session_id or bearer token>" }` frame before any `chat_start` frames; unauthenticated sockets close with code `4401`.
  - streaming events: `token`, `message_done`, `error`, `cancel_ack`, `trace` (router/workflow trace snapshot). SSE uses `event:` labels; WebSockets wrap as `{ "event": "token", "data": "..." }`.
  - minimal REST surface (kernel-stable):
    - `POST /v1/auth/login { email, password, mfa_code? } → { access_token, refresh_token, user }`.
    - `POST /v1/auth/refresh { refresh_token } → { access_token, refresh_token }`.
    - `POST /v1/chat { conversation_id?, message, context_ids?, artifact_ids?, stream: bool } → { conversation_id, message_id, stream_id? }`; stream events carry `{ event, data, request_id }` with `trace` payloads showing router/workflow steps.
    - `POST /v1/chat/cancel { request_id }`.
    - `GET /v1/conversations?cursor=...` and `GET /v1/conversations/{id}/messages?cursor=...` return paged lists.
    - `POST /v1/artifacts { type, name, schema, visibility?, fs_path? }` and `PATCH /v1/artifacts/{id}`; both emit a new `artifact_version` row and validate JSON Schema against `type` registry.
    - `POST /v1/config/patches { artifact_id, patch, justification }` queues a ConfigOps proposal; `POST /v1/config/apply { patch_id }` (admin-only) applies a validated patch.
    - `POST /v1/tools/run { tool_id, input }` executes a tool node outside a workflow (for testing) with the same retry/timeout caps.
  - errors MUST use stable `error.code` values: `unauthorized`, `forbidden`, `not_found`, `rate_limited`, `validation_error`, `conflict`, `server_error`; HTTP codes mirror the error (`401/403/404/429/400/409/500`).
  - constraint violations (FK/unique) return `conflict` with a short `details` map identifying the offending field/id; kernel surfaces storage errors instead of leaking database-specific messages.

- **auth/session flows (minimal, deterministic)**
  - password reset: `POST /v1/auth/request_reset { email }` stores a one-time token in Redis (15m TTL) and emails it; `POST /v1/auth/complete_reset { token, new_password }` rotates credentials and revokes sessions.
  - email verification: `POST /v1/auth/verify_email { token }` marks `user.meta.email_verified=true`; unverified accounts are limited to 24h and low rate limits.
  - MFA: `POST /v1/auth/mfa/enable` returns TOTP secret + QR; `POST /v1/auth/mfa/verify { code }` gates login/refresh when `user.meta.mfa_enabled=true`; 5 failed codes locks MFA for 5 minutes.
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
  - tool workers run under a fixed UID with cgroup limits (CPU shares, memory hard cap) and no filesystem access except a tmp scratch; `privileged:true` tools require admin-owned artifacts and are never called by default workflows.
  - JSON Schema validation enforced on tool inputs/outputs; outputs flagged `content_type: "html_untrusted"` must be sanitized by client before render.
  - retries: default 2 retries with exponential backoff (1s, 4s); per-node override allowed but capped at 3; node timeout default 15s, hard cap 60s.

- **inference/adapter cache discipline**
  - per-GPU adapter cache budget configured in bytes (e.g., 6GB); eviction LRU with pinning for persona adapter of active user; checksum of `params.json` verified against `schema.checksum` before activation.
  - per-request adapter cap = 3; if router selects more, lowest-weight adapters are dropped and the trace records the drop.
  - cancellation: orchestrator issues `{event:"cancel", request_id}`; worker aborts decode, frees KV cache and adapter refs, and emits `cancel_ack` with partial tokens if any.

- **adapter mode configuration**
  - `DEFAULT_ADAPTER_MODE` environment variable (default: `hybrid`): controls mode for newly created adapters.
  - valid values: `local`, `remote`, `prompt`, `hybrid` (see §5.0.1 for mode definitions).
  - `MODEL_BACKEND` determines which adapter modes are compatible:
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
