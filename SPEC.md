# adaptive chat system spec  
## (“small kernel, big data” with lora adapters, postgres, redis, cephfs, python, jax)

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
  - user files on CephFS
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
  - CephFS (files, adapters, artifacts)

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
  - File Service (CephFS abstraction)
  - Preference & Training Service
  - Clusterer & Skill Discovery
  - ConfigOps Service
- **data stores**
  - PostgreSQL
  - Redis
  - CephFS

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
  cephfs_path     TEXT,                          -- optional link to files
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  visibility      TEXT NOT NULL DEFAULT 'private', -- 'private','shared','global'
  meta            JSONB
);
```

**artifact versions** for history & rollbacks:

```sql
CREATE TABLE artifact_version (
  id              BIGSERIAL PRIMARY KEY,
  artifact_id     UUID NOT NULL REFERENCES artifact(id) ON DELETE CASCADE,
  version         INT NOT NULL,
  schema          JSONB NOT NULL,
  cephfs_path     TEXT,
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
  cephfs_path     TEXT NOT NULL,  -- directory or file
  recursive       BOOLEAN NOT NULL DEFAULT TRUE,
  meta            JSONB
);

CREATE TABLE knowledge_chunk (
  id              BIGSERIAL PRIMARY KEY,
  context_id      UUID NOT NULL REFERENCES knowledge_context(id) ON DELETE CASCADE,
  cephfs_path     TEXT NOT NULL,
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
  id                 BIGSERIAL PRIMARY KEY,
  adapter_artifact_id UUID NOT NULL REFERENCES artifact(id) ON DELETE CASCADE,
  user_id            UUID NOT NULL REFERENCES app_user(id) ON DELETE CASCADE,
  created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
  status             TEXT NOT NULL DEFAULT 'queued', -- 'queued','running','succeeded','failed'
  num_events         INT,
  loss               DOUBLE PRECISION,
  meta               JSONB
);
```

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

## 3. cephfs layout

### 3.1 directory structure

logical layout:

```text
/                             # cephfs root
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
        {adapter_artifact_id}/
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
  - `current_version`, `cephfs_dir`.

on CephFS in `/users/{user}/adapters/{artifact_id}/vNNNN/`:

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
W_{	ext{eff}} = W + \sum_j g_j \cdot lpha_j B_j A_j
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
  2. load corresponding LoRA parameter PyTrees from CephFS (cache hot ones in RAM).
  3. compose an effective view of weights:
     - for small K (top 2–3 adapters) this is cheap.
  4. run generation with sampling parameters (top-p, temperature, max tokens).
  5. stream tokens back to orchestrator.

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

5. run optimizer (Optax) for a few steps:

   - small learning rate, few epochs.
   - early stopping based on batch loss.

6. write new LoRA params to CephFS in new version directory.

7. update:

   - `adapter_router_state.centroid_vec` via EMA of event embeddings.
   - `adapter_router_state.last_trained_at`, `success_score`.

8. mark training job `status='succeeded'` with `loss`.

---

## 6. generic primitives in practice

### 6.1 artifact.schemas (examples)

**adapter.lora**:

```json
{
  "kind": "adapter.lora",
  "scope": "per-user",
  "user_id": "…",
  "rank": 8,
  "layers": [0,1,2,3,4,5],
  "matrices": ["attn_q", "attn_v"],
  "current_version": 3,
  "cephfs_dir": "/users/.../adapters/{id}",
  "cluster_id": "…",  // semantic cluster this adapter is tied to
  "applicability": {
    "natural_language": "Helps this user debug kernel panics via reproduce→bisect→log-analysis.",
    "embedding_centroid": null  // also in adapter_router_state; optional redundancy.
  }
}
```

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
      "tool": "rag.answer_with_context_v1"
    },
    {
      "id": "node_code_agent",
      "type": "tool_call",
      "tool": "agent.code_v1"
    },
    {
      "id": "node_plain",
      "type": "tool_call",
      "tool": "llm.generic_chat_v1"
    }
  ]
}
```

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
   - files on CephFS under `/users/{user}/files`.
   - embedded chunks in `knowledge_chunk` tied to `knowledge_context`s.

3. **behavioral memory**
   - preference_events in DB.
   - LoRA adapters (weights on CephFS).
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
   - group new preference_events per `(user, cluster)` → adapter_artifact.
   - create `training_job`s.
5. **adapter training** (offline):
   - TrainingService updates LoRA weights; writes new version to CephFS.
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

### 12.2 isolation

- **postgres**:
  - all queries must be filtered by `user_id` where appropriate.
  - Optionally: PostgreSQL Row-Level Security (RLS) to enforce `user_id = current_user_id()`.

- **cephfs**:
  - every access goes through FileService:
    - resolves `user_id` → root path `/users/{user_id}`.
    - rejects any path escape attempts (`..`).
    - enforces visibility of shared/global artifacts separately.

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

## 13. public api (minimal)

### 13.1 chat

`POST /v1/chat`

```json
{
  "conversation_id": "optional",
  "message": {
    "content": "string",
    "mode": "text"   // or 'voice' in future
  },
  "context_id": "optional knowledge_context id",
  "options": {
    "stream": true,
    "max_tokens": 512,
    "workflow_id": "optional artifact id override"
  }
}
```

- response: streaming SSE or WebSocket of partial tokens + final message metadata.

### 13.2 files & contexts

- `POST /v1/files/upload` — upload to CephFS under `/users/{u}/files`.
- `POST /v1/contexts` — create `knowledge_context` and attach paths.
- `GET /v1/contexts` — list contexts.
- ingestion service runs separately to chunk + embed.

### 13.3 artifacts

- `GET /v1/artifacts?type=workflow` — list accessible artifacts.
- `GET /v1/artifacts/{id}` — fetch.
- `POST /v1/artifacts` — create (user or system).
- `PATCH /v1/artifacts/{id}` — update (validated).

### 13.4 config ops

- as in §10.

---

## 14. implementation phases (minimal-first)

### phase 0: vanilla chat + files

- implement:
  - users, auth, conversations, messages.
  - FileService + CephFS.
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

logs:

- structured logs with correlation IDs for each chat request.

traces:

- optional OpenTelemetry traces:
  - gateway → orchestrator → router → workflow → inference → training.

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
