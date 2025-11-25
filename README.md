# liminallm

liminallm is an experiment in what a chatgpt-like system looks like if you **stop hard-coding product logic** and let the model help evolve itself.

it’s a small kernel wrapped around:

- a frozen base llm (jax)
- per-user / per-skill lora adapters
- emergent “skills” from clusters + preference events
- self-describing artifacts (workflows, routing policies, tools)
- notebooklm-style grounding over filesystem-backed files
- boring infra: postgres + redis + filesystem
- artifacts and adapter payloads live as JSON/weights on the shared filesystem

the code is just the glue. everything interesting lives as data.

---

## what it does (conceptually)

### feedback loop (at a glance)

```
User Feedback → Embeddings → Clustering → Skill Discovery
     ↑                                            ↓
Router Updates ← Adapter Training ← Promotion Decision
```

- **chatgpt-like web ui**
  - multi-user, password + pluggable auth
  - conversations, history, summaries
  - text first; voice later

- **deep behavioral memory**
  - per-user persona adapters (lora)
  - skill adapters born from usage: “when problems like this show up, start with this debugging workflow”
  - continuous micro-training jobs in jax, only on adapters, never on the base model

- **natural factual memory**
  - user files in the filesystem (`/users/{id}/files`)
  - ingestion → chunking → embeddings in postgres (pgvector)
  - notebooklm-style: bind “contexts” (collections of files/folders) to a chat and ask questions grounded in that corpus

- **small kernel, big data**
  - kernel only knows how to:
    - auth users
    - run workflows (graphs)
    - run routing policies
    - call the llm with optional lora adapters
    - talk to postgres / redis / filesystem
  - everything else (domains, skills, behaviors, tools, routing rules) is expressed as **artifacts**:
    - `adapter.lora`
    - `workflow.chat`
    - `policy.routing`
    - `tool.spec`
    - `context.knowledge`
    - etc.

- **emergent domains & skills**
  - no hard-coded `DEBUGGING`, `WRITING`, whatever
  - we cluster preference events in embedding space
  - llm labels clusters (“kernel panic debugging”, “multi-tenant billing schema design”, …)
  - when a cluster is big + consistently positive, we auto-propose a new skill adapter tied to that cluster

- **router as data, not code**
  - routing policies are artifacts (`policy.routing`) with a tiny expression language:
    - conditions over embeddings, clusters, safety flags
    - actions: activate/deactivate adapters, scale weights, etc.
  - the router engine is dumb and stable; policy is editable data

- **llm as architect (under guardrails)**
  - a config-ops api lets the llm propose patches to:
    - routing policies
    - workflows
    - adapter metadata
  - patches are stored, validated, can be auto- or human-approved, and are fully versioned

---

## architecture (stack)

- **language / runtime**
  - python (services, api, orchestration)
  - jax (base model + lora training)

- **storage**
  - postgres
    - users, auth, conversations, messages
    - artifacts & versions
    - semantic clusters
    - knowledge chunks (with pgvector)
    - preference events, training jobs, router state
  - redis
    - sessions
    - rate limiting
    - hot conversation summaries
    - router and workflow scratch state
  - filesystem
    - `/shared/models` – frozen base model weights
    - `/users/{id}/files` – user docs
    - `/users/{id}/adapters` – per-user lora weight files
    - `/users/{id}/artifacts` – generated notebooks, exports, etc.

- **services (logically)**
  - auth service
  - chat orchestrator
  - artifact service
  - workflow engine
  - router service
  - llm inference (jax + lora)
  - knowledge / rag service
  - preference + training service
  - clusterer + skill discovery
  - configops (patch proposals / approvals)

for v1 these can all live in one python app with clear module boundaries.

---

## current status

- **early design / prototyping**
  - do not treat as production-ready
  - interfaces & schemas are expected to change
- goal is to keep:
  - implementation minimal
  - all “product behavior” in data (artifacts / policies / workflows)
  - evolution driven by usage + llm suggestions, not constant code surgery

### implementation completeness (prototype)

- **implemented**
  - file upload endpoint writing to the shared filesystem and ingesting chunks into RAG contexts with configurable chunk sizes; default retrieval runs against pgvector with shared deterministic embeddings (optional in-process hybrid fallback for dev/test)
  - workflow execution with branching/parallel scheduling across `workflow.chat` graphs
  - router policies with a sandboxed evaluation engine (limited adapter gating usage)
  - pluggable model backend that can target external API fine-tune IDs or local JAX+LoRA adapter application
  - filesystem-backed LoRA adapter training that turns preference events into new adapter versions
  - preference capture with clustering + skill adapter promotion and routing integration
  - MFA with TOTP enrollment (otpauth URL), session gating, and login verification
  - HMAC-signed JWT access tokens with refresh rotation, tenant-aware sessions, and admin-only config endpoints
  - preference UI and rich routing feedback loop
  - LLM-as-architect auto-patch generation
  - voice interface
  - admin UI for patch approval

---

## getting started (high level)

> note: this is intentionally vague; exact commands depend on how you wire the codebase.

1. **bring your infra**
   - postgres (with pgvector installed)
   - redis
   - filesystem path accessible to the app
 - gpu / tpu for jax model if you expect to train adapters
 - backend selection is single-sourced from the SQL deployment config (editable from the web console when wired); env vars only override if you set them explicitly
  - set `MODEL_BACKEND=local_gpu_lora` to target the local JAX+LoRA path instead of external API fine-tune IDs; omit or leave as the default to use the OpenAI-style plug. The JAX backend (`LocalJaxLoRABackend` in `liminallm/service/model_backend.py`) loads adapters from the filesystem, tokenizes prompts, runs a JAX forward pass, and enforces conservative shapes; it requires a JAX runtime and optionally a Transformers tokenizer for decode parity. OpenAI plug secrets live under adapter-specific env vars (see below).

### frontend (chat + admin)

- A minimal, ChatGPT-style UI now lives in `/frontend` and is served by the FastAPI app at `/` with static assets mounted at `/static/*`.
- Authenticate with `/v1/auth/login`; the UI stores the issued bearer token/tenant ID locally and uses it for `/v1/chat`, `/v1/conversations`, and other API calls.
- The admin console is separate at `/admin` and is guarded by the `admin` role (FastAPI enforces the role before serving the HTML). It surfaces config patch proposal/approval flows backed by `/v1/config/*` endpoints, tenant-scoped user administration (list/add/delete, role changes), adapter visibility, and a read-only inspector for database objects.

### adapters: local LoRA vs remote fine-tune IDs vs prompt-distilled

- Router policies pick an adapter; the inference backend decides whether that means applying LoRA weights locally, swapping to a remote fine-tuned model ID, or injecting distilled prompt instructions on top of a black-box API.
- Each `adapter.lora` artifact carries a `backend` field describing where inference happens:

  ```json
  {
    "kind": "adapter.lora.remote",
    "provider": "zhipu",
    "backend": "api",
    "base_model": "glm-4-air",
    "remote_model_id": "glm-4-air-ft-2025-11-01-u123-debug",
    "region": "cn-beijing",
    "cluster_id": "…",
    "applicability": {
      "natural_language": "u123: kernel panic debugging skill on GLM-4-Air",
      "embedding_centroid": []
    }
  }
  ```

  ```json
  {
    "kind": "adapter.lora.local",
    "backend": "local",
    "provider": "aliyun",
    "base_model": "qwen2.5-32b-instruct",
    "cephfs_dir": "/users/u123/adapters/{id}",
    "rank": 8,
    "layers": [0, 1, 2, 3],
    "matrices": ["attn_q", "attn_v"],
    "cluster_id": "…"
  }
  ```

  ```json
  {
    "kind": "adapter.lora.prompt",
    "backend": "prompt",
    "provider": "api_only",
    "base_model": "glm-4-air",
    "prompt_instructions": "for kernel issues: reproduce → bisect → log inspection; keep replies terse",
    "cluster_id": "…",
    "applicability": {
      "natural_language": "prompt-distilled skill for kernel debugging",
      "embedding_centroid": []
    }
  }
  ```

- Remote adapters send requests to OpenAI-compatible fine-tuned model IDs (e.g., Zhipu BigModel or Alibaba DashScope). Local adapters resolve to filesystem-backed LoRA weights and are composable. Prompt-distilled adapters inject behavior as system messages without changing model IDs so you can still steer API-only providers.
- “Model-ID adapters” (fine-tuned endpoints) map 1:1 to model strings on providers like OpenAI/Azure (fine-tuned deployments), Vertex AI Gemini, or Bedrock custom models. Switching behavior = switching the `model` string; composition happens at routing time, not inside a single call.
- “Adapter-ID adapters” (multi-LoRA / adapter servers) surface `adapter_id` parameters on Together AI Serverless Multi-LoRA, LoRAX-style servers, or SageMaker adapter inference components. The backend keeps the base model string and passes `adapter_id` for one-or-more adapters per request when supported.
- Hybrid patterns (local adapter-enabled “controller” + external API “executor”) flow through the same artifacts: the controller uses a local LoRA backend to plan, then the API backend executes with prompt or remote-model adapters.

2. **configure env**
   - `DATABASE_URL` – postgres dsn
   - `REDIS_URL` – redis dsn
   - `SHARED_FS_ROOT` – filesystem root path
   - `MODEL_PATH` – model identifier for cloud mode (default `gpt-4o-mini`) or filesystem path when using an adapter server
   - `OPENAI_ADAPTER_API_KEY` – OpenAI plug API key (leave unset to use the echo fallback)
   - `OPENAI_ADAPTER_BASE_URL` – optional base URL override when pointing at an OpenAI-compatible endpoint
   - `ADAPTER_SERVER_MODEL` – model name when pointing at an OpenAI-compatible adapter server
   - `USE_MEMORY_STORE` – set to `true` to run without Postgres/Redis while testing the API and LLM calls
   - `RAG_CHUNK_SIZE` – default character window for knowledge ingestion; overrides can be provided per request
   - `RAG_MODE` – `pgvector` (default) uses the database index; `local_hybrid` forces the in-process BM25+cosine fallback for dev/test

3. **migrate db**
   - run the alembic / migration tool to create tables described in the spec.

4a. **preference_event → adapter dataset → tokenized batches**
   - `preference_event` rows (positive feedback) capture `context_embedding`, `score`, and optional `context_text`; they are clustered per-user to build adapter personas.
   - the training service reconstructs prompts from recent messages, appends any provided context snippet, and uses corrected text as targets while tracking cluster centroids.
   - dataset rows are written to `${SHARED_FS_ROOT}/users/{user_id}/adapters/{adapter_id}/jobs/{job_id}/dataset.jsonl`.
   - tokenized batches carry shapes for the downstream JAX/Optax loop (padding + masks, no base-model update), and training metadata records batch shapes + cluster summaries.
   - adapter metadata and params are stored under `${SHARED_FS_ROOT}/users/{user_id}/adapters/{adapter_id}/v####/`.

4. **start services**
   - run the api server (http + websocket for streaming)
   - run a background worker for:
     - ingestion / embeddings
     - clustering
     - adapter training
     - configops patch application

5. **open the web ui**
   - sign up / log in
   - create a conversation
   - upload a few files, create a knowledge context, and attach it to a chat
   - start talking to see basic chat + rag behavior
   - enable preference capture + adapters once that’s wired

---

## roadmap (rough)

- [x] minimal chat with postgres-backed conversations
- [x] file upload + filesystem + rag over pgvector chunks
- [x] artifacts for workflows + tools (no adapters yet)
- [x] preference events + single persona adapter per user
- [x] semantic clustering + skill adapters
- [x] router policies as data + simple editor
- [x] configops api + llm-generated patches
- [x] mobile / voice clients (optional layer)

---

## license

MIT

## dev quickstart (prototype)

- install dependencies: `pip install -e .`
- run migrations: `DATABASE_URL=postgres://... ./scripts/migrate.sh`
- start api: `uvicorn liminallm.app:app --reload`
- hit health check: `curl http://localhost:8000/healthz`
- call kernel endpoints (Bearer access token preferred; legacy `session_id` header supported for compatibility):
  - `POST /v1/auth/signup` → returns session + signed access/refresh tokens scoped to the tenant
  - `POST /v1/auth/login` → returns tokens, with MFA gating when enabled
  - `POST /v1/auth/refresh` → rotates refresh tokens and issues a new access token
  - `POST /v1/chat` → creates conversation + LLM reply (live if `OPENAI_ADAPTER_API_KEY` is set, echo otherwise)
  - `GET /v1/artifacts` → lists data-driven workflows/policies
  - admin config endpoints (`/v1/config/*`) require an admin-role token and are intended for the admin UI only
