# liminallm

liminallm is an experiment in what a chatgpt-like system looks like if you **stop hard-coding product logic** and let the model help evolve itself.

it’s a small kernel wrapped around:

- a frozen base llm (jax)
- per-user / per-skill lora adapters
- emergent “skills” from clusters + preference events
- self-describing artifacts (workflows, routing policies, tools)
- notebooklm-style grounding over cephfs-backed files
- boring infra: postgres + redis + cephfs

the code is just the glue. everything interesting lives as data.

---

## what it does (conceptually)

- **chatgpt-like web ui**
  - multi-user, password + pluggable auth
  - conversations, history, summaries
  - text first; voice later

- **deep behavioral memory**
  - per-user persona adapters (lora)
  - skill adapters born from usage: “when problems like this show up, start with this debugging workflow”
  - continuous micro-training jobs in jax, only on adapters, never on the base model

- **natural factual memory**
  - user files in cephfs (`/users/{id}/files`)
  - ingestion → chunking → embeddings in postgres (pgvector)
  - notebooklm-style: bind “contexts” (collections of files/folders) to a chat and ask questions grounded in that corpus

- **small kernel, big data**
  - kernel only knows how to:
    - auth users
    - run workflows (graphs)
    - run routing policies
    - call the llm with optional lora adapters
    - talk to postgres / redis / cephfs
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
  - cephfs
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

---

## getting started (high level)

> note: this is intentionally vague; exact commands depend on how you wire the codebase.

1. **bring your infra**
   - postgres (with pgvector installed)
   - redis
   - cephfs mount accessible to the app
   - gpu / tpu for jax model if you expect to train adapters

2. **configure env**
   - `DATABASE_URL` – postgres dsn
   - `REDIS_URL` – redis dsn
   - `CEPHFS_ROOT` – cephfs mount path
   - `MODEL_PATH` – base model directory under `CEPHFS_ROOT/shared/models/...`

3. **migrate db**
   - run the alembic / migration tool to create tables described in the spec.

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

- [ ] minimal chat with postgres-backed conversations
- [ ] file upload + cephfs + rag over pgvector chunks
- [ ] artifacts for workflows + tools (no adapters yet)
- [ ] preference events + single persona adapter per user
- [ ] semantic clustering + skill adapters
- [ ] router policies as data + simple editor
- [ ] configops api + llm-generated patches
- [ ] mobile / voice clients (optional layer)

---

## license

MIT
