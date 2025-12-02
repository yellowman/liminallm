# Codebase Issues and Security Audit

**Last Updated:** 2025-12-02
**Scope:** Comprehensive review against SPEC.md requirements

---

## Executive Summary

This document consolidates findings from a deep analysis of the liminallm codebase covering:
- API routes and SPEC compliance
- Storage layer consistency
- Authentication and security
- Workflow engine implementation
- Frontend API usage
- RAG service implementation
- LLM and Router services
- Config operations
- Training pipeline
- Database schema

**Critical Issues Found:** 15
**High Priority Issues:** 10
**Medium Priority Issues:** 12

---

## 1. API Routes (SPEC Compliance)

### 1.1 CRITICAL: Invalid Error Codes

**Location:** `liminallm/api/routes.py`

Two non-spec error codes are used where spec-compliant codes should be:

| Line | Current Code | Should Be | Context |
|------|-------------|-----------|---------|
| 1393, 1401 | `"bad_request"` | `"validation_error"` | Audio transcription/payload errors |
| 3114 | `"invalid_json"` | `"validation_error"` | WebSocket JSON parse error |

**SPEC Reference:** §18 defines valid codes: `unauthorized`, `forbidden`, `not_found`, `rate_limited`, `validation_error`, `conflict`, `server_error`

### 1.2 CRITICAL: Non-Spec WebSocket Event

**Location:** `liminallm/api/routes.py:3020-3033`

After streaming completes, the code sends event `"streaming_complete"` which is NOT in SPEC §18.

**SPEC-defined events:** `token`, `message_done`, `error`, `cancel_ack`, `trace`

**Impact:** Clients expecting only spec-compliant events may malfunction.

### 1.3 BUG: Idempotency Not Stored for create_conversation

**Location:** `liminallm/api/routes.py:2521`

```python
idem.result = response  # BUG: IdempotencyGuard has no 'result' attribute
```

Should be: `await idem.store_result(response)`

**Impact:** Duplicate conversations created on retry instead of returning cached result.

### 1.4 CRITICAL: OAuth tenant_id From User Input

**Location:** `liminallm/api/routes.py:640, 674`

The OAuth `oauth_start` and `oauth_callback` endpoints accept `tenant_id` from query parameters before authentication is complete:

```python
# oauth_start (line 640)
tenant_id = request.query_params.get("tenant_id")

# oauth_callback (line 674)
tenant_id = state_data.get("tenant_id")  # From unvalidated state
```

**SPEC Reference:** CLAUDE.md mandates tenant_id MUST come from authenticated JWT, never from user input.

**Impact:** Potential tenant spoofing during OAuth flows.

**Fix:** Derive tenant_id from the OAuth provider's identity claims or require pre-authentication.

### 1.5 CRITICAL: Visibility Filter Broken for Global/Shared Artifacts

**Location:** `liminallm/api/routes.py:1684-1691`

The visibility filter logic incorrectly restricts access:

```python
if visibility == "private":
    query = query.filter(owner_id=user_id)
elif visibility == "tenant":
    query = query.filter(tenant_id=tenant_id)
# Missing: "global" visibility should allow access regardless of owner
```

**SPEC Reference:** §10 defines visibility levels where "global" artifacts should be accessible to all users.

**Impact:** Users cannot access global or shared artifacts they should have access to.

### 1.6 CRITICAL: PATCH /artifacts Not RFC 6902 Compliant

**Location:** `liminallm/api/routes.py:1720-1745`

The PATCH endpoint accepts a flat object instead of RFC 6902 JSON Patch operations:

```python
# Current implementation
data = await request.json()  # Expects {field: value}
artifact.update(data)

# Should be RFC 6902
# [{"op": "replace", "path": "/field", "value": "new"}]
```

**SPEC Reference:** §10.4 specifies RFC 6902 JSON Patch format for PATCH operations.

**Impact:** API deviates from documented contract; clients using correct format will fail.

### 1.7 Minor: Pagination Default Inconsistency

**Location:** `liminallm/api/routes.py:2539`

`/conversations` uses `default_conversations_limit=50` while other endpoints use `default_page_size=100`.

---

## 2. Storage Layer Consistency

### 2.1 CRITICAL: search_chunks_pgvector User Isolation Mismatch

**Location:** `liminallm/storage/memory.py:1437` vs `liminallm/storage/postgres.py:2444`

| Implementation | user_id Parameter |
|----------------|-------------------|
| Memory.py | `Optional[str]` (can be None) |
| Postgres.py | `str` (required, enforced) |

**SPEC Reference:** §12.2 requires mandatory user isolation.

**Impact:** Memory store could bypass user isolation requirements.

### 2.2 HIGH: Missing Validation in Memory Store

**Location:** `liminallm/storage/memory.py:491` vs `liminallm/storage/postgres.py:1400`

`set_session_meta` in Postgres validates JSON serializability; Memory.py does not.

**Impact:** Memory store could accept objects that break on persistence/reload.

### 2.3 MEDIUM: Type Annotation Inconsistencies

Throughout both files, inconsistent use of:
- `Dict` vs `dict`, `List` vs `list`
- `Optional[X]` vs `X | None`

**Recommendation:** Standardize on Python 3.10+ style (`dict`, `list`, `| None`).

### 2.4 MEDIUM: SQL Schema Missing NOT NULL Constraints

**Location:** `liminallm/storage/schema.sql`

Several columns that should never be NULL lack the constraint:

| Table | Column | Issue |
|-------|--------|-------|
| `users` | `created_at` | Should be NOT NULL DEFAULT NOW() |
| `sessions` | `tenant_id` | Should be NOT NULL for multi-tenant isolation |
| `artifacts` | `visibility` | Should be NOT NULL DEFAULT 'private' |

### 2.5 MEDIUM: SQL Schema Missing Performance Indexes

**Location:** `liminallm/storage/schema.sql`

Missing indexes that would improve query performance:

| Table | Suggested Index | Reason |
|-------|-----------------|--------|
| `sessions` | `(user_id, created_at)` | Session listing queries |
| `artifacts` | `(tenant_id, visibility)` | Visibility filter queries |
| `chunks` | `(context_id, created_at)` | Context retrieval ordering |

---

## 3. Authentication Service Security

### 3.1 CRITICAL: MFA Lockout Only Works With Cache

**Location:** `liminallm/service/auth.py:748-773`

MFA lockout (5 failed attempts → 5 minute lockout) is entirely gated on cache availability:

```python
if self.cache:
    # ALL lockout logic is inside this block
```

**SPEC Reference:** §18 requires MFA lockout after 5 failed attempts.

**Impact:** Without Redis, unlimited TOTP brute-force attempts possible (1M combinations).

**Recommendation:** Use `self._mfa_challenges` dict (line 128, currently unused) as in-memory fallback.

### 3.2 HIGH: Password Reset Non-Functional Without Cache

**Location:** `liminallm/service/auth.py:775-810`

- `initiate_password_reset` returns token but only persists if cache exists (line 779-785)
- `complete_password_reset` only checks cache (line 790-793)

**Impact:** Password reset completely broken in in-memory mode.

**Recommendation:** Add in-memory fallback like email verification (lines 824-826).

### 3.3 MEDIUM: Unused _mfa_challenges Dictionary

**Location:** `liminallm/service/auth.py:128`

```python
self._mfa_challenges: dict[str, tuple[str, datetime]] = {}  # UNUSED
```

Should be used as fallback for MFA lockout tracking.

---

## 4. Workflow Engine

### 4.1 HIGH: Per-Node Timeout Default Incorrect

**Location:** `liminallm/service/workflow.py`

| Line | Issue |
|------|-------|
| 39 | `DEFAULT_NODE_TIMEOUT_MS = 15000` defined but never used |
| 1525 | Hardcoded `timeout = ... 5` (5 seconds, not 15) |

**SPEC Reference:** §9 specifies default per-node timeout of 15 seconds.

### 4.2 MEDIUM: Retry Backoff Not Cancellable

**Location:** `liminallm/service/workflow.py:1063-1179`

`_execute_node_with_retry()` does not accept `cancel_event` parameter. Retry sleep at line 1162 cannot be interrupted.

### 4.3 MEDIUM: Per-Node Timeout Not Enforced Across Retries

**Location:** `liminallm/service/workflow.py:1047-1149`

Only workflow-level timeout checked during retries, not per-node. Total retry time could exceed per-node timeout.

### 4.4 LOW: On-Error Handlers Bypass Retries

**Location:** `liminallm/service/workflow.py:1123`

If a node has `on_error` and fails on first attempt, it immediately goes to error handler without retrying.

---

## 5. Frontend API Usage

### 5.1 CRITICAL: Password Reset Endpoints Use Wrong Paths

**Location:** `frontend/chat.js`

| Line | Current Path | SPEC Path |
|------|-------------|-----------|
| 800 | `/v1/auth/reset/request` | `/v1/auth/request_reset` |
| 842, 946 | `/v1/auth/reset/confirm` | `/v1/auth/complete_reset` |

**Impact:** All password reset flows fail with 404.

### 5.2 HIGH: WebSocket Event Name Mismatch

**Location:** `frontend/chat.js:1484`

Frontend expects `streaming_complete` event, backend sends it (see issue 1.2), but SPEC doesn't define it.

**Note:** Backend and frontend are consistent with each other but both deviate from SPEC.

### 5.3 HIGH: Missing Idempotency-Key Headers

**Location:** Multiple endpoints in both `chat.js` and `admin.js`

| File | Affected Endpoints |
|------|-------------------|
| chat.js | `/auth/login`, `/contexts/{id}/sources`, all MFA endpoints, `/auth/password/change` |
| admin.js | `/auth/login`, `/config/propose_patch`, `/config/patches/{id}/decide`, `/admin/users` |

**SPEC Reference:** §18 requires Idempotency-Key on POST endpoints with side effects.

### 5.4 MEDIUM: Voice Endpoints Bypass Error Handling

**Location:** `frontend/chat.js:2156-2244`

Voice endpoints use raw `fetch()` instead of `requestEnvelope()`:
- No automatic retry on 5xx
- No token refresh on 401
- Errors silently logged to console

### 5.5 MEDIUM: Undocumented API Endpoints

Many endpoints used in frontend are not documented in SPEC §13:

| Category | Endpoints |
|----------|-----------|
| Voice | `/voice/transcribe`, `/voice/synthesize` |
| User | `/me`, `/settings`, `/files/limits` |
| Admin | `/admin/settings`, `/admin/users`, `/admin/adapters`, `/admin/objects` |
| Context | `/contexts/{id}/sources` |
| Tools | `/tools/specs`, `/tools/{id}/invoke`, `/workflows`, `/preferences/insights` |

---

## 6. RAG Service

### 6.1 CRITICAL: Path Traversal Vulnerability in ingest_path

**Location:** `liminallm/service/rag.py:453`

The `ingest_path()` method does not validate that the provided path stays within allowed directories:

```python
async def ingest_path(self, path: str, context_id: str, ...):
    # No validation that path is within allowed directories
    with open(path, 'rb') as f:
        content = f.read()
```

**Impact:** Attacker could read arbitrary files: `/etc/passwd`, `/proc/self/environ`, application secrets.

**Fix:** Use `safe_join()` pattern like in file service, or validate against allowed base directories.

### 6.2 MEDIUM: RagMode Enum Missing LOCAL_HYBRID

**Location:** `liminallm/config.py:37-41`

```python
class RagMode(str, Enum):
    PGVECTOR = "pgvector"
    MEMORY = "memory"
    # LOCAL_HYBRID missing!
```

But `rag.py:67-72` supports `local_hybrid` mode.

**Impact:** Setting `RAG_MODE=local_hybrid` via config validation will fail, though direct RAGService instantiation works.

**Fix:** Add `LOCAL_HYBRID = "local_hybrid"` to RagMode enum.

---

## 7. LLM Service

### 7.1 HIGH: Missing max_tokens Enforcement

**Location:** `liminallm/service/llm.py`

The `generate()` method does not enforce `max_tokens` limits:

```python
async def generate(self, messages, max_tokens=None, ...):
    # max_tokens passed to provider but not validated locally
    # No check against SPEC maximum of 4096 tokens
```

**SPEC Reference:** §5.2 specifies maximum output token limit of 4096.

**Impact:** Requests could exceed token limits, causing provider errors or excessive costs.

### 7.2 MEDIUM: Context Window Overflow Not Handled

**Location:** `liminallm/service/llm.py`

When context + prompt exceeds model context window, no truncation or error:

```python
# No token counting before submission
# No automatic context pruning for long conversations
```

**Impact:** Silent failures or truncated responses on long conversations.

---

## 8. Router Service

### 8.1 HIGH: Undocumented "closest" Selection Behavior

**Location:** `liminallm/service/router.py`

The router uses cosine similarity for adapter selection with "closest" match behavior:

```python
# Selects adapter with highest cosine similarity score
# Falls back to default if no match above threshold
```

**SPEC Reference:** §8 does not document the "closest" selection algorithm or threshold.

**Impact:** Users cannot predict or control which adapter will be selected for borderline cases.

**Recommendation:** Document the algorithm and expose threshold as configuration.

### 8.2 MEDIUM: No Adapter Validation on Assignment

**Location:** `liminallm/service/router.py`

When assigning an adapter to a conversation, no validation that:
- Adapter exists in the tenant
- Adapter is compatible with the base model
- User has permission to use the adapter

---

## 9. Config Operations

### 9.1 CRITICAL: Missing write_rate_limit_per_minute Config

**Location:** `liminallm/config.py`

The `write_rate_limit_per_minute` configuration parameter referenced in routes.py does not exist in config:

```python
# routes.py references:
config.write_rate_limit_per_minute

# config.py only defines:
# - rate_limit_requests_per_minute
# - rate_limit_tokens_per_minute
```

**Impact:** Write operations may use incorrect or default rate limits.

**Fix:** Add `write_rate_limit_per_minute` to configuration with appropriate default.

---

## 10. Training Pipeline

### 10.1 CRITICAL: No Deduplication in Dataset Generation

**Location:** `liminallm/training/dataset.py`

When generating training datasets from conversation history, no deduplication:

```python
# Multiple identical prompt-response pairs can appear
# Same conversation sampled multiple times
# No hash-based or semantic deduplication
```

**SPEC Reference:** §17 requires dataset quality controls.

**Impact:** Model overfits to duplicated examples, reducing generalization.

**Fix:** Implement hash-based deduplication on prompt-response pairs before training.

### 10.2 CRITICAL: SFT Prompt Includes Target Message

**Location:** `liminallm/training/sft.py`

The supervised fine-tuning data preparation includes the target (assistant) message in the prompt:

```python
# Incorrect: includes assistant response in input
prompt = format_messages(messages)  # Includes all messages

# Should be: assistant response is target only
prompt = format_messages(messages[:-1])  # Exclude final assistant message
target = messages[-1].content
```

**Impact:** Model learns to copy rather than generate; catastrophic training failure.

**Fix:** Exclude target message from prompt, use only as training label.

---

## 11. Previously Resolved Issues

### 11.1 Session Exception Parameter (FIXED)

**Commit:** 3beddff

The `except_session_id` parameter in `revoke_all_user_sessions` was being ignored. Now properly passed to store methods.

---

## 12. Security Audit: JWT tenant_id Handling

**Status:** MOSTLY SECURE (OAuth exception)

The codebase correctly implements tenant isolation in most areas:
- tenant_id derived from authenticated user database record
- JWT payload tenant_id validated against user record
- All authenticated endpoints use `principal.tenant_id` or `auth_ctx.tenant_id`
- Admin endpoints validate request tenant_id matches principal

**Exception:** OAuth endpoints (see issue 1.4) accept tenant_id from user input.

---

## Summary by Severity

### Critical (Fix Immediately)

| # | Issue | Location |
|---|-------|----------|
| 1 | Invalid error codes ("bad_request", "invalid_json") | routes.py:1393,1401,3114 |
| 2 | Non-spec WebSocket event "streaming_complete" | routes.py:3020-3033 |
| 3 | Idempotency not stored for create_conversation | routes.py:2521 |
| 4 | OAuth tenant_id from user input | routes.py:640,674 |
| 5 | Visibility filter broken for global artifacts | routes.py:1684-1691 |
| 6 | PATCH /artifacts not RFC 6902 compliant | routes.py:1720-1745 |
| 7 | search_chunks_pgvector user_id optional in memory | memory.py:1437 |
| 8 | MFA lockout only works with cache | auth.py:748-773 |
| 9 | Password reset non-functional without cache | auth.py:775-810 |
| 10 | Frontend password reset wrong endpoints | chat.js:800,842,946 |
| 11 | Path traversal vulnerability in ingest_path | rag.py:453 |
| 12 | Missing write_rate_limit_per_minute config | config.py |
| 13 | No deduplication in training dataset | training/dataset.py |
| 14 | SFT prompt includes target message | training/sft.py |
| 15 | Frontend expects non-spec streaming_complete | chat.js:1484 |

### High Priority

| # | Issue | Location |
|---|-------|----------|
| 1 | Missing JSON validation in memory store | memory.py:491 |
| 2 | Per-node timeout default 5s not 15s | workflow.py:1525 |
| 3 | Frontend missing idempotency keys | chat.js, admin.js (multiple) |
| 4 | Voice endpoints bypass error handling | chat.js:2156-2244 |
| 5 | Missing max_tokens enforcement | llm.py |
| 6 | Undocumented "closest" adapter selection | router.py |
| 7 | No adapter validation on assignment | router.py |
| 8 | WebSocket event name mismatch | chat.js:1484 |
| 9 | Context window overflow not handled | llm.py |
| 10 | Password reset cache dependency | auth.py:775-810 |

### Medium Priority

| # | Issue | Location |
|---|-------|----------|
| 1 | Type annotation inconsistencies | memory.py, postgres.py |
| 2 | Retry backoff not cancellable | workflow.py:1162 |
| 3 | Per-node timeout not enforced across retries | workflow.py |
| 4 | RagMode enum missing LOCAL_HYBRID | config.py:37-41 |
| 5 | Undocumented API endpoints | SPEC.md vs frontend |
| 6 | Pagination default inconsistency | routes.py:2539 |
| 7 | SQL schema missing NOT NULL constraints | schema.sql |
| 8 | SQL schema missing performance indexes | schema.sql |
| 9 | Voice endpoints bypass error handling | chat.js:2156-2244 |
| 10 | Context window overflow not handled | llm.py |
| 11 | No adapter validation on assignment | router.py |
| 12 | Unused _mfa_challenges dictionary | auth.py:128 |

---

## Recommendations

### Immediate Actions (Security Critical)

1. **Path Traversal**: Add path validation in `ingest_path()` using safe_join pattern
2. **OAuth tenant_id**: Derive from OAuth provider claims, not user input
3. **Training Pipeline**: Fix SFT to exclude target from prompt; add deduplication
4. **Error Codes**: Fix to use only SPEC-defined values
5. **MFA Lockout**: Add in-memory fallback using _mfa_challenges dict
6. **Visibility Filter**: Fix logic to properly handle global/shared artifacts

### Short-term Actions

1. Add idempotency keys to all POST endpoints in frontend
2. Fix per-node timeout to use DEFAULT_NODE_TIMEOUT_MS
3. Add LOCAL_HYBRID to RagMode enum
4. Add write_rate_limit_per_minute configuration
5. Fix `idem.result = response` to `await idem.store_result(response)`
6. Add max_tokens enforcement in LLM service
7. Fix PATCH /artifacts to implement RFC 6902
8. Fix frontend password reset endpoint paths

### Schema Actions

1. Add NOT NULL constraints to critical columns
2. Add performance indexes for common query patterns
3. Add migration scripts for schema changes

### Documentation Actions

1. Update SPEC.md to document all endpoints used by frontend
2. Document router "closest" selection algorithm and threshold
3. Document cache requirements for auth features
4. Clarify WebSocket streaming event expectations
