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

**Critical Issues Found:** 8
**High Priority Issues:** 7
**Medium Priority Issues:** 9

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

### 1.4 Minor: Pagination Default Inconsistency

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

### 6.1 MEDIUM: RagMode Enum Missing LOCAL_HYBRID

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

## 7. Previously Resolved Issues

### 7.1 Session Exception Parameter (FIXED)

**Commit:** 3beddff

The `except_session_id` parameter in `revoke_all_user_sessions` was being ignored. Now properly passed to store methods.

---

## 8. Security Audit: JWT tenant_id Handling

**Status:** SECURE

The codebase correctly implements tenant isolation:
- tenant_id derived from authenticated user database record
- JWT payload tenant_id validated against user record
- All authenticated endpoints use `principal.tenant_id` or `auth_ctx.tenant_id`
- Admin endpoints validate request tenant_id matches principal

See previous audit section for full details.

---

## Summary by Severity

### Critical (Fix Immediately)

| # | Issue | Location |
|---|-------|----------|
| 1 | Invalid error codes ("bad_request", "invalid_json") | routes.py:1393,1401,3114 |
| 2 | Non-spec WebSocket event "streaming_complete" | routes.py:3020-3033 |
| 3 | Idempotency not stored for create_conversation | routes.py:2521 |
| 4 | search_chunks_pgvector user_id optional in memory | memory.py:1437 |
| 5 | MFA lockout only works with cache | auth.py:748-773 |
| 6 | Password reset non-functional without cache | auth.py:775-810 |
| 7 | Frontend password reset wrong endpoints | chat.js:800,842,946 |
| 8 | Frontend expects non-spec streaming_complete | chat.js:1484 |

### High Priority

| # | Issue | Location |
|---|-------|----------|
| 1 | Missing JSON validation in memory store | memory.py:491 |
| 2 | Per-node timeout default 5s not 15s | workflow.py:1525 |
| 3 | Frontend missing idempotency keys | chat.js, admin.js (multiple) |
| 4 | Voice endpoints bypass error handling | chat.js:2156-2244 |

### Medium Priority

| # | Issue | Location |
|---|-------|----------|
| 1 | Type annotation inconsistencies | memory.py, postgres.py |
| 2 | Retry backoff not cancellable | workflow.py:1162 |
| 3 | Per-node timeout not enforced across retries | workflow.py |
| 4 | RagMode enum missing LOCAL_HYBRID | config.py:37-41 |
| 5 | Undocumented API endpoints | SPEC.md vs frontend |
| 6 | Pagination default inconsistency | routes.py:2539 |

---

## Recommendations

### Immediate Actions

1. Fix error codes in routes.py to use only SPEC-defined values
2. Rename `streaming_complete` to `message_done` or remove duplicate event
3. Fix `idem.result = response` to `await idem.store_result(response)`
4. Make `user_id` required in memory.py `search_chunks_pgvector`
5. Add in-memory fallbacks for MFA lockout and password reset
6. Fix frontend password reset endpoint paths

### Short-term Actions

1. Add idempotency keys to all POST endpoints in frontend
2. Fix per-node timeout to use DEFAULT_NODE_TIMEOUT_MS
3. Add LOCAL_HYBRID to RagMode enum
4. Standardize type annotations across storage layer

### Documentation Actions

1. Update SPEC.md to document all endpoints used by frontend
2. Clarify WebSocket streaming event expectations
3. Document cache requirements for auth features
