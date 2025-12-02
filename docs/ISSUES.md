# Codebase Issues and Security Audit

**Last Updated:** 2025-12-02
**Scope:** Comprehensive review against SPEC.md requirements (5th pass)

---

## Executive Summary

This document consolidates findings from deep analysis of the liminallm codebase covering:
- API routes and SPEC compliance
- Storage layer consistency
- Authentication and security
- Session management
- Rate limiting and concurrency
- File upload/download security
- WebSocket implementation
- Workflow engine implementation
- Tool sandboxing and circuit breakers
- Frontend API usage
- RAG service implementation
- LLM and Router services
- Config operations
- Training pipeline
- Preference/feedback handling
- Clusterer and skill discovery
- Redis usage and memory persistence
- Database schema
- Race conditions and concurrency bugs (4th pass)
- Error handling and partial failures (4th pass)
- Transaction safety and atomicity (4th pass)
- Cache invalidation and consistency (4th pass)
- Resource cleanup and memory management (4th pass)
- Edge cases handling (4th pass)
- Pagination and large payload handling (4th pass)
- State machine consistency (4th pass)
- API contract validation (5th pass)
- Service initialization issues (5th pass)
- Configuration validation (5th pass)
- Logging and observability gaps (5th pass)
- Business logic constraints (5th pass)
- Async/await anti-patterns (5th pass)
- Frontend-backend contract mismatches (5th pass)

**Critical Issues Found:** 66
**High Priority Issues:** 52
**Medium Priority Issues:** 34

---

## 1. API Routes (SPEC Compliance)

### 1.1 CRITICAL: Invalid Error Codes

**Location:** `liminallm/api/routes.py`

| Line | Current Code | Should Be | Context |
|------|-------------|-----------|---------|
| 1393, 1401 | `"bad_request"` | `"validation_error"` | Audio transcription/payload errors |
| 3114 | `"invalid_json"` | `"validation_error"` | WebSocket JSON parse error |

**SPEC Reference:** §18 defines valid codes: `unauthorized`, `forbidden`, `not_found`, `rate_limited`, `validation_error`, `conflict`, `server_error`

### 1.2 CRITICAL: Non-Spec WebSocket Event

**Location:** `liminallm/api/routes.py:3020-3033`

After streaming completes, the code sends event `"streaming_complete"` which is NOT in SPEC §18.

**SPEC-defined events:** `token`, `message_done`, `error`, `cancel_ack`, `trace`

### 1.3 BUG: Idempotency Not Stored for create_conversation

**Location:** `liminallm/api/routes.py:2521`

```python
idem.result = response  # BUG: IdempotencyGuard has no 'result' attribute
```

Should be: `await idem.store_result(response)`

### 1.4 CRITICAL: OAuth tenant_id From User Input

**Location:** `liminallm/api/routes.py:640, 674`

OAuth endpoints accept `tenant_id` from query parameters before authentication is complete, violating CLAUDE.md security guideline.

### 1.5 CRITICAL: Visibility Filter Broken for Global/Shared Artifacts

**Location:** `liminallm/api/routes.py:1684-1691`

The visibility filter logic incorrectly restricts access to global artifacts.

### 1.6 CRITICAL: PATCH /artifacts Not RFC 6902 Compliant

**Location:** `liminallm/api/routes.py:1720-1745`

The PATCH endpoint accepts a flat object instead of RFC 6902 JSON Patch operations.

### 1.7 Minor: Pagination Default Inconsistency

**Location:** `liminallm/api/routes.py:2539`

`/conversations` uses `default_conversations_limit=50` while other endpoints use `default_page_size=100`.

---

## 2. Session Management

### 2.1 CRITICAL: Session Rotation (24h Activity) NOT IMPLEMENTED

**Location:** `liminallm/service/auth.py:533-563`

**SPEC §12.1 requires:** "refresh `id`/`expires_at` every 24h of activity; invalidate old session id after grace period"

**Current:** Sessions created with static 24h TTL, never rotated. No grace period logic exists.

**Impact:** Sessions tied to initial creation time, not activity. User could be active for days but logged out.

### 2.2 CRITICAL: Single-Session Mode NOT IMPLEMENTED

**Location:** `liminallm/service/auth.py:533-563`

**SPEC §18 requires:** "login from new device invalidates prior refresh tokens if `meta.single_session=true`"

**Current:** No check for `meta.single_session` flag, no device detection logic.

**Impact:** Users cannot enforce "only one device" security policy.

### 2.3 CRITICAL: X-Session Header for WebSockets NOT IMPLEMENTED

**Location:** `liminallm/api/routes.py:2853-2875`

**SPEC §12.1 requires:** "WebSockets require `X-Session: <session id>` header or `Authorization: Bearer`"

**Current:** WebSocket accepts auth from JSON message body, not HTTP headers.

### 2.4 HIGH: Access Tokens Not Denylisted on Logout

**Location:** `liminallm/service/auth.py:591-605`

Only refresh tokens are revoked on logout. Access tokens remain valid for full 30 minutes.

**Impact:** Compromised access tokens remain valid even after logout.

### 2.5 MEDIUM: Session Expiry Not Differentiated by Device

**Location:** `liminallm/config.py:302-303`

**SPEC §18 requires:** "7 days web, 1 day mobile; configurable per plan"

**Current:** Single 24h TTL for all sessions, no web/mobile differentiation.

---

## 3. Rate Limiting and Concurrency

### 3.1 CRITICAL: Concurrency Caps NOT IMPLEMENTED

**Location:** Multiple files (search shows no implementation)

**SPEC §18 requires:**
- Max 3 concurrent workflows per user
- Max 2 concurrent inference decodes per user
- Return 409 "busy" when cap exceeded

**Current:** No tracking of active workflows/inference per user. No 409 responses.

### 3.2 CRITICAL: Per-Plan Adjustable Limits NOT IMPLEMENTED

**Location:** `liminallm/config.py`, `liminallm/api/routes.py`

**SPEC §18 requires:** Rate limits "adjustable per plan"

**Current:** Limits are global constants. User `plan_tier` field exists but never used for rate limit calculation.

### 3.3 MEDIUM: Token Bucket Is Fixed-Window Counter

**Location:** `liminallm/storage/redis_cache.py:42-73`

**SPEC §18 specifies:** "Redis token bucket"

**Current:** Uses fixed-window counter, not true token bucket. Vulnerable to boundary condition attacks (requests spike at window edges).

---

## 4. File Upload/Download Security

### 4.1 CRITICAL: No File Download Endpoint

**Location:** `liminallm/api/routes.py`

**SPEC §13.3 requires:** "GET /v1/files — list user files (paginated)"
**SPEC §12.2 requires:** "signed download URLs for browser fetch"

**Current:** Only `GET /files/limits` and `POST /files/upload` exist. No download capability.

### 4.2 CRITICAL: No Signed URLs (10-Minute Expiry)

**Location:** N/A (not implemented)

**SPEC §18 requires:** "downloads use signed URLs with 10m expiry and content-disposition set to prevent inline execution"

**Current:** No signed URL generation mechanism exists.

### 4.3 CRITICAL: Per-Plan Size Caps Not Enforced

**Location:** `liminallm/api/routes.py:2385-2388`

**SPEC §18 requires:** "free: 25MB/file, paid: 200MB/file"

**Current:** Single global `max_upload_bytes` (10MB default). AuthContext doesn't include `plan_tier`.

### 4.4 HIGH: Content-Disposition Header Missing

**SPEC §18 requires:** "content-disposition set to prevent inline execution"

**Current:** No Content-Disposition header implementation for downloads.

### 4.5 HIGH: MIME Type Validation Absent

**SPEC §2.5 requires:** "skip files over plan cap or unknown mime type"

**Current:** Upload endpoint accepts any file without MIME type validation.

### 4.6 MEDIUM: Temp File Cleanup Not Scheduled

**SPEC §18 requires:** "per-user scratch /users/{id}/tmp auto-cleans daily"

**Current:** No cleanup scheduler exists.

### 4.7 MEDIUM: File Checksum Validation Absent

**SPEC §2.5 requires:** "dedupe by (fs_path_checksum, path)"

**Current:** No checksum calculation on upload, no deduplication.

---

## 5. WebSocket Protocol Compliance

### 5.1 CRITICAL: Missing request_id in Stream Events

**Location:** `liminallm/api/routes.py:2954`

**SPEC §18 requires:** "stream events carry `{ event, data, request_id }`"

**Current:** Events sent without `request_id`: `{"event": event_type, "data": event_data}`

### 5.2 HIGH: No Per-User Connection Limits

**Location:** `liminallm/api/routes.py:2852`

No connection limit enforcement found. Single user could create unlimited concurrent WebSocket connections.

**Impact:** DoS vulnerability - user can exhaust server resources.

### 5.3 MEDIUM: No Mixed Transport Rejection

**SPEC §12.1 requires:** "reject mixed transports without fresh session"

**Current:** No validation that session is "fresh" for WebSocket use.

### 5.4 MEDIUM: Error Events Lack Details Field

**Location:** `liminallm/service/workflow.py:728-732`

**SPEC §18 requires:** Error envelope with `details: <object|array|null>`

**Current:** Error events only have `code` and `message`, no `details` field.

---

## 6. Tool Sandboxing and Circuit Breakers

### 6.1 CRITICAL: No Circuit Breaker Implementation

**SPEC §18 requires:** "circuit breaker opens for a tool after 5 failures in 1 minute"

**Current:** No circuit breaker found anywhere in codebase.

### 6.2 CRITICAL: No Tool Worker Cgroup Limits

**Location:** `liminallm/service/workflow.py:1510-1554`

**SPEC §18 requires:** "Tool workers run under a fixed UID with cgroup limits (CPU shares, memory hard cap)"

**Current:** Tools execute in same process as FastAPI app with no isolation.

### 6.3 CRITICAL: No Filesystem Isolation for Tools

**SPEC §18 requires:** "no filesystem access except tmp scratch"

**Current:** Tools have full access to application filesystem.

### 6.4 CRITICAL: No Allowlisted External Fetch Proxy

**SPEC §18 requires:** "External fetches from tools use a allowlisted proxy with 10s connect + 30s total timeout"

**Current:** No proxy implementation. Tools can make arbitrary outbound requests.

### 6.5 CRITICAL: No Network Egress Allowlist

**SPEC §18 requires:** "network egress allowlist enforcement"

**Current:** No allowlist implementation.

### 6.6 HIGH: No JSON Schema Validation on Tool Inputs/Outputs

**Location:** `liminallm/service/workflow.py:1292-1326`

**SPEC §9.2 requires:** "JSON Schema validation enforced on tool inputs/outputs"

**Current:** Tools invoked with unvalidated inputs, outputs not validated.

### 6.7 HIGH: No html_untrusted Content Sanitization

**SPEC §9.2 requires:** "outputs flagged `content_type: "html_untrusted"` must be sanitized"

**Current:** No content type tracking or sanitization.

### 6.8 HIGH: No Privileged Tool Access Controls

**SPEC §18 requires:** "privileged tools require admin-owned artifacts"

**Current:** No privilege levels. Any authenticated user can invoke any tool.

### 6.9 MEDIUM: Per-Node Timeout Hardcap Not Enforced

**Location:** `liminallm/service/workflow.py:1522-1525`

**SPEC §18 requires:** "per-node timeout default 15s, hard cap 60s"

**Current:** No hardcap validation. Tool specs can set arbitrary timeout_seconds.

---

## 7. Preference/Feedback Handling

### 7.1 CRITICAL: Missing Safety Filtering

**Location:** `liminallm/api/routes.py:1529-1572`

**SPEC §15.1 requires:** "only create `preference_event` if the interaction is policy-compliant; never train adapters on disallowed content"

**Current:** No safety checking before `record_feedback_event()`.

### 7.2 HIGH: Adapter Router State Never Updated After Training

**Location:** `liminallm/service/training_worker.py:150-212`

**SPEC §5.4 requires:** Update `adapter_router_state.centroid_vec` via EMA, `success_score`, `last_trained_at`

**Current:** No `update_adapter_router_state()` method exists in either storage backend.

### 7.3 HIGH: Missing explicit_signal Validation

**Location:** `liminallm/api/schemas.py:489-500`

**SPEC §2.6 specifies:** `explicit_signal` should be: 'like','dislike','always','never'

**Current:** Field accepts any string, no enum validation.

### 7.4 MEDIUM: Score Normalization Missing

**SPEC §2.6 requires:** "score normalized [-1,1]"

**Current:** Score validated but never normalized. No transformation from feedback type.

---

## 8. Clusterer and Skill Discovery

### 8.1 CRITICAL: No Global Clustering

**Location:** `liminallm/service/clustering.py:26-66`

**SPEC §7.1 requires:** Clustering "per user & globally"

**Current:** Only per-user clustering implemented. No code generates global clusters.

### 8.2 HIGH: No Incremental/Streaming Clustering

**SPEC §7.1 requires:** "streaming kmeans / HDBSCAN"

**Current:** Only basic mini-batch kmeans. Recalculates all clusters from scratch each call.

### 8.3 HIGH: No Approximate Clustering for Large Datasets

**SPEC §7.1 requires:** "for large datasets, approximate incremental clustering"

**Current:** All embeddings loaded into memory. No reservoir sampling, coresets, or sketch-based methods.

### 8.4 HIGH: Adapter Pruning/Merging NOT IMPLEMENTED

**Location:** N/A (not implemented)

**SPEC §7.4 requires:** Monitor adapter_router_state for low usage_count, poor success_score. Propose via ConfigOps to disable or merge adapters.

**Current:** No pruning or merging logic exists.

### 8.5 MEDIUM: No Periodic Clustering Batch Job

**SPEC §7.1 requires:** "periodic batch job"

**Current:** Clustering only triggered synchronously in request path.

### 8.6 MEDIUM: Skill Adapter Missing Schema Fields

**Location:** `liminallm/service/clustering.py:180-233`

**SPEC §7.3 requires:** `scope`, `rank`, `layers`, `matrices`, `applicability.natural_language`

**Current:** Many fields missing. Adapters created directly, not proposed via ConfigOps.

---

## 9. Redis Usage and Memory Store Persistence

### 9.1 CRITICAL: user_settings NOT Persisted in Memory Store

**Location:** `liminallm/storage/memory.py:1498-1559`

**SPEC §3.3 requires:** Snapshot must cover "users, auth sessions, credentials, conversations/messages, artifacts + versions, config patches, knowledge contexts, and chunks"

**Current:** `user_settings` stored in memory but NOT included in `_persist_state()` or `_load_state()`.

**Impact:** User settings (locale, timezone, voice, style) reset on restart.

### 9.2 CRITICAL: adapter_router_state NOT Persisted in Memory Store

**Location:** `liminallm/storage/memory.py:1498-1559`

**Current:** `adapter_router_state` stored in memory but NOT included in snapshot persistence.

**Impact:** Adapter routing statistics reset on restart.

### 9.3 HIGH: Missing Serialization Methods

**Location:** `liminallm/storage/memory.py`

Missing methods:
- `_serialize_user_settings()` / `_deserialize_user_settings()`
- `_serialize_adapter_router_state()` / `_deserialize_adapter_router_state()`

---

## 10. Storage Layer Consistency

### 10.1 CRITICAL: search_chunks_pgvector User Isolation Mismatch

**Location:** `liminallm/storage/memory.py:1437` vs `liminallm/storage/postgres.py:2444`

Memory.py has `user_id` as `Optional[str]`, Postgres.py requires it.

### 10.2 HIGH: Missing Validation in Memory Store

**Location:** `liminallm/storage/memory.py:491`

`set_session_meta` in Postgres validates JSON serializability; Memory.py does not.

### 10.3 MEDIUM: SQL Schema Missing NOT NULL Constraints

**Location:** `liminallm/storage/schema.sql`

| Table | Column | Issue |
|-------|--------|-------|
| `users` | `created_at` | Should be NOT NULL DEFAULT NOW() |
| `sessions` | `tenant_id` | Should be NOT NULL for multi-tenant isolation |
| `artifacts` | `visibility` | Should be NOT NULL DEFAULT 'private' |

### 10.4 MEDIUM: SQL Schema Missing Performance Indexes

Missing indexes for common query patterns on sessions, artifacts, chunks tables.

---

## 11. Authentication Service Security

### 11.1 CRITICAL: MFA Lockout Only Works With Cache

**Location:** `liminallm/service/auth.py:748-773`

MFA lockout logic entirely gated on cache availability. Without Redis, unlimited TOTP brute-force possible.

### 11.2 HIGH: Password Reset Non-Functional Without Cache

**Location:** `liminallm/service/auth.py:775-810`

Password reset tokens only stored if cache exists.

### 11.3 MEDIUM: Unused _mfa_challenges Dictionary

**Location:** `liminallm/service/auth.py:128`

Should be used as in-memory fallback for MFA lockout tracking.

---

## 12. Workflow Engine

### 12.1 HIGH: Per-Node Timeout Default Incorrect

**Location:** `liminallm/service/workflow.py:1525`

`DEFAULT_NODE_TIMEOUT_MS = 15000` defined but hardcoded `timeout = 5` used instead.

### 12.2 MEDIUM: Retry Backoff Not Cancellable

**Location:** `liminallm/service/workflow.py:1063-1179`

Retry sleep cannot be interrupted by cancel events.

### 12.3 MEDIUM: Per-Node Timeout Not Enforced Across Retries

Only workflow-level timeout checked during retries, not per-node.

---

## 13. Frontend API Usage

### 13.1 CRITICAL: Password Reset Endpoints Use Wrong Paths

**Location:** `frontend/chat.js`

| Line | Current Path | SPEC Path |
|------|-------------|-----------|
| 800 | `/v1/auth/reset/request` | `/v1/auth/request_reset` |
| 842, 946 | `/v1/auth/reset/confirm` | `/v1/auth/complete_reset` |

### 13.2 HIGH: Missing Idempotency-Key Headers

Multiple POST endpoints in chat.js and admin.js lack required Idempotency-Key headers.

### 13.3 MEDIUM: Voice Endpoints Bypass Error Handling

**Location:** `frontend/chat.js:2156-2244`

Voice endpoints use raw `fetch()` instead of `requestEnvelope()`.

---

## 14. RAG Service

### 14.1 CRITICAL: Path Traversal Vulnerability in ingest_path

**Location:** `liminallm/service/rag.py:453`

`ingest_path()` does not validate that path stays within allowed directories.

### 14.2 MEDIUM: RagMode Enum Missing LOCAL_HYBRID

**Location:** `liminallm/config.py:37-41`

---

## 15. LLM Service

### 15.1 HIGH: Missing max_tokens Enforcement

No validation against SPEC maximum of 4096 tokens.

### 15.2 MEDIUM: Context Window Overflow Not Handled

No truncation or error when context + prompt exceeds model window.

---

## 16. Router Service

### 16.1 HIGH: Undocumented "closest" Selection Behavior

Algorithm and threshold not documented in SPEC §8.

### 16.2 MEDIUM: No Adapter Validation on Assignment

No validation that adapter exists, is compatible, or user has permission.

---

## 17. Config Operations

### 17.1 CRITICAL: Missing write_rate_limit_per_minute Config

**Location:** `liminallm/config.py`

Configuration parameter referenced in routes.py does not exist.

---

## 18. Training Pipeline

### 18.1 CRITICAL: No Deduplication in Dataset Generation

**Location:** `liminallm/training/dataset.py`

Multiple identical prompt-response pairs can appear in training data.

### 18.2 CRITICAL: SFT Prompt Includes Target Message

**Location:** `liminallm/training/sft.py`

Target assistant message included in prompt, violating SFT principles.

---

## 19. Race Conditions and Concurrency Bugs (4th Pass)

### 19.1 CRITICAL: OAuth State TOCTOU Vulnerability

**Location:** `liminallm/storage/redis_cache.py:374-380`

```python
async def pop_oauth_state(self, state_key: str) -> Optional[dict]:
    data = await self.get(f"oauth:{state_key}")  # Read
    if data:
        await self.delete(f"oauth:{state_key}")  # Delete
    return data
```

**Issue:** Time-of-check-to-time-of-use race condition. Two concurrent callbacks with same state can both pass validation before either deletes. This enables OAuth replay attacks.

**Fix:** Use Redis atomic GETDEL operation or Lua script for atomic get-and-delete.

### 19.2 CRITICAL: MemoryStore Reads Without Lock

**Location:** `liminallm/storage/memory.py:557-565, 1041-1042, 786-787`

Multiple read methods access shared state dictionaries without acquiring `_data_lock`:
- `get_session()` reads `self._sessions` directly
- `get_user()` reads `self._users` without lock
- `list_conversations()` iterates `self._conversations`

**Impact:** Concurrent reads during writes can see partial/inconsistent data.

**Fix:** Wrap all read operations in `async with self._data_lock:` blocks.

### 19.3 CRITICAL: MFA Lockout Check-Then-Act Race

**Location:** `liminallm/service/auth.py:744-763`

```python
failures = await self.cache.get_mfa_failures(user_id)
if failures >= self.max_mfa_attempts:
    # ... lockout logic
await self.cache.increment_mfa_failures(user_id)
```

**Issue:** Attacker can flood concurrent TOTP attempts. Each request reads failures=N, all pass check, all increment. Can exceed lockout threshold significantly.

**Fix:** Use atomic Redis INCR with conditional check in Lua script.

### 19.4 CRITICAL: Idempotency Check-Then-Set Race

**Location:** `liminallm/api/routes.py:272-312`

```python
existing = await store.get_idempotency_record(key)
if existing:
    return existing.response
# ... process request
await store.set_idempotency_record(key, response)
```

**Issue:** Two concurrent requests with same idempotency key both see no existing record, both process, potentially creating duplicates.

**Fix:** Use atomic SETNX pattern or database advisory lock.

### 19.5 CRITICAL: Artifact Version Race Condition

**Location:** `liminallm/storage/postgres.py:1814-1818`

```python
result = await conn.fetchrow(
    "SELECT COALESCE(MAX(version), 0) + 1 FROM artifact_versions WHERE artifact_id = $1",
    artifact_id
)
next_version = result[0]
# ... insert with next_version
```

**Issue:** Two concurrent version creates both get same max, both insert same version number.

**Fix:** Use `SELECT ... FOR UPDATE` or auto-incrementing version column.

### 19.6 HIGH: Workflow Executor Thread Safety

**Location:** `liminallm/service/workflow.py:122, 428-600`

The shared `_executor` ThreadPoolExecutor is accessed by multiple async coroutines without synchronization. Additionally, `workflow_traces` and `context_lists` grow unbounded during execution.

**Impact:** Memory leaks and potential thread contention under high concurrency.

### 19.7 HIGH: Session Token Generation Race

**Location:** `liminallm/service/auth.py:533-563`

Token generation uses `secrets.token_urlsafe()` which is thread-safe, but session creation is not atomic. Two concurrent logins could theoretically create duplicate sessions.

### 19.8 MEDIUM: Router Last-Used State Race

**Location:** `liminallm/service/router.py:81`

`_last_used` dictionary updated without synchronization in async context.

---

## 20. Error Handling and Partial Failures (4th Pass)

### 20.1 CRITICAL: Swallowed Exceptions with Bare Pass

**Location:** Multiple files

```python
# liminallm/service/workflow.py:1580
except Exception:
    pass  # Tool cleanup failures silently ignored

# liminallm/storage/memory.py:1540
except Exception:
    pass  # Persistence failures silently lost
```

**Impact:** Data loss and silent failures make debugging impossible.

**Fix:** Log exceptions at minimum; propagate critical failures.

### 20.2 CRITICAL: Training Job Multi-Step Without Rollback

**Location:** `liminallm/service/training.py:259-380`

Training jobs perform multiple database operations:
1. Update job status to "running"
2. Generate dataset
3. Train model
4. Save artifacts
5. Update job status to "completed"

**Issue:** Failure at step 3 or 4 leaves job in "running" state forever with orphaned artifacts.

**Fix:** Implement saga pattern or compensating transactions.

### 20.3 HIGH: Unprotected File Operations

**Location:** `liminallm/service/training.py:684-690`

```python
os.symlink(source, dest)  # No try/except
```

Symlink operations can fail (permissions, existing file, invalid path) without error handling.

### 20.4 HIGH: WebSocket Send Without Error Handling

**Location:** `liminallm/api/routes.py:2954-2980`

```python
await websocket.send_json(event)  # Can raise WebSocketDisconnect
```

WebSocket send failures not caught individually, can terminate entire stream prematurely.

### 20.5 HIGH: Database Connection Errors Not Retried

**Location:** `liminallm/storage/postgres.py` (throughout)

Database operations fail immediately on connection errors without retry logic.

### 20.6 MEDIUM: Redis GET Returns None vs Missing Key

**Location:** `liminallm/storage/redis_cache.py:102-115`

No distinction between "key exists with None value" and "key does not exist".

---

## 21. Transaction Safety and Atomicity (4th Pass)

### 21.1 CRITICAL: Session Revocation Cache-DB Desync

**Location:** `liminallm/storage/postgres.py:1333-1352`

```python
async def revoke_all_user_sessions(...):
    # Delete from database
    await conn.execute("DELETE FROM sessions WHERE user_id = $1 ...")
    # Then invalidate cache
    await self._invalidate_session_cache(user_id)
```

**Issue:** If cache invalidation fails, DB shows no sessions but cache still has valid session tokens.

**Fix:** Use transaction with cache update in finally block, or two-phase approach.

### 21.2 CRITICAL: Config Patch Apply Not Atomic

**Location:** `liminallm/service/config_ops.py:89-99`

Applying a config patch involves multiple operations (validation, application, status update) without transaction wrapping.

### 21.3 HIGH: Artifact Create With Versions Not Atomic

**Location:** `liminallm/storage/postgres.py:1780-1830`

Creating artifact and first version are separate operations. Failure after artifact create leaves orphan.

### 21.4 HIGH: User Create With Settings Not Atomic

**Location:** `liminallm/storage/postgres.py:188-220`

User and initial settings created separately without transaction.

### 21.5 MEDIUM: Conversation Delete Leaves Orphan Messages

If message deletion fails partway through, conversation deleted but messages remain.

---

## 22. Cache Invalidation and Consistency (4th Pass)

### 22.1 CRITICAL: User Role Changes Not Invalidating Session Cache

**Location:** `liminallm/service/auth.py`, `liminallm/storage/redis_cache.py`

When user role/permissions change, existing cached sessions retain old permissions until TTL expires.

**Impact:** Privilege escalation window - demoted admin retains powers for cache TTL duration.

**Fix:** Invalidate all user sessions on permission change, or include version in session token.

### 22.2 CRITICAL: Missing tenant_id in Cache Keys

**Location:** `liminallm/storage/redis_cache.py:194`, `liminallm/service/router.py:81`

```python
f"router:last:{user_id}"  # Missing tenant_id
f"idemp:{key}"            # Missing tenant_id
```

**Impact:** Cross-tenant cache pollution in multi-tenant deployment.

**Fix:** Include tenant_id in all cache key prefixes.

### 22.3 CRITICAL: Password Reset Wrong Cache Key Format

**Location:** `liminallm/service/auth.py:632`

```python
f"user_sessions:{user_id}"  # Key pattern never created
```

**Issue:** Password reset attempts to invalidate sessions using wrong key pattern. Sessions not actually invalidated.

### 22.4 HIGH: Artifact Update Cache Invalidation Missing

**Location:** `liminallm/storage/postgres.py:1850-1890`

Artifact updates don't invalidate any caches. Stale artifact data served until TTL.

### 22.5 HIGH: Rate Limit Counter Not Tenant-Isolated

**Location:** `liminallm/storage/redis_cache.py:42-73`

Rate limit keys don't include tenant_id, allowing cross-tenant rate limit exhaustion.

### 22.6 MEDIUM: Conversation Cache TTL Mismatch

Conversation cached with 5m TTL but messages cached with 1m TTL. Can serve stale message counts.

---

## 23. Resource Cleanup and Memory Management (4th Pass)

### 23.1 CRITICAL: ThreadPoolExecutor Relies on __del__

**Location:** `liminallm/service/workflow.py:1869-1873`

```python
def __del__(self):
    if self._executor:
        self._executor.shutdown(wait=False)
```

**Issue:** `__del__` not guaranteed to run. Executor threads can leak on ungraceful shutdown.

**Fix:** Implement explicit cleanup method called during app shutdown.

### 23.2 CRITICAL: Unbounded _active_requests Dictionary

**Location:** `liminallm/api/routes.py:112-125`

```python
_active_requests: Dict[str, RequestState] = {}
```

**Issue:** Entries added on request start, removed on completion. Network disconnects or crashes leave orphan entries that grow unbounded.

**Fix:** Add TTL-based cleanup or use WeakValueDictionary.

### 23.3 HIGH: WebSocket Listener Not Cleaned Up

**Location:** `liminallm/api/routes.py:2852-3100`

WebSocket handlers don't always clean up listener tasks on disconnect. Task references leak.

### 23.4 HIGH: Workflow Trace Accumulation

**Location:** `liminallm/service/workflow.py:428-600`

```python
workflow_traces.append(trace)  # Never truncated during execution
```

Long-running workflows accumulate unbounded trace data in memory.

### 23.5 MEDIUM: Database Connection Pool Not Monitored

No metrics or alerts for connection pool exhaustion. Silent failures under load.

### 23.6 MEDIUM: File Handle Leaks in Training

**Location:** `liminallm/training/dataset.py`

Some file operations don't use context managers, risking handle leaks.

---

## 24. Edge Cases: Null/Empty/Encoding/Timezone (4th Pass)

### 24.1 CRITICAL: Naive vs Aware Datetime Mixing

**Location:** `liminallm/service/auth.py:1016`, `liminallm/storage/postgres.py` (multiple)

```python
expires_at = datetime.utcnow() + timedelta(...)  # Naive datetime
```

**Issue:** Mixing naive and timezone-aware datetimes causes comparison errors and incorrect expiry calculations.

**Fix:** Use `datetime.now(timezone.utc)` consistently throughout.

### 24.2 HIGH: Unsafe .get() Without None Handling

**Location:** Multiple files

```python
value = data.get("field")
value.strip()  # AttributeError if None
```

Many `.get()` calls followed by method calls without None checks.

### 24.3 HIGH: Float Conversion Without Error Handling

**Location:** `liminallm/service/router.py:145-160`

```python
score = float(raw_score)  # Can raise ValueError
```

Score conversions can fail on invalid input without try/except.

### 24.4 MEDIUM: Empty String vs None Inconsistency

**Location:** `liminallm/storage/memory.py`, `liminallm/storage/postgres.py`

Some methods treat empty string as falsy (skip), others store it. Behavior differs between backends.

### 24.5 MEDIUM: Unicode Normalization Missing

**Location:** `liminallm/service/rag.py:200-250`

Text ingestion doesn't normalize Unicode. Same text with different normalization forms treated as different.

### 24.6 MEDIUM: Locale-Dependent String Operations

**Location:** `liminallm/service/clustering.py:150-180`

`.lower()` and similar operations are locale-dependent, can give inconsistent results.

---

## 25. Pagination and Large Payload Handling (4th Pass)

### 25.1 CRITICAL: list_preference_events No LIMIT Clause

**Location:** `liminallm/storage/postgres.py:370-413`

```python
SELECT * FROM preference_events WHERE user_id = $1
```

**Issue:** No LIMIT clause. Users with many events exhaust memory/timeout.

**Fix:** Add mandatory pagination with max page size.

### 25.2 CRITICAL: Chat Loads All Messages Unbounded

**Location:** `liminallm/api/routes.py:1462-1466`

```python
messages = await store.list_messages(conversation_id)  # All messages
```

**Issue:** Conversations with thousands of messages loaded entirely into memory.

**Fix:** Implement cursor-based pagination for messages.

### 25.3 CRITICAL: search_chunks Loads All Before Scoring

**Location:** `liminallm/storage/postgres.py:2400-2450`

```python
# Load all matching chunks
chunks = await conn.fetch("SELECT * FROM chunks WHERE context_id = $1", ...)
# Then score in Python
scored = [(score(c), c) for c in chunks]
```

**Issue:** Large contexts load all chunks into memory before filtering.

**Fix:** Use database-side scoring and LIMIT.

### 25.4 HIGH: Artifact List No Default Limit

**Location:** `liminallm/storage/postgres.py:1700-1750`

`list_artifacts()` accepts optional limit but defaults to unlimited.

### 25.5 HIGH: Webhook Payload Size Unbounded

**Location:** `liminallm/service/webhooks.py`

Webhook payloads can be arbitrarily large. No truncation before sending.

### 25.6 MEDIUM: Offset Pagination Inefficient for Large Datasets

**Location:** Multiple endpoints

Using `OFFSET` for pagination. Performance degrades linearly with page number.

**Fix:** Implement keyset/cursor pagination.

---

## 26. State Machine Consistency (4th Pass)

### 26.1 CRITICAL: Config Patch Apply Bypasses Approval Check

**Location:** `liminallm/service/config_ops.py:89-99`

```python
async def apply_patch(self, patch_id: str):
    patch = await self.store.get_config_patch(patch_id)
    # BUG: No check that patch.status == "approved"
    await self._apply_operations(patch.operations)
    patch.status = "applied"
```

**Issue:** Anyone can apply a patch regardless of approval status. Skips entire approval workflow.

**Fix:** Add status validation: `if patch.status != "approved": raise ValidationError`

### 26.2 CRITICAL: Training Job Concurrent Processing Race

**Location:** `liminallm/service/training.py:200-260`

```python
job = await store.get_training_job(job_id)
if job.status == "queued":
    job.status = "running"
    await store.update_training_job(job)
    # ... process
```

**Issue:** Two workers can both see status="queued", both set to "running", both process same job.

**Fix:** Use atomic status transition with WHERE clause: `UPDATE ... SET status='running' WHERE status='queued' RETURNING *`

### 26.3 HIGH: No Visibility Transition Guards

**Location:** `liminallm/storage/postgres.py:1850-1890`

Artifact visibility can be changed from any state to any state. No validation of valid transitions (e.g., preventing `global` → `private` once shared).

### 26.4 HIGH: Conversation Status Inconsistent

**Location:** `liminallm/storage/postgres.py:1200-1250`

Conversations have `status` field but no state machine. Can jump between `active`, `archived`, `deleted` arbitrarily.

### 26.5 HIGH: decide_patch Doesn't Check Current Status

**Location:** `liminallm/service/config_ops.py:55-73`

```python
async def decide_patch(self, patch_id: str, approved: bool):
    patch = await self.store.get_config_patch(patch_id)
    # BUG: No check that patch.status == "pending"
    patch.status = "approved" if approved else "rejected"
```

**Issue:** Can approve/reject already-applied patches.

### 26.6 MEDIUM: Message Edit Allows Status Change

Messages can be edited to change their status field directly, bypassing any workflow.

---

## 27. API Contract Validation (5th Pass)

### 27.1 HIGH: Missing Path Parameter Validators

**Location:** `liminallm/api/routes.py` (multiple endpoints)

Multiple endpoints lack proper `Path(...)` validators on path parameters:
- `/conversations/{conversation_id}/messages` (line 2441)
- `/admin/users/{user_id}/role` (line 806)
- `/tools/{tool_id}/invoke` (line 1937)
- `/artifacts/{artifact_id}` PATCH (line 2058)
- `/config/patches/{patch_id}/decide` (line 2169)

**Issue:** No length/type validation on path parameters.

**Fix:** Add `Path(..., max_length=255)` validators.

### 27.2 HIGH: ArtifactRequest.type Claimed Optional But Required

**Location:** `liminallm/api/schemas.py:280-284`

```python
class ArtifactRequest(_SchemaPayload):
    type: Optional[str] = None  # Claims optional but actually required
```

**Issue:** Schema says optional but `routes.py:2020` raises error if missing.

### 27.3 MEDIUM: File Limits Response Missing Extensions

**Location:** `liminallm/api/routes.py:2330-2344`

`/files/limits` returns only `max_upload_bytes` but SPEC §17 mentions "allowed extensions from GET /v1/files/limits".

### 27.4 MEDIUM: Schema Validation Not JSON-Serializable Check

**Location:** `liminallm/api/schemas.py:268-277`

`_SchemaPayload.schema_` field accepts any dict without validating JSON serializability.

---

## 28. Service Initialization Issues (5th Pass)

### 28.1 CRITICAL: Thread-Unsafe Singleton in get_runtime()

**Location:** `liminallm/service/runtime.py:164-171`

```python
def get_runtime() -> Runtime:
    global runtime
    if runtime is None:  # TOCTOU race
        runtime = Runtime()
    return runtime
```

**Issue:** Non-atomic check-then-act without locks. Multiple threads can create multiple Runtime instances.

**Impact:** Duplicate database pools, memory leaks, lost state.

### 28.2 CRITICAL: Asyncio Lock at Module Import Time

**Location:** `liminallm/api/routes.py:113`

```python
_active_requests_lock = asyncio.Lock()  # Created before event loop exists
```

**Issue:** Lock created during module import, before any event loop. Can cause "No running event loop" errors.

### 28.3 HIGH: Missing Cleanup Hooks for Services

**Location:** Multiple files

- VoiceService has `close()` (voice.py:262) but never called
- PostgreSQL connection pool never explicitly closed
- Redis cache connections not cleaned on shutdown

**Impact:** Resource leaks on shutdown.

### 28.4 HIGH: AuthService Mutable State Not Thread-Safe

**Location:** `liminallm/service/auth.py:128-133`

```python
self._mfa_challenges: dict[str, tuple[str, datetime]] = {}
self._oauth_states: dict[str, tuple[str, datetime, Optional[str]]] = {}
```

**Issue:** Multiple unprotected mutable dictionaries accessed concurrently without locks.

### 28.5 HIGH: Config Validation Deferred to Runtime

**Location:** `liminallm/config.py:385-446`

JWT secret generation happens in field validator at first access, not at startup. File system errors occur at first auth request.

---

## 29. Configuration Validation Issues (5th Pass)

### 29.1 CRITICAL: Sensitive Config in Logs

**Location:** `liminallm/service/runtime.py:71`

```python
logger.warning("redis_disabled_fallback", redis_url=self.settings.redis_url)
```

**Issue:** Redis URL (may contain password) logged without masking.

### 29.2 CRITICAL: Undocumented Environment Variables

Multiple env vars read directly via `os.getenv()` but not in Settings class:
- `LOG_LEVEL`, `LOG_JSON`, `LOG_DEV_MODE` (logging.py:98-100)
- `BUILD_SHA` (app.py:22)
- `CORS_ALLOW_ORIGINS` (app.py:55)
- `ENABLE_HSTS` (app.py:123)
- `MFA_SECRET_KEY` (memory.py:113)

**Impact:** No centralized config discovery or validation.

### 29.3 HIGH: Missing Integer Range Validators

**Location:** `liminallm/config.py:294-341`

12+ config values (rate limits, TTLs, page sizes) have no min/max bounds:
- `chat_rate_limit_per_minute` - no bounds
- `training_worker_poll_interval` - `0` would loop infinitely
- `smtp_port` - no 1-65535 validation

### 29.4 HIGH: Inconsistent Boolean Parsing

**Location:** Multiple files

Boolean env vars parsed inconsistently:
- `app.py:72`: `flag.lower() in {"1", "true", "yes", "on"}`
- Pydantic uses stricter parsing

### 29.5 MEDIUM: Optional Config Dependencies Not Validated

OAuth and SMTP configs are optional individually but should require pairs:
- `oauth_google_client_id` set but not `oauth_google_client_secret`
- `smtp_host` set but not `smtp_password`

---

## 30. Logging and Observability Gaps (5th Pass)

### 30.1 HIGH: Missing Per-Node Latency in Workflow Traces

**Location:** `liminallm/service/workflow.py:881-882`

**SPEC §15.2 requires:** "workflow traces: per-node latency, retries, timeout counts"

**Current:** Traces only include node ID and result, not latency metrics.

### 30.2 HIGH: Routing/Workflow Trace Functions Never Called

**Location:** `liminallm/logging.py:117-126`

`log_routing_trace()` and `log_workflow_trace()` defined but never used anywhere in codebase.

### 30.3 HIGH: Missing SPEC §15.2 Metrics

**Location:** `liminallm/app.py:263-320`

`/metrics` endpoint missing:
- Request latency histograms
- Tokens in/out per call
- Adapter usage counts & success_score
- Preference event rates
- Training job metrics

### 30.4 HIGH: Silent Exception in Auth Cache Clear

**Location:** `liminallm/service/auth.py:633-634`

```python
except Exception:
    pass  # NO LOGGING
```

**Impact:** Redis failures invisible; debugging impossible.

### 30.5 MEDIUM: Chat Endpoint Minimal Logging

**Location:** `liminallm/api/routes.py:1336-1468`

Only 8 logging statements in 3,146 lines. Chat endpoint has no logging of:
- Request metadata with correlation IDs
- Token counts
- Adapter selection decisions

---

## 31. Business Logic Constraint Violations (5th Pass)

### 31.1 CRITICAL: Global Artifacts Inaccessible to Users

**Location:** `liminallm/api/routes.py:414-422`

`_get_owned_artifact()` blocks access to global artifacts (visibility='global') for non-admin users.

**SPEC §12.2 requires:** Global artifacts accessible to all users.

### 31.2 CRITICAL: list_artifacts Missing Global Items

**Location:** `liminallm/api/routes.py:1684-1690`

Query only returns artifacts owned by user, missing global system artifacts.

**Impact:** Users can't discover default workflows, policies, tool specs.

### 31.3 HIGH: RAG Cannot Access Shared Contexts

**Location:** `liminallm/service/rag.py:210, 229`

RAG filters out all contexts not owned by user, preventing shared knowledge base access.

### 31.4 HIGH: File Size Limits Not Plan-Differentiated

**Location:** `liminallm/api/routes.py:2385-2388`

**SPEC §18 requires:** Free: 25MB/file, Paid: 200MB/file

**Current:** Single global `max_upload_bytes` for all plans.

### 31.5 MEDIUM: Global Training Job Limit Missing

**Location:** `liminallm/service/training.py:419-428`

Per-user cooldown enforced but no global concurrency cap. Could exhaust GPU resources.

---

## 32. Async/Await Anti-Patterns (5th Pass)

### 32.1 CRITICAL: Blocking Tool Execution in Async Context

**Location:** `liminallm/service/workflow.py:1110, 1547`

```python
async def _execute_node_with_retry(...):
    result = self._execute_node(...)  # Sync call

def _execute_node(...):
    return future.result(timeout=timeout)  # BLOCKS event loop
```

**Issue:** Async function calls sync method that blocks on `future.result()` for up to 5 seconds.

**Impact:** Event loop stalls; WebSocket streaming and concurrent operations freeze.

**Fix:** Use `asyncio.to_thread()` for blocking operations.

### 32.2 HIGH: Fire-and-Forget Cache Close Task

**Location:** `liminallm/service/runtime.py:190`

```python
loop.create_task(runtime.cache.close())  # Task not stored or awaited
```

**Issue:** Cache close task may not complete before shutdown.

### 32.3 HIGH: Blocking File I/O in Async Upload

**Location:** `liminallm/api/routes.py:2417`

```python
async def upload_file(...):
    dest_path.write_bytes(contents)  # Sync file I/O blocks event loop
```

**Impact:** Large file uploads block all concurrent requests.

---

## 33. Frontend-Backend Contract Mismatches (5th Pass)

### 33.1 CRITICAL: content_struct.citations Field Mismatch

**Location:** Frontend `chat.js:1415` vs Backend `content_struct.py:55-57`

Frontend expects: `data.content_struct.citations` (top-level array)
Backend provides: Citations embedded in `content_struct.segments` as type="citation"

**Impact:** Citations never display in UI.

### 33.2 CRITICAL: WebSocket tenant_id From Message Body

**Location:** `chat.js:1571` vs `routes.py:2862-2872`

Frontend sends `tenant_id` in WebSocket message body. REST endpoints use `X-Tenant-ID` header.

**Security:** Per CLAUDE.md, tenant_id should be from JWT, not user input.

### 33.3 HIGH: Pagination Response Ignored

**Location:** `chat.js:1016` vs `routes.py:2525-2562`

Backend returns `has_next`, `next_page`, `total_count` but frontend ignores pagination.

**Impact:** Users can only see first 50 conversations.

### 33.4 HIGH: Admin.js Error Extraction Wrong Path

**Location:** `admin.js:141-147`

```javascript
const detail = payload?.detail || payload?.error || payload;
// Should be: payload?.error?.message
```

**Impact:** Admin console shows raw error objects instead of messages.

### 33.5 MEDIUM: VoiceSynthesis audio_path Fallback Missing

**Location:** `chat.js:2200` vs `routes.py:2827-2852`

Frontend only checks `audio_url`, ignoring `audio_path` fallback. Voice synthesis may fail unnecessarily.

---

## 34. Previously Resolved Issues

### 34.1 Session Exception Parameter (FIXED)

**Commit:** 3beddff

The `except_session_id` parameter in `revoke_all_user_sessions` now properly passed to store methods.

---

## Summary by Severity

### Critical (66 Issues)

| # | Issue | Location |
|---|-------|----------|
| 1 | Invalid error codes | routes.py:1393,1401,3114 |
| 2 | Non-spec WebSocket event "streaming_complete" | routes.py:3020-3033 |
| 3 | Idempotency not stored for create_conversation | routes.py:2521 |
| 4 | OAuth tenant_id from user input | routes.py:640,674 |
| 5 | Visibility filter broken for global artifacts | routes.py:1684-1691 |
| 6 | PATCH /artifacts not RFC 6902 compliant | routes.py:1720-1745 |
| 7 | Session rotation (24h activity) NOT IMPLEMENTED | auth.py:533-563 |
| 8 | Single-session mode NOT IMPLEMENTED | auth.py:533-563 |
| 9 | X-Session header for WebSockets NOT IMPLEMENTED | routes.py:2853-2875 |
| 10 | Concurrency caps NOT IMPLEMENTED | Multiple |
| 11 | Per-plan rate limits NOT IMPLEMENTED | config.py, routes.py |
| 12 | No file download endpoint | routes.py |
| 13 | No signed URLs (10m expiry) | N/A |
| 14 | Per-plan file size caps not enforced | routes.py:2385-2388 |
| 15 | Missing request_id in stream events | routes.py:2954 |
| 16 | No circuit breaker implementation | N/A |
| 17 | No tool worker cgroup limits | workflow.py:1510-1554 |
| 18 | No filesystem isolation for tools | N/A |
| 19 | No allowlisted external fetch proxy | N/A |
| 20 | No network egress allowlist | N/A |
| 21 | Missing safety filtering for preferences | routes.py:1529-1572 |
| 22 | No global clustering | clustering.py:26-66 |
| 23 | user_settings NOT persisted | memory.py:1498-1559 |
| 24 | adapter_router_state NOT persisted | memory.py:1498-1559 |
| 25 | search_chunks_pgvector user isolation mismatch | memory.py:1437 |
| 26 | MFA lockout only works with cache | auth.py:748-773 |
| 27 | Password reset non-functional without cache | auth.py:775-810 |
| 28 | Frontend password reset wrong endpoints | chat.js:800,842,946 |
| 29 | Path traversal vulnerability in ingest_path | rag.py:453 |
| 30 | Missing write_rate_limit_per_minute config | config.py |
| 31 | No deduplication in training dataset | training/dataset.py |
| 32 | SFT prompt includes target message | training/sft.py |
| 33 | Frontend expects non-spec streaming_complete | chat.js:1484 |
| 34 | OAuth state TOCTOU vulnerability | redis_cache.py:374-380 |
| 35 | MemoryStore reads without lock | memory.py:557-565 |
| 36 | MFA lockout check-then-act race | auth.py:744-763 |
| 37 | Idempotency check-then-set race | routes.py:272-312 |
| 38 | Artifact version race condition | postgres.py:1814-1818 |
| 39 | Swallowed exceptions with bare pass | workflow.py, memory.py |
| 40 | Training job multi-step without rollback | training.py:259-380 |
| 41 | Session revocation cache-DB desync | postgres.py:1333-1352 |
| 42 | Config patch apply not atomic | config_ops.py:89-99 |
| 43 | User role changes not invalidating cache | auth.py, redis_cache.py |
| 44 | Missing tenant_id in cache keys | redis_cache.py:194, router.py:81 |
| 45 | Password reset wrong cache key format | auth.py:632 |
| 46 | ThreadPoolExecutor relies on __del__ | workflow.py:1869-1873 |
| 47 | Unbounded _active_requests dictionary | routes.py:112-125 |
| 48 | Naive vs aware datetime mixing | auth.py:1016, postgres.py |
| 49 | list_preference_events no LIMIT | postgres.py:370-413 |
| 50 | Chat loads all messages unbounded | routes.py:1462-1466 |
| 51 | search_chunks loads all before scoring | postgres.py:2400-2450 |
| 52 | Config patch apply bypasses approval | config_ops.py:89-99 |
| 53 | Training job concurrent processing race | training.py:200-260 |
| 54 | Thread-unsafe singleton get_runtime() | runtime.py:164-171 |
| 55 | Asyncio Lock at module import time | routes.py:113 |
| 56 | Sensitive config (redis_url) in logs | runtime.py:71 |
| 57 | Undocumented environment variables | logging.py, app.py |
| 58 | Global artifacts inaccessible to users | routes.py:414-422 |
| 59 | list_artifacts missing global items | routes.py:1684-1690 |
| 60 | Blocking tool execution in async context | workflow.py:1110, 1547 |
| 61 | content_struct.citations field mismatch | chat.js:1415 vs content_struct.py |
| 62 | WebSocket tenant_id from message body | chat.js:1571 vs routes.py:2862 |

### High Priority (52 Issues)

| # | Issue | Location |
|---|-------|----------|
| 1 | Access tokens not denylisted on logout | auth.py:591-605 |
| 2 | No per-user WebSocket connection limits | routes.py:2852 |
| 3 | Content-disposition header missing | N/A |
| 4 | MIME type validation absent | routes.py upload |
| 5 | No JSON Schema validation on tool I/O | workflow.py:1292-1326 |
| 6 | No html_untrusted content sanitization | N/A |
| 7 | No privileged tool access controls | routes.py:1935-1975 |
| 8 | Adapter router state never updated | training_worker.py:150-212 |
| 9 | Missing explicit_signal validation | schemas.py:489-500 |
| 10 | No incremental/streaming clustering | clustering.py |
| 11 | No approximate clustering for large data | clustering.py |
| 12 | Adapter pruning/merging NOT IMPLEMENTED | N/A |
| 13 | Missing serialization methods | memory.py |
| 14 | Missing JSON validation in memory store | memory.py:491 |
| 15 | Per-node timeout default 5s not 15s | workflow.py:1525 |
| 16 | Frontend missing idempotency keys | chat.js, admin.js |
| 17 | Voice endpoints bypass error handling | chat.js:2156-2244 |
| 18 | Missing max_tokens enforcement | llm.py |
| 19 | Undocumented "closest" adapter selection | router.py |
| 20 | No adapter validation on assignment | router.py |
| 21 | WebSocket event name mismatch | chat.js:1484 |
| 22 | Context window overflow not handled | llm.py |
| 23 | Password reset cache dependency | auth.py:775-810 |
| 24 | Workflow executor thread safety | workflow.py:122, 428-600 |
| 25 | Session token generation race | auth.py:533-563 |
| 26 | Unprotected file operations | training.py:684-690 |
| 27 | WebSocket send without error handling | routes.py:2954-2980 |
| 28 | Database connection errors not retried | postgres.py |
| 29 | Artifact create with versions not atomic | postgres.py:1780-1830 |
| 30 | User create with settings not atomic | postgres.py:188-220 |
| 31 | Artifact update cache invalidation missing | postgres.py:1850-1890 |
| 32 | Rate limit counter not tenant-isolated | redis_cache.py:42-73 |
| 33 | WebSocket listener not cleaned up | routes.py:2852-3100 |
| 34 | Workflow trace accumulation | workflow.py:428-600 |
| 35 | Unsafe .get() without None handling | Multiple files |
| 36 | Missing Path parameter validators | routes.py (multiple) |
| 37 | ArtifactRequest.type optional but required | schemas.py:280-284 |
| 38 | Missing cleanup hooks for services | voice.py, postgres.py |
| 39 | AuthService mutable state not thread-safe | auth.py:128-133 |
| 40 | Config validation deferred to runtime | config.py:385-446 |
| 41 | Missing integer range validators | config.py:294-341 |
| 42 | Inconsistent boolean parsing | app.py, logging.py |
| 43 | Missing per-node latency in traces | workflow.py:881-882 |
| 44 | Routing/workflow trace functions unused | logging.py:117-126 |
| 45 | Missing SPEC §15.2 metrics | app.py:263-320 |
| 46 | Silent exception in auth cache clear | auth.py:633-634 |
| 47 | RAG cannot access shared contexts | rag.py:210, 229 |
| 48 | File size limits not plan-differentiated | routes.py:2385-2388 |
| 49 | Fire-and-forget cache close task | runtime.py:190 |
| 50 | Blocking file I/O in async upload | routes.py:2417 |
| 51 | Pagination response ignored by frontend | chat.js:1016 |
| 52 | Admin.js error extraction wrong path | admin.js:141-147 |

### Medium Priority (34 Issues)

| # | Issue | Location |
|---|-------|----------|
| 1 | Session expiry not differentiated by device | config.py:302-303 |
| 2 | Token bucket is fixed-window counter | redis_cache.py:42-73 |
| 3 | Temp file cleanup not scheduled | N/A |
| 4 | File checksum validation absent | routes.py upload |
| 5 | No mixed transport rejection | routes.py:2869 |
| 6 | Error events lack details field | workflow.py:728 |
| 7 | Per-node timeout hardcap not enforced | workflow.py:1522-1525 |
| 8 | Score normalization missing | training.py |
| 9 | No periodic clustering batch job | clustering.py |
| 10 | Skill adapter missing schema fields | clustering.py:180-233 |
| 11 | SQL schema missing NOT NULL constraints | schema.sql |
| 12 | SQL schema missing performance indexes | schema.sql |
| 13 | Unused _mfa_challenges dictionary | auth.py:128 |
| 14 | Retry backoff not cancellable | workflow.py:1162 |
| 15 | Per-node timeout not enforced across retries | workflow.py |
| 16 | RagMode enum missing LOCAL_HYBRID | config.py:37-41 |
| 17 | Undocumented API endpoints | SPEC.md vs frontend |
| 18 | Pagination default inconsistency | routes.py:2539 |
| 19 | Router last-used state race | router.py:81 |
| 20 | Redis GET None vs missing key | redis_cache.py:102-115 |
| 21 | Conversation delete leaves orphan messages | postgres.py |
| 22 | Conversation cache TTL mismatch | redis_cache.py |
| 23 | Database connection pool not monitored | postgres.py |
| 24 | File handle leaks in training | training/dataset.py |
| 25 | Float conversion without error handling | router.py:145-160 |
| 26 | Empty string vs None inconsistency | memory.py, postgres.py |
| 27 | File limits response missing extensions | routes.py:2330-2344 |
| 28 | Schema validation not JSON-serializable | schemas.py:268-277 |
| 29 | Optional config dependencies not validated | config.py (OAuth, SMTP) |
| 30 | Chat endpoint minimal logging | routes.py:1336-1468 |
| 31 | Global training job limit missing | training.py:419-428 |
| 32 | VoiceSynthesis audio_path fallback missing | chat.js:2200 |
| 33 | Adapter max policy-based not hardcapped | router.py:404-420 |
| 34 | Expression length limits missing | workflow.py:1846-1867 |

---

## Recommendations

### Immediate Actions (Security Critical)

1. **Tool Sandboxing**: Implement circuit breaker, cgroup limits, filesystem isolation
2. **Path Traversal**: Add path validation in `ingest_path()` using safe_join pattern
3. **OAuth tenant_id**: Derive from OAuth provider claims, not user input
4. **Safety Filtering**: Add policy compliance check before preference events
5. **Training Pipeline**: Fix SFT to exclude target from prompt; add deduplication
6. **MFA Lockout**: Add in-memory fallback using _mfa_challenges dict
7. **Concurrency Caps**: Implement 3 workflow / 2 inference limits with 409 responses
8. **OAuth TOCTOU**: Use atomic GETDEL for OAuth state validation
9. **Cache Tenant Isolation**: Add tenant_id prefix to all cache keys
10. **State Machine Guards**: Add status checks in config_ops apply/decide methods

### Session Management Actions

1. Implement session rotation (24h activity-based)
2. Add single-session mode enforcement
3. Add access token denylist on logout
4. Implement X-Session header for WebSockets
5. Invalidate sessions on permission/role changes

### File Service Actions

1. Implement file download endpoint with signed URLs
2. Add per-plan size cap enforcement
3. Add Content-Disposition headers
4. Implement MIME type validation
5. Add temp file cleanup scheduler

### Storage Actions

1. Add user_settings and adapter_router_state to memory store persistence
2. Add missing serialization methods
3. Add NOT NULL constraints to schema
4. Add performance indexes
5. Add MemoryStore read locks for concurrent access safety
6. Use SELECT FOR UPDATE for version number generation

### Clustering/Training Actions

1. Implement global clustering
2. Add incremental clustering algorithm
3. Implement adapter pruning/merging
4. Add periodic clustering batch job
5. Update adapter_router_state after training
6. Add saga pattern/rollback for multi-step training jobs

### Race Condition Fixes (4th Pass)

1. **Idempotency**: Use SETNX pattern for idempotency records
2. **Artifact Versions**: Use auto-increment or SELECT FOR UPDATE
3. **MFA Lockout**: Use atomic INCR with conditional check in Lua
4. **Training Jobs**: Atomic status transitions with WHERE clause

### Error Handling Improvements (4th Pass)

1. Replace bare `except: pass` with logged exceptions
2. Add retry logic for transient database errors
3. Wrap WebSocket sends in try/except
4. Add file operation error handling in training

### Resource Cleanup Actions (4th Pass)

1. Implement explicit cleanup method for ThreadPoolExecutor
2. Add TTL-based cleanup for _active_requests dictionary
3. Clean up WebSocket listener tasks on disconnect
4. Bound workflow trace accumulation during execution

### Pagination Actions (4th Pass)

1. Add mandatory LIMIT to list_preference_events
2. Implement cursor-based pagination for messages
3. Use database-side scoring for chunk search
4. Add default limits to all list endpoints

### Documentation Actions

1. Update SPEC.md to document all endpoints used by frontend
2. Document router "closest" selection algorithm
3. Document cache requirements for auth features
4. Document datetime handling (always use timezone-aware)

### Service Initialization Actions (5th Pass)

1. **Thread-safe Singleton**: Add threading.Lock to get_runtime()
2. **Lazy Lock Creation**: Create asyncio.Lock lazily, not at module import
3. **Shutdown Hooks**: Add explicit cleanup in app lifespan for VoiceService, DB pool, Redis
4. **AuthService Locks**: Add thread locks to mutable state dictionaries
5. **Early Config Validation**: Validate JWT secret and paths at startup, not first access

### Configuration Actions (5th Pass)

1. **Centralize Config**: Move all os.getenv() calls to Settings class
2. **Mask Sensitive Logs**: Filter redis_url and other secrets from log output
3. **Add Range Validators**: Add min/max bounds to all integer config values
4. **Consistent Boolean Parsing**: Use Pydantic's built-in boolean parsing everywhere
5. **Validate Config Pairs**: Ensure OAuth and SMTP configs are all-or-nothing

### Observability Actions (5th Pass)

1. **Per-Node Latency**: Add timing to workflow trace for each node execution
2. **Enable Trace Logging**: Call log_routing_trace() and log_workflow_trace()
3. **Add SPEC Metrics**: Implement latency, tokens, adapter usage, preference rate metrics
4. **Log All Exceptions**: Replace bare `except: pass` with logged exceptions

### Business Logic Actions (5th Pass)

1. **Fix Artifact Visibility**: Allow users to access global artifacts
2. **Fix list_artifacts**: Include global and shared artifacts in listing
3. **Fix RAG Context Access**: Allow shared context access per visibility rules
4. **Plan-Based Limits**: Implement per-plan file size limits (25MB free, 200MB paid)

### Async Pattern Actions (5th Pass)

1. **Async Tool Execution**: Use asyncio.to_thread() for blocking tool invocations
2. **Await Cache Close**: Store and await cache close task properly
3. **Async File I/O**: Use asyncio.to_thread() for file writes in upload endpoint

### Frontend Contract Actions (5th Pass)

1. **Fix Citations**: Extract from content_struct.segments, not top-level citations
2. **Fix Tenant ID**: Derive from JWT in WebSocket, not message body
3. **Implement Pagination**: Use has_next/next_page in frontend list views
4. **Fix Error Extraction**: Use payload.error.message in admin.js
