# Codebase Issues and Security Audit

**Last Updated:** 2025-12-03
**Scope:** Comprehensive review against SPEC.md requirements (8th pass)

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
- JWT/Authentication security (6th pass)
- Workflow graph security (6th pass)
- Data integrity issues (6th pass)
- DoS/Resource exhaustion (6th pass)
- Type coercion vulnerabilities (6th pass)
- CSRF/Session security (6th pass)
- Frontend XSS vulnerabilities (6th pass)
- Error handling security (6th pass)
- Adapter/LoRA security (7th pass)
- Multi-tenant isolation (7th pass)
- Embedding/vector security (7th pass)
- Input validation edge cases (7th pass)
- Content redaction security (7th pass)
- Deadlock/timeout patterns (7th pass)
- API versioning/compatibility (7th pass)
- RBAC and permission security (8th pass)
- Audit logging and compliance gaps (8th pass)
- HTTP security headers (8th pass)
- Business logic vulnerabilities (8th pass)
- Frontend security issues (8th pass)
- External API integration issues (8th pass)
- Cryptographic implementation issues (8th pass)

**Critical Issues Found:** 104 (83 from passes 1-7, 21 new in 8th pass)
**High Priority Issues:** 105 (90 from passes 1-7, 15 new in 8th pass)
**Medium Priority Issues:** 111 (75 from passes 1-7, 36 new in 8th pass)
**Total Issues:** 320
**False Positives Identified:** 4 (Issues 19.1, 33.2, 33.4, 33.5)

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

### 19.1 ~~CRITICAL: OAuth State TOCTOU Vulnerability~~ (FALSE POSITIVE - VERIFIED SAFE)

**Location:** `liminallm/storage/redis_cache.py:143-189`

**Original Claim:** TOCTOU race condition in OAuth state handling.

**Verification Result:** The actual implementation at lines 143-189 correctly uses atomic operations:

```python
async def pop_oauth_state(self, state: str) -> Optional[tuple[str, datetime, Optional[str]]]:
    """Atomically get and delete OAuth state to prevent replay attacks."""
    key = f"auth:oauth:{state}"
    # Try GETDEL first (Redis 6.2+) for atomic get-and-delete
    try:
        cached = await self.client.getdel(key)
    except AttributeError:
        # Fallback: use Lua script for atomicity
        lua_script = """
        local value = redis.call('GET', KEYS[1])
        if value then redis.call('DEL', KEYS[1]) end
        return value
        """
        cached = await self.client.eval(lua_script, 1, key)
```

**Status:** No vulnerability exists. Code uses atomic GETDEL or Lua script.

### 19.2 HIGH: MemoryStore Reads Without Lock (PARTIALLY CORRECTED)

**Location:** `liminallm/storage/memory.py:557-565, 786-787, 997-1002, 1041-1042`

**Verification Result:** Some claims were false positives:
- ~~`get_session()`~~ - DOES use lock (line 487-489) ✓
- ~~`get_user()`~~ - DOES use lock (line 281-283) ✓

**Confirmed issues (TRUE POSITIVES):**
- `get_conversation()` at lines 557-565 - NO lock
- `get_artifact()` at lines 1041-1042 - NO lock
- `get_semantic_cluster()` at lines 786-787 - NO lock
- `list_conversations()` at lines 997-1002 - NO lock

**Impact:** Concurrent reads during writes can see partial/inconsistent data for conversations, artifacts, and clusters.

**Fix:** Wrap remaining unprotected read operations in `with self._data_lock:` blocks.

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

### 33.2 ~~CRITICAL: WebSocket tenant_id From Message Body~~ (FALSE POSITIVE - VERIFIED SAFE)

**Location:** `chat.js:1571` vs `routes.py:2862-2872`

**Original Claim:** Backend accepts tenant_id from WebSocket message body.

**Verification Result:** While frontend DOES send `tenant_id` in the message body, the backend correctly IGNORES it. The backend uses `auth_ctx.tenant_id` derived from the authenticated JWT token (routes.py:2947), not from `init.get("tenant_id")`.

**Status:** Backend implementation follows CLAUDE.md security guideline. Frontend sends unnecessary data that is properly ignored.

### 33.3 HIGH: Pagination Response Ignored

**Location:** `chat.js:1016` vs `routes.py:2525-2562`

Backend returns `has_next`, `next_page`, `total_count` but frontend ignores pagination.

**Impact:** Users can only see first 50 conversations.

### 33.4 ~~HIGH: Admin.js Error Extraction Wrong Path~~ (FALSE POSITIVE - VERIFIED CORRECT)

**Location:** `admin.js:141-147`

**Original Claim:** Error extraction uses wrong path.

**Verification Result:** The actual code handles multiple fallback paths correctly:

```javascript
const extractError = (payload, fallback) => {
  const detail = payload?.detail || payload?.error || payload;
  if (typeof detail === 'string') return detail.trim() || fallback;
  if (detail?.message) return detail.message;
  if (detail?.error?.message) return detail.error.message;  // Line 145 - handles nested path
  return fallback;
};
```

**Status:** Error extraction is properly implemented with multiple fallback paths.

### 33.5 ~~MEDIUM: VoiceSynthesis audio_path Fallback Missing~~ (FALSE POSITIVE - VERIFIED CORRECT)

**Location:** `chat.js:2200` vs `routes.py:2827-2852`

**Original Claim:** Frontend should fallback to `audio_path` when `audio_url` missing.

**Verification Result:**
1. Backend ALWAYS returns `audio_url` (relative URL for browser fetch)
2. `audio_path` is a server filesystem path, NOT usable by browser
3. Frontend correctly checks `audio_url` and has browser speech synthesis fallback

**Status:** Implementation is correct. Using `audio_path` as fallback would not work since it's a filesystem path.

---

## 34. Previously Resolved Issues

### 34.1 Session Exception Parameter (FIXED)

**Commit:** 3beddff

The `except_session_id` parameter in `revoke_all_user_sessions` now properly passed to store methods.

---

## Summary by Severity

### Critical (63 Issues) - After False Positive Verification

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
| ~~34~~ | ~~OAuth state TOCTOU vulnerability~~ | **FALSE POSITIVE** - Uses atomic GETDEL/Lua |
| ~~35~~ | ~~MemoryStore reads without lock~~ | **DOWNGRADED to HIGH** - Only some methods |
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
| ~~62~~ | ~~WebSocket tenant_id from message body~~ | **FALSE POSITIVE** - Backend ignores it |

### High Priority (52 Issues) - After False Positive Verification

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
| ~~52~~ | ~~Admin.js error extraction wrong path~~ | **FALSE POSITIVE** - Handles multiple paths |
| 53 | MemoryStore reads without lock (partial) | memory.py:557-565, 786-787, 1041 |

### Medium Priority (33 Issues) - After False Positive Verification

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
| ~~32~~ | ~~VoiceSynthesis audio_path fallback missing~~ | **FALSE POSITIVE** - audio_url always provided |
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
8. ~~**OAuth TOCTOU**: Use atomic GETDEL for OAuth state validation~~ - **FALSE POSITIVE** (already uses atomic ops)
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
2. ~~**Fix Tenant ID**: Derive from JWT in WebSocket, not message body~~ - **FALSE POSITIVE** (already correct)
3. **Implement Pagination**: Use has_next/next_page in frontend list views
4. ~~**Fix Error Extraction**: Use payload.error.message in admin.js~~ - **FALSE POSITIVE** (already handles multiple paths)

---

## 35. JWT/Authentication Security (6th Pass)

### 35.1 CRITICAL: JWT Header Not Validated (Algorithm Confusion)

**Location:** `liminallm/service/auth.py:920-932`

JWT decoding does not validate the header's algorithm field. An attacker could craft a token with `"alg": "none"` or manipulate the algorithm to bypass signature verification.

**Impact:** Complete authentication bypass possible.

**Fix:** Decode and validate JWT header before signature verification; reject unexpected algorithms.

### 35.2 CRITICAL: Tenant ID Spoofing via Request Body

**Location:** `liminallm/api/routes.py:545-556, 591-603, 639-649`

Signup, login, and OAuth endpoints accept `tenant_id` from request body before authentication. Violates CLAUDE.md security guideline.

**Impact:** Users can create accounts in arbitrary tenants.

**Fix:** Remove tenant_id from unauthenticated request bodies; derive from server config or OAuth claims.

### 35.3 HIGH: Missing JWT Standard Claims (iat/nbf)

**Location:** `liminallm/service/auth.py:975-1008`

Tokens are issued without `iat` (issued at) or `nbf` (not before) claims.

**Impact:** Cannot detect token age or prevent pre-dated tokens.

### 35.4 HIGH: No JWT Clock Skew Tolerance

**Location:** `liminallm/service/auth.py:964-972`

Token expiration checked without clock skew tolerance.

**Impact:** Valid tokens rejected due to minor clock drift between servers.

### 35.5 HIGH: Weak Password Requirements

**Location:** `liminallm/api/schemas.py:90-96`

Password validation only checks length (8+ chars), no complexity requirements.

**Impact:** Vulnerable to dictionary attacks and credential stuffing.

### 35.6 HIGH: MFA Secret Uses JWT Key

**Location:** `liminallm/storage/memory.py:111-139`

MFA encryption cipher falls back to using JWT_SECRET when MFA_SECRET_KEY not set.

**Impact:** Single key compromise affects both JWT and MFA security.

---

## 36. Workflow Graph Security (6th Pass)

### 36.1 MEDIUM: Missing Cycle Detection in Workflow Graphs

**Location:** `liminallm/service/workflow.py:418-627`

No explicit cycle detection algorithm for workflow graphs. Cycles detected only through per-node visit limits at runtime.

**Impact:** Malicious workflows can exhaust resources before loop detection triggers.

**Fix:** Add DFS-based cycle detection at artifact creation time.

### 36.2 MEDIUM: Orphan Node Detection Missing

**Location:** `liminallm/service/workflow.py:418-425`

No reachability analysis from entrypoint. Orphan nodes silently ignored.

**Impact:** Configuration errors go unnoticed; dead code in workflows.

### 36.3 MEDIUM: Invalid Next Node References Not Validated

**Location:** `liminallm/service/workflow.py:1429-1435, 1485-1495`

Next node references not validated against node_map at creation time.

**Impact:** Invalid references cause silent runtime failures.

### 36.4 MEDIUM: Tool Output Directly Merged to State

**Location:** `liminallm/service/workflow.py:567, 602, 885`

Tool outputs merged directly into vars_scope without validation or whitelisting.

**Impact:** Malicious tools can pollute workflow state, overwrite critical variables.

### 36.5 MEDIUM: Missing Tool Input Size Limits

**Location:** `liminallm/api/routes.py:1935-1975`

No limit on total size of inputs passed to tools via API.

**Impact:** Memory exhaustion via large tool inputs.

---

## 37. Data Integrity Issues (6th Pass)

### 37.1 CRITICAL: Non-Atomic Dataset File Writes

**Location:** `liminallm/service/training.py:307-309`

Dataset files written directly without temp-then-rename atomic pattern.

**Impact:** Partial dataset on crash leads to invalid training jobs.

### 37.2 CRITICAL: config_patch_audit Table Does Not Exist

**Location:** `liminallm/storage/postgres.py:1231`

Code references `config_patch_audit` table but schema only defines `config_patch`.

**Impact:** User deletion fails at runtime; orphaned records.

### 37.3 HIGH: File I/O Outside Database Transaction

**Location:** `liminallm/storage/postgres.py:1741-1769`

Artifact file persisted BEFORE transaction starts. If INSERT fails, orphaned file remains.

**Impact:** Orphaned files accumulate on constraint violations.

### 37.4 HIGH: Manual Cascade Instead of DB Constraints

**Location:** `liminallm/storage/postgres.py:1151-1280`

15+ manual DELETE statements for user deletion, not using ON DELETE CASCADE.

**Impact:** Partial deletion leaves orphaned records if process crashes mid-operation.

### 37.5 HIGH: Non-Atomic Adapter Parameter Writes

**Location:** `liminallm/service/training.py:328-339`

Multiple files (params.json, metadata.json) written sequentially without atomicity.

**Impact:** Inconsistent adapter state if crash between writes.

### 37.6 MEDIUM: Message Sequence Using COUNT(*) Instead of MAX

**Location:** `liminallm/storage/postgres.py:1500-1504`

Uses `COUNT(*)` to calculate next sequence number. Inefficient and assumes no gaps.

**Impact:** Sequence collisions possible in concurrent scenarios.

### 37.7 MEDIUM: Session Cache Not Atomic with DB

**Location:** `liminallm/storage/postgres.py:1354-1363`

Cache update proceeds even if DB update fails (exception only logged, not raised).

**Impact:** Session cache diverges from database state.

---

## 38. DoS/Resource Exhaustion (6th Pass)

### 38.1 CRITICAL: No Per-User WebSocket Connection Limits

**Location:** `liminallm/api/routes.py:2853`

No limit on concurrent WebSocket connections per user.

**Impact:** Single user can open 1000+ connections, exhausting server resources.

### 38.2 CRITICAL: No Disk Quota Enforcement

**Location:** `liminallm/api/routes.py:2347-2436`

While individual files limited to 10MB, no per-user or per-tenant storage quota.

**Impact:** Users can upload unlimited files, exhausting disk space.

### 38.3 HIGH: Recursive File Ingestion Without Limits

**Location:** `liminallm/service/rag.py:431-506`

`ingest_path()` with recursive=True has no file count or depth limits.

**Impact:** Directory with 100K files causes memory exhaustion.

### 38.4 HIGH: Unbounded List Operations

**Location:** `liminallm/api/routes.py:1463`

`list_messages()` called without limit parameter, retrieves all messages.

**Impact:** Single conversation with 100K+ messages consumes unbounded memory.

### 38.5 HIGH: String Concatenation in WebSocket Streaming

**Location:** `liminallm/api/routes.py:2957`

Token accumulation uses `+=` string concatenation in loop.

**Impact:** Large responses cause O(n²) memory allocation pattern.

### 38.6 MEDIUM: PostgreSQL Connection Pool Too Small

**Location:** `liminallm/storage/postgres.py:63-68`

Connection pool max_size=10. With 100+ concurrent users, 90% block.

**Impact:** Database connection starvation under load.

### 38.7 MEDIUM: Email Validation ReDoS

**Location:** `liminallm/api/schemas.py:50-53`

Email regex with nested quantifiers vulnerable to catastrophic backtracking.

**Impact:** Malformed emails cause CPU exhaustion.

### 38.8 MEDIUM: Active Requests Registry Memory Leak

**Location:** `liminallm/api/routes.py:112-119`

`_active_requests` dict entries not cleaned up if WebSocket disconnects abnormally.

**Impact:** Memory leak from abandoned WebSocket connections.

---

## 39. Type Coercion Vulnerabilities (6th Pass)

### 39.1 CRITICAL: JSON Deserialization Without Error Handling

**Location:** `liminallm/service/model_backend.py:1023`, `liminallm/storage/memory.py:1565`

`json.loads()` called without try-except.

**Impact:** Invalid JSON crashes adapter loading or state restoration.

### 39.2 HIGH: datetime.fromisoformat Without Error Handling

**Location:** `liminallm/storage/redis_cache.py:185, 392`, `liminallm/storage/postgres.py:2137,2142,2147`

ISO datetime parsing without try-except.

**Impact:** Corrupted datetime strings crash deserialization.

### 39.3 HIGH: float()/int() Without Error Handling

**Location:** Multiple files - `router.py:177,399`, `model_backend.py:674,1281`, `postgres.py:409,440,464`, `training.py:318,324`

Numeric coercion without validation.

**Impact:** Non-numeric values crash request processing.

### 39.4 MEDIUM: No NaN/Infinity Validation

**Location:** `liminallm/service/training.py`, `liminallm/service/router.py`

Float values (embeddings, scores) never validated for NaN/Infinity.

**Impact:** Invalid JSON serialization when NaN/Infinity encountered.

---

## 40. CSRF/Session Security (6th Pass)

### 40.1 CRITICAL: CSRF Tokens Listed But Not Validated

**Location:** `liminallm/app.py:88` and all POST/PATCH/DELETE routes

CORS headers advertise `X-CSRF-Token` support but tokens are never generated or validated.

**Impact:** All state-changing endpoints vulnerable to CSRF attacks.

### 40.2 CRITICAL: Tokens Exposed in Email URLs

**Location:** `liminallm/service/email.py:108, 160`

Password reset and email verification tokens passed in URL query parameters.

**Impact:** Token exposure via browser history, Referer headers, server logs, proxies.

### 40.3 CRITICAL: Session Rotation Not Implemented

**Location:** `liminallm/service/auth.py` (missing functionality)

SPEC §12.1 requires session ID rotation every 24h of activity. Not implemented.

**Impact:** Session hijacking window remains open indefinitely.

### 40.4 HIGH: Access Tokens Not Revoked on Logout

**Location:** `liminallm/api/routes.py:1330-1331`

Only cookies deleted on logout; JWT access tokens remain valid for 30 minutes.

**Impact:** Compromised tokens usable after logout.

### 40.5 HIGH: No Origin/Referer Header Validation

**Location:** `liminallm/app.py` middleware

No validation that state-changing requests originate from allowed origins.

**Impact:** CSRF defense-in-depth compromised.

---

## 41. Frontend XSS Vulnerabilities (6th Pass)

### 41.1 CRITICAL: Dynamic onclick Handler

**Location:** `frontend/chat.js:192`

Inline event handler in innerHTML template. Anti-pattern that violates CSP.

**Fix:** Use addEventListener instead of inline handlers.

### 41.2 HIGH: innerHTML Injection in Patch Status

**Location:** `frontend/admin.js:171-174`

Patch status values from API inserted into innerHTML without escaping.

**Impact:** XSS via malicious patch status values.

### 41.3 HIGH: Unescaped JSON in Data Attributes

**Location:** `frontend/chat.js:1169-1176`

Citation data in data-citation attribute relies on fragile escaping.

**Impact:** XSS if escaping fails on malformed API data.

### 41.4 MEDIUM: Unvalidated URL Parameters

**Location:** `frontend/chat.js:648-651, 879-880, 972-973`

OAuth provider parameter used in API endpoint without validation.

**Impact:** Path traversal via `provider=../../../`.

### 41.5 MEDIUM: Sensitive Token Storage in sessionStorage

**Location:** `frontend/chat.js:13-19`, `frontend/admin.js:17-34`

Access tokens stored in sessionStorage accessible to any XSS.

**Impact:** Token theft if any XSS vulnerability exists.

---

## 42. Error Handling Security (6th Pass)

### 42.1 CRITICAL: Raw Exception Exposure in Responses

**Location:** `liminallm/api/routes.py:186, 2754`

`str(exc)` passed directly to client error responses.

**Impact:** Internal exception details (paths, schema, queries) exposed to attackers.

### 42.2 HIGH: Email Enumeration Vulnerability

**Location:** `liminallm/storage/postgres.py:865`, `memory.py:236`

Signup returns specific "email already exists" error on duplicate.

**Impact:** Attackers can enumerate valid email addresses.

### 42.3 HIGH: Full Stack Trace Logging

**Location:** `liminallm/api/error_handling.py:88`

`logger.exception(..., exc_info=exc)` logs full stack traces.

**Impact:** Sensitive data in variables exposed to log aggregation systems.

### 42.4 MEDIUM: Bare Exceptions Silently Swallowed

**Location:** `liminallm/api/routes.py:2935-2936, 3119-3120, 3144-3145, 2750-2751`

Multiple bare `except: pass` blocks without logging.

**Impact:** Failures go unnoticed; debugging impossible.

### 42.5 MEDIUM: Database Schema in Error Responses

**Location:** `liminallm/storage/postgres.py:2224-2228`

Database column names exposed in NOT NULL constraint violation errors.

**Impact:** Schema information disclosure aids SQL injection attempts.

---

## 43. Adapter/LoRA Security (7th Pass)

### 43.1 CRITICAL: Float Weight Injection via JSON Deserialization

**Location:** `liminallm/service/model_backend.py:1023`

Adapter weight files are deserialized using `json.loads()` without validation. JSON allows serialization of IEEE 754 special floating-point values (infinity, -infinity, NaN) via scientific notation.

**Impact:** Numerical instability, model corruption, NaN propagation through inference, DoS via malformed weights.

### 43.2 HIGH: Insufficient Gate Weight Bounds Checking

**Location:** `liminallm/service/model_backend.py:666-674`

Gate weights extracted and converted to float before bounds clamping. Float conversion at line 674 happens before clamping at line 1283.

**Impact:** Infinity/NaN in weight field bypass the clamping check; malformed weights passed to API backends.

### 43.3 HIGH: Missing File Size Limits on Weight Uploads

**Location:** `liminallm/service/model_backend.py:999`

No validation of file size before reading entire weight file into memory: `payload = params_path.read_bytes()`.

**Impact:** DoS via large adapter files causing OOM; resource exhaustion affecting other users.

### 43.4 HIGH: No Validation of Remote Model IDs

**Location:** `liminallm/service/model_backend.py:633, 659-664`

Remote model IDs passed through without validation to external providers.

**Impact:** Model injection, parameter pollution, access control bypass via crafted model IDs.

### 43.5 MEDIUM: Adapter Cache Poisoning (Cross-User Access)

**Location:** `liminallm/service/model_backend.py:825, 996-998, 1029`

Adapter cache keyed only by `adapter_id`, no user_id or tenant_id in cache key.

**Impact:** User B could load User A's private adapter weights if adapter IDs collide.

### 43.6 MEDIUM: Prompt Injection via adapter prompt_instructions

**Location:** `liminallm/service/model_backend.py:738-762, 764-788`

Adapter prompt instructions extracted and injected into system message without sanitization or length limits.

**Impact:** LLM jailbreak, model behavior hijacking, context window pollution.

### 43.7 MEDIUM: Missing Input Validation on Adapter Schema Fields

**Location:** `liminallm/api/schemas.py`, `model_backend.py`

No regex/pattern validation on remote_model_id, remote_adapter_id, adapter_id. Weight array dimensions loaded without shape validation.

**Impact:** Malformed data confuses downstream consumers; injection via provider APIs.

### 43.8 LOW: Cache Mtime Check Bypass

**Location:** `liminallm/service/model_backend.py:995-998`

Cache validation relies on file mtime which can be manipulated or affected by clock skew.

**Impact:** Stale or modified weights served from cache.

---

## 44. Multi-Tenant Isolation (7th Pass)

### 44.1 MEDIUM: X-Tenant-ID Header Parameter Accepted

**Location:** `liminallm/api/routes.py:339-370`

`X-Tenant-ID` header accepted as a "hint" and passed to `authenticate()`. Violates CLAUDE.md: "Always derive tenant_id from the authenticated JWT token, never from request parameters."

**Impact:** Information disclosure (tenant enumeration), confusion in audit logs.

### 44.2 MEDIUM: OAuth Callback tenant_id Parameter

**Location:** `liminallm/api/routes.py:658`

OAuth callback endpoint accepts `tenant_id` as a query parameter from attacker-controlled OAuth callback URL.

**Impact:** Potential tenant assignment manipulation during OAuth flow.

### 44.3 MEDIUM: Missing Tenant Isolation in Rate Limiting Cache Keys

**Location:** `liminallm/storage/redis_cache.py:58`

Rate limit keys constructed as `rate:{key}:{now_bucket}` without explicit tenant separation.

**Impact:** Cross-tenant rate limit bucket collisions possible; tenant-wide rate limits not feasible.

### 44.4 MEDIUM: Workflow State Cache Keys Lack Tenant Isolation

**Location:** `liminallm/storage/redis_cache.py:99-113`

Workflow state cache keys use only `state_key` without tenant/user isolation.

**Impact:** Cross-tenant workflow state collision if state_key is predictable.

### 44.5 LOW: Router Cache Keys Not Tenant-Prefixed

**Location:** `liminallm/storage/redis_cache.py:82, 95`

Cache keys include `user_id` but not `tenant_id`.

**Impact:** UUID collision unlikely but violates isolation principle.

---

## 45. Embedding/Vector Security (7th Pass)

### 45.1 CRITICAL: No NaN/Infinity Validation in Embeddings

**Location:** `liminallm/service/embeddings.py:29-36`

`cosine_similarity()` does not check for NaN or Infinity values before computing similarity.

**Impact:** Poisoned embeddings corrupt similarity scores, centroid calculations, vector search results.

### 45.2 HIGH: Missing Embedding Dimension Validation

**Location:** `liminallm/service/embeddings.py:39-47`

`ensure_embedding_dim()` silently pads or truncates embeddings without validating input dimensions.

**Impact:** Malformed embeddings cause inconsistent vector space geometry and clustering errors.

### 45.3 HIGH: Centroid Poisoning - No Validation on Cluster Centroids

**Location:** `liminallm/service/clustering.py:88-99`

Cluster centroids computed and stored without validating for NaN/Infinity. K-means update rule doesn't normalize results.

**Impact:** Malicious centroids break similarity calculations, skew cluster assignments.

### 45.4 HIGH: Embedding Injection in Preference Events

**Location:** `liminallm/storage/postgres.py:313-315`, `memory.py:648-650`

User-provided embeddings in preference events accepted without validation.

**Impact:** Training data poisoning, centroid corruption, clustering manipulation.

### 45.5 HIGH: No Bounds Checking on Cosine Similarity Scores

**Location:** `liminallm/service/router.py:307-319`

Similarity scores used directly for weight assignment without NaN validation.

**Impact:** NaN similarity scores propagate through adapter routing, undefined behavior.

### 45.6 MEDIUM: Centroid Exposure in Adapter Schema

**Location:** `liminallm/service/router.py:422-426`, `clustering.py:209`

Adapter centroids stored in artifact schema and exposed to users via API responses.

**Impact:** Embeddings leaked; attackers can craft malicious adapters with poisoned centroids.

### 45.7 MEDIUM: Chunk Search with Unvalidated Embeddings

**Location:** `liminallm/storage/common.py:86-91`, `memory.py:1487-1496`

Search functions don't validate embedding dimensions before computing similarity.

**Impact:** Dimension mismatch silently handled, incorrect search results.

### 45.8 MEDIUM: No Embedding Model Validation for pgvector Search

**Location:** `liminallm/storage/postgres.py:2472-2474`

Embedding model ID filtering is optional and happens after chunks selected.

**Impact:** Chunks from different embedding models mixed in search results.

### 45.9 MEDIUM: Unvalidated Centroid in Workflow Context Embedding

**Location:** `liminallm/service/workflow.py:1370, 1375`

Cluster centroids used directly in workflow vector alignment without validation.

**Impact:** Poisoned centroids corrupt workflow routing decisions.

### 45.10 MEDIUM: Missing Normalization in Centroid Update

**Location:** `liminallm/service/clustering.py:40-42, 88-95`

Mini-batch k-means centroids updated incrementally without normalization.

**Impact:** Centroids accumulate magnitude errors, incorrect cluster assignments.

---

## 46. Input Validation Edge Cases (7th Pass)

### 46.1 HIGH: Nested JSON Validation - Unbounded Depth

**Location:** `liminallm/api/schemas.py:184-185, 242, 258-265, 654, 662`

Multiple fields accept unrestricted `dict` with no nested depth or size validation: `default_style`, `flags`, `inputs`, `outputs`.

**Impact:** Memory exhaustion via deeply nested structures; JSON deserialization bomb attacks.

### 46.2 HIGH: Array Length Limits - Unbounded Arrays

**Location:** `liminallm/api/schemas.py:260-265`

`ChatResponse` list fields (`adapters`, `adapter_gates`, `context_snippets`, `routing_trace`, `workflow_trace`) have no `max_items` validation.

**Impact:** DoS via large arrays; memory exhaustion.

### 46.3 HIGH: String Length Limits - Missing maxLength

**Location:** `liminallm/api/schemas.py:281-282, 493, 655-657`

Multiple required/optional strings without `max_length`: `ArtifactRequest.name`, `type`, `explicit_signal`, `conversation_id`, `context_id`, `user_message`.

**Impact:** Memory exhaustion; buffer overflow in storage; unbounded query strings.

### 46.4 MEDIUM: Unicode Edge Cases - No Normalization

**Location:** `liminallm/api/schemas.py:59`

Email validation `strip().lower()` doesn't account for zero-width characters, RTL override, combining diacritics, or NFKC normalization.

**Impact:** IDN homoglyph attacks; Unicode normalization bypasses; spoofed identities.

### 46.5 MEDIUM: Numeric Bounds - Integer Overflow Potential

**Location:** `liminallm/api/routes.py:741, 1648-1650`

Integer query parameters have only `ge=1` bounds, no upper limits. Float fields don't prevent NaN/Infinity.

**Impact:** Integer overflow in offset/limit calculations; NaN propagation.

### 46.6 MEDIUM: Empty String vs Null Handling

**Location:** `liminallm/api/schemas.py:283`, `storage/postgres.py:1678`

Inconsistent empty string vs None handling between similar fields and between request/response.

**Impact:** Type confusion attacks; logic bypasses.

### 46.7 MEDIUM: Special Characters in Identifiers

**Location:** `liminallm/api/routes.py:1645-1646, 465`

`type`, `kind` query parameters and `fs_path` lack pattern validation.

**Impact:** NoSQL injection through JSONB operators; path traversal.

### 46.8 MEDIUM: Query Parameter Injection

**Location:** `liminallm/api/routes.py`

No protection against duplicate query parameters (FastAPI merges as lists). OAuth provider parameter not validated against allowed list.

**Impact:** Type confusion; parameter pollution; OAuth provider spoofing.

---

## 47. Content Redaction Security (7th Pass)

### 47.1 HIGH: Error Message Content Exposure

**Location:** `liminallm/service/workflow.py:600, 1002, 1024, 1470-1474`

Exception messages containing sensitive internal details exposed in workflow_trace and error responses.

**Impact:** Stack traces, SQL errors, API secrets, internal paths exposed to attackers.

### 47.2 HIGH: Citation Content Exposure

**Location:** `liminallm/service/workflow.py:1708, 1750, 1769`

Citation chunks extracted directly without content filtering, returned in API responses and traces.

**Impact:** Sensitive information from knowledge bases exposed; confidential documents leaked.

### 47.3 HIGH: Message Content Not Filtered

**Location:** `liminallm/api/routes.py:2458-2475`

Messages returned with full content and content_struct without any filtering or redaction.

**Impact:** User messages with sensitive info not filtered; tool outputs not sanitized.

### 47.4 MEDIUM: Tool Output Sanitization Incomplete

**Location:** `liminallm/service/workflow.py:1504-1507`

Tool outputs passed through without validation or filtering for sensitive data patterns.

**Impact:** Tools can leak credentials, API keys, or PII.

### 47.5 MEDIUM: Content Struct Meta Field Bypass

**Location:** `liminallm/content_struct.py:52, 61-95, 99-110`

`content_struct` normalization accepts arbitrary data in "meta" field without validation.

**Impact:** Malicious payloads injected; data exfiltration; security control bypass.

### 47.6 MEDIUM: Routing Trace & Debug Info Exposure

**Location:** `liminallm/api/routes.py:1441-1442`, `router.py:89-100`

Routing traces with rule evaluation details returned to clients.

**Impact:** Adapter selection logic, backend capabilities exposed; aids targeted attacks.

### 47.7 MEDIUM: PII Redaction Limited to Logging

**Location:** `liminallm/logging.py:37-48`

PII redaction only occurs in log entries, not in API responses or stored data.

**Impact:** PII exposed in API responses via message content, citations, tool outputs.

### 47.8 MEDIUM: Inconsistent Sanitization

**Location:** `liminallm/api/routes.py:2273-2327`

Admin settings endpoint has sanitization logic not applied to chat messages, tool outputs, workflow traces, citations.

**Impact:** No consistent sensitive field detection across API.

---

## 48. Deadlock/Timeout Patterns (7th Pass)

### 48.1 CRITICAL: Unprotected asyncio.Lock Ordering

**Location:** `liminallm/service/runtime.py:159-161`

Two separate asyncio.Lock instances without documented lock ordering discipline.

**Impact:** Complete service hang under concurrent requests if locks acquired in different orders.

### 48.2 CRITICAL: No Connection Pool Timeout Configuration

**Location:** `liminallm/storage/postgres.py:63-68`

ConnectionPool created with only 10 max connections and NO timeout parameters, no statement_timeout.

**Impact:** Long-running queries block new connections; pool exhaustion = service unresponsive.

### 48.3 CRITICAL: Redis Operations Without Explicit Timeouts

**Location:** `liminallm/storage/redis_cache.py` (multiple: 34, 37, 66, 162)

All Redis operations rely on connection-level timeout (if set), not operation-level timeouts.

**Impact:** Single slow Redis command stalls auth/rate-limiting/session system.

### 48.4 CRITICAL: No Timeout on Database Connection Acquisition

**Location:** `liminallm/storage/postgres.py:118`

`pool.connection()` can block indefinitely if no connections available.

**Impact:** Any of 100+ database operations using `with self._connect()` can deadlock.

### 48.5 HIGH: SyncRedisCache Race Condition in pop_oauth_state

**Location:** `liminallm/storage/redis_cache.py:378-380`

GET then DELETE is NOT atomic. Between operations, another coroutine can consume same OAuth state.

**Impact:** OAuth replay attacks; same state token could be used multiple times.

### 48.6 HIGH: ThreadPoolExecutor Resource Exhaustion

**Location:** `liminallm/service/workflow.py:122`

Fixed pool of 4 workers is too small; no queue monitoring or adaptive scaling.

**Impact:** Tools execute serially under load; requests queue indefinitely.

### 48.7 HIGH: Async/Sync Context Mixing in Training Worker

**Location:** `liminallm/service/training_worker.py:144`

`asyncio.to_thread()` offloads training but no timeout on the call.

**Impact:** If training takes too long, event loop can't process other tasks.

### 48.8 HIGH: Unbounded Parallel Node Execution Without Timeout

**Location:** `liminallm/service/workflow.py:323`

`asyncio.gather()` without timeout; all parallel nodes must complete or hang.

**Impact:** One stalled parallel node stalls entire workflow.

### 48.9 MEDIUM: Lock Contention in Idempotency/Rate-Limit Caching

**Location:** `liminallm/service/runtime.py:214-221, 278-289`

Both functions hold locks during dictionary operations; under high concurrency, this serializes access.

**Impact:** Rate limit and idempotency checks become bottleneck.

### 48.10 MEDIUM: No Timeout on Training Job Retry Loop

**Location:** `liminallm/service/training_worker.py:141-195`

Total retry duration unbounded. Each attempt can take 5+ minutes with retries=3.

**Impact:** Training jobs block queue indefinitely.

### 48.11 MEDIUM: OAuth State Cleanup Race Condition

**Location:** `liminallm/service/auth.py:138-183`

`cleanup_expired_states()` called without any lock while other methods read/write `_oauth_states` concurrently.

**Impact:** Data structure corruption, OAuth state loss.

---

## 49. API Versioning/Compatibility (7th Pass)

### 49.1 CRITICAL: Database Migration Safety - Breaking Column Rename

**Location:** `liminallm/sql/003_preferences.sql:53-61`

Column rename `adapter_artifact_id` to `adapter_id` without backward compatibility layer.

**Impact:** Old code referencing `adapter_artifact_id` breaks immediately; no rollback strategy.

### 49.2 HIGH: Missing /v1/ Prefix Enforcement

**Location:** `liminallm/app.py:174, 263`

Infrastructure endpoints bypass API versioning: `/healthz`, `/metrics` return raw dict/text, not Envelope.

**Impact:** Clients cannot consistently parse responses using same envelope format.

### 49.3 HIGH: Breaking Changes Protection - Schema Migration Handling

**Location:** `liminallm/storage/postgres.py:1701-1707`

Silent schema deserialization failure with data loss - returns empty `{}` on parse failure.

**Impact:** Corrupt or incompatible artifact schemas silently replaced with empty dict.

### 49.4 HIGH: Artifact Schema Versioning - Unsafe Old Schema Loading

**Location:** `liminallm/storage/postgres.py:1704, 1873, 1942`

Old artifact schemas loaded via `json.loads()` without schema validation or migration logic.

**Impact:** Old schemas may fail to load if internal structure changed.

### 49.5 MEDIUM: No Deprecation Headers

**Location:** `liminallm/api/routes.py` (all endpoints)

No deprecation headers (`Deprecation`, `Sunset`, `Link: rel="deprecation"`) sent to clients.

**Impact:** Old clients have no signal to migrate to new API versions.

### 49.6 MEDIUM: Response Format Stability - Inconsistent Envelope

**Location:** `liminallm/api/routes.py`

59 endpoints declare `response_model=Envelope` but only 31 explicitly return `Envelope(...)`.

**Impact:** Inconsistent response format across endpoints.

### 49.7 MEDIUM: No Accept-Version Header Support

**Location:** `liminallm/api/routes.py`

No `Accept-Version` header parsing or `API-Version` response header.

**Impact:** Clients cannot negotiate API version for backward compatibility.

### 49.8 MEDIUM: Response Envelope Consistency - Missing request_id

**Location:** `liminallm/api/routes.py:1973, 2051, 2434`

Only some endpoints explicitly pass `request_id` to Envelope; most rely on default UUID.

**Impact:** Idempotency keys and correlation IDs not properly propagated.

---

## Summary by Severity (Updated 7th Pass)

### Critical (83 Issues)

| # | Issue | Location |
|---|-------|----------|
| 1-62 | (Previous passes - see above) | Various |
| 63 | Float Weight Injection via JSON Deserialization | model_backend.py:1023 |
| 64 | No NaN/Infinity Validation in Embeddings | embeddings.py:29-36 |
| 65 | Unprotected asyncio.Lock Ordering | runtime.py:159-161 |
| 66 | No Connection Pool Timeout Configuration | postgres.py:63-68 |
| 67 | Redis Operations Without Explicit Timeouts | redis_cache.py (multiple) |
| 68 | No Timeout on Database Connection Acquisition | postgres.py:118 |
| 69 | Database Migration Safety - Breaking Column Rename | sql/003_preferences.sql:53-61 |
| 70 | (Numbered placeholder for prior issues) | Various |

### High Priority (90 Issues)

| # | Issue | Location |
|---|-------|----------|
| 1-52 | (Previous passes - see above) | Various |
| 53 | Insufficient Gate Weight Bounds Checking | model_backend.py:674 |
| 54 | Missing File Size Limits on Weight Uploads | model_backend.py:999 |
| 55 | No Validation of Remote Model IDs | model_backend.py:633, 659 |
| 56 | Missing Embedding Dimension Validation | embeddings.py:39-47 |
| 57 | Centroid Poisoning - No Validation | clustering.py:88-99 |
| 58 | Embedding Injection in Preference Events | postgres.py:313-315 |
| 59 | No Bounds Checking on Cosine Similarity | router.py:307-319 |
| 60 | Nested JSON Validation - Unbounded Depth | schemas.py:184-185 |
| 61 | Array Length Limits - Unbounded Arrays | schemas.py:260-265 |
| 62 | String Length Limits - Missing maxLength | schemas.py:281-282 |
| 63 | Error Message Content Exposure | workflow.py:600, 1002, 1024 |
| 64 | Citation Content Exposure | workflow.py:1708, 1750 |
| 65 | Message Content Not Filtered | routes.py:2458-2475 |
| 66 | SyncRedisCache Race Condition | redis_cache.py:378-380 |
| 67 | ThreadPoolExecutor Resource Exhaustion | workflow.py:122 |
| 68 | Async/Sync Context Mixing in Training | training_worker.py:144 |
| 69 | Unbounded Parallel Node Execution | workflow.py:323 |
| 70 | Missing /v1/ Prefix Enforcement | app.py:174, 263 |
| 71 | Schema Migration Handling | postgres.py:1701-1707 |
| 72 | Unsafe Old Schema Loading | postgres.py:1704, 1873 |

### Medium Priority (75 Issues)

| # | Issue | Location |
|---|-------|----------|
| 1-33 | (Previous passes - see above) | Various |
| 34 | Adapter Cache Poisoning | model_backend.py:825, 1029 |
| 35 | Prompt Injection via prompt_instructions | model_backend.py:738-788 |
| 36 | Missing Adapter Schema Field Validation | schemas.py, model_backend.py |
| 37 | X-Tenant-ID Header Parameter | routes.py:339-370 |
| 38 | OAuth Callback tenant_id Parameter | routes.py:658 |
| 39 | Rate Limiting Cache Keys Tenant Isolation | redis_cache.py:58 |
| 40 | Workflow State Cache Keys Tenant Isolation | redis_cache.py:99-113 |
| 41 | Centroid Exposure in Adapter Schema | router.py:422-426 |
| 42 | Chunk Search Unvalidated Embeddings | common.py:86-91 |
| 43 | No Embedding Model Validation for pgvector | postgres.py:2472-2474 |
| 44 | Unvalidated Centroid in Workflow | workflow.py:1370, 1375 |
| 45 | Missing Normalization in Centroid Update | clustering.py:92-95 |
| 46 | Unicode Edge Cases - No Normalization | schemas.py:59 |
| 47 | Numeric Bounds - Integer Overflow | routes.py:741, 1648 |
| 48 | Empty String vs Null Handling | schemas.py:283, postgres.py:1678 |
| 49 | Special Characters in Identifiers | routes.py:1645-1646 |
| 50 | Query Parameter Injection | routes.py |
| 51 | Tool Output Sanitization Incomplete | workflow.py:1504-1507 |
| 52 | Content Struct Meta Field Bypass | content_struct.py:52, 99-110 |
| 53 | Routing Trace & Debug Info Exposure | routes.py:1441, router.py:89-100 |
| 54 | PII Redaction Limited to Logging | logging.py:37-48 |
| 55 | Inconsistent Sanitization | routes.py:2273-2327 |
| 56 | Lock Contention in Caching | runtime.py:214-221, 278-289 |
| 57 | No Timeout on Training Retry Loop | training_worker.py:141-195 |
| 58 | OAuth State Cleanup Race | auth.py:138-183 |
| 59 | No Deprecation Headers | routes.py |
| 60 | Response Format Inconsistent Envelope | routes.py |
| 61 | No Accept-Version Header Support | routes.py |
| 62 | Missing request_id Propagation | routes.py:1973, 2051, 2434 |

---

## 7th Pass Recommendations

### Adapter/LoRA Security Actions

1. Validate all float values in loaded weights are finite (no inf/nan)
2. Implement file size limits on weight uploads
3. Validate remote model/adapter IDs against whitelist patterns
4. Use composite cache key (adapter_id + user_id + tenant_id)
5. Sanitize prompt_instructions for length and content

### Multi-Tenant Isolation Actions

1. Remove X-Tenant-ID header; derive only from JWT
2. Remove tenant_id parameter from OAuth callback
3. Add tenant_id prefix to all cache keys

### Embedding/Vector Security Actions

1. Check for NaN/Infinity in all vector operations
2. Add strict embedding dimension validation
3. Normalize centroids after each update
4. Validate all embeddings before similarity calculations

### Input Validation Actions

1. Add max_items to all list fields
2. Add max_length to all string fields
3. Implement Unicode normalization (NFKC)
4. Add upper bounds to all integer query parameters

### Content Redaction Actions

1. Filter error messages before returning to clients
2. Implement content filtering for citations and tool outputs
3. Validate/restrict content_struct meta field
4. Apply consistent sanitization across all endpoints

### Deadlock/Timeout Actions

1. Enforce strict lock ordering; document globally
2. Add connection pool and statement timeouts
3. Add Redis client socket timeout
4. Increase thread pool workers and add monitoring
5. Add timeout to asyncio.gather() calls

### API Versioning Actions

1. Wrap infrastructure endpoints in Envelope format
2. Add deprecation headers for deprecated endpoints
3. Implement schema validation and migration for artifacts
4. Standardize request_id propagation

---

## 8th Pass: Comprehensive Security Deep Dive (2025-12-03)

This pass focused on 8 specialized security audit areas:
- RBAC and permission checking
- Logging and audit trail compliance
- Data serialization security
- HTTP security headers
- Business logic vulnerabilities
- Frontend security (React/TypeScript)
- External API integrations
- Cryptographic implementations

---

## 50. RBAC and Permission Security

### 50.1 CRITICAL: MFA Request Endpoint Missing Authentication
**Location:** `liminallm/api/routes.py:997-1012`

The `POST /auth/mfa/request` endpoint does NOT require authentication and accepts an arbitrary `session_id` from the request body without validating ownership.

```python
@router.post("/auth/mfa/request", response_model=Envelope, tags=["auth"])
async def request_mfa(body: MFARequest):
    auth_ctx = await runtime.auth.resolve_session(
        body.session_id, allow_pending_mfa=True  # No ownership validation
    )
```

**Impact:** Attacker can enumerate session IDs and trigger MFA challenges for arbitrary other users' sessions.

### 50.2 CRITICAL: MFA Verify Endpoint Missing Session Ownership Validation
**Location:** `liminallm/api/routes.py:1015-1050`

The `POST /auth/mfa/verify` endpoint does NOT require authentication. An unauthenticated attacker can pass ANY session_id and attempt to verify MFA for that session without being the session owner.

**Impact:** **Account Takeover** - If a target user has MFA disabled, an attacker can receive valid tokens for the victim's session by enumerating session IDs.

### 50.3 CRITICAL: Admin User Role Modification Missing Tenant Isolation
**Location:** `liminallm/api/routes.py:804-820`

The `POST /admin/users/{user_id}/role` endpoint does NOT validate that the target `user_id` belongs to the admin's tenant.

**Impact:** Admins from Tenant A can promote/demote users from Tenant B, violating multi-tenant isolation.

### 50.4 CRITICAL: Admin User Deletion Missing Tenant Isolation
**Location:** `liminallm/api/routes.py:823-837`

The `DELETE /admin/users/{user_id}` endpoint does NOT validate tenant isolation.

**Impact:** Admins from one tenant can delete users from other tenants.

### 50.5 HIGH: Chat Request Cancellation Missing Ownership Validation
**Location:** `liminallm/api/routes.py:1472-1510`

The `POST /chat/cancel` endpoint accepts a `request_id` but does NOT validate that the request belongs to the authenticated user.

**Impact:** Attackers can cancel other users' active chat requests, causing denial of service.

### 50.6 MEDIUM: Inconsistent Tenant Validation Across Admin Endpoints
**Location:** Multiple admin endpoints in `liminallm/api/routes.py`

- ✅ `/admin/users` (GET, line 759) - VALIDATES tenant
- ✅ `/admin/users` (POST, line 781) - VALIDATES tenant
- ❌ `/admin/users/{user_id}/role` (POST, line 804) - MISSING validation
- ❌ `/admin/users/{user_id}` (DELETE, line 823) - MISSING validation

---

## 51. Audit Logging and Compliance Gaps

### 51.1 CRITICAL: Missing Audit Logging for User Signup
**Location:** `liminallm/api/routes.py:525-570`

User signup endpoint creates new accounts with **zero logging**. No record of success/failure, email created, timestamps, or tenant assignments.

**Compliance Impact:** FAILS GDPR requirement for user access logs and SOC2 accountability.

### 51.2 CRITICAL: Missing Audit Logging for Admin User Creation
**Location:** `liminallm/api/routes.py:771-801`

Privileged operation to create users has no logging of which admin created the user, user details, or timestamps.

**Compliance Impact:** CRITICAL for SOC2 Type II - no audit trail for privileged operations.

### 51.3 CRITICAL: Missing Audit Logging for User Deletion
**Location:** `liminallm/api/routes.py:824-837`

User deletion operations are completely unlogged.

**Compliance Impact:** GDPR & SOC2 require audit trail of data deletion.

### 51.4 CRITICAL: Missing Audit Logging for Permission Changes
**Location:** `liminallm/api/routes.py:805-820`

Role/permission changes are unlogged - no record of who changed what role.

**Compliance Impact:** SOC2 requires detailed access control change logs.

### 51.5 CRITICAL: Failed Login Attempts Not Logged
**Location:** `liminallm/api/routes.py:597-598`

Failed authentication attempts have no logging - cannot detect brute force attacks.

### 51.6 CRITICAL: Password Change Events Not Logged
**Location:** `liminallm/api/routes.py:1277-1305`

Credential changes completely unlogged - no audit trail for GDPR/SOC2 compliance.

### 51.7 CRITICAL: Email Verification Events Not Logged
**Location:** `liminallm/service/auth.py:828-855`

Email verification success/failure completely unlogged.

### 51.8 CRITICAL: Password Reset Completion Not Logged
**Location:** `liminallm/service/auth.py:788-810`

Password recovery operations unlogged.

### 51.9 CRITICAL: Session Revocation Not Logged
**Location:** `liminallm/api/routes.py:1308-1332`

Session termination completely unlogged - cannot audit user access patterns.

### 51.10 HIGH: PII (Email Addresses) Being Logged
**Location:** `liminallm/service/email.py:65, 99, 103`

Email addresses logged directly despite PII redaction being configured.

### 51.11 HIGH: No Logging for Failed Password Verification
**Location:** `liminallm/service/auth.py:862-873`

Failed password attempts silently fail with no logging - cannot detect brute force.

### 51.12 HIGH: Token Refresh Failures Not Logged
**Location:** `liminallm/api/routes.py:714-715`

Failed token refresh completely unlogged - could indicate token theft.

### 51.13 HIGH: Insufficient OAuth Exchange Logging
**Location:** `liminallm/service/auth.py:411-415`

OAuth success logs missing critical audit information (user_id, action type).

### 51.14 HIGH: Insufficient MFA Failure Logging
**Location:** `liminallm/service/auth.py:754-763`

Only MFA lockout is logged, not individual failed attempts.

### 51.15 MEDIUM: Inconsistent Log Levels for Security Events
**Location:** Multiple service files

No standard for security event log levels - makes alerting difficult.

### 51.16 MEDIUM: Insufficient Correlation ID Usage
**Location:** `liminallm/service/auth.py`

Service-layer logging doesn't consistently expose correlation IDs.

### 51.17 MEDIUM: Missing Config Patch Decision Logging
**Location:** `liminallm/api/routes.py:2104-2247`

Admin approval/rejection decisions lack detailed logs.

---

## 52. HTTP Security Headers

### 52.1 MEDIUM: X-Frame-Options Configuration Mismatch
**Location:** `nginx.conf:40` vs `liminallm/app.py:116`

Nginx sets `SAMEORIGIN` while app sets `DENY` - nginx takes precedence, weakening protection.

### 52.2 MEDIUM: Missing Cache-Control on Sensitive Endpoints
**Location:** `liminallm/app.py:174-260, 263-320`

`/healthz` and `/metrics` endpoints don't set Cache-Control headers. Build info could be cached.

### 52.3 MEDIUM: Missing Cache-Control on API Endpoints
**Location:** `liminallm/app.py:113-136`

No Cache-Control header set globally for API responses - intermediate proxies could cache sensitive data.

### 52.4 MEDIUM: HSTS Only Enabled via Environment Flag
**Location:** `liminallm/app.py:123-131`

HSTS is disabled by default (must enable via ENABLE_HSTS) - relies on nginx fallback.

### 52.5 MEDIUM: /healthz and /metrics Not Rate Limited
**Location:** `nginx.conf:119-123`

Health and metrics endpoints have no rate limiting - DoS vector.

### 52.6 MEDIUM: FileResponse Not Setting Cache Headers
**Location:** `liminallm/app.py:150-171`

FileResponse for HTML pages doesn't set Cache-Control - admin.html could be cached.

### 52.7 LOW: Missing Server Header Suppression
**Location:** `nginx.conf`

Missing `server_tokens off;` - nginx version information disclosure.

### 52.8 LOW: CORS Missing Max-Age Header
**Location:** `liminallm/app.py:75-91`

No explicit `max_age` configured for CORS preflight caching.

### 52.9 LOW: CORS Missing Expose-Headers
**Location:** `liminallm/app.py:75-91`

X-Request-ID header not exposed to frontend JavaScript.

### 52.10 LOW: Incomplete CSP Directives
**Location:** `liminallm/app.py:132-135`

CSP doesn't explicitly restrict `object-src`, `media-src`, `worker-src`.

---

## 53. Business Logic Vulnerabilities

### 53.1 CRITICAL: MFA Bypass via Silent Database Failure
**Location:** `liminallm/storage/postgres.py:1354-1363`

If database UPDATE fails during `mark_session_verified()`, the exception is caught but in-memory cache is STILL marked as verified.

```python
except Exception as exc:
    self.logger.warning("mark_session_verified_failed", error=str(exc))
self._update_cached_session(session_id, mfa_verified=True)  # ALWAYS EXECUTES
```

**Impact:** Complete MFA bypass - database transient failure enables MFA-protected account compromise.

### 53.2 CRITICAL: Tenant Spoofing via Signup Endpoint
**Location:** `liminallm/api/routes.py:545-549`, `liminallm/service/auth.py:202-220`

Signup endpoint accepts `tenant_id` directly from request body, violating CLAUDE.md guideline.

**Impact:** Attacker can register in ANY tenant by specifying arbitrary tenant_id.

### 53.3 CRITICAL: Tenant Spoofing via OAuth Complete
**Location:** `liminallm/service/auth.py:484-495`

OAuth callback accepts `tenant_id` parameter - can create account in attacker-specified tenant.

### 53.4 CRITICAL: TOCTOU Race Condition in Session Revocation
**Location:** `liminallm/api/routes.py:1325-1329`

Between ownership check and revoke call, session could be modified by concurrent request.

### 53.5 CRITICAL: Session Verification State Machine Inconsistency
**Location:** `liminallm/service/auth.py:591-605`

If `revoke_refresh_token()` throws, the session may remain in database while refresh token is revoked.

### 53.6 HIGH: MFA Race Condition - Lockout Bypass
**Location:** `liminallm/service/auth.py:756-763`

Multiple concurrent requests can increment MFA attempt counter before lockout check executes.

### 53.7 HIGH: Unsigned MFA Verification Failure
**Location:** `liminallm/service/auth.py:553-556`

`_mark_session_verified()` might fail silently but tokens are still issued.

### 53.8 HIGH: Missing Negative Value Validation in Rate Limiting
**Location:** `liminallm/service/runtime.py:278-289`

No validation that `limit` is positive - negative limit bypasses rate limiting.

### 53.9 MEDIUM: Insufficient Exception Handling in Cache Operations
**Location:** `liminallm/service/auth.py:630-634`

Cache deletion failures silently ignored - ghost sessions may persist.

### 53.10 MEDIUM: Race Condition in Session Cache Eviction
**Location:** `liminallm/storage/postgres.py:76-94`

Session may expire between sort and eviction decision.

---

## 54. Frontend Security Issues

### 54.1 CRITICAL: Sensitive MFA Secret Displayed in DOM
**Location:** `frontend/chat.js:2471`

MFA secret displayed via textContent - visible in DevTools, readable by extensions.

### 54.2 CRITICAL: OTP Authentication URI Exposed Without Escaping
**Location:** `frontend/chat.js:2480`

`otpauth_uri` interpolated directly into innerHTML without HTML escaping.

### 54.3 CRITICAL: Newly Created Passwords Displayed in DOM
**Location:** `frontend/admin.js:429`

Auto-generated passwords displayed in UI - window of exposure.

### 54.4 HIGH: Sensitive Runtime Config Displayed in Plaintext
**Location:** `frontend/admin.js:257`

Entire runtime configuration displayed via JSON.stringify.

### 54.5 HIGH: Admin Objects Inspect Displays Sensitive Data
**Location:** `frontend/admin.js:532-536`

Full object details displayed in JSON - could contain sensitive data.

### 54.6 HIGH: Unescaped Error Messages Displayed
**Location:** Multiple files - `chat.js:3064, 3079`, `admin.js:259, 295, 329, 539`

Error messages from API displayed directly - could expose SQL errors, file paths.

### 54.7 HIGH: Missing CSRF Protection on Forms
**Location:** `admin.html`, `index.html`

No forms or API requests include CSRF tokens.

### 54.8 HIGH: Sensitive Data in Session Storage
**Location:** `frontend/chat.js:13-20, 67-76`

Access/refresh tokens stored in sessionStorage - vulnerable to XSS.

### 54.9 MEDIUM: Preference Data Exposes Internal Routing
**Location:** `frontend/chat.js:2050, 2052-2055`

Internal routing traces and workflow details displayed in preference panel.

### 54.10 MEDIUM: URL Parameters Containing Sensitive Tokens
**Location:** `frontend/chat.js:880, 973`

Password reset and email verification tokens passed in URL parameters.

### 54.11 MEDIUM: Insufficient Input Validation on Admin Inputs
**Location:** `frontend/admin.js:264-276`

Minimal validation of patch body structure.

### 54.12 MEDIUM: Potential IDOR in Artifact/Conversation Access
**Location:** `frontend/chat.js:1061-1072, 1950-1966`

Frontend accesses resources by ID without validating authorization.

### 54.13 MEDIUM: Draft Data Stored in Plain LocalStorage
**Location:** `frontend/chat.js:23-52`

Conversation drafts stored unencrypted in localStorage.

### 54.14 MEDIUM: Tenant ID From Session Storage (Not Derived From Token)
**Location:** `frontend/admin.js:36-42, 55, 116`

Tenant ID read from sessionStorage and sent to backend - violates CLAUDE.md.

---

## 55. External API Integration Issues

### 55.1 MEDIUM: Secrets in URL Query Parameters
**Location:** `liminallm/service/email.py:108, 160`

Reset and verification tokens exposed in URL query parameters - logged in access logs, browser history.

### 55.2 MEDIUM: Missing Input Validation on OAuth Responses
**Location:** `liminallm/service/auth.py:373-416, 430-456`

OAuth token response structure not validated - could fail silently on malformed responses.

### 55.3 MEDIUM: Insecure Redirect Following on OAuth Calls
**Location:** `liminallm/service/auth.py:356`

Default httpx behavior follows redirects without limits - potential SSRF.

### 55.4 MEDIUM: No API Key Rotation Handling
**Location:** `liminallm/service/voice.py:32-42`, `liminallm/service/model_backend.py:354-371`

API keys cannot be rotated without restarting application.

### 55.5 MEDIUM: Missing Validation on OAuth Redirect URI
**Location:** `liminallm/service/auth.py:282-286, 348-351`

Redirect URI not validated for HTTPS or allowed domain.

### 55.6 MEDIUM: Default Insecure SMTP Configuration
**Location:** `liminallm/service/email.py:85-97`

Configuration naming confusing - `smtp_use_tls=False` uses SMTP_SSL.

### 55.7 LOW: No Timeout on OpenAI Client
**Location:** `liminallm/service/model_backend.py:368-371`

OpenAI client uses default timeout (may be infinite).

### 55.8 LOW: Missing Explicit Error Handling for JSON Parsing
**Location:** `liminallm/service/voice.py:95-100`

JSON parsing could fail even after raise_for_status().

---

## 56. Cryptographic Implementation Issues

### 56.1 MEDIUM: Weak JWT Test Secret
**Location:** `tests/test_auth_unit.py:25`

Test JWT secret only 27 characters - lower entropy than production requirement.

### 56.2 MEDIUM: MFA Encryption Key Fallback Chain
**Location:** `liminallm/storage/memory.py:111-114`

MFA encryption uses JWT_SECRET as fallback - violates key separation principle.

### 56.3 MEDIUM: OAuth State Parameter Not Redis-Backed
**Location:** `liminallm/service/auth.py:131, 276-278`

OAuth state stored in-memory - fails in multi-process deployments without Redis.

### 56.4 ADVISORY: SHA1 in TOTP Implementation
**Location:** `liminallm/service/auth.py:903`

TOTP uses SHA1 per RFC 6238 - acceptable but documented limitation.

---

## 8th Pass Summary Tables

### Critical Priority (104 Issues Total)

| # | Issue | Location |
|---|-------|----------|
| 84 | MFA Request Missing Authentication | routes.py:997-1012 |
| 85 | MFA Verify Missing Session Ownership | routes.py:1015-1050 |
| 86 | Admin Role Modification Missing Tenant Isolation | routes.py:804-820 |
| 87 | Admin User Deletion Missing Tenant Isolation | routes.py:823-837 |
| 88 | Missing Audit Logging - User Signup | routes.py:525-570 |
| 89 | Missing Audit Logging - Admin User Creation | routes.py:771-801 |
| 90 | Missing Audit Logging - User Deletion | routes.py:824-837 |
| 91 | Missing Audit Logging - Permission Changes | routes.py:805-820 |
| 92 | Failed Login Attempts Not Logged | routes.py:597-598 |
| 93 | Password Change Events Not Logged | routes.py:1277-1305 |
| 94 | Email Verification Events Not Logged | auth.py:828-855 |
| 95 | Password Reset Completion Not Logged | auth.py:788-810 |
| 96 | Session Revocation Not Logged | routes.py:1308-1332 |
| 97 | MFA Bypass via Silent Database Failure | postgres.py:1354-1363 |
| 98 | Tenant Spoofing via Signup Endpoint | routes.py:545-549 |
| 99 | Tenant Spoofing via OAuth Complete | auth.py:484-495 |
| 100 | TOCTOU Race in Session Revocation | routes.py:1325-1329 |
| 101 | Session Verification State Machine Issue | auth.py:591-605 |
| 102 | MFA Secret Displayed in DOM | chat.js:2471 |
| 103 | OTP URI Exposed Without Escaping | chat.js:2480 |
| 104 | Passwords Displayed in Admin UI | admin.js:429 |

### High Priority (105 Issues Total)

| # | Issue | Location |
|---|-------|----------|
| 91 | Chat Cancel Missing Ownership | routes.py:1472-1510 |
| 92 | PII (Emails) Being Logged | email.py:65, 99, 103 |
| 93 | Failed Password Verification Not Logged | auth.py:862-873 |
| 94 | Token Refresh Failures Not Logged | routes.py:714-715 |
| 95 | Insufficient OAuth Exchange Logging | auth.py:411-415 |
| 96 | Insufficient MFA Failure Logging | auth.py:754-763 |
| 97 | MFA Lockout Race Condition | auth.py:756-763 |
| 98 | Unsigned MFA Verification | auth.py:553-556 |
| 99 | Missing Negative Value Validation | runtime.py:278-289 |
| 100 | Runtime Config Displayed in Plaintext | admin.js:257 |
| 101 | Admin Inspect Displays Sensitive Data | admin.js:532-536 |
| 102 | Unescaped Error Messages | Multiple files |
| 103 | Missing CSRF Protection | admin.html, index.html |
| 104 | Tokens in Session Storage | chat.js:13-20 |
| 105 | Tenant Validation Inconsistency | routes.py (admin endpoints) |

### Medium Priority (111 Issues Total)

| # | Issue | Location |
|---|-------|----------|
| 76-85 | HTTP Header Issues | app.py, nginx.conf |
| 86-87 | Cache Error Handling, Eviction Race | auth.py, postgres.py |
| 88-93 | Frontend Issues | chat.js, admin.js |
| 94-99 | External API Issues | email.py, auth.py, voice.py |
| 100-103 | Cryptographic Issues | memory.py, auth.py, tests |
| 104-111 | Logging Consistency Issues | Multiple files |

---

## 8th Pass Recommendations

### RBAC Actions (Immediate)

1. Add authentication to MFA request/verify endpoints
2. Validate session ownership before MFA operations
3. Add tenant isolation checks to all admin user operations
4. Validate ownership before chat request cancellation

### Audit Logging Actions (Immediate)

1. Add logging to signup, login failure, password change endpoints
2. Add logging to all admin user operations (create, delete, role change)
3. Fix email address logging in email service (use PII redaction)
4. Add logging for failed password verification
5. Add logging for session revocation and token refresh failures
6. Standardize log levels for security events

### HTTP Security Actions (Short-term)

1. Add Cache-Control to sensitive endpoints (/healthz, /metrics, APIs)
2. Fix X-Frame-Options mismatch (use DENY consistently)
3. Rate limit /healthz and /metrics endpoints
4. Enable HSTS by default in production
5. Add explicit CSP directives for object-src, media-src

### Business Logic Actions (Immediate)

1. Only update MFA cache AFTER database update succeeds
2. Remove tenant_id from signup and OAuth callback parameters
3. Use atomic database operations for session revocation
4. Use Redis Lua scripts for atomic MFA attempt counting
5. Add positive validation for rate limit parameters

### Frontend Security Actions (Immediate)

1. Remove MFA secret display from DOM
2. Escape OTP URI before innerHTML insertion
3. Remove password display from admin UI
4. Implement CSRF token protection
5. Move tokens from sessionStorage to memory-only
6. Redact sensitive data from runtime config display
7. Remove tenant_id from frontend (derive from JWT only)

### External API Actions (Short-term)

1. Use POST for reset/verification tokens (not URL parameters)
2. Implement input validation for OAuth responses
3. Set follow_redirects=False on OAuth HTTP clients
4. Implement API key rotation mechanism
5. Add timeout parameter to OpenAI client

### Cryptographic Actions (Short-term)

1. Use production-strength secrets in tests
2. Require separate MFA_SECRET_KEY (remove JWT_SECRET fallback)
3. Make Redis mandatory for OAuth state in production

---

**Total Issues After 8th Pass:**
- **Critical:** 104 (83 + 21 new)
- **High:** 105 (90 + 15 new)
- **Medium:** 111 (75 + 36 new)
- **Total:** 320

