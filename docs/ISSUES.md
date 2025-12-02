# Codebase Issues and Security Audit

**Last Updated:** 2025-12-02
**Scope:** Comprehensive review against SPEC.md requirements (3rd pass)

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

**Critical Issues Found:** 38
**High Priority Issues:** 23
**Medium Priority Issues:** 18

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

## 19. Previously Resolved Issues

### 19.1 Session Exception Parameter (FIXED)

**Commit:** 3beddff

The `except_session_id` parameter in `revoke_all_user_sessions` now properly passed to store methods.

---

## Summary by Severity

### Critical (38 Issues)

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

### High Priority (23 Issues)

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

### Medium Priority (18 Issues)

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

### Session Management Actions

1. Implement session rotation (24h activity-based)
2. Add single-session mode enforcement
3. Add access token denylist on logout
4. Implement X-Session header for WebSockets

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

### Clustering/Training Actions

1. Implement global clustering
2. Add incremental clustering algorithm
3. Implement adapter pruning/merging
4. Add periodic clustering batch job
5. Update adapter_router_state after training

### Documentation Actions

1. Update SPEC.md to document all endpoints used by frontend
2. Document router "closest" selection algorithm
3. Document cache requirements for auth features
