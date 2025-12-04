# Rate Limiting Security Audit Report
**Date:** 2025-12-04
**Auditor:** Security Review
**Scope:** liminallm rate limiting implementation

---

## Executive Summary

This audit identified **12 security vulnerabilities** across 3 severity levels:
- **CRITICAL:** 2 issues
- **HIGH:** 3 issues
- **MEDIUM:** 7 issues

The most critical issues are:
1. Missing rate limits on token refresh endpoint (credential stuffing risk)
2. Missing rate limits on 7 other authenticated endpoints
3. INCR/EXPIRE race condition causing permanent rate limit lockout
4. Email-based DoS attacks on signup/login

---

## CRITICAL Severity Issues

### 1. Missing Rate Limit on Token Refresh Endpoint

**Severity:** CRITICAL
**File:** `/home/user/liminallm/liminallm/api/routes.py`
**Line:** 695

**Code:**
```python
@router.post("/auth/refresh", response_model=Envelope, tags=["auth"])
async def refresh_tokens(
    body: TokenRefreshRequest,
    response: Response,
    authorization: Optional[str] = Header(None),
    x_tenant_id: Optional[str] = Header(
        None, convert_underscores=False, alias="X-Tenant-ID"
    ),
):
    runtime = get_runtime()
    tenant_hint = body.tenant_id or x_tenant_id
    user, session, tokens = await runtime.auth.refresh_tokens(
        body.refresh_token, tenant_hint=tenant_hint
    )
    # NO RATE LIMIT CHECK HERE!
```

**Attack Scenario:**
1. Attacker obtains or guesses refresh tokens (leaked, brute force, etc.)
2. Can attempt unlimited token refresh attempts without rate limiting
3. Enables credential stuffing attacks on refresh tokens
4. Can test stolen refresh tokens at high speed
5. Can brute-force short/weak refresh tokens

**Impact:**
- Unlimited authentication bypass attempts
- Credential stuffing on refresh tokens
- Token enumeration attacks
- High server load from spam requests

**Recommended Fix:**
```python
@router.post("/auth/refresh", response_model=Envelope, tags=["auth"])
async def refresh_tokens(
    body: TokenRefreshRequest,
    response: Response,
    request: Request,  # Add request parameter
    authorization: Optional[str] = Header(None),
    x_tenant_id: Optional[str] = Header(
        None, convert_underscores=False, alias="X-Tenant-ID"
    ),
):
    runtime = get_runtime()

    # Add rate limiting by IP for unauthenticated endpoint
    client_ip = request.client.host if request.client else "unknown"
    await _enforce_rate_limit(
        runtime,
        f"refresh:{client_ip}",
        limit=20,  # 20 refresh attempts per minute per IP
        window_seconds=60,
    )

    tenant_hint = body.tenant_id or x_tenant_id
    user, session, tokens = await runtime.auth.refresh_tokens(
        body.refresh_token, tenant_hint=tenant_hint
    )
```

---

### 2. Missing Rate Limits on 7 Additional Endpoints

**Severity:** CRITICAL
**File:** `/home/user/liminallm/liminallm/api/routes.py`
**Lines:** Various

**Affected Endpoints:**
1. `POST /auth/logout` - Line ~704
2. `GET /artifacts` - Line ~1200s
3. `POST /tools/{tool_id}/invoke` - Line ~1300s
4. `POST /artifacts` - Line ~1250s
5. `PATCH /artifacts/{artifact_id}` - Line ~1270s
6. `POST /contexts` - Line ~1400s
7. `POST /contexts/{context_id}/sources` - Line ~1450s

**Attack Scenario (example: /artifacts):**
```bash
# Attacker can spam artifact creation without limits
for i in {1..10000}; do
    curl -X POST https://api.example.com/artifacts \
         -H "Authorization: Bearer $TOKEN" \
         -d '{"name":"spam'$i'","content":"x"}' &
done
```

**Impact:**
- Resource exhaustion via unlimited artifact/context creation
- Database bloat from spam data
- Storage exhaustion
- Service degradation for legitimate users
- Potential DoS via tool invocation spam

**Recommended Fix:**
Add rate limiting to each endpoint. Example for artifacts:

```python
@router.post("/artifacts", response_model=Envelope, status_code=201, tags=["artifacts"])
async def create_artifact(
    body: CreateArtifactRequest,
    principal: AuthContext = Depends(get_user)
):
    runtime = get_runtime()

    # Add rate limit
    await _enforce_rate_limit(
        runtime,
        f"artifacts:create:{principal.user_id}",
        runtime.settings.read_rate_limit_per_minute,  # or create a specific limit
        60,
    )

    # ... rest of function
```

---

## HIGH Severity Issues

### 3. INCR/EXPIRE Race Condition - Permanent Lockout

**Severity:** HIGH
**File:** `/home/user/liminallm/liminallm/storage/redis_cache.py`
**Lines:** 62-67

**Code:**
```python
# Use pipeline for atomic INCR + EXPIRE to prevent race conditions
pipe = self.client.pipeline()
pipe.incr(redis_key)
pipe.expire(redis_key, ttl)
results = await pipe.execute()
current = results[0]
```

**Issue:**
The comment claims "atomic operations" but `pipeline()` without `transaction=True` is **NOT atomic**. It only batches commands to reduce network round-trips. Commands can fail mid-execution.

**Attack Scenario:**
1. Client sends request, INCR executes (counter=1, no TTL set)
2. Network interruption or process crash before EXPIRE
3. Key exists in Redis with count but no TTL
4. Key never expires (permanent)
5. All future requests for this rate limit bucket are blocked
6. Example: `rate:chat:user123:28847` stuck forever at count=1

**Impact:**
- Permanent rate limit lockout for affected keys
- Requires manual Redis intervention to fix
- Can happen during network issues, deployments, crashes
- Affects user experience severely (legitimate users locked out)

**Proof of Concept:**
```python
# Simulate the race condition
import redis
import time

r = redis.Redis()
key = "rate:test:123"

# Simulate INCR succeeding but EXPIRE failing
r.incr(key)  # Counter is now 1
# <--- Process crashes here, EXPIRE never runs

# Later:
print(r.ttl(key))  # Returns -1 (no expiration)
print(r.get(key))  # Returns 1

# This key will never expire!
```

**Recommended Fix:**
Use a Lua script for atomic INCR+EXPIRE:

```python
async def check_rate_limit(
    self, key: str, limit: int, window_seconds: int, *, return_remaining: bool = False
) -> Union[bool, Tuple[bool, int]]:
    now = datetime.utcnow()
    now_bucket = int(now.timestamp() // window_seconds)
    redis_key = f"rate:{key}:{now_bucket}"
    bucket_end = (now_bucket + 1) * window_seconds
    ttl = max(1, int(bucket_end - now.timestamp()))

    # Use Lua script for atomic INCR+EXPIRE
    lua_script = """
    local current = redis.call('INCR', KEYS[1])
    redis.call('EXPIRE', KEYS[1], ARGV[1])
    return current
    """

    current = await self.client.eval(lua_script, 1, redis_key, ttl)

    allowed = current <= limit
    if return_remaining:
        remaining = max(0, limit - current)
        return (allowed, remaining)
    return allowed
```

---

### 4. Rate Limit Exhaustion DoS on Signup/Login

**Severity:** HIGH
**File:** `/home/user/liminallm/liminallm/api/routes.py`
**Lines:** 539-544, 585-589

**Code:**
```python
# Signup
await _enforce_rate_limit(
    runtime,
    f"signup:{body.email.lower()}",  # Only email-based
    runtime.settings.signup_rate_limit_per_minute,
    60,
)

# Login
await _enforce_rate_limit(
    runtime,
    f"login:{body.email.lower()}",  # Only email-based
    runtime.settings.login_rate_limit_per_minute,
    60,
)
```

**Attack Scenario:**
1. Attacker knows victim's email: `victim@company.com`
2. Attacker sends 10 signup requests with victim's email (limit is 5)
3. Victim's email is now rate-limited for 60 seconds
4. Victim attempts to sign up → blocked by rate limit
5. Attacker repeats attack every 60 seconds
6. Victim cannot create account or reset password

**Impact:**
- Account creation denial of service
- Login denial of service
- Password reset denial of service
- Targeted harassment of specific users
- No IP-based protection

**Recommended Fix:**
Implement dual rate limiting (email + IP):

```python
@router.post("/auth/signup", response_model=Envelope, status_code=201, tags=["auth"])
async def signup(body: SignupRequest, request: Request, response: Response):
    settings = get_settings()
    if not settings.allow_signup:
        raise _http_error("forbidden", "signup disabled", status_code=403)

    runtime = get_runtime()
    client_ip = request.client.host if request.client else "unknown"

    # Rate limit by both email AND IP
    await _enforce_rate_limit(
        runtime,
        f"signup:email:{body.email.lower()}",
        runtime.settings.signup_rate_limit_per_minute,
        60,
    )

    await _enforce_rate_limit(
        runtime,
        f"signup:ip:{client_ip}",
        limit=10,  # Allow more attempts per IP (covers multiple users)
        window_seconds=60,
    )

    # ... rest of function
```

---

### 5. No IP-Based Rate Limiting (Distributed Bypass)

**Severity:** HIGH
**File:** `/home/user/liminallm/liminallm/api/routes.py`
**Lines:** All rate limit implementations

**Issue:**
No client IP extraction or IP-based rate limiting anywhere in the codebase.

**Search Results:**
```bash
$ grep -r "X-Forwarded-For\|X-Real-IP\|request.client" liminallm/api/
# No results
```

**Attack Scenario:**
1. Attacker controls botnet with 1000 IPs
2. Each IP creates account and gets legitimate user_id
3. Each bot can make 60 chat requests/minute (user-based limit)
4. Total: 60,000 requests/minute from single attacker
5. Bypasses per-user rate limiting completely

**Impact:**
- Large-scale API abuse
- Resource exhaustion
- Cost inflation (LLM API costs)
- Service degradation
- No protection against distributed attacks

**Recommended Fix:**
1. Add client IP extraction utility:

```python
def get_client_ip(request: Request) -> str:
    """Extract client IP, respecting reverse proxy headers."""
    # Check X-Forwarded-For header (if behind proxy)
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        # Take first IP (original client)
        return forwarded.split(",")[0].strip()

    # Check X-Real-IP header
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip.strip()

    # Fallback to direct connection
    if request.client:
        return request.client.host

    return "unknown"
```

2. Add IP-based rate limits to expensive endpoints:

```python
@router.post("/chat", response_model=Envelope, tags=["chat"])
async def chat(
    body: ChatRequest,
    request: Request,
    response: Response,
    principal: AuthContext = Depends(get_user),
):
    runtime = get_runtime()
    user_id = principal.user_id
    client_ip = get_client_ip(request)

    # Rate limit by user
    await _enforce_rate_limit(
        runtime,
        f"chat:user:{user_id}",
        runtime.settings.chat_rate_limit_per_minute,
        runtime.settings.chat_rate_limit_window_seconds,
    )

    # Also rate limit by IP (higher limit)
    await _enforce_rate_limit(
        runtime,
        f"chat:ip:{client_ip}",
        limit=300,  # 5x user limit
        window_seconds=60,
    )
```

**Security Note:**
Be aware that `X-Forwarded-For` can be spoofed if not behind a trusted reverse proxy. Ensure your proxy (nginx, CloudFlare, etc.) is configured to set this header correctly.

---

## MEDIUM Severity Issues

### 6. Rate Limit Bypass via Email Variations

**Severity:** MEDIUM
**File:** `/home/user/liminallm/liminallm/api/routes.py`
**Lines:** 541, 587

**Code:**
```python
f"signup:{body.email.lower()}"
f"login:{body.email.lower()}"
```

**Issue:**
Only applies `.lower()` for normalization. Does not handle:
- Gmail dot addressing: `user.name@gmail.com` = `username@gmail.com`
- Plus addressing: `user+tag1@example.com` vs `user+tag2@example.com`
- Unicode homoglyphs: `test@example.com` vs `tеst@example.com` (Cyrillic е)
- Domain aliases

**Attack Scenario:**
```python
# Attacker bypasses 5 signup/minute limit:
emails = [
    "user@example.com",
    "user+1@example.com",
    "user+2@example.com",
    "user+3@example.com",
    "user+4@example.com",
    "user+5@example.com",
    # ... up to user+100@example.com
]

for email in emails:
    signup(email)  # Each has separate rate limit!
```

**Impact:**
- Bypass signup rate limits (5/min → 500/min with plus addressing)
- Bypass login rate limits
- Create many accounts quickly
- Spam prevention ineffective

**Recommended Fix:**
```python
def normalize_email(email: str) -> str:
    """Normalize email for rate limiting."""
    email = email.lower().strip()

    # Split into local and domain
    if "@" not in email:
        return email

    local, domain = email.rsplit("@", 1)

    # Remove plus addressing: user+tag@domain → user@domain
    if "+" in local:
        local = local.split("+")[0]

    # Gmail: remove dots from local part
    if domain in ("gmail.com", "googlemail.com"):
        local = local.replace(".", "")

    return f"{local}@{domain}"

# Usage:
await _enforce_rate_limit(
    runtime,
    f"signup:{normalize_email(body.email)}",
    runtime.settings.signup_rate_limit_per_minute,
    60,
)
```

---

### 7. Fixed Window Burst Attack

**Severity:** MEDIUM
**File:** `/home/user/liminallm/liminallm/storage/redis_cache.py`
**Lines:** 56-58

**Code:**
```python
now = datetime.utcnow()
now_bucket = int(now.timestamp() // window_seconds)
redis_key = f"rate:{key}:{now_bucket}"
```

**Issue:**
Fixed window implementation allows burst attacks at window boundaries.

**Attack Scenario:**
```
Limit: 10 requests per 60 seconds
Window 0: 00:00:00 - 00:00:59
Window 1: 00:01:00 - 00:01:59

Timeline:
00:00:59.0 - Send 10 requests → bucket 0, count=10 ✓
00:01:00.0 - Send 10 requests → bucket 1, count=10 ✓

Result: 20 requests in 1 second (200% of limit)
```

**Impact:**
- Can send 2x limit in short burst
- Defeats purpose of rate limiting
- Can cause resource spikes
- Inherent to fixed window algorithm

**Recommended Fix:**
Implement sliding window with sorted sets:

```python
async def check_rate_limit_sliding(
    self, key: str, limit: int, window_seconds: int, *, return_remaining: bool = False
) -> Union[bool, Tuple[bool, int]]:
    """Sliding window rate limit using sorted sets."""
    now = time.time()
    redis_key = f"rate:{key}"

    # Lua script for atomic sliding window check
    lua_script = """
    local key = KEYS[1]
    local now = tonumber(ARGV[1])
    local window = tonumber(ARGV[2])
    local limit = tonumber(ARGV[3])

    -- Remove old entries outside window
    redis.call('ZREMRANGEBYSCORE', key, 0, now - window)

    -- Count current entries
    local current = redis.call('ZCARD', key)

    if current < limit then
        -- Add new entry
        redis.call('ZADD', key, now, now)
        redis.call('EXPIRE', key, window)
        return {1, limit - current - 1}
    else
        return {0, 0}
    end
    """

    result = await self.client.eval(
        lua_script, 1, redis_key, now, window_seconds, limit
    )

    allowed = bool(result[0])
    remaining = result[1]

    if return_remaining:
        return (allowed, remaining)
    return allowed
```

---

### 8. Rate Limit Information Disclosure

**Severity:** MEDIUM
**File:** `/home/user/liminallm/liminallm/api/routes.py`
**Lines:** 235-239

**Code:**
```python
def apply_headers(self, response: Response) -> None:
    """Apply rate limit headers to response per IETF draft-polli-ratelimit-headers."""
    response.headers["X-RateLimit-Limit"] = str(self.limit)
    response.headers["X-RateLimit-Remaining"] = str(max(0, self.remaining))
    response.headers["X-RateLimit-Reset"] = str(self.reset_seconds)
```

**Issue:**
Exposes rate limit information to attackers.

**Attack Scenario:**
```bash
$ curl -i https://api.example.com/chat
HTTP/1.1 200 OK
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 59
X-RateLimit-Reset: 60

# Attacker now knows:
# 1. Exact limit (60 requests)
# 2. When window resets (60 seconds)
# 3. How many requests remaining (59)

# Can optimize attack:
# - Send exactly 60 requests per minute
# - Time requests to reset window
# - Know when to switch attack vectors
```

**Impact:**
- Helps attackers optimize rate limit bypass
- Reveals system capacity information
- Enables precise timing attacks
- Industry best practice debate (some argue transparency is good)

**Recommended Fix:**
Make headers optional via configuration:

```python
class Settings:
    # Add to config
    expose_rate_limit_headers: bool = env_field(False, "EXPOSE_RATE_LIMIT_HEADERS")

# In apply_headers:
def apply_headers(self, response: Response, expose: bool = False) -> None:
    if not expose:
        return

    response.headers["X-RateLimit-Limit"] = str(self.limit)
    response.headers["X-RateLimit-Remaining"] = str(max(0, self.remaining))
    response.headers["X-RateLimit-Reset"] = str(self.reset_seconds)
```

---

### 9. Negative/Zero Limit Bypass

**Severity:** MEDIUM (requires misconfiguration)
**File:** `/home/user/liminallm/liminallm/service/runtime.py`
**Lines:** 263-264

**Code:**
```python
if limit <= 0:
    return (True, limit) if return_remaining else True
```

**Issue:**
If limit is 0 or negative, ALL requests are allowed (not blocked).

**Attack Scenario:**
```python
# Admin mistakenly sets limit to 0 thinking it means "block all"
CHAT_RATE_LIMIT_PER_MINUTE=0

# Or attacker exploits config injection to set negative value
# Result: Unlimited requests allowed
```

**Impact:**
- Bypasses all rate limiting if misconfigured
- Counter-intuitive behavior (0 should mean no access)
- Could be exploited via config injection

**Recommended Fix:**
```python
if limit <= 0:
    # Raise error instead of silently allowing
    logger.error(
        "rate_limit_invalid_config",
        key=key,
        limit=limit,
        message="Rate limit must be positive integer"
    )
    raise ValueError(f"Invalid rate limit: {limit}. Must be > 0")
```

---

### 10. In-Memory Rate Limit State Loss

**Severity:** MEDIUM
**File:** `/home/user/liminallm/liminallm/service/runtime.py`
**Lines:** 160-161, 278-286

**Code:**
```python
# Runtime initialization
self._local_rate_limits: Dict[str, Tuple[datetime, int]] = {}
self._local_rate_limit_lock = asyncio.Lock()

# Fallback when Redis unavailable
async with runtime._local_rate_limit_lock:
    window_start, count = runtime._local_rate_limits.get(key, (now, 0))
    if now - window_start >= window:
        window_start, count = now, 0
    new_count = count + 1
    allowed = new_count <= limit
    if allowed:
        runtime._local_rate_limits[key] = (window_start, new_count)
```

**Issue:**
In-memory rate limits are lost on:
- Application restart
- Server crash
- Deployment
- Multiple application instances (not shared)

**Attack Scenario:**
```bash
# Attacker monitors for deployments
1. Send 60/60 chat requests (at limit)
2. Trigger app restart (or wait for deployment)
3. Rate limit counter resets to 0
4. Send another 60 requests immediately
5. Repeat on each restart

# With multiple app instances behind load balancer:
1. Each instance has separate in-memory counters
2. Round-robin to 10 instances
3. 60 requests/min per instance = 600 total requests/min
4. 10x bypass of intended limit
```

**Impact:**
- Rate limits reset on restart (attack window)
- Multi-instance deployments ineffective
- Can't enforce limits across fleet
- Only works in single-instance dev mode

**Current Mitigation:**
Documentation warns about this (lines 69-78):
```python
logger.warning(
    "redis_disabled_fallback",
    message=(
        f"Running without Redis under {fallback_mode}; rate limits, idempotency durability, and "
        "workflow/router caches are in-memory only."
    ),
)
```

**Recommended Fix:**
Require Redis in production:

```python
if not self.cache:
    if not self.settings.test_mode and not self.settings.allow_redis_fallback_dev:
        raise RuntimeError(
            "Redis is required for sessions, rate limits, idempotency, and workflow caches; "
            "start Redis or set TEST_MODE=true/ALLOW_REDIS_FALLBACK_DEV=true for local fallback."
        )
```

**Status:** Already enforced (lines 56-63), but environment variables could disable it.

---

### 11. Time Bucket Calculation Integer Overflow

**Severity:** MEDIUM (theoretical)
**File:** `/home/user/liminallm/liminallm/storage/redis_cache.py`
**Lines:** 57, 59-60

**Code:**
```python
now_bucket = int(now.timestamp() // window_seconds)
bucket_end = (now_bucket + 1) * window_seconds
ttl = max(1, int(bucket_end - now.timestamp()))
```

**Issue:**
Integer overflow possible with large timestamps or small windows.

**Attack Scenario:**
```python
# Year 2038 problem (32-bit systems)
now = datetime(2038, 1, 19, 3, 14, 8)  # Unix timestamp overflow
now_bucket = int(now.timestamp() // 60)
# Potential negative or wrapped value

# Or with very small window
window_seconds = 1
now_bucket = int(1733270400 // 1)  # Very large bucket number
bucket_end = (now_bucket + 1) * 1  # Potential overflow
```

**Impact:**
- Rate limit keys collide or fail
- TTL calculation incorrect
- Unlikely on 64-bit systems until year 292,277,026,596

**Recommended Fix:**
Add validation:

```python
async def check_rate_limit(
    self, key: str, limit: int, window_seconds: int, *, return_remaining: bool = False
) -> Union[bool, Tuple[bool, int]]:
    # Validate inputs
    if window_seconds <= 0 or window_seconds > 86400:  # Max 1 day
        raise ValueError(f"Invalid window_seconds: {window_seconds}")

    now = datetime.utcnow()
    now_ts = now.timestamp()

    # Use floor division with bounds check
    now_bucket = int(now_ts // window_seconds)
    if now_bucket < 0:
        raise ValueError(f"Invalid bucket calculation: {now_bucket}")

    # ... rest of function
```

---

### 12. No Rate Limit on Logout (Minor)

**Severity:** MEDIUM
**File:** `/home/user/liminallm/liminallm/api/routes.py`
**Line:** ~704

**Code:**
```python
@router.post("/auth/logout", response_model=Envelope, tags=["auth"])
async def logout(
    authorization: Optional[str] = Header(None),
    x_tenant_id: Optional[str] = Header(
        None, convert_underscores=False, alias="X-Tenant-ID"
    ),
):
    runtime = get_runtime()
    # NO RATE LIMIT
    session_id = _parse_session_cookie(authorization)
```

**Attack Scenario:**
```bash
# Spam logout endpoint to cause Redis operations
for i in {1..100000}; do
    curl -X POST https://api.example.com/auth/logout \
         -H "Cookie: session_id=fake_session_$i" &
done
```

**Impact:**
- Redis spam on session deletion
- Minor DoS vector
- Resource exhaustion

**Recommended Fix:**
```python
@router.post("/auth/logout", response_model=Envelope, tags=["auth"])
async def logout(
    request: Request,
    authorization: Optional[str] = Header(None),
    x_tenant_id: Optional[str] = Header(
        None, convert_underscores=False, alias="X-Tenant-ID"
    ),
):
    runtime = get_runtime()

    # Rate limit by IP
    client_ip = request.client.host if request.client else "unknown"
    await _enforce_rate_limit(
        runtime,
        f"logout:ip:{client_ip}",
        limit=30,
        window_seconds=60,
    )

    # ... rest of function
```

---

## Summary of Recommendations

### Immediate Actions (Critical/High)

1. **Add rate limiting to all missing endpoints:**
   - `/auth/refresh` (CRITICAL - credential stuffing risk)
   - `/auth/logout`
   - `/artifacts/*`
   - `/tools/{tool_id}/invoke`
   - `/contexts/*`

2. **Fix INCR/EXPIRE race condition:**
   - Replace pipeline with Lua script for atomic operations
   - Prevents permanent rate limit lockout

3. **Implement IP-based rate limiting:**
   - Add `get_client_ip()` helper
   - Add IP rate limits to all unauthenticated endpoints
   - Protect against distributed attacks

4. **Add dual rate limiting to signup/login:**
   - Prevent email-based DoS attacks
   - Rate limit by both email and IP

### Medium Priority

5. **Improve email normalization:**
   - Handle plus addressing
   - Handle Gmail dot addressing
   - Prevent rate limit bypass via email variations

6. **Consider sliding window algorithm:**
   - Prevents burst attacks at window boundaries
   - More expensive but more accurate

7. **Make rate limit headers optional:**
   - Reduce information disclosure
   - Configurable via environment variable

8. **Add input validation:**
   - Reject limit <= 0 instead of allowing
   - Validate window_seconds bounds
   - Add overflow checks

### Configuration Review

9. **Audit environment variable handling:**
   - Ensure rate limit configs cannot be negative
   - Add validation in Settings class
   - Document security implications

10. **Require Redis in production:**
    - Already enforced, but document clearly
    - Prevent TEST_MODE in production
    - Add monitoring for Redis failures

---

## Testing Recommendations

### Unit Tests Needed

```python
# Test rate limit bypass scenarios
async def test_refresh_endpoint_rate_limit():
    """Ensure refresh endpoint has rate limiting."""
    for i in range(25):  # Above limit
        response = await client.post("/auth/refresh", json={
            "refresh_token": "fake_token"
        })
        if i < 20:
            assert response.status_code in [200, 401]  # Allowed
        else:
            assert response.status_code == 429  # Rate limited

async def test_email_normalization():
    """Test email variations use same rate limit."""
    emails = ["user+1@x.com", "user+2@x.com", "user+3@x.com"]
    for email in emails[:5]:  # Limit is 5
        await signup(email)

    # 6th request with different variation should be rate limited
    response = await signup("user+6@x.com")
    assert response.status_code == 429

async def test_incr_expire_atomicity():
    """Verify INCR and EXPIRE are atomic."""
    # This requires integration test with Redis
    # Simulate network failure between INCR and EXPIRE
    pass
```

### Load Testing

```bash
# Test distributed bypass
ab -n 10000 -c 100 https://api.example.com/chat

# Test window boundary burst
# Send 60 at t=59s, then 60 at t=61s

# Test email DoS
for i in {1..100}; do
    curl -X POST /auth/signup -d '{"email":"victim@x.com"}'
done
```

---

## Appendix: Rate Limit Bypass Cheat Sheet

| Attack Vector | Severity | Mitigation |
|--------------|----------|------------|
| Missing endpoint rate limits | CRITICAL | Add rate limits to all endpoints |
| Refresh token stuffing | CRITICAL | Rate limit `/auth/refresh` |
| INCR/EXPIRE race | HIGH | Use Lua script |
| Email-based DoS | HIGH | Dual rate limiting (email + IP) |
| Distributed bypass (no IP limit) | HIGH | Add IP-based rate limiting |
| Email variations (+, dots) | MEDIUM | Normalize emails |
| Window boundary burst | MEDIUM | Use sliding window |
| Information disclosure | MEDIUM | Make headers optional |
| Negative limit bypass | MEDIUM | Validate limit > 0 |
| Multi-instance bypass | MEDIUM | Require Redis in prod |

---

## Conclusion

The rate limiting implementation has significant gaps that expose the system to:
- **Credential stuffing attacks** (missing rate limits)
- **Denial of service** (email-based DoS, distributed attacks)
- **Resource exhaustion** (unlimited endpoints)
- **Operational issues** (permanent lockouts from race conditions)

**Priority:** Implement fixes for CRITICAL and HIGH severity issues immediately before production deployment.

**Estimated effort:**
- Critical fixes: 2-3 days
- High severity fixes: 3-5 days
- Medium priority: 1-2 weeks
- Total: ~2-3 weeks for comprehensive fix

---

*End of Report*
