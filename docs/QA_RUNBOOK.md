# LiminalLM QA Runbook

This runbook documents the quality assurance process for validating LiminalLM releases.

## Quick Start

```bash
# Native (no Docker) - recommended for most testing
make qa-unit

# With Docker
make qa

# Run smoke tests against running instance
./scripts/smoke_test.sh http://localhost:8000
```

## Pre-requisites

### Native Testing (No Docker)

- Python 3.11+
- curl and jq (for smoke tests)
- No external dependencies required - uses in-memory stores

### Docker Testing

- Docker and Docker Compose
- curl and jq (for smoke tests)

## Native Testing Quick Start

Most QA testing can be done without Docker using in-memory stores:

```bash
# 1. Install dependencies
pip install -e ".[dev]"

# 2. Run all unit and integration tests (in-memory mode)
TEST_MODE=true USE_MEMORY_STORE=true pytest tests/ -v

# 3. Start server for manual testing
export JWT_SECRET="Test-Secret-Key-4-Testing-Only-Do-Not-Use-In-Production!"
export SHARED_FS_ROOT="/tmp/liminallm-data"
export USE_MEMORY_STORE=true
export TEST_MODE=true
uvicorn liminallm.app:app --reload --host 0.0.0.0 --port 8000

# 4. Run smoke tests (in another terminal)
./scripts/smoke_test.sh http://localhost:8000
```

## QA Procedure

### Step 1: Environment Setup

```bash
# Install dependencies
pip install -e ".[dev]"

# Verify installation
python -c "from liminallm.app import app; print('OK')"
```

### Step 2: Unit Tests (In-Memory Mode)

```bash
# Run all tests with in-memory store
TEST_MODE=true USE_MEMORY_STORE=true python -m pytest tests/ -v

# Expected: All tests pass (32+ tests)
```

### Step 3: Local API Smoke Test

```bash
# Start server
export JWT_SECRET="Test-Secret-Key-4-Testing-Only-Do-Not-Use-In-Production!"
export SHARED_FS_ROOT="/tmp/liminallm-data"
export USE_MEMORY_STORE=true
export TEST_MODE=true

python -m uvicorn liminallm.app:app --host 0.0.0.0 --port 8000

# In another terminal:
curl http://localhost:8000/healthz | jq .
curl http://localhost:8000/openapi.json > /dev/null && echo "OpenAPI OK"
```

### Step 4: Full API Flow Test

Test the complete user journey:

```bash
BASE_URL="http://localhost:8000"

# 1. Signup
SIGNUP=$(curl -s -X POST "$BASE_URL/v1/auth/signup" \
    -H "Content-Type: application/json" \
    -d '{"email": "test@example.com", "password": "TestPass123!"}')
TOKEN=$(echo "$SIGNUP" | jq -r '.data.access_token')

# 2. Create conversation
CONV=$(curl -s -X POST "$BASE_URL/v1/conversations" \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d '{"title": "Test"}')
CONV_ID=$(echo "$CONV" | jq -r '.data.id')

# 3. Chat
curl -s -X POST "$BASE_URL/v1/chat" \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d "{\"conversation_id\": \"$CONV_ID\", \"message\": {\"content\": \"Hello\"}, \"stream\": false}"
```

**Pass Criteria:**
- Signup returns access_token
- Chat returns status: ok

### Step 5: Integration Tests with PostgreSQL (Optional - Docker)

Skip this step if you don't have Docker. The in-memory tests in Step 2 cover the same functionality.

```bash
# Start PostgreSQL and Redis containers (credentials match docker-compose.test.yml)
docker compose -f docker-compose.test.yml up -d postgres redis
sleep 5

# Run migrations to create tables
docker compose -f docker-compose.test.yml run --rm migrate

# Run tests with real database (must disable memory store)
USE_MEMORY_STORE=false \
DATABASE_URL="postgresql://liminallm:testpassword123@localhost:5433/liminallm_test" \
REDIS_URL="redis://localhost:6380/0" \
    python -m pytest tests/ -v

# Cleanup
docker compose -f docker-compose.test.yml down -v
```

### Step 5 (Native Alternative): Integration Tests with Local PostgreSQL

If you have PostgreSQL installed locally without Docker:

```bash
# Create test database (requires pgvector extension)
sudo -u postgres psql -c "CREATE DATABASE liminallm_test;"
sudo -u postgres psql -d liminallm_test -c "CREATE EXTENSION IF NOT EXISTS vector;"
sudo -u postgres psql -d liminallm_test -c "CREATE EXTENSION IF NOT EXISTS citext;"

# Run migrations to create tables
DATABASE_URL="postgresql://postgres@localhost:5432/liminallm_test" \
    ./scripts/migrate.sh

# Run tests (must disable memory store)
USE_MEMORY_STORE=false \
DATABASE_URL="postgresql://postgres@localhost:5432/liminallm_test" \
    pytest tests/ -v
```

### Step 6: Full Docker Compose Test (Optional - Docker)

Skip this step if you don't have Docker. Steps 2-4 and Step 7+ work natively.

```bash
# Build and start all services
docker compose -f docker-compose.test.yml up --build -d
sleep 10

# Run smoke tests
./scripts/smoke_test.sh http://localhost:8000

# Cleanup
docker compose -f docker-compose.test.yml down -v
```

### Step 7: Admin Functionality

```bash
# Run admin integration tests
TEST_MODE=true USE_MEMORY_STORE=true python -m pytest tests/test_integration_admin.py -v

# Expected: 18 tests pass
```

### Step 8: Security Checks

```bash
# Run bandit security scanner
bandit -r liminallm/ -ll

# Check for common vulnerabilities
# - SQL injection: All queries use parameterized statements
# - XSS: All output is JSON-encoded
# - CSRF: Token-based auth with CSRF tokens for forms
```

## Post-Smoke Run Procedure

After your first successful smoke test, run these additional checks to catch idempotency, caching, race conditions, and isolation issues.

### Step 9: Repeat Smoke Test (3x)

Run the smoke script 3 times consecutively to catch:
- Idempotency key collisions
- Cache staleness issues
- Race conditions in concurrent requests

```bash
# Run 3 times
for i in 1 2 3; do
    echo "=== Smoke Run $i ==="
    ./scripts/smoke_test.sh http://localhost:8000
    sleep 2
done
```

**Pass Criteria:** All 3 runs pass without errors.

### Step 10: File Extension Upload Test

Test uploads for each allowed extension. Supported extensions:
- Text: `.txt`, `.md`, `.json`, `.csv`, `.tsv`
- Documents: `.pdf`
- Images: `.png`, `.jpg`, `.jpeg`, `.gif`, `.webp`
- Audio: `.mp3`, `.wav`, `.ogg`

```bash
# Create test files
echo "test content" > /tmp/test.txt
echo "# Markdown" > /tmp/test.md
echo '{"key": "value"}' > /tmp/test.json
echo "a,b,c" > /tmp/test.csv

# Upload each (requires valid $TOKEN)
for ext in txt md json csv; do
    curl -X POST "http://localhost:8000/v1/files/upload" \
        -H "Authorization: Bearer $TOKEN" \
        -F "file=@/tmp/test.$ext"
    echo ""
done
```

**Pass Criteria:** Each upload returns `{"status":"ok",...}` with a `file_id`.

### Step 11: Logout and Token Invalidation

Test that logout properly invalidates tokens:

```bash
# 1. Login and get token
LOGIN=$(curl -s -X POST "http://localhost:8000/v1/auth/login" \
    -H "Content-Type: application/json" \
    -d '{"email": "test@example.com", "password": "TestPass123!"}')
TOKEN=$(echo "$LOGIN" | jq -r '.data.access_token')

# 2. Verify token works
curl -s "http://localhost:8000/v1/conversations" \
    -H "Authorization: Bearer $TOKEN" | jq '.status'
# Should return "ok"

# 3. Logout
curl -s -X POST "http://localhost:8000/v1/auth/logout" \
    -H "Authorization: Bearer $TOKEN"

# 4. Verify token is invalidated
curl -s "http://localhost:8000/v1/conversations" \
    -H "Authorization: Bearer $TOKEN" | jq '.error.code'
# Should return "unauthorized" or "invalid session"
```

**Pass Criteria:** After logout, the token should be rejected with 401.

### Step 12: Password Reset Flow

```bash
# Request password reset (may require email configuration)
curl -s -X POST "http://localhost:8000/v1/auth/password/reset" \
    -H "Content-Type: application/json" \
    -d '{"email": "test@example.com"}'
# In test mode without email: token is returned in response
# In production: check email for reset link
```

### Step 13: Tenant Isolation Test

Create 2 users and verify they cannot access each other's data:

```bash
# Create User A
SIGNUP_A=$(curl -s -X POST "http://localhost:8000/v1/auth/signup" \
    -H "Content-Type: application/json" \
    -d '{"email": "userA@test.local", "password": "TestPass123!"}')
TOKEN_A=$(echo "$SIGNUP_A" | jq -r '.data.access_token')

# Create User B
SIGNUP_B=$(curl -s -X POST "http://localhost:8000/v1/auth/signup" \
    -H "Content-Type: application/json" \
    -d '{"email": "userB@test.local", "password": "TestPass123!"}')
TOKEN_B=$(echo "$SIGNUP_B" | jq -r '.data.access_token')

# User A creates a conversation
CONV_A=$(curl -s -X POST "http://localhost:8000/v1/conversations" \
    -H "Authorization: Bearer $TOKEN_A" \
    -H "Content-Type: application/json" \
    -d '{"title": "User A Private"}')
CONV_A_ID=$(echo "$CONV_A" | jq -r '.data.id')

# User B tries to access User A's conversation
RESULT=$(curl -s "http://localhost:8000/v1/conversations/$CONV_A_ID" \
    -H "Authorization: Bearer $TOKEN_B")
echo "$RESULT" | jq '.error'
# Should return 404 or 403, NOT the conversation data

# User B lists their own conversations (should be empty)
LIST_B=$(curl -s "http://localhost:8000/v1/conversations" \
    -H "Authorization: Bearer $TOKEN_B")
echo "$LIST_B" | jq '.data | length'
# Should return 0 (User B has no conversations)
```

**Pass Criteria:**
- User B cannot read User A's conversation (404 or 403)
- User B's conversation list does not include User A's data
- Each user can only see their own files and artifacts

### Automated Post-Smoke Tests

Run the automated test suite:

```bash
pytest tests/test_post_smoke.py -v
```

## Acceptance Criteria

| Check | Command | Expected |
|-------|---------|----------|
| Health endpoint | `curl /healthz` | `{"status":"healthy",...}` |
| Unit tests | `pytest tests/` | All pass |
| Admin tests | `pytest tests/test_integration_admin.py` | 18 pass |
| Smoke tests | `./scripts/smoke_test.sh` | All pass |
| Post-smoke tests | `pytest tests/test_post_smoke.py` | All pass |
| Security scan | `bandit -r liminallm/ -ll` | No high severity |

## Troubleshooting

### JWT_SECRET validation error

```bash
# Use a complex secret with mixed character classes
export JWT_SECRET="Test-Secret-Key-4-Testing-Only-Do-Not-Use-In-Production!"
```

### Rate limiting in tests

Rate limits apply even in test mode. Wait 60 seconds between repeated file uploads, or restart the server to clear in-memory rate limit state.

### Session "invalid session" errors

With `USE_MEMORY_STORE=true`, each server instance has its own memory store. Ensure all requests go to the same running instance.

### Admin bootstrap with in-memory store

The `bootstrap_admin.py` script creates its own runtime. For in-memory mode, use the test fixtures or run with PostgreSQL.

## CI/CD Integration

Add to your CI pipeline:

```yaml
test:
  script:
    - pip install -e ".[dev]"
    - make qa-unit

integration:
  services:
    - postgres:15
  script:
    - make qa-integration
```

## Related Documentation

- [TESTING.md](../TESTING.md) - Test environment setup
- [CONFIGURATION.md](CONFIGURATION.md) - Configuration sources
- [DEPLOYMENT.md](DEPLOYMENT.md) - Production deployment
