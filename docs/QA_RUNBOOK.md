# LiminalLM QA Runbook

This runbook documents the quality assurance process for validating LiminalLM releases.

## Quick Start

```bash
# Run all QA checks (requires Docker)
make qa

# Run unit tests only (no Docker)
make test

# Run smoke tests against running instance
./scripts/smoke_test.sh http://localhost:8000
```

## Pre-requisites

- Python 3.11+
- Docker and Docker Compose (for integration tests)
- curl and jq (for smoke tests)

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

### Step 5: Integration Tests with PostgreSQL

```bash
# Start PostgreSQL container
docker compose -f docker-compose.test.yml up -d postgres
sleep 5

# Run tests
DATABASE_URL="postgresql://testuser:testpass@localhost:5433/liminallm_test" \
    python -m pytest tests/ -v

# Cleanup
docker compose -f docker-compose.test.yml down -v
```

### Step 6: Full Docker Compose Test

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

## Acceptance Criteria

| Check | Command | Expected |
|-------|---------|----------|
| Health endpoint | `curl /healthz` | `{"status":"healthy",...}` |
| Unit tests | `pytest tests/` | All pass |
| Admin tests | `pytest tests/test_integration_admin.py` | 18 pass |
| Smoke tests | `./scripts/smoke_test.sh` | All pass |
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
