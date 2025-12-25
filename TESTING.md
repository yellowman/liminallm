# LiminalLM Testing Guide

This document describes how to set up and run tests for LiminalLM.

## Quick Start

### Unit Tests (In-Memory)

Run unit tests with the in-memory store (no external dependencies):

```bash
./scripts/run_tests.sh
```

Or with pytest directly:

```bash
TEST_MODE=true USE_MEMORY_STORE=true pytest tests/
```

### Integration Tests with Docker

Start the full test environment:

```bash
docker compose -f docker-compose.test.yml up --build
```

This starts:
- PostgreSQL database (port 5433)
- Redis cache (port 6380)
- LiminalLM app (port 8000)
- Runs migrations and bootstraps admin user

### Smoke Tests

Run the QA smoke test suite against a running instance:

```bash
# Against local Docker environment
./scripts/smoke_test.sh

# Against a custom URL
./scripts/smoke_test.sh http://staging.example.com:8000
```

## Test Credentials

When using `docker-compose.test.yml`, the following credentials are pre-configured:

| Role  | Email             | Password        |
|-------|-------------------|-----------------|
| Admin | admin@test.local  | TestAdmin123!   |

Regular test users are created dynamically during tests.

## Environment Configuration

### Test Mode Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `TEST_MODE` | `false` | Enables test mode (relaxed security, in-memory fallback) |
| `USE_MEMORY_STORE` | `false` | Use in-memory store instead of PostgreSQL |
| `ALLOW_REDIS_FALLBACK_DEV` | `false` | Allow in-memory cache when Redis unavailable |

### Test Database

For integration tests with a real database:

```bash
export DATABASE_URL="postgresql://user:pass@localhost:5432/liminallm_test"
export REDIS_URL="redis://localhost:6379/1"
export JWT_SECRET="Test-JWT-Secret-For-QA-Environment-Only-32chars!"
```

## Bootstrap Admin User

Create an admin user for testing:

```bash
# Using environment variables
ADMIN_EMAIL=admin@test.local \
ADMIN_PASSWORD=TestAdmin123! \
python scripts/bootstrap_admin.py

# Using command line arguments
python scripts/bootstrap_admin.py \
  --email admin@test.local \
  --password TestAdmin123!

# Dry run (show what would be done)
python scripts/bootstrap_admin.py --email admin@test.local --password TestAdmin123! --dry-run
```

Password requirements:
- At least 12 characters
- Contains 3+ character classes (uppercase, lowercase, digits, special)

## Smoke Test Details

The smoke test script (`scripts/smoke_test.sh`) performs these checks:

1. **Health Check** - Verifies `/healthz` endpoint responds
2. **Authentication Protection** - Verifies endpoints require auth (401)
3. **User Signup** - Creates a new test user
4. **User Login** - Logs in with the test user
5. **Admin Login** - Logs in with admin credentials
6. **Create Conversation** - Creates a new conversation
7. **Chat Message** - Sends a message (may return 503 without LLM backend)
8. **File Upload** - Tests file upload endpoint
9. **List Artifacts** - Verifies artifact listing works
10. **Admin Access Control** - Verifies regular users get 403 on admin endpoints
11. **Admin Endpoint Access** - Verifies admins get 200 on admin endpoints

### Running Smoke Tests in CI

```bash
# Start services
docker compose -f docker-compose.test.yml up -d

# Wait for services and run tests
./scripts/smoke_test.sh http://localhost:8000

# Cleanup
docker compose -f docker-compose.test.yml down -v
```

## Test File Structure

```
tests/
├── conftest.py              # Pytest fixtures and configuration
├── test_integration_admin.py    # Admin API integration tests
├── test_endpoint_rate_limits.py # Rate limiting tests
├── test_auth.py                 # Authentication tests
└── ...

scripts/
├── run_tests.sh           # Run unit tests
├── bootstrap_admin.py     # Create admin user
├── smoke_test.sh          # QA smoke test suite
└── migrate.sh             # Run database migrations
```

## Troubleshooting

### Tests fail with "JWT_SECRET must mix character classes"

The test JWT secret must contain uppercase, lowercase, digits, and special characters:

```bash
export JWT_SECRET="Test-Secret-Key-4-Testing-Only-Do-Not-Use-In-Production!"
```

### Redis connection errors in tests

Tests can run without Redis using the in-memory fallback:

```bash
export TEST_MODE=true
export ALLOW_REDIS_FALLBACK_DEV=true
```

### "testclient" IP address error

This is handled automatically - the `Session.new()` method gracefully handles invalid IP addresses from the test client.

### Admin login fails in smoke tests

Ensure the admin user was bootstrapped:

```bash
# Check if bootstrap completed
docker compose -f docker-compose.test.yml logs bootstrap

# Re-run bootstrap
docker compose -f docker-compose.test.yml run --rm bootstrap
```

## Coverage Reports

Generate test coverage:

```bash
pytest --cov=liminallm --cov-report=html tests/
open htmlcov/index.html
```
