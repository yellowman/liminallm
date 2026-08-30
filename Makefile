# LiminalLM Makefile
#
# Usage:
#   make test      - Run unit tests (in-memory), serially
#   make test-xdist - Every test but the browser lane, in parallel
#   make test-fast - Fast edit-loop tests; excludes @pytest.mark.slow
#   make test-fast-xdist - The same lane in parallel
#   make lint      - Run linters
#   make qa        - Full QA gate (lint + test + security)
#   make dev       - Start development server
#   make docker    - Build and test with Docker

.PHONY: help install test test-xdist test-fast test-fast-xdist test-browser lint security qa dev docker clean docker-clean

# Default target
help:
	@echo "LiminalLM Development Commands"
	@echo ""
	@echo "  make install    Install dependencies"
	@echo "  make test       Run unit tests (in-memory), serially"
	@echo "  make test-xdist Every test but the browser lane, across $(XDIST_WORKERS) processes"
	@echo "  make test-fast  Fast edit-loop tests; excludes @pytest.mark.slow"
	@echo "  make test-fast-xdist  The same lane across $(XDIST_WORKERS) processes"
	@echo "  make test-browser     Real browser against a real server (needs Chromium)"
	@echo "  make test-pg    Run tests with PostgreSQL (requires Docker)"
	@echo "  make lint       Run linters (ruff)"
	@echo "  make security   Run security scanner (bandit)"
	@echo "  make qa         Full QA gate (lint + test + security)"
	@echo "  make qa-unit    Fast QA (unit tests only)"
	@echo "  make dev        Start development server"
	@echo "  make docker     Build and run with docker-compose"
	@echo "  make smoke      Run smoke tests against localhost:8000"
	@echo "  make clean      Remove build artifacts"
	@echo "  make docker-clean  Remove test containers/volumes"

# Environment variables for testing
export TEST_MODE := true
export SHARED_FS_ROOT := /tmp/liminallm-data

# Install dependencies
install:
	pip install -e ".[dev]"

# Run unit tests with in-memory store
test:
	@mkdir -p $(SHARED_FS_ROOT)
	python -m pytest tests/ -v --tb=short -m 'not browser'

# Everything except the tests that run a real model or wait on a real clock.
# A convenience for the edit loop, never the gate: `make test` above and the
# pre-commit hook run all of it. Measured, this skips 5.7% of the tests and
# about 40% of the wall clock.
test-fast:
	@mkdir -p $(SHARED_FS_ROOT)
	python -m pytest tests/ -q --tb=short -m 'not slow and not browser'

# The same lane across several processes. Each worker gets a Postgres, a Redis
# database and a filesystem root of its own - see tests/test_worker_isolation.py
# - because the per-test TRUNCATE assumes it owns the database.
#
# A fixed default rather than `-n auto`: Redis has sixteen numbered databases,
# and on a large workstation `auto` would also start that many Postgres
# clusters. Override with `make test-fast-xdist XDIST_WORKERS=8`.
XDIST_WORKERS ?= 4
# `loadfile` keeps a file's tests on one worker. Measured on a 4-core box it is
# the same wall clock as the default per-test scheduler - 126.5s against 127.5s
# over three paired runs, which is noise - so it is chosen for what it makes
# predictable rather than for speed: a module-scoped fixture is built once per
# worker that sees the file, and tests written next to each other stay next to
# each other. Override with XDIST_DIST=load.
XDIST_DIST ?= loadfile
test-fast-xdist:
	@mkdir -p $(SHARED_FS_ROOT)
	python -m pytest tests/ -q --tb=short -m 'not slow and not browser' \
		-n $(XDIST_WORKERS) --dist $(XDIST_DIST)

# The same lane with nothing deselected. The slow set is not a different lane
# and does not need a different one: it is these tests plus the ones that run a
# real model or wait on a real clock, and the per-worker Postgres, Redis
# database and filesystem root that make the fast lane safe in parallel are not
# specific to a marker.
#
# Parallelism buys more here than it does in the fast lane, because what makes
# a test slow is usually waiting. Measured on a 4-core box: the 110 slow-marked
# tests alone take 5m37s serially and 1m43s at -n 4, and this whole lane -
# 2814 tests - takes 3m37s.
#
# This is the local gate, and what `make qa` runs. It is not the only signal:
# GitHub CI runs the same selection serially, once per supported Python
# version, which is a different question - whether the suite passes on an
# interpreter this machine does not have.
test-xdist:
	@mkdir -p $(SHARED_FS_ROOT)
	python -m pytest tests/ -q --tb=short -m 'not browser' \
		-n $(XDIST_WORKERS) --dist $(XDIST_DIST)

# The browser lane. Excluded from every lane above because it needs a Chromium
# binary, which `pip install playwright` does not provide - run
# `playwright install chromium` once, or point LIMINALLM_CHROMIUM at a build.
# It is not a second suite: it covers only what a browser can observe and a
# TestClient cannot, which is what the page persists where scripts can read it
# and what the browser actually puts on the wire.
test-browser:
	@mkdir -p $(SHARED_FS_ROOT)
	python -m pytest tests/ -q --tb=short -m browser

# Run tests with PostgreSQL (requires docker-compose)
# Credentials must match docker-compose.test.yml
# Uses trap to ensure cleanup runs even if tests fail
#
# `-m 'not browser'` for the same reason every other lane has it, and it was
# missed here when the marker was introduced: the dev extra installs Playwright
# but not a Chromium binary, and a browser test with the library present and no
# binary *errors* rather than skipping - measured. So this target collected the
# browser lane and failed on it after an ordinary `make install`.
test-pg:
	@bash -c '\
		trap "docker compose -f docker-compose.test.yml down" EXIT; \
		docker compose -f docker-compose.test.yml up -d --wait postgres redis && \
		docker compose -f docker-compose.test.yml run --rm migrate && \
		USE_MEMORY_STORE=false \
		DATABASE_URL="postgresql://liminallm:testpassword123@localhost:5433/liminallm_test" \
		REDIS_URL="redis://localhost:6380/0" \
		python -m pytest tests/ -v --tb=short -m "not browser" \
	'

# Lint with ruff (auto-fix safe issues)
#
# No `--select` and no `--ignore` on the first line: `[tool.ruff.lint]` in
# pyproject.toml already says `select = [E, F, W, I]` and `ignore = [E501]`,
# and CI's explicit flags only restate it. This line used to pass
# `--ignore E402`, which does not *add* to the configured ignore list - it
# replaces it. So locally E402 was suppressed and E501 was not, while CI had
# it the other way round, and five E402s and two unsorted import blocks sat on
# this branch through every local run and failed the first time CI saw them.
#
# `--extend-ignore` on the second line for the same reason: tests may keep an
# unused import or binding, and saying so must not silently drop E501 with it.
lint:
	ruff check liminallm/ --fix
	ruff check tests/ --extend-ignore F401,F841 --fix

# Security scan with bandit
security:
	bandit -r liminallm/ -ll -q

# Full QA gate - runs all checks
# The parallel lane, not the serial one: `test-xdist` runs the same tests and
# the gate should not cost four times the wall clock to say the same thing.
qa: lint security test-xdist
	@echo ""
	@echo "========================================="
	@echo " QA Gate PASSED"
	@echo "========================================="

# Fast QA - unit tests only (no Docker)
qa-unit: lint test-xdist
	@echo ""
	@echo "========================================="
	@echo " Unit QA Gate PASSED"
	@echo "========================================="

# Start development server
dev:
	@mkdir -p $(SHARED_FS_ROOT)
	python -m uvicorn liminallm.app:app --host 0.0.0.0 --port 8000 --reload

# Run smoke tests against running server
smoke:
	./scripts/smoke_test.sh http://localhost:8000

# Build and test with Docker
# Uses trap to ensure cleanup runs even if tests fail
docker:
	@bash -c '\
		trap "docker compose -f docker-compose.test.yml down -v" EXIT; \
		docker compose -f docker-compose.test.yml up --build -d --wait && \
		./scripts/smoke_test.sh http://localhost:8000 \
	'

# Clean build artifacts
clean:
	rm -rf .pytest_cache __pycache__ *.egg-info dist build
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Clean up Docker test containers and volumes (useful after interrupted tests)
docker-clean:
	docker compose -f docker-compose.test.yml down -v --remove-orphans 2>/dev/null || true
