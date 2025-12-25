# LiminalLM Makefile
#
# Usage:
#   make test      - Run unit tests (in-memory)
#   make lint      - Run linters
#   make qa        - Full QA gate (lint + test + security)
#   make dev       - Start development server
#   make docker    - Build and test with Docker

.PHONY: help install test lint security qa dev docker clean

# Default target
help:
	@echo "LiminalLM Development Commands"
	@echo ""
	@echo "  make install    Install dependencies"
	@echo "  make test       Run unit tests (in-memory)"
	@echo "  make test-pg    Run tests with PostgreSQL (requires Docker)"
	@echo "  make lint       Run linters (ruff)"
	@echo "  make security   Run security scanner (bandit)"
	@echo "  make qa         Full QA gate (lint + test + security)"
	@echo "  make qa-unit    Fast QA (unit tests only)"
	@echo "  make dev        Start development server"
	@echo "  make docker     Build and run with docker-compose"
	@echo "  make smoke      Run smoke tests against localhost:8000"
	@echo "  make clean      Remove build artifacts"

# Environment variables for testing
export TEST_MODE := true
export USE_MEMORY_STORE := true
export JWT_SECRET := Test-Secret-Key-4-Testing-Only-Do-Not-Use-In-Production!
export SHARED_FS_ROOT := /tmp/liminallm-data

# Install dependencies
install:
	pip install -e ".[dev]"

# Run unit tests with in-memory store
test:
	@mkdir -p $(SHARED_FS_ROOT)
	python -m pytest tests/ -v --tb=short

# Run tests with PostgreSQL (requires docker-compose)
test-pg:
	docker compose -f docker-compose.test.yml up -d postgres
	sleep 5
	DATABASE_URL="postgresql://testuser:testpass@localhost:5433/liminallm_test" \
		python -m pytest tests/ -v --tb=short
	docker compose -f docker-compose.test.yml down

# Lint with ruff (auto-fix safe issues)
lint:
	ruff check liminallm/ --fix --ignore E402
	ruff check tests/ --fix --ignore F401,F841

# Security scan with bandit
security:
	bandit -r liminallm/ -ll -q

# Full QA gate - runs all checks
qa: lint security test
	@echo ""
	@echo "========================================="
	@echo " QA Gate PASSED"
	@echo "========================================="

# Fast QA - unit tests only (no Docker)
qa-unit: lint test
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
docker:
	docker compose -f docker-compose.test.yml up --build -d
	sleep 10
	./scripts/smoke_test.sh http://localhost:8000
	docker compose -f docker-compose.test.yml down -v

# Clean build artifacts
clean:
	rm -rf .pytest_cache __pycache__ *.egg-info dist build
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
