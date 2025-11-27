#!/usr/bin/env bash
set -euo pipefail

# Default to the in-memory testing settings unless the caller overrides them.
export TEST_MODE="${TEST_MODE:-true}"
export USE_MEMORY_STORE="${USE_MEMORY_STORE:-true}"
export ALLOW_REDIS_FALLBACK_DEV="${ALLOW_REDIS_FALLBACK_DEV:-true}"

python -m compileall liminallm
pytest "$@"
