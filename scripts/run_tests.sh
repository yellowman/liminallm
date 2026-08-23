#!/usr/bin/env bash
set -euo pipefail

# The suite starts its own throwaway Postgres (tests/pgharness.py). Point
# TEST_DATABASE_URL at an existing database to use that instead.
export TEST_MODE="${TEST_MODE:-true}"

python -m compileall liminallm
pytest "$@"
