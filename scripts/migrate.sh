#!/usr/bin/env bash
set -euo pipefail

: "${DATABASE_URL:?DATABASE_URL must be set}"

psql "$DATABASE_URL" -f sql/000_base.sql
psql "$DATABASE_URL" -f sql/001_artifacts.sql
psql "$DATABASE_URL" -f sql/002_knowledge.sql
