#!/usr/bin/env bash
# Apply the liminallm schema (sql/schema.sql).
#
# The only thing that applies the schema. CI runs this same script, and then
# runs the test suite against the database it produced.
#
# Not a migration runner: there is no migration history. sql/schema.sql states
# the desired schema, and every operation in it is required to be safe to
# execute repeatedly, so this is safe to re-run against an existing database.
#
# Repeat-safe is not the same as inert. A declaration that already exists is
# skipped, which is why a second executor is dangerous: Docker used to also
# mount sql/ into the postgres image's /docker-entrypoint-initdb.d, applying
# the file first and without :embedding_dim. This run then found everything
# present, changed nothing, and reported success over a schema built at the
# wrong vector width.
set -euo pipefail

: "${DATABASE_URL:?DATABASE_URL must be set}"

cd "$(dirname "$0")/.."

# embedding_dim must match the configured encoder (EMBEDDING_VECTOR_DIM):
# pgvector cannot build an ivfflat index on a dimensionless column, and a
# vector of the wrong size is rejected on insert.
echo "Applying sql/schema.sql (embedding_dim=${EMBEDDING_VECTOR_DIM:-1536})"
psql "$DATABASE_URL" -v ON_ERROR_STOP=1 \
  -v embedding_dim="${EMBEDDING_VECTOR_DIM:-1536}" \
  --single-transaction -f sql/schema.sql

# Optional seed data, if any is present.
shopt -s nullglob
seed_files=(sql/seed/*.sql)
shopt -u nullglob
for seed_file in "${seed_files[@]}"; do
  echo "Seeding ${seed_file}"
  psql "$DATABASE_URL" -v ON_ERROR_STOP=1 --single-transaction -f "$seed_file"
done

echo "Schema applied."
