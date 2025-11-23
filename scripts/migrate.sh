#!/usr/bin/env bash
set -euo pipefail

: "${DATABASE_URL:?DATABASE_URL must be set}"

# Apply numbered migration files in order.
for sql_file in $(ls sql/*.sql | sort); do
  echo "Applying ${sql_file}"
  psql "$DATABASE_URL" -f "$sql_file"
done

# Apply optional seed files if present.
if ls sql/seed/*.sql >/dev/null 2>&1; then
  echo "Applying seed files"
  for seed_file in $(ls sql/seed/*.sql | sort); do
    echo "Applying ${seed_file}"
    psql "$DATABASE_URL" -f "$seed_file"
  done
fi
