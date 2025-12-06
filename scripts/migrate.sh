#!/usr/bin/env bash
set -euo pipefail

: "${DATABASE_URL:?DATABASE_URL must be set}"

# Change to script directory for consistent relative paths
cd "$(dirname "$0")/.."

# Apply numbered migration files in order (using glob pattern, sorted by filename)
shopt -s nullglob
sql_files=(sql/*.sql)
shopt -u nullglob

if [[ ${#sql_files[@]} -eq 0 ]]; then
  echo "No migration files found in sql/"
  exit 0
fi

# Sort files by name
IFS=$'\n' sorted_files=($(sort <<<"${sql_files[*]}"))
unset IFS

for sql_file in "${sorted_files[@]}"; do
  echo "Applying ${sql_file}"
  psql "$DATABASE_URL" -f "$sql_file"
done

# Apply optional seed files if present (using glob pattern)
shopt -s nullglob
seed_files=(sql/seed/*.sql)
shopt -u nullglob

if [[ ${#seed_files[@]} -gt 0 ]]; then
  echo "Applying seed files"
  IFS=$'\n' sorted_seeds=($(sort <<<"${seed_files[*]}"))
  unset IFS
  for seed_file in "${sorted_seeds[@]}"; do
    echo "Applying ${seed_file}"
    psql "$DATABASE_URL" -f "$seed_file"
  done
fi
