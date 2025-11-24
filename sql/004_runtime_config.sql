-- Runtime configuration overrides
CREATE TABLE IF NOT EXISTS instance_config (
  name        TEXT PRIMARY KEY,
  config      JSONB NOT NULL,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
