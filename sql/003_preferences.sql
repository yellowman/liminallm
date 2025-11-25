-- Preference events, semantic clusters, and training jobs
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;

-- Preference events capture explicit feedback with optional clustering context
CREATE TABLE IF NOT EXISTS preference_event (
  id                 UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id            UUID NOT NULL REFERENCES app_user(id) ON DELETE CASCADE,
  conversation_id    UUID NOT NULL REFERENCES conversation(id) ON DELETE CASCADE,
  message_id         UUID NOT NULL REFERENCES message(id) ON DELETE CASCADE,
  created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
  feedback           TEXT NOT NULL,
  explicit_signal    TEXT,
  score              DOUBLE PRECISION,
  context_embedding  VECTOR,
  context_text       TEXT,
  corrected_text     TEXT,
  cluster_id         UUID,
  weight             DOUBLE PRECISION DEFAULT 1.0,
  meta               JSONB
);

-- Semantic clusters for emergent skills/domains
CREATE TABLE IF NOT EXISTS semantic_cluster (
  id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id         UUID,
  centroid        VECTOR,
  size            INT NOT NULL,
  label           TEXT,
  description     TEXT,
  sample_message_ids UUID[],
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  meta            JSONB
);

-- Adapter routing state for centroids and usage
CREATE TABLE IF NOT EXISTS adapter_router_state (
  artifact_id     UUID PRIMARY KEY REFERENCES artifact(id) ON DELETE CASCADE,
  centroid_vec    VECTOR,
  usage_count     BIGINT NOT NULL DEFAULT 0,
  success_score   DOUBLE PRECISION DEFAULT 0.0,
  last_used_at    TIMESTAMPTZ,
  last_trained_at TIMESTAMPTZ,
  meta            JSONB
);

-- Training jobs generated from preference events
DO $$
BEGIN
  IF EXISTS (
    SELECT 1
    FROM information_schema.columns
    WHERE table_name = 'training_job'
      AND column_name = 'adapter_artifact_id'
  ) THEN
    ALTER TABLE training_job RENAME COLUMN adapter_artifact_id TO adapter_id;
  END IF;
END $$;

CREATE TABLE IF NOT EXISTS training_job (
  id                   UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  adapter_id           UUID NOT NULL REFERENCES artifact(id) ON DELETE CASCADE,
  user_id              UUID NOT NULL REFERENCES app_user(id) ON DELETE CASCADE,
  created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
  status               TEXT NOT NULL DEFAULT 'queued',
  num_events           INT,
  loss                 DOUBLE PRECISION,
  dataset_path         TEXT,
  new_version          INT,
  preference_event_ids UUID[],
  meta                 JSONB
);

-- MFA secrets for TOTP enrollment
CREATE TABLE IF NOT EXISTS user_mfa_secret (
  user_id     UUID PRIMARY KEY REFERENCES app_user(id) ON DELETE CASCADE,
  secret      TEXT NOT NULL,
  enabled     BOOLEAN NOT NULL DEFAULT FALSE,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
  meta        JSONB
);

-- Backfill MFA flags on sessions
ALTER TABLE auth_session
  ADD COLUMN IF NOT EXISTS mfa_required BOOLEAN NOT NULL DEFAULT FALSE,
  ADD COLUMN IF NOT EXISTS mfa_verified BOOLEAN NOT NULL DEFAULT FALSE;
