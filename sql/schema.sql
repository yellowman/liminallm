-- liminallm schema (SPEC §2).
--
-- One file, applied once. There is no migration history: this project has
-- never been deployed, so numbered migrations only carried the fiction of an
-- upgrade path that never existed — and with it a real hazard, since the
-- pgvector dimension change had to guess what a "previous encoder" left
-- behind. A single schema has no previous state to reconcile.
--
-- Apply with scripts/migrate.sh (which passes :embedding_dim from
-- EMBEDDING_VECTOR_DIM). Every statement is IF NOT EXISTS / idempotent, so
-- re-running is safe.

-- pgvector needs a fixed dimension to build an ivfflat index; a bare VECTOR
-- column fails with "column does not have dimensions". This must match the
-- configured encoder: 1536 for text-embedding-3-small, 64 for the built-in
-- hash fallback (EMBEDDING_VECTOR_DIM).
\if :{?embedding_dim}
\else
\set embedding_dim 1536
\endif


-- ===== from 000_base.sql =====
-- Core user and chat tables from SPEC phase 0
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE EXTENSION IF NOT EXISTS citext;

CREATE TABLE IF NOT EXISTS app_user (
  id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  email           CITEXT UNIQUE NOT NULL,
  handle          TEXT,
  role            TEXT NOT NULL DEFAULT 'user',
  tenant_id       TEXT NOT NULL DEFAULT 'public',
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  plan_tier       TEXT NOT NULL DEFAULT 'free',
  is_active       BOOLEAN NOT NULL DEFAULT TRUE,
  meta            JSONB
);

CREATE TABLE IF NOT EXISTS user_auth_credential (
  user_id         UUID PRIMARY KEY REFERENCES app_user(id) ON DELETE CASCADE,
  password_hash   TEXT,
  password_algo   TEXT,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  last_updated_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS user_auth_provider (
  id              BIGSERIAL PRIMARY KEY,
  user_id         UUID NOT NULL REFERENCES app_user(id) ON DELETE CASCADE,
  provider        TEXT NOT NULL,
  provider_uid    TEXT NOT NULL,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (provider, provider_uid)
);

CREATE TABLE IF NOT EXISTS user_settings (
  user_id         UUID PRIMARY KEY REFERENCES app_user(id) ON DELETE CASCADE,
  locale          TEXT,
  timezone        TEXT,
  default_voice   TEXT,
  default_style   JSONB,
  flags           JSONB
);

CREATE TABLE IF NOT EXISTS auth_session (
  id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id         UUID NOT NULL REFERENCES app_user(id) ON DELETE CASCADE,
  tenant_id       TEXT NOT NULL DEFAULT 'public',
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  expires_at      TIMESTAMPTZ NOT NULL,
  user_agent      TEXT,
  ip_addr         INET,
  mfa_required    BOOLEAN NOT NULL DEFAULT FALSE,
  mfa_verified    BOOLEAN NOT NULL DEFAULT FALSE,
  meta            JSONB
);

CREATE TABLE IF NOT EXISTS conversation (
  id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id         UUID NOT NULL REFERENCES app_user(id) ON DELETE CASCADE,
  title           TEXT,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  status          TEXT NOT NULL DEFAULT 'open',
  active_context_id UUID,  -- FK added in 002_knowledge.sql after knowledge_context exists
  meta            JSONB
);

CREATE TABLE IF NOT EXISTS message (
  id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  conversation_id UUID NOT NULL REFERENCES conversation(id) ON DELETE CASCADE,
  sender          TEXT NOT NULL,
  role            TEXT NOT NULL,
  content         TEXT NOT NULL,
  content_struct  JSONB,
  seq             INT NOT NULL,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  token_count_in  INT,
  token_count_out INT,
  meta            JSONB,
  UNIQUE (conversation_id, seq)
);

CREATE INDEX IF NOT EXISTS idx_app_user_tenant_id ON app_user(tenant_id);
CREATE INDEX IF NOT EXISTS idx_auth_session_user_id ON auth_session(user_id);
CREATE INDEX IF NOT EXISTS idx_auth_session_tenant_id ON auth_session(tenant_id);
CREATE INDEX IF NOT EXISTS idx_conversation_user_id ON conversation(user_id);
CREATE INDEX IF NOT EXISTS idx_message_conversation_id ON message(conversation_id);


-- ===== from 001_artifacts.sql =====
-- Artifact tables aligned to the SPEC kernel primitives
CREATE TABLE IF NOT EXISTS artifact (
  id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  owner_user_id   UUID REFERENCES app_user(id) ON DELETE CASCADE,
  type            TEXT NOT NULL,
  name            TEXT NOT NULL,
  description     TEXT,
  schema          JSONB NOT NULL,
  fs_path         TEXT,
  base_model      TEXT,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  visibility      TEXT NOT NULL DEFAULT 'private',
  meta            JSONB
);

CREATE TABLE IF NOT EXISTS artifact_version (
  id              BIGSERIAL PRIMARY KEY,
  artifact_id     UUID NOT NULL REFERENCES artifact(id) ON DELETE CASCADE,
  version         INT NOT NULL,
  schema          JSONB NOT NULL,
  fs_path         TEXT,
  base_model      TEXT,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  created_by      TEXT NOT NULL,
  change_note     TEXT,
  meta            JSONB,
  UNIQUE (artifact_id, version)
);

ALTER TABLE artifact_version
  ADD COLUMN IF NOT EXISTS change_note TEXT;
ALTER TABLE artifact
  ADD COLUMN IF NOT EXISTS base_model TEXT;
ALTER TABLE artifact_version
  ADD COLUMN IF NOT EXISTS base_model TEXT;

CREATE INDEX IF NOT EXISTS idx_artifact_owner_user_id ON artifact(owner_user_id);
CREATE INDEX IF NOT EXISTS idx_artifact_type ON artifact(type);
CREATE INDEX IF NOT EXISTS idx_artifact_kind ON artifact((schema->>'kind'));
CREATE INDEX IF NOT EXISTS idx_artifact_visibility ON artifact(visibility);
CREATE INDEX IF NOT EXISTS idx_artifact_owner_visibility ON artifact(owner_user_id, visibility);

CREATE TABLE IF NOT EXISTS config_patch (
  id              BIGSERIAL PRIMARY KEY,
  artifact_id     UUID NOT NULL REFERENCES artifact(id) ON DELETE CASCADE,
  proposer        TEXT NOT NULL,
  patch           JSONB NOT NULL,
  justification   TEXT,
  status          TEXT NOT NULL DEFAULT 'pending',
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  decided_at      TIMESTAMPTZ,
  applied_at      TIMESTAMPTZ,
  meta            JSONB
);


-- ===== from 002_knowledge.sql =====
-- Knowledge context and chunk tables for RAG
CREATE EXTENSION IF NOT EXISTS vector;

-- pgvector requires a fixed dimension to build an ivfflat index; a bare
-- VECTOR column fails with "column does not have dimensions". The size
-- must match the configured encoder (EMBEDDING_VECTOR_DIM: 1536 for
-- text-embedding-3-small, 64 for the built-in hash fallback).
CREATE TABLE IF NOT EXISTS knowledge_context (
  id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  owner_user_id   UUID NOT NULL REFERENCES app_user(id) ON DELETE CASCADE,
  name            TEXT NOT NULL,
  description     TEXT,
  fs_path         TEXT,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  meta            JSONB
);

CREATE TABLE IF NOT EXISTS context_source (
  id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  context_id      UUID NOT NULL REFERENCES knowledge_context(id) ON DELETE CASCADE,
  fs_path         TEXT NOT NULL,
  recursive       BOOLEAN NOT NULL DEFAULT TRUE,
  meta            JSONB
);

CREATE TABLE IF NOT EXISTS knowledge_chunk (
  id              BIGSERIAL PRIMARY KEY,
  context_id      UUID NOT NULL REFERENCES knowledge_context(id) ON DELETE CASCADE,
  fs_path         TEXT NOT NULL,
  chunk_index     INT NOT NULL,
  content         TEXT NOT NULL,
  embedding       VECTOR(:embedding_dim) NOT NULL,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  meta            JSONB
);

CREATE INDEX IF NOT EXISTS knowledge_chunk_context_idx ON knowledge_chunk (context_id);
CREATE INDEX IF NOT EXISTS knowledge_chunk_embedding_idx ON knowledge_chunk
USING ivfflat (embedding) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS knowledge_chunk_fs_path_idx ON knowledge_chunk (fs_path);
CREATE INDEX IF NOT EXISTS knowledge_chunk_context_chunk_idx ON knowledge_chunk (context_id, chunk_index);

-- Add FK constraint for conversation.active_context_id now that knowledge_context exists
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.table_constraints
    WHERE constraint_name = 'conversation_active_context_id_fkey'
      AND table_name = 'conversation'
  ) THEN
    ALTER TABLE conversation
      ADD CONSTRAINT conversation_active_context_id_fkey
      FOREIGN KEY (active_context_id) REFERENCES knowledge_context(id) ON DELETE SET NULL;
  END IF;
END $$;


-- ===== from 003_preferences.sql =====
-- Preference events, semantic clusters, and training jobs
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;

-- Semantic clusters for emergent skills/domains
-- user_id is nullable to allow global clusters per SPEC §2.4
-- MUST be created before preference_event which references it
CREATE TABLE IF NOT EXISTS semantic_cluster (
  id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id         UUID REFERENCES app_user(id) ON DELETE CASCADE,
  centroid        VECTOR,
  size            INT NOT NULL,
  label           TEXT,
  description     TEXT,
  sample_message_ids UUID[],
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  meta            JSONB
);

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
  cluster_id         UUID REFERENCES semantic_cluster(id) ON DELETE SET NULL,
  weight             DOUBLE PRECISION DEFAULT 1.0,
  meta               JSONB
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

CREATE INDEX IF NOT EXISTS idx_preference_event_user_id ON preference_event(user_id);
CREATE INDEX IF NOT EXISTS idx_preference_event_conversation_id ON preference_event(conversation_id);
CREATE INDEX IF NOT EXISTS idx_preference_event_cluster_id ON preference_event(cluster_id);
CREATE INDEX IF NOT EXISTS idx_training_job_user_id ON training_job(user_id);
CREATE INDEX IF NOT EXISTS idx_training_job_status ON training_job(status);
CREATE INDEX IF NOT EXISTS idx_training_job_adapter_id ON training_job(adapter_id);

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


-- ===== from 004_runtime_config.sql =====
-- Runtime configuration overrides
CREATE TABLE IF NOT EXISTS instance_config (
  name        TEXT PRIMARY KEY,
  config      JSONB NOT NULL,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);


-- ===== from 005_system_settings.sql =====
-- System settings for admin-managed configuration
-- These settings can be modified via the admin UI

-- Insert default system settings if not present
INSERT INTO instance_config (name, config, created_at, updated_at)
VALUES (
  'system_settings',
  '{
    "session_rotation_hours": 24,
    "session_rotation_grace_seconds": 300,
    "max_concurrent_workflows": 3,
    "max_concurrent_inference": 2,
    "rate_limit_multiplier_free": 1.0,
    "rate_limit_multiplier_paid": 2.0,
    "rate_limit_multiplier_enterprise": 5.0,
    "chat_rate_limit_per_minute": 60,
    "chat_rate_limit_window_seconds": 60,
    "login_rate_limit_per_minute": 10,
    "signup_rate_limit_per_minute": 5,
    "reset_rate_limit_per_minute": 5,
    "mfa_rate_limit_per_minute": 5,
    "admin_rate_limit_per_minute": 30,
    "admin_rate_limit_window_seconds": 60,
    "files_upload_rate_limit_per_minute": 10,
    "configops_rate_limit_per_hour": 30,
    "read_rate_limit_per_minute": 120,
    "default_page_size": 100,
    "max_page_size": 500,
    "default_conversations_limit": 50,
    "max_upload_bytes": 10485760,
    "rag_chunk_size": 400,
    "access_token_ttl_minutes": 30,
    "refresh_token_ttl_minutes": 1440,
    "enable_mfa": true,
    "allow_signup": true,
    "training_worker_enabled": true,
    "training_worker_poll_interval": 60,
    "smtp_host": "",
    "smtp_port": 587,
    "smtp_user": "",
    "smtp_password": "",
    "smtp_use_tls": true,
    "smtp_allow_insecure": false,
    "email_from_address": "",
    "email_from_name": "LiminalLM",
    "oauth_redirect_uri": "",
    "app_base_url": "http://localhost:8000",
    "voice_transcription_model": "whisper-1",
    "voice_synthesis_model": "tts-1",
    "voice_default_voice": "alloy",
    "rag_mode": "pgvector",
    "embedding_model_id": "text-embedding",
    "default_tenant_id": "public",
    "jwt_issuer": "liminallm",
    "jwt_audience": "liminal-clients",
    "model_path": "gpt-4o-mini",
    "model_backend": "openai",
    "default_adapter_mode": "hybrid"
  }'::jsonb,
  now(),
  now()
)
ON CONFLICT (name) DO NOTHING;


-- ===== from 006_notes.sql =====
-- Notes vault: linked notes with a graph the model can witness.
-- Embeddings live in JSONB (cosine computed in the app) so the vault works on
-- Postgres installs without pgvector; at personal-vault scale (~10k notes)
-- that trade is invisible and it keeps this migration dependency-free.

CREATE TABLE IF NOT EXISTS note (
  id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id         UUID NOT NULL REFERENCES app_user(id) ON DELETE CASCADE,
  title           TEXT NOT NULL,
  content         TEXT NOT NULL DEFAULT '',
  embedding       JSONB,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  meta            JSONB
);

-- Titles are the link namespace ([[Title]] must resolve to one note), so they
-- are unique per user, case-insensitively.
CREATE UNIQUE INDEX IF NOT EXISTS idx_note_user_title
  ON note (user_id, lower(title));
CREATE INDEX IF NOT EXISTS idx_note_user_updated
  ON note (user_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS note_link (
  src_note_id     UUID NOT NULL REFERENCES note(id) ON DELETE CASCADE,
  dst_note_id     UUID NOT NULL REFERENCES note(id) ON DELETE CASCADE,
  PRIMARY KEY (src_note_id, dst_note_id)
);

CREATE INDEX IF NOT EXISTS idx_note_link_dst ON note_link (dst_note_id);


-- ===== from 007_sweep_reports.sql =====
-- Witness sweep reports (SPEC §19.6). A sweep costs up to 30 model calls;
-- keeping the result turns it into a record ("what moved this year"), lets the
-- UI replay the last one for free, and lets a future scheduled sweep diff
-- against the previous run instead of re-judging unchanged pairs.
CREATE TABLE IF NOT EXISTS sweep_report (
  id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id         UUID NOT NULL REFERENCES app_user(id) ON DELETE CASCADE,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  report          JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sweep_report_user_created
  ON sweep_report (user_id, created_at DESC);
