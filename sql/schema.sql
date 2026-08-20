-- liminallm schema (SPEC §2).
--
-- This file states the desired schema, not a step in a history. It is applied
-- by scripts/migrate.sh, which is the only thing that applies it and which
-- passes :embedding_dim from EMBEDDING_VECTOR_DIM.
--
-- Every statement here must be safe to execute repeatedly against every
-- database state the project supports: declarations are IF NOT EXISTS, and a
-- data-repair block must reach the same result whether it runs once or many
-- times. That property is what makes re-running the schema safe, so it is a
-- requirement on anything added to this file, not an observation about what
-- the file currently happens to contain.
--
-- If a schema change ever cannot be written that way, add an ordered
-- migration mechanism before shipping the change rather than weakening the
-- rule here.

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
  -- Written by role changes and email verification; its absence was invisible
  -- while tests ran on the in-memory store, and broke every admin user
  -- mutation against real Postgres.
  updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
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
-- The lexical half of hybrid retrieval (SPEC §2.5). 'simple' takes no
-- stemming and no language, so an identifier indexes as itself; the two-arg
-- to_tsvector is IMMUTABLE, which is what makes it usable in a generated
-- column. Stored rather than computed per query: the WHERE clause is served
-- by the index either way, but ts_rank in the ORDER BY has to tokenize every
-- matching row, and on a large context that was the dominant cost of the
-- channel — paid on every grounded chat turn.
ALTER TABLE knowledge_chunk
  ADD COLUMN IF NOT EXISTS content_tsv tsvector
  GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED;
CREATE INDEX IF NOT EXISTS knowledge_chunk_content_fts_idx ON knowledge_chunk
USING gin (content_tsv);
CREATE INDEX IF NOT EXISTS knowledge_chunk_fs_path_idx ON knowledge_chunk (fs_path);

-- Late interaction (SPEC §2.5): several vectors per chunk, compared at query
-- time by MaxSim. A pooled chunk vector has to average everything the chunk
-- says into one point; these keep the parts separate. Same encoder and so the
-- same width as knowledge_chunk.embedding — a segment vector is only ever
-- compared against a query vector from the same encoder.
CREATE TABLE IF NOT EXISTS knowledge_chunk_vector (
  id              BIGSERIAL PRIMARY KEY,
  chunk_id        BIGINT NOT NULL REFERENCES knowledge_chunk(id) ON DELETE CASCADE,
  segment_index   INT NOT NULL,
  content         TEXT NOT NULL,
  embedding       VECTOR(:embedding_dim) NOT NULL,
  meta            JSONB
);
CREATE INDEX IF NOT EXISTS knowledge_chunk_vector_embedding_idx ON knowledge_chunk_vector
USING ivfflat (embedding) WITH (lists = 100);
-- (chunk_id, segment_index) also serves every lookup by chunk_id alone, so
-- there is no separate index on the leading column. Late interaction makes
-- this the hottest write path in ingestion; a second btree to maintain per
-- segment row would be paid on every insert and read by nothing.
CREATE UNIQUE INDEX IF NOT EXISTS knowledge_chunk_vector_segment_idx
ON knowledge_chunk_vector (chunk_id, segment_index);
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

-- Long-lived bearer credentials for the served Responses API (SPEC §13.1).
-- Only a SHA-256 of the key is stored; the plaintext is shown once at
-- creation. Revocation is a tombstone so the row keeps its audit trail.
CREATE TABLE IF NOT EXISTS user_api_key (
  id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id       UUID NOT NULL REFERENCES app_user(id) ON DELETE CASCADE,
  name          TEXT NOT NULL DEFAULT '',
  key_hash      TEXT NOT NULL UNIQUE,
  prefix        TEXT NOT NULL,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  last_used_at  TIMESTAMPTZ,
  revoked_at    TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_user_api_key_user ON user_api_key(user_id);

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
-- Admin-managed settings live in instance_config under the name
-- 'system_settings'. The row is deliberately NOT seeded: the store merges
-- SYSTEM_SETTINGS_DEFAULTS (config.py) under whatever is stored, and the
-- runtime gives env vars precedence over defaults for keys the admin never
-- set. Seeding every default would make each one look like an explicit
-- admin override and silently outrank the operator's environment.


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


-- ===== from 008_implicit_context_identity.sql =====
-- One conversation, one implicit attachment context (SPEC §19.5 scopes an
-- attachment to the chat that received it; §22 puts Postgres across
-- replicas). Identity used to be "the first row a 500-row listing matched",
-- and creation was an unconditional INSERT — so two first attachments racing
-- produced two hidden contexts, and a later lookup found only one of them
-- while the other kept chunks nothing could reach.
--
-- Duplicates that already exist are merged rather than dropped: the losers'
-- chunks move to the oldest row, which is the one any earlier lookup would
-- have returned. Deleting a loser outright would take chunks the winner does
-- not have.
DO $$
DECLARE
  winner RECORD;
BEGIN
  FOR winner IN
    SELECT owner_user_id,
           meta ->> 'conversation_id' AS conversation_id,
           MIN(created_at::text || '|' || id::text) AS oldest
    FROM knowledge_context
    WHERE COALESCE((meta ->> 'auto')::boolean, false)
      AND meta ->> 'conversation_id' IS NOT NULL
    GROUP BY 1, 2
    HAVING COUNT(*) > 1
  LOOP
    UPDATE knowledge_chunk
    SET context_id = split_part(winner.oldest, '|', 2)::uuid
    WHERE context_id IN (
      SELECT id FROM knowledge_context
      WHERE COALESCE((meta ->> 'auto')::boolean, false)
        AND owner_user_id = winner.owner_user_id
        AND meta ->> 'conversation_id' = winner.conversation_id
        AND id <> split_part(winner.oldest, '|', 2)::uuid
    );
    DELETE FROM knowledge_context
    WHERE COALESCE((meta ->> 'auto')::boolean, false)
      AND owner_user_id = winner.owner_user_id
      AND meta ->> 'conversation_id' = winner.conversation_id
      AND id <> split_part(winner.oldest, '|', 2)::uuid;

    -- Moving the rows is not enough. Both contexts could hold the *same*
    -- generation — two concurrent first attachments of one file, where the
    -- second was a disk dedupe hit into a context that was nonetheless new —
    -- and the merge bypasses replace_chunks_for_path, which is what normally
    -- keeps one fs_path meaning one complete current generation. Duplicate
    -- copies also spend candidate slots that belong to other attachments.
    -- Segment vectors cascade with the chunk rows removed here.
    DELETE FROM knowledge_chunk kc
    USING knowledge_chunk keep
    WHERE kc.context_id = split_part(winner.oldest, '|', 2)::uuid
      AND keep.context_id = kc.context_id
      AND keep.fs_path IS NOT DISTINCT FROM kc.fs_path
      AND keep.chunk_index = kc.chunk_index
      AND keep.id < kc.id;
  END LOOP;
END $$;

CREATE UNIQUE INDEX IF NOT EXISTS knowledge_context_auto_conversation_idx
  ON knowledge_context (owner_user_id, (meta ->> 'conversation_id'))
  WHERE COALESCE((meta ->> 'auto')::boolean, false);


-- A conversation's implicit attachment index belongs to that conversation's
-- lifetime, and that relationship is relational state rather than a string in
-- JSON. `meta.conversation_id` could not be enforced, could not cascade, and
-- could not serialize against a deletion: an upload that validated the chat,
-- did its work, and inserted afterwards left an index behind for a chat that
-- had been deleted in between, with the attached file's text still in it.
--
-- The foreign key makes PostgreSQL the arbiter of that race. Either the
-- insert commits while the conversation exists and the later delete cascades
-- it away, or the delete commits first and the insert cannot satisfy its
-- reference. There is no third outcome and no cleanup pass to get right.
ALTER TABLE knowledge_context
  ADD COLUMN IF NOT EXISTS conversation_id UUID
  REFERENCES conversation(id) ON DELETE CASCADE;

-- Adopt the contexts that already carry the relationship in JSON. Guarded on
-- the text being a UUID and on the conversation existing, so this is safe to
-- run against any state, and it does nothing once every row is adopted.
UPDATE knowledge_context kc
SET conversation_id = (kc.meta ->> 'conversation_id')::uuid
WHERE kc.conversation_id IS NULL
  AND COALESCE((kc.meta ->> 'auto')::boolean, false)
  AND kc.meta ->> 'conversation_id' ~*
      '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
  AND EXISTS (
    SELECT 1 FROM conversation c
    WHERE c.id = (kc.meta ->> 'conversation_id')::uuid
  );

-- What the previous deletion path left behind. An implicit context whose
-- conversation is gone is unreachable — every lookup goes through the
-- conversation — while its chunks still hold the text of files attached to
-- that chat and still spend candidate slots belonging to other attachments.
-- Chunks and segment vectors cascade with the context rows.
DELETE FROM knowledge_context kc
WHERE kc.conversation_id IS NULL
  AND COALESCE((kc.meta ->> 'auto')::boolean, false)
  AND kc.meta ->> 'conversation_id' IS NOT NULL;

-- One conversation, one implicit index. Keyed on the real identity, so no
-- expression can be substituted for it and no extra key can make it unique
-- for free. Ordinary contexts have a NULL conversation_id and are unaffected.
DROP INDEX IF EXISTS knowledge_context_auto_conversation_idx;
CREATE UNIQUE INDEX IF NOT EXISTS knowledge_context_conversation_idx
  ON knowledge_context (conversation_id)
  WHERE conversation_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS knowledge_context_owner_ordinary_idx
  ON knowledge_context (owner_user_id)
  WHERE conversation_id IS NULL;
