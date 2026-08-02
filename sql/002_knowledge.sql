-- Knowledge context and chunk tables for RAG
CREATE EXTENSION IF NOT EXISTS vector;

-- pgvector requires a fixed dimension to build an ivfflat index; a bare
-- VECTOR column fails with "column does not have dimensions". The size
-- must match the configured encoder (EMBEDDING_VECTOR_DIM: 1536 for
-- text-embedding-3-small, 64 for the built-in hash fallback).
\if :{?embedding_dim}
\else
\set embedding_dim 1536
\endif
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
