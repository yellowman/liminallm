-- Pin knowledge_chunk.embedding to the encoder's dimension.
--
-- Databases created before this migration have a dimensionless VECTOR column,
-- on which pgvector refuses to build an ivfflat index ("column does not have
-- dimensions"). With ON_ERROR_STOP that aborted migrate.sh at 002; without it
-- the index silently never existed and every similarity search was a
-- sequential scan.
--
-- Set EMBEDDING_VECTOR_DIM to match the configured encoder (1536 for
-- text-embedding-3-small, 64 for the built-in hash fallback). Changing
-- encoders later means re-running this; the app already treats a vector from
-- another encoder as "not embedded", and the training worker's re-embed sweep
-- backfills what this removes.
\if :{?embedding_dim}
\else
\set embedding_dim 1536
\endif

-- Vectors of another size cannot be cast to the pinned type. They came from a
-- previous encoder and are unusable either way, so drop them (psql reports the
-- count) and let re-ingestion restore them.
DELETE FROM knowledge_chunk WHERE vector_dims(embedding) <> :embedding_dim;

ALTER TABLE knowledge_chunk
  ALTER COLUMN embedding TYPE vector(:embedding_dim);

DROP INDEX IF EXISTS knowledge_chunk_embedding_idx;
CREATE INDEX knowledge_chunk_embedding_idx ON knowledge_chunk
USING ivfflat (embedding) WITH (lists = 100);
