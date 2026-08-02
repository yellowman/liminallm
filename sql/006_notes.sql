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
