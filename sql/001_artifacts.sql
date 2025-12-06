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
