-- Artifact tables aligned to the SPEC kernel primitives
CREATE TABLE IF NOT EXISTS artifact (
  id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  owner_user_id   UUID REFERENCES app_user(id),
  type            TEXT NOT NULL,
  name            TEXT NOT NULL,
  description     TEXT,
  schema          JSONB NOT NULL,
  fs_path         TEXT,
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
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  meta            JSONB,
  UNIQUE (artifact_id, version)
);

CREATE TABLE IF NOT EXISTS config_patch (
  id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  artifact_id     UUID NOT NULL REFERENCES artifact(id) ON DELETE CASCADE,
  proposer_user_id UUID REFERENCES app_user(id),
  patch           JSONB NOT NULL,
  justification   TEXT,
  status          TEXT NOT NULL DEFAULT 'pending',
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  meta            JSONB
);
