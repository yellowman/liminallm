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
