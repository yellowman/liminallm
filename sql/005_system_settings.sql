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
    "rate_limit_multiplier_enterprise": 5.0
  }'::jsonb,
  now(),
  now()
)
ON CONFLICT (name) DO NOTHING;
