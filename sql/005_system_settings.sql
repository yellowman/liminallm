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
    "rag_chunk_size": 400
  }'::jsonb,
  now(),
  now()
)
ON CONFLICT (name) DO NOTHING;
