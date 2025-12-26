# LiminalLM Configuration Architecture

This document clarifies the configuration sources and their intended uses.

## Configuration Sources

LiminalLM has three configuration sources, each serving a different purpose:

### 1. Environment Variables (`Settings` class)

**Location:** `liminallm/config.py`
**When loaded:** Application startup
**Mutability:** Immutable after startup

Used for:
- Secrets (JWT_SECRET, API keys, SMTP passwords)
- Infrastructure URLs (DATABASE_URL, REDIS_URL)
- Development flags (TEST_MODE, ALLOW_REDIS_FALLBACK_DEV)
- Default values for settings not in database

```bash
# Example
export JWT_SECRET="your-secure-secret-here"
export DATABASE_URL="postgresql://user:pass@localhost/liminallm"
export MODEL_BACKEND="openai"
```

### 2. System Settings (Admin-Managed)

**Storage:** Database `instance_config` table, name='system_settings'
**API:** `GET/PUT /v1/admin/settings`
**UI:** Admin Console → Settings tab

Used for operational settings that admins can modify at runtime:
- Rate limits (chat, auth, file uploads)
- Concurrency limits (workflows, inference)
- Session configuration (rotation, grace periods)
- Pagination defaults
- Feature flags (MFA, signup)
- SMTP configuration

```python
# Backend access
settings = runtime.store.get_system_settings()
rate_limit = settings.get("chat_rate_limit_per_minute", 60)
```

### 3. Runtime Config (Deployment Config)

**Storage:** Database `instance_config` table, name='default'
**API:** `GET /v1/config` (read-only)
**UI:** Admin Console → displays as JSON

Used for deployment-level configuration that's visible but not directly editable:
- Model path and backend
- Adapter mode
- RAG mode

## Resolution Order

When the runtime initializes, settings are resolved in this order:

1. **Database (system_settings)** - highest priority
2. **Environment variables** - fallback if not in database

```python
# Example from runtime.py
model_path = sys_settings.get("model_path") or self.settings.model_path
```

## Admin UI Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                     Admin Console                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │   Settings   │  │   Config     │  │   Patches    │       │
│  │   (editable) │  │   (readonly) │  │  (workflow)  │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
│         │                │                  │                │
│         ▼                ▼                  ▼                │
│  PUT /admin/settings  GET /config    POST /config/patches   │
│         │                │                  │                │
│         ▼                ▼                  ▼                │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              instance_config table                   │    │
│  │  ┌─────────────────┐  ┌─────────────────┐           │    │
│  │  │ system_settings │  │    default      │           │    │
│  │  └─────────────────┘  └─────────────────┘           │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Adding New Settings

### To add a runtime-configurable setting:

1. Add to `get_system_settings()` defaults in both:
   - `liminallm/storage/memory.py`
   - `liminallm/storage/postgres.py`

2. Use it in code with fallback:
   ```python
   sys_settings = runtime.store.get_system_settings()
   my_setting = sys_settings.get("my_setting") or self.settings.my_setting
   ```

3. Add UI controls in `frontend/admin.js`:
   - Add input field ID in `fetchSystemSettings()`
   - Add to settings object in `saveSystemSettings()`

### To add an environment-only setting:

1. Add field to `Settings` class in `liminallm/config.py`
2. Document in `.env.example`
3. Use via `get_settings().my_setting`

## Common Pitfalls

### Don't mix sources
```python
# BAD: Using env var when DB setting exists
rate_limit = self.settings.chat_rate_limit_per_minute

# GOOD: Check DB first, then fall back
rate_limit = sys_settings.get("chat_rate_limit_per_minute") or self.settings.chat_rate_limit_per_minute
```

### Don't cache settings
```python
# BAD: Caching at module level
RATE_LIMIT = get_settings().chat_rate_limit_per_minute

# GOOD: Read at runtime
def get_rate_limit():
    return runtime.store.get_system_settings().get("chat_rate_limit_per_minute", 60)
```

### Don't store secrets in system_settings
```python
# BAD: Putting API keys in system_settings (visible in admin UI)
sys_settings["openai_api_key"] = "sk-..."

# GOOD: Keep secrets in environment variables
os.environ.get("OPENAI_API_KEY")
```
