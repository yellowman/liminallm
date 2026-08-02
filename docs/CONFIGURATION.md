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

## Where a setting comes from

Two kinds of setting, and the distinction is the whole design:

**Environment** — secrets and bootstrap, nothing else. API keys, the JWT
signing key, `DATABASE_URL`, and things that describe the machine rather than
the install (`SHARED_FS_ROOT`, `EMBEDDING_VECTOR_DIM`, `TEST_MODE`). Declared
with `env_field`. Security controls live here too, deliberately: CORS, HSTS,
the SSRF allowlist, the SMTP downgrade flag. Those decide who may reach the
instance and who it will talk to, and an admin session should not be able to
widen them from a web form.

**Database** — everything else. Declared with `managed_field`, edited from the
admin console, and identical across replicas. Configuration that *can* live in
the database belongs there: it is auditable, it changes without a restart, and
one container cannot quietly disagree with another.

```
managed_field default  ->  what an admin saved  ->  runtime.settings
     (shipped)              (instance_config)        (what code reads)
```

`Runtime.refresh_settings()` builds that overlay once. Request handlers read
plain attributes off `runtime.settings`; they do not merge dicts and they do
not query the database per request. The settings watcher re-runs the overlay
when another worker writes a change, so a value edited in the admin UI reaches
every replica within `settings_watch_interval_seconds`.

Nothing seeds the defaults into the database. A stored value means an admin
chose it; a shipped default that had been written to the table would be
indistinguishable from a choice, and would silently outrank anything else.

### Declarative deploys

Managed settings have no environment variables, which would leave a
compose/k8s deploy unable to configure anything without a human opening the
admin UI. `INSTANCE_SETTINGS_JSON` is the one seam:

```yaml
environment:
  INSTANCE_SETTINGS_JSON: '{"model_backend": "stub", "notes_enabled": false}'
```

It is a *seed*, not an override: it is applied only when no admin has saved
anything yet. Once a value is in the database it wins, so a stale container
env cannot quietly revert what an operator changed. Unknown keys are logged
and dropped, and a malformed value never blocks boot.

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

1. Declare it once, on `Settings` in `liminallm/config.py`:

   ```python
   my_setting: int = env_field(30, "MY_SETTING", admin=True)
   ```

   `admin=True` puts it in the admin UI and flows its default into
   `SYSTEM_SETTINGS_DEFAULTS` automatically. Do not also write the default into
   that dict — the whole point is that there is one value. A setting with no
   env var (an operational limit tuned only from the UI) goes in
   `_ADMIN_ONLY_DEFAULTS` instead.

   Precedence at read time is: admin override > env var > declared default.
   Nothing seeds the defaults into the database, so a shipped default never
   masquerades as something an admin chose.

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
