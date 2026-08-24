# LiminalLM Configuration Architecture

This document clarifies the configuration sources and their intended uses.

## Configuration Sources

LiminalLM has three configuration sources, each serving a different purpose:

### 1. Environment Variables (`Settings` class)

**Location:** `liminallm/config.py`
**When loaded:** Application startup
**Mutability:** Immutable after startup

Reserved for what must be readable *before* the database is: `DATABASE_URL`,
plus `BUILD_SHA`, `TEST_MODE`, `EMBEDDING_VECTOR_DIM` and
`EXTRACT_READER_PLUGINS`, which are not settings. Provider credentials
(`OPENAI_API_KEY`, `GEMINI_API_KEY`, …) are read as a fallback when the
matching admin setting is blank, and `INSTANCE_SETTINGS_JSON` seeds managed
settings on first boot.

Nothing else is read from the environment. A variable that is not in that
list — `MODEL_BACKEND`, `REDIS_URL`, `SMTP_HOST` — is ignored.

```bash
# Example
export DATABASE_URL="postgresql://user:pass@localhost/liminallm"
export OPENAI_API_KEY="sk-..."
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

Almost everything lives in the database. A setting in an environment variable
is a value that differs per container, cannot be seen from inside the running
system, cannot be changed without a redeploy, and is invisible to the console
that claims to manage it. Restarting the application to change an SMTP
password is the wrong answer.

**Environment** — five variables, and only the first is configuration:

| Variable | Why it cannot live in the database |
|---|---|
| `DATABASE_URL` | It is how you reach the database |
| `BUILD_SHA` | Stamped by the build; provenance, not a setting |
| `TEST_MODE` | The harness tells the process it is a test before anything else, and it must not be flippable from a web form |
| `EMBEDDING_VECTOR_DIM` | A property of the schema that was applied; `scripts/migrate.sh` needs the same value |
| `EXTRACT_READER_PLUGINS` | Imports Python modules — settable from the console would mean remote code execution |

**Database** — everything else, declared with `managed_field` and edited from
the admin console.

**Database, write-only** — credentials, declared with `secret_field`: the SMTP
password, provider API keys, OAuth client secrets, and the JWT signing key.
Stored like any other setting so they can be rotated from the console, but
redacted from every read: `GET /admin/settings` returns them empty, the schema
endpoint reports only whether one is set, and a blank field on save means "not
retyped", never "erase it". The signing key generates itself on first boot —
which also fixed a quieter problem, since it used to be generated per process
into a file, so a replica that could not read that file rejected tokens the
others had issued.

```
managed_field default  ->  what an admin saved  ->  runtime.settings
     (shipped)              (instance_config)        (what code reads)
```

`Runtime.refresh_settings()` builds that overlay once. Request handlers read
plain attributes off `runtime.settings`; they do not merge dicts and they do
not query the database per request. The settings watcher re-runs the overlay
when another worker writes a change, so an edit reaches every replica within
`settings_watch_interval_seconds`. Services that capture configuration at
construction — the mailer and the voice service — are rebuilt, because handing
them a new settings object would not change the mail they send.

Nothing seeds the defaults into the database. A stored value means an admin
chose it; a shipped default written to the table would be indistinguishable
from a choice and would silently outrank everything else.

## Adding a setting

Declare it once, on `Settings` in `liminallm/config.py`:

```python
my_setting: int = managed_field(30, ge=1, le=600, description="What it does")
```

The bounds go on the field, not in the code that reads it: the admin API
validates against them, and the console renders a control that already
enforces them. The console builds itself from
`GET /v1/admin/settings/schema`, so there is nothing to add to the frontend.

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

### Declare a secret as a secret, not as an ordinary setting
`secret_field` stores the value in the database like any other managed
setting — so an operator can rotate a key without a redeploy — but redacts it
from every read path: `GET /admin/settings` returns an empty string and the
console renders a write-only control.

```python
# BAD: an ordinary managed_field echoes its value back to every admin
openai_api_key: str = managed_field("", description="...")

# GOOD
openai_api_key: str = secret_field(description="... Write-only.")
```

Bootstrap secrets are the exception: `DATABASE_URL` is needed before the
database can be read, so it stays outside it. `jwt_secret` is not one of
them — it is generated on first boot and stored in `instance_config` like
any other secret; a `JWT_SECRET` environment variable reaches nothing.
