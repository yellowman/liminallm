# Security Audit: JWT tenant_id Handling

**Audit Date:** 2025-12-02
**Scope:** All components handling tenant_id in authentication flows

## Executive Summary

The codebase follows secure practices for tenant_id handling. The authenticated tenant_id is always derived from the database user record after JWT/session validation, never from request parameters. Pre-authentication flows (signup, login, OAuth) appropriately accept tenant_id to determine tenant context.

## Components Audited

| Component | Location | Status |
|-----------|----------|--------|
| Python Backend (FastAPI) | `liminallm/` | Secure |
| Chat Frontend | `frontend/chat.js` | Secure |
| Admin Frontend | `frontend/admin.js` | Secure |

## Detailed Findings

### Python Backend

#### Authentication Flow (SECURE)

The `AuthService` properly derives tenant_id from the authenticated user record:

1. **JWT Token Authentication** (`liminallm/service/auth.py:1090-1129`)
   - Decodes JWT and extracts session_id
   - Loads user from database by `sub` claim
   - Validates `payload.tenant_id` matches `user.tenant_id`
   - Returns `AuthContext` with `tenant_id=user.tenant_id` from database

2. **Session Authentication** (`liminallm/service/auth.py:637-680`)
   - Loads session from database by session_id
   - Loads user from database by `session.user_id`
   - Returns `AuthContext` with `tenant_id=user.tenant_id` from database

#### Authenticated Endpoints (SECURE)

All authenticated endpoints use `principal.tenant_id` or `auth_ctx.tenant_id`:

- `POST /chat` - Uses `principal.tenant_id` (line 1421)
- `POST /tools/{tool_id}/invoke` - Uses `principal.tenant_id` (line 1962)
- `WebSocket /chat/stream` - Uses `auth_ctx.tenant_id` (lines 2947, 3043)
- `GET /admin/users` - Validates request tenant_id matches principal (lines 759-763)
- `POST /admin/users` - Validates request tenant_id matches principal (lines 781-785)

#### Pre-Authentication Flows (EXPECTED BEHAVIOR)

These endpoints accept tenant_id from request body to determine tenant context:

| Endpoint | Purpose |
|----------|---------|
| `POST /auth/signup` | Determines which tenant new user joins |
| `POST /auth/login` | Authenticates against specific tenant |
| `POST /auth/oauth/{provider}/start` | Routes OAuth to specific tenant |
| `GET /auth/oauth/{provider}/callback` | Completes OAuth for specific tenant |
| `POST /auth/refresh` | Uses tenant_hint for session lookup optimization |

#### X-Tenant-ID Header (INFORMATIONAL)

The `X-Tenant-ID` header is used as a "hint" for session resolution performance but does NOT grant authorization. The server validates:
- If tenant_hint doesn't match user.tenant_id, authentication fails
- Authorization always uses the database user record's tenant_id

### JavaScript Frontends

#### Chat Frontend (`frontend/chat.js`)

**SECURE** - Receives tenant_id from server, stores in state, sends back in requests:

1. Login response sets `state.tenantId = payload.tenant_id` (line 533)
2. Subsequent requests include `X-Tenant-ID: state.tenantId` (line 294)
3. WebSocket messages include `tenant_id: state.tenantId` (line 1571)

The frontend cannot spoof tenant_id because the server validates against the authenticated user record.

#### Admin Frontend (`frontend/admin.js`)

**SECURE** - Same pattern as chat frontend:

1. Login response sets `state.tenantId = payload.tenant_id` (line 111)
2. Subsequent requests include `X-Tenant-ID: state.tenantId` (line 55)

## Documentation Note

**SPEC.md line 1620** documents WebSocket messages accepting `tenant_id`, but the implementation correctly ignores this field and uses `auth_ctx.tenant_id` from authentication. Consider updating SPEC to clarify this is optional/ignored for security.

## Recommendations

### Completed

1. Added security guideline to CLAUDE.md requiring tenant_id derivation from authenticated JWT

### Future Considerations

1. **Consider removing tenant_id from WebSocket message spec** - It's not used and may confuse developers
2. **Consider removing X-Tenant-ID header support** - If not needed for performance, removing it reduces attack surface
3. **Add integration tests** - Verify tenant isolation by attempting cross-tenant access

## Conclusion

No security vulnerabilities found. The codebase correctly implements tenant isolation by always deriving the authoritative tenant_id from the authenticated user's database record, never from client-provided values in authenticated contexts.
