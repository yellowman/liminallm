// Shared by chat.js and admin.js. Loaded first (plain script, `defer` keeps
// the order), so everything here is a global by the time either runs.
//
// It exists because these functions were copy-pasted between the two files and
// had already started to drift: `persistAuth` diverged, and `extractErrorDetails`
// was added to the console alone, so the chat UI silently kept throwing away the
// structured field errors the API sends. One copy is the only way that stops.

// --------------------------------------------------------------------------
// Escaping and identifiers
// --------------------------------------------------------------------------

const escapeHtml = (str) => {
  if (str == null) return '';
  const text = String(str);
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

const randomIdempotencyKey = () => {
  if (window.crypto?.randomUUID) return window.crypto.randomUUID();
  // Fallback using crypto.getRandomValues() - cryptographically secure, broader browser support
  if (window.crypto?.getRandomValues) {
    const bytes = new Uint8Array(16);
    window.crypto.getRandomValues(bytes);
    bytes[6] = (bytes[6] & 0x0f) | 0x40; // UUID v4 version
    bytes[8] = (bytes[8] & 0x3f) | 0x80; // UUID v4 variant
    const hex = Array.from(bytes, b => b.toString(16).padStart(2, '0')).join('');
    return `${hex.slice(0,8)}-${hex.slice(8,12)}-${hex.slice(12,16)}-${hex.slice(16,20)}-${hex.slice(20)}`;
  }
  // Ultimate fallback for ancient browsers without crypto support
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

// --------------------------------------------------------------------------
// Requests
// --------------------------------------------------------------------------

const getCsrfToken = () => {
  const match = document.cookie.match(/(?:^|;\s*)csrf_token=([^;]+)/);
  return match ? decodeURIComponent(match[1]) : null;
};

// `state` is defined by whichever page loaded this; both keep the same auth
// fields on it.
//
// No tenant header. The server derives the tenant from the hostname the request
// arrived at (service/tenancy.py), so a client cannot name its own. This used
// to echo back the value the server itself had just sent at login, where the
// only two outcomes were "matches" and "401".
const authHeaders = (idempotencyKey) => {
  const h = {};
  if (state.accessToken) h['Authorization'] = `Bearer ${state.accessToken}`;
  if (state.sessionId) h['session_id'] = state.sessionId;
  const csrf = getCsrfToken();
  if (csrf) h['X-CSRF-Token'] = csrf;
  h['Idempotency-Key'] = idempotencyKey || randomIdempotencyKey();
  return h;
};

const headers = (idempotencyKey) => ({
  'Content-Type': 'application/json',
  ...authHeaders(idempotencyKey),
});

// --------------------------------------------------------------------------
// Errors
// --------------------------------------------------------------------------

const extractError = (payload, fallback) => {
  const detail = payload?.detail || payload?.error || payload;
  if (typeof detail === 'string') return detail.trim() || fallback;
  if (detail?.message) return detail.message;
  if (detail?.error?.message) return detail.error.message;
  return fallback;
};

// The API puts structured problems (e.g. which settings were rejected and why)
// under error.details. Flattening a response to its message string throws that
// away, and it is exactly what a form needs to mark the offending field.
const extractErrorDetails = (payload) => {
  const detail = payload?.detail || payload;
  return detail?.error?.details ?? detail?.details ?? null;
};

// Carries the status and the structured details onto the Error, so a caller
// can react to either without re-parsing the response.
const envelopeError = (payload, raw, fallbackMessage) => {
  const error = new Error(
    extractError(payload ?? raw, fallbackMessage || 'Request failed')
  );
  error.details = extractErrorDetails(payload);
  return error;
};


// --------------------------------------------------------------------------
// The request layer
//
// Both pages had their own copy and both had drifted. The console's version
// treated 429 as a fatal client error instead of backing off, and never
// refreshed an expired token on 401 — so an admin session simply failed where
// a chat session recovered. These are the chat versions, which had both.
//
// `state`, `apiBase` and `persistAuth` come from whichever page loaded this.
// --------------------------------------------------------------------------

const jsonHeaders = () => {
  const h = { 'Content-Type': 'application/json' };
  const csrf = getCsrfToken();
  if (csrf) h['X-CSRF-Token'] = csrf;
  return h;
}

const fetchWithRetry = async (url, options, retries = 3, backoffMs = 400) => {
  let lastError;
  for (let attempt = 0; attempt <= retries; attempt += 1) {
    try {
      const resp = await fetch(url, options);
      // Handle 429 rate limit with exponential backoff
      if (resp.status === 429) {
        const retryAfter = resp.headers.get('Retry-After');
        const waitMs = retryAfter ? parseInt(retryAfter, 10) * 1000 : backoffMs * Math.pow(2, attempt + 2);
        lastError = new Error('Rate limit exceeded');
        if (attempt === retries) return resp;
        await new Promise((r) => setTimeout(r, Math.min(waitMs, 30000)));
        continue;
      }
      if (resp.status >= 400 && resp.status < 500) return resp;
      if (!resp.ok && resp.status >= 500) {
        lastError = new Error(`Server error: ${resp.status}`);
        if (attempt === retries) return resp;
        await new Promise((r) => setTimeout(r, backoffMs * Math.pow(2, attempt)));
        continue;
      }
      return resp;
    } catch (err) {
      lastError = err;
      if (attempt === retries) break;
      await new Promise((r) => setTimeout(r, backoffMs * Math.pow(2, attempt)));
    }
  }
  throw new Error(`Request failed after ${retries + 1} attempts: ${lastError?.message || 'unknown'}`);
}

const tryRefreshToken = async () => {
  if (!state.refreshToken) return false;
  try {
    const resp = await fetch(`${apiBase}/auth/refresh`, {
      method: 'POST',
      headers: jsonHeaders(),
      body: JSON.stringify({ refresh_token: state.refreshToken, tenant_id: state.tenantId }),
    });
    if (!resp.ok) return false;
    const envelope = await resp.json();
    if (envelope.data?.access_token) {
      persistAuth(envelope.data);
      return true;
    }
  } catch {
    // Token refresh failed
  }
  return false;
}

const requestEnvelope = async (url, options, fallbackMessage) => {
  let resp = await fetchWithRetry(url, options);

  if (resp.status === 401 && state.refreshToken) {
    const refreshed = await tryRefreshToken();
    if (refreshed) {
      const newOptions = { ...options };
      if (newOptions.headers) {
        newOptions.headers = { ...newOptions.headers, Authorization: `Bearer ${state.accessToken}` };
      }
      resp = await fetchWithRetry(url, newOptions);
    }
  }

  const text = await resp.text();
  let payload;
  if (text.trim()) {
    try {
      payload = JSON.parse(text);
    } catch {
      if (!resp.ok) throw new Error(fallbackMessage || resp.statusText || 'Request failed');
      throw new Error('Invalid JSON response');
    }
  }
  if (!resp.ok) {
    // envelopeError carries error.details through, so a caller can act on the
    // API's per-field problems instead of only its message string.
    const error = envelopeError(payload, text, fallbackMessage);
    error.status = resp.status;
    throw error;
  }
  return payload ?? {};
}
