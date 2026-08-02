const apiBase = '/v1';
const authForm = document.getElementById('admin-auth-form');
const consolePanel = document.getElementById('admin-console');
const loginPanel = document.getElementById('admin-auth-panel');
const feedbackEl = document.getElementById('admin-feedback');
const errorEl = document.getElementById('admin-error');
const sessionIndicator = document.getElementById('admin-session-indicator');
const patchTableWrapper = document.getElementById('patch-table-wrapper');
const userTableWrapper = document.getElementById('user-table-wrapper');
const adapterTableWrapper = document.getElementById('adapter-table-wrapper');
const inspectOutput = document.getElementById('inspect-output');
const createdPasswordEl = document.getElementById('created-user-password');
const runtimeConfigEl = document.getElementById('runtime-config');
const decisionStatusInput = document.getElementById('decision-status');
const patchStatusOptions = document.getElementById('patch-status-options');
const settingsFeedbackEl = document.getElementById('settings-feedback');

const sessionStorageKey = (key) => `liminal.${key}`;
const readSession = (key) => sessionStorage.getItem(sessionStorageKey(key));

// XSS protection: escape HTML entities
const escapeHtml = (str) => {
  if (str == null) return '';
  const text = String(str);
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
};
const writeSession = (key, value) => {
  if (value) {
    sessionStorage.setItem(sessionStorageKey(key), value);
  } else {
    sessionStorage.removeItem(sessionStorageKey(key));
  }
};

const state = {
  accessToken: readSession('accessToken'),
  refreshToken: readSession('refreshToken'),
  sessionId: readSession('sessionId'),
  tenantId: readSession('tenantId'),
  role: readSession('role'),
};

const defaultPatchStatuses = new Set(['pending', 'approved', 'rejected', 'applied']);
let knownPatchStatuses = new Set(defaultPatchStatuses);

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
};

// Double-submit CSRF: echo the JS-readable csrf_token cookie on mutating requests.
const getCsrfToken = () => {
  const match = document.cookie.match(/(?:^|;\s*)csrf_token=([^;]+)/);
  return match ? decodeURIComponent(match[1]) : null;
};

const headers = (idempotencyKey) => {
  const h = { 'Content-Type': 'application/json' };
  if (state.accessToken) h['Authorization'] = `Bearer ${state.accessToken}`;
  if (state.tenantId) h['X-Tenant-ID'] = state.tenantId;
  if (state.sessionId) h['session_id'] = state.sessionId;
  const csrf = getCsrfToken();
  if (csrf) h['X-CSRF-Token'] = csrf;
  h['Idempotency-Key'] = idempotencyKey || randomIdempotencyKey();
  return h;
};

const fetchWithRetry = async (url, options, retries = 3, backoffMs = 400) => {
  let lastError;
  for (let attempt = 0; attempt <= retries; attempt += 1) {
    try {
      const resp = await fetch(url, options);
      // Don't retry client errors (4xx) - they won't succeed on retry
      if (resp.status >= 400 && resp.status < 500) {
        return resp;
      }
      // Only retry on server errors (5xx) or network failures
      if (!resp.ok && resp.status >= 500) {
        lastError = new Error(`Server error: ${resp.status}`);
        if (attempt === retries) return resp;
        const delay = backoffMs * Math.pow(2, attempt);
        await new Promise((resolve) => setTimeout(resolve, delay));
        continue;
      }
      return resp;
    } catch (err) {
      // Network error - retry
      lastError = err;
      if (attempt === retries) break;
      const delay = backoffMs * Math.pow(2, attempt);
      await new Promise((resolve) => setTimeout(resolve, delay));
    }
  }
  const attempts = retries + 1;
  const label = attempts === 1 ? 'attempt' : 'attempts';
  throw new Error(`Request failed after ${attempts} ${label}: ${lastError?.message || 'unknown error'}`);
};

const showError = (msg) => {
  if (!errorEl) return;
  errorEl.textContent = msg;
  errorEl.style.display = msg ? 'block' : 'none';
};

const showFeedback = (msg) => {
  if (!feedbackEl) return;
  feedbackEl.textContent = msg;
  feedbackEl.style.display = msg ? 'block' : 'none';
};

const requireAdmin = () => state.role === 'admin';

const persistAuth = (payload) => {
  state.accessToken = payload.access_token;
  state.refreshToken = payload.refresh_token;
  state.sessionId = payload.session_id;
  state.role = payload.role;
  state.tenantId = payload.tenant_id;
  writeSession('accessToken', state.accessToken);
  writeSession('refreshToken', state.refreshToken);
  writeSession('sessionId', state.sessionId);
  writeSession('role', state.role);
  writeSession('tenantId', state.tenantId);
};

const gatekeep = () => {
  if (!state.accessToken) {
    loginPanel.style.display = 'block';
    consolePanel.classList.add('hidden');
    showError('');
    showFeedback('Sign in with an admin account to manage patches and users.');
    return false;
  }
  if (!requireAdmin()) {
    loginPanel.style.display = 'block';
    consolePanel.classList.add('hidden');
    showError('Admin role required. Sign in with an admin account.');
    showFeedback('');
    return false;
  }
  loginPanel.style.display = 'none';
  consolePanel.classList.remove('hidden');
  sessionIndicator.textContent = `Signed in as admin (${state.tenantId || 'global'})`;
  showError('');
  return true;
};

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

const requestEnvelope = async (url, options, fallbackMessage) => {
  const resp = await fetchWithRetry(url, options);
  const text = await resp.text();
  const trimmed = text.trim();
  let payload;
  if (trimmed) {
    try {
      payload = JSON.parse(trimmed);
    } catch (err) {
      if (!resp.ok) throw new Error(fallbackMessage || resp.statusText || 'Request failed');
      throw err;
    }
  }
  if (!resp.ok) {
    const error = new Error(
      extractError(payload ?? trimmed, fallbackMessage || 'Request failed')
    );
    error.status = resp.status;
    error.details = extractErrorDetails(payload);
    throw error;
  }
  return payload ?? {};
};

const setPatchStatusOptions = (statuses = []) => {
  knownPatchStatuses = new Set([...defaultPatchStatuses, ...statuses.map((s) => (s || '').toLowerCase())]);
  if (!patchStatusOptions) return;
  // Escape status values to prevent XSS via malicious API data (Issue 69.7)
  patchStatusOptions.innerHTML = Array.from(knownPatchStatuses)
    .sort()
    .map((status) => `<option value="${escapeHtml(status)}">${escapeHtml(status)}</option>`)
    .join('');
};

const handleLogin = async (event) => {
  event.preventDefault();
  const body = {
    email: document.getElementById('admin-email').value,
    password: document.getElementById('admin-password').value,
    mfa_code: document.getElementById('admin-mfa')?.value || undefined,
    tenant_id: document.getElementById('admin-tenant').value || undefined,
  };
  try {
    const envelope = await requestEnvelope(
      `${apiBase}/auth/login`,
      {
        method: 'POST',
        headers: (() => {
          const h = { 'Content-Type': 'application/json' };
          const csrf = getCsrfToken();
          if (csrf) h['X-CSRF-Token'] = csrf;
          return h;
        })(),
        body: JSON.stringify(body),
      },
      'Login failed'
    );
    if (envelope.data?.mfa_required && !envelope.data?.access_token) {
      showError('MFA required. Enter your code and submit again.');
      return;
    }
    persistAuth(envelope.data);
    if (!gatekeep()) throw new Error('Admin role required');
    showFeedback('Authenticated');
  } catch (err) {
    showError(err.message);
  }
};

const renderPatchTable = (patches) => {
  if (!patches.length) {
    patchTableWrapper.innerHTML = '<div class="empty">No patch proposals</div>';
    setPatchStatusOptions([]);
    return;
  }
  setPatchStatusOptions(patches.map((p) => p.status).filter(Boolean));
  const rows = patches
    .map(
      (p) => `
        <tr>
          <td>${escapeHtml(p.id)}</td>
          <td>${escapeHtml(p.artifact_id)}</td>
          <td>${escapeHtml(p.status)}</td>
          <td>${escapeHtml(p.justification || '')}</td>
          <td>${escapeHtml(p.decided_at || '')}</td>
          <td>${escapeHtml(p.applied_at || '')}</td>
        </tr>`
    )
    .join('');
  patchTableWrapper.innerHTML = `
    <table class="table">
      <thead><tr><th>ID</th><th>Artifact</th><th>Status</th><th>Justification</th><th>Decided</th><th>Applied</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
};

const fetchPatches = async () => {
  try {
    const envelope = await requestEnvelope(
      `${apiBase}/config/patches`,
      { headers: headers() },
      'Unable to load patches'
    );
    renderPatchTable(envelope.data.items || []);
    showFeedback(`Loaded ${envelope.data.items?.length || 0} patches`);
  } catch (err) {
    showError(err.message);
  }
};

const fetchRuntimeConfig = async () => {
  if (!runtimeConfigEl) return;
  runtimeConfigEl.textContent = 'Loading config...';
  try {
    const envelope = await requestEnvelope(
      `${apiBase}/config`,
      { headers: headers() },
      'Unable to load config'
    );
    runtimeConfigEl.textContent = JSON.stringify(envelope.data, null, 2);
  } catch (err) {
    runtimeConfigEl.textContent = err.message;
  }
};

const proposePatch = async () => {
  const artifact = document.getElementById('patch-artifact').value;
  const justification = document.getElementById('patch-justification').value;
  const body = document.getElementById('patch-body').value;
  if (!artifact || !body || !justification) {
    showError('Artifact, justification, and patch body are required');
    return;
  }
  let parsed;
  try {
    parsed = JSON.parse(body);
  } catch (err) {
    showError('Patch body must be valid JSON');
    return;
  }
  try {
    await requestEnvelope(
      `${apiBase}/config/propose_patch`,
      {
        method: 'POST',
        headers: headers(),
        body: JSON.stringify({
          artifact_id: artifact,
          justification,
          patch: parsed,
        }),
      },
      'Unable to propose patch'
    );
    await fetchPatches();
    showFeedback('Patch proposed');
  } catch (err) {
    showError(err.message);
  }
};

const decidePatch = async (fallbackDecision) => {
  const patchId = document.getElementById('decision-id').value;
  if (!patchId) {
    showError('Provide a patch id');
    return;
  }
  const rawDecision = (decisionStatusInput?.value || fallbackDecision || '').trim();
  if (!rawDecision) {
    showError('Decision status is required');
    return;
  }
  const normalizedDecision = rawDecision.toLowerCase();
  knownPatchStatuses.add(normalizedDecision);
  try {
    await requestEnvelope(
      `${apiBase}/config/patches/${patchId}/decide`,
      {
        method: 'POST',
        headers: headers(),
        body: JSON.stringify({
          decision: normalizedDecision,
          reason: `console ${normalizedDecision}`,
        }),
      },
      'Unable to decide patch'
    );
    await fetchPatches();
    setPatchStatusOptions(Array.from(knownPatchStatuses));
    showFeedback(`Patch ${normalizedDecision}`);
  } catch (err) {
    showError(err.message);
  }
};

const applyPatch = async () => {
  const patchId = document.getElementById('decision-id').value;
  if (!patchId) {
    showError('Provide a patch id');
    return;
  }
  try {
    await requestEnvelope(
      `${apiBase}/config/patches/${patchId}/apply`,
      {
        method: 'POST',
        headers: headers(),
      },
      'Unable to apply patch'
    );
    await fetchPatches();
    showFeedback('Patch applied');
  } catch (err) {
    showError(err.message);
  }
};

const renderUsers = (users) => {
  if (!userTableWrapper) return;
  if (!users.length) {
    userTableWrapper.innerHTML = '<div class="empty">No users yet</div>';
    return;
  }
  const rows = users
    .map(
      (u) => `
        <tr>
          <td>${escapeHtml(u.id)}</td>
          <td>${escapeHtml(u.email)}</td>
          <td>${escapeHtml(u.role)}</td>
          <td>${escapeHtml(u.tenant_id)}</td>
          <td>${escapeHtml(u.plan_tier || '')}</td>
          <td>${u.is_active ? 'active' : 'disabled'}</td>
          <td>${escapeHtml(u.created_at)}</td>
        </tr>`
    )
    .join('');
  userTableWrapper.innerHTML = `
    <table class="table">
      <thead><tr><th>ID</th><th>Email</th><th>Role</th><th>Tenant</th><th>Plan</th><th>Status</th><th>Created</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
};

const fetchUsers = async () => {
  const limitInput = document.getElementById('user-limit');
  const limit = limitInput ? Number(limitInput.value || 50) : 50;
  try {
    const envelope = await requestEnvelope(
      `${apiBase}/admin/users?limit=${Math.max(1, limit)}`,
      { headers: headers() },
      'Unable to load users'
    );
    renderUsers(envelope.data.items || []);
  } catch (err) {
    showError(err.message);
  }
};

const createUser = async () => {
  const email = document.getElementById('new-user-email').value;
  const password = document.getElementById('new-user-password').value;
  const handle = document.getElementById('new-user-handle').value;
  const tenant = document.getElementById('new-user-tenant').value;
  const role = document.getElementById('new-user-role').value;
  const plan = document.getElementById('new-user-plan').value;
  const active = document.getElementById('new-user-active').checked;
  if (!email) {
    showError('Email required');
    return;
  }
  try {
    const envelope = await requestEnvelope(
      `${apiBase}/admin/users`,
      {
        method: 'POST',
        headers: headers(),
        body: JSON.stringify({
          email,
          password: password || undefined,
          handle: handle || undefined,
          tenant_id: tenant || undefined,
          role: role || undefined,
          plan_tier: plan || undefined,
          is_active: active,
        }),
      },
      'Unable to create user'
    );
    fetchUsers();
    if (createdPasswordEl) {
      createdPasswordEl.textContent = `Password: ${envelope.data.password}`;
    }
    showFeedback('User created');
  } catch (err) {
    showError(err.message);
  }
};

const setUserRole = async () => {
  const userId = document.getElementById('target-user-id').value;
  const role = document.getElementById('target-user-role').value;
  if (!userId || !role) {
    showError('User ID and role required');
    return;
  }
  try {
    await requestEnvelope(
      `${apiBase}/admin/users/${userId}/role`,
      {
        method: 'POST',
        headers: headers(),
        body: JSON.stringify({ role }),
      },
      'Unable to set role'
    );
    await fetchUsers();
    showFeedback('Role updated');
  } catch (err) {
    showError(err.message);
  }
};

const deleteUser = async () => {
  const userId = document.getElementById('target-user-id').value;
  if (!userId) {
    showError('User ID required');
    return;
  }
  try {
    await requestEnvelope(
      `${apiBase}/admin/users/${userId}`,
      { method: 'DELETE', headers: headers() },
      'Unable to delete user'
    );
    await fetchUsers();
    showFeedback('User deleted');
  } catch (err) {
    showError(err.message);
  }
};

const renderAdapters = (adapters) => {
  if (!adapterTableWrapper) return;
  if (!adapters.length) {
    adapterTableWrapper.innerHTML = '<div class="empty">No adapters</div>';
    return;
  }
  const rows = adapters
    .map(
      (a) => `
        <tr>
          <td>${escapeHtml(a.id)}</td>
          <td>${escapeHtml(a.name)}</td>
          <td>${escapeHtml(a.type)}</td>
          <td>${escapeHtml(a.kind || '')}</td>
          <td>${escapeHtml(a.owner_user_id || '')}</td>
          <td>${escapeHtml(a.updated_at)}</td>
        </tr>`
    )
    .join('');
  adapterTableWrapper.innerHTML = `
    <table class="table">
      <thead><tr><th>ID</th><th>Name</th><th>Type</th><th>Kind</th><th>Owner</th><th>Updated</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
};

const fetchAdapters = async () => {
  try {
    const envelope = await requestEnvelope(
      `${apiBase}/admin/adapters`,
      { headers: headers() },
      'Unable to load adapters'
    );
    renderAdapters(envelope.data.items || []);
  } catch (err) {
    showError(err.message);
  }
};

const runInspect = async () => {
  const kind = document.getElementById('inspect-kind')?.value;
  const limit = document.getElementById('inspect-limit')?.value || 50;
  if (inspectOutput) inspectOutput.textContent = 'Loading...';
  try {
    const envelope = await requestEnvelope(
      `${apiBase}/admin/objects?limit=${limit}${kind ? `&kind=${kind}` : ''}`,
      {
        headers: headers(),
      },
      'Unable to inspect objects'
    );
    if (inspectOutput)
      inspectOutput.textContent = JSON.stringify(
        { summary: envelope.data.summary, details: envelope.data.details },
        null,
        2
      );
  } catch (err) {
    showError(err.message);
    if (inspectOutput) inspectOutput.textContent = err.message;
  }
};

const showSettingsFeedback = (msg) => {
  if (!settingsFeedbackEl) return;
  settingsFeedbackEl.textContent = msg;
  settingsFeedbackEl.style.display = msg ? 'block' : 'none';
};

// ---------------------------------------------------------------------------
// System settings
//
// The form is built from GET /admin/settings/schema, which describes every
// managed setting: type, bounds, choices, default, group. Hard-coding the
// field list here is how thirty settings ended up with no control at all and
// one that had become environment-only was still posted, failing every save.

let settingsSchema = [];
let settingsCurrent = {};
const settingsEdits = new Map();

const settingsFormEl = document.getElementById('settings-form');
const settingsFilterEl = document.getElementById('settings-filter');
const settingsChangedOnlyEl = document.getElementById('settings-changed-only');
const settingsDirtyCountEl = document.getElementById('settings-dirty-count');

// Acronyms that should not be sentence-cased. Without this, settings read
// "Adapter openai api key" and "Jwt issuer".
const ACRONYMS = {
  api: 'API', url: 'URL', uri: 'URI', id: 'ID', ids: 'IDs', ttl: 'TTL',
  jwt: 'JWT', smtp: 'SMTP', mfa: 'MFA', rag: 'RAG', llm: 'LLM', tls: 'TLS',
  ui: 'UI', cors: 'CORS', hsts: 'HSTS', oauth: 'OAuth', openai: 'OpenAI',
  ws: 'WebSocket', json: 'JSON', dim: 'dimensions', pgvector: 'pgvector',
};

const labelFor = (name) => {
  const words = name.split('_').map((w) => ACRONYMS[w] || w);
  return words.join(' ').replace(/^./, (ch) => ch.toUpperCase());
};

const coerce = (field, raw) => {
  if (field.type === 'bool') return Boolean(raw);
  if (field.type === 'int') {
    if (raw === '' || raw == null) return null;
    return Number.isInteger(Number(raw)) ? Number(raw) : NaN;
  }
  if (field.type === 'float') {
    if (raw === '' || raw == null) return null;
    return Number(raw);
  }
  return String(raw ?? '');
};

// The same bounds the API enforces, so the control refuses a value before a
// round trip rather than after one.
const localError = (field, value) => {
  if (field.secret) return '';
  if (value === null) return 'required';
  if (field.type === 'int' && !Number.isInteger(value)) return 'must be a whole number';
  if ((field.type === 'int' || field.type === 'float') && Number.isNaN(value)) {
    return 'must be a number';
  }
  if (field.min != null && value < field.min) return `must be at least ${field.min}`;
  if (field.max != null && value > field.max) return `must be at most ${field.max}`;
  if (field.exclusive_min != null && value <= field.exclusive_min) {
    return `must be greater than ${field.exclusive_min}`;
  }
  if (field.exclusive_max != null && value >= field.exclusive_max) {
    return `must be less than ${field.exclusive_max}`;
  }
  if (field.type === 'choice' && !field.choices.includes(value)) {
    return `must be one of: ${field.choices.join(', ')}`;
  }
  return '';
};

const valueOf = (name) =>
  settingsEdits.has(name) ? settingsEdits.get(name) : settingsCurrent[name];

const setFieldError = (name, message) => {
  const row = settingsFormEl.querySelector(`[data-setting="${name}"]`);
  if (!row) return;
  const errorEl = row.querySelector('.field-error');
  errorEl.textContent = message || '';
  errorEl.style.display = message ? 'block' : 'none';
  row.classList.toggle('has-error', Boolean(message));
};

const refreshDirtyState = () => {
  const count = settingsEdits.size;
  const saveBtn = document.getElementById('save-settings');
  const revertBtn = document.getElementById('revert-settings');
  if (saveBtn) saveBtn.disabled = count === 0;
  if (revertBtn) revertBtn.disabled = count === 0;
  if (settingsDirtyCountEl) {
    const reloads = [...settingsEdits.keys()].some(
      (name) => settingsSchema.find((f) => f.name === name)?.reloads_model
    );
    settingsDirtyCountEl.textContent = count
      ? `${count} unsaved change${count === 1 ? '' : 's'}` +
        (reloads ? ' — saving will reload the model services' : '')
      : '';
  }
  settingsFormEl.querySelectorAll('[data-setting]').forEach((row) => {
    row.classList.toggle('is-dirty', settingsEdits.has(row.dataset.setting));
  });
};

const onFieldInput = (field, rawValue) => {
  const value = coerce(field, rawValue);
  // A secret has no readable current value to compare against; an empty box
  // is simply "unchanged".
  const current = field.secret ? '' : settingsCurrent[field.name];
  if (value === current) settingsEdits.delete(field.name);
  else settingsEdits.set(field.name, value);
  setFieldError(field.name, localError(field, value));
  refreshDirtyState();
};

const renderField = (field) => {
  const row = document.createElement('div');
  row.className = 'field setting-row';
  row.dataset.setting = field.name;
  row.dataset.group = field.group;

  const label = document.createElement('label');
  label.setAttribute('for', `setting-${field.name}`);
  label.textContent = labelFor(field.name);
  if (field.overridden) {
    const badge = document.createElement('span');
    badge.className = 'badge';
    badge.textContent = 'changed';
    badge.title = 'Saved in the database; differs from the shipped default';
    label.appendChild(badge);
  }
  row.appendChild(label);

  const value = valueOf(field.name);
  let input;
  if (field.type === 'choice') {
    input = document.createElement('select');
    field.choices.forEach((choice) => {
      const option = document.createElement('option');
      option.value = choice;
      option.textContent = choice;
      if (String(value) === choice) option.selected = true;
      input.appendChild(option);
    });
  } else if (field.type === 'bool') {
    input = document.createElement('input');
    input.type = 'checkbox';
    input.checked = Boolean(value);
  } else if (field.secret) {
    // Write-only: the server never sends the value back, so the control
    // starts empty and an empty box means "leave what is stored alone".
    input = document.createElement('input');
    input.type = 'password';
    input.autocomplete = 'new-password';
    input.placeholder = field.is_set ? 'Set — type to replace' : 'Not set';
    input.value = '';
  } else {
    input = document.createElement('input');
    input.type = field.type === 'text' ? 'text' : 'number';
    if (field.type === 'float') input.step = 'any';
    if (field.min != null) input.min = field.min;
    if (field.max != null) input.max = field.max;
    input.value = value ?? '';
  }
  input.id = `setting-${field.name}`;
  input.addEventListener(field.type === 'bool' ? 'change' : 'input', (event) => {
    const target = event.target;
    onFieldInput(field, field.type === 'bool' ? target.checked : target.value);
  });
  row.appendChild(input);

  const help = document.createElement('div');
  help.className = 'subtext';
  const defaultText = field.secret
    ? (field.is_set ? 'Currently set' : 'Not set')
    : `Default: ${JSON.stringify(field.default)}`;
  help.textContent = field.description
    ? `${field.description} (${defaultText})`
    : defaultText;
  row.appendChild(help);

  const error = document.createElement('div');
  error.className = 'field-error';
  error.style.display = 'none';
  error.setAttribute('role', 'alert');
  row.appendChild(error);
  return row;
};

const renderSettingsForm = () => {
  if (!settingsFormEl) return;
  const needle = (settingsFilterEl?.value || '').trim().toLowerCase();
  const changedOnly = Boolean(settingsChangedOnlyEl?.checked);
  settingsFormEl.textContent = '';

  let shown = 0;
  let lastGroup = null;
  settingsSchema.forEach((field) => {
    if (needle && !field.name.includes(needle) &&
        !field.description.toLowerCase().includes(needle)) return;
    if (changedOnly && !field.overridden && !settingsEdits.has(field.name)) return;
    if (field.group !== lastGroup) {
      const heading = document.createElement('h4');
      heading.textContent = field.group;
      settingsFormEl.appendChild(heading);
      lastGroup = field.group;
    }
    settingsFormEl.appendChild(renderField(field));
    shown += 1;
  });

  if (!shown) {
    const empty = document.createElement('div');
    empty.className = 'subtext';
    empty.textContent = 'No settings match this filter.';
    settingsFormEl.appendChild(empty);
  }
  refreshDirtyState();
};

const fetchSystemSettings = async () => {
  showSettingsFeedback('Loading settings...');
  try {
    const [schemaEnvelope, valuesEnvelope] = await Promise.all([
      requestEnvelope(
        `${apiBase}/admin/settings/schema`,
        { headers: headers() },
        'Unable to load the settings schema'
      ),
      requestEnvelope(
        `${apiBase}/admin/settings`,
        { headers: headers() },
        'Unable to load system settings'
      ),
    ]);
    settingsSchema = schemaEnvelope.data?.fields || [];
    settingsCurrent = valuesEnvelope.data || {};
    settingsEdits.clear();
    renderSettingsForm();
    showSettingsFeedback(`${settingsSchema.length} settings loaded`);
  } catch (err) {
    showSettingsFeedback(err.message);
  }
};

const saveSystemSettings = async () => {
  if (!settingsEdits.size) return;

  // Refuse locally first: the bounds here are the ones the API enforces, so a
  // value that cannot be saved should not cost a round trip to find out.
  let blocked = false;
  settingsEdits.forEach((value, name) => {
    const field = settingsSchema.find((f) => f.name === name);
    const message = field ? localError(field, value) : 'unknown setting';
    setFieldError(name, message);
    if (message) blocked = true;
  });
  if (blocked) {
    showSettingsFeedback('Some settings are not valid — see the fields marked below.');
    return;
  }

  const patch = Object.fromEntries(settingsEdits);
  const reloads = Object.keys(patch).some(
    (name) => settingsSchema.find((f) => f.name === name)?.reloads_model
  );
  if (reloads && !window.confirm(
    'This changes a model setting. Saving rebuilds the model services and ' +
    'briefly interrupts in-flight requests. Continue?'
  )) return;

  showSettingsFeedback('Saving...');
  try {
    await requestEnvelope(
      `${apiBase}/admin/settings`,
      { method: 'PUT', headers: headers(), body: JSON.stringify(patch) },
      'Unable to save settings'
    );
    const saved = Object.keys(patch).length;
    await fetchSystemSettings();
    showSettingsFeedback(`Saved ${saved} setting${saved === 1 ? '' : 's'}`);
  } catch (err) {
    // The API reports per-field problems; put each one on its own control
    // rather than leaving the operator to parse one long string.
    const fields = err.details?.fields;
    if (fields && typeof fields === 'object') {
      Object.entries(fields).forEach(([name, message]) => setFieldError(name, message));
      showSettingsFeedback('The server rejected some settings — see the fields below.');
    } else {
      showSettingsFeedback(err.message);
    }
  }
};

const revertSystemSettings = () => {
  settingsEdits.clear();
  renderSettingsForm();
  showSettingsFeedback('Changes discarded');
};

const logout = async () => {
  const tryRevoke = async () => {
    try {
      await requestEnvelope(
        `${apiBase}/auth/logout`,
        { method: 'POST', headers: headers(), keepalive: true },
        'Logout failed'
      );
    } catch (err) {
      console.warn('logout failed', err);
    }
  };

  await tryRevoke();
  ['accessToken', 'refreshToken', 'sessionId', 'role', 'tenantId'].forEach((k) =>
    sessionStorage.removeItem(sessionStorageKey(k))
  );
  window.location.href = '/';
};

setPatchStatusOptions([]);

if (authForm) authForm.addEventListener('submit', handleLogin);
const logoutBtn = document.getElementById('logout');
if (logoutBtn) logoutBtn.addEventListener('click', logout);
const refreshBtn = document.getElementById('refresh-patches');
if (refreshBtn) refreshBtn.addEventListener('click', fetchPatches);
const proposeBtn = document.getElementById('propose-patch');
if (proposeBtn) proposeBtn.addEventListener('click', proposePatch);
const approveBtn = document.getElementById('approve-patch');
if (approveBtn)
  approveBtn.addEventListener('click', () => {
    if (decisionStatusInput) decisionStatusInput.value = 'approve';
    decidePatch('approve');
  });
const rejectBtn = document.getElementById('reject-patch');
if (rejectBtn)
  rejectBtn.addEventListener('click', () => {
    if (decisionStatusInput) decisionStatusInput.value = 'reject';
    decidePatch('reject');
  });
const applyBtn = document.getElementById('apply-patch');
if (applyBtn) applyBtn.addEventListener('click', applyPatch);
const refreshUsersBtn = document.getElementById('refresh-users');
if (refreshUsersBtn) refreshUsersBtn.addEventListener('click', fetchUsers);
const createUserBtn = document.getElementById('create-user');
if (createUserBtn) createUserBtn.addEventListener('click', createUser);
const setRoleBtn = document.getElementById('set-user-role');
if (setRoleBtn) setRoleBtn.addEventListener('click', setUserRole);
const deleteUserBtn = document.getElementById('delete-user');
if (deleteUserBtn) deleteUserBtn.addEventListener('click', deleteUser);
const refreshAdaptersBtn = document.getElementById('refresh-adapters');
if (refreshAdaptersBtn) refreshAdaptersBtn.addEventListener('click', fetchAdapters);
const runInspectBtn = document.getElementById('run-inspect');
if (runInspectBtn) runInspectBtn.addEventListener('click', runInspect);
const refreshConfigBtn = document.getElementById('refresh-config');
if (refreshConfigBtn) refreshConfigBtn.addEventListener('click', fetchRuntimeConfig);
const refreshSettingsBtn = document.getElementById('refresh-settings');
if (refreshSettingsBtn) refreshSettingsBtn.addEventListener('click', fetchSystemSettings);
const saveSettingsBtn = document.getElementById('save-settings');
if (saveSettingsBtn) saveSettingsBtn.addEventListener('click', saveSystemSettings);
const revertSettingsBtn = document.getElementById('revert-settings');
if (revertSettingsBtn) revertSettingsBtn.addEventListener('click', revertSystemSettings);
if (settingsFilterEl) settingsFilterEl.addEventListener('input', renderSettingsForm);
if (settingsChangedOnlyEl) {
  settingsChangedOnlyEl.addEventListener('change', renderSettingsForm);
}
// Unsaved edits are easy to walk away from in a form this long.
window.addEventListener('beforeunload', (event) => {
  if (!settingsEdits.size) return;
  event.preventDefault();
  event.returnValue = '';
});

// Bootstrap existing session
if (state.accessToken) {
  if (gatekeep()) {
    fetchPatches();
    fetchRuntimeConfig();
    fetchSystemSettings();
    fetchUsers();
    fetchAdapters();
  }
} else {
  gatekeep();
}
