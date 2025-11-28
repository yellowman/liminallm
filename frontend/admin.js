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
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
};

const headers = (idempotencyKey) => {
  const h = { 'Content-Type': 'application/json' };
  if (state.accessToken) h['Authorization'] = `Bearer ${state.accessToken}`;
  if (state.tenantId) h['X-Tenant-ID'] = state.tenantId;
  if (state.sessionId) h['session_id'] = state.sessionId;
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
    throw new Error(extractError(payload ?? trimmed, fallbackMessage || 'Request failed'));
  }
  return payload ?? {};
};

const setPatchStatusOptions = (statuses = []) => {
  knownPatchStatuses = new Set([...defaultPatchStatuses, ...statuses.map((s) => (s || '').toLowerCase())]);
  if (!patchStatusOptions) return;
  patchStatusOptions.innerHTML = Array.from(knownPatchStatuses)
    .sort()
    .map((status) => `<option value="${status}"></option>`)
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
        headers: { 'Content-Type': 'application/json' },
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
  if (!artifact || !body) {
    showError('Artifact and patch body are required');
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
          justification: justification || undefined,
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

// Bootstrap existing session
if (state.accessToken) {
  if (gatekeep()) {
    fetchPatches();
    fetchRuntimeConfig();
    fetchUsers();
    fetchAdapters();
  }
} else {
  gatekeep();
}
