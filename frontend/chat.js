/**
 * LiminalLM Chat Frontend
 * Implements SPEC §17 frontend requirements with tabs, contexts, artifacts, and streaming.
 */

const apiBase = '/v1';

// =============================================================================
// Storage utilities
// =============================================================================

const storageKey = (key) => `liminal.${key}`;
const readStorage = (key) => sessionStorage.getItem(storageKey(key));
const writeStorage = (key, value) => {
  if (value) {
    sessionStorage.setItem(storageKey(key), value);
  } else {
    sessionStorage.removeItem(storageKey(key));
  }
};

// LocalStorage for drafts (offline-safe per SPEC §17)
const DRAFT_STORAGE_KEY = 'liminal.drafts';

const loadDrafts = () => {
  try {
    const data = localStorage.getItem(DRAFT_STORAGE_KEY);
    return data ? JSON.parse(data) : {};
  } catch {
    return {};
  }
};

const saveDraft = (conversationId, text) => {
  const drafts = loadDrafts();
  if (text && text.trim()) {
    drafts[conversationId || '_new'] = { text, savedAt: new Date().toISOString() };
  } else {
    delete drafts[conversationId || '_new'];
  }
  localStorage.setItem(DRAFT_STORAGE_KEY, JSON.stringify(drafts));
  updateDraftIndicator();
};

const getDraft = (conversationId) => {
  const drafts = loadDrafts();
  return drafts[conversationId || '_new']?.text || '';
};

const clearAllDrafts = () => {
  localStorage.removeItem(DRAFT_STORAGE_KEY);
  updateDraftIndicator();
};

const updateDraftIndicator = () => {
  const indicator = document.getElementById('draft-indicator');
  if (!indicator) return;
  const drafts = loadDrafts();
  const count = Object.keys(drafts).length;
  indicator.textContent = count > 0 ? `${count} draft${count > 1 ? 's' : ''} saved` : '';
};

// =============================================================================
// State management
// =============================================================================

const persistedKeys = ['accessToken', 'refreshToken', 'sessionId', 'tenantId', 'role', 'userId'];

const createState = (storage) => {
  const backing = {
    accessToken: storage.read('accessToken'),
    refreshToken: storage.read('refreshToken'),
    sessionId: storage.read('sessionId'),
    tenantId: storage.read('tenantId'),
    role: storage.read('role'),
    userId: storage.read('userId'),
    conversationId: null,
    lastAssistant: null,
    contexts: [],
    artifacts: [],
    conversations: [],
    selectedContext: null,
    selectedArtifact: null,
    isStreaming: false,
  };

  const sync = (key, value) => {
    if (!persistedKeys.includes(key)) return;
    storage.write(key, value);
  };

  const stateApi = {
    resetAuth() {
      persistedKeys.forEach((k) => {
        backing[k] = null;
        sync(k, null);
      });
      backing.lastAssistant = null;
      backing.conversationId = null;
    },
    snapshot() {
      return { ...backing };
    },
  };

  return new Proxy(stateApi, {
    get(target, prop) {
      if (prop in target) return target[prop];
      return backing[prop];
    },
    set(target, prop, value) {
      backing[prop] = value;
      sync(prop, value);
      return true;
    },
  });
};

const state = createState({ read: readStorage, write: writeStorage });

// =============================================================================
// Utility functions
// =============================================================================

const $ = (id) => document.getElementById(id);

const escapeHtml = (str) => {
  if (str == null) return '';
  const text = String(str);
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
};

// Citation modal for displaying source content
const showCitationModal = (element) => {
  try {
    const data = JSON.parse(element.dataset.citation || '{}');
    const modal = document.getElementById('citation-modal');
    const content = document.getElementById('citation-modal-content');
    const title = document.getElementById('citation-modal-title');

    if (!modal) {
      // Create modal dynamically if it doesn't exist
      createCitationModal();
      return showCitationModal(element);
    }

    // Set modal content
    const sourcePath = data.source_path || data.chunk_id || 'Unknown Source';
    title.textContent = sourcePath.split('/').pop() || sourcePath;

    // Build content display
    let html = '';
    if (data.source_path) {
      html += `<div class="citation-meta"><strong>Source:</strong> ${escapeHtml(data.source_path)}</div>`;
    }
    if (data.context_id) {
      html += `<div class="citation-meta"><strong>Context:</strong> ${escapeHtml(data.context_id)}</div>`;
    }
    if (data.chunk_index !== undefined) {
      html += `<div class="citation-meta"><strong>Chunk:</strong> #${data.chunk_index}</div>`;
    }
    if (data.content) {
      html += `<div class="citation-content"><pre>${escapeHtml(data.content)}</pre></div>`;
    } else {
      html += `<div class="citation-content"><em>No content preview available</em></div>`;
    }

    content.innerHTML = html;
    modal.classList.add('active');

    // Close on click outside
    modal.onclick = (e) => {
      if (e.target === modal) {
        modal.classList.remove('active');
      }
    };
  } catch (err) {
    console.error('Failed to parse citation data:', err);
  }
};

const createCitationModal = () => {
  const modal = document.createElement('div');
  modal.id = 'citation-modal';
  modal.className = 'modal-overlay';
  modal.innerHTML = `
    <div class="modal-content">
      <div class="modal-header">
        <h3 id="citation-modal-title">Citation</h3>
        <button class="modal-close" onclick="document.getElementById('citation-modal').classList.remove('active')">&times;</button>
      </div>
      <div id="citation-modal-content" class="modal-body"></div>
    </div>
  `;
  document.body.appendChild(modal);
};

// Escape key handler moved to initEventListeners() for consistent initialization

const randomIdempotencyKey = () => {
  if (window.crypto?.randomUUID) return window.crypto.randomUUID();
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
};

const stableHash = (str) => {
  let hash = 0;
  for (const ch of str) {
    hash = (hash << 5) - hash + ch.codePointAt(0);
    hash |= 0;
  }
  return Math.abs(hash >>> 0).toString(16);
};

const formatBytes = (bytes) => {
  if (!bytes && bytes !== 0) return '0 bytes';
  const thresh = 1024;
  if (Math.abs(bytes) < thresh) return `${bytes} bytes`;
  const units = ['KB', 'MB', 'GB'];
  let u = -1;
  let size = bytes;
  do {
    size /= thresh;
    ++u;
  } while (Math.abs(size) >= thresh && u < units.length - 1);
  return `${size.toFixed(1)} ${units[u]}`;
};

// =============================================================================
// DOM element references
// =============================================================================

const messagesEl = $('messages');
const messagesEmptyEl = $('messages-empty');
const authForm = $('auth-form');
const authPanel = $('auth-panel');
const chatForm = $('chat-form');
const statusEl = $('status');
const errorEl = $('error-banner');
const sessionIndicator = $('session-indicator');
const adminWarning = $('admin-warning');
const conversationLabel = $('conversation-label');
const adminLink = $('admin-link');
const authSubmit = $('auth-submit');
const sendBtn = $('send-btn');
const preferenceStatusEl = $('preference-status');
const preferenceMetaEl = $('preference-meta');
const preferenceRoutingEl = $('preference-routing');
const preferenceTargetEl = $('preference-target');
const preferenceHintEl = $('preference-hint');
const preferenceNotesEl = $('preference-notes');
const fileUploadInput = $('file-upload');
const fileUploadStatus = $('file-upload-status');
const fileUploadHint = $('file-upload-hint');
const fileUploadContextId = $('upload-context-id');
const fileUploadChunkSize = $('upload-chunk-size');
const fileUploadButton = $('upload-file-btn');
const mainTabs = $('main-tabs');
const conversationListEl = $('conversation-list');
const conversationSearchEl = $('conversation-search');

// =============================================================================
// API helpers
// =============================================================================

const DEFAULT_UPLOAD_BYTES = 10 * 1024 * 1024;
let uploadLimitBytes = null;
const ALLOWED_UPLOAD_TYPES = ['text/plain', 'text/markdown', 'application/pdf', 'application/json', 'text/csv'];
const ALLOWED_UPLOAD_EXTENSIONS = ['.txt', '.md', '.markdown', '.pdf', '.json', '.csv', '.yaml', '.yml'];

const getUploadLimit = () => uploadLimitBytes || DEFAULT_UPLOAD_BYTES;

const authHeaders = (idempotencyKey) => {
  const h = {};
  if (state.accessToken) h['Authorization'] = `Bearer ${state.accessToken}`;
  if (state.tenantId) h['X-Tenant-ID'] = state.tenantId;
  if (state.sessionId) h['session_id'] = state.sessionId;
  h['Idempotency-Key'] = idempotencyKey || randomIdempotencyKey();
  return h;
};

const headers = (idempotencyKey) => ({ 'Content-Type': 'application/json', ...authHeaders(idempotencyKey) });

const fetchWithRetry = async (url, options, retries = 3, backoffMs = 400) => {
  let lastError;
  for (let attempt = 0; attempt <= retries; attempt += 1) {
    try {
      const resp = await fetch(url, options);
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
};

const extractError = (payload, fallback) => {
  const detail = payload?.detail || payload?.error || payload;
  if (typeof detail === 'string') return detail.trim() || fallback;
  if (detail?.message) return detail.message;
  if (detail?.error?.message) return detail.error.message;
  return fallback;
};

const tryRefreshToken = async () => {
  if (!state.refreshToken) return false;
  try {
    const resp = await fetch(`${apiBase}/auth/refresh`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
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
};

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
    throw new Error(extractError(payload ?? text, fallbackMessage || 'Request failed'));
  }
  return payload ?? {};
};

// =============================================================================
// UI helpers
// =============================================================================

const showStatus = (message, isError = false) => {
  const target = isError ? errorEl : statusEl;
  if (target) {
    target.textContent = message;
    target.style.display = message ? 'block' : 'none';
  }
  if (isError && statusEl) statusEl.style.display = 'none';
  if (!isError && errorEl) errorEl.style.display = 'none';
};

const toggleButtonBusy = (button, isBusy, busyLabel = 'Working...') => {
  if (!button) return;
  if (isBusy) {
    button.dataset.label = button.textContent;
    button.textContent = busyLabel;
    button.disabled = true;
  } else {
    button.textContent = button.dataset.label || button.textContent;
    button.disabled = false;
    delete button.dataset.label;
  }
};

const updateEmptyState = () => {
  if (!messagesEmptyEl) return;
  const hasMessages = messagesEl?.children?.length;
  messagesEmptyEl.style.display = hasMessages ? 'none' : 'flex';
};

const updateAuthUI = () => {
  const isAuth = Boolean(state.accessToken);
  if (authPanel) authPanel.classList.toggle('hidden', isAuth);
  if (mainTabs) mainTabs.classList.toggle('hidden', !isAuth);

  document.querySelectorAll('.tab-panel').forEach((p) => {
    if (isAuth) {
      p.classList.remove('hidden');
    } else {
      p.classList.add('hidden');
    }
  });

  if (sessionIndicator) {
    sessionIndicator.textContent = isAuth
      ? `User: ${state.userId?.slice(0, 8) || 'unknown'}`
      : 'Not signed in';
  }

  // Update settings
  const settingUserId = $('setting-user-id');
  const settingRole = $('setting-role');
  const settingTenant = $('setting-tenant');
  const settingSessionId = $('setting-session-id');

  if (settingUserId) settingUserId.textContent = state.userId || '-';
  if (settingRole) settingRole.textContent = state.role || '-';
  if (settingTenant) settingTenant.textContent = state.tenantId || 'global';
  if (settingSessionId) settingSessionId.textContent = state.sessionId ? state.sessionId.slice(0, 16) + '...' : '-';
};

// =============================================================================
// Tab navigation
// =============================================================================

const initTabs = () => {
  if (!mainTabs) return;

  mainTabs.querySelectorAll('.tab-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      const tabId = btn.dataset.tab;

      mainTabs.querySelectorAll('.tab-btn').forEach((b) => {
        b.classList.toggle('active', b === btn);
        b.setAttribute('aria-selected', b === btn ? 'true' : 'false');
      });

      document.querySelectorAll('.tab-panel').forEach((panel) => {
        panel.classList.toggle('active', panel.id === tabId);
      });
    });
  });
};

// =============================================================================
// Collapsible sections
// =============================================================================

const initCollapsibleSections = () => {
  document.querySelectorAll('.panel-section .section-header.clickable').forEach((header) => {
    header.addEventListener('click', (e) => {
      if (e.target.tagName === 'BUTTON') return;
      const section = header.closest('.panel-section');
      section?.classList.toggle('collapsed');
    });
  });
};

// =============================================================================
// Auth form switching
// =============================================================================

const showAuthForm = (formName) => {
  const loginContainer = $('login-form-container');
  const signupContainer = $('signup-form-container');
  const resetContainer = $('reset-form-container');

  if (loginContainer) loginContainer.classList.toggle('hidden', formName !== 'login');
  if (signupContainer) signupContainer.classList.toggle('hidden', formName !== 'signup');
  if (resetContainer) resetContainer.classList.toggle('hidden', formName !== 'reset');
};

// =============================================================================
// Auth management
// =============================================================================

const renderAdminNotice = () => {
  if (state.role === 'admin') {
    if (adminWarning) {
      adminWarning.textContent = 'You are signed in as an admin. Use the Admin link to approve patches.';
      adminWarning.style.display = 'block';
    }
    if (adminLink) adminLink.style.display = 'inline-flex';
  } else {
    if (adminWarning) {
      adminWarning.textContent = '';
      adminWarning.style.display = 'none';
    }
    if (adminLink) adminLink.style.display = 'none';
  }
  // Show/hide admin settings section based on role
  renderAdminSettingsSection();
};

const persistAuth = (payload) => {
  state.accessToken = payload.access_token;
  state.refreshToken = payload.refresh_token;
  state.sessionId = payload.session_id;
  state.role = payload.role;
  state.tenantId = payload.tenant_id;
  state.userId = payload.user_id;
  updateAuthUI();
  renderAdminNotice();
  refreshUploadLimits();
};

const handleLogin = async (event) => {
  event.preventDefault();
  const body = {
    email: $('email')?.value,
    password: $('password')?.value,
    mfa_code: $('mfa')?.value || undefined,
    tenant_id: $('tenant')?.value || undefined,
  };

  try {
    toggleButtonBusy(authSubmit, true, 'Signing in...');
    const envelope = await requestEnvelope(
      `${apiBase}/auth/login`,
      { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) },
      'Login failed'
    );

    if (envelope.data?.mfa_required && !envelope.data?.access_token) {
      showStatus('MFA required. Enter the code from your authenticator.', true);
      return;
    }

    persistAuth(envelope.data);
    showStatus('Signed in');

    // Load initial data
    await Promise.all([
      fetchConversations(),
      fetchContexts(),
      fetchArtifacts(),
      fetchHealth(),
      fetchMfaStatus(),
      fetchEmailVerificationStatus(),
      fetchUserSettings(),
    ]);
  } catch (err) {
    showStatus(err.message, true);
  } finally {
    toggleButtonBusy(authSubmit, false);
  }
};

const logout = async () => {
  try {
    await requestEnvelope(
      `${apiBase}/auth/logout`,
      { method: 'POST', headers: headers(), keepalive: true },
      'Logout failed'
    );
  } catch {
    // Continue with local cleanup
  }

  state.resetAuth();
  setConversation(null);
  if (messagesEl) messagesEl.innerHTML = '';
  updateAuthUI();
  renderAdminNotice();
  updateEmptyState();
};

// =============================================================================
// OAuth
// =============================================================================

const startOAuth = async (provider) => {
  const oauthStatus = $('oauth-status');
  const btn = $(`oauth-${provider}`);

  try {
    if (btn) btn.disabled = true;
    if (oauthStatus) oauthStatus.textContent = `Connecting to ${provider}...`;

    const tenant = $('tenant')?.value?.trim() || undefined;
    const redirectUri = window.location.origin + window.location.pathname;

    const envelope = await requestEnvelope(
      `${apiBase}/auth/oauth/${provider}/start`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          redirect_uri: redirectUri,
          tenant_id: tenant,
        }),
      },
      `Failed to start ${provider} login`
    );

    const authUrl = envelope.data?.authorization_url;
    if (authUrl) {
      // Store the state for callback verification
      sessionStorage.setItem('oauth_state', envelope.data?.state || '');
      sessionStorage.setItem('oauth_provider', provider);
      // Redirect to OAuth provider
      window.location.href = authUrl;
    } else {
      if (oauthStatus) oauthStatus.textContent = 'No authorization URL returned';
    }
  } catch (err) {
    if (oauthStatus) oauthStatus.textContent = err.message;
  } finally {
    if (btn) btn.disabled = false;
  }
};

const handleOAuthCallback = async () => {
  // Check if this is an OAuth callback
  const urlParams = new URLSearchParams(window.location.search);
  const code = urlParams.get('code');
  const state = urlParams.get('state');
  const provider = urlParams.get('provider') || sessionStorage.getItem('oauth_provider');

  if (!code || !state || !provider) {
    return false;
  }

  const oauthStatus = $('oauth-status');
  const storedState = sessionStorage.getItem('oauth_state');

  // Verify state matches
  if (storedState && state !== storedState) {
    if (oauthStatus) oauthStatus.textContent = 'OAuth state mismatch. Please try again.';
    sessionStorage.removeItem('oauth_state');
    sessionStorage.removeItem('oauth_provider');
    return true;
  }

  try {
    if (oauthStatus) oauthStatus.textContent = 'Completing sign in...';

    const tenant = $('tenant')?.value?.trim() || undefined;

    const envelope = await requestEnvelope(
      `${apiBase}/auth/oauth/${provider}/callback?code=${encodeURIComponent(code)}&state=${encodeURIComponent(state)}`,
      {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
      },
      `OAuth callback failed`
    );

    // Clear OAuth session data
    sessionStorage.removeItem('oauth_state');
    sessionStorage.removeItem('oauth_provider');

    // Clear URL params
    window.history.replaceState({}, document.title, window.location.pathname);

    if (envelope.data?.access_token) {
      persistAuth(envelope.data);
      showStatus('Signed in with ' + provider);

      // Load initial data
      await Promise.all([
        fetchConversations(),
        fetchContexts(),
        fetchArtifacts(),
        fetchHealth(),
        fetchMfaStatus(),
        fetchEmailVerificationStatus(),
        fetchUserSettings(),
      ]);
    } else {
      if (oauthStatus) oauthStatus.textContent = 'OAuth completed but no token received';
    }
  } catch (err) {
    if (oauthStatus) oauthStatus.textContent = err.message;
    sessionStorage.removeItem('oauth_state');
    sessionStorage.removeItem('oauth_provider');
  }

  return true;
};

// =============================================================================
// Signup
// =============================================================================

const handleSignup = async (event) => {
  event.preventDefault();
  const signupStatus = $('signup-status');
  const signupSubmit = $('signup-submit');

  const email = $('signup-email')?.value?.trim();
  const password = $('signup-password')?.value;
  const confirm = $('signup-confirm')?.value;
  const tenant = $('signup-tenant')?.value?.trim() || undefined;

  if (!email || !password) {
    if (signupStatus) signupStatus.textContent = 'Email and password are required';
    return;
  }

  if (password !== confirm) {
    if (signupStatus) signupStatus.textContent = 'Passwords do not match';
    return;
  }

  if (password.length < 8) {
    if (signupStatus) signupStatus.textContent = 'Password must be at least 8 characters';
    return;
  }

  try {
    toggleButtonBusy(signupSubmit, true, 'Creating...');
    if (signupStatus) signupStatus.textContent = '';

    await requestEnvelope(
      `${apiBase}/auth/signup`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password, tenant_id: tenant }),
      },
      'Signup failed'
    );

    if (signupStatus) signupStatus.textContent = 'Account created! You can now sign in.';

    // Clear form and switch to login
    if ($('signup-email')) $('signup-email').value = '';
    if ($('signup-password')) $('signup-password').value = '';
    if ($('signup-confirm')) $('signup-confirm').value = '';

    // Pre-fill login email
    if ($('email')) $('email').value = email;

    setTimeout(() => showAuthForm('login'), 1500);
  } catch (err) {
    if (signupStatus) signupStatus.textContent = err.message;
  } finally {
    toggleButtonBusy(signupSubmit, false);
  }
};

// =============================================================================
// Password Reset
// =============================================================================

let resetEmailForConfirm = '';

const handleResetRequest = async (event) => {
  event.preventDefault();
  const resetStatus = $('reset-status');
  const resetSubmit = $('reset-request-submit');
  const resetCodeSection = $('reset-code-section');

  const email = $('reset-email')?.value?.trim();

  if (!email) {
    if (resetStatus) resetStatus.textContent = 'Email is required';
    return;
  }

  try {
    toggleButtonBusy(resetSubmit, true, 'Sending...');
    if (resetStatus) resetStatus.textContent = '';

    await requestEnvelope(
      `${apiBase}/auth/reset/request`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email }),
      },
      'Reset request failed'
    );

    resetEmailForConfirm = email;
    if (resetStatus) resetStatus.textContent = 'Reset code sent! Check your email.';
    if (resetCodeSection) resetCodeSection.classList.remove('hidden');
  } catch (err) {
    if (resetStatus) resetStatus.textContent = err.message;
  } finally {
    toggleButtonBusy(resetSubmit, false);
  }
};

const handleResetConfirm = async (event) => {
  event.preventDefault();
  const resetStatus = $('reset-status');
  const confirmSubmit = $('reset-confirm-submit');

  const code = $('reset-code')?.value?.trim();
  const newPassword = $('reset-new-password')?.value;

  if (!code || !newPassword) {
    if (resetStatus) resetStatus.textContent = 'Code and new password are required';
    return;
  }

  if (newPassword.length < 8) {
    if (resetStatus) resetStatus.textContent = 'Password must be at least 8 characters';
    return;
  }

  try {
    toggleButtonBusy(confirmSubmit, true, 'Resetting...');
    if (resetStatus) resetStatus.textContent = '';

    await requestEnvelope(
      `${apiBase}/auth/reset/confirm`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          token: code,
          new_password: newPassword,
        }),
      },
      'Password reset failed'
    );

    if (resetStatus) resetStatus.textContent = 'Password reset successful! You can now sign in.';

    // Clear form
    if ($('reset-code')) $('reset-code').value = '';
    if ($('reset-new-password')) $('reset-new-password').value = '';
    if ($('reset-code-section')) $('reset-code-section').classList.add('hidden');

    // Pre-fill login email
    if ($('email')) $('email').value = resetEmailForConfirm;

    setTimeout(() => showAuthForm('login'), 1500);
  } catch (err) {
    if (resetStatus) resetStatus.textContent = err.message;
  } finally {
    toggleButtonBusy(confirmSubmit, false);
  }
};

// =============================================================================
// Email Token URL Handlers
// =============================================================================

let pendingResetToken = '';

const handleResetTokenCallback = async () => {
  const urlParams = new URLSearchParams(window.location.search);
  const resetToken = urlParams.get('reset_token');

  if (!resetToken) {
    return false;
  }

  // Clear URL params
  window.history.replaceState({}, document.title, window.location.pathname);

  // Store the token for the form
  pendingResetToken = resetToken;

  // Show the reset form with just the new password field visible
  showAuthForm('reset');

  const resetStatus = $('reset-status');
  const resetCodeSection = $('reset-code-section');
  const resetRequestForm = $('reset-request-form');

  // Hide the email request section and show the password input
  if (resetRequestForm) resetRequestForm.classList.add('hidden');
  if (resetCodeSection) {
    resetCodeSection.classList.remove('hidden');
    // Hide the code input since we have the token from URL
    const codeField = resetCodeSection.querySelector('.field:first-child');
    if (codeField) codeField.classList.add('hidden');
  }

  // Update the description
  const subtext = resetCodeSection?.previousElementSibling;
  if (subtext && subtext.classList.contains('subtext')) {
    subtext.textContent = 'Enter your new password.';
  }

  if (resetStatus) resetStatus.textContent = 'Password reset link verified. Enter your new password.';

  return true;
};

const handleResetWithToken = async (event) => {
  event.preventDefault();
  const resetStatus = $('reset-status');
  const confirmSubmit = $('reset-confirm-submit');

  const newPassword = $('reset-new-password')?.value;

  if (!newPassword) {
    if (resetStatus) resetStatus.textContent = 'New password is required';
    return;
  }

  if (newPassword.length < 8) {
    if (resetStatus) resetStatus.textContent = 'Password must be at least 8 characters';
    return;
  }

  if (!pendingResetToken) {
    if (resetStatus) resetStatus.textContent = 'Reset token expired. Please request a new reset link.';
    return;
  }

  try {
    toggleButtonBusy(confirmSubmit, true, 'Resetting...');
    if (resetStatus) resetStatus.textContent = '';

    await requestEnvelope(
      `${apiBase}/auth/reset/confirm`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          token: pendingResetToken,
          new_password: newPassword,
        }),
      },
      'Password reset failed'
    );

    pendingResetToken = '';
    if (resetStatus) resetStatus.textContent = 'Password reset successful! You can now sign in.';

    // Clear form and show login after delay
    if ($('reset-new-password')) $('reset-new-password').value = '';
    setTimeout(() => showAuthForm('login'), 1500);
  } catch (err) {
    if (resetStatus) resetStatus.textContent = err.message;
  } finally {
    toggleButtonBusy(confirmSubmit, false);
  }
};

const handleVerifyTokenCallback = async () => {
  const urlParams = new URLSearchParams(window.location.search);
  const verifyToken = urlParams.get('verify_token');

  if (!verifyToken) {
    return false;
  }

  // Clear URL params
  window.history.replaceState({}, document.title, window.location.pathname);

  // Show a verification status message
  showStatus('Verifying email...');

  try {
    await requestEnvelope(
      `${apiBase}/auth/verify_email`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ token: verifyToken }),
      },
      'Email verification failed'
    );

    showStatus('Email verified successfully! You can now sign in.');

    // Show login form
    showAuthForm('login');
  } catch (err) {
    showStatus(`Email verification failed: ${err.message}`);
  }

  return true;
};

// =============================================================================
// Conversations
// =============================================================================

const fetchConversations = async () => {
  if (!state.accessToken) return;

  try {
    const envelope = await requestEnvelope(
      `${apiBase}/conversations?limit=50`,
      { headers: headers() },
      'Failed to load conversations'
    );
    state.conversations = envelope.data?.items || [];
    renderConversationList();
  } catch (err) {
    console.warn('Failed to fetch conversations:', err.message);
  }
};

const renderConversationList = () => {
  if (!conversationListEl) return;

  const search = conversationSearchEl?.value?.toLowerCase() || '';
  const filtered = state.conversations.filter((c) =>
    !search || (c.title || '').toLowerCase().includes(search) || c.id.toLowerCase().includes(search)
  );

  if (!filtered.length) {
    conversationListEl.innerHTML = state.accessToken
      ? '<div class="empty">No conversations</div>'
      : '<div class="empty">Sign in to see conversations</div>';
    return;
  }

  conversationListEl.innerHTML = filtered
    .map((c) => {
      const isActive = c.id === state.conversationId;
      const title = escapeHtml(c.title || 'Untitled conversation');
      const date = c.updated_at ? new Date(c.updated_at).toLocaleDateString() : '';
      return `
        <div class="conversation-item ${isActive ? 'active' : ''}" data-id="${escapeHtml(c.id)}">
          <div class="title">${title}</div>
          <div class="meta">${date}</div>
        </div>
      `;
    })
    .join('');

  conversationListEl.querySelectorAll('.conversation-item').forEach((item) => {
    item.addEventListener('click', () => loadConversation(item.dataset.id));
  });
};

const loadConversation = async (conversationId) => {
  if (!conversationId) return;

  state.conversationId = conversationId;
  state.lastAssistant = null;
  if (conversationLabel) conversationLabel.textContent = 'Loading...';

  try {
    const envelope = await requestEnvelope(
      `${apiBase}/conversations/${conversationId}`,
      { headers: headers() },
      'Failed to load conversation'
    );

    const convo = envelope.data;
    if (conversationLabel) conversationLabel.textContent = convo.title || 'Conversation';

    const messagesEnvelope = await requestEnvelope(
      `${apiBase}/conversations/${conversationId}/messages?limit=100`,
      { headers: headers() },
      'Failed to load messages'
    );

    renderMessages(messagesEnvelope.data?.messages || []);
    renderConversationList();

    // Load draft
    const messageInput = $('message-input');
    if (messageInput) {
      messageInput.value = getDraft(conversationId);
    }
  } catch (err) {
    showStatus(err.message, true);
    if (conversationLabel) conversationLabel.textContent = 'Error loading';
  }
};

const setConversation = (id) => {
  state.conversationId = id;
  if (conversationLabel) conversationLabel.textContent = id ? `Conversation ${id.slice(0, 8)}...` : 'New conversation';
  if (!id) {
    state.lastAssistant = null;
    renderPreferencePanel();
  }
};

const newConversation = () => {
  setConversation(null);
  if (messagesEl) messagesEl.innerHTML = '';
  showStatus('New thread ready');
  updateEmptyState();
  renderConversationList();

  const messageInput = $('message-input');
  if (messageInput) messageInput.value = getDraft(null);
};

// =============================================================================
// Messages
// =============================================================================

const renderMessages = (messages) => {
  if (!messagesEl) return;

  if (!messages.length) {
    messagesEl.innerHTML = '';
    updateEmptyState();
    return;
  }

  messagesEl.innerHTML = messages.map((m) => renderMessage(m)).join('');

  const lastAssistant = messages.filter((m) => m.role === 'assistant').pop();
  if (lastAssistant) {
    state.lastAssistant = {
      conversationId: state.conversationId,
      messageId: lastAssistant.id,
      adapters: lastAssistant.adapters || [],
      adapterGates: lastAssistant.adapter_gates || [],
      routingTrace: lastAssistant.routing_trace || [],
      workflowTrace: lastAssistant.workflow_trace || [],
      contextSnippets: lastAssistant.context_snippets || [],
    };
    renderPreferencePanel();
  }

  updateEmptyState();
  scrollToBottom();
};

const renderMessage = (m) => {
  const role = escapeHtml(m.role || 'unknown');
  const content = escapeHtml(m.content || '');

  const metaBits = [];
  if (m.token_count) metaBits.push(`${m.token_count} tokens`);
  if (m.model) metaBits.push(escapeHtml(m.model));

  // Render citations as clickable links per SPEC §17
  let citationsHtml = '';
  const citations = m.content_struct?.citations || [];
  if (citations.length) {
    citationsHtml = `
      <div class="citations-row">
        ${citations.map((c, i) => {
          const path = escapeHtml(c.source_path || c.chunk_id || `Citation ${i + 1}`);
          const label = path.split('/').pop() || path;
          // JSON.stringify escapes internal quotes; only need & and " for double-quoted attr
          const snippetData = JSON.stringify({
            source_path: c.source_path || '',
            chunk_id: c.chunk_id || '',
            content: c.content || c.snippet || '',
            context_id: c.context_id || '',
            chunk_index: c.chunk_index,
          }).replace(/&/g, '&amp;').replace(/"/g, '&quot;');
          return `<span class="citation-link" title="${path}" data-citation="${snippetData}" onclick="showCitationModal(this)">${escapeHtml(label)}</span>`;
        }).join('')}
      </div>
    `;
  }

  return `
    <div class="message ${role}" data-id="${escapeHtml(m.id || '')}">
      <div class="role">${role}</div>
      <div>
        <div class="bubble">${content}</div>
        ${citationsHtml}
        ${metaBits.length ? `<div class="meta">${metaBits.join(' · ')}</div>` : ''}
      </div>
    </div>
  `;
};

const appendMessage = (role, content, meta = '') => {
  const wrapper = document.createElement('div');
  wrapper.className = `message ${role}`;
  const roleEl = document.createElement('div');
  roleEl.className = 'role';
  roleEl.textContent = role;
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = content;
  const metaEl = document.createElement('div');
  metaEl.className = 'meta';
  metaEl.textContent = meta;
  wrapper.appendChild(roleEl);
  const contentWrap = document.createElement('div');
  contentWrap.appendChild(bubble);
  if (meta) contentWrap.appendChild(metaEl);
  wrapper.appendChild(contentWrap);
  if (messagesEl) {
    messagesEl.appendChild(wrapper);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }
  updateEmptyState();
  return wrapper;
};

/**
 * Create a streaming message element that can be updated token-by-token.
 * Returns an object with update() and finalize() methods.
 */
const createStreamingMessage = (role) => {
  const wrapper = document.createElement('div');
  wrapper.className = `message ${role} streaming`;
  const roleEl = document.createElement('div');
  roleEl.className = 'role';
  roleEl.textContent = role;
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = '';
  const metaEl = document.createElement('div');
  metaEl.className = 'meta';
  wrapper.appendChild(roleEl);
  const contentWrap = document.createElement('div');
  contentWrap.appendChild(bubble);
  contentWrap.appendChild(metaEl);
  wrapper.appendChild(contentWrap);

  if (messagesEl) {
    messagesEl.appendChild(wrapper);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }
  updateEmptyState();

  let content = '';

  return {
    /** Append a token to the message */
    update(token) {
      content += token;
      bubble.textContent = content;
      if (messagesEl) messagesEl.scrollTop = messagesEl.scrollHeight;
    },
    /** Finalize the message with optional meta info */
    finalize(meta = '') {
      wrapper.classList.remove('streaming');
      if (meta) metaEl.textContent = meta;
    },
    /** Get the accumulated content */
    getContent() {
      return content;
    },
    /** Get the wrapper element */
    getElement() {
      return wrapper;
    },
  };
};

const scrollToBottom = () => {
  if (messagesEl) messagesEl.scrollTop = messagesEl.scrollHeight;
};

// =============================================================================
// Chat submission
// =============================================================================

let chatSocket = null;
let chatSocketConnecting = false;
let chatSocketReconnectTimer = null;
let isStreaming = false;

const updateStreamingUI = (streaming) => {
  isStreaming = streaming;
  const sendBtn = $('send-btn');
  const stopBtn = $('stop-stream-btn');
  if (streaming) {
    if (sendBtn) sendBtn.classList.add('hidden');
    if (stopBtn) stopBtn.classList.remove('hidden');
  } else {
    if (sendBtn) sendBtn.classList.remove('hidden');
    if (stopBtn) stopBtn.classList.add('hidden');
  }
};

const cancelStreaming = () => {
  if (!isStreaming || !chatSocket || chatSocket.readyState !== WebSocket.OPEN) {
    return;
  }
  try {
    chatSocket.send(JSON.stringify({ action: 'cancel' }));
    showStatus('Cancelling...');
  } catch (err) {
    console.warn('Failed to send cancel:', err);
  }
};

const cleanupWebSocket = () => {
  if (chatSocketReconnectTimer) {
    clearTimeout(chatSocketReconnectTimer);
    chatSocketReconnectTimer = null;
  }
  if (chatSocket) {
    chatSocket.onopen = null;
    chatSocket.onerror = null;
    chatSocket.onclose = null;
    chatSocket.onmessage = null;
    if (chatSocket.readyState === WebSocket.OPEN || chatSocket.readyState === WebSocket.CONNECTING) {
      chatSocket.close();
    }
    chatSocket = null;
  }
  chatSocketConnecting = false;
};

window.addEventListener('beforeunload', cleanupWebSocket);

const connectWebSocket = () => {
  if (chatSocketConnecting) return chatSocket;
  if (chatSocket && [WebSocket.OPEN, WebSocket.CONNECTING].includes(chatSocket.readyState)) {
    return chatSocket;
  }
  chatSocketConnecting = true;
  const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
  const wsUrl = `${protocol}://${window.location.host}${apiBase}/chat/stream`;
  chatSocket = new WebSocket(wsUrl);
  chatSocket.onopen = () => { chatSocketConnecting = false; };
  chatSocket.onerror = () => { chatSocketConnecting = false; };
  chatSocket.onclose = () => {
    chatSocketConnecting = false;
    chatSocket = null;
    if (chatSocketReconnectTimer) clearTimeout(chatSocketReconnectTimer);
    chatSocketReconnectTimer = setTimeout(() => {
      chatSocketReconnectTimer = null;
      connectWebSocket();
    }, 2000);
  };
  return chatSocket;
};

const ensureWebSocket = () =>
  new Promise((resolve, reject) => {
    const socket = connectWebSocket();
    if (!socket) {
      reject(new Error('WebSocket unavailable'));
      return;
    }
    if (socket.readyState === WebSocket.OPEN) {
      resolve(socket);
      return;
    }
    const timeout = setTimeout(() => {
      cleanup();
      reject(new Error('WebSocket connection timeout'));
    }, 5000);
    const cleanup = () => {
      clearTimeout(timeout);
      socket.removeEventListener('open', handleOpen);
      socket.removeEventListener('error', handleError);
    };
    const handleOpen = () => { cleanup(); resolve(socket); };
    const handleError = () => { cleanup(); reject(new Error('WebSocket connection failed')); };
    socket.addEventListener('open', handleOpen);
    socket.addEventListener('error', handleError);
  });

const sendMessage = async (event) => {
  event.preventDefault();
  const messageInput = $('message-input');
  const content = messageInput?.value?.trim();
  if (!content) return;
  if (!state.accessToken) {
    showStatus('Sign in to chat.', true);
    return;
  }

  const payload = {
    conversation_id: state.conversationId || undefined,
    message: { content, mode: 'text' },
    context_id: $('context-id')?.value || undefined,
    workflow_id: $('workflow-id')?.value || undefined,
  };
  const idempotencyKey = `chat-${stableHash(JSON.stringify(payload))}`;

  const handleChatResponse = (data) => {
    setConversation(data.conversation_id);
    const structuredSegments = data.content_struct?.segments;
    const renderedContent =
      structuredSegments?.map((seg) => (typeof seg === 'string' ? seg : seg?.text || '')).join(' ')
      || data.content;
    const citations = data.content_struct?.citations || [];
    const metaBits = [];
    if (data.adapters?.length) metaBits.push(`adapters: ${data.adapters.join(', ')}`);
    if (data.context_snippets?.length) metaBits.push(`context: ${data.context_snippets.length} snippets`);
    if (data.usage?.total_tokens) metaBits.push(`usage: ${data.usage.total_tokens} tokens`);
    if (citations.length) metaBits.push(`citations: ${citations.length}`);
    appendMessage('assistant', renderedContent, metaBits.join(' · '));
    state.lastAssistant = {
      conversationId: data.conversation_id,
      messageId: data.message_id,
      adapters: data.adapters || [],
      adapterGates: data.adapter_gates || [],
      routingTrace: data.routing_trace || [],
      workflowTrace: data.workflow_trace || [],
      contextSnippets: data.context_snippets || [],
    };
    renderPreferencePanel();
    showStatus('');
    fetchConversations(); // Update sidebar
  };

  /**
   * SPEC §18: Streaming WebSocket chat with token events.
   * Handles events: token, trace, message_done, streaming_complete, error, cancel_ack
   */
  const chatViaWebSocketStreaming = async () => {
    const ws = await ensureWebSocket();
    return new Promise((resolve, reject) => {
      let settled = false;
      let streamingMsg = null;
      let messageDoneData = {};

      const cleanup = () => {
        ws.removeEventListener('message', handleMessage);
        ws.removeEventListener('error', handleError);
        ws.removeEventListener('close', handleClose);
      };

      const handleMessage = (event) => {
        if (settled) return;
        try {
          const msg = JSON.parse(event.data);

          // SPEC §18: Check for streaming events ({"event": "...", "data": "..."})
          if (msg.event) {
            switch (msg.event) {
              case 'token':
                // Create streaming message on first token
                if (!streamingMsg) {
                  streamingMsg = createStreamingMessage('assistant');
                  showStatus('');
                }
                streamingMsg.update(msg.data || '');
                break;

              case 'trace':
                // Optional: Could show trace info in UI
                console.debug('Workflow trace:', msg.data);
                break;

              case 'message_done':
                // Streaming visually complete, but wait for streaming_complete for IDs
                messageDoneData = msg.data || {};
                if (streamingMsg) {
                  const adapters = (messageDoneData.adapters || []).map(a => a?.name || a?.id || a).filter(Boolean);
                  streamingMsg.finalize(adapters.length ? `Adapters: ${adapters.join(', ')}` : '');
                }
                // Don't resolve yet - wait for streaming_complete with message_id
                break;

              case 'streaming_complete':
                // Final event with message_id and conversation_id after DB save
                settled = true;
                cleanup();
                // Merge with any data from message_done
                const finalData = { ...messageDoneData, ...(msg.data || {}) };
                resolve(finalData);
                break;

              case 'error':
                settled = true;
                cleanup();
                if (streamingMsg) {
                  streamingMsg.finalize('Error occurred');
                }
                reject(new Error(msg.data?.message || 'Streaming error'));
                break;

              case 'cancel_ack':
                settled = true;
                cleanup();
                if (streamingMsg) {
                  streamingMsg.finalize('Cancelled');
                }
                resolve({ cancelled: true });
                break;

              default:
                console.debug('Unknown streaming event:', msg.event);
            }
          } else if (msg.status) {
            // Legacy non-streaming response format
            settled = true;
            cleanup();
            if (msg.status === 'ok') {
              resolve(msg.data);
            } else {
              reject(new Error(extractError(msg.error, 'Chat failed')));
            }
          }
        } catch (err) {
          if (!settled) {
            settled = true;
            cleanup();
            reject(new Error(err instanceof SyntaxError ? 'Received invalid response' : err.message));
          }
        }
      };

      const handleError = () => {
        if (!settled) {
          settled = true;
          cleanup();
          if (streamingMsg) streamingMsg.finalize('Connection error');
          reject(new Error('WebSocket failed'));
        }
      };

      const handleClose = () => {
        if (!settled) {
          settled = true;
          cleanup();
          // If we got message_done but not streaming_complete, resolve with what we have
          if (Object.keys(messageDoneData).length > 0) {
            if (streamingMsg) streamingMsg.finalize('');
            resolve(messageDoneData);
          } else {
            if (streamingMsg) streamingMsg.finalize('Connection closed');
            reject(new Error('Connection closed'));
          }
        }
      };

      ws.addEventListener('message', handleMessage);
      ws.addEventListener('error', handleError);
      ws.addEventListener('close', handleClose);

      // SPEC §18: stream: true enables token streaming
      ws.send(JSON.stringify({
        idempotency_key: idempotencyKey,
        request_id: randomIdempotencyKey(),
        message: payload.message.content,
        workflow_id: payload.workflow_id,
        context_id: payload.context_id,
        conversation_id: payload.conversation_id,
        access_token: state.accessToken,
        session_id: state.sessionId,
        tenant_id: state.tenantId,
        stream: true,
      }));
    });
  };

  try {
    toggleButtonBusy(sendBtn, true, 'Sending...');
    if (messageInput) messageInput.value = '';
    saveDraft(state.conversationId, '');
    appendMessage('user', content);
    showStatus('Thinking...');
    updateStreamingUI(true);

    const data = await chatViaWebSocketStreaming().catch(async () => {
      // Fallback to REST API if WebSocket fails
      const envelope = await requestEnvelope(
        `${apiBase}/chat`,
        { method: 'POST', headers: headers(idempotencyKey), body: JSON.stringify({ ...payload, stream: false }) },
        'Chat failed'
      );
      return envelope.data;
    });

    // Only call handleChatResponse for non-streaming or fallback responses
    // Streaming messages are already rendered by createStreamingMessage
    if (data && !data.cancelled && data.message_id) {
      // For non-streaming fallback, render the message
      if (!document.querySelector('.message.assistant.streaming, .message.assistant:last-child')) {
        handleChatResponse(data);
      } else {
        // Update state for streaming (message already rendered)
        state.lastAssistant = {
          conversationId: data.conversation_id,
          messageId: data.message_id,
          adapters: data.adapters || [],
          adapterGates: data.adapter_gates || [],
          routingTrace: data.routing_trace || [],
          workflowTrace: data.workflow_trace || [],
          contextSnippets: data.context_snippets || [],
        };
        setConversation(data.conversation_id);
        renderPreferencePanel();
        fetchConversations();
      }
    }
  } catch (err) {
    showStatus(err.message, true);
  } finally {
    toggleButtonBusy(sendBtn, false, 'Send');
    updateStreamingUI(false);
  }
};

// =============================================================================
// Contexts
// =============================================================================

const fetchContexts = async () => {
  if (!state.accessToken) return;

  try {
    const envelope = await requestEnvelope(
      `${apiBase}/contexts?limit=100`,
      { headers: headers() },
      'Failed to load contexts'
    );
    state.contexts = envelope.data?.items || [];
    renderContextsList();
    updateContextSelects();
  } catch (err) {
    console.warn('Failed to fetch contexts:', err.message);
  }
};

const renderContextsList = () => {
  const list = $('contexts-list');
  if (!list) return;

  if (!state.contexts.length) {
    list.innerHTML = '<div class="empty">No contexts yet. Create one above.</div>';
    return;
  }

  list.innerHTML = state.contexts
    .map((ctx) => {
      const isSelected = ctx.id === state.selectedContext?.id;
      return `
        <div class="context-card ${isSelected ? 'selected' : ''}" data-id="${escapeHtml(ctx.id)}">
          <div class="name">${escapeHtml(ctx.name)}</div>
          <div class="description">${escapeHtml(ctx.description || 'No description')}</div>
          <div class="stats">
            <span class="stat">ID: ${escapeHtml(ctx.id.slice(0, 8))}...</span>
            <span class="stat">Created: ${new Date(ctx.created_at).toLocaleDateString()}</span>
          </div>
        </div>
      `;
    })
    .join('');

  list.querySelectorAll('.context-card').forEach((card) => {
    card.addEventListener('click', () => selectContext(card.dataset.id));
  });
};

const selectContext = async (contextId) => {
  const ctx = state.contexts.find((c) => c.id === contextId);
  if (!ctx) return;

  state.selectedContext = ctx;
  renderContextsList();

  const details = $('context-details');
  if (!details) return;

  try {
    const envelope = await requestEnvelope(
      `${apiBase}/contexts/${contextId}/chunks?limit=20`,
      { headers: headers() },
      'Failed to load context chunks'
    );

    const chunks = envelope.data?.items || [];

    details.innerHTML = `
      <div class="detail-header">
        <h4>${escapeHtml(ctx.name)}</h4>
        <span class="visibility-badge ${ctx.visibility || 'private'}">${ctx.visibility || 'private'}</span>
      </div>
      <div class="detail-row">
        <span class="detail-label">ID</span>
        <span class="monospace">${escapeHtml(ctx.id)}</span>
      </div>
      <div class="detail-row">
        <span class="detail-label">Description</span>
        <span>${escapeHtml(ctx.description || '-')}</span>
      </div>
      <div class="detail-row">
        <span class="detail-label">Chunks</span>
        <span>${chunks.length} chunks loaded</span>
      </div>
      <div class="detail-row">
        <span class="detail-label">Created</span>
        <span>${new Date(ctx.created_at).toLocaleString()}</span>
      </div>
      ${chunks.length ? `
        <div class="divider"></div>
        <h4>Recent chunks</h4>
        <div class="code-block">${chunks.slice(0, 5).map((c) =>
          `[${escapeHtml(String(c.id).slice(0, 8))}] ${escapeHtml((c.content || '').slice(0, 100))}...`
        ).join('\n\n')}</div>
      ` : ''}
    `;
  } catch (err) {
    details.innerHTML = `<div class="empty">Error loading context: ${escapeHtml(err.message)}</div>`;
  }

  // Also load sources for this context
  await fetchContextSources(contextId);

  // Show the add source section
  const addSourceSection = $('add-source-section');
  if (addSourceSection) addSourceSection.classList.remove('hidden');
};

// =============================================================================
// Context Sources
// =============================================================================

const fetchContextSources = async (contextId) => {
  const sourcesList = $('context-sources-list');
  if (!sourcesList) return;

  if (!contextId) {
    sourcesList.innerHTML = '<div class="empty">Select a context to view sources</div>';
    return;
  }

  try {
    sourcesList.innerHTML = '<div class="empty">Loading sources...</div>';

    const envelope = await requestEnvelope(
      `${apiBase}/contexts/${contextId}/sources`,
      { headers: headers() },
      'Failed to load sources'
    );

    const sources = envelope.data?.items || [];
    renderContextSources(sources);
  } catch (err) {
    sourcesList.innerHTML = `<div class="empty">Error: ${escapeHtml(err.message)}</div>`;
  }
};

const renderContextSources = (sources) => {
  const sourcesList = $('context-sources-list');
  if (!sourcesList) return;

  if (!sources.length) {
    sourcesList.innerHTML = '<div class="empty">No sources added yet</div>';
    return;
  }

  sourcesList.innerHTML = sources
    .map((s) => {
      const date = s.created_at ? new Date(s.created_at).toLocaleDateString() : '-';
      return `
        <div class="source-item">
          <div class="source-path monospace">${escapeHtml(s.fs_path || s.path || '-')}</div>
          <div class="source-meta">
            <span>${s.recursive ? 'Recursive' : 'Single file'}</span>
            <span>Added ${date}</span>
          </div>
        </div>
      `;
    })
    .join('');
};

const addContextSource = async (event) => {
  event.preventDefault();

  if (!state.selectedContext) {
    const statusEl = $('add-source-status');
    if (statusEl) statusEl.textContent = 'No context selected';
    return;
  }

  const pathEl = $('source-path');
  const recursiveEl = $('source-recursive');
  const statusEl = $('add-source-status');
  const submitBtn = $('add-source-btn');

  const fsPath = pathEl?.value?.trim();
  const recursive = recursiveEl?.checked ?? true;

  if (!fsPath) {
    if (statusEl) statusEl.textContent = 'Path is required';
    return;
  }

  try {
    toggleButtonBusy(submitBtn, true, 'Adding...');
    if (statusEl) statusEl.textContent = '';

    await requestEnvelope(
      `${apiBase}/contexts/${state.selectedContext.id}/sources`,
      {
        method: 'POST',
        headers: headers(),
        body: JSON.stringify({ fs_path: fsPath, recursive }),
      },
      'Failed to add source'
    );

    if (statusEl) statusEl.textContent = 'Source added and ingested!';
    if (pathEl) pathEl.value = '';

    // Reload sources
    await fetchContextSources(state.selectedContext.id);

    // Also reload the context details to update chunk count
    await selectContext(state.selectedContext.id);
  } catch (err) {
    if (statusEl) statusEl.textContent = err.message;
  } finally {
    toggleButtonBusy(submitBtn, false);
  }
};

const createContext = async () => {
  const nameEl = $('new-context-name');
  const descEl = $('new-context-description');
  const statusEl = $('context-create-status');

  const name = nameEl?.value?.trim();
  const description = descEl?.value?.trim();

  if (!name) {
    if (statusEl) statusEl.textContent = 'Name is required';
    return;
  }

  try {
    if (statusEl) statusEl.textContent = 'Creating...';

    await requestEnvelope(
      `${apiBase}/contexts`,
      { method: 'POST', headers: headers(), body: JSON.stringify({ name, description: description || undefined }) },
      'Failed to create context'
    );

    if (statusEl) statusEl.textContent = 'Context created!';
    if (nameEl) nameEl.value = '';
    if (descEl) descEl.value = '';

    await fetchContexts();
  } catch (err) {
    if (statusEl) statusEl.textContent = `Error: ${err.message}`;
  }
};

const updateContextSelects = () => {
  const selects = [$('context-id'), $('upload-context-id')];

  selects.forEach((select) => {
    if (!select) return;
    const currentValue = select.value;
    const firstOption = select.options[0]?.outerHTML || '<option value="">No context</option>';

    select.innerHTML = firstOption + state.contexts
      .map((ctx) => `<option value="${escapeHtml(ctx.id)}">${escapeHtml(ctx.name)}</option>`)
      .join('');

    select.value = currentValue;
  });
};

// =============================================================================
// Artifacts
// =============================================================================

const fetchArtifacts = async () => {
  if (!state.accessToken) return;

  const typeFilter = $('artifact-type-filter')?.value || '';
  const visibilityFilter = $('artifact-visibility-filter')?.value || '';

  let url = `${apiBase}/artifacts?limit=100`;
  if (typeFilter) url += `&type=${typeFilter}`;
  if (visibilityFilter) url += `&visibility=${visibilityFilter}`;

  try {
    const envelope = await requestEnvelope(url, { headers: headers() }, 'Failed to load artifacts');
    state.artifacts = envelope.data?.items || [];
    renderArtifactsList();
  } catch (err) {
    console.warn('Failed to fetch artifacts:', err.message);
    const list = $('artifacts-list');
    if (list) list.innerHTML = `<div class="empty">Error: ${escapeHtml(err.message)}</div>`;
  }
};

const renderArtifactsList = () => {
  const list = $('artifacts-list');
  if (!list) return;

  if (!state.artifacts.length) {
    list.innerHTML = '<div class="empty">No artifacts found</div>';
    return;
  }

  const rows = state.artifacts
    .map((a) => {
      const isSelected = a.id === state.selectedArtifact?.id;
      return `
        <tr class="clickable ${isSelected ? 'selected' : ''}" data-id="${escapeHtml(a.id)}">
          <td><span class="type-badge ${a.type || 'unknown'}">${escapeHtml(a.type || 'unknown')}</span></td>
          <td>${escapeHtml(a.name || a.id)}</td>
          <td><span class="visibility-badge ${a.visibility || 'private'}">${escapeHtml(a.visibility || 'private')}</span></td>
          <td>v${a.version || 1}</td>
          <td>${new Date(a.updated_at).toLocaleDateString()}</td>
        </tr>
      `;
    })
    .join('');

  list.innerHTML = `
    <table class="table">
      <thead><tr><th>Type</th><th>Name</th><th>Visibility</th><th>Version</th><th>Updated</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
  `;

  list.querySelectorAll('tr.clickable').forEach((row) => {
    row.addEventListener('click', () => selectArtifact(row.dataset.id));
  });
};

const selectArtifact = async (artifactId) => {
  const artifact = state.artifacts.find((a) => a.id === artifactId);
  if (!artifact) return;

  state.selectedArtifact = artifact;
  renderArtifactsList();

  const details = $('artifact-details');
  if (details) {
    details.innerHTML = `
      <div class="detail-header">
        <h4>${escapeHtml(artifact.name || artifact.id)}</h4>
        <span class="type-badge ${artifact.type || 'unknown'}">${escapeHtml(artifact.type || 'unknown')}</span>
      </div>
      <div class="detail-row">
        <span class="detail-label">ID</span>
        <span class="monospace">${escapeHtml(artifact.id)}</span>
      </div>
      <div class="detail-row">
        <span class="detail-label">Description</span>
        <span>${escapeHtml(artifact.description || '-')}</span>
      </div>
      <div class="detail-row">
        <span class="detail-label">Version</span>
        <span>v${artifact.version || 1}</span>
      </div>
      <div class="detail-row">
        <span class="detail-label">Owner</span>
        <span>${escapeHtml(artifact.owner_user_id || 'system')}</span>
      </div>
      <div class="divider"></div>
      <h4>Schema</h4>
      <pre class="schema-viewer">${escapeHtml(JSON.stringify(artifact.schema || {}, null, 2))}</pre>
    `;
  }

  await fetchArtifactVersions(artifactId);
};

const fetchArtifactVersions = async (artifactId) => {
  const versions = $('artifact-versions');
  if (!versions) return;

  try {
    const envelope = await requestEnvelope(
      `${apiBase}/artifacts/${artifactId}/versions?limit=10`,
      { headers: headers() },
      'Failed to load versions'
    );

    const items = envelope.data?.items || [];

    if (!items.length) {
      versions.innerHTML = '<div class="empty">No version history available</div>';
      return;
    }

    const rows = items
      .map((v) => `
        <tr>
          <td>v${v.version}</td>
          <td>${new Date(v.created_at).toLocaleString()}</td>
          <td>${escapeHtml(v.change_note || '-')}</td>
        </tr>
      `)
      .join('');

    versions.innerHTML = `
      <table class="table">
        <thead><tr><th>Version</th><th>Created</th><th>Changes</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    `;
  } catch (err) {
    versions.innerHTML = `<div class="empty">Error: ${escapeHtml(err.message)}</div>`;
  }
};

// =============================================================================
// Preferences / Feedback
// =============================================================================

const renderPreferencePanel = () => {
  if (!preferenceStatusEl || !preferenceMetaEl || !preferenceRoutingEl || !preferenceTargetEl || !preferenceHintEl) return;
  if (!state.lastAssistant) {
    preferenceStatusEl.textContent = '';
    preferenceMetaEl.textContent = '';
    preferenceRoutingEl.textContent = '';
    preferenceTargetEl.textContent = 'No assistant message selected yet.';
    preferenceHintEl.textContent = 'Send a message to enable thumbs up/down feedback.';
    return;
  }
  preferenceHintEl.textContent = 'Thumbs apply to the latest assistant response.';
  const { conversationId, messageId, adapters, contextSnippets } = state.lastAssistant;
  preferenceTargetEl.textContent = `Conversation ${conversationId?.slice(0, 8) || '?'}... · Message ${messageId?.slice(0, 8) || '?'}...`;
  const meta = {
    adapters: adapters || [],
    context_snippets: contextSnippets?.length || 0,
    adapter_gates: state.lastAssistant.adapterGates || [],
  };
  preferenceMetaEl.textContent = JSON.stringify(meta, null, 2);
  if (state.lastAssistant.routingTrace?.length || state.lastAssistant.workflowTrace?.length) {
    preferenceRoutingEl.textContent = JSON.stringify({
      routing_trace: state.lastAssistant.routingTrace || [],
      workflow_trace: state.lastAssistant.workflowTrace || [],
    }, null, 2);
  } else {
    preferenceRoutingEl.textContent = 'No routing trace';
  }
};

const sanitizeNotes = (notes) => {
  if (!notes || typeof notes !== 'string') return undefined;
  const trimmed = notes.trim().slice(0, 2000);
  return trimmed || undefined;
};

const sendPreference = async (isPositive) => {
  if (!state.lastAssistant) {
    if (preferenceStatusEl) preferenceStatusEl.textContent = 'No assistant message to rate yet.';
    return;
  }
  try {
    if (preferenceStatusEl) preferenceStatusEl.textContent = 'Sending feedback...';
    const body = {
      conversation_id: state.lastAssistant.conversationId,
      message_id: state.lastAssistant.messageId,
      feedback: isPositive ? 'positive' : 'negative',
      explicit_signal: isPositive ? 'thumbs_up' : 'thumbs_down',
      routing_trace: state.lastAssistant.routingTrace || undefined,
      adapter_gates: state.lastAssistant.adapterGates || undefined,
      notes: sanitizeNotes(preferenceNotesEl?.value),
    };
    const idempotencyKey = `pref-${stableHash(JSON.stringify({ cid: body.conversation_id, mid: body.message_id, fb: body.feedback }))}`;
    await requestEnvelope(
      `${apiBase}/preferences`,
      { method: 'POST', headers: headers(idempotencyKey), body: JSON.stringify(body) },
      'Unable to record preference'
    );
    if (preferenceStatusEl) preferenceStatusEl.textContent = 'Thanks for your feedback!';
  } catch (err) {
    if (preferenceStatusEl) preferenceStatusEl.textContent = err.message;
  }
};

// =============================================================================
// Voice Input/Output
// =============================================================================

let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;
let currentAudio = null;

// Voice button references - initialized lazily in initEventListeners
let voiceInputBtn = null;
let voiceOutputBtn = null;

const startVoiceRecording = async () => {
  if (isRecording) return;

  let stream = null;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);
    audioChunks = [];

    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) {
        audioChunks.push(e.data);
      }
    };

    mediaRecorder.onstop = async () => {
      const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
      stream.getTracks().forEach(track => track.stop());
      await transcribeAudio(audioBlob);
    };

    mediaRecorder.start();
    isRecording = true;
    if (voiceInputBtn) {
      voiceInputBtn.classList.add('recording');
      voiceInputBtn.title = 'Release to stop recording';
    }
  } catch (err) {
    // Clean up stream if it was obtained but recording failed to start
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
    }
    console.error('Microphone access denied:', err);
    alert('Could not access microphone. Please check permissions.');
  }
};

const stopVoiceRecording = () => {
  if (!isRecording || !mediaRecorder) return;

  mediaRecorder.stop();
  isRecording = false;
  if (voiceInputBtn) {
    voiceInputBtn.classList.remove('recording');
    voiceInputBtn.title = 'Hold to record';
  }
};

const transcribeAudio = async (audioBlob) => {
  if (!state.accessToken) {
    alert('Please sign in to use voice input.');
    return;
  }

  try {
    const formData = new FormData();
    formData.append('file', audioBlob, 'recording.webm');

    const response = await fetch(`${apiBase}/voice/transcribe`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${state.accessToken}`,
      },
      body: formData,
    });

    const envelope = await response.json();
    if (envelope.status === 'ok' && envelope.data?.transcript) {
      // Insert transcribed text into the message input
      const messageInput = $('message-input');
      if (messageInput) {
        messageInput.value = (messageInput.value + ' ' + envelope.data.transcript).trim();
        messageInput.focus();
      }
    } else {
      console.error('Transcription failed:', envelope);
    }
  } catch (err) {
    console.error('Transcription error:', err);
  }
};

const speakText = async (text) => {
  if (!text) return;

  // Stop any currently playing audio
  if (currentAudio) {
    currentAudio.pause();
    currentAudio = null;
  }

  if (!state.accessToken) {
    // Fall back to browser speech synthesis
    const utterance = new SpeechSynthesisUtterance(text);
    window.speechSynthesis.speak(utterance);
    return;
  }

  try {
    if (voiceOutputBtn) voiceOutputBtn.classList.add('playing');

    const response = await fetch(`${apiBase}/voice/synthesize`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${state.accessToken}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text }),
    });

    const envelope = await response.json();
    if (envelope.status === 'ok' && envelope.data?.audio_url) {
      currentAudio = new Audio(envelope.data.audio_url);
      currentAudio.onended = () => {
        if (voiceOutputBtn) voiceOutputBtn.classList.remove('playing');
      };
      currentAudio.onerror = () => {
        if (voiceOutputBtn) voiceOutputBtn.classList.remove('playing');
        // Fall back to browser speech synthesis
        const utterance = new SpeechSynthesisUtterance(text);
        window.speechSynthesis.speak(utterance);
      };
      currentAudio.play();
    } else {
      // Fall back to browser speech synthesis
      if (voiceOutputBtn) voiceOutputBtn.classList.remove('playing');
      const utterance = new SpeechSynthesisUtterance(text);
      window.speechSynthesis.speak(utterance);
    }
  } catch (err) {
    console.error('Speech synthesis error:', err);
    if (voiceOutputBtn) voiceOutputBtn.classList.remove('playing');
    // Fall back to browser speech synthesis
    const utterance = new SpeechSynthesisUtterance(text);
    window.speechSynthesis.speak(utterance);
  }
};

const readLastResponse = () => {
  // Get the last assistant message content from DOM
  const lastAssistantBubble = document.querySelector('.message.assistant:last-of-type .bubble');
  const content = lastAssistantBubble?.textContent?.trim();
  if (!content) {
    alert('No assistant response to read.');
    return;
  }
  speakText(content);
};

// Voice button event listeners are initialized in initEventListeners() after DOM ready

// =============================================================================
// File upload
// =============================================================================

const refreshUploadLimits = async () => {
  if (!state.accessToken) return;
  try {
    const envelope = await requestEnvelope(
      `${apiBase}/files/limits`,
      { headers: headers() },
      'Failed to load upload limits'
    );
    uploadLimitBytes = envelope.data?.max_upload_bytes || uploadLimitBytes;
    renderUploadHint();

    const settingMaxUpload = $('setting-max-upload');
    if (settingMaxUpload) {
      settingMaxUpload.textContent = formatBytes(getUploadLimit());
    }
  } catch {
    // Silently ignore
  }
};

const validateUploadFile = (file) => {
  if (!file) return { ok: false, message: 'Choose a file to upload.' };
  const limit = getUploadLimit();
  if (file.size > limit) {
    return { ok: false, message: `File too large (${formatBytes(file.size)}). Max allowed is ${formatBytes(limit)}.` };
  }
  const name = (file.name || '').toLowerCase();
  const matchesType = file.type && ALLOWED_UPLOAD_TYPES.some((t) => file.type.startsWith(t));
  const matchesExt = ALLOWED_UPLOAD_EXTENSIONS.some((ext) => name.endsWith(ext));
  if (!matchesType && !matchesExt) {
    return { ok: false, message: `Unsupported file type. Allowed: ${ALLOWED_UPLOAD_EXTENSIONS.join(', ')}` };
  }
  return { ok: true };
};

const renderUploadHint = () => {
  if (!fileUploadHint) return;
  const file = fileUploadInput?.files?.[0];
  if (!file) {
    fileUploadHint.textContent = `Up to ${formatBytes(getUploadLimit())}. Supported: ${ALLOWED_UPLOAD_EXTENSIONS.join(', ')}`;
    return;
  }
  fileUploadHint.textContent = `${file.name} · ${formatBytes(file.size)} · ${file.type || 'unknown type'}`;
};

const setUploadStatus = (message, isError = false) => {
  if (!fileUploadStatus) return;
  fileUploadStatus.textContent = message;
  fileUploadStatus.style.color = isError ? '#b00020' : 'inherit';
};

const handleFileUpload = async (event) => {
  event?.preventDefault?.();
  if (!fileUploadInput) return;
  const file = fileUploadInput.files?.[0];
  const validation = validateUploadFile(file);
  if (!validation.ok) {
    setUploadStatus(validation.message, true);
    return;
  }
  if (!state.accessToken) {
    setUploadStatus('Sign in before uploading files.', true);
    return;
  }

  const contextId = (fileUploadContextId?.value || $('context-id')?.value || '').trim();
  const chunkSizeRaw = (fileUploadChunkSize?.value || '').trim();
  const chunkSize = chunkSizeRaw ? Number(chunkSizeRaw) : null;
  if (chunkSizeRaw && (!Number.isFinite(chunkSize) || chunkSize < 64 || chunkSize > 4000)) {
    setUploadStatus('Chunk size must be between 64 and 4000.', true);
    return;
  }

  const formData = new FormData();
  formData.append('file', file);
  if (contextId) formData.append('context_id', contextId);
  if (chunkSize) formData.append('chunk_size', chunkSize);
  const idempotencyKey = `upload-${stableHash(`${file.name}-${file.size}-${contextId || 'global'}`)}`;

  setUploadStatus('Uploading...');
  try {
    const envelope = await requestEnvelope(
      `${apiBase}/files/upload`,
      { method: 'POST', headers: authHeaders(idempotencyKey), body: formData },
      'Upload failed'
    );
    const uploaded = envelope.data || {};
    const destLabel = uploaded.context_id ? `context ${uploaded.context_id.slice(0, 8)}...` : 'your files area';
    const chunkLabel = uploaded.chunk_count ? ` · ${uploaded.chunk_count} chunk(s) indexed` : '';
    setUploadStatus(`Uploaded ${file.name} to ${destLabel}${chunkLabel}.`);

    // Refresh contexts if uploaded to one
    if (contextId) await fetchContexts();
  } catch (err) {
    setUploadStatus(err.message, true);
  }
};

// =============================================================================
// Settings
// =============================================================================

const fetchHealth = async () => {
  try {
    const resp = await fetch('/healthz');
    const data = await resp.json();

    const settingVersion = $('setting-version');
    const settingBuild = $('setting-build');

    if (settingVersion) settingVersion.textContent = data.version || '-';
    if (settingBuild) settingBuild.textContent = data.build || '-';
  } catch {
    // Ignore
  }
};

const handleClearDrafts = () => {
  clearAllDrafts();
  const draftsStatus = $('drafts-status');
  if (draftsStatus) draftsStatus.textContent = 'All drafts cleared';
};

const handleExportDrafts = () => {
  const drafts = loadDrafts();
  const json = JSON.stringify(drafts, null, 2);
  const blob = new Blob([json], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'liminal-drafts.json';
  a.click();
  URL.revokeObjectURL(url);

  const draftsStatus = $('drafts-status');
  if (draftsStatus) draftsStatus.textContent = 'Drafts exported';
};

// =============================================================================
// MFA Settings
// =============================================================================

let pendingMfaSecret = null;

const fetchMfaStatus = async () => {
  const statusEl = $('setting-mfa-status');
  const enableBtn = $('mfa-enable-btn');
  const disableBtn = $('mfa-show-disable-btn');

  if (!state.accessToken) {
    if (statusEl) statusEl.textContent = 'Sign in to manage MFA';
    return;
  }

  try {
    const envelope = await requestEnvelope(
      `${apiBase}/auth/mfa/status`,
      { headers: headers() },
      'Failed to check MFA status'
    );

    const { enabled, configured } = envelope.data;
    if (statusEl) {
      statusEl.textContent = enabled ? 'Enabled' : 'Disabled';
      statusEl.style.color = enabled ? '#0a7' : 'inherit';
    }

    // Show/hide appropriate buttons
    if (enableBtn) enableBtn.classList.toggle('hidden', enabled);
    if (disableBtn) disableBtn.classList.toggle('hidden', !enabled);
  } catch (err) {
    if (statusEl) statusEl.textContent = 'Unable to check';
  }
};

const startMfaSetup = async () => {
  if (!state.accessToken || !state.sessionId) {
    setMfaSetupStatus('Sign in first', true);
    return;
  }

  try {
    const envelope = await requestEnvelope(
      `${apiBase}/auth/mfa/request`,
      {
        method: 'POST',
        headers: authHeaders(),
        body: JSON.stringify({ session_id: state.sessionId }),
      },
      'Failed to start MFA setup'
    );

    const { otpauth_uri, status } = envelope.data;
    if (status === 'disabled') {
      setMfaSetupStatus('MFA is disabled on this server', true);
      return;
    }

    // Extract secret from URI for manual entry
    const secretMatch = otpauth_uri?.match(/secret=([A-Z2-7]+)/i);
    pendingMfaSecret = secretMatch ? secretMatch[1] : null;

    // Show setup section
    $('mfa-setup-section')?.classList.remove('hidden');
    $('mfa-enable-btn')?.classList.add('hidden');

    // Display secret for manual entry
    const secretDisplay = $('mfa-secret-display');
    if (secretDisplay) secretDisplay.textContent = pendingMfaSecret || 'N/A';

    // Generate QR code using a simple text display (or use qrcode library if available)
    const qrContainer = $('mfa-qr-code');
    if (qrContainer) {
      // Create a simple link to the otpauth URI that mobile apps can scan
      qrContainer.innerHTML = `
        <div class="qr-placeholder">
          <p>Open your authenticator app and add a new account using this URI:</p>
          <code style="word-break: break-all; font-size: 0.75rem;">${otpauth_uri || 'N/A'}</code>
        </div>
      `;
    }

    setMfaSetupStatus('Enter the 6-digit code from your authenticator app');
  } catch (err) {
    setMfaSetupStatus(err.message || 'Failed to start MFA setup', true);
  }
};

const verifyMfaSetup = async (event) => {
  event.preventDefault();

  const codeInput = $('mfa-setup-code');
  const code = codeInput?.value?.trim();

  if (!code || code.length !== 6) {
    setMfaSetupStatus('Enter a 6-digit code', true);
    return;
  }

  if (!state.sessionId) {
    setMfaSetupStatus('No session. Please sign in again.', true);
    return;
  }

  try {
    await requestEnvelope(
      `${apiBase}/auth/mfa/verify`,
      {
        method: 'POST',
        headers: authHeaders(),
        body: JSON.stringify({ session_id: state.sessionId, code }),
      },
      'Invalid code. Try again.'
    );

    // Success - hide setup, refresh status
    $('mfa-setup-section')?.classList.add('hidden');
    if (codeInput) codeInput.value = '';
    pendingMfaSecret = null;

    setMfaSetupStatus('');
    await fetchMfaStatus();

    alert('MFA enabled successfully!');
  } catch (err) {
    setMfaSetupStatus(err.message || 'Verification failed', true);
  }
};

const cancelMfaSetup = () => {
  $('mfa-setup-section')?.classList.add('hidden');
  $('mfa-enable-btn')?.classList.remove('hidden');
  const codeInput = $('mfa-setup-code');
  if (codeInput) codeInput.value = '';
  pendingMfaSecret = null;
  setMfaSetupStatus('');
};

const showMfaDisable = () => {
  $('mfa-disable-section')?.classList.remove('hidden');
  $('mfa-show-disable-btn')?.classList.add('hidden');
};

const hideMfaDisable = () => {
  $('mfa-disable-section')?.classList.add('hidden');
  $('mfa-show-disable-btn')?.classList.remove('hidden');
  const codeInput = $('mfa-disable-code');
  if (codeInput) codeInput.value = '';
  setMfaDisableStatus('');
};

const disableMfa = async (event) => {
  event.preventDefault();

  const codeInput = $('mfa-disable-code');
  const code = codeInput?.value?.trim();

  if (!code || code.length !== 6) {
    setMfaDisableStatus('Enter your current 6-digit MFA code', true);
    return;
  }

  try {
    await requestEnvelope(
      `${apiBase}/auth/mfa/disable`,
      {
        method: 'POST',
        headers: authHeaders(),
        body: JSON.stringify({ code }),
      },
      'Invalid code. Try again.'
    );

    // Success
    hideMfaDisable();
    await fetchMfaStatus();

    alert('MFA disabled.');
  } catch (err) {
    setMfaDisableStatus(err.message || 'Failed to disable MFA', true);
  }
};

const setMfaSetupStatus = (message, isError = false) => {
  const el = $('mfa-setup-status');
  if (!el) return;
  el.textContent = message;
  el.style.color = isError ? '#b00020' : 'inherit';
};

const setMfaDisableStatus = (message, isError = false) => {
  const el = $('mfa-disable-status');
  if (!el) return;
  el.textContent = message;
  el.style.color = isError ? '#b00020' : 'inherit';
};

// =============================================================================
// Email Verification
// =============================================================================

const fetchEmailVerificationStatus = async () => {
  const statusEl = $('setting-email-verified');
  const emailEl = $('setting-email-address');
  const resendBtn = $('resend-verification-btn');

  if (!state.accessToken) {
    if (statusEl) statusEl.textContent = 'Sign in to check';
    return;
  }

  try {
    const envelope = await requestEnvelope(
      `${apiBase}/me`,
      { headers: headers() },
      'Failed to load profile'
    );

    const { email, meta } = envelope.data;
    const isVerified = meta?.email_verified === true;

    if (emailEl) emailEl.textContent = email || '-';
    if (statusEl) {
      statusEl.textContent = isVerified ? 'Verified' : 'Not verified';
      statusEl.style.color = isVerified ? '#0a7' : '#b00020';
    }

    // Show/hide resend button
    if (resendBtn) resendBtn.classList.toggle('hidden', isVerified);
  } catch (err) {
    if (statusEl) statusEl.textContent = 'Unable to check';
  }
};

const resendVerificationEmail = async () => {
  const statusEl = $('email-verify-status');
  const resendBtn = $('resend-verification-btn');

  if (!state.accessToken) {
    if (statusEl) {
      statusEl.textContent = 'Sign in first';
      statusEl.style.color = '#b00020';
    }
    return;
  }

  try {
    if (resendBtn) resendBtn.disabled = true;
    if (statusEl) {
      statusEl.textContent = 'Sending...';
      statusEl.style.color = 'inherit';
    }

    await requestEnvelope(
      `${apiBase}/auth/request_email_verification`,
      {
        method: 'POST',
        headers: authHeaders(),
      },
      'Failed to send verification email'
    );

    if (statusEl) {
      statusEl.textContent = 'Verification email sent! Check your inbox.';
      statusEl.style.color = '#0a7';
    }
  } catch (err) {
    if (statusEl) {
      statusEl.textContent = err.message || 'Failed to send';
      statusEl.style.color = '#b00020';
    }
  } finally {
    if (resendBtn) resendBtn.disabled = false;
  }
};

// =============================================================================
// Password Change
// =============================================================================

const changePassword = async (event) => {
  event.preventDefault();

  const statusEl = $('password-change-status');
  const submitBtn = $('change-password-btn');
  const currentPwd = $('current-password');
  const newPwd = $('new-password');
  const confirmPwd = $('confirm-password');

  const setStatus = (msg, isError = false) => {
    if (statusEl) {
      statusEl.textContent = msg;
      statusEl.style.color = isError ? '#b00020' : '#0a7';
    }
  };

  if (!state.accessToken) {
    setStatus('Sign in to change password', true);
    return;
  }

  const currentPassword = currentPwd?.value?.trim();
  const newPassword = newPwd?.value;
  const confirmPassword = confirmPwd?.value;

  if (!currentPassword) {
    setStatus('Enter your current password', true);
    return;
  }

  if (!newPassword || newPassword.length < 8) {
    setStatus('New password must be at least 8 characters', true);
    return;
  }

  if (newPassword !== confirmPassword) {
    setStatus('New passwords do not match', true);
    return;
  }

  if (currentPassword === newPassword) {
    setStatus('New password must be different from current password', true);
    return;
  }

  try {
    if (submitBtn) submitBtn.disabled = true;
    setStatus('Changing password...');

    await requestEnvelope(
      `${apiBase}/auth/password/change`,
      {
        method: 'POST',
        headers: authHeaders(),
        body: JSON.stringify({
          current_password: currentPassword,
          new_password: newPassword,
        }),
      },
      'Failed to change password'
    );

    // Clear form
    if (currentPwd) currentPwd.value = '';
    if (newPwd) newPwd.value = '';
    if (confirmPwd) confirmPwd.value = '';

    setStatus('Password changed successfully!');

    // Clear success message after a few seconds
    setTimeout(() => {
      if (statusEl) statusEl.textContent = '';
    }, 5000);
  } catch (err) {
    setStatus(err.message || 'Failed to change password', true);
  } finally {
    if (submitBtn) submitBtn.disabled = false;
  }
};

// =============================================================================
// User Settings (Preferences)
// =============================================================================

const fetchUserSettings = async () => {
  if (!state.accessToken) return;

  try {
    const envelope = await requestEnvelope(
      `${apiBase}/settings`,
      { headers: headers() },
      'Failed to load settings'
    );

    const data = envelope.data || {};
    const localeSelect = $('setting-locale');
    const timezoneSelect = $('setting-timezone');
    const voiceSelect = $('setting-default-voice');

    if (localeSelect) localeSelect.value = data.locale || '';
    if (timezoneSelect) timezoneSelect.value = data.timezone || '';
    if (voiceSelect) voiceSelect.value = data.default_voice || '';
  } catch (err) {
    // Silently fail - user might not have settings yet
  }
};

const saveUserSettings = async (event) => {
  event.preventDefault();

  const statusEl = $('user-settings-status');
  const saveBtn = $('save-user-settings-btn');

  if (!state.accessToken) {
    if (statusEl) {
      statusEl.textContent = 'Sign in to save settings';
      statusEl.style.color = '#b00020';
    }
    return;
  }

  const locale = $('setting-locale')?.value || null;
  const timezone = $('setting-timezone')?.value || null;
  const defaultVoice = $('setting-default-voice')?.value || null;

  try {
    if (saveBtn) saveBtn.disabled = true;
    if (statusEl) {
      statusEl.textContent = 'Saving...';
      statusEl.style.color = 'inherit';
    }

    await requestEnvelope(
      `${apiBase}/settings`,
      {
        method: 'PATCH',
        headers: authHeaders(),
        body: JSON.stringify({
          locale: locale || null,
          timezone: timezone || null,
          default_voice: defaultVoice || null,
        }),
      },
      'Failed to save settings'
    );

    if (statusEl) {
      statusEl.textContent = 'Settings saved!';
      statusEl.style.color = '#0a7';
    }

    // Clear success message after a few seconds
    setTimeout(() => {
      if (statusEl) statusEl.textContent = '';
    }, 3000);
  } catch (err) {
    if (statusEl) {
      statusEl.textContent = err.message || 'Failed to save';
      statusEl.style.color = '#b00020';
    }
  } finally {
    if (saveBtn) saveBtn.disabled = false;
  }
};

// =============================================================================
// Admin Settings
// =============================================================================

const fetchAdminSettings = async () => {
  if (state.role !== 'admin') return;

  try {
    const envelope = await requestEnvelope(
      `${apiBase}/admin/settings`,
      { method: 'GET', headers: authHeaders() },
      'Failed to load admin settings'
    );

    const data = envelope.data;
    if (data) {
      const defaultPageSize = $('admin-default-page-size');
      const maxPageSize = $('admin-max-page-size');
      const defaultConversationsLimit = $('admin-default-conversations-limit');

      if (defaultPageSize) defaultPageSize.value = data.default_page_size || 100;
      if (maxPageSize) maxPageSize.value = data.max_page_size || 500;
      if (defaultConversationsLimit) defaultConversationsLimit.value = data.default_conversations_limit || 50;
    }
  } catch {
    // Ignore - admin settings are optional
  }
};

const saveAdminSettings = async () => {
  if (state.role !== 'admin') return;

  const statusEl = $('admin-settings-status');
  const defaultPageSize = parseInt($('admin-default-page-size')?.value, 10);
  const maxPageSize = parseInt($('admin-max-page-size')?.value, 10);
  const defaultConversationsLimit = parseInt($('admin-default-conversations-limit')?.value, 10);

  // Validate inputs
  if (isNaN(defaultPageSize) || defaultPageSize < 1 || defaultPageSize > 1000) {
    if (statusEl) statusEl.textContent = 'Invalid default page size (1-1000)';
    return;
  }
  if (isNaN(maxPageSize) || maxPageSize < 1 || maxPageSize > 1000) {
    if (statusEl) statusEl.textContent = 'Invalid max page size (1-1000)';
    return;
  }
  if (isNaN(defaultConversationsLimit) || defaultConversationsLimit < 1 || defaultConversationsLimit > 500) {
    if (statusEl) statusEl.textContent = 'Invalid conversations limit (1-500)';
    return;
  }

  try {
    if (statusEl) statusEl.textContent = 'Saving...';

    await requestEnvelope(
      `${apiBase}/admin/settings`,
      {
        method: 'PATCH',
        headers: { ...authHeaders(), 'Content-Type': 'application/json' },
        body: JSON.stringify({
          default_page_size: defaultPageSize,
          max_page_size: maxPageSize,
          default_conversations_limit: defaultConversationsLimit,
        }),
      },
      'Failed to save admin settings'
    );

    if (statusEl) statusEl.textContent = 'Settings saved successfully';
  } catch (err) {
    if (statusEl) statusEl.textContent = `Error: ${err.message}`;
  }
};

const renderAdminSettingsSection = () => {
  const section = $('admin-settings-section');
  if (section) {
    section.style.display = state.role === 'admin' ? 'block' : 'none';
  }
  if (state.role === 'admin') {
    fetchAdminSettings();
    fetchAdminUsers();
    fetchAdminAdapters();
    fetchAdminObjects();
    fetchConfigPatches();
  }
};

// =============================================================================
// Admin Users Management
// =============================================================================

const fetchAdminUsers = async () => {
  if (state.role !== 'admin') return;

  const tbody = $('users-table-body');
  if (tbody) tbody.innerHTML = '<tr><td colspan="5" class="empty">Loading users...</td></tr>';

  try {
    const envelope = await requestEnvelope(
      `${apiBase}/admin/users`,
      { headers: authHeaders() },
      'Failed to load users'
    );

    const users = envelope.data?.items || [];
    renderAdminUsers(users);
  } catch (err) {
    if (tbody) tbody.innerHTML = `<tr><td colspan="5" class="empty">Error: ${escapeHtml(err.message)}</td></tr>`;
  }
};

const renderAdminUsers = (users) => {
  const tbody = $('users-table-body');
  if (!tbody) return;

  if (!users.length) {
    tbody.innerHTML = '<tr><td colspan="5" class="empty">No users found</td></tr>';
    return;
  }

  tbody.innerHTML = users
    .map((user) => {
      const created = user.created_at ? new Date(user.created_at).toLocaleDateString() : '-';
      const isSelf = user.id === state.userId;
      return `
        <tr data-user-id="${escapeHtml(user.id)}">
          <td>${escapeHtml(user.email || '-')}</td>
          <td>
            <select class="user-role-select" ${isSelf ? 'disabled title="Cannot change own role"' : ''}>
              <option value="user" ${user.role === 'user' ? 'selected' : ''}>User</option>
              <option value="admin" ${user.role === 'admin' ? 'selected' : ''}>Admin</option>
            </select>
          </td>
          <td class="monospace">${escapeHtml(user.tenant_id || 'default')}</td>
          <td>${escapeHtml(created)}</td>
          <td>
            <button class="ghost delete-user-btn" ${isSelf ? 'disabled title="Cannot delete self"' : ''}>Delete</button>
          </td>
        </tr>
      `;
    })
    .join('');

  // Add event listeners for role changes
  tbody.querySelectorAll('.user-role-select').forEach((select) => {
    select.addEventListener('change', async (e) => {
      const row = e.target.closest('tr');
      const userId = row?.dataset.userId;
      const newRole = e.target.value;
      if (userId) await changeUserRole(userId, newRole);
    });
  });

  // Add event listeners for delete buttons
  tbody.querySelectorAll('.delete-user-btn').forEach((btn) => {
    btn.addEventListener('click', async (e) => {
      const row = e.target.closest('tr');
      const userId = row?.dataset.userId;
      if (userId && confirm('Are you sure you want to delete this user?')) {
        await deleteUser(userId);
      }
    });
  });
};

const createAdminUser = async () => {
  const statusEl = $('create-user-status');
  const email = $('new-user-email')?.value?.trim();
  const password = $('new-user-password')?.value;
  const role = $('new-user-role')?.value || 'user';

  if (!email || !password) {
    if (statusEl) statusEl.textContent = 'Email and password are required';
    return;
  }

  try {
    if (statusEl) statusEl.textContent = 'Creating user...';

    await requestEnvelope(
      `${apiBase}/admin/users`,
      {
        method: 'POST',
        headers: { ...authHeaders(), 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password, role }),
      },
      'Failed to create user'
    );

    if (statusEl) statusEl.textContent = 'User created successfully';
    $('new-user-email').value = '';
    $('new-user-password').value = '';
    $('new-user-role').value = 'user';
    $('add-user-form-section')?.classList.add('hidden');

    fetchAdminUsers();
  } catch (err) {
    if (statusEl) statusEl.textContent = `Error: ${err.message}`;
  }
};

const changeUserRole = async (userId, newRole) => {
  try {
    await requestEnvelope(
      `${apiBase}/admin/users/${userId}/role`,
      {
        method: 'POST',
        headers: { ...authHeaders(), 'Content-Type': 'application/json' },
        body: JSON.stringify({ role: newRole }),
      },
      'Failed to change user role'
    );

    fetchAdminUsers();
  } catch (err) {
    alert(`Failed to change role: ${err.message}`);
    fetchAdminUsers();
  }
};

const deleteUser = async (userId) => {
  try {
    await requestEnvelope(
      `${apiBase}/admin/users/${userId}`,
      { method: 'DELETE', headers: authHeaders() },
      'Failed to delete user'
    );

    fetchAdminUsers();
  } catch (err) {
    alert(`Failed to delete user: ${err.message}`);
  }
};

// =============================================================================
// Admin Adapters List
// =============================================================================

const fetchAdminAdapters = async () => {
  if (state.role !== 'admin') return;

  const tbody = $('adapters-table-body');
  if (tbody) tbody.innerHTML = '<tr><td colspan="5" class="empty">Loading adapters...</td></tr>';

  try {
    const envelope = await requestEnvelope(
      `${apiBase}/admin/adapters`,
      { headers: authHeaders() },
      'Failed to load adapters'
    );

    const adapters = envelope.data?.items || [];
    renderAdminAdapters(adapters);
  } catch (err) {
    if (tbody) tbody.innerHTML = `<tr><td colspan="5" class="empty">Error: ${escapeHtml(err.message)}</td></tr>`;
  }
};

const renderAdminAdapters = (adapters) => {
  const tbody = $('adapters-table-body');
  if (!tbody) return;

  if (!adapters.length) {
    tbody.innerHTML = '<tr><td colspan="5" class="empty">No adapters found</td></tr>';
    return;
  }

  tbody.innerHTML = adapters
    .map((adapter) => {
      const created = adapter.created_at ? new Date(adapter.created_at).toLocaleDateString() : '-';
      return `
        <tr>
          <td class="monospace">${escapeHtml(adapter.id?.slice(0, 8) || '-')}...</td>
          <td class="monospace">${escapeHtml(adapter.user_id?.slice(0, 8) || '-')}...</td>
          <td class="monospace">${escapeHtml(adapter.cluster_id?.slice(0, 8) || '-')}...</td>
          <td>${escapeHtml(adapter.base_model_id || '-')}</td>
          <td>${escapeHtml(created)}</td>
        </tr>
      `;
    })
    .join('');
};

// =============================================================================
// Admin Storage Objects
// =============================================================================

const fetchAdminObjects = async () => {
  if (state.role !== 'admin') return;

  const tbody = $('objects-table-body');
  if (tbody) tbody.innerHTML = '<tr><td colspan="5" class="empty">Loading objects...</td></tr>';

  try {
    const envelope = await requestEnvelope(
      `${apiBase}/admin/objects`,
      { headers: authHeaders() },
      'Failed to load objects'
    );

    const objects = envelope.data?.items || [];
    renderAdminObjects(objects);
  } catch (err) {
    if (tbody) tbody.innerHTML = `<tr><td colspan="5" class="empty">Error: ${escapeHtml(err.message)}</td></tr>`;
  }
};

const renderAdminObjects = (objects) => {
  const tbody = $('objects-table-body');
  if (!tbody) return;

  if (!objects.length) {
    tbody.innerHTML = '<tr><td colspan="5" class="empty">No storage objects found</td></tr>';
    return;
  }

  tbody.innerHTML = objects
    .map((obj) => {
      const created = obj.created_at ? new Date(obj.created_at).toLocaleDateString() : '-';
      const sizeDisplay = obj.size_bytes
        ? obj.size_bytes > 1024 * 1024
          ? `${(obj.size_bytes / (1024 * 1024)).toFixed(1)} MB`
          : obj.size_bytes > 1024
          ? `${(obj.size_bytes / 1024).toFixed(1)} KB`
          : `${obj.size_bytes} B`
        : '-';
      return `
        <tr>
          <td class="monospace">${escapeHtml(obj.key || obj.object_key || '-')}</td>
          <td>${escapeHtml(obj.bucket || obj.bucket_name || 'default')}</td>
          <td>${escapeHtml(sizeDisplay)}</td>
          <td>${escapeHtml(obj.content_type || obj.mime_type || '-')}</td>
          <td>${escapeHtml(created)}</td>
        </tr>
      `;
    })
    .join('');
};

// =============================================================================
// Admin Config Patches
// =============================================================================

let selectedPatch = null;

const fetchConfigPatches = async () => {
  if (state.role !== 'admin') return;

  const tbody = $('patches-table-body');
  if (tbody) tbody.innerHTML = '<tr><td colspan="6" class="empty">Loading patches...</td></tr>';

  try {
    const statusFilter = $('patches-status-filter')?.value || '';
    const url = statusFilter
      ? `${apiBase}/config/patches?status=${statusFilter}`
      : `${apiBase}/config/patches`;

    const envelope = await requestEnvelope(url, { headers: authHeaders() }, 'Failed to load patches');

    const patches = envelope.data?.items || [];
    renderConfigPatches(patches);
  } catch (err) {
    if (tbody) tbody.innerHTML = `<tr><td colspan="6" class="empty">Error: ${escapeHtml(err.message)}</td></tr>`;
  }
};

const renderConfigPatches = (patches) => {
  const tbody = $('patches-table-body');
  if (!tbody) return;

  if (!patches.length) {
    tbody.innerHTML = '<tr><td colspan="6" class="empty">No config patches found</td></tr>';
    $('patch-details-section')?.classList.add('hidden');
    return;
  }

  tbody.innerHTML = patches
    .map((patch) => {
      const created = patch.created_at ? new Date(patch.created_at).toLocaleDateString() : '-';
      const statusClass = patch.status === 'pending' ? 'pending' : patch.status === 'approved' ? 'approved' : patch.status === 'applied' ? 'applied' : 'rejected';
      return `
        <tr class="clickable" data-patch-id="${patch.id}">
          <td>${escapeHtml(String(patch.id))}</td>
          <td class="monospace">${escapeHtml(patch.artifact_id?.slice(0, 8) || '-')}...</td>
          <td>${escapeHtml(patch.proposer || '-')}</td>
          <td><span class="patch-status ${statusClass}">${escapeHtml(patch.status)}</span></td>
          <td>${escapeHtml(created)}</td>
          <td>
            <button class="ghost view-patch-btn" data-patch-id="${patch.id}">View</button>
          </td>
        </tr>
      `;
    })
    .join('');

  // Add click handlers
  tbody.querySelectorAll('.view-patch-btn').forEach((btn) => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      const patchId = btn.dataset.patchId;
      const patch = patches.find((p) => String(p.id) === patchId);
      if (patch) selectPatch(patch);
    });
  });

  tbody.querySelectorAll('tr.clickable').forEach((row) => {
    row.addEventListener('click', () => {
      const patchId = row.dataset.patchId;
      const patch = patches.find((p) => String(p.id) === patchId);
      if (patch) selectPatch(patch);
    });
  });
};

const selectPatch = (patch) => {
  selectedPatch = patch;
  const detailsSection = $('patch-details-section');
  const detailsContent = $('patch-details-content');
  const approveBtn = $('approve-patch-btn');
  const rejectBtn = $('reject-patch-btn');
  const applyBtn = $('apply-patch-btn');

  if (!detailsSection || !detailsContent) return;

  detailsSection.classList.remove('hidden');

  // Render patch details
  const patchJson = JSON.stringify(patch.patch || {}, null, 2);
  detailsContent.innerHTML = `
    <div class="detail-row"><span class="detail-label">Patch ID</span><span>${escapeHtml(String(patch.id))}</span></div>
    <div class="detail-row"><span class="detail-label">Artifact</span><span class="monospace">${escapeHtml(patch.artifact_id || '-')}</span></div>
    <div class="detail-row"><span class="detail-label">Proposer</span><span>${escapeHtml(patch.proposer || '-')}</span></div>
    <div class="detail-row"><span class="detail-label">Status</span><span class="patch-status ${patch.status}">${escapeHtml(patch.status)}</span></div>
    <div class="detail-row"><span class="detail-label">Justification</span><span>${escapeHtml(patch.justification || 'None provided')}</span></div>
    <div class="detail-row"><span class="detail-label">Created</span><span>${patch.created_at ? new Date(patch.created_at).toLocaleString() : '-'}</span></div>
    ${patch.decided_at ? `<div class="detail-row"><span class="detail-label">Decided</span><span>${new Date(patch.decided_at).toLocaleString()}</span></div>` : ''}
    ${patch.applied_at ? `<div class="detail-row"><span class="detail-label">Applied</span><span>${new Date(patch.applied_at).toLocaleString()}</span></div>` : ''}
    <div class="patch-code">
      <label>Patch Content</label>
      <pre class="code-block">${escapeHtml(patchJson)}</pre>
    </div>
  `;

  // Update button states based on status
  if (approveBtn) approveBtn.disabled = patch.status !== 'pending';
  if (rejectBtn) rejectBtn.disabled = patch.status !== 'pending';
  if (applyBtn) applyBtn.disabled = patch.status !== 'approved';

  // Clear any previous status
  const statusEl = $('patch-action-status');
  if (statusEl) statusEl.textContent = '';
};

const decidePatch = async (decision) => {
  if (!selectedPatch) return;

  const statusEl = $('patch-action-status');

  try {
    if (statusEl) statusEl.textContent = `${decision === 'approve' ? 'Approving' : 'Rejecting'} patch...`;

    await requestEnvelope(
      `${apiBase}/config/patches/${selectedPatch.id}/decide`,
      {
        method: 'POST',
        headers: { ...authHeaders(), 'Content-Type': 'application/json' },
        body: JSON.stringify({ decision }),
      },
      `Failed to ${decision} patch`
    );

    if (statusEl) statusEl.textContent = `Patch ${decision}d successfully`;
    fetchConfigPatches();
  } catch (err) {
    if (statusEl) statusEl.textContent = `Error: ${err.message}`;
  }
};

const applyPatch = async () => {
  if (!selectedPatch || selectedPatch.status !== 'approved') return;

  const statusEl = $('patch-action-status');

  try {
    if (statusEl) statusEl.textContent = 'Applying patch...';

    await requestEnvelope(
      `${apiBase}/config/patches/${selectedPatch.id}/apply`,
      {
        method: 'POST',
        headers: authHeaders(),
      },
      'Failed to apply patch'
    );

    if (statusEl) statusEl.textContent = 'Patch applied successfully';
    fetchConfigPatches();
  } catch (err) {
    if (statusEl) statusEl.textContent = `Error: ${err.message}`;
  }
};

// =============================================================================
// Tools
// =============================================================================

let selectedTool = null;
let tools = [];
let workflows = [];

const fetchTools = async () => {
  if (!state.accessToken) return;

  const toolsList = $('tools-list');
  if (toolsList) toolsList.innerHTML = '<div class="empty">Loading tools...</div>';

  try {
    const envelope = await requestEnvelope(
      `${apiBase}/tools/specs`,
      { headers: headers() },
      'Failed to load tools'
    );

    tools = envelope.data?.items || [];
    renderToolsList();
  } catch (err) {
    if (toolsList) toolsList.innerHTML = `<div class="empty">Error: ${escapeHtml(err.message)}</div>`;
  }
};

const renderToolsList = () => {
  const toolsList = $('tools-list');
  if (!toolsList) return;

  if (!tools.length) {
    toolsList.innerHTML = '<div class="empty">No tools available</div>';
    return;
  }

  toolsList.innerHTML = tools
    .map((tool) => {
      const isSelected = selectedTool?.id === tool.id;
      const name = tool.name || tool.schema?.name || tool.id;
      const description = tool.description || tool.schema?.description || 'No description';
      return `
        <div class="tool-card ${isSelected ? 'selected' : ''}" data-id="${escapeHtml(tool.id)}">
          <div class="tool-name">${escapeHtml(name)}</div>
          <div class="tool-description">${escapeHtml(description)}</div>
        </div>
      `;
    })
    .join('');

  toolsList.querySelectorAll('.tool-card').forEach((card) => {
    card.addEventListener('click', () => selectTool(card.dataset.id));
  });
};

const selectTool = async (toolId) => {
  const tool = tools.find((t) => t.id === toolId);
  if (!tool) return;

  selectedTool = tool;
  renderToolsList();

  const details = $('tool-details');
  const invokeSection = $('tool-invoke-section');
  const invokePlaceholder = $('tool-invoke-placeholder');

  if (details) {
    const schema = tool.schema || {};
    const inputs = schema.inputs || {};

    details.innerHTML = `
      <div class="detail-header">
        <h4>${escapeHtml(tool.name || schema.name || tool.id)}</h4>
      </div>
      <div class="detail-row">
        <span class="detail-label">ID</span>
        <span class="monospace">${escapeHtml(tool.id)}</span>
      </div>
      <div class="detail-row">
        <span class="detail-label">Handler</span>
        <span class="monospace">${escapeHtml(schema.handler || '-')}</span>
      </div>
      <div class="detail-row">
        <span class="detail-label">Description</span>
        <span>${escapeHtml(tool.description || schema.description || '-')}</span>
      </div>
      <div class="divider"></div>
      <h4>Inputs</h4>
      <pre class="schema-viewer">${escapeHtml(JSON.stringify(inputs, null, 2))}</pre>
    `;
  }

  // Show invoke section
  if (invokeSection) invokeSection.classList.remove('hidden');
  if (invokePlaceholder) invokePlaceholder.style.display = 'none';

  // Pre-populate input template
  const invokeInput = $('tool-invoke-input');
  if (invokeInput) {
    const schema = tool.schema || {};
    const inputs = schema.inputs || {};
    const template = {};
    Object.keys(inputs).forEach((key) => {
      template[key] = inputs[key].type === 'string' ? '' : null;
    });
    invokeInput.value = JSON.stringify(template, null, 2);
  }
};

const invokeTool = async (event) => {
  event.preventDefault();

  if (!selectedTool) return;

  const statusEl = $('tool-invoke-status');
  const resultEl = $('tool-invoke-result');
  const inputEl = $('tool-invoke-input');
  const invokeBtn = $('tool-invoke-btn');

  let inputData;
  try {
    inputData = JSON.parse(inputEl?.value || '{}');
  } catch {
    if (statusEl) statusEl.textContent = 'Invalid JSON input';
    return;
  }

  try {
    toggleButtonBusy(invokeBtn, true, 'Invoking...');
    if (statusEl) statusEl.textContent = '';
    if (resultEl) resultEl.style.display = 'none';

    const envelope = await requestEnvelope(
      `${apiBase}/tools/${selectedTool.id}/invoke`,
      {
        method: 'POST',
        headers: headers(),
        body: JSON.stringify({ inputs: inputData }),
      },
      'Tool invocation failed'
    );

    if (statusEl) statusEl.textContent = 'Tool invoked successfully';
    if (resultEl) {
      resultEl.textContent = JSON.stringify(envelope.data, null, 2);
      resultEl.style.display = 'block';
    }
  } catch (err) {
    if (statusEl) statusEl.textContent = err.message;
  } finally {
    toggleButtonBusy(invokeBtn, false);
  }
};

const fetchWorkflows = async () => {
  if (!state.accessToken) return;

  const workflowsList = $('workflows-list');
  if (workflowsList) workflowsList.innerHTML = '<div class="empty">Loading workflows...</div>';

  try {
    const envelope = await requestEnvelope(
      `${apiBase}/workflows`,
      { headers: headers() },
      'Failed to load workflows'
    );

    workflows = envelope.data?.items || [];
    renderWorkflowsList();
  } catch (err) {
    if (workflowsList) workflowsList.innerHTML = `<div class="empty">Error: ${escapeHtml(err.message)}</div>`;
  }
};

let selectedWorkflow = null;

const renderWorkflowsList = () => {
  const workflowsList = $('workflows-list');
  if (!workflowsList) return;

  if (!workflows.length) {
    workflowsList.innerHTML = '<div class="empty">No workflows configured</div>';
    return;
  }

  workflowsList.innerHTML = workflows
    .map((wf) => {
      const isSelected = selectedWorkflow?.id === wf.id;
      return `
        <div class="workflow-card ${isSelected ? 'selected' : ''}" data-id="${escapeHtml(wf.id)}">
          <div class="workflow-name">${escapeHtml(wf.name || wf.id)}</div>
          <div class="workflow-meta">
            <span class="visibility-badge ${wf.visibility || 'private'}">${wf.visibility || 'private'}</span>
            <span>v${wf.version || 1}</span>
          </div>
        </div>
      `;
    })
    .join('');

  workflowsList.querySelectorAll('.workflow-card').forEach((card) => {
    card.addEventListener('click', () => selectWorkflow(card.dataset.id));
  });
};

const selectWorkflow = (workflowId) => {
  const wf = workflows.find((w) => w.id === workflowId);
  if (!wf) return;

  selectedWorkflow = wf;
  renderWorkflowsList();

  const details = $('workflow-details');
  if (details) {
    const schema = wf.schema || {};
    details.innerHTML = `
      <div class="detail-header">
        <h4>${escapeHtml(wf.name || wf.id)}</h4>
        <span class="visibility-badge ${wf.visibility || 'private'}">${wf.visibility || 'private'}</span>
      </div>
      <div class="detail-row">
        <span class="detail-label">ID</span>
        <span class="monospace">${escapeHtml(wf.id)}</span>
      </div>
      <div class="detail-row">
        <span class="detail-label">Entrypoint</span>
        <span class="monospace">${escapeHtml(schema.entrypoint || '-')}</span>
      </div>
      <div class="detail-row">
        <span class="detail-label">Nodes</span>
        <span>${(schema.nodes || []).length} nodes</span>
      </div>
      <div class="divider"></div>
      <h4>Schema</h4>
      <pre class="schema-viewer">${escapeHtml(JSON.stringify(schema, null, 2))}</pre>
    `;
  }
};

const refreshToolsAndWorkflows = async () => {
  await Promise.all([fetchTools(), fetchWorkflows()]);
};

// =============================================================================
// Preference Insights
// =============================================================================

const fetchInsights = async () => {
  if (!state.accessToken) return;

  try {
    const envelope = await requestEnvelope(
      `${apiBase}/preferences/insights`,
      { headers: headers() },
      'Failed to load insights'
    );

    renderInsights(envelope.data || {});
  } catch (err) {
    console.warn('Failed to fetch insights:', err.message);
    // Show error state
    const totalEl = $('insights-total-events');
    if (totalEl) totalEl.textContent = 'Error';
  }
};

const renderInsights = (data) => {
  // Summary stats
  const totalEl = $('insights-total-events');
  const positiveEl = $('insights-positive-count');
  const negativeEl = $('insights-negative-count');
  const neutralEl = $('insights-neutral-count');

  if (totalEl) totalEl.textContent = data.total_events ?? '-';
  if (positiveEl) positiveEl.textContent = data.positive_count ?? '-';
  if (negativeEl) negativeEl.textContent = data.negative_count ?? '-';
  if (neutralEl) neutralEl.textContent = data.neutral_count ?? '-';

  // Top adapters
  const adaptersEl = $('insights-top-adapters');
  if (adaptersEl) {
    const adapters = data.top_adapters || [];
    if (!adapters.length) {
      adaptersEl.innerHTML = '<div class="empty">No adapter data yet</div>';
    } else {
      adaptersEl.innerHTML = adapters
        .map((a) => `
          <div class="adapter-item">
            <span class="adapter-name">${escapeHtml(a.adapter_id || a.name || 'Unknown')}</span>
            <span class="adapter-score">+${a.positive_count || 0} / -${a.negative_count || 0}</span>
          </div>
        `)
        .join('');
    }
  }

  // Recent preferences
  const recentEl = $('insights-recent-list');
  if (recentEl) {
    const recent = data.recent_events || [];
    if (!recent.length) {
      recentEl.innerHTML = '<div class="empty">No preference events yet</div>';
    } else {
      recentEl.innerHTML = recent
        .map((e) => {
          const feedback = e.feedback || 'neutral';
          const date = e.created_at ? new Date(e.created_at).toLocaleDateString() : '-';
          const icon = feedback === 'positive' ? '+1' : feedback === 'negative' ? '-1' : '·';
          return `
            <div class="preference-item ${feedback}">
              <span class="feedback-icon">${icon}</span>
              <span class="preference-message">${escapeHtml((e.context_text || '').slice(0, 80))}${e.context_text?.length > 80 ? '...' : ''}</span>
              <span class="preference-date">${date}</span>
            </div>
          `;
        })
        .join('');
    }
  }

  // Clusters
  const clustersEl = $('insights-clusters');
  if (clustersEl) {
    const clusters = data.clusters || [];
    if (!clusters.length) {
      clustersEl.innerHTML = '<div class="empty">No clusters identified yet</div>';
    } else {
      clustersEl.innerHTML = clusters
        .map((c) => `
          <div class="cluster-card">
            <div class="cluster-label">${escapeHtml(c.label || 'Unlabeled')}</div>
            <div class="cluster-description">${escapeHtml(c.description || '-')}</div>
            <div class="cluster-meta">
              <span>${c.size || 0} events</span>
              ${c.adapter_id ? `<span class="has-adapter">Has adapter</span>` : ''}
            </div>
          </div>
        `)
        .join('');
    }
  }
};

// =============================================================================
// Auto-save drafts
// =============================================================================

let draftSaveTimeout = null;

const handleMessageInputChange = () => {
  clearTimeout(draftSaveTimeout);
  draftSaveTimeout = setTimeout(() => {
    const messageInput = $('message-input');
    const text = messageInput?.value || '';
    saveDraft(state.conversationId, text);
  }, 1000);
};

// =============================================================================
// Event listeners setup
// =============================================================================

const initEventListeners = () => {
  // Auth
  if (authForm) authForm.addEventListener('submit', handleLogin);
  $('logout')?.addEventListener('click', logout);

  // Auth form switching
  $('show-signup')?.addEventListener('click', () => showAuthForm('signup'));
  $('show-reset')?.addEventListener('click', () => showAuthForm('reset'));
  $('show-login-from-signup')?.addEventListener('click', () => showAuthForm('login'));
  $('show-login-from-reset')?.addEventListener('click', () => showAuthForm('login'));

  // Signup
  $('signup-form')?.addEventListener('submit', handleSignup);

  // Password reset
  $('reset-request-form')?.addEventListener('submit', handleResetRequest);

  // OAuth
  $('oauth-google')?.addEventListener('click', () => startOAuth('google'));
  $('oauth-github')?.addEventListener('click', () => startOAuth('github'));
  $('oauth-microsoft')?.addEventListener('click', () => startOAuth('microsoft'));
  $('reset-confirm-form')?.addEventListener('submit', (event) => {
    // Use token-based handler if we have a pending token from URL
    if (pendingResetToken) {
      handleResetWithToken(event);
    } else {
      handleResetConfirm(event);
    }
  });

  // Chat
  if (chatForm) chatForm.addEventListener('submit', sendMessage);
  $('message-input')?.addEventListener('input', handleMessageInputChange);
  $('new-thread')?.addEventListener('click', newConversation);
  $('new-thread-secondary')?.addEventListener('click', newConversation);
  $('stop-stream-btn')?.addEventListener('click', cancelStreaming);
  $('new-conversation-btn')?.addEventListener('click', newConversation);
  $('refresh-conversations')?.addEventListener('click', fetchConversations);

  // Conversation search
  conversationSearchEl?.addEventListener('input', renderConversationList);

  // Preferences
  $('thumbs-up')?.addEventListener('click', () => sendPreference(true));
  $('thumbs-down')?.addEventListener('click', () => sendPreference(false));

  // Contexts
  $('create-context-btn')?.addEventListener('click', createContext);
  $('refresh-contexts')?.addEventListener('click', fetchContexts);
  $('add-source-form')?.addEventListener('submit', addContextSource);

  // Artifacts
  $('refresh-artifacts')?.addEventListener('click', fetchArtifacts);
  $('artifact-type-filter')?.addEventListener('change', fetchArtifacts);
  $('artifact-visibility-filter')?.addEventListener('change', fetchArtifacts);

  // Tools
  $('refresh-tools')?.addEventListener('click', refreshToolsAndWorkflows);
  $('tool-invoke-form')?.addEventListener('submit', invokeTool);

  // Insights
  $('refresh-insights')?.addEventListener('click', fetchInsights);

  // File upload
  if (fileUploadInput) fileUploadInput.addEventListener('change', renderUploadHint);
  if (fileUploadButton) fileUploadButton.addEventListener('click', handleFileUpload);

  // Settings
  $('clear-drafts-btn')?.addEventListener('click', handleClearDrafts);
  $('export-drafts-btn')?.addEventListener('click', handleExportDrafts);

  // MFA settings
  $('mfa-enable-btn')?.addEventListener('click', startMfaSetup);
  $('mfa-verify-form')?.addEventListener('submit', verifyMfaSetup);
  $('mfa-cancel-btn')?.addEventListener('click', cancelMfaSetup);
  $('mfa-show-disable-btn')?.addEventListener('click', showMfaDisable);
  $('mfa-disable-form')?.addEventListener('submit', disableMfa);
  $('mfa-disable-cancel-btn')?.addEventListener('click', hideMfaDisable);

  // Email verification
  $('resend-verification-btn')?.addEventListener('click', resendVerificationEmail);

  // Password change
  $('password-change-form')?.addEventListener('submit', changePassword);

  // User settings (preferences)
  $('user-settings-form')?.addEventListener('submit', saveUserSettings);

  // Admin settings
  $('save-admin-settings-btn')?.addEventListener('click', saveAdminSettings);
  $('reload-admin-settings-btn')?.addEventListener('click', fetchAdminSettings);

  // Admin users management
  $('refresh-users-btn')?.addEventListener('click', fetchAdminUsers);
  $('show-add-user-btn')?.addEventListener('click', () => {
    $('add-user-form-section')?.classList.remove('hidden');
  });
  $('cancel-add-user-btn')?.addEventListener('click', () => {
    $('add-user-form-section')?.classList.add('hidden');
    $('create-user-status').textContent = '';
  });
  $('create-user-btn')?.addEventListener('click', createAdminUser);

  // Admin adapters and objects
  $('refresh-adapters-btn')?.addEventListener('click', fetchAdminAdapters);
  $('refresh-objects-btn')?.addEventListener('click', fetchAdminObjects);

  // Admin config patches
  $('refresh-patches-btn')?.addEventListener('click', fetchConfigPatches);
  $('patches-status-filter')?.addEventListener('change', fetchConfigPatches);
  $('approve-patch-btn')?.addEventListener('click', () => decidePatch('approve'));
  $('reject-patch-btn')?.addEventListener('click', () => decidePatch('reject'));
  $('apply-patch-btn')?.addEventListener('click', applyPatch);

  // Voice input/output buttons - must be initialized after DOM ready
  voiceInputBtn = $('voice-input-btn');
  voiceOutputBtn = $('voice-output-btn');

  if (voiceInputBtn) {
    voiceInputBtn.addEventListener('mousedown', startVoiceRecording);
    voiceInputBtn.addEventListener('mouseup', stopVoiceRecording);
    voiceInputBtn.addEventListener('mouseleave', stopVoiceRecording);
    voiceInputBtn.addEventListener('touchstart', (e) => {
      e.preventDefault();
      startVoiceRecording();
    });
    voiceInputBtn.addEventListener('touchend', (e) => {
      e.preventDefault();
      stopVoiceRecording();
    });
  }

  if (voiceOutputBtn) {
    voiceOutputBtn.addEventListener('click', readLastResponse);
  }

  // Close citation modal on escape key
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      const modal = document.getElementById('citation-modal');
      if (modal) modal.classList.remove('active');
    }
  });
};

// =============================================================================
// Initialization
// =============================================================================

const init = async () => {
  initTabs();
  initCollapsibleSections();
  initEventListeners();
  updateAuthUI();
  updateDraftIndicator();
  renderPreferencePanel();
  renderUploadHint();

  // Handle OAuth callback if present
  const isOAuthCallback = await handleOAuthCallback();
  if (isOAuthCallback) {
    // OAuth callback was handled, UI already updated
    updateEmptyState();
    return;
  }

  // Handle password reset token from email link
  const isResetCallback = await handleResetTokenCallback();
  if (isResetCallback) {
    updateEmptyState();
    return;
  }

  // Handle email verification token from email link
  const isVerifyCallback = await handleVerifyTokenCallback();
  if (isVerifyCallback) {
    updateEmptyState();
    return;
  }

  // Load draft for current conversation
  const messageInput = $('message-input');
  if (messageInput) {
    messageInput.value = getDraft(state.conversationId);
  }

  // If already authenticated, load data
  if (state.accessToken) {
    persistAuth({
      access_token: state.accessToken,
      refresh_token: state.refreshToken,
      session_id: state.sessionId,
      role: state.role,
      tenant_id: state.tenantId,
      user_id: state.userId,
    });
    await Promise.all([
      fetchConversations(),
      fetchContexts(),
      fetchArtifacts(),
      fetchTools(),
      fetchWorkflows(),
      fetchInsights(),
      fetchHealth(),
      fetchMfaStatus(),
      fetchEmailVerificationStatus(),
      fetchUserSettings(),
    ]);
  }

  updateEmptyState();
};

// Run on DOM ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
