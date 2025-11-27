const apiBase = '/v1';
const messagesEl = document.getElementById('messages');
const messagesEmptyEl = document.getElementById('messages-empty');
const authForm = document.getElementById('auth-form');
const chatForm = document.getElementById('chat-form');
const statusEl = document.getElementById('status');
const errorEl = document.getElementById('error-banner');
const sessionIndicator = document.getElementById('session-indicator');
const adminWarning = document.getElementById('admin-warning');
const conversationLabel = document.getElementById('conversation-label');
const adminLink = document.getElementById('admin-link');
const authSubmit = document.getElementById('auth-submit');
const sendBtn = document.getElementById('send-btn');

const sessionStorageKey = (key) => `liminal.${key}`;

const readSession = (key) => sessionStorage.getItem(sessionStorageKey(key));
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
  userId: readSession('userId'),
  conversationId: null,
};

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

const headers = (idempotencyKey) => {
  const h = { 'Content-Type': 'application/json' };
  if (state.accessToken) h['Authorization'] = `Bearer ${state.accessToken}`;
  if (state.tenantId) h['X-Tenant-ID'] = state.tenantId;
  if (state.sessionId) h['session_id'] = state.sessionId;
  h['Idempotency-Key'] = idempotencyKey || randomIdempotencyKey();
  return h;
};

const showStatus = (message, isError = false) => {
  const target = isError ? errorEl : statusEl;
  target.textContent = message;
  target.style.display = message ? 'block' : 'none';
  if (isError) {
    statusEl.style.display = 'none';
  } else {
    errorEl.style.display = 'none';
  }
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

const renderAdminNotice = () => {
  if (state.role === 'admin') {
    adminWarning.textContent = 'You are signed in as an admin. Use the Admin link to approve router or workflow patches.';
    adminWarning.style.display = 'block';
    adminLink.style.display = 'inline-flex';
  } else {
    adminWarning.textContent = '';
    adminWarning.style.display = 'none';
    adminLink.style.display = 'none';
  }
};

const persistAuth = (payload) => {
  state.accessToken = payload.access_token;
  state.refreshToken = payload.refresh_token;
  state.sessionId = payload.session_id;
  state.role = payload.role;
  state.tenantId = payload.tenant_id;
  state.userId = payload.user_id;
  writeSession('accessToken', state.accessToken);
  writeSession('refreshToken', state.refreshToken);
  writeSession('sessionId', state.sessionId);
  writeSession('role', state.role);
  writeSession('tenantId', state.tenantId);
  writeSession('userId', state.userId);
  sessionIndicator.textContent = state.accessToken
    ? `Signed in as ${state.userId || payload.user_id || 'current'} (${state.role || 'user'})`
    : 'Not signed in';
  renderAdminNotice();
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
  messagesEl.appendChild(wrapper);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  updateEmptyState();
};

const setConversation = (id) => {
  state.conversationId = id;
  conversationLabel.textContent = id ? `Conversation ${id}` : 'New conversation';
};

const extractError = (payload, fallback) => {
  const detail = payload?.detail || payload?.error || payload;
  if (typeof detail === 'string') return detail.trim() || fallback;
  if (detail?.message) return detail.message;
  if (detail?.error?.message) return detail.error.message;
  return fallback;
};

const requestEnvelope = async (url, options, fallbackMessage) => {
  const resp = await fetch(url, options);
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

const handleLogin = async (event) => {
  event.preventDefault();
  const body = {
    email: document.getElementById('email').value,
    password: document.getElementById('password').value,
    mfa_code: document.getElementById('mfa')?.value || undefined,
    tenant_id: document.getElementById('tenant').value || undefined,
  };
  try {
    toggleButtonBusy(authSubmit, true, 'Signing in...');
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
      showStatus('MFA required. Enter the code from your authenticator.', true);
      return;
    }
    persistAuth(envelope.data);
    showStatus('Signed in');
  } catch (err) {
    showStatus(err.message, true);
  } finally {
    toggleButtonBusy(authSubmit, false);
  }
};

const sendMessage = async (event) => {
  event.preventDefault();
  const content = document.getElementById('message-input').value.trim();
  if (!content) return;
  if (!state.accessToken) {
    showStatus('Sign in to chat.', true);
    return;
  }
  toggleButtonBusy(sendBtn, true, 'Sending...');
  document.getElementById('message-input').value = '';
  appendMessage('user', content);
  showStatus('Thinking...');
  const payload = {
    conversation_id: state.conversationId || undefined,
    message: { content, mode: 'text' },
    context_id: document.getElementById('context-id').value || undefined,
    workflow_id: document.getElementById('workflow-id').value || undefined,
    stream: false,
  };
  const idempotencyKey = `chat-${stableHash(JSON.stringify(payload))}`;

  try {
    const envelope = await requestEnvelope(
      `${apiBase}/chat`,
      {
        method: 'POST',
        headers: headers(idempotencyKey),
        body: JSON.stringify(payload),
      },
      'Chat failed'
    );
    const data = envelope.data;
    setConversation(data.conversation_id);
    const metaBits = [];
    if (data.adapters?.length) metaBits.push(`adapters: ${data.adapters.join(', ')}`);
    if (data.context_snippets?.length) metaBits.push(`context: ${data.context_snippets.length} snippets`);
    if (data.usage?.total_tokens) metaBits.push(`usage: ${data.usage.total_tokens} tokens`);
    appendMessage('assistant', data.content, metaBits.join(' · '));
    showStatus('');
  } catch (err) {
    showStatus(err.message, true);
  } finally {
    toggleButtonBusy(sendBtn, false, 'Send');
  }
};

const newConversation = () => {
  setConversation(null);
  messagesEl.innerHTML = '';
  showStatus('New thread ready');
  updateEmptyState();
};

const listConversations = async () => {
  if (!state.accessToken) return;
  try {
    const envelope = await requestEnvelope(
      `${apiBase}/conversations`,
      { headers: headers() },
      'Unable to load history'
    );
    const items = envelope.data.items || [];
    if (!items.length) {
      showStatus('No previous conversations for this user');
      return;
    }
    const latest = items.sort((a, b) => new Date(b.updated_at) - new Date(a.updated_at))[0];
    setConversation(latest.id);
    showStatus(`Resumed ${items.length} conversation(s); latest ${latest.id}`);
  } catch (err) {
    showStatus(err.message, true);
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
  state.accessToken = null;
  state.refreshToken = null;
  state.sessionId = null;
  state.role = null;
  state.tenantId = null;
  state.userId = null;
  ['accessToken', 'refreshToken', 'sessionId', 'role', 'tenantId', 'userId'].forEach((k) =>
    sessionStorage.removeItem(sessionStorageKey(k))
  );
  setConversation(null);
  messagesEl.innerHTML = '';
  sessionIndicator.textContent = 'Not signed in';
  renderAdminNotice();
  updateEmptyState();
};

// Wire up events
if (authForm) authForm.addEventListener('submit', handleLogin);
if (chatForm) chatForm.addEventListener('submit', sendMessage);
const newThreadBtn = document.getElementById('new-thread');
if (newThreadBtn) newThreadBtn.addEventListener('click', newConversation);
const newThreadSecondaryBtn = document.getElementById('new-thread-secondary');
if (newThreadSecondaryBtn) newThreadSecondaryBtn.addEventListener('click', newConversation);
const refreshBtn = document.getElementById('refresh-conversations');
if (refreshBtn) refreshBtn.addEventListener('click', listConversations);
const logoutBtn = document.getElementById('logout');
if (logoutBtn) logoutBtn.addEventListener('click', logout);

// Bootstrap from stored credentials
if (state.accessToken) {
  persistAuth({
    access_token: state.accessToken,
    refresh_token: state.refreshToken,
    session_id: state.sessionId,
    role: state.role,
    tenant_id: state.tenantId,
    user_id: state.userId,
  });
  listConversations();
} else {
  renderAdminNotice();
}
updateEmptyState();
