const apiBase = '/v1';
const messagesEl = document.getElementById('messages');
const authForm = document.getElementById('auth-form');
const chatForm = document.getElementById('chat-form');
const statusEl = document.getElementById('status');
const errorEl = document.getElementById('error-banner');
const sessionIndicator = document.getElementById('session-indicator');
const adminWarning = document.getElementById('admin-warning');
const conversationLabel = document.getElementById('conversation-label');
const adminLink = document.getElementById('admin-link');

const state = {
  accessToken: localStorage.getItem('liminal.accessToken'),
  refreshToken: localStorage.getItem('liminal.refreshToken'),
  sessionId: localStorage.getItem('liminal.sessionId'),
  tenantId: localStorage.getItem('liminal.tenantId'),
  role: localStorage.getItem('liminal.role'),
  conversationId: null,
};

const headers = () => {
  const h = { 'Content-Type': 'application/json' };
  if (state.accessToken) h['Authorization'] = `Bearer ${state.accessToken}`;
  if (state.tenantId) h['X-Tenant-ID'] = state.tenantId;
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
  localStorage.setItem('liminal.accessToken', state.accessToken || '');
  localStorage.setItem('liminal.refreshToken', state.refreshToken || '');
  localStorage.setItem('liminal.sessionId', state.sessionId || '');
  localStorage.setItem('liminal.role', state.role || '');
  localStorage.setItem('liminal.tenantId', state.tenantId || '');
  sessionIndicator.textContent = state.accessToken
    ? `Signed in as ${payload.user_id} (${state.role || 'user'})`
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
};

const setConversation = (id) => {
  state.conversationId = id;
  conversationLabel.textContent = id ? `Conversation ${id}` : 'New conversation';
};

const handleLogin = async (event) => {
  event.preventDefault();
  const body = {
    email: document.getElementById('email').value,
    password: document.getElementById('password').value,
    tenant_id: document.getElementById('tenant').value || undefined,
  };
  try {
    const resp = await fetch(`${apiBase}/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!resp.ok) throw new Error((await resp.json()).detail || 'Login failed');
    const envelope = await resp.json();
    persistAuth(envelope.data);
    showStatus('Signed in');
  } catch (err) {
    showStatus(err.message, true);
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
  document.getElementById('message-input').value = '';
  appendMessage('user', content);
  showStatus('Thinking...');
  try {
    const resp = await fetch(`${apiBase}/chat`, {
      method: 'POST',
      headers: headers(),
      body: JSON.stringify({
        conversation_id: state.conversationId || undefined,
        message: { content, mode: 'text' },
        context_id: document.getElementById('context-id').value || undefined,
        workflow_id: document.getElementById('workflow-id').value || undefined,
        stream: false,
      }),
    });
    if (!resp.ok) throw new Error((await resp.json()).detail || 'Chat failed');
    const envelope = await resp.json();
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
  }
};

const newConversation = () => {
  setConversation(null);
  messagesEl.innerHTML = '';
  showStatus('New thread ready');
};

const listConversations = async () => {
  if (!state.accessToken) return;
  try {
    const resp = await fetch(`${apiBase}/conversations`, { headers: headers() });
    if (!resp.ok) throw new Error((await resp.json()).detail || 'Unable to load history');
    const envelope = await resp.json();
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

const logout = () => {
  state.accessToken = null;
  state.refreshToken = null;
  state.sessionId = null;
  state.role = null;
  state.tenantId = null;
  ['liminal.accessToken', 'liminal.refreshToken', 'liminal.sessionId', 'liminal.role', 'liminal.tenantId'].forEach((k) =>
    localStorage.removeItem(k)
  );
  setConversation(null);
  messagesEl.innerHTML = '';
  sessionIndicator.textContent = 'Not signed in';
  renderAdminNotice();
};

// Wire up events
if (authForm) authForm.addEventListener('submit', handleLogin);
if (chatForm) chatForm.addEventListener('submit', sendMessage);
const newThreadBtn = document.getElementById('new-thread');
if (newThreadBtn) newThreadBtn.addEventListener('click', newConversation);
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
    user_id: 'current',
  });
  listConversations();
} else {
  renderAdminNotice();
}
