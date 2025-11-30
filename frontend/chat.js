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

// Close citation modal on escape key
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    const modal = document.getElementById('citation-modal');
    if (modal) modal.classList.remove('active');
  }
});

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
          const snippetData = escapeHtml(JSON.stringify({
            source_path: c.source_path || '',
            chunk_id: c.chunk_id || '',
            content: c.content || c.snippet || '',
            context_id: c.context_id || '',
            chunk_index: c.chunk_index,
          }));
          return `<span class="citation-link" title="${path}" data-citation='${snippetData}' onclick="showCitationModal(this)">${escapeHtml(label)}</span>`;
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
   * Handles events: token, trace, message_done, error, cancel_ack
   */
  const chatViaWebSocketStreaming = async () => {
    const ws = await ensureWebSocket();
    return new Promise((resolve, reject) => {
      let settled = false;
      let streamingMsg = null;
      let finalData = null;

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
                // Streaming complete
                settled = true;
                cleanup();
                if (streamingMsg) {
                  const adapters = (msg.data?.adapters || []).map(a => a?.name || a?.id || a).filter(Boolean);
                  streamingMsg.finalize(adapters.length ? `Adapters: ${adapters.join(', ')}` : '');
                }
                resolve(msg.data || {});
                break;

              case 'streaming_complete':
                // Final event with message_id and conversation_id after DB save
                if (!settled) {
                  settled = true;
                  cleanup();
                  finalData = msg.data || {};
                  resolve(finalData);
                }
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
          if (streamingMsg) streamingMsg.finalize('Connection closed');
          reject(new Error('Connection closed'));
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

const voiceInputBtn = $('voice-input-btn');
const voiceOutputBtn = $('voice-output-btn');

const startVoiceRecording = async () => {
  if (isRecording) return;

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
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
    if (envelope.status === 'ok' && envelope.data?.text) {
      // Insert transcribed text into the message input
      const messageInput = $('message-input');
      if (messageInput) {
        messageInput.value = (messageInput.value + ' ' + envelope.data.text).trim();
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
  if (!state.lastAssistant?.content) {
    alert('No assistant response to read.');
    return;
  }
  speakText(state.lastAssistant.content);
};

// Voice button event listeners
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

  // Chat
  if (chatForm) chatForm.addEventListener('submit', sendMessage);
  $('message-input')?.addEventListener('input', handleMessageInputChange);
  $('new-thread')?.addEventListener('click', newConversation);
  $('new-thread-secondary')?.addEventListener('click', newConversation);
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

  // Artifacts
  $('refresh-artifacts')?.addEventListener('click', fetchArtifacts);
  $('artifact-type-filter')?.addEventListener('change', fetchArtifacts);
  $('artifact-visibility-filter')?.addEventListener('change', fetchArtifacts);

  // File upload
  if (fileUploadInput) fileUploadInput.addEventListener('change', renderUploadHint);
  if (fileUploadButton) fileUploadButton.addEventListener('click', handleFileUpload);

  // Settings
  $('clear-drafts-btn')?.addEventListener('click', handleClearDrafts);
  $('export-drafts-btn')?.addEventListener('click', handleExportDrafts);

  // Admin settings
  $('save-admin-settings-btn')?.addEventListener('click', saveAdminSettings);
  $('reload-admin-settings-btn')?.addEventListener('click', fetchAdminSettings);
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
      fetchHealth(),
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
