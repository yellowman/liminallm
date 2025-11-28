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
const preferenceStatusEl = document.getElementById('preference-status');
const preferenceMetaEl = document.getElementById('preference-meta');
const preferenceRoutingEl = document.getElementById('preference-routing');
const preferenceTargetEl = document.getElementById('preference-target');
const preferenceHintEl = document.getElementById('preference-hint');
const preferenceNotesEl = document.getElementById('preference-notes');
const fileUploadInput = document.getElementById('file-upload');
const fileUploadStatus = document.getElementById('file-upload-status');
const fileUploadHint = document.getElementById('file-upload-hint');
const fileUploadContextId = document.getElementById('upload-context-id');
const fileUploadChunkSize = document.getElementById('upload-chunk-size');
const fileUploadButton = document.getElementById('upload-file-btn');

const sessionStorageKey = (key) => `liminal.${key}`;

const readSession = (key) => sessionStorage.getItem(sessionStorageKey(key));
const writeSession = (key, value) => {
  if (value) {
    sessionStorage.setItem(sessionStorageKey(key), value);
  } else {
    sessionStorage.removeItem(sessionStorageKey(key));
  }
};

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

const state = createState({ read: readSession, write: writeSession });

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

const DEFAULT_UPLOAD_BYTES = 10 * 1024 * 1024;
let uploadLimitBytes = null;
const ALLOWED_UPLOAD_TYPES = ['text/plain', 'text/markdown', 'application/pdf', 'application/json', 'text/csv'];
const ALLOWED_UPLOAD_EXTENSIONS = ['.txt', '.md', '.markdown', '.pdf', '.json', '.csv', '.yaml', '.yml'];

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

const getUploadLimit = () => uploadLimitBytes || DEFAULT_UPLOAD_BYTES;

const refreshUploadLimits = async () => {
  if (!state.accessToken) return;
  try {
    const envelope = await requestEnvelope(
      `${apiBase}/files/limits`,
      { method: 'GET', headers: headers() },
      'Failed to load upload limits'
    );
    uploadLimitBytes = envelope.data?.max_upload_bytes || uploadLimitBytes;
    renderUploadHint();
  } catch (err) {
    // Silently ignore upload limit refresh failures - non-critical
  }
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

const authHeaders = (idempotencyKey) => {
  const h = {};
  if (state.accessToken) h['Authorization'] = `Bearer ${state.accessToken}`;
  if (state.tenantId) h['X-Tenant-ID'] = state.tenantId;
  if (state.sessionId) h['session_id'] = state.sessionId;
  h['Idempotency-Key'] = idempotencyKey || randomIdempotencyKey();
  return h;
};

const headers = (idempotencyKey) => ({ 'Content-Type': 'application/json', ...authHeaders(idempotencyKey) });

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
  sessionIndicator.textContent = state.accessToken
    ? `Signed in as ${state.userId || payload.user_id || 'current'} (${state.role || 'user'})`
    : 'Not signed in';
  renderAdminNotice();
  refreshUploadLimits();
};

const setUploadStatus = (message, isError = false) => {
  if (!fileUploadStatus) return;
  fileUploadStatus.textContent = message;
  fileUploadStatus.style.color = isError ? '#b00020' : 'inherit';
};

const validateUploadFile = (file) => {
  if (!file) return { ok: false, message: 'Choose a file to upload.' };
  const limit = getUploadLimit();
  if (file.size > limit) {
    return {
      ok: false,
      message: `File too large (${formatBytes(file.size)}). Max allowed is ${formatBytes(limit)}.`,
    };
  }
  const name = (file.name || '').toLowerCase();
  const matchesType = file.type && ALLOWED_UPLOAD_TYPES.some((t) => file.type.startsWith(t));
  const matchesExt = ALLOWED_UPLOAD_EXTENSIONS.some((ext) => name.endsWith(ext));
  if (!matchesType && !matchesExt) {
    return {
      ok: false,
      message: `Unsupported file type. Allowed: ${ALLOWED_UPLOAD_EXTENSIONS.join(', ')}`,
    };
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

  const contextId = (fileUploadContextId?.value || document.getElementById('context-id')?.value || '').trim();
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
      {
        method: 'POST',
        headers: authHeaders(idempotencyKey),
        body: formData,
      },
      'Upload failed'
    );
    const uploaded = envelope.data || {};
    const destLabel = uploaded.context_id ? `context ${uploaded.context_id}` : 'your files area';
    const chunkLabel = uploaded.chunk_count ? ` · ${uploaded.chunk_count} chunk(s) indexed` : '';
    setUploadStatus(`Uploaded ${file.name} to ${destLabel}${chunkLabel}.`);
  } catch (err) {
    setUploadStatus(err.message, true);
  }
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
  if (!id) {
    state.lastAssistant = null;
    renderPreferencePanel();
  }
};

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
  preferenceTargetEl.textContent = `Conversation ${conversationId} · Message ${messageId}`;
  const meta = {
    adapters: adapters || [],
    context_snippets: contextSnippets?.length || 0,
    adapter_gates: state.lastAssistant.adapterGates || [],
  };
  preferenceMetaEl.textContent = JSON.stringify(meta, null, 2);
  if (state.lastAssistant.routingTrace?.length || state.lastAssistant.workflowTrace?.length) {
    preferenceRoutingEl.textContent = JSON.stringify(
      {
        routing_trace: state.lastAssistant.routingTrace || [],
        workflow_trace: state.lastAssistant.workflowTrace || [],
      },
      null,
      2
    );
  } else {
    preferenceRoutingEl.textContent = 'No routing trace';
  }
};

let chatSocket = null;
let chatSocketConnecting = false;
let chatSocketReconnectTimer = null;

// Cleanup WebSocket on page unload to prevent memory leaks
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
  chatSocket.onopen = () => {
    chatSocketConnecting = false;
  };
  chatSocket.onerror = () => {
    chatSocketConnecting = false;
  };
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
    const handleOpen = () => {
      cleanup();
      resolve(socket);
    };
    const handleError = () => {
      cleanup();
      reject(new Error('WebSocket connection failed'));
    };
    socket.addEventListener('open', handleOpen);
    socket.addEventListener('error', handleError);
  });

const extractError = (payload, fallback) => {
  const detail = payload?.detail || payload?.error || payload;
  if (typeof detail === 'string') return detail.trim() || fallback;
  if (detail?.message) return detail.message;
  if (detail?.error?.message) return detail.error.message;
  return fallback;
};

// Attempt to refresh access token using refresh token
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
  } catch (err) {
    // Token refresh failed - will fall through to require re-authentication
  }
  return false;
};

const requestEnvelope = async (url, options, fallbackMessage) => {
  let resp = await fetchWithRetry(url, options);

  // If unauthorized and we have a refresh token, try to refresh and retry
  if (resp.status === 401 && state.refreshToken) {
    const refreshed = await tryRefreshToken();
    if (refreshed) {
      // Update authorization header with new token and retry
      const newOptions = { ...options };
      if (newOptions.headers) {
        newOptions.headers = { ...newOptions.headers, Authorization: `Bearer ${state.accessToken}` };
      }
      resp = await fetchWithRetry(url, newOptions);
    }
  }

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
  const payload = {
    conversation_id: state.conversationId || undefined,
    message: { content, mode: 'text' },
    context_id: document.getElementById('context-id').value || undefined,
    workflow_id: document.getElementById('workflow-id').value || undefined,
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
  };

  const chatViaWebSocket = async () => {
    const ws = await ensureWebSocket();
    return new Promise((resolve, reject) => {
      let settled = false;
      const cleanup = () => {
        ws.removeEventListener('message', handleMessage);
        ws.removeEventListener('error', handleError);
        ws.removeEventListener('close', handleClose);
      };
      const handleMessage = (event) => {
        if (settled) return;
        settled = true;
        cleanup();
        try {
          const envelope = JSON.parse(event.data);
          if (envelope.status === 'ok') {
            resolve(envelope.data);
          } else {
            reject(new Error(extractError(envelope.error, 'Chat failed')));
          }
        } catch (err) {
          const isParseError = err instanceof SyntaxError;
          reject(new Error(isParseError ? 'Received invalid response from server' : err.message));
        }
      };
      const handleError = () => {
        if (settled) return;
        settled = true;
        cleanup();
        reject(new Error('WebSocket connection failed'));
      };
      const handleClose = () => {
        if (settled) return;
        settled = true;
        cleanup();
        reject(new Error('Connection closed'));
      };

      ws.addEventListener('message', handleMessage);
      ws.addEventListener('error', handleError);
      ws.addEventListener('close', handleClose);
      ws.send(
        JSON.stringify({
          idempotency_key: idempotencyKey,
          request_id: randomIdempotencyKey(),
          message: payload.message.content,
          workflow_id: payload.workflow_id,
          context_id: payload.context_id,
          conversation_id: payload.conversation_id,
          access_token: state.accessToken,
          session_id: state.sessionId,
          tenant_id: state.tenantId,
        })
      );
    });
  };

  try {
    // UI updates inside try block so finally can reset the button on any error
    toggleButtonBusy(sendBtn, true, 'Sending...');
    document.getElementById('message-input').value = '';
    appendMessage('user', content);
    showStatus('Thinking...');

    const data = await chatViaWebSocket().catch(async () => {
      const envelope = await requestEnvelope(
        `${apiBase}/chat`,
        {
          method: 'POST',
          headers: headers(idempotencyKey),
          body: JSON.stringify({ ...payload, stream: false }),
        },
        'Chat failed'
      );
      return envelope.data;
    });
    handleChatResponse(data);
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

const sanitizeNotes = (notes) => {
  if (!notes || typeof notes !== 'string') return undefined;
  // Trim and limit length to prevent abuse
  const trimmed = notes.trim().slice(0, 2000);
  if (!trimmed) return undefined;
  return trimmed;
};

const sendPreference = async (isPositive) => {
  if (!state.lastAssistant) {
    preferenceStatusEl.textContent = 'No assistant message to rate yet.';
    return;
  }
  try {
    preferenceStatusEl.textContent = 'Sending feedback...';
    const body = {
      conversation_id: state.lastAssistant.conversationId,
      message_id: state.lastAssistant.messageId,
      feedback: isPositive ? 'positive' : 'negative',
      explicit_signal: isPositive ? 'thumbs_up' : 'thumbs_down',
      routing_trace: state.lastAssistant.routingTrace || undefined,
      adapter_gates: state.lastAssistant.adapterGates || undefined,
      notes: sanitizeNotes(preferenceNotesEl?.value),
    };
    const idempotencyKey = `pref-${stableHash(JSON.stringify({
      cid: body.conversation_id,
      mid: body.message_id,
      fb: body.feedback,
    }))}`;
    await requestEnvelope(
      `${apiBase}/preferences`,
      {
        method: 'POST',
        headers: headers(idempotencyKey),
        body: JSON.stringify(body),
      },
      'Unable to record preference'
    );
    preferenceStatusEl.textContent = 'Thanks for your feedback!';
  } catch (err) {
    preferenceStatusEl.textContent = err.message;
  }
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
      // Logout API call failed - continue with local cleanup
    }
  };

  await tryRevoke();
  state.resetAuth();
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
const thumbsUpBtn = document.getElementById('thumbs-up');
if (thumbsUpBtn) thumbsUpBtn.addEventListener('click', () => sendPreference(true));
const thumbsDownBtn = document.getElementById('thumbs-down');
if (thumbsDownBtn) thumbsDownBtn.addEventListener('click', () => sendPreference(false));
if (fileUploadInput) fileUploadInput.addEventListener('change', renderUploadHint);
if (fileUploadButton) fileUploadButton.addEventListener('click', handleFileUpload);

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
renderPreferencePanel();
renderUploadHint();
