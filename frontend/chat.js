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

// conversationId shares the auth session's lifetime so a reload reopens the
// active thread; logout (resetAuth) clears it along with the token.
//
// The refresh token and the session id are deliberately absent (SPEC §17.10).
// The server sets both as HttpOnly cookies the page cannot read, so a copy
// here would be a durable credential any script on the page could take —
// removing the protection the cookie exists to provide, and outliving the
// short-lived access token it was meant to replace.
const persistedKeys = ['accessToken', 'tenantId', 'role', 'userId', 'conversationId'];

const createState = (storage) => {
  const backing = {
    accessToken: storage.read('accessToken'),
    tenantId: storage.read('tenantId'),
    role: storage.read('role'),
    userId: storage.read('userId'),
    conversationId: storage.read('conversationId'),
    conversationPublic: false,
    attachments: [],
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
      // No longer written, but a session that predates the move to cookies
      // still has them, and signing out is when they should go.
      ['refreshToken', 'sessionId'].forEach((k) => storage.write(k, null));
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

// escapeHtml leaves quotes alone (fine for text nodes); attribute values need
// them encoded too.
const escapeAttr = (str) => escapeHtml(str).replace(/"/g, '&quot;');

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
        <button class="modal-close" aria-label="Close">&times;</button>
      </div>
      <div id="citation-modal-content" class="modal-body"></div>
    </div>
  `;
  // Use addEventListener instead of inline onclick for CSP compliance
  const closeBtn = modal.querySelector('.modal-close');
  if (closeBtn) {
    closeBtn.addEventListener('click', () => modal.classList.remove('active'));
  }
  document.body.appendChild(modal);
};

// Escape key handler moved to initEventListeners() for consistent initialization

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
const approvePatches = $('approve-patches');
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
const ALLOWED_UPLOAD_TYPES = [
  'text/plain', 'text/markdown', 'application/pdf', 'application/json', 'text/csv',
  'application/zip', 'application/x-zip-compressed', 'application/gzip', 'application/x-gzip', 'application/x-tar',
];
const ALLOWED_UPLOAD_EXTENSIONS = ['.txt', '.md', '.markdown', '.pdf', '.json', '.csv', '.yaml', '.yml', '.zip', '.tar', '.tgz', '.gz'];
const ARCHIVE_EXTENSIONS = ['.zip', '.tar', '.tgz', '.tar.gz', '.gz'];
const isArchiveName = (name) => ARCHIVE_EXTENSIONS.some((ext) => name.toLowerCase().endsWith(ext));
// Encode a relative path for a URL, keeping the / separators.
const encodePath = (p) => p.split('/').map(encodeURIComponent).join('/');

const getUploadLimit = () => uploadLimitBytes || DEFAULT_UPLOAD_BYTES;

/**
 * Create a debounced version of a function.
 * @param {Function} fn - Function to debounce
 * @param {number} waitMs - Delay in milliseconds
 * @returns {Function} Debounced function
 */
const debounce = (fn, waitMs) => {
  let timeoutId = null;
  return (...args) => {
    if (timeoutId) clearTimeout(timeoutId);
    timeoutId = setTimeout(() => {
      timeoutId = null;
      fn(...args);
    }, waitMs);
  };
};

// Double-submit CSRF: the server sets a JS-readable csrf_token cookie and
// expects it echoed in X-CSRF-Token on mutating requests that rely on cookies.
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
  // The session id lives in an HttpOnly cookie this page cannot read
  // (SPEC §17.10), so there is nothing here to show.
  if (settingSessionId) settingSessionId.textContent = 'held in a secure cookie';
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

      // Lazy-load the data behind the tab; login only preloads a subset.
      if (state.accessToken) {
        if (tabId === 'notes-tab') fetchNotes();
        else if (tabId === 'contexts-tab') fetchContexts();
        else if (tabId === 'files-tab') fetchUserFiles();
        else if (tabId === 'artifacts-tab') fetchArtifacts();
        else if (tabId === 'tools-tab') refreshToolsAndWorkflows();
        else if (tabId === 'insights-tab') fetchInsights();
      }
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
      if (!section) return;
      section.classList.toggle('collapsed');
      // Lazy-load the files list the first time the section opens.
      if (section.id === 'files-section' && !section.classList.contains('collapsed')) {
        fetchUserFiles();
      }
    });
  });
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
    syncConversationTitle();
  } catch (err) {
    console.warn('Failed to fetch conversations:', err.message);
  }
};

// Keep the header in step with the (possibly just-generated) title.
const syncConversationTitle = () => {
  if (!conversationLabel || !state.conversationId) return;
  const active = (state.conversations || []).find((c) => c.id === state.conversationId);
  if (active?.title) conversationLabel.textContent = active.title;
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
      const apiTag = c.source === 'responses' ? '<span class="source-tag">api</span>' : '';
      return `
        <div class="conversation-item ${isActive ? 'active' : ''}" data-id="${escapeHtml(c.id)}">
          <div class="title">${title}${apiTag}</div>
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
  if (!conversationId) return false;

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
    state.conversationPublic = Boolean(convo.public);
    updateShareButton();
    fetchAttachments();

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
    return true;
  } catch (err) {
    showStatus(err.message, true);
    if (conversationLabel) conversationLabel.textContent = 'Error loading';
    return false;
  }
};

const setConversation = (id) => {
  // A different (or new) conversation starts from the private default until
  // loadConversation reports otherwise.
  if (id !== state.conversationId) {
    state.conversationPublic = false;
    state.attachments = [];
    renderAttachmentChips();
  }
  state.conversationId = id;
  if (conversationLabel) {
    // Never show a raw UUID: use the generated title once we know it.
    const known = (state.conversations || []).find((c) => c.id === id);
    conversationLabel.textContent = known?.title || (id ? 'Untitled conversation' : 'New conversation');
  }
  updateShareButton();
  if (!id) {
    state.lastAssistant = null;
    renderPreferencePanel();
  }
};

// =============================================================================
// Turn navigator — a rail of tick marks on the right, one per turn. At rest it
// is just the ticks; hovering or focusing it expands into a list of the
// model-written turn descriptions, and picking one jumps to that turn.
// =============================================================================

const turnRailEl = $('turn-rail');
const turnRailInnerEl = $('turn-rail-inner');

const renderTurnRail = () => {
  if (!turnRailInnerEl || !messagesEl) return;
  const turns = [...messagesEl.querySelectorAll('.message.user')];
  turnRailEl?.classList.toggle('hidden', turns.length < 2);
  if (turns.length < 2) {
    turnRailInnerEl.innerHTML = '';
    return;
  }
  turnRailInnerEl.innerHTML = turns
    .map((el, i) => {
      // Prefer the generated description; fall back to the message itself so
      // the list is never empty while labels are still being written.
      const label = el.dataset.turnLabel || (el.dataset.raw || '').trim() || `Turn ${i + 1}`;
      return `<button type="button" class="turn-tick" data-turn-index="${i}" title="${escapeAttr(label)}">
        <span class="tick-mark" aria-hidden="true"></span>
        <span class="tick-label">${escapeHtml(label)}</span>
      </button>`;
    })
    .join('');
  highlightActiveTurn();
};

// Mark the turn nearest the top of the reading area as current.
const highlightActiveTurn = () => {
  if (!turnRailInnerEl || !messagesEl) return;
  const turns = [...messagesEl.querySelectorAll('.message.user')];
  if (!turns.length) return;
  const anchor = messagesEl.getBoundingClientRect().top + 80;
  let active = 0;
  turns.forEach((el, i) => {
    if (el.getBoundingClientRect().top <= anchor) active = i;
  });
  turnRailInnerEl.querySelectorAll('.turn-tick').forEach((tick, i) => {
    tick.classList.toggle('active', i === active);
  });
};

/**
 * Turn descriptions and the conversation title are written by a background
 * model pass just after a reply, so fetch them once things settle and patch
 * them in without re-rendering the bubbles (which would flicker).
 */
const refreshTurnLabels = async () => {
  if (!state.accessToken || !state.conversationId) return;
  try {
    const envelope = await requestEnvelope(
      `${apiBase}/conversations/${state.conversationId}/messages?limit=100`,
      { headers: headers() },
      'Failed to refresh labels'
    );
    // Messages appended live during a send have no data-id (only
    // history-rendered ones do), so fall back to positional matching — but
    // align from the END, not the start: the API returns the newest page of
    // messages, so when either side is truncated the two lists share their
    // tail, not their head. Aligning from the start pinned label N onto turn 0.
    const userMessages = (envelope.data?.messages || [])
      .filter((m) => m.role === 'user')
      .sort((a, b) => (a.seq || 0) - (b.seq || 0));
    const rendered = [...(messagesEl?.querySelectorAll('.message.user') || [])];
    const offset = rendered.length - userMessages.length;
    let patched = false;
    rendered.forEach((el, i) => {
      const m = el.dataset.id
        ? userMessages.find((x) => x.id === el.dataset.id)
        : userMessages[i - offset];
      const label = m?.meta?.turn_label;
      if (label && el.dataset.turnLabel !== label) {
        el.dataset.turnLabel = label;
        patched = true;
      }
    });
    if (patched) renderTurnRail();
  } catch {
    /* labels are cosmetic */
  }
};

const initTurnRail = () => {
  if (!turnRailInnerEl || !messagesEl) return;

  turnRailInnerEl.addEventListener('click', (e) => {
    const tick = e.target.closest('.turn-tick');
    if (!tick) return;
    const turns = [...messagesEl.querySelectorAll('.message.user')];
    const target = turns[Number(tick.dataset.turnIndex)];
    if (target) {
      target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      target.classList.add('turn-flash');
      setTimeout(() => target.classList.remove('turn-flash'), 1200);
    }
    // A pointer click is "I'm done here": drop focus so the rail collapses back
    // to bars. Keyboard activation (detail === 0) keeps focus, so the list stays
    // open for continued arrow-key navigation.
    if (e.detail > 0) tick.blur();
  });

  messagesEl.addEventListener('scroll', debounce(highlightActiveTurn, 60));
  window.addEventListener('scroll', debounce(highlightActiveTurn, 60), { passive: true });
};

// =============================================================================
// Composer attachments — drop a file in the composer and it is usable in this
// chat immediately. No context to create or select: the server classifies the
// file and the model reaches it inline, via file_search, or via run_python.
// =============================================================================

const renderAttachmentChips = () => {
  const wrap = $('attachment-chips');
  if (!wrap) return;
  const items = state.attachments || [];
  wrap.innerHTML = items
    .map((a) => {
      const how = a.inline
        ? 'in prompt'
        : a.searchable
          ? 'searchable'
          : 'code';
      return `<span class="attachment-chip" title="${escapeAttr(a.name)} · ${how}">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/><polyline points="13 2 13 9 20 9"/>
        </svg>
        <span class="chip-name">${escapeHtml(a.name)}</span>
        <span class="chip-kind">${how}</span>
      </span>`;
    })
    .join('');
};

const fetchAttachments = async () => {
  if (!state.accessToken || !state.conversationId) {
    state.attachments = [];
    renderAttachmentChips();
    return;
  }
  try {
    const envelope = await requestEnvelope(
      `${apiBase}/conversations/${state.conversationId}/attachments`,
      { headers: headers() },
      'Failed to load attachments'
    );
    state.attachments = envelope.data?.items || [];
  } catch {
    state.attachments = [];
  }
  renderAttachmentChips();
};

/**
 * Attach a file to the current conversation, creating the conversation first
 * if the user has not sent anything yet (so a file can start a chat).
 */
const attachFileToConversation = async (file) => {
  if (!state.accessToken) {
    showStatus('Sign in to attach files.', true);
    return;
  }
  if (!file) return;

  const check = validateUploadFile(file);
  if (!check.ok) {
    showStatus(check.message, true);
    return;
  }

  try {
    showStatus(`Attaching ${file.name}...`);
    if (!state.conversationId) {
      const created = await requestEnvelope(
        `${apiBase}/conversations`,
        { method: 'POST', headers: headers(), body: JSON.stringify({ title: file.name }) },
        'Failed to start conversation'
      );
      setConversation(created.data?.id);
    }

    const form = new FormData();
    form.append('file', file);
    form.append('conversation_id', state.conversationId);
    const envelope = await requestEnvelope(
      `${apiBase}/files/upload`,
      // authHeaders (not headers) so the browser sets the multipart boundary.
      { method: 'POST', headers: authHeaders(), body: form },
      'Attach failed'
    );

    const attachment = envelope.data?.attachment;
    if (attachment) {
      state.attachments = [
        ...(state.attachments || []).filter((a) => a.name !== attachment.name),
        attachment,
      ];
      renderAttachmentChips();
    }
    const how = attachment?.inline
      ? 'included in the prompt'
      : attachment?.searchable
        ? `indexed for search (${envelope.data?.chunk_count || 0} chunks)`
        : 'available to the code interpreter';
    showStatus(`Attached ${file.name} — ${how}. Ask about it.`);
    fetchConversations();
  } catch (err) {
    showStatus(err.message, true);
  }
};

const initComposerAttachments = () => {
  const fileInput = $('composer-file');
  const dropZone = $('composer-drop');

  $('attach-btn')?.addEventListener('click', () => fileInput?.click());
  fileInput?.addEventListener('change', async () => {
    const file = fileInput.files?.[0];
    fileInput.value = '';
    await attachFileToConversation(file);
  });

  if (!dropZone) return;
  ['dragenter', 'dragover'].forEach((evt) =>
    dropZone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropZone.classList.add('drag-over');
    })
  );
  ['dragleave', 'drop'].forEach((evt) =>
    dropZone.addEventListener(evt, (e) => {
      e.preventDefault();
      if (evt === 'dragleave' && dropZone.contains(e.relatedTarget)) return;
      dropZone.classList.remove('drag-over');
    })
  );
  dropZone.addEventListener('drop', async (e) => {
    const file = e.dataTransfer?.files?.[0];
    if (file) await attachFileToConversation(file);
  });
};

// =============================================================================
// Conversation sharing (private by default; owner can publish to /share/{id})
// =============================================================================

const updateShareButton = () => {
  const btn = $('share-btn');
  if (!btn) return;
  btn.disabled = !state.conversationId;
  btn.textContent = state.conversationPublic ? 'Make Private' : 'Share It';
  btn.title = state.conversationPublic
    ? 'This conversation is public — click to make it private again'
    : 'Publish this conversation to a public read-only page';
  btn.classList.toggle('shared', Boolean(state.conversationPublic));
};

const toggleShareConversation = async () => {
  if (!state.conversationId) {
    showStatus('Start a conversation before sharing it.', true);
    return;
  }
  const btn = $('share-btn');
  const makePublic = !state.conversationPublic;
  try {
    toggleButtonBusy(btn, true, 'Working...');
    const envelope = await requestEnvelope(
      `${apiBase}/conversations/${state.conversationId}/share`,
      { method: 'POST', headers: headers(), body: JSON.stringify({ public: makePublic }) },
      'Sharing failed'
    );
    state.conversationPublic = Boolean(envelope.data?.public);
    if (state.conversationPublic) {
      const url = `${window.location.origin}${envelope.data.share_path}`;
      let copied = false;
      if (navigator.clipboard) {
        try {
          await navigator.clipboard.writeText(url);
          copied = true;
        } catch { /* clipboard unavailable */ }
      }
      showStatus(copied ? `Public link copied: ${url}` : `Public link: ${url}`);
    } else {
      showStatus('Conversation is private again.');
    }
  } catch (err) {
    showStatus(err.message, true);
  } finally {
    toggleButtonBusy(btn, false);
    updateShareButton();
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
    renderTurnRail();
    return;
  }

  messagesEl.innerHTML = messages.map((m) => renderMessage(m)).join('');
  renderTurnRail();

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
  const content = role === 'assistant' ? renderMarkdown(m.content || '') : escapeHtml(m.content || '');

  const metaBits = [];
  if (m.token_count) metaBits.push(`${m.token_count} tokens`);
  if (m.model) metaBits.push(escapeHtml(m.model));

  // Render citations as clickable links per SPEC §17
  // Note: Uses event delegation via messagesEl click handler (see initEventListeners)
  // Citations can be at content_struct.citations OR extracted from content_struct.segments
  let citationsHtml = '';
  let citations = m.content_struct?.citations || [];
  // Fallback: extract citations from segments if not at top level
  if (!citations.length && m.content_struct?.segments) {
    citations = m.content_struct.segments
      .filter(seg => seg.type === 'citation')
      .map(seg => ({
        source_path: seg.source_id || seg.locator || '',
        chunk_id: seg.chunk_id || '',
        content: seg.text || '',
        context_id: seg.context_id || '',
        chunk_index: seg.chunk_index,
        score: seg.score,
      }));
  }
  if (citations.length) {
    citationsHtml = `
      <div class="citations-row">
        ${citations.map((c, i) => {
          // Bug fix: Don't escape path here - only escape at output to prevent double-escaping
          const path = c.source_path || c.chunk_id || `Citation ${i + 1}`;
          const label = path.split('/').pop() || path;
          // JSON.stringify escapes internal quotes; only need & and " for double-quoted attr
          const snippetData = JSON.stringify({
            source_path: c.source_path || '',
            chunk_id: c.chunk_id || '',
            content: c.content || c.snippet || '',
            context_id: c.context_id || '',
            chunk_index: c.chunk_index,
          }).replace(/&/g, '&amp;').replace(/"/g, '&quot;');
          return `<span class="citation-link" title="${escapeHtml(path)}" data-citation="${snippetData}" tabindex="0" role="button">${escapeHtml(label)}</span>`;
        }).join('')}
      </div>
    `;
  }

  // The turn description (written by a quick model pass) rides on the user
  // message's meta and feeds the turn navigator.
  const turnLabel = m.meta?.turn_label ? ` data-turn-label="${escapeAttr(m.meta.turn_label)}"` : '';
  return `
    <div class="message ${role}" data-id="${escapeHtml(m.id || '')}" data-raw="${escapeAttr(m.content || '')}"${turnLabel}>
      <div class="role">${role}</div>
      <div>
        <div class="bubble">${content}</div>
        ${citationsHtml}
        <div class="msg-actions">${MSG_COPY_BUTTON_HTML}</div>
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
  if (role === 'assistant') {
    bubble.innerHTML = renderMarkdown(content);
  } else {
    bubble.textContent = content;
  }
  const metaEl = document.createElement('div');
  metaEl.className = 'meta';
  metaEl.textContent = meta;
  wrapper.dataset.raw = content;
  wrapper.appendChild(roleEl);
  const contentWrap = document.createElement('div');
  contentWrap.appendChild(bubble);
  const actionsEl = document.createElement('div');
  actionsEl.className = 'msg-actions';
  actionsEl.innerHTML = MSG_COPY_BUTTON_HTML;
  contentWrap.appendChild(actionsEl);
  if (meta) contentWrap.appendChild(metaEl);
  wrapper.appendChild(contentWrap);
  if (messagesEl) {
    messagesEl.appendChild(wrapper);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }
  updateEmptyState();
  return wrapper;
};

// Names of the tools the model called for a reply, for the message meta line.
// A node's extra result keys land under `outputs`, so check both places.
const toolNamesFromTrace = (trace) => {
  const names = [];
  for (const entry of Array.isArray(trace) ? trace : []) {
    const calls = [...(entry?.tool_calls || []), ...(entry?.outputs?.tool_calls || [])];
    for (const call of calls) {
      if (call?.tool && !names.includes(call.tool)) names.push(call.tool);
    }
  }
  return names;
};

// Only auto-scroll while the reader is already at the bottom, so scrolling
// up to reread earlier output is never fought by the stream.
const isNearBottom = () =>
  !messagesEl || messagesEl.scrollHeight - messagesEl.scrollTop - messagesEl.clientHeight < 120;

/**
 * Create a streaming message element that can be updated token-by-token.
 * Re-renders are batched per animation frame so fast token streams stay
 * smooth, and a dangling code fence is auto-closed mid-stream.
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
  let frame = null;

  const render = (final) => {
    frame = null;
    const stick = isNearBottom();
    if (role === 'assistant') {
      bubble.innerHTML = renderMarkdown(content, { stream: !final });
    } else {
      bubble.textContent = content;
    }
    if (stick) scrollToBottom();
  };

  return {
    /** Append a token to the message */
    update(token) {
      content += token;
      if (!frame) frame = requestAnimationFrame(() => render(false));
    },
    /** Finalize the message with optional meta info */
    finalize(meta = '') {
      if (frame) cancelAnimationFrame(frame);
      render(true);
      wrapper.classList.remove('streaming');
      // The copy affordance appears once the message is complete.
      wrapper.dataset.raw = content;
      const actionsEl = document.createElement('div');
      actionsEl.className = 'msg-actions';
      actionsEl.innerHTML = MSG_COPY_BUTTON_HTML;
      contentWrap.insertBefore(actionsEl, metaEl);
      if (meta) metaEl.textContent = meta;
    },
    /** Show a warning banner above the message meta */
    warn(text) {
      const el = document.createElement('div');
      el.className = 'msg-warning';
      el.textContent = text;
      contentWrap.insertBefore(el, metaEl);
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

// Three pulsing dots shown between sending a message and the first token,
// optionally labelled with what the model is currently doing.
let typingEl = null;

const TOOL_ACTIVITY_LABELS = {
  file_search: 'Searching your files',
  run_python: 'Running code',
  web_search: 'Searching the web',
  web_fetch: 'Reading a web page',
};

// Injection attempts found in fetched pages, surfaced from the workflow trace.
const injectionFindingsFromTrace = (trace) => {
  const kinds = [];
  for (const entry of Array.isArray(trace) ? trace : []) {
    const found = [
      ...(entry?.injection_findings || []),
      ...(entry?.outputs?.injection_findings || []),
    ];
    for (const kind of found) if (!kinds.includes(kind)) kinds.push(kind);
  }
  return kinds;
};

const showTypingIndicator = (label = '') => {
  if (!messagesEl) return;
  if (!typingEl) {
    typingEl = document.createElement('div');
    typingEl.className = 'message assistant typing';
    typingEl.innerHTML =
      '<div class="bubble"><span class="typing-dots"><span></span><span></span><span></span></span>' +
      '<span class="typing-label"></span></div>';
    messagesEl.appendChild(typingEl);
  }
  const labelEl = typingEl.querySelector('.typing-label');
  if (labelEl) labelEl.textContent = label;
  scrollToBottom();
};

// Tool activity arrives as trace events while the model works.
const showToolActivity = (tool) => {
  showTypingIndicator(`${TOOL_ACTIVITY_LABELS[tool] || tool}...`);
};

const hideTypingIndicator = () => {
  if (!typingEl) return;
  typingEl.remove();
  typingEl = null;
  updateEmptyState();
};

// =============================================================================
// Chat submission
// =============================================================================

// The server's /chat/stream endpoint handles exactly one message per
// connection (it reads a single init frame, streams the reply, and returns),
// so the client opens a fresh socket per send. chatSocket tracks the
// in-flight socket so Stop/teardown can reach it.
let chatSocket = null;
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
  if (chatSocket) {
    if (chatSocket.readyState === WebSocket.OPEN || chatSocket.readyState === WebSocket.CONNECTING) {
      chatSocket.close();
    }
    chatSocket = null;
  }
};

window.addEventListener('beforeunload', cleanupWebSocket);

// Open a fresh socket for one chat exchange; resolves once it is usable.
const openChatSocket = () =>
  new Promise((resolve, reject) => {
    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    const socket = new WebSocket(`${protocol}://${window.location.host}${apiBase}/chat/stream`);
    const timeout = setTimeout(() => {
      socket.close();
      reject(new Error('WebSocket connection timeout'));
    }, 5000);
    socket.addEventListener('open', () => {
      clearTimeout(timeout);
      chatSocket = socket;
      resolve(socket);
    });
    socket.addEventListener('error', () => {
      clearTimeout(timeout);
      reject(new Error('WebSocket connection failed'));
    });
  });

// Maximum message length (characters) - approximately 2k tokens per SPEC §18
const MAX_MESSAGE_LENGTH = 8000;

const sendMessage = async (event) => {
  event.preventDefault();
  const messageInput = $('message-input');
  const content = messageInput?.value?.trim();
  if (!content) return;
  if (!state.accessToken) {
    showStatus('Sign in to chat.', true);
    return;
  }

  // Client-side length validation per Issue 65.7
  if (content.length > MAX_MESSAGE_LENGTH) {
    showStatus(`Message too long (${content.length} chars). Maximum is ${MAX_MESSAGE_LENGTH} characters.`, true);
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
    const ws = await openChatSocket();
    return new Promise((resolve, reject) => {
      let settled = false;
      let streamingMsg = null;
      let messageDoneReceived = false;
      let messageDoneData = {};
      let idleTimer = null;

      const cleanup = () => {
        if (idleTimer) clearTimeout(idleTimer);
        ws.removeEventListener('message', handleMessage);
        ws.removeEventListener('error', handleError);
        ws.removeEventListener('close', handleClose);
        // One exchange per connection - the server won't read another message.
        try { ws.close(); } catch { /* already closed */ }
        if (chatSocket === ws) chatSocket = null;
      };

      // If nothing arrives for a while, give up rather than leaving the
      // send button stuck; the caller falls back to the REST endpoint.
      const armIdleTimer = () => {
        if (idleTimer) clearTimeout(idleTimer);
        idleTimer = setTimeout(() => {
          if (settled) return;
          settled = true;
          cleanup();
          if (streamingMsg) streamingMsg.finalize('Timed out');
          reject(new Error('Streaming timed out'));
        }, 120000);
      };
      armIdleTimer();

      const handleMessage = (event) => {
        if (settled) return;
        armIdleTimer();
        try {
          const msg = JSON.parse(event.data);

          // SPEC §18: Check for streaming events ({"event": "...", "data": "..."})
          if (msg.event) {
            switch (msg.event) {
              case 'token':
                // Create streaming message on first token
                if (!streamingMsg) {
                  hideTypingIndicator();
                  streamingMsg = createStreamingMessage('assistant');
                  showStatus('');
                }
                streamingMsg.update(msg.data || '');
                break;

              case 'trace':
                // The attachment agent reports each tool it runs, so the
                // indicator can say what is happening during the slow part.
                if (msg.data?.tool && !streamingMsg) {
                  showToolActivity(msg.data.tool);
                } else {
                  console.debug('Workflow trace:', msg.data);
                }
                break;

              case 'message_done':
                // Final event per SPEC §18 - carries message_id, conversation_id,
                // adapters, and usage. (The client used to wait for a
                // 'streaming_complete' event the server never sends, which left
                // the send button disabled forever after a streamed reply.)
                messageDoneReceived = true;
                messageDoneData = { ...messageDoneData, ...(msg.data || {}) };
                // Attachment answers come from a tool-calling node, which
                // returns the whole reply at once rather than as tokens — so
                // create the message here if no token ever arrived.
                if (!streamingMsg && messageDoneData.content) {
                  hideTypingIndicator();
                  streamingMsg = createStreamingMessage('assistant');
                  streamingMsg.update(messageDoneData.content);
                  showStatus('');
                }
                if (streamingMsg) {
                  const adapters = (messageDoneData.adapters || []).map(a => a?.name || a?.id || a).filter(Boolean);
                  const tools = toolNamesFromTrace(messageDoneData.workflow_trace);
                  const bits = [];
                  if (tools.length) bits.push(`Used: ${tools.join(', ')}`);
                  if (adapters.length) bits.push(`Adapters: ${adapters.join(', ')}`);
                  streamingMsg.finalize(bits.join(' · '));
                  // A page tried to hijack the model: say so where the user
                  // reads the answer, not just in the server log.
                  const injections = injectionFindingsFromTrace(messageDoneData.workflow_trace);
                  if (injections.length) {
                    streamingMsg.warn(
                      `A fetched page attempted a prompt injection (${injections.join(', ')}). ` +
                      'It was redacted — treat this answer with extra care.'
                    );
                  }
                }
                // Older servers relayed an interim message_done without IDs
                // before the persisted one; only settle once we have the
                // message_id (the close handler and idle timer cover servers
                // that never send it).
                if (messageDoneData.message_id) {
                  settled = true;
                  cleanup();
                  resolve(messageDoneData);
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
          // Safety net: if the socket closed right after message_done, resolve
          // with what we have rather than erroring a completed exchange.
          if (messageDoneReceived) {
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
        // The server rejects dual auth on the socket (fresh_session_required),
        // so send exactly one method — prefer the bearer token.
        access_token: state.accessToken || undefined,
        stream: true,
      }));
    });
  };

  try {
    toggleButtonBusy(sendBtn, true, 'Sending...');
    if (messageInput) messageInput.value = '';
    saveDraft(state.conversationId, '');
    appendMessage('user', content);
    showTypingIndicator();
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

    // The indicator must be gone before the fallback check below, or the
    // typing element would register as the last assistant message.
    hideTypingIndicator();

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
    hideTypingIndicator();
    toggleButtonBusy(sendBtn, false, 'Send');
    updateStreamingUI(false);
    renderTurnRail();
    // The labelling pass runs server-side after the reply; give it a moment.
    setTimeout(refreshTurnLabels, 2500);
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

    // Bug fix: Use authHeaders() consistently - don't duplicate Authorization header
    const response = await fetch(`${apiBase}/voice/transcribe`, {
      method: 'POST',
      headers: authHeaders(),
      body: formData,
    });

    if (!response.ok) {
      const errorText = await response.text().catch(() => 'Unknown error');
      console.error('Transcription failed:', response.status, errorText);
      showStatus('Voice transcription failed. Please try again.', true);
      return;
    }

    const envelope = await response.json();
    if (envelope.status === 'ok' && envelope.data?.transcript) {
      // Insert transcribed text into the message input
      const messageInput = $('message-input');
      if (messageInput) {
        messageInput.value = (messageInput.value + ' ' + envelope.data.transcript).trim();
        messageInput.focus();
      }
    } else {
      const errorMsg = extractError(envelope, 'Transcription failed');
      console.error('Transcription failed:', envelope);
      showStatus(errorMsg, true);
    }
  } catch (err) {
    console.error('Transcription error:', err);
    showStatus('Voice transcription error. Please try again.', true);
  }
};

const speakText = async (text) => {
  if (!text) return;

  // Stop any currently playing audio
  if (currentAudio) {
    currentAudio.pause();
    currentAudio = null;
  }

  // Helper for browser fallback
  const speakWithBrowser = () => {
    const utterance = new SpeechSynthesisUtterance(text);
    window.speechSynthesis.speak(utterance);
  };

  if (!state.accessToken) {
    // Fall back to browser speech synthesis
    speakWithBrowser();
    return;
  }

  try {
    if (voiceOutputBtn) voiceOutputBtn.classList.add('playing');

    const response = await fetch(`${apiBase}/voice/synthesize`, {
      method: 'POST',
      headers: {
        ...headers(),
      },
      body: JSON.stringify({ text }),
    });

    if (!response.ok) {
      console.warn('Voice synthesis API failed, using browser fallback');
      if (voiceOutputBtn) voiceOutputBtn.classList.remove('playing');
      speakWithBrowser();
      return;
    }

    const envelope = await response.json();
    if (envelope.status === 'ok' && envelope.data?.format === 'text/placeholder') {
      // Server has no TTS backend configured and returned a text stub;
      // don't try to play it as audio.
      if (voiceOutputBtn) voiceOutputBtn.classList.remove('playing');
      speakWithBrowser();
    } else if (envelope.status === 'ok' && envelope.data?.audio_url) {
      currentAudio = new Audio(envelope.data.audio_url);
      currentAudio.onended = () => {
        if (voiceOutputBtn) voiceOutputBtn.classList.remove('playing');
      };
      currentAudio.onerror = () => {
        if (voiceOutputBtn) voiceOutputBtn.classList.remove('playing');
        // Fall back to browser speech synthesis
        speakWithBrowser();
      };
      currentAudio.play();
    } else {
      // Fall back to browser speech synthesis
      if (voiceOutputBtn) voiceOutputBtn.classList.remove('playing');
      speakWithBrowser();
    }
  } catch (err) {
    console.error('Speech synthesis error:', err);
    if (voiceOutputBtn) voiceOutputBtn.classList.remove('playing');
    // Fall back to browser speech synthesis
    speakWithBrowser();
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
    // Refresh file list after upload
    await fetchUserFiles();
  } catch (err) {
    setUploadStatus(err.message, true);
  }
};

// =============================================================================
// File Browser (SPEC §13.3, §18)
// =============================================================================

const filesListEl = $('files-list');
const filesEmptyEl = $('files-empty');
const filesPaginationEl = $('files-pagination');
const refreshFilesBtn = $('refresh-files-btn');

let filesOffset = 0;
const FILES_LIMIT = 20;

const fetchUserFiles = async () => {
  if (!state.accessToken) {
    if (filesListEl) filesListEl.innerHTML = '';
    if (filesEmptyEl) filesEmptyEl.style.display = 'block';
    if (filesPaginationEl) filesPaginationEl.innerHTML = '';
    return;
  }

  try {
    const envelope = await requestEnvelope(
      `${apiBase}/files?limit=${FILES_LIMIT}&offset=${filesOffset}`,
      { headers: headers() },
      'Failed to load files'
    );

    const data = envelope.data || {};
    const files = data.files || [];
    const total = data.total || 0;
    const hasNext = data.has_next || false;

    renderFilesList(files, total, hasNext);
  } catch (err) {
    console.error('Failed to fetch files:', err);
    if (filesListEl) filesListEl.innerHTML = '<div class="small" style="color: #b00020;">Failed to load files</div>';
  }
};

const renderFilesList = (files, total, hasNext) => {
  if (!filesListEl) return;

  if (files.length === 0) {
    filesListEl.innerHTML = '';
    if (filesEmptyEl) filesEmptyEl.style.display = 'block';
    if (filesPaginationEl) filesPaginationEl.innerHTML = '';
    return;
  }

  if (filesEmptyEl) filesEmptyEl.style.display = 'none';

  filesListEl.innerHTML = files.map(file => `
    <div class="file-item" data-filename="${escapeHtml(file.name)}">
      <div class="file-info">
        <div class="file-name" title="${escapeHtml(file.name)}">${escapeHtml(file.name)}</div>
        <div class="file-meta">${formatBytes(file.size)} · ${formatRelativeTime(file.modified_at)}</div>
      </div>
      <div class="file-actions">
        ${isArchiveName(file.name) ? '<button type="button" class="download-btn" data-action="extract">Extract</button>' : ''}
        <button type="button" class="download-btn" data-action="vault" title="Copy this file's text into your notes vault">Vault</button>
        <button type="button" class="download-btn" data-action="download">Download</button>
        <button type="button" class="delete-btn" data-action="delete">Delete</button>
      </div>
    </div>
  `).join('');

  // Render pagination
  if (filesPaginationEl) {
    const hasPrev = filesOffset > 0;
    filesPaginationEl.innerHTML = `
      <button type="button" class="minor" ${!hasPrev ? 'disabled' : ''} data-action="prev">Previous</button>
      <span class="small">${filesOffset + 1}-${Math.min(filesOffset + files.length, total)} of ${total}</span>
      <button type="button" class="minor" ${!hasNext ? 'disabled' : ''} data-action="next">Next</button>
    `;
  }
};

const handleFileAction = async (event) => {
  const target = event.target;
  const action = target.dataset?.action;
  if (!action) return;

  if (action === 'prev') {
    filesOffset = Math.max(0, filesOffset - FILES_LIMIT);
    await fetchUserFiles();
    return;
  }

  if (action === 'next') {
    filesOffset += FILES_LIMIT;
    await fetchUserFiles();
    return;
  }

  const fileItem = target.closest('.file-item');
  if (!fileItem) return;
  const filename = fileItem.dataset.filename;
  if (!filename) return;

  if (action === 'download') {
    await downloadFile(filename);
  } else if (action === 'delete') {
    await deleteFile(filename);
  } else if (action === 'extract') {
    await extractFile(filename, target);
  } else if (action === 'vault') {
    try {
      const note = await notesApi('/notes/from-file', {
        method: 'POST', body: JSON.stringify({ name: filename }),
      });
      // Methods compose: "pdf+ocr" = text pages plus ocr'd image pages,
      // "docx-vision" = a doc whose only content was images the model read.
      const m = note.method || '';
      const how = m.includes('vision') ? (m.includes('+') ? ' (text + model-read images)' : ' (read by the model)')
        : m.includes('ocr') ? (m.includes('+') ? ' (text + ocr’d images)' : ' (ocr)')
        : m === 'pdf' || m === 'docx' || m === 'odt' ? ' (text extracted)' : '';
      showStatus(`Added to vault as "${note.title}"${how}${note.truncated ? ', truncated' : ''}`);
    } catch (err) {
      showStatus(err.message || 'Could not add to vault', true);
    }
  }
};

const extractFile = async (filename, button) => {
  if (!state.accessToken) return;

  try {
    toggleButtonBusy(button, true, 'Extracting...');
    if (fileUploadStatus) fileUploadStatus.textContent = `Extracting ${filename}...`;

    const envelope = await requestEnvelope(
      `${apiBase}/files/${encodePath(filename)}/extract`,
      { method: 'POST', headers: headers() },
      'Extraction failed'
    );

    const data = envelope.data || {};
    const skipped = data.skipped?.length ? `, ${data.skipped.length} skipped` : '';
    if (fileUploadStatus) {
      fileUploadStatus.textContent =
        `Extracted ${data.files?.length || 0} file(s) to ${data.extracted_to}/${skipped}`;
    }
    await fetchUserFiles();
  } catch (err) {
    if (fileUploadStatus) fileUploadStatus.textContent = err.message;
  } finally {
    toggleButtonBusy(button, false);
  }
};

const downloadFile = async (filename) => {
  if (!state.accessToken) return;

  try {
    // Get signed download URL
    const envelope = await requestEnvelope(
      `${apiBase}/files/${encodePath(filename)}/url`,
      { headers: headers() },
      'Failed to get download URL'
    );

    const downloadUrl = envelope.data?.download_url;
    if (!downloadUrl) throw new Error('No download URL returned');

    // Fetch the file using the signed URL (already contains full path)
    const response = await fetch(downloadUrl, {
      headers: headers(),
    });

    if (!response.ok) throw new Error('Download failed');

    // Create blob and trigger download
    const blob = await response.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    a.remove();
  } catch (err) {
    console.error('Download failed:', err);
    alert(`Failed to download file: ${err.message}`);
  }
};

const deleteFile = async (filename) => {
  if (!state.accessToken) return;
  if (!confirm(`Delete "${filename}"? This cannot be undone.`)) return;

  try {
    await requestEnvelope(
      `${apiBase}/files/${encodePath(filename)}`,
      {
        method: 'DELETE',
        headers: headers(),
      },
      'Failed to delete file'
    );

    // Refresh the file list
    await fetchUserFiles();
  } catch (err) {
    console.error('Delete failed:', err);
    alert(`Failed to delete file: ${err.message}`);
  }
};

const formatRelativeTime = (isoString) => {
  if (!isoString) return '';
  const date = new Date(isoString);
  const now = new Date();
  const diffMs = now - date;
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMs / 3600000);
  const diffDays = Math.floor(diffMs / 86400000);

  if (diffMins < 1) return 'just now';
  if (diffMins < 60) return `${diffMins}m ago`;
  if (diffHours < 24) return `${diffHours}h ago`;
  if (diffDays < 7) return `${diffDays}d ago`;
  return date.toLocaleDateString();
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
  if (!state.accessToken) {
    setMfaSetupStatus('Sign in first', true);
    return;
  }

  try {
    const envelope = await requestEnvelope(
      `${apiBase}/auth/mfa/request`,
      {
        method: 'POST',
        headers: headers(),
        // No session_id: the server reads its own HttpOnly cookie, which is
        // the only copy of it a browser has.
        body: JSON.stringify({}),
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
      // Create QR placeholder with properly escaped URI to prevent XSS
      const placeholder = document.createElement('div');
      placeholder.className = 'qr-placeholder';
      const instructions = document.createElement('p');
      instructions.textContent = 'Open your authenticator app and add a new account using this URI:';
      const uriCode = document.createElement('code');
      uriCode.style.cssText = 'word-break: break-all; font-size: 0.75rem;';
      uriCode.textContent = otpauth_uri || 'N/A';
      placeholder.appendChild(instructions);
      placeholder.appendChild(uriCode);
      qrContainer.innerHTML = '';
      qrContainer.appendChild(placeholder);
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

  if (!state.accessToken) {
    setMfaSetupStatus('No session. Please sign in again.', true);
    return;
  }

  try {
    await requestEnvelope(
      `${apiBase}/auth/mfa/verify`,
      {
        method: 'POST',
        headers: headers(),
        body: JSON.stringify({ code }),
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
        headers: headers(),
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
// API Keys (served Responses API)
// =============================================================================

const setApiKeyStatus = (message, isError = false) => {
  const el = $('api-key-status');
  if (!el) return;
  el.textContent = message;
  el.style.color = isError ? '#b00020' : 'inherit';
};

const loadApiKeys = async () => {
  const listEl = $('api-key-list');
  if (!listEl) return;
  if (!state.accessToken) {
    listEl.innerHTML = '';
    return;
  }
  try {
    const envelope = await requestEnvelope(
      `${apiBase}/auth/api-keys`,
      { headers: headers() },
      'Failed to load API keys'
    );
    const items = envelope.data?.items || [];
    if (!items.length) {
      listEl.innerHTML = '<div class="empty">No API keys yet</div>';
      return;
    }
    listEl.innerHTML = items
      .map((k) => {
        const created = k.created_at ? new Date(k.created_at).toLocaleDateString() : '';
        const lastUsed = k.last_used_at
          ? `last used ${new Date(k.last_used_at).toLocaleDateString()}`
          : 'never used';
        const stateText = k.revoked_at ? 'revoked' : lastUsed;
        const revokeBtn = k.revoked_at
          ? ''
          : `<button type="button" class="ghost api-key-revoke" data-id="${escapeHtml(k.id)}">Revoke</button>`;
        return `
          <div class="api-key-item ${k.revoked_at ? 'revoked' : ''}">
            <div class="api-key-info">
              <span class="api-key-name">${escapeHtml(k.name || 'unnamed key')}</span>
              <code class="api-key-prefix">${escapeHtml(k.prefix)}…</code>
              <div class="meta">created ${created} · ${stateText}</div>
            </div>
            ${revokeBtn}
          </div>`;
      })
      .join('');
    listEl.querySelectorAll('.api-key-revoke').forEach((btn) => {
      btn.addEventListener('click', () => revokeApiKey(btn.dataset.id));
    });
  } catch (err) {
    setApiKeyStatus(err.message || 'Failed to load API keys', true);
  }
};

const createApiKey = async (event) => {
  event.preventDefault();
  if (!state.accessToken) {
    setApiKeyStatus('Sign in first', true);
    return;
  }
  const nameInput = $('api-key-name');
  try {
    const envelope = await requestEnvelope(
      `${apiBase}/auth/api-keys`,
      {
        method: 'POST',
        headers: headers(),
        body: JSON.stringify({ name: nameInput?.value?.trim() || '' }),
      },
      'Failed to create API key'
    );
    // The one plaintext sighting; after this only the prefix survives.
    const plainEl = $('api-key-plaintext-value');
    if (plainEl) plainEl.textContent = envelope.data?.api_key || '';
    $('api-key-plaintext')?.classList.remove('hidden');
    if (nameInput) nameInput.value = '';
    setApiKeyStatus('Key created — copy it before leaving this page');
    await loadApiKeys();
  } catch (err) {
    setApiKeyStatus(err.message || 'Failed to create API key', true);
  }
};

const revokeApiKey = async (keyId) => {
  if (!keyId) return;
  if (!window.confirm('Revoke this API key? Agents using it stop working immediately.')) return;
  try {
    await requestEnvelope(
      `${apiBase}/auth/api-keys/${encodeURIComponent(keyId)}`,
      { method: 'DELETE', headers: headers() },
      'Failed to revoke API key'
    );
    setApiKeyStatus('Key revoked');
    await loadApiKeys();
  } catch (err) {
    setApiKeyStatus(err.message || 'Failed to revoke API key', true);
  }
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
        headers: headers(),
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
        headers: headers(),
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
    if (resultEl) resultEl.classList.add('hidden');

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
      resultEl.classList.remove('hidden');
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

  const totals = data.totals || {};
  const positive = totals.positive ?? 0;
  const negative = totals.negative ?? 0;
  const neutral = totals.neutral ?? 0;

  if (totalEl) totalEl.textContent = positive + negative + neutral;
  if (positiveEl) positiveEl.textContent = positive;
  if (negativeEl) negativeEl.textContent = negative;
  if (neutralEl) neutralEl.textContent = neutral;

  // Adapters
  const adaptersEl = $('insights-top-adapters');
  if (adaptersEl) {
    const adapters = data.adapters || [];
    if (!adapters.length) {
      adaptersEl.innerHTML = '<div class="empty">No adapter data yet</div>';
    } else {
      adaptersEl.innerHTML = adapters
        .map((a) => `
          <div class="adapter-item">
            <span class="adapter-name">${escapeHtml(a.name || a.id || 'Unknown')}</span>
            <span class="adapter-score">${escapeHtml(a.base_model || a.description || '')}</span>
          </div>
        `)
        .join('');
    }
  }

  // Recent preferences
  const recentEl = $('insights-recent-list');
  if (recentEl) {
    const recent = data.events || [];
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
            <div class="cluster-description">${escapeHtml(c.similarity_hint || c.description || '-')}</div>
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
  const messageInput = $('message-input');
  const text = messageInput?.value || '';
  const charCount = text.length;

  // Update character count indicator if it exists
  const charIndicator = $('char-count-indicator');
  if (charIndicator) {
    if (charCount > MAX_MESSAGE_LENGTH * 0.8) {
      // Show warning when approaching limit
      charIndicator.textContent = `${charCount}/${MAX_MESSAGE_LENGTH}`;
      charIndicator.className = charCount > MAX_MESSAGE_LENGTH ? 'char-count error' : 'char-count warning';
      charIndicator.style.display = 'block';
    } else {
      charIndicator.style.display = 'none';
    }
  }

  // Debounced draft save
  clearTimeout(draftSaveTimeout);
  draftSaveTimeout = setTimeout(() => {
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
  $('share-btn')?.addEventListener('click', toggleShareConversation);
  $('new-thread')?.addEventListener('click', newConversation);
  $('new-thread-secondary')?.addEventListener('click', newConversation);
  $('stop-stream-btn')?.addEventListener('click', cancelStreaming);
  $('new-conversation-btn')?.addEventListener('click', newConversation);
  $('refresh-conversations')?.addEventListener('click', fetchConversations);

  // Citation + copy click delegation (CSP-compliant instead of inline onclick)
  if (messagesEl) {
    messagesEl.addEventListener('click', (e) => {
      const msgCopy = e.target.closest('.msg-copy');
      if (msgCopy) {
        const msg = msgCopy.closest('.message');
        const raw = msg?.dataset.raw || msg?.querySelector('.bubble')?.innerText || '';
        if (raw && navigator.clipboard) {
          navigator.clipboard.writeText(raw).then(() => {
            msgCopy.classList.add('copied');
            setTimeout(() => msgCopy.classList.remove('copied'), 1600);
          }).catch(() => {});
        }
        return;
      }
      const copyBtn = e.target.closest('.code-copy');
      if (copyBtn) {
        const code = copyBtn.closest('.codeblock')?.querySelector('code');
        if (code && navigator.clipboard) {
          navigator.clipboard.writeText(code.textContent).then(() => {
            copyBtn.textContent = 'Copied';
            copyBtn.classList.add('copied');
            setTimeout(() => {
              copyBtn.textContent = 'Copy';
              copyBtn.classList.remove('copied');
            }, 1600);
          }).catch(() => {});
        }
        return;
      }
      const citationLink = e.target.closest('.citation-link');
      if (citationLink) {
        showCitationModal(citationLink);
      }
    });
    // Support keyboard activation for accessibility
    messagesEl.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        const citationLink = e.target.closest('.citation-link');
        if (citationLink) {
          e.preventDefault();
          showCitationModal(citationLink);
        }
      }
    });
  }

  // Conversation search (debounced to avoid excessive re-renders)
  conversationSearchEl?.addEventListener('input', debounce(renderConversationList, 150));

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

  // File browser
  if (refreshFilesBtn) refreshFilesBtn.addEventListener('click', fetchUserFiles);
  if (filesListEl) filesListEl.addEventListener('click', handleFileAction);
  if (filesPaginationEl) filesPaginationEl.addEventListener('click', handleFileAction);
  // Note: the files section expand/collapse (and its lazy fetch) is handled
  // by initCollapsibleSections; a second toggle listener here made every
  // click toggle twice, so the section could never be opened.

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
  $('api-key-create-form')?.addEventListener('submit', createApiKey);

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
  initComposerAttachments();
  initTurnRail();
  initNotes();
  updateAuthUI();
  updateShareButton();
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
      loadApiKeys(),
    ]);

    // Reopen the thread that was active before the reload; if it no longer
    // exists (deleted, or a stale id), fall back to a fresh conversation.
    if (state.conversationId && !(await loadConversation(state.conversationId))) {
      setConversation(null);
      showStatus('');
      if (messageInput) messageInput.value = getDraft(null);
    }
  }

  updateEmptyState();
};

// Run on DOM ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
