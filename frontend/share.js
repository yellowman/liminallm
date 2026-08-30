/**
 * Read-only viewer for publicly shared conversations, plus the share
 * directory at /share. Relies on common.js (escapeHtml) and markdown.js
 * (renderMarkdown, MSG_COPY_BUTTON_HTML), loaded first.
 */
(() => {
  const publicApi = '/v1/public/conversations';
  const el = (id) => document.getElementById(id);

  const buildMessage = (m) => {
    const isUser = m.role === 'user';
    const wrapper = document.createElement('div');
    wrapper.className = `message ${isUser ? 'user' : 'assistant'}`;
    wrapper.dataset.raw = m.content || '';
    const roleEl = document.createElement('div');
    roleEl.className = 'role';
    roleEl.textContent = m.role;
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    if (isUser) {
      bubble.textContent = m.content || '';
    } else {
      bubble.innerHTML = renderMarkdown(m.content || '');
    }
    const actions = document.createElement('div');
    actions.className = 'msg-actions';
    actions.innerHTML = MSG_COPY_BUTTON_HTML;
    const contentWrap = document.createElement('div');
    contentWrap.appendChild(bubble);
    contentWrap.appendChild(actions);
    wrapper.appendChild(roleEl);
    wrapper.appendChild(contentWrap);
    return wrapper;
  };

  const renderConversation = (data) => {
    el('share-title').textContent = data.title;
    document.title = `${data.title} · liminallm`;
    if (data.updated_at) {
      el('share-updated').textContent = new Date(data.updated_at).toLocaleString();
    }
    const wrap = el('share-messages');
    (data.messages || []).forEach((m) => wrap.appendChild(buildMessage(m)));
    if (data.messages?.length) {
      el('share-empty').style.display = 'none';
    } else {
      el('share-empty').textContent = 'No messages in this conversation.';
    }
  };

  const renderDirectory = (items) => {
    el('share-title').textContent = 'Shared conversations';
    document.title = 'Shared conversations · liminallm';
    if (!items.length) {
      el('share-empty').textContent = 'Nothing has been shared yet.';
      return;
    }
    el('share-empty').style.display = 'none';
    const wrap = el('share-messages');
    wrap.innerHTML = items
      .map(
        (c) => `
        <a class="conversation-item share-dir-item" href="/share/${encodeURIComponent(c.id)}">
          <div class="title">${escapeHtml(c.title)}</div>
          <div class="meta">${c.updated_at ? new Date(c.updated_at).toLocaleDateString() : ''}</div>
        </a>`
      )
      .join('');
  };

  // Copy actions (message + code) - the chat page wires these on #messages,
  // which does not exist here.
  document.addEventListener('click', (e) => {
    const flash = (btn) => {
      btn.classList.add('copied');
      setTimeout(() => btn.classList.remove('copied'), 1600);
    };
    const msgCopy = e.target.closest('.msg-copy');
    if (msgCopy && navigator.clipboard) {
      const raw = msgCopy.closest('.message')?.dataset.raw || '';
      if (raw) navigator.clipboard.writeText(raw).then(() => flash(msgCopy)).catch(() => {});
      return;
    }
    const codeCopy = e.target.closest('.code-copy');
    if (codeCopy && navigator.clipboard) {
      const code = codeCopy.closest('.codeblock')?.querySelector('code');
      if (code) {
        navigator.clipboard.writeText(code.textContent).then(() => {
          codeCopy.textContent = 'Copied';
          setTimeout(() => { codeCopy.textContent = 'Copy'; }, 1600);
        }).catch(() => {});
      }
    }
  });

  window.addEventListener('load', async () => {
    const parts = window.location.pathname.split('/').filter(Boolean);
    const conversationId = parts.length > 1 ? decodeURIComponent(parts[1]) : null;
    try {
      const resp = await fetch(
        conversationId ? `${publicApi}/${encodeURIComponent(conversationId)}` : publicApi
      );
      const envelope = await resp.json();
      if (!resp.ok) throw new Error(envelope?.error?.message || 'not available');
      if (conversationId) {
        renderConversation(envelope.data || {});
      } else {
        renderDirectory(envelope.data?.items || []);
      }
    } catch (err) {
      el('share-title').textContent = 'Not available';
      el('share-empty').textContent =
        'This conversation is private or does not exist.';
    }
  });
})();
