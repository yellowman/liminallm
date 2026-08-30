/**
 * The Notes vault tab: editor, wiki-links, backlinks, graph, witness runs,
 * and the vault-wide sweep.
 *
 * Definitions only - chat.js calls initNotes() at DOMContentLoaded.
 */

// =============================================================================
// Notes vault
// =============================================================================

const notesState = {
  notes: [],           // list metadata, newest first
  currentId: null,
  dirty: false,
  graph: null,         // {nodes, edges} when the graph view is open
  contradicted: new Set(), // note ids the witness flagged as contradicting
  evolved: new Set(),      // note ids whose position moved
  searchTimer: null,
};

const notesApi = async (path, options = {}) => {
  const resp = await fetchWithRetry(`${apiBase}${path}`, {
    headers: headers(),
    ...options,
  });
  const body = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    const message = body?.error?.message || body?.detail?.error?.message || `Request failed (${resp.status})`;
    throw new Error(message);
  }
  return body.data;
};

// [[Title]] → clickable wikilink. Runs on rendered (already-escaped) HTML;
// code blocks are left alone - [[x]] in code is code.
const linkifyWikiLinks = (html) =>
  html.split(/(<pre[\s\S]*?<\/pre>|<code[\s\S]*?<\/code>)/g)
    .map((seg, i) => (i % 2 ? seg : seg.replace(/\[\[([^\[\]\n]{1,200})\]\]/g, (_m, title) =>
      `<a href="#" class="wikilink" data-note-title="${escapeAttr(title.trim())}">${title.trim()}</a>`)))
    .join('');

const renderNoteList = () => {
  const list = $('note-list');
  if (!list) return;
  list.innerHTML = notesState.notes.map((n) => `
    <li class="note-item${n.id === notesState.currentId ? ' active' : ''}${notesState.contradicted.has(n.id) ? ' contradicted' : notesState.evolved.has(n.id) ? ' evolved' : ''}" data-id="${escapeAttr(n.id)}">
      <span class="note-item-title">${escapeHtml(n.title)}</span>
      <span class="note-item-date">${new Date(n.updated_at).toLocaleDateString()}</span>
    </li>`).join('');
  const count = $('note-count');
  if (count) count.textContent = notesState.notes.length ? `${notesState.notes.length} notes` : '';
};

const fetchNotes = async () => {
  try {
    const data = await notesApi('/notes?limit=500');
    notesState.notes = data.notes || [];
    renderNoteList();
    $('notes-empty')?.classList.toggle('hidden', notesState.notes.length > 0 || !!notesState.currentId);
  } catch (err) {
    if (String(err.message || '').includes('disabled')) {
      document.querySelector('[data-tab="notes-tab"]')?.classList.add('hidden');
    }
    console.warn('notes fetch failed', err);
  }
};

const showNoteEditor = (show) => {
  $('note-editor')?.classList.toggle('hidden', !show);
  $('notes-empty')?.classList.toggle('hidden', show || notesState.notes.length > 0);
  $('note-graph-wrap')?.classList.add('hidden');
  $('note-sweep-wrap')?.classList.add('hidden');
};

const openNote = async (noteId) => {
  try {
    const note = await notesApi(`/notes/${encodeURIComponent(noteId)}`);
    notesState.currentId = note.id;
    notesState.dirty = false;
    showNoteEditor(true);
    $('note-title').value = note.title;
    $('note-content').value = note.content;
    setNotePreview(false);
    $('note-witness-results')?.classList.add('hidden');
    const meta = $('note-meta');
    if (meta) {
      const backlinks = (note.backlinks || []).map((b) =>
        `<a href="#" class="wikilink" data-note-id="${escapeAttr(b.id)}">${escapeHtml(b.title)}</a>`).join(' · ');
      const dangling = (note.dangling || []).map((t) => `<span class="dangling" title="No note with this title yet">[[${escapeHtml(t)}]]</span>`).join(' ');
      meta.innerHTML = [
        backlinks ? `<span class="muted">Linked from:</span> ${backlinks}` : '',
        dangling ? `<span class="muted">Unresolved:</span> ${dangling}` : '',
      ].filter(Boolean).join('<br/>');
    }
    renderNoteList();
  } catch (err) {
    showStatus(err.message || 'Could not open note', true);
  }
};

const openNoteByTitle = async (title) => {
  const existing = notesState.notes.find((n) => n.title.toLowerCase() === title.toLowerCase());
  if (existing) return openNote(existing.id);
  // Follow the Obsidian convention: clicking a dangling link creates the note.
  try {
    const note = await notesApi('/notes', { method: 'POST', body: JSON.stringify({ title, content: '' }) });
    await fetchNotes();
    await openNote(note.id);
  } catch (err) {
    showStatus(err.message || 'Could not create note', true);
  }
};

const setNotePreview = (on) => {
  const btn = $('note-preview-btn');
  const preview = $('note-preview');
  const textarea = $('note-content');
  if (!btn || !preview || !textarea) return;
  btn.setAttribute('aria-pressed', on ? 'true' : 'false');
  btn.classList.toggle('active', on);
  preview.classList.toggle('hidden', !on);
  textarea.classList.toggle('hidden', on);
  if (on) preview.innerHTML = linkifyWikiLinks(renderMarkdown(textarea.value));
};

const saveCurrentNote = async () => {
  const title = $('note-title')?.value.trim();
  const content = $('note-content')?.value ?? '';
  if (!title) { showStatus('A note needs a title', true); return; }
  try {
    if (notesState.currentId) {
      await notesApi(`/notes/${encodeURIComponent(notesState.currentId)}`, {
        method: 'PATCH', body: JSON.stringify({ title, content }),
      });
    } else {
      const note = await notesApi('/notes', { method: 'POST', body: JSON.stringify({ title, content }) });
      notesState.currentId = note.id;
    }
    notesState.dirty = false;
    await fetchNotes();
    await openNote(notesState.currentId);
    showStatus('Saved');
  } catch (err) {
    showStatus(err.message || 'Save failed', true);
  }
};

const deleteCurrentNote = async () => {
  if (!notesState.currentId) return;
  try {
    await notesApi(`/notes/${encodeURIComponent(notesState.currentId)}`, { method: 'DELETE' });
    notesState.currentId = null;
    showNoteEditor(false);
    await fetchNotes();
  } catch (err) {
    showStatus(err.message || 'Delete failed', true);
  }
};

const runWitness = async () => {
  if (!notesState.currentId) return;
  const box = $('note-witness-results');
  const btn = $('note-witness-btn');
  if (!box || !btn) return;
  if (notesState.dirty) await saveCurrentNote();
  btn.disabled = true;
  box.classList.remove('hidden');
  box.innerHTML = '<div class="muted">The witness is reading the vault…</div>';
  try {
    const report = await notesApi(`/notes/${encodeURIComponent(notesState.currentId)}/witness`, {
      method: 'POST', body: JSON.stringify({}),
    });
    const rows = (report.findings || []).map((f) => witnessRowHtml(f, f.note_id, f.title, f.days_apart));
    (report.findings || []).forEach((f) => {
      if (f.verdict === 'CONTRADICTS') notesState.contradicted.add(f.note_id);
      if (f.verdict === 'EVOLVES') notesState.evolved.add(f.note_id);
    });
    if (notesState.contradicted.size) notesState.contradicted.add(report.note_id);
    box.innerHTML = report.checked === 0
      ? '<div class="muted">Nothing in the vault is close enough to compare yet.</div>'
      : `<div class="witness-summary">${witnessSummaryText(report, report.checked)}</div>${rows.join('')}`;
    renderNoteList();
  } catch (err) {
    box.innerHTML = `<div class="muted">${escapeHtml(err.message || 'Witness unavailable')}</div>`;
  } finally {
    btn.disabled = false;
  }
};

const VERDICT_CLASS = { CONTRADICTS: 'contradicts', EVOLVES: 'evolves', AGREES: 'agrees', UNRELATED: 'unrelated' };

// The process is comparison; the summary reports whatever fell out of it.
const witnessSummaryText = (report, checked) => {
  const parts = [];
  if (report.contradictions) parts.push(`${report.contradictions} contradiction${report.contradictions > 1 ? 's' : ''}`);
  if (report.evolutions) parts.push(`${report.evolutions} position${report.evolutions > 1 ? 's' : ''} that moved`);
  return parts.length
    ? `Compared ${checked} pairs of thoughts: ${parts.join(', ')}.`
    : `Compared ${checked} pairs of thoughts - your thinking holds together.`;
};

const witnessRowHtml = (f, noteId, title, daysApart) => {
  const cls = VERDICT_CLASS[f.verdict] || 'unrelated';
  const path = f.path_titles
    ? `<div class="witness-path">${f.path_titles.map((t) => escapeHtml(t)).join(' → ')}</div>` : '';
  return `<div class="witness-row ${cls}">
    <span class="witness-verdict">${f.verdict}</span>
    <a href="#" class="wikilink" data-note-id="${escapeAttr(noteId)}">${escapeHtml(title)}</a>
    <span class="muted">${daysApart} days apart</span>
    ${f.reason ? `<div class="witness-reason">${escapeHtml(f.reason)}</div>` : ''}
    ${path}
  </div>`;
};

const runVaultSweep = async () => {
  const wrap = $('note-sweep-wrap');
  const btn = $('note-sweep-btn');
  if (!wrap || !btn) return;
  $('note-editor')?.classList.add('hidden');
  $('note-graph-wrap')?.classList.add('hidden');
  $('notes-empty')?.classList.add('hidden');
  wrap.classList.remove('hidden');
  btn.disabled = true;
  wrap.innerHTML = '<div class="muted" style="padding:24px">The witness is reading the whole vault…</div>';
  try {
    const report = await notesApi('/notes/sweep', { method: 'POST', body: JSON.stringify({}) });
    report.findings.forEach((f) => {
      if (f.verdict === 'CONTRADICTS') { notesState.contradicted.add(f.a.id); notesState.contradicted.add(f.b.id); }
      if (f.verdict === 'EVOLVES') { notesState.evolved.add(f.a.id); notesState.evolved.add(f.b.id); }
    });
    const coverage = report.notes_scanned >= report.notes_cap || report.judged >= report.judgment_cap
      ? `<div class="muted small">Bounded pass: scanned ${report.notes_scanned} notes, judged the strongest ${report.judged} of ${report.pairs_considered} candidate pairs.</div>`
      : '';
    const rows = report.findings.map((f) => witnessRowHtml(
      f, f.b.id, `${f.a.title} ↔ ${f.b.title}`, f.days_apart));
    wrap.innerHTML = `<div class="note-sweep">
      <div class="witness-summary">${witnessSummaryText(report, report.judged)}</div>
      ${coverage}
      ${rows.join('') || '<div class="muted">No two notes were close enough to compare.</div>'}
    </div>`;
    renderNoteList();
  } catch (err) {
    wrap.innerHTML = `<div class="muted" style="padding:24px">${escapeHtml(err.message || 'Sweep unavailable')}</div>`;
  } finally {
    btn.disabled = false;
  }
};

const runNoteSearch = async (query) => {
  const box = $('note-search-results');
  if (!box) return;
  if (!query.trim()) { box.classList.add('hidden'); box.innerHTML = ''; return; }
  try {
    const data = await notesApi('/notes/search', { method: 'POST', body: JSON.stringify({ query, limit: 8 }) });
    const results = data.results || [];
    box.classList.remove('hidden');
    box.innerHTML = results.length
      ? results.map((r) => `<div class="note-search-hit" data-id="${escapeAttr(r.id)}">
          <span class="note-item-title">${escapeHtml(r.title)}</span>
          <span class="note-search-excerpt">${escapeHtml(r.excerpt || '')}</span>
        </div>`).join('')
      : '<div class="muted" style="padding:8px 10px">No matches.</div>';
  } catch (err) {
    console.warn('note search failed', err);
  }
};

// --- graph: a small force layout, enough to see the shape of the vault ---
const drawNoteGraph = async () => {
  const wrap = $('note-graph-wrap');
  const canvas = $('note-graph');
  if (!wrap || !canvas) return;
  $('note-editor')?.classList.add('hidden');
  $('notes-empty')?.classList.add('hidden');
  $('note-sweep-wrap')?.classList.add('hidden');
  wrap.classList.remove('hidden');
  const data = await notesApi('/notes/graph').catch(() => null);
  if (!data) return;
  const dpr = window.devicePixelRatio || 1;
  const width = wrap.clientWidth, height = Math.max(wrap.clientHeight, 420);
  canvas.width = width * dpr; canvas.height = height * dpr;
  canvas.style.width = `${width}px`; canvas.style.height = `${height}px`;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);

  const nodes = data.nodes.map((n, i) => ({
    ...n,
    x: width / 2 + Math.cos((i / Math.max(data.nodes.length, 1)) * Math.PI * 2) * Math.min(width, height) * 0.3,
    y: height / 2 + Math.sin((i / Math.max(data.nodes.length, 1)) * Math.PI * 2) * Math.min(width, height) * 0.3,
    vx: 0, vy: 0,
  }));
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const edges = data.edges.filter((e) => byId.has(e.src) && byId.has(e.dst));
  notesState.graph = { nodes, edges, byId };

  let ticks = 0;
  const step = () => {
    // Repulsion (O(n²) is fine at vault scale), springs along edges, mild centering.
    for (const a of nodes) {
      let fx = (width / 2 - a.x) * 0.002, fy = (height / 2 - a.y) * 0.002;
      for (const b of nodes) {
        if (a === b) continue;
        const dx = a.x - b.x, dy = a.y - b.y;
        const d2 = Math.max(dx * dx + dy * dy, 64);
        const rep = 900 / d2;
        fx += dx * rep / Math.sqrt(d2); fy += dy * rep / Math.sqrt(d2);
      }
      a.vx = (a.vx + fx) * 0.85; a.vy = (a.vy + fy) * 0.85;
    }
    for (const e of edges) {
      const s = byId.get(e.src), t = byId.get(e.dst);
      const dx = t.x - s.x, dy = t.y - s.y;
      const dist = Math.max(Math.hypot(dx, dy), 1);
      const pull = (dist - 90) * 0.004;
      s.vx += dx / dist * pull; s.vy += dy / dist * pull;
      t.vx -= dx / dist * pull; t.vy -= dy / dist * pull;
    }
    for (const n of nodes) {
      n.x = Math.min(Math.max(n.x + n.vx, 20), width - 20);
      n.y = Math.min(Math.max(n.y + n.vy, 20), height - 20);
    }

    ctx.clearRect(0, 0, width, height);
    const styles = getComputedStyle(document.documentElement);
    ctx.strokeStyle = styles.getPropertyValue('--border') || '#ddd';
    ctx.lineWidth = 1;
    for (const e of edges) {
      const s = byId.get(e.src), t = byId.get(e.dst);
      ctx.beginPath(); ctx.moveTo(s.x, s.y); ctx.lineTo(t.x, t.y); ctx.stroke();
    }
    for (const n of nodes) {
      const r = 4 + Math.min(n.degree || 0, 12);
      ctx.beginPath();
      ctx.fillStyle = notesState.contradicted.has(n.id) ? '#c0392b'
        : notesState.evolved.has(n.id) ? '#c07d10'
        : n.id === notesState.currentId ? (styles.getPropertyValue('--accent') || '#4a6fa5') : '#8a8f98';
      ctx.arc(n.x, n.y, r, 0, Math.PI * 2); ctx.fill();
      if ((n.degree || 0) > 2 || nodes.length <= 30) {
        ctx.fillStyle = styles.getPropertyValue('--fg-muted') || '#666';
        ctx.font = '11px sans-serif';
        ctx.fillText(n.title.slice(0, 24), n.x + r + 3, n.y + 3);
      }
    }
    if (ticks++ < 180 && !wrap.classList.contains('hidden')) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);

  canvas.onclick = (ev) => {
    const rect = canvas.getBoundingClientRect();
    const x = ev.clientX - rect.left, y = ev.clientY - rect.top;
    const hit = nodes.find((n) => Math.hypot(n.x - x, n.y - y) < 14);
    if (hit) openNote(hit.id);
  };
};

const initNotes = () => {
  $('note-new-btn')?.addEventListener('click', () => {
    notesState.currentId = null;
    notesState.dirty = false;
    showNoteEditor(true);
    $('note-title').value = '';
    $('note-content').value = '';
    $('note-meta').innerHTML = '';
    $('note-witness-results')?.classList.add('hidden');
    setNotePreview(false);
    $('note-title').focus();
  });
  $('note-save-btn')?.addEventListener('click', saveCurrentNote);
  $('note-delete-btn')?.addEventListener('click', deleteCurrentNote);
  $('note-witness-btn')?.addEventListener('click', runWitness);
  $('note-preview-btn')?.addEventListener('click', () =>
    setNotePreview($('note-preview')?.classList.contains('hidden')));
  $('note-graph-btn')?.addEventListener('click', drawNoteGraph);
  $('note-sweep-btn')?.addEventListener('click', runVaultSweep);
  $('note-content')?.addEventListener('input', () => { notesState.dirty = true; });
  $('note-title')?.addEventListener('input', () => { notesState.dirty = true; });
  $('note-content')?.addEventListener('keydown', (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 's') { e.preventDefault(); saveCurrentNote(); }
  });
  $('note-search-input')?.addEventListener('input', (e) => {
    clearTimeout(notesState.searchTimer);
    notesState.searchTimer = setTimeout(() => runNoteSearch(e.target.value), 250);
  });
  $('note-list')?.addEventListener('click', (e) => {
    const item = e.target.closest('.note-item');
    if (item) openNote(item.dataset.id);
  });
  $('note-search-results')?.addEventListener('click', (e) => {
    const hit = e.target.closest('.note-search-hit');
    if (hit) {
      $('note-search-input').value = '';
      $('note-search-results').classList.add('hidden');
      openNote(hit.dataset.id);
    }
  });
  // Wiki-links anywhere in the notes panel (preview, backlinks, witness rows).
  $('notes-tab')?.addEventListener('click', (e) => {
    const link = e.target.closest('.wikilink');
    if (!link) return;
    e.preventDefault();
    if (link.dataset.noteId) openNote(link.dataset.noteId);
    else if (link.dataset.noteTitle) openNoteByTitle(link.dataset.noteTitle);
  });
};

