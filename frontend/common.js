// Shared by chat.js and admin.js. Loaded first (plain script, `defer` keeps
// the order), so everything here is a global by the time either runs.
//
// It exists because these functions were copy-pasted between the two files and
// had already started to drift: `persistAuth` diverged, and `extractErrorDetails`
// was added to the console alone, so the chat UI silently kept throwing away the
// structured field errors the API sends. One copy is the only way that stops.

// --------------------------------------------------------------------------
// Escaping and identifiers
// --------------------------------------------------------------------------

const escapeHtml = (str) => {
  if (str == null) return '';
  const text = String(str);
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

// A stored offset counts Unicode code points; a JavaScript string index counts
// UTF-16 code units. The two agree until the text contains anything outside the
// Basic Multilingual Plane - an emoji, and a great many scripts - and then an
// anchor stored as 1 is index 2 here. SPEC §2.2 makes the stored unit
// normative, so the conversion belongs on this side.
//
// Returns the index in `text` of the given code-point offset, clamped to the
// end. Walking the string is what makes it correct: no arithmetic on the
// offset can know how many of the characters before it were pairs.
const utf16Index = (text, codePoints) => {
  const source = String(text ?? '');
  const wanted = Number(codePoints);
  // `Number.isFinite` is load-bearing, not decorative. NaN takes the same
  // path either way, because every comparison against it is false and the
  // loop below stops at once - but `Infinity` does not: without this it
  // satisfies `seen < wanted` for the whole string and returns its length.
  // An anchor that is not a position fails toward the start, where the worst
  // outcome is an empty lead rather than the entire answer.
  if (!Number.isFinite(wanted) || wanted <= 0) return 0;
  let index = 0;
  let seen = 0;
  while (index < source.length && seen < wanted) {
    index += source.codePointAt(index) > 0xffff ? 2 : 1;
    seen += 1;
  }
  return index;
}

// The last `limit` code points of `text`, so a tail never begins half way
// through a surrogate pair the way `slice(-limit)` can.
const lastCodePoints = (text, limit) => {
  const points = Array.from(String(text ?? ''));
  return points.length > limit ? points.slice(-limit).join('') : points.join('');
}

// --------------------------------------------------------------------------
// Citations
// --------------------------------------------------------------------------

//: A message's citation anchors, as a renderer wants them. Shared because two
//: pages show the same thing from two audiences: the signed-in chat reads the
//: stored citation, and the share page reads the projection of it that crosses
//: to a stranger. Both are anchors into `content`, so both convert the same
//: way, and one implementation is how they stay agreed.
//:
//: `lead` is the run-up to the citation in the answer itself. The passage a
//: citation rested on is deliberately not stored - deleting a document has to
//: reach what a citation shows - so what a reader gets before opening one is
//: the words it follows.
const CITATION_LEAD_MAX = 120;

//: A citation names a source, so the chip leads with what kind of source it
//: is and what it is called. Drawn rather than fetched: a real favicon means
//: a request to every cited domain from the reader's browser, which hands
//: those sites the reader's address and the fact that they were cited. These
//: are also mostly the reader's own files, which have no favicon at all.
const CITATION_ICONS = {
  web: '<circle cx="10" cy="10" r="6.5"/><path d="M3.5 10h13M10 3.5c2.8 3.6 2.8 9.4 0 13' +
       'c-2.8-3.6-2.8-9.4 0-13Z"/>',
  file: '<path d="M4.5 3.25h7L15.5 7v9.75a.75.75 0 0 1-.75.75H4.5a.75.75 0 0 1-.75-.75' +
        'V4a.75.75 0 0 1 .75-.75Z"/><path d="M11.25 3.5V7h3.75"/>',
};

//: Two icons, because two are all the set holds. The citation now carries the
//: source's kind, so this reads it instead of guessing from a string: an
//: earlier version read `.md`, `.markdown` and `.txt` as the notes vault, and
//: those are ordinary upload types here, so an attached `manual.md` was
//: presented to the reader as a note they had written.
const citationKind = (kind) => (kind === 'web' ? 'web' : 'file');

//: Long enough to recognise, short enough that eight fit on a line.
const CITATION_LABEL_MAX = 32;
//: After this many the row stops being a list and starts being a wall.
const CITATION_VISIBLE = 8;

//: The title is the label. It is the source's own name - a filename, a page
//: title, a note's heading - and the citation carries it precisely so a
//: reader is never shown an identifier or a path instead.
const citationLabel = (title) => {
  const tail = /^https?:\/\//i.test(title)
    ? title.replace(/^https?:\/\//i, '').replace(/\/$/, '')
    : title;
  return tail.length > CITATION_LABEL_MAX
    ? `${tail.slice(0, CITATION_LABEL_MAX - 1)}\u2026`
    : tail;
};

const escapeAttr = (str) => escapeHtml(str).replace(/"/g, '&quot;');

//: A message's citation chips, or nothing. `interactive` is what separates
//: the two pages: the signed-in chat opens a panel on click, so its chips
//: carry the citation and announce themselves as buttons, and the share page
//: has no panel - a `role="button"` there would be a promise to a screen
//: reader that nothing keeps. That page also shows every chip, since the
//: "and N more" control is wired on the chat's own message list.
const citationsRowHtml = (message, { interactive = true } = {}) => {
  const citations = citationViews(message);
  if (!citations.length) return '';
  const visible = interactive ? CITATION_VISIBLE : citations.length;
  const chips = citations.map((c, i) => {
    const label = c.title || `Citation ${i + 1}`;
    // JSON.stringify escapes internal quotes; only need & and " for
    // double-quoted attr - `escapeAttr`, not a second hand-written encoder.
    const data = escapeAttr(JSON.stringify({
      title: c.title || '',
      kind: c.kind || '',
      href: c.href || '',
      lead: c.lead || '',
      evidence: c.evidence || [],
    }));
    // The hover text is the source and the words it follows, which is what a
    // reader wants before deciding to open it.
    const lead = String(c.lead || '').replace(/\s+/g, ' ').trim();
    const hover = lead ? `${label}\n\n\u2026${lead}` : label;
    const extra = i >= visible ? ' is-extra' : '';
    const hide = i >= visible ? ' hidden' : '';
    const kind = citationKind(c.kind);
    // `escapeAttr`: the lead is the model's own answer, quoted back into an
    // attribute, and a quote in it would close this one and open others on
    // the element.
    return `<span class="citation-link${extra}"${hide} data-kind="${kind}" ` +
      `title="${escapeAttr(hover)}"` +
      (interactive ? ` data-citation="${data}" tabindex="0" role="button"` : '') +
      `>` +
      `<svg class="citation-icon" viewBox="0 0 20 20" aria-hidden="true" fill="none" ` +
      `stroke="currentColor" stroke-width="1.4" stroke-linecap="round" ` +
      `stroke-linejoin="round">${CITATION_ICONS[kind]}</svg>` +
      `<span class="citation-title">${escapeHtml(citationLabel(label))}</span></span>`;
  }).join('');
  const more = citations.length > visible
    ? `<button type="button" class="citation-more">and ${citations.length - visible} more</button>`
    : '';
  return `<div class="citations-row">${chips}${more}</div>`;
}

const citationViews = (message) => {
  const content = String(message?.content || '');
  return (message?.content_struct?.segments || [])
    .filter((seg) => seg && seg.type === 'citation')
    .map((seg) => ({
      title: seg.meta?.title || '',
      kind: seg.meta?.kind || '',
      href: seg.locator || '',
      lead: lastCodePoints(
        content.slice(0, utf16Index(content, seg.start)).trim(),
        CITATION_LEAD_MAX,
      ),
      // Absent on a share, where fingerprints of passages the reader cannot
      // read do not cross.
      evidence: seg.meta?.evidence || [],
    }));
}

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
}

// --------------------------------------------------------------------------
// Requests
// --------------------------------------------------------------------------

const getCsrfToken = () => {
  const match = document.cookie.match(/(?:^|;\s*)csrf_token=([^;]+)/);
  return match ? decodeURIComponent(match[1]) : null;
};

// `state` is defined by whichever page loaded this; both keep the same auth
// fields on it.
//
// No tenant header. The server derives the tenant from the hostname the request
// arrived at (service/tenancy.py), so a client cannot name its own. This used
// to echo back the value the server itself had just sent at login, where the
// only two outcomes were "matches" and "401".
// No `session_id` header either. The browser's session id is an HttpOnly
// cookie it cannot read (SPEC §17.10), and the access token is the credential
// this page actually holds; the header remains for API clients that
// authenticate that way.
const authHeaders = (idempotencyKey) => {
  const h = {};
  if (state.accessToken) h['Authorization'] = `Bearer ${state.accessToken}`;
  const csrf = getCsrfToken();
  if (csrf) h['X-CSRF-Token'] = csrf;
  h['Idempotency-Key'] = idempotencyKey || randomIdempotencyKey();
  return h;
};

const headers = (idempotencyKey) => ({
  'Content-Type': 'application/json',
  ...authHeaders(idempotencyKey),
});

// --------------------------------------------------------------------------
// Errors
// --------------------------------------------------------------------------

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

// Carries the status and the structured details onto the Error, so a caller
// can react to either without re-parsing the response.
const envelopeError = (payload, raw, fallbackMessage) => {
  const error = new Error(
    extractError(payload ?? raw, fallbackMessage || 'Request failed')
  );
  error.details = extractErrorDetails(payload);
  return error;
};


// --------------------------------------------------------------------------
// The request layer
//
// Both pages had their own copy and both had drifted. The console's version
// treated 429 as a fatal client error instead of backing off, and never
// refreshed an expired token on 401 - so an admin session simply failed where
// a chat session recovered. These are the chat versions, which had both.
//
// `state`, `apiBase` and `persistAuth` come from whichever page loaded this.
// --------------------------------------------------------------------------

const jsonHeaders = () => {
  const h = { 'Content-Type': 'application/json' };
  const csrf = getCsrfToken();
  if (csrf) h['X-CSRF-Token'] = csrf;
  return h;
}

const fetchWithRetry = async (url, options, retries = 3, backoffMs = 400) => {
  let lastError;
  for (let attempt = 0; attempt <= retries; attempt += 1) {
    try {
      const resp = await fetch(url, options);
      // Handle 429 rate limit with exponential backoff
      if (resp.status === 429) {
        const retryAfter = resp.headers.get('Retry-After');
        const waitMs = retryAfter ? parseInt(retryAfter, 10) * 1000 : backoffMs * Math.pow(2, attempt + 2);
        lastError = new Error('Rate limit exceeded');
        if (attempt === retries) return resp;
        await new Promise((r) => setTimeout(r, Math.min(waitMs, 30000)));
        continue;
      }
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
}

// The refresh credential is the HttpOnly cookie, which a same-origin request
// carries on its own and this code could not read if it wanted to. So the body
// is empty: `refresh_token` in it would require the page to hold a durable
// credential, which is the thing §17.10 moved into the cookie. `tenant_id` is
// gone with it - the server derives the tenant from the hostname, and echoing
// back the value it had just sent was never anything but "matches" or 401.
//
// `credentials: 'same-origin'` is the default, and stated rather than assumed
// because this request is now nothing but the cookie it carries.
const tryRefreshToken = async () => {
  try {
    const resp = await fetch(`${apiBase}/auth/refresh`, {
      method: 'POST',
      headers: jsonHeaders(),
      credentials: 'same-origin',
      body: JSON.stringify({}),
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
}

const requestEnvelope = async (url, options, fallbackMessage) => {
  let resp = await fetchWithRetry(url, options);

  // The trigger is "we had an authenticated session", not "a refresh token is
  // visible to JS" - which is no longer true of any browser session. One
  // attempt: a second would be retrying a cookie the server just rejected.
  if (resp.status === 401 && state.accessToken) {
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
    // envelopeError carries error.details through, so a caller can act on the
    // API's per-field problems instead of only its message string.
    const error = envelopeError(payload, text, fallbackMessage);
    error.status = resp.status;
    throw error;
  }
  return payload ?? {};
}
