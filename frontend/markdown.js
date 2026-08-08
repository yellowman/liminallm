/**
 * Message-rendering primitives shared by the chat page and the public share
 * page: the escape-first markdown renderer and the copy-message button.
 *
 * Load order (script defer preserves it): common.js (escapeHtml) ->
 * markdown.js -> the page driver (chat.js or share.js). The share page used
 * to pull in all of chat.js for renderMarkdown — and still broke, because
 * renderMarkdown calls escapeHtml from common.js, which share.html never
 * loaded: every assistant bubble died on a ReferenceError.
 */

// Copy-message button: overlapping-squares icon, swapped for a check when
// the copy lands. Shown under every user and assistant message.
const MSG_COPY_BUTTON_HTML =
  '<button type="button" class="msg-copy" title="Copy message" aria-label="Copy message">' +
  '<svg class="icon-copy" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
  '<rect x="9" y="9" width="12" height="12" rx="2"></rect>' +
  '<path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>' +
  '<svg class="icon-check" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
  '<polyline points="20 6 9 17 4 12"></polyline></svg>' +
  '</button>';

// =============================================================================
// Markdown renderer — GitHub-flavored subset, dependency-free, escape-first.
// The input is HTML-escaped FIRST, then markdown constructs are rewritten into
// a fixed set of safe tags, so message content can never inject markup.
// =============================================================================

const MD_PH = String.fromCharCode(0);
const MD_SLOT_RE = new RegExp('^' + MD_PH + 'B\\d+' + MD_PH + '$');
const MD_RESTORE_RE = new RegExp(MD_PH + '[BI](\\d+)' + MD_PH, 'g');

// Lightweight syntax highlighting: comments, strings, numbers, and keywords
// are enough to make code read like GitHub without shipping a highlighter.
const CODE_LANG_ALIASES = {
  javascript: 'js', jsx: 'js', ts: 'js', tsx: 'js', typescript: 'js', node: 'js',
  java: 'c', kotlin: 'c', swift: 'c', cpp: 'c', cc: 'c', h: 'c', cs: 'c', csharp: 'c', php: 'c', scala: 'c',
  py: 'python', python3: 'python', rb: 'python', ruby: 'python',
  sh: 'shell', bash: 'shell', zsh: 'shell', console: 'shell',
  yml: 'shell', yaml: 'shell', toml: 'shell', ini: 'shell', dockerfile: 'shell', makefile: 'shell',
  golang: 'go', rs: 'rust', postgres: 'sql', psql: 'sql', mysql: 'sql', sqlite: 'sql',
};

const CODE_PROFILES = {
  js: { comments: 'slash', keywords: 'const let var function class return if else for while do switch case break continue new delete typeof instanceof void yield async await try catch finally throw import export from default extends super this in of static get set null undefined true false' },
  c: { comments: 'slash', keywords: 'int long short char float double void bool unsigned signed struct enum union class interface public private protected static final const return if else for while do switch case break continue new delete this super try catch finally throw import package namespace using var val fun def true false null nullptr' },
  go: { comments: 'slash', keywords: 'func package import type struct interface map chan go defer return if else for range switch case break continue fallthrough var const nil true false select goto' },
  rust: { comments: 'slash', keywords: 'fn let mut pub struct enum impl trait use mod match if else for while loop return break continue crate super where async await move ref const static type dyn true false Some None Ok Err self Self' },
  python: { comments: 'hash', keywords: 'def class return if elif else for while break continue import from as with try except finally raise pass lambda global nonlocal assert yield async await del not and or in is None True False self require puts nil end begin rescue module attr_accessor' },
  shell: { comments: 'hash', keywords: 'if then else elif fi for in do done while until case esac function echo exit return local export readonly set unset shift source alias sudo cd true false' },
  sql: { comments: 'dash', ignoreCase: true, keywords: 'select from where insert into values update set delete create table view index drop alter add column join left right inner outer full cross on as and or not null primary foreign key references default group by order asc desc limit offset having distinct union all exists between like ilike in is case when then else end begin commit rollback transaction returning with' },
  json: { comments: null, keywords: 'true false null' },
  css: { comments: 'block', keywords: '' },
};

const CODE_COMMENT_PATTERNS = {
  slash: '\\/\\/[^\\n]*|\\/\\*[\\s\\S]*?\\*\\/',
  hash: '#[^\\n]*',
  dash: '--[^\\n]*',
  block: '\\/\\*[\\s\\S]*?\\*\\/',
};

// `code` is already HTML-escaped; wrap tokens in spans without re-escaping.
const highlightCode = (code, lang) => {
  const profile = CODE_PROFILES[CODE_LANG_ALIASES[lang] || lang];
  if (!profile) return code;
  const comment = profile.comments ? `(?<com>${CODE_COMMENT_PATTERNS[profile.comments]})` : '(?<com>\\u0001)';
  const string = '(?<str>"(?:\\\\.|[^"\\\\\\n])*"|\'(?:\\\\.|[^\'\\\\\\n])*\'|`(?:\\\\.|[^`\\\\])*`)';
  const number = '(?<num>\\b(?:0[xX][0-9a-fA-F]+|\\d[\\d_]*(?:\\.\\d+)?(?:[eE][+-]?\\d+)?)\\b)';
  const parts = [comment, string, number];
  if (profile.keywords) {
    parts.push(`(?<kw>\\b(?:${profile.keywords.trim().split(/\s+/).join('|')})\\b)`);
  }
  const re = new RegExp(parts.join('|'), profile.ignoreCase ? 'gi' : 'g');
  return code.replace(re, (...args) => {
    const g = args[args.length - 1];
    if (g.com) return `<span class="tok-com">${g.com}</span>`;
    if (g.str) return `<span class="tok-str">${g.str}</span>`;
    if (g.num) return `<span class="tok-num">${g.num}</span>`;
    return `<span class="tok-kw">${g.kw}</span>`;
  });
};

const renderCodeBlock = (code, lang) => (
  `<figure class="codeblock">` +
  `<figcaption><span class="codeblock-lang">${lang || 'code'}</span>` +
  `<button type="button" class="code-copy" aria-label="Copy code">Copy</button></figcaption>` +
  `<pre><code${lang ? ` class="lang-${lang}"` : ''}>${highlightCode(code, lang)}</code></pre></figure>`
);

// A URL we are willing to put in an href. The regexes below already require
// an http(s) scheme, so javascript: never matches — this adds the checks a
// pattern can't express: no quotes/brackets/backticks/whitespace, no control
// characters, and it must actually parse as a URL. (Merged from the web-UI
// polish branch, which centralized this more rigorously than the inline
// character classes did.)
const safeLinkHref = (raw) => {
  const url = String(raw || '').trim();
  if (!/^https?:\/\//i.test(url)) return null;
  if (/["'`<>\\\s]/.test(url)) return null;
  if (/[\u0000-\u001f\u007f]/.test(url)) return null;
  try {
    const parsed = new URL(url);
    if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') return null;
  } catch {
    return null;
  }
  return url;
};

// Inline formatting; operates on escaped text after code has been stashed.
// A URL failing safeLinkHref renders as plain text rather than a link.
const applyInlineMarkdown = (text) => text
  .replace(/!?\[([^\]\n]+)\]\((https?:\/\/[^)\s"']+)\)/g, (m, label, url) => {
    const safe = safeLinkHref(url);
    return safe ? `<a href="${safe}" target="_blank" rel="noopener noreferrer">${label}</a>` : m;
  })
  .replace(/(^|[\s(])(https?:\/\/[^\s"'<>]+[^\s"'<>.,;:!?)\]])/gm, (m, lead, url) => {
    const safe = safeLinkHref(url);
    return safe ? `${lead}<a href="${safe}" target="_blank" rel="noopener noreferrer">${safe}</a>` : m;
  })
  .replace(/\*\*\*([^*\n]+)\*\*\*/g, '<strong><em>$1</em></strong>')
  .replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>')
  .replace(/(^|[\s(])\*([^*\n]+)\*/gm, '$1<em>$2</em>')
  .replace(/__([^_\n]+)__/g, '<strong>$1</strong>')
  .replace(/(^|[\s(])_([^_\n]+)_/gm, '$1<em>$2</em>')
  .replace(/~~([^~\n]+)~~/g, '<del>$1</del>');

const MD_LIST_ITEM = /^(\s*)(?:([-*+])|(\d{1,9})[.)])\s+(.*)$/;
const MD_HR = /^ {0,3}([-*_])( *\1){2,} *$/;
const MD_TABLE_SEP = (l) => /^\s*\|?[\s:|-]+\|?\s*$/.test(l || '') && (l || '').includes('-');
const mdCells = (l) => l.replace(/^\s*\|/, '').replace(/\|\s*$/, '').split('|').map((c) => c.trim());

// Nested lists: items at the shallowest indent are siblings; deeper lines
// belong to the item above them and are parsed recursively.
const renderMdList = (lines, parseBlocks) => {
  const first = lines[0].match(MD_LIST_ITEM);
  const base = first[1].length;
  const ordered = first[3] !== undefined;
  const start = ordered ? parseInt(first[3], 10) : 1;
  const items = [];
  let cur = null;
  for (const line of lines) {
    const m = line.match(MD_LIST_ITEM);
    if (m && m[1].length <= base) {
      if (cur) items.push(cur);
      cur = { text: m[4], sub: [] };
    } else if (cur) {
      cur.sub.push(line);
    }
  }
  if (cur) items.push(cur);
  const body = items.map(({ text, sub }) => {
    let attrs = '';
    const task = text.match(/^\[([ xX])\]\s+(.*)$/);
    if (task) {
      attrs = ' class="task"';
      text = `<input type="checkbox" disabled${task[1] === ' ' ? '' : ' checked'}> ${task[2]}`;
    }
    return `<li${attrs}>${text}${sub.length ? parseBlocks(sub) : ''}</li>`;
  }).join('');
  return ordered
    ? `<ol${start !== 1 ? ` start="${start}"` : ''}>${body}</ol>`
    : `<ul>${body}</ul>`;
};

const renderMarkdown = (raw, opts = {}) => {
  // Strip NULs so message text can never reference the placeholder table.
  let src = String(raw || '').split(MD_PH).join('');
  // While streaming, close a dangling fence so partial code renders as code.
  if (opts.stream && ((src.match(/^\s*```/gm) || []).length % 2) === 1) src += '\n```';
  let text = escapeHtml(src);

  // Stash code and escaped punctuation so no later transform touches them.
  // 'B' placeholders are block-level (kept out of <p>), 'I' are inline.
  const slots = [];
  const stash = (html, block) => MD_PH + (block ? 'B' : 'I') + (slots.push(html) - 1) + MD_PH;

  text = text.replace(/```([^\n`]*)\n?([\s\S]*?)```/g, (m, info, code) => {
    const lang = (info.trim().split(/\s+/)[0] || '').toLowerCase().replace(/[^a-z0-9+#-]/g, '');
    return stash(renderCodeBlock(code.replace(/\n$/, ''), lang), true);
  });
  text = text.replace(/\\([\\`*_[\]()#+\-.!~|])/g, (m, ch) => stash(ch));
  text = text.replace(/`([^`\n]+)`/g, (m, code) => stash(`<code>${code}</code>`));

  text = applyInlineMarkdown(text);

  const isBlockStart = (line, next) =>
    !line || !line.trim() ||
    /^(#{1,6})\s/.test(line) ||
    /^ {0,3}&gt;/.test(line) ||
    MD_LIST_ITEM.test(line) ||
    MD_HR.test(line) ||
    MD_SLOT_RE.test(line.trim()) ||
    (line.includes('|') && MD_TABLE_SEP(next));

  const parseBlocks = (lines) => {
    const out = [];
    let i = 0;
    while (i < lines.length) {
      const line = lines[i];
      if (!line.trim()) { i += 1; continue; }
      let m;
      if (MD_SLOT_RE.test(line.trim())) {
        out.push(line.trim());
        i += 1;
      } else if ((m = line.match(/^(#{1,6})\s+(.*?)\s*#*\s*$/))) {
        const level = m[1].length;
        out.push(`<h${level}>${m[2]}</h${level}>`);
        i += 1;
      } else if (MD_HR.test(line)) {
        out.push('<hr>');
        i += 1;
      } else if (/^ {0,3}&gt;/.test(line)) {
        const quote = [];
        while (i < lines.length && /^ {0,3}&gt;/.test(lines[i])) {
          quote.push(lines[i].replace(/^ {0,3}&gt; ?/, ''));
          i += 1;
        }
        out.push(`<blockquote>${parseBlocks(quote)}</blockquote>`);
      } else if (MD_LIST_ITEM.test(line)) {
        const block = [];
        while (
          i < lines.length &&
          (MD_LIST_ITEM.test(lines[i]) || /^\s+\S/.test(lines[i]) ||
            (!lines[i].trim() && MD_LIST_ITEM.test(lines[i + 1] || '')))
        ) {
          if (lines[i].trim()) block.push(lines[i]);
          i += 1;
        }
        out.push(renderMdList(block, parseBlocks));
      } else if (line.includes('|') && MD_TABLE_SEP(lines[i + 1])) {
        const header = mdCells(line);
        const aligns = mdCells(lines[i + 1]).map((c) =>
          c.startsWith(':') && c.endsWith(':') ? 'center' : c.endsWith(':') ? 'right' : c.startsWith(':') ? 'left' : ''
        );
        const attr = (j) => (aligns[j] ? ` style="text-align:${aligns[j]}"` : '');
        i += 2;
        const rows = [];
        while (i < lines.length && lines[i].includes('|') && lines[i].trim()) {
          const r = mdCells(lines[i]).slice(0, header.length);
          while (r.length < header.length) r.push('');
          rows.push(r);
          i += 1;
        }
        out.push(
          `<div class="table-wrap"><table><thead><tr>${header.map((h, j) => `<th${attr(j)}>${h}</th>`).join('')}</tr></thead>` +
          `<tbody>${rows.map((r) => `<tr>${r.map((c, j) => `<td${attr(j)}>${c}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`
        );
      } else {
        const para = [line];
        i += 1;
        while (i < lines.length && !isBlockStart(lines[i], lines[i + 1])) {
          para.push(lines[i]);
          i += 1;
        }
        out.push(`<p>${para.join('<br>')}</p>`);
      }
    }
    return out.join('');
  };

  return parseBlocks(text.split('\n'))
    .replace(MD_RESTORE_RE, (m, n) => slots[Number(n)]);
};
