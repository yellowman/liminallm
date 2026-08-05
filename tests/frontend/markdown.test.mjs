/**
 * The escape-first markdown renderer, run as the browser runs it: common.js
 * then markdown.js in one shared scope. renderMarkdown feeds innerHTML on
 * every assistant message and on the public share page, so its contract is a
 * security boundary: model output must never become live markup.
 */

import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createContext, runInContext } from 'node:vm';
import test from 'node:test';
import assert from 'node:assert/strict';

const root = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
const source = ['frontend/common.js', 'frontend/markdown.js']
  .map((f) => readFileSync(join(root, f), 'utf8'))
  .join('\n;\n');

// escapeHtml escapes by writing textContent into a detached div and reading
// innerHTML back. This stub reproduces exactly what a browser text node
// escapes: & < > — and nothing else (quotes stay, as the comment in chat.js
// relies on).
const documentStub = {
  createElement: () => {
    let text = '';
    return {
      set textContent(v) { text = String(v); },
      get innerHTML() {
        return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
      },
    };
  },
};

const ctx = createContext({ URL, console, document: documentStub });
runInContext(source + '\n;({ renderMarkdown, safeLinkHref, highlightCode, escapeHtml })', ctx);
const { renderMarkdown, safeLinkHref } = runInContext(
  '({ renderMarkdown, safeLinkHref, highlightCode, escapeHtml })', ctx,
);

// ---------------------------------------------------------------------------
// Escape-first: the security contract
// ---------------------------------------------------------------------------

test('a script tag in a message renders as text, not markup', () => {
  const out = renderMarkdown('hello <script>alert(1)</script>');
  assert.ok(!out.includes('<script>'));
  assert.ok(out.includes('&lt;script&gt;'));
});

test('an img onerror payload is inert', () => {
  const out = renderMarkdown('<img src=x onerror=alert(1)>');
  assert.ok(!out.includes('<img'));
});

test('html smuggled inside a code fence stays escaped', () => {
  const out = renderMarkdown('```\n<script>alert(1)</script>\n```');
  assert.ok(!out.includes('<script>alert'));
  assert.ok(out.includes('&lt;script&gt;'));
});

test('markdown link labels cannot carry html', () => {
  const out = renderMarkdown('[<b>bold</b>](https://example.com)');
  assert.ok(!out.includes('<b>'));
});

test('a javascript: link renders as plain text, never an anchor', () => {
  const out = renderMarkdown('[click](javascript:alert(1))');
  assert.ok(!out.includes('href="javascript'));
});

test('only the fixed tag set ever appears in output', () => {
  const nasty = [
    '# h\n**b** *i* `c`\n', '> q\n', '- a\n- b\n', '1. a\n', '| a | b |\n|---|---|\n| 1 | 2 |\n',
    '```js\nconst x = "<svg onload=alert(1)>";\n```\n', '[x](https://e.com) ![y](https://e.com/i.png)',
    '<iframe src=x></iframe>', '<style>*{}</style>', 'text\n\n---\n\nmore',
  ].join('\n');
  const out = renderMarkdown(nasty);
  const tags = [...out.matchAll(/<\/?([a-zA-Z][a-zA-Z0-9-]*)/g)].map((m) => m[1].toLowerCase());
  const allowed = new Set(['p', 'br', 'h1', 'h2', 'h3', 'h4', 'strong', 'em', 'del', 'code',
    'pre', 'blockquote', 'ul', 'ol', 'li', 'hr', 'a', 'img', 'table', 'thead', 'tbody',
    'tr', 'th', 'td', 'span', 'div', 'button', 'svg', 'rect', 'path', 'polyline',
    'figure', 'figcaption']);
  for (const t of tags) assert.ok(allowed.has(t), `unexpected tag <${t}> in output`);
});

// ---------------------------------------------------------------------------
// safeLinkHref: what may become an anchor
// ---------------------------------------------------------------------------

test('http and https survive; everything else is refused', () => {
  assert.equal(safeLinkHref('https://example.com/a?b=1'), 'https://example.com/a?b=1');
  assert.equal(safeLinkHref('http://example.com'), 'http://example.com');
  for (const bad of ['javascript:alert(1)', 'data:text/html,x', 'ftp://x', 'file:///etc/passwd',
    'https://e.com/"onmouseover=alert(1)', 'https://e.com/<x>', 'https://e.com/a b', '', null]) {
    assert.equal(safeLinkHref(bad), null, `accepted: ${bad}`);
  }
});

// ---------------------------------------------------------------------------
// The GitHub-flavored subset renders
// ---------------------------------------------------------------------------

test('headings, emphasis, and inline code', () => {
  assert.match(renderMarkdown('# Title'), /<h1>Title<\/h1>/);
  assert.match(renderMarkdown('**bold** and *italic*'), /<strong>bold<\/strong>/);
  assert.match(renderMarkdown('`x = 1`'), /<code>x = 1<\/code>/);
});

test('fenced code keeps its language and its verbatim body', () => {
  const out = renderMarkdown('```python\ndef f():\n    return 1\n```');
  assert.match(out, /<pre/);
  assert.ok(out.includes('def'));
  assert.ok(out.includes('return'));
});

test('lists nest and ordered lists count', () => {
  const out = renderMarkdown('1. one\n2. two\n   - nested\n');
  assert.match(out, /<ol>/);
  assert.match(out, /<ul>/);
  assert.ok(out.includes('nested'));
});

test('tables become tables', () => {
  const out = renderMarkdown('| a | b |\n|---|---|\n| 1 | 2 |');
  assert.match(out, /<table/);
  assert.match(out, /<th>a<\/th>/);
  assert.match(out, /<td>1<\/td>/);
});

test('blockquotes and rules', () => {
  assert.match(renderMarkdown('> quoted'), /<blockquote>/);
  assert.match(renderMarkdown('a\n\n---\n\nb'), /<hr>/);
});

test('plain text is a paragraph with soft breaks', () => {
  assert.equal(renderMarkdown('line one\nline two'), '<p>line one<br>line two</p>');
});

test('empty input renders to nothing harmful', () => {
  assert.equal(renderMarkdown(''), '');
  assert.equal(renderMarkdown(null), '');
});

// ---------------------------------------------------------------------------
// Streaming mode: a dangling fence must still render as code
// ---------------------------------------------------------------------------

test('an unclosed fence in stream mode renders as code, not as text', () => {
  const out = renderMarkdown('```js\nconst x = 1;', { stream: true });
  assert.match(out, /<pre/);
  assert.ok(out.includes('const'));
});

// ---------------------------------------------------------------------------
// Highlighting never introduces markup from the payload
// ---------------------------------------------------------------------------

test('code highlighting output only ever adds tok spans', () => {
  const out = renderMarkdown('```js\nconst s = "<script>x</script>"; // <b>note</b>\n```');
  assert.ok(!out.includes('<script>'));
  assert.ok(!out.includes('<b>'));
  assert.match(out, /tok-(kw|str|com)/);
});
