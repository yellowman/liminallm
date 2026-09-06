/**
 * The stored citation anchor, converted for the side that renders it.
 *
 * SPEC §2.2 makes the stored offset a count of Unicode code points, because
 * the producer counts them and this side does not: a JavaScript string index
 * counts UTF-16 code units, so every character outside the Basic Multilingual
 * Plane makes the two disagree by one more. An anchor placed by the naive
 * arithmetic lands inside a following word, or inside a surrogate pair.
 */

import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createContext, runInContext } from 'node:vm';
import test from 'node:test';
import assert from 'node:assert/strict';

const root = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
const source = readFileSync(join(root, 'frontend/common.js'), 'utf8');

// The same stub markdown.test.mjs uses, and for the same reason: `escapeHtml`
// escapes by writing textContent into a detached div and reading innerHTML
// back, so a stub that returned the text unchanged would make every escaping
// assertion in this file pass without escaping anything. A browser text node
// escapes `&`, `<` and `>` and leaves quotes alone - which is exactly why
// `escapeAttr` exists on top of it.
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
runInContext(source, ctx);
const { utf16Index, lastCodePoints } = runInContext(
  '({ utf16Index, lastCodePoints })', ctx,
);

//: One emoji, two code units, one code point. The answer a citation would be
//: anchored in.
const ANSWER = '\u{1F600} Alpha beta';

test('plain text needs no conversion', () => {
  assert.equal(utf16Index('Alpha beta', 5), 5);
  assert.equal('Alpha beta'.slice(0, utf16Index('Alpha beta', 5)), 'Alpha');
});

test('an anchor after an astral character lands after it', () => {
  // The stored anchor is 1: one code point consumed.
  const index = utf16Index(ANSWER, 1);
  assert.equal(index, 2);
  assert.equal(ANSWER.slice(0, index), '\u{1F600}');
  // What the naive reading would have produced instead.
  assert.equal(ANSWER.slice(0, 1), '\ud83d');
});

test('an anchor at the end of the text is the end of the text', () => {
  const points = Array.from(ANSWER).length;
  assert.equal(utf16Index(ANSWER, points), ANSWER.length);
});

test('an anchor past the end clamps rather than running off', () => {
  assert.equal(utf16Index(ANSWER, 999), ANSWER.length);
});

test('a zero, negative or missing anchor is the start', () => {
  for (const offset of [0, -1, undefined, null, NaN, 'x']) {
    assert.equal(utf16Index(ANSWER, offset), 0);
  }
});

test('an anchor that is not a position fails toward the start', () => {
  // Infinity is the case that separates the guard from the loop: `seen <
  // Infinity` holds for the whole string, so without the finiteness check
  // this returns the length instead of refusing.
  assert.equal(utf16Index(ANSWER, Infinity), 0);
  assert.equal(utf16Index(ANSWER, -Infinity), 0);
});

test('every anchor in a mixed string round-trips through code points', () => {
  const text = 'a\u{1F600}b\u{1F1EC}\u{1F1E7}céd';
  const points = Array.from(text);
  for (let n = 0; n <= points.length; n += 1) {
    assert.equal(text.slice(0, utf16Index(text, n)), points.slice(0, n).join(''));
  }
});

test('a tail never begins half way through a pair', () => {
  const text = 'ab\u{1F600}cd';
  assert.equal(lastCodePoints(text, 3), '\u{1F600}cd');
  // `slice(-3)` would take three code units and open on a lone low surrogate.
  assert.equal(text.slice(-3).charCodeAt(0), 0xde00);
  assert.equal(lastCodePoints(text, 99), text);
  assert.equal(lastCodePoints(null, 3), '');
});

//: The chip row, from the two audiences that render it.
const rowCtx = createContext({ URL, console, document: documentStub });
runInContext(
  ['frontend/common.js', 'frontend/markdown.js']
    .map((f) => readFileSync(join(root, f), 'utf8')).join('\n;\n'),
  rowCtx,
);
const { citationsRowHtml } = runInContext('({ citationsRowHtml })', rowCtx);

const cited = (segment) => ({
  content: 'Four hundred hours',
  content_struct: { segments: [{ type: 'citation', start: 18, end: 18, ...segment }] },
});

test('the chip is labelled with the title, never an identifier', () => {
  const html = citationsRowHtml(cited({
    locator: '', meta: { kind: 'file', title: 'manual.md' },
  }));
  assert.match(html, /manual\.md/);
  assert.match(html, /data-kind="file"/);
});

test('a web citation is drawn as one', () => {
  const html = citationsRowHtml(cited({
    locator: 'https://example.test/handbook',
    meta: { kind: 'web', title: 'Turbine handbook' },
  }));
  assert.match(html, /data-kind="web"/);
  assert.match(html, /Turbine handbook/);
});

test('the hover carries the words the citation follows', () => {
  const html = citationsRowHtml(cited({
    locator: '', meta: { kind: 'file', title: 'manual.md' },
  }));
  assert.match(html, /Four hundred hours/);
});

test('a share chip announces nothing it cannot do', () => {
  const html = citationsRowHtml(
    cited({ locator: '', meta: { kind: 'file', title: 'manual.md' } }),
    { interactive: false },
  );
  assert.doesNotMatch(html, /role="button"/);
  assert.doesNotMatch(html, /tabindex/);
  assert.doesNotMatch(html, /data-citation/);
  assert.match(html, /manual\.md/);
});

test('a message with no citations renders no row', () => {
  assert.equal(citationsRowHtml({ content: 'plain', content_struct: null }), '');
  assert.equal(citationsRowHtml({}), '');
});

test('a hostile title cannot close the attribute it sits in', () => {
  // A source's title is not the reader's text: a page title, a filename, a
  // note heading - all of them arrive from somewhere else.
  const html = citationsRowHtml(cited({
    locator: '', meta: { kind: 'file', title: '" onmouseover="alert(1)' },
  }));
  // Every quote in the title is an entity, so neither attribute it appears
  // in can be closed early.
  assert.match(html, /title="&quot; onmouseover=&quot;alert\(1\)/);
  assert.doesNotMatch(html, /title="" onmouseover/);
});

test('a hostile title cannot open a tag of its own', () => {
  const html = citationsRowHtml(cited({
    locator: '', meta: { kind: 'file', title: '<img src=x onerror=alert(1)>' },
  }));
  assert.doesNotMatch(html, /<img/);
  assert.match(html, /&lt;img src=x onerror=alert\(1\)&gt;/);
});
