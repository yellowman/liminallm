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

const documentStub = {
  createElement: () => {
    let text = '';
    return {
      set textContent(v) { text = String(v); },
      get innerHTML() { return text; },
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
