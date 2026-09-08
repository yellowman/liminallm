"""The streamed scrub reads each character once, and still says what the
whole-string scrub says.

`tests/test_citation_stream.py` is the semantic corpus: a marker never becomes
observable, however the provider cuts it. This file is about the second claim
the reader now makes - that the work it does is proportional to the answer and
not to how finely the answer was chopped - and about the wider differential
that claim has to survive.

The reader used to scrub the whole canonical text again after every chunk. A
provider sending one character at a time therefore paid `1 + 2 + ... + N`, and
the module said so and accepted it. What replaced it is a chain of passes,
each an automaton over arriving characters, which is the same shape
`_scrub_text` has: remove every leftmost non-overlapping match, hand what
survives to the next pass, and stop when a pass removes nothing.

`scrub_positions` is still what the answer *is*. It is asked once, at
completion, for the finished text and the origin map, and the reader's
released text is checked against it there.
"""

from __future__ import annotations

import random

import pytest

from liminallm.service import citation_stream
from liminallm.service.citation_stream import (
    CanonicalCitationStream,
    ScrubbedTokenStream,
)
from liminallm.service.citations import scrub_positions

NONCE = "K7Q2ABCD"

#: A nonce that overlaps itself. Eight characters of the real alphabet, so it
#: is a nonce `mint_nonce` can draw: `SKSKSKSK` has borders at every even
#: length, which is what makes a matcher that restarts at the failing
#: character - rather than at the longest border - miss occurrences.
OVERLAPPING = "SKSKSKSK"

#: A nonce whose longest border is three characters. `mint_nonce` can draw it
#: - `A` and `B` are both in the alphabet - and it is what distinguishes a
#: matcher that falls back through the borders from one that restarts at the
#: character that failed. In `AAABAAABA` the occurrence begins at index 1,
#: inside the run the first candidate had already consumed, so a restart at
#: the failing character walks past it and never sees it at all.
BORDERED = "AABAAABA"

#: The other fold pair in the alphabet, to keep the Unicode cases in the
#: differential rather than only in the examples.
S_NONCE = "K7Q2SBCD"

KELVIN = "K"
LONG_S = "ſ"
DOTLESS_I = "ı"
DOTTED_I = "İ"


def _pieces(nonce):
    """Material the grammar can and cannot be built from, in fragments."""
    return [
        "The interval is 400 hours", ".", " ", "\t", "\n", "[cite:", nonce,
        nonce.lower(), "-1", "-12", "]", "[", ":", "cite", "CITE", "-",
        "[cite:OTHER-1]", "see [1]", "a[b]c", "  ", "-999", "999", "   ",
        nonce[:1], nonce[:3], nonce[:7], nonce[3:], f"[CITE:{nonce}-2]",
        f"[cite:{nonce}]", f" {nonce}-3", "attic bit tidier", "\t\t",
        KELVIN, LONG_S, DOTLESS_I, DOTTED_I, KELVIN + nonce[1:],
        f"[c{DOTTED_I}te:", f"[C{DOTLESS_I}TE:{nonce}-1]", nonce[:-1],
        "1234567890", "     ", f"{nonce}-", f"[cite:{nonce}-", "]]", "[[",
    ]


def _chunkings(text, rng):
    """One chunk, one character, fixed sizes, every split, and random cuts."""
    yield [text] if text else [""]
    yield list(text) or [""]
    for size in (2, 4, 8, 32):
        yield [text[i:i + size] for i in range(0, len(text), size)] or [""]
    if len(text) <= 24:
        for index in range(len(text) + 1):
            yield [text[:index], text[index:]]
    for _ in range(2):
        cuts = sorted(rng.sample(range(len(text) + 1),
                                 min(len(text), rng.randint(0, 5))))
        bounds = [0] + cuts + [len(text)]
        yield [text[a:b] for a, b in zip(bounds, bounds[1:]) if b > a] or [""]


def _agrees(nonce, text, chunks):
    """Run one stream and say how it disagreed with the oracle, or None."""
    expected, expected_origins = scrub_positions(text, nonce)
    reader = CanonicalCitationStream(nonce)
    out = []
    seen = ""
    for chunk in chunks:
        fresh = reader.push(chunk)
        out.append(fresh)
        seen += fresh
        if nonce.lower() in seen.lower():
            return f"the namespace was observable mid-stream: {seen!r}"
        if f"[cite:{nonce}".lower() in seen.lower():
            return f"a marker was observable mid-stream: {seen!r}"
    tail, origins = reader.finish()
    out.append(tail)
    public = "".join(out)
    if public != expected:
        return f"released {public!r}, oracle says {expected!r}"
    if origins != expected_origins:
        return "the origin map disagrees with the oracle"
    if reader.canonical != text:
        return "the canonical text drifted from what was pushed"
    if reader.released != public:
        return "released text and the pieces handed back disagree"
    if not reader.intact():
        return "the reader did not call itself intact"
    return None


class TestTheOracleIsTheAnswerOnAnythingAtAll:
    """The differential, widened: more material, more chunkings, more nonces.

    The failures this class exists for were never the shapes anyone lists.
    The one that produced the pass chain is named below; it was found here.
    """

    @pytest.mark.parametrize("nonce", [NONCE, OVERLAPPING, BORDERED, S_NONCE])
    def test_generated_texts_under_every_chunking(self, nonce):
        rng = random.Random(20260908)
        pieces = _pieces(nonce)
        for _ in range(220):
            text = "".join(rng.choice(pieces) for _ in range(rng.randint(1, 6)))
            for chunks in _chunkings(text, rng):
                problem = _agrees(nonce, text, chunks)
                assert problem is None, (nonce, text, chunks, problem)

    @pytest.mark.parametrize("nonce", [NONCE, OVERLAPPING, BORDERED])
    @pytest.mark.parametrize("family", [
        "long ordinary prose",
        "spaces",
        "tabs",
        "digits",
        "nonce prefixes of every length",
        "cite prefixes of every length",
        "a very long possible handle",
        "adjacent markers",
        "many removals",
        "cascading splices",
        "case variants",
        "grammar lookalikes",
    ])
    def test_the_adversarial_families(self, nonce, family):
        """Named rather than drawn, because a random corpus finds these only
        when the seed is kind."""
        text = _family_text(family, nonce)
        rng = random.Random(7)
        for chunks in _chunkings(text, rng):
            problem = _agrees(nonce, text, chunks)
            assert problem is None, (family, chunks[:4], problem)


def _family_text(family, nonce):
    lower = nonce.lower()
    if family == "long ordinary prose":
        return "The service interval is four hundred hours. " * 6
    if family == "spaces":
        return "answer" + " " * 300 + "end"
    if family == "tabs":
        return "answer" + "\t" * 300 + "end"
    if family == "digits":
        return f"{nonce}-" + "9" * 300 + " end"
    if family == "nonce prefixes of every length":
        return "".join(nonce[:k] + "x" for k in range(len(nonce) + 1)) * 3
    if family == "cite prefixes of every length":
        return "".join("[cite:"[:k] + "x" for k in range(7)) * 3
    if family == "a very long possible handle":
        return f"[cite:{nonce}-" + "1" * 200
    if family == "adjacent markers":
        return f"[cite:{nonce}-1][cite:{nonce}-2][cite:{nonce}]" * 3
    if family == "many removals":
        return f"a {nonce} b {nonce}-1 c [cite:{nonce}-2] d " * 4
    if family == "cascading splices":
        # Each removal splices its neighbours into the next occurrence.
        return (nonce[:4] + lower + nonce[4:]) * 3
    if family == "case variants":
        return (
            f"{lower} {nonce.upper()} [CITE:{lower}-1] "
            f"[c{DOTTED_I}te:{nonce}] [c{DOTLESS_I}te:{lower}-2] "
            + (KELVIN + nonce[1:] if nonce[0] == "K" else LONG_S + nonce[1:])
        )
    if family == "grammar lookalikes":
        return "[cite:] [cite:OTHER-1] see [1] a[b]c -999 999 : - [[]] cite"
    raise AssertionError(family)


class TestAPassIsWhereASpliceIsReconsidered:
    """The regression the pass chain exists for.

    `_scrub_text` removes every leftmost non-overlapping match and *then*
    repeats. Its scan therefore passes over the junction a removal makes: the
    text after the match is matched on its own terms first, and only the next
    pass sees the two halves as neighbours.

    A reader that treated the junction as available immediately - resuming
    the matcher with the state from before the removal - found an earlier
    occurrence overlapping the one the oracle takes, and released the wrong
    two characters. It survived every example in the corpus and was caught by
    the differential on a self-overlapping nonce.
    """

    def test_a_junction_is_not_matched_before_the_pass_that_owns_it(self):
        text = "SK SKSKSKSK-3sksksksk"
        expected, _origins = scrub_positions(text, OVERLAPPING)
        assert expected == "SK", expected
        for chunks in ([text], list(text), ["SK SKSKSKSK-3", "sksksksk"]):
            problem = _agrees(OVERLAPPING, text, chunks)
            assert problem is None, (chunks, problem)

    def test_an_occurrence_beginning_inside_a_failed_candidate_is_found(self):
        """The matcher falls back through the nonce's borders.

        In `AAABAAABA` with `AABAAABA`, the occurrence begins at index 1 -
        inside the run the first candidate had already consumed. A matcher
        that restarts at the character that failed begins at index 3 and
        never sees it, so a nonce the model wrote survives into the answer.
        Both letters are in the nonce alphabet, so this is a nonce a turn can
        actually be given.
        """
        text = "AAABAAABA"
        expected, _origins = scrub_positions(text, BORDERED)
        assert expected == "A", expected
        for chunks in ([text], list(text), ["AAAB", "AAABA"], ["AAABAAAB", "A"]):
            problem = _agrees(BORDERED, text, chunks)
            assert problem is None, (chunks, problem)

    def test_the_splice_that_does_span_a_pass_still_spans_it(self):
        """The other half of the rule, and the reason a chain is needed at
        all: what one pass leaves adjacent, the next pass matches."""
        text = "K7Q2K7Q2k7q2abcdABCD"
        expected, _origins = scrub_positions(text, NONCE)
        assert expected == "K7Q2", expected
        for chunks in ([text], list(text), ["K7Q2K7Q2k7q2ab", "cdABCD"]):
            problem = _agrees(NONCE, text, chunks)
            assert problem is None, (chunks, problem)


# ---------------------------------------------------------------------------
# The complexity gate
# ---------------------------------------------------------------------------

#: Families whose cost must be linear in the characters received. Each is a
#: shape that the previous reader, or a naive frontier, pays quadratically
#: for: a run it would rescan per character, or a hold that never settles.
LINEAR_FAMILIES = {
    "prose": lambda n: ("The service interval is four hundred hours. "
                        * (n // 44 + 1))[:n],
    "spaces": lambda n: " " * n,
    "digits": lambda n: ("-" + "9" * (n - 1))[:n],
    "partials": lambda n: ("K7Q2" * (n // 4 + 1))[:n],
    "brackets": lambda n: ("[cite:" * (n // 6 + 1))[:n],
    "markers": lambda n: (f"ok [cite:{NONCE}-12] " * (n // 21 + 1))[:n],
    "splices": lambda n: (("K7Q2" + NONCE.lower() + "ABCD")
                          * (n // 16 + 1))[:n],
    "prefixes": lambda n: ("".join(NONCE[:k] + "x" for k in range(9))
                           * (n // 44 + 1))[:n],
}

#: Measured worst case is 4.95 characters of work per character received, over
#: every family and size below. The bound is loose enough not to be a
#: tripwire for an ordinary change and tight enough that a rescan of anything
#: growing breaks it: a reader that rescanned the unresolved run would be at
#: N/2 per character on `spaces` alone.
WORK_PER_CHARACTER = 12
WORK_CONSTANT = 512


def _work(text, chunk_size, nonce=NONCE):
    """Characters of variable-length work in the incremental path.

    Read before `finish`, because the gate is about the streaming half: the
    whole-string scrub at completion is allowed, once.
    """
    reader = CanonicalCitationStream(nonce)
    if chunk_size:
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    else:
        chunks = [text]
    for chunk in chunks:
        reader.push(chunk)
    return reader._work


class TestTheWorkIsLinearInTheAnswer:
    @pytest.mark.parametrize("family", sorted(LINEAR_FAMILIES))
    def test_character_at_a_time_stays_under_a_constant_factor(self, family):
        build = LINEAR_FAMILIES[family]
        for size in (1000, 2000, 4000, 8000, 16000):
            work = _work(build(size), 1)
            assert work <= WORK_PER_CHARACTER * size + WORK_CONSTANT, (
                f"{family} at N={size} did {work} characters of work"
            )

    @pytest.mark.parametrize("family", sorted(LINEAR_FAMILIES))
    def test_doubling_the_answer_doubles_the_work(self, family):
        """The shape, not the constant. Quadratic work would quadruple."""
        build = LINEAR_FAMILIES[family]
        for size in (2000, 4000, 8000):
            small = _work(build(size), 1)
            large = _work(build(size * 2), 1)
            ratio = large / max(small, 1)
            assert 1.6 <= ratio <= 2.6, (
                f"{family}: N={size} did {small}, N={size * 2} did {large} "
                f"(ratio {ratio:.2f})"
            )

    @pytest.mark.parametrize("family", sorted(LINEAR_FAMILIES))
    def test_the_provider_cannot_choose_the_cost(self, family):
        """The whole claim, as one comparison: the same answer costs the same
        whether it arrives in one chunk or sixteen thousand."""
        text = LINEAR_FAMILIES[family](8000)
        counts = {size: _work(text, size) for size in (1, 2, 4, 8, 32, 0)}
        assert len(set(counts.values())) == 1, counts

    def test_a_long_held_run_is_held_rather_than_rescanned(self):
        """The specific gate: an unresolved suffix that grows with the answer.

        A run of spaces is a run a marker could still claim, so none of it can
        be released until the answer ends. Holding it is correct; rescanning
        it per character is what made the old reader quadratic.
        """
        for size in (2000, 8000, 32000):
            reader = CanonicalCitationStream(NONCE)
            for character in " " * size:
                assert reader.push(character) == ""
            assert reader._work <= 2 * size + WORK_CONSTANT
            tail, _origins = reader.finish()
            assert tail == " " * size
            assert reader.intact()


class TestTheWholeStringScrubIsAskedOnce:
    """`scrub_positions` is the oracle, and an oracle asked per chunk is the
    cost this tranche removed."""

    @pytest.fixture()
    def counted(self, monkeypatch):
        calls = []
        real = citation_stream.scrub_positions

        def counting(text, nonce):
            calls.append(len(text))
            return real(text, nonce)

        monkeypatch.setattr(citation_stream, "scrub_positions", counting)
        return calls

    def test_pushing_never_asks_it(self, counted):
        reader = CanonicalCitationStream(NONCE)
        for character in f"400 hours [cite:{NONCE}-1] exactly, and more prose":
            reader.push(character)
        assert counted == [], f"the oracle was asked {len(counted)} times"

    def test_completion_asks_it_once(self, counted):
        reader = CanonicalCitationStream(NONCE)
        reader.push(f"400 hours [cite:{NONCE}-1] exactly")
        reader.finish()
        assert len(counted) == 1, counted

    def test_asking_whether_it_is_intact_again_asks_nobody(self, counted):
        reader = CanonicalCitationStream(NONCE)
        reader.push(f"400 hours [cite:{NONCE}-1]")
        reader.finish()
        before = len(counted)
        assert reader.intact()
        assert reader.intact()
        assert reader.intact()
        assert len(counted) == before, "intact() scrubbed the answer again"


class TestTheWrapperDoesNotRebuildTheAnswerPerToken:
    """The ceiling check ran per token and asked for `len(canonical)`, which
    joined every chunk received so far to count them."""

    @staticmethod
    def _events(chunks, content):
        out = [{"event": "token", "data": chunk} for chunk in chunks]
        out.append({"event": "message_done", "data": {"content": content}})
        return out

    def test_the_ceiling_check_counts_without_joining(self, monkeypatch):
        joins = []
        original = CanonicalCitationStream.canonical

        def counting(self):
            # A real materialization, not an ask: the completion asks twice -
            # the provider's own content is compared against it, and the
            # scrub reads it - and the second ask is answered from the first.
            if self._canonical_text is None:
                joins.append(self.canonical_length)
            return original.fget(self)

        monkeypatch.setattr(
            CanonicalCitationStream, "canonical", property(counting)
        )
        chunks = ["word "] * 200
        stream = ScrubbedTokenStream(
            self._events(chunks, "".join(chunks)), NONCE
        )
        events = list(stream)
        assert events[-1]["event"] == "message_done"
        # Once, for the provider's own final content. Not once per token.
        assert len(joins) <= 1, f"the answer was materialized {len(joins)} times"

    def test_the_ceiling_still_fires(self):
        chunks = ["x" * 40] * 10
        stream = ScrubbedTokenStream(
            self._events(chunks, "".join(chunks)), NONCE, max_canonical_chars=100
        )
        with pytest.raises(citation_stream.CanonicalStreamTooLong):
            list(stream)
        assert not stream.reader.intact()
        assert stream.reader.canonical_length <= 100
