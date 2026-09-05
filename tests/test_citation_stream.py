"""A streamed answer loses its citation namespace before anyone sees it.

The whole-string scrub already has witnesses. What is under test here is that
the same removal survives being performed a chunk at a time - that a marker cut
in half by a provider's tokenizer is still a marker, and that text released
early is never text the finished scrub would have taken out.
"""

from __future__ import annotations

import random

import pytest

from liminallm.service.citation_stream import CanonicalCitationStream
from liminallm.service.citations import scrub_positions

NONCE = "K7Q2ABCD"
MARKER = f"[cite:{NONCE}-1]"

#: A nonce holding the other letter Python folds a non-ASCII character into.
#: `K` and `S` are both in the real nonce alphabet, so neither of these is a
#: contrived input.
S_NONCE = "K7Q2SBCD"

#: Characters `re.IGNORECASE` treats as the ASCII letter beside them and
#: `.lower() + .upper()` does not. Named rather than pasted, because a reader
#: comparing the two spellings by eye would see the same letter twice.
KELVIN = "\u212a"       # KELVIN SIGN, folds to K
LONG_S = "\u017f"       # LATIN SMALL LETTER LONG S, folds to S
DOTLESS_I = "\u0131"    # LATIN SMALL LETTER DOTLESS I, folds to I
DOTTED_I = "\u0130"     # LATIN CAPITAL LETTER I WITH DOT ABOVE, folds to I


def _read(nonce, chunks):
    """Run a stream and return (released, origins)."""
    reader = CanonicalCitationStream(nonce)
    out = [reader.push(chunk) for chunk in chunks]
    tail, origins = reader.finish()
    out.append(tail)
    assert "".join(out) == reader.released
    return reader, "".join(out), origins


def _every_split(text):
    """Every two-way cut of `text`, plus the character-by-character one."""
    for index in range(len(text) + 1):
        yield [text[:index], text[index:]]
    yield list(text)


class TestAMarkerNeverBecomesObservable:
    def test_a_marker_inside_one_chunk(self):
        _reader, public, _origins = _read(NONCE, [f"400 hours {MARKER} exactly"])
        assert public == "400 hours exactly"

    @pytest.mark.parametrize("text", [
        f"400 hours {MARKER} exactly",
        f"{MARKER} leads",
        f"trails {MARKER}",
        f"a {MARKER}{MARKER} b",
        f"a {NONCE} b",
        f"a {NONCE}-12 b",
        f"[cite:{NONCE}]",
        f"[CITE:{NONCE}-2]",
        f"a {NONCE.lower()} b",
    ])
    def test_every_split_agrees_with_the_finished_scrub(self, text):
        """The split is the whole problem: a provider may cut anywhere, and
        the same text must scrub to the same thing however it arrives."""
        expected, expected_origins = scrub_positions(text, NONCE)
        for chunks in _every_split(text):
            _reader, public, origins = _read(NONCE, chunks)
            assert public == expected, (chunks, public, expected)
            assert origins == expected_origins, chunks

    @pytest.mark.parametrize("text", [
        f"400 hours {MARKER} exactly",
        f"a {NONCE} b",
        f"[cite:{NONCE}-99]",
        f"  {NONCE}-3  ",
    ])
    def test_no_split_ever_lets_the_namespace_out(self, text):
        """Not "it is gone by the end" - never present. A reader that renders
        as it receives has already shown whatever reached it, so a marker
        removed at `message_done` was still a marker on someone's screen."""
        for chunks in _every_split(text):
            reader = CanonicalCitationStream(NONCE)
            seen = ""
            for chunk in chunks:
                seen += reader.push(chunk)
                assert NONCE.lower() not in seen.lower(), (chunks, seen)
                assert f"[cite:{NONCE}".lower() not in seen.lower(), (chunks, seen)

    def test_text_released_before_a_removal_cannot_be_spliced_into_a_nonce(self):
        """The second defect this reader had, pinned by name.

        `K7Q2` `K7Q2` `k7q2ab` has no complete occurrence in it, and the last
        six characters are the only ones a marker could be growing from. So a
        hold that asked only "is the tail a partial occurrence" released the
        first eight - and then `cdABCD` arrived, the middle became a nonce and
        vanished, and the two halves left behind spliced into another one.

        It lived in the random corpus for exactly as long as the seed happened
        to draw it. Named here because a defect that has been seen once should
        not depend on a lottery.
        """
        text = "K7Q2K7Q2k7q2abcdABCD"
        expected, _origins = scrub_positions(text, NONCE)
        assert expected == "K7Q2", expected
        for chunks in [["K7Q2K7Q2k7q2ab", "cdABCD"], list(text),
                       ["K7Q2K7Q2k7q2", "abcd", "ABCD"]]:
            _reader, public, _o = _read(NONCE, chunks)
            assert public == expected, chunks

    def test_a_stream_cut_off_mid_marker_emits_none_of_it(self):
        """Cancellation. `finish` is what releases the hold, so an answer that
        stops halfway through a marker has emitted no part of it - and has no
        finished text to read a citation out of either."""
        reader = CanonicalCitationStream(NONCE)
        released = "".join(
            reader.push(chunk) for chunk in ["400 hours ", "[cite:", NONCE, "-"]
        )
        assert released == "400 hours"
        assert NONCE not in released
        # The canonical side still has everything, which is what makes the
        # difference between "not shown" and "not received" legible.
        assert reader.canonical == f"400 hours [cite:{NONCE}-"

    def test_ordinary_text_is_not_held_hostage(self):
        """The hold is for what a marker could still claim, not a buffer. A
        sentence with nothing citation-shaped in it streams as it arrives."""
        prose = "The service interval is four hundred hours."
        reader = CanonicalCitationStream(NONCE)
        released = "".join(reader.push(character) for character in prose)
        assert released == prose


class TestTheHoldAsksTheSameEngineTheScrubAsks:
    """Case-insensitivity is the regex engine's question, not `str`'s.

    `re.IGNORECASE` folds characters into the namespace alphabet that
    `.lower() + .upper()` does not, and two of them - KELVIN SIGN for `K`,
    LONG S for `S` - fold into letters the real nonce alphabet contains. A
    hold that decided which characters could be part of a marker by building
    a set walked straight past them, released half a nonce, and had the other
    half delete it.

    These are the whole namespace to the finished scrub, so they must be the
    whole namespace to the reader at every split.
    """

    @pytest.mark.parametrize("nonce,text", [
        (NONCE, KELVIN + "7Q2ABCD"),
        (NONCE, "before " + KELVIN + "7Q2ABCD after"),
        (NONCE, f"[c{DOTTED_I}te:{NONCE}-1]"),
        (NONCE, f"[c{DOTLESS_I}te:{NONCE}-1]"),
        (NONCE, f"a [C{DOTLESS_I}TE:{NONCE}-2] b"),
        (S_NONCE, "K7Q2" + LONG_S + "BCD"),
        (S_NONCE, f"held [cite:K7Q2{LONG_S}BCD-1] here"),
        (S_NONCE, "K" + KELVIN.upper() + "7Q2" + LONG_S + "BCD"),
    ])
    def test_every_split_still_agrees_with_the_finished_scrub(self, nonce, text):
        expected, expected_origins = scrub_positions(text, nonce)
        assert expected != text, "the fixture is not a namespace occurrence"
        for chunks in _every_split(text):
            _reader, public, origins = _read(nonce, chunks)
            assert public == expected, (chunks, public, expected)
            assert origins == expected_origins, chunks

    @pytest.mark.parametrize("nonce,text", [
        (NONCE, KELVIN + "7Q2ABCD"),
        (NONCE, f"[c{DOTTED_I}te:{NONCE}-1]"),
        (S_NONCE, "K7Q2" + LONG_S + "BCD"),
    ])
    def test_no_split_lets_a_folded_half_out(self, nonce, text):
        """The failure was not that the marker survived. It was that half of
        it was released and the rest of the answer then contradicted it."""
        for chunks in _every_split(text):
            reader = CanonicalCitationStream(nonce)
            seen = ""
            for chunk in chunks:
                seen += reader.push(chunk)
            tail, _origins = reader.finish()
            seen += tail
            assert seen == scrub_positions(text, nonce)[0], chunks
            assert reader.intact()


class TestWhatIsNotThisTurnsNamespaceIsLeftAlone:
    @pytest.mark.parametrize("text", [
        "see [1] and [2]",
        "index a[b]c",
        "[cite:OLDTURN-1] was another turn",
        "[cite:] is not a handle",
        "brackets [ and ] alone",
        "a colon: and a dash - and 999",
        f"[cite:{NONCE[:4]}NOPE-1] is not this turn",
    ])
    def test_it_survives_every_split(self, text):
        expected, _origins = scrub_positions(text, NONCE)
        assert expected == text, "the fixture is not testing what it says"
        for chunks in _every_split(text):
            _reader, public, _o = _read(NONCE, chunks)
            assert public == text, chunks


class TestTheOriginMapIsTheOneCitationsAreReadWith:
    def test_offsets_match_the_finished_scrub(self):
        """`citation_payload` turns canonical positions into public ones with
        this map. A stream that produced its own would be a second
        implementation of the rule that decides where a citation points."""
        text = f"alpha {NONCE} beta {MARKER} gamma"
        expected, expected_origins = scrub_positions(text, NONCE)
        for chunks in _every_split(text):
            _reader, public, origins = _read(NONCE, chunks)
            assert public == expected
            assert origins == expected_origins, chunks

    def test_a_marker_after_a_bare_nonce_removal(self):
        """The second removal's offsets depend on the first having happened,
        which is the case a per-chunk map would get wrong."""
        text = f"one {NONCE} two {MARKER} three"
        expected, expected_origins = scrub_positions(text, NONCE)
        _reader, public, origins = _read(NONCE, list(text))
        assert public == expected == "one two three"
        assert origins == expected_origins
        # Every surviving character really came from where the map says.
        assert "".join(text[i] for i in origins) == public


class TestTheStreamAgreesWithTheFinishedScrubOnAnythingAtAll:
    """The contract as a differential rather than as examples.

    Chosen because the failures here were not the shapes anyone lists. Two
    were: an uppercase `[CITE:` whose keyword the hold's character walk did
    not know, and text released before a *later* removal spliced it into a
    nonce that neither half had been. Both were found by running this, not by
    reading the code.
    """

    PIECES = [
        "The interval is 400 hours", ".", " ", "\t", "\n", "[cite:", NONCE,
        NONCE.lower(), "-1", "-12", "]", "[", ":", "cite", "CITE",
        "[cite:OTHER-1]", "see [1]", "a[b]c", "  ", "-999", "999",
        NONCE[:1], NONCE[:3], NONCE[:7], NONCE[3:], f"[CITE:{NONCE}-2]",
        f"[cite:{NONCE}]", f" {NONCE}-3", "attic bit tidier",
        KELVIN, LONG_S, DOTLESS_I, DOTTED_I, KELVIN + "7Q2ABCD",
        f"[c{DOTTED_I}te:", f"[C{DOTLESS_I}TE:{NONCE}-1]",
    ]

    def test_random_texts_and_random_chunkings(self):
        rng = random.Random(20260905)
        for _ in range(1500):
            text = "".join(rng.choice(self.PIECES) for _ in range(rng.randint(1, 9)))
            cuts = sorted(rng.sample(
                range(len(text) + 1), min(len(text), rng.randint(0, 6))
            ))
            bounds = [0] + cuts + [len(text)]
            chunks = [text[a:b] for a, b in zip(bounds, bounds[1:]) if b > a]
            expected, expected_origins = scrub_positions(text, NONCE)
            reader, public, origins = _read(NONCE, chunks or [""])
            assert public == expected, (text, chunks)
            assert origins == expected_origins, (text, chunks)
            assert reader.canonical == text
            assert reader.intact()

    def test_character_by_character_is_the_worst_case_and_still_agrees(self):
        rng = random.Random(451)
        for _ in range(300):
            text = "".join(rng.choice(self.PIECES) for _ in range(rng.randint(1, 6)))
            expected, _origins = scrub_positions(text, NONCE)
            _reader, public, _o = _read(NONCE, list(text) or [""])
            assert public == expected, text


class TestTheStreamRefusesToBeUsedWrongly:
    def test_pushing_after_finishing_is_an_error(self):
        reader = CanonicalCitationStream(NONCE)
        reader.push("done")
        reader.finish()
        with pytest.raises(RuntimeError):
            reader.push(" more")

    def test_intact_is_the_contract_stated_as_a_comparison(self):
        reader = CanonicalCitationStream(NONCE)
        reader.push(f"400 hours {MARKER}")
        reader.finish()
        assert reader.intact()
        # Forcing the failure the check exists for: text released that the
        # finished scrub does not begin with.
        reader._released = "800 hours"
        assert not reader.intact()


class TestTheGuardsOnStateThatShouldNotHappen:
    """Three checks that the mechanism above is meant to make unreachable.

    Forced rather than provoked. Each one exists because the alternative to
    raising is streaming an answer whose public form is already wrong, and a
    guard nobody has ever seen fire is a guard nobody knows the behaviour of.
    """

    def test_released_text_that_the_scrub_later_removes_raises(self):
        reader = CanonicalCitationStream(NONCE)
        reader.push("400 hours ")
        # What a hold that was too short would have left behind.
        reader._released = f"400 hours [cite:{NONCE}"
        with pytest.raises(ValueError):
            reader.push("-1] exactly")

    def test_a_stream_that_has_not_finished_is_not_intact(self):
        """The gate this is about to sit in asks one question - may citations
        be read out of this answer - and an unfinished stream's answer is no.

        Ordinary prose holds nothing back, so its released text equals the
        scrub of what has arrived so far. Answering on the text alone would
        have told a cancelled turn that its public form was vouched for."""
        reader = CanonicalCitationStream(NONCE)
        reader.push("The service interval is four hundred hours.")
        assert reader.released == "The service interval is four hundred hours."
        assert not reader.intact()
        reader.finish()
        assert reader.intact()

    def test_intact_is_equality_and_not_containment(self):
        """A public form that merely contains what was released is an answer
        with text missing from the front of it."""
        reader = CanonicalCitationStream(NONCE)
        reader.push(f"400 hours {MARKER}")
        reader.finish()
        reader._released = "hours"
        assert "hours" in scrub_positions(reader.canonical, NONCE)[0]
        assert not reader.intact()
