"""A streamed answer loses its citation namespace before anyone sees it.

The whole-string scrub already has witnesses. What is under test here is that
the same removal survives being performed a chunk at a time - that a marker cut
in half by a provider's tokenizer is still a marker, and that text released
early is never text the finished scrub would have taken out.
"""

from __future__ import annotations

import asyncio
import random
import threading
from json import dumps as json_dumps

import pytest

from liminallm.service.citation_stream import (
    MAX_CANONICAL_CHARS,
    CanonicalCitationStream,
    CanonicalStreamTooLong,
    ScrubbedTokenStream,
)
from liminallm.service.citations import scrub_positions
from liminallm.service.node_attempt import StreamPump
from liminallm.service.tokenizer_utils import MAX_GENERATION_TOKENS

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


class TestTheFilterKeepsTheHandlesThePumpReachesFor:
    """`StreamPump` owns its iterator and reaches into it.

    On `stop` it looks for `abort`, because a cancellable backend can
    interrupt a read already in flight and the stop flag is only read between
    events; `cancellation_proven` reads `armed` to decide whether that death
    can be presumed prompt; and the producer thread calls `close` on the way
    out. A generator wrapping the provider would have none of the three, and
    the loss would be silent - a `timeout_ms` quietly back to waiting out the
    provider client's own timeout.
    """

    class _Provider:
        """A backend stream with the handles the pump uses."""

        def __init__(self, events, armed=True):
            self._events = iter(events)
            self._armed = armed
            self.aborted = False
            self.closed = False

        def __iter__(self):
            return self

        def __next__(self):
            if self.aborted:
                raise StopIteration
            return next(self._events)

        @property
        def armed(self):
            return self._armed

        def abort(self):
            self.aborted = True

        def close(self):
            self.closed = True

    def test_abort_reaches_the_backend(self):
        provider = self._Provider([{"event": "token", "data": "x"}])
        stream = ScrubbedTokenStream(provider, NONCE)
        stream.abort()
        assert provider.aborted

    def test_armed_is_the_backends_answer(self):
        assert ScrubbedTokenStream(self._Provider([], armed=True), NONCE).armed
        assert not ScrubbedTokenStream(self._Provider([], armed=False), NONCE).armed

    def test_close_reaches_the_backend(self):
        provider = self._Provider([])
        ScrubbedTokenStream(provider, NONCE).close()
        assert provider.closed

    def test_a_backend_without_the_handles_is_still_iterable(self):
        """Plain in-memory doubles carry none of them, and the pump already
        treats an unarmed producer as one whose death it will not presume."""
        stream = ScrubbedTokenStream(
            iter([{"event": "token", "data": "plain"}]), NONCE
        )
        assert not stream.armed
        stream.abort()
        stream.close()
        assert [event["data"] for event in stream] == ["plain"]


class TestWhatTheFilterLetsThrough:
    @staticmethod
    def _events(*chunks, content=None):
        out = [{"event": "token", "data": chunk} for chunk in chunks]
        data = {"usage": {}}
        if content is not None:
            data["content"] = content
        out.append({"event": "message_done", "data": data})
        return out

    def test_a_marker_split_across_tokens_never_leaves(self):
        raw = ["400 hours ", "[ci", "te:" + NONCE, "-1]", " exactly"]
        stream = ScrubbedTokenStream(self._events(*raw, content="".join(raw)), NONCE)
        events = list(stream)
        tokens = [e["data"] for e in events if e["event"] == "token"]
        assert "".join(tokens) == "400 hours exactly"
        assert NONCE not in "".join(tokens)
        done = events[-1]
        assert done["event"] == "message_done"
        assert done["data"]["content"] == "400 hours exactly"
        assert stream.reader.intact()

    def test_no_empty_token_is_forwarded(self):
        """A chunk wholly inside a marker produces nothing safe. Forwarding an
        empty token would read to a consumer counting events as the model
        having said something."""
        raw = ["[cite:", NONCE, "-1]", "done"]
        events = list(ScrubbedTokenStream(self._events(*raw, content="".join(raw)),
                                          NONCE))
        tokens = [e for e in events if e["event"] == "token"]
        assert all(e["data"] for e in tokens), tokens
        assert "".join(e["data"] for e in tokens) == "done"

    def test_the_held_tail_goes_before_the_completion(self):
        """A consumer that replaces its accumulated tokens with the final
        content has to be handed a final content those tokens add up to.

        Trailing spaces are the everyday case: `[ \t]*` in front of a marker
        is part of the match, so a run of them at the end of an answer is held
        until there is nothing left that could claim it."""
        raw = ["answer", "   "]
        events = list(ScrubbedTokenStream(self._events(*raw, content="".join(raw)),
                                          NONCE))
        assert [e["event"] for e in events] == ["token", "token", "message_done"]
        assert events[0]["data"] == "answer"
        assert events[1]["data"] == "   ", "the tail was not released"
        tokens = "".join(e["data"] for e in events if e["event"] == "token")
        assert tokens == events[-1]["data"]["content"] == "answer   "

    def test_an_unclosed_marker_is_not_this_turns_namespace(self):
        """`[cite:` with the handle taken out of it names nothing, and the
        scrub deliberately leaves everything that is not this namespace. The
        filter must not invent a stricter rule than the oracle it serves."""
        raw = ["answer", " [cite:" + NONCE]
        text = "".join(raw)
        stream = ScrubbedTokenStream(self._events(*raw, content=text), NONCE)
        events = list(stream)
        tokens = "".join(e["data"] for e in events if e["event"] == "token")
        assert tokens == scrub_positions(text, NONCE)[0] == "answer [cite:"
        assert NONCE not in tokens
        assert stream.reader.intact()

    def test_events_that_are_not_tokens_pass_through(self):
        error = {"event": "error", "data": {"code": "server_error"}}
        events = list(ScrubbedTokenStream([error], NONCE))
        assert events == [error]

    def test_a_provider_that_contradicts_its_own_tokens_is_refused(self):
        """Two claims about one answer. The tokens are what the client was
        shown, so a final content that disagrees is not a correction the
        parent may accept - and an answer nobody can vouch for carries no
        citations."""
        raw = ["400 hours ", MARKER]
        stream = ScrubbedTokenStream(
            self._events(*raw, content="800 hours " + MARKER), NONCE
        )
        events = list(stream)
        tokens = "".join(e["data"] for e in events if e["event"] == "token")
        assert tokens == "400 hours"
        assert "800" not in json_dumps(events)
        assert NONCE not in json_dumps(events)
        assert stream.contradicted
        assert not stream.reader.intact()

    def test_a_contradiction_is_not_a_completion(self):
        """Zero citations is not the whole of it.

        `message_done` is what the streamed node treats as the answer
        boundary, so a completion carrying correctly scrubbed but truncated
        text is a partial answer wearing a success stamp - ready to be
        returned as the turn's reply and persisted. There is no completion
        here at all; the turn ends the way a backend failure ends, with
        whatever the client was already shown.
        """
        raw = ["400 hours ", MARKER]
        stream = ScrubbedTokenStream(
            self._events(*raw, content="800 hours " + MARKER), NONCE
        )
        events = list(stream)
        kinds = [event["event"] for event in events]
        assert "message_done" not in kinds, events
        assert kinds[-1] == "error", events
        assert events[-1]["data"]["code"] == "server_error"

    def test_a_stream_with_no_reported_content_still_completes(self):
        raw = ["400 hours ", MARKER, " exactly"]
        stream = ScrubbedTokenStream(self._events(*raw), NONCE)
        events = list(stream)
        assert events[-1]["data"]["content"] == "400 hours exactly"
        assert stream.reader.intact()


class TestTheCeilingStopsRatherThanTruncates:
    def test_it_fires_and_takes_the_provider_with_it(self):
        """Truncating would hand the client a shorter answer that looks
        finished, when tokens from it have already been rendered."""
        provider = TestTheFilterKeepsTheHandlesThePumpReachesFor._Provider(
            [{"event": "token", "data": "x" * 40} for _ in range(10)]
        )
        stream = ScrubbedTokenStream(provider, NONCE, max_canonical_chars=100)
        with pytest.raises(CanonicalStreamTooLong):
            list(stream)
        assert provider.aborted
        assert not stream.reader.intact()
        assert len(stream.reader.canonical) <= 100

    def test_an_answer_inside_the_ceiling_is_untouched(self):
        events = [{"event": "token", "data": "fits"},
                  {"event": "message_done", "data": {"content": "fits"}}]
        stream = ScrubbedTokenStream(events, NONCE, max_canonical_chars=100)
        assert [e["data"] for e in stream if e["event"] == "token"] == ["fits"]
        assert stream.reader.intact()

    def test_the_default_is_the_length_the_system_calls_a_whole_reply(self):
        assert MAX_CANONICAL_CHARS == 4 * MAX_GENERATION_TOKENS


class TestTheRealPumpCanStillStopABlockedProvider:
    """The composition, not the delegation.

    The tests above prove the wrapper forwards `abort`, `armed` and `close`.
    This proves the thing that matters: a `StreamPump` owning the wrapper
    owning a cancellable backend can still interrupt a read in flight, and
    still knows that it can.

    Without it a generator would pass every delegation test by not existing,
    and a node timeout would quietly go back to waiting out the provider
    client's own thirty-second one.
    """

    class _BlockingBackend:
        """A backend stream that stops between events until aborted."""

        def __init__(self):
            self._released = threading.Event()
            self._sent = False
            self.aborted = False
            self.closed = False

        def __iter__(self):
            return self

        def __next__(self):
            if not self._sent:
                self._sent = True
                return {"event": "token", "data": "400 hours"}
            # The read a stop has to interrupt. Without `abort` reaching it,
            # nothing here checks a flag and the thread stays put.
            self._released.wait(timeout=10)
            if self.aborted:
                # What a cancellable backend really does: the shutdown socket
                # raises out of the read rather than ending it tidily. Ending
                # tidily would be recorded as a natural completion, which is
                # the opposite of what a stop means.
                raise ConnectionError("stream aborted")
            raise StopIteration

        @property
        def armed(self):
            return True

        def abort(self):
            self.aborted = True
            self._released.set()

        def close(self):
            self.closed = True

    @pytest.mark.asyncio
    async def test_stopping_the_pump_reaches_the_backend_through_the_wrapper(self):
        backend = self._BlockingBackend()
        wrapper = ScrubbedTokenStream(backend, NONCE)
        pump = StreamPump(lambda: wrapper, label="citation-stream").start()

        events = pump.events()
        first = await asyncio.wait_for(events.__anext__(), timeout=5)
        assert first == {"event": "token", "data": "400 hours"}

        # The pump can say the death will be prompt, which it reads off the
        # iterator it owns - the wrapper, which answers for the backend.
        assert pump.cancellation_proven()

        assert await asyncio.wait_for(pump.wait_dead(2.0), timeout=5)
        assert backend.aborted, "the stop never reached the backend"
        assert backend.closed, "the producer thread did not close the backend"
        assert pump.interrupted
