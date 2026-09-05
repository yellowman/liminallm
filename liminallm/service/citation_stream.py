"""Taking a turn's citation namespace out of an answer as it is written.

`scrub_positions` answers the question for a finished string: what crossed,
and where every character came from. A streamed answer has no finished string
until it is over, and the tokens have already reached the reader by then. So
this is the same transformation performed incrementally, by a parent that
holds the raw text and releases only what can no longer change.

The rule it exists to keep is that a marker never becomes observable. Emitting
`[cite:K7Q2ABCD-1]` and cleaning it up at the end is not a boundary: once a
token has left, the removal is a correction, and a reader that renders as it
receives has already shown it.

What makes this more than a search-and-replace is that the answer arrives cut
at arbitrary points. A provider may send `[ci`, then `te:K7Q2`, then
`ABCD-1]`; a bare nonce may straddle three chunks; and a removal can splice
its neighbours into a fresh occurrence, so text that looked safe stops being
safe when what follows it disappears. A regex over each chunk sees none of
this.

The answer is to emit a prefix and hold a suffix. After each chunk the whole
canonical text is scrubbed from scratch - the transformation is defined on the
whole string and reproducing it approximately would be a second implementation
that disagrees - and the result is split: everything a future chunk could
still change is held back, and the rest is released. What has been released is
therefore always a prefix of what the finished string would scrub to, which is
the property `finish` then asserts outright.

Rescanning the whole answer per chunk is quadratic, and that is a deliberate
choice rather than an oversight. Measured on four-character chunks: 2,000
characters cost 0.02s in total, 6,000 cost 0.18s, 12,000 cost 0.72s. An
ordinary answer is well inside the first of those and the cost is spread
across the stream rather than paid at the end, so the incremental scrubber
this would be replaced with buys milliseconds and owes a proof that it agrees
with `scrub_positions` on every input. If answers get long enough for it to
matter, the number to beat is here.

Whoever wires this owes it a ceiling, because nothing else provides one today.
`MAX_GENERATION_TOKENS` is only ever subtracted from the context window to
leave room for a reply; no backend here sends a max-output parameter of any
kind, so a reply's length is the provider's to choose. Extrapolating the
numbers above, a 100,000-character answer is tens of seconds of regex. Two
things make that safe and neither is this module's to do: run the scan off the
event loop, and bound the canonical text - by an output cap on the request, or
by refusing to keep scrubbing past a size the caller names.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from liminallm.service.citations import scrub_positions


#: What a namespace occurrence can be built from, before it is complete.
#:
#: Read against the *scrubbed* text rather than the raw text, because the
#: scrub is what creates these: `[cite:K7Q2ABCD-1` with its bracket still
#: unclosed loses the bare handle inside it and becomes a bare `[cite:`, and
#: that is what the next chunk's `]` would complete.
#:
#: Nothing here is a complete occurrence - the scrub has already removed those
#: - so every branch stops one character short of closing, or trails an
#: optional part that has not arrived. The whitespace run is unbounded on
#: purpose: `[ \t]*` before a marker is part of the match, so a hundred spaces
#: at the end of a chunk are a hundred characters a marker could still claim.
def _partial_pattern(nonce: str) -> "re.Pattern[str]":
    prefixes = "|".join(
        re.escape(nonce[:length]) for length in range(len(nonce), -1, -1)
    )
    handle = rf"(?:{prefixes})(?:-\d*)?"
    bracketed = rf"\[(?:c(?:i(?:t(?:e(?::(?:{handle})?)?)?)?)?)?"
    return re.compile(rf"[ \t]*(?:{bracketed}|{handle})?", re.IGNORECASE)


#: Characters a partial occurrence can be made of. Walking back past one of
#: these ends the search immediately, so the scan is the length of the run
#: rather than of the answer.
#:
#: A pattern rather than a set, compiled with the flag the matcher above uses,
#: because "the same character ignoring case" is the regex engine's question
#: and it does not answer it the way `str` does. `re.IGNORECASE` folds four
#: characters into this alphabet that `.lower() + .upper()` leaves out - U+212A
#: KELVIN SIGN for `K`, U+017F LATIN SMALL LETTER LONG S for `S`, and U+0131
#: and U+0130 for `i` - and `K` and `S` are both in the nonce alphabet.
#:
#: What that cost was not a slow search. `K7Q2` `ABCD` split across two chunks
#: is a nonce to the finished scrub and was four ordinary characters to this
#: walk, so the first half went out and the second half deleted it. The one
#: before it was the same mistake in ASCII - a walk that knew only lowercase
#: `cite` stopped inside `[CITE:` - which is why this now asks rather than
#: enumerates.
def _partial_characters(nonce: str) -> "re.Pattern[str]":
    letters = sorted(set(" \t[:-0123456789cite" + nonce))
    joined = "|".join(re.escape(character) for character in letters)
    return re.compile(rf"(?:{joined})", re.IGNORECASE)


class CanonicalCitationStream:
    """One streamed answer, in both representations at once.

    The canonical side is every raw chunk the provider sent, concatenated and
    never edited: it is what a citation is read out of, and the only text that
    can honestly say what the model wrote.

    The public side is what has been released. It is produced only by
    `scrub_positions` over the canonical text, so the two representations
    cannot drift apart the way two implementations of one rule would.

    Not a general filter. It removes exactly this turn's namespace, in the
    forms `scrub_positions` removes it, and leaves every other bracketed
    thing - another turn's marker, prose about citations, an array index -
    exactly as the model wrote it.
    """

    def __init__(self, nonce: str) -> None:
        self.nonce = nonce
        self._canonical: List[str] = []
        self._released = ""
        self._partial = _partial_pattern(nonce)
        self._characters = _partial_characters(nonce)
        self._finished = False

    @property
    def canonical(self) -> str:
        """Everything the provider sent, unedited."""
        return "".join(self._canonical)

    @property
    def released(self) -> str:
        """Everything that has been handed to the reader."""
        return self._released

    def push(self, chunk: str) -> str:
        """Take one provider chunk, and return what is now safe to emit.

        The empty string is a normal answer: a chunk may be entirely inside a
        marker, or may extend a run of trailing spaces a marker could still
        claim. Nothing is owed per chunk.
        """
        if self._finished:
            raise RuntimeError("chunk pushed after the stream was finished")
        self._canonical.append(chunk)
        public, _origins = scrub_positions(self.canonical, self.nonce)
        return self._release(public, len(public) - self._held(public))

    def finish(self) -> Tuple[str, List[int]]:
        """Close the stream: the last of the text, and the origin map.

        Nothing is held any more - there is no future chunk to claim it - so
        what comes back is whatever the hold was covering, which is empty for
        an answer that did not end mid-marker.

        `origins[i]` is the index in the canonical text of the character at
        `i` in the public text, the same map `scrub_positions` returns and
        `citation_payload` reads. It is produced here rather than accumulated
        as chunks arrive because it describes the finished string.
        """
        public, origins = scrub_positions(self.canonical, self.nonce)
        tail = self._release(public, len(public))
        self._finished = True
        return tail, origins

    def intact(self) -> bool:
        """Whether this stream finished, and released what it should have.

        The contract as one comparison, checked rather than argued. Everything
        above is an argument that the released text is always a prefix of this
        one; a caller that gets `False` has an answer whose public form nobody
        can vouch for, and no citation may be read out of it.

        A stream that has not finished is not intact, whatever its text says.
        Saying "only meaningful after `finish`" and then answering anyway put
        the burden on every caller: an answer whose held tail happens to be
        empty - which is most ordinary prose - would have told a cancelled or
        failed turn that its public form was vouched for. This is about to sit
        in an authority gate, so it answers the question the gate is asking.
        """
        if not self._finished:
            return False
        public, _origins = scrub_positions(self.canonical, self.nonce)
        return self._released == public

    def _release(self, public: str, safe: int) -> str:
        """Hand over `public` up to `safe`, minus what has gone already.

        The check is against the whole scrubbed text, not against the safe
        part of it. Those are different questions: the hold moving back over
        text already released only means nothing more can go out this time,
        while the scrub no longer *starting* with what went out means an
        occurrence has eaten it.

        Comparing against the safe part conflated the two and made an
        ordinary answer - a chunk ending in `-999`, whose digits a handle
        could still have claimed - look like a boundary failure.
        """
        if not public.startswith(self._released):
            # The hold was too short: text already handed over turned out to
            # be inside an occurrence, and there is no taking it back. Raising
            # is the only honest outcome - the alternative is continuing to
            # stream an answer whose public form is already wrong.
            raise ValueError(
                "citation stream released text the scrub later removed"
            )
        if safe <= len(self._released):
            # Measured equivalent, and kept as the statement of what the two
            # marks mean. Instrumented over 187,000 releases of the corpus in
            # the tests, `safe` never moved back over released text at all,
            # and the equal case - which is most of them - slices empty and
            # reassigns the same string either way.
            #
            # What it says that the arithmetic does not is that the released
            # mark only ever moves forward. Without it, a hold that did reach
            # back would shorten the mark and send that text a second time,
            # which is a worse failure than the one it would be recovering
            # from.
            return ""
        fresh = public[len(self._released):safe]
        self._released = public[:safe]
        return fresh

    def _held(self, public: str) -> int:
        """How much of `public` a future chunk could still absorb.

        Two ways it can, and the second is why this is a loop.

        Directly: the tail is a partial occurrence, and the rest of it is
        still coming. That is `_partial_suffix` below.

        By splicing: an occurrence that has not arrived yet will be removed
        when it does, and removing it joins the text on either side. So text
        *before* the partial tail can end up adjacent to text that has not
        been sent, and the two together can form an occurrence that neither
        was. `K7Q2` `K7Q2` `k7q2ab` releases the first eight characters if
        only the direct rule applies - and then `cdABCD` arrives, the middle
        disappears, and what was released spliced into a nonce.

        So the hold is extended over whatever the text before it could
        contribute to such a junction, and again over whatever precedes
        *that*, until it stops growing. Each pass either reaches back or ends,
        and it cannot reach past the start.
        """
        held = 0
        while True:
            grown = self._partial_suffix(public[: len(public) - held])
            if not grown:
                return held
            held += grown

    def _partial_suffix(self, public: str) -> int:
        """The longest suffix of `public` that a match could begin with.

        A suffix search rather than a fixed window because the parts have no
        common bound: the digits of a handle are a run, and so is the
        whitespace a marker eats in front of itself.

        The walk stops at the first character no occurrence can contain,
        which is what keeps this proportional to the run at the end of the
        text rather than to the answer.
        """
        start = len(public)
        while start > 0 and self._characters.fullmatch(public[start - 1]):
            start -= 1
        for index in range(start, len(public)):
            if self._partial.fullmatch(public[index:]):
                return len(public) - index
        return 0


#: How much canonical text one streamed answer may accumulate.
#:
#: A ceiling is needed because nothing else supplies one. `MAX_GENERATION_TOKENS`
#: is only ever subtracted from the context window to leave room for a reply,
#: and no backend here sends a max-output parameter, so a provider decides how
#: long an answer runs. The scan above is quadratic, so "as long as it likes"
#: is a way to spend a minute of CPU on one turn.
#:
#: Four characters per token against that same 4,096, which is the length the
#: rest of the system already treats as a whole reply. Cutting it finer would
#: buy time by killing answers the deployment considers legitimate, which is
#: the wrong trade to make silently.
#:
#: Measured at the limit rather than extrapolated: 16,384 characters in 4,097
#: events costs 2.1s. That is producer-thread CPU spread across the stream -
#: half a millisecond per token, invisible as latency - so what the ceiling
#: bounds is how much of a worker one long answer can occupy, not how fast a
#: short one feels. A deployment that wants less passes a smaller number.
#:
#: The constant factor is the thing worth attacking if this ever matters, and
#: the shape of the fix is known: everything before the released mark is
#: settled and contains no occurrences, so the rescan could start at the
#: frontier instead of at zero. It is not done here because it is a second
#: implementation of the transformation and would owe the differential proof
#: over again - which is at least a proof this module already knows how to
#: produce.
MAX_CANONICAL_CHARS = 4 * 4096


class CanonicalStreamTooLong(RuntimeError):
    """A streamed answer went past what the parent will scrub."""


class ScrubbedTokenStream:
    """A provider's event iterator with this turn's namespace taken out.

    Wraps the iterator rather than the consumer, so the scrubbing happens on
    whichever thread pulls the provider - which is `StreamPump`'s own producer
    thread, not the event loop. A quadratic scan on the loop would stall every
    other request the worker is serving for the length of one long answer.

    A wrapper, not a generator, and that distinction is the reason this class
    exists at all. `StreamPump` reaches into its iterator for `abort` when it
    stops - a cancellable backend can interrupt a read already in flight,
    which the stop flag alone cannot - and for `armed` to decide whether that
    death can be presumed prompt. A generator has neither, so wrapping the
    provider in one would silently take a `timeout_ms` back to waiting out the
    provider client's own 30-60 second timeout. Both are proxied, and so is
    `close`, which the pump calls from the producer thread on the way out.

    What reaches the consumer is only ever `reader.released` text. A token
    event carrying nothing safe yet is not forwarded as an empty one: the
    provider is pulled again, so a marker split across five chunks costs five
    reads rather than five empty events.
    """

    def __init__(
        self,
        events: Any,
        nonce: str,
        *,
        max_canonical_chars: int = MAX_CANONICAL_CHARS,
    ) -> None:
        self._events = iter(events)
        self.reader = CanonicalCitationStream(nonce)
        self._limit = max_canonical_chars
        self._pending: List[Dict[str, Any]] = []
        #: Set when the provider's own final content did not match the tokens
        #: it sent. The completion is refused rather than believed, and the
        #: reader is left unfinished, which is what `intact` reports.
        self.contradicted = False

    # -- what the pump reaches for ----------------------------------------

    def __iter__(self) -> "ScrubbedTokenStream":
        return self

    @property
    def armed(self) -> bool:
        return bool(getattr(self._events, "armed", False))

    def abort(self) -> None:
        abort = getattr(self._events, "abort", None)
        if callable(abort):
            abort()

    def close(self) -> None:
        close = getattr(self._events, "close", None)
        if callable(close):
            close()

    # -- the filter -------------------------------------------------------

    def __next__(self) -> Dict[str, Any]:
        while True:
            if self._pending:
                return self._pending.pop(0)
            event = next(self._events)
            if not isinstance(event, dict):
                return event
            kind = event.get("event")
            if kind == "token":
                public = self._take(str(event.get("data") or ""))
                if not public:
                    # Nothing has cleared the hold. Pull again rather than
                    # forward an empty token, which a consumer counting
                    # events would read as the model having said nothing.
                    continue
                return {"event": "token", "data": public}
            if kind == "message_done":
                self._pending.extend(self._complete(event))
                continue
            return event

    def _take(self, chunk: str) -> str:
        """One raw chunk in, whatever is safe to show out."""
        if len(self.reader.canonical) + len(chunk) > self._limit:
            # Past the ceiling. Not truncated: earlier public tokens have
            # already reached the client, so quietly stopping here would hand
            # them a shorter answer that looks finished. The provider is cut
            # off, the reader never finishes, and the turn ends as the error
            # it is.
            self.abort()
            raise CanonicalStreamTooLong(
                f"streamed answer exceeded {self._limit} characters"
            )
        return self.reader.push(chunk)

    def _complete(self, event: Dict[str, Any]) -> List[Dict[str, Any]]:
        """The end of the stream, in public terms.

        The provider's own `content` is checked against the tokens it sent
        rather than trusted in place of them. They are two claims about one
        answer, and a provider that contradicts itself has not given the
        parent an answer it can read citations out of - so the reader is left
        unfinished, `intact` stays false, and the held tail is never flushed.

        Otherwise the tail goes first. A consumer that replaces its
        accumulated tokens with the final content has to be given a final
        content those tokens add up to.
        """
        data = dict(event.get("data") or {})
        reported = data.get("content")
        if reported is not None and str(reported) != self.reader.canonical:
            self.contradicted = True
            data["content"] = self.reader.released
            return [{**event, "data": data}]
        tail, _origins = self.reader.finish()
        data["content"] = self.reader.released
        done = {**event, "data": data}
        if tail:
            return [{"event": "token", "data": tail}, done]
        return [done]
