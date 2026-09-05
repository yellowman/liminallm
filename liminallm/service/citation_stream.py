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
"""

from __future__ import annotations

import re
from typing import List, Tuple

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
#: Both cases of every one of them. The keyword is matched case-insensitively
#: and the nonce alphabet is uppercase, so a walk that knew only the lowercase
#: `cite` stopped at the `I` of `[CITE:` and never offered that suffix to the
#: check above - which then released `[CITE:` and had it removed under itself
#: when the closing bracket arrived. A missing character here is not a slow
#: search, it is a marker on the wire.
def _partial_alphabet(nonce: str) -> frozenset:
    letters = " \t[:-0123456789cite" + nonce
    return frozenset(letters.lower() + letters.upper())


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
        self._alphabet = _partial_alphabet(nonce)
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
        """Whether what was released is what the finished text scrubs to.

        The contract as one comparison, checked rather than argued. Everything
        above is an argument that the released text is always a prefix of this
        one; a caller that gets `False` has an answer whose public form nobody
        can vouch for, and no citation may be read out of it.

        Only meaningful once `finish` has run - before that the released text
        is a proper prefix by design.
        """
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
        while start > 0 and public[start - 1] in self._alphabet:
            start -= 1
        for index in range(start, len(public)):
            if self._partial.fullmatch(public[index:]):
                return len(public) - index
        return 0
