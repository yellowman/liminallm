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

The reader below emits a prefix and holds a suffix, and reads each character
once. The first version of this module scrubbed the whole canonical text
again after every chunk, which was correct and quadratic in the number of
chunks: a provider sending one character at a time paid `1 + 2 + ... + N` for
an answer it could have sent whole.

What replaced it has the same shape as `_scrub_text`, one character at a time.
That function removes every leftmost non-overlapping match and then repeats,
and the repetition is the interesting half: a removal splices its neighbours,
and the pair can be an occurrence that neither was - but the pass that made
the junction has already scanned past it, so only the *next* pass sees it. So
a pass here is an automaton over arriving characters, it hands what survives
to the pass below it, and a pass is added exactly when the one above removes
something. Getting that order wrong is not a performance bug: a reader that
matched across a junction immediately found an occurrence overlapping the one
the oracle takes, and released two characters the finished scrub does not
contain.

The transformation itself is unchanged. `scrub_positions` is still what
defines it, still the only thing that produces the finished public text and
the origin map, and `finish` asks it once and checks that what went out is
what it says.

Whoever wires this owes it a ceiling anyway, because nothing else provides
one: `MAX_GENERATION_TOKENS` is only ever subtracted from the context window
to leave room for a reply, and no backend here sends a max-output parameter,
so a reply's length is the provider's to choose.
"""

from __future__ import annotations

import re
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

from liminallm.service.citations import scrub_positions

#: The keyword a bracketed marker is written with, between `[` and the handle.
_CITE = "[cite:"


class _Alphabet:
    """Character questions, asked of the engine that defines the answer.

    `re.IGNORECASE` folds characters into the namespace alphabet that
    `.lower()` and `.upper()` leave out - U+212A KELVIN SIGN for `K`, U+017F
    LATIN SMALL LETTER LONG S for `S`, U+0131 and U+0130 for `i` - and `K` and
    `S` are both in the nonce alphabet. A reader that decided "is this the
    same letter" with `str` released half a nonce and had the other half
    delete it, which is why every comparison here goes through a compiled
    pattern instead.

    Memoized per character, so a repeated character costs a dict lookup: the
    engine is asked once per distinct character per position, and the answers
    do not change.
    """

    def __init__(self, nonce: str) -> None:
        self._nonce = nonce
        self._patterns = [
            re.compile(re.escape(character), re.IGNORECASE) for character in nonce
        ]
        self._cite = [
            re.compile(re.escape(character), re.IGNORECASE) for character in _CITE
        ]
        self._digit = re.compile(r"\d")
        self._memo: Dict[Tuple[str, int, str], bool] = {}

    def nonce_at(self, index: int, character: str) -> bool:
        key = ("n", index, character)
        answer = self._memo.get(key)
        if answer is None:
            answer = bool(self._patterns[index].fullmatch(character))
            self._memo[key] = answer
        return answer

    def cite_at(self, index: int, character: str) -> bool:
        key = ("c", index, character)
        answer = self._memo.get(key)
        if answer is None:
            answer = bool(self._cite[index].fullmatch(character))
            self._memo[key] = answer
        return answer

    def is_digit(self, character: str) -> bool:
        key = ("d", 0, character)
        answer = self._memo.get(key)
        if answer is None:
            answer = bool(self._digit.fullmatch(character))
            self._memo[key] = answer
        return answer

    @staticmethod
    def is_space(character: str) -> bool:
        # `[ \t]` exactly, which no case fold widens.
        return character == " " or character == "\t"

    def failure(self) -> List[int]:
        """The KMP failure function over the nonce, under the same equality.

        Needed rather than a single candidate because the nonce alphabet can
        draw a self-overlapping nonce - `ABABABAB` is eight legal characters -
        and then `ABABAB` + `AB` contains an occurrence starting at the third
        character that a matcher restarting at the failing character misses.
        The fold classes are an equivalence relation, so the usual argument
        for KMP still holds.
        """
        nonce = self._nonce
        table = [0] * len(nonce)
        length = 0
        for index in range(1, len(nonce)):
            while length and not self.nonce_at(length, nonce[index]):
                length = table[length - 1]
            if self.nonce_at(length, nonce[index]):
                length += 1
            table[index] = length
        return table


class _State:
    """What the automaton knows at one position of the pending text.

    Immutable and small: one is recorded per pending character so that a
    removal can restore the position before it in constant time.

    * `spaces` - the run of spaces and tabs ending here. A marker's `[ \t]*`
      is part of the match, so a run at the end of the text is a run a marker
      could still claim.
    * `cite` - how much of `[cite:` ends here, so an arriving `e` knows
      whether it is continuing a keyword.
    * `nonce` - KMP progress: the longest prefix of the nonce that is a suffix
      of the text here.
    * `dash` and `digits` - a completed handle's optional `-\\d+`, being
      collected. Meaningful only while `handle` is set.
    * `handle` - the pending text ends with a complete handle, whose nonce
      starts at this index. Set exactly when `nonce` reached the full length.

    "Clean" is the whole frontier rule. A clean state can begin no match and
    can be reached back through by none, so everything at or before it is
    settled - including against a splice, because a splice-created occurrence
    still has to start at a character that was already a candidate here.
    """

    __slots__ = ("spaces", "cite", "nonce", "dash", "digits", "handle")

    def __init__(
        self,
        spaces: int = 0,
        cite: int = 0,
        nonce: int = 0,
        dash: bool = False,
        digits: int = 0,
        handle: Optional[int] = None,
    ) -> None:
        self.spaces = spaces
        self.cite = cite
        self.nonce = nonce
        self.dash = dash
        self.digits = digits
        self.handle = handle

    @property
    def clean(self) -> bool:
        return (
            self.spaces == 0
            and self.cite == 0
            and self.nonce == 0
            and self.handle is None
        )


_CLEAN = _State()


class _Pass:
    """One pass of the scrub, performed left to right as text arrives.

    `_scrub_text` removes every leftmost non-overlapping match, then repeats
    over what is left until a pass finds nothing. The repetition is not a
    detail: removing a match splices its neighbours, and the pair can be an
    occurrence that neither was - but only the *next* pass sees it, because
    the pass that made the junction has already scanned past it.

    Reproducing that faithfully is what this class is. One pass is a scan
    with a scan pointer that only moves forward, so it is an automaton over
    the arriving characters:

    * text that no future match in *this* pass can reach is handed on;
    * a match is dropped, and the pass resumes as `finditer` does - after
      the match, remembering nothing before it;
    * whatever it hands on becomes the next pass's input, which is where a
      splice is finally reconsidered.

    A pass hands its output to the pass below it, and the bottom of the
    chain is what the reader releases. A pass with no successor has never
    removed anything, so its state is a function of exactly the text it has
    handed on: it hands text on only when clean, and a fresh pass reading
    that same text would be clean there too. That is what makes releasing
    from the bottom safe even though a new pass may be added later.
    """

    __slots__ = ("_reader", "_successor", "_pending", "_states", "_state", "queue")

    def __init__(self, reader: "CanonicalCitationStream") -> None:
        self._reader = reader
        self._successor: Optional["_Pass"] = None
        #: The characters this pass has not handed on, and the state after
        #: each. Both are emptied whole, so index 0 is always the first.
        self._pending: List[str] = []
        self._states: List[_State] = []
        self._state = _CLEAN
        #: What this pass has been handed and not yet read. A deque because
        #: it is read from the front and a whole chunk goes in at once. It is
        #: also what lets the reader drive and close the chain a pass at a
        #: time rather than by recursion: a pass hands text down by queueing
        #: it, and a long answer can need more passes than Python has stack.
        self.queue: Deque[str] = deque()

    # -- reading ------------------------------------------------------------

    def step(self) -> None:
        """Read one queued character, with whatever it settles or removes."""
        rereads = [self.queue.popleft()]
        while rereads:
            current = rereads.pop()
            state = self._state
            if state.handle is not None:
                if self._extends_handle(current):
                    self._append(current, self._extended(state, current))
                    continue
                if current == "]" and self._closes_marker(state):
                    self._append(current, state)
                    self._remove_marker(state)
                    continue
                dash = self._settle_handle()
                # The trigger goes back under the dash the handle gave up,
                # so the dash is read first: it is the earlier character.
                rereads.append(current)
                if dash is not None:
                    rereads.append(dash)
                continue
            self._append(current, self._advance(state, current))
        if self._state.clean and self._pending:
            self._hand_on()

    def close(self) -> None:
        """No more input: this pass's last handle goes, and its tail with it.

        Its own queue first. A pass above hands text down by queueing it, so
        closing without reading that queue would drop everything the pass
        above settled on its way out.

        This pass only. What it hands down lands in the successor's queue,
        and the reader closes the chain in order - closing the successor from
        here would be one Python frame per pass, and how many passes an
        answer needs is the model's choice, not this module's.
        """
        while self.queue:
            self.step()
        while self._state.handle is not None:
            dash = self._settle_handle()
            if dash is not None:
                self.queue.append(dash)
            while self.queue:
                self.step()
        if self._pending:
            self._hand_on()

    # -- the automaton ------------------------------------------------------

    def _advance(self, state: _State, character: str) -> _State:
        """The state after `character`, with no handle pending."""
        alphabet = self._reader._alphabet
        if alphabet.is_space(character):
            # A space cannot continue a keyword or a nonce, and starts a run
            # a later marker may claim.
            return _State(spaces=state.spaces + 1)

        # The keyword, one character at a time. A complete `[cite:` is not a
        # match by itself - what follows decides - so it is carried as a
        # candidate until something that is not the nonce arrives. A second
        # `[` starts the keyword over wherever it appears.
        cite = 0
        if 0 < state.cite < len(_CITE) and alphabet.cite_at(state.cite, character):
            cite = state.cite + 1
        elif alphabet.cite_at(0, character):
            cite = 1

        nonce = state.nonce
        failure = self._reader._failure
        self._reader._work += 1
        while nonce and not alphabet.nonce_at(nonce, character):
            nonce = failure[nonce - 1]
            self._reader._work += 1
        if alphabet.nonce_at(nonce, character):
            nonce += 1
        if nonce == len(self._reader.nonce):
            # A complete handle, ending here. Not removed yet: `-12` may
            # still be part of it, and so may a closing bracket. `handle` is
            # where the nonce *starts*: this character is about to be
            # appended at `len(self._pending)`, and the nonce is the run
            # ending with it.
            return _State(
                cite=cite,
                nonce=nonce,
                handle=len(self._pending) - len(self._reader.nonce) + 1,
            )
        return _State(cite=cite, nonce=nonce)

    def _extends_handle(self, character: str) -> bool:
        state = self._state
        if character == "-":
            return not state.dash
        return state.dash and self._reader._alphabet.is_digit(character)

    @staticmethod
    def _extended(state: _State, character: str) -> _State:
        if character == "-":
            return _State(
                cite=state.cite, nonce=state.nonce, dash=True,
                digits=0, handle=state.handle,
            )
        return _State(
            cite=state.cite, nonce=state.nonce, dash=True,
            digits=state.digits + 1, handle=state.handle,
        )

    def _closes_marker(self, state: _State) -> bool:
        """Whether a `]` here closes `[cite:` + handle.

        The keyword is read out of the pending text rather than tracked
        alongside the handle: it sits immediately before the nonce, so this is
        six characters at a known offset. `-` with no digits is not a marker -
        `-\\d+` needs a digit - and the bare handle inside it is what the
        finished scrub removes.
        """
        if state.dash and state.digits == 0:
            return False
        start = state.handle
        if start is None or start < len(_CITE):
            return False
        self._reader._work += len(_CITE)
        alphabet = self._reader._alphabet
        for offset in range(len(_CITE)):
            if not alphabet.cite_at(
                offset, self._pending[start - len(_CITE) + offset]
            ):
                return False
        return True

    # -- the tape -----------------------------------------------------------

    def _append(self, character: str, state: _State) -> None:
        self._pending.append(character)
        self._states.append(state)
        self._state = state

    def _settle_handle(self) -> Optional[str]:
        """Remove the complete handle at the end of the pending tail.

        The bare form: `[ \t]*` then the nonce then an optional `-\\d+`. A
        dash with no digits after it is not part of the handle - `-\\d+`
        needs a digit - so it is handed back to be read again, after the
        removal, as the character following it.
        """
        state = self._state
        start = state.handle
        assert start is not None
        dash: Optional[str] = None
        if state.dash and state.digits == 0:
            dash = self._pending[-1]
            self._drop(1)
        spaces = self._states[start - 1].spaces if start else 0
        self._remove(start - spaces)
        return dash

    def _remove_marker(self, state: _State) -> None:
        """Remove `[ \t]*[cite:` + handle + `]`, ending at the pending tail."""
        start = state.handle
        assert start is not None
        opening = start - len(_CITE)
        spaces = self._states[opening - 1].spaces if opening else 0
        self._remove(opening - spaces)

    def _remove(self, start: int) -> None:
        """Drop `pending[start:]`, which is the match, and resume after it.

        Everything before it is settled *for this pass*: the scan pointer is
        past it and a pass never looks back. It is handed on rather than
        released, to a pass that reads it against what follows the removal -
        which is the splice, and the reason `_scrub_text` repeats.
        """
        if self._successor is None:
            self._successor = _Pass(self._reader)
            self._reader._passes.append(self._successor)
        self._drop(len(self._pending) - start)
        # `finditer` remembers nothing before the match it just passed, and
        # neither does this: handing the text on empties the tail and clears
        # the state, and a removal that emptied the tail already cleared it.
        # Stating it a third time here would be a line no test can reach.
        if self._pending:
            self._hand_on()

    def _drop(self, count: int) -> None:
        """Take `count` characters off the pending tail, state and all.

        Every character is appended once and dropped at most once, so the
        total cost of removal over a pass is the length of the pass's input.
        """
        if count <= 0:
            return
        self._reader._work += count
        del self._pending[len(self._pending) - count:]
        del self._states[len(self._states) - count:]
        self._state = self._states[-1] if self._states else _CLEAN

    def _hand_on(self) -> None:
        """Give the pending text to the pass below, or release it."""
        self._reader._work += len(self._pending)
        text = "".join(self._pending)
        self._pending.clear()
        self._states.clear()
        self._state = _CLEAN
        if self._successor is not None:
            self._successor.queue.extend(text)
        else:
            self._reader._release(text)


class CanonicalCitationStream:
    """One streamed answer, in both representations at once.

    The canonical side is every raw chunk the provider sent, concatenated and
    never edited: it is what a citation is read out of, and the only text that
    can honestly say what the model wrote.

    The public side is what has been released. `scrub_positions` remains the
    definition of what that is - `finish` produces the finished text and the
    origin map with it, and checks that what went out is exactly it - and the
    passes above are how the same answer is reached one character at a time,
    without rescanning what is already settled.

    Not a general filter. It removes exactly this turn's namespace, in the
    forms `scrub_positions` removes it, and leaves every other bracketed
    thing - another turn's marker, prose about citations, an array index -
    exactly as the model wrote it.
    """

    def __init__(self, nonce: str) -> None:
        self.nonce = nonce
        self._alphabet = _Alphabet(nonce)
        self._failure = self._alphabet.failure()
        self._canonical: List[str] = []
        self._canonical_len = 0
        self._canonical_text: Optional[str] = None
        #: Released text as the pieces it went out in. Joined when somebody
        #: asks, never per chunk: rebuilding the prefix on every release is
        #: one of the growing-prefix costs this reader exists without.
        self._released_parts: List[str] = []
        self._released_len = 0
        #: The passes, top first. One is enough for an answer with nothing to
        #: remove; a pass is added when the one above it removes something,
        #: exactly as `_scrub_text` repeats only when a pass found a match.
        self._passes: List[_Pass] = [_Pass(self)]
        self._finished = False
        self._verdict: Optional[bool] = None
        #: Where a release lands while a call is collecting one. Set by
        #: `push` and `finish`, so a pass can hand text out from any depth
        #: without knowing which call it is answering.
        self._fresh: Optional[List[str]] = None
        #: Characters of variable-length work, for the complexity witness.
        #: Every loop and every slice below adds what it actually touched.
        self._work = 0

    # -- what the stream is ------------------------------------------------

    @property
    def canonical(self) -> str:
        """Everything the provider sent, unedited.

        Joined at most once between chunks: the completion asks for it twice -
        the provider's own final content is checked against it, and the scrub
        reads it - and joining the answer per ask is the same growing-prefix
        cost the ceiling check used to pay per token.
        """
        if self._canonical_text is None:
            self._work += self._canonical_len
            self._canonical_text = "".join(self._canonical)
        return self._canonical_text

    @property
    def canonical_length(self) -> int:
        """How much the provider has sent, without materializing it.

        The ceiling check runs per token and asked `len(self.canonical)`,
        which joined the whole answer to count it.
        """
        return self._canonical_len

    @property
    def released(self) -> str:
        """Everything that has been handed to the reader."""
        self._work += self._released_len
        return "".join(self._released_parts)

    # -- the stream ---------------------------------------------------------

    def push(self, chunk: str) -> str:
        """Take one provider chunk, and return what is now safe to emit.

        The empty string is a normal answer: a chunk may be entirely inside a
        marker, or may extend a run of trailing spaces a marker could still
        claim. Nothing is owed per chunk.
        """
        if self._finished:
            raise RuntimeError("chunk pushed after the stream was finished")
        self._canonical.append(chunk)
        self._canonical_len += len(chunk)
        self._canonical_text = None
        fresh: List[str] = []
        self._fresh = fresh
        try:
            self._work += len(chunk)
            self._passes[0].queue.extend(chunk)
            self._drive()
        finally:
            self._fresh = None
        return "".join(fresh)

    def _drive(self) -> None:
        """Read what is queued, one pass at a time, top to bottom.

        A pass only ever hands text downwards, so one sweep is enough - and a
        sweep stops at the first pass with nothing waiting, so an answer with
        many passes costs nothing per character in the ones it does not
        reach.
        """
        index = 0
        while index < len(self._passes):
            current = self._passes[index]
            if not current.queue:
                break
            while current.queue:
                current.step()
            index += 1

    def _release(self, text: str) -> None:
        """Text that has fallen out of the bottom pass."""
        if not text:
            return
        self._released_parts.append(text)
        self._released_len += len(text)
        if self._fresh is not None:
            self._fresh.append(text)

    def finish(self) -> Tuple[str, List[int]]:
        """Close the stream: the last of the text, and the origin map.

        Nothing is held any more - there is no future chunk to claim it - so
        each pass settles the handle it was holding and hands the rest down.

        `origins[i]` is the index in the canonical text of the character at
        `i` in the public text, the same map `scrub_positions` returns and
        `citation_payload` reads. Both come from the whole-string scrub, which
        stays the authority on the finished answer: this is where the two are
        compared - as equality, in both directions - and a reader that
        disagreed with it has no answer anyone can vouch for.
        """
        fresh: List[str] = []
        self._fresh = fresh
        try:
            # Top down, over a list that grows while it is walked: closing a
            # pass can settle a handle, which is a removal, which adds the
            # pass below it. Indexed rather than iterated for exactly that.
            index = 0
            while index < len(self._passes):
                self._passes[index].close()
                index += 1
        finally:
            self._fresh = None
        tail = "".join(fresh)
        public, origins = scrub_positions(self.canonical, self.nonce)
        self._finished = True
        released = self.released
        if released != public:
            # Equality, in both directions, and neither is repairable here.
            #
            # Released text the scrub removes cannot be taken back: the client
            # has seen it. Released text the scrub keeps cannot be supplied:
            # the caller builds the completion out of what went out, so an
            # answer that stops early is a truncated reply with a success
            # stamp on it - and appending the difference would be this module
            # writing an answer nobody streamed. Both are the same failure,
            # which is that the reader and the oracle do not agree about what
            # the answer is, and the honest response to that is to have no
            # answer.
            self._verdict = False
            raise ValueError(
                "citation stream released text the whole-string scrub "
                "does not agree with"
            )
        self._verdict = True
        return tail, origins

    def intact(self) -> bool:
        """Whether this stream finished, and released what it should have.

        The contract as one comparison, made once. `finish` performs it
        against the whole-string scrub and records the answer; asking again
        reads the record rather than scrubbing the answer a second time.

        A stream that has not finished is not intact, whatever its text says.
        Saying "only meaningful after `finish`" and then answering anyway put
        the burden on every caller: an answer whose held tail happens to be
        empty - which is most ordinary prose - would have told a cancelled or
        failed turn that its public form was vouched for. This sits in an
        authority gate, so it answers the question the gate is asking.
        """
        return bool(self._verdict)


#: How much canonical text one streamed answer may accumulate.
#:
#: A ceiling is needed because nothing else supplies one. `MAX_GENERATION_TOKENS`
#: is only ever subtracted from the context window to leave room for a reply,
#: and no backend here sends a max-output parameter, so a provider decides how
#: long an answer runs.
#:
#: Four characters per token against that same 4,096, which is the length the
#: rest of the system already treats as a whole reply. Cutting it finer would
#: buy time by killing answers the deployment considers legitimate, which is
#: the wrong trade to make silently.
#:
#: What the ceiling bounds now is memory and the one whole-string scrub at the
#: end, rather than a quadratic scan: the reader reads each character once
#: however the provider cuts the answer. A deployment that wants less passes a
#: smaller number.
MAX_CANONICAL_CHARS = 4 * 4096


class CanonicalStreamTooLong(RuntimeError):
    """A streamed answer went past what the parent will scrub."""


class ScrubbedTokenStream:
    """A provider's event iterator with this turn's namespace taken out.

    Wraps the iterator rather than the consumer, so the scrubbing happens on
    whichever thread pulls the provider - which is `StreamPump`'s own producer
    thread, not the event loop. Work on the loop would stall every other
    request the worker is serving for the length of one long answer.

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
        #: The origin map this stream's own completion produced, kept so a
        #: caller reading citations out of it never scrubs the text a second
        #: time. Two scrubs are two chances to disagree about where a marker
        #: was, which is the whole ambiguity the reader exists to remove.
        self.origins: List[int] = []

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
        if self.reader.canonical_length + len(chunk) > self._limit:
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
        parent an answer it can read citations out of.

        A contradiction becomes an error rather than a quieter completion.
        Refusing the citations is not enough on its own: `message_done` is
        what the streamed node treats as the answer boundary, so a completion
        carrying the released text - however correctly scrubbed - is a
        truncated answer with a success stamp on it, ready to be persisted as
        the turn's reply. There is no completion here to mistake for one. What
        the client has already been shown stands, by the same partial-answer
        handling a backend failure gets, and the reader is left unfinished so
        nothing downstream reads authority out of it.

        `finish` refuses the same completion for the same reason when the
        reader and the whole-string scrub disagree about the answer, in
        either direction: `content` below is the released text, so a reader
        that stopped short would be completed here as a shorter reply.

        Otherwise the tail goes first. A consumer that replaces its
        accumulated tokens with the final content has to be given a final
        content those tokens add up to.
        """
        data = dict(event.get("data") or {})
        reported = data.get("content")
        if reported is not None and str(reported) != self.reader.canonical:
            self.contradicted = True
            return [{
                "event": "error",
                "data": {
                    "code": "server_error",
                    "message": "provider stream contradicted its own tokens",
                },
            }]
        tail, self.origins = self.reader.finish()
        data["content"] = self.reader.released
        done = {**event, "data": data}
        if tail:
            return [{"event": "token", "data": tail}, done]
        return [done]
