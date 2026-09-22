"""A marker removal joins its neighbours, and the pair can be a live handle.

`reader_positions` used to be two steps: scrub this turn's namespace, then
remove every closed marker. Each step repeats over its *own* splices and
neither reconsidered the other's, so:

    K7Q2[cite:]ABCD   ->   K7Q2ABCD

The namespace pass finds no handle - the nonce is in halves. The marker
cleanup then takes `[cite:]` out, joins the halves, and hands a reader this
turn's live namespace. A model needs no access to the nonce to arrange it,
only to write its own handle in two pieces around any marker at all, and
`[cite:OTHER-1]` and `[cite:x]` work as well as the empty one.

The fix is a third step: the namespace scrub runs again over what the marker
cleanup spliced. That step runs to its own fixed point, so what leaves it
holds no representation of the namespace, and nothing after it removes
anything that could splice one back.

Three steps rather than alternating to a joint fixed point, and the reason is
in `TestWhyTheMarkerCleanupIsNotRepeated` below. It is not tidiness - a
repeated marker cleanup cannot be implemented in a stream at all.

Every property here is asked of both implementations, because they have to
agree: `CanonicalCitationStream.finish` raises when they do not.
"""

from __future__ import annotations

import pytest

from liminallm.service.citation_stream import CanonicalCitationStream
from liminallm.service.citations import (
    CITATION_STRIP_RE,
    _namespace_pattern,
    _scrub_text,
    reader_answer,
    reader_positions,
    replaced_answer,
    scrub_namespace,
    strip_citation_positions,
)

NONCE = "K7Q2ABCD"

#: A handle written in halves around a marker. The marker's own body does not
#: matter - what matters is that removing it closes the gap.
SPLICES = [
    f"{NONCE[:4]}[cite:]{NONCE[4:]}",
    f"{NONCE[:4]}[cite:x]{NONCE[4:]}",
    f"{NONCE[:1]}[cite:]{NONCE[1:]}",
    f"{NONCE[:7]}[cite:]{NONCE[7:]}",
    f"{NONCE[:4]} [cite:OTHER-1]{NONCE[4:]}",
    f"{NONCE[:4]}[CITE:]{NONCE[4:]}",
    f"prose {NONCE[:2]}[cite:]{NONCE[2:]} more prose",
]


def two_step(text: str) -> str:
    """Reader cleanup as it was: scrub, then remove markers, and stop.

    Kept here rather than described, so that what this file exists to
    prevent is executed on every run. A witness that only asserts the
    current behaviour cannot tell a fix from a rewrite that never had the
    defect, and this one states the defect and requires it to be gone.
    """
    scrubbed, _ = _scrub_text(text, _namespace_pattern(NONCE))
    stripped, _ = strip_citation_positions(scrubbed)
    return stripped


def stream(text: str, chunks=None):
    reader = CanonicalCitationStream(NONCE)
    out = [reader.push(chunk) for chunk in (chunks or list(text))]
    tail, origins = reader.finish()
    return "".join(out) + tail, origins, reader


class TestAHandleSplicedTogetherDoesNotReachTheReader:
    @pytest.mark.parametrize("text", SPLICES)
    def test_the_two_step_cleanup_hands_over_the_namespace(self, text):
        """The defect, stated as a measurement rather than as prose.

        If this ever stops finding the nonce, these inputs have stopped
        reaching the thing the rest of the class is about, and the
        assertions below would be passing over text that never held a
        handle.
        """
        assert NONCE.lower() in two_step(text).lower()

    @pytest.mark.parametrize("text", SPLICES)
    def test_the_finished_string_does_not(self, text):
        public, origins = reader_positions(text, NONCE)
        assert NONCE.lower() not in public.lower(), public
        assert len(origins) == len(public)
        assert "".join(text[index] for index in origins) == public

    @pytest.mark.parametrize("text", SPLICES)
    def test_the_stream_agrees_at_every_chunk_boundary(self, text):
        expected, expected_origins = reader_positions(text, NONCE)
        for cut in range(len(text) + 1):
            released, origins, reader = stream(
                text, [text[:cut], text[cut:]]
            )
            assert released == expected, f"cut at {cut}"
            assert origins == expected_origins, f"cut at {cut}"
            assert reader.intact()

    @pytest.mark.parametrize("text", SPLICES)
    def test_an_abandoned_stream_releases_no_handle(self, text):
        """`fail` flushes what every stage was holding.

        Read off `released`, because a disagreement raises after the flush
        has released its text and before `fail` can return it.
        """
        for upto in range(len(text) + 1):
            reader = CanonicalCitationStream(NONCE)
            for character in text[:upto]:
                reader.push(character)
            try:
                reader.fail()
            except ValueError:
                pass
            assert NONCE.lower() not in reader.released.lower(), (
                f"abandoned after {upto} characters: {reader.released!r}"
            )

    def test_text_with_nothing_to_splice_is_untouched(self):
        """The control. A cleanup that removed more would also pass above."""
        for text in ("plain prose, no markers", "a [cite:OTHER-1] marker"):
            assert reader_positions(text, NONCE)[0] == two_step(text)

    def test_an_ordinary_handle_still_goes(self):
        """The other control: the narrow rule still does its own job."""
        assert reader_positions(f"see [cite:{NONCE}-1] here", NONCE)[0] == (
            "see here"
        )
        assert reader_positions(f"bare {NONCE} here", NONCE)[0] == "bare here"


class TestWhyTheMarkerCleanupIsNotRepeated:
    """The marker cleanup runs once, and that is a constraint, not a choice.

    Repeating it is what a joint fixed point would need, and a stream cannot
    follow it: a removal revives a match start that is already behind the
    reader, and how far behind is not bounded. So the finished-string helper
    is held to exactly what the stream can implement, and the residue is
    stale marker syntax - never a handle.
    """

    def test_a_removal_revives_a_match_start_arbitrarily_far_back(self):
        """The measurement the constraint rests on.

        Each `[ci` is three characters that only become a marker once
        everything between them and a `te:]` has gone. Repeating the
        cleanup consumes the lot, and the first `[` is three characters
        further back every time the text grows by seven.

        An earlier version of this computed `reach = 3 * depth` and then
        asserted `reach[16] == 48`, which is `3 * 16 == 48` - true whatever
        the grammar does, and therefore silent if the reach ever changed.
        The distance is measured off the string instead: how far the first
        `[` sits from the only marker in it, and that one pass leaves that
        first `[` alone while repetition consumes it.
        """
        reach = {}
        for depth in (2, 4, 8, 16):
            text = "[ci" * depth + "[cite:]" + "te:]" * depth
            marker = text.index("[cite:]")

            once, _ = strip_citation_positions(text)
            assert once.startswith("[ci"), (depth, once)

            repeated, _ = _scrub_text(text, CITATION_STRIP_RE)
            assert repeated == "", (depth, repeated)

            # The first `[` survives one pass and does not survive repetition,
            # so repetition reached back at least this far.
            reach[depth] = marker - text.index("[")

        assert reach == {2: 6, 4: 12, 8: 24, 16: 48}, reach
        assert reach[16] > reach[2], reach

    def test_one_pass_leaves_the_residue_and_no_handle(self):
        """What the reader actually gets, and what it does not."""
        text = "[ci[cite:]te:x]"
        public, _ = reader_positions(text, NONCE)
        assert public == "[cite:x]"
        assert NONCE.lower() not in public.lower()

    @pytest.mark.parametrize("depth", [1, 2, 4, 16])
    def test_the_stream_implements_exactly_that(self, depth):
        text = "[ci" * depth + "[cite:]" + "te:]" * depth
        expected, expected_origins = reader_positions(text, NONCE)
        released, origins, reader = stream(text)
        assert released == expected
        assert origins == expected_origins
        assert reader.intact()


class TestTheAnswerBoundaryTakesTheThirdStepToo:
    """`reader_positions` is not the only place the two steps were performed.

    `reader_answer` is the shared answer boundary: the workflow runners hand
    it a node's content and it removes the markers before that content
    becomes a token or a stored row. What it is handed is the worker/public
    copy - already through the narrow scrub - so its marker removal splices
    text the scrub has read past, and the pair can be a handle. The same
    defect, on a live path, found by reviewing the commit that fixed the
    other one.

    It takes the nonce now and performs the same three steps. A caller that
    cannot name the turn's namespace gets the old two, because removing a
    namespace you cannot name is not a thing a function can do; those
    callers sit downstream of one that can.
    """

    #: What the model wrote, as the worker copy would hold it. The narrow
    #: scrub finds nothing in any of these - the handle is in halves.
    HALVES = [
        f"{NONCE[:4]}[cite:]{NONCE[4:]}",
        f"{NONCE[:4]}[cite:x]{NONCE[4:]}",
        f"{NONCE[:4]} [cite:OTHER-1]{NONCE[4:]}",
        f"prose {NONCE[:2]}[cite:]{NONCE[2:]} and more",
    ]

    @pytest.mark.parametrize("raw", HALVES)
    def test_without_the_nonce_the_boundary_still_hands_it_over(self, raw):
        """The defect, executed rather than described.

        This is what every caller got before, and what a caller that cannot
        name the namespace still gets. If this stops finding the handle,
        these inputs have stopped reaching the property below.
        """
        public = scrub_namespace(raw, NONCE)
        assert public == raw, "the narrow scrub was supposed to find nothing"
        assert NONCE.lower() in reader_answer(public, [], []).content.lower()

    @pytest.mark.parametrize("raw", HALVES)
    def test_with_the_nonce_it_does_not(self, raw):
        public = scrub_namespace(raw, NONCE)
        cleaned = reader_answer(public, [], [], NONCE)
        assert NONCE.lower() not in cleaned.content.lower(), cleaned.content

    @pytest.mark.parametrize("raw", HALVES)
    def test_the_replacement_wrapper_carries_it_through(self, raw):
        """`replaced_answer` is what the runners actually call."""
        public = scrub_namespace(raw, NONCE)
        answer = replaced_answer(public, [], [], NONCE)
        content = answer.content if answer else ""
        assert NONCE.lower() not in content.lower(), content

    def test_the_boundary_agrees_with_the_finished_string_helper(self):
        """Two implementations of one rule, so they are compared."""
        for raw in self.HALVES + [
            f"see [cite:{NONCE}-1] here",
            f"bare {NONCE} here",
            "ordinary prose with no markers",
            "[ci[cite:]te:x]",
        ]:
            assert reader_answer(raw, [], [], NONCE).content == (
                reader_positions(raw, NONCE)[0]
            ), raw

    def test_an_offset_still_indexes_the_string_it_is_returned_with(self):
        """The third step moves text, so it has to move the coordinates."""
        raw = f"see [cite:{NONCE}-1] here and {NONCE[:4]}[cite:]{NONCE[4:]}"
        citations = [{"public_offset": len(raw), "source_id": "s1"}]
        cleaned = reader_answer(raw, [], citations, NONCE)
        assert NONCE.lower() not in cleaned.content.lower()
        for citation in cleaned.citations:
            assert 0 <= citation["public_offset"] <= len(cleaned.content)

    def test_clean_content_is_untouched(self):
        """The control: a cleanup that removed more would pass above too."""
        for raw in ("nothing to remove here", "a [cite:OTHER-1] marker"):
            assert reader_answer(raw, [], [], NONCE).content == (
                reader_answer(raw, [], []).content
            )
