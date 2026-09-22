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
    reader_positions,
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
        """
        reach = {}
        for depth in (2, 4, 8, 16):
            text = "[ci" * depth + "[cite:]" + "te:]" * depth
            repeated, _ = _scrub_text(text, CITATION_STRIP_RE)
            assert repeated == "", (depth, repeated)
            reach[depth] = 3 * depth
        assert reach[16] > reach[2], reach
        assert reach[16] == 48, reach

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
