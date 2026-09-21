"""A marker carrying several handles is one match while it is streamed too.

`_namespace_pattern` removes `[cite:H-1, H-2]` whole. The streaming reader
implements that same language by hand, and it used to implement only part of
it: it settled each handle on its own and handed the keyword down, which left
`[cite:,]` for the bounded reader-side stripper to sweep up afterwards.

The sweep is bounded at `MAX_CITATION_MARKER_BODY`, and the residue is one
comma per handle. So the two halves agreed only while the marker held fewer
handles than that bound. Past it the stripper had to leave the comma run as
ordinary text, `finish` found the stream and the whole-string scrub disagreed
about what the answer was, and the turn ended with no answer at all.

Two properties here, both regressions:

* the reader agrees with the whole-string scrub on a merged marker of any
  size, and
* a stream abandoned while it holds one still releases no live handle.

The second is why this was not done earlier. Holding a handle across a comma
in order to recognise the merged form would mean a cancelled turn could flush
a live handle to the reader. It is not held: each handle is still removed as
it is settled, and what an open marker keeps is the keyword, the commas and
horizontal space. The test below is the check on that claim rather than the
claim itself.
"""

from __future__ import annotations

import pytest

from liminallm.service.citation_stream import CanonicalCitationStream
from liminallm.service.citations import (
    CITATION_RE,
    MAX_CITATION_MARKER_BODY,
    reader_positions,
)

NONCE = "K7Q2ABCD"

#: The first handle count whose comma residue is longer than the bound, and
#: the last that fits under it. `n` handles leave `n - 1` commas, and the
#: stripper's body is those commas alone, so 65 handles is the last body it
#: can still match and 66 is the first it cannot.
OVER_THE_BOUND = MAX_CITATION_MARKER_BODY + 2
UNDER_THE_BOUND = MAX_CITATION_MARKER_BODY + 1


def merged(handles: int, separator: str = ", ") -> str:
    return "[cite: " + separator.join(f"{NONCE}-1" for _ in range(handles)) + "]"


def stream(text: str, *, chunks=None, **kwargs):
    """Push `text` and finish, returning what the reader released."""
    reader = CanonicalCitationStream(NONCE, **kwargs)
    out = [reader.push(chunk) for chunk in (chunks or list(text))]
    tail, origins = reader.finish()
    return "".join(out) + tail, origins, reader


class TestAMergedMarkerOfAnySizeAgreesWithTheFinishedScrub:
    @pytest.mark.parametrize(
        "handles", [1, 2, 3, 10, UNDER_THE_BOUND, OVER_THE_BOUND, 120]
    )
    @pytest.mark.parametrize("separator", [",", ", ", " , ", ",\t"])
    def test_the_whole_marker_goes(self, handles, separator):
        text = f"Before {merged(handles, separator)} after."
        expected, expected_origins = reader_positions(text, NONCE)
        assert expected == "Before after."
        released, origins, reader = stream(text)
        assert released == expected
        assert origins == expected_origins
        assert reader.intact()

    def test_the_residue_really_does_pass_the_bound(self):
        """The reason the large case is not the same test as the small one.

        What the reader used to leave behind is `[cite:` plus one comma per
        handle and a bracket, and the reader-side stripper is what had to
        sweep it. So the two counts above are only meaningful if the
        stripper can still match one residue and not the other - which is
        built and matched here rather than asserted from arithmetic. An
        earlier version of this compared two constants and reduced to
        `MAX + 1 > MAX`, which is true whatever the bound is and therefore
        said nothing about either of them.
        """
        def residue(handles):
            return "[cite:" + "," * (handles - 1) + "]"

        assert CITATION_RE.fullmatch(residue(UNDER_THE_BOUND)), (
            f"{UNDER_THE_BOUND} handles leave a residue the stripper cannot "
            f"match, so the small case is already past the bound and the "
            f"two parametrizations test the same thing"
        )
        assert not CITATION_RE.fullmatch(residue(OVER_THE_BOUND)), (
            f"{OVER_THE_BOUND} handles leave a residue the stripper can "
            f"still match, so the large case does not reach the defect"
        )

    @pytest.mark.parametrize(
        "text",
        [
            f"[cite:{NONCE}-1, OTHER-2]",
            f"[cite:{NONCE}-1 {NONCE}-2]",
            f"[cite:{NONCE}-1,, {NONCE}-2]",
            f"[cite:{NONCE}-1,]",
            f"[cite:, {NONCE}-1]",
            f"[cite:[cite:{NONCE}-1 {NONCE}-2]",
            f"[cite:{NONCE}-1, {NONCE}-2",
        ],
        ids=[
            "one handle is not ours", "no comma between them", "two commas",
            "trailing comma", "leading comma", "keyword twice", "never closed",
        ],
    )
    def test_what_is_not_a_merged_marker_is_not_treated_as_one(self, text):
        """The near-misses, which is where a hand-written grammar goes wrong.

        Each of these fails the pattern's merged branch, so its handles come
        out one at a time and whatever is left is the reader-side stripper's
        problem. The reader has to reach the same answer by a different
        route, and the answer - not the route - is what is compared.
        """
        expected, expected_origins = reader_positions(text, NONCE)
        released, origins, _reader = stream(text)
        assert released == expected
        assert origins == expected_origins

    def test_a_chunk_boundary_anywhere_inside_it_changes_nothing(self):
        """A provider's tokenizer cuts wherever it likes."""
        text = f"a {merged(3)} b"
        expected, _origins = reader_positions(text, NONCE)
        for cut in range(len(text) + 1):
            released, _o, _r = stream(text, chunks=[text[:cut], text[cut:]])
            assert released == expected, f"cut at {cut}"


class TestAnAbandonedStreamReleasesNoLiveHandle:
    """The property that decides whether the reader may hold across a comma.

    `fail` is what a cancelled turn and a backend error both reach. It flushes
    what every pass was holding, because those bytes can no longer become
    anything else and the caller builds its partial reply out of what went
    out. So whatever an open marker holds arrives at the reader.
    """

    def _abandon(self, text: str, upto: int) -> str:
        """Everything the reader released, the flush included.

        Read off `released` rather than accumulated from the return values,
        which is not a detail: a disagreement between the reader and the
        whole-string scrub raises out of the flush, and it raises *after*
        the flush has released its text and before `fail` can return it. A
        caller that adds up return values therefore cannot see exactly the
        bytes this is looking for. Measured: collecting them that way made
        this class pass against a build that leaked a handle at fifteen
        separate abandonment points.

        The disagreement itself is the other class's finding, so it is
        swallowed here and this one measures one property.
        """
        reader = CanonicalCitationStream(NONCE)
        for character in text[:upto]:
            reader.push(character)
        try:
            reader.fail()
        except ValueError:
            pass
        assert not reader.intact(), "a failed stream must vouch for nothing"
        return reader.released

    @pytest.mark.parametrize("handles", [2, 3, OVER_THE_BOUND])
    def test_abandoning_at_every_point_inside_a_marker_leaks_nothing(
        self, handles
    ):
        text = f"Before {merged(handles)} after."
        for upto in range(len(text) + 1):
            released = self._abandon(text, upto)
            assert NONCE.lower() not in released.lower(), (
                f"a handle reached the reader when the stream was abandoned "
                f"after {upto} characters: {released!r}"
            )

    def test_the_check_can_see_a_handle_that_does_reach_the_reader(self):
        """The control.

        A stream told the nonce is not this turn's syntax releases it as
        ordinary text, which is correct and is also the state the assertion
        above is looking for. If this did not find it there, that assertion
        would be reading nothing.
        """
        reader = CanonicalCitationStream(NONCE, scrub_namespace=False)
        released = reader.push(f"the word {NONCE} here")
        released += reader.fail()
        assert NONCE.lower() in released.lower()
