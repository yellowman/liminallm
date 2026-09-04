"""Showing the model which passage is which.

`citations.py` owns identity: which sources this turn may cite, what each is
called, and whether a marker in an answer is one this turn issued. This module
owns the other half - putting those names into the text the model reads, so
that copying one back is a thing it can do.

The two are kept apart because they fail differently. An identity mistake
grants authority that was never earned. A materialization mistake shows the
model a marker in the wrong place, and the model then cites the wrong source
correctly. Both are wrong; only the first is a hole in the gate, and mixing
prompt mechanics into the gate is how a reader stops being able to see which
one they are reading.

Nothing here mints a handle. It is handed a table and it reads from it, so a
source without a handle stays unciteable no matter how the passage is
rendered.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

from liminallm.service.citations import CitationTable
from liminallm.service.provenance import GroundedSpan

#: What separates a passage's text from the marker offered for it. One space,
#: and the marker inline at the end of the grounded run rather than on a line
#: of its own.
#:
#: Two reasons, both about the rest of the system rather than about looks. The
#: model has to write this token after a claim in prose, so the demonstration
#: it is given should be the shape it is being asked to produce. And
#: `strip_citations` removes `[ \t]*` before a marker, so a space-separated one
#: is taken back out cleanly by the reader-side cleanup that already exists,
#: while a newline-separated one would leave the blank line behind.
MARKER_SEPARATOR = " "


def handle_marker(handle: str) -> str:
    """The model-facing form of one handle."""
    return f"[cite:{handle}]"


def marker_cost(counter: Any, marker: str) -> int:
    """What to charge the prompt budget for one marker.

    A handle is a random-looking run of mixed-case letters and digits, which
    is not what the fallback estimator was calibrated on: it prices text at
    four characters per token, and a token that fragments harder than prose
    would then be budgeted as though it were prose. Being wrong here does not
    show up as a bad estimate, it shows up as a prompt the provider refuses
    after the offers have already been committed.

    So when the counter is estimating, the marker is charged at no less than
    its own length in bytes - one token per character, the worst a tokenizer
    can do with it. The rest of the prompt keeps the calibrated estimate;
    only the new random-looking token gets the floor.

    When the counter owns a real tokenizer its answer is the answer, and no
    floor is applied: a measured count is not improved by a guess about it.

    Markers are ASCII by construction - the nonce alphabet, digits, and
    `[cite:` - so the byte length is the character length.
    """
    counted = counter.count(marker)
    if getattr(counter, "exact", False):
        return counted
    return max(counted, len(marker.encode("ascii")))


def _eligible(table: CitationTable, span: GroundedSpan) -> Optional[str]:
    """The handle this span may be labelled with, if it may be labelled.

    Both halves, not just the source. A span naming a source that has a
    handle, beside a passage filed under a different source, is exactly the
    relation `build_citation_table` refuses to grant authority for; accepting
    it here would put the first source's name on the second one's text and
    reach the model as a demonstration that the two go together.
    """
    handle = table.handle_for(span.source_id)
    if handle is None:
        return None
    if span.evidence_id not in table.evidence_for(span.source_id):
        return None
    return handle


def _placeable(text: str, span: GroundedSpan) -> bool:
    """Whether this span describes a run of `text` a marker can follow.

    A span that runs past the end of the string, or backwards, or covers no
    characters, is not a position - it is a record that does not describe this
    text. Such a span is dropped rather than clamped: clamping puts the marker
    at whatever offset survives, which attributes a passage the span never
    covered, and that is the failure this whole layer exists to prevent.

    Dropped rather than raised, because the blast radius is one marker. The
    passage is still shown, the model simply is not offered a name for that
    piece of it, and under-offering is the safe direction.
    """
    return 0 <= span.start < span.end <= len(text)


def label_passage(
    text: str,
    spans: Sequence[GroundedSpan],
    table: CitationTable,
) -> str:
    """`text` with a citation marker after each passage that has earned one.

    This is the offer. Every marker in the returned string is a handle this
    turn issued for the source whose evidence covers the run of text directly
    in front of it, so a model that copies one is naming the passage it was
    reading.

    Right to left, so that inserting a marker cannot move a position not yet
    used. Left to right needs every later offset shifted by the width of every
    marker already written, which is a second calculation that has to agree
    with the first, and the two disagree the first time a span is skipped.

    One handle per source, from the table, so the same document cited from two
    passages is named identically in both. A span whose relation is not
    eligible, and one whose offsets do not describe this text, get no marker at
    all - never a guessed position.

    Returns a new string and reads its inputs. The passage records, the
    transcript entries and the base prompt snapshot are all parent state that
    later stages read again; a materializer that edited them in place would
    make the second materialization of one assembly differ from the first.
    """
    placements: List[Tuple[int, str]] = []
    for span in spans:
        if not _placeable(text, span):
            continue
        handle = _eligible(table, span)
        if handle is None:
            continue
        placements.append((span.end, handle_marker(handle)))
    # Descending by offset. `reverse=True` on the offset alone is stable, so
    # two markers at one position keep the order their spans were recorded in
    # - reversed, since they are applied from the right.
    placements.sort(key=lambda item: item[0], reverse=True)
    labelled = text
    for offset, marker in placements:
        labelled = (
            labelled[:offset] + MARKER_SEPARATOR + marker + labelled[offset:]
        )
    return labelled
