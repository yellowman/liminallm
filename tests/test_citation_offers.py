"""Putting a turn's citation names into the text the model reads.

`test_citation_identity` asks who may be cited. These ask what the model is
actually shown, which is the other half of the same question: a handle nobody
is offered is never copied back, and a handle offered against the wrong run of
text is copied back correctly and still names the wrong document.

Nothing here mints. Every table is built first and read second, so a witness
that passed by allocating a handle at render time would fail.
"""

from __future__ import annotations

import pytest

from liminallm.service.citation_offers import (
    MARKER_SEPARATOR,
    handle_marker,
    label_passage,
    marker_cost,
)
from liminallm.service.citations import (
    CitationTable,
    build_citation_table,
    strip_citations,
)
from liminallm.service.provenance import GroundedSpan, SourceRegistry, binding
from liminallm.service.token_counting import TokenCounter

NONCE = "K7Q2ABCD"

ALPHA = "the inspection interval is 400 hours"
BETA = "the overhaul interval is 800 hours"
TEXT = f"[a.md]\n{ALPHA}\n\n[b.md]\n{BETA}\n"


def _table(*titles, nonce=NONCE):
    """A registry holding one passage per title, and a table over all of them.

    Real objects throughout: the eligibility this module reads is the
    relation the registry recorded, so a hand-built table would be a witness
    about my belief rather than about the table.
    """
    registry = SourceRegistry()
    bindings = []
    for title in titles:
        source = registry.register_source(
            kind="file", title=title, locator=f"/files/{title}"
        )
        evidence = registry.add_evidence(
            source.source_id, text=ALPHA if title == "a.md" else BETA
        )
        bindings.append(binding(source.source_id, evidence.evidence_id))
    return registry, bindings, build_citation_table(registry, bindings, nonce=nonce)


def _span(registry, table, source_id, text=TEXT, covered=None):
    """A span over `covered` in `text`, naming that source's own evidence."""
    body = covered if covered is not None else (
        ALPHA if source_id == "src_1" else BETA
    )
    start = text.index(body)
    (evidence_id,) = table.evidence_for(source_id)
    return GroundedSpan(
        start=start,
        end=start + len(body),
        source_id=source_id,
        evidence_id=evidence_id,
    )


class TestAMarkerFollowsThePassageItNames:
    def test_the_offer_is_written_the_one_agreed_way(self):
        """Spelled out, not built from the constants under test.

        Every other witness here interpolates `MARKER_SEPARATOR`, which makes
        them true of whatever the separator happens to be - including none at
        all, which runs the handle into the last word of the passage. The
        representation is a decision, so one witness states it literally.
        """
        registry, _bindings, table = _table("a.md")
        span = _span(registry, table, "src_1")

        labelled = label_passage(TEXT, [span], table)

        assert "400 hours [cite:K7Q2ABCD-1]\n" in labelled

    def test_the_offer_sits_at_the_end_of_the_run_it_covers(self):
        registry, _bindings, table = _table("a.md")
        span = _span(registry, table, "src_1")

        labelled = label_passage(TEXT, [span], table)

        marker = handle_marker(table.handle_for("src_1"))
        assert f"{ALPHA}{MARKER_SEPARATOR}{marker}" in labelled
        # And nothing else moved: the only difference is what was inserted.
        assert labelled.replace(f"{MARKER_SEPARATOR}{marker}", "", 1) == TEXT

    def test_two_passages_are_each_named_where_they_end(self):
        """The right-to-left property, stated as what a reader sees: the run
        in front of every marker is the passage that marker names."""
        registry, _bindings, table = _table("a.md", "b.md")
        spans = [
            _span(registry, table, "src_1"),
            _span(registry, table, "src_2"),
        ]

        labelled = label_passage(TEXT, spans, table)

        for source_id, body in (("src_1", ALPHA), ("src_2", BETA)):
            marker = handle_marker(table.handle_for(source_id))
            assert f"{body}{MARKER_SEPARATOR}{marker}" in labelled, source_id

    def test_the_order_the_spans_arrive_in_does_not_change_the_result(self):
        """Insertion is by offset, not by argument order. A producer that
        recorded its spans out of order would otherwise place every marker
        after the first one at a stale offset."""
        registry, _bindings, table = _table("a.md", "b.md")
        spans = [
            _span(registry, table, "src_1"),
            _span(registry, table, "src_2"),
        ]

        assert label_passage(TEXT, spans, table) == label_passage(
            TEXT, list(reversed(spans)), table
        )

    def test_one_source_read_in_two_places_is_named_once(self):
        """Not two handles. The model is shown one document under one name,
        so a claim resting on either half cites the same source."""
        registry, _bindings, table = _table("a.md")
        first = _span(registry, table, "src_1", covered="the inspection")
        second = _span(registry, table, "src_1", covered="400 hours")

        labelled = label_passage(TEXT, [first, second], table)

        marker = handle_marker(table.handle_for("src_1"))
        assert labelled.count(marker) == 2
        assert f"the inspection{MARKER_SEPARATOR}{marker}" in labelled
        assert f"400 hours{MARKER_SEPARATOR}{marker}" in labelled


class TestWhatEarnsNoNameGetsNone:
    """Never a guessed placement. A span this layer cannot honour is left
    unlabelled, and the passage is still shown - under-offering costs a
    citation, mis-offering grants one to the wrong document."""

    def test_a_source_the_table_does_not_cite_is_not_named(self):
        registry, bindings, _full = _table("a.md", "b.md")
        # Only the first source grounded the answer.
        table = build_citation_table(registry, bindings[:1], nonce=NONCE)
        span = GroundedSpan(
            start=TEXT.index(BETA),
            end=TEXT.index(BETA) + len(BETA),
            source_id="src_2",
            evidence_id="ev_2",
        )

        assert label_passage(TEXT, [span], table) == TEXT

    def test_a_passage_filed_under_another_source_is_not_named(self):
        """The relation, not just the source. `src_1` has a handle and `ev_2`
        is real, and putting the first one's name on the second one's text
        would show the model that the two belong together."""
        registry, _bindings, table = _table("a.md", "b.md")
        (foreign,) = table.evidence_for("src_2")
        span = GroundedSpan(
            start=TEXT.index(ALPHA),
            end=TEXT.index(ALPHA) + len(ALPHA),
            source_id="src_1",
            evidence_id=foreign,
        )

        assert label_passage(TEXT, [span], table) == TEXT

    @pytest.mark.parametrize(
        "start,end",
        [
            (0, len(TEXT) + 1),  # runs off the end
            (len(TEXT) - 4, 2),  # backwards
            (5, 5),  # covers nothing
            (-1, 8),  # before the start
        ],
    )
    def test_a_span_that_does_not_describe_this_text_is_dropped(self, start, end):
        registry, _bindings, table = _table("a.md")
        (evidence_id,) = table.evidence_for("src_1")
        span = GroundedSpan(
            start=start, end=end, source_id="src_1", evidence_id=evidence_id
        )

        assert label_passage(TEXT, [span], table) == TEXT

    def test_a_turn_with_no_handles_shows_no_markers(self):
        empty = CitationTable(nonce=NONCE)
        span = GroundedSpan(start=0, end=6, source_id="src_1", evidence_id="ev_1")

        assert label_passage(TEXT, [span], empty) == TEXT

    def test_evidence_filed_under_a_source_with_no_handle_names_nothing(self):
        """A table whose two maps disagree.

        The builder and the extender cannot produce one - `_require_consistent`
        refuses evidence filed under a source the table does not cite - so this
        is a table arriving from somewhere else. That is the case an offer gate
        must not inherit an upstream invariant for: reading the passage
        relation alone would find `ev_1` filed under `src_1` and label it with
        a name the turn never issued.
        """
        table = CitationTable(nonce=NONCE, evidence={"src_1": ("ev_1",)})
        span = GroundedSpan(
            start=TEXT.index(ALPHA),
            end=TEXT.index(ALPHA) + len(ALPHA),
            source_id="src_1",
            evidence_id="ev_1",
        )

        assert label_passage(TEXT, [span], table) == TEXT


class TestTheRepresentationIsOneTheRestOfTheSystemAgreesWith:
    def test_the_reader_cleanup_takes_the_offer_back_out(self):
        """The separator is chosen for this. `strip_citations` removes a
        marker together with the spacing in front of it, so a labelled
        passage returns to exactly the text it was built from - which is what
        S6 will run over anything a reader sees."""
        registry, _bindings, table = _table("a.md", "b.md")
        spans = [
            _span(registry, table, "src_1"),
            _span(registry, table, "src_2"),
        ]

        labelled = label_passage(TEXT, spans, table)

        assert labelled != TEXT
        assert strip_citations(labelled) == TEXT

    def test_the_marker_is_the_shape_the_validator_reads(self):
        """One spelling across the system: what is offered is what a model
        copying it produces, and what the validator resolves."""
        from liminallm.service.citations import validate_citations

        registry, _bindings, table = _table("a.md")
        span = _span(registry, table, "src_1")

        labelled = label_passage(TEXT, [span], table)

        found = validate_citations(labelled, table)
        assert [item.source_id for item in found] == ["src_1"]


class TestNothingIsEditedInPlace:
    def test_the_inputs_come_back_as_they_went_in(self):
        """The passage records, the transcript and the base prompt are read
        again by the second materialization of the same assembly. One that
        edited them would make the two disagree."""
        registry, _bindings, table = _table("a.md")
        span = _span(registry, table, "src_1")
        spans = [span]

        labelled = label_passage(TEXT, spans, table)

        assert labelled != TEXT
        assert spans == [span]
        assert (span.start, span.end, span.source_id) == (
            TEXT.index(ALPHA),
            TEXT.index(ALPHA) + len(ALPHA),
            "src_1",
        )

    def test_labelling_twice_gives_the_same_answer(self):
        registry, _bindings, table = _table("a.md", "b.md")
        spans = [
            _span(registry, table, "src_1"),
            _span(registry, table, "src_2"),
        ]

        first = label_passage(TEXT, spans, table)
        second = label_passage(TEXT, spans, table)

        assert first == second


class _Tokenizer:
    """A real tokenizer's shape: `encode` returns the ids of the text.

    Three characters per token, so its answer for a marker is well under the
    marker's length and the two rules are distinguishable.
    """

    def encode(self, text):
        return list(range(max(1, len(text) // 3)))


class TestAHandleIsNotBudgetedAsProse:
    def test_an_estimating_counter_charges_at_least_the_markers_length(self):
        """The measurement said a marker is 18 characters and the fallback
        estimator prices it at 5, because it prices everything at four
        characters per token. A random-looking run of mixed-case letters and
        digits is the text that estimate is least true for, and being wrong
        here is a prompt the provider refuses after the offers are already
        committed."""
        counter = TokenCounter(model="a-model-with-no-tokenizer")
        marker = handle_marker(f"{NONCE}-1")
        assert not counter.exact

        assert counter.count(marker) < len(marker)
        assert marker_cost(counter, marker) == len(marker)

    def test_a_real_tokenizer_is_believed(self):
        """A measured count is not improved by a guess about it."""
        counter = TokenCounter(model="local", tokenizer=_Tokenizer())
        marker = handle_marker(f"{NONCE}-1")
        assert counter.exact

        assert marker_cost(counter, marker) == counter.count(marker)
        assert marker_cost(counter, marker) < len(marker)
