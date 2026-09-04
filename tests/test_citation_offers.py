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


def _one_source_two_passages(nonce=NONCE):
    """One document read in two places: two passages of one file.

    What the shared-handle rule is actually about. Two spans over parts of a
    single passage is a different thing, and under the placement rule neither
    of them covers what it names.
    """
    registry = SourceRegistry()
    source = registry.register_source(
        kind="file", title="a.md", locator="/files/a.md"
    )
    bindings = [
        binding(source.source_id, registry.add_evidence(
            source.source_id, text=body).evidence_id)
        for body in (ALPHA, BETA)
    ]
    return registry, build_citation_table(registry, bindings, nonce=nonce)


def _handed_table(*, cites, evidence, nonce=NONCE):
    """A table built by hand rather than by the builder.

    `build_citation_table` and `extend_citation_table` both resolve every
    entry against the registry, so neither produces one of these. That is the
    point: the offer gate is handed a table and must not assume the builder
    made it - the same reason the builder itself re-checks a table it inherits.
    """
    handle = f"{nonce}-1"
    return CitationTable(
        nonce=nonce,
        by_handle={handle: cites},
        by_source={cites: handle},
        evidence={cites: tuple(evidence)},
    )


def _span(registry, table, source_id, text=TEXT, covered=None, evidence_id=None):
    """A span over `covered` in `text`, naming that source's own evidence."""
    body = covered if covered is not None else (
        ALPHA if source_id == "src_1" else BETA
    )
    start = text.index(body)
    if evidence_id is None:
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

        labelled = label_passage(TEXT, [span], table, registry)

        assert "400 hours [cite:K7Q2ABCD-1]\n" in labelled

    def test_the_offer_sits_at_the_end_of_the_run_it_covers(self):
        registry, _bindings, table = _table("a.md")
        span = _span(registry, table, "src_1")

        labelled = label_passage(TEXT, [span], table, registry)

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

        labelled = label_passage(TEXT, spans, table, registry)

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

        assert label_passage(TEXT, spans, table, registry) == label_passage(
            TEXT, list(reversed(spans)), table, registry
        )

    def test_one_source_read_in_two_places_is_named_once(self):
        """Not two handles. The model is shown one document under one name,
        so a claim resting on either passage cites the same source."""
        registry, table = _one_source_two_passages()
        first, second = table.evidence_for("src_1")
        spans = [
            _span(registry, table, "src_1", covered=ALPHA, evidence_id=first),
            _span(registry, table, "src_1", covered=BETA, evidence_id=second),
        ]

        labelled = label_passage(TEXT, spans, table, registry)

        marker = handle_marker(table.handle_for("src_1"))
        assert labelled.count(marker) == 2
        assert f"{ALPHA}{MARKER_SEPARATOR}{marker}" in labelled
        assert f"{BETA}{MARKER_SEPARATOR}{marker}" in labelled


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

        assert label_passage(TEXT, [span], table, registry) == TEXT

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

        assert label_passage(TEXT, [span], table, registry) == TEXT

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

        assert label_passage(TEXT, [span], table, registry) == TEXT

    def test_a_turn_with_no_handles_shows_no_markers(self):
        registry, _bindings, built = _table("a.md")
        empty = CitationTable(nonce=NONCE)
        span = _span(registry, built, "src_1")

        assert label_passage(TEXT, [span], empty, registry) == TEXT

    def test_evidence_filed_under_a_source_with_no_handle_names_nothing(self):
        """A table whose two maps disagree.

        The builder and the extender cannot produce one - `_require_consistent`
        refuses evidence filed under a source the table does not cite - so this
        is a table arriving from somewhere else. That is the case an offer gate
        must not inherit an upstream invariant for: reading the passage
        relation alone would find the passage filed under `src_1` and label it
        with a name the turn never issued.

        Everything else about this span is correct - the registry holds the
        passage, it belongs to `src_1`, and the offsets cover it - so the
        missing handle is the only thing that can refuse it.
        """
        registry, _bindings, built = _table("a.md")
        (evidence_id,) = built.evidence_for("src_1")
        table = CitationTable(nonce=NONCE, evidence={"src_1": (evidence_id,)})
        span = _span(registry, built, "src_1")

        assert label_passage(TEXT, [span], table, registry) == TEXT


class TestTheSpanMustCoverThePassageItNames:
    """The relation being valid and the offsets being right are two different
    questions, and a span can pass the first while failing the second.

    `GroundedSpan` is parent-owned and the producers are witnessed, so this is
    not a worker reaching anything. It is the offer gate declining to inherit
    a placement invariant it does not own, at the last point where a wrong
    offset is still only a wrong offset rather than a citation the model was
    taught.
    """

    def test_a_valid_relation_over_another_sources_text_is_not_named(self):
        """The hole this class closes. Individually:

            the offsets are inside the text      yes
            `src_1` has a handle                 yes
            its passage is eligible for `src_1`  yes

        and the run those offsets cover is `src_2`'s sentence. Labelling it
        shows the model 800 hours under the name of the document that says
        400 - a legitimate source name on the wrong text, which is exactly
        what this module says it prevents.
        """
        registry, _bindings, table = _table("a.md", "b.md")
        (own,) = table.evidence_for("src_1")
        span = GroundedSpan(
            start=TEXT.index(BETA),
            end=TEXT.index(BETA) + len(BETA),
            source_id="src_1",
            evidence_id=own,
        )

        assert label_passage(TEXT, [span], table, registry) == TEXT

    def test_a_span_covering_half_its_passage_is_not_named(self):
        """A truncated span names a claim the passage only partly supports,
        and the marker would sit in the middle of the sentence it came
        from."""
        registry, _bindings, table = _table("a.md")
        span = _span(registry, table, "src_1", covered="the inspection interval")

        assert label_passage(TEXT, [span], table, registry) == TEXT

    def test_an_envelope_around_the_passage_still_names_it(self):
        """Containment, not equality. Every producer wraps its evidence in
        text of the parent's own - an untrusted-data envelope, a `source:`
        header, a result number - and the span covers the rendered run."""
        registry, _bindings, table = _table("a.md")
        wrapped = f"[a.md]\n{ALPHA}"
        span = _span(registry, table, "src_1", covered=wrapped)

        labelled = label_passage(TEXT, [span], table, registry)

        marker = handle_marker(table.handle_for("src_1"))
        assert f"{wrapped}{MARKER_SEPARATOR}{marker}" in labelled

    def test_a_passage_with_no_text_is_named_nowhere(self):
        """`""` is inside every string, so an empty passage would be
        placeable at any valid offset - the containment test passes and the
        marker lands wherever the span happened to point.

        Reachable through the ordinary route, not a hand-built table.
        `add_evidence` requires a `str` and not a non-empty one, so an empty
        passage is registered, binds like any other, and `build_citation_table`
        gives its source a handle.
        """
        registry = SourceRegistry()
        source = registry.register_source(
            kind="file", title="a.md", locator="/files/a.md"
        )
        empty = registry.add_evidence(source.source_id, text="")
        table = build_citation_table(
            registry,
            [binding(source.source_id, empty.evidence_id)],
            nonce=NONCE,
        )
        assert table.handle_for(source.source_id), "the source earned a handle"
        span = GroundedSpan(
            start=TEXT.index(ALPHA),
            end=TEXT.index(ALPHA) + len(ALPHA),
            source_id=source.source_id,
            evidence_id=empty.evidence_id,
        )

        assert label_passage(TEXT, [span], table, registry) == TEXT

    def test_a_passage_the_registry_does_not_hold_is_not_named(self):
        """A table that says a passage is eligible, and a registry with no
        such passage.

        The table alone cannot answer this: it stores evidence ids, so it can
        list one that resolves to nothing. Reading only the table would find
        the id eligible for a source that has a handle and place a marker
        against text nothing was ever recorded for.
        """
        registry, _bindings, _built = _table("a.md")
        table = _handed_table(cites="src_1", evidence=("ev_404",))
        span = GroundedSpan(
            start=TEXT.index(ALPHA),
            end=TEXT.index(ALPHA) + len(ALPHA),
            source_id="src_1",
            evidence_id="ev_404",
        )

        assert label_passage(TEXT, [span], table, registry) == TEXT

    def test_a_passage_the_table_files_under_the_wrong_source_is_not_named(
        self,
    ):
        """The registry is the authority on which source a passage belongs to.

        Here the table has been handed a passage of `src_2` filed under
        `src_1`, so the eligibility check agrees with itself and is wrong. The
        offsets are right for that passage, which is what makes it dangerous:
        the marker would land on the sentence the passage really is, under the
        other document's name.
        """
        registry, _bindings, built = _table("a.md", "b.md")
        (foreign,) = built.evidence_for("src_2")
        table = _handed_table(cites="src_1", evidence=(foreign,))
        span = GroundedSpan(
            start=TEXT.index(BETA),
            end=TEXT.index(BETA) + len(BETA),
            source_id="src_1",
            evidence_id=foreign,
        )

        assert label_passage(TEXT, [span], table, registry) == TEXT


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

        labelled = label_passage(TEXT, spans, table, registry)

        assert labelled != TEXT
        assert strip_citations(labelled) == TEXT

    def test_the_marker_is_the_shape_the_validator_reads(self):
        """One spelling across the system: what is offered is what a model
        copying it produces, and what the validator resolves."""
        from liminallm.service.citations import validate_citations

        registry, _bindings, table = _table("a.md")
        span = _span(registry, table, "src_1")

        labelled = label_passage(TEXT, [span], table, registry)

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

        labelled = label_passage(TEXT, spans, table, registry)

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

        first = label_passage(TEXT, spans, table, registry)
        second = label_passage(TEXT, spans, table, registry)

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
