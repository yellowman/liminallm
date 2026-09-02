"""Which citations a turn issued, and which it must refuse.

The conservation boundary: a citation in model output resolves only to a
source this turn's bindings already grounded on. Everything here is about the
two ways that can fail - a handle that names something the turn did not
ground on, and a handle that looks right because a previous turn or a
retrieved document wrote it.
"""

from __future__ import annotations

import pytest

from liminallm.service.citations import (
    ALPHABET,
    CITATION_RE,
    MIN_NONCE_BITS,
    NONCE_LENGTH,
    CitationTable,
    build_citation_table,
    mint_nonce,
    strip_citations,
    validate_citations,
)
from liminallm.service.provenance import SourceRegistry, binding


def _turn(*titles, nonce="K7Q2"):
    """A registry holding these sources, and a table over all of them."""
    registry = SourceRegistry()
    bindings = []
    for index, title in enumerate(titles):
        source = registry.register_source(
            kind="file", title=title, locator=f"/files/{title}"
        )
        evidence = registry.add_evidence(source.source_id, text=f"passage {index}")
        bindings.append(binding(source.source_id, evidence.evidence_id))
    return registry, bindings, build_citation_table(registry, bindings, nonce=nonce)


class TestOnlyWhatGroundedTheAnswerIsCitable:
    def test_a_registered_source_is_cited(self):
        _registry, _bindings, table = _turn("manual.md")
        handle = table.handle_for("src_1")
        found = validate_citations(f"The interval is 400 hours [cite:{handle}].", table)

        assert [(c.handle, c.source_id) for c in found] == [(handle, "src_1")]

    def test_a_consulted_but_unbound_source_gets_no_handle(self):
        """The registry is what the turn consulted - including what the prompt
        budget dropped and what a failed node retrieved. Only bindings say
        what may support this answer."""
        registry = SourceRegistry()
        kept = registry.register_source(kind="file", title="kept.md", locator="/a")
        dropped = registry.register_source(kind="file", title="dropped.md", locator="/b")
        evidence = registry.add_evidence(kept.source_id, text="passage")
        table = build_citation_table(
            registry, [binding(kept.source_id, evidence.evidence_id)], nonce="K7Q2"
        )

        assert table.handle_for(kept.source_id) is not None
        assert table.handle_for(dropped.source_id) is None, (
            "a source the answer never rested on became citable"
        )

    def test_an_unknown_handle_is_not_a_citation(self):
        _registry, _bindings, table = _turn("manual.md")
        assert validate_citations("Claimed [cite:K7Q2-9].", table) == []

    def test_a_binding_the_registry_cannot_resolve_gets_no_handle(self):
        """A citation nobody can follow is worse than a missing one: there is
        no title, kind or locator to show for a source that is not there."""
        registry = SourceRegistry()
        table = build_citation_table(
            registry, [binding("src_404", "ev_404")], nonce="K7Q2"
        )
        assert not table
        assert validate_citations("Claimed [cite:K7Q2-1].", table) == []


class TestABindingIsCheckedAsARelation:
    """The gate that grants citation authority is the wrong place to inherit
    an upstream invariant. Today's producers do not emit these shapes; that is
    a fact about today's producers, not a property of this function."""

    @staticmethod
    def _two_sources():
        registry = SourceRegistry()
        first = registry.register_source(kind="file", title="a.md", locator="/a")
        second = registry.register_source(kind="file", title="b.md", locator="/b")
        return (
            registry,
            first,
            second,
            registry.add_evidence(first.source_id, text="from a"),
            registry.add_evidence(second.source_id, text="from b"),
        )

    def test_a_real_source_with_no_such_evidence_gets_no_handle(self):
        registry, first, _second, _ea, _eb = self._two_sources()
        table = build_citation_table(
            registry, [binding(first.source_id, "ev_404")], nonce="K7Q2"
        )
        assert not table, dict(table.by_handle)

    def test_evidence_belonging_to_another_source_gets_no_handle(self):
        """The worst of the two: a real source paired with a real passage that
        is not its own would attach a citation to text it never contained."""
        registry, first, _second, _ea, from_b = self._two_sources()
        table = build_citation_table(
            registry, [binding(first.source_id, from_b.evidence_id)], nonce="K7Q2"
        )
        assert not table, dict(table.by_handle)

    def test_a_bad_binding_does_not_spoil_a_good_one(self):
        """Failing closed per binding, not per source: the source is still
        citable through the binding that resolves, and only that one is
        eligible under it."""
        registry, first, _second, from_a, from_b = self._two_sources()
        table = build_citation_table(
            registry,
            [
                binding(first.source_id, "ev_404"),
                binding(first.source_id, from_b.evidence_id),
                binding(first.source_id, from_a.evidence_id),
            ],
            nonce="K7Q2",
        )
        assert len(table.by_handle) == 1, dict(table.by_handle)
        assert table.evidence_for(first.source_id) == (from_a.evidence_id,)

    def test_a_binding_with_no_evidence_grants_nothing(self):
        registry, first, _second, _ea, _eb = self._two_sources()
        table = build_citation_table(
            registry, [{"source_id": first.source_id, "evidence_id": ""}], nonce="K7Q2"
        )
        assert not table


class TestTheTableCannotBeEditedAfterItIsBuilt:
    """A frozen dataclass does not freeze what its attributes point at. This
    is the object the next stage makes authority, so a retained reference must
    not be able to add a handle to it."""

    @pytest.mark.parametrize("attribute", ["by_handle", "by_source", "evidence"])
    def test_no_mapping_can_be_written_through(self, attribute):
        _registry, _bindings, table = _turn("manual.md")
        with pytest.raises(TypeError):
            getattr(table, attribute)["forged"] = "src_1"

    def test_an_empty_table_is_frozen_too(self):
        """Otherwise the default is the one writable table in the system."""
        with pytest.raises(TypeError):
            CitationTable(nonce="K7Q2").by_handle["forged"] = "src_1"


class TestAHandleFromAnotherTurnDoesNotResolve:
    """`source_id` restarts at `src_1` in every registry, and history is
    replayed verbatim into later prompts. A handle built from the internal id
    would let yesterday's citation name today's unrelated document."""

    def test_yesterdays_handle_is_refused_today(self):
        _r1, _b1, yesterday = _turn("rates-2024.md", nonce="AAAA")
        _r2, _b2, today = _turn("rates-2025.md", nonce="BBBB")
        stale = yesterday.handle_for("src_1")

        # Both turns minted `src_1`; only the nonce tells them apart.
        assert yesterday.source_for(stale) == "src_1"
        assert today.source_for(stale) is None
        assert validate_citations(f"Still 400 hours [cite:{stale}].", today) == []

    def test_a_source_authored_handle_does_not_resolve(self):
        """Retrieved text reaches the model as data, and a hostile page can
        write a citation marker as easily as any other string. It cannot write
        this turn's nonce, which is minted after the corpus was."""
        _registry, _bindings, table = _turn("manual.md", nonce="K7Q2")
        answer = "The page said [cite:ZZZZ-1] and also [cite:src_1-1]."
        assert validate_citations(answer, table) == []

    def test_two_turns_do_not_share_a_nonce(self):
        assert len({mint_nonce() for _ in range(200)}) > 1

    def test_the_namespace_is_too_large_to_guess(self):
        """The property that matters is the size of the space, not that the
        nonces differ. A retrieved page is attacker-controlled text that can
        carry a thousand candidate markers and pay nothing for a miss, so a
        namespace it could plausibly cover is not a boundary at all."""
        assert len(ALPHABET) ** NONCE_LENGTH >= 2 ** MIN_NONCE_BITS, (
            f"{len(ALPHABET)}**{NONCE_LENGTH} is under the {MIN_NONCE_BITS}-bit floor"
        )
        assert len(set(ALPHABET)) == len(ALPHABET), "a repeated symbol shrinks it"


class TestTheModelCannotNameASourceItself:
    def test_a_filename_is_not_a_citation(self):
        _registry, _bindings, table = _turn("foo.md")
        assert validate_citations("As foo.md says, 400 hours.", table) == []

    @pytest.mark.parametrize(
        "answer",
        [
            "[cite K7Q2-1]",      # no colon
            "cite:K7Q2-1",        # no brackets
            "[cite:]",            # no handle
            "[cite:K7Q2]",        # a nonce with no source number
            "[cite:K7Q2-]",       # a source number that is not one
        ],
    )
    def test_malformed_syntax_does_not_become_evidence(self, answer):
        _registry, _bindings, table = _turn("foo.md")
        assert validate_citations(answer, table) == [], (
            f"a malformed marker was accepted: {answer!r}"
        )

    def test_punctuation_around_a_marker_is_not_part_of_it(self):
        """`[[cite:X]` and `(see [cite:X])` both carry a real handle. The
        marker is the exact token wherever it appears, so a stray bracket is
        a typo in the prose rather than a different citation - and the safety
        property is unchanged either way, because only an issued handle
        resolves at all."""
        _registry, _bindings, table = _turn("foo.md")
        handle = table.handle_for("src_1")
        for answer in (f"[[cite:{handle}]", f"(see [cite:{handle}])"):
            found = validate_citations(answer, table)
            assert [c.source_id for c in found] == ["src_1"], answer
            assert answer[found[0].start : found[0].end] == f"[cite:{handle}]"


class TestOneSourceHoweverManyTimesItIsCited:
    def test_two_routes_to_one_source_share_one_identity(self):
        """A context retrieval and an explicit search reaching the same file
        dedupe to one Source in S1, so they must not offer the model two names
        for one document."""
        registry = SourceRegistry()
        source = registry.register_source(
            kind="file", title="manual.md", locator="/files/manual.md"
        )
        first = registry.add_evidence(source.source_id, text="passage one")
        second = registry.add_evidence(source.source_id, text="passage two")
        table = build_citation_table(
            registry,
            [
                binding(source.source_id, first.evidence_id, context_id="ctx_a"),
                binding(source.source_id, second.evidence_id, context_id="ctx_b"),
            ],
            nonce="K7Q2",
        )

        assert len(table.by_handle) == 1, table.by_handle
        # Both passages stay eligible under the one identity.
        assert table.evidence_for(source.source_id) == (
            first.evidence_id,
            second.evidence_id,
        )

    def test_one_source_cited_twice_is_two_occurrences(self):
        """One cited source, two citations. Collapsing them would lose the
        second position before anything could render or persist it."""
        _registry, _bindings, table = _turn("manual.md")
        handle = table.handle_for("src_1")
        found = validate_citations(
            f"Alpha [cite:{handle}]. Beta [cite:{handle}].", table
        )

        assert [c.source_id for c in found] == ["src_1", "src_1"]
        assert found[0].start < found[1].start
        assert len({(c.start, c.end) for c in found}) == 2

    def test_occurrences_carry_the_marker_span(self):
        _registry, _bindings, table = _turn("manual.md")
        handle = table.handle_for("src_1")
        answer = f"Four hundred hours [cite:{handle}] exactly."
        found = validate_citations(answer, table)

        assert answer[found[0].start : found[0].end] == f"[cite:{handle}]"

    def test_citations_come_back_in_the_order_they_appear(self):
        _registry, _bindings, table = _turn("a.md", "b.md")
        first, second = table.handle_for("src_1"), table.handle_for("src_2")
        found = validate_citations(
            f"Beta [cite:{second}]. Alpha [cite:{first}].", table
        )
        assert [c.source_id for c in found] == ["src_2", "src_1"]


class TestUncitedProseIsFine:
    def test_an_answer_with_no_citations_is_not_an_error(self):
        _registry, _bindings, table = _turn("manual.md")
        assert validate_citations("Nothing to cite here.", table) == []


class TestTheMarkersCanBeTakenBackOut:
    """Until citations are rendered, a handle reaching a reader is an internal
    token in their chat. Stripping is what keeps the offer machinery dormant
    without a second parser."""

    def test_every_marker_is_removed_including_the_invalid_ones(self):
        _registry, _bindings, table = _turn("manual.md")
        handle = table.handle_for("src_1")
        answer = f"Alpha [cite:{handle}]. Beta [cite:ZZZZ-9]."
        assert strip_citations(answer) == "Alpha. Beta."
        assert CITATION_RE.search(strip_citations(answer)) is None

    @pytest.mark.parametrize(
        "answer",
        ["Alpha [cite:K7Q2].", "Alpha [cite:].", "Alpha [cite:K7Q2-]."],
    )
    def test_a_mistyped_marker_is_removed_too(self, answer):
        """Stripping is deliberately wider than validation. A marker the
        model mistyped resolves to nothing, so it is the one most certain to
        be left in front of a reader if only well-formed markers are taken
        out."""
        assert strip_citations(answer) == "Alpha.", answer

    @pytest.mark.parametrize("marker", ["[CITE:K7Q2-1]", "[Cite:K7Q2-1]"])
    def test_a_mistyped_keyword_is_removed_too(self, marker):
        """`[CITE:...]` is the same weak-model typo the broad stripper exists
        for; only the handle inside has to be exact."""
        assert strip_citations(f"Alpha {marker}.") == "Alpha."

    def test_an_unclosed_marker_does_not_eat_the_sentence(self):
        assert strip_citations("Alpha [cite:K7Q2 and the rest.") == (
            "Alpha [cite:K7Q2 and the rest."
        )

    def test_text_with_no_markers_is_unchanged(self):
        assert strip_citations("plain answer") == "plain answer"
