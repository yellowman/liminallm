"""Which citations a turn issued, and which it must refuse.

The conservation boundary: a citation in model output resolves only to a
source this turn's bindings already grounded on. Everything here is about the
two ways that can fail - a handle that names something the turn did not
ground on, and a handle that looks right because a previous turn or a
retrieved document wrote it.
"""

from __future__ import annotations

import uuid

import pytest

from liminallm.service.citations import (
    ALPHABET,
    CITATION_RE,
    MIN_NONCE_BITS,
    NONCE_LENGTH,
    CitationTable,
    build_citation_table,
    extend_citation_table,
    mint_nonce,
    strip_citations,
    validate_citations,
)
from liminallm.service.provenance import ProvenanceError, SourceRegistry, binding

#: A nonce of the real width, so no test exercises a namespace the type refuses.
NONCE = "K7Q2ABCD"


def _turn(*titles, nonce=NONCE):
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
            registry, [binding(kept.source_id, evidence.evidence_id)], nonce=NONCE
        )

        assert table.handle_for(kept.source_id) is not None
        assert table.handle_for(dropped.source_id) is None, (
            "a source the answer never rested on became citable"
        )

    def test_an_unknown_handle_is_not_a_citation(self):
        _registry, _bindings, table = _turn("manual.md")
        assert validate_citations(f"Claimed [cite:{NONCE}-9].", table) == []

    def test_a_binding_the_registry_cannot_resolve_gets_no_handle(self):
        """A citation nobody can follow is worse than a missing one: there is
        no title, kind or locator to show for a source that is not there."""
        registry = SourceRegistry()
        table = build_citation_table(
            registry, [binding("src_404", "ev_404")], nonce=NONCE
        )
        assert not table
        assert validate_citations(f"Claimed [cite:{NONCE}-1].", table) == []


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
            registry, [binding(first.source_id, "ev_404")], nonce=NONCE
        )
        assert not table, dict(table.by_handle)

    def test_evidence_belonging_to_another_source_gets_no_handle(self):
        """The worst of the two: a real source paired with a real passage that
        is not its own would attach a citation to text it never contained."""
        registry, first, _second, _ea, from_b = self._two_sources()
        table = build_citation_table(
            registry, [binding(first.source_id, from_b.evidence_id)], nonce=NONCE
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
            nonce=NONCE,
        )
        assert len(table.by_handle) == 1, dict(table.by_handle)
        assert table.evidence_for(first.source_id) == (from_a.evidence_id,)

    def test_a_binding_with_no_evidence_grants_nothing(self):
        registry, first, _second, _ea, _eb = self._two_sources()
        table = build_citation_table(
            registry, [{"source_id": first.source_id, "evidence_id": ""}], nonce=NONCE
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
            CitationTable(nonce=NONCE).by_handle["forged"] = "src_1"

    def test_a_table_built_directly_is_frozen_too(self):
        """The builder is not the only way to get one of these. A rule kept
        there is a rule the constructor does not have, and this is the object
        the next stage makes authority."""
        table = CitationTable(nonce=NONCE, by_handle={f"{NONCE}-1": "src_1"})
        with pytest.raises(TypeError):
            table.by_handle["forged"] = "src_2"

    def test_the_callers_own_mapping_is_not_kept(self):
        """Wrapping a caller's dict without copying it leaves the caller
        holding the writable original."""
        mine = {f"{NONCE}-1": "src_1"}
        table = CitationTable(nonce=NONCE, by_handle=mine)
        mine["forged"] = "src_2"
        assert table.source_for("forged") is None, dict(table.by_handle)


class TestTheNamespaceCannotBeNarrowedByACaller:
    """The default mint is 40 bits. An override that skips the floor would
    make the guarantee a convention rather than a property - and the next
    stage needs one nonce reused across a growing assembly, so the override
    becomes a production path rather than a test convenience."""

    @pytest.mark.parametrize(
        "nonce",
        [
            "K7Q2",        # the old width
            "A",           # one character
            "",            # none at all
            "k7q2abcd",    # outside the alphabet
            "K7Q2ABC0",    # a digit the alphabet excludes
            "K7Q2ABCDE",   # one too many
        ],
    )
    def test_a_nonce_off_the_alphabet_or_the_width_is_refused(self, nonce):
        registry = SourceRegistry()
        with pytest.raises(ProvenanceError):
            build_citation_table(registry, [], nonce=nonce)

    def test_a_minted_nonce_is_always_accepted(self):
        registry = SourceRegistry()
        for _ in range(50):
            build_citation_table(registry, [], nonce=mint_nonce())


class TestAHandleFromAnotherTurnDoesNotResolve:
    """`source_id` restarts at `src_1` in every registry, and history is
    replayed verbatim into later prompts. A handle built from the internal id
    would let yesterday's citation name today's unrelated document."""

    def test_yesterdays_handle_is_refused_today(self):
        _r1, _b1, yesterday = _turn("rates-2024.md", nonce="AAAAAAAA")
        _r2, _b2, today = _turn("rates-2025.md", nonce="BBBBBBBB")
        stale = yesterday.handle_for("src_1")

        # Both turns minted `src_1`; only the nonce tells them apart.
        assert yesterday.source_for(stale) == "src_1"
        assert today.source_for(stale) is None
        assert validate_citations(f"Still 400 hours [cite:{stale}].", today) == []

    def test_a_source_authored_handle_does_not_resolve(self):
        """Retrieved text reaches the model as data, and a hostile page can
        write a citation marker as easily as any other string. It cannot write
        this turn's nonce, which is minted after the corpus was."""
        _registry, _bindings, table = _turn("manual.md", nonce=NONCE)
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
            "[cite K7Q2ABCD-1]",      # no colon
            "cite:K7Q2ABCD-1",        # no brackets
            "[cite:]",            # no handle
            "[cite:K7Q2ABCD]",        # a nonce with no source number
            "[cite:K7Q2ABCD-]",       # a source number that is not one
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
            nonce=NONCE,
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
        ["Alpha [cite:K7Q2ABCD].", "Alpha [cite:].", "Alpha [cite:K7Q2ABCD-]."],
    )
    def test_a_mistyped_marker_is_removed_too(self, answer):
        """Stripping is deliberately wider than validation. A marker the
        model mistyped resolves to nothing, so it is the one most certain to
        be left in front of a reader if only well-formed markers are taken
        out."""
        assert strip_citations(answer) == "Alpha.", answer

    @pytest.mark.parametrize("marker", ["[CITE:K7Q2ABCD-1]", "[Cite:K7Q2ABCD-1]"])
    def test_a_mistyped_keyword_is_removed_too(self, marker):
        """`[CITE:...]` is the same weak-model typo the broad stripper exists
        for; only the handle inside has to be exact."""
        assert strip_citations(f"Alpha {marker}.") == "Alpha."

    def test_an_unclosed_marker_does_not_eat_the_sentence(self):
        assert strip_citations("Alpha [cite:K7Q2ABCD and the rest.") == (
            "Alpha [cite:K7Q2ABCD and the rest."
        )

    def test_text_with_no_markers_is_unchanged(self):
        assert strip_citations("plain answer") == "plain answer"


class TestOneNamespacePerLogicalExecution:
    """A namespace belongs to the invocation, not to an attempt.

    The ledger spans retries: a replacement attempt can be handed a committed
    model response without the handler running again, and that text quotes the
    handles the *first* attempt was offered. A per-attempt namespace would
    give attempt B its predecessor's citations and nothing to resolve them
    against.
    """

    @staticmethod
    def _invocation(user_id="u"):
        from liminallm.service.invocation import InvocationRegistry

        return InvocationRegistry().open(
            uuid.uuid4().hex, tool="agent.files_v1", user_id=user_id, tenant_id=None
        )

    def test_a_replayed_answer_still_resolves_on_the_next_attempt(self):
        """Attempt A is offered a handle and the model quotes it. Attempt B
        replays that committed text; the handle has to mean the same source."""
        registry = SourceRegistry()
        source = registry.register_source(
            kind="file", title="manual.md", locator="/files/manual.md"
        )
        evidence = registry.add_evidence(source.source_id, text="400 hours")
        invocation = self._invocation()

        # Attempt A: the offer, and the model's answer quoting it.
        table_a = invocation.extend_citations(
            registry, [binding(source.source_id, evidence.evidence_id)]
        )
        handle = table_a.handle_for(source.source_id)
        replayed = f"400 hours [cite:{handle}]."

        # Attempt B: the same invocation, the same ledger, the replayed text.
        table_b = invocation.citations
        found = validate_citations(replayed, table_b)
        assert [c.source_id for c in found] == [source.source_id], (
            "a replayed answer's citation did not survive the retry"
        )

    def test_two_executions_do_not_share_a_namespace(self):
        first, second = self._invocation(), self._invocation()
        assert first.citations.nonce != second.citations.nonce

    def test_a_fresh_invocation_can_cite_nothing(self):
        assert not self._invocation().citations


class TestTheNamespaceGrowsRatherThanRestarting:
    """Two offers in one assembly - a round adds sources, then another does.
    Building a second table from the second round alone would allocate `-1`
    again, to a different source, under the same nonce."""

    @staticmethod
    def _registry_with(*titles):
        registry = SourceRegistry()
        pairs = []
        for title in titles:
            source = registry.register_source(
                kind="file", title=title, locator=f"/files/{title}"
            )
            evidence = registry.add_evidence(source.source_id, text=f"from {title}")
            pairs.append((source, evidence))
        return registry, pairs

    def test_a_new_source_takes_the_next_number(self):
        registry, pairs = self._registry_with("a.md", "b.md")
        (first, ea), (second, eb) = pairs
        table = build_citation_table(
            registry, [binding(first.source_id, ea.evidence_id)], nonce=NONCE
        )
        grown = extend_citation_table(
            registry, table, [binding(second.source_id, eb.evidence_id)]
        )

        assert grown.nonce == table.nonce
        assert grown.handle_for(first.source_id) == f"{NONCE}-1"
        assert grown.handle_for(second.source_id) == f"{NONCE}-2", (
            "the second offer restarted the numbering"
        )

    def test_an_existing_source_keeps_its_handle(self):
        registry, pairs = self._registry_with("a.md")
        (first, ea) = pairs[0]
        second_passage = registry.add_evidence(first.source_id, text="another passage")
        table = build_citation_table(
            registry, [binding(first.source_id, ea.evidence_id)], nonce=NONCE
        )
        grown = extend_citation_table(
            registry, table, [binding(first.source_id, second_passage.evidence_id)]
        )

        assert grown.handle_for(first.source_id) == table.handle_for(first.source_id)
        # Seen again only widens what may be cited within it.
        assert grown.evidence_for(first.source_id) == (
            ea.evidence_id,
            second_passage.evidence_id,
        )

    def test_an_invalid_binding_changes_nothing(self):
        registry, pairs = self._registry_with("a.md", "b.md")
        (first, ea), (second, eb) = pairs
        table = build_citation_table(
            registry, [binding(first.source_id, ea.evidence_id)], nonce=NONCE
        )
        grown = extend_citation_table(
            registry,
            table,
            [
                binding(second.source_id, "ev_404"),
                binding(second.source_id, ea.evidence_id),
            ],
        )
        assert dict(grown.by_handle) == dict(table.by_handle)

    def test_the_grown_table_is_frozen_too(self):
        registry, pairs = self._registry_with("a.md")
        (first, ea) = pairs[0]
        grown = extend_citation_table(
            registry,
            CitationTable(nonce=NONCE),
            [binding(first.source_id, ea.evidence_id)],
        )
        with pytest.raises(TypeError):
            grown.by_handle["forged"] = "src_1"

    def test_a_table_from_another_registry_is_refused(self):
        """Immutability is not validity. Extension is another authority gate,
        and a table whose entries do not resolve here would renumber a
        namespace that text already quotes."""
        registry, pairs = self._registry_with("a.md")
        stranger = CitationTable(
            nonce=NONCE,
            by_handle={f"{NONCE}-1": "src_99"},
            by_source={"src_99": f"{NONCE}-1"},
        )
        with pytest.raises(ProvenanceError):
            extend_citation_table(registry, stranger, [])

    def test_a_handle_from_another_namespace_is_refused(self):
        registry, pairs = self._registry_with("a.md")
        (first, _ea) = pairs[0]
        stranger = CitationTable(
            nonce=NONCE,
            by_handle={"ZZZZZZZZ-1": first.source_id},
            by_source={first.source_id: "ZZZZZZZZ-1"},
        )
        with pytest.raises(ProvenanceError):
            extend_citation_table(registry, stranger, [])


class TestAnInheritedTableIsCheckedWhole:
    """Extension validated only what its own loop read: the handle maps. The
    evidence map was copied through untouched, so state the builder would
    refuse in a new binding was preserved by arriving already built."""

    @staticmethod
    def _registry():
        registry = SourceRegistry()
        first = registry.register_source(kind="file", title="a.md", locator="/a")
        second = registry.register_source(kind="file", title="b.md", locator="/b")
        return (
            registry,
            registry.add_evidence(first.source_id, text="from a"),
            registry.add_evidence(second.source_id, text="from b"),
        )

    def test_inherited_evidence_that_does_not_exist_is_refused(self):
        registry, _from_a, _from_b = self._registry()
        table = CitationTable(
            nonce=NONCE,
            by_handle={f"{NONCE}-1": "src_1"},
            by_source={"src_1": f"{NONCE}-1"},
            evidence={"src_1": ("ev_404",)},
        )
        with pytest.raises(ProvenanceError):
            extend_citation_table(registry, table, [])

    def test_inherited_evidence_belonging_to_another_source_is_refused(self):
        """The relation defect fixed for new bindings, arriving by
        inheritance instead."""
        registry, _from_a, from_b = self._registry()
        table = CitationTable(
            nonce=NONCE,
            by_handle={f"{NONCE}-1": "src_1"},
            by_source={"src_1": f"{NONCE}-1"},
            evidence={"src_1": (from_b.evidence_id,)},
        )
        with pytest.raises(ProvenanceError):
            extend_citation_table(registry, table, [])

    def test_a_source_with_no_handle_of_its_own_is_refused(self):
        """Validation read `by_handle` only, so a stray reverse entry was
        never inspected. It suppresses allocation when that source later
        becomes genuinely eligible - and reached the loop as a `KeyError`."""
        registry, _from_a, from_b = self._registry()
        table = CitationTable(
            nonce=NONCE,
            by_handle={f"{NONCE}-1": "src_1"},
            by_source={"src_1": f"{NONCE}-1", "src_2": f"{NONCE}-2"},
        )
        with pytest.raises(ProvenanceError):
            extend_citation_table(
                registry, table, [binding("src_2", from_b.evidence_id)]
            )

    def test_evidence_filed_under_an_uncited_source_is_refused(self):
        registry, _from_a, from_b = self._registry()
        table = CitationTable(
            nonce=NONCE,
            by_handle={f"{NONCE}-1": "src_1"},
            by_source={"src_1": f"{NONCE}-1"},
            evidence={"src_2": (from_b.evidence_id,)},
        )
        with pytest.raises(ProvenanceError):
            extend_citation_table(registry, table, [])

    def test_a_consistent_table_still_grows(self):
        """The check has to refuse the malformed without refusing the real."""
        registry, from_a, from_b = self._registry()
        table = CitationTable(
            nonce=NONCE,
            by_handle={f"{NONCE}-1": "src_1"},
            by_source={"src_1": f"{NONCE}-1"},
            evidence={"src_1": (from_a.evidence_id,)},
        )
        grown = extend_citation_table(
            registry, table, [binding("src_2", from_b.evidence_id)]
        )
        assert grown.handle_for("src_1") == f"{NONCE}-1"
        assert grown.handle_for("src_2") == f"{NONCE}-2"
        assert grown.evidence_for("src_1") == (from_a.evidence_id,)
