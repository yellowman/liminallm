"""The provenance vocabulary, before anything speaks it.

Nothing in the application imports `provenance` yet. These tests are the
review surface for the shape itself: each one pins an invariant that four
future producers - web search, web fetch, context retrieval and note search -
will all depend on, and that would be expensive to change once they do.
"""

from __future__ import annotations

import json

import pytest

from liminallm.service.provenance import (
    Evidence,
    EvidenceLocator,
    ProvenanceError,
    Source,
    SourceRegistry,
)


class TestIdentityBelongsToTheRegistry:
    """A tool worker handles untrusted text. It must never be able to say
    "this is src_7" - it hands evidence over and is told what the id is."""

    def test_a_registry_is_the_turn_and_nothing_wider(self):
        """No module-level instance, no process-wide default: two registries
        do not share a counter, so one turn's `src_1` is not another's."""
        first, second = SourceRegistry(), SourceRegistry()
        a = first.register_source(kind="web", title="A", locator="https://a.test/")
        b = second.register_source(kind="web", title="B", locator="https://b.test/")
        assert a.source_id == b.source_id == "src_1"
        assert first.get_source("src_1").title == "A"
        assert second.get_source("src_1").title == "B"

    def test_ids_are_assigned_in_the_order_things_arrive(self):
        r = SourceRegistry()
        ids = [
            r.register_source(kind="file", title=f"f{i}", origin_id=f"o{i}").source_id
            for i in range(3)
        ]
        assert ids == ["src_1", "src_2", "src_3"]
        ev = [
            r.add_evidence("src_1", text=f"passage {i}").evidence_id
            for i in range(3)
        ]
        assert ev == ["ev_1", "ev_2", "ev_3"]

    def test_evidence_must_name_a_source_this_turn_registered(self):
        r = SourceRegistry()
        with pytest.raises(ProvenanceError, match="no such source"):
            r.add_evidence("src_99", text="from nowhere")


class TestTheSameThingIsOneSource:
    """A document retrieved twice is one entry, or a citation list repeats
    itself and a reader cannot tell two sources from one."""

    def test_an_origin_id_is_what_makes_two_retrievals_the_same(self):
        r = SourceRegistry()
        first = r.register_source(kind="note", title="Gonzo", origin_id="note_1")
        again = r.register_source(
            kind="note", title="Gonzo (again)", origin_id="note_1"
        )
        assert again.source_id == first.source_id
        assert len(r.sources) == 1
        assert r.get_source("src_1").title == "Gonzo", "the first registration wins"

    def test_without_an_origin_id_the_locator_decides(self):
        r = SourceRegistry()
        a = r.register_source(kind="web", title="P", locator="https://ex.test/p")
        b = r.register_source(kind="web", title="P", locator="https://ex.test/p")
        assert a.source_id == b.source_id
        assert len(r.sources) == 1

    def test_scheme_and_host_are_folded_and_nothing_else_is(self):
        """Case-insensitive per RFC 3986. A trailing slash or a query string
        is left alone: tidying those merges pages that differ, and a merged
        source attaches a citation to the wrong document."""
        r = SourceRegistry()
        a = r.register_source(kind="web", title="P", locator="HTTPS://Ex.TEST/Path")
        b = r.register_source(kind="web", title="P", locator="https://ex.test/Path")
        assert a.source_id == b.source_id, "host case should not make two sources"

        c = r.register_source(kind="web", title="P", locator="https://ex.test/Path/")
        d = r.register_source(kind="web", title="P", locator="https://ex.test/path")
        assert c.source_id != a.source_id, "a trailing slash may be a different page"
        assert d.source_id != a.source_id, "a path is case-sensitive"

    def test_a_source_with_no_identity_at_all_never_merges(self):
        """Nothing to match on. Matching on the title instead would merge two
        unrelated documents that happen to share a name."""
        r = SourceRegistry()
        a = r.register_source(kind="unknown", title="untitled")
        b = r.register_source(kind="unknown", title="untitled")
        assert a.source_id != b.source_id
        assert len(r.sources) == 2

    def test_one_identity_cannot_be_two_kinds(self):
        """A producer disagreeing with itself about what a thing is. Merging
        would pick one silently; refusing says which two claims conflict."""
        r = SourceRegistry()
        r.register_source(kind="note", title="thing", origin_id="x1")
        with pytest.raises(ProvenanceError, match="cannot also be"):
            r.register_source(kind="file", title="thing", origin_id="x1")


class TestEvidenceIsHashedHereNotElsewhere:
    def test_the_registry_computes_the_hash_from_the_text(self):
        """A stored citation is checked against this later, so a producer
        that could supply it could supply one that does not match."""
        import hashlib

        r = SourceRegistry()
        r.register_source(kind="file", title="m", origin_id="f1")
        record = r.add_evidence("src_1", text="the passage")
        assert record.content_hash == hashlib.sha256(b"the passage").hexdigest()

    def test_add_evidence_takes_no_hash_from_the_caller(self):
        r = SourceRegistry()
        r.register_source(kind="file", title="m", origin_id="f1")
        with pytest.raises(TypeError):
            r.add_evidence("src_1", text="x", content_hash="deadbeef")

    def test_the_same_passage_in_the_same_place_is_one_piece_of_evidence(self):
        r = SourceRegistry()
        r.register_source(kind="file", title="m", origin_id="f1")
        where = EvidenceLocator(chunk_id="c9", chunk_index=9)
        first = r.add_evidence("src_1", text="same", locator=where)
        again = r.add_evidence("src_1", text="same", locator=EvidenceLocator(
            chunk_id="c9", chunk_index=9
        ))
        assert again.evidence_id == first.evidence_id
        assert len(r.evidence) == 1

    def test_the_same_text_elsewhere_in_the_source_is_not(self):
        r = SourceRegistry()
        r.register_source(kind="file", title="m", origin_id="f1")
        a = r.add_evidence("src_1", text="repeated", locator=EvidenceLocator(page=1))
        b = r.add_evidence("src_1", text="repeated", locator=EvidenceLocator(page=7))
        assert a.evidence_id != b.evidence_id

    def test_the_same_text_in_a_different_source_is_not(self):
        r = SourceRegistry()
        r.register_source(kind="file", title="a", origin_id="f1")
        r.register_source(kind="file", title="b", origin_id="f2")
        a = r.add_evidence("src_1", text="shared wording")
        b = r.add_evidence("src_2", text="shared wording")
        assert a.evidence_id != b.evidence_id
        assert r.evidence_for("src_1") == (a,)


class TestKindIsStatedNeverInferred:
    """The defect this whole layer exists to prevent: the renderer guessing
    that `.md` means a note, and telling a reader their uploaded manual is
    something they wrote."""

    def test_a_markdown_upload_is_a_file_because_the_producer_says_so(self):
        r = SourceRegistry()
        source = r.register_source(
            kind="file",
            title="manual.md",
            origin_id="filegen_7",
            locator="/uploads/manual.md",
            metadata={"context_id": "ctx_1"},
        )
        assert source.kind == "file"

    def test_a_context_is_a_scope_on_a_source_not_a_kind_of_one(self):
        """A knowledge context is a corpus you search. The citable thing is
        the file inside it, so the context rides in metadata."""
        r = SourceRegistry()
        with pytest.raises(ProvenanceError, match="unknown source kind"):
            r.register_source(kind="context", title="Gonzo", origin_id="ctx_1")

        source = r.register_source(
            kind="file", title="a.pdf", origin_id="f1", metadata={"context_id": "ctx_1"}
        )
        assert source.metadata["context_id"] == "ctx_1"

    def test_unknown_provenance_stays_unknown(self):
        """The explicit neutral exists so that a producer which cannot
        establish provenance has somewhere honest to put it, instead of
        picking the nearest-looking kind."""
        r = SourceRegistry()
        source = r.register_source(kind="unknown", title="pasted text")
        assert source.kind == "unknown"

    def test_a_kind_outside_the_vocabulary_is_refused(self):
        r = SourceRegistry()
        with pytest.raises(ProvenanceError, match="unknown source kind"):
            r.register_source(kind="website", title="x")


class TestThereIsNoCommonScore:
    """BM25, cosine similarity, a fused rank, a search-result position and an
    eventual support score are different quantities that happen to be
    numbers. One field named `score` would eventually compare two of them,
    and the comparison would look reasonable."""

    def test_evidence_has_no_score_field(self):
        assert "score" not in Evidence.__dataclass_fields__

    def test_neither_does_a_source_or_a_locator(self):
        assert "score" not in Source.__dataclass_fields__
        assert "score" not in EvidenceLocator.__dataclass_fields__

    def test_a_retrieval_number_can_still_be_recorded_by_its_own_name(self):
        r = SourceRegistry()
        source = r.register_source(
            kind="web",
            title="P",
            locator="https://ex.test/p",
            metadata={"search_rank": 2, "provider": "brave"},
        )
        assert source.metadata["search_rank"] == 2


class TestTheSnapshotIsPlainData:
    def test_it_round_trips_through_json_in_registration_order(self):
        r = SourceRegistry()
        r.register_source(kind="web", title="first", locator="https://a.test/")
        r.register_source(kind="note", title="second", origin_id="note_2")
        r.add_evidence("src_1", text="from the web")
        r.add_evidence("src_2", text="from a note", locator=EvidenceLocator(start=0, end=11))

        snapshot = r.snapshot()
        assert json.loads(json.dumps(snapshot)) == snapshot
        assert list(snapshot["sources"]) == ["src_1", "src_2"]
        assert [e["evidence_id"] for e in snapshot["evidence"]] == ["ev_1", "ev_2"]
        assert snapshot["evidence"][1]["locator"]["end"] == 11

    def test_metadata_that_could_not_survive_the_snapshot_is_refused_early(self):
        """Checked where it enters, so a bad value names the producer that
        supplied it rather than surfacing as a failure to serialise the turn."""
        r = SourceRegistry()
        with pytest.raises(ProvenanceError, match="JSON-safe"):
            r.register_source(
                kind="web", title="P", locator="https://ex.test/",
                metadata={"fetched": object()},
            )

    def test_a_registered_source_cannot_be_edited_through_the_snapshot(self):
        r = SourceRegistry()
        r.register_source(kind="web", title="P", locator="https://ex.test/")
        r.snapshot()["sources"]["src_1"]["title"] = "tampered"
        assert r.get_source("src_1").title == "P"

    def test_records_are_frozen(self):
        r = SourceRegistry()
        source = r.register_source(kind="web", title="P", locator="https://ex.test/")
        with pytest.raises(Exception):
            source.title = "tampered"
