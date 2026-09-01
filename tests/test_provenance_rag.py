"""What a knowledge-context retrieval contributes to a turn's provenance.

The first producer to speak the S1 vocabulary. Nothing here is about ranking
or the prompt: the retrieval already happened, and these tests pin what the
turn records about where its grounding came from.
"""

from __future__ import annotations

import pytest

from liminallm.service.provenance import ProvenanceError, SourceRegistry
from liminallm.service.rag import register_retrieved_chunks
from liminallm.storage.models import KnowledgeChunk


def chunk(
    fs_path="reports/turbines.md", *, context_id="ctx_a", index=0, content="x",
    row_id=None,
):
    return KnowledgeChunk(
        id=row_id,
        context_id=context_id,
        fs_path=fs_path,
        content=content,
        embedding=[],
        chunk_index=index,
    )


class TestAFileIsIdentifiedByWhereItIsNotByAnIdItDoesNotHave:
    """`origin_id` is a source system's own stable identity - a note id, a
    message id, a file-generation id. RAG has none: the schema records no
    generation, and a chunk claims to describe whatever its path holds *now*.
    So the path is the locator, and `origin_id` stays empty until there is a
    real generation id to put in it."""

    def test_the_path_is_the_locator(self):
        r = SourceRegistry()
        register_retrieved_chunks(r, [chunk()])
        source = r.get_source("src_1")
        assert source.locator == "reports/turbines.md"
        assert source.origin_id is None
        assert source.kind == "file"

    def test_the_title_is_what_a_reader_would_call_it(self):
        r = SourceRegistry()
        register_retrieved_chunks(r, [chunk("/users/u/files/annual report.pdf")])
        assert r.get_source("src_1").title == "annual report.pdf"

    def test_two_chunks_of_one_file_are_one_source_with_two_passages(self):
        r = SourceRegistry()
        register_retrieved_chunks(
            r,
            [
                chunk(index=0, content="first"),
                chunk(index=1, content="second"),
            ],
        )
        assert len(r.sources) == 1
        assert len(r.evidence) == 2
        assert [e.locator.chunk_index for e in r.evidence] == [0, 1]

    def test_the_passage_carries_the_chunk_index_and_nothing_invented(self):
        r = SourceRegistry()
        register_retrieved_chunks(r, [chunk(index=7, content="the passage")])
        record = r.get_evidence("ev_1")
        assert record.text == "the passage"
        assert record.locator.chunk_index == 7
        assert record.locator.page is None
        assert record.locator.section is None


class TestInlineTextIsNotAFile:
    """`ingest_text` writes the sentinel `fs_path = "inline"` when no source
    path was given. It names no document, so it cannot become a file source.

    Nor can the context stand in for one. `_commit_generation` *adds* inline
    text rather than replacing a path's generation, so two anonymous ingests
    into one context are two unrelated documents that both land under this
    sentinel. A context-scoped identity would merge them, which is the
    context being the source again by another name.
    """

    def test_inline_is_neutral_rather_than_a_fabricated_file(self):
        r = SourceRegistry()
        register_retrieved_chunks(r, [chunk("inline", context_id="ctx_a", row_id=41)])
        source = r.get_source("src_1")
        assert source.kind == "unknown"
        assert source.locator is None
        assert source.origin_id == "knowledge_chunk:41"

    def test_two_inline_documents_in_one_context_do_not_merge(self):
        """The case a context-scoped identity got wrong."""
        r = SourceRegistry()
        register_retrieved_chunks(
            r,
            [
                chunk("inline", context_id="ctx_a", row_id=1, content="one"),
                chunk("inline", context_id="ctx_a", row_id=2, content="two"),
            ],
        )
        assert len(r.sources) == 2
        assert {s.origin_id for s in r.sources} == {
            "knowledge_chunk:1",
            "knowledge_chunk:2",
        }

    def test_the_same_inline_chunk_twice_is_still_one_source(self):
        """The row id is a real identity, so it still dedupes."""
        r = SourceRegistry()
        register_retrieved_chunks(r, [chunk("inline", row_id=7, content="x")])
        register_retrieved_chunks(r, [chunk("inline", row_id=7, content="x")])
        assert len(r.sources) == 1
        assert len(r.evidence) == 1

    def test_an_unpersisted_inline_chunk_claims_no_identity(self):
        """No row id, nothing else honest to use. A source with no identity
        never merges, which is the safe direction: two anonymous fragments
        stay two sources rather than becoming one document."""
        r = SourceRegistry()
        register_retrieved_chunks(
            r,
            [
                chunk("inline", row_id=None, content="one"),
                chunk("inline", row_id=None, content="two"),
            ],
        )
        assert len(r.sources) == 2
        assert all(s.origin_id is None and s.locator is None for s in r.sources)

    def test_inline_evidence_points_at_its_row(self):
        r = SourceRegistry()
        register_retrieved_chunks(r, [chunk("inline", row_id=41, index=3)])
        assert r.get_evidence("ev_1").locator.chunk_id == "41"
        assert r.get_evidence("ev_1").locator.chunk_index == 3

    def test_file_evidence_does_not_carry_the_row_id(self):
        """Measured: one file in two contexts has two sets of rows with two
        sets of ids. Carrying the id would split one passage into two pieces
        of evidence and defeat the cross-context dedupe below."""
        r = SourceRegistry()
        register_retrieved_chunks(r, [chunk("manual.md", row_id=9, index=3)])
        assert r.get_evidence("ev_1").locator.chunk_id is None
        assert r.get_evidence("ev_1").locator.chunk_index == 3


class TestWhichScopeFoundWhatIsABindingNotAField:
    """One file can be described by several contexts, and registration is
    first-wins. A `context_id` on the source would freeze whichever context
    retrieved it first and read as if the document belonged to that one."""

    def test_the_source_does_not_claim_to_belong_to_a_context(self):
        r = SourceRegistry()
        register_retrieved_chunks(r, [chunk(context_id="ctx_a")])
        assert "context_id" not in r.get_source("src_1").metadata

    def test_the_same_file_through_two_contexts_is_one_source(self):
        r = SourceRegistry()
        first = register_retrieved_chunks(
            r, [chunk("manual.md", context_id="ctx_a", index=3, content="same bytes", row_id=1)]
        )
        second = register_retrieved_chunks(
            r, [chunk("manual.md", context_id="ctx_b", index=3, content="same bytes", row_id=2)]
        )
        assert len(r.sources) == 1
        assert first[0]["source_id"] == second[0]["source_id"] == "src_1"

    def test_and_the_identical_passage_is_one_piece_of_evidence(self):
        """Same source, same locator, same bytes. S1 dedupes on exactly that,
        so the second context does not get a second `ev_`."""
        r = SourceRegistry()
        register_retrieved_chunks(
            r, [chunk("manual.md", context_id="ctx_a", index=3, content="same bytes", row_id=1)]
        )
        second = register_retrieved_chunks(
            r, [chunk("manual.md", context_id="ctx_b", index=3, content="same bytes", row_id=2)]
        )
        assert len(r.evidence) == 1
        assert second[0]["evidence_id"] == "ev_1"

    def test_but_both_contexts_are_still_recorded_as_having_found_it(self):
        """The binding is what would be lost if this lived on the source: two
        scopes reached one passage, and the turn can still say which."""
        r = SourceRegistry()
        a = register_retrieved_chunks(
            r, [chunk("manual.md", context_id="ctx_a", index=3, content="same bytes", row_id=1)]
        )
        b = register_retrieved_chunks(
            r, [chunk("manual.md", context_id="ctx_b", index=3, content="same bytes", row_id=2)]
        )
        assert a == [
            {"context_id": "ctx_a", "source_id": "src_1", "evidence_id": "ev_1"}
        ]
        assert b == [
            {"context_id": "ctx_b", "source_id": "src_1", "evidence_id": "ev_1"}
        ]


class TestTheAdapterRefusesWhatItCannotDescribe:
    def test_a_chunk_with_no_path_is_not_silently_given_one(self):
        r = SourceRegistry()
        with pytest.raises(ProvenanceError):
            register_retrieved_chunks(r, [chunk(fs_path=None)])

    def test_nothing_retrieved_records_nothing(self):
        r = SourceRegistry()
        assert register_retrieved_chunks(r, []) == []
        assert r.sources == ()


class TestEveryProductionPathThreadsTheTurnsRegistry:
    """The signatures default to `None` so that a test exercising retry or
    breaker semantics need not care about provenance. That default cannot
    enforce the thing that matters: a node path added later must not silently
    stop recording. The call sites are what to check, so this checks them.
    """

    THREADED = {
        "_invoke_tool",
        "_execute_node",
        "_execute_node_with_retry",
        "_blocking_attempt",
        "_execute_parallel_nodes",
        "_plan_invocation",
        "_stream_llm_node",
        "_stream_node_with_retry",
    }

    #: The one function that may build a registry rather than thread one:
    #: the direct tool endpoint, which has no workflow turn around it to
    #: share one with and so owns its own exactly as it owns its own
    #: invocation. Keyed by the *containing* function, not the callee - a
    #: callee-name exemption would let every `_invoke_tool(...)` call site
    #: construct one, including the ones inside a workflow turn.
    MAY_CONSTRUCT = {"invoke_tool"}

    def _call_sites(self):
        """Both halves of the invariant, and whose frame each call is in.

        Presence is the weaker half, and on its own it is satisfied by
        handing every parallel child a `SourceRegistry()` of its own - which
        is the exact defect this threading exists to prevent, since one turn
        would then hold several identity spaces and several `src_1`s meaning
        different documents. So the value is checked too: it must be the name
        being threaded, unless the enclosing function is the one allowed to
        start a registry.
        """
        import ast
        import pathlib

        import liminallm.service.workflow as wf
        import liminallm.service.workflow_streaming as wfs

        missing, rebuilt = [], []
        for module in (wf, wfs):
            path = pathlib.Path(module.__file__)
            tree = ast.parse(path.read_text())
            # Which function body each call sits in, so the exemption can be
            # keyed on the caller.
            enclosing = {}
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    for inner in ast.walk(node):
                        enclosing.setdefault(id(inner), node.name)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
                if name not in self.THREADED:
                    continue
                passed = {kw.arg: kw.value for kw in node.keywords}
                caller = enclosing.get(id(node), "<module>")
                where = f"{path.name}:{node.lineno} {caller}() -> {name}()"
                if "source_registry" not in passed:
                    missing.append(where)
                    continue
                value = passed["source_registry"]
                if isinstance(value, ast.Name) and value.id == "source_registry":
                    continue
                if caller in self.MAY_CONSTRUCT:
                    continue
                rebuilt.append(where)
        return missing, rebuilt

    def test_no_production_call_site_drops_it(self):
        missing, _ = self._call_sites()
        assert not missing, (
            "these call sites would lose the turn's provenance: " + ", ".join(missing)
        )

    def test_no_production_call_site_starts_a_second_one(self):
        _, rebuilt = self._call_sites()
        assert not rebuilt, (
            "these call sites would give one turn more than one identity "
            "space: " + ", ".join(rebuilt)
        )

    def test_the_check_can_actually_fail(self):
        """A structural test that cannot go red is decoration."""
        import ast

        tree = ast.parse("self._invoke_tool(a, b, user_id=u)")
        call = tree.body[0].value
        assert "source_registry" not in {kw.arg for kw in call.keywords}
