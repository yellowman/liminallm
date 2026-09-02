"""What the model is told about the excerpts explicit `file_search` returns.

The tool renders a header per excerpt so the model can tell one document from
another. That header is the whole of what it knows about where an excerpt
came from, so a wrong one is not cosmetic: two different files become
indistinguishable in the same answer.
"""

from __future__ import annotations

import uuid

import pytest

from liminallm.service import agent_tools
from liminallm.service.runtime import get_runtime

MANUAL = "Turbine blade inspection happens every 400 flight hours. " * 40
LOGBOOK = "The logbook records each inspection against its airframe. " * 40
ENGINE_A = "Left engine inspection: compressor stage two within limits. " * 40
ENGINE_B = "Right engine inspection: compressor stage two out of limits. " * 40


def _context_with(store, files):
    """A real user, a real context, and files ingested under real paths."""
    user = store.create_user(email=f"fs_{uuid.uuid4().hex[:8]}@t.local")
    ctx = store.upsert_context(
        name=f"fs-{uuid.uuid4().hex[:6]}",
        description="two documents",
        owner_user_id=user.id,
    )
    for path, text in files.items():
        written = get_runtime().rag.ingest_text(ctx.id, text, source_path=path)
        assert written > 0, f"the fixture failed to index {path}"
    return user.id, ctx.id


def _headers(rendered):
    return [line for line in rendered.split("\n") if line.startswith("[")]


class TestTheModelIsToldWhichFileAnExcerptCameFrom:
    def test_an_excerpt_is_labelled_with_its_file(self, store):
        """`run_file_search` reads `chunk.meta["source_path"]`, and no ingest
        path writes that key - `fs_path` is where the path lives. So every
        excerpt fell through to the literal `attachment`."""
        user_id, ctx_id = _context_with(
            store, {"reports/turbines.md": MANUAL}
        )
        rendered, _, _chunks = agent_tools.run_file_search(
            "turbine blade inspection", 3, [ctx_id],
            rag=get_runtime().rag, user_id=user_id, tenant_id=None,
        )
        assert _headers(rendered), f"nothing was retrieved: {rendered!r}"
        assert all(h == "[turbines.md]" for h in _headers(rendered)), (
            f"the model was told the wrong source: {_headers(rendered)}"
        )

    def test_two_files_are_distinguishable_in_one_answer(self, store):
        """The consequence, and the reason the label matters: an answer
        resting on two documents must be able to say which said what."""
        user_id, ctx_id = _context_with(
            store,
            {"reports/turbines.md": MANUAL, "reports/logbook.md": LOGBOOK},
        )
        rendered, snippets, _chunks = agent_tools.run_file_search(
            "inspection", 6, [ctx_id],
            rag=get_runtime().rag, user_id=user_id, tenant_id=None,
        )
        # A fixture property, checked rather than assumed: the defect is only
        # visible if both documents were actually retrieved.
        chunks = get_runtime().rag.retrieve(
            [ctx_id], "inspection", limit=6, user_id=user_id, tenant_id=None
        )
        assert len({c.fs_path for c in chunks}) >= 2, (
            f"fixture retrieved one document only: {[c.fs_path for c in chunks]}"
        )
        assert len(snippets) >= 2, f"fixture retrieved too little: {snippets}"
        # Exactly the file names: two documents that are already told apart by
        # their names must not be widened to tell them apart.
        assert set(_headers(rendered)) == {"[turbines.md]", "[logbook.md]"}, (
            "two documents were rendered under one indistinguishable label: "
            f"{_headers(rendered)}"
        )

    def test_one_name_in_two_directories_is_two_files(self, store):
        """A corpus ingested from a directory tree holds the same file name
        more than once - `**/*` is a supported source. The name alone then
        says nothing, and an answer resting on both reports would attribute
        each one's finding to the other."""
        user_id, ctx_id = _context_with(
            store,
            {
                "reports/engine-a/status.md": ENGINE_A,
                "reports/engine-b/status.md": ENGINE_B,
            },
        )
        rendered, snippets, _chunks = agent_tools.run_file_search(
            "compressor inspection", 6, [ctx_id],
            rag=get_runtime().rag, user_id=user_id, tenant_id=None,
        )
        chunks = get_runtime().rag.retrieve(
            [ctx_id], "compressor inspection", limit=6,
            user_id=user_id, tenant_id=None,
        )
        assert len({c.fs_path for c in chunks}) == 2, (
            f"fixture retrieved one document only: {[c.fs_path for c in chunks]}"
        )
        headers = _headers(rendered)
        # More excerpts than files, so this also says that several excerpts
        # from one file share one label.
        assert len(headers) > 2, f"fixture retrieved too little: {headers}"
        assert set(headers) == {"[engine-a/status.md]", "[engine-b/status.md]"}, (
            f"two reports were rendered under one label: {headers}"
        )
        assert len(snippets) == len(headers)

    def test_inline_text_is_not_given_a_filename_it_does_not_have(self, store):
        """`ingest_text` with no source path writes the `inline` sentinel. It
        names no file, so the header must not pretend it does."""
        user_id, ctx_id = _context_with(store, {})
        get_runtime().rag.ingest_text(ctx_id, MANUAL)
        rendered, _, _chunks = agent_tools.run_file_search(
            "turbine blade inspection", 3, [ctx_id],
            rag=get_runtime().rag, user_id=user_id, tenant_id=None,
        )
        headers = _headers(rendered)
        assert headers, f"nothing was retrieved: {rendered!r}"
        assert all(h == "[inline text]" for h in headers), (
            f"inline text was labelled as a file: {headers}"
        )


class TestTheLabelInventsNothing:
    """`chunk_label` takes whatever it is handed. `knowledge_chunk.fs_path`
    is `TEXT NOT NULL`, so a chunk read from the database always has a path -
    but the column does not constrain one built in memory, and the RAG
    provenance adapter already refuses a pathless chunk rather than inventing
    an identity for it. The label is the same rule in the text the model
    reads."""

    def test_a_chunk_with_no_path_is_named_as_unknown(self):
        from types import SimpleNamespace

        from liminallm.service.agent_tools import chunk_label

        assert chunk_label(SimpleNamespace(fs_path=None)) == "unknown source"
        assert chunk_label(SimpleNamespace(fs_path="")) == "unknown source"
        assert chunk_label(SimpleNamespace()) == "unknown source"


class TestTheLabelGrowsOnlyAsFarAsItMust:
    """The widening rule itself, on path shapes a corpus fixture cannot
    produce cheaply. A label the model reads is a claim about identity, so it
    has to be wide enough to separate the files in front of it and no wider."""

    @staticmethod
    def _labels(*paths):
        from types import SimpleNamespace

        from liminallm.service.agent_tools import chunk_labels

        return chunk_labels([SimpleNamespace(fs_path=path) for path in paths])

    def test_one_directory_is_enough_when_one_directory_separates_them(self):
        assert self._labels("a/b/c/report.md", "a/b/d/report.md") == [
            "c/report.md",
            "d/report.md",
        ]

    def test_a_shared_name_does_not_widen_an_unshared_one(self):
        assert self._labels("x/status.md", "y/status.md", "notes.md") == [
            "x/status.md",
            "y/status.md",
            "notes.md",
        ]

    def test_one_path_ending_another_falls_back_to_the_whole_path(self):
        """Every trailing run of `a/report.md` is also a trailing run of
        `b/a/report.md`, so no suffix separates the two."""
        assert self._labels("a/report.md", "b/a/report.md") == [
            "a/report.md",
            "b/a/report.md",
        ]

    def test_inline_text_takes_no_part_in_the_widening(self):
        from liminallm.service.rag import INLINE_PATH

        assert self._labels(INLINE_PATH, "reports/status.md") == [
            "inline text",
            "status.md",
        ]


class TestNothingRetrievedSaysSo:
    def test_an_empty_result_is_still_an_answer(self, store):
        user_id, ctx_id = _context_with(store, {"reports/turbines.md": MANUAL})
        rendered, snippets, _chunks = agent_tools.run_file_search(
            "zzzz nonexistent phrase zzzz", 3, [ctx_id],
            rag=get_runtime().rag, user_id=user_id, tenant_id=None,
        )
        assert snippets == [] or _headers(rendered)


@pytest.fixture
def store():
    return get_runtime().store
