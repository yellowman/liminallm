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
        assert len(set(_headers(rendered))) > 1, (
            "two documents were rendered under one indistinguishable label: "
            f"{_headers(rendered)}"
        )

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
