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
from liminallm.service.provenance import SourceRegistry
from liminallm.service.runtime import get_runtime

MANUAL = "Turbine blade inspection happens every 400 flight hours. " * 40
LOGBOOK = "The logbook records each inspection against its airframe. " * 40
ENGINE_A = "Left engine inspection: compressor stage two within limits. " * 40
ENGINE_B = "Right engine inspection: compressor stage two out of limits. " * 40
#: Past `INLINE_MAX_BYTES`, which is what makes an attachment searchable at
#: all - a small text file is carried into the prompt whole and never indexed,
#: so a smaller fixture would exercise no retrieval.
SEARCHABLE = MANUAL * 7


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


class TestAnAttachmentIsNamedByTheNameTheChatGaveIt:
    """A searchable attachment's `fs_path` is not a path at all.

    Attachment ingestion keys the index by `generation_key()` - one *reading*
    of an object, spelled `attachment-generation:<sha256>:<extension>` -
    because the same bytes attached under two names parse two ways. The
    filename is not encoded in it. So anything that treats every non-inline
    `fs_path` as a filesystem path shows the model a checksum instead of a
    document, and records the reading as a locator the attachment subsystem
    says it is not.

    The parent has the missing half: the conversation's own records name
    every generation it holds.
    """

    def _account(self, client):
        email = f"prov_{uuid.uuid4().hex[:8]}@example.com"
        resp = client.post(
            "/v1/auth/signup", json={"email": email, "password": "TestPassword123!"}
        )
        assert resp.status_code == 201, resp.text
        data = resp.json()["data"]
        return data["user_id"], {"Authorization": f"Bearer {data['access_token']}"}

    def _attached(self, client, headers, name, body):
        """One conversation holding one searchable attachment, uploaded the
        way the chat uploads it."""
        resp = client.post(
            "/v1/conversations", headers=headers, json={"title": "attachment chat"}
        )
        assert resp.status_code in (200, 201), resp.text
        conversation_id = resp.json()["data"]["id"]
        resp = client.post(
            "/v1/files/upload",
            headers={**headers, "Idempotency-Key": uuid.uuid4().hex},
            files={"file": (name, body, "text/markdown")},
            data={"conversation_id": conversation_id},
        )
        assert resp.status_code in (200, 201), resp.text
        return conversation_id

    def _generation(self, runtime, conversation_id, user_id):
        """The reading this conversation authorizes, and its checksum."""
        from liminallm.service.attachments import generation_key, list_attachments

        conversation = runtime.store.get_conversation(
            conversation_id, user_id=user_id
        )
        records = list_attachments(conversation)
        assert len(records) == 1, records
        key = generation_key(records[0].get("checksum"), records[0].get("name"))
        assert key, records
        return key, str(records[0].get("checksum"))

    def test_the_model_reads_the_filename_and_not_the_reading(self, client):
        user_id, headers = self._account(client)
        runtime = get_runtime()
        conversation_id = self._attached(
            client, headers, "flight-manual.pdf.md", SEARCHABLE
        )
        key, checksum = self._generation(runtime, conversation_id, user_id)

        rendered, snippets, _chunks, _hints = runtime.workflow._run_file_search(
            "turbine blade inspection", 4,
            conversation_id=conversation_id, context_id=None,
            user_id=user_id, tenant_id=None,
        )
        assert snippets, f"the fixture attached nothing searchable: {rendered!r}"
        headers_read = _headers(rendered)
        assert headers_read, f"nothing was retrieved: {rendered!r}"
        assert all(h == "[flight-manual.pdf.md]" for h in headers_read), (
            f"the model was told a generation key: {headers_read}"
        )
        # Not only the header: the key names the object by digest, and a
        # digest in the model's context is a string it can quote back.
        assert "attachment-generation:" not in rendered, rendered[:200]
        assert checksum not in rendered, "the model was shown a checksum"
        assert key not in rendered

    def _selected_context(self, client, headers, files):
        """A knowledge context the chat can name, holding real paths."""
        resp = client.post(
            "/v1/contexts",
            headers=headers,
            json={"name": f"ctx-{uuid.uuid4().hex[:6]}", "description": "manuals"},
        )
        assert resp.status_code in (200, 201), resp.text
        context_id = resp.json()["data"]["id"]
        for path, text in files.items():
            written = get_runtime().rag.ingest_text(
                context_id, text, source_path=path
            )
            assert written > 0, f"the fixture failed to index {path}"
        return context_id

    def test_an_attachment_and_a_context_file_of_one_name_stay_two(
        self, client
    ):
        """The two scopes are searched together - `_run_file_search` starts
        with the conversation's attachment context and adds the selected
        knowledge context to the same list, and the agent offers `file_search`
        whenever either exists. So a chat holding `report.md` beside a context
        holding `manuals/report.md` is an ordinary configuration that puts
        both in one answer.

        The attachment keeps the name its owner gave it; the path widens
        around it, because a path has parts to widen with and a name does
        not."""
        user_id, headers = self._account(client)
        runtime = get_runtime()
        conversation_id = self._attached(client, headers, "report.md", SEARCHABLE)
        context_id = self._selected_context(
            client, headers, {"manuals/report.md": LOGBOOK * 4}
        )

        # A term from each document, so the ranking has a reason to return
        # both rather than filling every slot from whichever is longer.
        rendered, _snippets, chunks, _hints = runtime.workflow._run_file_search(
            "turbine logbook", 6,
            conversation_id=conversation_id, context_id=context_id,
            user_id=user_id, tenant_id=None,
        )
        # Checked rather than assumed: the collision is only reachable if both
        # scopes actually answered.
        assert len({c.fs_path for c in chunks}) == 2, (
            f"the fixture searched one scope only: {[c.fs_path for c in chunks]}"
        )
        assert set(_headers(rendered)) == {"[report.md]", "[manuals/report.md]"}, (
            "an attachment and a context file were rendered under one label: "
            f"{_headers(rendered)}"
        )

    def test_the_reading_is_an_origin_and_never_a_locator(self, client):
        """`Source.locator` says where a document is. The generation key is
        not a place, it is the attachment subsystem's own stable identity for
        one reading - which is what `origin_id` is for."""
        from liminallm.service.broker import CapabilityBroker, InvocationContext
        from liminallm.service.invocation import InvocationRegistry

        user_id, headers = self._account(client)
        runtime = get_runtime()
        conversation_id = self._attached(
            client, headers, "flight-manual.pdf.md", SEARCHABLE
        )
        key, _checksum = self._generation(runtime, conversation_id, user_id)

        registry = SourceRegistry()
        context = InvocationContext(
            user_id=user_id,
            conversation_id=conversation_id,
            source_registry=registry,
            provenance_bindings=[],
        )
        broker = CapabilityBroker(runtime.workflow, context)
        reply = broker._answer(
            InvocationRegistry().open(
                uuid.uuid4().hex, tool="file.search_v1", user_id=user_id,
                tenant_id=None,
            ),
            {
                "capability": "rag.retrieve",
                "operation_seq": 1,
                "payload": {"query": "turbine blade inspection", "limit": 4},
            },
        )
        assert reply["ok"] and reply["result"].get("text"), reply
        assert context.provenance_bindings, "the search grounded nothing"

        sources = [
            registry.get_source(b["source_id"])
            for b in context.provenance_bindings
        ]
        assert len(set(s.source_id for s in sources)) == 1, (
            f"one attachment became several sources: {sources}"
        )
        source = sources[0]
        assert source.title == "flight-manual.pdf.md", source
        assert source.origin_id == key, source
        assert source.locator is None, (
            f"a reading was recorded as a filesystem path: {source.locator}"
        )


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

    def test_the_whole_path_fallback_is_not_an_absolute_path(self):
        """The fallback's input is routinely absolute: a context source is
        `authorize_path`-resolved before `ingest_path` sees it, and the
        resolved path is what RAG stores when no separate identity is given.
        A label is text the model can quote back, so the rooted spelling must
        not be what it is handed."""
        labels = self._labels(
            "/srv/liminallm/users/u/a/report.md",
            "/srv/liminallm/users/u/srv/liminallm/users/u/a/report.md",
        )
        assert not any(label.startswith("/") for label in labels), labels
        assert labels[0] != labels[1], (
            f"the two reports are still indistinguishable: {labels}"
        )

    def test_inline_text_takes_no_part_in_the_widening(self):
        from liminallm.service.rag import INLINE_PATH

        assert self._labels(INLINE_PATH, "reports/status.md") == [
            "inline text",
            "status.md",
        ]

    @staticmethod
    def _labels_with(hints, *paths):
        from types import SimpleNamespace

        from liminallm.service.agent_tools import chunk_labels
        from liminallm.service.rag import SourceHint

        return chunk_labels(
            [SimpleNamespace(fs_path=path) for path in paths],
            {
                key: SourceHint(title=title, origin_id=key)
                for key, title in hints.items()
            },
        )

    def test_a_hinted_title_holds_the_label_it_uses(self):
        """A hinted source has no suffixes to compete with, but it does take
        a label, and a path that would render the same one has to widen."""
        assert self._labels_with(
            {"reading:1": "report.md"}, "reading:1", "manuals/report.md"
        ) == ["report.md", "manuals/report.md"]

    def test_a_hinted_title_that_collides_with_nothing_widens_nothing(self):
        assert self._labels_with(
            {"reading:1": "notes.md"}, "reading:1", "manuals/report.md"
        ) == ["notes.md", "report.md"]

    def test_a_hint_for_a_document_not_shown_holds_nothing(self):
        """The hints cover every reading the conversation authorizes, not
        only what this search returned. A document the model was not shown
        must not widen one it was."""
        assert self._labels_with(
            {"reading:1": "report.md"}, "manuals/report.md"
        ) == ["report.md"]


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
