"""Every remaining producer says where its answer came from.

S2A gave automatic context grounding a record and S2B gave explicit file
search one. Four producers were still flattening their results into anonymous
text: the web, the vault, the conversation's own record, and remote MCP tools.
Each one has an identity of a different shape, and the point of these is that
the shape is honest - a page is a place, a note is a document with an id, a
chat is one source with many passages, and a remote tool is neither a place
nor a document.

The doubles here are as real as they can be: a live HTTP server for a fetch,
the provider parser the search path actually uses, the store for notes and
messages, and the SDK's own server over Streamable HTTP for MCP.
"""

from __future__ import annotations

import http.server
import threading
import uuid

import pytest

from liminallm.service import agent_tools, mcp_client, web
from liminallm.service import notes as notes_service
from liminallm.service.provenance import SourceRegistry
from liminallm.service.runtime import get_runtime
from tests.mcpfixture import MCPFixture, allow_local

BRAVE_PAYLOAD = {
    "web": {
        "results": [
            {
                "title": "Turbine inspection intervals",
                "url": "https://example.test/turbines",
                "description": "Blades are inspected every 400 flight hours.",
            },
            {
                "title": "Airframe logbook practice",
                "url": "https://example.test/logbook",
                "description": "Each inspection is recorded against its airframe.",
            },
        ]
    }
}


def _results():
    """Result dicts from the parser the search path really uses."""
    return web._results_from_brave(BRAVE_PAYLOAD)


class TestAPageIsIdentifiedByWhereItIs:
    def test_each_result_becomes_a_source_at_its_url(self):
        registry = SourceRegistry()
        bindings = web.register_search_results(registry, _results())

        assert len(bindings) == 2, bindings
        by_url = {s.locator: s for s in registry.sources}
        assert set(by_url) == {
            "https://example.test/turbines",
            "https://example.test/logbook",
        }
        for source in registry.sources:
            assert source.kind == "web"
            # A page has no identity the web gives it beyond its address.
            assert source.origin_id is None
        assert by_url["https://example.test/turbines"].title == (
            "Turbine inspection intervals"
        )
        assert {e.text for e in registry.evidence} == {
            "Blades are inspected every 400 flight hours.",
            "Each inspection is recorded against its airframe.",
        }

    def test_a_result_with_no_url_is_skipped_rather_than_invented(self):
        registry = SourceRegistry()
        bindings = web.register_search_results(
            registry, [{"title": "no address", "url": "", "snippet": "text"}]
        )
        assert bindings == []
        assert registry.sources == ()

    def test_one_page_reached_twice_in_a_turn_is_one_source(self):
        registry = SourceRegistry()
        web.register_search_results(registry, _results())
        web.register_search_results(registry, _results())
        assert len(registry.sources) == 2, registry.sources

    def test_a_fetched_page_records_what_the_model_read(self):
        """Through a real server, so the page dict is the one `fetch_url`
        builds rather than one written to match this test."""

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                body = (
                    b"<html><head><title>Blade manual</title></head>"
                    b"<body><p>Inspect every 400 hours.</p></body></html>"
                )
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            url = f"http://127.0.0.1:{server.server_port}/manual"
            page = web.fetch_url(url, allow_private=True)
            registry = SourceRegistry()
            bindings = web.register_fetched_page(registry, page)
        finally:
            server.shutdown()

        assert len(bindings) == 1, bindings
        source = registry.get_source(bindings[0]["source_id"])
        assert source.kind == "web"
        assert source.locator == page["url"]
        assert source.title == "Blade manual"
        evidence = registry.get_evidence(bindings[0]["evidence_id"])
        assert "Inspect every 400 hours." in evidence.text
        # The envelope is this system's words about the page, not the page.
        assert web.UNTRUSTED_OPEN not in evidence.text


class TestANoteIsADocumentWithAnIdentity:
    def test_the_note_id_is_the_origin_and_the_excerpt_is_the_evidence(
        self, store
    ):
        user = store.create_user(email=f"n_{uuid.uuid4().hex[:8]}@t.local")
        note = store.create_note(
            user_id=user.id,
            title="Inspection cadence",
            content="Blades are inspected every 400 flight hours. " * 20,
        )
        registry = SourceRegistry()
        bindings = notes_service.register_note_results(registry, [(note, 1.0)])

        assert len(bindings) == 1, bindings
        source = registry.get_source(bindings[0]["source_id"])
        assert source.kind == "note"
        assert source.origin_id == f"note:{note.id}"
        # A note is not a place: nothing can be opened at a path.
        assert source.locator is None
        assert source.title == "Inspection cadence"

    def test_the_recorded_passage_is_the_passage_the_model_was_shown(
        self, store
    ):
        """The render caps the excerpt. Recording a longer one would make the
        evidence something the answer could not actually have rested on."""
        user = store.create_user(email=f"n_{uuid.uuid4().hex[:8]}@t.local")
        note = store.create_note(
            user_id=user.id, title="Long note", content="word " * 2000
        )
        registry = SourceRegistry()
        notes_service.register_note_results(registry, [(note, 1.0)])
        shown = notes_service.format_note_results([(note, 1.0)])

        recorded = registry.evidence[0].text
        assert len(recorded) <= notes_service.NOTE_SEARCH_EXCERPT_CHARS
        assert recorded in shown, "the passage recorded was never displayed"


class TestTheConversationIsOneSourceWithManyPassages:
    def test_messages_are_passages_and_not_separate_documents(self):
        from types import SimpleNamespace

        registry = SourceRegistry()
        messages = [
            SimpleNamespace(id="m1", role="user", content="what is the cadence"),
            SimpleNamespace(id="m2", role="assistant", content="every 400 hours"),
        ]
        bindings = agent_tools.register_history_matches(registry, "conv-7", messages)

        assert len(bindings) == 2, bindings
        assert len(registry.sources) == 1, registry.sources
        source = registry.sources[0]
        assert source.kind == "conversation"
        assert source.origin_id == "conversation:conv-7"
        assert [e.locator.block_id for e in registry.evidence] == ["m1", "m2"]

    def test_a_conversation_with_no_id_records_nothing(self):
        """The source would have no identity, and one derived from the turn
        would merge two unrelated chats."""
        from types import SimpleNamespace

        registry = SourceRegistry()
        assert agent_tools.register_history_matches(
            registry, None, [SimpleNamespace(id="m1", content="text")]
        ) == []
        assert registry.sources == ()


class TestARemoteToolIsNeitherAPlaceNorADocument:
    def test_the_tool_is_the_source_and_each_answer_a_passage(self):
        with MCPFixture("inventory", {"lookup_part": "part A1 in stock"}) as fixture:
            tools = mcp_client.run_sync(
                mcp_client.discover([fixture.as_server()], policy=allow_local())
            )
            registry = SourceRegistry()
            sink: list = []
            first = mcp_client.run_sync(
                mcp_client.call(
                    tools[0], {"sku": "A1"}, policy=allow_local(),
                    source_registry=registry, bindings_sink=sink,
                )
            )
            mcp_client.run_sync(
                mcp_client.call(
                    tools[0], {"sku": "A2"}, policy=allow_local(),
                    source_registry=registry, bindings_sink=sink,
                )
            )

        assert "part A1 in stock" in first
        assert len(sink) == 2, sink
        # Two calls to one tool are one source with two passages: the tool is
        # what is stable, and its answers are what it said.
        assert len(registry.sources) == 1, registry.sources
        source = registry.sources[0]
        assert source.kind == "mcp"
        # The artifact row id, not the display name - see the collision case
        # below. Written out rather than recomputed, so the encoding itself is
        # pinned and not just compared with its own generator.
        assert source.origin_id == 'mcp:["fixture-inventory","lookup_part"]'
        assert source.title == "inventory :: lookup_part", source
        # The server URL is where the tool lives, not where the answer can be
        # read again.
        assert source.locator is None
        assert "part A1 in stock" in registry.evidence[0].text
        # The envelope is added after the recording, so it is not in it.
        assert web.UNTRUSTED_OPEN not in registry.evidence[0].text

    def test_two_servers_of_one_name_are_two_sources(self):
        """A display name is not an identity. `mcp.server` artifacts are not
        unique by name in the schema, so two installations of the same tool
        can legitimately be configured as `inventory`. The model-side name
        projection already tells them apart; the registry must too, or one
        server's evidence is attached to the other."""
        with MCPFixture("inventory", {"lookup": "part A1 from warehouse east"}) as a, \
                MCPFixture("inventory", {"lookup": "part A1 from warehouse west"}) as b:
            first = {**a.as_server(), "artifact_id": "artifact-uuid-a"}
            second = {**b.as_server(), "artifact_id": "artifact-uuid-b"}
            tools = mcp_client.run_sync(
                mcp_client.discover([first, second], policy=allow_local())
            )
            assert len({t.model_name for t in tools}) == 2, (
                f"the fixture did not produce two callable tools: {tools}"
            )
            registry = SourceRegistry()
            sink: list = []
            for tool in tools:
                mcp_client.run_sync(
                    mcp_client.call(
                        tool, {"sku": "A1"}, policy=allow_local(),
                        source_registry=registry, bindings_sink=sink,
                    )
                )

        assert len(sink) == 2, sink
        assert len(registry.sources) == 2, (
            "two servers collapsed into one source: "
            f"{[(s.origin_id, s.title) for s in registry.sources]}"
        )
        # Each source holds its own server's answer, and only that one.
        answers = {
            registry.get_source(b["source_id"]).source_id:
                registry.get_evidence(b["evidence_id"]).text
            for b in sink
        }
        assert len(answers) == 2
        assert {"east" in text for text in answers.values()} == {True, False}

    def test_a_refused_call_grounds_nothing(self):
        """A turn that has read hostile input loses egress. It read nothing
        it could rest on, so it records nothing."""
        from liminallm.service import taint

        with MCPFixture("inventory", {"lookup_part": "part A1 in stock"}) as fixture:
            tools = mcp_client.run_sync(
                mcp_client.discover([fixture.as_server()], policy=allow_local())
            )
            session: dict = {}
            taint.record_findings(session, [{"type": "persona-hijack"}])
            registry = SourceRegistry()
            sink: list = []
            answer = mcp_client.run_sync(
                mcp_client.call(
                    tools[0], {"sku": "A1"}, policy=allow_local(), session=session,
                    source_registry=registry, bindings_sink=sink,
                )
            )

        assert "part A1 in stock" not in answer
        assert sink == [], sink
        assert registry.sources == (), registry.sources


class TestTheEvidenceIsTheTextTheModelWasGiven:
    """Grounding describes what reached the model, and `wrap_untrusted`
    neutralizes control markers on the way. A passage recorded before that
    step is a passage the model never saw: the source wrote a marker, the
    model read `[filtered]`, and the evidence would still hold the marker a
    citation is later checked against."""

    MARKER = "Fact " + web.UNTRUSTED_OPEN + " remains"

    def test_a_remote_answer_is_recorded_as_the_model_reads_it(self):
        with MCPFixture("inventory", {"lookup": self.MARKER}) as fixture:
            tools = mcp_client.run_sync(
                mcp_client.discover([fixture.as_server()], policy=allow_local())
            )
            registry = SourceRegistry()
            sink: list = []
            shown = mcp_client.run_sync(
                mcp_client.call(
                    tools[0], {}, policy=allow_local(),
                    source_registry=registry, bindings_sink=sink,
                )
            )

        # The envelope opens with the marker itself, so "the model never saw
        # one" is "it appears once, as the delimiter" rather than "not at all".
        assert shown.count(web.UNTRUSTED_OPEN) == 1, shown[:300]
        assert "[filtered]" in shown, shown[:300]
        recorded = registry.get_evidence(sink[0]["evidence_id"]).text
        assert web.UNTRUSTED_OPEN not in recorded, (
            f"the evidence holds a marker the model never read: {recorded!r}"
        )
        assert "[filtered]" in recorded, recorded

    def test_a_fetched_page_is_recorded_as_the_model_reads_it(self):
        # Plain text, not HTML: the extractor treats `<<<MARKER>>>` as a tag
        # and strips it to `<<>>` before neutralization is ever reached, so an
        # HTML fixture would be red for the wrong reason. A served text file
        # is a page too, and it is where the marker survives to the step under
        # test.
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                body = self.server.page.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        server.page = self.MARKER
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            registry = SourceRegistry()
            sink: list = []
            shown, _findings = agent_tools.run_web_fetch(
                f"http://127.0.0.1:{server.server_port}/page",
                settings=_web_enabled_settings(),
                logger=get_runtime().workflow.logger,
                source_registry=registry,
                bindings_sink=sink,
            )
        finally:
            server.shutdown()

        assert sink, "the fetch grounded nothing"
        assert "[filtered]" in shown, shown[:300]
        recorded = registry.get_evidence(sink[0]["evidence_id"]).text
        assert web.UNTRUSTED_OPEN not in recorded, (
            f"the evidence holds a marker the model never read: {recorded!r}"
        )
        assert "[filtered]" in recorded, recorded


def _web_enabled_settings():
    """The real Settings object with web access on, not a stand-in.

    `web_settings()` reads declared fields off it, so a hand-made namespace
    would encode this test's belief about that list rather than the list.
    """
    import copy

    settings = copy.copy(get_runtime().workflow.settings)
    settings.web_tools_enabled = True
    settings.web_fetch_allow_private = True
    return settings


def _conversation_with_older_turns(store, monkeypatch):
    """A chat with turns the model can no longer see verbatim.

    `history_search` searches only the span outside the recent window, so a
    conversation whose whole history still fits in the prompt returns "no
    earlier turns" and grounds nothing - measured, which is why the budget is
    shrunk here rather than the message count guessed at.
    """
    user = store.create_user(email=f"h_{uuid.uuid4().hex[:8]}@t.local")
    conversation = store.create_conversation(user_id=user.id, title="chat")
    for index in range(20):
        store.append_message(
            conversation.id, user.id, "user",
            f"turbine inspection question {index} " * 40,
        )
        store.append_message(
            conversation.id, user.id, "assistant",
            f"answer about inspection {index} " * 40,
        )
    monkeypatch.setattr(
        get_runtime().workflow, "history_budget", lambda: 200, raising=False
    )
    return user, conversation.id


def _broker(registry, *, user_id=None, conversation_id=None):
    from liminallm.service.broker import CapabilityBroker, InvocationContext

    context = InvocationContext(
        user_id=user_id,
        conversation_id=conversation_id,
        source_registry=registry,
        provenance_bindings=[],
    )
    return CapabilityBroker(get_runtime().workflow, context), context


def _open(user_id=None):
    from liminallm.service.invocation import InvocationRegistry

    return InvocationRegistry().open(
        uuid.uuid4().hex, tool="agent.files_v1", user_id=user_id, tenant_id=None
    )


def _ask(broker, invocation, capability, payload, seq=1):
    return broker._answer(
        invocation,
        {"capability": capability, "operation_seq": seq, "payload": payload},
    )


class TestTheWorkerGetsTheTextAndNotTheAuthority:
    """The rule S2B established at one capability, now at every one of them.
    A reply crosses the pipe to an untrusted worker; an id in it is an id the
    worker can quote back as a citation it never earned."""

    @staticmethod
    def _no_ids(reply, context):
        """The reply carries none of the ids, and the parent kept them all.

        Both halves together, because either alone can pass for the wrong
        reason: a capability that grounded nothing leaks nothing.
        """
        assert reply["ok"], reply
        assert context.provenance_bindings, "this capability grounded nothing"
        flat = repr(reply["result"])
        for forbidden in ("provenance_bindings", "source_id", "evidence_id"):
            assert forbidden not in flat, f"{forbidden} crossed the pipe: {flat[:200]}"

    def test_a_web_search_reply_carries_no_ids(self, store, monkeypatch):
        monkeypatch.setattr(web, "search_web", lambda *a, **kw: _results())
        monkeypatch.setattr(
            get_runtime().workflow.settings, "web_tools_enabled", True,
            raising=False,
        )
        user = store.create_user(email=f"s_{uuid.uuid4().hex[:8]}@t.local")
        registry = SourceRegistry()
        broker, context = _broker(registry, user_id=user.id)
        reply = _ask(
            broker, _open(user.id), "web.search",
            {"query": "turbine inspection", "limit": 2},
        )
        self._no_ids(reply, context)

    def test_a_note_search_reply_carries_no_ids(self, store):
        user = store.create_user(email=f"s_{uuid.uuid4().hex[:8]}@t.local")
        store.create_note(
            user_id=user.id,
            title="Cadence",
            content="Blades are inspected every 400 flight hours.",
        )
        registry = SourceRegistry()
        broker, context = _broker(registry, user_id=user.id)
        reply = _ask(
            broker, _open(user.id), "notes.search",
            {"query": "inspected", "limit": 4},
        )
        self._no_ids(reply, context)

    def test_a_history_search_reply_carries_no_ids(self, store, monkeypatch):
        user, conversation_id = _conversation_with_older_turns(store, monkeypatch)
        registry = SourceRegistry()
        broker, context = _broker(
            registry, user_id=user.id, conversation_id=conversation_id
        )
        reply = _ask(
            broker, _open(user.id), "history.search",
            {"query": "turbine inspection", "limit": 4},
        )
        self._no_ids(reply, context)
        # One conversation, many passages - and each names the message it is.
        assert len(registry.sources) == 1, registry.sources
        assert all(e.locator.block_id for e in registry.evidence)

    def test_the_parent_keeps_what_the_vault_grounded(self, store):
        user = store.create_user(email=f"s_{uuid.uuid4().hex[:8]}@t.local")
        note = store.create_note(
            user_id=user.id,
            title="Cadence",
            content="Blades are inspected every 400 flight hours.",
        )
        registry = SourceRegistry()
        broker, context = _broker(registry, user_id=user.id)
        _ask(broker, _open(user.id), "notes.search", {"query": "inspected", "limit": 4})

        assert context.provenance_bindings, "the vault grounded nothing"
        assert all(
            set(b) == {"context_id", "source_id", "evidence_id"}
            for b in context.provenance_bindings
        )
        for entry in context.provenance_bindings:
            # A note was not retrieved through a knowledge context, and saying
            # it was would put it in a scope it never belonged to.
            assert entry["context_id"] is None, entry
            assert registry.get_source(entry["source_id"]) is not None
            assert registry.get_evidence(entry["evidence_id"]) is not None
        assert registry.sources[0].origin_id == f"note:{note.id}"


class TestAReplayedProducerStillCarriesItsProvenance:
    """The ledger returns a committed reply to a replacement attempt without
    running the handler. Every capability that grounds has to come back with
    its record, not only the one that first needed it."""

    def test_a_replayed_note_search_grounds_the_second_attempt(self, store):
        user = store.create_user(email=f"s_{uuid.uuid4().hex[:8]}@t.local")
        store.create_note(
            user_id=user.id,
            title="Cadence",
            content="Blades are inspected every 400 flight hours.",
        )
        registry = SourceRegistry()
        invocation = _open(user.id)
        payload = {"query": "inspected", "limit": 4}

        broker_a, ctx_a = _broker(registry, user_id=user.id)
        _ask(broker_a, invocation, "notes.search", payload)
        assert ctx_a.provenance_bindings, "attempt A recorded nothing"

        broker_b, ctx_b = _broker(registry, user_id=user.id)
        ran = {"handler": False}
        real = broker_b._notes_search

        def _tripwire(*args, **kwargs):
            ran["handler"] = True
            return real(*args, **kwargs)

        broker_b._notes_search = _tripwire
        reply_b = _ask(broker_b, invocation, "notes.search", payload)

        assert reply_b.get("replayed"), "the fixture did not exercise a replay"
        assert not ran["handler"], "the handler ran again on replay"
        assert ctx_b.provenance_bindings == ctx_a.provenance_bindings, (
            f"the replayed attempt received text with no provenance: "
            f"{ctx_b.provenance_bindings}"
        )
        for entry in ctx_b.provenance_bindings:
            assert registry.get_source(entry["source_id"]) is not None
            assert registry.get_evidence(entry["evidence_id"]) is not None


class TestARoundOfMixedProducersRecordsThemInCallOrder:
    def test_the_web_and_the_vault_land_in_the_order_they_were_called(
        self, store, monkeypatch
    ):
        """The binding list is read in order, so it must follow the calls the
        model made rather than whichever producer answered first."""
        engine = get_runtime().workflow
        monkeypatch.setattr(web, "search_web", lambda *a, **kw: _results())
        monkeypatch.setattr(
            engine.settings, "web_tools_enabled", True, raising=False
        )
        user = store.create_user(email=f"r_{uuid.uuid4().hex[:8]}@t.local")
        store.create_note(
            user_id=user.id,
            title="Cadence",
            content="Blades are inspected every 400 flight hours.",
        )
        registry = SourceRegistry()
        collected: list = []
        engine._run_round_tools(
            [
                ("id0", "web_search", {"query": "turbine inspection"}),
                ("id1", "note_search", {"query": "inspected"}),
            ],
            conversation_id=None,
            context_id=None,
            user_id=user.id,
            tenant_id=None,
            session={},
            snippets=[],
            fallback_query="inspection",
            source_registry=registry,
            bindings=collected,
        )

        kinds = [
            registry.get_source(b["source_id"]).kind for b in collected
        ]
        assert kinds, "the round recorded nothing"
        assert kinds[0] == "web", f"the round folded out of call order: {kinds}"
        assert "note" in kinds, f"the vault recorded nothing: {kinds}"


@pytest.fixture
def store():
    return get_runtime().store
