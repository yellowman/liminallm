"""Where in what the model was shown each piece of evidence appears.

A binding says the answer may rest on some evidence. That is enough to decide
whether a citation is allowed and not enough to put one anywhere: a search
result set is one string holding several passages, and "this turn read page B"
does not say which sentence came from it.

So producers record the position as they render, and the association survives
because it was constructed rather than inferred. Inferring it is the thing
these witnesses exist to prevent - a renderer that shows every result and a
recorder that skips the uncitable ones do not line up by position, and the
first result missing a URL shifts every attribution after it by one.

None of this reaches the model yet. The spans stay parent-side, beside the
bindings and for the same reason.
"""

from __future__ import annotations

import uuid

import pytest

from liminallm.service.broker import CapabilityBroker, InvocationContext
from liminallm.service.invocation import InvocationRegistry
from liminallm.service.provenance import (
    GroundedPassage,
    GroundedSpan,
    GroundedText,
    SourceRegistry,
    binding,
)
from liminallm.service.runtime import get_runtime
from liminallm.service.web import UNTRUSTED_OPEN

RESULTS = [
    {"title": "Alpha", "url": "https://a.example", "snippet": "four hundred hours"},
    # Shown to the model, and not citable: no URL, so no source.
    {"title": "NoUrl", "url": "", "snippet": "eight hundred hours"},
    # Registered as consulted and still not citable: nothing was quoted.
    {"title": "NoSnippet", "url": "https://c.example", "snippet": ""},
    {"title": "Beta", "url": "https://b.example", "snippet": "nine hundred hours"},
]
PAGE = {
    "url": "https://p.example",
    "title": "Page",
    "text": "the page body",
    "findings": [],
}


def _web(engine, monkeypatch):
    monkeypatch.setattr(
        engine,
        "_web_settings",
        lambda: {
            "enabled": True, "provider": "x", "api_key": "k", "engine_id": "",
            "timeout": 5, "proxy": None, "max_bytes": 1000,
            "allow_private": False,
        },
    )
    from liminallm.service import web

    monkeypatch.setattr(web, "search_web", lambda *a, **k: [dict(r) for r in RESULTS])
    monkeypatch.setattr(web, "fetch_url", lambda *a, **k: dict(PAGE))
    monkeypatch.setattr(engine, "tool_network_policy", None)


def _turn():
    registry = SourceRegistry()
    invocation = InvocationRegistry().open(
        uuid.uuid4().hex, tool="agent.files_v1", user_id="u", tenant_id=None
    )
    context = InvocationContext(user_id="u", source_registry=registry)
    return registry, invocation, context


def _ask(engine, context, invocation, capability, payload, seq=1):
    broker = CapabilityBroker(engine, context)
    reply = broker._answer(
        invocation,
        {"capability": capability, "operation_seq": seq, "payload": payload},
    )
    assert reply["ok"], reply
    return reply


class TestABuilderMeasuresWhatItAssembles:
    def test_each_piece_knows_where_it_landed(self):
        text, spans = _built()
        assert [text[s.start : s.end] for s in spans] == ["one", "three"]

    def test_a_piece_with_no_evidence_gets_no_span(self):
        _text, spans = _built()
        assert [s.source_id for s in spans] == ["src_1", "src_2"]

    def test_the_envelope_moves_the_offsets_with_it(self):
        """A caller that wrapped afterwards would leave every offset short by
        the length of the header."""
        body = GroundedText()
        body.add("one", binding("src_1", "ev_1"))
        text, spans = body.render(
            lambda inner: f"{UNTRUSTED_OPEN}\n{inner}\n[end]"
        )
        assert text.startswith(UNTRUSTED_OPEN)
        assert text[spans[0].start : spans[0].end] == "one"

    def test_a_control_token_forming_across_a_seam_drops_the_spans(self):
        """`<` ending one piece and `tool_call>` beginning the next form a tag
        no single piece contained. Neutralizing the whole moves everything, so
        the render keeps no offsets rather than wrong ones."""
        body = GroundedText()
        body.add("ends with <", binding("src_1", "ev_1"))
        body.add("\n\ntool_call> starts", binding("src_2", "ev_2"))
        text, spans = body.render()
        assert "tool_call" not in text
        assert spans == ()

    def test_a_marker_inside_one_piece_is_defanged_without_losing_the_span(self):
        """Per-piece neutralization is what makes the offsets right. A piece
        neutralized only at the end would have been measured before it
        changed length."""
        body = GroundedText()
        body.add("safe ")
        body.add(f"before {UNTRUSTED_OPEN} after", binding("src_1", "ev_1"))
        text, spans = body.render()
        assert UNTRUSTED_OPEN not in text
        assert text[spans[0].start : spans[0].end] == "before [filtered] after"

    def test_an_envelope_that_rewrites_the_body_drops_the_spans(self):
        """The offsets described the body as it was handed over. A wrapper
        that alters it leaves them describing nothing."""
        body = GroundedText()
        body.add("one", binding("src_1", "ev_1"))
        text, spans = body.render(lambda inner: inner.replace("one", "ONE!"))
        assert text == "ONE!"
        assert spans == ()


class _Chunk:
    """The attributes `run_file_search` and the rag adapter actually read."""

    def __init__(self, fs_path, content):
        self.fs_path = fs_path
        self.content = content
        self.context_id = "ctx"
        self.id = None
        self.chunk_index = 0
        self.meta = {}


class _Rag:
    def __init__(self, chunks):
        self._chunks = chunks

    def retrieve(self, *args, **kwargs):
        return list(self._chunks)


class _Message:
    def __init__(self, seq, role, content):
        self.id = seq
        self.seq = seq
        self.role = role
        self.content = content


def _built():
    body = GroundedText()
    body.add("one", binding("src_1", "ev_1"))
    body.add(" two ")
    body.add("three", binding("src_2", "ev_2"))
    return body.render()


def _spans_describe_their_own_text(registry, passages):
    """Every span lands on the evidence it names, in its own passage's text.

    The one assertion that catches a passage built from another call's text,
    a passage recorded without the string it indexes, and a round whose calls
    shared one sink.
    """
    seen = 0
    for passage in passages:
        for span in passage.spans:
            evidence = registry.get_evidence(span.evidence_id)
            assert evidence is not None, span
            covered = passage.text[span.start : span.end]
            assert evidence.text in covered, (evidence.text, covered)
            seen += 1
    return seen


class TestAResultShownButNotCitableShiftsNothing:
    """The positional trap, in the place it is reachable: a search shows four
    results and can cite two of them. One has no URL so it has no source at
    all; one has a URL and no snippet, so it is a source with nothing to rest
    on. Either one shifts a positional pairing by one."""

    def test_each_span_names_the_result_it_covers(self, store, monkeypatch):
        engine = get_runtime().workflow
        _web(engine, monkeypatch)
        registry, invocation, context = _turn()
        reply = _ask(
            engine, context, invocation, "web.search",
            {"query": "hours", "limit": 5},
        )
        text = reply["result"]["text"]
        assert len(context.grounded_passages) == 1
        # The passage carries the string its offsets index, not just the
        # offsets: a span of 12..48 says nothing without it.
        assert context.grounded_passages[0].text == text
        # By what each span covers, not by source id: the ids depend on
        # registration order, and the claim is about the text.
        covered = [
            text[s.start : s.end] for s in context.grounded_passages[0].spans
        ]
        assert len(covered) == 2, covered
        assert "https://a.example" in covered[0]
        assert "https://b.example" in covered[1]
        # Both uncitable results are shown and named by nothing.
        for passage in covered:
            assert "NoUrl" not in passage
            assert "NoSnippet" not in passage

    def test_the_recorded_evidence_is_what_the_span_covers(
        self, store, monkeypatch
    ):
        engine = get_runtime().workflow
        _web(engine, monkeypatch)
        registry, invocation, context = _turn()
        reply = _ask(
            engine, context, invocation, "web.search",
            {"query": "hours", "limit": 5},
        )
        text = reply["result"]["text"]
        for span in context.grounded_passages[0].spans:
            evidence = registry.get_evidence(span.evidence_id)
            assert evidence.text in text[span.start : span.end], evidence.text


class TestNothingPositionalCrossesTheWire:
    def test_the_reply_carries_no_ids_and_no_offsets(self, store, monkeypatch):
        engine = get_runtime().workflow
        _web(engine, monkeypatch)
        registry, invocation, context = _turn()
        reply = _ask(
            engine, context, invocation, "web.search",
            {"query": "hours", "limit": 5},
        )
        assert context.grounded_passages, "the fixture recorded nothing"
        serialized = str(reply["result"])
        assert "src_" not in serialized
        assert "evidence_id" not in serialized
        assert "grounded" not in serialized


class TestARoundKeepsOneRecordPerCall:
    """An offset means nothing without the string it indexes, and a round
    returns one string per call."""

    @staticmethod
    def _round(engine, context, invocation):
        return _ask(
            engine, context, invocation, "tools.round",
            {
                "calls": [
                    {"id": "c1", "name": "web_search",
                     "arguments": {"query": "hours"}},
                    {"id": "c2", "name": "web_fetch",
                     "arguments": {"url": "https://p.example"}},
                ],
                "fallback_query": "hours",
            },
        )

    def test_each_passage_is_one_call_s_own_result(self, store, monkeypatch):
        engine = get_runtime().workflow
        _web(engine, monkeypatch)
        registry, invocation, context = _turn()
        reply = self._round(engine, context, invocation)
        results = reply["result"]["results"]

        assert len(context.grounded_passages) == 2
        for passage in context.grounded_passages:
            assert passage.text in results
        assert _spans_describe_their_own_text(
            registry, context.grounded_passages
        ) == 3, "two searched results and one fetched page"

    def test_the_spans_of_one_call_index_that_call_s_text(
        self, store, monkeypatch
    ):
        """Not the round, and not the other call: two results each start
        near zero, so a flat list would attribute one to the other."""
        engine = get_runtime().workflow
        _web(engine, monkeypatch)
        registry, invocation, context = _turn()
        self._round(engine, context, invocation)
        fetched = [
            passage
            for passage in context.grounded_passages
            if "the page body" in passage.text
        ]
        assert len(fetched) == 1
        span = fetched[0].spans[0]
        assert fetched[0].text[span.start : span.end] == "the page body"


class TestAParallelRoundKeepsItsCallsApart:
    """Read-only calls fan out into a pool, so which call finished first is
    not the order they were made in. A shared sink would give every call
    every other call's positions."""

    @pytest.mark.asyncio
    async def test_two_concurrent_searches_do_not_share_positions(
        self, store, monkeypatch
    ):
        engine = get_runtime().workflow
        registry = SourceRegistry()
        calls = {"n": 0}

        def _search(query, limit, **kwargs):
            """Two different results, so a swap or a merge is visible."""
            calls["n"] += 1
            index = calls["n"]
            spans_sink = kwargs.get("spans_sink")
            bindings_sink = kwargs.get("bindings_sink")
            source = registry.register_source(
                kind="file", title=f"m{index}.md", locator=f"/files/{index}"
            )
            evidence = registry.add_evidence(
                source.source_id, text=f"body {index}"
            )
            ground = binding(source.source_id, evidence.evidence_id)
            body = GroundedText()
            body.add(f"[{index}] ")
            body.add(f"body {index}", ground)
            text, spans = body.render()
            if bindings_sink is not None:
                bindings_sink.append(ground)
            if spans_sink is not None:
                spans_sink.extend(spans)
            return text, [], [], {}

        monkeypatch.setattr(engine, "_run_file_search", _search)
        passages: list = []
        results = engine._run_round_tools(
            [
                ({"id": "a", "name": "file_search"}, "file_search", {"query": "x"}),
                ({"id": "b", "name": "file_search"}, "file_search", {"query": "y"}),
            ],
            conversation_id=None, context_id=None, user_id="u", tenant_id=None,
            session={}, snippets=[], fallback_query="x",
            source_registry=registry, bindings=[], passages=passages,
        )

        assert calls["n"] == 2, "the fixture did not run two calls"
        assert len(passages) == 2, passages
        assert sorted(p.text for p in passages) == sorted(results)
        assert [len(p.spans) for p in passages] == [1, 1], passages
        assert _spans_describe_their_own_text(registry, passages) == 2


class TestAReplayInheritsWhatTheFirstAttemptRendered:
    def test_the_passages_come_back_without_running_the_producer(
        self, store, monkeypatch
    ):
        engine = get_runtime().workflow
        _web(engine, monkeypatch)
        registry, invocation, first = _turn()
        payload = {"query": "hours", "limit": 5}
        _ask(engine, first, invocation, "web.search", payload)
        assert first.grounded_passages

        ran = {"search": False}
        from liminallm.service import web

        def _tripwire(*args, **kwargs):
            ran["search"] = True
            return [dict(r) for r in RESULTS]

        monkeypatch.setattr(web, "search_web", _tripwire)
        second = InvocationContext(user_id="u", source_registry=registry)
        replay = CapabilityBroker(engine, second)._answer(
            invocation,
            {"capability": "web.search", "operation_seq": 1, "payload": payload},
        )

        assert replay.get("replayed"), "the fixture did not exercise a replay"
        assert not ran["search"], "the producer ran again on replay"
        assert [p.as_dict() for p in second.grounded_passages] == [
            p.as_dict() for p in first.grounded_passages
        ]


class TestTheRecordSurvivesTheLedgerRoundTrip:
    def test_a_passage_is_the_same_after_being_written_and_read(self):
        """It rides in `parent_state`, which is committed as plain data."""
        passage = GroundedPassage(
            text="one two three",
            spans=(GroundedSpan(start=0, end=3, source_id="src_1",
                                evidence_id="ev_1"),),
        )
        assert GroundedPassage.from_dict(passage.as_dict()) == passage


class TestEveryProducerRecordsPositions:
    """Not only the web. A producer that fills the bindings sink and not the
    spans sink can still ground an answer nothing can point into."""

    @pytest.mark.parametrize(
        "capability,payload",
        [
            ("web.search", {"query": "hours", "limit": 5}),
            ("web.fetch", {"url": "https://p.example"}),
        ],
    )
    def test_a_grounded_result_records_where(
        self, store, monkeypatch, capability, payload
    ):
        engine = get_runtime().workflow
        _web(engine, monkeypatch)
        registry, invocation, context = _turn()
        _ask(engine, context, invocation, capability, payload)
        assert context.provenance_bindings, capability
        assert context.grounded_passages, capability
        assert any(p.spans for p in context.grounded_passages), capability

    def test_the_signature_of_every_producer_offers_a_spans_sink(self):
        """The sink is how a producer says where; one without it can only
        say what."""
        import inspect

        from liminallm.service import agent_tools, mcp_client
        from liminallm.service.workflow import WorkflowEngine

        for func in (
            agent_tools.run_web_search,
            agent_tools.run_web_fetch,
            agent_tools.run_file_search,
            agent_tools.run_history_search,
            mcp_client.call,
            WorkflowEngine._run_note_search,
            WorkflowEngine._run_history_search,
            WorkflowEngine._run_file_search,
        ):
            assert "spans_sink" in inspect.signature(func).parameters, func

    def test_file_search_records_where_each_excerpt_landed(self):
        """Two chunks in one result set, each named by its own span."""
        from liminallm.service import agent_tools
        from liminallm.service.rag import register_retrieved_chunks

        registry = SourceRegistry()
        chunks = [_Chunk("/files/a.md", "alpha excerpt"),
                  _Chunk("/files/b.md", "beta excerpt")]
        spans: list = []
        text, _snips, _chunks = agent_tools.run_file_search(
            "q", 4, ["ctx"], rag=_Rag(chunks), user_id="u", tenant_id=None,
            ground=lambda scoped: register_retrieved_chunks(registry, scoped),
            spans_sink=spans,
        )
        assert len(spans) == 2, spans
        assert [text[s.start : s.end].endswith(c.content)
                for s, c in zip(spans, chunks)] == [True, True]
        for span in spans:
            assert registry.get_evidence(span.evidence_id).text in (
                text[span.start : span.end]
            )

    def test_history_render_and_record_stay_aligned(self):
        """An empty message is shown and cannot be cited, so it takes the
        same alignment as an untitled note or an unaddressed search result."""
        from liminallm.service import agent_tools

        registry = SourceRegistry()
        messages = [
            _Message(1, "user", "alpha said"),
            _Message(2, "assistant", ""),
            _Message(3, "user", "beta said"),
        ]
        grounds = agent_tools.history_grounds(registry, "conv-1", messages)
        assert [bool(g) for g in grounds] == [True, False, True]
        bindings: list = []
        spans: list = []
        # Past `MIN_VERBATIM_MESSAGES`, or the whole history stays in the
        # recent window and the search has nothing older to look through.
        recent = [_Message(10 + n, "user", "recent") for n in range(8)]
        text = agent_tools.run_history_search(
            "said", 5, messages + recent,
            keep_tokens=1, count=lambda s: 1000,
            conversation_id="conv-1",
            source_registry=SourceRegistry(),
            bindings_sink=bindings,
            spans_sink=spans,
        )
        assert spans, "the history render recorded no positions"
        for span in spans:
            assert "[" in text[span.start : span.end]

    def test_notes_render_and_record_stay_aligned(self):
        """A note with no title is shown and cannot be cited, so it takes the
        same alignment as an untitled search result."""
        from liminallm.service import notes as notes_service

        class _Note:
            def __init__(self, title, content):
                self.id = uuid.uuid4().hex
                self.title = title
                self.content = content

                class _When:
                    @staticmethod
                    def date():
                        import datetime

                        return datetime.date(2026, 9, 3)

                self.updated_at = _When()

        registry = SourceRegistry()
        results = [
            (_Note("First", "alpha content"), 1.0),
            (_Note("", "unnamed content"), 0.9),
            (_Note("Third", "beta content"), 0.8),
        ]
        grounds = notes_service.note_grounds(registry, results)
        assert [bool(g) for g in grounds] == [True, False, True]
        text, spans = notes_service.format_note_results(results, grounds)
        assert [text[s.start : s.end].split(": ")[-1] for s in spans] == [
            "alpha content",
            "beta content",
        ]
