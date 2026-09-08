"""The workflow trace is execution evidence, not conversation history.

`workflow_trace` says which nodes ran, what each returned, which edge was
taken and what failed. That is worth having while a request executes, and it
is what the engine's own tests read. It was also being written into the
assistant message's metadata, returned in the chat response, streamed to the
browser as `event: trace`, and read back out of history for every later turn -
so a node's outputs, a tool's arguments and a failure's text became durable
client-visible state nobody asked for.

The lifetime this pins:

    workflow executes -> transient trace
        -> sanitized diagnostic summary, internal callers -> gone

A log is a retention system too, so the execution log gets a summary built
from the trace rather than the trace itself.

The boundary is `chat_turn.public()`. Above it the trace exists; below it -
the message row, the chat response, the stream, the history API - it does not.
The two things the browser genuinely needs are projected on purpose instead:
the tool names it shows as icons, through `tool_progress` events, and the
injection finding *kinds*, through a field carrying only those kinds.
"""

from __future__ import annotations

import inspect
import json
import uuid

import pytest
from structlog.testing import CapturingLogger

from liminallm.api import chat_turn, routes, schemas
from liminallm.logging import log_workflow_trace
from liminallm.service.auth import AuthContext
from liminallm.service.runtime import get_runtime
from liminallm.service.workflow_streaming import WorkflowStreamingMixin

#: A value no legitimate public projection can have a reason to carry. Put in
#: the places a trace entry nests things: a node's outputs, a failure's text,
#: and a tool call's arguments and result.
SENTINEL = "kL9-trace-must-not-escape-7pQ"

TRACE_WITH_SECRETS = [
    {
        "node": "retrieve",
        "status": "ok",
        "outputs": {"scratch": SENTINEL, "rows": [{"passage": SENTINEL}]},
    },
    {
        "node": "answer",
        "status": "error",
        "error": f"backend said: {SENTINEL}",
        "tool_calls": [
            {"tool": "web_fetch", "args": {"url": f"https://x.test/{SENTINEL}"},
             "result": SENTINEL},
        ],
    },
]

#: What the row legitimately keeps. Unchanged by this tranche, and asserted
#: alongside the removal so "delete the metadata" cannot pass for "delete the
#: trace".
LEGITIMATE_META = {
    "adapters": [{"id": "a1", "name": "helper"}],
    "adapter_gates": [{"id": "a1", "gate": "on"}],
    "routing_trace": [{"rule": "default", "matched": True}],
    "usage": {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14},
}


def _orchestration(**extra):
    return {
        "content": "The kettle takes three minutes.",
        "workflow_trace": [dict(entry) for entry in TRACE_WITH_SECRETS],
        **LEGITIMATE_META,
        **extra,
    }


def _turn(store):
    user = store.create_user(email=f"trace_{uuid.uuid4().hex[:8]}@example.com")
    conversation = store.create_conversation(title="trace lifetime", user_id=user.id)
    return chat_turn.Turn(
        principal=AuthContext(user_id=user.id, role="user", tenant_id=None),
        conversation_id=conversation.id,
        context_id=None,
        workflow_id=None,
        user_content="how long does the kettle take",
        user_message=None,
        needs_title=False,
    )


async def _finished(store, orchestration=None):
    """Finish a turn and read the row back, not the object that was returned."""
    turn = _turn(store)
    message = await chat_turn.finish(
        get_runtime(), turn, orchestration or _orchestration()
    )
    stored = store.get_message(message.id)
    assert stored is not None, "the assistant message was never persisted"
    return turn, stored


class _ToolCallingBackend:
    """A backend whose completion reports a tool call with a secret argument.

    The streamed node copies the backend's `message_done` data into the trace
    entry, so this is how a real turn gets a nested secret into the trace
    without inventing one: it is the shape `web_fetch` produces.

    `supports_stream_cancel` is the declaration a backend must make before the
    engine will stream it (`LLMService.stream_is_cancellable`). True here for
    the reason it is true of the stub: this one never blocks inside a read.
    """

    supports_stream_cancel = True

    def generate_stream(self, messages, adapters, *, user_id=None):
        yield {"event": "token", "data": "Three "}
        yield {"event": "token", "data": "minutes."}
        yield {
            "event": "message_done",
            "data": {
                "content": "Three minutes.",
                "usage": {"total_tokens": 4},
                "tool_calls": [
                    {"tool": "web_fetch",
                     "args": {"url": f"https://x.test/{SENTINEL}"}},
                ],
                "injection_findings": ["override-instructions"],
            },
        }

    def generate(self, messages, adapters, *, user_id=None):
        return {"content": "Three minutes.", "usage": {"total_tokens": 4}}


@pytest.fixture
def tool_calling_backend():
    """The backend above, on the plain streamed path.

    Web tools off so the turn takes `llm.generic` rather than the tool agent:
    the agent would go looking for real pages, and what is under test is what
    crosses the API boundary, not which workflow a turn picks.
    """
    runtime = get_runtime()
    previous = runtime.llm.backend
    settings = runtime.workflow.settings
    was_web = getattr(settings, "web_tools_enabled", None)
    runtime.llm.backend = _ToolCallingBackend()
    settings.web_tools_enabled = False
    try:
        yield
    finally:
        runtime.llm.backend = previous
        settings.web_tools_enabled = was_web


class TestTheEngineKeepsItsTransientTrace:
    """The trace is not deleted from the executor. It is useful evidence."""

    @pytest.mark.asyncio
    async def test_a_blocking_run_still_says_which_nodes_ran(self):
        result = await get_runtime().workflow.run(
            None, None, "hello", None, user_id=None
        )
        trace = result.get("workflow_trace")
        assert trace, "the engine stopped recording which nodes ran"
        assert [entry.get("node") for entry in trace if entry.get("node")]

    @pytest.mark.asyncio
    async def test_a_streamed_run_can_be_observed_by_its_caller(self):
        """The error paths never reach a completion, so the caller needs a
        live sink to see which nodes ran. Internal only: no route passes one."""
        sink: list = []
        events = [
            event
            async for event in get_runtime().workflow.run_streaming(
                None, None, "hello", None, user_id=None, trace_sink=sink
            )
        ]
        assert events, "the stream produced nothing"
        assert sink, "the caller's trace sink was never filled"
        assert [entry.get("node") for entry in sink if entry.get("node")]

    def test_no_route_asks_for_the_sink(self):
        assert "trace_sink" not in inspect.getsource(routes), (
            "a transport is collecting the internal trace; the sink is for "
            "the engine's own callers"
        )


class TestNothingDurableCarriesTheTrace:
    @pytest.mark.asyncio
    async def test_the_stored_message_has_no_workflow_trace_key(self, store):
        _, stored = await _finished(store)
        assert "workflow_trace" not in (stored.meta or {}), (
            "the assistant row still carries the execution trace"
        )

    @pytest.mark.asyncio
    async def test_the_metadata_that_belongs_there_survives_intact(self, store):
        _, stored = await _finished(store)
        for key, value in LEGITIMATE_META.items():
            assert (stored.meta or {}).get(key) == value, (
                f"{key} was lost or altered while removing the trace"
            )

    @pytest.mark.asyncio
    async def test_no_nested_trace_value_reaches_the_row(self, store):
        _, stored = await _finished(store)
        assert SENTINEL not in json.dumps(stored.meta or {}, default=str)
        assert SENTINEL not in json.dumps(stored.content_struct or {}, default=str)
        assert SENTINEL not in (stored.content or "")

    @pytest.mark.asyncio
    async def test_the_turn_does_not_keep_the_trace_after_it_is_finished(
        self, store
    ):
        """`Turn.orchestration` outlives the workflow and is read afterwards."""
        turn, _ = await _finished(store)
        assert "workflow_trace" not in turn.orchestration


class TestTheExecutionLogIsASummaryNotTheTrace:
    """A log is a retention system too.

    The trace was handed to structlog whole. The redaction processor there
    walks top-level event keys and rewrites string values under sensitive-
    looking names, so `trace=[{outputs, tool_calls, ...}]` passed through it
    untouched: the same node outputs and tool arguments this tranche keeps out
    of rows and responses were being copied into a file that outlives the
    request. What is logged now is an allowlist - which nodes ran and how they
    ended - assembled from the trace rather than filtered out of it.
    """

    def _logged(self, trace):
        capture = CapturingLogger()
        log_workflow_trace(trace, logger=capture)
        assert len(capture.calls) == 1, capture.calls
        return capture.calls[0]

    def test_no_nested_value_reaches_the_logger(self):
        call = self._logged(TRACE_WITH_SECRETS)
        assert SENTINEL not in json.dumps(
            {"args": call.args, "kwargs": call.kwargs}, default=str
        ), call

    def test_an_ordinary_looking_key_is_dropped_too(self):
        """`outputs.scratch` names nothing sensitive. It is dropped because it
        was never on the list, which is the difference between an allowlist
        and a redaction pass."""
        call = self._logged(TRACE_WITH_SECRETS)
        emitted = json.dumps(call.kwargs, default=str)
        for absent in ("scratch", "outputs", "tool_calls", "args", "result",
                       "error\":", "backend said", "web_fetch"):
            assert absent not in emitted, f"{absent} survived: {emitted}"

    def test_what_the_log_is_for_survives(self):
        call = self._logged(TRACE_WITH_SECRETS)
        assert call.args == ("workflow_trace",)
        assert call.kwargs == {
            "trace_length": 2,
            "error_count": 1,
            "nodes": [
                {"node": "retrieve", "status": "ok"},
                {"node": "answer", "status": "error"},
            ],
        }

    def test_a_status_the_engine_never_sets_is_not_logged_verbatim(self):
        """`status` arrives inside a node result, and a result is assembled
        from a handler's return value - so the value is bounded here rather
        than trusted."""
        call = self._logged([{"node": "n", "status": f"ok {SENTINEL}"}])
        assert call.kwargs["nodes"] == [{"node": "n", "status": "other"}]

    def test_an_empty_trace_still_logs_a_shape(self):
        call = self._logged([])
        assert call.kwargs == {"trace_length": 0, "error_count": 0, "nodes": []}

    def test_a_real_streamed_turn_logs_no_secret_anywhere(
        self, client, auth_headers, tool_calling_backend
    ):
        """Not only this function: nothing the engine logs during a turn whose
        trace nests a secret in a tool argument may carry it."""
        engine = get_runtime().workflow
        capture = CapturingLogger()
        previous = engine.logger
        engine.logger = capture
        try:
            with client.websocket_connect("/v1/chat/stream") as ws:
                ws.send_json({
                    "access_token": auth_headers["Authorization"].split()[1],
                    "message": "how long does the kettle take",
                    "stream": True,
                })
                for _ in range(200):
                    if ws.receive_json().get("event") in ("message_done", "error"):
                        break
        finally:
            engine.logger = previous
        assert capture.calls, "the engine logged nothing; the witness is vacuous"
        assert any(call.args[:1] == ("workflow_trace",) for call in capture.calls), (
            f"the turn logged no workflow trace: {[c.args for c in capture.calls]}"
        )
        emitted = json.dumps(
            [{"args": c.args, "kwargs": c.kwargs} for c in capture.calls], default=str
        )
        assert SENTINEL not in emitted, emitted


class TestAFailedWorkflowParksNoTraceInTheCache:
    """The rollback record is a record that a workflow stopped, not a copy of
    what it was doing.

    The state written here goes to Redis with its own TTL, and the delete that
    follows it is best effort - so a trace written into it survives a cache
    the engine could not reach. `_retire_workflow_state` already states the
    rule for the successful path: no terminal state is written, because it
    would be a second copy of a conversation's content.
    """

    @pytest.mark.asyncio
    async def test_the_rollback_state_carries_a_length_not_a_trace(
        self, monkeypatch
    ):
        engine = get_runtime().workflow
        if engine.cache is None:
            pytest.skip("no cache configured; nothing is persisted to inspect")
        written: list = []
        real = engine.cache.set_workflow_state

        async def spy(state_key, state):
            written.append((state_key, state))
            return await real(state_key, state)

        monkeypatch.setattr(engine.cache, "set_workflow_state", spy)
        rollback = await engine._rollback_workflow(
            f"wf-{uuid.uuid4().hex[:8]}",
            [dict(entry) for entry in TRACE_WITH_SECRETS],
            {"answer": 3},
        )
        assert written, "the rollback wrote no state to inspect"
        for _, state in written:
            assert "workflow_trace" not in state, state
            assert SENTINEL not in json.dumps(state, default=str)
        # What the caller gets back is unchanged: it already said how long the
        # trace was rather than what was in it.
        assert (rollback or {}).get("trace_length") == len(TRACE_WITH_SECRETS)


class TestNoOrdinaryResponseCarriesTheTrace:
    def test_the_chat_response_has_no_such_field(self):
        assert "workflow_trace" not in schemas.ChatResponse.model_fields, (
            "the public chat response still declares the execution trace"
        )

    @pytest.mark.asyncio
    async def test_the_response_body_carries_none_of_it(self, store):
        turn, stored = await _finished(store)
        body = chat_turn.response(turn, stored, ["helper"]).model_dump()
        assert "workflow_trace" not in body
        assert SENTINEL not in json.dumps(body, default=str)
        # The control: the reply is still a reply.
        assert body["content"] == stored.content
        assert body["routing_trace"] == LEGITIMATE_META["routing_trace"]
        assert body["usage"] == LEGITIMATE_META["usage"]

    def test_a_blocking_turn_answers_without_it(self, client, auth_headers):
        response = client.post(
            "/v1/chat",
            headers=auth_headers,
            json={"message": {"content": "hello"}, "stream": False},
        )
        assert response.status_code == 200, response.text
        assert "workflow_trace" not in response.json()["data"]

    def test_no_transport_names_the_trace_at_all(self):
        assert "workflow_trace" not in inspect.getsource(routes), (
            "a transport still reads or projects the execution trace"
        )

    @pytest.mark.parametrize("name", ["websocket_chat", "_responses_stream"])
    def test_a_streaming_transport_projects_the_completion_first(self, name):
        """Both streaming transports take the engine's completed result and
        build their own payload from it. That result is the engine's object,
        so it crosses the boundary before it is used - the same rule the
        blocking transports get from `finish`, stated where they receive it.
        A transport that keeps the raw object leaks nothing today only because
        every key it projects is named by hand."""
        source = inspect.getsource(getattr(routes, name))
        assert "chat_turn.public(" in source, (
            f"{name} uses the engine's completion without projecting it"
        )


class TestNoPublicStreamEventCarriesTheTrace:
    @pytest.mark.asyncio
    async def test_the_engine_emits_no_generic_trace_event(self):
        events = [
            event
            async for event in get_runtime().workflow.run_streaming(
                None, None, "hello", None, user_id=None
            )
        ]
        assert not [e for e in events if e.get("event") == "trace"], (
            "the generic trace event is still an escape hatch for internal "
            "execution state"
        )
        carriers = [
            e for e in events
            if e.get("event") != "message_done"
            and "workflow_trace" in (e.get("data") or {})
        ]
        assert not carriers, f"an event carried the trace: {carriers}"

    def test_a_streamed_turn_shows_the_client_no_trace_and_no_secret(
        self, client, auth_headers, tool_calling_backend
    ):
        """The whole public surface of one turn: every frame, the row it
        stored, and the history the next page load reads."""
        frames = []
        with client.websocket_connect("/v1/chat/stream") as ws:
            ws.send_json({
                "access_token": auth_headers["Authorization"].split()[1],
                "message": "how long does the kettle take",
                "stream": True,
            })
            for _ in range(200):
                frame = ws.receive_json()
                frames.append(frame)
                if frame.get("event") in ("message_done", "error"):
                    break

        done = frames[-1]
        assert done.get("event") == "message_done", frames
        assert "workflow_trace" not in done["data"], (
            "the streaming completion still carries the execution trace"
        )
        assert not [f for f in frames if f.get("event") == "trace"]
        as_json = json.dumps(frames, default=str)
        assert SENTINEL not in as_json, "a tool argument reached the client"

        conversation_id = done["data"]["conversation_id"]
        listing = client.get(
            f"/v1/conversations/{conversation_id}/messages", headers=auth_headers
        )
        assert listing.status_code == 200, listing.text
        history = json.dumps(listing.json(), default=str)
        assert SENTINEL not in history, "a tool argument reached history"
        assert "workflow_trace" not in history


class TestTheRepairErasesTheTracesAlreadyStored:
    """Stopping the writes does not undo the ones already made.

    The repair is the block in `sql/schema.sql`, run the way `migrate.sh`
    runs it. Rows are inserted as an old build wrote them - raw SQL, because
    that is what an old database is.
    """

    @pytest.fixture()
    def repaired(self, store):
        import psycopg

        from tests.harness import apply_schema

        user = store.create_user(email=f"repair_{uuid.uuid4().hex[:8]}@example.com")
        conversation = store.create_conversation(title="old rows", user_id=user.id)
        rows = {
            "with_trace": {**LEGITIMATE_META, "workflow_trace": TRACE_WITH_SECRETS},
            "trace_only": {"workflow_trace": TRACE_WITH_SECRETS},
            "no_trace": dict(LEGITIMATE_META),
            "null_meta": None,
        }
        ids = {}
        with psycopg.connect(store.dsn, autocommit=True) as conn:
            for index, (name, meta) in enumerate(rows.items()):
                row_id = str(uuid.uuid4())
                ids[name] = row_id
                conn.execute(
                    "INSERT INTO message (id, conversation_id, sender, role, "
                    "content, seq, meta) VALUES (%s, %s, 'assistant', "
                    "'assistant', 'an old answer', %s, %s)",
                    (
                        row_id,
                        conversation.id,
                        index + 1,
                        json.dumps(meta) if meta is not None else None,
                    ),
                )
        apply_schema(store.dsn, embedding_dim=64)
        yield ids
        with psycopg.connect(store.dsn, autocommit=True) as conn:
            conn.execute(
                "DELETE FROM message WHERE id = ANY(%s)", (list(ids.values()),)
            )

    def _meta(self, store, row_id):
        return store.get_message(row_id).meta

    def test_the_trace_is_gone_from_an_old_row(self, store, repaired):
        for name in ("with_trace", "trace_only"):
            meta = self._meta(store, repaired[name]) or {}
            assert "workflow_trace" not in meta, name
            assert SENTINEL not in json.dumps(meta, default=str), name

    def test_the_rest_of_the_metadata_is_untouched(self, store, repaired):
        assert self._meta(store, repaired["with_trace"]) == LEGITIMATE_META
        assert self._meta(store, repaired["no_trace"]) == LEGITIMATE_META

    def test_a_row_that_had_only_a_trace_keeps_an_empty_object(
        self, store, repaired
    ):
        """Removing the last key leaves `{}`, not NULL: the column still
        describes a row that has no metadata rather than an unknown one."""
        assert self._meta(store, repaired["trace_only"]) == {}

    def test_null_metadata_survives_the_repair(self, store, repaired):
        assert self._meta(store, repaired["null_meta"]) is None

    def test_running_it_twice_changes_nothing(self, store, repaired):
        from tests.harness import apply_schema

        before = {name: self._meta(store, row) for name, row in repaired.items()}
        apply_schema(store.dsn, embedding_dim=64)
        after = {name: self._meta(store, row) for name, row in repaired.items()}
        assert before == after

    def test_the_repair_is_written_to_match_only_rows_that_have_the_key(self):
        """The idempotence above is a property of this predicate."""
        schema = (
            __import__("pathlib").Path(__file__).resolve().parent.parent
            / "sql" / "schema.sql"
        ).read_text()
        assert "SET meta = meta - 'workflow_trace'" in schema
        assert "WHERE meta ? 'workflow_trace'" in schema


class TestTheInjectionWarningSurvivesTheTrace:
    """The browser warns when a fetched page tried to hijack the turn. It read
    that out of the trace; it reads the classified kinds instead."""

    def test_the_completion_carries_the_finding_kinds(
        self, client, auth_headers, tool_calling_backend
    ):
        frames = []
        with client.websocket_connect("/v1/chat/stream") as ws:
            ws.send_json({
                "access_token": auth_headers["Authorization"].split()[1],
                "message": "read that page for me",
                "stream": True,
            })
            for _ in range(200):
                frame = ws.receive_json()
                frames.append(frame)
                if frame.get("event") in ("message_done", "error"):
                    break

        done = frames[-1]
        assert done.get("event") == "message_done", frames
        assert done["data"].get("injection_findings") == ["override-instructions"], (
            "the sanitized finding kinds no longer reach the client, so the "
            "warning cannot be rendered"
        )

    def test_the_field_carries_kinds_and_nothing_else(
        self, client, auth_headers, tool_calling_backend
    ):
        """Kinds are classifications. Evidence is not."""
        frames = []
        with client.websocket_connect("/v1/chat/stream") as ws:
            ws.send_json({
                "access_token": auth_headers["Authorization"].split()[1],
                "message": "read that page for me",
                "stream": True,
            })
            for _ in range(200):
                frame = ws.receive_json()
                frames.append(frame)
                if frame.get("event") in ("message_done", "error"):
                    break
        findings = frames[-1]["data"].get("injection_findings") or []
        assert findings, "nothing to judge: the turn reported no finding"
        assert all(isinstance(kind, str) for kind in findings), findings
        assert SENTINEL not in json.dumps(findings)

    def test_a_finding_object_is_not_a_kind_and_is_not_projected(self):
        """A scanner finding carries the text it matched. A kind does not, and
        only kinds are projected - so a node that reports whole findings under
        that key contributes nothing rather than the evidence inside them."""
        collected: list = []
        WorkflowStreamingMixin._record_findings(
            collected,
            ["override-instructions", {"type": "persona-hijack", "match": SENTINEL}],
        )
        assert collected == ["override-instructions"]

    def test_a_kind_found_twice_is_reported_once(self):
        collected: list = []
        WorkflowStreamingMixin._record_findings(collected, ["a", "b"])
        WorkflowStreamingMixin._record_findings(collected, ["b", "c", ""])
        assert collected == ["a", "b", "c"]
