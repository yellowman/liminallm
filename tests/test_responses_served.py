"""The served Responses API: OpenAI's wire shape over the kernel's chat turn.

Two shape rules carry this surface, so the tests read at the wire level:
success bodies are the bare Responses object (never the Envelope), error
bodies are OpenAI's ``{"error": {...}}`` (never ours). Everything behind the
shape - conversations, ownership, context binding - is the same chat turn
``/v1/chat`` runs, so continuity is asserted through the native routes.
"""

import uuid

import pytest


def _respond(client, headers, body):
    return client.post("/v1/responses", headers=headers, json=body)


def _assert_openai_error(resp, *, status, param=None):
    assert resp.status_code == status, resp.text
    body = resp.json()
    # The whole body is the error object's envelope - nothing of ours beside it.
    assert set(body) == {"error"}
    error = body["error"]
    assert error["message"]
    assert error["type"] == "invalid_request_error"
    if param is not None:
        assert error["param"] == param
    return error


class TestResponsesSuccessShape:
    def test_string_input_returns_bare_responses_object(self, client, auth_headers):
        resp = _respond(client, auth_headers, {"input": "What is a knowledge context?"})

        assert resp.status_code == 200, resp.text
        body = resp.json()
        # Bare Responses object: no Envelope keys anywhere.
        assert "data" not in body and "status_code" not in body
        assert body["object"] == "response"
        assert body["id"].startswith("resp_")
        assert body["status"] == "completed"
        assert body["error"] is None
        assert body["store"] is True
        assert body["previous_response_id"] is None
        assert isinstance(body["model"], str) and body["model"]
        assert isinstance(body["created_at"], int)

        (message,) = body["output"]
        assert message["type"] == "message"
        assert message["role"] == "assistant"
        assert message["id"].startswith("msg_")
        (part,) = message["content"]
        assert part["type"] == "output_text"
        assert isinstance(part["text"], str) and part["text"]
        assert part["annotations"] == []

        usage = body["usage"]
        assert set(usage) == {
            "input_tokens",
            "input_tokens_details",
            "output_tokens",
            "output_tokens_details",
            "total_tokens",
        }
        assert all(
            isinstance(v, int) for k, v in usage.items() if not k.endswith("_details")
        )
        # Typed SDKs require the details objects, zeros when unknown.
        assert isinstance(usage["input_tokens_details"]["cached_tokens"], int)
        assert isinstance(usage["output_tokens_details"]["reasoning_tokens"], int)

    def test_message_items_input(self, client, auth_headers):
        resp = _respond(
            client,
            auth_headers,
            {
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "First part."},
                            {"type": "input_text", "text": "Second part."},
                        ],
                    },
                    {"role": "user", "content": "String-content item."},
                ]
            },
        )

        assert resp.status_code == 200, resp.text
        assert resp.json()["output"][0]["content"][0]["text"]

    def test_metadata_is_echoed(self, client, auth_headers):
        resp = _respond(
            client,
            auth_headers,
            {"input": "hello", "metadata": {"trace": "abc-123"}},
        )

        assert resp.status_code == 200, resp.text
        assert resp.json()["metadata"] == {"trace": "abc-123"}


class TestResponsesContinuity:
    def test_previous_response_id_continues_the_conversation(
        self, client, auth_headers
    ):
        first = _respond(client, auth_headers, {"input": "Remember the number 41."})
        assert first.status_code == 200, first.text
        first_id = first.json()["id"]

        second = _respond(
            client,
            auth_headers,
            {"input": "What number did I mention?", "previous_response_id": first_id},
        )
        assert second.status_code == 200, second.text
        assert second.json()["previous_response_id"] == first_id
        assert second.json()["id"] != first_id

        # Both turns landed in one conversation, visible on the native surface.
        convs = client.get("/v1/conversations", headers=auth_headers).json()["data"][
            "items"
        ]
        assert len(convs) == 1
        messages = client.get(
            f"/v1/conversations/{convs[0]['id']}/messages", headers=auth_headers
        ).json()["data"]["messages"]
        user_texts = [m["content"] for m in messages if m["role"] == "user"]
        assert user_texts == [
            "Remember the number 41.",
            "What number did I mention?",
        ]
        assert sum(1 for m in messages if m["role"] == "assistant") == 2

    def test_context_id_binds_the_thread(self, client, auth_headers):
        ctx = client.post(
            "/v1/contexts",
            headers=auth_headers,
            json={
                "name": "Responses ctx",
                "description": "grounding",
                "text": "The launch code is stored in the blue cabinet.",
            },
        )
        assert ctx.status_code == 201, ctx.text
        context_id = ctx.json()["data"]["id"]

        resp = _respond(
            client,
            auth_headers,
            {"input": "Where is the launch code stored?", "context_id": context_id},
        )
        assert resp.status_code == 200, resp.text

        convs = client.get("/v1/conversations", headers=auth_headers).json()["data"][
            "items"
        ]
        assert len(convs) == 1
        assert convs[0]["active_context_id"] == context_id

    def test_foreign_context_id_is_refused(self, client, auth_headers):
        other = client.post(
            "/v1/auth/signup",
            json={
                "email": f"other_{uuid.uuid4().hex[:8]}@example.com",
                "password": "TestPassword123!",
            },
        )
        other_headers = {
            "Authorization": f"Bearer {other.json()['data']['access_token']}"
        }
        ctx = client.post(
            "/v1/contexts",
            headers=other_headers,
            json={"name": "Not yours", "description": "x"},
        )
        context_id = ctx.json()["data"]["id"]

        resp = _respond(
            client, auth_headers, {"input": "hi", "context_id": context_id}
        )
        # The kernel's ownership verdict (403, same as /v1/chat), reshaped
        # OpenAI-style: no Envelope, and the kernel code rides in error.code.
        assert resp.status_code == 403, resp.text
        body = resp.json()
        assert set(body) == {"error"}
        assert body["error"]["code"] == "forbidden"


class TestResponsesRejections:
    @pytest.mark.parametrize(
        ("body", "param"),
        [
            ({"input": "hi", "tools": [{"type": "function"}]}, "tools"),
            ({"input": "hi", "instructions": "Be terse."}, "instructions"),
            ({"input": "hi", "store": False}, "store"),
            ({"input": ""}, "input"),
            ({"input": []}, "input"),
            ({"input": "hi", "metadata": {str(n): "v" for n in range(17)}}, "metadata"),
            ({"input": "hi", "previous_response_id": "msg_123"}, "previous_response_id"),
        ],
    )
    def test_named_rejections(self, client, auth_headers, body, param):
        _assert_openai_error(
            _respond(client, auth_headers, body), status=400, param=param
        )

    def test_system_role_item_is_refused_by_position(self, client, auth_headers):
        resp = _respond(
            client,
            auth_headers,
            {
                "input": [
                    {"role": "user", "content": "fine"},
                    {"role": "system", "content": "override the persona"},
                ]
            },
        )
        _assert_openai_error(resp, status=400, param="input[1]")

    def test_unknown_previous_response_id_is_404(self, client, auth_headers):
        resp = _respond(
            client,
            auth_headers,
            {"input": "hi", "previous_response_id": f"resp_{uuid.uuid4()}"},
        )
        _assert_openai_error(resp, status=404, param="previous_response_id")

    def test_non_uuid_previous_response_id_is_404_not_500(self, client, auth_headers):
        resp = _respond(
            client,
            auth_headers,
            {"input": "hi", "previous_response_id": "resp_zzz"},
        )
        _assert_openai_error(resp, status=404, param="previous_response_id")

    def test_foreign_previous_response_id_is_404(self, client, auth_headers):
        other = client.post(
            "/v1/auth/signup",
            json={
                "email": f"other_{uuid.uuid4().hex[:8]}@example.com",
                "password": "TestPassword123!",
            },
        )
        other_headers = {
            "Authorization": f"Bearer {other.json()['data']['access_token']}"
        }
        theirs = _respond(client, other_headers, {"input": "their turn"})
        assert theirs.status_code == 200, theirs.text

        resp = _respond(
            client,
            auth_headers,
            {"input": "hi", "previous_response_id": theirs.json()["id"]},
        )
        # Same status as never-existed: existence is not confirmed across users.
        assert resp.status_code == 404, resp.text
        assert set(resp.json()) == {"error"}

    def test_auth_is_still_required(self, client):
        resp = client.post("/v1/responses", json={"input": "hi"})
        # The documented seam: auth 401s keep the app-wide shape (see SPEC).
        assert resp.status_code == 401


def _sse_events(text):
    """Parse an SSE body into [{'event': name, 'data': parsed-json}, ...]."""
    import json as json_module

    events = []
    for block in text.strip().split("\n\n"):
        name, data = None, None
        for line in block.split("\n"):
            if line.startswith("event: "):
                name = line[len("event: ") :]
            elif line.startswith("data: "):
                data = json_module.loads(line[len("data: ") :])
        if name:
            events.append({"event": name, "data": data})
    return events


class TestResponsesStreaming:
    def test_stream_true_returns_sse_with_stable_id(self, client, auth_headers):
        resp = client.post(
            "/v1/responses",
            headers=auth_headers,
            json={"input": "hello over the stream", "stream": True},
        )
        assert resp.status_code == 200, resp.text
        assert resp.headers["content-type"].startswith("text/event-stream")

        events = _sse_events(resp.text)
        names = [e["event"] for e in events]
        assert names[0] == "response.created"
        assert names[-1] == "response.completed"
        assert "response.output_item.added" in names
        assert "response.output_text.done" in names

        created = events[0]["data"]["response"]
        completed = events[-1]["data"]["response"]
        # The id announced first is the id the reply is persisted under.
        assert created["id"] == completed["id"]
        assert created["status"] == "in_progress"
        assert completed["status"] == "completed"

        text = completed["output"][0]["content"][0]["text"]
        assert isinstance(text, str) and text
        done_text = next(
            e["data"]["text"]
            for e in events
            if e["event"] == "response.output_text.done"
        )
        assert done_text == text
        deltas = "".join(
            e["data"]["delta"]
            for e in events
            if e["event"] == "response.output_text.delta"
        )
        if deltas:
            assert deltas == text
        assert isinstance(completed["usage"], dict)

        seqs = [e["data"]["sequence_number"] for e in events]
        assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)

    def test_streamed_turn_is_a_real_conversation_turn(self, client, auth_headers):
        first = client.post(
            "/v1/responses",
            headers=auth_headers,
            json={"input": "Remember the word crimson.", "stream": True},
        )
        assert first.status_code == 200, first.text
        completed = _sse_events(first.text)[-1]["data"]["response"]
        assert completed["status"] == "completed"

        # The streamed reply chains exactly like a blocking one.
        second = client.post(
            "/v1/responses",
            headers=auth_headers,
            json={
                "input": "What word did I mention?",
                "previous_response_id": completed["id"],
            },
        )
        assert second.status_code == 200, second.text
        assert second.json()["previous_response_id"] == completed["id"]

        convs = client.get("/v1/conversations", headers=auth_headers).json()["data"][
            "items"
        ]
        assert len(convs) == 1
        messages = client.get(
            f"/v1/conversations/{convs[0]['id']}/messages", headers=auth_headers
        ).json()["data"]["messages"]
        assert sum(1 for m in messages if m["role"] == "assistant") == 2
        # The persisted streamed reply carries the announced id.
        streamed_id = completed["id"][len("resp_") :]
        assert any(m["id"] == streamed_id for m in messages)

    def test_stream_crash_emits_response_failed(
        self, client, auth_headers, monkeypatch
    ):
        from liminallm.service.runtime import get_runtime

        async def boom(*args, **kwargs):
            raise RuntimeError("secret internals: db=10.0.0.7")
            yield  # pragma: no cover - makes this an async generator

        monkeypatch.setattr(get_runtime().workflow, "run_streaming", boom)
        resp = client.post(
            "/v1/responses",
            headers=auth_headers,
            json={"input": "hi", "stream": True},
        )
        assert resp.status_code == 200  # status was spent when the stream began
        events = _sse_events(resp.text)
        assert events[-1]["event"] == "response.failed"
        failed = events[-1]["data"]["response"]
        assert failed["status"] == "failed"
        assert failed["error"]["code"] == "server_error"
        assert "10.0.0.7" not in resp.text

    def test_stream_error_event_carries_kernel_message(
        self, client, auth_headers, monkeypatch
    ):
        from liminallm.service.runtime import get_runtime

        async def erroring(*args, **kwargs):
            yield {"event": "error", "data": {"message": "model backend unavailable"}}

        monkeypatch.setattr(get_runtime().workflow, "run_streaming", erroring)
        resp = client.post(
            "/v1/responses",
            headers=auth_headers,
            json={"input": "hi", "stream": True},
        )
        events = _sse_events(resp.text)
        assert events[-1]["event"] == "response.failed"
        assert (
            events[-1]["data"]["response"]["error"]["message"]
            == "model backend unavailable"
        )


class TestResponsesUpstreamParity:
    """What an upstream parent's Responses API would serve, ours serves too:
    usage details, server-side tool items, and provenance - without faking
    what it cannot honestly provide (citation anchors, file ids)."""

    def test_usage_details_and_local_tokenizer_total(
        self, client, auth_headers, monkeypatch
    ):
        from liminallm.service.runtime import get_runtime

        async def run(*args, **kwargs):
            # The local-tokenizer shape: real parts, no total - plus the
            # detail keys the compat layer carries through from upstream.
            return {
                "content": "counted",
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "reasoning_tokens": 7,
                    "cached_tokens": 3,
                },
            }

        monkeypatch.setattr(get_runtime().workflow, "run", run)
        body = client.post(
            "/v1/responses", headers=auth_headers, json={"input": "hi"}
        ).json()
        usage = body["usage"]
        assert usage["total_tokens"] == 15
        assert usage["input_tokens_details"]["cached_tokens"] == 3
        assert usage["output_tokens_details"]["reasoning_tokens"] == 7

    def test_tool_runs_become_dialect_items_and_extension_trace(
        self, client, auth_headers, monkeypatch
    ):
        from liminallm.service.runtime import get_runtime

        async def run(*args, **kwargs):
            return {
                "content": "grounded answer",
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                "context_snippets": ["the vermilion cabinet"],
                "tool_calls": [
                    {"tool": "file_search", "arguments": {"query": "cabinet"}},
                    {"tool": "note_search", "arguments": {"query": "cabinet"}},
                ],
            }

        monkeypatch.setattr(get_runtime().workflow, "run", run)
        body = client.post(
            "/v1/responses", headers=auth_headers, json={"input": "hi"}
        ).json()

        # file_search is a dialect-native item; note_search is not dressed up
        # as one - it stays in the extension's full trace.
        assert [o["type"] for o in body["output"]] == ["file_search_call", "message"]
        assert body["output"][0]["status"] == "completed"
        assert body["output"][0]["queries"] == ["cabinet"]
        assert body["output"][1]["content"][0]["annotations"] == []

        ext = body["liminallm"]
        assert ext["context_snippets"] == ["the vermilion cabinet"]
        assert [t["tool"] for t in ext["tool_trace"]] == ["file_search", "note_search"]

    def test_streamed_tool_items_close_before_the_text_opens(
        self, client, auth_headers, monkeypatch
    ):
        from liminallm.service.runtime import get_runtime

        async def run_streaming(*args, **kwargs):
            yield {"event": "trace", "data": {"tool": "file_search", "status": "running"}}
            yield {"event": "trace", "data": {"tool": "note_search", "status": "running"}}
            yield {"event": "token", "data": "Answer."}
            yield {
                "event": "message_done",
                "data": {
                    "content": "Answer.",
                    "usage": {"prompt_tokens": 2, "completion_tokens": 1},
                    "tool_calls": [
                        {"tool": "file_search", "arguments": {"query": "x"}}
                    ],
                },
            }

        monkeypatch.setattr(get_runtime().workflow, "run_streaming", run_streaming)
        resp = client.post(
            "/v1/responses",
            headers=auth_headers,
            json={"input": "hi", "stream": True},
        )
        events = _sse_events(resp.text)

        added = [e for e in events if e["event"] == "response.output_item.added"]
        assert [a["data"]["item"]["type"] for a in added] == [
            "file_search_call",
            "message",
        ]
        fs_done = next(
            e
            for e in events
            if e["event"] == "response.output_item.done"
            and e["data"]["item"]["type"] == "file_search_call"
        )
        assert fs_done["data"]["item"]["status"] == "completed"
        # The tool item closed before the message item opened.
        assert (
            fs_done["data"]["sequence_number"]
            < added[1]["data"]["sequence_number"]
        )
        delta = next(
            e for e in events if e["event"] == "response.output_text.delta"
        )
        assert delta["data"]["output_index"] == 1

        completed = events[-1]["data"]["response"]
        assert [o["type"] for o in completed["output"]] == [
            "file_search_call",
            "message",
        ]
        assert completed["output"][1]["content"][0]["text"] == "Answer."
        assert completed["usage"]["total_tokens"] == 3
        assert completed["liminallm"]["tool_trace"][0]["tool"] == "file_search"

    def test_chat_completions_usage_keeps_the_rich_keys(self):
        """Duck-typed like the compat layer itself: getattr is the interface.
        vLLM's prefix caching and OpenAI both report the details on the chat
        transport; they must survive into the internal usage shape."""
        from types import SimpleNamespace

        from liminallm.service.model_backend import ApiAdapterBackend

        rich = SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            prompt_tokens_details=SimpleNamespace(cached_tokens=64),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=8),
        )
        assert ApiAdapterBackend._chat_usage(rich) == {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "cached_tokens": 64,
            "reasoning_tokens": 8,
        }

        # No details, no total: parts survive, total falls back to the sum.
        bare = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=0)
        assert ApiAdapterBackend._chat_usage(bare) == {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }
        assert ApiAdapterBackend._chat_usage(None)["total_tokens"] == 0

    def test_refusal_part_survives_ingestion(self):
        """Duck-typed on purpose: responses_compat reads SDK objects via
        getattr, so the duck interface IS the real interface here."""
        from types import SimpleNamespace

        from liminallm.service import responses_compat

        response = SimpleNamespace(
            output_text="",
            output=[
                SimpleNamespace(
                    type="message",
                    content=[
                        SimpleNamespace(
                            type="refusal", refusal="I can't help with that."
                        )
                    ],
                )
            ],
        )
        assert responses_compat.output_text(response) == "I can't help with that."


class TestResponsesWireShapeUnderFailure:
    """The shape rule holds even when FastAPI or the kernel would answer
    in its own vocabulary - the exact leaks a review probe reproduced."""

    def test_array_body_is_openai_shaped_not_fastapi_422(self, client, auth_headers):
        resp = client.post("/v1/responses", headers=auth_headers, json=[])
        _assert_openai_error(resp, status=400)

    def test_malformed_json_is_openai_shaped(self, client, auth_headers):
        resp = client.post(
            "/v1/responses",
            headers={**auth_headers, "Content-Type": "application/json"},
            content=b"not json{",
        )
        _assert_openai_error(resp, status=400)

    def test_string_input_shares_chats_dos_cap(self, client, auth_headers):
        resp = client.post(
            "/v1/responses", headers=auth_headers, json={"input": "x" * 100_001}
        )
        _assert_openai_error(resp, status=400, param="input")

    def test_item_inputs_cap_the_accumulated_total(self, client, auth_headers):
        items = [
            {"role": "user", "content": "y" * 60_000},
            {"role": "user", "content": "y" * 60_000},
        ]
        resp = client.post(
            "/v1/responses", headers=auth_headers, json={"input": items}
        )
        _assert_openai_error(resp, status=400, param="input")

    def test_service_error_mid_turn_leaves_openai_shaped(
        self, client, auth_headers, monkeypatch
    ):
        from liminallm.service.errors import ServiceError
        from liminallm.service.runtime import get_runtime

        async def boom(*args, **kwargs):
            raise ServiceError("provider exploded", status_code=502)

        monkeypatch.setattr(get_runtime().workflow, "run", boom)
        resp = client.post(
            "/v1/responses", headers=auth_headers, json={"input": "hi"}
        )
        assert resp.status_code == 502, resp.text
        body = resp.json()
        assert set(body) == {"error"}
        assert body["error"]["type"] == "server_error"
        assert body["error"]["message"] == "provider exploded"

    def test_crash_mid_turn_leaves_openai_shaped(
        self, client, auth_headers, monkeypatch
    ):
        from liminallm.service.runtime import get_runtime

        async def boom(*args, **kwargs):
            raise RuntimeError("secret internals: db=10.0.0.7")

        monkeypatch.setattr(get_runtime().workflow, "run", boom)
        resp = client.post(
            "/v1/responses", headers=auth_headers, json={"input": "hi"}
        )
        assert resp.status_code == 500, resp.text
        body = resp.json()
        assert set(body) == {"error"}
        # Generic message only: internals never reach the wire.
        assert "10.0.0.7" not in resp.text
        assert body["error"]["code"] == "server_error"


class TestTheWireIsTheDialectsOwnTypes:
    """The promise is a base-URL swap, so the arbiter is OpenAI's own model.

    SPEC §16 exists so an agent framework can change only its base URL, and
    says wire shapes are OpenAI's both ways. A test that transcribes what we
    believe those shapes are proves we were consistent with ourselves, so
    these validate against the installed SDK's generated types instead -
    built from OpenAI's OpenAPI schema, and the thing a caller's client
    actually is.

    `model_validate` rather than the SDK's own response parser: that parser
    constructs models permissively and fills in absent fields, so "the Python
    client happens to deserialize it" is a weaker claim than the one the SPEC
    makes.
    """

    def test_a_web_search_run_carries_the_action_it_performed(
        self, client, auth_headers, monkeypatch
    ):
        """`action` is required, and says which of three operations this was.

        `file_search_call` got its `queries` and `web_search_call` got
        nothing, so the item said a web search happened without saying what
        was searched for. `run_web_search` is a search, and the trace already
        carries the query, so nothing has to be invented.
        """
        from openai.types.responses import ResponseFunctionWebSearch

        from liminallm.service.runtime import get_runtime

        async def run(*args, **kwargs):
            return {
                "content": "grounded answer",
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                "tool_calls": [
                    {"tool": "web_search", "arguments": {"query": "needle"}},
                ],
            }

        monkeypatch.setattr(get_runtime().workflow, "run", run)
        body = _respond(client, auth_headers, {"input": "hi"}).json()

        item = next(o for o in body["output"] if o["type"] == "web_search_call")
        assert item["action"] == {"type": "search", "query": "needle"}
        ResponseFunctionWebSearch.model_validate(item)

    def test_a_web_search_with_no_query_still_names_its_action(
        self, client, auth_headers, monkeypatch
    ):
        """The field is required, so an unrecorded query cannot omit it."""
        from openai.types.responses import ResponseFunctionWebSearch

        from liminallm.service.runtime import get_runtime

        async def run(*args, **kwargs):
            return {
                "content": "answer",
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                "tool_calls": [{"tool": "web_search"}],
            }

        monkeypatch.setattr(get_runtime().workflow, "run", run)
        body = _respond(client, auth_headers, {"input": "hi"}).json()

        item = next(o for o in body["output"] if o["type"] == "web_search_call")
        assert item["action"]["type"] == "search"
        ResponseFunctionWebSearch.model_validate(item)

    def test_the_streamed_text_events_carry_the_logprobs_field(
        self, client, auth_headers, monkeypatch
    ):
        """Required on both, and the SDK's own accumulator reads it.

        There are no token logprobs on this surface, so the honest wire value
        is the empty list - the same answer already given for `annotations`
        and the zero-valued usage detail objects: the typed shape is present,
        and empty because the information does not exist.
        """
        from openai.types.responses import ResponseTextDeltaEvent, ResponseTextDoneEvent

        from liminallm.service.runtime import get_runtime

        async def run_streaming(*args, **kwargs):
            yield {"event": "token", "data": "Ans"}
            yield {"event": "token", "data": "wer."}
            yield {
                "event": "message_done",
                "data": {
                    "content": "Answer.",
                    "usage": {"prompt_tokens": 2, "completion_tokens": 1},
                },
            }

        monkeypatch.setattr(get_runtime().workflow, "run_streaming", run_streaming)
        resp = client.post(
            "/v1/responses",
            headers=auth_headers,
            json={"input": "hi", "stream": True},
        )
        events = _sse_events(resp.text)

        deltas = [e for e in events if e["event"] == "response.output_text.delta"]
        dones = [e for e in events if e["event"] == "response.output_text.done"]
        assert deltas and dones, [e["event"] for e in events]
        for event in deltas:
            assert event["data"]["logprobs"] == []
            ResponseTextDeltaEvent.model_validate(event["data"])
        for event in dones:
            assert event["data"]["logprobs"] == []
            ResponseTextDoneEvent.model_validate(event["data"])

    def test_the_completed_response_validates_as_one(self, client, auth_headers):
        """The whole object, against the type a caller's client will use."""
        from openai.types.responses import Response

        body = _respond(client, auth_headers, {"input": "hi"}).json()

        Response.model_validate(body)

    def test_a_streamed_web_search_item_is_valid_when_it_opens(
        self, client, auth_headers, monkeypatch
    ):
        """The streamed item is built by different code from the blocking one.

        `response.output_item.added` carries the item before the arguments
        are known, so it is the empty-query form; a typed reader validates it
        at that moment all the same.
        """
        from openai.types.responses import Response, ResponseFunctionWebSearch

        from liminallm.service.runtime import get_runtime

        async def run_streaming(*args, **kwargs):
            yield {"event": "trace", "data": {"tool": "web_search", "status": "running"}}
            yield {"event": "token", "data": "Answer."}
            yield {
                "event": "message_done",
                "data": {
                    "content": "Answer.",
                    "usage": {"prompt_tokens": 2, "completion_tokens": 1},
                    "tool_calls": [
                        {"tool": "web_search", "arguments": {"query": "needle"}}
                    ],
                },
            }

        monkeypatch.setattr(get_runtime().workflow, "run_streaming", run_streaming)
        resp = client.post(
            "/v1/responses",
            headers=auth_headers,
            json={"input": "hi", "stream": True},
        )
        events = _sse_events(resp.text)

        for name in ("response.output_item.added", "response.output_item.done"):
            item = next(
                e["data"]["item"]
                for e in events
                if e["event"] == name
                and e["data"]["item"]["type"] == "web_search_call"
            )
            ResponseFunctionWebSearch.model_validate(item)

        completed = events[-1]["data"]["response"]
        Response.model_validate(completed)


class TestEveryStreamedEventValidatesAsItsDialectType:
    """One stream, every event, one external arbiter.

    The earlier reds validated the shapes we had reason to doubt. That is
    backwards for a finite public protocol: several independent required-field
    omissions in one surface is reason to check the whole surface, not the
    parts we already suspected. `ResponseStreamEvent` is the dialect's own
    discriminated union over every server event, so each payload is handed to
    it whole - measured to reject an unknown `type`, a missing required field,
    and an invalid nested item, so it is an arbiter rather than a formality.
    """

    def _validate(self, events):
        import pydantic
        from openai.types.responses import ResponseStreamEvent

        adapter = pydantic.TypeAdapter(ResponseStreamEvent)
        for event in events:
            try:
                adapter.validate_python(event["data"])
            except pydantic.ValidationError as exc:
                raise AssertionError(
                    f"{event['event']} is not a valid Responses stream event: "
                    f"{exc.errors()[:3]}"
                ) from exc

    def test_a_successful_stream_speaks_the_dialect_throughout(
        self, client, auth_headers, monkeypatch
    ):
        from liminallm.service.runtime import get_runtime

        async def run_streaming(*args, **kwargs):
            yield {"event": "trace", "data": {"tool": "web_search", "status": "running"}}
            yield {"event": "token", "data": "An"}
            yield {"event": "token", "data": "swer."}
            yield {
                "event": "message_done",
                "data": {
                    "content": "Answer.",
                    "usage": {"prompt_tokens": 2, "completion_tokens": 1},
                    "tool_calls": [
                        {"tool": "web_search", "arguments": {"query": "needle"}}
                    ],
                },
            }

        monkeypatch.setattr(get_runtime().workflow, "run_streaming", run_streaming)
        resp = client.post(
            "/v1/responses",
            headers=auth_headers,
            json={"input": "hi", "stream": True},
        )
        events = _sse_events(resp.text)

        # Named, so this cannot pass by emitting two events and validating
        # both. Every kind the surface promises has to be here.
        assert set(e["event"] for e in events) == {
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.output_item.done",
            "response.content_part.added",
            "response.content_part.done",
            "response.output_text.delta",
            "response.output_text.done",
            "response.completed",
        }, sorted(set(e["event"] for e in events))
        self._validate(events)

    def test_a_failed_stream_speaks_it_too(
        self, client, auth_headers, monkeypatch
    ):
        """The failure event is part of the wire, not an escape from it."""
        from liminallm.service.runtime import get_runtime

        async def run_streaming(*args, **kwargs):
            yield {"event": "token", "data": "partial"}
            yield {"event": "error", "data": {"message": "provider exploded"}}

        monkeypatch.setattr(get_runtime().workflow, "run_streaming", run_streaming)
        resp = client.post(
            "/v1/responses",
            headers=auth_headers,
            json={"input": "hi", "stream": True},
        )
        events = _sse_events(resp.text)

        assert "response.failed" in [e["event"] for e in events]
        self._validate(events)

    @pytest.mark.parametrize(
        "tool, item_type, empty, filled",
        [
            (
                "web_search",
                "web_search_call",
                {"action": {"type": "search", "query": ""}},
                {"action": {"type": "search", "query": "needle"}},
            ),
            ("file_search", "file_search_call", {"queries": []}, {"queries": ["needle"]}),
        ],
        ids=["web_search", "file_search"],
    )
    def test_the_opened_item_and_the_finished_one_are_both_valid(
        self, client, auth_headers, monkeypatch, tool, item_type, empty, filled
    ):
        """The item is opened before its arguments are known, then enriched.

        An empty required query is honest while the information does not
        exist. What must hold is that each intermediate object is itself
        valid, and that the finished response carries the query the run
        actually used - measured, both item types reported an empty one for a
        run whose trace named it, because the item was built when the trace
        event opened it and never revisited.
        """
        from openai.types.responses import Response

        from liminallm.service.runtime import get_runtime

        async def run_streaming(*args, **kwargs):
            yield {"event": "trace", "data": {"tool": tool, "status": "running"}}
            yield {"event": "token", "data": "Answer."}
            yield {
                "event": "message_done",
                "data": {
                    "content": "Answer.",
                    "usage": {"prompt_tokens": 2, "completion_tokens": 1},
                    "tool_calls": [{"tool": tool, "arguments": {"query": "needle"}}],
                },
            }

        monkeypatch.setattr(get_runtime().workflow, "run_streaming", run_streaming)
        resp = client.post(
            "/v1/responses",
            headers=auth_headers,
            json={"input": "hi", "stream": True},
        )
        events = _sse_events(resp.text)

        opened = next(
            e["data"]["item"]
            for e in events
            if e["event"] == "response.output_item.added"
            and e["data"]["item"]["type"] == item_type
        )
        for key, value in empty.items():
            assert opened[key] == value, opened
        self._validate(events)

        completed = events[-1]["data"]["response"]
        Response.model_validate(completed)
        final = next(o for o in completed["output"] if o["type"] == item_type)
        assert final["id"] == opened["id"], (
            "the finished item is a different item from the one that opened"
        )
        for key, value in filled.items():
            assert final[key] == value, (
                "the finished response does not say what the run searched "
                f"for, although the trace recorded it: {final}"
            )
