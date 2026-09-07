"""The OpenAI Responses adapter keeps the provider's own tape.

Under `store=false` the provider keeps nothing, so what it said last time has
to be handed back next time in its own terms - reasoning items with their
encrypted content, messages, calls, and whatever item types it invents later.
The adapter's job is to preserve that tape exactly and replay it exactly. It
reads none of it.

Driven against the real SDK's `Response` type where the SDK has one, so the
serializer is exercised on the objects it will meet rather than on a
namespace written to match it.
"""

from __future__ import annotations

import json
import warnings

import pytest
from openai._models import construct_type
from openai.types.responses import Response

from liminallm.service import responses_compat as rc
from liminallm.service.continuation import (
    CHAT_STRUCTURED_V1,
    OPENAI_RESPONSES_NATIVE_V1,
    ContinuationMismatch,
    ModelTurnRejected,
    ProviderContinuation,
)
from liminallm.service.model_backend import supports_reasoning_context
from tests.test_responses_endpoint import _Unsupported, _backend, _client

SENTINEL = "gAAAAB+/x9Q==étape"

TOOLS = [{"type": "function", "function": {
    "name": "web_search", "description": "search", "parameters": {"type": "object"}}}]


def _reasoning(ident="rs_1", encrypted=SENTINEL):
    return {"type": "reasoning", "id": ident, "summary": [],
            "encrypted_content": encrypted, "status": "completed"}


def _call(ident="fc_1", call_id="call_1", name="web_search", arguments='{"q": "x"}'):
    return {"type": "function_call", "id": ident, "call_id": call_id, "name": name,
            "arguments": arguments, "status": "completed"}


def _message(ident="msg_1", text="found it"):
    return {"type": "message", "id": ident, "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}]}


def _raw(output, model, status="completed", incomplete=None):
    raw = {
        "id": "resp_1", "object": "response", "created_at": 0, "model": model,
        "status": status, "parallel_tool_calls": True, "tool_choice": "auto",
        "tools": [], "output": output,
        # The usage block as the wire sends it today: a newer SDK's
        # validating constructor requires `cache_write_tokens`, and the
        # fixture has to be a reply that SDK accepts, not only this one.
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15,
                  "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                  "output_tokens_details": {"reasoning_tokens": 4}},
    }
    if incomplete is not None:
        raw["incomplete_details"] = incomplete
    return raw


def _sdk_response(output, model="gpt-6-astra", **shape):
    """A real `Response`, validated by the SDK the backend actually uses."""
    return Response.model_validate(_raw(output, model, **shape))


def _wire_response(output, model="gpt-6-astra", **shape):
    """A `Response` as the client itself builds one from the wire: without
    validation, optional fields it did not receive filled in as None, an item
    type it has no class for held in the first class that takes it."""
    return construct_type(type_=Response, value=_raw(output, model, **shape))


def _stream_events(text="done"):
    """A responses stream as the SDK yields it: deltas, then completion."""
    from types import SimpleNamespace as NS

    return iter([
        NS(type="response.output_text.delta", delta=text),
        NS(type="response.completed", response=_sdk_response([_message(text=text)])),
    ])


def _native(create, chat_create=None, *, mode="openai", model="gpt-6-astra"):
    backend = _backend(_client(create, chat_create))
    backend.backend_mode = mode
    backend.base_model = model
    return backend


USER = [{"role": "user", "content": "find x"}]


class TestTheRequestIsStatelessByContract:
    def test_native_mode_sends_store_false_and_asks_for_encrypted_reasoning(self):
        """Explicit even where the provider's current default might agree:
        the adapter's contract should not depend on a default moving."""
        seen = {}

        def create(**kw):
            seen.update(kw)
            return _sdk_response([_reasoning(), _message()])

        _native(create).generate_with_tools(USER, TOOLS, [])

        assert seen["store"] is False
        assert "reasoning.encrypted_content" in seen["include"]
        # One authority for the conversation, and it is ours.
        assert "previous_response_id" not in seen
        assert "conversation" not in seen
        assert seen.get("background") is not True

    def test_every_responses_call_of_a_native_backend_keeps_nothing_on_the_provider(self):
        """Not only the calls that keep a tape. The plain completion and the
        stream build their kwargs the same way, and a compatible provider's
        carry no such flag at all."""
        seen = {}

        def create(**kw):
            seen.update(kw)
            return _sdk_response([_message()])

        backend = _native(create)
        backend.generate(USER, [])
        assert seen["store"] is False
        assert backend._responses_kwargs("gpt-6-astra", None)["store"] is False
        assert "store" not in _native(create, mode="xai")._responses_kwargs("grok-4.5", None)

    def test_the_openai_mode_pointed_at_another_host_is_a_compatible_endpoint(self):
        """`adapter_openai_base_url` is the documented way to point this
        client at any compatible endpoint. Under the `openai` mode that is
        still a compatible endpoint: no tape, nothing asked of it that only
        OpenAI's own endpoint answers."""
        seen = {}

        def create(**kw):
            seen.update(kw)
            return _sdk_response([_reasoning(), _message()])

        backend = _native(create)
        backend._base_url = "https://gateway.example/v1"
        out = backend.generate_with_tools(USER, TOOLS, [])
        assert "store" not in seen and "include" not in seen
        assert out["continuation"]["strategy"] == CHAT_STRUCTURED_V1
        assert backend.declared_continuation() == CHAT_STRUCTURED_V1

        backend._base_url = "https://api.openai.com/v1"
        seen.clear()
        out = backend.generate_with_tools(USER, TOOLS, [])
        assert seen["store"] is False and "include" in seen
        assert out["continuation"]["strategy"] == OPENAI_RESPONSES_NATIVE_V1

    def test_a_reasoning_context_model_is_asked_to_keep_its_context(self):
        """`reasoning.context` is what makes a replayed encrypted reasoning
        item worth replaying, and it is asked for by profile: the models
        known to honour it, under the native strategy only."""
        seen = {}

        def create(**kw):
            seen.update(kw)
            return _sdk_response([_reasoning(), _message()])

        _native(create).generate_with_tools(USER, TOOLS, [])
        assert seen["reasoning"] == {"context": "auto"}

        backend = _native(create)
        backend._reasoning_effort = "high"
        backend.generate_with_tools(USER, TOOLS, [])
        assert seen["reasoning"] == {"effort": "high", "context": "auto"}

        assert supports_reasoning_context("gpt-5.6") and supports_reasoning_context("gpt-6-astra")
        assert not supports_reasoning_context("gpt-5.4")

    def test_a_conventional_model_is_not_sent_a_context_it_was_not_measured_against(self):
        seen = {}

        def create(**kw):
            seen.update(kw)
            return _sdk_response([_message()])

        _native(create, model="gpt-4o-mini").generate_with_tools(USER, TOOLS, [])
        assert "reasoning" not in seen
        # The wire alone does not qualify a model: a compatible provider
        # serving the same name is sent nothing of it either.
        _native(create, mode="xai", model="gpt-6-astra").generate_with_tools(USER, TOOLS, [])
        assert "reasoning" not in seen

    def test_a_compatible_provider_on_responses_gets_none_of_it(self):
        """Answering `/responses` is the wire, not the entitlement: a
        gateway declared as a compatible provider is not asked for encrypted
        reasoning, is not told to keep nothing, and its candidate says what
        it is - the chat-shaped strategy on the responses wire, with no
        opaque state at all."""
        seen = {}

        def create(**kw):
            seen.update(kw)
            return _sdk_response([_reasoning(), _message()])

        out = _native(create, mode="xai", model="grok-4.5").generate_with_tools(
            USER, TOOLS, [])

        assert "store" not in seen and "include" not in seen
        assert out["continuation"] == {
            "strategy": CHAT_STRUCTURED_V1, "provider": "openai",
            "transport": "responses", "model": "grok-4.5", "payload": {},
        }

    def test_a_compatible_provider_on_chat_says_so_and_keeps_nothing(self):
        from tests.test_responses_endpoint import _chat_completion

        def responses_create(**kw):
            raise _Unsupported(404)

        out = _native(responses_create, lambda **kw: _chat_completion(),
                      mode="xai", model="grok-4.5").generate_with_tools(USER, [], [])

        assert out["continuation"] == {
            "strategy": CHAT_STRUCTURED_V1, "provider": "openai",
            "transport": "chat", "model": "grok-4.5", "payload": {},
        }

    def test_an_unnamed_backend_returns_no_candidate(self):
        """A backend built without a mode - the endpoint tests' own fixture
        - declares nothing, and the parent keeps nothing for it."""
        backend = _backend(_client(lambda **kw: _sdk_response([_message()])))

        out = backend.generate_with_tools(USER, TOOLS, [])

        assert "continuation" not in out


class TestOnlyAFinishedReplyIsATurn:
    """The wire's own word decides. A call that looks whole inside a reply
    the provider reports as cut off is part of a reply that was cut off, and
    nothing of it - reasoning, text, calls, tape - is accepted."""

    def test_an_incomplete_reply_with_a_whole_looking_call_is_refused(self):
        calls = []

        def create(**kw):
            calls.append(kw)
            return _wire_response(
                [_reasoning(), _call()], status="incomplete",
                incomplete={"reason": "max_output_tokens"},
            )

        with pytest.raises(ModelTurnRejected, match="incomplete.*max_output_tokens"):
            _native(create).generate_with_tools(USER, TOOLS, [])
        assert len(calls) == 1

    @pytest.mark.parametrize("status", ["failed", "cancelled", "in_progress", "queued"])
    def test_a_reply_in_any_other_state_is_refused(self, status):
        def create(**kw):
            return _wire_response([_reasoning(), _message()], status=status)

        with pytest.raises(ModelTurnRejected, match=status):
            _native(create).generate_with_tools(USER, TOOLS, [])

    def test_the_native_contract_requires_the_status_to_say_completed(self):
        """A reply that says nothing about whether it finished is not one
        the native contract accepts; only a compatible provider is allowed
        that silence."""
        def silent(**kw):
            return _wire_response([_message()], status=None)

        with pytest.raises(ModelTurnRejected, match="status None"):
            _native(silent).generate_with_tools(USER, TOOLS, [])

    def test_a_completed_reply_with_no_output_is_refused(self):
        def create(**kw):
            return _wire_response([], status="completed")

        with pytest.raises(ModelTurnRejected, match="no output"):
            _native(create).generate_with_tools(USER, TOOLS, [])

    def test_a_compatible_provider_may_omit_the_status_but_not_report_another(self):
        """Strictness is the native contract. A compatible provider on this
        wire is held to what it says: nothing said is accepted, a state
        that is not completion is refused, and an empty output is still
        nothing to accept."""
        def silent(**kw):
            return _wire_response([_message()], status=None)

        out = _native(silent, mode="xai", model="grok-4.5").generate_with_tools(USER, TOOLS, [])
        assert out["content"] == "found it"

        def cut_off(**kw):
            return _wire_response([_message()], status="incomplete")

        with pytest.raises(ModelTurnRejected):
            _native(cut_off, mode="xai", model="grok-4.5").generate_with_tools(USER, TOOLS, [])

        def empty(**kw):
            return _wire_response([], status=None)

        with pytest.raises(ModelTurnRejected):
            _native(empty, mode="xai", model="grok-4.5").generate_with_tools(USER, TOOLS, [])

    def test_a_chat_reply_cut_off_at_its_limit_is_refused(self):
        """The chat wire says the same thing in its own words."""
        from types import SimpleNamespace as NS

        def responses_create(**kw):
            raise _Unsupported(404)

        def cut_off(**kw):
            return NS(choices=[NS(message=NS(content="partial", tool_calls=None),
                                  finish_reason="length")], usage=None)

        with pytest.raises(ModelTurnRejected, match="cut off"):
            _native(responses_create, cut_off, mode="xai", model="grok-4.5").generate_with_tools(
                USER, [], [])

        def nothing(**kw):
            return NS(choices=[], usage=None)

        with pytest.raises(ModelTurnRejected, match="no choices"):
            _native(responses_create, nothing, mode="xai", model="grok-4.5").generate_with_tools(
                USER, [], [])


class TestTheStreamConsumesTheTape:
    """The final answer streams from the accepted state: the whole tape
    first, then only the record past it, on the responses endpoint alone.
    Consumed and not advanced - nothing comes back from a stream to accept."""

    ACCEPTED = ProviderContinuation(
        strategy=OPENAI_RESPONSES_NATIVE_V1, provider="openai", transport="responses",
        model="gpt-6-astra", through_operation_seq=3,
        payload={"items": [
            {"role": "user", "content": [{"type": "input_text", "text": "find x"}]},
            _reasoning(), _call(), _message(text=""),
        ]},
    )
    TAIL = [{"role": "tool", "tool_call_id": "call_1", "name": "web_search",
             "content": "result text"}]

    def test_the_tape_goes_first_and_only_the_tail_follows(self):
        seen = {}

        def create(**kw):
            seen.update(kw)
            return _stream_events("done")

        events = list(_native(create).generate_stream(self.TAIL, [], continuation=self.ACCEPTED))

        assert [e["event"] for e in events] == ["token", "message_done"]
        assert seen["input"][:4] == self.ACCEPTED.payload["items"]
        assert seen["input"][4:] == [{"type": "function_call_output",
                                      "call_id": "call_1", "output": "result text"}]
        assert seen["stream"] is True and seen["store"] is False
        # Nothing is asked back: the stream produces no continuation.
        assert "include" not in seen

    def test_a_tape_never_streams_over_chat(self):
        chat_calls = []

        def responses_create(**kw):
            raise _Unsupported(404)

        def chat_create(**kw):
            chat_calls.append(kw)
            return iter([])

        events = list(_native(responses_create, chat_create).generate_stream(
            self.TAIL, [], continuation=self.ACCEPTED))

        assert events[-1]["event"] == "error"
        assert events[-1]["data"]["code"] == "continuation_mismatch"
        assert chat_calls == []

    def test_a_tape_is_refused_by_a_backend_already_known_chat_only(self):
        """The verdict about the endpoint was reached earlier in the process.
        A tape still has no chat form, and the stream says so rather than
        sending the tail alone."""
        chat_calls = []

        def chat_create(**kw):
            chat_calls.append(kw)
            return iter([])

        backend = _native(lambda **kw: _stream_events(), chat_create)
        backend._responses_ok = False
        events = list(backend.generate_stream(self.TAIL, [], continuation=self.ACCEPTED))

        assert events[-1]["event"] == "error"
        assert events[-1]["data"]["code"] == "continuation_mismatch"
        assert chat_calls == []

    def test_a_tape_for_another_model_is_refused_before_the_call(self):
        calls = []

        def create(**kw):
            calls.append(kw)
            return _stream_events()

        events = list(_native(create, model="gpt-6-other").generate_stream(
            self.TAIL, [], continuation=self.ACCEPTED))

        assert events == [{"event": "error", "data": {
            "code": "continuation_mismatch",
            "message": "the accepted continuation is for 'gpt-6-astra' and this "
                       "backend serves 'gpt-6-other'",
        }}]
        assert calls == []


class TestTheCandidateIsTheWholeTape:
    def test_every_output_item_survives_in_order_with_its_fields(self):
        response = _wire_response([_reasoning(), _call(), _message()])
        seen = {}

        def create(**kw):
            seen.update(kw)
            return response

        out = _native(create).generate_with_tools(USER, TOOLS, [])

        candidate = out["continuation"]
        assert candidate["strategy"] == OPENAI_RESPONSES_NATIVE_V1
        assert candidate["transport"] == "responses"
        assert candidate["model"] == "gpt-6-astra"
        items = candidate["payload"]["items"]
        # The input that was sent, then the output that came back, in order.
        assert items[: len(seen["input"])] == seen["input"]
        tail = items[len(seen["input"]):]
        assert [i["type"] for i in tail] == ["reasoning", "function_call", "message"]
        assert tail[0]["encrypted_content"] == SENTINEL
        assert tail[0]["id"] == "rs_1"
        assert tail[1]["call_id"] == "call_1" and tail[1]["id"] == "fc_1"
        assert tail[0]["summary"] == []
        # A field the provider never sent is not invented for it: the client
        # fills the reasoning item's optional `content` and the text part's
        # `logprobs` with None, and those are the SDK's, not the wire's.
        assert "content" not in tail[0]
        assert "logprobs" not in tail[2]["content"][0]

    def test_a_null_the_provider_sent_stays_and_a_default_is_not_added(self):
        """The difference is whether the field was set: a null the wire
        carried was, a null the SDK filled in was not."""
        sent = {**_reasoning(), "content": None, "phase": None}

        out = _native(lambda **kw: _wire_response([sent, _message()])).generate_with_tools(
            USER, TOOLS, [])

        item = out["continuation"]["payload"]["items"][-2]
        assert "content" in item and item["content"] is None
        assert "phase" in item and item["phase"] is None
        assert "status" not in item
        part = out["continuation"]["payload"]["items"][-1]["content"][0]
        assert "logprobs" not in part and part["text"] == "found it"

    def test_an_unknown_item_type_through_the_client_parse_path_keeps_its_wire_keys(self):
        """The client holds an item type it has no class for in the first
        class that takes it. What is replayed is the item's own keys and no
        defaults of that class - and no serializer warning printing the
        item, opaque value and all, to stderr."""
        future = {"type": "future_item", "id": "fi_1",
                  "opaque": {"x": [1, None, "é"]}, "encrypted_content": SENTINEL}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = _native(
                lambda **kw: _wire_response([_reasoning(), future, _message()])
            ).generate_with_tools(USER, TOOLS, [])

        items = out["continuation"]["payload"]["items"]
        assert items[-2] == future
        assert [i["type"] for i in items[-3:]] == ["reasoning", "future_item", "message"]
        assert [w for w in caught if issubclass(w.category, UserWarning)] == []

    def test_replay_strips_only_the_documented_output_only_fields(self):
        """`reasoning.status` and `compaction.created_by` are the two the
        wire refuses back. Nothing else is touched - `status` on a message or
        a call goes back as it came, and the encrypted content is not
        re-encoded on the way through."""
        class _Duck:
            def model_dump(self, mode="json", **_kw):
                return {"output": [
                    _reasoning(),
                    {"type": "compaction", "id": "cmp_1",
                     "encrypted_content": SENTINEL, "created_by": "server"},
                    _call(),
                    _message(),
                ]}

        items = rc.replayable_output(_Duck())

        assert "status" not in items[0]
        assert items[0]["encrypted_content"] == SENTINEL
        assert "created_by" not in items[1]
        assert items[1]["encrypted_content"] == SENTINEL
        assert items[2]["status"] == "completed"
        assert items[3]["status"] == "completed"

    def test_an_item_type_nobody_has_seen_passes_through_untouched(self):
        """Ordering and content are both semantic, so the serializer is not
        an allowlist of interesting fields: it removes what is documented as
        output-only and keeps everything else, including what it cannot
        name."""
        future = {"type": "future_item", "id": "fi_1",
                  "opaque": {"x": [1, None, "é"]}, "phase": None}

        class _Duck:
            def model_dump(self, mode="json", **_kw):
                return {"output": [_reasoning(), future, _message()]}

        items = rc.replayable_output(_Duck())

        assert items[1] == future
        assert [i["type"] for i in items] == ["reasoning", "future_item", "message"]

    def test_the_serializer_refuses_what_it_cannot_replay(self):
        """A response the SDK did not model, or an output that is not a list
        of mappings, is not a candidate. Refusing here is what keeps a broken
        turn out of the next request."""
        class _NoDump:
            output = []

        with pytest.raises(ValueError):
            rc.replayable_output(_NoDump())

        class _Odd:
            def model_dump(self, mode="json", **_kw):
                return {"output": ["not a mapping"]}

        with pytest.raises(ValueError):
            rc.replayable_output(_Odd())


class TestTheNextCallReplaysTheTape:
    def test_the_accepted_items_go_first_and_only_the_new_input_follows(self):
        """Stateless replay: the whole accepted tape, then what the parent
        accepted since. The base prompt is in the tape already and does not
        go again."""
        accepted = ProviderContinuation(
            strategy=OPENAI_RESPONSES_NATIVE_V1, provider="openai",
            transport="responses", model="gpt-6-astra", through_operation_seq=1,
            payload={"items": [
                {"role": "user", "content": [{"type": "input_text", "text": "find x"}]},
                _reasoning(), _call(), _message(text=""),
            ]},
        )
        seen = {}

        def create(**kw):
            seen.update(kw)
            return _sdk_response([_reasoning("rs_2", "second"), _message("msg_2", "done")])

        new_input = [{"role": "tool", "tool_call_id": "call_1", "name": "web_search",
                      "content": "result text"}]
        out = _native(create).generate_with_tools(
            new_input, TOOLS, [], continuation=accepted)

        assert seen["input"][:4] == accepted.payload["items"]
        assert seen["input"][4:] == [{"type": "function_call_output",
                                      "call_id": "call_1", "output": "result text"}]
        # And the new candidate is that whole input plus this turn's output.
        items = out["continuation"]["payload"]["items"]
        assert items[:5] == seen["input"]
        assert [i["type"] for i in items[5:]] == ["reasoning", "message"]
        assert items[5]["encrypted_content"] == "second"

    def test_an_accepted_tape_is_never_continued_over_chat(self):
        """A replacement process negotiates the endpoint afresh. If the
        provider now answers 404, the tail alone must not go to
        chat/completions as if it continued anything: the accepted tape has
        no chat form, so the round is refused, the provider is not asked, and
        the tape is left for the parent to decide about."""
        from tests.test_responses_endpoint import _chat_completion

        accepted = ProviderContinuation(
            strategy=OPENAI_RESPONSES_NATIVE_V1, provider="openai",
            transport="responses", model="gpt-6-astra", through_operation_seq=1,
            payload={"items": [_reasoning(), _message(text="")]},
        )
        chat_calls = []

        def responses_create(**kw):
            raise _Unsupported(404)

        def chat_create(**kw):
            chat_calls.append(kw)
            return _chat_completion()

        backend = _native(responses_create, chat_create)
        with pytest.raises(ContinuationMismatch, match="responses"):
            backend.generate_with_tools(USER, [], [], continuation=accepted)

        assert chat_calls == []

    def test_an_accepted_tape_for_another_model_is_refused_before_the_call(self):
        """A replacement process configured for a different model would put
        one model's items in front of another. Refused before the provider
        is asked, not after a paid call the parent would refuse anyway."""
        accepted = ProviderContinuation(
            strategy=OPENAI_RESPONSES_NATIVE_V1, provider="openai",
            transport="responses", model="gpt-6-astra", through_operation_seq=1,
            payload={"items": [_reasoning(), _message(text="")]},
        )
        calls = []

        def create(**kw):
            calls.append(kw)
            return _sdk_response([_message()])

        backend = _native(create, model="gpt-6-other")
        with pytest.raises(ContinuationMismatch, match="gpt-6-other"):
            backend.generate_with_tools(USER, TOOLS, [], continuation=accepted)

        assert calls == []

    def test_a_chat_structured_record_is_not_taken_native_before_the_call(self):
        """The reverse of the chat guard: a record accepted on chat is the
        chat-shaped conversation, and a backend that would now go native
        refuses before the call rather than after the parent refuses the
        reply - and every retry with it."""
        accepted = ProviderContinuation(
            strategy=CHAT_STRUCTURED_V1, provider="openai", transport="chat",
            model="gpt-6-astra", through_operation_seq=1, payload={},
        )
        calls = []

        def create(**kw):
            calls.append(kw)
            return _sdk_response([_message()])

        with pytest.raises(ContinuationMismatch, match="chat-structured"):
            _native(create).generate_with_tools(USER, TOOLS, [], continuation=accepted)

        assert calls == []

    def test_a_continuation_written_by_another_strategy_is_not_replayed(self):
        """An accepted state is only usable by the strategy that wrote it.
        Handed something else, the adapter starts from the new input alone
        and says so in the candidate it returns."""
        from liminallm.service.continuation import GEMINI_NATIVE_V1

        # Another strategy's payload, in a shape this adapter would read if
        # it read shapes. It reads the strategy.
        foreign = ProviderContinuation(
            strategy=GEMINI_NATIVE_V1, provider="gemini", transport="native",
            model="gemini-3-pro", through_operation_seq=1,
            payload={"items": [{"role": "user", "content": "from elsewhere"}]},
        )
        seen = {}

        def create(**kw):
            seen.update(kw)
            return _sdk_response([_message()])

        out = _native(create).generate_with_tools(USER, TOOLS, [], continuation=foreign)

        assert seen["input"] == rc.to_input_items(USER)
        assert out["continuation"]["strategy"] == OPENAI_RESPONSES_NATIVE_V1
        assert out["continuation"]["payload"]["items"][0] == seen["input"][0]


class TestTheChatFallbackDeclaresItself:
    def test_before_any_state_the_chat_path_returns_a_chat_structured_candidate(self):
        """Negotiation may find a chat-only endpoint. That is a legal start,
        and the candidate says which strategy this invocation is now on so
        the parent can hold it there."""
        from tests.test_responses_endpoint import _chat_completion

        def responses_create(**kw):
            raise _Unsupported(404)

        out = _native(responses_create, lambda **kw: _chat_completion()).generate_with_tools(
            USER, [], [])

        assert out["continuation"]["strategy"] == CHAT_STRUCTURED_V1
        assert out["continuation"]["transport"] == "chat"
        assert out["continuation"]["payload"] == {}
