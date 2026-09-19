"""A model that refuses has answered, and the reader must see the refusal.

`responses_compat.output_text` already says this, in a docstring recording
the defect it was written for: a reply that was entirely a refusal flattened
to `""`, "and the turn then fabricated 'No response generated.' over the
model's actual words. The refusal IS the answer."

That fix reached the two blocking readers that call `output_text`. It never
reached their siblings, which read the wire directly:

* `_stream_via_responses` matched only `response.output_text.delta`, so a
  reply delivered as `response.refusal.delta` produced no tokens and empty
  content. This is the streaming path every API provider takes first, since
  `_responses_available()` is true whenever the client has a `responses`
  attribute.
* the three chat/completions readers took `message.content` and
  `delta.content` alone, and a refusal on that wire arrives with `content`
  None and the words in `refusal`. Reached once a provider has answered 404
  or 405 on `/responses`.

Downstream is the same in every case: empty content reaches the workflow
layer, which substitutes its placeholder, and the reader is shown a
malfunction instead of what the model said.

The provider doubles are built from the installed SDK's own event and
message classes rather than from namespaces shaped like them, because the
belief under test is precisely which field the wire uses. The backend is the
real one, driven through its real entry points.
"""

from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

pytest.importorskip("openai")

from liminallm.service.model_backend import ApiAdapterBackend  # noqa: E402

REFUSAL = "I can't help with that request, and here is why."
PLAIN = "A plain answer."


def _backend(client) -> ApiAdapterBackend:
    """The real class, with the provider end faked - as the suite does."""
    made = ApiAdapterBackend.__new__(ApiAdapterBackend)
    made.base_model = "gpt-4o-mini"
    made.adapter_server_model = None
    made.adapter_mode = "api_adapters"
    made._api_key = "k"
    made._base_url = None
    made._reasoning_effort = None
    made._temperature = None
    made._api_key_env = "OPENAI_API_KEY"
    made._client_timeout = 30.0
    made._active_api_key = "k"
    made._responses_ok = None
    made.client = client
    made._stream_client = None
    made.provider = "openai"
    from liminallm.config import get_provider_capabilities

    made.capabilities = get_provider_capabilities("openai")
    made._context_window = 128000
    return made


def _client(responses_create=None, chat_create=None):
    client = NS()
    if responses_create is not None:
        client.responses = NS(create=responses_create)
    client.chat = NS(completions=NS(create=chat_create or (lambda **kw: None)))
    return client


def _refusal_delta(text: str):
    from openai.types.responses import ResponseRefusalDeltaEvent

    return ResponseRefusalDeltaEvent(
        type="response.refusal.delta",
        delta=text,
        content_index=0,
        item_id="i1",
        output_index=0,
        sequence_number=1,
    )


def _text_delta(text: str):
    from openai.types.responses import ResponseTextDeltaEvent

    return ResponseTextDeltaEvent(
        type="response.output_text.delta",
        delta=text,
        content_index=0,
        item_id="i1",
        output_index=0,
        sequence_number=1,
        logprobs=[],
    )


def _completed():
    return NS(
        type="response.completed",
        response=NS(usage=NS(input_tokens=3, output_tokens=4, total_tokens=7)),
    )


def _run_stream(backend):
    return list(backend.generate_stream([{"role": "user", "content": "hi"}], []))


def _tokens(events):
    return "".join(e["data"] for e in events if e["event"] == "token")


def _content(events):
    done = [e for e in events if e["event"] == "message_done"]
    assert done, [e["event"] for e in events]
    return done[-1]["data"].get("content", "")


class TestTheResponsesStream:
    def test_a_refusal_is_streamed_and_recorded(self):
        backend = _backend(
            _client(lambda **kw: iter([_refusal_delta(REFUSAL), _completed()]))
        )

        events = _run_stream(backend)

        assert _content(events) == REFUSAL, (
            "the refusal flattened to empty, which reaches the reader as the "
            "server's own placeholder"
        )
        assert _tokens(events) == REFUSAL, (
            "the refusal was recorded but never shown; a client that renders "
            "the token stream displays an empty bubble"
        )

    def test_an_ordinary_answer_is_unaffected(self):
        """The control. Matching every event type would satisfy the test
        above while corrupting ordinary replies."""
        backend = _backend(
            _client(lambda **kw: iter([_text_delta(PLAIN), _completed()]))
        )

        events = _run_stream(backend)

        assert _content(events) == PLAIN
        assert _tokens(events) == PLAIN

    def test_the_refusal_event_does_not_disable_the_endpoint(self):
        """An AttributeError inside that reader is what
        `responses_compat.is_unsupported` reads as "this provider has no
        /responses", which would move the process to chat completions for
        good. Reading through `delta` avoids it; this pins that it did."""
        backend = _backend(
            _client(lambda **kw: iter([_refusal_delta(REFUSAL), _completed()]))
        )

        _run_stream(backend)

        assert backend._responses_ok is not False, (
            "a refusal was mistaken for an unsupported endpoint"
        )


class TestTheChatWire:
    def _chunks(self, *deltas):
        from openai.types.chat.chat_completion_chunk import ChoiceDelta

        return iter(
            [
                NS(choices=[NS(delta=d)], usage=None)
                for d in deltas
            ]
            + [
                NS(
                    choices=[NS(delta=ChoiceDelta(content=None, refusal=None))],
                    usage=NS(prompt_tokens=2, completion_tokens=2),
                )
            ]
        )

    def _refusing_client(self, *deltas):
        def responses_create(**kw):
            raise _Unsupported()

        return _client(responses_create, lambda **kw: self._chunks(*deltas))

    def test_a_streamed_refusal_reaches_the_reader(self):
        from openai.types.chat.chat_completion_chunk import ChoiceDelta

        backend = _backend(
            self._refusing_client(ChoiceDelta(content=None, refusal=REFUSAL))
        )

        events = _run_stream(backend)

        assert backend._responses_ok is False, "the fixture did not fall back"
        assert _tokens(events) == REFUSAL
        assert _content(events) == REFUSAL

    def test_a_streamed_answer_is_unaffected(self):
        """The control on the same wire."""
        from openai.types.chat.chat_completion_chunk import ChoiceDelta

        backend = _backend(
            self._refusing_client(ChoiceDelta(content=PLAIN, refusal=None))
        )

        events = _run_stream(backend)

        assert _tokens(events) == PLAIN
        assert _content(events) == PLAIN

    def _blocking_client(self, message):
        def responses_create(**kw):
            raise _Unsupported()

        def chat_create(**kw):
            return NS(
                choices=[NS(message=message, tool_calls=None)],
                usage=NS(prompt_tokens=3, completion_tokens=2, total_tokens=5),
            )

        return _client(responses_create, chat_create)

    def test_a_blocking_refusal_is_the_content(self):
        from openai.types.chat import ChatCompletionMessage

        backend = _backend(
            self._blocking_client(
                ChatCompletionMessage(
                    role="assistant", content=None, refusal=REFUSAL
                )
            )
        )

        out = backend.generate([{"role": "user", "content": "hi"}], [])

        assert out["content"] == REFUSAL, (
            "the model's refusal was dropped and the turn will report having "
            "generated nothing"
        )

    def test_a_blocking_answer_is_unaffected(self):
        """The control on the blocking reader."""
        from openai.types.chat import ChatCompletionMessage

        backend = _backend(
            self._blocking_client(
                ChatCompletionMessage(role="assistant", content=PLAIN, refusal=None)
            )
        )

        out = backend.generate([{"role": "user", "content": "hi"}], [])

        assert out["content"] == PLAIN

    def test_a_double_carrying_no_refusal_field_still_streams(self):
        """`getattr` is load-bearing, not defensive style: the doubles
        already in this suite are namespaces with `content` and nothing
        else, and plain attribute access would raise against them."""
        backend = _backend(self._refusing_client(NS(content="chat ")))

        events = _run_stream(backend)

        assert _tokens(events) == "chat "


class _Unsupported(Exception):
    def __init__(self, status=404, message="Not Found"):
        super().__init__(message)
        self.status_code = status
