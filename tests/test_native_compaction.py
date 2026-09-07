"""Provider-native compaction, behind the strategy boundary.

The provider may compact its own tape: a Responses reply can carry a
`compaction` item that stands in for everything before it. That is the
provider's continuation being compacted in the provider's terms, and it is
the adapter's to handle - the tape the next request replays runs from the
latest compaction item on, in order, with every item type kept. The readable
record is not touched by it, the parent's lifecycle is not changed by it, and
nothing in the common layer learns the word.

Three things are pinned beside the tape rule. Compaction is asked for by
profile and priced from the resolved window, never from a borrowed constant.
It is transactional: a compacting reply the parent refuses compacts nothing.
And what the record says its state costs to replay is the current state's
cost, so a compacted tape is not reserved against as though it still carried
everything it no longer does.

Driven against the real SDK `Response` on both parse paths, and through the
broker with the real adapter behind it.
"""

from __future__ import annotations

import json
import uuid

import pytest

from liminallm.service.broker import CapabilityBroker, InvocationContext
from liminallm.service.citation_offers import rebuild_agent_messages
from liminallm.service.continuation import (
    CHAT_STRUCTURED_V1,
    OPENAI_RESPONSES_NATIVE_V1,
    ProviderContinuation,
)
from liminallm.service.invocation import COMMITTED, FAILED, InvocationRegistry
from liminallm.service.provenance import (
    GroundedMessage,
    GroundedSpan,
    SourceRegistry,
    binding,
)
from liminallm.service.runtime import get_runtime
from liminallm.service.transcript import ModelTurn, TrustedTranscript
from tests.test_continuation_lifecycle import (
    MARK,
    SENTINEL,
    _declare,
    _model,
    _replacement,
    _round,
    _tool_messages,
    _turn,
    _unmarked,
    _watch_rounds,
    native,
)
from tests.test_openai_native_adapter import (
    TOOLS,
    USER,
    _call,
    _message,
    _native,
    _reasoning,
    _sdk_response,
    _wire_response,
)
from tests.test_trusted_transcript import SEARCH, _web

#: The compaction item's opaque value, distinct from the reasoning one so a
#: containment check can tell which of the two leaked.
COMPACTED = "cmp/9Q==résumé"
CMARK = "cmp/9Q=="


def _compaction(ident="cmp_1", encrypted=COMPACTED, **extra):
    return {"type": "compaction", "id": ident, "encrypted_content": encrypted,
            "created_by": "system", **extra}


def _accepted(items, *, replay_tokens=0, through=1, model="gpt-6-astra"):
    return ProviderContinuation(
        strategy=OPENAI_RESPONSES_NATIVE_V1, provider="openai", transport="responses",
        model=model, through_operation_seq=through, payload={"items": items},
        replay_tokens=replay_tokens,
    )


#: A tape as an earlier turn left it: the opening, a reasoning item, a call
#: and its result - the shape the adapter replays first on the next request.
EARLIER = [
    {"role": "system", "content": "the parent's own prompt"},
    {"role": "user", "content": "find x"},
    {"type": "reasoning", "id": "rs_0", "encrypted_content": SENTINEL},
    {"type": "function_call", "id": "fc_0", "call_id": "c0", "name": "web_search",
     "arguments": '{"q": "x"}'},
    {"type": "function_call_output", "call_id": "c0", "output": "four hundred"},
]

TAIL = [{"role": "tool", "tool_call_id": "c0", "name": "web_search",
         "content": "four hundred"}]

try:  # The pinned SDK predates the item; its client still carries one.
    from openai.types.responses import ResponseCompactionItem  # noqa: F401
    KNOWS_COMPACTION = True
except ImportError:  # pragma: no cover - which SDK is installed decides
    KNOWS_COMPACTION = False

#: Both ways a reply reaches the adapter. The validating constructor only
#: on an SDK with a class for the item; the client's own parse path always.
BOTH_PATHS = pytest.mark.parametrize("response", [
    pytest.param(_sdk_response, id="validated", marks=pytest.mark.skipif(
        not KNOWS_COMPACTION, reason="this SDK has no class for a compaction item")),
    pytest.param(_wire_response, id="client-parse-path"),
])


def _stripped(item):
    """An output item as the tape keeps it: less the documented output-only
    field of its type, and nothing else."""
    kept = dict(item)
    kept.pop({"reasoning": "status", "compaction": "created_by"}.get(item["type"], ""), None)
    return kept


class TestTheTapeRunsFromTheLatestCompactionItem:
    @BOTH_PATHS
    def test_a_compacting_reply_replaces_everything_before_the_compaction_item(
        self, response
    ):
        """Accepted tape, then the new input, then the reply as it came: the
        candidate is that sequence cut at the compaction item. Nothing the
        item stands in for is replayed again - not the accepted items, not
        the input this call sent."""
        output = [_compaction(), _reasoning("rs_1", "fresh"), _call("fc_1", "c1")]
        seen = {}

        def create(**kw):
            seen.update(kw)
            return response(output)

        out = _native(create).generate_with_tools(
            TAIL, TOOLS, [], continuation=_accepted(EARLIER)
        )

        # The request still replayed the whole accepted tape first.
        assert seen["input"][: len(EARLIER)] == EARLIER
        items = out["continuation"]["payload"]["items"]
        assert items == [_stripped(i) for i in output]
        assert items[0]["encrypted_content"] == COMPACTED
        assert "created_by" not in items[0]
        assert items[1]["encrypted_content"] == "fresh"
        # The earlier reasoning, and the opening, are what the item stands
        # in for: gone from the tape.
        assert SENTINEL not in json.dumps(items, ensure_ascii=False)
        assert "the parent's own prompt" not in json.dumps(items)

    @BOTH_PATHS
    def test_only_the_latest_compaction_item_and_its_suffix_survive(self, response):
        """A tape already compacted once, compacted again: the older item
        stood in for its prefix, the newer stands in for that and more, and
        one of them is what goes back. Two in one reply, the same."""
        already = [_stripped(_compaction("cmp_old", "older")), *EARLIER[2:]]
        output = [_reasoning("rs_1"), _compaction("cmp_mid", "mid"),
                  _message("msg_1", "and then"), _compaction("cmp_new"),
                  _call("fc_1", "c1")]

        out = _native(lambda **kw: response(output)).generate_with_tools(
            TAIL, TOOLS, [], continuation=_accepted(already)
        )

        items = out["continuation"]["payload"]["items"]
        assert [i["type"] for i in items] == ["compaction", "function_call"]
        assert items[0]["id"] == "cmp_new" and items[0]["encrypted_content"] == COMPACTED
        dumped = json.dumps(items, ensure_ascii=False)
        assert "older" not in dumped and "mid" not in dumped and "and then" not in dumped

    @BOTH_PATHS
    def test_compaction_followed_by_reasoning_message_and_call_keeps_their_order(
        self, response
    ):
        output = [_compaction(), _reasoning("rs_1"), _message("msg_1", "on it"),
                  _call("fc_1", "c1")]

        out = _native(lambda **kw: response(output)).generate_with_tools(
            TAIL, TOOLS, [], continuation=_accepted(EARLIER)
        )

        assert out["continuation"]["payload"]["items"] == [_stripped(i) for i in output]
        assert out["content"] == "on it"
        assert [c["id"] for c in out["tool_calls"]] == ["c1"]

    def test_unknown_items_around_the_compaction_item_survive_as_they_came(self):
        """The client's own parse path holds an item type it has no class
        for in the first class that takes it. Before the compaction item it
        goes with the rest of the prefix; after it, it goes back with its
        own keys and nothing of that class's defaults."""
        before = {"type": "future_item", "id": "fi_0", "opaque": {"x": [1, None, "é"]}}
        after = {"type": "future_item", "id": "fi_1", "opaque": {"y": None},
                 "encrypted_content": SENTINEL}
        output = [before, _compaction(extra=None), after, _reasoning("rs_1"),
                  _call("fc_1", "c1")]

        out = _native(lambda **kw: _wire_response(output)).generate_with_tools(
            TAIL, TOOLS, [], continuation=_accepted(EARLIER)
        )

        items = out["continuation"]["payload"]["items"]
        assert [i["type"] for i in items] == [
            "compaction", "future_item", "reasoning", "function_call",
        ]
        assert items[1] == after
        # A null the provider sent on the compaction item is set and stays.
        assert "extra" in items[0] and items[0]["extra"] is None
        assert "fi_0" not in json.dumps(items)

    @BOTH_PATHS
    def test_a_reply_without_a_compaction_item_grows_the_tape_as_before(self, response):
        output = [_reasoning("rs_1"), _call("fc_1", "c1")]

        out = _native(lambda **kw: response(output)).generate_with_tools(
            TAIL, TOOLS, [], continuation=_accepted(EARLIER)
        )

        items = out["continuation"]["payload"]["items"]
        assert items[: len(EARLIER)] == EARLIER
        assert items[len(EARLIER):] == [
            {"type": "function_call_output", "call_id": "c0", "output": "four hundred"},
            *[_stripped(i) for i in output],
        ]

    def test_the_rule_is_the_wire_modules_and_reads_only_the_type(self):
        """What the adapter applies, stated once where the wire's words
        live: from the latest compaction item on, whatever else is there."""
        from liminallm.service import responses_compat as rc

        tape = [{"type": "message", "id": "m0"}, {"type": "compaction", "id": "c0"},
                {"type": "unknown", "id": "u0"}, {"type": "compaction", "id": "c1"},
                {"type": "reasoning", "id": "r1"}, {"id": "typeless"}]
        assert rc.replay_items(tape) == tape[3:]
        assert rc.replay_items(tape[:1]) == tape[:1]
        assert rc.replay_items([]) == []


class TestCompactionIsAskedForByProfileAndPricedFromTheWindow:
    def test_a_profiled_model_on_the_native_tool_path_is_told_where_to_compact(self):
        """`context_management` rides in the request body - through
        `extra_body`, the one spelling both the pinned SDK and the current
        one carry to the wire - with a threshold derived from the window
        the caller resolved and handed down with the request, not from
        anyone's benchmark constant."""
        from liminallm.service.model_backend import compact_threshold

        seen = {}

        def create(**kw):
            seen.update(kw)
            return _sdk_response([_reasoning(), _call()])

        backend = _native(create)
        backend.generate_with_tools(USER, TOOLS, [], context_window=128_000)

        assert seen["extra_body"]["context_management"] == [
            {"type": "compaction", "compact_threshold": compact_threshold(128_000)}
        ]
        assert compact_threshold(128_000) == 83_136
        assert "context_management" not in seen
        assert "previous_response_id" not in seen and seen["store"] is False

    def test_the_adapter_resolves_no_window_of_its_own(self):
        """One window fact, resolved by the parent for the request. The
        adapter's own discovery is one input to that resolution and is not
        consulted here: handed nothing, it asks for no compaction; handed
        the parent's answer, it sizes the threshold from that and not from
        what it discovered."""
        from liminallm.service.model_backend import compact_threshold

        seen = {}

        def create(**kw):
            seen.update(kw)
            return _sdk_response([_reasoning(), _call()])

        backend = _native(create)
        backend._context_window = 1_050_000

        backend.generate_with_tools(USER, TOOLS, [])
        assert "extra_body" not in seen

        backend.generate_with_tools(USER, TOOLS, [], context_window=128_000)
        assert seen["extra_body"]["context_management"][0]["compact_threshold"] == (
            compact_threshold(128_000)
        )
        assert compact_threshold(128_000) != compact_threshold(1_050_000)

    def test_the_threshold_is_the_window_less_explicit_headroom(self):
        """Reply, next input, and a share for the provider counting
        differently from the tokenizer here - each named, none borrowed.
        Below the floor the window is too small for compaction to buy
        anything, and none is asked for."""
        from liminallm.service.model_backend import (
            COMPACTION_INPUT_HEADROOM,
            COMPACTION_SAFETY_DIVISOR,
            MIN_COMPACT_THRESHOLD,
            compact_threshold,
        )
        from liminallm.service.tokenizer_utils import MAX_GENERATION_TOKENS

        for window in (1_050_000, 400_000, 128_000):
            assert compact_threshold(window) == (
                window - window // COMPACTION_SAFETY_DIVISOR
                - MAX_GENERATION_TOKENS - COMPACTION_INPUT_HEADROOM
            )
        assert compact_threshold(1_050_000) == 947_511
        assert compact_threshold(400_000) == 338_136
        assert compact_threshold(8_192) is None
        assert MIN_COMPACT_THRESHOLD == 32_768
        assert compact_threshold(74_275) == 32_769
        assert compact_threshold(74_272) is None
        assert compact_threshold(0) is None

    def test_a_window_below_the_floor_asks_for_no_compaction(self):
        seen = {}

        def create(**kw):
            seen.update(kw)
            return _sdk_response([_reasoning(), _call()])

        backend = _native(create)
        backend.generate_with_tools(USER, TOOLS, [], context_window=8_192)

        assert "extra_body" not in seen
        assert seen["store"] is False and "include" in seen

    def test_an_unprofiled_model_and_a_compatible_provider_are_sent_none_of_it(self):
        """The profile qualifies the model and the declaration qualifies the
        endpoint. A conventional model under the native strategy, or the
        same model served by a compatible provider, is sent nothing it was
        not established for."""
        from liminallm.service.model_backend import supports_native_compaction

        seen = {}

        def create(**kw):
            seen.update(kw)
            return _sdk_response([_message()])

        _native(create, model="gpt-4o-mini").generate_with_tools(
            USER, TOOLS, [], context_window=128_000)
        assert "extra_body" not in seen
        _native(create, mode="xai", model="gpt-6-astra").generate_with_tools(
            USER, TOOLS, [], context_window=128_000)
        assert "extra_body" not in seen

        assert supports_native_compaction("gpt-5.6") and supports_native_compaction("gpt-6-astra")
        assert not supports_native_compaction("gpt-5.4") and not supports_native_compaction("o3")

    def test_the_plain_completion_and_the_stream_keep_no_tape_and_ask_for_none(self):
        """Only the tool path captures a candidate. A compaction item in a
        reply nothing continues from would be paid for and thrown away."""
        from tests.test_openai_native_adapter import _stream_events

        seen = []

        def create(**kw):
            seen.append(kw)
            return _stream_events() if kw.get("stream") else _sdk_response([_message()])

        backend = _native(create)
        backend.generate(USER, [])
        list(backend.generate_stream(USER, []))

        assert seen and all("extra_body" not in kw for kw in seen)

    def test_compaction_merges_into_an_extra_body_the_request_already_carries(self):
        """Adapter weights and the like travel in `extra_body` too. One
        more key, not a replacement."""
        seen = {}

        def create(**kw):
            seen.update(kw)
            return _sdk_response([_reasoning(), _call()])

        backend = _native(create)
        real = backend._process_adapters_for_provider

        def with_extra(adapters):
            processed = real(adapters)
            processed["extra_body"] = {"adapter_weights": {"a": 1.0}}
            return processed

        backend._process_adapters_for_provider = with_extra
        backend.generate_with_tools(USER, TOOLS, [], context_window=128_000)

        assert seen["extra_body"]["adapter_weights"] == {"a": 1.0}
        assert seen["extra_body"]["context_management"][0]["type"] == "compaction"

    def test_the_chat_path_carries_no_context_management(self):
        from tests.test_responses_endpoint import _chat_completion, _Unsupported

        seen = {}

        def chat_create(**kw):
            seen.update(kw)
            return _chat_completion("from chat")

        def create(**kw):
            raise _Unsupported()

        backend = _native(create, chat_create, model="gpt-4o-mini")
        backend.generate_with_tools(USER, TOOLS, [])

        assert seen and "context_management" not in (seen.get("extra_body") or {})


class TestWhatTheStateCostsToReplayIsTheCurrentStates:
    """`replay_tokens` is the adapter's estimate of what the candidate state
    costs on the next request beyond the rendered conversation: the
    reasoning the retained items carry. It grows by a turn's reasoning while
    the tape grows, and a compacting reply restarts it at that turn's own,
    because what the compaction item stands in for is no longer replayed."""

    def test_the_openai_estimate_grows_by_the_turns_reasoning_and_restarts_at_a_compaction(
        self,
    ):
        grown = _native(
            lambda **kw: _sdk_response([_reasoning(), _call()], reasoning=30)
        ).generate_with_tools(TAIL, TOOLS, [], continuation=_accepted(EARLIER, replay_tokens=150))
        assert grown["continuation"]["replay_tokens"] == 180

        compacted = _native(
            lambda **kw: _wire_response([_compaction(), _reasoning(), _call()], reasoning=30)
        ).generate_with_tools(TAIL, TOOLS, [], continuation=_accepted(EARLIER, replay_tokens=150))
        assert compacted["continuation"]["replay_tokens"] == 30

        opening = _native(
            lambda **kw: _sdk_response([_reasoning(), _call()], reasoning=30)
        ).generate_with_tools(USER, TOOLS, [])
        assert opening["continuation"]["replay_tokens"] == 30

    def test_a_compatible_provider_on_the_responses_wire_reserves_nothing(self):
        """The transcript is its whole state; a reported reasoning count is
        what the turn cost, not what its replay will."""
        out = _native(
            lambda **kw: _sdk_response([_reasoning(), _call()], reasoning=30),
            mode="xai", model="grok-4.5",
        ).generate_with_tools(USER, TOOLS, [])

        assert out["continuation"]["strategy"] == CHAT_STRUCTURED_V1
        assert int(out["continuation"].get("replay_tokens") or 0) == 0

    def test_the_gemini_estimate_is_the_accepted_one_plus_this_turns_thoughts(self):
        """A tape that only grows: the parent's reserve grows with it, by
        what the provider says this turn's thinking cost."""
        from liminallm.service.continuation import GEMINI_NATIVE_V1
        from tests.test_gemini_native_continuation import OPENING, PARTS, _scripted

        reply = {
            "candidates": [{"content": {"role": "model", "parts": PARTS},
                            "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 8, "candidatesTokenCount": 4,
                              "thoughtsTokenCount": 7, "totalTokenCount": 19},
        }
        backend, _bodies = _scripted(reply, reply)

        first = backend.generate_with_tools(OPENING, TOOLS, [])
        assert first["continuation"]["replay_tokens"] == 7
        accepted = ProviderContinuation(
            strategy=GEMINI_NATIVE_V1, provider="gemini", transport="generateContent",
            model="gemini-2.5-flash", through_operation_seq=1,
            payload=first["continuation"]["payload"], replay_tokens=100,
        )
        again = backend.generate_with_tools(TAIL, TOOLS, [], continuation=accepted)
        assert again["continuation"]["replay_tokens"] == 107

    def test_the_parent_records_the_adapters_estimate_and_adds_nothing_of_its_own(
        self, store, monkeypatch
    ):
        """The reasoning count in `usage` is what the turn cost. What the
        state costs to replay is the adapter's to say, and the parent takes
        that number as it is."""
        from tests.test_continuation_lifecycle import _Provider

        engine = get_runtime().workflow
        _registry, invocation, context, broker = _turn(engine, monkeypatch)
        stated = native("looking", [SEARCH], tag="1", reasoning=100)

        def estimated(messages, continuation):
            reply = stated(messages, continuation)
            reply["continuation"]["replay_tokens"] = 7
            return reply

        _Provider(engine, monkeypatch, [estimated])
        assert _model(broker, invocation, 1)["ok"]

        assert context.continuation.replay_tokens == 7
        assert ProviderContinuation.from_dict(
            context.continuation.as_dict()
        ).replay_tokens == 7


def _openai(monkeypatch, engine, replies):
    """The real adapter behind the broker, with the provider scripted and
    every request it was sent kept. The engine's window cache is cleared,
    so the window it resolves for these calls is this backend's."""
    queue = list(replies)
    requests = []

    def create(**kw):
        requests.append(kw)
        return queue.pop(0)

    backend = _native(create)
    monkeypatch.setattr(engine.llm, "backend", backend)
    engine._budget_cache = None
    return backend, requests


def _search(ident, call_id, reasoning=4):
    return _wire_response(
        [_reasoning(ident), _call(f"fc_{call_id}", call_id, "web_search",
                                  '{"query": "hours"}')],
        reasoning=reasoning,
    )


def _budgets(monkeypatch):
    """What each offer pricing was given as its budget, in order."""
    from liminallm.service import workflow as wf

    budgets = []
    real = wf.choose_offers

    def priced(**kw):
        budgets.append(kw["budget"])
        return real(**kw)

    monkeypatch.setattr(wf, "choose_offers", priced)
    return budgets


class TestTheReserveFollowsTheStateThroughTheBroker:
    def test_the_reserve_is_pinned_across_a_compaction(self, store, monkeypatch):
        """Through the real adapter. Two turns accumulate; the compacting
        turn restarts the reserve at its own reasoning; the next grows from
        there. The budgets the offers are priced against follow."""
        engine = get_runtime().workflow
        _registry, invocation, context, broker = _turn(engine, monkeypatch)
        _backend, requests = _openai(monkeypatch, engine, [
            _search("rs_1", "c1", reasoning=100),
            _search("rs_2", "c2", reasoning=50),
            _wire_response([_compaction(), _reasoning("rs_3"),
                            _call("fc_c3", "c3", "web_search", '{"query": "hours"}')],
                           reasoning=30),
            _wire_response([_reasoning("rs_4"), _message("msg_4", "400 hours")],
                           reasoning=20),
        ])
        budgets = _budgets(monkeypatch)
        reserves = []
        for seq in (1, 3, 5, 7):
            assert _model(broker, invocation, seq)["ok"]
            reserves.append(context.continuation.replay_tokens)
            if seq < 7:
                submitted = {"id": f"c{(seq + 1) // 2}", "name": "web_search",
                             "arguments": {"query": "hours"}}
                assert _round(broker, invocation, seq + 1, calls=[submitted])["ok"]

        assert reserves == [100, 150, 30, 50]
        full = engine.prompt_budget()
        assert budgets == [full, full - 100, full - 150, full - 30]
        # The compacting turn's request carried the threshold, and the
        # request after it replayed only what the compaction item left.
        assert all(kw["extra_body"]["context_management"][0]["type"] == "compaction"
                   for kw in requests)
        assert [i["type"] for i in requests[3]["input"]][:2] == [
            "compaction", "reasoning",
        ]
        assert requests[3]["input"][-1]["type"] == "function_call_output"


def _threshold_sent(request):
    return request["extra_body"]["context_management"][0]["compact_threshold"]


class TestTheProviderCompactsInsideTheWindowTheParentResolved:
    """One window fact. The parent resolves it - the admin override first,
    which exists to correct a discovery that guessed wrong, then discovery
    - prices the prompt from it, and hands it down with the request, so
    the provider is told to compact inside the same window the parent
    believes it is operating in. Not two answers to one question."""

    def test_the_admin_override_is_the_window_the_provider_compacts_at(
        self, store, monkeypatch
    ):
        from liminallm.service.model_backend import compact_threshold
        from liminallm.service.tokenizer_utils import MAX_GENERATION_TOKENS

        engine = get_runtime().workflow
        _registry, invocation, _context, broker = _turn(engine, monkeypatch)
        backend, requests = _openai(monkeypatch, engine, [_search("rs_1", "c1")])
        backend._context_window = 1_050_000
        monkeypatch.setattr(engine.settings, "model_context_window", 128_000)

        assert _model(broker, invocation, 1)["ok"]

        window = engine.resolved_context_window()
        assert window == 128_000
        assert _threshold_sent(requests[0]) == compact_threshold(window)
        assert engine.prompt_budget() == window - MAX_GENERATION_TOKENS
        # Not the discovered window: the override corrected it.
        assert _threshold_sent(requests[0]) != compact_threshold(backend.context_window)

    def test_without_an_override_discovery_is_the_window(self, store, monkeypatch):
        from liminallm.service.model_backend import compact_threshold
        from liminallm.service.tokenizer_utils import MAX_GENERATION_TOKENS

        engine = get_runtime().workflow
        _registry, invocation, _context, broker = _turn(engine, monkeypatch)
        backend, requests = _openai(monkeypatch, engine, [_search("rs_1", "c1")])
        backend._context_window = 1_050_000
        monkeypatch.setattr(engine.settings, "model_context_window", 0)

        assert _model(broker, invocation, 1)["ok"]

        window = engine.resolved_context_window()
        assert window == backend.context_window == 1_050_000
        assert _threshold_sent(requests[0]) == compact_threshold(window)
        assert engine.prompt_budget() == window - MAX_GENERATION_TOKENS

    def test_a_changed_override_reaches_both_under_one_cache(self, store, monkeypatch):
        """The parent caches the window it resolved, briefly, so an admin
        change applies without a restart and a turn does not pay a settings
        read. The threshold lives under that same cache: while the budget
        is still priced from the old window, so is the threshold, and when
        the cache turns over both move together."""
        from liminallm.service.model_backend import compact_threshold
        from liminallm.service.tokenizer_utils import MAX_GENERATION_TOKENS

        engine = get_runtime().workflow
        _registry, invocation, _context, broker = _turn(engine, monkeypatch)
        backend, requests = _openai(monkeypatch, engine, [
            _search("rs_1", "c1"), _search("rs_2", "c2"), _search("rs_3", "c3"),
        ])
        backend._context_window = 1_050_000
        monkeypatch.setattr(engine.settings, "model_context_window", 128_000)

        assert _model(broker, invocation, 1)["ok"]
        assert _threshold_sent(requests[0]) == compact_threshold(128_000)

        monkeypatch.setattr(engine.settings, "model_context_window", 200_000)
        assert _round(broker, invocation, 2)["ok"]
        assert _model(broker, invocation, 3)["ok"]
        # Still the cached window, for the budget and the threshold alike.
        assert engine.prompt_budget() == 128_000 - MAX_GENERATION_TOKENS
        assert _threshold_sent(requests[1]) == compact_threshold(128_000)

        engine._budget_cache = None
        assert _round(broker, invocation, 4, calls=[
            {"id": "c2", "name": "web_search", "arguments": {"query": "hours"}}])["ok"]
        assert _model(broker, invocation, 5)["ok"]
        assert engine.resolved_context_window() == 200_000
        assert engine.prompt_budget() == 200_000 - MAX_GENERATION_TOKENS
        assert _threshold_sent(requests[2]) == compact_threshold(200_000)


class TestACallTheCompactionLeftBehindIsNotRun:
    """The wire orders output items as the model's and promises nothing
    about where a compaction item falls relative to a call. A call the cut
    removed from the replay state is a call the state cannot answer: the
    reply is refused whole, before anything runs, rather than run on an
    order nobody measured."""

    def test_a_call_after_the_compaction_item_is_retained_and_accepted(self):
        out = _native(
            lambda **kw: _wire_response([_compaction(), _reasoning("rs_1"), _call("fc_1", "c1")])
        ).generate_with_tools(TAIL, TOOLS, [], continuation=_accepted(EARLIER))

        assert [c["id"] for c in out["tool_calls"]] == ["c1"]
        retained = [i for i in out["continuation"]["payload"]["items"]
                    if i["type"] == "function_call"]
        assert [i["call_id"] for i in retained] == ["c1"]

    @pytest.mark.parametrize("output", [
        [_call("fc_1", "c1"), _compaction()],
        [_call("fc_1", "c1"), _compaction(), _reasoning("rs_1")],
        [_reasoning("rs_1"), _call("fc_1", "c1"), _compaction(), _call("fc_2", "c2")],
    ], ids=["call-then-compaction", "call-compaction-reasoning", "calls-either-side"])
    def test_a_call_before_the_compaction_item_refuses_the_whole_reply(self, output):
        from liminallm.service.continuation import ModelTurnRejected

        with pytest.raises(ModelTurnRejected, match="call"):
            _native(lambda **kw: _wire_response(output)).generate_with_tools(
                TAIL, TOOLS, [], continuation=_accepted(EARLIER)
            )

    @pytest.mark.parametrize("output", [
        [_call("fc_2", "c2", "web_search", '{"query": "hours"}'), _compaction()],
        [_reasoning("rs_2"), _call("fc_2", "c2", "web_search", '{"query": "hours"}'),
         _compaction(), _call("fc_3", "c3", "web_fetch", '{"url": "https://a.example"}')],
    ], ids=["one-call", "calls-either-side"])
    def test_nothing_runs_and_nothing_moves_and_the_retry_starts_from_the_old_state(
        self, store, monkeypatch, output
    ):
        engine = get_runtime().workflow
        registry, invocation, context, broker = _turn(engine, monkeypatch)
        _backend, requests = _openai(monkeypatch, engine, [
            _search("rs_1", "c1"),
            _wire_response(output),
            _wire_response([_reasoning("rs_3"), _message("msg_3", "400 hours")]),
        ])
        ran = _watch_rounds(engine, monkeypatch)
        assert _model(broker, invocation, 1)["ok"]
        assert _round(broker, invocation, 2)["ok"]
        accepted = context.continuation
        before = [entry.as_dict() for entry in context.transcript.entries]

        reply = _model(broker, invocation, 3)

        assert reply["ok"] is False and reply["code"] == "model_turn_rejected"
        assert invocation.ledger.get(3).state == FAILED
        assert ran == [["web_search"]], "a call of the refused reply ran"
        assert [entry.as_dict() for entry in context.transcript.entries] == before
        assert context.continuation == accepted
        assert CMARK not in json.dumps(accepted.as_dict())

        second, replacement = _replacement(engine, registry)
        assert _model(replacement, invocation, 1).get("replayed")
        assert _round(replacement, invocation, 2).get("replayed")
        retry = _model(replacement, invocation, 3)

        assert retry["ok"] and not retry.get("replayed")
        handed = requests[-1]["input"]
        assert handed[: len(accepted.payload["items"])] == accepted.payload["items"]
        assert CMARK not in json.dumps(handed) and "c2" not in json.dumps(handed)
        assert second.continuation.through_operation_seq == 3
        assert ran == [["web_search"]]


class TestCompactionIsTransactional:
    def test_a_compacting_reply_the_wire_reports_unfinished_compacts_nothing(
        self, store, monkeypatch
    ):
        """The provider compacted and then ran out of room. The parent
        refuses the whole reply: the accepted tape is not cut, the
        compaction item is nowhere in it, the retry starts from exactly
        the state before the attempt."""
        engine = get_runtime().workflow
        registry, invocation, context, broker = _turn(engine, monkeypatch)
        _backend, requests = _openai(monkeypatch, engine, [
            _search("rs_1", "c1"),
            _wire_response([_compaction(), _reasoning("rs_2"),
                            _call("fc_2", "c2", "web_fetch", '{"url": "https://a.example"}')],
                           status="incomplete", incomplete={"reason": "max_output_tokens"}),
            _wire_response([_reasoning("rs_3"), _message("msg_3", "400 hours")]),
        ])
        ran = _watch_rounds(engine, monkeypatch)
        assert _model(broker, invocation, 1)["ok"]
        assert _round(broker, invocation, 2)["ok"]
        accepted = context.continuation

        reply = _model(broker, invocation, 3)

        assert reply["ok"] is False and reply["code"] == "model_turn_rejected"
        assert invocation.ledger.get(3).state == FAILED
        assert context.continuation == accepted
        assert "compaction" not in json.dumps(accepted.as_dict())
        assert len(context.transcript.entries) == 2
        assert ran == [["web_search"]]

        second, replacement = _replacement(engine, registry)
        assert _model(replacement, invocation, 1).get("replayed")
        assert _round(replacement, invocation, 2).get("replayed")
        retry = _model(replacement, invocation, 3)

        assert retry["ok"] and not retry.get("replayed")
        handed = requests[-1]["input"]
        assert handed[: len(accepted.payload["items"])] == accepted.payload["items"]
        assert CMARK not in json.dumps(handed) and "rs_2" not in json.dumps(handed)
        assert second.continuation.through_operation_seq == 3

    def test_a_compacting_reply_the_parent_refuses_for_its_shape_compacts_nothing(
        self, store, monkeypatch
    ):
        """The wire said the reply finished; one of its calls is not a
        call. The candidate was built, cut at the compaction item, and is
        thrown away with the rest of the turn."""
        engine = get_runtime().workflow
        _registry, invocation, context, broker = _turn(engine, monkeypatch)
        _openai(monkeypatch, engine, [
            _search("rs_1", "c1"),
            _wire_response([_compaction(), _reasoning("rs_2"),
                            _call("fc_2", "c2", "web_fetch", '{"url": ')]),
        ])
        assert _model(broker, invocation, 1)["ok"]
        assert _round(broker, invocation, 2)["ok"]
        accepted = context.continuation

        reply = _model(broker, invocation, 3)

        assert reply["ok"] is False and reply["code"] == "model_turn_rejected"
        assert context.continuation == accepted
        assert "compaction" not in json.dumps(context.continuation.as_dict())
        assert context.continuation.replay_tokens == accepted.replay_tokens
        assert len(context.transcript.entries) == 2


class TestTheReadableRecordIsNotCompacted:
    def test_an_accepted_compaction_leaves_every_transcript_entry_as_it_was(
        self, store, monkeypatch
    ):
        """Two records of one conversation. The provider's is cut; the
        parent's grows by the turn and by nothing else, and the
        conversation rebuilt from it is the one it always was."""
        engine = get_runtime().workflow
        registry, invocation, context, broker = _turn(engine, monkeypatch)
        _openai(monkeypatch, engine, [
            _search("rs_1", "c1"),
            _wire_response([_compaction(), _reasoning("rs_2"),
                            _message("msg_2", "400 hours")]),
        ])
        assert _model(broker, invocation, 1)["ok"]
        assert _round(broker, invocation, 2)["ok"]
        before = [entry.as_dict() for entry in context.transcript.entries]

        assert _model(broker, invocation, 3)["ok"]

        entries = context.transcript.entries
        assert [entry.as_dict() for entry in entries[:2]] == before
        assert isinstance(entries[2], ModelTurn) and entries[2].content == "400 hours"
        assert "compaction" not in json.dumps([entry.as_dict() for entry in entries])
        # The conversation rebuilt from the record, under the one table the
        # turn ended with: the prefix the earlier entries render to, then
        # the turn, and nothing of the provider's cut in either.
        rebuilt_before, _m, _p = rebuild_agent_messages(
            context.initial_messages, context.initial_grounded_messages,
            TrustedTranscript(entries=list(entries[:2])), invocation.citations, registry,
        )
        rebuilt, _m, _p = rebuild_agent_messages(
            context.initial_messages, context.initial_grounded_messages,
            context.transcript, invocation.citations, registry,
        )
        assert rebuilt[: len(rebuilt_before)] == rebuilt_before
        assert rebuilt[len(rebuilt_before):] == [entries[2].assistant_message]
        # The provider's record, by contrast, starts at the compaction item.
        assert [i["type"] for i in context.continuation.payload["items"]] == [
            "compaction", "reasoning", "message",
        ]
        assert invocation.ledger.get(3).state == COMMITTED


class TestNothingOfTheCompactionCrosses:
    def test_the_reply_and_the_logs_carry_no_compaction_state(
        self, store, monkeypatch, capfd
    ):
        engine = get_runtime().workflow
        _registry, invocation, context, broker = _turn(engine, monkeypatch)
        _openai(monkeypatch, engine, [
            _search("rs_1", "c1"),
            _wire_response([_compaction(), _reasoning("rs_2"),
                            _call("fc_2", "c2", "web_search", '{"query": "hours"}')]),
        ])
        assert _model(broker, invocation, 1)["ok"]
        assert _round(broker, invocation, 2)["ok"]

        reply = _model(broker, invocation, 3)

        assert reply["ok"]
        crossed = json.dumps(reply, ensure_ascii=False)
        assert "continuation" not in reply["result"]
        assert CMARK not in crossed and MARK not in crossed
        assert "compaction" not in crossed and "encrypted_content" not in crossed
        assert COMPACTED in json.dumps(context.continuation.as_dict(), ensure_ascii=False)
        out, err = capfd.readouterr()
        assert CMARK not in out + err and MARK not in out + err
        assert "continuation_candidate" in out + err
        # Sized after the cut: the log counts what the state holds.
        assert "entries=3" in out + err or '"entries": 3' in out + err


#: The passage the parent measured into its opening, and the binding that
#: says the answer may rest on it.
PASSAGE = "the service interval is 400 hours"


def _grounded_turn(engine, monkeypatch):
    """A native turn whose opening carries one citable relation: the
    parent's prompt with a passage it measured as it wrote it."""
    _web(engine, monkeypatch)
    _declare(engine, monkeypatch, "openai")
    registry = SourceRegistry()
    source = registry.register_source(kind="file", title="manual.md",
                                      locator="/files/manual.md")
    evidence = registry.add_evidence(source.source_id, text=PASSAGE)
    content = f"the parent's own prompt\n\nContext: {PASSAGE}"
    start = content.index(PASSAGE)
    context = InvocationContext(user_id="u", source_registry=registry)
    context.provenance_bindings = [binding(source.source_id, evidence.evidence_id)]
    context.remember_base_prompt(
        [{"role": "system", "content": content}], TOOLS,
        grounded_messages=[GroundedMessage(
            message_index=0, text=content,
            spans=(GroundedSpan(start=start, end=start + len(PASSAGE),
                                source_id=source.source_id,
                                evidence_id=evidence.evidence_id),),
        )],
    )
    invocation = InvocationRegistry().open(
        uuid.uuid4().hex, tool="agent.files_v1", user_id="u", tenant_id=None
    )
    broker = CapabilityBroker(engine, context, worker_tool="agent.files_v1")
    return source.source_id, registry, invocation, context, broker


def _dictated(monkeypatch, plans):
    """`choose_offers` with the budget the test dictates per pricing:
    `"tight"` is one token short of the prompt with every fresh offer in
    it, so the last fresh candidate is withheld and nothing new fits;
    `"plenty"` is more than any prompt. What each pricing granted is kept."""
    from liminallm.service import workflow as wf

    real = wf.choose_offers
    choices = []

    def priced(**kw):
        plan = plans.pop(0)
        if plan == "tight":
            kw["budget"] = real(**{**kw, "budget": 10 ** 9}).tokens - 1
        else:
            kw["budget"] = 10 ** 9
        choice = real(**kw)
        choices.append(choice)
        return choice

    monkeypatch.setattr(wf, "choose_offers", priced)
    return choices


class TestAWithheldOpeningRelationStaysWithheld:
    def test_a_relation_the_budget_withheld_from_the_opening_is_never_granted_later(
        self, store, monkeypatch
    ):
        """The opening went to the provider without the marker, and the
        provider holds that opening: no later call sends it again. However
        the budget moves afterwards - here, the reserve falling to nothing
        at a compaction - a marker granted now would be granted for text
        the model was never shown. What the tail can still show is offered;
        what only the opening could have shown is not."""
        engine = get_runtime().workflow
        source_id, _registry, invocation, context, broker = _grounded_turn(
            engine, monkeypatch
        )
        _backend, requests = _openai(monkeypatch, engine, [
            _search("rs_1", "c1", reasoning=100),
            _wire_response([_compaction(), _reasoning("rs_2"),
                            _call("fc_2", "c2", "web_search", '{"query": "hours"}')],
                           reasoning=0),
            _wire_response([_reasoning("rs_3"), _message("msg_3", "400 hours")]),
        ])
        choices = _dictated(monkeypatch, ["tight", "plenty", "plenty"])

        assert _model(broker, invocation, 1)["ok"]
        assert choices[0].fits and choices[0].granted == ()
        assert invocation.citations.handle_for(source_id) is None
        assert "[cite:" not in requests[0]["input"][0]["content"]
        assert context.continuation.replay_tokens == 100

        assert _round(broker, invocation, 2)["ok"]
        assert _model(broker, invocation, 3)["ok"]
        assert context.continuation.replay_tokens == 0, "the compaction freed the reserve"
        assert _round(broker, invocation, 4)["ok"]
        assert _model(broker, invocation, 5)["ok"]

        # Never the opening's relation, on either call that had room for it.
        assert invocation.citations.handle_for(source_id) is None
        for choice in choices[1:]:
            assert source_id not in {b["source_id"] for b in choice.granted}
        # The round's own results were offered and their markers sent.
        assert invocation.citations, "the tail's relations were granted"
        tails = [m for kw in requests[1:] for m in kw["input"]
                 if m.get("type") == "function_call_output"]
        assert tails and all("[cite:" in m["output"] for m in tails)
        assert _unmarked([{"content": m["output"]} for m in tails[-1:]])[0]["content"] == (
            _tool_messages(context)[-1]["content"]
        )

    def test_a_relation_granted_in_the_opening_stays_granted(self, store, monkeypatch):
        """The other direction: a committed relation is the floor, and the
        cursor does not take it back. Its marker went with the opening and
        the provider holds it."""
        engine = get_runtime().workflow
        source_id, _registry, invocation, context, broker = _grounded_turn(
            engine, monkeypatch
        )
        _backend, requests = _openai(monkeypatch, engine, [
            _search("rs_1", "c1"),
            _wire_response([_reasoning("rs_2"), _message("msg_2", "400 hours")]),
        ])
        _dictated(monkeypatch, ["plenty", "plenty"])

        assert _model(broker, invocation, 1)["ok"]
        handle = invocation.citations.handle_for(source_id)
        assert handle and f"[cite:{handle}]" in requests[0]["input"][0]["content"]
        accepted = context.continuation

        assert _round(broker, invocation, 2)["ok"]
        assert _model(broker, invocation, 3)["ok"]

        assert invocation.citations.handle_for(source_id) == handle
        # The accepted tape first - the marked opening inside it - and past
        # it only the round; the opening itself is not sent again.
        sent = requests[1]["input"]
        assert sent[: len(accepted.payload["items"])] == accepted.payload["items"]
        assert [i["type"] for i in sent[len(accepted.payload["items"]):]] == [
            "function_call_output",
        ]
        rebuilt, _markers, _placed = rebuild_agent_messages(
            context.initial_messages, context.initial_grounded_messages,
            context.transcript, invocation.citations, _registry,
        )
        assert f"[cite:{handle}]" in rebuilt[0]["content"]
