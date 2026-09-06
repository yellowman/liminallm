"""What a worker may ask for, and what a round may cause.

Two authorizations, and neither is the other. A worker is a process running an
implementation the parent chose: `file.search_v1` retrieves, `code.python_v1`
runs code, and the agent loop drives a model. What each one's body actually
asks the broker for is a short, fixed list, so a request outside that list is
not a capability the worker needs and cannot use - it is a compromised worker
asking for something else. The broker knows which implementation it is
serving, and the worker never sends that name.

Inside the agent loop, a round is a claim about what the model asked for. The
parent watched the model turn happen and knows both halves: which tools it
offered, and which calls came back. So a round may cause effects only for the
exact calls the parent saw, using tools the parent actually offered - and
"exact" is `calls_match`, which compares name, decoded arguments, order and
count while ignoring the provider's own call ids.

The two catch different attacks. A faithfully relayed round of a tool that was
never offered is the model or the provider misbehaving; a round whose calls
differ from the ones the parent recorded is the worker misbehaving. They are
refused differently for that reason.
"""

from __future__ import annotations

import uuid

import pytest

from liminallm.service.broker import CapabilityBroker, InvocationContext
from liminallm.service.invocation import Invocation
from liminallm.service.runtime import get_runtime
from liminallm.service.transcript import ModelTurn

#: The one tool these witnesses offer, in the shape a model is given it.
FILE_SEARCH = {
    "type": "function",
    "function": {
        "name": "file_search",
        "description": "search the user's files",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
    },
}

#: A second, so a round can name a tool that was never in the parent's set.
NOTE_SEARCH = {
    "type": "function",
    "function": {
        "name": "note_search",
        "description": "search the user's notes",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
    },
}


@pytest.fixture
def engine():
    return get_runtime().workflow


def _invocation(tool="agent.files_v1"):
    invocation = Invocation(uuid.uuid4().hex, tool=tool)
    invocation.begin_attempt()
    return invocation


def _context(engine, *, tools=(FILE_SEARCH,)):
    """A context whose parent-side base prompt offered exactly `tools`."""
    context = InvocationContext(user_id="u-1", tenant_id=None, user_message="find it")
    context.remember_base_prompt(
        [{"role": "system", "content": "the parent's own prompt"}], list(tools)
    )
    return context


def _broker(engine, context, *, worker_tool="agent.files_v1", on_capability=None):
    """The broker as `_serve_invocation` builds it, told which body it serves.

    The implementation, not the tool spec's name: a persisted spec may carry a
    `handler` alias, and `_resolve_worker_tool` follows it, so the external
    spelling and the body that runs can differ.
    """
    return CapabilityBroker(
        engine, context, worker_tool=worker_tool, on_capability=on_capability
    )


def _ask(broker, invocation, capability, payload, seq):
    return broker._answer(
        invocation,
        {"operation_seq": seq, "capability": capability, "payload": payload},
    )


def _model_asks(engine, monkeypatch, calls, content=""):
    """The backend answering one turn with the tool calls it wants."""
    seen: dict = {}

    def _generate_with_tools(messages, tools, adapters, *, user_id=None):
        seen["tools"] = list(tools or [])
        return {
            "content": content,
            "tool_calls": list(calls),
            "assistant_message": {"role": "assistant", "content": content},
            "usage": {},
        }

    monkeypatch.setattr(
        engine.llm, "generate_with_tools", _generate_with_tools, raising=False
    )
    return seen


def _watch_round(engine, monkeypatch):
    """Every round the parent actually executed. Empty is the claim."""
    ran: list = []
    real = engine._run_round_tools

    def _watched(parsed, **kwargs):
        ran.append([name for _call, name, _args in parsed])
        return real(parsed, **kwargs)

    monkeypatch.setattr(engine, "_run_round_tools", _watched)
    return ran


def _call(name, arguments, call_id="call_1"):
    """One tool call as a provider sends it: arguments are a JSON string."""
    import json

    return {"id": call_id, "name": name, "arguments": json.dumps(arguments)}


def _submitted(name, arguments, call_id="call_1"):
    """The same call as a worker relays it: arguments already parsed."""
    return {"id": call_id, "name": name, "arguments": dict(arguments)}


class TestARoundRunsOnlyTheCallsTheParentSaw:
    """The agent loop's authorization, from the parent's own record of the
    model turn rather than from what the worker says happened."""

    def _turn(self, engine, monkeypatch, context, invocation, calls, *, tools):
        """Run one `llm.generate_with_tools`, so the parent records the turn."""
        seen = _model_asks(engine, monkeypatch, calls)
        broker = _broker(engine, context)
        reply = _ask(
            broker, invocation, "llm.generate_with_tools",
            {"messages": [{"role": "user", "content": "find it"}], "tools": tools},
            1,
        )
        assert reply["ok"], reply
        return broker, seen

    def test_a_tool_the_parent_never_offered_does_not_run(
        self, engine, monkeypatch
    ):
        """The worker reduced itself to no tools, the model asked anyway, and
        the worker relayed it faithfully.

        `calls_match` is true - the worker did exactly as it was told - so the
        only thing standing between the request and the corpus is whether the
        parent offered that tool on that turn.
        """
        context = _context(engine)
        invocation = _invocation()
        ran = _watch_round(engine, monkeypatch)
        broker, seen = self._turn(
            engine, monkeypatch, context, invocation,
            [_call("file_search", {"query": "secrets"})], tools=[],
        )
        assert seen["tools"] == [], "the fixture offered a tool after all"

        reply = _ask(
            broker, invocation, "tools.round",
            {"calls": [_submitted("file_search", {"query": "secrets"})]}, 2,
        )

        assert ran == [], f"an unoffered tool executed: {ran}"
        assert reply["ok"], reply

    def test_a_worker_invented_schema_confers_no_authority(
        self, engine, monkeypatch
    ):
        """With citation offers off the parent forwards the worker's own tool
        schemas, which is the legacy prompt shape. That must not become a way
        for a worker to declare its own authority: the names that count are
        the ones in the parent's set, whatever bytes were sent."""
        monkeypatch.setattr(
            type(engine), "CITATION_OFFERS_ENABLED", False, raising=False
        )
        context = _context(engine, tools=(NOTE_SEARCH,))
        invocation = _invocation()
        ran = _watch_round(engine, monkeypatch)
        broker, seen = self._turn(
            engine, monkeypatch, context, invocation,
            [_call("file_search", {"query": "secrets"})], tools=[FILE_SEARCH],
        )
        # The worker's bytes really did reach the model, which is what makes
        # this the interesting case.
        assert seen["tools"] == [FILE_SEARCH]

        reply = _ask(
            broker, invocation, "tools.round",
            {"calls": [_submitted("file_search", {"query": "secrets"})]}, 2,
        )

        assert ran == [], f"a worker's own schema authorized a call: {ran}"
        assert reply["ok"], reply

    def test_one_unoffered_call_stops_the_offered_ones_too(
        self, engine, monkeypatch
    ):
        """A mixed round, which is the shape the rule is for.

        The model asked for two calls and the parent offered only one of the
        tools, so the round is half authorized. Running that half is not a
        smaller version of running the round - it is one operation with one
        ledger entry and one transcript entry, and "which half ran" is not a
        question either can answer afterwards.
        """
        context = _context(engine)
        invocation = _invocation()
        ran = _watch_round(engine, monkeypatch)
        broker, _seen = self._turn(
            engine, monkeypatch, context, invocation,
            [
                _call("file_search", {"query": "manual"}),
                _call("note_search", {"query": "secrets"}, "call_2"),
            ],
            tools=[FILE_SEARCH],
        )

        reply = _ask(
            broker, invocation, "tools.round",
            {"calls": [
                _submitted("file_search", {"query": "manual"}),
                _submitted("note_search", {"query": "secrets"}, "call_2"),
            ]}, 2,
        )

        assert ran == [], f"the offered half of a mixed round ran: {ran}"
        assert reply["ok"], reply

    def test_changed_arguments_run_nothing(self, engine, monkeypatch):
        context = _context(engine)
        invocation = _invocation()
        ran = _watch_round(engine, monkeypatch)
        broker, _seen = self._turn(
            engine, monkeypatch, context, invocation,
            [_call("file_search", {"query": "A"})], tools=[FILE_SEARCH],
        )

        reply = _ask(
            broker, invocation, "tools.round",
            {"calls": [_submitted("file_search", {"query": "B"})]}, 2,
        )

        assert ran == [], f"the worker's arguments ran: {ran}"
        assert reply["ok"] is False or reply.get("result", {}).get("error")

    def test_an_added_call_voids_the_whole_round(self, engine, monkeypatch):
        """Atomic on purpose: the honest half of a tampered round is still a
        round the parent cannot vouch for, and partial effects make a retry
        impossible to reason about."""
        context = _context(engine, tools=(FILE_SEARCH, NOTE_SEARCH))
        invocation = _invocation()
        ran = _watch_round(engine, monkeypatch)
        broker, _seen = self._turn(
            engine, monkeypatch, context, invocation,
            [_call("file_search", {"query": "A"})],
            tools=[FILE_SEARCH, NOTE_SEARCH],
        )

        reply = _ask(
            broker, invocation, "tools.round",
            {"calls": [
                _submitted("file_search", {"query": "A"}),
                _submitted("note_search", {"query": "A"}, "call_2"),
            ]}, 2,
        )

        assert ran == [], f"a tampered round executed: {ran}"
        assert reply["ok"] is False or reply.get("result", {}).get("error")

    def test_a_dropped_or_reordered_call_voids_the_round(
        self, engine, monkeypatch
    ):
        context = _context(engine, tools=(FILE_SEARCH, NOTE_SEARCH))
        invocation = _invocation()
        ran = _watch_round(engine, monkeypatch)
        broker, _seen = self._turn(
            engine, monkeypatch, context, invocation,
            [
                _call("file_search", {"query": "A"}),
                _call("note_search", {"query": "B"}, "call_2"),
            ],
            tools=[FILE_SEARCH, NOTE_SEARCH],
        )

        reply = _ask(
            broker, invocation, "tools.round",
            {"calls": [
                _submitted("note_search", {"query": "B"}, "call_2"),
                _submitted("file_search", {"query": "A"}),
            ]}, 2,
        )

        assert ran == [], f"a reordered round executed: {ran}"
        assert reply["ok"] is False or reply.get("result", {}).get("error")

    def test_a_round_no_model_turn_asked_for_is_refused(
        self, engine, monkeypatch
    ):
        """`calls_match([], [])` is true, so an empty round after a terminal
        answer would otherwise authorize itself. It has no effects today, but
        it moves the transcript and the sequence, and a round the protocol
        never asked for should not do either."""
        context = _context(engine)
        invocation = _invocation()
        ran = _watch_round(engine, monkeypatch)
        broker, _seen = self._turn(
            engine, monkeypatch, context, invocation, [], tools=[FILE_SEARCH],
        )

        reply = _ask(broker, invocation, "tools.round", {"calls": []}, 2)

        assert ran == []
        assert reply["ok"] is False or reply.get("result", {}).get("error")
        assert context.transcript.rounds() == [], "a refused round was recorded"

    def test_the_exact_round_the_model_asked_for_runs(self, engine, monkeypatch):
        """The positive case, so the three refusals above are not the whole
        behaviour: an offered tool, relayed exactly, executes."""
        context = _context(engine)
        invocation = _invocation()
        ran = _watch_round(engine, monkeypatch)
        broker, _seen = self._turn(
            engine, monkeypatch, context, invocation,
            [_call("file_search", {"query": "turbines"})], tools=[FILE_SEARCH],
        )

        reply = _ask(
            broker, invocation, "tools.round",
            {"calls": [_submitted("file_search", {"query": "turbines"})]}, 2,
        )

        assert ran == [["file_search"]], ran
        assert reply["ok"], reply

    def test_a_renamed_call_id_is_still_the_same_round(
        self, engine, monkeypatch
    ):
        """`calls_match` ignores provider ids on purpose, and the parent
        substitutes the model turn's own id when it rebuilds tool messages. An
        implementation that started comparing whole call dicts would break
        this, so it is pinned here rather than left to the old test."""
        context = _context(engine)
        invocation = _invocation()
        ran = _watch_round(engine, monkeypatch)
        broker, _seen = self._turn(
            engine, monkeypatch, context, invocation,
            [_call("file_search", {"query": "turbines"}, "provider_id")],
            tools=[FILE_SEARCH],
        )

        reply = _ask(
            broker, invocation, "tools.round",
            {"calls": [
                _submitted("file_search", {"query": "turbines"}, "worker_id")
            ]}, 2,
        )

        assert ran == [["file_search"]], ran
        assert reply["ok"], reply


class TestAWorkerAsksOnlyForWhatItsBodyNeeds:
    """The other authorization: which broker capabilities this worker
    implementation may reach at all, whatever protocol it claims to follow."""

    def test_the_agent_cannot_retrieve_directly(self, engine, monkeypatch):
        """The attack the round protocol does not cover. `agent.files_v1`'s
        body calls two capabilities; a compromised one asking for a third is
        not following the protocol at all, so it is refused before the
        handler, before the ledger and before any withdrawal check."""
        reached: list = []
        monkeypatch.setattr(
            CapabilityBroker, "_rag_retrieve",
            lambda self, *a, **k: reached.append(a) or {"text": "", "snippets": []},
        )
        context = _context(engine)
        invocation = _invocation("agent.files_v1")
        broker = _broker(engine, context, worker_tool="agent.files_v1")

        reply = _ask(broker, invocation, "rag.retrieve", {"query": "secrets"}, 1)

        assert reached == [], "an unauthorized capability reached its handler"
        assert reply["ok"] is False, reply
        assert reply.get("code") == "capability_not_allowed", reply

    def test_the_worker_whose_body_needs_it_still_retrieves(
        self, engine, monkeypatch
    ):
        reached: list = []
        monkeypatch.setattr(
            CapabilityBroker, "_rag_retrieve",
            lambda self, *a, **k: reached.append(a) or {"text": "ok", "snippets": []},
        )
        context = _context(engine)
        invocation = _invocation("file.search_v1")
        broker = _broker(engine, context, worker_tool="file.search_v1")

        reply = _ask(broker, invocation, "rag.retrieve", {"query": "turbines"}, 1)

        assert reached, "the worker's own capability was refused"
        assert reply["ok"], reply

    def test_an_aliased_spec_is_judged_by_the_body_it_resolves_to(
        self, engine, monkeypatch
    ):
        """A persisted spec's `handler` decides which body runs, so the ACL
        reads the resolved implementation and not the name the node used. The
        invocation still carries the external spelling."""
        reached: list = []
        monkeypatch.setattr(
            CapabilityBroker, "_rag_retrieve",
            lambda self, *a, **k: reached.append(a) or {"text": "ok", "snippets": []},
        )
        context = _context(engine)
        invocation = _invocation("acme.lookup_v3")
        broker = _broker(engine, context, worker_tool="file.search_v1")

        reply = _ask(broker, invocation, "rag.retrieve", {"query": "turbines"}, 1)

        assert invocation.tool == "acme.lookup_v3"
        assert reached, "an aliased spec lost its body's own capability"
        assert reply["ok"], reply

    def test_a_worker_with_no_declared_body_gets_the_host_call_only(
        self, engine, monkeypatch
    ):
        """`_worker_main` falls back to the host body for any tool that is not
        one of the six, so that default row is `tool.host` and nothing else."""
        context = _context(engine)
        invocation = _invocation("llm.generic")
        broker = _broker(engine, context, worker_tool="llm.generic")

        refused = _ask(broker, invocation, "web.fetch", {"url": "http://x/"}, 1)

        assert refused["ok"] is False, refused
        assert refused.get("code") == "capability_not_allowed", refused

    def test_a_committed_result_does_not_replay_for_a_worker_denied_it(
        self, engine, monkeypatch
    ):
        """A denied capability must not reach the ledger either. Replay is how
        a retry inherits the first attempt's work, and inheriting a capability
        this worker may not ask for is the same grant by another route."""
        context = _context(engine)
        invocation = _invocation("agent.files_v1")
        broker = _broker(engine, context, worker_tool="agent.files_v1")
        payload = {"query": "secrets"}
        from liminallm.service.invocation import payload_hash

        digest = payload_hash(payload)
        invocation.ledger.begin(1, "rag.retrieve", digest)
        invocation.ledger.commit(1, {"text": "committed earlier", "snippets": []})

        reply = _ask(broker, invocation, "rag.retrieve", payload, 1)

        assert reply["ok"] is False, reply
        assert reply.get("code") == "capability_not_allowed", reply
        assert "committed earlier" not in str(reply)

    def test_a_denied_request_does_not_destroy_the_record_it_was_denied(
        self, engine, monkeypatch
    ):
        """Reading the committed result is not the only way to reach it.

        A refusal that marks the position failed lets a worker wreck what it
        may not read: the entry it was denied stops being committed, and the
        next attempt - an authorized one - finds nothing to inherit and runs
        the operation again. For a read that is a second, later reading of a
        corpus that may have moved; for a durable step it is the repeat the
        ledger exists to prevent.

        The refusal never began this operation. It is not its outcome to
        record.
        """
        from liminallm.service.invocation import COMMITTED, payload_hash

        context = _context(engine)
        invocation = _invocation("agent.files_v1")
        payload = {"query": "secrets"}
        digest = payload_hash(payload)
        committed = {"text": "committed earlier", "snippets": []}
        invocation.ledger.begin(1, "rag.retrieve", digest)
        invocation.ledger.commit(1, committed)

        denied = _ask(
            _broker(engine, context, worker_tool="agent.files_v1"),
            invocation, "rag.retrieve", payload, 1,
        )
        assert denied.get("code") == "capability_not_allowed", denied

        entry = invocation.ledger.get(1)
        assert entry.state == COMMITTED, entry.state
        assert entry.result == committed, entry.result

        # And the worker whose body does ask for it still inherits the work.
        ran: list = []
        monkeypatch.setattr(
            CapabilityBroker, "_rag_retrieve",
            lambda self, *a, **k: ran.append(a) or {"text": "ran again"},
        )
        replayed = _ask(
            _broker(engine, _context(engine), worker_tool="file.search_v1"),
            invocation, "rag.retrieve", payload, 1,
        )

        assert replayed.get("replayed") is True, replayed
        assert replayed["result"] == committed, replayed
        assert ran == [], "the committed operation ran a second time"


class TestTheOfferedNamesSurviveARestore:
    """A replay restores the parent's record rather than running the operation
    again, so what a round is authorized against has to survive that."""

    def test_the_offered_names_are_part_of_the_recorded_turn(self):
        turn = ModelTurn(
            operation_seq=1,
            content="",
            tool_calls=({"name": "file_search", "arguments": "{}"},),
            offered_tools=("file_search",),
        )

        restored = ModelTurn.from_dict(turn.as_dict())

        assert restored.offered_tools == ("file_search",)
        assert "offered_tools" in turn.as_dict()
