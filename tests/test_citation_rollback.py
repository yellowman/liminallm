"""Citation offers are an operator's switch, and turning it off is a rollback.

The switch answers one question: may an execution grant citation authority
now. It does not answer whether a namespace already shown to the model may
stop being removed from that execution's output, and it does not answer
whether citations already recorded should cease to exist. Those are the two
ways a rollback control over this feature turns into a security regression,
so most of this file is about the things the switch must *not* do.

The asymmetry, in one line:

    authority off, immediately and for good; containment unchanged.

`Invocation.citation_offers_intact` is the execution's own copy of the
policy, taken when it opens and monotonic afterwards. That is what makes a
rollback one-way within a turn: half a turn with authority and half without
would produce an answer quoting handles from prompts the parent can no longer
describe as one conversation.
"""

from __future__ import annotations

import asyncio
import json
import threading
import uuid
from types import SimpleNamespace

import pytest

from liminallm.config import (
    MODEL_AFFECTING_SETTINGS,
    SYSTEM_SETTINGS_DEFAULTS,
    Settings,
    managed_settings_schema,
)
from liminallm.service.broker import CapabilityBroker, InvocationContext
from liminallm.service.citation_offers import CITATION_INSTRUCTION
from liminallm.service.citation_stream import ScrubbedTokenStream
from liminallm.service.citations import scrub_positions
from liminallm.service.invocation import Invocation, InvocationRegistry
from liminallm.service.provenance import (
    GroundedMessage,
    GroundedSpan,
    SourceRegistry,
    binding,
)
from liminallm.service.runtime import get_runtime
from liminallm.storage.models import KnowledgeChunk

ANSWER = "400 hours."


def _grounded_registry(text=ANSWER):
    """One source, one passage, and the binding that reaches it."""
    registry = SourceRegistry()
    source = registry.register_source(
        kind="file", title="manual.md", locator="/files/manual.md"
    )
    evidence = registry.add_evidence(source.source_id, text=text)
    return registry, binding(source.source_id, evidence.evidence_id), source, evidence


def _agent_turn(engine, *, offers=True):
    """An agent execution with a grounded opening, as the parent builds one."""
    registry, ground, source, evidence = _grounded_registry()
    content = f"answer from the source\n\nContext: {ANSWER}"
    start = content.index(ANSWER)
    context = InvocationContext(user_id="u", source_registry=registry)
    context.provenance_bindings = [ground]
    context.remember_base_prompt(
        [
            {"role": "system", "content": content},
            {"role": "user", "content": "how long"},
        ],
        [{"type": "function", "function": {"name": "file_search"}}],
        grounded_messages=[
            GroundedMessage(
                message_index=0,
                text=content,
                spans=(GroundedSpan(
                    start=start,
                    end=start + len(ANSWER),
                    source_id=source.source_id,
                    evidence_id=evidence.evidence_id,
                ),),
            )
        ],
    )
    engine.invocations.configure_citation_offers(offers)
    invocation = engine.invocations.open(
        uuid.uuid4().hex, tool="agent.files_v1", user_id="u", tenant_id=None
    )
    broker = CapabilityBroker(engine, context, worker_tool="agent.files_v1")
    return registry, invocation, context, broker


class TestTheOperatorSurface:
    """The setting as an admin console and a declarative deploy see it."""

    def test_the_shipped_default_offers_citations(self):
        assert Settings().citation_offers_enabled is True
        assert SYSTEM_SETTINGS_DEFAULTS["citation_offers_enabled"] is True

    def test_it_is_a_feature_toggle_the_console_can_render(self):
        entry = next(
            item for item in managed_settings_schema()
            if item["name"] == "citation_offers_enabled"
        )
        assert entry["group"] == "Features"
        assert entry["type"] == "bool"
        assert entry["secret"] is False
        assert entry["default"] is True

    def test_changing_it_does_not_rebuild_the_model_stack(self):
        """A citation toggle that tore down the LLM, embeddings, training and
        workflow services would interrupt every in-flight turn to change a
        boolean - and would hand the rollback a fresh engine, which is a
        second way for a live execution to escape it."""
        assert "citation_offers_enabled" not in MODEL_AFFECTING_SETTINGS
        entry = next(
            item for item in managed_settings_schema()
            if item["name"] == "citation_offers_enabled"
        )
        assert entry["reloads_model"] is False

    def test_no_environment_variable_reaches_it(self, monkeypatch):
        """Managed means the database is the only source. An environment
        variable would let one container disagree with the cluster about
        whether the feature is on, which is the opposite of a rollback."""
        for name in ("CITATION_OFFERS_ENABLED", "LIMINALLM_CITATION_OFFERS_ENABLED"):
            monkeypatch.setenv(name, "false")
        assert Settings().citation_offers_enabled is True
        field = Settings.model_fields["citation_offers_enabled"]
        extra = field.json_schema_extra or {}
        assert extra.get("admin") is True
        assert "env" not in extra


class TestTheAdminWriteReachesTheRunningWorkflow:
    """The existing settings mechanism, used rather than duplicated."""

    def test_the_write_is_live_before_the_response_returns(
        self, client, admin_headers
    ):
        runtime = get_runtime()
        assert runtime.workflow.invocations.citation_offers is True

        headers = admin_headers
        response = client.put(
            "/v1/admin/settings",
            json={"citation_offers_enabled": False},
            headers=headers,
        )
        assert response.status_code == 200

        # The worker that served the write, without waiting for its own
        # watcher: an admin told "saved" and then handed a turn that still
        # cites has been told something untrue.
        assert runtime.settings.citation_offers_enabled is False
        assert runtime.workflow.invocations.citation_offers is False

        read = client.get("/v1/admin/settings", headers=headers)
        assert read.json()["data"]["citation_offers_enabled"] is False

    def test_turning_it_back_on_reaches_new_executions_only(
        self, client, admin_headers
    ):
        runtime = get_runtime()
        headers = admin_headers
        caught = runtime.workflow.invocations.open(uuid.uuid4().hex, tool="t")

        client.put(
            "/v1/admin/settings",
            json={"citation_offers_enabled": False},
            headers=headers,
        )
        assert caught.citation_offers_intact is False

        client.put(
            "/v1/admin/settings",
            json={"citation_offers_enabled": True},
            headers=headers,
        )
        assert runtime.workflow.invocations.citation_offers is True
        # The one that lived through the rollback stays rolled back.
        assert caught.citation_offers_intact is False
        fresh = runtime.workflow.invocations.open(uuid.uuid4().hex, tool="t")
        assert fresh.citation_offers_intact is True

    def test_a_declarative_deploy_can_seed_it(self, monkeypatch, store):
        """`INSTANCE_SETTINGS_JSON` is the existing first-boot seed, and this
        setting is seedable through it because it is an ordinary managed one.
        Nothing citation-specific is added.

        Driven against a runtime whose stored settings are empty, which is
        what "first boot" means to the seeder: once an operator has chosen
        anything, a container's environment does not get to revert it.
        """
        runtime = get_runtime()
        monkeypatch.setenv(
            "INSTANCE_SETTINGS_JSON", json.dumps({"citation_offers_enabled": False})
        )
        # First boot means nothing an operator chose is stored yet. Patched
        # only across the seed: `refresh_settings` below reads the same method
        # and must see the real stored value, which is what the seed wrote.
        monkeypatch.setattr(runtime, "_system_settings_overrides", lambda: {})
        runtime._seed_settings_from_env()
        monkeypatch.undo()

        assert store.get_system_settings()["citation_offers_enabled"] is False
        runtime.refresh_settings()
        assert runtime.settings.citation_offers_enabled is False
        assert runtime.workflow.invocations.citation_offers is False


class TestTheStateMachineOnOneExecution:
    def test_an_execution_is_born_with_the_policy_of_its_registry(self):
        assert InvocationRegistry().open("a").citation_offers_intact is True
        registry = InvocationRegistry(citation_offers=False)
        assert registry.open("b").citation_offers_intact is False

    def test_disabling_is_one_way(self):
        invocation = Invocation("a")
        invocation.disable_citation_offers()
        assert invocation.citation_offers_intact is False
        # There is no re-enable. The registry's policy going back to true is
        # about the executions after this one.
        assert not hasattr(invocation, "enable_citation_offers")

    def test_it_is_a_different_fact_from_the_budget(self):
        """Two failure domains. Collapsed into one bit, an operator's rollback
        would read in the logs as a prompt that did not fit."""
        rolled_back = Invocation("a")
        rolled_back.disable_citation_offers()
        assert rolled_back.citation_budget_intact is True

        overrun = Invocation("b")
        overrun.poison_citation_budget()
        assert overrun.citation_offers_intact is True

    def test_the_table_survives_the_rollback(self):
        """Forgetting the namespace would only mean the parent could no longer
        recognize a handle the model repeats."""
        registry, ground, _source, _evidence = _grounded_registry()
        invocation = Invocation("a")
        table = invocation.extend_citations(registry, [ground])
        assert table
        nonce = invocation.citations.nonce
        handles = dict(table.by_handle)

        invocation.disable_citation_offers()

        assert invocation.citations is table
        assert invocation.citations.nonce == nonce
        assert dict(invocation.citations.by_handle) == handles

    def test_extend_citations_refuses_after_a_rollback(self):
        """The backstop. Every caller checks first today, so this refuses
        nothing - and it is here because the setting can change between a
        caller's check and its extension, which no caller can prevent."""
        registry, ground, _source, _evidence = _grounded_registry()
        invocation = Invocation("a")
        invocation.disable_citation_offers()

        table = invocation.extend_citations(registry, [ground])

        assert not table
        assert not invocation.citations


class TestOpeningAndRollingBackAreLinearized:
    """The registry owns the live set and the lock around it, so it is where
    the two orders are decided. There is no third one."""

    def test_an_execution_opened_before_a_rollback_is_caught_by_it(self):
        registry = InvocationRegistry()
        early = registry.open("early")
        registry.configure_citation_offers(False)
        assert early.citation_offers_intact is False

    def test_an_execution_opened_after_a_rollback_is_born_caught(self):
        registry = InvocationRegistry()
        registry.configure_citation_offers(False)
        assert registry.open("late").citation_offers_intact is False

    def test_re_enabling_does_not_revive_a_caught_execution(self):
        registry = InvocationRegistry()
        caught = registry.open("caught")
        registry.configure_citation_offers(False)
        registry.configure_citation_offers(True)
        assert caught.citation_offers_intact is False
        assert registry.open("fresh").citation_offers_intact is True

    def test_opening_concurrently_with_a_rollback_never_escapes_it(self):
        """The race the lock exists for, run rather than argued.

        Openers and one rollback are released together, many times over. An
        execution opened before the flip is in the live map when the sweep
        runs; one opened after is born disabled. Both are disabled, so every
        execution the openers produced is disabled - and any third outcome is
        an execution that escaped.
        """
        for round_index in range(60):
            registry = InvocationRegistry()
            opened: list = []
            failures: list = []
            start = threading.Barrier(9)

            def opener(worker):
                start.wait()
                for index in range(20):
                    try:
                        opened.append(
                            registry.open(f"{round_index}-{worker}-{index}")
                        )
                    except Exception as exc:  # noqa: BLE001 - reported below
                        failures.append(exc)

            def roll_back():
                start.wait()
                registry.configure_citation_offers(False)

            threads = [
                threading.Thread(target=opener, args=(worker,))
                for worker in range(8)
            ]
            threads.append(threading.Thread(target=roll_back))
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            assert not failures, failures
            escaped = [
                invocation for invocation in opened
                if invocation.citation_offers_intact
            ]
            assert not escaped, (
                f"{len(escaped)} of {len(opened)} executions escaped the "
                f"rollback in round {round_index}"
            )

    def test_the_disabling_happens_under_the_lock_open_takes(self):
        """The linearization, pinned deterministically rather than by luck.

        The window a snapshot-then-disable leaves open is a handful of
        bytecodes wide, so a racing witness catches it only when the
        scheduler is kind. This asks the property directly: while the sweep
        is disabling live executions, `open` must be waiting - because if it
        is not, an execution can be born between the policy flip and the
        sweep and belong to neither.
        """
        registry = InvocationRegistry()
        live = registry.open("live")
        sweeping = threading.Event()
        release = threading.Event()
        opened = threading.Event()

        real_disable = live.disable_citation_offers

        def slow_disable():
            sweeping.set()
            release.wait(timeout=5)
            real_disable()

        live.disable_citation_offers = slow_disable

        rollback = threading.Thread(
            target=registry.configure_citation_offers, args=(False,)
        )
        opener = threading.Thread(
            target=lambda: (registry.open("late"), opened.set())
        )
        rollback.start()
        assert sweeping.wait(timeout=5), "the sweep never started"
        opener.start()

        # The sweep holds the lock, so nothing can be born while it runs.
        assert not opened.wait(timeout=0.3), (
            "an execution was opened while the rollback was still sweeping"
        )

        release.set()
        opener.join(timeout=5)
        rollback.join(timeout=5)
        assert opened.is_set()
        assert live.citation_offers_intact is False
        assert registry.get("late").citation_offers_intact is False

    def test_enabling_concurrently_never_revives_a_caught_execution(self):
        """The other direction of the same race. Turning the policy back on
        is prospective, so an execution live across it keeps what it had."""
        for round_index in range(60):
            registry = InvocationRegistry(citation_offers=False)
            caught = [registry.open(f"{round_index}-pre-{i}") for i in range(4)]
            start = threading.Barrier(5)
            opened: list = []

            def opener(worker):
                start.wait()
                for index in range(20):
                    opened.append(registry.open(f"{round_index}-{worker}-{index}"))

            def enable():
                start.wait()
                registry.configure_citation_offers(True)

            threads = [
                threading.Thread(target=opener, args=(worker,))
                for worker in range(4)
            ]
            threads.append(threading.Thread(target=enable))
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            assert all(not inv.citation_offers_intact for inv in caught)


class TestTheRuntimeAppliesTheTransition:
    """`refresh_settings` is the one path, and every route goes through it."""

    def test_refreshing_settings_moves_the_live_registry(self, store):
        runtime = get_runtime()
        live = runtime.workflow.invocations.open(uuid.uuid4().hex, tool="t")

        store.set_system_settings({"citation_offers_enabled": False})
        runtime.refresh_settings()

        assert runtime.workflow.invocations.citation_offers is False
        assert live.citation_offers_intact is False

    def test_a_peer_worker_picks_it_up_through_the_settings_version(self, store):
        """Two runtimes over one store, which is the deployment: worker A
        writes, worker B observes the persisted settings version and applies
        the same transition to the execution it already has running.

        No citation-specific distribution. The version watcher is the existing
        cross-worker contract and stays the correctness fallback.
        """
        from liminallm.service.runtime import Runtime  # noqa: PLC0415

        worker_a = get_runtime()
        worker_b = Runtime()
        assert worker_b.workflow.invocations.citation_offers is True
        live_on_b = worker_b.workflow.invocations.open(uuid.uuid4().hex, tool="t")

        store.set_system_settings({"citation_offers_enabled": False})
        worker_a.refresh_settings()
        assert worker_a.workflow.invocations.citation_offers is False
        # B has not looked yet.
        assert live_on_b.citation_offers_intact is True

        worker_b.maybe_reload_model_services()

        assert worker_b.settings.citation_offers_enabled is False
        assert worker_b.workflow.invocations.citation_offers is False
        assert live_on_b.citation_offers_intact is False

    def test_the_watcher_interval_is_the_existing_managed_contract(self):
        assert Settings().settings_watch_interval_seconds == 10

    def test_a_model_rebuild_is_not_a_way_around_a_rollback(self, store):
        """The combined case: citations off and the backend changed in one
        write. The engine being retired must lose authority on its way out,
        and the replacement must be built already disabled - otherwise a
        rollback plus a model change is a rollback that did not happen."""
        runtime = get_runtime()
        old_engine = runtime.workflow
        live = old_engine.invocations.open(uuid.uuid4().hex, tool="t")

        store.set_system_settings({
            "citation_offers_enabled": False,
            "model_path": "another-model",
        })
        runtime.reload_model_services()

        assert runtime.workflow is not old_engine
        assert live.citation_offers_intact is False
        assert runtime.workflow.invocations.citation_offers is False
        assert runtime.workflow.invocations.open(
            uuid.uuid4().hex, tool="t"
        ).citation_offers_intact is False

    def test_a_citation_only_change_rebuilds_nothing(self, store):
        runtime = get_runtime()
        engine = runtime.workflow
        llm = runtime.llm

        store.set_system_settings({"citation_offers_enabled": False})
        runtime.refresh_settings()

        assert runtime.workflow is engine
        assert runtime.llm is llm


class TestAuthorityStopsAtTheRollback:
    """The blocking transport. A handle was issued, then the switch moved."""

    def test_no_further_handle_is_granted(self, store, monkeypatch):
        engine = get_runtime().workflow
        registry, invocation, context, broker = _agent_turn(engine)

        first = engine.agent_prompt(invocation, context)
        assert first is not None
        issued = dict(invocation.citations.by_handle)
        assert issued

        engine.invocations.configure_citation_offers(False)

        assert engine.agent_prompt(invocation, context) is None
        assert dict(invocation.citations.by_handle) == issued

    def test_no_instruction_or_label_is_materialized_afterwards(
        self, store, monkeypatch
    ):
        engine = get_runtime().workflow
        _registry, invocation, context, _broker = _agent_turn(engine)
        before = engine.agent_prompt(invocation, context)
        assert any(
            CITATION_INSTRUCTION in (message.get("content") or "")
            for message in before
        )

        engine.invocations.configure_citation_offers(False)

        assert engine.agent_prompt(invocation, context) is None

    def test_the_grounded_snippets_go_back_to_what_retrieval_wrote(self):
        """The automatic route, whose whole offer is the labelling."""
        engine = get_runtime().workflow
        registry, ground, _source, _evidence = _grounded_registry()
        invocation = engine.invocations.open(uuid.uuid4().hex, tool="t")
        invocation.extend_citations(registry, [ground])

        invocation.disable_citation_offers()

        snippets, instruction = engine._offered_context(
            invocation, registry, [ANSWER], [ground],
            prompt="how long", adapters=[], history=[],
        )

        assert snippets == [ANSWER]
        assert instruction is None


class TestContainmentOutlivesTheRollback:
    """The load-bearing half. The provider cannot be made to unsee a handle."""

    @staticmethod
    def _reply(engine, monkeypatch, content):
        """The model's blocking reply, as `llm.generate_with_tools` returns
        one. Scripted rather than stubbed away: the scrub sits between this
        and the worker, which is exactly what is under test."""
        def _generate(messages, tools, adapters, *, user_id=None,
                      continuation=None, context_window=None):
            return {"content": content, "tool_calls": [], "usage": {}}

        monkeypatch.setattr(
            engine.llm, "generate_with_tools", _generate, raising=False
        )

    def test_the_blocking_reply_still_loses_an_issued_namespace(
        self, store, monkeypatch
    ):
        engine = get_runtime().workflow
        registry, invocation, context, broker = _agent_turn(engine)
        assert engine.agent_prompt(invocation, context) is not None
        table = invocation.citations
        assert table
        handle = next(iter(table.by_handle))
        nonce = table.nonce
        written = f"{ANSWER} [cite:{handle}]"
        self._reply(engine, monkeypatch, written)

        engine.invocations.configure_citation_offers(False)
        assert invocation.citation_offers_intact is False

        result = broker._answer(invocation, {
            "capability": "llm.generate_with_tools",
            "operation_seq": 1,
            "payload": {"messages": [{"role": "user", "content": "how long"}],
                        "tools": []},
        })

        assert result["ok"], result
        # Containment: the worker is handed the answer with the namespace
        # taken out, exactly as it would have been before the rollback.
        assert nonce.lower() not in json.dumps(result).lower()
        assert result["result"]["content"] == scrub_positions(written, nonce)[0]

    def test_an_unused_nonce_is_still_not_scrubbed(self, store, monkeypatch):
        """The opposite case, and the reason containment is keyed on issued
        handles rather than on the namespace existing. Every execution mints
        one; an answer that happens to contain an unissued one is a
        coincidence, and editing it would be editing prose."""
        engine = get_runtime().workflow
        _registry, invocation, _context, broker = _agent_turn(engine, offers=False)
        assert not invocation.citations
        nonce = invocation.citations.nonce
        written = f"the code is {nonce} and [cite:{nonce}-1] besides"
        self._reply(engine, monkeypatch, written)

        result = broker._answer(invocation, {
            "capability": "llm.generate_with_tools",
            "operation_seq": 1,
            "payload": {"messages": [{"role": "user", "content": "how long"}],
                        "tools": []},
        })

        assert result["ok"], result
        assert result["result"]["content"] == written


class TestTheStreamedTransportBehavesTheSame:
    @staticmethod
    def _streaming(engine, monkeypatch, store, *, contents=(ANSWER,)):
        monkeypatch.setattr(
            type(engine.llm.backend), "supports_tools", property(lambda _s: False)
        )
        chunks = [
            KnowledgeChunk(
                context_id="ctx", fs_path=f"/files/m{index}.md", content=text,
                embedding=[], chunk_index=index,
            )
            for index, text in enumerate(contents)
        ]
        monkeypatch.setattr(
            engine, "rag", SimpleNamespace(retrieve=lambda *a, **k: chunks)
        )
        monkeypatch.setattr(engine, "_validate_context_scope", lambda ids, **k: ["ctx"])
        monkeypatch.setattr(engine, "_resolve_context_ids", lambda a, b: ["ctx"])
        user_id = store.create_user(
            email=f"roll_{uuid.uuid4().hex[:8]}@example.com"
        ).id
        opened: list = []
        real_open = engine.invocations.open

        def _open(*a, **k):
            invocation = real_open(*a, **k)
            opened.append(invocation)
            return invocation

        monkeypatch.setattr(engine.invocations, "open", _open)
        return user_id, opened

    @staticmethod
    async def _run(engine, user_id):
        return [
            event async for event in engine.run_streaming(
                None, None, "how long", "ctx", user_id
            )
        ]

    @pytest.mark.asyncio
    async def test_a_rollback_mid_answer_keeps_the_filter_and_drops_the_citations(
        self, store, monkeypatch
    ):
        """The causal witness, on the transport where the tokens have already
        left. The handle is in the prompt, the model writes it, the operator
        rolls back while it is being written - and the namespace still never
        reaches the client while the citations do not survive."""
        engine = get_runtime().workflow
        user_id, opened = self._streaming(engine, monkeypatch, store)
        rolled_back: list = []
        offering: list = []

        def _generate_stream(prompt, adapters=None, context_snippets=None,
                             history=None, *, user_id=None, instruction=None):
            # The execution this answer is being written for: the one holding
            # the handles the prompt was labelled with. Others open around it.
            invocation = next(inv for inv in opened if inv.citations)
            offering.append(invocation)
            handle = next(iter(invocation.citations.by_handle))
            yield {"event": "token", "data": "400 hours"}
            # The operator, mid-answer.
            engine.invocations.configure_citation_offers(False)
            rolled_back.append(True)
            marked = f"400 hours [cite:{handle}]"
            yield {"event": "token", "data": f" [cite:{handle}]"}
            yield {"event": "message_done", "data": {"content": marked}}

        monkeypatch.setattr(
            engine.llm, "generate_stream", _generate_stream, raising=False
        )

        events = await self._run(engine, user_id)

        assert rolled_back, "the stub never ran"
        invocation = offering[0]
        assert invocation.citations, "the handle was issued before the rollback"
        assert invocation.citation_offers_intact is False
        nonce = invocation.citations.nonce
        # Containment: nothing of the namespace reaches the client.
        assert nonce.lower() not in json.dumps(events).lower()
        # The answer still completes, as an ordinary uncited one.
        done = [event for event in events if event.get("event") == "message_done"]
        assert done, events
        assert done[-1]["data"]["content"] == "400 hours"
        # Present and empty is the outcome, not absent: the answer completed
        # as an ordinary one, and the field a citing turn would have filled is
        # filled with nothing.
        cited = [
            (event.get("data") or {}).get("validated_citations")
            for event in events
            if isinstance(event.get("data"), dict)
        ]
        assert all(not entry for entry in cited), cited

    @pytest.mark.asyncio
    async def test_a_rollback_before_the_answer_still_installs_the_filter(
        self, store, monkeypatch
    ):
        """The case the whole asymmetry exists for, and the one a mechanical
        replacement of the old class attribute gets wrong.

        The handle is issued while labelling the prompt. The operator rolls
        back *before the answer is generated*, so by the time the stream is
        built the execution has no authority left - and the model has still
        been shown the namespace, so the filter must go on anyway. A
        containment decision that read the policy would send the marker to
        the client.
        """
        engine = get_runtime().workflow
        user_id, opened = self._streaming(engine, monkeypatch, store)
        issued: list = []

        real_offered = engine._offered_context

        def offered(*args, **kwargs):
            result = real_offered(*args, **kwargs)
            invocation = args[0]
            if invocation is not None and invocation.citations:
                issued.append(invocation)
                # The operator, after the prompt was labelled and before a
                # single token exists.
                engine.invocations.configure_citation_offers(False)
            return result

        monkeypatch.setattr(engine, "_offered_context", offered)

        def _generate_stream(prompt, adapters=None, context_snippets=None,
                             history=None, *, user_id=None, instruction=None):
            handle = next(iter(issued[0].citations.by_handle))
            written = f"400 hours [cite:{handle}]"
            yield {"event": "token", "data": written}
            yield {"event": "message_done", "data": {"content": written}}

        monkeypatch.setattr(
            engine.llm, "generate_stream", _generate_stream, raising=False
        )

        events = await self._run(engine, user_id)

        assert issued, "no handle was issued before the rollback"
        invocation = issued[0]
        assert invocation.citation_offers_intact is False
        assert invocation.citations
        nonce = invocation.citations.nonce
        assert nonce.lower() not in json.dumps(events).lower()
        assert "[cite:" not in json.dumps(events)
        tokens = "".join(
            event["data"] for event in events if event.get("event") == "token"
        )
        assert tokens == "400 hours"

    @pytest.mark.asyncio
    async def test_with_the_policy_off_the_answer_is_byte_identical(
        self, store, monkeypatch
    ):
        """Gate off, and an answer that happens to contain the execution's own
        unused nonce. No handle was ever issued, so there is no namespace to
        contain and nothing may touch the text."""
        engine = get_runtime().workflow
        engine.invocations.configure_citation_offers(False)
        user_id, opened = self._streaming(engine, monkeypatch, store)
        written: list = []

        def _generate_stream(prompt, adapters=None, context_snippets=None,
                             history=None, *, user_id=None, instruction=None):
            nonce = opened[-1].citations.nonce
            text = f"400 hours, ref {nonce} and [cite:{nonce}-1]"
            written.append(text)
            yield {"event": "token", "data": text}
            yield {"event": "message_done", "data": {"content": text}}

        monkeypatch.setattr(
            engine.llm, "generate_stream", _generate_stream, raising=False
        )

        events = await self._run(engine, user_id)

        assert written
        assert not opened[-1].citations
        tokens = "".join(
            event["data"] for event in events if event.get("event") == "token"
        )
        assert tokens == written[0]


class TestTheFinalTransferReturnsNothingOnBothTransports:
    """The last authority gate on each path, called directly.

    Both read the answer the model actually wrote and resolve handles out of
    it. After a rollback both must return nothing - the answer stands, it
    simply carries no citations.
    """

    def test_a_delivered_accepted_answer_transfers_nothing(self, store):
        engine = get_runtime().workflow
        _registry, invocation, context, _broker = _agent_turn(engine)
        assert engine.agent_prompt(invocation, context) is not None
        handle = next(iter(invocation.citations.by_handle))
        written = f"{ANSWER} [cite:{handle}]"
        # The canonical copy of the turn, in the shape the broker records it.
        context.canonical_model_response = {"content": written, "tool_calls": []}
        public = scrub_positions(written, invocation.citations.nonce)[0]

        assert engine._recorded_citations(context, invocation, public)

        engine.invocations.configure_citation_offers(False)

        assert engine._recorded_citations(context, invocation, public) == []

    def test_a_streamed_answer_transfers_nothing(self, store):
        """The stream is the real reader over the real text, not a double:
        what a citation is read out of is the canonical copy this producer
        accumulated, and a hand-made stand-in would not have one."""
        engine = get_runtime().workflow
        _registry, invocation, context, _broker = _agent_turn(engine)
        assert engine.agent_prompt(invocation, context) is not None
        handle = next(iter(invocation.citations.by_handle))
        written = f"{ANSWER} [cite:{handle}]"
        events = [
            {"event": "token", "data": written},
            {"event": "message_done", "data": {"content": written}},
        ]
        stream = ScrubbedTokenStream(events, invocation.citations.nonce)
        list(stream)
        assert stream.reader.intact()

        assert engine._streamed_citations(stream, invocation)

        engine.invocations.configure_citation_offers(False)

        assert engine._streamed_citations(stream, invocation) == []


class TestTheWholeLifecycleOnOneExecution:
    """Off, then on again, over one execution and the one after it."""

    def test_a_caught_execution_never_cites_again_but_the_next_one_does(
        self, store
    ):
        engine = get_runtime().workflow
        _registry, caught, context, _broker = _agent_turn(engine)

        # 1-4: grounded, offered, and the model has seen the handle.
        offered = engine.agent_prompt(caught, context)
        assert offered is not None
        issued = dict(caught.citations.by_handle)
        assert issued
        assert any(
            CITATION_INSTRUCTION in (message.get("content") or "")
            for message in offered
        )

        # 5-6: the operator rolls back, and this execution is caught.
        engine.invocations.configure_citation_offers(False)
        assert caught.citation_offers_intact is False

        # 7: it continues, without authority.
        assert engine.agent_prompt(caught, context) is None
        written = f"{ANSWER} [cite:{next(iter(issued))}]"
        context.canonical_model_response = {"content": written, "tool_calls": []}
        public = scrub_positions(written, caught.citations.nonce)[0]
        assert engine._recorded_citations(context, caught, public) == []
        # The namespace it was issued is untouched, so what it was shown can
        # still be recognized - and taken back out.
        assert dict(caught.citations.by_handle) == issued
        assert caught.citations.nonce not in public

        # The switch goes back on before this execution ends.
        engine.invocations.configure_citation_offers(True)
        assert caught.citation_offers_intact is False
        assert engine.agent_prompt(caught, context) is None
        assert engine._recorded_citations(context, caught, public) == []

        # A new execution opened afterwards cites again.
        _registry2, fresh, context2, _broker2 = _agent_turn(engine)
        assert fresh is not caught
        assert engine.agent_prompt(fresh, context2) is not None
        assert fresh.citations


class TestARetryCannotRecoverAuthority:
    """Citation state belongs to the logical execution, not to one attempt."""

    def test_the_replacement_attempt_finds_the_rolled_back_execution(self):
        registry = InvocationRegistry()
        first = registry.open("inv-1", tool="agent.files_v1")
        registry.configure_citation_offers(False)
        assert first.citation_offers_intact is False

        # The policy is back on before the retry, which is the case that
        # would otherwise let a replacement worker read it afresh.
        registry.configure_citation_offers(True)
        second = registry.open("inv-1", tool="agent.files_v1")

        assert second is first
        assert second.citation_offers_intact is False

    def test_a_replay_grants_no_new_handle(self):
        registry_of_sources, ground, _source, _evidence = _grounded_registry()
        registry = InvocationRegistry()
        invocation = registry.open("inv-1")
        invocation.extend_citations(registry_of_sources, [ground])
        issued = dict(invocation.citations.by_handle)

        registry.configure_citation_offers(False)
        registry.configure_citation_offers(True)
        replayed = registry.open("inv-1")

        assert replayed is invocation
        replayed.extend_citations(registry_of_sources, [ground])
        assert dict(replayed.citations.by_handle) == issued


class TestNoProductionPathReadsTheOldClassAttribute:
    """The census, as a witness. The old switch was one attribute read from
    five places; a sixth that kept reading it would be a path the rollback
    never reached."""

    def test_the_name_is_gone_from_the_package(self):
        from pathlib import Path  # noqa: PLC0415

        root = Path(__file__).resolve().parent.parent / "liminallm"
        offenders = [
            str(path.relative_to(root))
            for path in root.rglob("*.py")
            if "CITATION_OFFERS_ENABLED" in path.read_text()
        ]
        assert not offenders, offenders

    def test_every_authority_path_reads_the_execution(self):
        """Named rather than counted: each of these decides whether authority
        is granted, and each must ask the execution that would grant it."""
        from pathlib import Path  # noqa: PLC0415

        root = Path(__file__).resolve().parent.parent / "liminallm" / "service"
        workflow = (root / "workflow.py").read_text()
        streaming = (root / "workflow_streaming.py").read_text()
        assert workflow.count("invocation.citation_offers_intact") >= 3
        assert streaming.count("invocation.citation_offers_intact") >= 2

    def test_no_containment_decision_reads_the_policy(self):
        """The regression this whole tranche is most likely to produce: a
        rollback that also turns the scrubber off for a namespace the model
        has already been shown."""
        from pathlib import Path  # noqa: PLC0415

        root = Path(__file__).resolve().parent.parent / "liminallm" / "service"
        streaming = (root / "workflow_streaming.py").read_text()
        for line in streaming.splitlines():
            if "ScrubbedTokenStream(" not in line:
                continue
            assert "citation_offers" not in line, line
        # The two installation guards, as they must read.
        assert streaming.count("if not invocation.citations:\n") >= 0
        broker = (root / "broker.py").read_text()
        assert "citation_offers_intact" not in broker
