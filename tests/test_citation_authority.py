"""Citations come from what the model said, never from what the worker returns.

The parent keeps the model's answer with its handles; the worker gets that
answer scrubbed and sends back what it claims the result is. Those two strings
are compared exactly, and only then are markers read - out of the parent's
copy. A worker that changed one word transfers nothing, and a worker that
writes a marker of its own is writing into a string nobody parses.
"""

from __future__ import annotations

import asyncio
import json
import random
import string
import uuid

import pytest

from liminallm.service.citations import (
    CitationTable,
    build_citation_table,
    mint_nonce,
    scrub_positions,
    transfer_citations,
)
from liminallm.service.provenance import SourceRegistry, binding
from liminallm.service.runtime import get_runtime
from tests.mcpfixture import allow_local

NONCE = "K7Q2ABCD"
ANSWER = "400 hours."


def _tools_on(monkeypatch, engine):
    """The agent path runs only when the backend declares tool support; a
    plan with no tools falls back to `llm.generic` and never reaches the
    seam. Patched on the real backend rather than replaced with a double."""
    monkeypatch.setattr(
        type(engine.llm.backend), "supports_tools", property(lambda _self: True)
    )
    monkeypatch.setattr(engine, "tool_network_policy", allow_local())


def _registry(count=1):
    """A registry holding `count` grounded sources, and its bindings."""
    registry = SourceRegistry()
    bindings = []
    for index in range(count):
        source = registry.register_source(
            kind="file", title=f"manual{index}.md", locator=f"/files/m{index}.md"
        )
        evidence = registry.add_evidence(source.source_id, text=ANSWER)
        bindings.append(binding(source.source_id, evidence.evidence_id))
    return registry, bindings


def _table(count=1):
    registry, bindings = _registry(count)
    return registry, build_citation_table(registry, bindings, nonce=NONCE)


class TestTheAnswerMustBeTheOneTheModelWrote:
    def test_an_unedited_answer_transfers_its_citations(self):
        _, table = _table()
        handle = table.handle_for("src_1")
        found = transfer_citations(
            {"content": f"{ANSWER} [cite:{handle}]"}, table, ANSWER
        )
        assert [item["source_id"] for item in found] == ["src_1"]

    def test_a_worker_that_changed_one_word_transfers_nothing(self):
        """The whole boundary in one case: the worker holds the handle, and
        the answer it returns is not the answer that carried it."""
        _, table = _table()
        handle = table.handle_for("src_1")
        assert (
            transfer_citations(
                {"content": f"400 hours [cite:{handle}]."}, table, "800 hours."
            )
            == []
        )

    def test_a_reformatted_answer_transfers_nothing(self):
        """Exact, with no normalization. Losing a citation is the safe
        direction; accepting an answer the model did not write is not."""
        _, table = _table()
        handle = table.handle_for("src_1")
        canonical = {"content": f"{ANSWER} [cite:{handle}]"}
        assert transfer_citations(canonical, table, ANSWER + "\n") == []
        assert transfer_citations(canonical, table, " " + ANSWER) == []

    def test_the_default_answer_the_worker_invents_transfers_nothing(self):
        """When an assembly produces no content the worker substitutes its
        own sentence. That is not an answer the model wrote."""
        _, table = _table()
        handle = table.handle_for("src_1")
        assert (
            transfer_citations(
                {"content": f"{ANSWER} [cite:{handle}]"},
                table,
                "I could not derive an answer from the available sources.",
            )
            == []
        )


class TestAuthorityIsReadOnlyFromTheParentsCopy:
    def test_a_marker_the_worker_wrote_is_never_parsed(self):
        """The worker's text is compared, not read. Here it carries a valid
        handle and still yields nothing, because the string that matched has
        no marker in it."""
        _, table = _table()
        handle = table.handle_for("src_1")
        assert transfer_citations(
            {"content": ANSWER}, table, f"{ANSWER} [cite:{handle}]"
        ) == []

    def test_the_canonical_offsets_span_the_marker_in_the_model_s_answer(self):
        """Named for the string they index, because it is not the string the
        caller holds - that one has no marker in it at all."""
        _, table = _table()
        handle = table.handle_for("src_1")
        canonical = f"{ANSWER} [cite:{handle}]"
        found = transfer_citations({"content": canonical}, table, ANSWER)
        span = canonical[found[0]["canonical_start"] : found[0]["canonical_end"]]
        assert span == f"[cite:{handle}]"

    def test_the_handle_stops_at_the_authority_boundary(self):
        """It named a source for the model and resolved to `source_id`. Past
        that it is only the nonce, travelling further than the invocation
        that minted it."""
        _, table = _table()
        handle = table.handle_for("src_1")
        found = transfer_citations(
            {"content": f"{ANSWER} [cite:{handle}]"}, table, ANSWER
        )
        assert found and "handle" not in found[0], found
        assert NONCE not in json.dumps(found)


class TestThePublicOffsetPointsIntoTheAnswerTheCallerHolds:
    """The marker is gone from that string, so the offset is where it was -
    an insertion point. Taken from the scrub that removed it, not from
    arithmetic over marker widths."""

    @staticmethod
    def _split(canonical, table):
        public, _ = scrub_positions(canonical, table.nonce)
        found = transfer_citations({"content": canonical}, table, public)
        return public, [
            (public[: item["public_offset"]], public[item["public_offset"] :])
            for item in found
        ]

    def test_each_marker_reopens_where_it_was_taken_out(self):
        _, table = _table(count=2)
        canonical = (
            f"Alpha [cite:{table.handle_for('src_1')}]. "
            f"Beta [cite:{table.handle_for('src_2')}]."
        )
        public, splits = self._split(canonical, table)
        assert public == "Alpha. Beta."
        assert splits == [("Alpha", ". Beta."), ("Alpha. Beta", ".")]

    def test_a_bare_nonce_earlier_in_the_answer_shifts_the_offsets(self):
        """The case that separates a real mapping from subtracting marker
        widths: nothing here is a marker, and everything after it moves."""
        _, table = _table()
        canonical = f"{NONCE} Alpha [cite:{table.handle_for('src_1')}]."
        public, splits = self._split(canonical, table)
        assert public == " Alpha."
        assert splits == [(" Alpha", ".")]
        # Marker-width arithmetic would have said 15 - 0 = 15, past the end.
        found = transfer_citations({"content": canonical}, table, public)
        assert found[0]["canonical_start"] == 15
        assert found[0]["public_offset"] == 6

    def test_a_repeated_splice_before_a_marker_shifts_the_offsets(self):
        """Two passes of removal, so the offsets have to compose rather than
        describe one pass."""
        _, table = _table()
        spliced = NONCE[:4] + NONCE + NONCE[4:]
        canonical = f"{spliced}Alpha [cite:{table.handle_for('src_1')}]."
        public, splits = self._split(canonical, table)
        assert public == "Alpha."
        assert splits == [("Alpha", ".")]

    def test_every_origin_names_the_character_it_kept(self):
        """The invariant the whole mapping rests on: reading the original
        text at the recorded positions reproduces the scrubbed string, in
        order. Measured over generated input because a mapping that composes
        wrongly across passes is right on any one example someone picks."""
        random.seed(20260903)
        pool = string.printable + NONCE + "[cite:]"
        for index in range(2000):
            nonce = mint_nonce()
            filler = [
                "".join(random.choice(pool) for _ in range(random.randint(0, 8)))
                for _ in range(3)
            ]
            spliced = nonce[:4] + nonce + nonce[4:]
            shape = index % 3
            if shape == 0:
                text = spliced + filler[0]
            elif shape == 1:
                # A first-pass removal, then surviving text, then a splice
                # that only forms on the second pass. That is the shape where
                # a later pass has to read positions through the earlier one
                # rather than through the string in front of it.
                text = nonce + filler[0] + spliced + filler[1]
            else:
                text = (
                    filler[0]
                    + f"[cite:{nonce}-1]"
                    + filler[1]
                    + nonce.lower()
                    + filler[2]
                )
            public, origins = scrub_positions(text, nonce)
            assert public == "".join(text[at] for at in origins), (text, origins)
            assert origins == sorted(origins), (text, origins)


class TestThereIsNoCanonicalResponseToTransferFrom:
    def test_no_canonical_response_transfers_nothing(self):
        _, table = _table()
        assert transfer_citations(None, table, ANSWER) == []

    def test_an_empty_canonical_answer_transfers_nothing(self):
        """Stated rather than left to the comparison: two empty strings are
        equal, so a malformed canonical response carrying markers somewhere
        other than `content` must be refused explicitly."""
        _, table = _table()
        handle = table.handle_for("src_1")
        canonical = {"content": "", "assistant_message": f"[cite:{handle}]"}
        assert transfer_citations(canonical, table, "") == []

    def test_a_turn_that_issued_no_handles_transfers_nothing(self):
        registry = SourceRegistry()
        table = build_citation_table(registry, [], nonce=NONCE)
        assert transfer_citations({"content": ANSWER}, table, ANSWER) == []


class TestOnlyTheLastCanonicalResponseIsEligible:
    """Two model turns whose public text is identical and whose citations are
    not. `canonical_model_response` is replacement state, so the answer is
    matched against the last one and never searched for among the rest."""

    def test_an_earlier_response_is_not_reachable_by_matching_prose(self):
        _, table = _table(count=2)
        first = f"{ANSWER} [cite:{table.handle_for('src_1')}]"
        last = f"{ANSWER} [cite:{table.handle_for('src_2')}]"
        # Both scrub to exactly the worker's answer.
        assert transfer_citations({"content": first}, table, ANSWER)
        assert transfer_citations({"content": last}, table, ANSWER)
        # The seam is handed one canonical response, so only its sources can
        # be reached. The check is that the caller passes the last, which is
        # what `_apply_parent_state` replacement guarantees.
        found = transfer_citations({"content": last}, table, ANSWER)
        assert [item["source_id"] for item in found] == ["src_2"]


class TestThroughTheRealAgentLoop:
    """The helper above is exact by construction. This settles whether the
    blocking path actually returns the answer unchanged, which is the only
    reason exact comparison is usable."""

    @staticmethod
    def _seed(engine, monkeypatch):
        registry, bindings = _registry()
        invocation = engine.invocations.open(
            uuid.uuid4().hex,
            tool="agent.files_v1",
            user_id="u",
            tenant_id=None,
        )
        invocation.extend_citations(registry, bindings)
        _tools_on(monkeypatch, engine)
        return registry, invocation

    @staticmethod
    async def _run(engine, registry, invocation, tool="agent.files_v1"):
        return await engine._invoke_tool(
            tool,
            {"message": "how long"},
            [],
            [],
            None,
            uuid.uuid4().hex,
            "how long",
            source_registry=registry,
            user_id="u",
            tenant_id=None,
            invocation=invocation,
        )

    @pytest.mark.asyncio
    async def test_a_cited_answer_survives_the_worker_unchanged(
        self, store, monkeypatch
    ):
        engine = get_runtime().workflow
        registry, invocation = self._seed(engine, monkeypatch)
        handle = invocation.citations.handle_for("src_1")
        monkeypatch.setattr(
            engine.llm,
            "generate_with_tools",
            lambda *a, **k: {
                "content": f"{ANSWER} [cite:{handle}]",
                "tool_calls": [],
                "assistant_message": None,
                "usage": {},
            },
            raising=False,
        )
        result = await self._run(engine, registry, invocation)

        assert result.get("status") != "error", result
        # The empirical claim exact comparison rests on: the terminal model
        # response reaches the parent byte-for-byte, minus the namespace.
        assert result.get("content") == ANSWER, repr(result.get("content"))
        assert result.get("validated_citations") == [
            {
                "source_id": "src_1",
                "canonical_start": len(ANSWER) + 1,
                "canonical_end": len(ANSWER) + 1 + len(f"[cite:{handle}]"),
                # The marker sat at the very end, after a space that goes
                # with it, so it reopens exactly where the answer stops.
                "public_offset": len(ANSWER),
            }
        ], result.get("validated_citations")

    @pytest.mark.asyncio
    async def test_a_worker_that_rewrites_the_answer_gets_no_citations(
        self, store, monkeypatch
    ):
        """The worker is the untrusted half, so this is the case that
        matters: it holds a real handle and returns different prose."""
        engine = get_runtime().workflow
        registry, invocation = self._seed(engine, monkeypatch)
        handle = invocation.citations.handle_for("src_1")
        monkeypatch.setattr(
            engine.llm,
            "generate_with_tools",
            lambda *a, **k: {
                "content": f"{ANSWER} [cite:{handle}]",
                "tool_calls": [],
                "assistant_message": None,
                "usage": {},
            },
            raising=False,
        )
        real = engine.tool_postflight

        def _rewrite(result, *args, **kwargs):
            result = dict(result)
            result["content"] = "800 hours."
            return real(result, *args, **kwargs)

        monkeypatch.setattr(engine, "tool_postflight", _rewrite)
        result = await self._run(engine, registry, invocation)

        assert result.get("content") == "800 hours."
        assert not result.get("validated_citations")

    @pytest.mark.asyncio
    async def test_the_namespace_still_does_not_reach_the_result(
        self, store, monkeypatch
    ):
        engine = get_runtime().workflow
        registry, invocation = self._seed(engine, monkeypatch)
        handle = invocation.citations.handle_for("src_1")
        monkeypatch.setattr(
            engine.llm,
            "generate_with_tools",
            lambda *a, **k: {
                "content": f"{ANSWER} [cite:{handle}]",
                "tool_calls": [],
                "assistant_message": None,
                "usage": {},
            },
            raising=False,
        )
        result = await self._run(engine, registry, invocation)
        assert invocation.citations.nonce not in str(result.get("content"))

    @pytest.mark.asyncio
    async def test_a_diverged_assembly_transfers_nothing_it_still_matches(
        self, store, monkeypatch
    ):
        """The attack the round-level check alone does not stop.

        The model writes a handle it genuinely learned, and the answer comes
        back byte-for-byte, so every other gate here passes. What the parent
        cannot say is which conversation the model wrote it in: once a round
        diverged from the turn that asked for it, the surrounding prompt was
        the worker's to compose.

        Divergence is forced at the point it is decided, rather than by
        pretending a worker misbehaved, because what is under test is what
        happens after it is detected - the detection has its own witnesses.
        The same run without the divergence is asserted beside it, so this
        cannot pass by transferring nothing for some unrelated reason.
        """
        from liminallm.service import broker as broker_module

        engine = get_runtime().workflow

        def _asks_then_answers(handle):
            """A model that runs one tool round and then answers, citing."""
            state = {"turns": 0}

            def _generate(*_a, **_k):
                state["turns"] += 1
                if state["turns"] == 1:
                    return {
                        "content": "",
                        "tool_calls": [{
                            "id": "c1",
                            "name": "file_search",
                            "arguments": '{"query": "hours"}',
                        }],
                        "assistant_message": None,
                        "usage": {},
                    }
                return {
                    "content": f"{ANSWER} [cite:{handle}]",
                    "tool_calls": [],
                    "assistant_message": None,
                    "usage": {},
                }

            return _generate

        registry, invocation = self._seed(engine, monkeypatch)
        monkeypatch.setattr(
            engine.llm,
            "generate_with_tools",
            _asks_then_answers(invocation.citations.handle_for("src_1")),
            raising=False,
        )
        honest = await self._run(engine, registry, invocation)
        assert honest.get("validated_citations"), honest

        registry, invocation = self._seed(engine, monkeypatch)
        monkeypatch.setattr(
            engine.llm,
            "generate_with_tools",
            _asks_then_answers(invocation.citations.handle_for("src_1")),
            raising=False,
        )
        monkeypatch.setattr(
            broker_module, "calls_match", lambda offered, submitted: False
        )
        diverged = await self._run(engine, registry, invocation)

        # The answer is still the model's own, unedited - the transfer's
        # other gate passes and this one is what refuses.
        assert diverged.get("content") == ANSWER, repr(diverged.get("content"))
        assert not diverged.get("validated_citations"), diverged


class TestTheGateIsTheResolvedWorkerBody:
    @pytest.mark.asyncio
    async def test_a_non_agent_tool_transfers_no_citations(
        self, store, monkeypatch
    ):
        """`llm.generic` never makes `llm.generate_with_tools` broker calls,
        so it has no canonical response. Gated on the resolved body anyway,
        so a later host tool filling in similar state inherits nothing."""
        engine = get_runtime().workflow
        registry, invocation = self._seed_generic(engine, monkeypatch)
        result = await engine._invoke_tool(
            "llm.generic",
            {"message": "how long"},
            [],
            [],
            None,
            uuid.uuid4().hex,
            "how long",
            source_registry=registry,
            user_id="u",
            tenant_id=None,
            invocation=invocation,
        )
        assert result.get("status") != "error", result
        assert not result.get("validated_citations")

    @staticmethod
    def _seed_generic(engine, monkeypatch):
        registry, bindings = _registry()
        invocation = engine.invocations.open(
            uuid.uuid4().hex, tool="llm.generic", user_id="u", tenant_id=None
        )
        invocation.extend_citations(registry, bindings)
        handle = invocation.citations.handle_for("src_1")
        monkeypatch.setattr(
            engine.llm,
            "generate",
            lambda *a, **k: {
                "content": f"{ANSWER} [cite:{handle}]",
                "usage": {},
            },
            raising=False,
        )
        return registry, invocation

    @pytest.mark.asyncio
    async def test_a_failed_agent_result_transfers_no_citations(
        self, store, monkeypatch
    ):
        engine = get_runtime().workflow
        registry, bindings = _registry()
        invocation = engine.invocations.open(
            uuid.uuid4().hex, tool="agent.files_v1", user_id="u", tenant_id=None
        )
        invocation.extend_citations(registry, bindings)
        handle = invocation.citations.handle_for("src_1")
        _tools_on(monkeypatch, engine)
        monkeypatch.setattr(
            engine.llm,
            "generate_with_tools",
            lambda *a, **k: {
                "content": f"{ANSWER} [cite:{handle}]",
                "tool_calls": [],
                "assistant_message": None,
                "usage": {},
            },
            raising=False,
        )
        real = engine.tool_postflight

        def _fail(result, *args, **kwargs):
            sanitized, refusal = real(result, *args, **kwargs)
            sanitized["status"] = "error"
            return sanitized, refusal

        monkeypatch.setattr(engine, "tool_postflight", _fail)
        result = await TestThroughTheRealAgentLoop._run(
            engine, registry, invocation
        )
        assert result.get("status") == "error"
        assert not result.get("validated_citations")


class TestTheStreamedAnswerIsNotThisWorkersToCite:
    """With `stream_final` the worker stops before the final answer and the
    parent streams it, so the worker's `content` is the last *tool* round's
    text rather than the answer. That path calls `_serve_invocation` directly
    and never reaches the transfer seam, which is what makes the seam's
    `stream_final` guard unkillable - and the second test here is why it is
    kept anyway.
    """

    @staticmethod
    def _streamed(engine, monkeypatch, store, spy=None):
        _tools_on(monkeypatch, engine)
        user_id = store.create_user(
            email=f"cite_{uuid.uuid4().hex[:8]}@example.com"
        ).id
        rounds = {"n": 0}

        def _generate(*args, **kwargs):
            # One tool-calling round carrying prose, then a terminal round
            # carrying the same prose. That is what makes the worker's
            # returned content and the canonical response coincide.
            rounds["n"] += 1
            return {
                "content": ANSWER,
                "assistant_message": None,
                "usage": {},
                "tool_calls": (
                    [
                        {
                            "id": "c1",
                            "name": "web_search",
                            "arguments": '{"query":"how long"}',
                        }
                    ]
                    if rounds["n"] == 1
                    else []
                ),
            }

        monkeypatch.setattr(
            engine.llm, "generate_with_tools", _generate, raising=False
        )

        def _stream(messages, adapters, **kwargs):
            yield {"event": "token", "data": ANSWER}
            yield {"event": "message_done", "data": {"content": ANSWER}}

        monkeypatch.setattr(engine.llm, "stream_messages", _stream, raising=False)
        if spy is not None:
            real = type(engine)._serve_invocation

            def _watch(self, invocation, worker_tool, plan, context, *a, **kw):
                result = real(self, invocation, worker_tool, plan, context, *a, **kw)
                spy.append((plan, result, context))
                return result

            monkeypatch.setattr(type(engine), "_serve_invocation", _watch)
        return user_id

    @pytest.mark.asyncio
    async def test_a_streamed_turn_carries_no_validated_citations(
        self, store, monkeypatch
    ):
        engine = get_runtime().workflow
        user_id = self._streamed(engine, monkeypatch, store)
        events = [
            event
            async for event in engine.run_streaming(
                None, None, "how long", None, user_id
            )
        ]

        done = [e for e in events if e.get("event") == "message_done"]
        assert done, [e.get("event") for e in events]
        for event in events:
            payload = event.get("data")
            if isinstance(payload, dict):
                assert not payload.get("validated_citations"), event

    @pytest.mark.asyncio
    async def test_the_streamed_workers_content_can_equal_the_answer_it_did_not_write(
        self, store, monkeypatch
    ):
        """Why the guard is kept although nothing can kill it: on this path
        the two strings an exact comparison would compare do coincide. A
        refactor routing streaming through the seam would hand the streamed
        answer citations from a response that is not it."""
        engine = get_runtime().workflow
        served = []
        user_id = self._streamed(engine, monkeypatch, store, spy=served)
        async for _event in engine.run_streaming(
            None, None, "how long", None, user_id
        ):
            pass

        agent = [entry for entry in served if entry[0].get("stream_final")]
        assert agent, [entry[0].keys() for entry in served]
        plan, result, context = agent[-1]
        canonical = (context.canonical_model_response or {}).get("content")
        assert result.get("content") == canonical, (result.get("content"), canonical)


class TestTheNamesTravelWithTheCitations:
    """A citation says `src_3`, which means nothing once the turn's registry
    goes out of scope. Whatever resolves the name has to travel beside it."""

    def test_a_turn_that_validated_citations_carries_the_names(
        self, store, monkeypatch
    ):
        engine = get_runtime().workflow
        _tools_on(monkeypatch, engine)
        user_id = store.create_user(
            email=f"carry_{uuid.uuid4().hex[:8]}@example.com"
        ).id
        seeded = {}
        real_record = engine._record_grounding

        def _seed(registry, *args, **kwargs):
            # Ride the turn's own registry rather than a second one: the
            # snapshot has to resolve names minted in the registry the node
            # actually used.
            bindings = real_record(registry, *args, **kwargs)
            source = registry.register_source(
                kind="file", title="manual.md", locator="/files/manual.md"
            )
            evidence = registry.add_evidence(source.source_id, text=ANSWER)
            seeded["registry"] = registry
            seeded["source_id"] = source.source_id
            return list(bindings) + [
                binding(source.source_id, evidence.evidence_id)
            ]

        monkeypatch.setattr(engine, "_record_grounding", _seed)
        real_open = engine.invocations.open

        def _open(*args, **kwargs):
            invocation = real_open(*args, **kwargs)
            # By tool, not by recency: the turn opens another invocation of
            # its own after this one, and seeding that table would issue a
            # handle the seam never sees.
            if kwargs.get("tool") == "agent.files_v1":
                seeded["invocation"] = invocation
            return invocation

        monkeypatch.setattr(engine.invocations, "open", _open)

        def _generate(*args, **kwargs):
            # Standing in for S6's offer: planning already grounded, and the
            # invocation exists by the time the model is called, so this is
            # where a handle can be issued and then cited.
            invocation = seeded["invocation"]
            registry = seeded["registry"]
            source_id = seeded["source_id"]
            invocation.extend_citations(
                registry,
                [
                    binding(source_id, item.evidence_id)
                    for item in registry.evidence_for(source_id)
                ],
            )
            handle = invocation.citations.handle_for(source_id)
            return {
                "content": f"{ANSWER} [cite:{handle}]",
                "tool_calls": [],
                "assistant_message": None,
                "usage": {},
            }

        monkeypatch.setattr(
            engine.llm, "generate_with_tools", _generate, raising=False
        )
        result = asyncio.run(engine.run(None, None, "how long", None, user_id))

        citations = result.get("validated_citations")
        assert citations, f"the turn validated nothing: {sorted(result)}"
        snapshot = result.get("provenance_snapshot")
        assert snapshot, "citations travelled with no way to resolve their names"
        for item in citations:
            assert item["source_id"] in snapshot["sources"], (item, snapshot)
        # Parent bookkeeping, like the bindings beside it - not a node output
        # a graph author reads or a later node interpolates.
        assert "validated_citations" not in (result.get("vars") or {}), result["vars"]

    def test_a_turn_with_no_citations_carries_no_snapshot(self, store, monkeypatch):
        """Transport for something, or not present. An empty citation list
        needs no lookup table."""
        engine = get_runtime().workflow
        _tools_on(monkeypatch, engine)
        user_id = store.create_user(
            email=f"bare_{uuid.uuid4().hex[:8]}@example.com"
        ).id
        monkeypatch.setattr(
            engine.llm,
            "generate_with_tools",
            lambda *a, **k: {
                "content": ANSWER,
                "tool_calls": [],
                "assistant_message": None,
                "usage": {},
            },
            raising=False,
        )
        result = asyncio.run(engine.run(None, None, "how long", None, user_id))
        assert result.get("validated_citations") == []
        assert "provenance_snapshot" not in result

    def test_the_snapshot_resolves_every_validated_source(self):
        registry, table = _table(count=2)
        handle = table.handle_for("src_2")
        found = transfer_citations(
            {"content": f"{ANSWER} [cite:{handle}]"}, table, ANSWER
        )
        snapshot = registry.snapshot()
        for item in found:
            assert item["source_id"] in snapshot["sources"], snapshot

    def test_an_empty_table_is_falsey(self):
        assert not CitationTable(nonce=NONCE)
