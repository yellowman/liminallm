"""Resolving the context window: one discovery, one answer, no frozen worker.

The window is resolved lazily by probing the provider, and that probe is
synchronous HTTP. Two things follow, and only one of them was a defect.

**The worker keeps serving.** A turn that triggers discovery waits for it -
that is inherent. What must not happen is the rest of the worker waiting too.
It does not today, and the reason is worth knowing but not worth depending
on: planning runs off the event loop and asks for the budget on its way
through, so the streamed nodes find the answer already cached. That is
ordering, not design - a new node, a reordering, or a turn shape that skips
planning would put a blocking probe back on the loop. So the first witness
pins the behaviour a worker's other requests actually care about, and says
nothing about which function resolves first.

**One discovery, not one per caller.** This one was a real hole. Turns
starting together on a cold engine each resolved independently, and
discovery need not answer them the same way: a probe that times out falls
back to the table while one that succeeds does not. Two turns could then
hold two windows - the prompt priced against one, the provider told to
compact at a threshold derived from the other - which is precisely what
sizing everything from a single resolved fact exists to prevent.

What prompt budgeting and provider-native compaction do with that one value
is pinned in `tests/test_native_compaction.py` and is not repeated here.
"""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from types import SimpleNamespace

import pytest

from liminallm.service.runtime import get_runtime
from liminallm.service.tokenizer_utils import MAX_GENERATION_TOKENS
from liminallm.storage.models import KnowledgeChunk

#: Long enough to dwarf the heartbeat, short enough to keep the suite quick.
STALL = 1.0
HEARTBEAT = 0.02
ANSWER = "400 hours."


def _stalled_backend(engine, monkeypatch, *, window=128000, stall=STALL, fail=False):
    """The real backend, with discovery made slow.

    The property is replaced rather than the service, so what runs is the
    engine's own resolution path - `resolved_context_window`, its cache and
    its fallbacks - with only the network made observable.
    """
    calls: list = []

    def slow_probe(_self):
        calls.append(time.perf_counter())
        time.sleep(stall)
        if fail:
            raise RuntimeError("the endpoint did not answer")
        return window

    monkeypatch.setattr(
        type(engine.llm.backend), "context_window", property(slow_probe)
    )
    engine._budget_cache = None
    return calls


async def _heartbeat(stop, gaps):
    """Tick steadily; a gap is the loop failing to come back."""
    last = time.perf_counter()
    while not stop.is_set():
        await asyncio.sleep(HEARTBEAT)
        now = time.perf_counter()
        gaps.append(now - last)
        last = now


def _streaming(engine, monkeypatch, store):
    """A plain streamed turn with retrieval stubbed and one token to send."""
    monkeypatch.setattr(
        type(engine.llm.backend), "supports_tools", property(lambda _s: False)
    )
    chunk = KnowledgeChunk(
        context_id="ctx", fs_path="/files/m.md", content=ANSWER,
        embedding=[], chunk_index=0,
    )
    monkeypatch.setattr(
        engine, "rag", SimpleNamespace(retrieve=lambda *a, **k: [chunk])
    )
    monkeypatch.setattr(engine, "_validate_context_scope", lambda ids, **k: ["ctx"])
    monkeypatch.setattr(engine, "_resolve_context_ids", lambda a, b: ["ctx"])

    def _generate_stream(prompt, adapters=None, context_snippets=None,
                         history=None, *, user_id=None, instruction=None):
        yield {"event": "token", "data": ANSWER}
        yield {"event": "message_done", "data": {"content": ANSWER}}

    monkeypatch.setattr(
        engine.llm, "generate_stream", _generate_stream, raising=False
    )
    return store.create_user(email=f"cw_{uuid.uuid4().hex[:8]}@example.com").id


async def _run_with_heartbeat(engine, user_id):
    """Drive one streamed turn while something else wants the loop."""
    stop = asyncio.Event()
    gaps: list = []
    beat = asyncio.create_task(_heartbeat(stop, gaps))
    await asyncio.sleep(0.05)
    events = [
        event async for event in engine.run_streaming(
            None, None, "how long", "ctx", user_id
        )
    ]
    stop.set()
    await beat
    return events, (max(gaps) if gaps else 0.0)


class TestTheLoopKeepsRunningWhileDiscoveryDoes:
    @pytest.mark.asyncio
    async def test_a_streamed_turn_does_not_freeze_the_worker(
        self, store, monkeypatch
    ):
        """The defect, as a worker's other requests experience it.

        The turn that triggered discovery waits - that is inherent, and fine.
        What must not happen is the loop refusing to run anything else while
        it waits.
        """
        engine = get_runtime().workflow
        calls = _stalled_backend(engine, monkeypatch)
        user_id = _streaming(engine, monkeypatch, store)

        started = time.perf_counter()
        events, worst_gap = await _run_with_heartbeat(engine, user_id)
        elapsed = time.perf_counter() - started

        assert calls, "discovery never ran, so this witness proves nothing"
        # The turn waited for it.
        assert elapsed >= STALL
        # Everything else did not.
        assert worst_gap < STALL / 2, (
            f"the event loop stalled for {worst_gap:.2f}s while the context "
            f"window was discovered; nothing else on this worker ran"
        )
        assert any(e.get("event") == "message_done" for e in events)


class TestOneDiscoveryForEveryoneWhoAsksAtOnce:
    """The hole this tranche closes.

    Sequential execution never reached it - the second caller finds a cache.
    Turns starting together on a cold engine do, and discovery is allowed to
    answer them differently.
    """

    @staticmethod
    def _race(engine, callers=4):
        """Every caller asks at once; return what each one got."""
        answers: list = []
        start = threading.Barrier(callers)

        def resolve():
            start.wait()
            answers.append(engine.resolved_context_window())

        threads = [threading.Thread(target=resolve) for _ in range(callers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        return answers

    def test_only_one_of_them_asks_the_provider(self, monkeypatch):
        engine = get_runtime().workflow
        calls = _stalled_backend(engine, monkeypatch, stall=0.3)

        answers = self._race(engine)

        assert len(answers) == 4
        assert len(calls) == 1, (
            f"{len(calls)} concurrent first turns each probed the provider"
        )

    def test_they_all_get_the_same_window(self, monkeypatch):
        """The PN5 reason for the lock, with a provider that can disagree.

        A deterministic probe cannot witness this - four of them return the
        same number whether or not they ran. So discovery answers differently
        each time it is asked, which is what a real one does when one attempt
        times out into the table and the next succeeds. One discovery means
        one of those answers reaches everybody.
        """
        engine = get_runtime().workflow
        windows = iter([1_000_000, 128_000, 8_192, 4_096])

        def shifting_probe(_self):
            time.sleep(0.3)
            return next(windows)

        monkeypatch.setattr(
            type(engine.llm.backend), "context_window", property(shifting_probe)
        )
        engine._budget_cache = None

        answers = self._race(engine)

        assert len(set(answers)) == 1, (
            f"concurrent turns were priced against different windows: "
            f"{sorted(set(answers))}"
        )

    def test_they_agree_when_discovery_fails_too(self, monkeypatch):
        """The case that made disagreement possible: one caller's probe
        failing while another's succeeds would give the fallback to some and
        the discovered window to the rest. With one discovery there is one
        outcome, whichever it was."""
        from liminallm.service.model_backend import (  # noqa: PLC0415
            DEFAULT_CONTEXT_WINDOW,
        )

        engine = get_runtime().workflow
        calls = _stalled_backend(engine, monkeypatch, stall=0.3, fail=True)

        answers = self._race(engine)

        assert set(answers) == {DEFAULT_CONTEXT_WINDOW}
        assert len(calls) == 1

    def test_a_slow_discovery_does_not_publish_an_already_stale_answer(
        self, monkeypatch
    ):
        """The cache lifetime starts when the answer exists.

        Discovery has no wall-clock bound - the probe's timeouts are
        per-operation, not for the request as a whole - so it can outlast the
        cache interval. Timestamped when the search *began*, the winner would
        publish a value already too old to use: the next caller discards it
        and probes again, which is the stampede the lock exists to prevent,
        and on the streamed path that second probe is synchronous and reached
        from the event loop.

        Written by shrinking the interval rather than stalling the suite for
        a minute; the relation under test is the same one.
        """
        engine = get_runtime().workflow
        monkeypatch.setattr(type(engine), "_BUDGET_CACHE_SECONDS", 0.05)
        calls = _stalled_backend(engine, monkeypatch, stall=0.10)

        answers = self._race(engine)

        assert len(calls) == 1, (
            f"discovery outlasted the cache interval and was repeated "
            f"{len(calls)} times"
        )
        assert len(set(answers)) == 1, sorted(set(answers))
