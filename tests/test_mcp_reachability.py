"""An ordinary chat turn, with MCP as the only reason to have tools.

The seam these cover is one layer above `test_mcp_turn.py`. That file proves
the tools are assembled correctly *once something has chosen the tool-agent
path*. This one proves an ordinary turn chooses it.

They are separable, and the gap between them was real: the selector asked for
attachments or an enabled web tool and knew nothing about MCP, so the exact
configuration an operator gets after publishing one server - tool-capable
backend, web off, nothing attached - took the plain-chat workflow and never
discovered anything.

Driven through `run` and `run_streaming`, which are what the chat routes call.
"""

from __future__ import annotations

import asyncio
import time
import uuid

import pytest

from liminallm.service.runtime import get_runtime
from tests.mcpfixture import MCPFixture, allow_local

_CONVERSATION = "00000000-0000-4000-8000-0000000000e1"


@pytest.fixture
def engine(monkeypatch):
    """A tool-capable deployment with web off and loopback reachable.

    Web off is the point, not a convenience: with it on, the selector reaches
    the tool agent for a reason that has nothing to do with MCP, and every
    test here would pass against the unfixed selector.
    """
    engine = get_runtime().workflow
    monkeypatch.setattr(engine, "tool_network_policy", allow_local())
    monkeypatch.setattr(
        type(engine.llm.backend), "supports_tools", property(lambda _self: True)
    )
    monkeypatch.setattr(
        engine, "_web_settings", lambda: {"enabled": False, "provider": "none"}
    )
    return engine


def _publish(store, fixture):
    user = store.create_user(email=f"reach_{uuid.uuid4().hex[:8]}@example.com")
    store.update_user_role(user.id, role="admin")
    return store.create_artifact(
        "mcp",
        fixture.name,
        fixture.as_artifact_schema(taint_class="local_read"),
        owner_user_id=user.id,
        visibility="global",
    )


class TestAnMCPOnlyTurnReachesTheToolAgent:
    def test_the_blocking_turn_discovers_the_server(self, engine, store):
        """Asserted on the fixture's own records.

        Not on the workflow name and not on the reply: the question is whether
        a real remote server was listed during an ordinary turn, and only the
        server can answer that.
        """
        name = f"blk{uuid.uuid4().hex[:6]}"
        with MCPFixture(name, {"lookup": "found it"}) as fixture:
            _publish(store, fixture)

            asyncio.run(
                engine.run(
                    None,
                    _CONVERSATION,
                    "what does the tool say?",
                    None,
                    user_id=None,
                    tenant_id=None,
                )
            )

            assert fixture.listed, (
                "an ordinary turn never listed the configured server, so the "
                "selector kept it out of the tool-agent path"
            )

    def test_the_streaming_turn_discovers_the_server(self, engine, store):
        """The chat window's path, which selects independently."""
        name = f"str{uuid.uuid4().hex[:6]}"
        with MCPFixture(name, {"lookup": "found it"}) as fixture:
            _publish(store, fixture)

            async def _drain():
                async for _event in engine.run_streaming(
                    None,
                    _CONVERSATION,
                    "what does the tool say?",
                    None,
                    user_id=None,
                    tenant_id=None,
                ):
                    pass

            asyncio.run(_drain())

            assert fixture.listed, (
                "the streaming selector kept the configured server out of the "
                "tool-agent path"
            )

    def test_no_server_configured_still_takes_the_plain_path(self, engine):
        """The selector must not send every turn through the agent.

        With nothing attached, web off and no server published, a plain chat
        is the right answer - the agent path costs a worker process and a
        round of tool offers for nothing.
        """
        selected = engine._turn_needs_tools(_CONVERSATION, None)

        assert selected is False

    def test_the_selector_reads_persisted_state_not_the_wire(self, engine, store):
        """No probe in the decision, measured on the server's own records.

        Choosing a workflow must not depend on a third party answering: an
        unreachable server would otherwise decide, per request and after a
        timeout, whether the turn can use its attachments. Discovery stays
        inside the agent context, where one server being down already costs
        only its own tools.
        """
        name = f"sel{uuid.uuid4().hex[:6]}"
        with MCPFixture(name) as fixture:
            _publish(store, fixture)

            assert engine._turn_needs_tools(_CONVERSATION, None) is True
            assert not fixture.listed, (
                "the selector connected to the server to make its decision"
            )


class TestDiscoveryDoesNotHoldTheEventLoop:
    """A slow third party must cost its own turn, not the worker.

    `run_sync` answers an already-running loop by starting a thread and then
    joining it, so the loop thread sits in `join()` for the whole listing.
    With a 10-second discovery timeout and servers listed one after another,
    a couple of unhealthy servers can stall every unrelated request on that
    worker for tens of seconds.

    The heartbeat is the instrument: it is an ordinary asyncio task, so it
    ticks only while the loop is free to run it.
    """

    DELAY = 1.0
    #: The instrument is the longest gap between heartbeat ticks, not the tick
    #: count. Counting ticks over the whole turn measures nothing: a turn does
    #: plenty of other awaiting, so the count reaches any threshold from the
    #: parts that were never blocked - measured, the first version of these
    #: tests passed against the defect for exactly that reason. A gap is local
    #: to the stall and cannot be paid for elsewhere.
    TICK = 0.02
    MAX_STALL = 0.5

    async def _longest_stall(self, work) -> float:
        worst = 0.0

        async def _heartbeat():
            nonlocal worst
            last = time.monotonic()
            while True:
                await asyncio.sleep(self.TICK)
                now = time.monotonic()
                worst = max(worst, now - last)
                last = now

        beat = asyncio.create_task(_heartbeat())
        try:
            await work
        finally:
            beat.cancel()
        return worst

    def test_a_turn_leaves_the_loop_free_while_a_server_is_slow(
        self, engine, store
    ):
        """`_invoke_tool` is the call site that stalled, measured.

        Only one test here, not two, because only one path reproduces. The
        streaming path was the one named as suspect, and it turned out to
        reach `_build_agent_context` from a worker thread already: with both
        offloads reverted its worst loop gap was 0.021s across a 1.0s
        listing, while this path's was 1.10s. The streaming offload stays as
        the right discipline for blocking I/O in an `async def`, and is
        honestly recorded as having no witness.
        """
        name = f"slowb{uuid.uuid4().hex[:6]}"
        with MCPFixture(name, {"read": "ok"}, list_delay=self.DELAY) as fixture:
            _publish(store, fixture)

            work = engine.run(
                None, _CONVERSATION, "hello", None, user_id=None, tenant_id=None
            )
            stall = asyncio.run(self._longest_stall(work))

            assert fixture.listed, "the server was never listed"
            assert stall < self.MAX_STALL, (
                f"the loop was blocked for {stall:.2f}s during a "
                f"{self.DELAY}s discovery"
            )
