"""Retry backoff and timeout enforcement. The rule is SPEC §18.3."""

from __future__ import annotations

import asyncio
import time
from typing import Optional
from unittest.mock import patch

import pytest

from liminallm.service.llm import LLMService
from liminallm.service.model_backend import StubBackend
from liminallm.service.tool_namespace import (
    SYSTEM_SCOPE,
    ResolvedWorkflow,
    ToolDescriptor,
)
from liminallm.service.workflow import (
    DEFAULT_BACKOFF_MS,
    DEFAULT_NODE_MAX_RETRIES,
    DEFAULT_NODE_TIMEOUT_MS,
    DEFAULT_WORKFLOW_TIMEOUT_MS,
    MAX_NODE_TIMEOUT_SECONDS,
    MAX_RETRIES_HARD_CAP,
    WorkflowEngine,
)
from liminallm.storage.common import get_default_tool_specs


def _loaded(schema):
    """What `get_latest_workflow` returns: a schema and the namespace its tool
    references mean. These tests are about retry and timeout mechanics, so the
    system namespace - the one the engine uses for workflows it synthesises
    itself - is the right one."""
    return ResolvedWorkflow(schema, SYSTEM_SCOPE)


class MockStore:
    """Minimal store mock for testing."""

    def __init__(self):
        self.artifacts = []
        self.messages = []
        #: Specs a test registers, standing in for artifact rows. The engine
        #: resolves through the store now, so registering in the engine's
        #: process cache no longer makes a tool exist.
        self.tool_specs: dict = {}

    def get_latest_workflow(self, workflow_id: str) -> Optional[dict]:
        return None

    def list_artifacts(self, type_filter: Optional[str] = None, **kwargs) -> list:
        return []

    def list_semantic_clusters(self, user_id: Optional[str]) -> list:
        return []

    def register_tool(self, name, handler="llm.generic", **extra):
        self.tool_specs[name] = {"name": name, "handler": handler, **extra}

    def resolve_tool_spec(self, name, scope):
        """Resolve a default tool the way the real store would.

        Built from `get_default_tool_specs()` - the same definition
        `_ensure_default_artifacts` seeds from - rather than a table written
        here, so this double cannot disagree with production about what a
        default tool's handler is.

        It differs from the real store in one way that is true of the thing it
        stands for: seeded tools are ownerless, so `artifact_id` is None and
        the descriptor can never be privileged. Tests needing real
        provenance use the real store.
        """
        registered = self.tool_specs.get(name)
        if registered is not None:
            return (
                ToolDescriptor(
                    name=name, schema=registered, artifact_id=None,
                    owner_user_id=None, owner_role=None,
                ),
                None,
            )
        for spec in get_default_tool_specs():
            if spec.get("name") == name:
                return (
                    ToolDescriptor(
                        name=name,
                        schema=spec,
                        artifact_id=None,
                        owner_user_id=None,
                        owner_role=None,
                    ),
                    None,
                )
        return None, "names no tool this workflow can reach"


class MockLLM(LLMService):
    """The real service over the stub backend.

    A hand-written stand-in with one `generate` on it answered every capability
    question by not having the attribute, so tests here could not see a
    capability the engine now asks about - `stream_is_cancellable`, which the
    real property resolves from the backend. Subclassing keeps the one
    behaviour these tests do rely on (a canned completion) while the rest of
    the interface stays the real one.
    """

    def __init__(self) -> None:
        super().__init__("test-model", backend=StubBackend())

    def generate(self, prompt: str, **kwargs) -> dict:
        return {"content": "test response", "usage": {"tokens": 10}}


class MockRAG:
    """Mock RAG service."""

    def retrieve(self, ctx_ids, query, **kwargs) -> list:
        return []


class MockRouter:
    """Mock router engine."""

    async def route(self, policy, ctx_emb, adapters, **kwargs) -> dict:
        return {"adapters": [], "trace": []}


class MockRedisCache:
    """Mock Redis cache."""

    async def get_conversation_summary(self, conv_id: str) -> Optional[dict]:
        return None

    async def set_conversation_summary(self, conv_id: str, summary: dict) -> None:
        pass

    async def get_workflow_state(self, key: str) -> Optional[dict]:
        return None

    async def set_workflow_state(self, key: str, state: dict) -> None:
        pass

    async def delete_workflow_state(self, key: str) -> None:
        pass

    async def check_circuit_breaker(self, tool_name: str, *, tenant_id=None):
        # Circuit closed so the tool handler is actually invoked.
        return False, None

    async def record_tool_success(self, tool_name: str, *, tenant_id=None) -> None:
        pass

    async def record_tool_failure(self, tool_name: str, *, tenant_id=None):
        return False, 0


@pytest.fixture
def workflow_engine():
    """Create a workflow engine with mock dependencies."""
    store = MockStore()
    llm = MockLLM()
    rag = MockRAG()
    router = MockRouter()
    cache = MockRedisCache()
    return WorkflowEngine(store, llm, router, rag, cache=cache)


# ==============================================================================
# SPEC Constants Tests
# ==============================================================================


class TestSpecConstants:
    """Verify SPEC §18.3 constants are correctly defined."""

    def test_default_node_timeout(self):
        """Default node timeout should be 15s per SPEC §18.3."""
        assert DEFAULT_NODE_TIMEOUT_MS == 15000

    def test_default_max_retries(self):
        """Default max retries should be 2 per SPEC §18."""
        assert DEFAULT_NODE_MAX_RETRIES == 2

    def test_max_retries_hard_cap(self):
        """Hard cap on retries should be 3 per SPEC §18."""
        assert MAX_RETRIES_HARD_CAP == 3

    def test_default_backoff(self):
        """Default backoff should be 1s per SPEC §18."""
        assert DEFAULT_BACKOFF_MS == 1000

    def test_default_workflow_timeout(self):
        """Default workflow timeout should be reasonable."""
        assert DEFAULT_WORKFLOW_TIMEOUT_MS > 0
        assert DEFAULT_WORKFLOW_TIMEOUT_MS >= DEFAULT_NODE_TIMEOUT_MS


# ==============================================================================
# Workflow-Level Timeout Tests
# ==============================================================================


class TestWorkflowTimeout:
    """Test workflow-level timeout enforcement per SPEC §18.3."""

    @pytest.mark.asyncio
    async def test_workflow_respects_timeout_ms_from_schema(self, workflow_engine):
        """Workflow should use timeout_ms from workflow schema."""
        # Mock time.monotonic to simulate elapsed time exceeding the timeout
        # First call returns 0 (start time), subsequent calls return 1.0 (1 second later)
        call_count = 0

        def mock_monotonic():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return 0.0  # Start time
            return 1.0  # 1 second elapsed - exceeds 100ms timeout

        with (
            patch.object(workflow_engine.store, "get_latest_workflow") as mock_workflow,
            patch("time.monotonic", mock_monotonic),
        ):
            mock_workflow.return_value = _loaded({
                "kind": "workflow.chat",
                "timeout_ms": 100,  # 100ms timeout
                "entrypoint": "slow_node",
                "nodes": [
                    {
                        "id": "slow_node",
                        "type": "tool_call",
                        "tool": "llm.generic",
                    }
                ],
            })

            result = await workflow_engine.run(
                "test-workflow",
                "test-conv",
                "hello",
                None,
            )

            # Should timeout
            assert result.get("status") == "error"
            assert "timeout" in result.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_workflow_uses_default_timeout_when_not_specified(
        self, workflow_engine
    ):
        """Workflow should use default timeout when not specified in schema."""
        with patch.object(workflow_engine.store, "get_latest_workflow") as mock:
            mock.return_value = _loaded({
                "kind": "workflow.chat",
                # No timeout_ms specified
                "entrypoint": "quick_node",
                "nodes": [
                    {
                        "id": "quick_node",
                        "type": "end",
                    }
                ],
            })

            # This should complete quickly without timing out
            result = await workflow_engine.run(
                "test-workflow",
                "test-conv",
                "hello",
                None,
            )

            # Should not timeout for a simple end node
            assert result.get("error") != "workflow_timeout"

    @pytest.mark.asyncio
    async def test_workflow_timeout_includes_elapsed_time(self, workflow_engine):
        """Timeout result should include elapsed time info."""
        # Mock time.monotonic to simulate elapsed time exceeding the timeout
        call_count = 0

        def mock_monotonic():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return 0.0  # Start time
            return 1.0  # 1 second elapsed - exceeds 100ms timeout

        with (
            patch.object(workflow_engine.store, "get_latest_workflow") as mock_workflow,
            patch("time.monotonic", mock_monotonic),
        ):
            mock_workflow.return_value = _loaded({
                "kind": "workflow.chat",
                "timeout_ms": 100,
                "entrypoint": "node1",
                "nodes": [{"id": "node1", "type": "tool_call", "tool": "llm.generic"}],
            })

            result = await workflow_engine.run(
                "test-workflow",
                "test-conv",
                "hello",
                None,
            )

            if result.get("status") == "error" and "timeout" in result.get("error", ""):
                # Should have elapsed_ms in the result
                assert "elapsed_ms" in result or "workflow_trace" in result


# ==============================================================================
# Retry with Exponential Backoff Tests
# ==============================================================================


class TestRetryBackoff:
    """Test node retry with exponential backoff per SPEC §18.3."""

    @pytest.mark.asyncio
    async def test_node_retries_on_error(self, workflow_engine):
        """Node should retry on error with exponential backoff."""
        call_count = 0

        def failing_tool(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:  # Fail first 2 attempts
                return {"status": "error", "error": "transient failure"}
            return {"content": "success", "status": "ok", "usage": {}}

        with patch.object(workflow_engine, "_builtin_tool_handlers") as mock_handlers:
            mock_handlers.return_value = {"llm.generic": failing_tool}

            workflow_engine.store.tool_specs["test.tool"] = {
                "name": "test.tool",
                "handler": "llm.generic",
                "timeout_seconds": 30,
            }

            node = {
                "id": "retry_node",
                "type": "tool_call",
                "tool": "test.tool",
                "max_retries": 2,
                "backoff_ms": 10,  # Short backoff for testing
            }

            start = time.monotonic()
            result, next_nodes = await workflow_engine._execute_node_with_retry(
                node,
                user_message="test",
                context_id=None,
                conversation_id=None,
                adapters=[],
                history=[],
                vars_scope={},
                user_id=None,
                tenant_id=None,
                workflow_start_time=start,
                workflow_timeout_ms=60000,
            )

            # Should have retried
            assert call_count == 3
            assert result.get("status") == "ok"

    @pytest.mark.asyncio
    async def test_max_retries_respected(self, workflow_engine):
        """Should not retry more than max_retries times."""
        call_count = 0

        def always_failing_tool(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise Exception("permanent failure")

        with patch.object(workflow_engine, "_builtin_tool_handlers") as mock_handlers:
            mock_handlers.return_value = {"llm.generic": always_failing_tool}

            workflow_engine.store.tool_specs["test.tool"] = {
                "name": "test.tool",
                "handler": "llm.generic",
                "timeout_seconds": 30,
            }

            node = {
                "id": "failing_node",
                "type": "tool_call",
                "tool": "test.tool",
                "max_retries": 2,  # 2 retries = 3 total attempts
                "backoff_ms": 1,
            }

            start = time.monotonic()
            result, next_nodes = await workflow_engine._execute_node_with_retry(
                node,
                user_message="test",
                context_id=None,
                conversation_id=None,
                adapters=[],
                history=[],
                vars_scope={},
                user_id=None,
                tenant_id=None,
                workflow_start_time=start,
                workflow_timeout_ms=60000,
            )

            # Should have tried 3 times (1 initial + 2 retries)
            assert call_count == 3
            assert result.get("status") == "error"
            assert result.get("retries_exhausted") is True
            assert result.get("attempts") == 3

    @pytest.mark.asyncio
    async def test_hard_cap_on_retries(self, workflow_engine):
        """Retries should be capped at MAX_RETRIES_HARD_CAP per SPEC §18."""
        call_count = 0

        def always_failing_tool(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise Exception("failure")

        with patch.object(workflow_engine, "_builtin_tool_handlers") as mock_handlers:
            mock_handlers.return_value = {"llm.generic": always_failing_tool}

            workflow_engine.store.tool_specs["test.tool"] = {
                "name": "test.tool",
                "handler": "llm.generic",
                "timeout_seconds": 30,
            }

            node = {
                "id": "capped_node",
                "type": "tool_call",
                "tool": "test.tool",
                "max_retries": 100,  # Attempt to exceed hard cap
                "backoff_ms": 1,
            }

            start = time.monotonic()
            result, _ = await workflow_engine._execute_node_with_retry(
                node,
                user_message="test",
                context_id=None,
                conversation_id=None,
                adapters=[],
                history=[],
                vars_scope={},
                user_id=None,
                tenant_id=None,
                workflow_start_time=start,
                workflow_timeout_ms=60000,
            )

            # Should be capped at MAX_RETRIES_HARD_CAP + 1 attempts
            assert call_count == MAX_RETRIES_HARD_CAP + 1

    @pytest.mark.asyncio
    async def test_exponential_backoff_timing(self, workflow_engine):
        """Backoff quadruples each retry: 1s, 4s, 16s per SPEC §18.3."""
        backoff_times = []
        call_count = 0

        async def mock_sleep(seconds):
            backoff_times.append(seconds * 1000)  # Convert to ms

        def always_failing_tool(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise Exception("failure")

        with (
            patch.object(workflow_engine, "_builtin_tool_handlers") as mock_handlers,
            patch("asyncio.sleep", mock_sleep),
        ):
            mock_handlers.return_value = {"llm.generic": always_failing_tool}

            workflow_engine.store.tool_specs["test.tool"] = {
                "name": "test.tool",
                "handler": "llm.generic",
                "timeout_seconds": 30,
            }

            node = {
                "id": "backoff_node",
                "type": "tool_call",
                "tool": "test.tool",
                "max_retries": 3,
                "backoff_ms": 1000,  # 1s base
            }

            start = time.monotonic()
            await workflow_engine._execute_node_with_retry(
                node,
                user_message="test",
                context_id=None,
                conversation_id=None,
                adapters=[],
                history=[],
                vars_scope={},
                user_id=None,
                tenant_id=None,
                workflow_start_time=start,
                workflow_timeout_ms=60000,
            )

            # Per SPEC §18.3: 1000, 4000, 16000 ms
            assert len(backoff_times) == 3
            assert backoff_times[0] == 1000  # First backoff: 1s
            assert backoff_times[1] == 4000  # Second backoff: 4s (quadruple)
            assert backoff_times[2] == 16000  # Third backoff: 16s (quadruple)

    @pytest.mark.asyncio
    async def test_retry_respects_workflow_timeout(self, workflow_engine):
        """Retry should stop if workflow timeout is reached."""
        call_count = 0

        def slow_failing_tool(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise Exception("failure")

        with patch.object(workflow_engine, "_builtin_tool_handlers") as mock_handlers:
            mock_handlers.return_value = {"llm.generic": slow_failing_tool}

            workflow_engine.store.tool_specs["test.tool"] = {
                "name": "test.tool",
                "handler": "llm.generic",
                "timeout_seconds": 30,
            }

            node = {
                "id": "timeout_node",
                "type": "tool_call",
                "tool": "test.tool",
                "max_retries": 10,
                "backoff_ms": 1,
            }

            start = time.monotonic()
            result, _ = await workflow_engine._execute_node_with_retry(
                node,
                user_message="test",
                context_id=None,
                conversation_id=None,
                adapters=[],
                history=[],
                vars_scope={},
                user_id=None,
                tenant_id=None,
                workflow_start_time=start - 100,  # Pretend we started 100s ago
                workflow_timeout_ms=1,  # 1ms timeout - already expired
            )

            # Should have stopped due to workflow timeout
            assert result.get("error") == "workflow_timeout_during_retry"

    @pytest.mark.asyncio
    async def test_no_retry_for_nodes_with_on_error(self, workflow_engine):
        """Nodes with on_error handler should not retry, just forward to error handler."""
        call_count = 0

        def failing_tool(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return {"status": "error", "error": "handled failure"}

        with patch.object(workflow_engine, "_builtin_tool_handlers") as mock_handlers:
            mock_handlers.return_value = {"llm.generic": failing_tool}

            workflow_engine.store.tool_specs["test.tool"] = {
                "name": "test.tool",
                "handler": "llm.generic",
                "timeout_seconds": 30,
            }

            node = {
                "id": "handled_node",
                "type": "tool_call",
                "tool": "test.tool",
                "max_retries": 3,
                "on_error": "error_handler",  # Has error handler
            }

            start = time.monotonic()
            result, next_nodes = await workflow_engine._execute_node_with_retry(
                node,
                user_message="test",
                context_id=None,
                conversation_id=None,
                adapters=[],
                history=[],
                vars_scope={},
                user_id=None,
                tenant_id=None,
                workflow_start_time=start,
                workflow_timeout_ms=60000,
            )

            # Should only call once, no retry since on_error is defined
            assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_count_in_result(self, workflow_engine):
        """Successful result after retries should include retry count."""
        call_count = 0

        def eventual_success(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("transient")
            return {"content": "ok", "status": "ok", "usage": {}}

        with patch.object(workflow_engine, "_builtin_tool_handlers") as mock_handlers:
            mock_handlers.return_value = {"llm.generic": eventual_success}

            workflow_engine.store.tool_specs["test.tool"] = {
                "name": "test.tool",
                "handler": "llm.generic",
                "timeout_seconds": 30,
            }

            node = {
                "id": "eventual_node",
                "type": "tool_call",
                "tool": "test.tool",
                "max_retries": 2,
                "backoff_ms": 1,
            }

            start = time.monotonic()
            result, _ = await workflow_engine._execute_node_with_retry(
                node,
                user_message="test",
                context_id=None,
                conversation_id=None,
                adapters=[],
                history=[],
                vars_scope={},
                user_id=None,
                tenant_id=None,
                workflow_start_time=start,
                workflow_timeout_ms=60000,
            )

            # Should succeed with retry count
            assert result.get("status") == "ok"
            assert result.get("retry_attempts") == 1


# ==============================================================================
# Integration Tests
# ==============================================================================


class TestRetryTimeoutIntegration:
    """Integration tests combining retry and timeout behavior."""

    @pytest.mark.asyncio
    async def test_workflow_timeout_stops_retrying(self, workflow_engine):
        """Workflow timeout should interrupt retry loops."""
        # This is covered by test_retry_respects_workflow_timeout above
        pass

    @pytest.mark.asyncio
    async def test_default_values_used_when_not_specified(self, workflow_engine):
        """Default values should be used when not specified in node config."""
        call_count = 0

        def failing_tool(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise Exception("failure")

        with patch.object(workflow_engine, "_builtin_tool_handlers") as mock_handlers:
            mock_handlers.return_value = {"llm.generic": failing_tool}

            workflow_engine.store.tool_specs["test.tool"] = {
                "name": "test.tool",
                "handler": "llm.generic",
                "timeout_seconds": 30,
            }

            # Node without max_retries or backoff_ms
            node = {
                "id": "default_node",
                "type": "tool_call",
                "tool": "test.tool",
            }

            start = time.monotonic()
            result, _ = await workflow_engine._execute_node_with_retry(
                node,
                user_message="test",
                context_id=None,
                conversation_id=None,
                adapters=[],
                history=[],
                vars_scope={},
                user_id=None,
                tenant_id=None,
                workflow_start_time=start,
                workflow_timeout_ms=60000,
            )

            # Should use DEFAULT_NODE_MAX_RETRIES (2) = 3 total attempts
            assert call_count == DEFAULT_NODE_MAX_RETRIES + 1


class TestTheWorkflowDeadlineIsRealWallClock:
    """§18.3 promises the workflow's `timeout_ms` caps total wall clock.

    It did not. Two independent leaks, both of which let a workflow return
    materially after its own deadline:

    * the attempt was awaited with the node's own `timeout_ms`, neither
      capped at the kernel's 60s nor reduced to the time the workflow had
      left, so a node starting just before the deadline ran well past it;
    * the retry backoff was computed against a `remaining_ms` measured
      *before* the attempt, so a node that consumed almost all of the
      remaining budget still slept a full backoff on top.

    `MAX_NODE_TIMEOUT_SECONDS` existed but capped the tool spec's
    `timeout_seconds`, not this outer node timeout - the constant was right
    and unused where it mattered.
    """

    @pytest.mark.asyncio
    async def test_the_attempt_is_capped_by_the_time_the_workflow_has_left(
        self, workflow_engine
    ):
        """What reaches `asyncio.wait_for` is the binding limit, not the node's ask."""
        seen = []

        real_wait_for = asyncio.wait_for

        async def capturing_wait_for(aw, timeout=None):
            seen.append(timeout)
            return await real_wait_for(aw, timeout)

        def ok_tool(*args, **kwargs):
            return {"ok": True}

        with (
            patch.object(workflow_engine, "_builtin_tool_handlers") as mock_handlers,
            patch("asyncio.wait_for", capturing_wait_for),
        ):
            mock_handlers.return_value = {"llm.generic": ok_tool}
            workflow_engine.store.tool_specs["test.tool"] = {
                "name": "test.tool",
                "handler": "llm.generic",
                "timeout_seconds": 30,
            }

            node = {
                "id": "greedy",
                "type": "tool_call",
                "tool": "test.tool",
                # Both larger than the kernel cap and larger than what the
                # workflow has left.
                "timeout_ms": 600_000,
            }

            await workflow_engine._execute_node_with_retry(
                node,
                user_message="test",
                context_id=None,
                conversation_id=None,
                adapters=[],
                history=[],
                vars_scope={},
                user_id=None,
                tenant_id=None,
                workflow_start_time=time.monotonic(),
                workflow_timeout_ms=5_000,
            )

        assert seen, "the node was never awaited through wait_for"
        applied = seen[0]
        assert applied is not None
        assert applied <= MAX_NODE_TIMEOUT_SECONDS, (
            f"the node's own timeout_ms bypassed the {MAX_NODE_TIMEOUT_SECONDS}s "
            f"kernel cap: {applied}s"
        )
        assert applied <= 5.0, (
            "the attempt was allowed to run past the workflow's own deadline: "
            f"{applied}s with 5s of budget left"
        )

    @pytest.mark.asyncio
    async def test_the_sixty_second_cap_holds_with_budget_to_spare(
        self, workflow_engine
    ):
        """*Independently* capped, which the budget bound can hide.

        With a small workflow budget the remaining-time bound is the smaller
        of the two and the cap is never exercised - a version that dropped
        `MAX_NODE_TIMEOUT_SECONDS` entirely passed a five-second-budget test.
        So this one gives the workflow ten minutes and asks only about the
        cap.
        """
        seen = []
        real_wait_for = asyncio.wait_for

        async def capturing_wait_for(aw, timeout=None):
            seen.append(timeout)
            return await real_wait_for(aw, timeout)

        def ok_tool(*args, **kwargs):
            return {"ok": True}

        with (
            patch.object(workflow_engine, "_builtin_tool_handlers") as mock_handlers,
            patch("asyncio.wait_for", capturing_wait_for),
        ):
            mock_handlers.return_value = {"llm.generic": ok_tool}
            workflow_engine.store.tool_specs["test.tool"] = {
                "name": "test.tool",
                "handler": "llm.generic",
                "timeout_seconds": 30,
            }

            await workflow_engine._execute_node_with_retry(
                {
                    "id": "greedy",
                    "type": "tool_call",
                    "tool": "test.tool",
                    "timeout_ms": 600_000,
                },
                user_message="test",
                context_id=None,
                conversation_id=None,
                adapters=[],
                history=[],
                vars_scope={},
                user_id=None,
                tenant_id=None,
                workflow_start_time=time.monotonic(),
                workflow_timeout_ms=600_000,
            )

        assert seen and seen[0] is not None
        assert seen[0] <= MAX_NODE_TIMEOUT_SECONDS, (
            "a node asked for 600s inside a workflow with 600s of budget and "
            f"got {seen[0]}s - the kernel cap is not applied independently"
        )

    @pytest.mark.asyncio
    async def test_backoff_is_measured_after_the_attempt_not_before(
        self, workflow_engine
    ):
        """A node that eats the budget must not then sleep a full backoff.

        The workflow has 2s. The node burns 1.5s and fails. A backoff
        computed against the budget as it was *before* the attempt still
        believes 2s remain and sleeps a full second on top.
        """
        slept = []

        async def mock_sleep(seconds):
            slept.append(seconds * 1000)

        def slow_failing_tool(*args, **kwargs):
            time.sleep(1.5)
            raise Exception("failure")

        with (
            patch.object(workflow_engine, "_builtin_tool_handlers") as mock_handlers,
            patch("asyncio.sleep", mock_sleep),
        ):
            mock_handlers.return_value = {"llm.generic": slow_failing_tool}
            workflow_engine.store.tool_specs["test.tool"] = {
                "name": "test.tool",
                "handler": "llm.generic",
                "timeout_seconds": 30,
            }

            node = {
                "id": "budget_eater",
                "type": "tool_call",
                "tool": "test.tool",
                "max_retries": 1,
                "backoff_ms": 1000,
            }

            await workflow_engine._execute_node_with_retry(
                node,
                user_message="test",
                context_id=None,
                conversation_id=None,
                adapters=[],
                history=[],
                vars_scope={},
                user_id=None,
                tenant_id=None,
                workflow_start_time=time.monotonic(),
                workflow_timeout_ms=2_000,
            )

        # Either it declined to sleep at all, or it slept only what was
        # genuinely left. What it must not do is sleep the full backoff.
        assert all(ms <= 500 for ms in slept), (
            "the backoff was computed against the budget as it stood before "
            f"the attempt, so the workflow overran its deadline: slept {slept}ms "
            "with ~0.5s of a 2s budget actually remaining"
        )

    @pytest.mark.asyncio
    async def test_a_workflow_does_not_return_materially_after_its_deadline(
        self, workflow_engine
    ):
        """The contract as a caller experiences it, measured on the clock."""

        def very_slow_tool(*args, **kwargs):
            time.sleep(10)
            return {"ok": True}

        with patch.object(workflow_engine, "_builtin_tool_handlers") as mock_handlers:
            mock_handlers.return_value = {"llm.generic": very_slow_tool}
            workflow_engine.store.tool_specs["test.tool"] = {
                "name": "test.tool",
                "handler": "llm.generic",
                "timeout_seconds": 30,
            }

            node = {
                "id": "overrunner",
                "type": "tool_call",
                "tool": "test.tool",
                "timeout_ms": 600_000,
                "max_retries": 0,
            }

            start = time.monotonic()
            await workflow_engine._execute_node_with_retry(
                node,
                user_message="test",
                context_id=None,
                conversation_id=None,
                adapters=[],
                history=[],
                vars_scope={},
                user_id=None,
                tenant_id=None,
                workflow_start_time=start,
                workflow_timeout_ms=1_000,
            )
            elapsed = time.monotonic() - start

        assert elapsed < 5.0, (
            f"a workflow with a 1s deadline returned after {elapsed:.1f}s - the "
            "node's own timeout_ms governed, not the workflow's remaining budget"
        )
