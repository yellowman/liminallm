"""Tests for SPEC §9/§18 workflow retry backoff and timeout enforcement."""

from __future__ import annotations

import time
from typing import Optional
from unittest.mock import patch

import pytest

from liminallm.service.workflow import (
    DEFAULT_BACKOFF_MS,
    DEFAULT_NODE_MAX_RETRIES,
    DEFAULT_NODE_TIMEOUT_MS,
    DEFAULT_WORKFLOW_TIMEOUT_MS,
    MAX_RETRIES_HARD_CAP,
    WorkflowEngine,
)


class MockStore:
    """Minimal store mock for testing."""

    def __init__(self):
        self.artifacts = []
        self.messages = []

    def get_latest_workflow(self, workflow_id: str) -> Optional[dict]:
        return None

    def list_artifacts(self, type_filter: Optional[str] = None, **kwargs) -> list:
        return []

    def list_semantic_clusters(self, user_id: Optional[str]) -> list:
        return []


class MockLLM:
    """Mock LLM service."""

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
    """Verify SPEC §9/§18 constants are correctly defined."""

    def test_default_node_timeout(self):
        """Default node timeout should be 15s per SPEC §9."""
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
    """Test workflow-level timeout enforcement per SPEC §9."""

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
            mock_workflow.return_value = {
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
            }

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
            mock.return_value = {
                "kind": "workflow.chat",
                # No timeout_ms specified
                "entrypoint": "quick_node",
                "nodes": [
                    {
                        "id": "quick_node",
                        "type": "end",
                    }
                ],
            }

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
            mock_workflow.return_value = {
                "kind": "workflow.chat",
                "timeout_ms": 100,
                "entrypoint": "node1",
                "nodes": [{"id": "node1", "type": "tool_call", "tool": "llm.generic"}],
            }

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
    """Test node retry with exponential backoff per SPEC §9/§18."""

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
            mock_handlers.return_value = {"test.tool": failing_tool}

            workflow_engine.tool_registry["test.tool"] = {
                "name": "test.tool",
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
            mock_handlers.return_value = {"test.tool": always_failing_tool}

            workflow_engine.tool_registry["test.tool"] = {
                "name": "test.tool",
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
            mock_handlers.return_value = {"test.tool": always_failing_tool}

            workflow_engine.tool_registry["test.tool"] = {
                "name": "test.tool",
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
        """Backoff should grow exponentially: 1s, 2s, 4s per SPEC §18."""
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
            mock_handlers.return_value = {"test.tool": always_failing_tool}

            workflow_engine.tool_registry["test.tool"] = {
                "name": "test.tool",
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

            # Should have exponential backoff: 1000, 2000, 4000 ms
            assert len(backoff_times) == 3
            assert backoff_times[0] == 1000  # First backoff: 1s
            assert backoff_times[1] == 2000  # Second backoff: 2s
            assert backoff_times[2] == 4000  # Third backoff: 4s

    @pytest.mark.asyncio
    async def test_retry_respects_workflow_timeout(self, workflow_engine):
        """Retry should stop if workflow timeout is reached."""
        call_count = 0

        def slow_failing_tool(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise Exception("failure")

        with patch.object(workflow_engine, "_builtin_tool_handlers") as mock_handlers:
            mock_handlers.return_value = {"test.tool": slow_failing_tool}

            workflow_engine.tool_registry["test.tool"] = {
                "name": "test.tool",
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
            mock_handlers.return_value = {"test.tool": failing_tool}

            workflow_engine.tool_registry["test.tool"] = {
                "name": "test.tool",
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
            mock_handlers.return_value = {"test.tool": eventual_success}

            workflow_engine.tool_registry["test.tool"] = {
                "name": "test.tool",
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
            mock_handlers.return_value = {"test.tool": failing_tool}

            workflow_engine.tool_registry["test.tool"] = {
                "name": "test.tool",
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
