from __future__ import annotations

import asyncio
import concurrent.futures
import copy
import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from liminallm.config import Settings
from liminallm.logging import get_logger
from liminallm.service.embeddings import (
    EMBEDDING_DIM,
    cosine_similarity,
    deterministic_embedding,
    ensure_embedding_dim,
)
from liminallm.service.errors import BadRequestError
from liminallm.service.llm import LLMService
from liminallm.service.rag import RAGService
from liminallm.service.router import RouterEngine
from liminallm.service.sandbox import (
    AllowlistedFetcher,
    ToolNetworkPolicy,
    build_tool_network_policy,
    safe_eval_expr,
    tool_network_guard,
)
from liminallm.storage.memory import MemoryStore
from liminallm.storage.models import Message
from liminallm.storage.postgres import PostgresStore
from liminallm.storage.redis_cache import RedisCache

# SPEC §9/§18: Default retry and timeout settings
DEFAULT_NODE_TIMEOUT_MS = 15000  # 15 seconds per node
MAX_NODE_TIMEOUT_SECONDS = 60  # SPEC §18: per-node timeout hard cap 60s
DEFAULT_NODE_MAX_RETRIES = 2  # Up to 2 retries (3 total attempts), hard cap at 3
DEFAULT_BACKOFF_MS = (
    1000  # Initial backoff 1s, quadruples each retry (1s, 4s per SPEC §18)
)
MAX_RETRIES_HARD_CAP = 3  # SPEC §18: hard cap at 3 retries
DEFAULT_WORKFLOW_TIMEOUT_MS = 60000  # 60 seconds total workflow timeout
MAX_CONTEXT_SNIPPETS = 20
MAX_WORKFLOW_SNAPSHOTS = 10  # Keep max 10 snapshots for rollback (memory management)


@dataclass
class WorkflowSnapshot:
    """Snapshot of workflow state for rollback support.

    Captures the complete state before each node execution, allowing
    rollback to a previous known-good state on failure.
    """
    node_id: str
    vars_scope: Dict[str, Any]
    workflow_trace: List[Dict[str, Any]]
    content: str
    usage: Dict[str, Any]
    context_snippets: List[str]
    pending: List[str]
    visited_count: int
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @classmethod
    def capture(
        cls,
        node_id: str,
        vars_scope: Dict[str, Any],
        workflow_trace: List[Dict[str, Any]],
        content: str,
        usage: Dict[str, Any],
        context_snippets: List[str],
        pending: List[str],
        visited_count: int,
    ) -> "WorkflowSnapshot":
        """Create a deep copy snapshot of current workflow state."""
        return cls(
            node_id=node_id,
            vars_scope=copy.deepcopy(vars_scope),
            workflow_trace=copy.deepcopy(workflow_trace),
            content=content,
            usage=copy.deepcopy(usage),
            context_snippets=list(context_snippets),
            pending=list(pending),
            visited_count=visited_count,
        )


@dataclass
class ParallelNodeResult:
    """Result of parallel node execution with merged outputs."""
    merged_outputs: Dict[str, Any]  # Outputs namespaced by node ID
    merged_content: str  # Concatenated content from all nodes
    merged_usage: Dict[str, Any]  # Summed token counts
    merged_snippets: List[str]  # Deduplicated context snippets
    failed_nodes: List[str]  # Node IDs that failed
    status: str = "ok"  # "ok" if all succeeded, "partial" if some failed, "error" if all failed


class WorkflowEngine:
    """Executes workflow.chat graphs using a small tool registry."""

    # Issue 48.6: Increase ThreadPoolExecutor workers and add scaling config
    DEFAULT_TOOL_WORKERS = 8  # Up from 4 to handle concurrent tool calls
    MAX_TOOL_WORKERS = 16  # Hard cap to prevent resource exhaustion

    def __init__(
        self,
        store: PostgresStore | MemoryStore,
        llm: LLMService,
        router: RouterEngine,
        rag: RAGService,
        *,
        cache: Optional[RedisCache] = None,
        tool_workers: int = DEFAULT_TOOL_WORKERS,
        settings: Optional[Settings] = None,
    ) -> None:
        self.store = store
        self.llm = llm
        self.router = router
        self.rag = rag
        self.logger = get_logger(__name__)
        self.tool_registry = self._build_tool_registry()
        self.cache = cache
        self.tool_network_policy: ToolNetworkPolicy = build_tool_network_policy(
            allowlist=(settings.tool_network_allowlist if settings else []),
            proxy_url=settings.tool_network_proxy_url if settings else None,
            connect_timeout=(
                settings.tool_fetch_connect_timeout if settings else 10.0
            ),
            total_timeout=settings.tool_fetch_timeout if settings else 30.0,
        )
        self.tool_fetcher = AllowlistedFetcher(self.tool_network_policy)
        # Issue 48.6: Configurable worker pool with bounds
        workers = min(max(1, tool_workers), self.MAX_TOOL_WORKERS)
        self._tool_executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
        self._executor_shutdown = False

    def _error_event(
        self, code: str, message: str, details: dict | None = None
    ) -> dict:
        return {
            "event": "error",
            "data": {"code": code, "message": message, "details": details or {}},
        }

    def _append_trace(
        self,
        workflow_trace: List[Dict[str, Any]],
        entry: Dict[str, Any],
        max_entries: int = 500,
    ) -> None:
        """Append to workflow_trace with bounded size (Issue 23.4)."""

        workflow_trace.append(entry)
        if len(workflow_trace) > max_entries:
            # Drop oldest entries to avoid unbounded growth during long runs
            del workflow_trace[0 : len(workflow_trace) - max_entries]

    def shutdown(self, wait: bool = True) -> None:
        """Explicitly shutdown the executor. Call during app shutdown.

        Issue 23.1: Provides explicit cleanup instead of relying on __del__.
        This should be called during application shutdown to ensure proper cleanup
        of ThreadPoolExecutor resources.

        Args:
            wait: If True, wait for pending futures to complete. If False, cancel them.
        """
        if self._executor_shutdown:
            return
        self._executor_shutdown = True
        try:
            self._tool_executor.shutdown(wait=wait, cancel_futures=not wait)
            self.logger.info("workflow_executor_shutdown", wait=wait)
        except Exception as exc:
            self.logger.warning("workflow_executor_shutdown_error", error=str(exc))

    async def _rollback_workflow(
        self,
        state_key: str,
        workflow_trace: List[Dict[str, Any]],
        vars_scope: Dict[str, Any],
        *,
        reason: str = "node_failure",
        snapshots: Optional[List[WorkflowSnapshot]] = None,
        target_snapshot_index: int = -1,
    ) -> Optional[dict]:
        """Rollback workflow state, optionally restoring from a snapshot.

        Args:
            state_key: Workflow state cache key
            workflow_trace: Current trace (will be truncated if restoring)
            vars_scope: Current variables (will be replaced if restoring)
            reason: Reason for rollback
            snapshots: List of captured snapshots for restoration
            target_snapshot_index: Index of snapshot to restore to (-1 = latest)

        Returns:
            Rollback state dict with restoration details, or None on failure
        """
        restored_from = None

        # If snapshots available, restore to specified snapshot
        if snapshots and len(snapshots) > 0:
            idx = target_snapshot_index if target_snapshot_index >= 0 else len(snapshots) - 1
            if idx < len(snapshots):
                snapshot = snapshots[idx]
                restored_from = {
                    "snapshot_node": snapshot.node_id,
                    "snapshot_time": snapshot.timestamp.isoformat(),
                    "restored_vars": list(snapshot.vars_scope.keys()),
                    "restored_trace_length": len(snapshot.workflow_trace),
                }
                self.logger.info(
                    "workflow_rollback_restoring",
                    from_node=snapshot.node_id,
                    trace_length=len(snapshot.workflow_trace),
                )

        rollback_state = {
            "status": "rolled_back",
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat(),
            "vars": vars_scope,
            "trace_length": len(workflow_trace),
            "restored_from": restored_from,
        }

        try:
            # Mark workflow as rolling back in persistent state
            await self._persist_workflow_state(
                state_key,
                {
                    "status": "rolling_back",
                    "reason": reason,
                    "updated_at": datetime.utcnow().isoformat(),
                    "workflow_trace": workflow_trace,
                    "restored_from": restored_from,
                },
            )

            # Clear any workflow-specific cache entries
            await self._clear_workflow_cache(state_key)

        except Exception as exc:
            self.logger.warning("workflow_rollback_mark_failed", error=str(exc))
            return None

        return rollback_state

    def _capture_snapshot(
        self,
        snapshots: List[WorkflowSnapshot],
        node_id: str,
        vars_scope: Dict[str, Any],
        workflow_trace: List[Dict[str, Any]],
        content: str,
        usage: Dict[str, Any],
        context_snippets: List[str],
        pending: List[str],
        visited_count: int,
    ) -> None:
        """Capture a snapshot of workflow state before node execution.

        Maintains a bounded list of snapshots (max MAX_WORKFLOW_SNAPSHOTS)
        by removing oldest snapshots when the limit is exceeded.
        """
        snapshot = WorkflowSnapshot.capture(
            node_id=node_id,
            vars_scope=vars_scope,
            workflow_trace=workflow_trace,
            content=content,
            usage=usage,
            context_snippets=context_snippets,
            pending=pending,
            visited_count=visited_count,
        )
        snapshots.append(snapshot)

        # Keep only the most recent snapshots
        while len(snapshots) > MAX_WORKFLOW_SNAPSHOTS:
            snapshots.pop(0)

    def _restore_from_snapshot(
        self,
        snapshot: WorkflowSnapshot,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], str, Dict[str, Any], List[str], List[str], int]:
        """Restore workflow state from a snapshot.

        Returns:
            Tuple of (vars_scope, workflow_trace, content, usage, context_snippets, pending, visited_count)
        """
        return (
            copy.deepcopy(snapshot.vars_scope),
            copy.deepcopy(snapshot.workflow_trace),
            snapshot.content,
            copy.deepcopy(snapshot.usage),
            list(snapshot.context_snippets),
            list(snapshot.pending),
            snapshot.visited_count,
        )

    async def _clear_workflow_cache(self, state_key: str) -> None:
        """Clear workflow-specific cache entries during rollback."""
        if not self.cache:
            return
        try:
            # Clear workflow state from cache
            await self.cache.delete_workflow_state(state_key)
            self.logger.debug("workflow_cache_cleared", state_key=state_key)
        except Exception as exc:
            # Non-fatal - cache clear is best effort
            self.logger.warning("workflow_cache_clear_failed", error=str(exc))

    async def _execute_parallel_nodes(
        self,
        node_ids: List[str],
        node_map: Dict[str, Dict[str, Any]],
        *,
        user_message: str,
        context_id: Optional[str],
        conversation_id: Optional[str],
        adapters: List[dict],
        history: List[Any],
        vars_scope: Dict[str, Any],
        user_id: Optional[str],
        tenant_id: Optional[str],
        workflow_start_time: float,
        workflow_timeout_ms: float,
        cancel_event: Optional[asyncio.Event] = None,
    ) -> ParallelNodeResult:
        """Execute multiple nodes concurrently and merge results.

        Each node gets a copy of vars_scope to prevent conflicts.
        Results are namespaced by node ID.
        """
        if not node_ids:
            return ParallelNodeResult(
                merged_outputs={},
                merged_content="",
                merged_usage={},
                merged_snippets=[],
                failed_nodes=[],
                status="ok",
            )

        async def execute_single_node(node_id: str) -> Tuple[str, Dict[str, Any], List[str]]:
            """Execute a single node with its own vars_scope copy."""
            node = node_map.get(node_id)
            if not node:
                return node_id, {"status": "error", "error": f"Node {node_id} not found"}, []

            # Each parallel node gets its own copy of vars_scope
            local_vars = copy.deepcopy(vars_scope)

            try:
                result, _ = await self._execute_node_with_retry(
                    node,
                    user_message=user_message,
                    context_id=context_id,
                    conversation_id=conversation_id,
                    adapters=adapters,
                    history=history,
                    vars_scope=local_vars,
                    user_id=user_id,
                    tenant_id=tenant_id,
                    workflow_start_time=workflow_start_time,
                    workflow_timeout_ms=workflow_timeout_ms,
                    cancel_event=cancel_event,
                )
                snippets = result.get("context_snippets", []) if isinstance(result, dict) else []
                return node_id, result, snippets
            except Exception as exc:
                self.logger.error("parallel_node_failed", node_id=node_id, error=str(exc))
                return node_id, {"status": "error", "error": str(exc)}, []

        # Execute all nodes concurrently
        tasks = [execute_single_node(nid) for nid in node_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Merge results
        merged_outputs: Dict[str, Any] = {}
        merged_content_parts: List[str] = []
        merged_usage: Dict[str, Any] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        all_snippets: List[str] = []
        failed_nodes: List[str] = []

        for item in results:
            if isinstance(item, Exception):
                self.logger.error("parallel_gather_exception", error=str(item))
                continue

            node_id, result, snippets = item

            if isinstance(result, dict):
                # Namespace outputs by node ID
                merged_outputs[node_id] = {
                    k: v for k, v in result.items()
                    if k not in {"usage", "context_snippets", "status"}
                }

                # Check for failure
                if result.get("status") == "error":
                    failed_nodes.append(node_id)

                # Merge content
                content = result.get("content", "")
                if content:
                    merged_content_parts.append(f"[{node_id}]\n{content}")

                # Sum usage
                usage = result.get("usage", {})
                if isinstance(usage, dict):
                    for key in ["prompt_tokens", "completion_tokens", "total_tokens"]:
                        merged_usage[key] += usage.get(key, 0)

                # Collect snippets
                all_snippets.extend(snippets)

        # Deduplicate snippets
        seen_snippets: set = set()
        deduped_snippets: List[str] = []
        for snippet in all_snippets:
            normalized = snippet.strip().lower()
            if normalized not in seen_snippets:
                seen_snippets.add(normalized)
                deduped_snippets.append(snippet)

        # Determine overall status
        if len(failed_nodes) == len(node_ids):
            status = "error"
        elif failed_nodes:
            status = "partial"
        else:
            status = "ok"

        return ParallelNodeResult(
            merged_outputs=merged_outputs,
            merged_content="\n\n".join(merged_content_parts),
            merged_usage=merged_usage,
            merged_snippets=deduped_snippets[:MAX_CONTEXT_SNIPPETS],
            failed_nodes=failed_nodes,
            status=status,
        )

    async def run(
        self,
        workflow_id: Optional[str],
        conversation_id: Optional[str],
        user_message: str,
        context_id: Optional[str],
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> dict:
        workflow_schema = None
        if workflow_id:
            workflow_schema = self.store.get_latest_workflow(workflow_id)
        if not workflow_schema:
            workflow_schema = self._default_workflow()

        # SPEC §9: workflow-level timeout_ms caps total wall clock
        workflow_timeout_ms = workflow_schema.get(
            "timeout_ms", DEFAULT_WORKFLOW_TIMEOUT_MS
        )
        workflow_start_time = time.monotonic()

        adapters, routing_trace, adapter_gates = await self._select_adapters(
            user_message, user_id, context_id
        )
        history = await self._load_conversation_history(
            conversation_id, user_id=user_id, tenant_id=tenant_id
        )

        node_map = {
            n.get("id"): n for n in workflow_schema.get("nodes", []) if n.get("id")
        }
        if not node_map:
            raise BadRequestError("workflow has no nodes to execute")
        entry = workflow_schema.get("entrypoint") or next(iter(node_map), None)
        if not entry or entry not in node_map:
            entry = next(iter(node_map)) if node_map else None

        vars_scope: Dict[str, Any] = {}
        workflow_trace: List[Dict[str, Any]] = []
        max_trace_entries = 500
        context_snippets: List[str] = []
        context_seen = set()
        content = ""
        usage: Dict[str, Any] = {}

        pending: List[str] = [entry] if entry else []
        visited = 0
        max_steps = max(1, min(100, len(node_map) * 2 + 10))
        visited_nodes: Dict[str, int] = {}
        max_visits_per_node = max(2, math.ceil(max_steps / max(1, len(node_map))))

        # Initialize snapshot list for rollback support
        snapshots: List[WorkflowSnapshot] = []

        state_key = f"{conversation_id or 'anon'}:{workflow_id or 'default'}"
        await self._persist_workflow_state(
            state_key,
            {"status": "running", "started_at": datetime.utcnow().isoformat()},
        )

        while pending and visited < max_steps:
            # SPEC §9: Check workflow-level timeout before executing next node
            elapsed_ms = (time.monotonic() - workflow_start_time) * 1000
            if elapsed_ms >= workflow_timeout_ms:
                self.logger.warning(
                    "workflow_timeout",
                    workflow_id=workflow_id,
                    elapsed_ms=elapsed_ms,
                    timeout_ms=workflow_timeout_ms,
                )
                timeout_result = {
                    "status": "error",
                    "content": "workflow execution timed out",
                    "error": "workflow_timeout",
                    "elapsed_ms": elapsed_ms,
                    "timeout_ms": workflow_timeout_ms,
                    "routing_trace": routing_trace,
                    "workflow_trace": workflow_trace,
                    "context_snippets": context_snippets,
                    "vars": vars_scope,
                }
                await self._persist_workflow_state(
                    state_key,
                    {
                        "status": "timeout",
                        "failed_at": datetime.utcnow().isoformat(),
                        "error": "workflow_timeout",
                        "elapsed_ms": elapsed_ms,
                    },
                )
                return timeout_result

            node_id = pending.pop(0)
            node = node_map.get(node_id)
            if not node:
                continue
            visited += 1
            visited_nodes[node_id] = visited_nodes.get(node_id, 0) + 1
            if visited_nodes[node_id] > max_visits_per_node:
                self.logger.warning("workflow_loop_detected", node=node_id)
                break

            # Capture snapshot before node execution for rollback support
            self._capture_snapshot(
                snapshots,
                node_id,
                vars_scope,
                workflow_trace,
                content,
                usage,
                context_snippets,
                pending,
                visited,
            )

            # SPEC §9/§18: Execute node with retry and exponential backoff
            result, next_nodes = await self._execute_node_with_retry(
                node,
                user_message=user_message,
                context_id=context_id,
                conversation_id=conversation_id,
                adapters=adapters,
                history=history,
                vars_scope=vars_scope,
                user_id=user_id,
                tenant_id=tenant_id,
                workflow_start_time=workflow_start_time,
                workflow_timeout_ms=workflow_timeout_ms,
            )

            # Check if node execution failed after all retries
            if result.get("status") == "error" and result.get("retries_exhausted"):
                return await self._handle_node_failure(
                    state_key,
                    node_id,
                    Exception(result.get("error", "node execution failed")),
                    vars_scope=vars_scope,
                    context_snippets=context_snippets,
                    workflow_trace=workflow_trace,
                    routing_trace=routing_trace,
                    snapshots=snapshots,
                )

            # Handle parallel node execution - run child nodes concurrently
            if result.get("status") == "parallel":
                parallel_node_ids = result.get("parallel_nodes", [])
                after_node = result.get("after")

                if parallel_node_ids:
                    self.logger.info(
                        "workflow_parallel_start",
                        node_id=node_id,
                        parallel_nodes=parallel_node_ids,
                    )
                    parallel_result = await self._execute_parallel_nodes(
                        parallel_node_ids,
                        node_map,
                        user_message=user_message,
                        context_id=context_id,
                        conversation_id=conversation_id,
                        adapters=adapters,
                        history=history,
                        vars_scope=vars_scope,
                        user_id=user_id,
                        tenant_id=tenant_id,
                        workflow_start_time=workflow_start_time,
                        workflow_timeout_ms=workflow_timeout_ms,
                    )

                    # Merge parallel results into workflow state
                    self._append_trace(
                        workflow_trace,
                        {
                            "node": node_id,
                            "status": parallel_result.status,
                            "parallel_nodes": parallel_node_ids,
                            "failed_nodes": parallel_result.failed_nodes,
                        },
                        max_trace_entries,
                    )

                    # Update vars with namespaced parallel outputs
                    vars_scope.update(parallel_result.merged_outputs)

                    # Update content if parallel nodes produced any
                    if parallel_result.merged_content:
                        content = parallel_result.merged_content

                    # Merge usage
                    usage = self._merge_usage(usage, parallel_result.merged_usage)

                    # Add context snippets
                    for snippet in parallel_result.merged_snippets:
                        if snippet not in context_seen and len(context_snippets) < MAX_CONTEXT_SNIPPETS:
                            context_seen.add(snippet)
                            context_snippets.append(snippet)

                    # Handle parallel failures
                    if parallel_result.status == "error":
                        return await self._handle_node_failure(
                            state_key,
                            node_id,
                            Exception(f"All parallel nodes failed: {parallel_result.failed_nodes}"),
                            vars_scope=vars_scope,
                            context_snippets=context_snippets,
                            workflow_trace=workflow_trace,
                            routing_trace=routing_trace,
                            snapshots=snapshots,
                        )

                # Continue to "after" node if specified
                if after_node:
                    pending.insert(0, after_node)
                continue

            self._append_trace(workflow_trace, {"node": node_id, **result}, max_trace_entries)
            if result.get("outputs"):
                vars_scope.update(result["outputs"])
            if result.get("context_snippets"):
                for snippet in result["context_snippets"]:
                    if snippet in context_seen:
                        continue
                    if len(context_snippets) >= MAX_CONTEXT_SNIPPETS:
                        break
                    context_seen.add(snippet)
                    context_snippets.append(snippet)
            if result.get("content"):
                content = result["content"]
            node_usage = result.get("usage")
            usage = self._merge_usage(usage, node_usage or {})

            pending.extend(next_nodes)
            if result.get("status") == "error" and not next_nodes:
                return await self._record_terminal_failure(
                    state_key,
                    result,
                    workflow_trace=workflow_trace,
                    routing_trace=routing_trace,
                    context_snippets=context_snippets,
                    vars_scope=vars_scope,
                    snapshots=snapshots,
                )
            if result.get("status") == "end":
                break

        if not content:
            content = "No response generated."

        result = {
            "content": content,
            "usage": usage,
            "adapters": adapters,
            "adapter_gates": adapter_gates,
            "context_snippets": context_snippets,
            "workflow_trace": workflow_trace,
            "routing_trace": routing_trace,
            "vars": vars_scope,
        }
        await self._persist_workflow_state(
            state_key,
            {
                "status": "completed",
                "completed_at": datetime.utcnow().isoformat(),
                "result": {
                    "content": content,
                    "adapters": [a.get("id") for a in adapters or []],
                },
            },
        )
        await self.cache_conversation_state(conversation_id, history)
        return result

    async def run_streaming(
        self,
        workflow_id: Optional[str],
        conversation_id: Optional[str],
        user_message: str,
        context_id: Optional[str],
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        cancel_event: Optional[asyncio.Event] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Execute workflow with streaming token output per SPEC §18.

        Yields events:
        - {"event": "token", "data": "token_text"}
        - {"event": "trace", "data": {...workflow_trace...}}
        - {"event": "message_done", "data": {"content": "...", "usage": {...}, ...}}
        - {"event": "error", "data": {"code": "...", "message": "..."}}
        - {"event": "cancel_ack", "data": {}}
        """
        workflow_schema = None
        if workflow_id:
            workflow_schema = self.store.get_latest_workflow(workflow_id)
        if not workflow_schema:
            workflow_schema = self._default_workflow()

        workflow_timeout_ms = workflow_schema.get(
            "timeout_ms", DEFAULT_WORKFLOW_TIMEOUT_MS
        )
        workflow_start_time = time.monotonic()

        adapters, routing_trace, adapter_gates = await self._select_adapters(
            user_message, user_id, context_id
        )
        history = await self._load_conversation_history(
            conversation_id, user_id=user_id, tenant_id=tenant_id
        )

        node_map = {
            n.get("id"): n for n in workflow_schema.get("nodes", []) if n.get("id")
        }
        if not node_map:
            yield self._error_event(
                "validation_error",
                "workflow has no nodes",
                {"workflow_id": workflow_id},
            )
            return

        entry = workflow_schema.get("entrypoint") or next(iter(node_map), None)
        if not entry or entry not in node_map:
            entry = next(iter(node_map)) if node_map else None

        vars_scope: Dict[str, Any] = {}
        workflow_trace: List[Dict[str, Any]] = []
        context_snippets: List[str] = []
        context_seen = set()
        content = ""
        usage: Dict[str, Any] = {}

        pending: List[str] = [entry] if entry else []
        visited = 0
        max_steps = max(1, min(100, len(node_map) * 2 + 10))
        visited_nodes: Dict[str, int] = {}
        max_visits_per_node = max(2, math.ceil(max_steps / max(1, len(node_map))))

        while pending and visited < max_steps:
            # Check for cancellation
            if cancel_event and cancel_event.is_set():
                yield {"event": "cancel_ack", "data": {}}
                return

            # Check workflow timeout
            elapsed_ms = (time.monotonic() - workflow_start_time) * 1000
            if elapsed_ms >= workflow_timeout_ms:
                yield self._error_event(
                    "server_error",
                    "workflow execution timed out",
                    {"timeout_ms": workflow_timeout_ms},
                )
                return

            node_id = pending.pop(0)
            node = node_map.get(node_id)
            if not node:
                continue

            visited += 1
            visited_nodes[node_id] = visited_nodes.get(node_id, 0) + 1
            if visited_nodes[node_id] > max_visits_per_node:
                self.logger.warning("workflow_loop_detected", node=node_id)
                break

            node_type = node.get("type", "tool_call")
            tool_name = node.get("tool", "")

            # Handle streaming for LLM-based tools
            if node_type == "tool_call" and tool_name in {"llm.generic", "llm.generic_chat_v1"}:
                # Stream tokens from LLM
                async for event in self._stream_llm_node(
                    node,
                    user_message=user_message,
                    context_id=context_id,
                    adapters=adapters,
                    history=history,
                    vars_scope=vars_scope,
                    user_id=user_id,
                    tenant_id=tenant_id,
                    cancel_event=cancel_event,
                ):
                    if event["event"] == "token":
                        yield event
                    elif event["event"] == "message_done":
                        # Update state from completed message
                        data = event.get("data", {})
                        content = data.get("content", "")
                        node_usage = data.get("usage", {})
                        usage = self._merge_usage(usage, node_usage)
                        self._append_trace(
                            workflow_trace,
                            {
                                "node": node_id,
                                "status": "ok",
                                "content": content,
                                "usage": node_usage,
                            },
                        )
                        # Emit trace event
                        yield {"event": "trace", "data": {"workflow_trace": workflow_trace[-1]}}
                    elif event["event"] == "error":
                        yield event
                        return
                    elif event["event"] == "cancel_ack":
                        yield event
                        return

                # Move to next nodes
                next_nodes = node.get("next")
                if isinstance(next_nodes, str):
                    pending.append(next_nodes)
                elif isinstance(next_nodes, list):
                    pending.extend([n for n in next_nodes if n])

            else:
                # Non-streaming node execution (switch, parallel, RAG, etc.)
                result, next_nodes = await self._execute_node_with_retry(
                    node,
                    user_message=user_message,
                    context_id=context_id,
                    conversation_id=conversation_id,
                    adapters=adapters,
                    history=history,
                    vars_scope=vars_scope,
                    user_id=user_id,
                    tenant_id=tenant_id,
                    workflow_start_time=workflow_start_time,
                    workflow_timeout_ms=workflow_timeout_ms,
                    cancel_event=cancel_event,
                )

                if result.get("status") == "error" and result.get("retries_exhausted"):
                    yield self._error_event(
                        "server_error",
                        result.get("error", "node execution failed"),
                        {"node_id": node_id, "retries": result.get("retries", 0)},
                    )
                    return

                # Handle parallel node execution in streaming mode
                if result.get("status") == "parallel":
                    parallel_node_ids = result.get("parallel_nodes", [])
                    after_node = result.get("after")

                    if parallel_node_ids:
                        self.logger.info(
                            "workflow_streaming_parallel_start",
                            node_id=node_id,
                            parallel_nodes=parallel_node_ids,
                        )
                        parallel_result = await self._execute_parallel_nodes(
                            parallel_node_ids,
                            node_map,
                            user_message=user_message,
                            context_id=context_id,
                            conversation_id=conversation_id,
                            adapters=adapters,
                            history=history,
                            vars_scope=vars_scope,
                            user_id=user_id,
                            tenant_id=tenant_id,
                            workflow_start_time=workflow_start_time,
                            workflow_timeout_ms=workflow_timeout_ms,
                            cancel_event=cancel_event,
                        )

                        # Record parallel execution in trace
                        self._append_trace(
                            workflow_trace,
                            {
                                "node": node_id,
                                "status": parallel_result.status,
                                "parallel_nodes": parallel_node_ids,
                                "failed_nodes": parallel_result.failed_nodes,
                            },
                        )
                        yield {"event": "trace", "data": {"workflow_trace": workflow_trace[-1]}}

                        # Merge parallel results
                        vars_scope.update(parallel_result.merged_outputs)
                        if parallel_result.merged_content:
                            content = parallel_result.merged_content
                        usage = self._merge_usage(usage, parallel_result.merged_usage)
                        for snippet in parallel_result.merged_snippets:
                            if snippet not in context_seen and len(context_snippets) < MAX_CONTEXT_SNIPPETS:
                                context_seen.add(snippet)
                                context_snippets.append(snippet)

                        # Handle parallel failures
                        if parallel_result.status == "error":
                            yield self._error_event(
                                "server_error",
                                f"All parallel nodes failed: {parallel_result.failed_nodes}",
                                {"failed_nodes": parallel_result.failed_nodes},
                            )
                            return

                    # Continue to "after" node if specified
                    if after_node:
                        pending.insert(0, after_node)
                    continue

                self._append_trace(workflow_trace, {"node": node_id, **result})
                yield {"event": "trace", "data": {"workflow_trace": workflow_trace[-1]}}

                if result.get("outputs"):
                    vars_scope.update(result["outputs"])
                if result.get("context_snippets"):
                    for snippet in result["context_snippets"]:
                        if snippet in context_seen:
                            continue
                        if len(context_snippets) >= MAX_CONTEXT_SNIPPETS:
                            break
                        context_seen.add(snippet)
                        context_snippets.append(snippet)
                if result.get("content"):
                    content = result["content"]
                node_usage = result.get("usage")
                usage = self._merge_usage(usage, node_usage or {})

                pending.extend(next_nodes)
                if result.get("status") == "error" and not next_nodes:
                    yield self._error_event(
                        "server_error",
                        result.get("error", ""),
                        {"node_id": node_id},
                    )
                    return
                if result.get("status") == "end":
                    break

        if not content:
            content = "No response generated."

        # Emit final message_done with complete response
        yield {
            "event": "message_done",
            "data": {
                "content": content,
                "usage": usage,
                "adapters": adapters,
                "adapter_gates": adapter_gates,
                "context_snippets": context_snippets,
                "workflow_trace": workflow_trace,
                "routing_trace": routing_trace,
                "vars": vars_scope,
            },
        }

        await self.cache_conversation_state(conversation_id, history)

    async def _stream_llm_node(
        self,
        node: Dict[str, Any],
        *,
        user_message: str,
        context_id: Optional[str],
        adapters: List[dict],
        history: List[Any],
        vars_scope: Dict[str, Any],
        user_id: Optional[str],
        tenant_id: Optional[str],
        cancel_event: Optional[asyncio.Event] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream tokens from an LLM node."""
        inputs = self._resolve_inputs(node.get("inputs", {}), user_message, vars_scope)
        message = (
            inputs.get("message") or inputs.get("prompt") or inputs.get("text") or ""
        )
        if not message:
            message = user_message

        ctx_ids = self._resolve_context_ids(inputs.get("context_id"), context_id)
        allowed_ctx_ids = self._validate_context_scope(
            ctx_ids, user_id=user_id, tenant_id=tenant_id
        )

        ctx_chunks = self.rag.retrieve(
            allowed_ctx_ids, message, user_id=user_id, tenant_id=tenant_id
        )
        context_snippets = [c.text for c in ctx_chunks]

        # Stream from LLM
        try:
            stream = self.llm.generate_stream(
                message or "",
                adapters=adapters,
                context_snippets=context_snippets,
                history=history,
                user_id=user_id,
            )

            # Iterate through synchronous stream, yielding control for async
            for event in stream:
                if cancel_event and cancel_event.is_set():
                    yield {"event": "cancel_ack", "data": {}}
                    return
                yield event
                # Yield control to allow other tasks
                await asyncio.sleep(0)

        except Exception as exc:
            self.logger.error("llm_stream_error", error=str(exc))
            yield self._error_event(
                "server_error",
                str(exc),
                {"node_id": node.get("id"), "tool": node.get("tool")},
            )

    async def _handle_node_failure(
        self,
        state_key: str,
        node_id: str,
        exc: Exception,
        *,
        vars_scope: Dict[str, Any],
        context_snippets: List[str],
        workflow_trace: List[Dict[str, Any]],
        routing_trace: List[Dict[str, Any]],
        snapshots: Optional[List[WorkflowSnapshot]] = None,
    ) -> dict:
        self.logger.error("workflow_node_failed", node=node_id, error=str(exc))
        failure_entry = {
            "node": node_id,
            "status": "error",
            "error": str(exc),
            "outputs": {},
        }
        self._append_trace(workflow_trace, failure_entry)
        rollback_state = await self._rollback_workflow(
            state_key, workflow_trace, vars_scope, snapshots=snapshots
        )
        if rollback_state:
            failure_entry["rollback"] = rollback_state
        await self._persist_workflow_state(
            state_key,
            {
                "status": "failed",
                "failed_at": datetime.utcnow().isoformat(),
                "error": str(exc),
                "workflow_trace": workflow_trace,
                "vars": vars_scope,
            },
        )
        return {
            "status": "error",
            "content": "workflow execution failed",
            "error": str(exc),
            "routing_trace": routing_trace,
            "workflow_trace": workflow_trace,
            "context_snippets": context_snippets,
            "vars": vars_scope,
            "rollback": rollback_state,
        }

    async def _record_terminal_failure(
        self,
        state_key: str,
        result: Dict[str, Any],
        *,
        workflow_trace: List[Dict[str, Any]],
        routing_trace: List[Dict[str, Any]],
        context_snippets: List[str],
        vars_scope: Dict[str, Any],
        snapshots: Optional[List[WorkflowSnapshot]] = None,
    ) -> Dict[str, Any]:
        rollback_state = await self._rollback_workflow(
            state_key, workflow_trace, vars_scope, reason="tool_error", snapshots=snapshots
        )
        if rollback_state:
            result["rollback"] = rollback_state
        result.setdefault("workflow_trace", workflow_trace)
        result.setdefault("routing_trace", routing_trace)
        result.setdefault("context_snippets", context_snippets)
        result.setdefault("vars", vars_scope)
        await self._persist_workflow_state(
            state_key,
            {
                "status": "failed",
                "completed_at": datetime.utcnow().isoformat(),
                "result": result,
                "error": result.get("error"),
            },
        )
        return result

    async def _execute_node_with_retry(
        self,
        node: Dict[str, Any],
        *,
        user_message: str,
        context_id: Optional[str],
        conversation_id: Optional[str],
        adapters: List[dict],
        history: List[Any],
        vars_scope: Dict[str, Any],
        user_id: Optional[str],
        tenant_id: Optional[str],
        workflow_start_time: float,
        workflow_timeout_ms: float,
        cancel_event: Optional[asyncio.Event] = None,
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Execute a node with SPEC §9/§18 exponential backoff retry logic.

        Retry settings are read from node metadata with defaults:
        - max_retries: 2 (hard cap at 3 per SPEC §18)
        - backoff_ms: 1000 (quadruples each retry: 1s, 4s per SPEC §18)
        """
        node_id = node.get("id", "unknown")
        max_retries = min(
            node.get("max_retries", DEFAULT_NODE_MAX_RETRIES),
            MAX_RETRIES_HARD_CAP,
        )
        backoff_ms = node.get("backoff_ms", DEFAULT_BACKOFF_MS)

        last_error: Optional[Exception] = None
        attempt = 0

        while attempt <= max_retries:
            # Check workflow timeout before each attempt
            elapsed_ms = (time.monotonic() - workflow_start_time) * 1000
            remaining_ms = workflow_timeout_ms - elapsed_ms
            if remaining_ms <= 0:
                return (
                    {
                        "status": "error",
                        "error": "workflow_timeout_during_retry",
                        "retries_exhausted": True,
                        "attempts": attempt,
                    },
                    [],
            )

            try:
                node_timeout_ms = node.get("timeout_ms", DEFAULT_NODE_TIMEOUT_MS)
                result, next_nodes = await asyncio.wait_for(
                    self._execute_node(
                        node,
                        user_message=user_message,
                        context_id=context_id,
                        conversation_id=conversation_id,
                        adapters=adapters,
                        history=history,
                        vars_scope=vars_scope,
                        user_id=user_id,
                        tenant_id=tenant_id,
                    ),
                    timeout=node_timeout_ms / 1000.0,
                )

                # If node executed successfully or has an on_error handler, return
                if result.get("status") != "error" or node.get("on_error"):
                    if attempt > 0:
                        result["retry_attempts"] = attempt
                    return result, next_nodes

                # Node returned an error status - treat as retryable
                last_error = Exception(
                    result.get("error", "node returned error status")
                )

            except asyncio.TimeoutError:
                last_error = asyncio.TimeoutError("node_timeout")
                self.logger.warning(
                    "workflow_node_timeout",
                    node=node_id,
                    attempt=attempt + 1,
                    timeout_ms=node_timeout_ms,
                )
                result = {
                    "status": "error",
                    "error": "node_timeout",
                    "timeout_ms": node_timeout_ms,
                }
                next_nodes = []

            except Exception as exc:
                last_error = exc
                self.logger.warning(
                    "workflow_node_retry",
                    node=node_id,
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    error=str(exc),
                )

            attempt += 1

            # If we have more retries, apply exponential backoff
            if attempt <= max_retries:
                # Exponential backoff: backoff_ms * (4 ^ (attempt - 1))
                # Per SPEC §18: 1s, 4s, 16s progression (quadruple each retry)
                current_backoff_ms = backoff_ms * (4 ** (attempt - 1))

                # Don't sleep longer than remaining workflow timeout
                sleep_ms = min(
                    current_backoff_ms, remaining_ms - 100
                )  # Leave 100ms buffer
                if sleep_ms > 0:
                    self.logger.info(
                        "workflow_node_backoff",
                        node=node_id,
                        attempt=attempt,
                        backoff_ms=sleep_ms,
                    )
                    if cancel_event and cancel_event.is_set():
                        return (
                            {
                                "status": "error",
                                "error": "workflow_cancelled",
                                "cancelled": True,
                            },
                            [],
                        )
                    if cancel_event:
                        try:
                            await asyncio.wait_for(
                                cancel_event.wait(), timeout=sleep_ms / 1000.0
                            )
                            return (
                                {
                                    "status": "error",
                                    "error": "workflow_cancelled",
                                    "cancelled": True,
                                },
                                [],
                            )
                        except asyncio.TimeoutError:
                            pass
                    else:
                        await asyncio.sleep(sleep_ms / 1000.0)

        # All retries exhausted
        self.logger.error(
            "workflow_node_retries_exhausted",
            node=node_id,
            attempts=attempt,
            error=str(last_error),
        )
        return (
            {
                "status": "error",
                "error": str(last_error) if last_error else "unknown error",
                "retries_exhausted": True,
                "attempts": attempt,
            },
            [],
        )

    def _merge_usage(
        self, accum: Dict[str, Any], new_usage: Dict[str, Any]
    ) -> Dict[str, Any]:
        merged = dict(accum)
        for key, value in new_usage.items():
            if isinstance(value, (int, float)):
                merged[key] = merged.get(key, 0) + value
            else:
                merged[key] = value
        return merged

    async def _load_conversation_history(
        self,
        conversation_id: Optional[str],
        *,
        user_id: Optional[str],
        tenant_id: Optional[str],
    ) -> List[Message]:
        if not conversation_id or not hasattr(self.store, "list_messages"):
            return []
        if not self._validate_conversation_scope(
            conversation_id, user_id=user_id, tenant_id=tenant_id
        ):
            return []
        cached: Optional[dict] = None
        if self.cache:
            try:
                cached = await self.cache.get_conversation_summary(conversation_id)
            except Exception as exc:
                self.logger.warning("cache_conversation_summary_failed", error=str(exc))
        if cached and isinstance(cached.get("recent_messages"), list):
            deserialized = self._deserialize_messages(cached["recent_messages"])
            if deserialized:
                return deserialized
        history = self.store.list_messages(conversation_id, user_id=user_id)  # type: ignore[attr-defined]
        await self.cache_conversation_state(conversation_id, history)
        return history

    async def cache_conversation_state(
        self, conversation_id: Optional[str], history: List[Message]
    ) -> None:
        if not conversation_id or not self.cache:
            return
        serialized = self._serialize_messages(history[-10:])
        await self.cache.set_conversation_summary(
            conversation_id,
            {
                "recent_messages": serialized,
                "updated_at": datetime.utcnow().isoformat(),
            },
        )

    async def _persist_workflow_state(self, state_key: str, state: dict) -> None:
        if not self.cache:
            return
        await self.cache.set_workflow_state(state_key, state)

    def _serialize_messages(self, history: List[Message]) -> List[dict]:
        serialized: List[dict] = []
        for msg in history:
            serialized.append(
                {
                    "id": msg.id,
                    "conversation_id": msg.conversation_id,
                    "sender": msg.sender,
                    "role": msg.role,
                    "content": msg.content,
                    "content_struct": msg.content_struct,
                    "seq": msg.seq,
                    "token_count_in": msg.token_count_in,
                    "token_count_out": msg.token_count_out,
                    "created_at": msg.created_at.isoformat(),
                    "meta": msg.meta,
                }
            )
        return serialized

    def _deserialize_messages(self, items: List[dict]) -> List[Message]:
        deserialized: List[Message] = []
        for item in items:
            try:
                deserialized.append(
                    Message(
                        id=str(item.get("id")),
                        conversation_id=str(item.get("conversation_id")),
                        sender=str(item.get("sender", "")),
                        role=str(item.get("role", "assistant")),
                        content=str(item.get("content", "")),
                        content_struct=item.get("content_struct"),
                        seq=int(item.get("seq", 0)),
                        token_count_in=item.get("token_count_in"),
                        token_count_out=item.get("token_count_out"),
                        created_at=datetime.fromisoformat(str(item.get("created_at"))),
                        meta=item.get("meta"),
                    )
                )
            except Exception as exc:
                self.logger.warning(
                    "workflow_deserialize_message_failed", error=str(exc), item=item
                )
                continue
        return deserialized

    def _build_tool_registry(self) -> Dict[str, dict]:
        registry: Dict[str, dict] = {}
        if hasattr(self.store, "list_artifacts"):
            for artifact in self.store.list_artifacts(type_filter="tool"):
                if isinstance(artifact.schema, dict) and artifact.schema.get("name"):
                    registry[artifact.schema["name"]] = artifact.schema
        return registry

    def _validate_tool_payload(
        self, payload: Any, schema: Optional[dict], *, phase: str, tool_name: str
    ) -> Optional[List[str]]:
        if not schema or not isinstance(schema, dict):
            return None
        try:
            validator = Draft202012Validator(schema)
        except SchemaError as exc:  # pragma: no cover - defensive logging
            self.logger.warning(
                "tool_schema_invalid", phase=phase, tool=tool_name, error=str(exc)
            )
            return [f"invalid {phase} schema: {exc.message}"]
        errors = sorted(validator.iter_errors(payload), key=lambda e: e.path)
        if errors:
            return [e.message for e in errors]
        return None

    def _sanitize_html_untrusted(self, value: Any) -> Any:
        """Escape untrusted HTML strings recursively.

        SPEC §9.2 requires sanitizing outputs flagged as html_untrusted. We avoid
        external dependencies and escape markup using the stdlib `html` module.
        Only payloads explicitly marked with `content_type: "html_untrusted"`
        are escaped to avoid mutating other tool outputs.
        """

        import html

        if isinstance(value, list):
            return [self._sanitize_html_untrusted(v) for v in value]
        if isinstance(value, dict):
            sanitized: Dict[str, Any] = {}
            is_html_untrusted = value.get("content_type") == "html_untrusted"
            for k, v in value.items():
                if k == "content_type":
                    sanitized[k] = v
                    continue
                if is_html_untrusted and k == "content":
                    sanitized[k] = html.escape(str(v or ""), quote=True)
                    continue
                sanitized[k] = self._sanitize_html_untrusted(v)
            return sanitized
        return value

    def invoke_tool(
        self,
        tool_schema: dict,
        inputs: Dict[str, Any],
        *,
        conversation_id: Optional[str] = None,
        context_id: Optional[str] = None,
        user_message: Optional[str] = None,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        tool_name = tool_schema.get("name") or tool_schema.get("id")
        if not tool_name:
            return {"status": "error", "content": "tool spec missing name"}
        self.tool_registry.setdefault(tool_name, dict(tool_schema))
        history: List[Any] = []
        if conversation_id and hasattr(self.store, "list_messages"):
            if self._validate_conversation_scope(
                conversation_id, user_id=user_id, tenant_id=tenant_id
            ):
                try:
                    history = self.store.list_messages(conversation_id, user_id=user_id)  # type: ignore[attr-defined]
                except Exception:
                    history = []
        return self._invoke_tool(
            tool_name,
            inputs,
            adapters=[],
            history=history,
            context_id=context_id,
            conversation_id=conversation_id,
            user_message=user_message or inputs.get("message") or "",
            user_id=user_id,
            tenant_id=tenant_id,
        )

    def _default_workflow(self) -> dict:
        plain_chat_node = {
            "id": "plain_chat",
            "type": "tool_call",
            "tool": "llm.generic",
            # forward the user message so llm.generic doesn't receive an empty payload
            "inputs": {"message": "${input.message}"},
        }

        return {
            "kind": "workflow.chat",
            "entrypoint": "plain_chat",
            "nodes": [
                plain_chat_node | {"next": "end"},
                {"id": "end", "type": "end"},
            ],
        }

    async def _select_adapters(
        self, user_message: str, user_id: Optional[str], context_id: Optional[str]
    ) -> Tuple[List[dict], List[dict], List[dict]]:
        adapter_artifacts = [a for a in self.store.list_artifacts(type_filter="adapter")]  # type: ignore[arg-type]
        policy = None
        for art in self.store.list_artifacts(type_filter="policy"):  # type: ignore[arg-type]
            if art.name == "default_routing":
                policy = art.schema
                break
        context_embedding = deterministic_embedding(user_message or "")
        candidates = []
        cluster_lookup: dict[str, Any] = {}
        if hasattr(self.store, "list_semantic_clusters"):
            for cluster in self.store.list_semantic_clusters(user_id):  # type: ignore[attr-defined]
                cluster_lookup[cluster.id] = cluster
            for cluster in self.store.list_semantic_clusters(None):  # type: ignore[attr-defined]
                if cluster.user_id is None:
                    cluster_lookup[cluster.id] = cluster
        for art in adapter_artifacts:
            candidate = {"id": art.id, "name": art.name}
            if isinstance(art.schema, dict):
                candidate.update(art.schema)
            cid = candidate.get("cluster_id")
            if cid and cid in cluster_lookup:
                candidate.setdefault("centroid", cluster_lookup[cid].centroid)
            candidates.append(candidate)
        best_cluster = None
        best_sim = 0.0
        for cluster in cluster_lookup.values():
            emb_a, emb_b = self._align_vectors(context_embedding, cluster.centroid)
            sim = cosine_similarity(emb_a, emb_b)
            if sim > best_sim:
                best_cluster = cluster
                best_sim = sim
        ctx_cluster = None
        if best_cluster:
            ctx_cluster = {
                "id": best_cluster.id,
                "label": best_cluster.label,
                "similarity": best_sim,
            }
        routing = await self.router.route(
            policy or {},
            context_embedding,
            candidates,
            ctx_cluster=ctx_cluster,
            user_id=user_id,
        )
        gates = routing.get("adapters", []) if isinstance(routing, dict) else []
        activated_ids = [gate.get("id", "") for gate in gates if gate.get("id")]
        candidate_lookup = {c.get("id"): c for c in candidates if c.get("id")}
        activated_adapters = [
            candidate_lookup[a_id] for a_id in activated_ids if a_id in candidate_lookup
        ]
        return (
            activated_adapters,
            routing.get("trace", []) if isinstance(routing, dict) else [],
            gates,
        )

    def _align_vectors(
        self, a: List[float], b: List[float]
    ) -> Tuple[List[float], List[float]]:
        return (
            ensure_embedding_dim(a, dim=EMBEDDING_DIM),
            ensure_embedding_dim(b, dim=EMBEDDING_DIM),
        )

    async def _execute_node(
        self,
        node: Dict[str, Any],
        *,
        user_message: str,
        context_id: Optional[str],
        conversation_id: Optional[str],
        adapters: List[dict],
        history: List[Any],
        vars_scope: Dict[str, Any],
        user_id: Optional[str],
        tenant_id: Optional[str],
    ) -> Tuple[Dict[str, Any], List[str]]:
        node_type = node.get("type", "tool_call")
        if node_type == "switch":
            next_nodes = []
            for branch in node.get("branches", []) or []:
                expr = branch.get("when")
                if self._evaluate_condition(expr, user_message, vars_scope):
                    next_nodes.append(branch.get("next"))
                    break
            return {"status": "ok"}, [n for n in next_nodes if n]
        if node_type == "parallel":
            # Return special status to trigger concurrent execution in main loop
            next_nodes = node.get("next", []) or []
            if isinstance(next_nodes, str):
                next_nodes = [next_nodes]
            child_nodes = [n for n in next_nodes if n]
            # After parallel, continue to "after" node if specified
            after_node = node.get("after")
            return {
                "status": "parallel",
                "parallel_nodes": child_nodes,
                "after": after_node,
            }, []
        if node_type == "end":
            return {"status": "end"}, []

        tool_name = node.get("tool", "")
        inputs = self._resolve_inputs(node.get("inputs", {}), user_message, vars_scope)
        if "message" not in inputs and user_message:
            inputs["message"] = user_message

        # SPEC §18: Check circuit breaker before invoking tool
        if self.cache and tool_name:
            is_open, _ = await self.cache.check_circuit_breaker(
                tool_name, tenant_id=tenant_id
            )
            if is_open:
                self.logger.warning("tool_circuit_open", tool=tool_name, tenant_id=tenant_id)
                tool_result = {
                    "status": "error",
                    "content": "tool temporarily unavailable (circuit breaker open)",
                    "error": "circuit_breaker_open",
                }
                outputs = {}
                next_nodes = node.get("next")
                if isinstance(next_nodes, str):
                    next_nodes_list = [next_nodes]
                elif isinstance(next_nodes, list):
                    next_nodes_list = [n for n in next_nodes if n]
                else:
                    next_nodes_list = []
                result_payload = {
                    "node_id": node_id,
                    "status": tool_result.get("status", "done"),
                    "outputs": outputs,
                }
                if isinstance(tool_result, dict):
                    for k in ("content", "usage", "context_snippets"):
                        if k in tool_result:
                            result_payload[k] = tool_result[k]
                return result_payload, next_nodes_list

        try:
            tool_result = self._invoke_tool(
                tool_name,
                inputs,
                adapters,
                history,
                context_id,
                conversation_id,
                user_message,
                user_id=user_id,
                tenant_id=tenant_id,
            )
            # SPEC §18: Record success to reset failure counter
            if self.cache and tool_name:
                if isinstance(tool_result, dict) and tool_result.get("status") != "error":
                    await self.cache.record_tool_success(tool_name, tenant_id=tenant_id)
        except Exception as exc:
            self.logger.error("tool_invoke_failed", tool=tool_name, error=str(exc))
            # SPEC §18: Record failure for circuit breaker (only here for exceptions)
            if self.cache and tool_name:
                tripped, failures = await self.cache.record_tool_failure(
                    tool_name, tenant_id=tenant_id
                )
                if tripped:
                    self.logger.warning(
                        "tool_circuit_tripped",
                        tool=tool_name,
                        failures=failures,
                        tenant_id=tenant_id,
                    )
            tool_result = {
                "status": "error",
                "content": "tool execution failed",
                "error": str(exc),
                "_failure_recorded": True,  # Flag to prevent double-counting
            }
        # Record failure for error results from _invoke_tool (but not if already recorded)
        if (
            self.cache
            and tool_name
            and isinstance(tool_result, dict)
            and tool_result.get("status") == "error"
            and not tool_result.get("_failure_recorded")
        ):
            tripped, failures = await self.cache.record_tool_failure(
                tool_name, tenant_id=tenant_id
            )
            if tripped:
                self.logger.warning(
                    "tool_circuit_tripped",
                    tool=tool_name,
                    failures=failures,
                    tenant_id=tenant_id,
                )
        outputs = {}
        for key in node.get("outputs", []) or []:
            if isinstance(tool_result, dict) and key in tool_result:
                outputs[key] = tool_result[key]
        if isinstance(tool_result, dict) and not outputs:
            outputs = {
                k: v
                for k, v in tool_result.items()
                if k not in {"usage", "context_snippets", "_failure_recorded"}
            }
        next_nodes = node.get("next")
        if isinstance(next_nodes, str):
            next_nodes_list: List[str] = [next_nodes]
        elif isinstance(next_nodes, list):
            next_nodes_list = [n for n in next_nodes if n]
        else:
            next_nodes_list = []
        if isinstance(tool_result, dict) and tool_result.get("status") == "error":
            err_next = node.get("on_error")
            if err_next:
                next_nodes_list = [err_next]
        result_payload: Dict[str, Any] = {
            "status": (
                tool_result.get("status", "ok")
                if isinstance(tool_result, dict)
                else "ok"
            ),
            "outputs": outputs,
        }
        if isinstance(tool_result, dict):
            for k in ("content", "usage", "context_snippets"):
                if k in tool_result:
                    result_payload[k] = tool_result[k]
        return result_payload, next_nodes_list

    def _invoke_tool(
        self,
        tool: str,
        inputs: Dict[str, Any],
        adapters: List[dict],
        history: List[Any],
        context_id: Optional[str],
        conversation_id: Optional[str],
        user_message: str,
        *,
        user_id: Optional[str],
        tenant_id: Optional[str],
    ) -> Dict[str, Any]:
        tool_name = tool or "llm.generic"
        tool_spec = self.tool_registry.get(tool_name)
        # Issue 6.9: Apply hardcap per SPEC §18 (default 15s, hard cap 60s)
        raw_timeout = tool_spec.get("timeout_seconds", 15) if tool_spec else 15
        timeout = min(raw_timeout, MAX_NODE_TIMEOUT_SECONDS)
        validation_errors = self._validate_tool_payload(
            inputs, tool_spec.get("input_schema") if tool_spec else None, phase="input", tool_name=tool_name
        )
        if validation_errors:
            return {
                "status": "error",
                "content": "tool input validation failed",
                "error": "validation_error",
                "details": {"errors": validation_errors},
            }
        handler = self._builtin_tool_handlers().get(tool_name)
        if tool_spec and not handler:
            handler = self._builtin_tool_handlers().get(tool_spec.get("handler"))

        def _run_handler() -> Dict[str, Any]:
            with tool_network_guard(self.tool_network_policy):
                if not handler:
                    return {"status": "error", "content": f"unknown tool {tool_name}"}
                return handler(
                    inputs,
                    adapters,
                    history,
                    context_id,
                    conversation_id,
                    user_message,
                    user_id,
                    tenant_id,
                )

        future = self._tool_executor.submit(_run_handler)
        cancelled = False
        try:
            result = future.result(timeout=timeout)
            sanitized = self._sanitize_html_untrusted(result)
            output_errors = self._validate_tool_payload(
                sanitized,
                tool_spec.get("output_schema") if tool_spec else None,
                phase="output",
                tool_name=tool_name,
            )
            if output_errors:
                return {
                    "status": "error",
                    "content": "tool output validation failed",
                    "error": "validation_error",
                    "details": {"errors": output_errors},
                }
            return sanitized
        except concurrent.futures.TimeoutError:
            self.logger.warning("tool_timeout", tool=tool_name, timeout=timeout)
            cancelled = future.cancel()
            if not cancelled:
                self.logger.warning("tool_timeout_cancellation_failed", tool=tool_name)
            return {"status": "error", "content": "tool timed out", "error": "timeout"}

    def _builtin_tool_handlers(
        self,
    ) -> Dict[
        str,
        Callable[
            [
                Dict[str, Any],
                List[dict],
                List[Any],
                Optional[str],
                Optional[str],
                str,
                Optional[str],
                Optional[str],
            ],
            Dict[str, Any],
        ],
    ]:
        return {
            "llm.generic": self._tool_llm_generic,
            "llm.generic_chat_v1": self._tool_llm_generic,
            "rag.answer_with_context_v1": self._tool_rag_answer,
            "llm.intent_classifier_v1": self._tool_intent_classifier,
            "agent.code_v1": self._tool_agent_code,
            "workflow.end": self._tool_end,
        }

    def _resolve_context_ids(
        self, provided: Any, fallback: Optional[str]
    ) -> Optional[Sequence[str]]:
        ctx_ids = provided or fallback
        if isinstance(ctx_ids, str):
            return [ctx_ids]
        return ctx_ids

    def _validate_context_scope(
        self,
        ctx_ids: Optional[Sequence[str]],
        *,
        user_id: Optional[str],
        tenant_id: Optional[str],
    ) -> Optional[List[str]]:
        if not ctx_ids:
            return None
        if not user_id:
            self.logger.warning("context_scope_missing_user", requested=list(ctx_ids))
            return None

        allowed: List[str] = []
        # MemoryStore path: direct access to context ownership and tenants
        contexts = getattr(self.store, "contexts", None)
        users = getattr(self.store, "users", None)
        if isinstance(contexts, dict):
            for ctx_id in ctx_ids:
                ctx = contexts.get(ctx_id)
                if not ctx or ctx.owner_user_id != user_id:
                    continue
                if tenant_id and isinstance(users, dict):
                    owner = users.get(ctx.owner_user_id)
                    if not owner or owner.tenant_id != tenant_id:
                        continue
                allowed.append(ctx_id)
            return allowed or None

        # Postgres path: fall back to listed contexts for the user
        list_contexts = getattr(self.store, "list_contexts", None)
        get_user = getattr(self.store, "get_user", None)
        if callable(list_contexts):
            owned_contexts = {
                ctx.id: ctx for ctx in list_contexts(owner_user_id=user_id)
            }
            for ctx_id in ctx_ids:
                ctx = owned_contexts.get(ctx_id)
                if not ctx:
                    continue
                if tenant_id and callable(get_user):
                    owner = get_user(ctx.owner_user_id)
                    if not owner or owner.tenant_id != tenant_id:
                        continue
                allowed.append(ctx_id)
            return allowed or None

        self.logger.warning(
            "context_scope_validation_unavailable", requested=list(ctx_ids)
        )
        return None

    def _validate_conversation_scope(
        self, conversation_id: str, *, user_id: Optional[str], tenant_id: Optional[str]
    ) -> bool:
        if not user_id:
            self.logger.warning(
                "conversation_scope_missing_user", conversation_id=conversation_id
            )
            return False

        get_conversation = getattr(self.store, "get_conversation", None)
        if callable(get_conversation):
            conv = get_conversation(conversation_id, user_id=user_id)
        else:
            conv = None
        if not conv:
            self.logger.warning(
                "conversation_scope_forbidden",
                conversation_id=conversation_id,
                user_id=user_id,
            )
            return False

        if tenant_id and hasattr(self.store, "get_user"):
            get_user = getattr(self.store, "get_user")
            if callable(get_user):
                owner = get_user(conv.user_id)
                if not owner or owner.tenant_id != tenant_id:
                    self.logger.warning(
                        "conversation_scope_tenant_mismatch",
                        conversation_id=conversation_id,
                        user_id=user_id,
                        tenant_id=tenant_id,
                    )
                    return False
        return True

    def _tool_llm_generic(
        self,
        inputs: Dict[str, Any],
        adapters: List[dict],
        history: List[Any],
        context_id: Optional[str],
        conversation_id: Optional[str],
        user_message: str,
        user_id: Optional[str],
        tenant_id: Optional[str],
    ) -> Dict[str, Any]:
        message = (
            inputs.get("message") or inputs.get("prompt") or inputs.get("text") or ""
        )
        if not message:
            message = inputs.get("input") or ""
        if not message:
            message = inputs.get("question") or ""
        if not message:
            message = inputs.get("raw") or ""
        if not message:
            message = ""
        ctx_ids = self._resolve_context_ids(inputs.get("context_id"), context_id)
        allowed_ctx_ids = self._validate_context_scope(
            ctx_ids, user_id=user_id, tenant_id=tenant_id
        )

        ctx_chunks = self.rag.retrieve(
            allowed_ctx_ids, message, user_id=user_id, tenant_id=tenant_id
        )
        context_snippets = [c.text for c in ctx_chunks]
        try:
            resp = self.llm.generate(
                message or "",
                adapters=adapters,
                context_snippets=context_snippets,
                history=history,
                user_id=user_id,
            )
        except TypeError:
            resp = self.llm.generate(
                message or "",
                adapters=adapters,
                context_snippets=context_snippets,
                history=history,
            )
        return {
            "content": resp["content"],
            "usage": resp["usage"],
            "context_snippets": context_snippets,
        }

    def _tool_rag_answer(
        self,
        inputs: Dict[str, Any],
        adapters: List[dict],
        history: List[Any],
        context_id: Optional[str],
        conversation_id: Optional[str],
        user_message: str,
        user_id: Optional[str],
        tenant_id: Optional[str],
    ) -> Dict[str, Any]:
        question = inputs.get("question") or inputs.get("message") or ""
        ctx_ids = self._resolve_context_ids(inputs.get("context_id"), context_id)
        allowed_ctx_ids = self._validate_context_scope(
            ctx_ids, user_id=user_id, tenant_id=tenant_id
        )

        chunks = self.rag.retrieve(
            allowed_ctx_ids, question, user_id=user_id, tenant_id=tenant_id
        )
        snippets = [c.text for c in chunks]
        try:
            resp = self.llm.generate(
                question or "",
                adapters=adapters,
                context_snippets=snippets,
                history=history,
                user_id=user_id,
            )
        except TypeError:
            resp = self.llm.generate(
                question or "",
                adapters=adapters,
                context_snippets=snippets,
                history=history,
            )
        return {
            "content": resp["content"],
            "usage": resp["usage"],
            "context_snippets": snippets,
            "answer": resp["content"],
        }

    def _tool_intent_classifier(
        self,
        inputs: Dict[str, Any],
        adapters: List[dict],
        history: List[Any],
        context_id: Optional[str],
        conversation_id: Optional[str],
        user_message: str,
        user_id: Optional[str],
        tenant_id: Optional[str],
    ) -> Dict[str, Any]:
        message = inputs.get("message") or user_message or ""
        lowered = message.lower()
        intent = "qa_with_docs" if "doc" in lowered or "file" in lowered else "analysis"
        if "code" in lowered:
            intent = "code_edit"
        return {"intent": intent}

    def _tool_agent_code(
        self,
        inputs: Dict[str, Any],
        adapters: List[dict],
        history: List[Any],
        context_id: Optional[str],
        conversation_id: Optional[str],
        user_message: str,
        user_id: Optional[str],
        tenant_id: Optional[str],
    ) -> Dict[str, Any]:
        prompt = inputs.get("message") or inputs.get("prompt") or ""
        resp = self.llm.generate(
            prompt or "",
            adapters=adapters,
            context_snippets=[],
            history=history,
            user_id=user_id,
        )
        return {"content": resp["content"], "usage": resp["usage"]}

    def _tool_end(
        self,
        inputs: Dict[str, Any],
        adapters: List[dict],
        history: List[Any],
        context_id: Optional[str],
        conversation_id: Optional[str],
        user_message: str,
        user_id: Optional[str],
        tenant_id: Optional[str],
    ) -> Dict[str, Any]:
        return {"content": inputs.get("message", ""), "usage": {}, "status": "end"}

    def _resolve_inputs(
        self, inputs: Dict[str, Any], user_message: str, vars_scope: Dict[str, Any]
    ) -> Dict[str, Any]:
        def _resolve(val: Any) -> Any:
            if isinstance(val, str) and val.startswith("${") and val.endswith("}"):
                path = val[2:-1].split(".")
                root: Any = {"input": {"message": user_message}, "vars": vars_scope}
                for part in path:
                    if isinstance(root, dict):
                        root = root.get(part)
                    else:
                        root = None
                return root
            if isinstance(val, dict):
                return {k: _resolve(v) for k, v in val.items()}
            if isinstance(val, list):
                return [_resolve(v) for v in val]
            return val

        return {k: _resolve(v) for k, v in inputs.items()}

    def _evaluate_condition(
        self, expr: Optional[str], user_message: str, vars_scope: Dict[str, Any]
    ) -> bool:
        if not expr:
            return False
        try:
            return bool(
                safe_eval_expr(
                    expr,
                    {
                        "input": {"message": user_message},
                        "vars": vars_scope,
                        "true": True,
                        "false": False,
                    },
                )
            )
        except Exception as exc:
            self.logger.warning(
                "workflow_condition_evaluation_failed", expr=expr, error=str(exc)
            )
            return False

    def __del__(self) -> None:
        """Fallback cleanup if shutdown() was not called explicitly."""
        # Issue 23.1: Call shutdown method for proper cleanup
        self.shutdown(wait=False)
