from __future__ import annotations

import asyncio
import concurrent.futures
import math
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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
from liminallm.service.sandbox import safe_eval_expr
from liminallm.storage.memory import MemoryStore
from liminallm.storage.models import Message
from liminallm.storage.postgres import PostgresStore
from liminallm.storage.redis_cache import RedisCache

# SPEC §9/§18: Default retry and timeout settings
DEFAULT_NODE_TIMEOUT_MS = 15000  # 15 seconds per node
DEFAULT_NODE_MAX_RETRIES = 2  # Up to 2 retries (3 total attempts), hard cap at 3
DEFAULT_BACKOFF_MS = (
    1000  # Initial backoff 1s, quadruples each retry (1s, 4s per SPEC §18)
)
MAX_RETRIES_HARD_CAP = 3  # SPEC §18: hard cap at 3 retries
DEFAULT_WORKFLOW_TIMEOUT_MS = 60000  # 60 seconds total workflow timeout
MAX_CONTEXT_SNIPPETS = 20


class WorkflowEngine:
    """Executes workflow.chat graphs using a small tool registry."""

    def __init__(
        self,
        store: PostgresStore | MemoryStore,
        llm: LLMService,
        router: RouterEngine,
        rag: RAGService,
        *,
        cache: Optional[RedisCache] = None,
    ) -> None:
        self.store = store
        self.llm = llm
        self.router = router
        self.rag = rag
        self.logger = get_logger(__name__)
        self.tool_registry = self._build_tool_registry()
        self.cache = cache
        self._tool_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    async def _rollback_workflow(
        self,
        state_key: str,
        workflow_trace: List[Dict[str, Any]],
        vars_scope: Dict[str, Any],
        *,
        reason: str = "node_failure",
    ) -> Optional[dict]:
        rollback_state = {
            "status": "rolled_back",
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat(),
            "vars": vars_scope,
            "trace_length": len(workflow_trace),
        }
        try:
            await self._persist_workflow_state(
                state_key,
                {
                    "status": "rolling_back",
                    "reason": reason,
                    "updated_at": datetime.utcnow().isoformat(),
                    "workflow_trace": workflow_trace,
                },
            )
        except Exception as exc:
            self.logger.warning("workflow_rollback_mark_failed", error=str(exc))
            return None
        return rollback_state

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
        context_snippets: List[str] = []
        context_seen = set()
        content = ""
        usage: Dict[str, Any] = {}

        pending: List[str] = [entry] if entry else []
        visited = 0
        max_steps = max(1, min(100, len(node_map) * 2 + 10))
        visited_nodes: Dict[str, int] = {}
        max_visits_per_node = max(2, math.ceil(max_steps / max(1, len(node_map))))

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
                )
            workflow_trace.append({"node": node_id, **result})
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
    ) -> dict:
        self.logger.error("workflow_node_failed", node=node_id, error=str(exc))
        failure_entry = {
            "node": node_id,
            "status": "error",
            "error": str(exc),
            "outputs": {},
        }
        workflow_trace.append(failure_entry)
        rollback_state = await self._rollback_workflow(
            state_key, workflow_trace, vars_scope
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
    ) -> Dict[str, Any]:
        rollback_state = await self._rollback_workflow(
            state_key, workflow_trace, vars_scope, reason="tool_error"
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
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Execute a node with SPEC §9/§18 exponential backoff retry logic.

        Retry settings are read from node metadata with defaults:
        - max_retries: 2 (hard cap at 3 per SPEC §18)
        - backoff_ms: 1000 (doubles each retry: 1s, 4s)
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
                result, next_nodes = self._execute_node(
                    node,
                    user_message=user_message,
                    context_id=context_id,
                    conversation_id=conversation_id,
                    adapters=adapters,
                    history=history,
                    vars_scope=vars_scope,
                    user_id=user_id,
                    tenant_id=tenant_id,
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

    def _execute_node(
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
            next_nodes = node.get("next", []) or []
            if isinstance(next_nodes, str):
                next_nodes = [next_nodes]
            return {"status": "ok"}, [n for n in next_nodes if n]
        if node_type == "end":
            return {"status": "end"}, []

        tool_name = node.get("tool", "")
        inputs = self._resolve_inputs(node.get("inputs", {}), user_message, vars_scope)
        if "message" not in inputs and user_message:
            inputs["message"] = user_message
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
        except Exception as exc:
            self.logger.error("tool_invoke_failed", tool=tool_name, error=str(exc))
            tool_result = {
                "status": "error",
                "content": "tool execution failed",
                "error": str(exc),
            }
        outputs = {}
        for key in node.get("outputs", []) or []:
            if isinstance(tool_result, dict) and key in tool_result:
                outputs[key] = tool_result[key]
        if isinstance(tool_result, dict) and not outputs:
            outputs = {
                k: v
                for k, v in tool_result.items()
                if k not in {"usage", "context_snippets"}
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
        timeout = tool_spec.get("timeout_seconds", 5) if tool_spec else 5
        handler = self._builtin_tool_handlers().get(tool_name)
        if tool_spec and not handler:
            handler = self._builtin_tool_handlers().get(tool_spec.get("handler"))

        def _run_handler() -> Dict[str, Any]:
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
            return future.result(timeout=timeout)
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
        try:
            self._tool_executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
