"""Token-by-token execution of a workflow.

The same graph as the non-streaming path, run so that partial output reaches
the user as it is produced. That difference is not cosmetic: a stream has
failure modes the batch path does not — cancellation mid-token, a node failing
after output has already been sent, tool traces that must arrive before the
tokens they explain — and they are easier to get right, and to review, in a
file about nothing else.

This is a mixin, not a standalone service. Streaming *is* the engine's
execution path: it drives node retries, adapter selection, the prompt budget
and the conversation cache, all of which are the engine's own. Splitting it
into free functions would mean threading the engine through every call, which
is this with extra steps. The split here buys a readable file, not decoupling
— WorkflowEngine remains the single object at runtime.
"""

from __future__ import annotations

import asyncio
import math
import time
import uuid
from contextlib import aclosing
from functools import partial
from typing import Any, AsyncIterator, Dict, List, Optional

from liminallm.logging import log_routing_trace, log_workflow_trace
from liminallm.service.broker import InvocationContext
from liminallm.service.invocation import Invocation
from liminallm.service.node_attempt import (
    NodeOutcome,
    StreamedNodeAttempt,
    StreamPump,
)

# Shared with the batch path in workflow.py; imported rather than re-declared so
# a stream and a non-stream run of the same graph cannot diverge.
from liminallm.service.tool_namespace import SYSTEM_SCOPE, ResolvedWorkflow
from liminallm.service.workflow_graph import graph_problems
from liminallm.service.workflow_limits import (
    DEFAULT_WORKFLOW_TIMEOUT_MS,
    MAX_CONTEXT_SNIPPETS,
    ExecutionBudget,
)
from liminallm.storage.common import get_default_attachment_workflow_schema


class _StreamFailed(Exception):
    """A producer's error event, raised so one handler covers both shapes."""


class WorkflowStreamingMixin:
    """Streaming execution for WorkflowEngine. Not usable on its own."""

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
        """Execute workflow with streaming token output per SPEC §13.7.

        Yields events:
        - {"event": "token", "data": "token_text"}
        - {"event": "trace", "data": {...workflow_trace...}}
        - {"event": "message_done", "data": {"content": "...", "usage": {...}, ...}}
        - {"event": "error", "data": {"code": "...", "message": "..."}}
        - {"event": "cancel_ack", "data": {}}
        """
        loaded = None
        if workflow_id:
            # Same ownership check as the blocking path: a workflow is an
            # artifact, and `workflow_id` comes from the request body. Loading
            # it by id alone let any authenticated user stream another user's
            # private workflow.
            loaded = self._load_workflow_for(
                workflow_id, user_id=user_id, tenant_id=tenant_id
            )
        if loaded is None:
            # Same question the blocking path asks, and the same function, so
            # the two cannot answer it differently. See `_turn_needs_tools`.
            if self._turn_needs_tools(conversation_id, user_id):
                loaded = ResolvedWorkflow(
                    get_default_attachment_workflow_schema(), SYSTEM_SCOPE
                )
            else:
                loaded = ResolvedWorkflow(self._default_workflow(), SYSTEM_SCOPE)
        workflow_schema = loaded.schema
        # The same namespace the blocking path uses. A published workflow must
        # name the same capability whether or not the request asked to stream.
        tool_scope = loaded.tool_scope

        workflow_timeout_ms = workflow_schema.get(
            "timeout_ms", DEFAULT_WORKFLOW_TIMEOUT_MS
        )
        workflow_start_time = time.monotonic()

        adapters, routing_trace, adapter_gates = await self._select_adapters(
            user_message, user_id, context_id, tenant_id
        )
        history = await self._load_conversation_history(
            conversation_id, user_id=user_id, tenant_id=tenant_id
        )

        # Before `node_map`, for the reason `run` checks there: this path
        # carried its own copy of the repair semantics, so an invalid row
        # failed closed in blocking chat and silently ran a different graph
        # here. Same rule, this path's vocabulary — blocking raises, streaming
        # emits and stops before a token or a trace reaches anyone.
        problems = graph_problems(workflow_schema)
        if problems:
            yield self._error_event(
                "validation_error",
                "workflow graph is not consistent",
                {"problems": problems},
            )
            return

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

        # `graph_problems` has already refused an entrypoint that names
        # nothing, so this only chooses a start when none was named.
        entry = workflow_schema.get("entrypoint") or next(iter(node_map), None)

        vars_scope: Dict[str, Any] = {}
        workflow_trace: List[Dict[str, Any]] = []
        context_snippets: List[str] = []
        context_seen = set()
        content = ""
        usage: Dict[str, Any] = {}

        pending: List[str] = [entry] if entry else []
        max_steps = max(1, min(100, len(node_map) * 2 + 10))
        visited_nodes: Dict[str, int] = {}
        max_visits_per_node = max(2, math.ceil(max_steps / max(1, len(node_map))))
        # One budget for the whole run, shared with the fan-out this loop
        # dispatches — `_execute_parallel_nodes` is the same method the
        # blocking path calls, and its children were free to both.
        budget = ExecutionBudget(max_steps)
        exhausted: Optional[str] = None

        while pending:
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

            if not budget.reserve():
                exhausted = "workflow_step_limit"
                break
            visited_nodes[node_id] = visited_nodes.get(node_id, 0) + 1
            if visited_nodes[node_id] > max_visits_per_node:
                self.logger.warning("workflow_loop_detected", node=node_id)
                exhausted = "workflow_node_revisit_limit"
                break

            node_type = node.get("type", "tool_call")
            tool_name = node.get("tool", "")

            # Resolve before choosing a capability. This compared the
            # reference *spelling* against a fixed list, so a tenant-shared
            # spec overriding the name `llm.generic` streamed the model while
            # the blocking path ran the override's real body: the `stream`
            # flag decided which capability a published workflow executed.
            # Deciding on the resolved handler also lets an aliased name whose
            # handler *is* the LLM stream, which a spelling comparison could
            # not express.
            descriptor = (
                self._resolve_tool(tool_name, tool_scope)
                if node_type == "tool_call" and tool_name
                else None
            )

            # Handle streaming for LLM-based tools. The attachment agent streams
            # too: its tool rounds emit trace events, then the answer streams.
            #
            # A backend that has not proven it can be stopped does not stream
            # at all — undeclared means no (see
            # `LLMService.stream_is_cancellable`). The branch below runs the
            # node on the ordinary executor: the driver enforces the deadline,
            # the retry waits for the previous attempt to be confirmed dead,
            # and the answer reaches the client in the final `message_done`.
            # What the kill ends there is the worker; a host-tool body such as
            # `llm.generic` runs in the parent's serve thread, so a generation
            # past its deadline is reported failed at the deadline while the
            # body runs on as bounded, authorityless work — the retry is then
            # refused until it returns, not run beside it.
            if (
                descriptor is not None
                and descriptor.streamable
                and self.llm.stream_is_cancellable
            ):
                # The control plane around the call is the blocking path's,
                # shared rather than copied: an open breaker refuses the call
                # (SPEC §18), and a call that fails takes `on_error` (SPEC §9).
                # Both used to live only inside `_execute_node`, which this
                # branch does not call — so an open breaker did not stop a
                # streamed LLM call at all, and a graph declaring
                # `tool -> recover` on failure ended the turn instead. Only
                # token production below is streaming-specific.
                # Everything a resolved tool must pass before any body runs,
                # through the same function the blocking path uses. Streaming
                # specialises token production and nothing above it: this path
                # entered `_stream_llm_node` with only a node, so an ordinary
                # user's own private spec claiming `privileged: true` was
                # refused on one path and ran the model on the other.
                preflight_inputs = self._resolve_inputs(
                    node.get("inputs", {}), user_message, vars_scope
                )
                # The same fallback `_execute_node` applies before the
                # blocking preflight: a node that names no `message` runs on
                # the user's turn, and `_stream_llm_node` will read exactly
                # that — so validation must see the inputs the node executes
                # with, or a schema requiring `message` refuses only here.
                if "message" not in preflight_inputs and user_message:
                    preflight_inputs["message"] = user_message
                refusal = self.tool_preflight(
                    descriptor,
                    preflight_inputs,
                    user_id=user_id,
                    tool_name=tool_name,
                )
                # Keyed by the resolved identity, exactly as the blocking
                # path keys it: the reference's spelling names a different
                # tool per scope, and the artifact behind it is the thing
                # whose health the ledger tracks (SPEC §18).
                tool_result = refusal or await self._circuit_open_result(
                    descriptor.artifact_id or descriptor.name, tenant_id=tenant_id
                )
                failure_event = None
                # Once a token has reached the client it is on their screen,
                # so recovery would append a second answer to the same bubble
                # rather than replace the first. `_stream_agent_files_node`
                # already keeps this boundary for the same reason; the
                # `on_error` handoff needs it too. The driver decides it now,
                # because it is also the retry boundary.
                emitted_tokens = False
                cancelled = False

                if tool_result is None:
                    async with aclosing(
                        self._stream_node_with_retry(
                            node,
                            descriptor=descriptor,
                            tool_name=tool_name,
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
                    ) as driver:
                        async for event in driver:
                            if isinstance(event, NodeOutcome):
                                emitted_tokens = event.emitted
                                failure_event = event.failure_event
                                if event.result.get("status") == "error":
                                    tool_result = event.result
                                continue
                            if event["event"] == "token":
                                yield event
                            elif event["event"] == "trace":
                                # Tool-activity notices from the attachment
                                # agent pass straight through for the UI.
                                yield event
                            elif event["event"] == "message_done":
                                # Update state from completed message
                                data = event.get("data", {})
                                content = data.get("content", "")
                                node_usage = data.get("usage", {})
                                usage = self._merge_usage(usage, node_usage)
                                for snippet in data.get("context_snippets") or []:
                                    if (
                                        snippet not in context_seen
                                        and len(context_snippets) < MAX_CONTEXT_SNIPPETS
                                    ):
                                        context_seen.add(snippet)
                                        context_snippets.append(snippet)
                                trace_entry: Dict[str, Any] = {
                                    "node": node_id,
                                    "status": "ok",
                                    "content": content,
                                    "usage": node_usage,
                                }
                                if data.get("tool_calls"):
                                    trace_entry["tool_calls"] = data["tool_calls"]
                                if data.get("injection_findings"):
                                    trace_entry["injection_findings"] = data[
                                        "injection_findings"
                                    ]
                                self._append_trace(workflow_trace, trace_entry)
                                # Emit trace event
                                yield {
                                    "event": "trace",
                                    "data": {"workflow_trace": workflow_trace[-1]},
                                }
                            elif event["event"] == "cancel_ack":
                                yield event
                                cancelled = True
                                break
                    if cancelled:
                        return

                if tool_result is not None:
                    self._append_trace(
                        workflow_trace, {"node": node_id, **tool_result}
                    )
                    yield {
                        "event": "trace",
                        "data": {"workflow_trace": workflow_trace[-1]},
                    }
                    if self._error_edge(node) and not emitted_tokens:
                        pending.extend(self._successors(node, tool_result))
                        continue
                    # Nowhere to go, or nowhere left to go. The stream ends
                    # where it always did, rather than falling through to
                    # `next`: the chooser answers `next` when no error edge
                    # exists, and handing a failure to the success path gives
                    # it outputs the node never produced.
                    yield failure_event or self._error_event(
                        self._refusal_code(tool_result),
                        tool_result.get("content") or tool_result.get("error", ""),
                        {"node_id": node_id, "tool": tool_name},
                    )
                    return

                pending.extend(self._successors(node, {"status": "ok"}))

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
                    tool_scope=tool_scope,
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
                            budget=budget,
                            tool_scope=tool_scope,
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

                        if parallel_result.status == "budget_exhausted":
                            # Refused before any child began, so there is
                            # nothing to trace and nothing partially done.
                            exhausted = "workflow_step_limit"
                            break

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

        # A run that stopped early must not be handed to the caller as a
        # finished answer. Same rule as the blocking path, said in the
        # streaming vocabulary.
        if exhausted is not None:
            self.logger.warning(
                "workflow_budget_exhausted",
                workflow_id=workflow_id,
                reason=exhausted,
                visited=budget.spent,
                pending=len(pending),
            )
            yield self._error_event(
                "server_error",
                "workflow did not reach an end node",
                {"reason": exhausted, "visited": budget.spent,
                 "max_steps": max_steps},
            )
            return

        if not content:
            content = "No response generated."

        # Emit structured traces for observability (Issue 30.x)
        log_workflow_trace(workflow_trace, logger=self.logger)
        if routing_trace:
            log_routing_trace(routing_trace, logger=self.logger)

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

        await self.cache_conversation_state(conversation_id, history, user_id)

    #: Node failures the client is entitled to see by name. A schema the tool
    #: declared and its own answer failed is not a server fault, and neither is
    #: a refusal: reporting both as `server_error` tells the caller to retry
    #: something that will fail identically. Anything else stays generic — an
    #: error string from a backend is not a code, and the streamed graph errors
    #: already use this vocabulary (`validation_error` for a bad graph).
    REFUSAL_CODES = frozenset({"validation_error", "forbidden"})

    def _refusal_code(self, result: Dict[str, Any]) -> str:
        error = (result or {}).get("error")
        return error if error in self.REFUSAL_CODES else "server_error"

    async def _stream_node_with_retry(
        self,
        node: Dict[str, Any],
        *,
        descriptor,
        tool_name: str,
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
    ) -> AsyncIterator[Any]:
        """A streamed node, under the node contract the blocking path obeys.

        The same driver, the same retry cap, the same three-way deadline and
        the same logical execution — only the attempt body differs. Yields the
        node's stream events, then one `NodeOutcome`, last.
        """
        from liminallm.service.workflow import (
            DEFAULT_BACKOFF_MS,
            DEFAULT_NODE_MAX_RETRIES,
            MAX_RETRIES_HARD_CAP,
        )

        node_id = node.get("id", "unknown")
        max_retries = min(
            node.get("max_retries", DEFAULT_NODE_MAX_RETRIES),
            MAX_RETRIES_HARD_CAP,
        )
        backoff_ms = node.get("backoff_ms", DEFAULT_BACKOFF_MS)

        # The postflight is the blocking path's own, applied to the canonical
        # completed result the streaming implementation hands over — one
        # transformation boundary, not merely a shared predicate, so one
        # schema cannot pass one transport and fail the other, and what
        # proceeds downstream is the sanitized object on both. The schema's
        # *presence* additionally decides buffering: a validated output
        # cannot be incremental (SPEC §9.2), so a node with a schema holds
        # its tokens until the finished answer passes.
        output_schema = (descriptor.schema or {}).get("output_schema")

        def finalize(result: Dict[str, Any]):
            return self.tool_postflight(
                result, descriptor.schema, tool_name=tool_name
            )

        invocation = self.invocations.open(
            uuid.uuid4().hex,
            tool=str(node.get("tool") or ""),
            user_id=user_id,
            tenant_id=tenant_id,
        )

        def make_attempt():
            body = (
                self._stream_agent_files_node
                if descriptor.handler == "agent.files_v1"
                else self._stream_llm_node
            )
            return StreamedNodeAttempt(
                body(
                    node,
                    user_message=user_message,
                    context_id=context_id,
                    conversation_id=conversation_id,
                    adapters=adapters,
                    history=history,
                    vars_scope=vars_scope,
                    user_id=user_id,
                    tenant_id=tenant_id,
                    invocation=invocation,
                    cancel_event=cancel_event,
                ),
                finalize=finalize,
                buffer=bool(output_schema),
            )

        try:
            async with self._cancel_revokes(invocation, cancel_event):
                async with aclosing(
                    self._drive_node_attempts(
                        node,
                        invocation=invocation,
                        node_id=node_id,
                        max_retries=max_retries,
                        backoff_ms=backoff_ms,
                        make_attempt=make_attempt,
                        workflow_start_time=workflow_start_time,
                        workflow_timeout_ms=workflow_timeout_ms,
                        cancel_event=cancel_event,
                        breaker_identity=descriptor.artifact_id or descriptor.name,
                        tenant_id=tenant_id,
                    )
                ) as driver:
                    async for item in driver:
                        yield item
        finally:
            # Reached on success, failure, timeout and cancellation alike, and
            # also when the caller closes this generator early — a client that
            # disconnects mid-stream. Killing and reaping block, so off the
            # loop, exactly as the blocking path does it.
            await asyncio.to_thread(invocation.close)

    async def _stream_llm_node(
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
        invocation: Invocation,
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
        context_snippets = [c.content for c in ctx_chunks]
        # The digest of turns older than the window rides in front of the
        # retrieved context, so it survives pruning longest.
        digest = self._digest_snippet(conversation_id)
        if digest:
            context_snippets.insert(0, digest)
        # Assembled window: relevance-recalled turns ride behind the digest;
        # both are snippets, so the pruner drops them before the verbatim tail.
        recall = self._recall_snippet(conversation_id, user_id, message or "", history)
        if recall:
            context_snippets.insert(1 if digest else 0, recall)
        context_snippets, history = self._apply_prompt_budget(
            message or "", context_snippets, history
        )

        # `generate_stream` is a synchronous iterator, and iterating it here
        # ran the model on the event loop: every other request the worker was
        # serving waited on this one's tokens, and a node past its `timeout_ms`
        # could not be stopped because nothing was watching the clock. The pump
        # owns the iterator on a thread of its own and is registered on the
        # execution, so one revoke reaches it — see `StreamPump`.
        async for event in self._pumped(
            invocation,
            partial(
                self.llm.generate_stream,
                message or "",
                adapters=adapters,
                context_snippets=context_snippets,
                history=history,
                user_id=user_id,
            ),
            label=str(node.get("id") or "llm"),
            cancel_event=cancel_event,
        ):
            if event.get("event") == "message_done":
                # The grounding this node retrieved, on the node's answer —
                # the same key the blocking `llm.generic` result carries. The
                # backend cannot put it there: it never saw the retrieval.
                # Without it the streamed turn reported no context, and an
                # `output_schema` requiring `context_snippets` validated a
                # different object per transport.
                data = dict(event.get("data") or {})
                data.setdefault("context_snippets", list(context_snippets))
                # The canonical completed result, as its own event — exactly
                # the keys blocking `llm.generic` returns. The handler names
                # its result's fields; `StreamedNodeAttempt` consumes this
                # and refuses to reconstruct one from the client event.
                yield {
                    "event": "tool_result",
                    "data": {
                        "content": data.get("content", ""),
                        "usage": data.get("usage") or {},
                        "context_snippets": list(data.get("context_snippets") or []),
                    },
                }
                event = {"event": "message_done", "data": data}
            yield event

    async def _pumped(
        self,
        invocation: Invocation,
        factory,
        *,
        label: str,
        cancel_event: Optional[asyncio.Event],
    ) -> AsyncIterator[Dict[str, Any]]:
        """Drive a synchronous token producer off the loop.

        The registration is the whole of the stop. `asyncio.wait_for` around
        this generator cancels the await inside it and nothing else, so what
        actually stops the thread is the execution being revoked — on the node
        timeout, on a cancel, before a retry, and on the way out. Stopping the
        pump here as well was a second route to the same stop, and mutation
        found it: with two, removing either changed nothing that any test
        could see. One authority for what an attempt started, which is the
        `Invocation`.
        """
        pump = StreamPump(factory, label=label).start()
        invocation.resources.add_producer(pump, f"stream:{label}")
        async for event in pump.events():
            if cancel_event and cancel_event.is_set():
                yield {"event": "cancel_ack", "data": {}}
                return
            yield event

    async def _stream_agent_files_node(
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
        invocation: Invocation,
        cancel_event: Optional[asyncio.Event] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Attachment agent with a streamed final answer.

        The tool-calling rounds run to completion first (they return function
        calls, not prose), each emitting a trace event so the UI can say what
        the model is doing; the answer itself is then streamed token by token.
        """
        inputs = self._resolve_inputs(node.get("inputs", {}), user_message, vars_scope)
        message = inputs.get("message") or user_message or ""
        attachments = self._conversation_attachments(conversation_id, user_id)

        # Retrieval for an explicitly selected knowledge context. Off the loop
        # for the plainer of the two reasons below: it is a database round
        # trip.
        explicit_ids, grounding = await asyncio.to_thread(
            self._explicit_context_grounding,
            message, context_id, user_id=user_id, tenant_id=tenant_id,
        )

        # Off the event loop. Assembling the prompt now includes listing every
        # configured MCP server, and `mcp_client.run_sync` answers an
        # already-running loop by starting a thread and joining it — a join
        # on the loop thread blocks every other request the worker is serving.
        #
        # Honest about the evidence: this path did not reproduce the stall.
        # Reverted, its worst loop gap across a 1.0s listing was 0.021s, while
        # the blocking path's was 1.10s — so this call already reaches a
        # worker thread by some route, and there is no test that fails without
        # this line. It stays because a synchronous network call in an
        # `async def` is a stall waiting for a caller to change, not because a
        # measurement demands it.
        messages, tools, _, mcp_tools, grounded = await asyncio.to_thread(
            partial(
                self._build_agent_context,
                explicit_context_ids=explicit_ids,
                grounding=grounding,
            ),
            message, attachments, history, user_id, conversation_id,
        )
        if not tools or not self.llm.supports_tools:
            async for event in self._stream_llm_node(
                node,
                user_message=user_message,
                context_id=context_id,
                conversation_id=conversation_id,
                adapters=adapters,
                history=history,
                vars_scope=vars_scope,
                user_id=user_id,
                tenant_id=tenant_id,
                invocation=invocation,
                cancel_event=cancel_event,
            ):
                yield event
            return

        session: dict = {}
        snippets: List[str] = []
        tool_trace: List[dict] = []
        usage: Dict[str, Any] = {}
        content = ""
        # Once a token has reached the client, restarting on the plain node
        # would append a second answer to the same bubble.
        emitted_tokens = False

        try:
            # The tool rounds run in the worker, exactly as they do on the
            # batch path — a second copy of the agent loop here is a second
            # copy of its defects. `stream_final` stops the worker once the
            # tools are done and hands back the conversation it built; the
            # final turn offers no tools, so there is no model-chosen control
            # flow left in it to contain, and this side streams it.
            traces: List[dict] = []
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    self._serve_invocation,
                    invocation,
                    "agent.files_v1",
                    {
                        "inputs": dict(inputs or {}),
                        "messages": messages,
                        "tools": tools,
                        "message": message,
                        "max_rounds": self.MAX_AGENT_ROUNDS,
                        "deadline_seconds": self.AGENT_DEADLINE_SECONDS,
                        "stream_final": True,
                        # What survived budgeting, so it is exactly what is
                        # in `messages`. Carried so the worker returns it among
                        # its own and the streamed turn reports the grounding
                        # it used, not only what a tool fetched.
                        "context_snippets": list(grounded),
                    },
                    InvocationContext(
                        user_id=user_id,
                        tenant_id=tenant_id,
                        conversation_id=conversation_id,
                        context_id=context_id,
                        adapters=list(adapters or []),
                        history=list(history or []),
                        user_message=user_message,
                        # On the context, never in the plan above: the plan is
                        # what the worker reads, and an entry there carries the
                        # server's URL and its taint class.
                        mcp_tools=mcp_tools,
                    ),
                    self._worker_limits(self.tool_registry.get("agent.files_v1")),
                    on_capability=traces.append,
                ),
                timeout=self.AGENT_DEADLINE_SECONDS,
            )
            for entry in traces:
                yield {"event": "trace", "data": entry}
            if cancel_event and cancel_event.is_set():
                yield {"event": "cancel_ack", "data": {}}
                return
            messages = result.get("messages") or messages
            usage = self._merge_usage(usage, result.get("usage") or {})
            snippets.extend(result.get("context_snippets") or [])
            tool_trace.extend(result.get("tool_calls") or [])
            session["artifacts"] = list(result.get("artifacts") or [])
            session["injection_findings"] = list(
                result.get("injection_findings") or []
            )

            # Final turn: no tools offered, so the model must answer — streamed.
            # Through the same pump as the plain node: `to_thread` moved the
            # *call* off the loop and then iterated the result on it, which is
            # where the tokens actually arrive.
            content_parts: List[str] = []
            async for event in self._pumped(
                invocation,
                partial(
                    self.llm.stream_messages, messages, adapters, user_id=user_id
                ),
                label="agent.files_v1",
                cancel_event=cancel_event,
            ):
                kind = event.get("event")
                if kind == "token":
                    content_parts.append(str(event.get("data") or ""))
                    emitted_tokens = True
                    yield event
                elif kind == "message_done":
                    data = event.get("data") or {}
                    usage = self._merge_usage(usage, data.get("usage") or {})
                    if data.get("content"):
                        content_parts = [str(data["content"])]
                elif kind == "error":
                    if emitted_tokens:
                        # A backend failure used to reach the handler below as
                        # an exception; the pump reports it as an event,
                        # because it happens on a thread. Same handling, so
                        # the answer already on the client's screen still gets
                        # its turn closed instead of a bare error after it.
                        raise _StreamFailed(
                            (event.get("data") or {}).get("message", "stream failed")
                        )
                    yield event
                    return
                elif kind == "cancel_ack":
                    yield event
                    return
            content = "".join(content_parts)
        except Exception as exc:  # noqa: BLE001 - degrade to a plain answer
            # Revoke before degrading, not after: the fallback below runs
            # inside this handler, so a worker left alive would be answering
            # capability requests while a second answer is being produced.
            await asyncio.to_thread(invocation.revoke, "agent_stream_failed")
            self.logger.warning(
                "attachment_agent_stream_failed",
                conversation_id=conversation_id,
                error=str(exc),
                emitted_tokens=emitted_tokens,
            )
            if emitted_tokens:
                # Keep the partial answer rather than gluing a second one after
                # it; the caller stores what was streamed.
                content = "".join(content_parts)
                partial_result = {
                    "content": content,
                    "usage": usage,
                    "context_snippets": snippets,
                    "tool_calls": tool_trace,
                    "artifacts": session.get("artifacts", []),
                    "injection_findings": session.get("injection_findings", []),
                }
                yield {"event": "tool_result", "data": dict(partial_result)}
                yield {"event": "message_done", "data": partial_result}
                return
            async for event in self._stream_llm_node(
                node,
                user_message=user_message,
                context_id=context_id,
                conversation_id=conversation_id,
                adapters=adapters,
                history=history,
                vars_scope=vars_scope,
                user_id=user_id,
                tenant_id=tenant_id,
                invocation=invocation,
                cancel_event=cancel_event,
            ):
                yield event
            return

        # The canonical completed result — the same six keys the worker's
        # agent loop returns on the blocking path — as its own event for
        # `StreamedNodeAttempt`, then the client's `message_done`. The
        # handler names its result's fields; the attempt must not.
        completed = {
            "content": content,
            "usage": usage,
            "context_snippets": snippets,
            "tool_calls": tool_trace,
            "artifacts": session.get("artifacts", []),
            "injection_findings": session.get("injection_findings", []),
        }
        yield {"event": "tool_result", "data": dict(completed)}
        yield {"event": "message_done", "data": completed}
