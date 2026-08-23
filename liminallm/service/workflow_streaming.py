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
from typing import Any, AsyncIterator, Dict, List, Optional

from liminallm.logging import log_routing_trace, log_workflow_trace
from liminallm.service.broker import InvocationContext

# Shared with the batch path in workflow.py; imported rather than re-declared so
# a stream and a non-stream run of the same graph cannot diverge.
from liminallm.service.workflow_limits import (
    DEFAULT_WORKFLOW_TIMEOUT_MS,
    MAX_CONTEXT_SNIPPETS,
)
from liminallm.storage.common import get_default_attachment_workflow_schema


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
        workflow_schema = None
        if workflow_id:
            # Same ownership check as the blocking path: a workflow is an
            # artifact, and `workflow_id` comes from the request body. Loading
            # it by id alone let any authenticated user stream another user's
            # private workflow.
            workflow_schema = self._load_workflow_for(
                workflow_id, user_id=user_id, tenant_id=tenant_id
            )
        if not workflow_schema:
            # Same question the blocking path asks, and the same function, so
            # the two cannot answer it differently. See `_turn_needs_tools`.
            if self._turn_needs_tools(conversation_id, user_id):
                workflow_schema = get_default_attachment_workflow_schema()
            else:
                workflow_schema = self._default_workflow()

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

            # Handle streaming for LLM-based tools. The attachment agent streams
            # too: its tool rounds emit trace events, then the answer streams.
            if node_type == "tool_call" and tool_name in {
                "llm.generic",
                "llm.generic_chat_v1",
                "agent.files_v1",
            }:
                node_stream = (
                    self._stream_agent_files_node(
                        node,
                        user_message=user_message,
                        context_id=context_id,
                        conversation_id=conversation_id,
                        adapters=adapters,
                        history=history,
                        vars_scope=vars_scope,
                        user_id=user_id,
                        tenant_id=tenant_id,
                        cancel_event=cancel_event,
                    )
                    if tool_name == "agent.files_v1"
                    else self._stream_llm_node(
                        node,
                        user_message=user_message,
                        context_id=context_id,
                        conversation_id=conversation_id,
                        adapters=adapters,
                        history=history,
                        vars_scope=vars_scope,
                        user_id=user_id,
                        tenant_id=tenant_id,
                        cancel_event=cancel_event,
                    )
                )
                async for event in node_stream:
                    if event["event"] == "token":
                        yield event
                    elif event["event"] == "trace":
                        # Tool-activity notices from the attachment agent pass
                        # straight through for the UI to display.
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
                        entry: Dict[str, Any] = {
                            "node": node_id,
                            "status": "ok",
                            "content": content,
                            "usage": node_usage,
                        }
                        if data.get("tool_calls"):
                            entry["tool_calls"] = data["tool_calls"]
                        if data.get("injection_findings"):
                            entry["injection_findings"] = data["injection_findings"]
                        self._append_trace(workflow_trace, entry)
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
        messages, tools, _, mcp_tools = await asyncio.to_thread(
            self._build_agent_context,
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
        invocation = self.invocations.open(
            uuid.uuid4().hex,
            tool="agent.files_v1",
            user_id=user_id,
            tenant_id=tenant_id,
        )

        # A cancel arriving mid-round must stop the work, not only the waiting:
        # the watcher cancels the execution, which takes the worker's process
        # tree down with it.
        cancel_watch = self._watch_for_cancel(invocation, cancel_event)

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
            content_parts: List[str] = []
            stream = await asyncio.to_thread(
                self.llm.stream_messages, messages, adapters, user_id=user_id
            )
            for event in stream:
                if cancel_event and cancel_event.is_set():
                    yield {"event": "cancel_ack", "data": {}}
                    return
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
                    yield event
                    return
                await asyncio.sleep(0)
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
                yield {
                    "event": "message_done",
                    "data": {
                        "content": content,
                        "usage": usage,
                        "context_snippets": snippets,
                        "tool_calls": tool_trace,
                        "injection_findings": session.get("injection_findings", []),
                    },
                }
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
                cancel_event=cancel_event,
            ):
                yield event
            return
        finally:
            if cancel_watch is not None:
                cancel_watch.cancel()
            # Closing the execution kills what it started and removes the
            # scratch, on every path out of here — including the one where a
            # token had already reached the client. Off the loop: it blocks.
            await asyncio.to_thread(invocation.close)

        yield {
            "event": "message_done",
            "data": {
                "content": content,
                "usage": usage,
                "context_snippets": snippets,
                "tool_calls": tool_trace,
                "artifacts": session.get("artifacts", []),
                "injection_findings": session.get("injection_findings", []),
            },
        }
