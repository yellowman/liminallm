from __future__ import annotations

import concurrent.futures
from typing import Any, Callable, Dict, List, Optional, Tuple

from liminallm.logging import get_logger

from liminallm.service.llm import LLMService
from liminallm.service.rag import RAGService
from liminallm.service.router import RouterEngine
from liminallm.service.sandbox import safe_eval_expr
from liminallm.service.embeddings import cosine_similarity, deterministic_embedding
from liminallm.storage.memory import MemoryStore
from liminallm.storage.postgres import PostgresStore


class WorkflowEngine:
    """Executes workflow.chat graphs using a small tool registry."""

    def __init__(self, store: PostgresStore | MemoryStore, llm: LLMService, router: RouterEngine, rag: RAGService) -> None:
        self.store = store
        self.llm = llm
        self.router = router
        self.rag = rag
        self.logger = get_logger(__name__)
        self.tool_registry = self._build_tool_registry()

    def run(
        self,
        workflow_id: Optional[str],
        conversation_id: Optional[str],
        user_message: str,
        context_id: Optional[str],
        user_id: Optional[str] = None,
    ) -> dict:
        workflow_schema = None
        if workflow_id:
            workflow_schema = self.store.get_latest_workflow(workflow_id) if hasattr(self.store, "get_latest_workflow") else None
        if not workflow_schema:
            workflow_schema = self._default_workflow()

        adapters, routing_trace, adapter_gates = self._select_adapters(user_message, user_id, context_id)
        history = []
        if conversation_id and hasattr(self.store, "list_messages"):
            history = self.store.list_messages(conversation_id)  # type: ignore[attr-defined]

        node_map = {n.get("id"): n for n in workflow_schema.get("nodes", []) if n.get("id")}
        entry = workflow_schema.get("entrypoint") or next(iter(node_map), None)
        if not entry or entry not in node_map:
            entry = next(iter(node_map)) if node_map else None

        vars_scope: Dict[str, Any] = {}
        workflow_trace: List[Dict[str, Any]] = []
        context_snippets: List[str] = []
        content = ""
        usage: Dict[str, Any] = {}

        pending: List[str] = [entry] if entry else []
        visited = 0
        max_steps = max(1, min(100, len(node_map) * 2 + 10))
        visited_nodes: Dict[str, int] = {}

        while pending and visited < max_steps:
            node_id = pending.pop(0)
            node = node_map.get(node_id)
            if not node:
                continue
            visited += 1
            visited_nodes[node_id] = visited_nodes.get(node_id, 0) + 1
            if visited_nodes[node_id] > len(node_map) + 2:
                self.logger.warning("workflow_loop_detected", node=node_id)
                break

            result, next_nodes = self._execute_node(
                node,
                user_message=user_message,
                context_id=context_id,
                conversation_id=conversation_id,
                adapters=adapters,
                history=history,
                vars_scope=vars_scope,
            )
            workflow_trace.append({"node": node_id, **result})
            if result.get("outputs"):
                vars_scope.update(result["outputs"])
            if result.get("context_snippets"):
                context_snippets.extend(result["context_snippets"])
            if result.get("content"):
                content = result["content"]
            node_usage = result.get("usage")
            usage = self._merge_usage(usage, node_usage or {})

            pending.extend(next_nodes)
            if result.get("status") in {"end", "error"}:
                break

        if not content:
            content = "No response generated."

        return {
            "content": content,
            "usage": usage,
            "adapters": adapters,
            "adapter_gates": adapter_gates,
            "context_snippets": context_snippets,
            "workflow_trace": workflow_trace,
            "routing_trace": routing_trace,
            "vars": vars_scope,
        }

    def _merge_usage(self, accum: Dict[str, Any], new_usage: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(accum)
        for key, value in new_usage.items():
            if isinstance(value, (int, float)):
                merged[key] = merged.get(key, 0) + value
            else:
                merged[key] = value
        return merged

    def _build_tool_registry(self) -> Dict[str, dict]:
        registry: Dict[str, dict] = {}
        if hasattr(self.store, "list_artifacts"):
            for artifact in self.store.list_artifacts(type_filter="tool"):
                if isinstance(artifact.schema, dict) and artifact.schema.get("name"):
                    registry[artifact.schema["name"]] = artifact.schema
        return registry

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

    def _select_adapters(
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
            ctx_cluster = {"id": best_cluster.id, "label": best_cluster.label, "similarity": best_sim}
        routing = self.router.route(policy or {}, context_embedding, candidates, ctx_cluster=ctx_cluster)
        gates = routing.get("adapters", []) if isinstance(routing, dict) else []
        activated_ids = [gate.get("id", "") for gate in gates if gate.get("id")]
        candidate_lookup = {c.get("id"): c for c in candidates if c.get("id")}
        activated_adapters = [candidate_lookup[a_id] for a_id in activated_ids if a_id in candidate_lookup]
        return activated_adapters, routing.get("trace", []) if isinstance(routing, dict) else [], gates

    def _align_vectors(self, a: List[float], b: List[float]) -> Tuple[List[float], List[float]]:
        dim = max(len(a), len(b)) or 1
        padded_a = list(a) + [0.0] * (dim - len(a))
        padded_b = list(b) + [0.0] * (dim - len(b))
        return padded_a, padded_b

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
            return {"status": "ok"}, [n for n in next_nodes if n]
        if node_type == "end":
            return {"status": "end"}, []

        tool_name = node.get("tool", "")
        inputs = self._resolve_inputs(node.get("inputs", {}), user_message, vars_scope)
        if "message" not in inputs and user_message:
            inputs["message"] = user_message
        try:
            tool_result = self._invoke_tool(tool_name, inputs, adapters, history, context_id, conversation_id, user_message)
        except Exception as exc:
            self.logger.error("tool_invoke_failed", tool=tool_name, error=str(exc))
            tool_result = {"status": "error", "content": "tool execution failed", "error": str(exc)}
        outputs = {}
        for key in node.get("outputs", []) or []:
            if isinstance(tool_result, dict) and key in tool_result:
                outputs[key] = tool_result[key]
        if isinstance(tool_result, dict) and not outputs:
            outputs = {k: v for k, v in tool_result.items() if k not in {"usage", "context_snippets"}}
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
        result_payload: Dict[str, Any] = {"status": tool_result.get("status", "ok") if isinstance(tool_result, dict) else "ok", "outputs": outputs}
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
            return handler(inputs, adapters, history, context_id, conversation_id, user_message)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run_handler)
            try:
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                self.logger.warning("tool_timeout", tool=tool_name, timeout=timeout)
                return {"status": "error", "content": "tool timed out", "error": "timeout"}

    def _builtin_tool_handlers(self) -> Dict[str, Callable[[Dict[str, Any], List[dict], List[Any], Optional[str], Optional[str], str], Dict[str, Any]]]:
        return {
            "llm.generic": self._tool_llm_generic,
            "llm.generic_chat_v1": self._tool_llm_generic,
            "rag.answer_with_context_v1": self._tool_rag_answer,
            "llm.intent_classifier_v1": self._tool_intent_classifier,
            "agent.code_v1": self._tool_agent_code,
            "workflow.end": self._tool_end,
        }

    def _tool_llm_generic(
        self,
        inputs: Dict[str, Any],
        adapters: List[dict],
        history: List[Any],
        context_id: Optional[str],
        conversation_id: Optional[str],
        user_message: str,
    ) -> Dict[str, Any]:
        message = inputs.get("message") or inputs.get("prompt") or inputs.get("text") or ""
        if not message:
            message = inputs.get("input") or ""
        if not message:
            message = inputs.get("question") or ""
        if not message:
            message = inputs.get("raw") or ""
        if not message:
            message = ""
        ctx_chunks = self.rag.retrieve(inputs.get("context_id", context_id), message)
        context_snippets = [c.text for c in ctx_chunks]
        resp = self.llm.generate(message or "", adapters=adapters, context_snippets=context_snippets, history=history)
        return {"content": resp["content"], "usage": resp["usage"], "context_snippets": context_snippets}

    def _tool_rag_answer(
        self,
        inputs: Dict[str, Any],
        adapters: List[dict],
        history: List[Any],
        context_id: Optional[str],
        conversation_id: Optional[str],
        user_message: str,
    ) -> Dict[str, Any]:
        question = inputs.get("question") or inputs.get("message") or ""
        ctx_id = inputs.get("context_id") or context_id
        chunks = self.rag.retrieve(ctx_id, question)
        snippets = [c.text for c in chunks]
        resp = self.llm.generate(question or "", adapters=adapters, context_snippets=snippets, history=history)
        return {"content": resp["content"], "usage": resp["usage"], "context_snippets": snippets, "answer": resp["content"]}

    def _tool_intent_classifier(
        self,
        inputs: Dict[str, Any],
        adapters: List[dict],
        history: List[Any],
        context_id: Optional[str],
        conversation_id: Optional[str],
        user_message: str,
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
    ) -> Dict[str, Any]:
        prompt = inputs.get("message") or inputs.get("prompt") or ""
        resp = self.llm.generate(prompt or "", adapters=adapters, context_snippets=[], history=history)
        return {"content": resp["content"], "usage": resp["usage"]}

    def _tool_end(
        self,
        inputs: Dict[str, Any],
        adapters: List[dict],
        history: List[Any],
        context_id: Optional[str],
        conversation_id: Optional[str],
        user_message: str,
    ) -> Dict[str, Any]:
        return {"content": inputs.get("message", ""), "usage": {}, "status": "end"}

    def _resolve_inputs(self, inputs: Dict[str, Any], user_message: str, vars_scope: Dict[str, Any]) -> Dict[str, Any]:
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

    def _evaluate_condition(self, expr: Optional[str], user_message: str, vars_scope: Dict[str, Any]) -> bool:
        if not expr:
            return False
        try:
            return bool(
                safe_eval_expr(expr, {"input": {"message": user_message}, "vars": vars_scope, "true": True, "false": False})
            )
        except Exception as exc:
            self.logger.warning("Workflow condition evaluation failed for '%s': %s", expr, exc)
            return False
