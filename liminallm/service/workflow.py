from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from liminallm.service.llm import LLMService
from liminallm.service.rag import RAGService
from liminallm.service.router import RouterEngine
from liminallm.storage.memory import MemoryStore
from liminallm.storage.postgres import PostgresStore


class WorkflowEngine:
    """Executes workflow.chat graphs using a small tool registry."""

    def __init__(self, store: PostgresStore | MemoryStore, llm: LLMService, router: RouterEngine, rag: RAGService) -> None:
        self.store = store
        self.llm = llm
        self.router = router
        self.rag = rag

    def run(self, workflow_id: Optional[str], conversation_id: Optional[str], user_message: str, context_id: Optional[str]) -> dict:
        workflow_schema = None
        if workflow_id:
            workflow_schema = self.store.get_latest_workflow(workflow_id) if hasattr(self.store, "get_latest_workflow") else None
        if not workflow_schema:
            workflow_schema = self._default_workflow()

        adapters, routing_trace, adapter_gates = self._select_adapters(context_id)
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
        max_steps = max(1, len(node_map) * 3)

        while pending and visited < max_steps:
            node_id = pending.pop(0)
            node = node_map.get(node_id)
            if not node:
                continue
            visited += 1

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
            if result.get("usage"):
                usage = result["usage"]

            pending.extend(next_nodes)
            if result.get("status") == "end":
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

    def _default_workflow(self) -> dict:
        return {
            "kind": "workflow.chat",
            "entrypoint": "plain_chat",
            "nodes": [
                {
                    "id": "plain_chat",
                    "type": "tool_call",
                    "tool": "llm.generic",
                    "inputs": {"message": "${input.message}"},
                },
                {"id": "end", "type": "end"},
            ],
        }

    def _select_adapters(self, context_id: Optional[str]) -> Tuple[List[str], List[dict], List[dict]]:
        adapter_artifacts = [a for a in self.store.list_artifacts(type_filter="adapter")]  # type: ignore[arg-type]
        policy = None
        for art in self.store.list_artifacts(type_filter="policy"):  # type: ignore[arg-type]
            if art.name == "default_routing":
                policy = art.schema
                break
        context_embedding = None
        candidates = []
        for art in adapter_artifacts:
            candidate = {"id": art.id}
            if isinstance(art.schema, dict):
                candidate.update(art.schema)
            candidates.append(candidate)
        routing = self.router.route(policy or {}, context_embedding, candidates)
        gates = routing.get("adapters", []) if isinstance(routing, dict) else []
        activated = [gate.get("id", "") for gate in gates if gate.get("id")]
        return [a for a in activated if a], routing.get("trace", []) if isinstance(routing, dict) else [], gates

    def _execute_node(
        self,
        node: Dict[str, Any],
        *,
        user_message: str,
        context_id: Optional[str],
        conversation_id: Optional[str],
        adapters: List[str],
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
        tool_result = self._invoke_tool(tool_name, inputs, adapters, history, context_id, conversation_id)
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
        result_payload: Dict[str, Any] = {"status": "ok", "outputs": outputs}
        if isinstance(tool_result, dict):
            for k in ("content", "usage", "context_snippets"):
                if k in tool_result:
                    result_payload[k] = tool_result[k]
        return result_payload, next_nodes_list

    def _invoke_tool(
        self,
        tool: str,
        inputs: Dict[str, Any],
        adapters: List[str],
        history: List[Any],
        context_id: Optional[str],
        conversation_id: Optional[str],
    ) -> Dict[str, Any]:
        tool_name = tool or "llm.generic"
        if tool_name in {"llm.generic", "llm.generic_chat_v1"}:
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
        if tool_name == "rag.answer_with_context_v1":
            question = inputs.get("question") or inputs.get("message") or ""
            ctx_id = inputs.get("context_id") or context_id
            chunks = self.rag.retrieve(ctx_id, question)
            snippets = [c.text for c in chunks]
            resp = self.llm.generate(question or "", adapters=adapters, context_snippets=snippets, history=history)
            return {"content": resp["content"], "usage": resp["usage"], "context_snippets": snippets, "answer": resp["content"]}
        if tool_name == "llm.intent_classifier_v1":
            message = inputs.get("message") or ""
            lowered = message.lower()
            intent = "qa_with_docs" if "doc" in lowered or "file" in lowered else "analysis"
            if "code" in lowered:
                intent = "code_edit"
            return {"intent": intent}
        if tool_name == "agent.code_v1":
            prompt = inputs.get("message") or inputs.get("prompt") or ""
            resp = self.llm.generate(prompt or "", adapters=adapters, context_snippets=[], history=history)
            return {"content": resp["content"], "usage": resp["usage"]}
        if tool_name == "workflow.end":
            return {"content": inputs.get("message", ""), "usage": {}}
        return {"content": inputs.get("message", ""), "usage": {}}

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
        safe_globals: Dict[str, Any] = {"__builtins__": {}}
        safe_locals = {"input": {"message": user_message}, "vars": vars_scope, "true": True, "false": False}
        try:
            return bool(eval(expr, safe_globals, safe_locals))
        except Exception:
            return False
