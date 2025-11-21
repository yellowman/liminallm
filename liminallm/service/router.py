from __future__ import annotations

from typing import Any, Dict, List


class RouterEngine:
    """Evaluate simple routing policy artifacts to choose adapters/tools."""

    def __init__(self) -> None:
        self.safe_functions = {"cosine_similarity": self._cosine_similarity}

    def route(self, policy: dict, context_embedding: List[float] | None, adapters: List[dict]) -> List[dict]:
        activations: List[dict] = []
        rules = policy.get("rules", []) if policy else []
        for rule in rules:
            condition = rule.get("when", "true")
            if self._eval_condition(condition, context_embedding, adapters):
                activations.append(rule.get("action", {}))
        return activations

    def _eval_condition(self, expr: str, context_embedding: List[float] | None, adapters: List[dict]) -> bool:
        if expr.strip() in ("true", "True", "1"):
            return True
        # Extremely small sandbox: only expose cosine_similarity
        local_scope: Dict[str, Any] = {
            "ctx_embedding": context_embedding or [],
            "adapters": adapters,
            "cosine_similarity": self._cosine_similarity,
        }
        try:
            return bool(eval(expr, {"__builtins__": {}}, local_scope))
        except Exception:
            return False

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        num = sum(x * y for x, y in zip(a, b))
        denom = (sum(x * x for x in a) ** 0.5) * (sum(y * y for y in b) ** 0.5)
        return num / denom if denom else 0.0
