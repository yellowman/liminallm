from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .sandbox import safe_eval_expr


class RouterEngine:
    """Evaluate routing policy artifacts to choose adapters/tools."""

    def __init__(self) -> None:
        self.safe_functions = {
            "cosine_similarity": self._cosine_similarity,
            "contains": lambda haystack, needle: needle in haystack if haystack is not None else False,
            "len": len,
        }

    def route(
        self, policy: dict, context_embedding: List[float] | None, adapters: List[dict], *, safety_risk: Optional[str] = None
    ) -> Dict[str, Any]:
        ctx_emb = context_embedding or []
        candidates = adapters or []
        weights: Dict[str, float] = {}
        trace: List[Dict[str, Any]] = []
        rules = policy.get("rules", []) if policy else []
        for rule in rules:
            condition = rule.get("when", "true")
            fired = self._eval_condition(condition, ctx_emb, candidates, safety_risk)
            effect = None
            if fired:
                effect = self._apply_action(rule.get("action", {}), candidates, ctx_emb, weights)
            trace.append({"id": rule.get("id"), "when": condition, "fired": fired, "action": rule.get("action"), "effect": effect})
        normalized = self._normalize_weights(weights, policy)
        return {"adapters": normalized, "trace": trace}

    def _eval_condition(
        self, expr: str, context_embedding: List[float], adapters: List[dict], safety_risk: Optional[str] = None
    ) -> bool:
        expr = expr or ""
        if expr.strip() in {"true", "True", "1"}:
            return True
        local_scope: Dict[str, Any] = {
            "ctx_embedding": context_embedding,
            "adapters": adapters,
            "safety_risk": safety_risk,
            **self.safe_functions,
            "true": True,
            "false": False,
            "none": None,
        }
        try:
            return bool(safe_eval_expr(expr, local_scope))
        except Exception:
            return False

    def _apply_action(
        self, action: dict, adapters: List[dict], ctx_emb: List[float], weights: Dict[str, float]
    ) -> Dict[str, Any]:
        action_type = action.get("type")
        weight = action.get("weight", 1.0)
        overwrite = action.get("overwrite", False)

        def _record(target_id: Optional[str], applied_weight: float) -> Dict[str, Any]:
            return {"target": target_id, "weight": applied_weight}

        if action_type == "deactivate_all_adapters":
            weights.clear()
            return {"cleared": True}

        target_id: Optional[str] = None
        target_similarity: Optional[float] = None

        if action_type == "activate_adapter_by_id":
            target_id, target_similarity = self._resolve_adapter(action.get("adapter_id"), adapters, ctx_emb)
        elif action_type == "activate_adapter_by_type":
            target_id, target_similarity = self._resolve_adapter_by_field("adapter_type", action.get("adapter_type"), adapters, ctx_emb)
        elif action_type == "activate_adapter_by_cluster":
            target_id, target_similarity = self._resolve_adapter_by_field("cluster_id", action.get("cluster_id"), adapters, ctx_emb)
        elif action_type == "deactivate_adapter":
            target_id, _ = self._resolve_adapter(action.get("adapter_id"), adapters, ctx_emb)
            if target_id:
                weights.pop(target_id, None)
            return _record(target_id, 0.0)
        elif action_type == "scale_adapter_weight":
            target_id, _ = self._resolve_adapter(action.get("adapter_id"), adapters, ctx_emb)
            if target_id and target_id in weights:
                weights[target_id] *= action.get("scale", 1.0)
            return _record(target_id, weights.get(target_id, 0.0))

        if not target_id:
            return {}

        applied_weight = self._resolve_weight(weight, target_similarity)
        if overwrite or target_id not in weights:
            weights[target_id] = applied_weight
        else:
            weights[target_id] = max(weights[target_id], applied_weight)
        return _record(target_id, applied_weight)

    def _resolve_weight(self, configured_weight: Any, similarity: Optional[float]) -> float:
        if isinstance(configured_weight, (int, float)):
            return float(configured_weight)
        if isinstance(configured_weight, str) and configured_weight == "similarity":
            return float(similarity or 0.0)
        return 1.0

    def _resolve_adapter(
        self, adapter_id: Optional[str], adapters: List[dict], ctx_emb: List[float]
    ) -> Tuple[Optional[str], Optional[float]]:
        if not adapters:
            return None, None
        if adapter_id == "closest":
            best_id: Optional[str] = None
            best_sim = -1.0
            for candidate in adapters:
                cand_id = candidate.get("id") or candidate.get("name")
                sim = self._cosine_similarity(ctx_emb, self._adapter_embedding(candidate))
                if sim > best_sim:
                    best_id, best_sim = cand_id, sim
            return best_id, best_sim if best_sim >= 0 else None
        for candidate in adapters:
            cand_id = candidate.get("id") or candidate.get("name")
            if cand_id == adapter_id:
                sim = self._cosine_similarity(ctx_emb, self._adapter_embedding(candidate))
                return cand_id, sim
        return None, None

    def _resolve_adapter_by_field(
        self, field: str, value: Optional[str], adapters: List[dict], ctx_emb: List[float]
    ) -> Tuple[Optional[str], Optional[float]]:
        if value == "closest":
            return self._resolve_adapter(value, adapters, ctx_emb)
        for candidate in adapters:
            if candidate.get(field) == value:
                cand_id = candidate.get("id") or candidate.get("name")
                return cand_id, self._cosine_similarity(ctx_emb, self._adapter_embedding(candidate))
        return None, None

    def _normalize_weights(self, weights: Dict[str, float], policy: dict) -> List[Dict[str, Any]]:
        max_active = policy.get("max_active_adapters", 3)
        weight_floor = policy.get("weight_floor", 0.05)
        # clamp to 0..1
        clamped = {k: min(1.0, max(0.0, v)) for k, v in weights.items()}
        if not clamped:
            return []
        total = sum(clamped.values())
        if total > 1.0:
            clamped = {k: v / total for k, v in clamped.items()}
        filtered = {k: v for k, v in clamped.items() if v >= weight_floor}
        if not filtered:
            return []
        ranked = sorted(filtered.items(), key=lambda kv: kv[1], reverse=True)
        return [{"id": adapter_id, "weight": weight} for adapter_id, weight in ranked[:max_active]]

    def _adapter_embedding(self, adapter: dict) -> List[float]:
        emb = adapter.get("embedding") or adapter.get("centroid") or []
        return emb if isinstance(emb, list) else []

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        num = sum(x * y for x, y in zip(a, b))
        denom = (sum(x * x for x in a) ** 0.5) * (sum(y * y for y in b) ** 0.5)
        return num / denom if denom else 0.0
