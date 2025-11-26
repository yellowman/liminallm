from __future__ import annotations

import hashlib
import struct
from typing import Any, Dict, List, Optional, Tuple

from liminallm.logging import get_logger

from .embeddings import EMBEDDING_DIM, cosine_similarity, ensure_embedding_dim
from .sandbox import safe_eval_expr
from liminallm.storage.redis_cache import RedisCache

logger = get_logger(__name__)


class RouterEngine:
    """Evaluate routing policy artifacts to choose adapters/tools."""

    def __init__(self, cache: Optional[RedisCache] = None) -> None:
        self.safe_functions = {
            "cosine_similarity": cosine_similarity,
            "contains": lambda haystack, needle: needle in haystack if haystack is not None else False,
            "len": len,
        }
        self.cache = cache

    async def route(
        self,
        policy: dict,
        context_embedding: List[float] | None,
        adapters: List[dict],
        *,
        safety_risk: Optional[str] = None,
        ctx_cluster: Optional[dict] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        ctx_emb = ensure_embedding_dim(context_embedding, dim=EMBEDDING_DIM) if context_embedding else []
        candidates = adapters or []
        cached = None
        ctx_hash = self._hash_embedding(ctx_emb)
        if self.cache and user_id and ctx_hash:
            cached = await self.cache.get_router_cache(user_id, ctx_hash)
        if cached:
            return cached
        weights: Dict[str, float] = {}
        trace: List[Dict[str, Any]] = []
        rules = policy.get("rules", []) if policy else []
        for rule in rules:
            condition = rule.get("when", "true")
            fired = self._eval_condition(condition, ctx_emb, candidates, safety_risk, ctx_cluster)
            effect = None
            if fired:
                effect = self._apply_action(rule.get("action", {}), candidates, ctx_emb, weights)
            trace.append({"id": rule.get("id"), "when": condition, "fired": fired, "action": rule.get("action"), "effect": effect})
        if not weights and ctx_emb:
            similarity_effects = self._apply_similarity_boost(candidates, ctx_emb, weights)
            if similarity_effects:
                trace.append({"id": "default_similarity_boost", "when": "auto", "fired": True, "action": {"type": "boost_by_similarity"}, "effect": similarity_effects})
        normalized = self._normalize_weights(weights, policy)
        routing = {"adapters": normalized, "trace": trace, "ctx_cluster": ctx_cluster}
        if self.cache and user_id and ctx_hash:
            await self.cache.set_router_cache(user_id, ctx_hash, routing)
        return routing

    def _hash_embedding(self, embedding: List[float]) -> str:
        """Return a deterministic hash for a numeric embedding.

        Uses the full double precision of each component instead of rounding so
        that small numeric differences still produce distinct cache keys.
        """

        if not embedding:
            return ""
        packed = struct.pack(f">{len(embedding)}d", *[float(v) for v in embedding])
        return hashlib.blake2b(packed, digest_size=32).hexdigest()

    def _eval_condition(
        self, expr: str, context_embedding: List[float], adapters: List[dict], safety_risk: Optional[str] = None, ctx_cluster: Optional[dict] = None
    ) -> bool:
        expr = expr or ""
        if expr.strip() in {"true", "True", "1"}:
            return True
        local_scope: Dict[str, Any] = {
            "ctx_embedding": context_embedding,
            "adapters": adapters,
            "safety_risk": safety_risk,
            "ctx_cluster": ctx_cluster,
            **self.safe_functions,
            "true": True,
            "false": False,
            "none": None,
        }
        try:
            return bool(safe_eval_expr(expr, local_scope))
        except Exception as exc:
            logger.warning("routing_condition_evaluation_failed", expr=expr, error=str(exc))
            return False

    def _apply_action(
        self, action: dict, adapters: List[dict], ctx_emb: List[float], weights: Dict[str, float]
    ) -> Dict[str, Any]:
        action_type = action.get("type")
        weight = action.get("weight", 1.0)
        overwrite = action.get("overwrite", False)

        def _record(target_id: Optional[str], applied_weight: float, similarity: Optional[float] = None, ranked: Optional[List[dict]] = None) -> Dict[str, Any]:
            record: Dict[str, Any] = {"target": target_id, "weight": applied_weight}
            if similarity is not None:
                record["similarity"] = similarity
            if ranked:
                record["ranked_candidates"] = ranked
            return record

        if action_type == "deactivate_all_adapters":
            weights.clear()
            return {"cleared": True}

        target_id: Optional[str] = None
        target_similarity: Optional[float] = None

        ranked_candidates: List[dict] = []
        if action_type == "activate_adapter_by_id":
            target_id, target_similarity = self._resolve_adapter(action.get("adapter_id"), adapters, ctx_emb)
        elif action_type == "activate_adapter_by_type":
            target_id, target_similarity, ranked_candidates = self._resolve_adapter_by_field("adapter_type", action.get("adapter_type"), adapters, ctx_emb)
        elif action_type == "activate_adapter_by_cluster":
            target_id, target_similarity, ranked_candidates = self._resolve_adapter_by_field("cluster_id", action.get("cluster_id"), adapters, ctx_emb)
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
        return _record(target_id, applied_weight, target_similarity, ranked_candidates)

    def _apply_similarity_boost(self, adapters: List[dict], ctx_emb: List[float], weights: Dict[str, float]) -> List[dict]:
        effects: List[dict] = []
        for candidate in adapters:
            cand_id = candidate.get("id") or candidate.get("name")
            similarity = cosine_similarity(ctx_emb, self._adapter_embedding(candidate))
            if similarity > 0.6 and cand_id:
                weights[cand_id] = max(weights.get(cand_id, 0.0), similarity)
                effects.append({"target": cand_id, "weight": similarity, "similarity": similarity})
        return effects

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
                sim = cosine_similarity(ctx_emb, self._adapter_embedding(candidate))
                if sim > best_sim:
                    best_id, best_sim = cand_id, sim
            return best_id, best_sim if best_sim >= 0 else None
        for candidate in adapters:
            cand_id = candidate.get("id") or candidate.get("name")
            if cand_id == adapter_id:
                sim = cosine_similarity(ctx_emb, self._adapter_embedding(candidate))
                return cand_id, sim
        return None, None

    def _resolve_adapter_by_field(
        self, field: str, value: Optional[str], adapters: List[dict], ctx_emb: List[float]
    ) -> Tuple[Optional[str], Optional[float], List[dict]]:
        if value == "closest":
            cand_id, sim = self._resolve_adapter(value, adapters, ctx_emb)
            return cand_id, sim, []
        ranked: List[dict] = []
        for candidate in adapters:
            if candidate.get(field) == value:
                cand_id = candidate.get("id") or candidate.get("name")
                ranked.append(
                    {
                        "id": cand_id,
                        "similarity": cosine_similarity(ctx_emb, self._adapter_embedding(candidate)),
                    }
                )
        ranked = sorted(
            [c for c in ranked if c.get("id")],
            key=lambda entry: (-(entry.get("similarity") or 0.0), str(entry.get("id"))),
        )
        if not ranked:
            return None, None, []
        top = ranked[0]
        return str(top.get("id")), float(top.get("similarity") or 0.0), ranked

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
        if not isinstance(emb, list):
            return ensure_embedding_dim([], dim=EMBEDDING_DIM)
        return ensure_embedding_dim(emb, dim=EMBEDDING_DIM)

