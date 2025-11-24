from __future__ import annotations

import random
from datetime import datetime
import json
from typing import Dict, Iterable, List, Sequence, Tuple

from liminallm.service.embeddings import cosine_similarity, pad_vectors
from liminallm.storage.models import PreferenceEvent, SemanticCluster


class SemanticClusterer:
    """Cluster preference events and label emergent skills."""

    def __init__(self, store, llm=None, training=None) -> None:
        self.store = store
        self.llm = llm
        self.training = training

    def cluster_user_preferences(
        self, user_id: str, *, k: int = 3, batch_size: int = 8, min_events: int = 3
    ) -> List[SemanticCluster]:
        events = [e for e in self.store.list_preference_events(user_id=user_id, feedback="positive") if e.context_embedding]
        if len(events) < min_events:
            return []
        embeddings = pad_vectors([list(e.context_embedding) for e in events])
        k = min(k, len(embeddings))
        centroids, assignments = self._mini_batch_kmeans(embeddings, k=k, batch_size=batch_size)
        cluster_events: Dict[int, list[PreferenceEvent]] = {i: [] for i in range(len(centroids))}
        for idx, event in enumerate(events):
            cluster_idx = assignments[idx]
            cluster_events.setdefault(cluster_idx, []).append(event)
        results: List[SemanticCluster] = []
        for idx, centroid in enumerate(centroids):
            members = cluster_events.get(idx, [])
            if not members:
                continue
            sample_messages = [m.message_id for m in members[:5]]
            cluster = self.store.upsert_semantic_cluster(
                user_id=user_id,
                centroid=centroid,
                size=len(members),
                sample_message_ids=sample_messages,
                meta={"method": "mini_batch_kmeans", "k": k},
            )
            for evt in members:
                self.store.update_preference_event(evt.id, cluster_id=cluster.id)
            results.append(cluster)
        self.label_clusters(results, events)
        return results

    def _mini_batch_kmeans(
        self, embeddings: Sequence[Sequence[float]], k: int, batch_size: int = 8, iters: int = 10
    ) -> Tuple[List[List[float]], List[int]]:
        centroids = [list(vec) for vec in random.sample(list(embeddings), k)]
        assignments = [0 for _ in embeddings]
        for _ in range(iters):
            batch_indices = random.sample(range(len(embeddings)), min(batch_size, len(embeddings)))
            for idx in batch_indices:
                vec = embeddings[idx]
                sims = [cosine_similarity(vec, c) for c in centroids]
                best = max(range(len(sims)), key=lambda i: sims[i])
                assignments[idx] = best
                # simple centroid update
                lr = 0.2
                centroids[best] = [c + lr * (v - c) for c, v in zip(centroids[best], vec)]
        # final assignment
        for idx, vec in enumerate(embeddings):
            sims = [cosine_similarity(vec, c) for c in centroids]
            assignments[idx] = max(range(len(sims)), key=lambda i: sims[i])
        return centroids, assignments

    def label_clusters(self, clusters: Iterable[SemanticCluster], events: Sequence[PreferenceEvent], samples: int = 3) -> None:
        event_lookup = {e.cluster_id: [] for e in events if e.cluster_id}
        for evt in events:
            if evt.cluster_id in event_lookup:
                event_lookup[evt.cluster_id].append(evt)
        for cluster in clusters:
            texts: List[str] = []
            for evt in event_lookup.get(cluster.id, [])[:samples]:
                if evt.corrected_text:
                    texts.append(evt.corrected_text)
                elif evt.context_text:
                    texts.append(evt.context_text)
            summary_prompt = "\n".join(texts) if texts else "No examples"
            label, description = self._label_with_llm(cluster, summary_prompt)
            self.store.update_semantic_cluster(
                cluster.id,
                label=label,
                description=description,
                meta={"labeled_at": datetime.utcnow().isoformat()},
            )

    def _label_with_llm(self, cluster: SemanticCluster, text: str) -> Tuple[str, str]:
        if self.llm:
            prompt = (
                "You label semantic clusters of user preference events. "
                "Return JSON with fields 'label' (3-5 words) and 'description' (one sentence).\n"
                f"Cluster size: {cluster.size}\nSample messages:\n{text}"
            )
            resp = self.llm.generate(prompt, adapters=[], context_snippets=[], history=None)
            content = resp.get("content", "") if isinstance(resp, dict) else ""
            parsed = self._parse_label_response(content)
            if parsed:
                return parsed
        fallback = text.split(" ")[:5]
        return (" ".join(fallback) or "Unlabeled cluster", text[:140] or "No description")

    def _parse_label_response(self, content: str) -> Tuple[str, str] | None:
        if not content:
            return None
        try:
            payload = json.loads(content)
            label = str(payload.get("label", "")).strip()
            description = str(payload.get("description", "")).strip()
            if label or description:
                return label or description[:64] or "Unlabeled cluster", description or label
        except (TypeError, ValueError):
            pass
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        if not lines:
            return None
        label = lines[0][:64]
        description = lines[1] if len(lines) > 1 else ""
        return label, description or label

    def promote_skill_adapters(
        self,
        *,
        min_size: int = 5,
        positive_ratio: float = 0.7,
    ) -> List[str]:
        promoted: List[str] = []
        clusters = self.store.list_semantic_clusters()
        adapters = self.store.list_artifacts(type_filter="adapter")  # type: ignore[arg-type]
        bound_clusters = {a.schema.get("cluster_id") for a in adapters if isinstance(a.schema, dict)}
        for cluster in clusters:
            events = self.store.list_preference_events(cluster_id=cluster.id)
            if len(events) < min_size or cluster.id in bound_clusters:
                continue
            positive = [e for e in events if e.feedback in {"positive", "like"} or (e.score or 0) > 0]
            ratio = len(positive) / len(events) if events else 0.0
            if ratio < positive_ratio:
                continue
            owner_id = cluster.user_id or events[0].user_id
            schema = {
                "kind": "adapter.lora",
                "backend": "local",
                "cluster_id": cluster.id,
                "centroid": cluster.centroid,
                "label": cluster.label,
                "description": cluster.description,
                "current_version": 0,
            }
            adapter = self.store.create_artifact(
                type_="adapter",
                name=f"cluster_skill_{cluster.id[:8]}",
                schema=schema,
                description=cluster.description or "Cluster skill adapter",
                owner_user_id=owner_id,
            )
            if self.training:
                self.training.ensure_user_adapter(owner_id, adapter_id_override=adapter.id)
                self.store.create_training_job(
                    user_id=owner_id,
                    adapter_id=adapter.id,
                    preference_event_ids=[e.id for e in events],
                )
            promoted.append(adapter.id)
        return promoted
