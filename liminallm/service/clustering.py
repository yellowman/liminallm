from __future__ import annotations

import asyncio
import inspect
import json
import random
from datetime import datetime
from typing import Dict, Iterable, List, Sequence, Tuple

from liminallm.logging import get_logger
from liminallm.service.embeddings import (
    EMBEDDING_DIM,
    cosine_similarity,
    normalize_vector,
    sanitize_embedding,
    validated_embedding,
    validate_centroid,
)
from liminallm.storage.models import (
    POSITIVE_FEEDBACK_VALUES,
    PreferenceEvent,
    SemanticCluster,
)


class SemanticClusterer:
    """Cluster preference events and label emergent skills."""

    logger = get_logger(__name__)

    def __init__(self, store, llm=None, training=None) -> None:
        self.store = store
        self.llm = llm
        self.training = training

    async def cluster_user_preferences(
        self,
        user_id: str,
        *,
        k: int = 3,
        batch_size: int = 8,
        min_events: int = 3,
        max_events: int = 500,
        streaming: bool = True,
        approximate: bool = True,
        tenant_id: str | None = None,
    ) -> List[SemanticCluster]:
        events = await self._fetch_preference_events(
            user_id=user_id,
            tenant_id=tenant_id,
            max_events=max_events,
        )
        return await self._cluster_preferences(
            events,
            scope_user_id=user_id,
            k=k,
            batch_size=batch_size,
            min_events=min_events,
            streaming=streaming,
            approximate=approximate,
            meta_extra={"scope": "per-user", "tenant_id": tenant_id},
        )

    async def cluster_global_preferences(
        self,
        *,
        k: int = 5,
        batch_size: int = 16,
        min_events: int = 5,
        max_events: int = 1200,
        streaming: bool = True,
        approximate: bool = True,
        tenant_id: str | None = None,
    ) -> List[SemanticCluster]:
        events = await self._fetch_preference_events(
            user_id=None, tenant_id=tenant_id, max_events=max_events
        )
        return await self._cluster_preferences(
            events,
            scope_user_id=None,
            k=k,
            batch_size=batch_size,
            min_events=min_events,
            streaming=streaming,
            approximate=approximate,
            meta_extra={"scope": "global", "tenant_id": tenant_id},
        )

    async def _fetch_preference_events(
        self,
        *,
        user_id: str | None,
        tenant_id: str | None,
        max_events: int,
    ) -> list[PreferenceEvent]:
        events_raw = self.store.list_preference_events(
            user_id=user_id, feedback=POSITIVE_FEEDBACK_VALUES, tenant_id=tenant_id, limit=max_events * 4
        )
        events = events_raw
        if inspect.isawaitable(events_raw):
            events = await events_raw
        filtered = [e for e in events if e.context_embedding]
        if len(filtered) > max_events:
            filtered = self._reservoir_sample(filtered, max_events)
        return filtered

    def _reservoir_sample(
        self, events: Sequence[PreferenceEvent], k: int
    ) -> list[PreferenceEvent]:
        reservoir: list[PreferenceEvent] = []
        for i, evt in enumerate(events):
            if i < k:
                reservoir.append(evt)
                continue
            j = random.randint(0, i)
            if j < k:
                reservoir[j] = evt
        return reservoir

    async def _cluster_preferences(
        self,
        events: Sequence[PreferenceEvent],
        *,
        scope_user_id: str | None,
        k: int,
        batch_size: int,
        min_events: int,
        streaming: bool,
        approximate: bool,
        meta_extra: Dict | None = None,
    ) -> List[SemanticCluster]:
        embeddings: list[list[float]] = []
        valid_events: list[PreferenceEvent] = []
        for evt in events:
            try:
                embeddings.append(
                    validated_embedding(
                        evt.context_embedding,
                        expected_dim=EMBEDDING_DIM,
                        name="context_embedding",
                    )
                )
                valid_events.append(evt)
            except ValueError as exc:
                self.logger.warning(
                    "cluster_embedding_invalid", event_id=evt.id, error=str(exc)
                )

        if len(valid_events) < min_events:
            return []

        k = min(k, len(embeddings))

        initial_centroids = self._warm_start_centroids(scope_user_id, k)
        centroids, assignments = self._mini_batch_kmeans(
            embeddings,
            k=k,
            batch_size=batch_size,
            initial_centroids=initial_centroids,
            streaming=streaming,
        )
        cluster_events: Dict[int, list[PreferenceEvent]] = {
            i: [] for i in range(len(centroids))
        }
        for idx, event in enumerate(valid_events):
            cluster_idx = assignments[idx]
            cluster_events.setdefault(cluster_idx, []).append(event)
        results: List[SemanticCluster] = []
        for idx, centroid in enumerate(centroids):
            members = cluster_events.get(idx, [])
            if not members:
                continue
            sample_messages = [m.message_id for m in members[:5]]
            meta: Dict = {
                "method": "mini_batch_kmeans",
                "k": k,
                "approximate": approximate or len(events) > len(members),
                "streaming": streaming,
            }
            if meta_extra:
                meta.update({k_: v for k_, v in meta_extra.items() if v is not None})
            cluster = self.store.upsert_semantic_cluster(
                user_id=scope_user_id,
                centroid=centroid,
                size=len(members),
                sample_message_ids=sample_messages,
                meta=meta,
            )
            for evt in members:
                evt.cluster_id = cluster.id
                self.store.update_preference_event(evt.id, cluster_id=cluster.id)
            results.append(cluster)
        await self.label_clusters(results, valid_events)
        return results

    def _mini_batch_kmeans(
        self,
        embeddings: Sequence[Sequence[float]],
        k: int,
        batch_size: int = 8,
        iters: int = 10,
        *,
        initial_centroids: list[list[float]] | None = None,
        streaming: bool = False,
    ) -> Tuple[List[List[float]], List[int]]:
        if not embeddings or k <= 0:
            return [], []
        k = min(k, len(embeddings))

        seed_source: list[list[float]] = []
        for centroid in list(initial_centroids or []):
            try:
                seed_source.append(
                    validate_centroid(
                        list(centroid), expected_dim=EMBEDDING_DIM, name="initial_centroid"
                    )
                )
            except ValueError as exc:
                self.logger.warning("cluster_centroid_seed_invalid", error=str(exc))
        if len(seed_source) < k:
            expanded = list(embeddings)
            random.shuffle(expanded)
            seed_source.extend(expanded[: k - len(seed_source)])
        if not seed_source:
            seed_source = list(embeddings)
        centroids = []
        for vec in seed_source[:k]:
            try:
                centroids.append(
                    validate_centroid(
                        sanitize_embedding(vec),
                        expected_dim=EMBEDDING_DIM,
                        name="seed_centroid",
                    )
                )
            except ValueError as exc:
                self.logger.warning("cluster_centroid_invalid", error=str(exc))
                centroids.append([0.0] * EMBEDDING_DIM)
        if not centroids:
            return [], []
        assignments: list[int] = [0 for _ in embeddings]

        if streaming:
            for idx, vec in enumerate(embeddings):
                sims = [cosine_similarity(vec, c) for c in centroids]
                best = max(range(len(sims)), key=lambda i: sims[i])
                assignments[idx] = best
                lr = 1.0 / max(1, assignments.count(best))
                updated = [
                    c + lr * (v - c) for c, v in zip(centroids[best], vec)
                ]
                centroids[best] = normalize_vector(updated)
        else:
            for _ in range(iters):
                batch_indices = random.sample(
                    range(len(embeddings)), min(batch_size, len(embeddings))
                )
                for idx in batch_indices:
                    vec = embeddings[idx]
                    sims = [cosine_similarity(vec, c) for c in centroids]
                    best = max(range(len(sims)), key=lambda i: sims[i])
                    assignments[idx] = best
                    lr = 0.2
                    updated = [
                        c + lr * (v - c) for c, v in zip(centroids[best], vec)
                    ]
                    centroids[best] = normalize_vector(updated)

        cleaned_centroids: list[list[float]] = []
        for centroid in centroids:
            try:
                cleaned_centroids.append(
                    validate_centroid(
                        centroid,
                        expected_dim=EMBEDDING_DIM,
                        name="cluster_centroid",
                    )
                )
            except ValueError as exc:
                self.logger.warning("cluster_centroid_invalid", error=str(exc))
                cleaned_centroids.append([0.0] * EMBEDDING_DIM)
        centroids = cleaned_centroids

        for idx, vec in enumerate(embeddings):
            sims = [cosine_similarity(vec, c) for c in centroids]
            assignments[idx] = max(range(len(sims)), key=lambda i: sims[i])
        return centroids, assignments

    def _warm_start_centroids(
        self, user_id: str | None, k: int
    ) -> list[list[float]]:
        if not hasattr(self.store, "list_semantic_clusters"):
            return []
        clusters = self.store.list_semantic_clusters(user_id=user_id)
        if inspect.isawaitable(clusters):
            # Avoid blocking event loop if an async implementation is provided.
            return []
        sorted_clusters = sorted(
            clusters,
            key=lambda c: c.size,
            reverse=True,
        )
        return [list(c.centroid) for c in sorted_clusters[:k]]

    async def label_clusters(
        self,
        clusters: Iterable[SemanticCluster],
        events: Sequence[PreferenceEvent],
        samples: int = 3,
    ) -> None:
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
            label, description = await self._label_with_llm(cluster, summary_prompt)
            self.store.update_semantic_cluster(
                cluster.id,
                label=label,
                description=description,
                meta={"labeled_at": datetime.utcnow().isoformat()},
            )

    async def _label_with_llm(
        self, cluster: SemanticCluster, text: str
    ) -> Tuple[str, str]:
        if self.llm:
            prompt = (
                "You label semantic clusters of user preference events. "
                "Return JSON with fields 'label' (3-5 words) and 'description' (one sentence).\n"
                f"Cluster size: {cluster.size}\nSample messages:\n{text}"
            )
            generate_fn = getattr(self.llm, "generate", None)
            if inspect.iscoroutinefunction(generate_fn):
                resp = await generate_fn(
                    prompt, adapters=[], context_snippets=[], history=None
                )
            elif callable(getattr(asyncio, "to_thread", None)):
                resp = await asyncio.to_thread(generate_fn, prompt, [], [], None)
            else:
                resp = generate_fn(
                    prompt, adapters=[], context_snippets=[], history=None
                )
            content = resp.get("content", "") if isinstance(resp, dict) else ""
            parsed = self._parse_label_response(content)
            if parsed:
                return parsed
        fallback = text.split(" ")[:5]
        return (
            " ".join(fallback) or "Unlabeled cluster",
            text[:140] or "No description",
        )

    def _parse_label_response(self, content: str) -> Tuple[str, str] | None:
        if not content:
            return None
        try:
            payload = json.loads(content)
            if not isinstance(payload, dict):
                raise ValueError("Label payload must be a JSON object")
            label = str(payload.get("label", "")).strip()
            description = str(payload.get("description", "")).strip()
            if label or description:
                final_label = label or description[:64] or "Unlabeled cluster"
                final_description = description or label or "Unlabeled cluster"
                return final_label, final_description
        except (TypeError, ValueError, AttributeError):
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
        bound_clusters = {
            a.schema.get("cluster_id") for a in adapters if isinstance(a.schema, dict)
        }
        for cluster in clusters:
            events = self.store.list_preference_events(cluster_id=cluster.id)
            if len(events) < min_size or cluster.id in bound_clusters:
                continue
            positive = [
                e
                for e in events
                if e.feedback in POSITIVE_FEEDBACK_VALUES or (e.score or 0) > 0
            ]
            ratio = len(positive) / len(events) if events else 0.0
            if ratio < positive_ratio:
                continue
            owner_id = cluster.user_id or events[0].user_id
            schema = {
                "kind": "adapter.lora",
                "scope": "per-user" if owner_id else "global",
                "backend": "local",
                "rank": 4,
                "layers": [0, 1, 2],
                "matrices": ["attn_q"],
                "cluster_id": cluster.id,
                "centroid": cluster.centroid,
                "label": cluster.label,
                "description": cluster.description,
                "current_version": 0,
                "applicability": {
                    "natural_language": "Skill: "
                    + (cluster.label or "")
                    + " – "
                    + (cluster.description or ""),
                },
            }
            adapter = self.store.create_artifact(
                type_="adapter",
                name=f"cluster_skill_{cluster.id[:8]}",
                schema=schema,
                description=cluster.description or "Cluster skill adapter",
                owner_user_id=owner_id,
            )
            if self.training:
                self.training.ensure_user_adapter(
                    owner_id, adapter_id_override=adapter.id
                )
                create_training_job = getattr(self.store, "create_training_job", None)
                if callable(create_training_job):
                    create_training_job(
                        user_id=owner_id,
                        adapter_id=adapter.id,
                        preference_event_ids=[e.id for e in events],
                    )
            promoted.append(adapter.id)
        return promoted
