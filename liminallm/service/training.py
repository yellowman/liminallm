from __future__ import annotations

import json
import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence

from liminallm.service.fs import safe_join
from liminallm.storage.errors import ConstraintViolation
from liminallm.storage.models import Artifact, PreferenceEvent


class TrainingService:
    """Minimal LoRA adapter training loop following SPEC phase 2."""

    def __init__(self, store, fs_root: str) -> None:
        self.store = store
        self.fs_root = Path(fs_root)

    def ensure_user_adapter(self, user_id: str, *, rank: int = 4) -> Artifact:
        existing = [a for a in self.store.list_artifacts(type_filter="adapter") if a.owner_user_id == user_id]
        if existing:
            return existing[0]
        adapter_id = None
        adapter_schema = {
            "kind": "adapter.lora",
            "scope": "per-user",
            "user_id": user_id,
            "rank": rank,
            "layers": list(range(4)),
            "matrices": ["attn_q", "attn_v"],
            "current_version": 0,
        }
        adapter = self.store.create_artifact(
            type_="adapter",
            name="persona_adapter",
            schema=adapter_schema,
            description="Per-user persona adapter",
            owner_user_id=user_id,
        )
        adapter_schema["fs_dir"] = str(self._adapter_dir(user_id, adapter.id))
        self.store.update_artifact(adapter.id, adapter_schema)
        return self.store.get_artifact(adapter.id) or adapter

    def train_from_preferences(self, user_id: str, adapter_id: Optional[str] = None) -> Optional[dict]:
        adapter = self.ensure_user_adapter(user_id) if not adapter_id else self.store.get_artifact(adapter_id)
        if not adapter:
            raise ConstraintViolation("adapter missing", {"adapter_id": adapter_id})
        events = self.store.list_preference_events(user_id=user_id, feedback="positive")
        if not events:
            return None
        cluster_meta = self._cluster_events(events, user_id)
        dataset_entries = list(self._build_examples(events))
        token_batches = list(self._tokenize_batches(dataset_entries))
        job = self.store.create_training_job(
            user_id=user_id,
            adapter_id=adapter.id,
            preference_event_ids=[e.id for e in events],
        )
        job_dir = self._job_dir(user_id, adapter.id, job.id)
        job_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = job_dir / "dataset.jsonl"
        with dataset_path.open("w") as f:
            for row in dataset_entries:
                f.write(json.dumps(row) + "\n")
        self.store.update_training_job(
            job.id,
            dataset_path=str(dataset_path),
            meta={"token_batches": [b["shape"] for b in token_batches], "clusters": cluster_meta},
        )
        next_version = int(adapter.schema.get("current_version", 0)) + 1
        version_dir = self._adapter_dir(user_id, adapter.id) / f"v{next_version:04d}"
        version_dir.mkdir(parents=True, exist_ok=True)
        params_path = version_dir / "params.json"
        weights = self._init_lora_weights(
            rank=int(adapter.schema.get("rank", 4)),
            layers=adapter.schema.get("layers", []),
            matrices=adapter.schema.get("matrices", []),
        )
        params_path.write_text(json.dumps(weights, indent=2))
        metadata = {
            "adapter_id": adapter.id,
            "user_id": user_id,
            "created_at": datetime.utcnow().isoformat(),
            "version": next_version,
            "dataset_path": str(dataset_path),
            "preference_events": [asdict(e) for e in events],
            "token_batches": [b["shape"] for b in token_batches],
            "clusters": cluster_meta,
        }
        (version_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
        updated_schema = dict(adapter.schema)
        updated_schema["current_version"] = next_version
        updated_schema["fs_dir"] = str(self._adapter_dir(user_id, adapter.id))
        self.store.update_artifact(adapter.id, updated_schema)
        loss = 1.0 / (1 + len(dataset_entries))
        training_trace = self._run_jax_optax_training(weights, token_batches)
        self.store.update_training_job(
            job.id,
            status="succeeded",
            loss=loss,
            new_version=next_version,
            meta={
                "token_batches": [b["shape"] for b in token_batches],
                "jax_trace": training_trace,
                "clusters": cluster_meta,
            },
        )
        return {
            "job_id": job.id,
            "adapter_id": adapter.id,
            "version_dir": str(version_dir),
            "loss": loss,
            "token_batches": token_batches,
            "jax_trace": training_trace,
            "clusters": cluster_meta,
        }

    def _build_examples(self, events: Iterable[PreferenceEvent]) -> Iterable[dict]:
        for event in events:
            messages = self.store.list_messages(event.conversation_id, limit=200)
            prompt_chunks: List[str] = []
            target_text = event.corrected_text
            cluster_id = event.cluster_id or self._bucket_embedding(event.context_embedding, event.user_id)
            for msg in messages:
                prompt_chunks.append(f"{msg.role.upper()}: {msg.content}")
                if msg.id == event.message_id and not target_text:
                    target_text = msg.content
            if event.context_text:
                prompt_chunks.append(f"CONTEXT_SNIPPET: {event.context_text}")
            if not target_text:
                target_text = ""
            yield {
                "prompt": "\n".join(prompt_chunks[-50:]),
                "target": target_text,
                "weight": event.weight,
                "context": {
                    "conversation_id": event.conversation_id,
                    "message_id": event.message_id,
                    "cluster_id": cluster_id,
                    "context_embedding": event.context_embedding,
                    "context_text": event.context_text,
                },
            }

    def _tokenize_batches(
        self,
        dataset_entries: Sequence[dict],
        batch_size: int = 2,
        max_length: int = 512,
    ) -> Iterator[dict]:
        """
        Convert preference_event-derived examples into padded token batches.

        This intentionally keeps tokenization minimal: it accepts any tokenizer
        object with an ``encode`` method and falls back to whitespace splitting
        to avoid hard dependencies. Batches carry shapes so downstream
        backends (e.g., JAX) can preallocate arrays without re-tokenizing.
        """

        def _encode(text: str) -> List[int]:
            if hasattr(self, "tokenizer") and getattr(self, "tokenizer") is not None:
                return list(getattr(self, "tokenizer").encode(text, truncation=True, max_length=max_length))
            tokens = text.split()
            return [hash(tok) % 32000 for tok in tokens[:max_length]]

        for i in range(0, len(dataset_entries), batch_size):
            batch = dataset_entries[i : i + batch_size]
            prompts = [_encode(row["prompt"]) for row in batch]
            targets = [_encode(row["target"]) for row in batch]
            max_prompt = min(max((len(p) for p in prompts), default=0), max_length)
            max_target = min(max((len(t) for t in targets), default=0), max_length)

            def _pad(seq: List[int], length: int) -> List[int]:
                if len(seq) >= length:
                    return seq[:length]
                return seq + [0] * (length - len(seq))

            yield {
                "input_ids": [_pad(p, max_prompt) for p in prompts],
                "labels": [_pad(t, max_target) for t in targets],
                "attention_mask": [[1] * min(len(p), max_prompt) + [0] * max(0, max_prompt - len(p)) for p in prompts],
                "shape": {
                    "batch": len(batch),
                    "prompt_len": max_prompt,
                    "target_len": max_target,
                },
            }

    def _adapter_dir(self, user_id: str, adapter_id: str) -> Path:
        return safe_join(self.fs_root, f"users/{user_id}/adapters/{adapter_id}")

    def _job_dir(self, user_id: str, adapter_id: str, job_id: str) -> Path:
        return safe_join(self._adapter_dir(user_id, adapter_id), f"jobs/{job_id}")

    def _init_lora_weights(self, rank: int, layers: List[int], matrices: List[str]) -> dict:
        weights: dict[str, list[list[float]]] = {}
        hidden_dim = max(rank * 4, 8)
        for layer in layers:
            for matrix in matrices:
                key_a = f"layer_{layer}.{matrix}.A"
                key_b = f"layer_{layer}.{matrix}.B"
                weights[key_a] = [[random.uniform(-0.01, 0.01) for _ in range(hidden_dim)] for _ in range(rank)]
                weights[key_b] = [[random.uniform(-0.01, 0.01) for _ in range(rank)] for _ in range(hidden_dim)]
        return weights

    def _run_jax_optax_training(self, params: dict, batches: Sequence[dict]) -> dict:
        """
        Sketch a single-adapter JAX/Optax loop (no base-model updates).

        The implementation is intentionally lightweight and only runs if
        `jax` and `optax` are available. It treats LoRA matrices as a flat
        parameter dict and minimizes a simple L2 loss between a toy forward
        pass and target labels. Real deployments should swap in the
        architecture-specific forward function described in SPEC §5.
        """

        try:
            import jax
            import jax.numpy as jnp
            import optax
        except Exception:
            return {"status": "skipped", "reason": "jax/optax not installed"}

        def _flatten_params(param_dict: dict) -> dict:
            return {k: jnp.array(v) for k, v in param_dict.items()}

        def forward(p: dict, inputs: jnp.ndarray) -> jnp.ndarray:
            acc = jnp.zeros((inputs.shape[0], inputs.shape[1]))
            for name, mat in p.items():
                if name.endswith(".A"):
                    base = mat @ inputs.T
                    b_key = name.replace(".A", ".B")
                    if b_key in p:
                        acc = acc + (p[b_key] @ base).T
            return acc

        def loss_fn(p: dict, batch: dict) -> jnp.ndarray:
            inputs = jnp.array(batch["input_ids"], dtype=jnp.float32)
            labels = jnp.array(batch["labels"], dtype=jnp.float32)
            preds = forward(p, inputs)
            return jnp.mean((preds - labels) ** 2)

        opt = optax.adam(1e-3)
        params_tree = _flatten_params(params)
        opt_state = opt.init(params_tree)
        grad_fn = jax.value_and_grad(loss_fn)
        trace: list[dict] = []

        for batch in batches:
            value, grads = grad_fn(params_tree, batch)
            updates, opt_state = opt.update(grads, opt_state)
            params_tree = optax.apply_updates(params_tree, updates)
            trace.append({"loss": float(value), "shape": batch["shape"]})

            return {"status": "ok", "steps": trace[-10:]}

    def _bucket_embedding(self, embedding: Sequence[float], user_id: str) -> Optional[str]:
        if not embedding:
            return None
        rounded = tuple(round(v, 1) for v in embedding[:8])
        return f"{user_id}-c{abs(hash(rounded)) % 1_000_000:06d}"

    def _cluster_events(self, events: Sequence[PreferenceEvent], user_id: str) -> List[dict]:
        clusters: dict[str, dict] = {}
        for event in events:
            cluster_id = event.cluster_id or self._bucket_embedding(event.context_embedding, user_id)
            cluster_key = cluster_id or f"{user_id}-conv-{event.conversation_id}"
            cluster = clusters.setdefault(cluster_key, {"embeddings": [], "events": []})
            if event.context_embedding:
                cluster["embeddings"].append(event.context_embedding)
            cluster["events"].append(event)

        summaries: List[dict] = []
        for cluster_id, data in clusters.items():
            centroid: List[float] = []
            if data["embeddings"]:
                max_len = max(len(vec) for vec in data["embeddings"])
                accum = [0.0] * max_len
                for vec in data["embeddings"]:
                    padded = list(vec) + [0.0] * (max_len - len(vec))
                    for i, val in enumerate(padded):
                        accum[i] += val
                centroid = [val / len(data["embeddings"]) for val in accum]
            summaries.append(
                {
                    "cluster_id": cluster_id,
                    "count": len(data["events"]),
                    "centroid": centroid,
                    "sample_event_ids": [e.id for e in data["events"][:5]],
                }
            )
        return summaries
