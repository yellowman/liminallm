from __future__ import annotations

import json
import random
import shutil
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence

from liminallm.service.embeddings import deterministic_embedding
from liminallm.service.fs import safe_join
from liminallm.service.tokenizer_utils import DEFAULT_VOCAB_SIZE, vocab_size_from_tokenizer
from liminallm.storage.errors import ConstraintViolation
from liminallm.storage.models import Artifact, PreferenceEvent

from liminallm.logging import get_logger

logger = get_logger(__name__)


class TrainingService:
    """Minimal LoRA adapter training loop following SPEC phase 2."""

    def __init__(self, store, fs_root: str, *, runtime_base_model: Optional[str] = None) -> None:
        self.store = store
        self.fs_root = Path(fs_root)
        self.default_vocab_size = DEFAULT_VOCAB_SIZE
        self._base_vocab_size = DEFAULT_VOCAB_SIZE
        self._adapter_vocab_size: Optional[int] = None
        self.tokenizer = None
        self._tokenizer_error: Optional[str] = None
        self._tokenizer_model: Optional[str] = None
        self.runtime_base_model = runtime_base_model
        self.training_job_cooldown_seconds = 300

    def _ensure_tokenizer(self, base_model: Optional[str]) -> None:
        model_name = base_model or self.runtime_base_model
        if not model_name:
            return
        if self.tokenizer is not None and self._tokenizer_model == model_name:
            return
        if self._tokenizer_error is not None and self._tokenizer_model == model_name:
            return
        try:  # pragma: no cover - optional dependency
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._base_vocab_size = vocab_size_from_tokenizer(
                self.tokenizer, fallback=self.default_vocab_size
            )
            self._tokenizer_model = model_name
            self._tokenizer_error = None
        except Exception as exc:  # pragma: no cover - optional dependency
            self.tokenizer = None
            self._tokenizer_model = model_name
            self._tokenizer_error = str(exc)
            self._base_vocab_size = self.default_vocab_size
            logger.warning("tokenizer_load_failed", base_model=model_name, error=str(exc))

    def _vocab_size(self) -> int:
        if isinstance(self._adapter_vocab_size, int) and self._adapter_vocab_size > 0:
            return self._adapter_vocab_size
        return self._base_vocab_size

    def _apply_adapter_vocab_size(self, adapter: Artifact) -> None:
        self._adapter_vocab_size = None
        if not adapter:
            return
        vocab_size = (adapter.schema or {}).get("vocab_size")
        if isinstance(vocab_size, int) and vocab_size > 0:
            self._adapter_vocab_size = vocab_size

    def _assert_adapter_base(self, adapter: Artifact) -> Artifact:
        runtime_base = self.runtime_base_model
        stored_base = adapter.schema.get("base_model") if adapter and adapter.schema else None
        if runtime_base and stored_base and stored_base != runtime_base:
            migration_plan = {
                "expected_base": runtime_base,
                "stored_base": stored_base,
                "plan": "retrain adapter on active base or distill weights",
            }
            logger.warning(
                "adapter_base_model_mismatch",
                adapter_id=adapter.id,
                stored_base_model=stored_base,
                runtime_base_model=runtime_base,
                migration_plan=migration_plan,
            )
            raise ConstraintViolation("adapter base incompatible", migration_plan)
        if runtime_base and not stored_base:
            updated_schema = dict(adapter.schema)
            updated_schema["base_model"] = runtime_base
            self.store.update_artifact(adapter.id, updated_schema)
            refreshed = self.store.get_artifact(adapter.id)
            return refreshed or adapter
        return adapter

    def ensure_user_adapter(self, user_id: str, *, rank: int = 4, adapter_id_override: Optional[str] = None) -> Artifact:
        existing = [a for a in self.store.list_artifacts(type_filter="adapter") if a.owner_user_id == user_id]
        if adapter_id_override:
            existing = [a for a in existing if a.id == adapter_id_override]
            if not existing:
                raise ConstraintViolation(
                    "adapter override missing",
                    {"adapter_id": adapter_id_override, "user_id": user_id},
                )
        if existing:
            adapter = existing[0]
            runtime_base = self.runtime_base_model
            stored_base = adapter.schema.get("base_model") if adapter.schema else None
            if runtime_base and stored_base and stored_base != runtime_base:
                migration_plan = {
                    "expected_base": runtime_base,
                    "stored_base": stored_base,
                    "plan": "retrain adapter on active base or distill weights",
                }
                logger.warning(
                    "adapter_base_model_mismatch",
                    adapter_id=adapter.id,
                    stored_base_model=stored_base,
                    runtime_base_model=runtime_base,
                    migration_plan=migration_plan,
                )
                raise ConstraintViolation("adapter base incompatible", migration_plan)
            if runtime_base and not stored_base:
                updated_schema = dict(adapter.schema)
                updated_schema["base_model"] = runtime_base
                self.store.update_artifact(adapter.id, updated_schema)
                adapter = self.store.get_artifact(adapter.id) or adapter
            return adapter
        adapter_id = adapter_id_override
        adapter_schema = {
            "kind": "adapter.lora",
            "backend": "local",
            "provider": "local",
            "scope": "per-user",
            "user_id": user_id,
            "base_model": self.runtime_base_model or "jax-base",
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
        adapter_fs_dir = self._adapter_dir(user_id, adapter.id, adapter_schema)
        adapter_schema["fs_dir"] = str(adapter_fs_dir)
        adapter_schema.setdefault("cephfs_dir", str(adapter_fs_dir))
        self.store.update_artifact(adapter.id, adapter_schema)
        return self.store.get_artifact(adapter.id) or adapter

    def train_from_preferences(
        self, user_id: str, adapter_id: Optional[str] = None, cluster_id: Optional[str] = None
    ) -> Optional[dict]:
        adapter = self.ensure_user_adapter(user_id) if not adapter_id else self.store.get_artifact(adapter_id)
        if not adapter:
            raise ConstraintViolation("adapter missing", {"adapter_id": adapter_id})
        adapter = self._assert_adapter_base(adapter)
        self._apply_adapter_vocab_size(adapter)
        events = self.store.list_preference_events(user_id=user_id, feedback="positive", cluster_id=cluster_id)
        if not events:
            return None
        cluster_meta = self._cluster_events(events, user_id)
        dataset_entries = list(self._build_examples(events))
        token_batches = list(self._tokenize_batches(dataset_entries, base_model=adapter.schema.get("base_model")))
        vocab_size = self._vocab_size()
        if adapter.schema.get("vocab_size") != vocab_size:
            adapter_schema = dict(adapter.schema)
            adapter_schema["vocab_size"] = vocab_size
            self.store.update_artifact(adapter.id, adapter_schema)
            adapter = self.store.get_artifact(adapter.id) or adapter
            self._apply_adapter_vocab_size(adapter)
        job = self.store.create_training_job(
            user_id=user_id,
            adapter_id=adapter.id,
            preference_event_ids=[e.id for e in events],
        )
        job_dir = self._job_dir(user_id, adapter.id, job.id, adapter.schema)
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
        adapter_dir = self._adapter_dir(user_id, adapter.id, adapter.schema)
        version_dir = adapter_dir / f"v{next_version:04d}"
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
        updated_schema["fs_dir"] = str(adapter_dir)
        self.store.update_artifact(adapter.id, updated_schema)
        self._update_latest_symlink(adapter_dir, version_dir)
        loss = 1.0 / (1 + len(dataset_entries))
        training_trace = self._run_jax_optax_training(
            weights,
            token_batches,
            params_path=params_path,
            checkpoint_dir=version_dir / "checkpoints",
        )
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

    def record_feedback_event(
        self,
        *,
        user_id: str,
        conversation_id: str,
        message_id: str,
        feedback: str,
        score: Optional[float] = None,
        context_text: Optional[str] = None,
        corrected_text: Optional[str] = None,
        weight: Optional[float] = None,
        explicit_signal: Optional[str] = None,
        routing_trace: Optional[List[dict]] = None,
        adapter_gates: Optional[List[dict]] = None,
    ) -> PreferenceEvent:
        embedding = deterministic_embedding(context_text or "")
        meta = {}
        if routing_trace:
            meta["routing_trace"] = routing_trace
        if adapter_gates:
            meta["adapter_gates"] = adapter_gates
        event = self.store.record_preference_event(
            user_id=user_id,
            conversation_id=conversation_id,
            message_id=message_id,
            feedback=feedback,
            score=score,
            corrected_text=corrected_text,
            context_text=context_text,
            weight=weight,
            context_embedding=embedding,
            explicit_signal=explicit_signal,
            meta=meta or None,
        )
        if feedback in {"positive", "like"} and self._should_enqueue_training_job(user_id):
            adapter = self.ensure_user_adapter(user_id)
            self.store.create_training_job(
                user_id=user_id, adapter_id=adapter.id, preference_event_ids=[event.id], dataset_path=None
            )
        return event

    def _should_enqueue_training_job(self, user_id: str) -> bool:
        active_statuses = {"queued", "running"}
        recent_jobs = self._list_user_training_jobs(user_id)
        for job in recent_jobs:
            if job.status in active_statuses:
                return False
        if not recent_jobs:
            return True
        most_recent = recent_jobs[0]
        cooldown_elapsed = (
            datetime.utcnow() - (most_recent.updated_at or most_recent.created_at)
        ).total_seconds()
        return cooldown_elapsed >= self.training_job_cooldown_seconds

    def _list_user_training_jobs(self, user_id: str) -> List:
        list_fn = getattr(self.store, "list_training_jobs", None)
        if callable(list_fn):
            jobs = list_fn(user_id=user_id)
        elif hasattr(self.store, "training_jobs"):
            jobs = [j for j in getattr(self.store, "training_jobs", {}).values() if j.user_id == user_id]
        else:
            jobs = []
        try:
            jobs.sort(key=lambda j: j.updated_at or j.created_at, reverse=True)
        except Exception as exc:
            logger.warning("training_job_sort_failed", error=str(exc))
            raise
        return jobs

    def summarize_preferences(self, user_id: Optional[str]) -> dict:
        events: List[PreferenceEvent] = []
        status: str = "ok"
        error_detail: Optional[str] = None
        try:
            events = self.store.list_preference_events(user_id=user_id)  # type: ignore[attr-defined]
        except Exception as exc:
            status = "error"
            error_detail = f"preference retrieval failed: {exc}"
            logger.warning("list_preference_events_failed", user_id=user_id, error=str(exc))
        totals = {"positive": 0, "negative": 0, "neutral": 0}
        routing_feedback: dict[str, int] = {}
        for event in events:
            totals[event.feedback] = totals.get(event.feedback, 0) + 1
            if event.meta and event.meta.get("routing_trace"):
                routing_feedback["routing_trace_present"] = routing_feedback.get("routing_trace_present", 0) + 1
        clusters: List[dict] = []
        clusters_error: Optional[str] = None
        if hasattr(self.store, "list_semantic_clusters"):
            try:
                for cluster in self.store.list_semantic_clusters(user_id):  # type: ignore[attr-defined]
                    clusters.append(
                        {
                            "id": cluster.id,
                            "size": cluster.size,
                            "label": cluster.label,
                            "similarity_hint": cluster.description,
                        }
                    )
            except Exception as exc:
                clusters_error = str(exc)
                logger.warning("list_semantic_clusters_failed", user_id=user_id, error=str(exc))
        adapter_candidates = [a for a in self.store.list_artifacts(type_filter="adapter")]  # type: ignore[arg-type]
        adapters = [
            {
                "id": a.id,
                "name": a.name,
                "description": a.description,
                "cluster_id": a.schema.get("cluster_id"),
                "base_model": a.schema.get("base_model"),
            }
            for a in adapter_candidates
        ]
        adapter_state = self._collect_adapter_router_state(user_id)
        if status == "ok" and not events and not clusters and (
            adapter_state.get("status") in {"no_data", "unavailable"}
            or not adapter_state.get("entries")
        ):
            status = "no_data"
        return {
            "status": status,
            "error": error_detail,
            "totals": totals,
            "events": [asdict(e) for e in events[-10:]],
            "clusters": clusters,
            "clusters_status": "error" if clusters_error else ("ok" if clusters else "no_data"),
            "clusters_error": clusters_error,
            "adapters": adapters,
            "routing_feedback": routing_feedback,
            "adapter_router_state": adapter_state,
        }

    def _collect_adapter_router_state(self, user_id: Optional[str]) -> dict:
        if hasattr(self.store, "list_adapter_router_state"):
            try:
                states = self.store.list_adapter_router_state(user_id=user_id)  # type: ignore[attr-defined]
            except Exception as exc:
                logger.warning("list_adapter_router_state_failed", user_id=user_id, error=str(exc))
                return {"status": "error", "error": str(exc)}
            parsed: List[dict] = []
            for state in states or []:
                parsed.append(
                    {
                        "adapter_id": getattr(state, "adapter_id", None),
                        "centroid_vec": getattr(state, "centroid_vec", None),
                        "base_model": getattr(state, "base_model", None),
                        "success_score": getattr(state, "success_score", None),
                        "last_trained_at": getattr(state, "last_trained_at", None),
                    }
                )
            return {"status": "ok" if parsed else "no_data", "entries": parsed}
        return {"status": "unavailable"}

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
        base_model: Optional[str] = None,
    ) -> Iterator[dict]:
        """
        Convert preference_event-derived examples into padded token batches.

        This intentionally keeps tokenization minimal: it accepts any tokenizer
        object with an ``encode`` method and falls back to whitespace splitting
        to avoid hard dependencies. Batches carry shapes so downstream
        backends (e.g., JAX) can preallocate arrays without re-tokenizing.
        """

        self._ensure_tokenizer(base_model)
        vocab_size = max(self._vocab_size(), 1)

        def _encode(text: str) -> List[int]:
            if self.tokenizer is not None:
                return list(self.tokenizer.encode(text, truncation=True, max_length=max_length))
            tokens = text.split()
            return [hash(tok) % vocab_size for tok in tokens[:max_length]]

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

    def _adapter_dir(self, user_id: str, adapter_id: str, adapter_schema: Optional[dict] = None) -> Path:
        adapter_schema = adapter_schema or {}
        explicit = adapter_schema.get("cephfs_dir") or adapter_schema.get("fs_dir")
        if explicit:
            return Path(explicit)
        return safe_join(self.fs_root, f"adapters/{adapter_id}")

    def _job_dir(self, user_id: str, adapter_id: str, job_id: str, adapter_schema: Optional[dict] = None) -> Path:
        return safe_join(self._adapter_dir(user_id, adapter_id, adapter_schema), f"jobs/{job_id}")

    def _update_latest_symlink(self, adapter_dir: Path, version_dir: Path) -> None:
        latest = adapter_dir / "latest"
        if latest.exists() or latest.is_symlink():
            if latest.is_dir() and not latest.is_symlink():
                shutil.rmtree(latest)
            else:
                latest.unlink()
        latest.symlink_to(version_dir, target_is_directory=True)

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

    def _run_jax_optax_training(
        self,
        params: dict,
        batches: Sequence[dict],
        *,
        params_path: Path,
        checkpoint_dir: Optional[Path] = None,
        accumulation_steps: int = 4,
    ) -> dict:
        """
        Train a single LoRA adapter with a supervised loss and checkpoints.

        The loop mirrors the lightweight JAX forward pass used by the
        ``LocalJaxLoRABackend``: embeddings are projected through paired
        ``.A`` / ``.B`` matrices to produce logits and a masked
        cross-entropy loss. Gradients are accumulated across
        ``accumulation_steps`` microbatches before each optimizer update and
        checkpoints are written so the backend can reload trained weights.
        """

        try:
            import jax
            import jax.numpy as jnp
            import optax
        except Exception as exc:
            logger.warning("training_loop_skipped", reason="jax_optax_unavailable", error=str(exc))
            return {"status": "skipped", "reason": "jax/optax not installed"}

        vocab_size = max(self._vocab_size(), 1)
        max_token_id = 0
        for batch in batches:
            for key in ("input_ids", "labels"):
                seqs = batch.get(key) or []
                for seq in seqs:
                    if seq:
                        max_token_id = max(max_token_id, max(seq))
        vocab_size = max(vocab_size, max_token_id + 1)
        hidden_dim = 0
        for name, value in params.items():
            if name.endswith(".A"):
                hidden_dim = max(hidden_dim, len(value[0]) if value else 0)
        hidden_dim = hidden_dim or 16

        emb_table = jnp.sin(
            jnp.arange(vocab_size * hidden_dim, dtype=jnp.float32).reshape(vocab_size, hidden_dim)
            / float(hidden_dim)
        )

        def _flatten_params(param_dict: dict) -> dict:
            return {k: jnp.array(v, dtype=jnp.float32) for k, v in param_dict.items()}

        def _to_python(tree: dict) -> dict:
            return {k: v.tolist() for k, v in tree.items()}

        def _checkpoint(step: int, tree: dict) -> None:
            if not checkpoint_dir:
                return
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            path = checkpoint_dir / f"step_{step:04d}.json"
            path.write_text(json.dumps(_to_python(tree)))

        def _apply_lora(p: dict, embeds: jnp.ndarray) -> jnp.ndarray:
            acc = jnp.zeros_like(embeds)
            for name, mat in p.items():
                if not name.endswith(".A"):
                    continue
                b_key = name.replace(".A", ".B")
                if b_key not in p:
                    continue
                base = embeds @ mat.T
                update = base @ p[b_key].T
                if update.shape[-1] != embeds.shape[-1]:
                    width = embeds.shape[-1]
                    pad = width - update.shape[-1]
                    if pad > 0:
                        update = jnp.pad(update, ((0, 0), (0, 0), (0, pad)))
                    else:
                        update = update[:, :, :width]
                acc = acc + update
            return embeds + acc

        def forward(p: dict, batch: dict) -> jnp.ndarray:
            input_ids = jnp.array(batch["input_ids"], dtype=jnp.int32)
            labels = jnp.array(batch["labels"], dtype=jnp.int32)
            mask = jnp.array(batch.get("attention_mask") or [[1]], dtype=jnp.float32)
            clipped_ids = jnp.clip(input_ids, 0, vocab_size - 1)
            embeds = emb_table[clipped_ids]
            lora_embeds = _apply_lora(p, embeds)
            logits = jnp.einsum("bsh,vh->bsv", lora_embeds, emb_table)
            labels = jnp.clip(labels, 0, vocab_size - 1)
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            nll = -jnp.take_along_axis(log_probs, labels[..., None], axis=-1).squeeze(-1)
            masked = nll * mask
            denom = jnp.maximum(jnp.sum(mask), 1.0)
            return jnp.sum(masked) / denom

        opt = optax.adam(2e-3)
        params_tree = _flatten_params(params)
        opt_state = opt.init(params_tree)
        grad_fn = jax.value_and_grad(forward)
        trace: list[dict] = []
        accum_grads = None
        accum_count = 0

        for step, batch in enumerate(batches, start=1):
            value, grads = grad_fn(params_tree, batch)
            accum_grads = grads if accum_grads is None else jax.tree_util.tree_map(lambda a, b: a + b, accum_grads, grads)
            accum_count += 1
            if accum_count < accumulation_steps and step < len(batches):
                trace.append({"loss": float(value), "shape": batch["shape"], "accumulating": True})
                continue

            mean_grads = jax.tree_util.tree_map(lambda g: g / float(accum_count), accum_grads)
            updates, opt_state = opt.update(mean_grads, opt_state, params_tree)
            params_tree = optax.apply_updates(params_tree, updates)
            trace.append({"loss": float(value), "shape": batch["shape"], "accumulated": accum_count})
            _checkpoint(step, params_tree)
            accum_grads = None
            accum_count = 0

        params_path.write_text(json.dumps(_to_python(params_tree), indent=2))
        return {"status": "ok", "steps": trace[-10:], "final_params_path": str(params_path)}

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
