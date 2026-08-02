from __future__ import annotations

import json
import random
import shutil
import uuid
from contextlib import suppress
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence

from liminallm.config import AdapterMode, get_compatible_adapter_modes
from liminallm.logging import get_logger
from liminallm.service.embeddings import deterministic_embedding
from liminallm.service.fs import PathTraversalError, safe_join
from liminallm.service.tokenizer_utils import (
    DEFAULT_VOCAB_SIZE,
    vocab_size_from_tokenizer,
)
from liminallm.storage.errors import ConstraintViolation
from liminallm.storage.models import POSITIVE_FEEDBACK_VALUES, Artifact, PreferenceEvent

logger = get_logger(__name__)

DEFAULT_TRAIN_BATCH_SIZE = 2
DEFAULT_MAX_TOKEN_LENGTH = 512
DEFAULT_GRAD_ACCUM_STEPS = 4
DEFAULT_LORA_LEARNING_RATE = 2e-3
# SPEC §5.4: eval gate. Every Nth example is held out once the dataset is big
# enough, and promotion requires the holdout loss to improve by at least the
# relative margin below.
DEFAULT_TRAIN_EPOCHS = 5
HOLDOUT_MIN_EXAMPLES = 5
HOLDOUT_EVERY_N = 5
EVAL_MIN_RELATIVE_IMPROVEMENT = 0.01
# SPEC §7.5: cap teacher-distillation calls per job to bound cost.
MAX_DISTILL_EXAMPLES = 32


class TrainingService:
    """Minimal LoRA adapter training loop following SPEC phase 2.

    Supports dual-mode operation per SPEC §5 clarification:
    - LOCAL mode: Train and store weights on filesystem for LocalJaxLoRABackend
    - REMOTE mode: Train locally, then sync to external adapter server
    - PROMPT mode: No weights; inject behavior via system prompt
    - HYBRID mode: Store local weights + prompt fallback for API backends
    """

    def __init__(
        self,
        store,
        fs_root: str,
        *,
        runtime_base_model: Optional[str] = None,
        default_adapter_mode: str = AdapterMode.HYBRID,
        backend_mode: Optional[str] = None,
        max_active_training_jobs: int = 10,
        teacher=None,
        distillation_enabled: bool = False,
    ) -> None:
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
        self.default_adapter_mode = default_adapter_mode
        self.backend_mode = backend_mode
        self._compatible_modes = get_compatible_adapter_modes(backend_mode or "openai")
        self.max_active_training_jobs = max_active_training_jobs
        # SPEC §7.5: optional teacher LLM used to distill preference events
        # into cleaner training targets before tokenization.
        self.teacher = teacher
        self.distillation_enabled = distillation_enabled

    def _safe_int(self, value: object, default: int, *, context: str) -> int:
        """Coerce values to int with fallback to avoid ValueError crashes (Issue 39.3)."""

        try:
            return int(value)
        except (TypeError, ValueError):
            logger.warning("training_int_parse_failed", context=context, value=value)
            return default

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
            # Fall back to the deterministic hash tokenizer in
            # _tokenize_batches rather than failing the job: a missing
            # optional dependency (or a base model without a HF tokenizer)
            # degrades fidelity, and the eval gate still judges the result.
            self.tokenizer = None
            self._tokenizer_model = model_name
            self._tokenizer_error = str(exc)
            self._base_vocab_size = self.default_vocab_size
            logger.warning(
                "tokenizer_load_failed", base_model=model_name, error=str(exc)
            )

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
        stored_base = (
            adapter.schema.get("base_model") if adapter and adapter.schema else None
        )
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

    def ensure_user_adapter(
        self,
        user_id: str,
        *,
        rank: int = 4,
        adapter_id_override: Optional[str] = None,
        adapter_mode: Optional[str] = None,
    ) -> Artifact:
        """Ensure a user has an adapter, creating one if needed.

        Args:
            user_id: Owner of the adapter
            rank: LoRA rank for new adapters
            adapter_id_override: Specific adapter ID to use
            adapter_mode: Override default adapter mode (local/remote/prompt/hybrid)
        """
        # Pass owner_user_id so the user's own private adapters are returned;
        # without it list_artifacts only yields global artifacts and this method
        # would never find an existing adapter (creating a duplicate each call).
        existing = [
            a
            for a in self.store.list_artifacts(
                type_filter="adapter", owner_user_id=user_id
            )
            if a.owner_user_id == user_id
        ]
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
            needs_update = False
            updated_schema = dict(adapter.schema)
            if runtime_base and not stored_base:
                updated_schema["base_model"] = runtime_base
                needs_update = True
            # Migrate existing adapters to include mode if missing
            if "mode" not in updated_schema:
                updated_schema["mode"] = self._infer_adapter_mode(updated_schema)
                needs_update = True
            if needs_update:
                self.store.update_artifact(adapter.id, updated_schema)
                adapter = self.store.get_artifact(adapter.id) or adapter
            return adapter

        # Create new adapter with explicit mode
        resolved_mode = adapter_mode or self.default_adapter_mode
        _ = adapter_id_override  # Reserved for future use in adapter creation
        adapter_schema = {
            "kind": "adapter.lora",
            "mode": resolved_mode,  # Explicit adapter mode
            "backend": self._mode_to_backend(resolved_mode),
            "provider": self._mode_to_provider(resolved_mode),
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
        # Only set fs_dir for modes that use local storage
        if resolved_mode in {AdapterMode.LOCAL, AdapterMode.HYBRID}:
            adapter_fs_dir = self._adapter_dir(user_id, adapter.id, adapter_schema)
            adapter_schema["fs_dir"] = str(adapter_fs_dir)
            adapter_schema.setdefault("cephfs_dir", str(adapter_fs_dir))
        self.store.update_artifact(adapter.id, adapter_schema)
        return self.store.get_artifact(adapter.id) or adapter

    def _infer_adapter_mode(self, schema: dict) -> str:
        """Infer adapter mode from legacy schema fields."""
        backend = (schema.get("backend") or "").lower()
        provider = (schema.get("provider") or "").lower()

        if backend in {"prompt", "prompt_distill"}:
            return AdapterMode.PROMPT
        if backend in {"local", "local_lora"} or provider == "local":
            # If has prompt_instructions, it's hybrid
            if schema.get("prompt_instructions") or schema.get("behavior_prompt"):
                return AdapterMode.HYBRID
            return AdapterMode.LOCAL
        if backend in {"api", "remote"} or schema.get("remote_model_id"):
            return AdapterMode.REMOTE
        # Default to hybrid for backwards compatibility
        return AdapterMode.HYBRID

    def _mode_to_backend(self, mode: str) -> str:
        """Map adapter mode to backend field value."""
        if mode == AdapterMode.LOCAL:
            return "local"
        if mode == AdapterMode.REMOTE:
            return "api"
        if mode == AdapterMode.PROMPT:
            return "prompt"
        # HYBRID uses local backend with prompt fallback
        return "hybrid"

    def _mode_to_provider(self, mode: str) -> str:
        """Map adapter mode to provider field value."""
        if mode == AdapterMode.LOCAL:
            return "local"
        if mode == AdapterMode.REMOTE:
            return self.backend_mode or "api"
        if mode == AdapterMode.PROMPT:
            return "prompt"
        return "hybrid"

    def train_from_preferences(
        self,
        user_id: str,
        adapter_id: Optional[str] = None,
        cluster_id: Optional[str] = None,
        *,
        job_id: Optional[str] = None,
    ) -> Optional[dict]:
        adapter = (
            self.ensure_user_adapter(user_id)
            if not adapter_id
            else self.store.get_artifact(adapter_id)
        )
        if not adapter:
            raise ConstraintViolation("adapter missing", {"adapter_id": adapter_id})
        adapter_scope = (adapter.schema or {}).get("scope")
        caller = self.store.get_user(user_id)
        caller_tenant = caller.tenant_id if caller else None
        if adapter.owner_user_id and adapter.owner_user_id != user_id:
            # Tenant-scoped skill adapters are pooled across contributors, so
            # their nominal owner is bookkeeping; any caller inside the same
            # tenant may train them. Everything else stays owner-only.
            owner = self.store.get_user(adapter.owner_user_id)
            owner_tenant = owner.tenant_id if owner else None
            same_tenant = bool(caller_tenant) and caller_tenant == owner_tenant
            if not (adapter_scope == "tenant" and same_tenant):
                raise ConstraintViolation(
                    "adapter ownership mismatch", {"adapter_id": adapter.id}
                )
        adapter = self._assert_adapter_base(adapter)
        self._apply_adapter_vocab_size(adapter)
        adapter_schema_now = adapter.schema or {}
        adapter_cluster = adapter_schema_now.get("cluster_id") or cluster_id
        # SPEC §7.3: skill adapters (cluster-bound, not owned by a single user)
        # pool positive events across every contributor to the cluster - one
        # user's thumbs are too sparse to train weights on. Persona adapters
        # stay strictly per-user.
        # Pooling is decided by scope, not ownership: tenant-scoped skill
        # adapters carry a nominal owner for visibility purposes.
        pooled_skill = bool(adapter_cluster) and (
            adapter_scope == "tenant" or not adapter.owner_user_id
        )
        if pooled_skill:
            # CLAUDE.md / SPEC §12: pooling must never cross a tenant boundary.
            # Scope to the requesting user's tenant so one tenant's content can
            # never end up in weights another tenant is served.
            tenant_id = (adapter.schema or {}).get("tenant_id") or caller_tenant
            if not tenant_id:
                logger.warning(
                    "pooled_training_tenant_unresolved",
                    user_id=user_id,
                    adapter_id=adapter.id,
                )
                raise ConstraintViolation(
                    "cannot resolve tenant for pooled training",
                    {"user_id": user_id, "adapter_id": adapter.id},
                )
            events = self.store.list_preference_events(
                user_id=None,
                feedback=POSITIVE_FEEDBACK_VALUES,
                cluster_id=adapter_cluster,
                tenant_id=tenant_id,
            )
        else:
            events = self.store.list_preference_events(
                user_id=user_id,
                feedback=POSITIVE_FEEDBACK_VALUES,
                cluster_id=cluster_id,
            )
        if not events:
            return None
        cluster_meta = self._cluster_events(events, user_id)
        dataset_entries = list(self._build_examples(events))
        dataset_entries = self._maybe_distill(dataset_entries)
        # SPEC §5.4: hold out a slice of examples so promotion is gated on
        # measured improvement rather than "training ran without raising".
        train_entries, holdout_entries = self._split_holdout(dataset_entries)
        token_batches = list(
            self._tokenize_batches(
                train_entries, base_model=adapter.schema.get("base_model")
            )
        )
        eval_batches = list(
            self._tokenize_batches(
                holdout_entries, base_model=adapter.schema.get("base_model")
            )
        ) if holdout_entries else []
        vocab_size = self._vocab_size()
        if adapter.schema.get("vocab_size") != vocab_size:
            adapter_schema = dict(adapter.schema)
            adapter_schema["vocab_size"] = vocab_size
            self.store.update_artifact(adapter.id, adapter_schema)
            adapter = self.store.get_artifact(adapter.id) or adapter
            self._apply_adapter_vocab_size(adapter)
        # Reuse a job the worker already claimed; only create one when invoked
        # directly (job_id is None), so the worker path can't spawn duplicates.
        if job_id is None:
            job = self.store.create_training_job(
                user_id=user_id,
                adapter_id=adapter.id,
                preference_event_ids=[e.id for e in events],
            )
            job_id = job.id
        job_dir = self._job_dir(user_id, adapter.id, job_id, adapter.schema)
        job_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = job_dir / "dataset.jsonl"
        with dataset_path.open("w") as f:
            for row in dataset_entries:
                f.write(json.dumps(row) + "\n")
        self.store.update_training_job(
            job_id,
            dataset_path=str(dataset_path),
            meta={
                "token_batches": [b["shape"] for b in token_batches],
                "clusters": cluster_meta,
            },
        )
        next_version = self._safe_int(
            adapter.schema.get("current_version", 0),
            default=0,
            context="adapter_current_version",
        ) + 1
        adapter_dir = self._adapter_dir(user_id, adapter.id, adapter.schema)
        version_dir = adapter_dir / f"v{next_version:04d}"
        version_dir.mkdir(parents=True, exist_ok=True)
        params_path = version_dir / "params.json"
        weights = self._init_lora_weights(
            rank=self._safe_int(
                adapter.schema.get("rank", 4),
                default=4,
                context="adapter_rank",
            ),
            layers=adapter.schema.get("layers", []),
            matrices=adapter.schema.get("matrices", []),
        )
        params_path.write_text(json.dumps(weights, indent=2))
        metadata = {
            "adapter_id": adapter.id,
            "user_id": user_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "version": next_version,
            "dataset_path": str(dataset_path),
            "preference_events": [asdict(e) for e in events],
            "token_batches": [b["shape"] for b in token_batches],
            "clusters": cluster_meta,
        }
        # default=str: preference events carry datetime fields
        (version_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, default=str)
        )
        training_trace = self._run_jax_optax_training(
            weights,
            token_batches,
            eval_batches=eval_batches,
            params_path=params_path,
            checkpoint_dir=version_dir / "checkpoints",
        )
        # SPEC §5.4/§10.2: promotion is gated on the holdout eval, not on
        # "training ran without raising". A skipped or regressed run leaves
        # the artifact untouched (prompt-mode skill adapters simply stay on
        # the prompt rung of the ladder).
        gate = self._promotion_gate(training_trace, holdout_count=len(holdout_entries))
        if gate["promoted"]:
            updated_schema = dict(adapter.schema)
            updated_schema["current_version"] = next_version
            updated_schema["fs_dir"] = str(adapter_dir)
            # Adapter ladder (SPEC §5.6): a prompt-born skill adapter that
            # passes its first eval graduates to hybrid - trained weights
            # with the prompt instructions kept as portable fallback.
            if updated_schema.get("mode") == AdapterMode.PROMPT:
                updated_schema["mode"] = AdapterMode.HYBRID
                updated_schema["backend"] = self._mode_to_backend(AdapterMode.HYBRID)
                updated_schema["provider"] = self._mode_to_provider(AdapterMode.HYBRID)
                updated_schema.setdefault("lifecycle", {})
                if isinstance(updated_schema["lifecycle"], dict):
                    updated_schema["lifecycle"]["stage"] = "weights"
            self.store.update_artifact(adapter.id, updated_schema)
            self._update_latest_symlink(adapter_dir, version_dir)
        else:
            # Quarantine the un-promoted weights. Leaving them in place as
            # vNNNN/ let the serving backend pick them up as "newest on disk",
            # which silently defeated the gate; they are kept under rejected/
            # for debugging instead of deleted.
            try:
                rejected_root = adapter_dir / "rejected"
                rejected_root.mkdir(parents=True, exist_ok=True)
                destination = rejected_root / version_dir.name
                if destination.exists():
                    shutil.rmtree(destination, ignore_errors=True)
                shutil.move(str(version_dir), str(destination))
            except Exception as exc:
                # Never fail the job over cleanup, but do not leave live
                # weights behind either - drop them and log loudly.
                logger.warning(
                    "adapter_rejected_quarantine_failed",
                    adapter_id=adapter.id,
                    version_dir=str(version_dir),
                    error=str(exc),
                )
                shutil.rmtree(version_dir, ignore_errors=True)
            logger.info(
                "adapter_promotion_gated",
                adapter_id=adapter.id,
                job_id=job_id,
                reason=gate["reason"],
            )
        # Extract actual loss from JAX training instead of using heuristic
        # Per SPEC §5.4: metrics = loss and preference alignment rate
        loss = 1.0 / (1 + len(dataset_entries))  # Fallback heuristic
        if training_trace.get("status") == "ok" and training_trace.get("steps"):
            # Use actual final loss from training if available
            final_step = training_trace["steps"][-1]
            if isinstance(final_step, dict) and "loss" in final_step:
                actual_loss = final_step.get("loss")
                if isinstance(actual_loss, (int, float)) and actual_loss >= 0:
                    loss = float(actual_loss)
        self.store.update_training_job(
            job_id,
            status="succeeded",
            loss=loss,
            new_version=next_version if gate["promoted"] else None,
            meta={
                "token_batches": [b["shape"] for b in token_batches],
                "jax_trace": training_trace,
                "clusters": cluster_meta,
                "eval_gate": gate,
                "pooled_skill": pooled_skill,
                "distilled": sum(1 for e in dataset_entries if e.get("distilled")),
            },
        )
        return {
            "job_id": job_id,
            "adapter_id": adapter.id,
            "version_dir": str(version_dir),
            "loss": loss,
            "token_batches": token_batches,
            "jax_trace": training_trace,
            "clusters": cluster_meta,
            "eval_gate": gate,
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
        notes: Optional[str] = None,
    ) -> PreferenceEvent:
        embedding = deterministic_embedding(context_text or "")
        meta = {}
        if routing_trace:
            meta["routing_trace"] = routing_trace
        if adapter_gates:
            meta["adapter_gates"] = adapter_gates
        if notes:
            meta["notes"] = notes
        normalized_score = self._normalize_feedback_score(
            feedback=feedback, explicit_signal=explicit_signal, score=score
        )
        event = self.store.record_preference_event(
            user_id=user_id,
            conversation_id=conversation_id,
            message_id=message_id,
            feedback=feedback,
            score=normalized_score,
            corrected_text=corrected_text,
            context_text=context_text,
            weight=weight,
            context_embedding=embedding,
            explicit_signal=explicit_signal,
            meta=meta or None,
        )
        if feedback in POSITIVE_FEEDBACK_VALUES and self._should_enqueue_training_job(
            user_id
        ):
            adapter = self.ensure_user_adapter(user_id)
            self.store.create_training_job(
                user_id=user_id,
                adapter_id=adapter.id,
                preference_event_ids=[event.id],
                dataset_path=None,
            )
        return event

    def _should_enqueue_training_job(self, user_id: str) -> bool:
        active_statuses = {"queued", "running"}
        recent_jobs = self._list_user_training_jobs(user_id)
        for job in recent_jobs:
            if job.status in active_statuses:
                return False
        global_jobs = self.store.list_training_jobs(status=None)
        if global_jobs:
            active_global = [j for j in global_jobs if j.status in active_statuses]
            if len(active_global) >= self.max_active_training_jobs:
                logger.info(
                    "training_job_global_limit_reached",
                    active=len(active_global),
                    max_allowed=self.max_active_training_jobs,
                )
                return False
        if not recent_jobs:
            return True
        most_recent = recent_jobs[0]
        cooldown_elapsed = (
            datetime.now(timezone.utc) - (most_recent.updated_at or most_recent.created_at)
        ).total_seconds()
        return cooldown_elapsed >= self.training_job_cooldown_seconds

    def _list_user_training_jobs(self, user_id: str) -> List:
        jobs = self.store.list_training_jobs(user_id=user_id)
        try:
            jobs.sort(key=lambda j: j.updated_at or j.created_at, reverse=True)
        except Exception as exc:
            logger.warning("training_job_sort_failed", error=str(exc))
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
            logger.warning(
                "list_preference_events_failed", user_id=user_id, error=str(exc)
            )
        totals = {"positive": 0, "negative": 0, "neutral": 0}
        routing_feedback: dict[str, int] = {}
        for event in events:
            totals[event.feedback] = totals.get(event.feedback, 0) + 1
            if event.meta and event.meta.get("routing_trace"):
                routing_feedback["routing_trace_present"] = (
                    routing_feedback.get("routing_trace_present", 0) + 1
                )
        clusters: List[dict] = []
        clusters_error: Optional[str] = None
        try:
            for cluster in self.store.list_semantic_clusters(user_id):
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
            logger.warning(
                "list_semantic_clusters_failed", user_id=user_id, error=str(exc)
            )
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
        if (
            status == "ok"
            and not events
            and not clusters
            and (
                adapter_state.get("status") in {"no_data", "unavailable"}
                or not adapter_state.get("entries")
            )
        ):
            status = "no_data"
        return {
            "status": status,
            "error": error_detail,
            "totals": totals,
            "events": [asdict(e) for e in events[-10:]],
            "clusters": clusters,
            "clusters_status": (
                "error" if clusters_error else ("ok" if clusters else "no_data")
            ),
            "clusters_error": clusters_error,
            "adapters": adapters,
            "routing_feedback": routing_feedback,
            "adapter_router_state": adapter_state,
        }

    def _collect_adapter_router_state(self, user_id: Optional[str]) -> dict:
        try:
            states = self.store.list_adapter_router_state(user_id=user_id)
        except Exception as exc:
            logger.warning(
                "list_adapter_router_state_failed", user_id=user_id, error=str(exc)
            )
            return {"status": "error", "error": str(exc)}
        parsed = [
            {
                "adapter_id": state.adapter_id,
                "centroid_vec": state.centroid_vec,
                "base_model": state.base_model,
                "success_score": state.success_score,
                "last_trained_at": state.last_trained_at,
            }
            for state in states or []
        ]
        return {"status": "ok" if parsed else "no_data", "entries": parsed}

    def _maybe_distill(self, entries: List[dict]) -> List[dict]:
        """Rewrite training targets through the teacher LLM (SPEC §7.5).

        Small student models train better on clean exemplars than on raw chat
        transcripts. When a teacher is configured, each example's target is
        rewritten into an ideal response; failures fall back to the raw target
        so distillation can never lose data. Capped to bound teacher cost.
        """
        if not (self.distillation_enabled and self.teacher):
            return entries
        out: List[dict] = []
        for idx, entry in enumerate(entries):
            if idx >= MAX_DISTILL_EXAMPLES:
                out.append(entry)
                continue
            try:
                resp = self.teacher.generate(
                    "You are preparing supervised training data for a smaller "
                    "model. Rewrite the assistant response below into an ideal, "
                    "concise exemplar that preserves its meaning and any facts. "
                    "Reply with the improved response only.\n\n"
                    f"CONVERSATION:\n{entry.get('prompt', '')[-4000:]}\n\n"
                    f"RESPONSE TO IMPROVE:\n{entry.get('target', '')[:4000]}",
                    adapters=[],
                    context_snippets=[],
                    history=[],
                )
                improved = (resp or {}).get("content", "").strip()
                if improved:
                    entry = {**entry, "target": improved, "distilled": True}
            except Exception as exc:
                logger.warning("distillation_failed", error=str(exc))
            out.append(entry)
        return out

    def _split_holdout(self, entries: List[dict]) -> tuple[List[dict], List[dict]]:
        """Deterministically hold out every Nth example for the eval gate."""
        if len(entries) < HOLDOUT_MIN_EXAMPLES:
            return entries, []
        train: List[dict] = []
        holdout: List[dict] = []
        for idx, entry in enumerate(entries):
            if idx % HOLDOUT_EVERY_N == HOLDOUT_EVERY_N - 1:
                holdout.append(entry)
            else:
                train.append(entry)
        return train, holdout

    def _promotion_gate(self, trace: dict, *, holdout_count: int) -> dict:
        """Decide whether trained weights may be promoted (SPEC §5.4).

        Rules:
        - training skipped/failed -> never promote (a zero-init adapter must
          not become "latest"; prompt-mode adapters stay on the prompt rung).
        - holdout present -> promote only when holdout loss improved by at
          least EVAL_MIN_RELATIVE_IMPROVEMENT.
        - dataset too small for a holdout -> promote on completed training,
          recording that the gate ran without an eval.
        """
        status = trace.get("status")
        if status != "ok":
            return {
                "promoted": False,
                "reason": f"training did not run (status={status or 'unknown'})",
                "holdout_examples": holdout_count,
            }
        eval_before = trace.get("eval_before")
        eval_after = trace.get("eval_after")
        if holdout_count and isinstance(eval_before, (int, float)) and isinstance(
            eval_after, (int, float)
        ):
            denom = max(abs(eval_before), 1e-8)
            improvement = (eval_before - eval_after) / denom
            promoted = improvement >= EVAL_MIN_RELATIVE_IMPROVEMENT
            return {
                "promoted": promoted,
                "reason": (
                    f"holdout loss {eval_before:.4f} -> {eval_after:.4f} "
                    f"({improvement:+.2%})"
                ),
                "holdout_examples": holdout_count,
                "eval_before": eval_before,
                "eval_after": eval_after,
                "improvement": improvement,
            }
        return {
            "promoted": True,
            "reason": "no holdout (dataset below eval threshold)",
            "holdout_examples": holdout_count,
        }

    def _build_examples(self, events: Iterable[PreferenceEvent]) -> Iterable[dict]:
        # Issue 18.1: Dedupe by (conversation_id, message_id) per SPEC §18
        seen: set[tuple[str, str]] = set()
        for event in events:
            key = (event.conversation_id or "", event.message_id or "")
            if key in seen:
                continue
            seen.add(key)
            messages = self.store.list_messages(
                event.conversation_id, limit=200, user_id=event.user_id
            )
            prompt_chunks: List[str] = []
            target_text = event.corrected_text
            cluster_id = event.cluster_id or self._bucket_embedding(
                event.context_embedding, event.user_id
            )
            # Issue 18.2: Exclude target message from prompt (SFT principle)
            for msg in messages:
                if msg.id == event.message_id:
                    # This is the target message - extract for training target, not prompt
                    if not target_text:
                        target_text = msg.content
                    continue  # Don't include target in prompt
                prompt_chunks.append(f"{msg.role.upper()}: {msg.content}")
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
        batch_size: int = DEFAULT_TRAIN_BATCH_SIZE,
        max_length: int = DEFAULT_MAX_TOKEN_LENGTH,
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
                return list(
                    self.tokenizer.encode(text, truncation=True, max_length=max_length)
                )
            # Use deterministic hashing instead of Python's randomized hash()
            tokens = text.split()

            def stable_hash(s: str) -> int:
                h = 0
                for ch in s:
                    h = (h * 31 + ord(ch)) & 0xFFFFFFFF
                return h

            return [stable_hash(tok) % vocab_size for tok in tokens[:max_length]]

        for i in range(0, len(dataset_entries), batch_size):
            batch = dataset_entries[i : i + batch_size]
            # Causal-LM SFT layout: one sequence of prompt+target per example,
            # next-token labels, and the loss masked to the target span so the
            # adapter learns the response, not the prompt. (The previous
            # layout padded inputs to prompt length and labels to target
            # length, which could never broadcast in the loss.)
            seqs: List[tuple[List[int], int]] = []
            for row in batch:
                prompt_tokens = _encode(row["prompt"])
                target_tokens = _encode(row["target"]) or [0]
                full = (prompt_tokens + target_tokens)[: max_length + 1]
                if len(full) < 2:
                    full = full + [0] * (2 - len(full))
                prompt_len = min(len(prompt_tokens), len(full) - 1)
                seqs.append((full, prompt_len))
            seq_len = max(len(full) - 1 for full, _ in seqs)
            input_ids: List[List[int]] = []
            labels: List[List[int]] = []
            loss_mask: List[List[float]] = []
            for full, prompt_len in seqs:
                inp = full[:-1]
                lab = full[1:]
                # Label position j predicts token full[j+1]; that token is
                # part of the target once j+1 >= prompt_len.
                mask = [1.0 if (j + 1) >= prompt_len else 0.0 for j in range(len(lab))]
                pad = seq_len - len(inp)
                input_ids.append(inp + [0] * pad)
                labels.append(lab + [0] * pad)
                loss_mask.append(mask + [0.0] * pad)
            yield {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": loss_mask,
                "shape": {
                    "batch": len(batch),
                    "seq_len": seq_len,
                },
            }

    def _adapter_dir(
        self, user_id: str, adapter_id: str, adapter_schema: Optional[dict] = None
    ) -> Path:
        adapter_schema = adapter_schema or {}
        explicit = adapter_schema.get("cephfs_dir") or adapter_schema.get("fs_dir")
        if explicit:
            candidate = Path(explicit)
            base_root = Path(self.fs_root).resolve()
            resolved = (
                candidate
                if candidate.is_absolute()
                else safe_join(base_root, str(candidate))
            )
            if base_root not in resolved.parents and resolved != base_root:
                raise PathTraversalError(
                    "adapter directory must reside under shared_fs_root"
                )
            return resolved
        return safe_join(self.fs_root, f"adapters/{adapter_id}")

    def _job_dir(
        self,
        user_id: str,
        adapter_id: str,
        job_id: str,
        adapter_schema: Optional[dict] = None,
    ) -> Path:
        return safe_join(
            self._adapter_dir(user_id, adapter_id, adapter_schema), f"jobs/{job_id}"
        )

    def _update_latest_symlink(self, adapter_dir: Path, version_dir: Path) -> None:
        latest = adapter_dir / "latest"
        temp = adapter_dir / f".latest.{uuid.uuid4()}"
        try:
            if temp.exists() or temp.is_symlink():
                temp.unlink()
            temp.symlink_to(version_dir, target_is_directory=True)
            temp.replace(latest)
        except OSError as exc:
            logger.warning(
                "update_latest_symlink_failed",
                adapter_dir=str(adapter_dir),
                version_dir=str(version_dir),
                error=str(exc),
            )
            with suppress(FileNotFoundError):
                temp.unlink(missing_ok=True)  # type: ignore[arg-type]
            raise

    @staticmethod
    def _normalize_feedback_score(
        *, feedback: str, explicit_signal: Optional[str], score: Optional[float]
    ) -> Optional[float]:
        """Normalize feedback scores to [-1, 1] per SPEC §2.6.

        - If a score is provided, clamp to [-1, 1].
        - Otherwise, derive from feedback/explicit_signal strings.
        """

        if score is not None:
            try:
                normalized = float(score)
            except (TypeError, ValueError):
                return None
            return max(-1.0, min(1.0, normalized))

        signal = (explicit_signal or feedback or "").lower()
        if signal in {"positive", "like", "always"}:
            return 1.0
        if signal in {"negative", "dislike", "never"}:
            return -1.0
        if signal == "neutral":
            return 0.0
        return None

    def _init_lora_weights(
        self, rank: int, layers: List[int], matrices: List[str]
    ) -> dict:
        weights: dict[str, list[list[float]]] = {}
        hidden_dim = max(rank * 4, 8)
        for layer in layers:
            for matrix in matrices:
                key_a = f"layer_{layer}.{matrix}.A"
                key_b = f"layer_{layer}.{matrix}.B"
                weights[key_a] = [
                    [random.uniform(-0.01, 0.01) for _ in range(hidden_dim)]
                    for _ in range(rank)
                ]
                weights[key_b] = [
                    [random.uniform(-0.01, 0.01) for _ in range(rank)]
                    for _ in range(hidden_dim)
                ]
        return weights

    def _run_jax_optax_training(
        self,
        params: dict,
        batches: Sequence[dict],
        *,
        eval_batches: Sequence[dict] = (),
        params_path: Path,
        checkpoint_dir: Optional[Path] = None,
        accumulation_steps: int = DEFAULT_GRAD_ACCUM_STEPS,
        learning_rate: float = DEFAULT_LORA_LEARNING_RATE,
        epochs: int = DEFAULT_TRAIN_EPOCHS,
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
            logger.warning(
                "training_loop_skipped", reason="jax_optax_unavailable", error=str(exc)
            )
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
            jnp.arange(vocab_size * hidden_dim, dtype=jnp.float32).reshape(
                vocab_size, hidden_dim
            )
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
            nll = -jnp.take_along_axis(log_probs, labels[..., None], axis=-1).squeeze(
                -1
            )
            masked = nll * mask
            denom = jnp.maximum(jnp.sum(mask), 1.0)
            return jnp.sum(masked) / denom

        def _eval_loss(p: dict) -> Optional[float]:
            """Mean forward loss over the holdout batches (no gradients)."""
            if not eval_batches:
                return None
            losses = [float(forward(p, batch)) for batch in eval_batches]
            return sum(losses) / len(losses)

        opt = optax.adam(learning_rate)
        params_tree = _flatten_params(params)
        opt_state = opt.init(params_tree)
        grad_fn = jax.value_and_grad(forward)
        trace: list[dict] = []
        accum_grads = None
        accum_count = 0
        eval_before = _eval_loss(params_tree)

        # Small adapters on small datasets need several passes to converge;
        # accumulation flushes at each epoch boundary so an epoch always ends
        # in an optimizer update.
        global_step = 0
        for epoch in range(1, max(epochs, 1) + 1):
            for step, batch in enumerate(batches, start=1):
                global_step += 1
                value, grads = grad_fn(params_tree, batch)
                accum_grads = (
                    grads
                    if accum_grads is None
                    else jax.tree_util.tree_map(lambda a, b: a + b, accum_grads, grads)
                )
                accum_count += 1
                if accum_count < accumulation_steps and step < len(batches):
                    trace.append(
                        {
                            "loss": float(value),
                            "epoch": epoch,
                            "shape": batch["shape"],
                            "accumulating": True,
                        }
                    )
                    continue

                mean_grads = jax.tree_util.tree_map(
                    lambda g: g / float(accum_count), accum_grads
                )
                updates, opt_state = opt.update(mean_grads, opt_state, params_tree)
                params_tree = optax.apply_updates(params_tree, updates)
                trace.append(
                    {
                        "loss": float(value),
                        "epoch": epoch,
                        "shape": batch["shape"],
                        "accumulated": accum_count,
                    }
                )
                _checkpoint(global_step, params_tree)
                accum_grads = None
                accum_count = 0

        params_path.write_text(json.dumps(_to_python(params_tree), indent=2))
        return {
            "status": "ok",
            "steps": trace[-10:],
            "final_params_path": str(params_path),
            "eval_before": eval_before,
            "eval_after": _eval_loss(params_tree),
        }

    def _bucket_embedding(
        self, embedding: Sequence[float], user_id: str
    ) -> Optional[str]:
        if not embedding:
            return None
        # Use deterministic hashing instead of Python's randomized hash()
        rounded = tuple(round(v, 1) for v in embedding[:8])
        h = 0
        for val in rounded:
            # Convert float to string for consistent hashing
            for ch in str(val):
                h = (h * 31 + ord(ch)) & 0xFFFFFFFF
        return f"{user_id}-c{h % 1_000_000:06d}"

    def _cluster_events(
        self, events: Sequence[PreferenceEvent], user_id: str
    ) -> List[dict]:
        clusters: dict[str, dict] = {}
        for event in events:
            cluster_id = event.cluster_id or self._bucket_embedding(
                event.context_embedding, user_id
            )
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
