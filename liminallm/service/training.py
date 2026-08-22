from __future__ import annotations

import json
import random
import shutil
import uuid
from contextlib import suppress
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Optional, Sequence

from liminallm.config import AdapterMode, get_compatible_adapter_modes
from liminallm.logging import get_logger
from liminallm.service import local_format, transformer
from liminallm.service.embeddings import deterministic_embedding
from liminallm.service.fs import adapter_root, safe_join
from liminallm.service.tokenizer_utils import (
    DEFAULT_VOCAB_SIZE,
    vocab_size_from_tokenizer,
)
from liminallm.storage.errors import ConstraintViolation
from liminallm.storage.models import (
    POSITIVE_FEEDBACK_VALUES,
    Artifact,
    ConfigPatchAudit,
    PreferenceEvent,
)

logger = get_logger(__name__)

DEFAULT_TRAIN_BATCH_SIZE = 2
DEFAULT_MAX_TOKEN_LENGTH = 512
DEFAULT_GRAD_ACCUM_STEPS = 4
DEFAULT_LORA_LEARNING_RATE = 2e-3
# SPEC §5.4.4: the loss carries an L2 term over the LoRA parameters. Small
# adapters on small datasets overfit quickly, and the regularizer is what
# keeps a passing eval gate meaningful rather than memorized.
LORA_L2_LAMBDA = 1e-4
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
        # Resolved lazily: False = not looked at yet, None = looked and absent.
        self._base_config_state: Any = False
        self._base_checkpoint_state: Any = False

    def _base_model_dir(self) -> Optional[str]:
        return self.runtime_base_model

    def _base_config(self):
        """The frozen base model's config, or None. Cheap and cached."""
        if self._base_config_state is not False:
            return self._base_config_state
        directory = self._base_model_dir()
        self._base_config_state = (
            transformer.load_config(directory) if directory else None
        )
        return self._base_config_state

    def _base_checkpoint(self):
        """(config, params) for the frozen base model, or None.

        Loaded once per service: training needs the real weights to compute a
        real gradient, and re-reading gigabytes per job would dominate the
        run.
        """
        if self._base_checkpoint_state is not False:
            return self._base_checkpoint_state
        directory = self._base_model_dir()
        self._base_checkpoint_state = None
        if directory and transformer.checkpoint_available(directory):
            try:
                self._base_checkpoint_state = transformer.load_checkpoint(directory)
            except Exception as exc:  # noqa: BLE001 - skip, never train blind
                logger.error(
                    "training_base_checkpoint_failed",
                    base_model=directory,
                    error=str(exc),
                )
        return self._base_checkpoint_state

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
        config = self._base_config()
        if config is not None:
            # Same rule serving follows: the checkpoint's embedding table
            # defines the only ids that mean anything, so tokenization must
            # land inside it rather than inside a default.
            return config.vocab_size
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
        # One identity rule for both ends of the ladder (SPEC §5.1). Raw
        # string inequality made a checkout path and its own directory name
        # two different models here while serving called them one, so which
        # spelling a deployment happened to store decided whether an adapter
        # could be trained at all.
        if (
            runtime_base
            and stored_base
            and not transformer.same_base_model(stored_base, runtime_base)
        ):
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
            if (
                runtime_base
                and stored_base
                and not transformer.same_base_model(stored_base, runtime_base)
            ):
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
            if needs_update:
                self.store.update_artifact(adapter.id, updated_schema)
                adapter = self.store.get_artifact(adapter.id) or adapter
            return adapter

        # Create new adapter with explicit mode
        resolved_mode = adapter_mode or self.default_adapter_mode
        _ = adapter_id_override  # Reserved for future use in adapter creation
        adapter_schema = {
            "kind": "adapter.lora",
            "mode": resolved_mode,
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
        self.store.update_artifact(adapter.id, adapter_schema)
        return self.store.get_artifact(adapter.id) or adapter

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
            # SPEC §5.2 restricts LoRA to the attention projections; an
            # adapter that names none gets the standard Q/V pair rather than
            # an empty weight set that could only be skipped.
            matrices=adapter.schema.get("matrices") or ["attn_q", "attn_v"],
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

    @staticmethod
    def _aggregate_cluster_centroid(clusters: Optional[List[dict]]) -> List[float]:
        if not clusters:
            return []
        accum: List[float] = []
        weight_sum = 0
        for cluster in clusters:
            centroid = cluster.get("centroid") or []
            count = cluster.get("count") or 0
            if not isinstance(centroid, list) or not count:
                continue
            if len(accum) < len(centroid):
                accum += [0.0] * (len(centroid) - len(accum))
            padded = list(centroid) + [0.0] * (len(accum) - len(centroid))
            accum = [a + c * count for a, c in zip(accum, padded)]
            weight_sum += count
        if not weight_sum:
            return []
        return [val / weight_sum for val in accum]

    @staticmethod
    def _score_from_loss(loss: Optional[float]) -> float:
        if loss is None or not isinstance(loss, (int, float)):
            return 0.0
        if loss < 0:
            return 0.0
        return 1.0 / (1.0 + float(loss))

    @staticmethod
    def describe_run(result: Optional[dict]) -> Optional[dict]:
        """Normalise what train_from_preferences returned into a run summary.

        The version has to be recovered from the job directory name, and this
        service is what created that directory (see _job_dir). A caller should
        not have to know the layout to find out which version it just trained.
        """
        if not result:
            return None
        version = result.get("new_version")
        version_dir = result.get("version_dir", "")
        if version is None and "v" in version_dir:
            try:
                version = int(version_dir.split("/")[-1].replace("v", ""))
            except (ValueError, IndexError):
                version = None
        return {
            "loss": result.get("loss"),
            "version": version,
            "jax_trace": result.get("jax_trace"),
            "clusters": result.get("clusters"),
            # The gate decision travels with the summary. Dropping it let the
            # worker default `promoted` to True (SPEC §5.4.6 requires the
            # opposite), marking gate-rejected runs succeeded and crediting
            # the adapter's router state for a rollout that never happened.
            "eval_gate": result.get("eval_gate"),
        }

    def record_training_outcome(
        self, *, adapter_id: str, loss: Optional[float], clusters: Optional[List[dict]]
    ) -> None:
        """Fold a finished run into the adapter's router state (SPEC §5.4).

        The router picks adapters by comparing a request against each one's
        centroid and success score, so those numbers are how training feeds
        back into serving. Turning a training loss into a score is a judgement
        about training, which is why it lives here and not in the worker that
        happened to run the job.
        """

        centroid_vec = self._aggregate_cluster_centroid(clusters)
        try:
            self.store.update_adapter_router_state(
                adapter_id,
                centroid_vec=centroid_vec,
                success_score=self._score_from_loss(loss),
                last_trained_at=datetime.now(timezone.utc),
            )
        except Exception as exc:
            logger.warning(
                "adapter_router_state_update_failed",
                adapter_id=adapter_id,
                error=str(exc),
            )


    # An adapter is a prune candidate when it is barely used, scores badly,
    # and has been idle. All three, so a new adapter is never proposed for
    # retirement before it has had a chance to be used.
    ADAPTER_PRUNE_MIN_USAGE = 2
    ADAPTER_PRUNE_MAX_SUCCESS = 0.25
    ADAPTER_PRUNE_STALE_DAYS = 7

    def recommend_adapter_pruning(self) -> int:
        """Propose retiring adapters that are used little and score badly.

        A ConfigOps patch, not a deletion: the recommendation goes in front of
        a human, because an adapter with poor measured numbers may still be the
        one a small group of users depends on. Returns how many were proposed.

        Lives here rather than in the worker because deciding an adapter has
        stopped earning its keep is a judgement about adapters, and this
        service already owns what an adapter is, how it is scored and when it
        is promoted. The worker's job is to decide *when* to ask.
        """
        try:
            states = self.store.list_adapter_router_state()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("adapter_prune_state_fetch_failed", error=str(exc))
            return

        proposed = 0

        def _is_auto_prune_patch(patch: ConfigPatchAudit) -> bool:
            if getattr(patch, "status", "pending") != "pending":
                return False
            ops = []
            if isinstance(getattr(patch, "patch", None), dict):
                ops = patch.patch.get("ops") or []
            for op in ops:
                if isinstance(op, dict) and op.get("path") == "/meta/auto_prune":
                    return True
            return False

        existing_targets: set[str] = set()
        try:
            for patch in self.store.list_config_patches():
                if _is_auto_prune_patch(patch):
                    existing_targets.add(patch.artifact_id)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("adapter_prune_patch_scan_failed", error=str(exc))

        stale_cutoff = datetime.now(timezone.utc) - timedelta(days=self.ADAPTER_PRUNE_STALE_DAYS)
        for state in states:
            artifact = self.store.get_artifact(state.artifact_id)
            if not artifact or artifact.type != "adapter":
                continue

            last_used = state.last_used_at or state.last_trained_at or artifact.updated_at
            if not last_used:
                last_used = artifact.created_at
            if last_used and last_used.tzinfo is None:
                last_used = last_used.replace(tzinfo=timezone.utc)

            if (
                state.artifact_id not in existing_targets
                and state.usage_count < self.ADAPTER_PRUNE_MIN_USAGE
                and state.success_score < self.ADAPTER_PRUNE_MAX_SUCCESS
                and (not last_used or last_used < stale_cutoff)
            ):
                patch = {
                    "ops": [
                        {
                            "op": "add",
                            "path": "/meta/auto_prune",
                            "value": {
                                "recommended": True,
                                "reason": "low_usage_and_success_score",
                                "usage_count": state.usage_count,
                                "success_score": state.success_score,
                                "last_used_at": last_used.isoformat() if last_used else None,
                            },
                        }
                    ]
                }
                try:
                    self.store.record_config_patch(
                        artifact_id=state.artifact_id,
                        proposer="system_llm",
                        patch=patch,
                        justification=(
                            "Auto-prune recommendation for low-usage adapter; consider disabling or merging."
                        ),
                    )
                    existing_targets.add(state.artifact_id)
                    proposed += 1
                    logger.info(
                        "adapter_prune_recommendation_created",
                        adapter_id=state.artifact_id,
                        usage_count=state.usage_count,
                        success_score=state.success_score,
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning(
                        "adapter_prune_patch_failed",
                        adapter_id=state.artifact_id,
                        error=str(exc),
                    )
        return proposed

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
        - dataset too small for a holdout -> never promote: improvement
          cannot be shown, and unevaluated weights now change the model.
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
        # No holdout means the ≥1% improvement of §5.4.6 cannot be shown, and
        # "promoted only when it improves" refuses what it cannot measure.
        # This used to promote: harmless while trained weights were inert,
        # but they now change the model, so an unevaluated version could
        # regress it — which is exactly what §5.5's "nothing regresses"
        # forbids. The adapter stays on the prompt rung until it has enough
        # data to prove itself.
        return {
            "promoted": False,
            "reason": "no holdout (dataset below eval threshold); nothing to prove improvement",
            "holdout_examples": holdout_count,
        }

    def _build_examples(self, events: Iterable[PreferenceEvent]) -> Iterable[dict]:
        # Issue 18.1: Dedupe by (conversation_id, message_id) per SPEC §5.4.3
        seen: set[tuple[str, str]] = set()
        for event in events:
            key = (event.conversation_id or "", event.message_id or "")
            if key in seen:
                continue
            seen.add(key)
            # SPEC §5.4.2: the boundary is the target's sequence number, and
            # it is resolved from the store rather than searched for inside a
            # fetch window. Asking for the newest 200 messages and looking for
            # the target among them silently disabled the bound for any older
            # event — the target simply was not in the window, and every later
            # turn became training context.
            target_message = self.store.get_message(event.message_id or "")
            if target_message is None:
                logger.warning(
                    "sft_example_dropped_unresolvable_target",
                    conversation_id=event.conversation_id,
                    message_id=event.message_id,
                )
                continue
            # The query does the bounding, so the prompt cannot contain a turn
            # written after the answer being taught.
            messages = self.store.list_messages_before(
                event.conversation_id, target_message.seq, limit=200
            )
            prompt_chunks: List[str] = []
            target_text = event.corrected_text or target_message.content
            cluster_id = event.cluster_id or self._bucket_embedding(
                event.context_embedding, event.user_id
            )
            # Same placement rule serving uses (local_format.place_context):
            # appending the context after every message put it *after* the
            # question at training time and *before* it at serving time —
            # one marker, two token orders, which is two inputs.
            turns = [
                {"role": msg.role, "content": msg.content} for msg in messages
            ]
            if event.context_text:
                turns = local_format.place_context(turns, [event.context_text])
            prompt_chunks = [
                local_format.format_turn(turn["role"], turn["content"])
                for turn in turns
            ]
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

        def _encode(
            text: str, *, continuation: bool = False, limit: Optional[int] = None
        ) -> List[int]:
            """Tokens for one span. ``continuation`` suppresses special tokens.

            The target continues the prompt inside one sequence, so encoding
            it with specials would splice a second BOS into the middle —
            training the model on a sequence shape it never sees at serving
            time.

            ``limit`` is applied by this function, never by the tokenizer's
            own ``truncation``: tokenizers truncate from the right, keeping
            the OLDEST tokens. Letting it pre-truncate a long prompt threw
            away the newest context before the caller could ask to keep it,
            so the deliberate "trim oldest first" below operated on tokens
            that had already lost the part it was trying to preserve.
            """
            if self.tokenizer is not None:
                try:
                    tokens = list(
                        self.tokenizer.encode(
                            text,
                            truncation=False,
                            add_special_tokens=not continuation,
                        )
                    )
                except TypeError:  # pragma: no cover - minimal tokenizer objects
                    tokens = list(self.tokenizer.encode(text))
                return tokens[:limit] if limit is not None else tokens
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
                budget = max_length + 1
                # Target first: it is the supervised span, so it claims its
                # room before the prompt gets any. An empty target is dropped,
                # not padded to [0] — a zero there is not "no supervision", it
                # is supervision teaching the model to emit token 0, and it
                # carries positive mask weight so no later check can catch it.
                target_tokens = _encode(
                    row.get("target") or "", continuation=True, limit=budget - 1
                )
                if not target_tokens:
                    logger.warning(
                        "sft_example_without_target_dropped", reason="empty_target"
                    )
                    continue
                room = budget - len(target_tokens)
                # Trim the OLDEST prompt context; the newest turn is the one
                # the target responds to.
                prompt_tokens = _encode(row.get("prompt") or "")
                if len(prompt_tokens) > room:
                    prompt_tokens = prompt_tokens[-room:]
                if not prompt_tokens:
                    # Nothing to condition on: give the target one token of
                    # ground rather than inventing content.
                    if len(target_tokens) < 2:
                        logger.warning(
                            "sft_example_without_target_dropped", reason="no_prompt"
                        )
                        continue
                    prompt_tokens = [target_tokens.pop(0)]
                full = prompt_tokens + target_tokens
                prompt_len = len(prompt_tokens)
                seqs.append((full, prompt_len))
            if not seqs:
                continue
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
                if sum(mask) <= 0.0:
                    # Nothing supervised: the example would train on nothing
                    # while reporting a loss of zero, which reads as a
                    # perfectly learned example. Construction above prevents
                    # it; this refuses to emit one if that ever stops holding.
                    logger.warning(
                        "sft_example_without_target_dropped", prompt_len=prompt_len
                    )
                    continue
                pad = seq_len - len(inp)
                input_ids.append(inp + [0] * pad)
                labels.append(lab + [0] * pad)
                loss_mask.append(mask + [0.0] * pad)
            if not input_ids:
                continue
            yield {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": loss_mask,
                "shape": {
                    # The rows actually emitted, not the source slice: the
                    # shape exists so a consumer can preallocate, and it lied
                    # by the number of examples dropped for having no target.
                    "batch": len(input_ids),
                    "seq_len": seq_len,
                },
            }

    def _adapter_dir(
        self, user_id: str, adapter_id: str, adapter_schema: Optional[dict] = None
    ) -> Path:
        adapter_schema = adapter_schema or {}
        explicit = adapter_schema.get("fs_dir")
        # The same identity binding serving uses (§5.5). Containment alone let
        # an explicit root name *another* adapter's directory, and on this
        # side that writes A's new version into B's tree — where serving would
        # then find it under B's own id, with B's promotion authorizing it.
        return adapter_root(Path(self.fs_root), adapter_id, explicit)

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
            # Best-effort, deliberately. `latest` is convenience state that
            # serving does not consult (SPEC §5.5); `current_version` is the
            # authority and has already been written by the time this runs.
            # Re-raising meant a failed symlink aborted the run *after* the
            # adapter was promoted, so the gate decision §5.4.6 requires for
            # audit was never recorded — and on the worker path the job then
            # retried against weights that were already authoritative.
            logger.warning(
                "update_latest_symlink_failed",
                adapter_dir=str(adapter_dir),
                version_dir=str(version_dir),
                error=str(exc),
                detail="promotion stands; latest is convenience state only",
            )
            with suppress(FileNotFoundError):
                temp.unlink(missing_ok=True)  # type: ignore[arg-type]

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
        """Zero-initialized LoRA matrices sized to the base model (SPEC §5.2).

        Three things here are load-bearing:

        * **Shapes come from the checkpoint.** `A ∈ ℝ^{r × d_in}` and
          `B ∈ ℝ^{d_out × r}` where the projection decides d_in/d_out — k and
          v are narrower than q under grouped-query attention. Sizing both
          from a made-up hidden width produced matrices that fit no model.
        * **B starts at zero.** `B @ A` is then exactly zero, so a freshly
          created adapter is the identity. That is what lets §5.5 put an
          adapter on the prompt rung without it perturbing the model before
          the data has earned any weights.
        * **No base model, no weights.** LoRA hooks the base model's matrices;
          without a checkpoint there is nothing to hook, so the adapter stays
          prompt-only rather than carrying numbers that mean nothing.
        """
        config = self._base_config()
        if config is None:
            logger.warning(
                "lora_init_without_base_model",
                detail="no base checkpoint; adapter stays on the prompt rung",
            )
            return {}
        weights: dict[str, list[list[float]]] = {}
        for layer in layers:
            if not isinstance(layer, int) or not 0 <= layer < config.num_layers:
                logger.warning(
                    "lora_init_layer_out_of_range",
                    layer=layer,
                    num_layers=config.num_layers,
                )
                continue
            for matrix in matrices:
                shape = transformer.projection_shape(config, matrix)
                if shape is None:
                    logger.warning("lora_init_unknown_target", target=matrix)
                    continue
                d_out, d_in = shape
                weights[f"layers.{layer}.{matrix}.A"] = [
                    [random.uniform(-0.01, 0.01) for _ in range(d_in)]
                    for _ in range(rank)
                ]
                weights[f"layers.{layer}.{matrix}.B"] = [
                    [0.0 for _ in range(rank)] for _ in range(d_out)
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
        ``LocalJaxLoRABackend``: the frozen base checkpoint is loaded once,
        the LoRA matrices are applied inside its attention projections, and
        the gradient is taken with respect to those matrices alone — the base
        parameters are closed over, never differentiated, so "only on
        adapters, never on the base model" is structural rather than a
        promise. Gradients are accumulated across ``accumulation_steps``
        microbatches before each optimizer update and checkpoints are written
        so the backend can reload trained weights.
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

        # SPEC §5.4.4: the loss is over `model_apply(params_base, lora_params,
        # inputs)`. Without a base model there is no such loss to compute, so
        # the run is *skipped* — and §5.4.6 makes a skipped run unpromotable,
        # which leaves the adapter on the prompt rung. Training something
        # else and calling it a success is the failure this branch prevents.
        checkpoint = self._base_checkpoint()
        if checkpoint is None:
            logger.warning(
                "training_loop_skipped",
                reason="no_base_checkpoint",
                base_model=self._base_model_dir(),
            )
            return {"status": "skipped", "reason": "no base checkpoint to train against"}
        if not params:
            return {"status": "skipped", "reason": "adapter has no LoRA matrices"}
        # Resolved here rather than assumed: the invariant must not depend on
        # whether a caller happened to tokenize through this service first.
        self._ensure_tokenizer(self._base_model_dir())
        if self.tokenizer is None:
            # Gradients through the real transformer are worth nothing if the
            # text reached it through an invented token space: the adapter
            # would be fitted to ids that serving never produces, and the
            # holdout — tokenized the same wrong way — would happily agree.
            # "Train against the model that will serve it" includes its
            # tokenizer.
            logger.warning(
                "training_loop_skipped",
                reason="no_real_tokenizer",
                base_model=self._base_model_dir(),
                error=self._tokenizer_error,
            )
            return {
                "status": "skipped",
                "reason": "base checkpoint present but its tokenizer failed to load",
            }
        config, base_params = checkpoint
        vocab_size = config.vocab_size

        # A token the model has no embedding for means the tokenizer and the
        # checkpoint disagree. Clipping it to vocab_size-1 would train on a
        # token nobody wrote — the mismatch has to be refused, the same way a
        # missing checkpoint tensor is.
        for batch in list(batches) + list(eval_batches):
            for key in ("input_ids", "labels"):
                for sequence in batch.get(key) or []:
                    # Both ends of [0, vocab_size), symmetrically with
                    # serving: a negative id is as far outside the vocabulary
                    # as an oversized one, and it does not even fail loudly —
                    # array indexing reads it from the end of the table, so
                    # training would fit the adapter to a token nobody wrote.
                    if sequence and (min(sequence) < 0 or max(sequence) >= vocab_size):
                        logger.error(
                            "training_loop_skipped",
                            reason="tokenizer_vocab_mismatch",
                            checkpoint_vocab=vocab_size,
                            observed=[min(sequence), max(sequence)],
                        )
                        return {
                            "status": "skipped",
                            "reason": "tokenizer and checkpoint disagree on vocabulary",
                        }

        # SPEC §5.2: any malformed or foreign name refuses the whole adapter.
        # Training skips rather than raises — a skipped run cannot promote, so
        # the adapter waits on the prompt rung (§5.4.6).
        try:
            transformer.validate_lora_weights(config, params)
        except ValueError as exc:
            logger.warning("training_loop_skipped", reason="invalid_lora_weights", error=str(exc))
            return {"status": "skipped", "reason": f"invalid LoRA weights: {exc}"}
        # Parsed once, out here: the assembly inside the loss must be pure
        # array plumbing so it stays traceable and differentiable.
        lora_index = transformer.lora_layer_index(list(params))
        if not lora_index:
            return {"status": "skipped", "reason": "no LoRA matrices matched the model"}
        # α (SPEC §5.2) is a fixed hyperparameter, so it is carried as a
        # constant rather than handed to the optimizer as something to learn.
        scale_index = transformer.lora_layer_index(list(params), slots=("scale",))
        constants = {
            name: jnp.asarray(params[name], dtype=jnp.float32) for name in scale_index
        }

        def _assemble(tree: dict) -> List[dict]:
            layers: List[dict] = [{} for _ in range(config.num_layers)]
            for name, (layer_index, slot) in lora_index.items():
                layers[layer_index][slot] = tree[name]
            for name, (layer_index, slot) in scale_index.items():
                layers[layer_index][slot] = constants[name]
            return layers

        def _flatten_params(param_dict: dict) -> dict:
            """Only the trainable matrices enter the optimizer tree.

            Including α here made the "constant" comment above a lie: Optax
            updated it (the L2 term alone gives it a gradient), so the eval
            ran with the original α while the file written afterwards carried
            the modified one — passing a gate under one model and serving
            another.
            """
            return {
                name: jnp.array(param_dict[name], dtype=jnp.float32)
                for name in lora_index
            }

        def _to_python(tree: dict) -> dict:
            # Constants are reattached unchanged on the way out, so the
            # artifact keeps the α it was trained and evaluated under.
            serialized = {k: v.tolist() for k, v in tree.items()}
            for name, value in constants.items():
                serialized[name] = (
                    value.tolist() if hasattr(value, "tolist") else value
                )
            return serialized

        def _checkpoint(step: int, tree: dict) -> None:
            if not checkpoint_dir:
                return
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            path = checkpoint_dir / f"step_{step:04d}.json"
            path.write_text(json.dumps(_to_python(tree)))

        def cross_entropy(p: dict, batch: dict) -> jnp.ndarray:
            """Masked token-level CE over the real model (SPEC §5.4.4).

            The mask is the target span only, so the adapter learns the
            answer and not the prompt it was given. Normalized by mask weight
            rather than by batch, so a long prompt cannot dilute a short
            correction.
            """
            # No clipping: ids are validated against the checkpoint's vocab
            # before training starts, and a mismatch is refused there rather
            # than folded into a token the user never wrote.
            input_ids = jnp.array(batch["input_ids"], dtype=jnp.int32)
            labels = jnp.array(batch["labels"], dtype=jnp.int32)
            mask = jnp.array(batch.get("attention_mask") or [[1]], dtype=jnp.float32)
            logits, _ = transformer.forward(
                jnp, config, base_params, input_ids, lora=_assemble(p)
            )
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            nll = -jnp.take_along_axis(log_probs, labels[..., None], axis=-1).squeeze(
                -1
            )
            denom = jnp.maximum(jnp.sum(mask), 1.0)
            return jnp.sum(nll * mask) / denom

        def forward(p: dict, batch: dict) -> jnp.ndarray:
            """The training objective: CE plus the SPEC §5.4.4 L2 term."""
            l2 = sum(jnp.sum(jnp.square(value)) for value in p.values())
            return cross_entropy(p, batch) + LORA_L2_LAMBDA * l2

        def _eval_loss(p: dict) -> Optional[float]:
            """Holdout cross-entropy, without the regularizer.

            The gate asks whether predictions improved. Including the L2 term
            would let a shrinking weight norm register as progress, and — since
            B starts at zero and can only grow — it would count honest learning
            as a penalty against promotion.
            """
            if not eval_batches:
                return None
            losses = [float(cross_entropy(p, batch)) for batch in eval_batches]
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
