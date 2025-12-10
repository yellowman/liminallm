"""Background worker for processing training jobs.

This module provides a worker that periodically checks for queued training jobs
and executes them using the TrainingService. It handles:
- Picking up queued jobs
- Running JAX/Optax training
- Updating job status
- Error handling and retries
- Connecting emergent skills to training
"""

from __future__ import annotations

import asyncio
import inspect
import time
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, List, Optional

from liminallm.logging import get_logger

if TYPE_CHECKING:
    from liminallm.service.clustering import SemanticClusterer
    from liminallm.service.training import TrainingService
    from liminallm.storage.memory import MemoryStore
    from liminallm.storage.postgres import PostgresStore

logger = get_logger(__name__)

# Worker configuration
DEFAULT_POLL_INTERVAL_SECONDS = 60
DEFAULT_BATCH_SIZE = 5
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY_SECONDS = 30
MAX_QUEUE_DEPTH = 100
DEFAULT_CLUSTER_INTERVAL_SECONDS = 15 * 60
DEFAULT_CLUSTER_USER_LIMIT = 50
DEFAULT_CLUSTER_EVENT_LIMIT = 500
DEFAULT_ADAPTER_PRUNE_INTERVAL_SECONDS = 6 * 60 * 60
ADAPTER_PRUNE_MIN_USAGE = 2
ADAPTER_PRUNE_MAX_SUCCESS = 0.25
ADAPTER_PRUNE_STALE_DAYS = 7


class TrainingWorker:
    """Background worker for processing training jobs.

    The worker runs in a loop, periodically checking for queued jobs
    and processing them using the TrainingService.
    """

    def __init__(
        self,
        store: "PostgresStore | MemoryStore",
        training_service: "TrainingService",
        clusterer: Optional["SemanticClusterer"] = None,
        *,
        poll_interval: int = DEFAULT_POLL_INTERVAL_SECONDS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_delay: int = DEFAULT_RETRY_DELAY_SECONDS,
        cluster_interval: int = DEFAULT_CLUSTER_INTERVAL_SECONDS,
        cluster_user_limit: int = DEFAULT_CLUSTER_USER_LIMIT,
        cluster_event_limit: int = DEFAULT_CLUSTER_EVENT_LIMIT,
        adapter_prune_interval: int = DEFAULT_ADAPTER_PRUNE_INTERVAL_SECONDS,
    ) -> None:
        self.store = store
        self.training = training_service
        self.clusterer = clusterer
        self.poll_interval = poll_interval
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.cluster_interval = cluster_interval
        self.cluster_user_limit = cluster_user_limit
        self.cluster_event_limit = cluster_event_limit
        self.adapter_prune_interval = adapter_prune_interval
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_cluster_run: float = 0.0
        self._last_prune_run: float = 0.0

    async def start(self) -> None:
        """Start the background worker."""
        if self._running:
            logger.warning("training_worker_already_running")
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("training_worker_started", poll_interval=self.poll_interval)

    async def stop(self) -> None:
        """Stop the background worker."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("training_worker_stopped")

    async def _run_loop(self) -> None:
        """Main worker loop."""
        consecutive_errors = 0
        while self._running:
            try:
                await self._process_queued_jobs()
                await self._maybe_run_periodic_clustering()
                await self._maybe_recommend_adapter_pruning()
                consecutive_errors = 0
            except Exception as exc:
                consecutive_errors += 1
                logger.error(
                    "training_worker_loop_error",
                    error=str(exc),
                    error_type=type(exc).__name__,
                    consecutive_errors=consecutive_errors,
                )
                # Exponential backoff on repeated errors
                if consecutive_errors > 3:
                    backoff = min(300, self.poll_interval * (2 ** (consecutive_errors - 3)))
                    logger.warning(
                        "training_worker_backoff",
                        backoff_seconds=backoff,
                        consecutive_errors=consecutive_errors,
                    )
                    await asyncio.sleep(backoff)
                    continue

            await asyncio.sleep(self.poll_interval)

    async def _maybe_run_periodic_clustering(self) -> None:
        if not self.clusterer or self.cluster_interval <= 0:
            return

        now = time.monotonic()
        if self._last_cluster_run and (now - self._last_cluster_run) < self.cluster_interval:
            return

        self._last_cluster_run = now
        users = []
        if hasattr(self.store, "list_users"):
            users_raw = self.store.list_users(limit=self.cluster_user_limit)
            if inspect.isawaitable(users_raw):
                users = list(await users_raw)
            else:
                users = list(users_raw)

        for user in users:
            try:
                await self.clusterer.cluster_user_preferences(
                    user.id,
                    max_events=self.cluster_event_limit,
                    streaming=True,
                    approximate=True,
                )
            except Exception as exc:
                logger.warning(
                    "periodic_user_clustering_failed",
                    user_id=user.id,
                    error=str(exc),
                )

        try:
            await self.clusterer.cluster_global_preferences(
                max_events=self.cluster_event_limit,
                streaming=True,
                approximate=True,
            )
        except Exception as exc:
            logger.warning("periodic_global_clustering_failed", error=str(exc))

    async def _maybe_recommend_adapter_pruning(self) -> None:
        """Surface low-quality adapters via ConfigOps auto-proposals."""

        if self.adapter_prune_interval <= 0:
            return

        now = time.monotonic()
        if self._last_prune_run and (now - self._last_prune_run) < self.adapter_prune_interval:
            return
        self._last_prune_run = now

        list_states = getattr(self.store, "list_adapter_router_state", None)
        record_patch = getattr(self.store, "record_config_patch", None)
        list_patches = getattr(self.store, "list_config_patches", None)
        if not callable(list_states) or not callable(record_patch):
            return

        try:
            states = list_states()  # type: ignore[call-arg]
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("adapter_prune_state_fetch_failed", error=str(exc))
            return

        existing_targets: set[str] = set()
        if callable(list_patches):
            try:
                for patch in list_patches():
                    if (
                        isinstance(patch.meta, dict)
                        and patch.meta.get("auto_prune")
                        and getattr(patch, "status", "pending") == "pending"
                    ):
                        existing_targets.add(patch.artifact_id)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("adapter_prune_patch_scan_failed", error=str(exc))

        stale_cutoff = datetime.utcnow() - timedelta(days=ADAPTER_PRUNE_STALE_DAYS)
        for state in states:
            artifact = self.store.get_artifact(state.artifact_id)
            if not artifact or artifact.type != "adapter":
                continue

            last_used = state.last_used_at or state.last_trained_at or artifact.updated_at
            if not last_used:
                last_used = artifact.created_at
            if last_used and last_used.tzinfo:
                last_used = last_used.astimezone(timezone.utc).replace(tzinfo=None)

            if (
                state.artifact_id not in existing_targets
                and state.usage_count < ADAPTER_PRUNE_MIN_USAGE
                and state.success_score < ADAPTER_PRUNE_MAX_SUCCESS
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
                    record_patch(
                        artifact_id=state.artifact_id,
                        proposer="system_llm",
                        patch=patch,
                        justification=(
                            "Auto-prune recommendation for low-usage adapter; consider disabling or merging."
                        ),
                    )
                    existing_targets.add(state.artifact_id)
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

    async def _process_queued_jobs(self) -> None:
        """Process a batch of queued training jobs."""
        jobs = self._get_queued_jobs()
        if not jobs:
            return

        logger.info("training_worker_processing", job_count=len(jobs))

        for job in jobs[:self.batch_size]:
            await self._process_job(job)

    def _get_queued_jobs(self) -> List:
        """Get queued training jobs from the store."""
        list_fn = getattr(self.store, "list_training_jobs", None)
        if callable(list_fn):
            all_jobs = list_fn()
            queued = [j for j in all_jobs if j.status == "queued"]
            if len(queued) > MAX_QUEUE_DEPTH:
                logger.warning(
                    "training_queue_depth_capped",
                    queued=len(queued),
                    capped=MAX_QUEUE_DEPTH,
                )
            return queued[:MAX_QUEUE_DEPTH]

        # MemoryStore fallback
        if hasattr(self.store, "training_jobs"):
            queued = [
                j for j in self.store.training_jobs.values()
                if j.status == "queued"
            ]
            return queued[:MAX_QUEUE_DEPTH]

        return []

    async def _process_job(self, job) -> None:
        """Process a single training job."""
        job_id = job.id
        user_id = job.user_id
        adapter_id = job.adapter_id

        # Issue 26.2: Atomically claim the job to prevent duplicate processing
        # Use claim_training_job if available (PostgresStore), fallback to update
        claim_fn = getattr(self.store, "claim_training_job", None)
        if callable(claim_fn):
            claimed_job = claim_fn(job_id)
            if not claimed_job:
                # Job already claimed by another worker or doesn't exist
                logger.info("training_job_already_claimed", job_id=job_id)
                return
        else:
            # Fallback for MemoryStore - use regular update (less safe)
            self.store.update_training_job(job_id, status="running")

        logger.info(
            "training_job_starting",
            job_id=job_id,
            user_id=user_id,
            adapter_id=adapter_id,
        )

        attempt = 0
        last_error: Optional[str] = None

        while attempt < self.max_retries:
            try:
                # Run the actual training
                result = await asyncio.to_thread(
                    self._execute_training,
                    user_id=user_id,
                    adapter_id=adapter_id,
                    cluster_id=self._get_cluster_id(job),
                )

                if result:
                    # Training succeeded
                    self.store.update_training_job(
                        job_id,
                        status="succeeded",
                        loss=result.get("loss"),
                        new_version=result.get("version"),
                        meta={
                            "jax_trace": result.get("jax_trace"),
                            "clusters": result.get("clusters"),
                            "completed_at": datetime.utcnow().isoformat(),
                        },
                    )
                    logger.info(
                        "training_job_succeeded",
                        job_id=job_id,
                        loss=result.get("loss"),
                        version=result.get("version"),
                    )

                    self._update_adapter_router_state(
                        adapter_id=adapter_id,
                        loss=result.get("loss"),
                        clusters=result.get("clusters"),
                    )
                    # Trigger clustering after successful training
                    await self._run_post_training_clustering(user_id)
                    return
                else:
                    # No events to train on
                    self.store.update_training_job(
                        job_id,
                        status="skipped",
                        meta={"reason": "no_preference_events"},
                    )
                    logger.info("training_job_skipped", job_id=job_id, reason="no_events")
                    return

            except Exception as exc:
                attempt += 1
                last_error = str(exc)
                logger.warning(
                    "training_job_attempt_failed",
                    job_id=job_id,
                    user_id=user_id,
                    adapter_id=adapter_id,
                    attempt=attempt,
                    max_retries=self.max_retries,
                    error_type=type(exc).__name__,
                    error=last_error,
                )

                if attempt < self.max_retries:
                    backoff = min(self.retry_delay * (2 ** (attempt - 1)), 300)
                    logger.debug(
                        "training_job_retry_wait",
                        job_id=job_id,
                        retry_delay=backoff,
                        next_attempt=attempt + 1,
                    )
                    await asyncio.sleep(backoff)

        # All retries exhausted
        self.store.update_training_job(
            job_id,
            status="dead_letter",
            meta={
                "error": last_error,
                "attempts": attempt,
                "failed_at": datetime.utcnow().isoformat(),
            },
        )
        logger.error(
            "training_job_failed",
            job_id=job_id,
            error=last_error,
            attempts=attempt,
        )

    def _execute_training(
        self,
        user_id: str,
        adapter_id: str,
        cluster_id: Optional[str] = None,
    ) -> Optional[dict]:
        """Execute the actual training via TrainingService."""
        result = self.training.train_from_preferences(
            user_id=user_id,
            adapter_id=adapter_id,
            cluster_id=cluster_id,
        )

        if result:
            # Extract version from result
            version_dir = result.get("version_dir", "")
            version = None
            if "v" in version_dir:
                try:
                    version_str = version_dir.split("/")[-1].replace("v", "")
                    version = int(version_str)
                except (ValueError, IndexError):
                    pass

            return {
                "loss": result.get("loss"),
                "version": version or result.get("new_version"),
                "jax_trace": result.get("jax_trace"),
                "clusters": result.get("clusters"),
            }

        return None

    def _get_cluster_id(self, job) -> Optional[str]:
        """Extract cluster_id from job metadata if present."""
        if job.meta and isinstance(job.meta, dict):
            return job.meta.get("cluster_id")
        return None

    async def _run_post_training_clustering(self, user_id: str) -> None:
        """Run clustering after successful training to detect emergent skills."""
        if not self.clusterer:
            return

        try:
            clusters = await self.clusterer.cluster_user_preferences(user_id)
            if clusters:
                logger.info(
                    "post_training_clustering_complete",
                    user_id=user_id,
                    cluster_count=len(clusters),
                )

                # Check for skill promotion opportunities
                promoted = self.clusterer.promote_skill_adapters(
                    min_size=5,
                    positive_ratio=0.7,
                )
                if promoted:
                    logger.info(
                        "emergent_skills_promoted",
                        user_id=user_id,
                        adapter_ids=promoted,
                    )
        except Exception as exc:
            logger.warning(
                "post_training_clustering_failed",
                user_id=user_id,
                error=str(exc),
            )

    def _aggregate_cluster_centroid(self, clusters: Optional[List[dict]]) -> List[float]:
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

    def _score_from_loss(self, loss: Optional[float]) -> float:
        if loss is None or not isinstance(loss, (int, float)):
            return 0.0
        if loss < 0:
            return 0.0
        return 1.0 / (1.0 + float(loss))

    def _update_adapter_router_state(
        self, *, adapter_id: str, loss: Optional[float], clusters: Optional[List[dict]]
    ) -> None:
        """Update adapter router state after training (SPEC §5.4)."""

        if not hasattr(self.store, "update_adapter_router_state"):
            return
        centroid_vec = self._aggregate_cluster_centroid(clusters)
        try:
            self.store.update_adapter_router_state(
                adapter_id,
                centroid_vec=centroid_vec,
                success_score=self._score_from_loss(loss),
                last_trained_at=datetime.utcnow(),
            )
        except Exception as exc:
            logger.warning(
                "adapter_router_state_update_failed",
                adapter_id=adapter_id,
                error=str(exc),
            )

    async def process_emergent_skills(self) -> List[str]:
        """Manually trigger emergent skill detection and training.

        This scans all users for cluster promotion opportunities
        and creates training jobs for newly created skill adapters.
        """
        if not self.clusterer:
            logger.warning("emergent_skills_no_clusterer")
            return []

        promoted_adapters: List[str] = []

        # Get all users with preference events
        users = self._get_users_with_preferences()

        for user_id in users:
            try:
                # Run clustering
                clusters = await self.clusterer.cluster_user_preferences(user_id)

                if clusters:
                    # Promote eligible clusters to skill adapters
                    promoted = self.clusterer.promote_skill_adapters(
                        min_size=5,
                        positive_ratio=0.7,
                    )
                    promoted_adapters.extend(promoted)

            except Exception as exc:
                logger.warning(
                    "emergent_skill_processing_failed",
                    user_id=user_id,
                    error=str(exc),
                )

        logger.info(
            "emergent_skills_processed",
            promoted_count=len(promoted_adapters),
            adapter_ids=promoted_adapters,
        )

        return promoted_adapters

    def _get_users_with_preferences(self) -> List[str]:
        """Get list of users who have preference events."""
        users: set[str] = set()

        list_fn = getattr(self.store, "list_preference_events", None)
        if callable(list_fn):
            events = list_fn()
            for event in events:
                if event.user_id:
                    users.add(event.user_id)
        elif hasattr(self.store, "preference_events"):
            for event in self.store.preference_events.values():
                if event.user_id:
                    users.add(event.user_id)

        return list(users)


async def create_training_worker(
    store: "PostgresStore | MemoryStore",
    training_service: "TrainingService",
    clusterer: Optional["SemanticClusterer"] = None,
    *,
    auto_start: bool = True,
    poll_interval: int = DEFAULT_POLL_INTERVAL_SECONDS,
) -> TrainingWorker:
    """Factory function to create and optionally start a training worker.

    Args:
        store: Database store
        training_service: TrainingService instance
        clusterer: Optional SemanticClusterer for emergent skills
        auto_start: Whether to start the worker immediately
        poll_interval: How often to check for queued jobs

    Returns:
        TrainingWorker instance
    """
    worker = TrainingWorker(
        store=store,
        training_service=training_service,
        clusterer=clusterer,
        poll_interval=poll_interval,
    )

    if auto_start:
        await worker.start()

    return worker
