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
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, List, Optional

from liminallm.logging import get_logger
from liminallm.service import notes as notes_service
from liminallm.service.replication import AdvisoryLock

if TYPE_CHECKING:
    from liminallm.service.clustering import SemanticClusterer
    from liminallm.service.training import TrainingService
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
DEFAULT_REEMBED_INTERVAL_SECONDS = 60 * 60
# Bounded per pass: a vault that changed encoders converges over several
# passes instead of stalling the worker (or the provider) on one.


class TrainingWorker:
    """Background worker for processing training jobs.

    The worker runs in a loop, periodically checking for queued jobs
    and processing them using the TrainingService.
    """

    def __init__(
        self,
        store: "PostgresStore",
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
        reembed_interval: int = DEFAULT_REEMBED_INTERVAL_SECONDS,
        embeddings=None,
        leader_lock: Optional["AdvisoryLock"] = None,
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
        self.reembed_interval = reembed_interval
        # Needed to re-embed after an encoder change; None disables the sweep.
        self.embeddings = embeddings
        # Periodic clustering and prune proposals are cluster-wide work, not
        # per-replica work: without this lock every replica repeats them.
        # Queued jobs need no lock — claim_training_job() is an atomic
        # conditional UPDATE, so exactly one replica wins each job.
        self.leader_lock = leader_lock or AdvisoryLock(None)
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_cluster_run: float = 0.0
        self._last_prune_run: float = 0.0
        self._last_reembed_run: float = 0.0

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
                await self._maybe_reembed_stale_vectors()
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
        async with self.leader_lock.try_hold("training_worker:clustering") as leader:
            if not leader:
                logger.debug("periodic_clustering_skipped_not_leader")
                return
            await self.clusterer.cluster_everyone(
                user_limit=self.cluster_user_limit,
                max_events=self.cluster_event_limit,
            )

    async def _maybe_recommend_adapter_pruning(self) -> None:
        """Surface low-quality adapters via ConfigOps auto-proposals."""

        if self.adapter_prune_interval <= 0:
            return

        now = time.monotonic()
        if self._last_prune_run and (now - self._last_prune_run) < self.adapter_prune_interval:
            return
        self._last_prune_run = now

        async with self.leader_lock.try_hold("training_worker:adapter_prune") as leader:
            if not leader:
                logger.debug("adapter_prune_skipped_not_leader")
                return
            await asyncio.to_thread(self.training.recommend_adapter_pruning)

    async def _maybe_reembed_stale_vectors(self) -> None:
        """Re-embed vectors written by a previous encoder, cluster-wide once.

        Encoder changes are otherwise handled lazily: a vector whose recorded
        encoder differs reads as "not embedded" and is recomputed only when
        something reads it. Notes and messages nobody opens would keep stale
        vectors indefinitely and quietly drop out of semantic search. This is
        the sweep that closes that gap.
        """
        embeddings = self.embeddings
        if embeddings is None or not getattr(embeddings, "is_semantic", False):
            return  # hash vectors have no encoder identity to go stale
        if self.reembed_interval <= 0:
            return
        now = time.monotonic()
        if self._last_reembed_run and (now - self._last_reembed_run) < self.reembed_interval:
            return
        self._last_reembed_run = now
        async with self.leader_lock.try_hold("training_worker:reembed") as leader:
            if not leader:
                logger.debug("reembed_skipped_not_leader")
                return
            done = await asyncio.to_thread(
                notes_service.reembed_stale,
                self.store,
                self.embeddings,
                user_limit=self.cluster_user_limit,
            )
            if done:
                logger.info("reembed_sweep_completed", vectors=done)

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
        queued = [j for j in self.store.list_training_jobs() if j.status == "queued"]
        if len(queued) > MAX_QUEUE_DEPTH:
            logger.warning(
                "training_queue_depth_capped",
                queued=len(queued),
                capped=MAX_QUEUE_DEPTH,
            )
        return queued[:MAX_QUEUE_DEPTH]

    async def _process_job(self, job) -> None:
        """Process a single training job."""
        job_id = job.id
        user_id = job.user_id
        adapter_id = job.adapter_id

        # Issue 26.2: Atomically claim the job to prevent duplicate processing
        if not self.store.claim_training_job(job_id):
            logger.info("training_job_already_claimed", job_id=job_id)
            return

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
                    job_id=job_id,
                )

                if result:
                    # SPEC §5.4.6: a run that trained but failed the eval gate
                    # did NOT ship weights - record it as gate-rejected rather
                    # than "succeeded", and leave router state alone so an
                    # un-promoted adapter is not credited with a training pass.
                    # A run that never trained is neither: see
                    # `TrainingService.terminal_status`, which owns the rule
                    # for both this and the service's own write.
                    gate = result.get("eval_gate") or {}
                    # Absent means unknown, and unknown is not approval: the
                    # summary used to drop eval_gate entirely, so this
                    # defaulted to True and credited every rejected run.
                    promoted = bool(gate.get("promoted", False))
                    if "promoted" not in gate:
                        logger.warning(
                            "training_gate_decision_missing",
                            job_id=job_id,
                            detail="treating as not promoted",
                        )
                    # Merge into the meta TrainingService already wrote (it
                    # holds eval_gate/pooled_skill/distilled); replacing it
                    # would destroy the gate audit trail.
                    existing_meta = {}
                    refreshed = self.store.get_training_job(job_id)
                    if refreshed and isinstance(refreshed.meta, dict):
                        existing_meta = dict(refreshed.meta)
                    existing_meta.update(
                        {
                            "jax_trace": result.get("jax_trace"),
                            "clusters": result.get("clusters"),
                            "completed_at": datetime.now(timezone.utc).isoformat(),
                        }
                    )
                    self.store.update_training_job(
                        job_id,
                        status=self.training.terminal_status(
                            result.get("jax_trace"), gate
                        ),
                        loss=result.get("loss"),
                        # TrainingService already set new_version on promotion;
                        # the result exposes the directory, not the number.
                        new_version=None,
                        meta=existing_meta,
                    )
                    logger.info(
                        "training_job_finished",
                        job_id=job_id,
                        loss=result.get("loss"),
                        promoted=promoted,
                        gate_reason=gate.get("reason"),
                        version_dir=result.get("version_dir"),
                    )

                    if promoted:
                        self.training.record_training_outcome(
                            adapter_id=adapter_id,
                            loss=result.get("loss"),
                            clusters=result.get("clusters"),
                        )
                    # Trigger clustering after successful training
                    if self.clusterer:
                        await self.clusterer.cluster_after_training(user_id)
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
                "failed_at": datetime.now(timezone.utc).isoformat(),
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
        job_id: Optional[str] = None,
    ) -> Optional[dict]:
        """Run the job through TrainingService and summarise the result."""
        # Pass the already-claimed job_id so training reuses it instead of
        # creating a duplicate queued job on every worker run.
        return self.training.describe_run(
            self.training.train_from_preferences(
                user_id=user_id,
                adapter_id=adapter_id,
                cluster_id=cluster_id,
                job_id=job_id,
            )
        )

    def _get_cluster_id(self, job) -> Optional[str]:
        """Extract cluster_id from job metadata if present."""
        if job.meta and isinstance(job.meta, dict):
            return job.meta.get("cluster_id")
        return None
