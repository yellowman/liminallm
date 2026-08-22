from __future__ import annotations

import asyncio
import contextlib
import json
import os
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple, Union
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from liminallm.config import (
    MODEL_AFFECTING_SETTINGS as _MODEL_AFFECTING,
)
from liminallm.config import (
    SYSTEM_SETTINGS_DEFAULTS,
    apply_managed_settings,
    generate_jwt_secret,
    get_settings,
    reset_settings_cache,
)
from liminallm.logging import get_logger
from liminallm.service import token_counting
from liminallm.service.auth import AuthService
from liminallm.service.clustering import SemanticClusterer
from liminallm.service.config_ops import ConfigOpsService
from liminallm.service.email import EmailService
from liminallm.service.embeddings import (
    EmbeddingsService,
    make_provider_batch_encoder,
    make_provider_encoder,
)
from liminallm.service.llm import LLMService
from liminallm.service.rag import RAGService
from liminallm.service.replication import AdvisoryLock, ClusterBus
from liminallm.service.rerank import make_llm_reranker
from liminallm.service.router import RouterEngine
from liminallm.service.training import TrainingService
from liminallm.service.training_worker import TrainingWorker
from liminallm.service.voice import VoiceService
from liminallm.service.workflow import WorkflowEngine
from liminallm.storage.postgres import PostgresStore
from liminallm.storage.redis_cache import RedisCache

logger = get_logger(__name__)

#: Sorted so the signature tuple has a stable shape; config.py owns the set,
#: because the admin console labels settings from the same one.
MODEL_AFFECTING_SETTINGS: Tuple[str, ...] = tuple(sorted(_MODEL_AFFECTING))


#: Query keys that carry a password. Both drivers read connection keywords
#: from the query string — `?password=` for redis-py and libpq alike, plus
#: libpq's `sslpassword` — so a mask that rewrites only the userinfo
#: publishes the same secret through the other spelling.
_PASSWORD_QUERY_KEYS = frozenset({"password", "sslpassword"})


def _mask_url_password(url: Optional[str]) -> Optional[str]:
    """Mask passwords in a URL for safe logging (Issue 29.1).

    Both places a URL can carry one: the userinfo
    (redis://:secret@host -> redis://:***@host) and the query string
    (postgresql://host/db?password=secret -> ...?password=***), which
    redis-py and libpq honour just as they do the userinfo.
    """
    if not url:
        return url
    try:
        parsed = urlparse(url)
        query_pairs = parse_qsl(parsed.query, keep_blank_values=True)
        query_masked = any(
            key.lower() in _PASSWORD_QUERY_KEYS and value for key, value in query_pairs
        )
        if not parsed.password and not query_masked:
            return url
        netloc = parsed.netloc
        if parsed.password:
            netloc = parsed.hostname or ""
            if parsed.port:
                netloc = f"{netloc}:{parsed.port}"
            if parsed.username:
                netloc = f"{parsed.username}:***@{netloc}"
            else:
                netloc = f":***@{netloc}"
        query = parsed.query
        if query_masked:
            query = urlencode(
                [
                    (key, "***" if key.lower() in _PASSWORD_QUERY_KEYS else value)
                    for key, value in query_pairs
                ],
                # `*` is not special in a query string, and percent-encoding
                # the replacement turned every masked value into `%2A%2A%2A`.
                # The secret was gone either way; this is about the log line
                # being readable, and about the function producing the `***`
                # its own docstring promises.
                safe="*",
            )
        return urlunparse((
            parsed.scheme,
            netloc,
            parsed.path,
            parsed.params,
            query,
            parsed.fragment,
        ))
    except Exception:
        # If parsing fails, return masked placeholder
        return "***url_parse_error***"


#: A store for `Runtime` to adopt instead of building its own. Only the test
#: harness sets it, and only in TEST_MODE.
#:
#: The suite rebuilds the runtime before and after every test for state
#: isolation, and building one used to mean a new connection pool and a full
#: re-run of the schema verification each time. Measured at 2636 tests: 38.5ms
#: per `Runtime()`, twice per test, about a quarter of the suite's wall clock
#: spent proving the same schema over and over against the same database.
#:
#: The store is the one thing in the runtime with no per-test state to
#: isolate — it holds a pool and a fs_root — so sharing one across the session
#: costs nothing that the reset was buying. Tests that need a store built from
#: scratch, such as the startup-verification reds, construct `PostgresStore`
#: directly and are unaffected.
_shared_store = None


def use_shared_store(store) -> None:
    """Have `Runtime` adopt `store` rather than construct one. Test-mode only."""
    global _shared_store
    if not get_settings().test_mode:
        raise RuntimeError("a shared store is only allowed in TEST_MODE")
    _shared_store = store


class Runtime:
    """Holds singleton service instances for the FastAPI app."""

    def __init__(self):
        self.settings = get_settings()
        logger.info(
            "runtime_init_started",
            test_mode=self.settings.test_mode,
        )

        try:
            self.store = _shared_store or PostgresStore(
                self.settings.database_url, fs_root=self.settings.shared_fs_root
            )
            logger.info(
                "runtime_store_initialized",
                store_type="postgres",
            )
        except Exception as exc:
            logger.error(
                "runtime_store_init_failed",
                store_type="postgres",
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise

        # Everything below is configured from the database, so resolve the
        # settings before building anything that depends on them.
        # Order matters: the seed only applies to an unconfigured instance, and
        # the generated signing key would otherwise make it look configured.
        self._seed_settings_from_env()
        self._ensure_signing_key()
        self.refresh_settings()

        self.cache = None
        redis_error: Exception | None = None
        if self.settings.redis_url:
            try:
                cache = RedisCache(self.settings.redis_url)
                cache.verify_connection()
                self.cache = cache
            except Exception as exc:
                redis_error = exc
                self.cache = None

        if not self.cache:
            if (
                not self.settings.test_mode
                and not self.settings.allow_redis_fallback_dev
            ):
                raise RuntimeError(
                    "Redis is required for sessions, rate limits, idempotency, and workflow caches; "
                    "start Redis, set TEST_MODE=true, or enable the "
                    "allow_redis_fallback_dev setting for local fallback."
                ) from redis_error

            # test_mode is an environment variable; the other is an admin
            # setting, and naming it in caps sent operators to export a
            # variable nothing reads.
            fallback_mode = (
                "TEST_MODE" if self.settings.test_mode else "allow_redis_fallback_dev"
            )

            logger.warning(
                "redis_disabled_fallback",
                # Issue 29.1: Mask password in URL for safe logging
                redis_url=_mask_url_password(self.settings.redis_url),
                error=str(redis_error) if redis_error else "redis_url_missing",
                message=(
                    f"Running without Redis under {fallback_mode}; rate limits, idempotency durability, and "
                    "workflow/router caches are in-memory only."
                ),
                mode=fallback_mode,
            )
        # Build all backend-dependent services (router, llm, rag, training,
        # clusterer, workflow, config_ops). Extracted into a helper so
        # reload_model_services() can rebuild them when an admin changes
        # model_backend/model_path without restarting the process.
        self._build_model_services()

        self.auth = AuthService(
            self.store,
            self.cache,
            self.settings,
            mfa_enabled=self.settings.enable_mfa,
        )
        self._build_credentialed_services()
        # Cross-replica coordination. Both primitives are best-effort and
        # neither makes Redis required: the bus falls back to Postgres
        # LISTEN/NOTIFY, and a single-process deployment gets local semantics.
        db_url = self.settings.database_url
        # The cache object is the signal that Redis really exists here (the
        # setting has a localhost default even on redis-less deployments) —
        # but an explicit backend=redis is the operator's word over that
        # heuristic, e.g. when Redis was briefly down at boot.
        bus_backend = self.settings.cluster_bus_backend
        bus_redis_url = (
            self.settings.redis_url
            if (self.cache is not None or bus_backend == "redis")
            else None
        )
        self.bus = ClusterBus(
            redis_url=bus_redis_url,
            database_url=db_url,
            backend=bus_backend,
        )
        self.leader_lock = AdvisoryLock(db_url)
        self._install_calibration_sharing()

        # Training worker for background job processing
        self.training_worker = TrainingWorker(
            store=self.store,
            training_service=self.training,
            clusterer=self.clusterer,
            poll_interval=self.settings.training_worker_poll_interval,
            leader_lock=self.leader_lock,
            # Lets the worker re-embed vectors left behind by a previous
            # encoder; a hash encoder makes the sweep a no-op.
            embeddings=self.embeddings,
        )
        self._local_idempotency: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        self._local_idempotency_lock = asyncio.Lock()
        self._local_idempotency_last_cleanup = datetime.now(timezone.utc)
        self._local_idempotency_max_entries = 5000
        self._local_rate_limits: Dict[str, Tuple[float, datetime]] = {}
        self._local_rate_limit_lock = asyncio.Lock()
        self._local_rate_limit_last_cleanup = datetime.now(timezone.utc)
        self._local_rate_limit_max_entries = 5000
        # Cross-worker settings reload coordination: the version this worker's
        # model stack was built from, and a lock serializing rebuilds.
        self._reload_lock = threading.Lock()
        self._applied_settings_version = self._read_settings_version()

        # Log successful initialization with summary
        logger.info(
            "runtime_initialized",
            model_path=self.resolved_base_model,
            model_backend=str(self.backend_mode),
            rag_mode=str(self.rag_mode),
            adapter_mode=str(self.default_adapter_mode),
            redis_enabled=self.cache is not None,
            email_configured=self.email.is_configured,
            voice_configured=self.voice.is_configured,
            mfa_enabled=self.settings.enable_mfa,
        )

    def _install_calibration_sharing(self) -> None:
        """Share learned token-calibration factors across replicas.

        Durable in the store (survives restart, works with Redis absent) and
        broadcast on the cluster bus so peers converge immediately instead of
        each re-learning the same factor. Entirely best-effort: without it
        calibration is per-process, which is correct but slower.
        """
        store = self.store

        def load(model: str):
            stored = store.get_instance_config(token_counting.CALIBRATION_CONFIG_NAME)
            entry = (stored or {}).get(model)
            if isinstance(entry, dict):
                return entry.get("factor")
            return entry if isinstance(entry, (int, float)) else None

        def publish(model: str, factor: float, observations: int) -> None:
            store.merge_instance_config(
                token_counting.CALIBRATION_CONFIG_NAME,
                {model: {"factor": round(factor, 4), "observations": observations}},
            )
            bus = getattr(self, "bus", None)
            if bus is None:
                return
            payload = {
                "model": model,
                "factor": round(factor, 4),
                "observations": observations,
            }
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return  # no loop (sync context): the store write already landed
            loop.create_task(
                bus.publish(token_counting.CALIBRATION_CHANNEL, payload)
            )

        token_counting.install_sharing(load=load, publish=publish)

    def _system_settings_overrides(self) -> dict:
        """Only what an admin actually saved, without defaults merged in.

        get_system_settings() merges the shipped defaults over the stored
        values; overlaying that would make a default indistinguishable from a
        choice. This returns the choices.
        """
        try:
            return self.store.get_system_settings_overrides() or {}
        except Exception as exc:
            logger.warning("settings_overrides_read_failed", error=str(exc))
            return {}

    #: Written by the instance itself, not chosen by anyone.
    GENERATED_SETTINGS = frozenset({"jwt_secret"})

    def _seed_settings_from_env(self) -> None:
        """Apply INSTANCE_SETTINGS_JSON on first boot, once.

        Managed settings have no environment variables, which would otherwise
        make a declarative deploy (compose, k8s) unable to configure anything
        without a human opening the admin UI. This is the one seam: a JSON
        object of managed settings, written to the database only when no admin
        has saved anything yet. It is a seed, not an override — once a value is
        in the database it is authoritative, so a stale container env cannot
        quietly revert what an operator changed.
        """
        raw = os.environ.get("INSTANCE_SETTINGS_JSON")
        if not raw:
            return
        try:
            seed = json.loads(raw)
        except ValueError as exc:
            logger.error("instance_settings_seed_invalid_json", error=str(exc))
            return
        if not isinstance(seed, dict):
            logger.error("instance_settings_seed_not_an_object")
            return
        known = set(SYSTEM_SETTINGS_DEFAULTS)
        unknown = sorted(set(seed) - known)
        if unknown:
            logger.warning("instance_settings_seed_unknown_keys", keys=unknown)
        seed = {k: v for k, v in seed.items() if k in known}
        if not seed:
            return
        try:
            # "Configured" means an operator chose something. The signing key
            # writes itself on first boot, so it does not count.
            chosen = set(self._system_settings_overrides()) - self.GENERATED_SETTINGS
            if chosen:
                logger.info("instance_settings_seed_skipped_already_configured")
                return
            self.store.merge_instance_config("system_settings", seed)
            logger.info("instance_settings_seeded", settings=sorted(seed))
        except Exception as exc:  # noqa: BLE001 - a failed seed must not block boot
            logger.warning("instance_settings_seed_failed", error=str(exc))

    def _ensure_signing_key(self) -> None:
        """Put a signing key in the database on first boot.

        It used to be generated into a file under shared_fs_root, which no
        longer works: that path is itself a database setting now. Storing the
        key alongside it also fixes a quieter problem — every replica that
        could not read the file generated its own, so tokens issued by one
        worker were rejected by the next.
        """
        try:
            stored = self._system_settings_overrides()
            if stored.get("jwt_secret"):
                return
            self.store.merge_instance_config(
                "system_settings", {"jwt_secret": generate_jwt_secret()}
            )
            logger.info("jwt_secret_generated")
        except Exception as exc:  # noqa: BLE001 - surfaced by the auth service
            logger.error("jwt_secret_generation_failed", error=str(exc))

    def _build_credentialed_services(self) -> None:
        """(Re)build the services that capture credentials at construction.

        Both take their configuration as constructor arguments, so handing them
        a new settings object is not enough — rotating an SMTP password would
        update self.settings and change nothing about the mail that gets sent.
        Neither opens a connection when constructed, so rebuilding is cheap and
        a rotation applies to the next message rather than the next restart.
        """
        settings = self.settings
        self.voice = VoiceService(
            settings.shared_fs_root,
            api_key=settings.voice_api_key,
            transcription_model=settings.voice_transcription_model,
            synthesis_model=settings.voice_synthesis_model,
            default_voice=settings.voice_default_voice,
        )
        self.email = EmailService(
            smtp_host=settings.smtp_host,
            smtp_port=settings.smtp_port,
            smtp_user=settings.smtp_user,
            smtp_password=settings.smtp_password,
            smtp_security=settings.smtp_security,
            from_email=settings.email_from_address,
            from_name=settings.email_from_name,
            base_url=settings.app_base_url,
        )

    def refresh_settings(self) -> None:
        """Rebuild self.settings from the declared defaults plus what is stored.

        Everything downstream reads plain attributes off self.settings. Before
        this, sixty call sites each merged the store's dict by hand and wrote
        the default out a third time inline; they disagreed, and nothing
        noticed.
        """
        self.settings = apply_managed_settings(
            get_settings(), self._system_settings_overrides()
        )
        # Services built earlier hold a reference to the old object. They only
        # read attributes off it, so handing them the new one is enough — and
        # without this an admin's change would not reach a running auth or
        # email service until the process restarted.
        for service in ("auth", "workflow", "training"):
            existing = getattr(self, service, None)
            if existing is not None and hasattr(existing, "settings"):
                existing.settings = self.settings
        # These capture their configuration rather than reading it, so they
        # have to be rebuilt for a change to reach them.
        if getattr(self, "email", None) is not None:
            self._build_credentialed_services()

    def _build_model_services(self) -> None:
        """Construct all backend-dependent services from system settings.

        Shared by __init__ and reload_model_services so a runtime
        model_backend / model_path change takes effect without a restart.
        """
        self.refresh_settings()
        self.resolved_base_model = self.settings.model_path
        self.backend_mode = self.settings.model_backend
        self.default_adapter_mode = self.settings.default_adapter_mode
        self.rag_mode = self.settings.rag_mode
        embedding_model_id = self.settings.embedding_model_id
        rag_chunk_size = self.settings.rag_chunk_size
        adapter_configs = {
            "openai": {
                "api_key": self.settings.adapter_openai_api_key,
                "base_url": self.settings.adapter_openai_base_url,
                "adapter_server_model": self.settings.adapter_server_model,
            },
            "gemini": {
                "api_key": self.settings.gemini_api_key or None,
            },
        }
        self.router = RouterEngine(cache=self.cache, backend_mode=self.backend_mode)
        self.llm = LLMService(
            base_model=self.resolved_base_model,
            backend_mode=self.backend_mode,
            adapter_configs=adapter_configs,
            api_key=self.settings.adapter_openai_api_key,
            base_url=self.settings.adapter_openai_base_url,
            adapter_server_model=self.settings.adapter_server_model,
            fs_root=self.settings.shared_fs_root,
            reasoning_effort=self.settings.model_reasoning_effort,
            temperature=self.settings.model_temperature,
        )
        # Real embeddings when the backend has an OpenAI-compatible client
        # (its /embeddings endpoint serves OpenAI, Gemini-compat, and
        # self-hosted alike). Without one, the deterministic hash keeps the
        # kernel self-contained — and is_semantic=False tells every consumer
        # that cosine over those vectors is noise, so rankings stay BM25-only.
        embed_client = getattr(self.llm.backend, "client", None)
        if embed_client is not None and hasattr(embed_client, "embeddings"):
            self.embeddings = EmbeddingsService(
                embedding_model_id,
                encoder=make_provider_encoder(embed_client, embedding_model_id),
                batch_encoder=make_provider_batch_encoder(
                    embed_client, embedding_model_id
                ),
                semantic=True,
            )
        else:
            self.embeddings = EmbeddingsService(embedding_model_id)
        logger.info(
            "embeddings_configured",
            model=embedding_model_id,
            semantic=self.embeddings.is_semantic,
        )
        self.rag = RAGService(
            self.store,
            default_chunk_size=rag_chunk_size,
            rag_mode=self.rag_mode,
            embed=self.embeddings.embed,
            embed_many=self.embeddings.embed_many,
            embedding_model_id=embedding_model_id,
            semantic=self.embeddings.is_semantic,
            # A live reader, not a captured value: the reranker asks
            # self.settings on every retrieval, so its two settings take
            # effect on the next turn without rebuilding this stack.
            rerank=make_llm_reranker(self.llm, lambda: self.settings),
            late_interaction=self.settings.rag_late_interaction,
            late_segments=self.settings.rag_late_segments,
        )
        self.training = TrainingService(
            self.store,
            self.settings.shared_fs_root,
            runtime_base_model=self.resolved_base_model,
            default_adapter_mode=self.default_adapter_mode,
            backend_mode=self.backend_mode,
            max_active_training_jobs=self.settings.max_active_training_jobs,
            # SPEC §7.5: the serving LLM doubles as distillation teacher when
            # enabled; targets are rewritten into clean exemplars pre-training.
            teacher=self.llm,
            distillation_enabled=self.settings.training_distillation_enabled,
        )
        self.clusterer = SemanticClusterer(self.store, self.llm, self.training)
        self.workflow = WorkflowEngine(
            self.store,
            self.llm,
            self.router,
            self.rag,
            cache=self.cache,
            settings=self.settings,
            embeddings=self.embeddings,
        )
        self.config_ops = ConfigOpsService(
            self.store, self.llm, self.router, self.training
        )
        # Record the model-affecting settings this stack was built from so a
        # watcher can tell whether a later settings write actually changed them.
        self._model_settings_signature = self._model_signature()

    def _model_signature(self) -> tuple:
        """Signature of the settings that affect the model service stack."""
        effective = apply_managed_settings(
            get_settings(), self._system_settings_overrides()
        )
        return tuple(
            str(getattr(effective, name)) for name in MODEL_AFFECTING_SETTINGS
        )

    def _read_settings_version(self) -> Optional[str]:
        try:
            return self.store.get_system_settings_version()
        except Exception as exc:
            logger.warning("settings_version_read_failed", error=str(exc))
            return None

    def maybe_reload_model_services(self) -> bool:
        """Reload model services if another worker changed model settings.

        Each Uvicorn worker holds its own in-process Runtime, so a settings
        change made in one worker (and persisted to the store) must be observed
        by the others. Compares the persisted settings version to the one this
        worker last applied and, only when a model-affecting value actually
        differs, rebuilds the stack. Returns True if a reload occurred.
        """
        version = self._read_settings_version()
        if version is not None and version == self._applied_settings_version:
            return False
        target = self._model_signature()
        # Pick up every managed setting, not just the model ones: a rate-limit
        # or feature-flag change has to reach self.settings too, since that is
        # now what request handlers read.
        self.refresh_settings()
        # Record the version even when nothing model-relevant changed, so an
        # unrelated settings write isn't rechecked.
        self._applied_settings_version = version
        if target != self._model_settings_signature:
            self.reload_model_services()
            return True
        return False

    # Attributes set by _build_model_services; snapshotted for atomic reload.
    _MODEL_SERVICE_ATTRS = (
        "resolved_base_model",
        "backend_mode",
        "default_adapter_mode",
        "rag_mode",
        "router",
        "embeddings",
        "llm",
        "rag",
        "training",
        "clusterer",
        "workflow",
        "config_ops",
    )

    def reload_model_services(self) -> None:
        """Rebuild backend-dependent services after a system-settings change.

        Atomic: on any failure the previous services are restored so the runtime
        never runs on a half-rebuilt stack, and the exception propagates so the
        caller can report that the change did not take effect live. In-flight
        workflow threads finish on the old engine; new requests use the rebuilt
        services.
        """
        # Serialize rebuilds so a request-triggered reload and the background
        # settings watcher on the same worker can't rebuild concurrently.
        with self._reload_lock:
            snapshot = {
                name: getattr(self, name, None) for name in self._MODEL_SERVICE_ATTRS
            }
            old_workflow = getattr(self, "workflow", None)
            version = self._read_settings_version()
            try:
                self._build_model_services()
            except Exception as exc:
                # Restore the previous stack so a partial rebuild can't leave the
                # runtime inconsistent, then let the caller surface the failure.
                for name, value in snapshot.items():
                    setattr(self, name, value)
                logger.error(
                    "runtime_model_services_reload_failed",
                    model_backend=str(
                        self.settings.model_backend
                    ),
                    error=str(exc),
                )
                raise
            # Mark this version applied so the watcher doesn't reload again.
            self._applied_settings_version = version
            if getattr(self, "training_worker", None):
                self.training_worker.training = self.training
                self.training_worker.clusterer = self.clusterer
            if old_workflow is not None and old_workflow is not self.workflow:
                with contextlib.suppress(Exception):
                    old_workflow.shutdown(wait=False)
        logger.info(
            "runtime_model_services_reloaded",
            model_path=self.resolved_base_model,
            model_backend=str(self.backend_mode),
            rag_mode=str(self.rag_mode),
        )

    async def close(self) -> None:
        """Cleanup resources for graceful shutdown (Issues 57.7, 59.1)."""

        if getattr(self, "training_worker", None):
            with contextlib.suppress(Exception):
                await self.training_worker.stop()

        if getattr(self, "bus", None):
            with contextlib.suppress(Exception):
                await self.bus.stop()

        if getattr(self, "workflow", None):
            with contextlib.suppress(Exception):
                self.workflow.shutdown(wait=True)

        if getattr(self, "voice", None):
            with contextlib.suppress(Exception):
                await self.voice.close()

        if getattr(self, "cache", None):
            with contextlib.suppress(Exception):
                await self.cache.close()

        # A runtime closes what it built. Under test the store is shared
        # across the session, and the app's lifespan shutdown runs whenever a
        # test exercises it — which closed the pool every later test needed.
        if getattr(self, "store", None) and self.store is not _shared_store:
            with contextlib.suppress(Exception):
                await self.store.close()


runtime: Runtime | None = None
# Issue 28.1: Thread-safe singleton pattern using a lock
_runtime_lock = threading.Lock()


def get_runtime() -> Runtime:
    """Get or create the Runtime singleton in a thread-safe manner.

    Uses double-checked locking pattern for efficiency:
    - First check without lock (fast path for existing runtime)
    - Second check with lock to prevent race condition during creation
    """
    global runtime
    with _runtime_lock:
        if runtime is None:
            runtime = Runtime()
        return runtime


def reset_runtime_for_tests() -> Runtime:
    """Reinitialize the runtime singleton for isolated test runs.

    Uses the same thread lock as get_runtime for safety (Issue 28.1).
    """
    import asyncio

    global runtime

    with _runtime_lock:
        # Close existing Redis connections to avoid event loop issues
        if runtime is not None and runtime.cache is not None:
            try:
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(runtime.cache.close())
                except RuntimeError:
                    asyncio.run(runtime.cache.close())
            except Exception:
                # Ignore errors during cleanup - connection may already be closed
                pass

        # Close the store's connection pool too. Replacing the runtime without
        # this leaks a pool per reset — invisible while the store was in-memory,
        # and it exhausts Postgres connections the moment it is not.
        # Not the shared one: it outlives every reset by design, and closing
        # its pool here would make the next test pay to rebuild it.
        if (
            runtime is not None
            and getattr(runtime, "store", None) is not None
            and runtime.store is not _shared_store
        ):
            runtime.store.close_pool()

        reset_settings_cache()
        settings = get_settings()
        if not settings.test_mode:
            raise RuntimeError("runtime reset is only allowed in TEST_MODE")
        runtime = Runtime()
        return runtime


IDEMPOTENCY_TTL_SECONDS = 60 * 60 * 24


def _cleanup_local_idempotency(
    runtime: Runtime, now: datetime
) -> tuple[int, int]:
    """Prune expired or excess in-memory idempotency records.

    Returns a tuple of (expired_removed, evicted_removed).
    """

    expired = [
        key
        for key, record in runtime._local_idempotency.items()
        if record.get("expires_at") and record["expires_at"] < now
    ]
    for key in expired:
        runtime._local_idempotency.pop(key, None)

    evicted = 0
    if len(runtime._local_idempotency) > runtime._local_idempotency_max_entries:
        # Evict oldest entries first
        sorted_items = sorted(
            runtime._local_idempotency.items(),
            key=lambda item: item[1].get("expires_at", now),
        )
        excess = len(runtime._local_idempotency) - runtime._local_idempotency_max_entries
        for i in range(excess):
            runtime._local_idempotency.pop(sorted_items[i][0], None)
            evicted += 1
    return len(expired), evicted


async def _set_cached_idempotency_record(
    runtime: Runtime,
    route: str,
    user_id: str,
    key: str,
    record: dict,
    *,
    ttl_seconds: int = IDEMPOTENCY_TTL_SECONDS,
    tenant_id: Optional[str] = None,
) -> None:
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)
    # Under the account's lifetime lock, because this record holds a completed
    # API response and lives for a day. A request authorized before an erasure
    # is allowed to finish, and finishing here after the erasure's purge put
    # the account's own content back under a key naming the account. See
    # `PostgresStore.hold_live_user`; the whole write is inside the guard, not
    # after a question asked before it.
    with runtime.store.hold_live_user(user_id) as live:
        if not live:
            return
        if runtime.cache:
            # Issue 22.2: Pass tenant_id for multi-tenant isolation
            await runtime.cache.set_idempotency_record(
                route, user_id, key, record, ttl_seconds=ttl_seconds, tenant_id=tenant_id
            )
            return
        async with runtime._local_idempotency_lock:
            # Include tenant_id in in-memory key for multi-tenant isolation
            cache_key = (
                (tenant_id, route, user_id, key) if tenant_id else (route, user_id, key)
            )
            _cleanup_local_idempotency(runtime, datetime.now(timezone.utc))
            runtime._local_idempotency[cache_key] = {**record, "expires_at": expires_at}


async def _acquire_idempotency_slot(
    runtime: Runtime,
    route: str,
    user_id: str,
    key: str,
    record: dict,
    *,
    ttl_seconds: int = IDEMPOTENCY_TTL_SECONDS,
    tenant_id: Optional[str] = None,
) -> tuple[bool, Optional[dict]]:
    """Atomically acquire an idempotency slot (Issue 19.4).

    Uses SETNX for Redis or lock-protected check-and-set for in-memory fallback.

    Args:
        runtime: Application runtime
        route: Route/operation name
        user_id: User ID
        key: Idempotency key
        record: Record to set if slot acquired
        ttl_seconds: TTL for the record
        tenant_id: Optional tenant ID for multi-tenant isolation (Issue 22.2)

    Returns:
        Tuple of (acquired: bool, existing_record: Optional[dict])
    """
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(seconds=ttl_seconds)

    # Same guard as storing the result, and for the same reason: this claim is
    # also a key naming the account, kept for a day. Reporting the slot as
    # acquired lets a request whose account has just gone still finish; it
    # simply records nothing, here or when its result comes back.
    #
    # The whole claim is inside the guard, not a question asked before it. An
    # answer released before the write is a check-then-act across the
    # deletion — the erasure commits and purges in the gap, and the claim
    # lands afterwards under a key the purge has already been past.
    with runtime.store.hold_live_user(user_id) as live:
        if not live:
            return (True, None)

        if runtime.cache:
            # Issue 22.2: Pass tenant_id for multi-tenant isolation
            return await runtime.cache.acquire_idempotency_slot(
                route, user_id, key, record, ttl_seconds=ttl_seconds, tenant_id=tenant_id
            )

        # In-memory fallback with atomic check-and-set within lock
        async with runtime._local_idempotency_lock:
            # Include tenant_id in in-memory key for multi-tenant isolation
            cache_key = (
                (tenant_id, route, user_id, key) if tenant_id else (route, user_id, key)
            )
            expired, evicted = _cleanup_local_idempotency(runtime, now)
            if expired or evicted:
                logger.info(
                    "idempotency_local_cleanup",
                    expired=expired,
                    evicted=evicted,
                    remaining=len(runtime._local_idempotency),
                )
            existing = runtime._local_idempotency.get(cache_key)

            if existing:
                is_expired = existing.get("expires_at") and existing["expires_at"] < now
                # Reclaim a prior failed attempt atomically (within this lock)
                # so a retry proceeds without a separate racy overwrite.
                is_failed = existing.get("status") == "failed"
                if is_expired or is_failed:
                    runtime._local_idempotency.pop(cache_key, None)
                else:
                    # Live in-progress/completed record: return it.
                    return (False, existing)

            # No existing record or it was expired, claim the slot
            runtime._local_idempotency[cache_key] = {
                **record,
                "expires_at": expires_at,
            }
            return (True, None)


async def check_rate_limit(
    runtime: Runtime,
    key: str,
    limit: int,
    window_seconds: int,
    *,
    return_remaining: bool = False,
    cost: int = 1,
) -> Union[bool, Tuple[bool, int, int]]:
    """Enforce rate limits even when Redis is unavailable.

    Per SPEC §18, rate limits use Redis token bucket with configurable defaults.

    Args:
        runtime: Runtime instance with cache
        key: Rate limit key
        limit: Maximum requests per window
        window_seconds: Window duration in seconds
        return_remaining: If True, return tuple of (allowed, remaining)

    Returns:
        bool if return_remaining is False, else (bool, int, int) tuple
    """
    if limit <= 0:
        return (True, limit, 0) if return_remaining else True
    rate_limit_key = key
    if window_seconds <= 0:
        logger.warning(
            "rate_limit_invalid_window",
            key=rate_limit_key,
            window_seconds=window_seconds,
            message="Invalid rate limit window_seconds; defaulting to 60 seconds",
        )
        window_seconds = 60  # Default to 1 minute if the configured window is invalid
    now = datetime.now(timezone.utc)
    if runtime.cache:
        result = await runtime.cache.check_rate_limit(
            rate_limit_key,
            limit,
            window_seconds,
            return_remaining=return_remaining,
            cost=cost,
        )
        return result
    refill_rate = float(limit) / float(window_seconds)
    async with runtime._local_rate_limit_lock:
        now = datetime.now(timezone.utc)
        # Cleanup old or excess entries to prevent unbounded growth (Issue 57.2)
        max_age_seconds = max(window_seconds * 2, 3600)
        if (now - runtime._local_rate_limit_last_cleanup).total_seconds() > 300:
            runtime._local_rate_limit_last_cleanup = now
            expired_keys = [
                stale_key
                for stale_key, (_, ts) in runtime._local_rate_limits.items()
                if (now - ts).total_seconds() > max_age_seconds
            ]
            for stale_key in expired_keys:
                runtime._local_rate_limits.pop(stale_key, None)
            if len(runtime._local_rate_limits) > runtime._local_rate_limit_max_entries:
                # Evict oldest by timestamp
                sorted_items = sorted(
                    runtime._local_rate_limits.items(), key=lambda item: item[1][1]
                )
                excess = len(runtime._local_rate_limits) - runtime._local_rate_limit_max_entries
                for i in range(excess):
                    runtime._local_rate_limits.pop(sorted_items[i][0], None)
            if expired_keys:
                logger.info(
                    "rate_limit_local_cleanup",
                    cleaned=len(expired_keys),
                    remaining=len(runtime._local_rate_limits),
                )
        tokens, last_ts = runtime._local_rate_limits.get(rate_limit_key, (float(limit), now))
        elapsed = max(0.0, (now - last_ts).total_seconds())
        tokens = min(float(limit), tokens + elapsed * refill_rate)
        allowed = tokens >= cost
        if allowed:
            tokens -= cost
            runtime._local_rate_limits[rate_limit_key] = (tokens, now)
        reset_seconds = int(((cost - tokens) / refill_rate)) if not allowed and refill_rate > 0 else 0
        remaining = int(tokens)
    if return_remaining:
        return (allowed, remaining, reset_seconds)
    return allowed
