from __future__ import annotations

import logging
import os
import sys
import uuid
from contextvars import ContextVar
from typing import Any, Dict, Optional

import structlog

# Context variable for correlation ID (per-request tracking per SPEC §15.2)
correlation_id_var: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)


def get_correlation_id() -> Optional[str]:
    """Get the current correlation ID for request tracing."""
    return correlation_id_var.get()


def set_correlation_id(correlation_id: Optional[str] = None) -> str:
    """Set or generate a correlation ID for the current request context."""
    cid = correlation_id or str(uuid.uuid4())
    correlation_id_var.set(cid)
    return cid


def _add_correlation_id(
    logger: Any, method_name: str, event_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """Processor to add correlation_id to all log entries."""
    cid = get_correlation_id()
    if cid:
        event_dict["correlation_id"] = cid
    return event_dict


def _redact_pii(
    logger: Any, method_name: str, event_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """Processor to redact PII from log entries per SPEC §15.2."""
    pii_keys = {"password", "secret", "token", "api_key", "authorization", "email", "ssn"}
    for key in list(event_dict.keys()):
        lower_key = key.lower()
        if any(pii in lower_key for pii in pii_keys):
            if isinstance(event_dict[key], str) and len(event_dict[key]) > 4:
                # Redact but preserve first/last 2 chars for debugging
                event_dict[key] = event_dict[key][:2] + "***" + event_dict[key][-2:]
    return event_dict


def _configure_structlog(
    log_level: str = "INFO",
    json_output: bool = True,
    development_mode: bool = False,
) -> None:
    """Configure structlog with processors per SPEC §15.2.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        json_output: If True, output JSON; if False, output human-readable
        development_mode: If True, use pretty console output
    """
    # Shared processors for all modes
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        _add_correlation_id,
        _redact_pii,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if development_mode or not json_output:
        # Human-readable output for development
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True),
        ]
    else:
        # JSON output for production per SPEC §15.2
        processors = shared_processors + [
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


# Initialize logging on module import
_log_level = os.getenv("LOG_LEVEL", "INFO").upper()
_json_output = os.getenv("LOG_JSON", "true").lower() in {"1", "true", "yes", "on"}
_dev_mode = os.getenv("LOG_DEV_MODE", "false").lower() in {"1", "true", "yes", "on"}

_configure_structlog(
    log_level=_log_level,
    json_output=_json_output,
    development_mode=_dev_mode,
)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a configured logger with correlation ID support.

    Per SPEC §15.2: Structured logs with correlation IDs for each chat request.
    """
    return structlog.get_logger(name)


def log_routing_trace(trace: list, logger: Optional[Any] = None) -> None:
    """Log routing trace per SPEC §15.2: rules fired, adapters activated."""
    log = logger or get_logger("routing")
    log.info("routing_trace", trace=trace)


def log_workflow_trace(trace: list, logger: Optional[Any] = None) -> None:
    """Log workflow trace per SPEC §15.2: nodes executed, errors."""
    log = logger or get_logger("workflow")
    log.info("workflow_trace", trace=trace)
