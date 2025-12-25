from __future__ import annotations

import logging
import os
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


# Issue 47.1-47.8: Content sanitization for API responses
# Patterns that indicate sensitive information in error messages
_SENSITIVE_ERROR_PATTERNS = [
    # Database/SQL related
    r'(?i)(sql|query|select|insert|update|delete|where|from|join)\s+.{0,50}',
    r'(?i)database\s+error',
    r'(?i)connection\s+.*\s+(failed|refused|timeout)',
    # Path/file related
    r'(?i)/(?:home|var|etc|usr|opt|tmp)/[^\s]+',
    r'(?i)[a-z]:\\[^\s]+',
    # Credential patterns
    r'(?i)(password|secret|token|key|credential|api.?key)\s*[:=]\s*[^\s]+',
    # Stack traces
    r'(?i)traceback\s*\(most recent call last\)',
    r'(?i)at\s+\S+\.\S+\(\S+:\d+\)',
    # Internal function names that might leak implementation
    r'(?i)_internal_|_private_|__[a-z]+__',
]

# Compiled patterns for performance
import re

_SENSITIVE_PATTERNS_COMPILED = [re.compile(p) for p in _SENSITIVE_ERROR_PATTERNS]

# Keys that should be redacted in response data
_SENSITIVE_RESPONSE_KEYS = frozenset({
    'password', 'secret', 'token', 'api_key', 'apikey', 'api-key',
    'authorization', 'auth', 'credentials', 'private_key', 'privatekey',
    'ssn', 'social_security', 'credit_card', 'creditcard', 'cvv',
    'secret_key', 'secretkey', 'access_key', 'accesskey',
})


def sanitize_error_message(error: str, *, replacement: str = "[redacted]") -> str:
    """Sanitize error message for API responses (Issue 47.1).

    Removes sensitive information like:
    - SQL queries and database errors
    - File paths and internal paths
    - Credentials and secrets
    - Stack traces

    Args:
        error: Original error message
        replacement: String to replace sensitive content with

    Returns:
        Sanitized error message safe for API responses
    """
    if not error or not isinstance(error, str):
        return "An error occurred"

    result = error
    for pattern in _SENSITIVE_PATTERNS_COMPILED:
        result = pattern.sub(replacement, result)

    # Limit length to prevent excessive error messages
    if len(result) > 500:
        result = result[:497] + "..."

    return result


def sanitize_response_data(data: Any, *, depth: int = 0, max_depth: int = 20) -> Any:
    """Sanitize response data by redacting sensitive keys (Issue 47.7).

    Args:
        data: Data to sanitize (dict, list, or primitive)
        depth: Current recursion depth
        max_depth: Maximum recursion depth

    Returns:
        Sanitized data with sensitive values redacted
    """
    if depth > max_depth:
        return "[max depth exceeded]"

    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            lower_key = key.lower().replace('-', '_').replace(' ', '_')
            if any(sensitive in lower_key for sensitive in _SENSITIVE_RESPONSE_KEYS):
                # Redact sensitive values but preserve type hint
                if isinstance(value, str):
                    result[key] = "[REDACTED]"
                elif isinstance(value, (int, float)):
                    result[key] = 0
                elif isinstance(value, bool):
                    result[key] = False
                else:
                    result[key] = "[REDACTED]"
            else:
                result[key] = sanitize_response_data(value, depth=depth + 1, max_depth=max_depth)
        return result
    elif isinstance(data, list):
        return [sanitize_response_data(item, depth=depth + 1, max_depth=max_depth) for item in data]
    else:
        return data


def sanitize_workflow_trace(trace: list) -> list:
    """Sanitize workflow trace for API responses (Issue 47.6).

    Removes internal details like:
    - Detailed error messages with stack traces
    - Internal node configuration
    - Debugging information

    Args:
        trace: Raw workflow trace

    Returns:
        Sanitized trace safe for API responses
    """
    sanitized = []
    for entry in trace:
        if not isinstance(entry, dict):
            continue

        safe_entry = {
            "node_id": entry.get("node_id"),
            "status": entry.get("status"),
            "duration_ms": entry.get("duration_ms"),
        }

        # Include error message only if sanitized
        if "error" in entry:
            safe_entry["error"] = sanitize_error_message(str(entry.get("error", "")))

        # Include output keys but not values (for debugging without data exposure)
        if "output" in entry and isinstance(entry["output"], dict):
            safe_entry["output_keys"] = list(entry["output"].keys())

        sanitized.append(safe_entry)

    return sanitized
