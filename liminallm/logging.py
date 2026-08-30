from __future__ import annotations

import logging
import os
import re
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
) -> None:
    """Configure structlog with processors per SPEC §15.2.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        json_output: If True, output JSON; if False, pretty console output
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

    if not json_output:
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
_configure_structlog(log_level=_log_level, json_output=_json_output)


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
    # A SQL statement pasted into an exception message. Matching a bare verb
    # would eat ordinary prose - "delete conversation failed", "update the
    # adapter first" - so a clause keyword has to follow the verb before this
    # counts as SQL. These messages are shown to users; over-redaction makes
    # the field useless, which is its own failure.
    r'(?i)\b(select|insert|update|delete|replace)\b[^\n]{0,200}?'
    r'\b(from|into|where|values|set|join)\b[^\n]*',
    r'(?i)database\s+error',
    r'(?i)connection\s+.*\s+(failed|refused|timeout)',
    # Path/file related
    r'(?i)/(?:home|var|etc|usr|opt|tmp)/[^\s]+',
    r'(?i)[a-z]:\\[^\s]+',
    # Credential patterns
    r'(?i)(password|secret|token|key|credential|api.?key)\s*[:=]\s*[^\s]+',
    # An Authorization header pasted into an exception message. Not caught by
    # the line above: "Authorization" is not one of its words, and
    # "Bearer eyJ..." has no ':' or '=' after a matching one.
    r'(?i)\bauthorization\s*:\s*\S+',
    r'(?i)\b(bearer|basic)\s+[A-Za-z0-9._~+/=-]{8,}',
    # A bare JWT, which reaches a log without its header often enough.
    r'\beyJ[A-Za-z0-9_-]{4,}\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+',
    # A connection URI carrying credentials - psycopg puts the DSN in its
    # exception text, so this is the common way a database password escapes.
    r'(?i)\b[a-z][a-z0-9+.\-]*://[^\s:@/]+:[^\s@]+@\S*',
    # Stack traces
    r'(?i)traceback\s*\(most recent call last\)',
    r'(?i)at\s+\S+\.\S+\(\S+:\d+\)',
    # Internal function names that might leak implementation
    r'(?i)_internal_|_private_|__[a-z]+__',
]

# Compiled patterns for performance
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
                # bool before int: bool is a subclass of int, so the numeric
                # branch used to swallow it and a redacted flag came back as 0.
                if isinstance(value, bool):
                    result[key] = False
                elif isinstance(value, (int, float)):
                    result[key] = 0
                else:
                    result[key] = "[REDACTED]"
            else:
                result[key] = sanitize_response_data(value, depth=depth + 1, max_depth=max_depth)
        return result
    elif isinstance(data, list):
        return [sanitize_response_data(item, depth=depth + 1, max_depth=max_depth) for item in data]
    else:
        return data


