"""What an error body tells the caller.

`sanitize_error_message` and `sanitize_response_data` were written for exactly
this and then never called from anything but their own tests, so two things
leaked in production: the server's filesystem root, on any traversal probe,
and the regex of the policy blocklist, which turns a moderation filter into an
oracle you can iterate against. Both are fixed at the raise site; `_error_response`
now runs the sanitizers as well, because it is the one place an error leaves
the process.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi import HTTPException

from liminallm.api.error_handling import _error_response
from liminallm.service.errors import ServiceError
from liminallm.service.runtime import get_runtime
from liminallm.storage.common import ensure_policy_compliant_texts
from liminallm.storage.errors import ConstraintViolation


def _body(response):
    import json

    return json.loads(bytes(response.body))


# ---------------------------------------------------------------------------
# The envelope itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "leak",
    [
        "/home/liminal/shared/uploads",
        "postgresql://liminal:hunter2@db.internal:5432/liminallm",
        "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.abc.def",
        "SELECT id FROM app_user WHERE email = 'alice@example.com'",
    ],
    ids=["path", "dsn", "bearer", "sql"],
)
def test_a_secret_in_a_message_does_not_reach_the_body(leak):
    body = _body(_error_response(400, f"failed: {leak}"))
    assert leak not in body["error"]["message"]


def test_a_secret_in_the_details_does_not_reach_the_body():
    """`details` is the machine-built half — a driver's exception text, a
    resolved path, a matched pattern — and leaked while only `message` was
    ever considered."""
    body = _error_response(409, "conflict", {"password": "hunter2"})
    assert _body(body)["error"]["details"]["password"] == "[REDACTED]"


def test_an_ordinary_message_survives_intact():
    """Over-redaction makes the field useless, which is its own failure. The
    SQL pattern used to match a bare verb, so this message came back as
    "[redacted]"."""
    for message in (
        "conversation not found",
        "delete the adapter first",
        "cannot update a promoted adapter",
        "select a context before asking",
    ):
        assert _body(_error_response(400, message))["error"]["message"] == message


def test_a_real_sql_statement_is_still_caught():
    body = _body(_error_response(500, "UPDATE app_user SET role = 'admin' WHERE id = 1"))
    assert "app_user" not in body["error"]["message"]


def test_no_details_stays_no_details():
    assert _body(_error_response(404, "not found"))["error"]["details"] is None


# ---------------------------------------------------------------------------
# The two leaks, at the source
# ---------------------------------------------------------------------------


def test_the_policy_filter_does_not_name_the_pattern_that_matched(caplog):
    """Otherwise: submit, read which rule fired, edit, repeat until nothing
    fires. The operator still gets it, in the log."""
    with pytest.raises(ConstraintViolation) as caught:
        ensure_policy_compliant_texts(["<script>alert(1)</script>"], violation_context={"user_id": "u1"})

    rendered = repr(caught.value.detail)
    assert "script" not in rendered
    assert caught.value.detail["user_id"] == "u1"


def test_a_policy_violation_still_says_which_request_it_was():
    """Redacting the pattern must not take the correlation fields with it."""
    with pytest.raises(ConstraintViolation) as caught:
        ensure_policy_compliant_texts(
            ["build a bomb"], violation_context={"message_id": "m1", "conversation_id": "c1"}
        )
    assert caught.value.detail == {"message_id": "m1", "conversation_id": "c1"}


def test_compliant_text_raises_nothing():
    ensure_policy_compliant_texts(["a perfectly ordinary sentence", None, ""])


def test_a_traversal_probe_is_not_told_the_servers_root():
    """`ingest_path` puts the base in the log line; the 400 body must not
    repeat it, or probing with `../` is free reconnaissance."""
    from liminallm.service.fs import PathTraversalError

    rag = get_runtime().rag
    with pytest.raises(PathTraversalError) as caught:
        rag.ingest_path("ctx", "/etc/passwd", allowed_base="/srv/liminal/shared")

    assert "/srv/liminal/shared" not in str(caught.value)


# ---------------------------------------------------------------------------
# Reaching the handlers through a real request
# ---------------------------------------------------------------------------


def test_a_service_error_reaches_the_client_sanitized(client, auth_headers):
    """End to end: whatever a route raises, the envelope is scrubbed."""
    resp = client.get(f"/v1/conversations/{uuid.uuid4()}", headers=auth_headers)
    assert resp.status_code in (403, 404)
    body = resp.json()
    assert body["status"] == "error"
    assert "/home/" not in body["error"]["message"]


def test_the_handler_survives_an_error_with_no_details():
    """The sanitizer must not turn a clean 4xx into a 500 by tripping on
    None — it runs inside the exception handler."""
    exc = ServiceError("plain failure")
    assert _error_response(exc.status_code, exc.message, None).status_code >= 400


def test_an_unknown_error_code_still_produces_a_valid_envelope():
    body = _body(_error_response(409, "conflict", None, code="not_a_real_code"))
    assert body["error"]["code"] == "conflict"


def test_an_http_exception_detail_dict_is_scrubbed():
    exc = HTTPException(status_code=400, detail={"detail": "path /home/liminal/x denied"})
    body = _body(_error_response(exc.status_code, exc.detail["detail"]))
    assert "/home/liminal/x" not in body["error"]["message"]
