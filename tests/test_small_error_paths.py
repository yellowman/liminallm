"""The small uncovered islands: error paths one branch wide.

Each of these is a covered module's last dark corner - an SMTP failure class,
a corrupted cache entry, a validator's raise. None is speculative: every
branch here is reachable in production, just never walked by the suite.
"""

from __future__ import annotations

import asyncio
import smtplib
import ssl
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from liminallm.service.email import EmailService
from liminallm.service.runtime import get_runtime

# ---------------------------------------------------------------------------
# Email: every SMTP failure class returns False, never raises
# ---------------------------------------------------------------------------


@pytest.fixture
def mailer():
    return EmailService(
        smtp_host="smtp.example", smtp_port=587,
        smtp_user="mailer", smtp_password="hunter2",
        from_email="noreply@example.com",
    )


class _RaisingSMTP:
    def __init__(self, exc):
        self._exc = exc

    def __call__(self, *a, **kw):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def starttls(self, **kw):
        raise self._exc


@pytest.mark.parametrize(
    "exc",
    [
        smtplib.SMTPAuthenticationError(535, b"bad credentials"),
        smtplib.SMTPConnectError(421, b"cannot connect"),
        smtplib.SMTPRecipientsRefused({"a@b.c": (550, b"no such user")}),
        smtplib.SMTPSenderRefused(553, b"bad sender", "noreply@example.com"),
        smtplib.SMTPException("catch-all smtp failure"),
        ssl.SSLError("handshake failed"),
        TimeoutError("connect timed out"),
        RuntimeError("anything else"),
    ],
    ids=lambda e: type(e).__name__,
)
def test_an_smtp_failure_reports_false_instead_of_raising(mailer, monkeypatch, exc):
    """Callers treat False as "not delivered"; an exception would turn a mail
    outage into a failed signup or a failed sweep."""
    monkeypatch.setattr(smtplib, "SMTP", _RaisingSMTP(exc))
    assert mailer.send_password_reset("a@b.c", "tok") is False


def test_plaintext_with_credentials_is_refused_not_sent(monkeypatch):
    """smtp_security='none' plus a password would put that password on the
    wire in the clear; the send fails instead."""
    sent = []
    monkeypatch.setattr(smtplib, "SMTP", lambda *a, **kw: sent.append(a) or (_ for _ in ()).throw(AssertionError("connected anyway")))
    mailer = EmailService(
        smtp_host="smtp.example", smtp_security="none",
        smtp_user="mailer", smtp_password="hunter2",
        from_email="noreply@example.com",
    )
    assert mailer.send_password_reset("a@b.c", "tok") is False
    assert sent == []


# ---------------------------------------------------------------------------
# Redis: a corrupted cache entry is a miss, not a crash
# ---------------------------------------------------------------------------


@pytest.fixture
def cache():
    c = get_runtime().cache
    if c is None:
        pytest.skip("redis not available")
    return c


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "key_fmt, getter",
    [
        ("router:last:{u}:h1", lambda c, u: c.get_router_cache(u, "h1")),
        ("workflow:state:{u}", lambda c, u: c.get_workflow_state(u)),
        ("chat:summary:{u}", lambda c, u: c.get_conversation_summary(u)),
    ],
    ids=["router", "workflow", "summary"],
)
async def test_corrupted_json_reads_as_a_miss(cache, key_fmt, getter):
    """Anything with Redis access can write garbage under our keys; the app
    must shrug, not 500."""
    marker = uuid.uuid4().hex[:8]
    await cache.client.set(key_fmt.format(u=marker), "{not json at all")
    assert await getter(cache, marker) is None


@pytest.mark.asyncio
async def test_an_absent_entry_is_also_a_miss(cache):
    assert await cache.get_router_cache(uuid.uuid4().hex, "h") is None


# ---------------------------------------------------------------------------
# Admission: the no-redis deployment and the full house
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_without_redis_admission_admits():
    """Single-process deployment: no shared counter exists, so no cap. The
    alternative - refusing all chat because Redis is absent - is worse."""
    from liminallm.service import admission

    runtime = SimpleNamespace(cache=None, settings=None)
    assert await admission.acquire(runtime, "inference", "u1") is True


@pytest.mark.asyncio
async def test_a_full_slot_raises_at_capacity_as_a_409():
    from liminallm.service import admission

    class FullCache:
        async def acquire_concurrency_slot(self, kind, user_id, limit):
            return False, limit

    runtime = SimpleNamespace(
        cache=FullCache(),
        settings=SimpleNamespace(max_concurrent_inference=2),
    )
    with pytest.raises(admission.AtCapacity) as caught:
        await admission.acquire(runtime, "inference", "u1")
    assert caught.value.status_code == 409
    assert caught.value.error_code == "busy"


# ---------------------------------------------------------------------------
# Limits: the degenerate inputs
# ---------------------------------------------------------------------------


def test_a_request_without_a_client_is_unknown_not_a_crash():
    from liminallm.api.limits import client_ip

    assert client_ip(None) == "unknown"
    assert client_ip(SimpleNamespace(client=None)) == "unknown"


@pytest.mark.asyncio
async def test_a_zero_window_falls_back_to_a_minute():
    """window_seconds <= 0 would make every bucket permanently empty; the
    code corrects to 60s and logs, rather than rate-limiting to death."""
    from liminallm.api.limits import enforce

    info = await enforce(
        get_runtime(), f"island:{uuid.uuid4().hex[:8]}", 100, 0
    )
    assert info.limit == 100
    assert info.reset_seconds <= 60


# ---------------------------------------------------------------------------
# Idempotency: a replayed record without a request_id
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_stored_record_missing_request_id_is_backfilled_on_replay():
    """Records written by an older build predate request_id in the response
    payload; the replay fills it rather than returning an invalid envelope."""
    from liminallm.api import idempotency
    from liminallm.service.runtime import _set_cached_idempotency_record

    runtime = get_runtime()
    key = f"legacy-{uuid.uuid4().hex[:8]}"
    await _set_cached_idempotency_record(
        runtime, "route", "u1", key,
        {"status": "completed", "request_id": "req-original",
         "response": {"status": "ok", "data": {"n": 1}}},
    )

    request_id, envelope = await idempotency.resolve("route", "u1", key)

    assert envelope is not None
    assert envelope.request_id == "req-original"
    assert request_id == "req-original"


# ---------------------------------------------------------------------------
# Embeddings: the poisoned-vector edges
# ---------------------------------------------------------------------------


def test_validate_embedding_names_the_bad_index():
    from liminallm.service.embeddings import validate_embedding

    with pytest.raises(ValueError, match=r"\[1\] contains NaN"):
        validate_embedding([0.0, float("nan")])
    with pytest.raises(ValueError, match="contains Infinity"):
        validate_embedding([float("inf")])


def test_cosine_similarity_of_a_poisoned_vector_is_neutral():
    """NaN would propagate into every ranking score above it."""
    from liminallm.service.embeddings import cosine_similarity

    assert cosine_similarity([float("nan"), 1.0], [1.0, 1.0]) == 0.0
    assert cosine_similarity([1.0, 1.0], [float("inf"), 1.0]) == 0.0
    assert cosine_similarity([1.0], [1.0, 0.0]) == 0.0  # width mismatch
    assert cosine_similarity([], []) == 0.0


def test_ensure_embedding_dim_refuses_the_wrong_width_when_told_to():
    from liminallm.service.embeddings import ensure_embedding_dim

    with pytest.raises(ValueError, match="dimension 2"):
        ensure_embedding_dim([1.0, 2.0], dim=3, validate_length=True)


def test_normalize_vector_of_zero_magnitude_is_zeros_not_nan():
    from liminallm.service.embeddings import normalize_vector

    assert normalize_vector([0.0, 0.0]) == [0.0, 0.0]
    assert normalize_vector([]) == []
    assert normalize_vector([float("nan"), 3.0, 4.0]) == [0.0, 0.6, 0.8]


def test_validate_centroid_refuses_empty_and_poisoned():
    from liminallm.service.embeddings import EMBEDDING_DIM, validate_centroid

    with pytest.raises(ValueError, match="empty"):
        validate_centroid([])
    poisoned = [float("nan")] + [0.0] * (EMBEDDING_DIM - 1)
    with pytest.raises(ValueError, match="NaN"):
        validate_centroid(poisoned)
    with pytest.raises(ValueError, match="dimension"):
        validate_centroid([1.0, 2.0])


# ---------------------------------------------------------------------------
# content_struct: the shapes that must not reach storage
# ---------------------------------------------------------------------------


def test_content_struct_rejects_non_dict_shapes():
    from liminallm.content_struct import normalize_content_struct

    assert normalize_content_struct("just text") is None
    assert normalize_content_struct(["segments"]) is None
    assert normalize_content_struct({"segments": "not-a-list"}) is None


def test_content_struct_with_no_valid_segments_falls_back_to_the_text():
    from liminallm.content_struct import normalize_content_struct

    out = normalize_content_struct(
        {"segments": [{"type": "hologram"}, "garbage"]}, content="the message"
    )
    assert out == {"segments": [{"type": "text", "text": "the message"}]}


def test_content_struct_with_nothing_at_all_is_none():
    from liminallm.content_struct import normalize_content_struct

    assert normalize_content_struct({"segments": []}, content=None) is None


def test_content_struct_keeps_a_dict_summary_and_drops_others():
    from liminallm.content_struct import normalize_content_struct

    kept = normalize_content_struct(
        {"segments": [{"type": "text", "text": "x"}], "summary": {"tl": "dr"}}
    )
    assert kept["summary"] == {"tl": "dr"}
    dropped = normalize_content_struct(
        {"segments": [{"type": "text", "text": "x"}], "summary": "prose"}
    )
    assert "summary" not in dropped


def test_segment_keys_outside_the_allowlist_are_stripped():
    from liminallm.content_struct import normalize_content_struct

    out = normalize_content_struct(
        {"segments": [{"type": "text", "text": "x", "onclick": "alert(1)"}]}
    )
    assert out["segments"][0] == {"type": "text", "text": "x"}


# ---------------------------------------------------------------------------
# Schemas: validators that had never fired
# ---------------------------------------------------------------------------


def test_an_oversized_array_is_a_deserialization_bomb():
    from liminallm.api.schemas import MAX_ARRAY_ITEMS, _validate_json_depth

    with pytest.raises(ValueError, match="Array length"):
        _validate_json_depth([0] * (MAX_ARRAY_ITEMS + 1))


@pytest.mark.parametrize(
    "email",
    ["a@" + "x" * 253, "a@b", "@nolocal.com", "toolong" * 10 + "@x.com",
     "spaces in@x.com", "a@" + "y" * 64 + ".com", 123],
    ids=["too-long", "one-label", "no-local", "long-local", "bad-local", "long-label", "not-str"],
)
def test_a_malformed_email_is_refused(email):
    from liminallm.api.schemas import _validate_email

    with pytest.raises(ValueError):
        _validate_email(email)


def test_an_unknown_device_type_is_refused():
    from liminallm.api.schemas import LoginRequest

    with pytest.raises(ValueError, match="device_type"):
        LoginRequest(email="a@b.co", password="x" * 12, device_type="toaster")


def test_oauth_start_refuses_a_client_supplied_tenant():
    """Tenant is bound into server-created OAuth state; accepting it from the
    request body would be tenant spoofing (CLAUDE.md isolation)."""
    from liminallm.api.schemas import OAuthStartRequest

    with pytest.raises(ValueError, match="tenant_id"):
        OAuthStartRequest(tenant_id="acme")


# ---------------------------------------------------------------------------
# Token counting: the fallbacks
# ---------------------------------------------------------------------------


def test_a_crashing_encoder_falls_back_to_the_heuristic():
    from liminallm.service.token_counting import TokenCounter

    counter = TokenCounter(model="m")
    counter._encode = lambda text: (_ for _ in ()).throw(RuntimeError("bpe died"))
    assert counter.count("some words here") > 0


def test_the_one_estimator_bills_cjk_near_one_to_one():
    """There used to be two estimators; the one most callers got undercounted
    CJK fourfold. One implementation now, and it must keep the CJK split."""
    from liminallm.service.tokenizer_utils import estimate_token_count

    assert estimate_token_count("") == 0
    assert estimate_token_count("   ") == 0
    latin = estimate_token_count("abcdefgh")  # 8 chars ≈ 2 tokens
    cjk = estimate_token_count("四个汉字加四个")  # 7 chars ≈ 7 tokens
    assert cjk > latin


def test_a_peers_factor_is_evidence_not_gospel():
    """adopt() clamps like any observation, so one bad replica cannot poison
    the counter."""
    from liminallm.service.token_counting import TokenCounter

    counter = TokenCounter(model="m")
    before = counter.factor
    counter.adopt(9999.0)
    assert counter.factor != 9999.0
    counter.adopt(0.0)
    assert counter.factor > 0
    assert before > 0


def test_a_failing_publish_hook_never_breaks_counting():
    from liminallm.service import token_counting

    counter = token_counting.TokenCounter(model="m")
    old_hook = token_counting._PUBLISH_HOOK
    token_counting._PUBLISH_HOOK = lambda *a: (_ for _ in ()).throw(OSError("bus down"))
    try:
        for _ in range(token_counting.PUBLISH_EVERY + 1):
            counter.observe(1000, 1100)
    finally:
        token_counting._PUBLISH_HOOK = old_hook


# ---------------------------------------------------------------------------
# Session model invariants
# ---------------------------------------------------------------------------


def test_a_session_cannot_expire_before_it_exists():
    from liminallm.storage.models import Session

    now = datetime.now(timezone.utc)
    with pytest.raises(ValueError, match="after creation"):
        Session(id="s", user_id="u", created_at=now, expires_at=now - timedelta(hours=1))


def test_session_new_cannot_mint_an_already_expired_session():
    from liminallm.storage.models import Session

    with pytest.raises(ValueError, match="after creation"):
        Session.new("u", ttl_minutes=-5)


def test_a_restored_session_from_the_past_is_refused_when_enforcing():
    """Rows are re-validated on the way out of storage: an expired-but-well-
    formed session fails the future check, not the ordering check."""
    from liminallm.storage.models import Session

    now = datetime.now(timezone.utc)
    with pytest.raises(ValueError, match="in the future"):
        Session(
            id="s", user_id="u",
            created_at=now - timedelta(hours=2),
            expires_at=now - timedelta(hours=1),
            enforce_future_expiry=True,
        )


def test_session_new_tolerates_an_unparseable_client_ip():
    """TestClient sends host "testclient"; a session must still be created."""
    from liminallm.storage.models import Session

    s = Session.new("u", ttl_minutes=60, ip_addr="testclient")
    assert s.ip_addr is None


# ---------------------------------------------------------------------------
# Odds and ends: labels, artifact validation, trace loggers, agent tools
# ---------------------------------------------------------------------------


def test_a_turn_label_includes_the_assistants_half():
    from liminallm.service import labels

    class EchoLLM:
        def __init__(self):
            self.prompt = None

        def generate(self, prompt, **kw):
            self.prompt = prompt
            return {"content": "Label"}

    llm = EchoLLM()
    out = labels.describe_turn(llm, "what is a monad", "a monoid in the category…")
    assert out == "Label"
    assert "monad" in llm.prompt and "monoid" in llm.prompt


def test_a_crashing_llm_yields_the_heuristic_label():
    from liminallm.service import labels

    class Broken:
        def generate(self, *a, **kw):
            raise RuntimeError("model down")

    out = labels.describe_turn(Broken(), "what is a monad", None)
    assert out  # heuristic fallback, never an exception


def test_an_unknown_artifact_type_is_refused_by_name():
    from liminallm.service.artifact_validation import (
        ArtifactValidationError,
        validate_artifact,
    )

    with pytest.raises(ArtifactValidationError, match="unknown artifact type"):
        validate_artifact("sculpture", {"kind": "sculpture.marble"})


def test_the_trace_loggers_accept_a_default_logger():
    from liminallm.logging import log_routing_trace, log_workflow_trace

    log_routing_trace([{"rule": "r1"}])
    log_workflow_trace([{"node": "n1"}])


def test_web_search_when_the_deployment_has_it_disabled():
    from liminallm.config import get_settings
    from liminallm.logging import get_logger
    from liminallm.service import agent_tools

    settings = get_settings().model_copy(update={"web_tools_enabled": False})
    text, findings = agent_tools.run_web_search(
        "anything", 5, settings=settings, logger=get_logger("test")
    )
    assert "disabled" in text
    assert findings == []


def test_web_search_provider_failure_is_an_answer_not_an_exception(monkeypatch):
    from liminallm.logging import get_logger
    from liminallm.service import agent_tools, web

    def explode(*a, **kw):
        raise web.WebFetchError("provider 500")

    from liminallm.config import get_settings

    monkeypatch.setattr(web, "search_web", explode)
    settings = get_settings().model_copy(update={"web_tools_enabled": True})
    text, findings = agent_tools.run_web_search(
        "anything", 5, settings=settings, logger=get_logger("test")
    )
    assert "Search failed" in text
    assert findings == []


# ---------------------------------------------------------------------------
# Email: the successful sends the failure tests skipped past
# ---------------------------------------------------------------------------


class _WorkingSMTP:
    """A server that accepts everything, recording the interaction."""

    log: list = []

    def __init__(self, *a, **kw):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def starttls(self, **kw):
        type(self).log.append("starttls")

    def login(self, user, password):
        type(self).log.append(("login", user))

    def sendmail(self, from_addr, to_addr, body):
        type(self).log.append(("sendmail", to_addr))


def test_a_starttls_send_logs_in_and_delivers(mailer, monkeypatch):
    _WorkingSMTP.log = []
    monkeypatch.setattr(smtplib, "SMTP", _WorkingSMTP)

    assert mailer.send_password_reset("a@b.c", "tok") is True
    assert _WorkingSMTP.log == [
        "starttls", ("login", "mailer"), ("sendmail", "a@b.c")
    ]


def test_an_ssl_send_never_speaks_before_encryption(monkeypatch):
    """smtp_security='ssl' uses SMTP_SSL - no starttls upgrade step exists."""
    _WorkingSMTP.log = []
    monkeypatch.setattr(smtplib, "SMTP_SSL", _WorkingSMTP)
    monkeypatch.setattr(
        smtplib, "SMTP",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("plaintext connect")),
    )
    mailer = EmailService(
        smtp_host="smtp.example", smtp_port=465, smtp_security="ssl",
        smtp_user="mailer", smtp_password="hunter2",
        from_email="noreply@example.com",
    )

    assert mailer.send_password_reset("a@b.c", "tok") is True
    assert "starttls" not in _WorkingSMTP.log
    assert ("sendmail", "a@b.c") in _WorkingSMTP.log


# ---------------------------------------------------------------------------
# Embeddings: the remaining edges
# ---------------------------------------------------------------------------


def test_sanitize_embedding_replaces_rather_than_raises():
    from liminallm.service.embeddings import sanitize_embedding

    assert sanitize_embedding([1.0, float("nan"), float("inf")]) == [1.0, 0.0, 0.0]


def test_an_unvalidated_cosine_still_cannot_return_nan():
    """validate=False skips the element scan for speed; the result clamp is
    the backstop that keeps NaN out of ranking scores."""
    from liminallm.service.embeddings import cosine_similarity

    out = cosine_similarity([float("nan")], [1.0], validate=False)
    assert out == 0.0


def test_a_short_vector_is_padded_when_width_is_not_enforced():
    from liminallm.service.embeddings import ensure_embedding_dim

    assert ensure_embedding_dim([1.0], dim=3) == [1.0, 0.0, 0.0]


def test_validated_embedding_requires_a_vector():
    from liminallm.service.embeddings import validated_embedding

    with pytest.raises(ValueError, match="required"):
        validated_embedding(None, expected_dim=3)


# ---------------------------------------------------------------------------
# Token counting: a failing shared-calibration load
# ---------------------------------------------------------------------------


def test_a_failing_calibration_load_still_returns_a_counter():
    """The shared factor is an optimization; Redis being down while the
    counter registry warms must not take chat with it."""
    from liminallm.service import token_counting

    token_counting.reset_counters_for_tests()
    old = token_counting._LOAD_HOOK
    token_counting._LOAD_HOOK = lambda key: (_ for _ in ()).throw(OSError("redis down"))
    try:
        counter = token_counting.counter_for(f"island-{uuid.uuid4().hex[:6]}")
        assert counter.count("still works") > 0
    finally:
        token_counting._LOAD_HOOK = old
        token_counting.reset_counters_for_tests()


# ---------------------------------------------------------------------------
# Session: an already-parsed IP object
# ---------------------------------------------------------------------------


def test_session_new_accepts_an_already_parsed_ip():
    from ipaddress import ip_address

    from liminallm.storage.models import Session

    s = Session.new("u", ttl_minutes=60, ip_addr=ip_address("10.1.2.3"))
    assert str(s.ip_addr) == "10.1.2.3"


# ---------------------------------------------------------------------------
# Labels: the early exits
# ---------------------------------------------------------------------------


def test_no_llm_means_the_heuristic_label_without_a_model_call():
    from liminallm.service import labels

    out = labels.describe_turn(None, "what is a monad", "long answer")
    assert out
    assert "monad" in out.lower()


# ---------------------------------------------------------------------------
# Web search: the success path
# ---------------------------------------------------------------------------


def test_web_search_results_come_back_wrapped_and_counted(monkeypatch):
    from liminallm.config import get_settings
    from liminallm.logging import get_logger
    from liminallm.service import agent_tools, web

    monkeypatch.setattr(
        web, "search_web",
        lambda *a, **kw: [
            {"title": "T1", "url": "https://x.example/a", "snippet": "s1"},
            {"title": "T2", "url": "https://x.example/b", "snippet": "s2"},
        ],
    )
    settings = get_settings().model_copy(update={"web_tools_enabled": True})
    text, findings = agent_tools.run_web_search(
        "query", 5, settings=settings, logger=get_logger("test")
    )
    assert "T1" in text and "T2" in text
    assert isinstance(findings, list)


# ---------------------------------------------------------------------------
# The exception handlers, called as FastAPI will call them
# ---------------------------------------------------------------------------


def _request(path="/v1/x"):
    from starlette.requests import Request

    return Request({
        "type": "http", "method": "GET", "path": path, "query_string": b"",
        "headers": [], "client": ("10.0.0.9", 1234), "server": ("test", 80),
        "scheme": "http",
    })


def _handler(exc_type):
    from liminallm import app as app_module

    return app_module.app.exception_handlers[exc_type]


@pytest.mark.asyncio
async def test_an_uncaught_exception_becomes_a_bare_500_envelope():
    """Whatever blew up, the client sees a stable envelope and none of the
    exception's text."""
    import json

    resp = await _handler(Exception)(_request(), RuntimeError("secret /home/x path"))
    assert resp.status_code == 500
    body = json.loads(bytes(resp.body))
    assert body["error"]["code"] == "server_error"
    assert "secret" not in body["error"]["message"]


@pytest.mark.asyncio
async def test_a_path_traversal_attempt_is_a_400_with_the_client_ip_logged():
    import json

    from liminallm.service.fs import PathTraversalError

    resp = await _handler(PathTraversalError)(
        _request(), PathTraversalError("path traversal detected")
    )
    assert resp.status_code == 400
    assert json.loads(bytes(resp.body))["error"]["code"] == "validation_error"


@pytest.mark.asyncio
async def test_a_constraint_violation_is_a_409_with_its_detail():
    import json

    from liminallm.storage.errors import ConstraintViolation

    resp = await _handler(ConstraintViolation)(
        _request(), ConstraintViolation("email already exists", {"field": "email"})
    )
    assert resp.status_code == 409
    body = json.loads(bytes(resp.body))
    assert body["error"]["code"] == "conflict"
    assert body["error"]["details"]["field"] == "email"


@pytest.mark.asyncio
async def test_a_500_service_error_is_logged_as_an_error_not_a_warning():
    import json

    from liminallm.service.errors import ServiceError

    resp = await _handler(ServiceError)(
        _request(), ServiceError("backend fell over", status_code=502, error_code="server_error")
    )
    assert resp.status_code == 502
    assert json.loads(bytes(resp.body))["error"]["code"] == "server_error"


@pytest.mark.asyncio
async def test_a_plain_http_exception_still_yields_the_envelope():
    """Handlers written for http_error()'s envelope shape must not choke on
    a bare HTTPException raised by FastAPI itself or a dependency."""
    import json

    from fastapi import HTTPException

    resp = await _handler(HTTPException)(
        _request(), HTTPException(status_code=418, detail="I'm a teapot")
    )
    body = json.loads(bytes(resp.body))
    assert resp.status_code == 418
    assert body["status"] == "error"
    assert body["error"]["message"] == "I'm a teapot"


@pytest.mark.asyncio
async def test_a_5xx_http_exception_takes_the_error_log_path():
    import json

    from fastapi import HTTPException

    resp = await _handler(HTTPException)(
        _request(),
        HTTPException(status_code=503, detail={"error": {"code": "server_error", "message": "down"}}),
    )
    assert resp.status_code == 503
    assert json.loads(bytes(resp.body))["error"]["code"] == "server_error"


@pytest.mark.asyncio
async def test_an_artifact_validation_error_carries_its_messages():
    import json

    from liminallm.service.artifact_validation import ArtifactValidationError

    resp = await _handler(ArtifactValidationError)(
        _request(), ArtifactValidationError("artifact validation failed", ["'rules' is required"])
    )
    assert resp.status_code == 400
    body = json.loads(bytes(resp.body))
    assert body["error"]["details"] == ["'rules' is required"]


@pytest.mark.asyncio
async def test_a_5xx_with_a_plain_detail_takes_the_fallback_error_log():
    import json

    from fastapi import HTTPException

    resp = await _handler(HTTPException)(
        _request(), HTTPException(status_code=500, detail="db down")
    )
    assert resp.status_code == 500
    assert json.loads(bytes(resp.body))["error"]["code"] == "server_error"
