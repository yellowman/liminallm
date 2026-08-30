"""Two things that must not leak: log output, and other people's contexts.

Both were uncovered. `sanitize_*` decides what an error message and a response
body look like once secrets are stripped, and `_allowed_context_ids` is the
SPEC §12.2 isolation check standing between a requested context id and someone
else's documents.
"""

from __future__ import annotations

import uuid

import pytest

from liminallm.logging import (
    sanitize_error_message,
    sanitize_response_data,
)
from liminallm.service.runtime import get_runtime
from liminallm.storage.models import KnowledgeChunk

# ---------------------------------------------------------------------------
# Error messages reaching a client
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "leak",
    [
        'password="hunter2"',
        "api_key=sk-live-abcdef123456",
        "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.abc.def",
        "postgresql://user:secret@db.internal:5432/liminallm",
        "SELECT * FROM app_user WHERE email = 'alice@example.com'",
        "/home/liminal/users/u-123/files/private.pdf",
        'File "/opt/app/liminallm/service/auth.py", line 412, in login',
    ],
    ids=lambda s: s.split()[0][:22],
)
def test_a_secret_does_not_survive_into_an_error_message(leak):
    """Whatever else it does, the original text must not come back out."""
    out = sanitize_error_message(f"boom: {leak}")
    assert leak not in out


def test_an_error_message_is_bounded():
    """A stack trace pasted into a response body is both a leak and a payload."""
    assert len(sanitize_error_message("x" * 5000)) <= 500


@pytest.mark.parametrize("empty", ["", None, 123, [], {}])
def test_a_non_string_error_becomes_a_safe_default(empty):
    assert sanitize_error_message(empty) == "An error occurred"


def test_an_ordinary_message_survives():
    """Over-redaction makes the field useless, which is its own failure."""
    assert sanitize_error_message("conversation not found") == "conversation not found"


# ---------------------------------------------------------------------------
# Response bodies reaching a log
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key",
    ["password", "secret", "token", "api_key", "apiKey", "API-KEY",
     "authorization", "credentials", "private_key", "access_key",
     "smtp_password", "refresh_token", "csrf_token"],
)
def test_a_sensitive_key_is_redacted_however_it_is_spelled(key):
    """Keys arrive from provider payloads and client bodies, so the match is
    on a normalized name rather than an exact one."""
    out = sanitize_response_data({key: "the-actual-value"})
    assert out[key] == "[REDACTED]"


def test_redaction_reaches_into_nesting():
    out = sanitize_response_data(
        {"user": {"name": "alice", "credentials": {"password": "hunter2"}}}
    )
    assert out["user"]["name"] == "alice"
    assert out["user"]["credentials"] == "[REDACTED]"


def test_redaction_reaches_into_lists():
    out = sanitize_response_data([{"token": "abc"}, {"safe": "kept"}])
    assert out[0]["token"] == "[REDACTED]"
    assert out[1]["safe"] == "kept"


def test_a_numeric_secret_keeps_its_type():
    """The shape of the response should survive redaction, so a consumer
    reading it does not trip over a string where a number was."""
    out = sanitize_response_data({"cvv": 123, "auth": True})
    assert out["cvv"] == 0
    assert out["auth"] is False


def test_ordinary_data_is_untouched():
    payload = {"conversation_id": "c1", "messages": [{"role": "user", "content": "hi"}]}
    assert sanitize_response_data(payload) == payload


def test_deep_nesting_stops_rather_than_recursing_forever():
    """A cyclic-looking payload from a provider must not blow the stack."""
    deep = current = {}
    for _ in range(60):
        current["next"] = {}
        current = current["next"]
    out = sanitize_response_data(deep, max_depth=10)
    text = repr(out)
    assert "max depth exceeded" in text


# ---------------------------------------------------------------------------
# RAG isolation: SPEC §12.2
# ---------------------------------------------------------------------------


@pytest.fixture
def rag():
    return get_runtime().rag


@pytest.fixture
def store():
    return get_runtime().store


def _user(store, tenant_id="public"):
    return store.create_user(
        email=f"rag_{uuid.uuid4().hex[:8]}@example.com", tenant_id=tenant_id
    )


def _context(store, owner, *, visibility=None):
    meta = {"visibility": visibility} if visibility else None
    return store.upsert_context(
        owner.id, f"ctx-{uuid.uuid4().hex[:6]}", "fixture", meta=meta
    )


def test_a_request_without_a_user_gets_nothing(rag, store):
    """No user means no way to check ownership, so the answer is no contexts -
    not "all of them"."""
    owner = _user(store)
    ctx = _context(store, owner)
    assert rag._allowed_context_ids([ctx.id], user_id=None, tenant_id=None) == []


def test_an_owner_reaches_their_own_context(rag, store):
    owner = _user(store)
    ctx = _context(store, owner)
    assert rag._allowed_context_ids([ctx.id], user_id=owner.id, tenant_id=None) == [ctx.id]


def test_a_stranger_is_kept_out_of_a_private_context(rag, store):
    owner, stranger = _user(store), _user(store)
    ctx = _context(store, owner)
    assert rag._allowed_context_ids([ctx.id], user_id=stranger.id, tenant_id=None) == []


def test_a_global_context_is_reachable_by_anyone(rag, store):
    owner, stranger = _user(store), _user(store)
    ctx = _context(store, owner, visibility="global")
    assert rag._allowed_context_ids([ctx.id], user_id=stranger.id, tenant_id=None) == [ctx.id]


def test_a_shared_context_stays_inside_its_tenant(rag, store):
    """"global" is cross-tenant by design; "shared" is not."""
    owner = _user(store, tenant_id="acme")
    insider = _user(store, tenant_id="acme")
    outsider = _user(store, tenant_id="globex")
    ctx = _context(store, owner, visibility="shared")

    assert rag._allowed_context_ids([ctx.id], user_id=insider.id, tenant_id="acme") == [ctx.id]
    assert rag._allowed_context_ids([ctx.id], user_id=outsider.id, tenant_id="globex") == []


def test_a_global_context_crosses_tenants(rag, store):
    owner = _user(store, tenant_id="acme")
    outsider = _user(store, tenant_id="globex")
    ctx = _context(store, owner, visibility="global")
    assert rag._allowed_context_ids([ctx.id], user_id=outsider.id, tenant_id="globex") == [ctx.id]


def test_an_unknown_context_is_dropped_not_fatal(rag, store):
    owner = _user(store)
    ctx = _context(store, owner)
    allowed = rag._allowed_context_ids(
        [ctx.id, str(uuid.uuid4())], user_id=owner.id, tenant_id=None
    )
    assert allowed == [ctx.id]


def test_one_forbidden_id_does_not_forfeit_the_rest(rag, store):
    """A request naming several contexts should return the ones it may have."""
    owner, stranger = _user(store), _user(store)
    mine = _context(store, stranger)
    theirs = _context(store, owner)
    allowed = rag._allowed_context_ids(
        [theirs.id, mine.id], user_id=stranger.id, tenant_id=None
    )
    assert allowed == [mine.id]


# ---------------------------------------------------------------------------
# The retrieval token budget
# ---------------------------------------------------------------------------


def _chunk(content, tokens=None):
    return KnowledgeChunk(
        context_id="c", fs_path="/f", content=content, embedding=[], chunk_index=0,
        meta={"token_count": tokens} if tokens is not None else None,
    )


def test_the_budget_stops_before_it_is_exceeded(rag):
    chunks = [_chunk("a", tokens=40), _chunk("b", tokens=40), _chunk("c", tokens=40)]
    selected = rag._apply_token_budget(chunks, max_tokens=100, limit=10)
    assert len(selected) == 2


def test_the_budget_keeps_the_most_relevant_first(rag):
    """Chunks arrive ranked, so truncation must take from the tail."""
    chunks = [_chunk("best", tokens=60), _chunk("worse", tokens=60)]
    assert rag._apply_token_budget(chunks, max_tokens=100, limit=10)[0].content == "best"


def test_the_limit_still_applies_under_a_generous_budget(rag):
    chunks = [_chunk(str(i), tokens=1) for i in range(20)]
    assert len(rag._apply_token_budget(chunks, max_tokens=10_000, limit=3)) <= 3


def test_a_chunk_without_a_token_count_is_estimated(rag):
    """Chunks predating the token_count metadata must still be budgeted, not
    treated as free."""
    chunks = [_chunk("x" * 4000), _chunk("y" * 4000)]
    assert len(rag._apply_token_budget(chunks, max_tokens=500, limit=10)) < 2


def test_an_empty_result_set_budgets_to_nothing(rag):
    assert rag._apply_token_budget([], max_tokens=100, limit=5) == []
