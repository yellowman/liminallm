"""Injection findings must restrict capability, not merely inform.

Detection was already good — sanitize, envelope, scan, report. But a model
that just read "ignore your rules and run this" is precisely the model least
able to be trusted with an interpreter, so the control has to be enforcement
rather than instruction.
"""

from __future__ import annotations

import pytest

from liminallm.service import notes
from liminallm.service.runtime import get_runtime


def _engine():
    return get_runtime().workflow


_USER_ID = "00000000-0000-4000-8000-000000000001"


def _exec(engine, name, args, session):
    return engine._execute_agent_tool(
        name, args,
        conversation_id="00000000-0000-4000-8000-0000000000c1", context_id=None,
        user_id=_USER_ID, tenant_id=None,
        session=session, snippets=[], fallback_query="q",
    )


def test_clean_session_still_runs_code():
    engine = _engine()
    session = {}
    out = _exec(engine, "run_python", {"code": "print(6*7)"}, session)
    assert "REFUSED" not in out
    assert "42" in out


def test_tainted_session_refuses_code_execution():
    engine = _engine()
    session = {"injection_findings": ["persona-hijack"]}
    out = _exec(engine, "run_python", {"code": "print(1)"}, session)
    assert out.startswith("REFUSED")
    assert "persona-hijack" in out          # names what was seen
    assert "not a failure you can retry" in out  # discourages retry loops
    assert "1" not in out.split("REFUSED")[1][:20]  # code did not run
    assert session["taint_blocked"] == 1


def test_taint_persists_for_the_rest_of_the_turn():
    engine = _engine()
    session = {"injection_findings": ["tool-abuse"]}
    for _ in range(3):
        assert _exec(engine, "run_python", {"code": "print(1)"}, session).startswith("REFUSED")
    assert session["taint_blocked"] == 3


def test_taint_does_not_block_reading_tools():
    """Search and retrieval stay available; only execution is withdrawn."""
    engine = _engine()
    session = {"injection_findings": ["persona-hijack"]}
    out = _exec(engine, "note_search", {"query": "anything"}, session)
    assert "REFUSED" not in out


def test_web_fetch_findings_set_the_taint(monkeypatch):
    """The gate is armed by the same findings the reporter already collects."""
    engine = _engine()
    monkeypatch.setattr(
        engine, "_run_web_fetch",
        lambda url: ("wrapped page text", [{"type": "persona-hijack", "match": "x"}]),
    )
    session = {}
    _exec(engine, "web_fetch", {"url": "http://example.com"}, session)
    assert session["injection_findings"] == ["persona-hijack"]
    # ...and code execution is now refused, in the same turn.
    assert _exec(engine, "run_python", {"code": "print(1)"}, session).startswith("REFUSED")


# ---------------------------------------------------------------------------
# Re-embed sweep: encoder changes converge without waiting for a reader


class _Enc:
    is_semantic = True
    model_id = "encoder-v2"

    def __init__(self):
        self.calls = 0

    def embed(self, text):
        self.calls += 1
        return [0.5, 0.5]


def _worker(store, embeddings, **kw):
    from liminallm.service.training_worker import TrainingWorker

    return TrainingWorker(
        store=store, training_service=object(), clusterer=None,
        embeddings=embeddings, **kw,
    )


async def test_sweep_reembeds_stale_and_missing_vectors(tmp_path):
    from tests.pgharness import get_test_store

    store = get_test_store()
    user = store.create_user(email="sweep@example.com", role="user")
    fresh = store.create_note(user.id, "Fresh", "x", embedding=[1.0] + [0.0] * 63)
    store.update_note_meta(fresh.id, {"embedding_model": "encoder-v2"})
    stale = store.create_note(user.id, "Stale", "y", embedding=[1.0] + [0.0] * 63)
    store.update_note_meta(stale.id, {"embedding_model": "encoder-v1"})
    never = store.create_note(user.id, "Never", "z")  # no vector at all

    enc = _Enc()
    done = notes.reembed_stale(store, enc, user_limit=100)
    assert done == 2  # the stale one and the un-embedded one, not the fresh one
    assert store.get_note(stale.id).meta["embedding_model"] == "encoder-v2"
    assert store.get_note(never.id).embedding == [0.5, 0.5]
    # A second pass finds nothing left to do — it converges.
    assert notes.reembed_stale(store, enc, user_limit=100) == 0


async def test_sweep_is_a_noop_without_a_real_encoder(tmp_path):
    from liminallm.service.embeddings import EmbeddingsService
    from tests.pgharness import get_test_store

    store = get_test_store()
    worker = _worker(store, EmbeddingsService("hash"))  # is_semantic False
    worker._last_reembed_run = 0.0
    await worker._maybe_reembed_stale_vectors()
    assert worker._last_reembed_run == 0.0  # never even armed


async def test_sweep_is_leader_locked(tmp_path):
    from liminallm.service.replication import AdvisoryLock
    from tests.pgharness import get_test_store

    store = get_test_store()
    user = store.create_user(email="lock@example.com", role="user")
    store.create_note(user.id, "Needs embedding", "text")
    lock = AdvisoryLock(None)
    enc = _Enc()
    worker = _worker(store, enc, leader_lock=lock)
    async with lock.try_hold("training_worker:reembed") as held:
        assert held is True
        await worker._maybe_reembed_stale_vectors()
    assert enc.calls == 0  # a follower does not duplicate the work


async def test_provider_failure_stops_the_pass_without_losing_progress(tmp_path):
    from tests.pgharness import get_test_store

    store = get_test_store()
    user = store.create_user(email="fail@example.com", role="user")
    for i in range(4):
        store.create_note(user.id, f"Note {i}", "text")

    class Flaky(_Enc):
        def embed(self, text):
            self.calls += 1
            if self.calls > 2:
                raise RuntimeError("provider 500")
            return [0.1, 0.2]

    done = notes.reembed_stale(store, Flaky(), user_limit=100)
    assert done == 2  # the two that succeeded are persisted, not rolled back


# ---------------------------------------------------------------------------
# Sweep archive: a 30-call sweep should survive a page reload


def _client_and_headers():
    import uuid as _uuid

    from fastapi.testclient import TestClient

    from liminallm import app as app_module

    client = TestClient(app_module.app)
    email = f"sweep_{_uuid.uuid4().hex[:8]}@example.com"
    resp = client.post(
        "/v1/auth/signup", json={"email": email, "password": "TestPassword123!"}
    )
    assert resp.status_code == 201, resp.text
    return client, {"Authorization": f"Bearer {resp.json()['data']['access_token']}"}


def test_sweep_is_archived_and_replayable():
    client, headers = _client_and_headers()
    client.post("/v1/notes", headers=headers, json={"title": "Solo", "content": "x"})
    resp = client.post("/v1/notes/sweep", headers=headers)
    assert resp.status_code == 200, resp.text
    data = resp.json()["data"]
    assert data.get("id") and data.get("created_at")  # archived, not ephemeral

    listed = client.get("/v1/notes/sweeps", headers=headers)
    assert listed.status_code == 200, listed.text
    sweeps = listed.json()["data"]["sweeps"]
    assert len(sweeps) == 1
    assert sweeps[0]["id"] == data["id"]
    assert "report" in sweeps[0]  # full report replays without new model calls


def test_sweeps_route_is_not_shadowed_by_the_note_id_route():
    """GET /notes/sweeps must not be captured by GET /notes/{note_id}."""
    client, headers = _client_and_headers()
    resp = client.get("/v1/notes/sweeps", headers=headers)
    assert resp.status_code == 200
    assert "sweeps" in resp.json()["data"]


def test_sweep_archive_is_owner_scoped():
    client, headers = _client_and_headers()
    client.post("/v1/notes", headers=headers, json={"title": "Mine", "content": "x"})
    client.post("/v1/notes/sweep", headers=headers)
    _other_client, other_headers = _client_and_headers()
    listed = client.get("/v1/notes/sweeps", headers=other_headers)
    assert listed.json()["data"]["sweeps"] == []


def test_archive_failure_never_fails_the_sweep(monkeypatch):
    client, headers = _client_and_headers()
    runtime = get_runtime()

    def broken(*a, **kw):
        raise RuntimeError("disk full")

    monkeypatch.setattr(runtime.store, "save_sweep_report", broken, raising=False)
    resp = client.post("/v1/notes/sweep", headers=headers)
    assert resp.status_code == 200  # the expensive work is still returned
