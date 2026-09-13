"""Toggling a retrieval flag changes the RAG service that serves the next turn.

`RAGService` captures `late_interaction` and `late_segments` at construction
and never re-reads them, which is the shape that left MFA challenges issuing
after an admin switched `enable_mfa` off. Here it is not a defect, because
both settings are in `MODEL_AFFECTING_SETTINGS`: the admin write rebuilds the
model stack, and the replacement is built from the stored value. That is a
claim about two mechanisms agreeing, so it is measured through the real
endpoint rather than read off the construction site.
"""

from __future__ import annotations

import uuid

import pytest

from liminallm.config import MODEL_AFFECTING_SETTINGS
from liminallm.service import runtime as runtime_module
from liminallm.service.embeddings import EmbeddingsService
from liminallm.service.runtime import get_runtime


@pytest.fixture
def admin_headers():
    runtime = get_runtime()
    user = runtime.store.create_user(
        email=f"r_{uuid.uuid4().hex[:8]}@t.local", role="admin"
    )
    session = runtime.store.create_session(user.id, tenant_id=user.tenant_id)
    _u, _s, tokens = runtime.auth.issue_tokens_for_session(session.id)
    return {"Authorization": f"Bearer {tokens['access_token']}"}


@pytest.fixture
def semantic_runtime(monkeypatch):
    """A runtime whose encoder claims to be semantic, then put back.

    `RAGService` stores `late_interaction and semantic`, because MaxSim over
    hash vectors is noise - and the test kernel's encoder is the hash. Without
    this the flag is pinned to False whatever an admin sets, and a test
    asserting False would pass against a rebuild that never happened. The
    double is the real `EmbeddingsService` with its honesty flag set, not a
    stand-in: what is under test is whether the setting reaches the rebuilt
    object, not what MaxSim then does with it.

    Both retrieval settings are captured and restored, because this rebuilds
    the process-wide model stack.
    """
    runtime = get_runtime()
    before = {
        "rag_late_interaction": runtime.settings.rag_late_interaction,
        "rag_late_segments": runtime.settings.rag_late_segments,
    }

    def semantic_embeddings(model_id, **kwargs):
        kwargs["semantic"] = True
        return EmbeddingsService(model_id, **kwargs)

    monkeypatch.setattr(runtime_module, "EmbeddingsService", semantic_embeddings)
    runtime.reload_model_services()
    assert runtime.rag.semantic, "the fixture did not reach the rebuilt stack"
    yield runtime
    monkeypatch.undo()
    runtime.store.set_system_settings(before)
    runtime.reload_model_services()


def _put(client, headers, body):
    resp = client.put("/v1/admin/settings", headers=headers, json=body)
    assert resp.status_code == 200, resp.text
    return resp


def test_both_flags_reach_the_active_rag_service(
    client, admin_headers, semantic_runtime
):
    """One write, both values, asserted on the object that serves retrieval."""
    before = semantic_runtime.rag
    target_segments = 2 if before.late_segments != 2 else 5

    _put(client, admin_headers, {
        "rag_late_interaction": True,
        "rag_late_segments": target_segments,
    })

    live = get_runtime().rag
    assert live is not before, "the model stack was not rebuilt"
    assert live.late_interaction is True
    assert live.late_segments == target_segments


def test_turning_late_interaction_off_reaches_it_too(
    client, admin_headers, semantic_runtime
):
    """The off direction is the one that matters for a withdrawal.

    A flag that only ever switches on leaves the feature running after an
    operator turns it off, which is how the MFA defect read.
    """
    _put(client, admin_headers, {"rag_late_interaction": True})
    assert get_runtime().rag.late_interaction is True

    _put(client, admin_headers, {"rag_late_interaction": False})

    assert get_runtime().rag.late_interaction is False


def test_the_two_flags_are_declared_model_affecting():
    """The rebuild is the only thing that delivers them, so it must be asked
    for. Dropping either name from the list would leave the captured value in
    place, with nothing failing until an operator noticed the feature had not
    changed - and the console would stop warning about the interruption."""
    assert "rag_late_interaction" in MODEL_AFFECTING_SETTINGS
    assert "rag_late_segments" in MODEL_AFFECTING_SETTINGS
