"""The two chat transports do the same turn.

They had drifted three ways, all silently — no inference slot, no cache
warming, no post-turn work on the non-streaming branch — because there were no
WebSocket tests at all. The structural checks catch a new divergence where it
is written; the end-to-end ones prove the turn completes on the socket.
"""

from __future__ import annotations

import inspect
import uuid

import pytest
from fastapi.testclient import TestClient

from liminallm import app as app_module
from liminallm.api import chat_turn, routes
from liminallm.service import admission


@pytest.fixture
def client():
    return TestClient(app_module.app)


@pytest.fixture
def auth(client):
    email = f"wschat_{uuid.uuid4().hex[:8]}@example.com"
    resp = client.post(
        "/v1/auth/signup", json={"email": email, "password": "TestPassword123!"}
    )
    assert resp.status_code == 201, resp.text
    data = resp.json()["data"]
    return {
        "headers": {"Authorization": f"Bearer {data['access_token']}"},
        "access_token": data["access_token"],
    }


# ---------------------------------------------------------------------------
# Structure: neither transport may reimplement the turn
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["chat", "websocket_chat"])
def test_transport_delegates_the_turn(name):
    source = inspect.getsource(getattr(routes, name))
    for step in ("chat_turn.begin(", "chat_turn.finish(", "chat_turn.response("):
        assert step in source, f"{name} does not call {step}"


@pytest.mark.parametrize("name", ["chat", "websocket_chat"])
def test_transport_persists_no_messages_of_its_own(name):
    source = inspect.getsource(getattr(routes, name))
    assert "append_message(" not in source, (
        f"{name} persists a message directly. Both transports must go through "
        f"chat_turn so the post-turn work cannot be forgotten on one of them."
    )


def test_a_chat_turn_costs_the_same_on_both_transports():
    assert set(admission.CHAT_SLOTS) == {"workflow", "inference"}
    for name in ("chat", "websocket_chat"):
        source = inspect.getsource(getattr(routes, name))
        assert "admission.CHAT_SLOTS" in source, f"{name} names its own slots"


def test_no_route_reaches_past_admission_to_the_cache():
    """Raw acquire/release is how the two paths ended up taking different slots."""
    source = inspect.getsource(routes)
    for call in ("acquire_concurrency_slot", "release_concurrency_slot"):
        assert call not in source, (
            f"routes.py calls cache.{call} directly; go through service.admission"
        )


def test_finish_schedules_every_post_turn_effect():
    source = inspect.getsource(chat_turn.finish)
    assert "turn_effects.schedule_all(" in source
    assert "cache_conversation_state(" in source


# ---------------------------------------------------------------------------
# Behaviour: the turn actually completes over the WebSocket
# ---------------------------------------------------------------------------


def _open(client, auth):
    return client.websocket_connect("/v1/chat/stream")


def test_non_streaming_websocket_turn_completes(client, auth):
    """`stream: false` is the branch that used to schedule nothing."""
    with _open(client, auth) as ws:
        ws.send_json(
            {
                "access_token": auth["access_token"],
                "message": "hello over the socket",
                "stream": False,
            }
        )
        envelope = ws.receive_json()

    assert envelope["status"] == "ok", envelope
    data = envelope["data"]
    assert data["message_id"]
    assert data["conversation_id"]

    # The turn is durable: both messages landed on the conversation the socket
    # created, which is what proves finish() ran rather than the socket simply
    # answering and forgetting.
    listing = client.get(
        f"/v1/conversations/{data['conversation_id']}/messages", headers=auth["headers"]
    )
    assert listing.status_code == 200, listing.text
    roles = [m["role"] for m in listing.json()["data"]["messages"]]
    assert "user" in roles and "assistant" in roles


def test_streaming_websocket_turn_completes(client, auth):
    with _open(client, auth) as ws:
        ws.send_json(
            {
                "access_token": auth["access_token"],
                "message": "hello streaming",
                "stream": True,
            }
        )
        done = None
        for _ in range(200):
            event = ws.receive_json()
            if event.get("event") == "message_done":
                done = event
                break
            if event.get("event") == "error":
                pytest.fail(f"stream errored: {event}")

    assert done is not None, "stream never sent message_done"
    assert done["data"]["message_id"]
    assert done["data"]["conversation_id"]


def test_websocket_conversation_records_its_context(client, auth):
    """A socket-created conversation keeps its context for the next turn.

    It used to be created without active_context_id, so the context applied to
    the first turn and silently vanished afterwards.
    """
    created = client.post(
        "/v1/contexts",
        headers=auth["headers"],
        json={
            "name": f"ctx-{uuid.uuid4().hex[:6]}",
            "description": "context for the websocket turn",
            "scope": "user",
        },
    )
    assert created.status_code in (200, 201), created.text
    context_id = created.json()["data"]["id"]

    with _open(client, auth) as ws:
        ws.send_json(
            {
                "access_token": auth["access_token"],
                "message": "with a context",
                "context_id": context_id,
                "stream": False,
            }
        )
        envelope = ws.receive_json()

    assert envelope["status"] == "ok", envelope
    conversation_id = envelope["data"]["conversation_id"]
    detail = client.get(
        f"/v1/conversations/{conversation_id}", headers=auth["headers"]
    )
    assert detail.status_code == 200, detail.text
    assert detail.json()["data"]["active_context_id"] == context_id
