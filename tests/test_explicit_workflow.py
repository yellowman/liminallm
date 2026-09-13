"""An explicit `workflow_id` either runs that workflow or fails.

`workflow_id` on a chat turn is an optional override, which gives it two
meanings that must not collapse into one: omitted means "choose the default,
as always"; given means "this one, or nothing". They had collapsed. A name the
caller could not reach - absent, another user's private, another tenant's -
was quietly replaced by the default workflow, the turn ran, and the reply
reported the requested id as though it had run. Measured on both transports
before the fix.

Two defenses, each witnessed on its own: the transports resolve the override
before anything durable exists, and the engine refuses rather than
substitutes for a workflow that disappears between that check and execution.
"""

from __future__ import annotations

import uuid

import pytest

from liminallm.service.errors import NotFoundError
from liminallm.service.runtime import get_runtime

ACME, GLOBEX = "acme.test", "globex.test"


def _http_account(client, prefix="u"):
    resp = client.post(
        "/v1/auth/signup",
        json={"email": f"{prefix}_{uuid.uuid4().hex[:8]}@example.com",
              "password": "TestPassword123!"},
    )
    assert resp.status_code == 201, resp.text
    data = resp.json()["data"]
    return data["user_id"], data["access_token"], {
        "Authorization": f"Bearer {data['access_token']}"
    }


def _tenant_account(tenant_id, host):
    """A user in a named tenant, with a token minted the production way."""
    runtime = get_runtime()
    user = runtime.store.create_user(
        email=f"t_{uuid.uuid4().hex[:8]}@t.local", tenant_id=tenant_id
    )
    session = runtime.store.create_session(user.id, tenant_id=tenant_id)
    _user, _sess, tokens = runtime.auth.issue_tokens_for_session(session.id)
    return user.id, {"Authorization": f"Bearer {tokens['access_token']}",
                     "host": host}


#: The engine's own default, restated: one generic model call and an end. The
#: smallest workflow that actually executes, so a witness that expects the
#: named workflow to *run* can tell running from being quietly replaced.
RUNNABLE = {
    "kind": "workflow.chat",
    "entrypoint": "plain_chat",
    "nodes": [
        {"id": "plain_chat", "type": "tool_call", "tool": "llm.generic",
         "inputs": {"message": "${input.message}"}, "next": "end"},
        {"id": "end", "type": "end"},
    ],
}


def _workflow(client, headers) -> str:
    resp = client.post(
        "/v1/artifacts",
        headers={**headers, "Idempotency-Key": uuid.uuid4().hex},
        json={"type": "workflow", "name": f"wf-{uuid.uuid4().hex[:6]}",
              "description": "a private workflow", "schema": RUNNABLE},
    )
    assert resp.status_code in (200, 201), resp.text
    return resp.json()["data"]["id"]


def _chat(client, headers, **body):
    return client.post(
        "/v1/chat", headers=headers,
        json={"message": {"content": "hi", "mode": "text"}, "stream": False,
              **body},
    )


def _refused(resp) -> dict:
    assert resp.status_code == 404, resp.text
    error = resp.json()["error"]
    assert error["code"] == "not_found"
    assert error["message"] == "workflow not found"
    return error


class TestAnExplicitWorkflowRunsOrFails:
    def test_the_owners_workflow_runs_and_is_the_one_reported(self, client):
        _uid, _tok, owner = _http_account(client)
        wf_id = _workflow(client, owner)

        resp = _chat(client, owner, workflow_id=wf_id)

        assert resp.status_code == 200, resp.text
        assert resp.json()["data"]["workflow_id"] == wf_id

    def test_omitting_it_still_means_the_default(self, client):
        _uid, _tok, owner = _http_account(client)

        resp = _chat(client, owner)

        assert resp.status_code == 200, resp.text
        assert resp.json()["data"]["workflow_id"] is None

    def test_absent_foreign_and_cross_tenant_all_read_the_same(
        self, client, monkeypatch
    ):
        """One answer for every way of naming what you cannot run, so the id
        confirms nothing about what exists or who owns it."""
        runtime = get_runtime()
        monkeypatch.setattr(
            runtime.settings, "tenant_domains", {ACME: "acme", GLOBEX: "globex"}
        )
        _oid, owner = _tenant_account("acme", ACME)
        _sid, stranger = _tenant_account("acme", ACME)
        _xid, outsider = _tenant_account("globex", GLOBEX)
        wf_id = _workflow(client, owner)

        absent = _refused(_chat(client, owner, workflow_id=str(uuid.uuid4())))
        foreign = _refused(_chat(client, stranger, workflow_id=wf_id))
        cross = _refused(_chat(client, outsider, workflow_id=wf_id))

        assert absent == foreign == cross

    def test_a_refusal_leaves_no_message_and_no_conversation_behind(
        self, client
    ):
        """The check runs before anything durable is written.

        The engine refuses too, but it runs after `begin()` has appended the
        user's message; on its own it would leave a message that was never
        answered on a turn that never happened.
        """
        _uid, _tok, owner = _http_account(client)
        opened = _chat(client, owner)
        assert opened.status_code == 200, opened.text
        conversation_id = opened.json()["data"]["conversation_id"]

        def messages():
            listing = client.get(
                f"/v1/conversations/{conversation_id}/messages", headers=owner
            )
            return len(listing.json()["data"]["messages"])

        def conversations():
            return len(client.get("/v1/conversations", headers=owner)
                       .json()["data"]["items"])

        before_messages, before_conversations = messages(), conversations()

        _refused(_chat(client, owner, conversation_id=conversation_id,
                       workflow_id=str(uuid.uuid4())))
        _refused(_chat(client, owner, workflow_id=str(uuid.uuid4())))

        assert messages() == before_messages, "a refused turn persisted its message"
        assert conversations() == before_conversations, (
            "a refused turn created a conversation"
        )

    @pytest.mark.asyncio
    async def test_the_engine_refuses_rather_than_substitutes(self, client):
        """The backstop, asked directly: with the transport check bypassed, an
        unreachable explicit workflow is still an error and not the default."""
        runtime = get_runtime()
        user_id, _tok, _owner = _http_account(client)
        conversation = runtime.store.create_conversation(user_id=user_id)

        with pytest.raises(NotFoundError):
            await runtime.workflow.run(
                str(uuid.uuid4()), conversation.id, "hi", None, user_id,
                tenant_id="public",
            )


class TestTheSocketKeepsTheSameContract:
    def _turn(self, client, token, **init):
        with client.websocket_connect("/v1/chat/stream") as ws:
            ws.send_json({"access_token": token, "message": "hi",
                          "stream": False, **init})
            return ws.receive_json()

    def test_a_foreign_workflow_is_refused_over_the_socket(self, client):
        _oid, _otok, owner = _http_account(client, "owner")
        _sid, stranger_token, _stranger = _http_account(client, "stranger")
        wf_id = _workflow(client, owner)

        envelope = self._turn(client, stranger_token, workflow_id=wf_id)

        assert envelope["status"] == "error", envelope
        assert envelope["error"]["code"] == "not_found"
        assert envelope["error"]["message"] == "workflow not found"

    def test_the_owners_workflow_runs_over_the_socket(self, client):
        _oid, owner_token, owner = _http_account(client, "owner")
        wf_id = _workflow(client, owner)

        envelope = self._turn(client, owner_token, workflow_id=wf_id)

        assert envelope["status"] == "ok", envelope
        assert envelope["data"]["workflow_id"] == wf_id
