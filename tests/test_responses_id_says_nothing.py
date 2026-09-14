"""A `previous_response_id` that is not yours reads exactly like one that
does not exist.

SPEC §13.1: "a foreign or unknown id is 404 either way, so existence is not
confirmed across users." The status was 404 either way; the message was not.
A foreign id reached the owned-conversation check and answered `conversation
not found`, while an unknown one answered `No response found with id ...` -
so a caller holding a response id could learn whether it was somebody's.
Measured by the release-qualification sweep, on both a same-tenant stranger
and a cross-tenant outsider.
"""

from __future__ import annotations

import uuid

from liminallm.service.runtime import get_runtime

ACME, GLOBEX = "acme.test", "globex.test"


def _account(tenant_id, host):
    runtime = get_runtime()
    user = runtime.store.create_user(
        email=f"r_{uuid.uuid4().hex[:8]}@t.local", tenant_id=tenant_id
    )
    session = runtime.store.create_session(user.id, tenant_id=tenant_id)
    _u, _s, tokens = runtime.auth.issue_tokens_for_session(session.id)
    return {"Authorization": f"Bearer {tokens['access_token']}", "host": host}


def _turn(client, headers, **body):
    return client.post(
        "/v1/responses", headers=headers,
        json={"model": "gpt-4o-mini", "input": "hello", **body},
    )


def _error(resp) -> dict:
    assert resp.status_code == 404, resp.text
    return resp.json()["error"]


class TestAResponseIdConfirmsNothing:
    def test_foreign_and_unknown_answer_byte_for_byte_alike(self, client, monkeypatch):
        runtime = get_runtime()
        monkeypatch.setattr(
            runtime.settings, "tenant_domains", {ACME: "acme", GLOBEX: "globex"}
        )
        owner = _account("acme", ACME)
        stranger = _account("acme", ACME)
        outsider = _account("globex", GLOBEX)
        first = _turn(client, owner)
        assert first.status_code == 200, first.text
        theirs = first.json()["id"]
        assert theirs.startswith("resp_")
        ghost = f"resp_{uuid.uuid4()}"

        foreign = _error(_turn(client, stranger, previous_response_id=theirs))
        cross = _error(_turn(client, outsider, previous_response_id=theirs))
        unknown = _error(_turn(client, stranger, previous_response_id=ghost))

        # The id the caller sent is the only thing allowed to differ.
        def scrub(error, sent):
            return {**error, "message": error["message"].replace(sent, "<id>")}

        assert scrub(foreign, theirs) == scrub(cross, theirs) == scrub(unknown, ghost)

    def test_the_owner_still_continues(self, client):
        runtime = get_runtime()
        owner = _account(runtime.settings.default_tenant_id, "testserver")
        first = _turn(client, owner)
        assert first.status_code == 200, first.text

        again = _turn(client, owner, previous_response_id=first.json()["id"])

        assert again.status_code == 200, again.text
        assert again.json()["id"] != first.json()["id"]
