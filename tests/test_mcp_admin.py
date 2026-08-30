"""Configuring a remote MCP server through the API an operator actually has.

`servers_for_turn` requires a globally visible, admin-owned artifact. Nothing
persisted through `POST /v1/artifacts` was ever global - the route did not
accept a visibility and the store defaults to private - so the whole feature
was reachable only by writing the row by hand. These are the reds for the path
that closes it, and for the privilege boundary that path opens.
"""

from __future__ import annotations

import uuid

from liminallm.service import mcp_client
from liminallm.service.runtime import get_runtime
from tests.mcpfixture import MCPFixture


def _server_body(fixture, **over):
    body = {
        "type": "mcp",
        "name": fixture.name,
        "description": "a remote tool server",
        "schema": fixture.as_artifact_schema(),
    }
    body.update(over)
    return body


class TestAnAdminCanMakeAServerACapability:
    def test_a_created_server_is_visible_to_a_turn(self, client, admin_headers):
        """End to end, through the route: create, then ask the turn.

        Asserted on `servers_for_turn` rather than on the response body,
        because the response saying `global` and the resolver disagreeing is
        exactly the failure this is here to catch.
        """
        with MCPFixture(f"api{uuid.uuid4().hex[:6]}") as fixture:
            resp = client.post(
                "/v1/artifacts",
                json=_server_body(fixture, visibility="global"),
                headers=admin_headers,
            )

            assert resp.status_code == 201, resp.text
            assert resp.json()["data"]["visibility"] == "global"
            servers = mcp_client.servers_for_turn(get_runtime().store)
            assert any(s["url"] == fixture.url for s in servers), servers

    def test_the_default_is_still_private(self, client, admin_headers):
        """Omitting the field must not publish anything.

        Every other caller of this route omits it, and a default that
        published would turn every existing private artifact into everyone's.
        """
        with MCPFixture(f"def{uuid.uuid4().hex[:6]}") as fixture:
            resp = client.post(
                "/v1/artifacts", json=_server_body(fixture), headers=admin_headers
            )

            assert resp.status_code == 201, resp.text
            assert resp.json()["data"]["visibility"] == "private"
            servers = mcp_client.servers_for_turn(get_runtime().store)
            assert not any(s["url"] == fixture.url for s in servers), servers


class TestPublishingIsAdminsOnly:
    def test_a_user_cannot_publish_globally(self, client, auth_headers):
        """Global visibility is a privilege, not a preference.

        A globally visible `tool` artifact enters the process-wide registry
        every turn resolves against, so this field is the difference between
        "my configuration" and "everyone's capability" for more than MCP.
        """
        with MCPFixture(f"esc{uuid.uuid4().hex[:6]}") as fixture:
            resp = client.post(
                "/v1/artifacts",
                json=_server_body(fixture, visibility="global"),
                headers=auth_headers,
            )

            assert resp.status_code == 403, resp.text
            servers = mcp_client.servers_for_turn(get_runtime().store)
            assert not any(s["url"] == fixture.url for s in servers), (
                "the refusal did not stop the write"
            )

    def test_a_user_cannot_publish_to_their_tenant_either(
        self, client, auth_headers
    ):
        with MCPFixture(f"sh{uuid.uuid4().hex[:6]}") as fixture:
            resp = client.post(
                "/v1/artifacts",
                json=_server_body(fixture, visibility="shared"),
                headers=auth_headers,
            )

            assert resp.status_code == 403, resp.text

    def test_a_user_can_still_create_their_own(self, client, auth_headers):
        """The gate is on publishing, not on creating.

        A private artifact is the ordinary case and must stay open to
        everybody, or this becomes an admin-only endpoint by accident.
        """
        resp = client.post(
            "/v1/artifacts",
            json={
                "type": "workflow",
                "name": "mine",
                "schema": {"kind": "workflow.linear", "nodes": []},
            },
            headers=auth_headers,
        )

        assert resp.status_code == 201, resp.text
        assert resp.json()["data"]["visibility"] == "private"


class TestTheRouteRefusesWhatTheResolverWouldRefuse:
    def test_a_non_http_server_url_is_a_bad_request(self, client, admin_headers):
        """The artifact schema already refuses it; this is the route's answer.

        A 500 here would be the validator working and the route reporting it
        as a server fault, which sends an operator looking in the wrong place.
        """
        resp = client.post(
            "/v1/artifacts",
            json={
                "type": "mcp",
                "name": "local",
                "visibility": "global",
                "schema": {
                    "kind": "mcp.server",
                    "name": "local",
                    "url": "file:///etc/passwd",
                },
            },
            headers=admin_headers,
        )

        assert resp.status_code == 400, resp.text

    def test_an_unknown_visibility_is_refused_by_the_schema(
        self, client, admin_headers
    ):
        with MCPFixture(f"bad{uuid.uuid4().hex[:6]}") as fixture:
            resp = client.post(
                "/v1/artifacts",
                json=_server_body(fixture, visibility="everyone"),
                headers=admin_headers,
            )

            assert resp.status_code == 422, resp.text


class TestRetiringAServerGoesThroughConfigOps:
    """Publishing moves an artifact out of its owner's sole control.

    `_get_private_artifact` says so and refuses PATCH and DELETE for anything
    published - "changed and retired through config ops, not here". So the way
    an operator turns a server off has to be the review flow, and that has to
    actually work on this artifact type rather than be a sentence in a
    docstring. Both halves are checked here: the CRUD route refusing, and the
    reviewed path succeeding.
    """

    def _publish(self, client, admin_headers, fixture):
        resp = client.post(
            "/v1/artifacts",
            json=_server_body(fixture, visibility="global"),
            headers=admin_headers,
        )
        assert resp.status_code == 201, resp.text
        return resp.json()["data"]["id"]

    def test_the_crud_route_refuses_a_published_server(
        self, client, admin_headers
    ):
        with MCPFixture(f"pub{uuid.uuid4().hex[:6]}") as fixture:
            artifact_id = self._publish(client, admin_headers, fixture)

            patched = client.patch(
                f"/v1/artifacts/{artifact_id}",
                json={"patch": [{"op": "replace", "path": "/enabled", "value": False}]},
                headers=admin_headers,
            )
            deleted = client.delete(
                f"/v1/artifacts/{artifact_id}", headers=admin_headers
            )

            assert patched.status_code == 403, patched.text
            assert deleted.status_code == 403, deleted.text
            servers = mcp_client.servers_for_turn(get_runtime().store)
            assert any(s["url"] == fixture.url for s in servers), (
                "the refusal did not leave the server standing"
            )

    def test_a_reviewed_patch_takes_the_server_out_of_the_turn(
        self, client, admin_headers
    ):
        """Propose, approve, apply - then ask the turn, not the response."""
        with MCPFixture(f"off{uuid.uuid4().hex[:6]}") as fixture:
            artifact_id = self._publish(client, admin_headers, fixture)

            proposed = client.post(
                "/v1/config/propose_patch",
                json={
                    "artifact_id": artifact_id,
                    "patch": [
                        {"op": "replace", "path": "/enabled", "value": False}
                    ],
                    "justification": "the vendor retired this server",
                },
                headers=admin_headers,
            )
            assert proposed.status_code == 200, proposed.text
            patch_id = proposed.json()["data"]["id"]
            decided = client.post(
                f"/v1/config/patches/{patch_id}/decide",
                json={"decision": "approve"},
                headers=admin_headers,
            )
            assert decided.status_code == 200, decided.text
            applied = client.post(
                f"/v1/config/patches/{patch_id}/apply", headers=admin_headers
            )

            assert applied.status_code == 200, applied.text
            servers = mcp_client.servers_for_turn(get_runtime().store)
            assert not any(s["url"] == fixture.url for s in servers), servers
