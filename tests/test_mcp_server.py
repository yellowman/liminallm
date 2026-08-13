"""The MCP server: the kernel's retrieval, spoken to outside agents.

Wire-level tests: JSON-RPC 2.0 over one POST endpoint, the 2025-06-18
protocol subset (initialize, ping, tools/list, tools/call; notifications
202; batching rejected by name), and the same API-key/session auth and
ownership verdicts as the rest of the agent surface.
"""

import uuid

from liminallm.service.runtime import get_runtime


def _rpc(client, headers, method, params=None, request_id=1):
    body = {"jsonrpc": "2.0", "id": request_id, "method": method}
    if params is not None:
        body["params"] = params
    return client.post("/v1/mcp", headers=headers, json=body)


def _tool_text(resp):
    result = resp.json()["result"]
    assert result["isError"] is False, result
    return result["content"][0]["text"]


class TestMcpProtocol:
    def test_initialize_handshake(self, client, auth_headers):
        resp = _rpc(
            client,
            auth_headers,
            "initialize",
            {"protocolVersion": "2025-06-18", "capabilities": {}},
        )
        assert resp.status_code == 200, resp.text
        result = resp.json()["result"]
        assert result["protocolVersion"] == "2025-06-18"
        assert "tools" in result["capabilities"]
        assert result["serverInfo"]["name"] == "liminallm"

    def test_initialize_with_unknown_version_offers_ours(self, client, auth_headers):
        resp = _rpc(
            client, auth_headers, "initialize", {"protocolVersion": "2024-11-05"}
        )
        assert resp.json()["result"]["protocolVersion"] == "2025-06-18"

    def test_notification_is_202_with_no_body(self, client, auth_headers):
        resp = client.post(
            "/v1/mcp",
            headers=auth_headers,
            json={"jsonrpc": "2.0", "method": "notifications/initialized"},
        )
        assert resp.status_code == 202
        assert resp.content == b""

    def test_tools_list(self, client, auth_headers):
        resp = _rpc(client, auth_headers, "tools/list")
        tools = {t["name"]: t for t in resp.json()["result"]["tools"]}
        assert set(tools) == {"note_search", "knowledge_search"}
        for tool in tools.values():
            assert tool["inputSchema"]["required"] == ["query"]

    def test_ping(self, client, auth_headers):
        assert _rpc(client, auth_headers, "ping").json()["result"] == {}

    def test_unknown_method_is_32601(self, client, auth_headers):
        resp = _rpc(client, auth_headers, "resources/list")
        assert resp.json()["error"]["code"] == -32601

    def test_unknown_tool_is_32602(self, client, auth_headers):
        resp = _rpc(client, auth_headers, "tools/call", {"name": "shell_exec"})
        assert resp.json()["error"]["code"] == -32602

    def test_batch_is_rejected_by_name(self, client, auth_headers):
        resp = client.post(
            "/v1/mcp",
            headers=auth_headers,
            json=[{"jsonrpc": "2.0", "id": 1, "method": "ping"}],
        )
        error = resp.json()["error"]
        assert error["code"] == -32600
        assert "2025-06-18" in error["message"]

    def test_malformed_json_is_parse_error(self, client, auth_headers):
        resp = client.post(
            "/v1/mcp",
            headers={**auth_headers, "Content-Type": "application/json"},
            content=b"not json{",
        )
        assert resp.json()["error"]["code"] == -32700

    def test_get_is_405(self, client):
        assert client.get("/v1/mcp").status_code == 405


class TestMcpTools:
    def test_note_search_finds_a_seeded_note(self, client, auth_headers):
        # Resolve the user id from a minted key record (headers carry no id).
        signup = client.post(
            "/v1/auth/signup",
            json={
                "email": f"mcp_{uuid.uuid4().hex[:8]}@example.com",
                "password": "TestPassword123!",
            },
        )
        headers = {
            "Authorization": f"Bearer {signup.json()['data']['access_token']}"
        }
        user_id = signup.json()["data"]["user_id"]
        get_runtime().store.create_note(
            user_id,
            "Deployment runbook",
            "The zephyrine cluster restarts every wednesday.",
        )

        text = _tool_text(
            _rpc(client, headers, "tools/call",
                 {"name": "note_search", "arguments": {"query": "zephyrine"}})
        )
        assert "Deployment runbook" in text

    def test_knowledge_search_grounds_in_a_context(self, client, auth_headers):
        ctx = client.post(
            "/v1/contexts",
            headers=auth_headers,
            json={
                "name": "MCP ctx",
                "description": "grounding",
                "text": "The launch code is stored in the vermilion cabinet.",
            },
        )
        assert ctx.status_code == 201, ctx.text
        context_id = ctx.json()["data"]["id"]

        scoped = _tool_text(
            _rpc(client, auth_headers, "tools/call",
                 {"name": "knowledge_search",
                  "arguments": {"query": "vermilion cabinet", "context_id": context_id}})
        )
        assert "vermilion" in scoped

        unscoped = _tool_text(
            _rpc(client, auth_headers, "tools/call",
                 {"name": "knowledge_search", "arguments": {"query": "vermilion cabinet"}})
        )
        assert "vermilion" in unscoped

    def test_foreign_context_is_a_tool_error(self, client, auth_headers):
        other = client.post(
            "/v1/auth/signup",
            json={
                "email": f"other_{uuid.uuid4().hex[:8]}@example.com",
                "password": "TestPassword123!",
            },
        )
        other_headers = {
            "Authorization": f"Bearer {other.json()['data']['access_token']}"
        }
        ctx = client.post(
            "/v1/contexts",
            headers=other_headers,
            json={"name": "Not yours", "description": "x"},
        )
        context_id = ctx.json()["data"]["id"]

        resp = _rpc(client, auth_headers, "tools/call",
                    {"name": "knowledge_search",
                     "arguments": {"query": "anything", "context_id": context_id}})
        result = resp.json()["result"]
        assert result["isError"] is True
        assert "another user" in result["content"][0]["text"]

    def test_missing_query_is_a_tool_error(self, client, auth_headers):
        resp = _rpc(client, auth_headers, "tools/call",
                    {"name": "note_search", "arguments": {}})
        result = resp.json()["result"]
        assert result["isError"] is True
        assert "query" in result["content"][0]["text"]


class TestMcpAuth:
    def test_api_key_authenticates_mcp(self, client, auth_headers):
        key = client.post(
            "/v1/auth/api-keys", headers=auth_headers, json={"name": "mcp agent"}
        ).json()["data"]["api_key"]
        resp = _rpc(
            client, {"Authorization": f"Bearer {key}"}, "tools/list"
        )
        assert resp.status_code == 200, resp.text
        assert len(resp.json()["result"]["tools"]) == 2

    def test_no_auth_is_401(self, client):
        resp = client.post(
            "/v1/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "ping"}
        )
        assert resp.status_code == 401
