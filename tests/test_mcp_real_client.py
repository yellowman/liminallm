"""The MCP server, driven by the MCP client library this project depends on.

The rest of the MCP suite speaks JSON-RPC to the endpoint directly, which
pins what the server does but not whether a real client agrees. This file is
the other half, and it exists because of two things the direct tests could
not have told us.

The server refuses a post-initialize request that carries no
`MCP-Protocol-Version`. That is only safe if real clients send it - so this
watches the wire and checks that they do, rather than assuming it.

And the server counter-offers `2025-06-18` to a client asking for something
newer. That is only useful if the client accepts a downgrade instead of
hanging up, which is a decision made inside the client library.

Marked slow: it binds a port and runs a real server, which the fast lane
does not need on every change.
"""

from __future__ import annotations

import socket
import threading
import time
import uuid

import httpx
import pytest
import uvicorn

from liminallm.api import mcp as mcp_server

#: The client library's own HTTP layer, which `streamable_http_client` takes
#: and this file needs directly to watch the headers going out. It arrives
#: with `mcp` rather than being declared, so it is fetched the way the
#: dependency guard requires: a skip where it is absent, never a collection
#: error that would take the whole lane down.
httpx2 = pytest.importorskip("httpx2")

pytestmark = pytest.mark.slow


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def live_server():
    """The real app over real HTTP. `TestClient` would not exercise the
    transport, and the transport is what this file is about."""
    from liminallm import app as app_module

    config = uvicorn.Config(
        app_module.app, host="127.0.0.1", port=_free_port(), log_level="error"
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 30
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    assert server.started, "the server under test never came up"
    try:
        yield f"http://127.0.0.1:{config.port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def _account(base_url: str) -> dict:
    """Created inside the test: the suite truncates between tests, so an
    account made at setup time is gone before the body runs."""
    resp = httpx.post(
        f"{base_url}/v1/auth/signup",
        json={
            "email": f"sdk_{uuid.uuid4().hex[:8]}@example.com",
            "password": "TestPassword123!",
        },
        timeout=30,
    )
    assert resp.status_code == 201, resp.text
    return {"Authorization": f"Bearer {resp.json()['data']['access_token']}"}


@pytest.mark.asyncio
async def test_the_client_library_completes_a_whole_session(live_server):
    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    sent: list[tuple[str, str | None]] = []

    async def record(request):
        body = request.content.decode("utf-8", "replace")
        method = None
        for name in ("initialize", "notifications/initialized", "tools/list",
                     "tools/call"):
            if f'"{name}"' in body:
                method = name
                break
        sent.append(
            (method, request.headers.get(mcp_server.PROTOCOL_VERSION_HEADER))
        )

    async with httpx2.AsyncClient(
        headers=_account(live_server), timeout=30,
        event_hooks={"request": [record]},
    ) as http_client:
        url = f"{live_server}/v1/mcp"
        async with streamable_http_client(url, http_client=http_client) as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                initialized = await session.initialize()
                listed = await session.list_tools()
                called = await session.call_tool(
                    "note_search", {"query": "anything"}
                )

    # The client asked for something newer and took the counter-offer rather
    # than hanging up, so the whole session ran.
    assert initialized.protocol_version == mcp_server.PROTOCOL_VERSION
    assert {tool.name for tool in listed.tools} == {
        "note_search", "knowledge_search"
    }
    assert called.is_error is False
    # Structured output reaches a real client, not only our own JSON-RPC.
    assert called.structured_content == {"notes": []}

    # The header rule is safe because of this: the handshake carries no
    # version, and everything after it does.
    by_method = dict(sent)
    assert by_method["initialize"] is None
    for method in ("notifications/initialized", "tools/list", "tools/call"):
        assert by_method[method] == mcp_server.PROTOCOL_VERSION, (
            f"{method} carried {by_method[method]!r}; the server now refuses "
            f"a post-initialize request without the negotiated version"
        )
