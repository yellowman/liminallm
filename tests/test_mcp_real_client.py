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

The resource surface is here for the same reason. The direct tests pin what
the server puts on the wire; this pins that a client holding nothing but the
advertised template can build an address the server accepts, and that the
`_meta` marking a passage untrusted survives the client's own parsing rather
than being dropped as an unknown field. It shares the one session on purpose:
a second live server would cost another port and another startup to witness
the same handshake twice.

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
from liminallm.storage.models import KnowledgeChunk
from tests.test_mcp_resources import HOSTILE_PATH, expand

#: Written out rather than read from `mcp_server`, so a change to the
#: advertised address has to be made here too and is visible in the diff.
CHUNK_TEMPLATE = "liminal://context/{context_id}/doc/~{fs_path}/chunk/{chunk_index}"

#: Short and exact. What matters is that it arrives unchanged through an
#: address built from the template above, not that it is long.
CHUNK_TEXT = "Crate 19 left the bonded warehouse before the audit."

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


def _account(base_url: str) -> tuple[str, dict]:
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
    data = resp.json()["data"]
    return data["user_id"], {"Authorization": f"Bearer {data['access_token']}"}


#: Every method this session sends, in the order it sends them. Each is
#: matched with its quotes, so `resources/templates/list` cannot be recorded
#: as `resources/list`: the needle is the JSON string, not a bare name.
_METHODS = (
    "initialize",
    "notifications/initialized",
    "tools/list",
    "tools/call",
    "resources/list",
    "resources/templates/list",
    "resources/read",
)


@pytest.mark.asyncio
async def test_the_client_library_completes_a_whole_session(live_server, store):
    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    sent: list[tuple[str, str | None]] = []

    async def record(request):
        body = request.content.decode("utf-8", "replace")
        method = None
        for name in _METHODS:
            if f'"{name}"' in body:
                method = name
                break
        sent.append(
            (method, request.headers.get(mcp_server.PROTOCOL_VERSION_HEADER))
        )

    user_id, auth = _account(live_server)
    context_id = store.upsert_context(user_id, "sdk", "one document").id
    store.add_chunks(
        context_id,
        [
            KnowledgeChunk(
                context_id=context_id,
                fs_path=HOSTILE_PATH,
                content=CHUNK_TEXT,
                embedding=[0.0] * 64,
                chunk_index=0,
            )
        ],
    )

    async with httpx2.AsyncClient(
        headers=auth, timeout=30, event_hooks={"request": [record]},
    ) as http_client:
        url = f"{live_server}/v1/mcp"
        async with streamable_http_client(url, http_client=http_client) as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                initialized = await session.initialize()
                listed = await session.list_tools()
                called = await session.call_tool(
                    "note_search", {"query": "anything"}
                )
                resources = await session.list_resources()
                templates = await session.list_resource_templates()
                # Built from what the server advertised, expanded by this
                # suite's own RFC 6570. A client that knows only the template
                # has to be able to reach the passage.
                address = expand(
                    templates.resource_templates[0].uri_template,
                    context_id=context_id,
                    fs_path=HOSTILE_PATH,
                    chunk_index=0,
                )
                read = await session.read_resource(address)

    # The client asked for something newer and took the counter-offer rather
    # than hanging up, so the whole session ran.
    assert initialized.protocol_version == mcp_server.PROTOCOL_VERSION
    assert {tool.name for tool in listed.tools} == {
        "note_search", "knowledge_search"
    }
    assert called.is_error is False
    # Structured output reaches a real client, not only our own JSON-RPC.
    assert called.structured_content == {"notes": []}

    # One context holding one document, so the listing is exactly its one
    # document-level address - and the client parsed the URI as given.
    assert [resource.uri for resource in resources.resources] == [
        mcp_server.document_uri(context_id, HOSTILE_PATH)
    ]
    assert [t.uri_template for t in templates.resource_templates] == [
        CHUNK_TEMPLATE
    ]

    contents = read.contents[0]
    assert contents.uri == address, "the server renamed the address it accepted"
    assert contents.text == CHUNK_TEXT
    assert contents.mime_type == "text/plain"
    # The mark that says this text is data reaches the client. `_meta` is an
    # extension point, and a client model that dropped unknown keys would
    # leave a caller no way to tell a passage from an instruction.
    assert contents.meta == {"liminallm.dev/content-role": "untrusted-data"}

    # The header rule is safe because of this: the handshake carries no
    # version, and everything after it does.
    by_method = dict(sent)
    assert by_method["initialize"] is None
    for method in _METHODS[1:]:
        assert by_method[method] == mcp_server.PROTOCOL_VERSION, (
            f"{method} carried {by_method[method]!r}; the server now refuses "
            f"a post-initialize request without the negotiated version"
        )
