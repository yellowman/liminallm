"""A peer that vanishes mid-request is not an application failure.

Two routes read their request body by hand, because both need a wire shape
FastAPI's own body parsing would not give them: `/v1/mcp` answers JSON-RPC and
`/v1/responses` answers OpenAI's error object. Reading it by hand means owning
what happens when the body never finishes arriving.

It raises `ClientDisconnect`, which is not a `ValueError` and so escaped the
parse guard on `/v1/mcp` entirely - an unhandled ASGI exception, six frames of
traceback, and a 500 for an event the server did not cause. On
`/v1/responses` it was caught by the catch-all and recorded as
`responses_turn_failed`, which is quieter and still wrong: a caller who
cancels an upload did not fail a turn.

Routes whose body FastAPI parses - `/v1/chat` and every typed-body endpoint -
were measured and are unaffected: the framework handles the disconnect before
the endpoint runs. So this is two sites, not a class, and the fix is at both
of them rather than in a body-reading framework invented for two callers.

The live-socket tests are marked slow: a real socket has to open and then
stop talking, which needs a real server rather than `TestClient`. The two at
the end are not - they drive the route function over a real disconnecting
receive channel, which needs no server and no port.
"""

from __future__ import annotations

import socket
import threading
import time
import uuid

import httpx
import pytest
import uvicorn

#: Long enough that the server is still waiting for the rest when we hang up.
PROMISED_EXTRA_BYTES = 50


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def live():
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
        yield "127.0.0.1", config.port
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def _token(host, port) -> str:
    resp = httpx.post(
        f"http://{host}:{port}/v1/auth/signup",
        json={"email": f"dc_{uuid.uuid4().hex[:8]}@example.com",
              "password": "TestPassword123!"},
        timeout=30,
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["data"]["access_token"]


def _vanish_mid_body(host, port, path, token):
    """Announce a body, send part of it, then hang up."""
    started = b'{"jsonrpc":"2.0","method":"notifications/ini'
    head = (
        f"POST {path} HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        f"Authorization: Bearer {token}\r\n"
        f"Content-Type: application/json\r\n"
        f"MCP-Protocol-Version: 2025-06-18\r\n"
        f"Content-Length: {len(started) + PROMISED_EXTRA_BYTES}\r\n"
        f"\r\n"
    ).encode()
    sock = socket.create_connection((host, port), timeout=10)
    sock.sendall(head + started)
    time.sleep(0.3)
    sock.close()
    time.sleep(0.6)


@pytest.mark.slow
@pytest.mark.parametrize("path", ["/v1/mcp", "/v1/responses"])
def test_a_vanished_peer_leaves_no_wreckage(live, capfd, path):
    """The whole point, measured the way an operator meets it: what the
    process prints.

    Captured at the file descriptor, because the two noises come out of
    different places - uvicorn's unhandled-exception report through its own
    logger, and the app's own traceback through structlog to stdout - and no
    single logging handler sees both.
    """
    host, port = live
    token = _token(host, port)
    capfd.readouterr()

    _vanish_mid_body(host, port, path, token)

    printed = "".join(capfd.readouterr())
    assert "Exception in ASGI application" not in printed
    assert "Traceback (most recent call last)" not in printed
    assert "responses_turn_failed" not in printed


@pytest.mark.slow
def test_a_body_that_arrived_malformed_is_still_the_caller_s_mistake(live):
    """The other half of the pair, so the two stay told apart.

    A body that arrived and is unparseable gets the parse error it always
    got. Only a body that never finished arriving is silent. Folding the two
    together - widening the parse guard, or reaching for `except Exception` -
    would answer a real fault with a shrug.
    """
    host, port = live
    token = _token(host, port)

    resp = httpx.post(
        f"http://{host}:{port}/v1/mcp",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "MCP-Protocol-Version": "2025-06-18",
        },
        content=b"not json{",
        timeout=30,
    )

    assert resp.status_code == 200, resp.text
    assert resp.json()["error"]["code"] == -32700


@pytest.mark.slow
def test_a_genuine_failure_still_surfaces(live, monkeypatch, capfd):
    """What stops somebody closing this by widening the catch.

    The disconnect handler is narrow on purpose. If it ever grows into
    `except Exception`, a real fault on this route becomes a quiet 499 with
    nothing in the log - so a real fault is raised here and required to still
    be loud.
    """
    from liminallm.api import mcp as mcp_module

    host, port = live
    token = _token(host, port)

    def explode(*_args, **_kwargs):
        raise RuntimeError("a genuine bug, not a disconnect")

    monkeypatch.setattr(mcp_module, "handle_message", explode)
    capfd.readouterr()

    resp = httpx.post(
        f"http://{host}:{port}/v1/mcp",
        headers={
            "Authorization": f"Bearer {token}",
            "MCP-Protocol-Version": "2025-06-18",
        },
        json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
        timeout=30,
    )

    assert resp.status_code == 500, resp.text
    assert "a genuine bug" in "".join(capfd.readouterr())


def _request(receive):
    """A real `Request` over a receive channel that behaves as given.

    Not a stand-in for one: `await request.json()` runs Starlette's own body
    reader, so `ClientDisconnect` is raised by the code that raises it in
    production rather than by a double imitating it.
    """
    from starlette.requests import Request

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/mcp",
        "root_path": "",
        "query_string": b"",
        "headers": [
            (b"content-type", b"application/json"),
            (b"mcp-protocol-version", b"2025-06-18"),
        ],
        "client": ("127.0.0.1", 51234),
    }
    return Request(scope, receive)


@pytest.mark.asyncio
async def test_the_two_endings_are_told_apart_by_what_they_answer(store):
    """Absence of noise is not enough to pin this.

    Widening the parse guard to `except Exception` also silences the
    disconnect - quietly answering it with the parse error meant for a body
    that actually arrived. The difference is only visible in what comes back,
    so that is what this asserts: a vanished peer gets 499, and a malformed
    body still gets -32700.
    """
    from liminallm.api.routes import mcp_endpoint
    from liminallm.service.auth import AuthContext

    user = store.create_user(email=f"seam_{uuid.uuid4().hex[:8]}@example.com")
    principal = AuthContext(
        user_id=user.id, role=user.role, tenant_id=user.tenant_id, session_id=None
    )

    async def vanishes():
        return {"type": "http.disconnect"}

    async def arrives_malformed():
        return {"type": "http.request", "body": b"not json{", "more_body": False}

    gone = await mcp_endpoint(_request(vanishes), principal)
    malformed = await mcp_endpoint(_request(arrives_malformed), principal)

    assert gone.status_code == 499
    assert malformed.status_code == 200
    assert b"-32700" in malformed.body
