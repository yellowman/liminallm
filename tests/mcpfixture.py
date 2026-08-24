"""A real MCP server on a real port, with the knobs these tests need.

Not a stub of the protocol: the point of using the official SDK on the client
side is that the wire is somebody else's problem, and a hand-written fake
would put it back. This is the SDK's own server, so a test that passes here
passed against a real handshake, a real `tools/list` and a real `tools/call`.
"""

from __future__ import annotations

import socket
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class MCPFixture:
    """One server, its tools, and a record of what was actually called.

    `calls` is the load-bearing part of several reds: proving a withdrawn tool
    returned a refusal is weaker than proving the remote server never heard
    from us at all.
    """

    def __init__(
        self,
        name: str = "fixture",
        tools: Optional[Dict[str, Any]] = None,
        *,
        redirect_from: Optional[str] = None,
        metadata: Optional[Dict[str, dict]] = None,
        list_delay: float = 0.0,
    ) -> None:
        self.name = name
        #: remote tool name -> str, or a callable taking the arguments dict.
        self.tools: Dict[str, Any] = dict(tools or {"echo": "ok"})
        self.calls: List[Tuple[str, dict]] = []
        #: How many times this server has been asked for its tools. Separate
        #: from `calls` because the two answer different questions: `listed`
        #: is whether a turn reached discovery at all, which is the only way
        #: to tell a workflow that chose the tool-agent path from one that
        #: never did.
        self.listed = 0
        #: remote tool name -> the `description`/`inputSchema` this server
        #: advertises for it. The point of a hostile server is that it writes
        #: its own metadata, so a test that wants one has to be able to.
        self.metadata: Dict[str, dict] = dict(metadata or {})
        #: Seconds this server takes to answer `tools/list`. A slow third
        #: party is the ordinary case, not an exotic one, and it is the only
        #: way to see whether discovery is holding a thread somebody else
        #: needs.
        self.list_delay = list_delay
        self.redirect_from = redirect_from
        self.port = free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.url = f"{self.base_url}/mcp"
        self._server = None
        self._thread: Optional[threading.Thread] = None

    def _app(self):
        from mcp import types
        from mcp.server import Server

        async def on_list_tools(ctx, params):
            self.listed += 1
            if self.list_delay:
                import asyncio

                await asyncio.sleep(self.list_delay)
            return types.ListToolsResult(
                tools=[
                    types.Tool(
                        name=tool,
                        description=self.metadata.get(tool, {}).get(
                            "description", f"{tool} on {self.name}"
                        ),
                        inputSchema=self.metadata.get(tool, {}).get(
                            "inputSchema", {"type": "object", "properties": {}}
                        ),
                    )
                    for tool in self.tools
                ]
            )

        async def on_call_tool(ctx, params):
            arguments = dict(params.arguments or {})
            self.calls.append((params.name, arguments))
            body = self.tools.get(params.name, "")
            text = body(arguments) if callable(body) else str(body)
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=text)]
            )

        server = Server(
            self.name,
            version="1.0",
            on_list_tools=on_list_tools,
            on_call_tool=on_call_tool,
        )
        app = server.streamable_http_app()
        if self.redirect_from:
            from starlette.responses import RedirectResponse
            from starlette.routing import Route

            target = self.redirect_from

            async def _redirect(request):
                # 307: the client must repeat the POST, which is what makes
                # this a redirect the transport actually follows.
                return RedirectResponse(target, status_code=307)

            app.routes.insert(0, Route("/mcp", _redirect, methods=["GET", "POST"]))
        return app

    def start(self, *, timeout: float = 30.0) -> "MCPFixture":
        import uvicorn

        config = uvicorn.Config(
            self._app(), host="127.0.0.1", port=self.port, log_level="error"
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        deadline = time.time() + timeout
        while time.time() < deadline:
            if getattr(self._server, "started", False):
                return self
            time.sleep(0.05)
        raise RuntimeError("the MCP fixture did not start")

    def stop(self, *, timeout: float = 10.0) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def __enter__(self) -> "MCPFixture":
        return self.start()

    def __exit__(self, *_exc) -> None:
        self.stop()

    def as_artifact_schema(self, *, taint_class: str = "egress") -> dict:
        return {
            "kind": "mcp.server",
            "name": self.name,
            "url": self.url,
            "enabled": True,
            "taint_class": taint_class,
        }

    def as_server(self, *, taint_class: str = "egress") -> dict:
        """The resolved shape `discover` takes, for tests that skip the store."""
        return {
            "artifact_id": f"fixture-{self.name}",
            "name": self.name,
            "url": self.url,
            "taint_class": taint_class,
        }


def dead_server(name: str = "dead") -> dict:
    """A configured server that is not listening. Nothing binds this port."""
    return {
        "artifact_id": f"fixture-{name}",
        "name": name,
        "url": f"http://127.0.0.1:{free_port()}/mcp",
        "taint_class": "egress",
    }


def allow_local(*extra: str):
    from liminallm.service.sandbox import ToolNetworkPolicy

    return ToolNetworkPolicy(allowlist=["127.0.0.1", *extra])


Handler = Callable[[dict], str]
