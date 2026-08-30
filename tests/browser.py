"""A real browser against the real server, for the seams only a browser has.

Two defect classes live here and nowhere else: what the SPA persists where
scripts can read it, and what the browser actually puts on the wire once
cookies, CSRF and same-origin rules are in play. Neither is reachable from
`TestClient`, which has no cookie jar policy, no script context, and no
opinion about `HttpOnly`.

The server runs in a thread rather than a subprocess so it shares this
process's already-configured runtime - the same Postgres and Redis the rest
of the suite uses, with no environment plumbing to keep in step.
"""

from __future__ import annotations

import os
import pathlib
import socket
import threading
import time
from typing import Optional


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def chromium_executable() -> Optional[str]:
    """The browser to drive, or None to let Playwright find its own.

    A pre-provisioned image ships one build of Chromium and the installed
    Playwright may expect another, so the path is resolved rather than
    assumed. `LIMINALLM_CHROMIUM` wins; otherwise the newest build under
    `PLAYWRIGHT_BROWSERS_PATH`; otherwise Playwright's own lookup, which is
    what a developer who ran `playwright install` has.
    """
    explicit = os.environ.get("LIMINALLM_CHROMIUM")
    if explicit:
        return explicit
    root = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    if not root:
        return None
    builds = sorted(pathlib.Path(root).glob("chromium-*/chrome-linux/chrome"))
    return str(builds[-1]) if builds else None


class LiveServer:
    """The real ASGI app on a real port, started and stopped by a test."""

    def __init__(self) -> None:
        self.port = free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        self._server = None
        self._thread: Optional[threading.Thread] = None

    def start(self, *, timeout: float = 30.0) -> "LiveServer":
        import uvicorn

        from liminallm import app as app_module

        config = uvicorn.Config(
            app_module.app,
            host="127.0.0.1",
            port=self.port,
            log_level="error",
            # The app is already imported and configured in this process.
            lifespan="on",
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        deadline = time.time() + timeout
        while time.time() < deadline:
            if getattr(self._server, "started", False):
                return self
            time.sleep(0.05)
        raise RuntimeError(f"the server did not start within {timeout}s")

    def stop(self, *, timeout: float = 10.0) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=timeout)
