"""Regressions for bugs found reviewing the tool/attachment/web branch."""
import http.server
import threading

import pytest

from liminallm.service.interpreter import publish_artifacts
from liminallm.service.upload_policy import ALLOWED_UPLOAD_EXTENSIONS
from liminallm.service.web import WebFetchError, _unwrap_ddg_url, fetch_url


# ---------------------------------------------------------------------------
# fetch_url must not buffer an oversized body before truncating
# ---------------------------------------------------------------------------

_HUGE = 64 * 1024 * 1024  # far more than the cap under test


class _LyingSizeHandler(http.server.BaseHTTPRequestHandler):
    """Advertises text/plain, then streams far more than the cap."""

    served = 0

    def do_GET(self):  # noqa: N802
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        chunk = b"x" * 65536
        try:
            for _ in range(_HUGE // len(chunk)):
                self.wfile.write(chunk)
                type(self).served += len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass  # expected: the client stops reading at the cap

    def log_message(self, *args):
        return


def test_fetch_stops_reading_at_the_cap():
    """A hostile server must not be able to make us buffer its whole body.

    Before the fix the entire response was read into memory and only then
    truncated, so a multi-gigabyte body could exhaust the worker.
    """
    _LyingSizeHandler.served = 0
    server = http.server.HTTPServer(("127.0.0.1", 0), _LyingSizeHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        page = fetch_url(
            f"http://127.0.0.1:{server.server_port}/huge",
            allow_private=True,
            max_bytes=256 * 1024,
        )
    finally:
        server.shutdown()
        server.server_close()

    assert len(page["text"]) <= 256 * 1024
    # Well short of the 64MB the server was willing to send.
    assert _LyingSizeHandler.served < 8 * 1024 * 1024, (
        f"read {_LyingSizeHandler.served} bytes; the cap is not bounding the download"
    )


# ---------------------------------------------------------------------------
# Interpreter artifacts must obey the upload extension policy
# ---------------------------------------------------------------------------


def test_published_artifacts_obey_the_upload_allowlist(tmp_path):
    workdir = tmp_path / "session"
    workdir.mkdir()
    dest = tmp_path / "files"
    for name in ("report.csv", "notes.txt", "payload.exe", "page.html", "lib.py"):
        (workdir / name).write_text("x")
    created = [{"name": n} for n in ("report.csv", "notes.txt", "payload.exe", "page.html", "lib.py")]

    published = publish_artifacts(
        str(workdir), str(dest), created, allowed_extensions=ALLOWED_UPLOAD_EXTENSIONS
    )

    assert sorted(published) == ["notes.txt", "report.csv"]
    assert not (dest / "payload.exe").exists()
    assert not (dest / "page.html").exists()
    assert not (dest / "lib.py").exists()


def test_published_artifacts_still_reject_paths_and_dotfiles(tmp_path):
    workdir = tmp_path / "s"
    workdir.mkdir()
    (workdir / "ok.txt").write_text("x")
    dest = tmp_path / "d"
    published = publish_artifacts(
        str(workdir),
        str(dest),
        [{"name": "../escape.txt"}, {"name": ".hidden.txt"}, {"name": "ok.txt"}],
        allowed_extensions=ALLOWED_UPLOAD_EXTENSIONS,
    )
    assert published == ["ok.txt"]
    assert not (tmp_path / "escape.txt").exists()


# ---------------------------------------------------------------------------
# DuckDuckGo wraps result links in a protocol-relative redirect
# ---------------------------------------------------------------------------


def test_ddg_redirect_wrapper_is_unwrapped():
    """Taking the href verbatim dropped every result (it fails the http check)."""
    href = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage&rut=abc"
    assert _unwrap_ddg_url(href) == "https://example.com/page"


def test_ddg_protocol_relative_url_becomes_absolute():
    assert _unwrap_ddg_url("//example.org/x").startswith("https://")


def test_ddg_direct_url_is_left_alone():
    assert _unwrap_ddg_url("https://direct.example/p") == "https://direct.example/p"


def test_ddg_parsed_results_survive_the_url_filter():
    from liminallm.service.web import _results_from_ddg_html

    html = (
        '<a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fex.com%2Fa">'
        "First result</a>"
    )
    results = _results_from_ddg_html(html)
    assert results and results[0]["url"] == "https://ex.com/a"
    assert results[0]["url"].startswith("https://")  # would be dropped otherwise


# ---------------------------------------------------------------------------
# A mid-stream failure must not append a second, complete answer
# ---------------------------------------------------------------------------


class _FailsMidStreamBackend:
    """Streams a few tokens, then the provider drops the connection."""

    @property
    def supports_tools(self):
        return True

    def generate_with_tools(self, messages, tools, adapters, *, user_id=None):
        return {
            "content": "",
            "tool_calls": [],
            "assistant_message": {"role": "assistant", "content": ""},
            "usage": {},
        }

    def generate_stream(self, messages, adapters, *, user_id=None):
        yield {"event": "token", "data": "Partial "}
        yield {"event": "token", "data": "answer"}
        raise RuntimeError("provider dropped the stream")

    def generate(self, messages, adapters, *, user_id=None):
        return {"content": "A WHOLE SECOND ANSWER", "usage": {}}


def test_mid_stream_failure_does_not_duplicate_the_answer(monkeypatch):
    """Falling back after tokens were sent glued two answers into one bubble."""
    import asyncio

    from liminallm.service.runtime import get_runtime

    runtime = get_runtime()
    engine = runtime.workflow
    previous = runtime.llm.backend
    runtime.llm.backend = _FailsMidStreamBackend()
    monkeypatch.setattr(engine.settings, "web_tools_enabled", True, raising=False)
    try:
        events = asyncio.run(_collect(engine))
    finally:
        runtime.llm.backend = previous

    tokens = "".join(e["data"] for e in events if e["event"] == "token")
    done = [e for e in events if e["event"] == "message_done"]
    assert tokens == "Partial answer"
    # The fallback's full answer must not have been streamed on top.
    assert "SECOND ANSWER" not in tokens
    assert done, "the turn must still be completed"
    assert done[-1]["data"]["content"] == "Partial answer"


async def _collect(engine):
    return [
        event
        async for event in engine.run_streaming(
            workflow_id=None,
            conversation_id=None,
            user_message="anything",
            context_id=None,
            user_id=None,
            tenant_id=None,
        )
    ]
