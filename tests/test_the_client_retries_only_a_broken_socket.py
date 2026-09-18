"""REST fallback is for a socket that failed, not for a server that answered.

`sendMessage` wraps the streaming call in `.catch()` and falls back to the REST
`/chat` endpoint. That catch took no argument, so it could not tell a dead
socket from a server that had successfully delivered an application error. A
`conversation not found` error event was therefore retried over REST, the
retry succeeded against a different conversation, and the outer handler that
puts the cause in the error banner never ran.

Found in CI rather than by reading. The browser witness for #233 timed out on
its *second* assertion: the partial bubble was correctly finalized, so that
fix held, and the banner that should have read `conversation not found` was
empty, because the turn had been silently replayed.

The rule these witnesses pin:

  socket open failure, socket error, premature close, idle timeout
      -> the transport failed to carry the exchange; REST fallback is right
  server `event: error`, legacy `status != ok`, a frame that will not parse
      -> the server answered; never replay the turn

Fallback is opt-in, so the default is propagation: an untagged error surfaces
instead of silently re-running a turn. A malformed frame counts as answered,
not as broken transport - the bytes arrived, so the connection worked, and
what failed is the protocol the two sides are meant to share. By then the
server may already have performed part or all of the turn, and an idempotency
key narrows that risk without making it safe to assume.

The frames are driven with `route_web_socket` rather than by coaxing the
server into producing each shape. What is under test is how the client
classifies an answer it has already received, so handing it that answer
directly is both the smaller test and the more faithful one.

Four witnesses: the three shapes that must never be retried, and one that must
still be retried. Without the last, deleting the fallback entirely would pass
this file - and it is also the positive control for the REST counter, which
otherwise reads empty whether or not it can record anything at all.
"""

from __future__ import annotations

import json
import re
import uuid

import pytest

from tests.browser import LiveServer, chromium_executable

pytest.importorskip(
    "playwright",
    reason="the browser lane needs the dev extra: uv pip install playwright",
)

pytestmark = pytest.mark.browser

PASSWORD = "TestPassword123!"

#: The socket the client opens for one chat exchange.
CHAT_SOCKET = re.compile(r"/v1/chat/stream$")

#: The REST endpoint the fallback posts to. Anchored, so it does not also
#: match the socket above.
REST_CHAT = re.compile(r"/v1/chat$")


@pytest.fixture(scope="module")
def server():
    live = LiveServer().start()
    try:
        yield live
    finally:
        live.stop()


@pytest.fixture(scope="module")
def browser():
    from playwright.sync_api import sync_playwright

    with sync_playwright() as play:
        launched = play.chromium.launch(executable_path=chromium_executable())
        try:
            yield launched
        finally:
            launched.close()


class _Turn:
    """One signed-in page, plus control over what its chat socket answers."""

    def __init__(self, page):
        self.page = page
        self.frame = None
        self.close_after = False
        self.rest_calls: list = []

    def answers(self, frame, *, close_after=False):
        self.frame = frame
        self.close_after = close_after

    def send(self):
        self.page.fill("#message-input", "the question this turn asks")
        self.page.click("#chat-form button[type=submit]")

    def banner(self) -> str:
        self.page.wait_for_function(
            """() => {
                const el = document.getElementById('error-banner');
                return el && el.textContent.trim().length > 0;
            }""",
            timeout=30000,
        )
        return (self.page.text_content("#error-banner") or "").strip()


@pytest.fixture
def turn(browser, server):
    """A signed-in page whose chat socket answers whatever the test sets.

    Both routes are installed before the first navigation, and the socket one
    has to be: `route_web_socket` patches `window.WebSocket` through an init
    script, so a route added after the page has loaded never sees the socket
    the client opens. Measured rather than assumed - installed after login,
    the handler was simply never called and the turn went to the real server.
    """
    import httpx

    email = f"wsc_{uuid.uuid4().hex[:8]}@example.com"
    signup = httpx.post(
        f"{server.base_url}/v1/auth/signup",
        json={"email": email, "password": PASSWORD},
        timeout=30,
    )
    assert signup.status_code == 201, signup.text

    context = browser.new_context(viewport={"width": 1440, "height": 900})
    page = context.new_page()
    state = _Turn(page)

    def _handler(ws):
        def _on_message(_sent):
            if state.frame is not None:
                ws.send(state.frame)
            if state.close_after:
                ws.close()

        ws.on_message(_on_message)

    # Nothing calls `connect_to_server`, so the real endpoint is never
    # reached and the client sees exactly the answer the test chose.
    page.route_web_socket(CHAT_SOCKET, _handler)

    def _record(route, request):
        if request.method == "POST":
            state.rest_calls.append(request.url)
        route.continue_()

    page.route(REST_CHAT, _record)

    page.goto(f"{server.base_url}/", wait_until="domcontentloaded")
    page.fill("#email", email)
    page.fill("#password", PASSWORD)
    page.click("#auth-form button[type=submit]")
    page.wait_for_function(
        "() => !!sessionStorage.getItem('liminal.accessToken')", timeout=30000
    )
    page.wait_for_selector("#main-tabs", state="visible")
    try:
        yield state
    finally:
        context.close()


class TestAnAnsweredTurnIsNeverReplayed:
    """The server spoke. Whatever it said, the turn has already happened."""

    def test_a_server_error_event_reaches_the_banner(self, turn):
        """The shape the CI failure actually produced."""
        turn.answers(json.dumps({
            "event": "error",
            "data": {"message": "conversation not found"},
        }))

        turn.send()

        assert turn.banner() == "conversation not found", (
            "the server's own reason did not reach the screen"
        )
        assert turn.rest_calls == [], (
            "the client replayed an answered turn over REST, so a delivered "
            f"application error was treated as a broken socket: "
            f"{turn.rest_calls}"
        )

    def test_a_legacy_error_envelope_reaches_the_banner(self, turn):
        """The same defect through the other protocol shape.

        `event: error` and `status != ok` are two branches of one handler.
        Fixing only the branch CI happened to exercise moves the hole rather
        than closing it.
        """
        turn.answers(json.dumps({
            "status": "error",
            "error": {"message": "the legacy envelope said no"},
        }))

        turn.send()

        assert turn.banner() == "the legacy envelope said no"
        assert turn.rest_calls == [], (
            f"the legacy error envelope was retried over REST: "
            f"{turn.rest_calls}"
        )

    def test_a_frame_that_will_not_parse_reaches_the_banner(self, turn):
        """A protocol failure, not a transport one.

        The bytes arrived, so the connection worked. Bad JSON means a server
        bug, version skew, or something rewriting frames in between - none of
        which a retry should hide, and any of which may already have run the
        turn.
        """
        turn.answers("{this is not json")

        turn.send()

        assert turn.banner() == "Received invalid response"
        assert turn.rest_calls == [], (
            f"an unparseable frame was retried over REST: {turn.rest_calls}"
        )


class TestABrokenSocketStillFallsBack:
    def test_a_close_without_an_answer_is_retried_over_rest(self, turn):
        """The other direction, and the reason the three above mean anything.

        If fallback were simply removed, every witness in this file would pass
        while the feature it guards was gone. It is also the control for the
        counter: an empty `rest_calls` is only evidence once the same counter
        has been shown to record something.
        """
        turn.answers(None, close_after=True)

        turn.send()

        turn.page.wait_for_selector(
            ".message.assistant:not(.streaming)", timeout=60000
        )
        assert turn.rest_calls, (
            "a socket that closed without answering did not fall back to "
            "REST, so the fix withdrew a retry that should still happen"
        )
