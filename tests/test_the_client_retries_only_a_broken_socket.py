"""REST fallback is only for a request that never reached the server.

`sendMessage` wraps the streaming call in `.catch()` and falls back to the REST
`/chat` endpoint. That catch took no argument, so it could not tell a socket
that failed from a server that had answered. Two different defects follow, and
the boundary between them is not "did a terminal frame arrive" but "was the
request handed to the socket".

**An answer is not a failure.** A `conversation not found` error event was
retried over REST, the retry succeeded against a different conversation, and
the outer handler that puts the cause in the error banner never ran. Found in
CI: the browser witness for #233 timed out on its *second* assertion, so the
bubble was correctly finalized and only the truthful banner was missing.

**A turn already sent must never be replayed.** The two transports do not
share an idempotency slot - the socket claims `chat:ws` and the REST route
claims `chat`, and `redis_cache.acquire_idempotency_slot` builds the key as
`idemp:{tenant}{route}:{user}:{key}`, so neither can see the other's claim.
The socket appends the user's message in `chat_turn.begin` (routes.py:6115)
and stores its result only after `chat_turn.finish` (routes.py:6243). A
disconnect in between therefore leaves the message written, the socket's slot
merely in progress, and no completed response for a retry to replay. The REST
attempt appends the message a second time and runs the workflow again -
duplicated tool effects, not only duplicated inference.

So the rule these witnesses pin:

  the socket would not open, or failed before the turn was sent
      -> the request never reached the server; REST fallback is right
  anything after the request is on the socket - error, close, idle timeout,
  an unparseable frame, `event: error`, legacy `status != ok`
      -> the outcome is ambiguous or already decided; never replay

Fallback is opt-in, so the default is propagation. The tag lives only inside
`openChatSocket`, which returns before the request exists, so the boundary is
enforced by where the function can be called rather than by a flag each new
reject site has to remember.

The frames are driven with `route_web_socket` rather than by coaxing the
server into producing each shape. What is under test is how the client
classifies an outcome, so handing it that outcome directly is both the smaller
test and the more faithful one.
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

#: A socket that never opens. `openChatSocket` sees `error` before it can send
#: the turn, which is the one case where nothing reached the server and a
#: fallback is safe. Injected in the client rather than by breaking the
#: network, because the reject site is what is being classified.
FAILS_TO_OPEN = """
class FailingSocket extends EventTarget {
  constructor(url) {
    super();
    this.url = url;
    this.readyState = 3;
    setTimeout(() => this.dispatchEvent(new Event('error')), 0);
  }
  send() {}
  close() {}
}
window.WebSocket = FailingSocket;
"""


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
    """One signed-in page, plus control over what its chat socket does."""

    def __init__(self, page):
        self.page = page
        self.frames: list = []
        self.close_after = False
        self.rest_calls: list = []

    def answers(self, *frames, close_after=False):
        self.frames = [f for f in frames if f is not None]
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
def signed_in(browser, server):
    """Build a signed-in page with the socket behaviour a test needs.

    Both routes go on before the first navigation, and the socket one has to:
    `route_web_socket` patches `window.WebSocket` through an init script, so a
    route added after the page has loaded never sees the socket the client
    opens. Measured rather than assumed - installed after login, the handler
    was never called and the turn went to the real server.
    """
    import httpx

    contexts = []

    def _make(*, fails_to_open=False):
        email = f"wsc_{uuid.uuid4().hex[:8]}@example.com"
        signup = httpx.post(
            f"{server.base_url}/v1/auth/signup",
            json={"email": email, "password": PASSWORD},
            timeout=30,
        )
        assert signup.status_code == 201, signup.text

        context = browser.new_context(viewport={"width": 1440, "height": 900})
        contexts.append(context)
        page = context.new_page()
        state = _Turn(page)

        if fails_to_open:
            page.add_init_script(FAILS_TO_OPEN)
        else:
            def _handler(ws):
                def _on_message(_sent):
                    for frame in state.frames:
                        ws.send(frame)
                    if state.close_after:
                        ws.close()

                ws.on_message(_on_message)

            # Nothing calls `connect_to_server`, so the real endpoint is never
            # reached and the client sees exactly what the test chose.
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
        return state

    try:
        yield _make
    finally:
        for context in contexts:
            context.close()


class TestAnAnsweredTurnIsNeverReplayed:
    """The server spoke. Whatever it said, the turn has already happened."""

    def test_a_server_error_event_reaches_the_banner(self, signed_in):
        """The shape the CI failure actually produced."""
        turn = signed_in()
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

    def test_a_legacy_error_envelope_reaches_the_banner(self, signed_in):
        """The same defect through the other protocol shape.

        `event: error` and `status != ok` are two branches of one handler.
        Fixing only the branch CI happened to exercise moves the hole rather
        than closing it.
        """
        turn = signed_in()
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

    def test_a_frame_that_will_not_parse_reaches_the_banner(self, signed_in):
        """A protocol failure, not a transport one.

        The bytes arrived, so the connection worked. Bad JSON means a server
        bug, version skew, or something rewriting frames in between - none of
        which a retry should hide, and any of which may already have run the
        turn.
        """
        turn = signed_in()
        turn.answers("{this is not json")

        turn.send()

        assert turn.banner() == "Received invalid response"
        assert turn.rest_calls == [], (
            f"an unparseable frame was retried over REST: {turn.rest_calls}"
        )


class TestATurnAlreadySentIsNeverReplayed:
    """The socket died mid-turn. What the server did with it is unknown."""

    def test_a_close_after_a_token_is_not_retried(self, signed_in):
        """The clearest case: the turn demonstrably started.

        A token proves `chat_turn.begin` has already appended the user's
        message and the workflow is running. Replaying over REST claims a
        different idempotency slot, appends that message again, and runs the
        workflow a second time.
        """
        turn = signed_in()
        turn.answers(
            json.dumps({"event": "token", "data": "half an ans"}),
            close_after=True,
        )

        turn.send()

        assert turn.banner() == "Connection closed"
        assert turn.rest_calls == [], (
            "a turn that had already produced a token was replayed over "
            f"REST, duplicating the user message and the workflow: "
            f"{turn.rest_calls}"
        )

    def test_a_close_with_no_answer_is_not_retried(self, signed_in):
        """Silence after the send is ambiguous, not proof of nothing.

        The request is on the socket, so the server may have begun the turn
        and said nothing yet. Ambiguity fails closed.
        """
        turn = signed_in()
        turn.answers(close_after=True)

        turn.send()

        assert turn.banner() == "Connection closed"
        assert turn.rest_calls == [], (
            "a socket that died after the request was sent was replayed over "
            f"REST, although the turn may already have run: {turn.rest_calls}"
        )


class TestAFailureBeforeTheRequestStillFallsBack:
    def test_a_socket_that_never_opens_falls_back_to_rest(self, signed_in):
        """The other direction, and the reason the five above mean anything.

        If fallback were simply removed, every witness above would pass while
        the feature they bound was gone. It is also the positive control for
        the REST counter: an empty `rest_calls` is evidence only once the same
        counter has been shown to record something.
        """
        turn = signed_in(fails_to_open=True)

        turn.send()

        turn.page.wait_for_selector(
            ".message.assistant:not(.streaming)", timeout=60000
        )
        assert turn.rest_calls, (
            "a socket that never opened did not fall back to REST, so the "
            "fix withdrew the one retry that is still safe"
        )
