"""A stream that fails leaves a finished message, and says why.

The server sends a streaming failure as an `error` event (SPEC §13.7). The
client has a branch built for exactly that - it finalizes the partial bubble
with "Error occurred" and surfaces the server's own message. It also has a
legacy branch for the `{status, error}` envelope, and that one cleans up and
rejects without finalizing anything.

While the route answered a mid-turn failure with an envelope, a streamed reply
that failed took the legacy branch: the half-written bubble kept its
`streaming` class, and the text said an internal error had occurred when the
real cause was that the user had deleted the chat in another tab.

This is the visible half of that fix, and it is a browser test because the
residue is a DOM state - which branch ran is not observable from the wire
alone.
"""

from __future__ import annotations

import uuid

import pytest

from liminallm.service.runtime import get_runtime
from tests.browser import LiveServer, chromium_executable

pytest.importorskip(
    "playwright",
    reason="the browser lane needs the dev extra: uv pip install playwright",
)

pytestmark = pytest.mark.browser

PASSWORD = "TestPassword123!"


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


class TestAFailedStreamFinishesItsMessage:
    def test_deleting_the_chat_mid_stream_finalizes_the_bubble(
        self, browser, server
    ):
        import httpx

        email = f"wsb_{uuid.uuid4().hex[:8]}@example.com"
        signup = httpx.post(
            f"{server.base_url}/v1/auth/signup",
            json={"email": email, "password": PASSWORD},
            timeout=30,
        )
        assert signup.status_code == 201, signup.text
        token = signup.json()["data"]["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()
        runtime = get_runtime()
        real_stream = runtime.llm.generate_stream
        real_needs_tools = runtime.workflow._turn_needs_tools
        state: dict = {}

        # Take the plain-chat workflow, not the tool-agent one. This test is
        # about the socket's terminal frame, not about tool routing, and the
        # attachment agent's `files` node reaches the network: on a runner that
        # refuses that egress it exhausts its retries and the turn ends with
        # the workflow's own error event, before `chat_turn.finish` is ever
        # reached. That error is already event-shaped, so the bubble would be
        # finalized by the very path this test exists to prove is not taken -
        # the witness would pass with the defect present. Measured on CI:
        # `workflow_node_retries_exhausted` on `files`, then a 30s timeout here.
        runtime.workflow._turn_needs_tools = lambda *a, **k: False

        try:
            page.goto(f"{server.base_url}/", wait_until="domcontentloaded")
            page.fill("#email", email)
            page.fill("#password", PASSWORD)
            page.click("#auth-form button[type=submit]")
            page.wait_for_function(
                "() => !!sessionStorage.getItem('liminal.accessToken')",
                timeout=30000,
            )
            page.wait_for_selector("#main-tabs", state="visible")

            # One ordinary turn first, so there is a conversation to delete and
            # the socket has a live chat to stream into.
            page.fill("#message-input", "the question asked first")
            page.click("#chat-form button[type=submit]")
            page.wait_for_selector(
                ".message.assistant:not(.streaming)", timeout=60000
            )

            listed = httpx.get(
                f"{server.base_url}/v1/conversations", headers=headers, timeout=30
            )
            items = listed.json()["data"]["items"]
            assert items, listed.text
            conversation_id = items[0]["id"]

            # The delete lands from inside the model call of the next turn:
            # after it has begun, before the reply can be persisted. The hook
            # does one thing and records one value.
            def _delete_then_stream(*args, **kwargs):
                if "deleted" not in state:
                    state["deleted"] = httpx.delete(
                        f"{server.base_url}/v1/conversations/{conversation_id}",
                        headers=headers,
                        timeout=30,
                    ).status_code
                return real_stream(*args, **kwargs)

            runtime.llm.generate_stream = _delete_then_stream

            page.fill("#message-input", "the doomed question")
            page.click("#chat-form button[type=submit]")

            # The outcome, not a duration: the partial bubble is finished.
            page.wait_for_function(
                """() => {
                    const streaming = document.querySelector(
                        '.message.assistant.streaming');
                    const metas = Array.from(
                        document.querySelectorAll('.message.assistant .meta'));
                    return !streaming && metas.some(
                        (m) => m.textContent.includes('Error occurred'));
                }""",
                timeout=60000,
            )

            assert state.get("deleted") == 200, (
                f"the delete never landed: {state.get('deleted')}"
            )
            assert page.query_selector(".message.assistant.streaming") is None, (
                "the half-written reply is still marked as streaming, so the "
                "client took the legacy envelope branch"
            )

            # The truthful cause reached the screen, not "an internal error".
            # `showStatus(err.message, true)` puts it in the error banner, and
            # that runs after the bubble is finalized, so it is waited for
            # separately rather than assumed to have arrived with it.
            page.wait_for_function(
                """() => {
                    const el = document.getElementById('error-banner');
                    return el && el.textContent.trim().length > 0;
                }""",
                timeout=30000,
            )
            banner = (page.text_content("#error-banner") or "").strip()
            assert banner == "conversation not found", (
                f"the banner never named the real cause: {banner!r}"
            )
        finally:
            runtime.llm.generate_stream = real_stream
            runtime.workflow._turn_needs_tools = real_needs_tools
            context.close()
