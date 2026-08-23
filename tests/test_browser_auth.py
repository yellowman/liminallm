"""The browser holds one credential, and it is the short-lived one.

SPEC §17.10: the SPA keeps the access token; `session_id` and `refresh_token`
ride as `HttpOnly` cookies the page cannot read. A durable credential in
`sessionStorage` is readable by any script that reaches the page, and it
outlives the access token it was meant to replace — which is the whole reason
the cookie exists.

These run a real browser against the real server because that is the only
place the property is observable: `TestClient` has no script context, no
`HttpOnly` enforcement, and no same-origin cookie policy. The previous defects
at this seam were all of that shape.
"""

from __future__ import annotations

import uuid

import pytest

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


@pytest.fixture
def page(browser):
    context = browser.new_context()
    opened = context.new_page()
    try:
        yield opened
    finally:
        context.close()


def _account(server) -> tuple[str, str]:
    """A real account, made through the API the SPA talks to."""
    import httpx

    email = f"br_{uuid.uuid4().hex[:8]}@example.com"
    resp = httpx.post(
        f"{server.base_url}/v1/auth/signup",
        json={"email": email, "password": PASSWORD},
        timeout=30,
    )
    assert resp.status_code == 201, resp.text
    return email, PASSWORD


def _log_in(page, server, email, password) -> None:
    page.goto(f"{server.base_url}/", wait_until="domcontentloaded")
    page.fill("#email", email)
    page.fill("#password", password)
    page.click("#auth-form button[type=submit]")
    # The app is signed in once it holds an access token; waiting on the token
    # rather than on a pixel keeps this about auth.
    page.wait_for_function(
        "() => !!sessionStorage.getItem('liminal.accessToken')", timeout=15000
    )


def _session_storage(page) -> dict:
    return page.evaluate(
        "() => Object.fromEntries("
        "  Object.keys(sessionStorage).map(k => [k, sessionStorage.getItem(k)])"
        ")"
    )


class TestTheBrowserHoldsOneCredential:
    def test_login_leaves_only_the_access_token_where_scripts_can_read(
        self, page, server
    ):
        """The access token is short-lived on purpose; the others are not.

        A `refresh_token` in `sessionStorage` is a durable credential any
        script on the page can read, which is exactly what moving it into an
        `HttpOnly` cookie was for — the cookie is set either way, so keeping
        the copy only removes the protection.

        `session_id` is the same argument: it is the handle MFA and the
        WebSocket authenticate with, and the server already sets it as a
        cookie the page cannot read.
        """
        email, password = _account(server)
        _log_in(page, server, email, password)

        # The login really worked: a protected call, made by the app itself.
        conversations = page.evaluate(
            "async () => (await fetch('/v1/conversations?limit=1', {"
            "  headers: {"
            "    'Authorization': 'Bearer ' + sessionStorage.getItem('liminal.accessToken')"
            "  }"
            "})).status"
        )
        assert conversations == 200, conversations

        stored = _session_storage(page)
        assert stored.get("liminal.accessToken"), stored
        assert "liminal.refreshToken" not in stored, (
            "the refresh token is readable by any script on the page, and it "
            "outlives the access token it was meant to replace"
        )
        assert "liminal.sessionId" not in stored, (
            "the session id is readable by any script on the page, although "
            "the server already sets it as an HttpOnly cookie"
        )

    def test_the_credentials_the_page_cannot_read_are_the_ones_that_matter(
        self, page, server
    ):
        """The cookies exist, and are the reason the copies are not needed."""
        email, password = _account(server)
        _log_in(page, server, email, password)

        cookies = {c["name"]: c for c in page.context.cookies()}
        assert cookies["session_id"]["httpOnly"] is True, cookies["session_id"]
        assert cookies["refresh_token"]["httpOnly"] is True, cookies["refresh_token"]
        # The CSRF cookie is deliberately readable: the page has to echo it.
        assert cookies["csrf_token"]["httpOnly"] is False, cookies["csrf_token"]

        visible = page.evaluate("() => document.cookie")
        assert "session_id" not in visible and "refresh_token" not in visible, visible


class TestTheRefreshIsTheCookiesJob:
    """An expired access token recovers without the page holding a secret.

    The whole point of moving the refresh token out of reach is that the
    recovery still works. So the witness is the lifecycle, not the storage
    snapshot: sign in, break the access token, make the app do something, and
    require that it recovered — using a credential it could not read, sending
    nothing durable of its own, and exactly once.
    """

    def test_an_expired_token_recovers_on_the_cookie_alone(self, page, server):
        email, password = _account(server)
        _log_in(page, server, email, password)

        posts = []
        page.on(
            "request",
            lambda req: posts.append((req.url, req.post_data))
            if "/v1/auth/refresh" in req.url
            else None,
        )
        refreshes = []
        page.on(
            "response",
            lambda resp: refreshes.append(resp.status)
            if "/v1/auth/refresh" in resp.url
            else None,
        )

        before = page.evaluate("() => sessionStorage.getItem('liminal.accessToken')")
        # A token the server will refuse, in the shape the app stores.
        page.evaluate(
            "() => sessionStorage.setItem('liminal.accessToken', 'not.a.token')"
        )
        page.evaluate("() => { state.accessToken = 'not.a.token'; }")

        # A real application operation, through the app's own request layer.
        page.evaluate("async () => { await fetchConversations(); }")

        assert refreshes == [200], (
            f"expected exactly one successful cookie refresh, saw {refreshes}"
        )
        assert len(posts) == 1, posts
        _url, body = posts[0]
        assert "refresh_token" not in (body or ""), (
            f"the page sent a refresh token it should not be able to read: {body}"
        )
        assert "tenant_id" not in (body or ""), (
            f"the page named its own tenant, which the server derives: {body}"
        )

        after = page.evaluate("() => sessionStorage.getItem('liminal.accessToken')")
        assert after and after != "not.a.token", "the access token was not replaced"
        assert after != before, "the refresh returned the same access token"

        # The operation that hit 401 completed: the app holds a usable token
        # again, and the protected call it retried now answers.
        status = page.evaluate(
            "async () => (await fetch('/v1/conversations?limit=1', {"
            "  headers: { 'Authorization': 'Bearer ' + state.accessToken }"
            "})).status"
        )
        assert status == 200, status

        stored = _session_storage(page)
        assert "liminal.refreshToken" not in stored, stored
        assert "liminal.sessionId" not in stored, stored


class TestTheAdminConsoleHoldsTheSameLine:
    """A second page with its own copy of the rule is a second place to break.

    The console has its own `persistAuth` and its own storage keys, so the
    chat SPA's witness says nothing about it — measured, re-persisting the
    refresh token here left the chat tests green. Both pages are the same
    origin and the same cookies; the rule has to hold on both.
    """

    def test_an_admin_login_leaves_only_the_access_token(self, page, server):
        from liminallm.service.runtime import get_runtime

        email, password = _account(server)
        user = get_runtime().store.get_user_by_email(email)
        get_runtime().store.update_user_role(user.id, role="admin")

        # `/admin`, not `/static/admin.html`: the direct path is blocked on
        # purpose, so using it would test a route nobody has.
        page.goto(f"{server.base_url}/admin", wait_until="domcontentloaded")
        page.fill("#admin-email", email)
        page.fill("#admin-password", password)
        page.click("#admin-auth-form button[type=submit]")
        page.wait_for_function(
            "() => !!sessionStorage.getItem('liminal.accessToken')", timeout=15000
        )

        # The console really opened: an admin-only call, with its token.
        status = page.evaluate(
            "async () => (await fetch('/v1/admin/users?limit=1', {"
            "  headers: {"
            "    'Authorization': 'Bearer ' + sessionStorage.getItem('liminal.accessToken')"
            "  }"
            "})).status"
        )
        assert status == 200, status

        stored = _session_storage(page)
        assert stored.get("liminal.accessToken"), stored
        assert "liminal.refreshToken" not in stored, stored
        assert "liminal.sessionId" not in stored, stored


class TestSigningOutTakesWhatAnOlderSessionLeft:
    """The keys are no longer written, which is not the same as gone.

    A tab open across the change, or a returning user whose `sessionStorage`
    predates it, still holds the old credentials. Signing out is when they
    should go, so logout clears them even though nothing writes them.
    """

    def test_logout_clears_credentials_an_older_version_persisted(
        self, page, server
    ):
        email, password = _account(server)
        _log_in(page, server, email, password)

        # Exactly what a session from before this change would be holding.
        page.evaluate(
            "() => {"
            "  sessionStorage.setItem('liminal.refreshToken', 'left-behind');"
            "  sessionStorage.setItem('liminal.sessionId', 'left-behind');"
            "}"
        )
        page.evaluate("() => state.resetAuth()")

        stored = _session_storage(page)
        assert "liminal.refreshToken" not in stored, (
            f"signing out left a durable credential behind: {stored}"
        )
        assert "liminal.sessionId" not in stored, stored
        assert "liminal.accessToken" not in stored, stored
