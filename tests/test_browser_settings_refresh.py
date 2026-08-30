"""Opening the Settings tab reloads what the panel shows.

Every other tab reloads its data when it is opened. Settings did not, and
its fields are fetched once at start-up, so a single failed request left
the panel showing an error for the rest of the session with no way back
except a full page reload.

That is not a hypothetical. A privilege change invalidates the access
token that was minted before it, which is correct, and the page recovers
by refreshing. But any request already in flight fails first, and the
screenshot checked into the README caught exactly that: "Unable to check"
next to an empty email address, on an account whose email was readable
the whole time.

The test forces the same shape deterministically by failing the first
profile request and then opening the tab.
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


class TestTheSettingsTabRecoversFromAFailedLoad:
    def test_opening_settings_refetches_the_profile(self, browser, server):
        email = f"set_{uuid.uuid4().hex[:8]}@example.com"

        import httpx

        resp = httpx.post(
            f"{server.base_url}/v1/auth/signup",
            json={"email": email, "password": PASSWORD},
            timeout=30,
        )
        assert resp.status_code == 201, resp.text

        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()

        # Fail the profile request for the whole of start-up, then let it
        # through. One aborted attempt proves nothing: the fetch helper
        # retries, so a single failure heals itself and never reaches the
        # screen. The real outage outlives those retries, which is why the
        # block is lifted only once the page has settled.
        state = {"block": True, "attempts": 0}

        def profile_route(route):
            state["attempts"] += 1
            if state["block"]:
                route.abort("failed")
                return
            route.continue_()

        page.route("**/v1/me", profile_route)

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
            # The fetch helper makes four attempts with a growing backoff,
            # so wait for the outcome rather than for a duration.
            page.wait_for_function(
                """() => {
                    const el = document.getElementById(
                        'setting-email-verified');
                    return el && el.textContent.trim() !== 'Loading...';
                }""",
                timeout=30000,
            )
            assert state["attempts"] > 0, (
                "the profile request was never made, so this test never "
                "reproduced the failure it is about"
            )

            # The failure is on screen: this is the state the README caught.
            stale = (page.text_content("#setting-email-verified") or "").strip()
            assert stale == "Unable to check", (
                f"expected the failed load to show, got {stale!r}; the test "
                "has stopped reproducing the condition it exists for"
            )

            # The outage is over. Nothing else reloads the panel, so whether
            # the screen recovers is entirely down to opening the tab.
            state["block"] = False
            page.click("#main-tabs .tab-btn[data-tab='settings-tab']")
            page.wait_for_selector("#settings-tab.active", state="visible")
            page.wait_for_timeout(2500)

            status = (page.text_content("#setting-email-verified") or "").strip()
            address = (page.text_content("#setting-email-address") or "").strip()

            assert status != "Unable to check", (
                "the Settings tab still shows the failed start-up request; "
                "opening the panel has to reload what it displays"
            )
            assert status == "Not verified", f"unexpected email status: {status!r}"
            assert address == email, (
                f"the email address never loaded: {address!r}; the panel is "
                "showing the placeholder from the request that failed"
            )
        finally:
            context.close()

    def test_an_unsaved_preference_survives_leaving_the_tab(
        self, browser, server
    ):
        """Reloading the panel must not reach the editable preferences.

        The status rows above are display-only, so reloading them can only
        replace a stale value with a fresh one. The preference selects are
        the user's own uncommitted input, and reloading those on every visit
        would silently discard an edit made before stepping away.
        """
        email = f"pref_{uuid.uuid4().hex[:8]}@example.com"

        import httpx

        resp = httpx.post(
            f"{server.base_url}/v1/auth/signup",
            json={"email": email, "password": PASSWORD},
            timeout=30,
        )
        assert resp.status_code == 201, resp.text

        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()
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
            page.wait_for_timeout(1500)

            page.click("#main-tabs .tab-btn[data-tab='settings-tab']")
            page.wait_for_selector("#settings-tab.active", state="visible")
            page.select_option("#setting-locale", "ja-JP")

            # Step away without saving, then come back.
            page.click("#main-tabs .tab-btn[data-tab='chat-tab']")
            page.wait_for_selector("#chat-tab.active", state="visible")
            page.click("#main-tabs .tab-btn[data-tab='settings-tab']")
            page.wait_for_selector("#settings-tab.active", state="visible")
            page.wait_for_timeout(2500)

            kept = page.input_value("#setting-locale")
            assert kept == "ja-JP", (
                f"the unsaved locale went back to {kept!r}; reopening the "
                "tab overwrote the user's own edit"
            )
        finally:
            context.close()
