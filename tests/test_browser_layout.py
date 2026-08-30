"""The chat chrome stays one row per band, at every width it claims to support.

The main navigation lives inside a fixed-height top bar, so it must never
wrap: a second line has nowhere to go and overlaps the bar's own border. The
conversation title shares a row with the thread controls on a desktop width,
which is what stopped that row costing the chat column twice. Neither
property is observable without a real layout engine - they are about wrapping
and overflow, which only a browser computes - so they live in the browser
lane.

A stale `flex-wrap: wrap` override survived the move into the top bar and
defeated the first property below 1080px, which is why these assert geometry
at two widths rather than one.
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
#: A laptop and a phone, the two shapes the stylesheet has rules for.
DESKTOP = {"width": 1440, "height": 900}
PHONE = {"width": 390, "height": 844}


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


def _signed_in_page(browser, server, viewport):
    """A real account on a real page, sized to `viewport`."""
    import httpx

    email = f"lay_{uuid.uuid4().hex[:8]}@example.com"
    resp = httpx.post(
        f"{server.base_url}/v1/auth/signup",
        json={"email": email, "password": PASSWORD},
        timeout=30,
    )
    assert resp.status_code == 201, resp.text

    context = browser.new_context(viewport=viewport)
    page = context.new_page()
    page.goto(f"{server.base_url}/", wait_until="domcontentloaded")
    page.fill("#email", email)
    page.fill("#password", PASSWORD)
    page.click("#auth-form button[type=submit]")
    page.wait_for_function(
        "() => !!sessionStorage.getItem('liminal.accessToken')", timeout=30000
    )
    page.wait_for_selector("#main-tabs", state="visible")
    page.wait_for_timeout(700)
    return context, page


def _tab_rows(page) -> int:
    """How many distinct rows the tab buttons occupy."""
    return page.evaluate(
        """() => new Set(
            [...document.querySelectorAll('#main-tabs .tab-btn')]
                .map(el => Math.round(el.getBoundingClientRect().top))
        ).size"""
    )


class TestTheTopBarNavigationStaysOnOneLine:
    """Eight tabs in a bar with room for one line of them."""

    @pytest.mark.parametrize("viewport", [DESKTOP, PHONE], ids=["desktop", "phone"])
    def test_the_tabs_never_wrap_inside_the_fixed_height_bar(
        self, browser, server, viewport
    ):
        context, page = _signed_in_page(browser, server, viewport)
        try:
            rows = _tab_rows(page)
            assert rows == 1, (
                f"the tab buttons occupy {rows} rows at "
                f"{viewport['width']}px; the bar has height for one, so the "
                "extra line overlaps its border instead of scrolling"
            )
            fits = page.evaluate(
                """() => {
                    const nav = document.querySelector('#main-tabs');
                    const bar = nav.closest('header');
                    return nav.getBoundingClientRect().height
                        <= bar.getBoundingClientRect().height + 1;
                }"""
            )
            assert fits, "the navigation is taller than the bar containing it"
        finally:
            context.close()

    @pytest.mark.parametrize("viewport", [DESKTOP, PHONE], ids=["desktop", "phone"])
    def test_the_page_never_scrolls_sideways(self, browser, server, viewport):
        """Whatever does not fit scrolls inside its own row, not the page."""
        context, page = _signed_in_page(browser, server, viewport)
        try:
            overflow = page.evaluate(
                "() => document.documentElement.scrollWidth - window.innerWidth"
            )
            assert overflow <= 1, (
                f"the page scrolls {overflow}px sideways at "
                f"{viewport['width']}px: a control row is pushing the "
                "document wider than the viewport"
            )
        finally:
            context.close()


class TestTheConversationHeaderSpendsOneRowOnDesktop:
    def test_the_title_and_controls_share_a_row(self, browser, server):
        """The reason the header stopped costing the chat column twice.

        The title shrinks and truncates; the controls keep their size.
        """
        context, page = _signed_in_page(browser, server, DESKTOP)
        try:
            same_row = page.evaluate(
                """() => {
                    const b = document.querySelector(
                        '.conversation-header .badge');
                    const p = document.querySelector(
                        '.conversation-header .pill-row');
                    if (!b || !p) return null;
                    return Math.abs(
                        b.getBoundingClientRect().top
                        - p.getBoundingClientRect().top) < 4;
                }"""
            )
            assert same_row is True, (
                "the conversation title wrapped onto its own line, which is "
                "the row the layout change removed"
            )
        finally:
            context.close()

    def test_the_first_message_starts_high_on_the_page(self, browser, server):
        """The whole point of the change, stated as a number.

        Before the chrome was consolidated the message list started 207px
        down an empty thread. The bound is deliberately loose: it fails on a
        band coming back, not on a few pixels of padding.
        """
        context, page = _signed_in_page(browser, server, DESKTOP)
        try:
            top = page.evaluate(
                "() => Math.round("
                "document.querySelector('#messages').getBoundingClientRect().top)"
            )
            assert top <= 170, (
                f"the message list starts {top}px down; the chat column is "
                "paying for a band of chrome again"
            )
        finally:
            context.close()
