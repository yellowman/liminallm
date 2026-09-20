"""The settings index has to land a reader where it says it will.

Two defects a review found and this file pins, both of which need a real
layout engine to see - one is about sticky positioning and the other about
where a page stops scrolling, and neither is visible from the stylesheet or
the script.

* A jump link scrolls its section to the top of the viewport, which is
  behind `.topbar`. Measured before the fix: 34px of a 48px bar sat over the
  `.section-band` the jump exists to reveal, on every section that had room
  to scroll. Both pages carrying this pattern wear the same sticky bar.
* The mark follows the last section whose top has crossed a line near the
  top of the viewport. The last sections on a page are short, so the scroll
  runs out before their tops ever reach it: at the foot of Settings the
  index said "Users" while Config patches filled the screen.
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
DESKTOP = {"width": 1440, "height": 900}


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


def _signed_in_page(browser, server, *, admin: bool = False):
    import httpx

    email = f"nav_{uuid.uuid4().hex[:8]}@example.com"
    resp = httpx.post(
        f"{server.base_url}/v1/auth/signup",
        json={"email": email, "password": PASSWORD},
        timeout=30,
    )
    assert resp.status_code == 201, resp.text
    if admin:
        from liminallm.service.runtime import get_runtime

        # Before the browser signs in: a role change invalidates a token
        # minted before it, so promoting afterwards breaks the session the
        # test is about to drive.
        get_runtime().store.update_user_role(
            resp.json()["data"]["user_id"], role="admin"
        )

    context = browser.new_context(viewport=DESKTOP)
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


def _open_settings(page):
    page.click("#main-tabs .rail-btn[data-tab='settings-tab']")
    page.wait_for_selector("#settings-tab.active", state="visible")
    page.wait_for_timeout(900)


def _bar_bottom(page) -> int:
    return page.eval_on_selector(
        ".topbar", "el => Math.round(el.getBoundingClientRect().bottom)"
    )


class TestAJumpLandsWhereItPointed:
    def test_no_section_lands_behind_the_sticky_bar(self, browser, server):
        """The band is the whole point of the jump, so it has to be the
        thing the reader sees when they arrive."""
        context, page = _signed_in_page(browser, server)
        try:
            _open_settings(page)
            floor = _bar_bottom(page)
            assert floor > 0, "no sticky bar, so this test measures nothing"

            hidden = []
            hrefs = page.eval_on_selector_all(
                "#settings-index a:not(.hidden)",
                "els => els.map(e => e.getAttribute('href'))",
            )
            assert len(hrefs) >= 9, hrefs
            for href in hrefs:
                page.click(f"#settings-index a[href='{href}']")
                page.wait_for_timeout(400)
                measured = page.eval_on_selector(
                    f"{href} .section-band",
                    """el => {
                        const r = el.getBoundingClientRect();
                        return {top: Math.round(r.top),
                                bottom: Math.round(r.bottom)};
                    }""",
                )
                # A section at the foot of the page cannot scroll to the
                # line, and lands wherever the document ends - which is
                # below the bar, not behind it. Only a section the browser
                # actually scrolled to is being measured here.
                if measured["top"] < floor and measured["bottom"] > 0:
                    hidden.append((href, floor - measured["top"]))

            assert not hidden, (
                "these sections land behind the "
                f"{floor}px bar, so the jump hides the title it aimed at: "
                + ", ".join(f"{href} by {px}px" for href, px in hidden)
            )
        finally:
            context.close()

    def test_the_admin_console_index_lands_clear_of_the_same_bar(
        self, browser, server
    ):
        """The other page with this pattern, and the same sticky bar. A fix
        that stopped at the reported page would leave this one behind."""
        context, page = _signed_in_page(browser, server, admin=True)
        try:
            page.goto(f"{server.base_url}/admin", wait_until="domcontentloaded")
            page.wait_for_selector("#settings-form .setting-group", timeout=30000)
            page.wait_for_timeout(900)
            floor = _bar_bottom(page)
            assert floor > 0

            hrefs = page.eval_on_selector_all(
                "#settings-nav a", "els => els.map(e => e.getAttribute('href'))"
            )
            assert len(hrefs) >= 5, hrefs
            hidden = []
            # The first few have room to scroll; the tail runs out of page,
            # which the loop above explains.
            for href in hrefs[:5]:
                page.click(f"#settings-nav a[href='{href}']")
                page.wait_for_timeout(400)
                measured = page.eval_on_selector(
                    f"{href} .section-band",
                    """el => {
                        const r = el.getBoundingClientRect();
                        return {top: Math.round(r.top),
                                bottom: Math.round(r.bottom)};
                    }""",
                )
                if measured["top"] < floor and measured["bottom"] > 0:
                    hidden.append((href, floor - measured["top"]))
            assert not hidden, (
                f"these groups land behind the {floor}px bar: "
                + ", ".join(f"{href} by {px}px" for href, px in hidden)
            )
        finally:
            context.close()


    @pytest.mark.parametrize("admin_console", [False, True], ids=["app", "console"])
    def test_the_index_itself_sticks_below_the_bar(
        self, browser, server, admin_console
    ):
        """The third instance of the same shape. The index is sticky, the
        bar is sticky above it, and an offset that does not account for the
        bar puts the index's first links behind it once the page scrolls."""
        context, page = _signed_in_page(browser, server, admin=True)
        try:
            if admin_console:
                page.goto(f"{server.base_url}/admin", wait_until="domcontentloaded")
                page.wait_for_selector(
                    "#settings-form .setting-group", timeout=30000
                )
                page.wait_for_timeout(900)
                index = "#settings-nav"
            else:
                _open_settings(page)
                index = "#settings-index"

            page.evaluate("() => window.scrollTo(0, 1200)")
            page.wait_for_timeout(500)
            floor = _bar_bottom(page)
            measured = page.eval_on_selector(
                index,
                """el => {
                    const r = el.getBoundingClientRect();
                    return {top: Math.round(r.top),
                            sticking: getComputedStyle(el).position};
                }""",
            )
            assert measured["sticking"] == "sticky", measured
            assert measured["top"] >= floor, (
                f"the index sits at {measured['top']}px under a bar whose "
                f"bottom is {floor}px, so its first links are behind it"
            )
        finally:
            context.close()


class TestTheMarkFollowsTheReaderToTheEnd:
    def test_the_foot_of_the_page_marks_a_section_that_is_on_screen(
        self, browser, server
    ):
        """At the end of the scroll the index named a section that had left
        the screen three sections earlier."""
        context, page = _signed_in_page(browser, server, admin=True)
        try:
            _open_settings(page)
            page.evaluate("() => window.scrollTo(0, document.body.scrollHeight)")
            page.wait_for_timeout(700)

            state = page.evaluate(
                """() => {
                  const groups = [...document.querySelectorAll(
                    '#settings-tab .setting-group')]
                    .filter((g) => g.getClientRects().length);
                  const last = groups[groups.length - 1];
                  const marked = [...document.querySelectorAll(
                    '#settings-index a.current')]
                    .map((a) => a.getAttribute('href').slice(1));
                  return {
                    marked,
                    last: last.id,
                    lastOnScreen: last.getBoundingClientRect().top
                                  < window.innerHeight,
                    atEnd: Math.abs(
                      window.scrollY + window.innerHeight
                      - document.documentElement.scrollHeight) < 3,
                  };
                }"""
            )
            # The positive control for the measurement itself: if the page
            # did not reach its end, or the last section is not on screen,
            # this test is not looking at the state it claims to.
            assert state["atEnd"], "the page did not scroll to its end"
            assert state["lastOnScreen"], (
                "the last section is not on screen, so nothing here is about "
                "what the reader is looking at"
            )
            assert state["marked"] == [state["last"]], (
                f"at the foot of the page the index marks {state['marked']} "
                f"while {state['last']} is the section on screen"
            )
        finally:
            context.close()

    def test_a_section_mid_page_is_still_the_one_marked(self, browser, server):
        """The control for the fix above: teaching the mark about the end of
        the page must not stop it tracking the middle."""
        context, page = _signed_in_page(browser, server)
        try:
            _open_settings(page)
            page.click("#settings-index a[href='#settings-api-keys']")
            page.wait_for_timeout(600)
            marked = page.eval_on_selector_all(
                "#settings-index a.current",
                "els => els.map(e => e.getAttribute('href'))",
            )
            assert marked == ["#settings-api-keys"], marked
        finally:
            context.close()
