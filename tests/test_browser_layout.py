"""The shell's three bands hold their shape at every width it claims.

Navigation lives in a 48px rail down the left edge rather than a strip
across the top, so the properties worth pinning changed with it. A strip
could wrap onto a second line and overlap its own border; a rail cannot
wrap, but it can outgrow the viewport it is pinned to, and the two columns
beside it can push the document wider than the screen.

What survives from the strip era is the reason the rail exists: the chat
column pays for vertical chrome and not for horizontal, so the message list
must still start high on the page.

None of this is observable without a real layout engine - it is about
wrapping, overflow and sticky columns, which only a browser computes - so it
lives in the browser lane.
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

#: The sections whose list already drove a detail view, and so have a pane.
WITH_PANE = ["chat-tab", "notes-tab", "contexts-tab", "artifacts-tab", "tools-tab"]
#: The sections that do not. A pane here would need navigation inventing to
#: fill it: Files' list is a download list, and the other two are forms.
WITHOUT_PANE = ["files-tab", "insights-tab", "settings-tab"]


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


class TestTheRailIsOneColumnThatFits:
    """Eight destinations down a 48px edge."""

    @pytest.mark.parametrize("viewport", [DESKTOP, PHONE], ids=["desktop", "phone"])
    def test_the_rail_is_a_single_column(self, browser, server, viewport):
        """A second column would be the rail's version of a wrapped strip."""
        context, page = _signed_in_page(browser, server, viewport)
        try:
            # Visible buttons only: the admin link is in the markup for
            # everyone and displayed for admins, and a hidden element
            # reports a zero rect that would read as a second column.
            columns = page.evaluate(
                """() => new Set(
                    [...document.querySelectorAll('#main-tabs .rail-btn')]
                        .filter(el => el.getBoundingClientRect().width > 0)
                        .map(el => Math.round(el.getBoundingClientRect().left))
                ).size"""
            )
            assert columns == 1, (
                f"the rail's buttons occupy {columns} columns at "
                f"{viewport['width']}px; it has width for one"
            )
        finally:
            context.close()

    @pytest.mark.parametrize("viewport", [DESKTOP, PHONE], ids=["desktop", "phone"])
    def test_every_destination_is_reachable_without_scrolling_the_rail(
        self, browser, server, viewport
    ):
        """The rail is pinned to the viewport, so anything past its bottom
        edge is unreachable rather than merely below the fold."""
        context, page = _signed_in_page(browser, server, viewport)
        try:
            escaped = page.evaluate(
                """() => {
                    const rail = document.querySelector('#main-tabs');
                    const box = rail.getBoundingClientRect();
                    return [...rail.querySelectorAll('.rail-btn')]
                        .filter(el => {
                            const r = el.getBoundingClientRect();
                            return r.bottom > box.bottom + 1 || r.top < box.top - 1;
                        })
                        .map(el => el.getAttribute('aria-label'));
                }"""
            )
            assert escaped == [], (
                f"{escaped} sit outside the rail at {viewport['height']}px tall, "
                "so they cannot be clicked"
            )
        finally:
            context.close()

    @pytest.mark.parametrize("viewport", [DESKTOP, PHONE], ids=["desktop", "phone"])
    def test_the_page_never_scrolls_sideways(self, browser, server, viewport):
        """Three bands must still add up to the viewport, not more."""
        context, page = _signed_in_page(browser, server, viewport)
        try:
            overflow = page.evaluate(
                "() => document.documentElement.scrollWidth - window.innerWidth"
            )
            assert overflow <= 1, (
                f"the shell is {overflow}px wider than the viewport at "
                f"{viewport['width']}px"
            )
        finally:
            context.close()


class TestThePaneAppearsWhereNavigationAlreadyExisted:
    """The pane holds a list that drives a detail view, or it is not there.

    An empty 240px column is worse than no column: it spends the width and
    answers nothing. Files, Insights and Settings have no such list, so the
    middle band collapses for them.
    """

    def test_sections_with_a_list_show_a_pane(self, browser, server):
        context, page = _signed_in_page(browser, server, DESKTOP)
        try:
            for tab in WITH_PANE:
                page.click(f'.rail-btn[data-tab="{tab}"]')
                page.wait_for_timeout(120)
                width = page.evaluate(
                    "() => document.querySelector('.context-pane')"
                    ".getBoundingClientRect().width"
                )
                assert width > 100, f"{tab} lost its pane (width {width})"
        finally:
            context.close()

    def test_sections_without_one_collapse_the_band(self, browser, server):
        context, page = _signed_in_page(browser, server, DESKTOP)
        try:
            for tab in WITHOUT_PANE:
                page.click(f'.rail-btn[data-tab="{tab}"]')
                page.wait_for_timeout(120)
                width = page.evaluate(
                    "() => document.querySelector('.context-pane')"
                    ".getBoundingClientRect().width"
                )
                assert width == 0, (
                    f"{tab} shows a {width}px pane with nothing in it"
                )
                # And no control offering to hide it. The click also wrote
                # the remembered preference, so an idle click on Files took
                # Chat's conversations away on the next visit.
                toggle = page.evaluate(
                    "() => document.querySelector('.pane-toggle')"
                    ".getBoundingClientRect().width"
                )
                assert toggle == 0, (
                    f"{tab} offers to hide a list it does not have"
                )
        finally:
            context.close()

    def test_hiding_the_pane_keeps_the_rail(self, browser, server):
        """The common case gives back the list, not the way to other sections."""
        context, page = _signed_in_page(browser, server, DESKTOP)
        try:
            page.click("#pane-toggle")
            page.wait_for_timeout(150)
            state = page.evaluate(
                """() => ({
                    pane: document.querySelector('.context-pane')
                        .getBoundingClientRect().width,
                    rail: document.querySelector('.app-rail')
                        .getBoundingClientRect().width,
                })"""
            )
            assert state["pane"] == 0, "the toggle left the pane on screen"
            assert state["rail"] > 0, (
                "hiding the list also took away the way back to the other "
                "sections; that is the shift-click case, not the plain one"
            )
        finally:
            context.close()


class TestTheChatColumnStillStartsHigh:
    """The reason the strip moved to the edge in the first place."""

    def test_the_title_and_controls_share_a_row(self, browser, server):
        """The conversation's name and the bar's actions, on one line.

        They used to be siblings inside `.conversation-header`; the actions
        now sit in the bar's own group beside it. The property is the same
        one either way - the bar must not wrap into a second band - so it is
        measured between the two elements that actually carry them.
        """
        context, page = _signed_in_page(browser, server, DESKTOP)
        try:
            same_row = page.evaluate(
                """() => {
                    const b = document.querySelector(
                        '.topbar .conversation-header .badge');
                    const a = document.querySelector('.topbar .bar-actions');
                    if (!b || !a) return null;
                    return Math.abs(
                        b.getBoundingClientRect().top
                        - a.getBoundingClientRect().top) < 8;
                }"""
            )
            assert same_row is True, (
                "the conversation name and the bar's actions are on "
                "different lines, so the bar has become two bands"
            )
        finally:
            context.close()

    def test_the_first_message_starts_high_on_the_page(self, browser, server):
        """Before the chrome was consolidated the message list started 207px
        down an empty thread. The bound is deliberately loose: it fails on a
        band coming back, not on a few pixels of padding."""
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


class TestTheThreadSpendsTheWidthItIsGiven:
    """Chat is the one section whose value is the space itself.

    Two width caps used to stack - 1100px on the content column and 860px on
    the panel inside it - so a 1440px screen spent nearly 300px of the
    workspace on margin, and hiding the pane bought margin rather than
    thread. Both are lifted for Chat only; every other section keeps a
    reading-width cap, where a full-width form is worse, not better.
    """

    def _widths(self, page):
        return page.evaluate(
            """() => {
                const w = document.querySelector('.workspace')
                    .getBoundingClientRect();
                const c = document.querySelector('.chat-panel')
                    .getBoundingClientRect();
                return {
                    workspace: Math.round(w.width),
                    chat: Math.round(c.width),
                    slack: Math.round((c.left - w.left) + (w.right - c.right)),
                };
            }"""
        )

    def test_the_thread_fills_the_workspace(self, browser, server):
        context, page = _signed_in_page(browser, server, DESKTOP)
        try:
            seen = self._widths(page)
            assert seen["slack"] <= 48, (
                f"the thread leaves {seen['slack']}px of the workspace empty "
                f"({seen['chat']}px inside {seen['workspace']}px)"
            )
        finally:
            context.close()

    def test_hiding_the_pane_gives_its_width_to_the_thread(self, browser, server):
        context, page = _signed_in_page(browser, server, DESKTOP)
        try:
            before = self._widths(page)
            page.click("#pane-toggle")
            page.wait_for_timeout(250)
            after = self._widths(page)
            gained = after["chat"] - before["chat"]
            assert gained > 150, (
                f"hiding the pane freed {after['workspace'] - before['workspace']}px "
                f"of workspace but the thread grew by only {gained}px, so the "
                "space became margin instead of thread"
            )
        finally:
            context.close()


class TestTheSignedOutShellShowsOneThing:
    """Before signing in there is one action, so there is one band.

    The rail navigates nowhere, the pane would list conversations nobody can
    open, and the thread controls would sit above a sign-in form. All three
    collapse; the sign-in card and the wordmark are what is left.
    """

    @pytest.mark.parametrize("viewport", [DESKTOP, PHONE], ids=["desktop", "phone"])
    def test_the_rail_and_pane_are_absent(self, browser, server, viewport):
        context = browser.new_context(viewport=viewport)
        page = context.new_page()
        try:
            page.goto(f"{server.base_url}/", wait_until="domcontentloaded")
            page.wait_for_selector("#auth-panel", state="visible")
            page.wait_for_timeout(400)
            seen = page.evaluate(
                """() => {
                    const box = sel => {
                        const el = document.querySelector(sel);
                        return el ? el.getBoundingClientRect().width : 0;
                    };
                    const height = sel => {
                        const el = document.querySelector(sel);
                        return el ? el.getBoundingClientRect().height : 0;
                    };
                    return {
                        rail: box('.app-rail'),
                        pane: box('.context-pane'),
                        bar: height('.topbar'),
                        overflow: document.documentElement.scrollWidth
                            - window.innerWidth,
                    };
                }"""
            )
            assert seen["rail"] == 0, "the rail navigates nowhere signed out"
            assert seen["pane"] == 0, "the pane lists conversations nobody can open"
            assert seen["bar"] == 0, (
                "the bar names a conversation there is not one of; emptying "
                "it leaves a band of white above the sign-in card"
            )
            assert seen["overflow"] <= 1
        finally:
            context.close()


class TestTheBarStaysSmallInEveryView:
    """A bar is for what you are looking at and the one action it is about.

    Chat's bar once carried eight controls doing four different jobs: the
    thread's identity, its settings, list maintenance, and the session. The
    settings and the session moved behind an overflow menu, and each list's
    refresh moved beside the list it refreshes, which is in the pane. What
    is left is the pane toggle, at most one primary action, and the menu.
    """

    #: Toggle, menu, and at most one action for the view.
    LIMIT = 3

    ALL_TABS = WITH_PANE + WITHOUT_PANE

    def _visible_controls(self, page):
        return page.evaluate(
            """() => [...document.querySelectorAll(
                '.topbar button, .topbar select, .topbar input, .topbar a')]
                .filter(el => el.getBoundingClientRect().width > 0)
                .map(el => el.id || el.className)"""
        )

    def test_no_view_crowds_the_bar(self, browser, server):
        context, page = _signed_in_page(browser, server, DESKTOP)
        try:
            for tab in self.ALL_TABS:
                page.click(f'.rail-btn[data-tab="{tab}"]')
                page.wait_for_timeout(150)
                seen = self._visible_controls(page)
                assert len(seen) <= self.LIMIT, (
                    f"{tab} puts {len(seen)} controls in the bar: {seen}"
                )
        finally:
            context.close()

    def test_the_menu_holds_what_left_the_bar(self, browser, server):
        """Moved, not deleted. Each one still has to be reachable."""
        context, page = _signed_in_page(browser, server, DESKTOP)
        try:
            page.click("#bar-menu-btn")
            page.wait_for_timeout(200)
            reachable = page.evaluate(
                """() => ['context-id', 'workflow-id', 'share-btn', 'logout']
                    .filter(id => {
                        const el = document.getElementById(id);
                        return el && el.getBoundingClientRect().width > 0;
                    })"""
            )
            assert sorted(reachable) == [
                "context-id",
                "logout",
                "share-btn",
                "workflow-id",
            ], f"the menu is missing {reachable}"
        finally:
            context.close()

    def test_escape_closes_the_menu(self, browser, server):
        context, page = _signed_in_page(browser, server, DESKTOP)
        try:
            page.click("#bar-menu-btn")
            page.wait_for_timeout(150)
            page.keyboard.press("Escape")
            page.wait_for_timeout(150)
            hidden = page.evaluate(
                "() => document.getElementById('bar-menu-panel')"
                ".classList.contains('hidden')"
            )
            assert hidden, "the menu stayed open after Escape"
        finally:
            context.close()

    def test_each_list_refreshes_from_its_own_pane(self, browser, server):
        """The control that reloads a list belongs beside the list."""
        context, page = _signed_in_page(browser, server, DESKTOP)
        try:
            for tab, button in [
                ("chat-tab", "refresh-conversations"),
                ("contexts-tab", "refresh-contexts"),
                ("artifacts-tab", "refresh-artifacts"),
                ("tools-tab", "refresh-tools"),
            ]:
                page.click(f'.rail-btn[data-tab="{tab}"]')
                page.wait_for_timeout(150)
                inside = page.evaluate(
                    """(id) => {
                        const el = document.getElementById(id);
                        return !!el && !!el.closest('.context-pane')
                            && el.getBoundingClientRect().width > 0;
                    }""",
                    button,
                )
                assert inside, f"{button} is not in {tab}'s pane"
        finally:
            context.close()


class TestThePaneCannotTrapAPhone:
    """Below 900px the pane is an overlay, and an overlay must be closable.

    It is fixed at `z-index: 30` over a bar at `10`, so it covers the very
    control that opened it. With no close of its own and no reaction to a
    row being chosen, a phone went: open Chats, choose a conversation, and
    keep looking at the list.
    """

    def _phone(self, browser, server):
        return _signed_in_page(browser, server, PHONE)

    def test_the_pane_carries_its_own_close(self, browser, server):
        context, page = self._phone(browser, server)
        try:
            page.click("#pane-toggle")
            page.wait_for_timeout(250)
            assert page.evaluate(
                "() => document.querySelector('.context-pane')"
                ".getBoundingClientRect().width > 0"
            ), "the pane did not open"

            close = page.evaluate(
                """() => {
                    const el = document.getElementById('pane-close');
                    if (!el) return null;
                    const r = el.getBoundingClientRect();
                    if (r.width === 0) return null;
                    // Nothing may sit on top of the way out.
                    const hit = document.elementFromPoint(
                        r.left + r.width / 2, r.top + r.height / 2);
                    return !!hit && !!hit.closest('#pane-close');
                }"""
            )
            assert close is True, (
                "the overlay has no close of its own, and the bar's toggle "
                "is underneath it"
            )
            page.click("#pane-close")
            page.wait_for_timeout(250)
            assert page.evaluate(
                "() => document.querySelector('.context-pane')"
                ".getBoundingClientRect().width === 0"
            )
        finally:
            context.close()

    def test_choosing_a_conversation_gets_out_of_the_way(self, browser, server):
        """The choice is invisible if the list is still on top of it."""
        import httpx

        context, page = self._phone(browser, server)
        try:
            token = page.evaluate(
                "() => sessionStorage.getItem('liminal.accessToken')"
            )
            made = httpx.post(
                f"{server.base_url}/v1/conversations",
                json={"title": "Network diagnostics"},
                headers={"Authorization": f"Bearer {token}"},
                timeout=30,
            )
            assert made.status_code in (200, 201), made.text

            page.click("#pane-toggle")
            page.wait_for_timeout(250)
            page.click("#refresh-conversations")
            page.wait_for_selector(".conversation-item", timeout=15000)
            page.locator(".conversation-item").first.click()
            page.wait_for_timeout(400)

            assert page.evaluate(
                "() => document.querySelector('.context-pane')"
                ".getBoundingClientRect().width === 0"
            ), (
                "the list stayed over the conversation it was used to choose"
            )
        finally:
            context.close()


class TestTheRailCollapsesInTheOpen:
    """Hiding the rail was a shift-click: undiscoverable, and forgotten on
    reload because only the pane's state was stored."""

    def test_the_rail_has_a_control_and_a_way_back(self, browser, server):
        context, page = _signed_in_page(browser, server, DESKTOP)
        try:
            page.click("#rail-toggle")
            page.wait_for_timeout(250)
            state = page.evaluate(
                """() => ({
                    rail: document.querySelector('.app-rail')
                        .getBoundingClientRect().width,
                    restore: document.getElementById('rail-restore')
                        .getBoundingClientRect().width,
                })"""
            )
            assert state["rail"] == 0, "the rail's own control did not hide it"
            assert state["restore"] > 0, "there is no way back to navigation"

            page.click("#rail-restore")
            page.wait_for_timeout(250)
            assert page.evaluate(
                "() => document.querySelector('.app-rail')"
                ".getBoundingClientRect().width > 0"
            )
        finally:
            context.close()

    def test_the_choice_survives_a_reload(self, browser, server):
        context, page = _signed_in_page(browser, server, DESKTOP)
        try:
            page.click("#rail-toggle")
            page.wait_for_timeout(250)
            page.reload(wait_until="domcontentloaded")
            page.wait_for_timeout(900)
            assert page.evaluate(
                "() => document.querySelector('.app-rail')"
                ".getBoundingClientRect().width === 0"
            ), "the rail came back on reload; the choice was not remembered"
        finally:
            context.close()
