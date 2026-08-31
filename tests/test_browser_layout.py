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
                # By id. `.pane-toggle` is a shape three buttons share -
                # the pane's, the rail's restore, and the overflow menu -
                # and `querySelector` returns the first in document order,
                # which is the restore button. Asking that way measured a
                # button that is hidden anyway and proved nothing.
                seen = page.evaluate(
                    """() => {
                        const w = id => {
                            const el = document.getElementById(id);
                            return el ? el.getBoundingClientRect().width : 0;
                        };
                        return {
                            toggle: w('pane-toggle'),
                            menu: w('bar-menu-btn'),
                        };
                    }"""
                )
                assert seen["toggle"] == 0, (
                    f"{tab} offers to hide a list it does not have, and the "
                    "click writes the remembered preference, so it takes "
                    "Chat's conversations away later"
                )
                assert seen["menu"] > 0, (
                    f"{tab} lost its overflow menu: the rule hiding the pane "
                    "button matched every button sharing its class"
                )
                page.click("#bar-menu-btn")
                page.wait_for_timeout(150)
                assert page.evaluate(
                    "() => document.getElementById('logout')"
                    ".getBoundingClientRect().width > 0"
                ), f"there is no way to sign out from {tab}"
                page.keyboard.press("Escape")
                page.wait_for_timeout(100)
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
                "sections; that is the rail's own control, not this one"
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
    """The rail once hid only through a modifier gesture on the pane's
    toggle: undiscoverable, and forgotten on reload because only the pane's
    state was stored. It has its own control and its own memory now."""

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


class TestTheTwoCollapsesAreIndependent:
    """Two controls, two stored keys, and so two geometries.

    Hiding the rail used to collapse the pane with it. That was right while
    one control meant both; it is wrong now that the rail has its own,
    because hiding 48px of navigation silently took away 240px of
    conversation list as well.
    """

    def _bands(self, page):
        return page.evaluate(
            """() => ({
                rail: Math.round(document.querySelector('.app-rail')
                    .getBoundingClientRect().width),
                pane: Math.round(document.querySelector('.context-pane')
                    .getBoundingClientRect().width),
            })"""
        )

    def test_hiding_the_rail_keeps_the_pane(self, browser, server):
        context, page = _signed_in_page(browser, server, DESKTOP)
        try:
            page.click("#rail-toggle")
            page.wait_for_timeout(250)
            seen = self._bands(page)
            assert seen["rail"] == 0, "the rail's control did not hide it"
            assert seen["pane"] > 100, (
                "hiding the rail took the conversation list with it; they "
                f"are separate controls, and the pane is {seen['pane']}px"
            )
        finally:
            context.close()

    def test_hiding_the_pane_keeps_the_rail(self, browser, server):
        context, page = _signed_in_page(browser, server, DESKTOP)
        try:
            page.click("#pane-toggle")
            page.wait_for_timeout(250)
            seen = self._bands(page)
            assert seen["pane"] == 0
            assert seen["rail"] > 0, "hiding the list took the rail with it"
        finally:
            context.close()

    def test_both_can_be_hidden_at_once(self, browser, server):
        context, page = _signed_in_page(browser, server, DESKTOP)
        try:
            page.click("#pane-toggle")
            page.wait_for_timeout(150)
            page.click("#rail-toggle")
            page.wait_for_timeout(250)
            assert self._bands(page) == {"rail": 0, "pane": 0}
        finally:
            context.close()

    def test_a_phone_overlay_takes_the_width_the_rail_gave_up(
        self, browser, server
    ):
        """The overlay is offset by the rail, so it must stop being offset
        when the rail is gone, or it leaves a 48px strip of dead ground."""
        context, page = _signed_in_page(browser, server, PHONE)
        try:
            page.click("#rail-toggle")
            page.wait_for_timeout(200)
            page.click("#pane-toggle")
            page.wait_for_timeout(300)
            box = page.evaluate(
                """() => {
                    const r = document.querySelector('.context-pane')
                        .getBoundingClientRect();
                    return {left: Math.round(r.left), width: Math.round(r.width)};
                }"""
            )
            assert box["left"] == 0, (
                f"the overlay starts {box['left']}px in, holding a gap for a "
                "rail that is not there"
            )
            assert box["width"] >= PHONE["width"] - 1, (
                f"the overlay is {box['width']}px of a {PHONE['width']}px "
                "viewport it now has to itself"
            )
        finally:
            context.close()


class TestProseKeepsAMeasureTheWorkspaceDoesNot:
    """The wide workspace earns its width on code, tables and tool output.

    Ordinary prose does not: a paragraph set across 1400px is unpleasant to
    read however welcome the space is for a diff. So the measure is on the
    prose children of an answer, not on the panel or the bubble, and `pre`
    and `table` are deliberately left alone.

    The markup here is the template `renderMessage` writes (chat.js), so the
    rule is exercised against the shape the app actually produces rather
    than against a shape this test invented.
    """

    LONG = "Prose that has to wrap somewhere. " * 40

    def test_paragraphs_are_measured_and_code_is_not(self, browser, server):
        context, page = _signed_in_page(browser, server, DESKTOP)
        try:
            page.evaluate(
                """(long) => {
                    const list = document.getElementById('messages');
                    list.innerHTML = `
                      <div class="message assistant">
                        <div class="content">
                          <div class="bubble">
                            <p id="probe-p">${long}</p>
                            <pre id="probe-pre"><code>${long}</code></pre>
                            <table id="probe-table"><tbody><tr><td>${long}</td></tr></tbody></table>
                          </div>
                        </div>
                      </div>`;
                }""",
                self.LONG,
            )
            page.wait_for_timeout(200)
            seen = page.evaluate(
                """() => {
                    const w = id => document.getElementById(id)
                        .getBoundingClientRect().width;
                    // A real ch in the paragraph's own font, measured
                    // rather than guessed from the em: the ratio differs
                    // per face, and the bound is in characters.
                    const probe = document.getElementById('probe-p');
                    const ruler = document.createElement('span');
                    ruler.style.cssText =
                        'position:absolute;visibility:hidden;width:100ch';
                    probe.appendChild(ruler);
                    const ch = ruler.getBoundingClientRect().width / 100;
                    ruler.remove();
                    return {
                        prose: w('probe-p'),
                        pre: w('probe-pre'),
                        table: w('probe-table'),
                        ch,
                    };
                }"""
            )
            measure = seen["prose"] / seen["ch"]
            assert measure <= 92, (
                f"a paragraph runs about {measure:.0f} characters "
                f"({seen['prose']:.0f}px); the measure is meant to be 88ch"
            )
            assert seen["pre"] > seen["prose"] + 50, (
                f"code is held to the prose measure ({seen['pre']:.0f}px vs "
                f"{seen['prose']:.0f}px); the width is the point of a code block"
            )
            assert seen["table"] > seen["prose"] + 50, (
                "a table is held to the prose measure"
            )
        finally:
            context.close()


class TestTheResponsiveDefaultIsNotAChoice:
    """A width decides the default; only a click decides a preference.

    Initialisation used to write the responsive default through the
    remembering path, which turned "this is what a phone opens with" into
    "this is what the reader wants everywhere". A first visit on a desktop
    stored `paneHidden=0` and the same browser on a phone then opened the
    overlay on top of the thread; a first visit on a phone stored `1` and
    the desktop came back with no conversation list.

    Every other test here builds a fresh `BrowserContext` per viewport, so
    `localStorage` never survives the transition and none of them can see
    this. These reuse one context across both shapes, which is what a person
    with one browser actually does.
    """

    PANE_KEY = "liminal.paneHidden"
    RAIL_KEY = "liminal.railHidden"

    def _signed_in_context(self, browser, server, viewport):
        """Like `_signed_in_page`, but the caller keeps the context."""
        import httpx

        email = f"pref_{uuid.uuid4().hex[:8]}@example.com"
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

    def _pane_width(self, page):
        return page.evaluate(
            "() => document.querySelector('.context-pane')"
            ".getBoundingClientRect().width"
        )

    def _stored(self, page, key=None):
        return page.evaluate(
            "(k) => localStorage.getItem(k)", key or self.PANE_KEY
        )

    def _reload_at(self, page, viewport, server):
        # Waits for the shell to have chosen a section, not for the rail to
        # be visible: a reader who hid the rail reloads into a page where it
        # is legitimately absent, and waiting for it there can only time out.
        page.set_viewport_size(viewport)
        page.goto(f"{server.base_url}/", wait_until="domcontentloaded")
        page.wait_for_function(
            "() => !!document.querySelector('.app-shell[data-section]')",
            timeout=30000,
        )
        page.wait_for_timeout(700)

    def test_a_desktop_first_visit_does_not_decide_for_the_phone(
        self, browser, server
    ):
        context, page = self._signed_in_context(browser, server, DESKTOP)
        try:
            assert self._pane_width(page) > 100, "the desktop default is open"
            assert self._stored(page) is None, (
                "opening on a desktop recorded a pane preference the reader "
                "never expressed"
            )
            self._reload_at(page, PHONE, server)
            assert self._pane_width(page) == 0, (
                "the phone opened the overlay on top of the thread, because "
                "the desktop visit had stored its default as a choice"
            )
        finally:
            context.close()

    def test_a_phone_first_visit_does_not_decide_for_the_desktop(
        self, browser, server
    ):
        context, page = self._signed_in_context(browser, server, PHONE)
        try:
            assert self._pane_width(page) == 0, "the phone default is hidden"
            assert self._stored(page) is None, (
                "opening on a phone recorded a pane preference the reader "
                "never expressed"
            )
            self._reload_at(page, DESKTOP, server)
            assert self._pane_width(page) > 100, (
                "the desktop came back with no conversation list, because "
                "the phone visit had stored its default as a choice"
            )
        finally:
            context.close()

    def test_an_actual_choice_does_persist(self, browser, server):
        """The other half: the fix must not stop preferences working."""
        context, page = self._signed_in_context(browser, server, DESKTOP)
        try:
            page.click("#pane-toggle")
            page.wait_for_timeout(200)
            assert self._pane_width(page) == 0
            assert self._stored(page) == "1", (
                "clicking the toggle did not record the choice"
            )
            self._reload_at(page, DESKTOP, server)
            assert self._pane_width(page) == 0, (
                "the reader hid the pane and it came back on reload"
            )
        finally:
            context.close()

    def test_an_untouched_load_stores_no_shell_preference_at_all(
        self, browser, server
    ):
        """Both halves of one contract, not one of them.

        The rail's default does not depend on the viewport, so writing it on
        load produced no visible bug - it just meant the two keys followed
        different rules, and the pane's version of that rule was the defect
        above.
        """
        context, page = self._signed_in_context(browser, server, DESKTOP)
        try:
            assert self._stored(page, self.PANE_KEY) is None
            assert self._stored(page, self.RAIL_KEY) is None, (
                "an untouched load recorded a rail preference the reader "
                "never expressed"
            )
        finally:
            context.close()

    def test_the_rail_remembers_a_real_choice_both_ways(self, browser, server):
        context, page = self._signed_in_context(browser, server, DESKTOP)
        try:
            page.click("#rail-toggle")
            page.wait_for_timeout(200)
            assert self._stored(page, self.RAIL_KEY) == "1"
            self._reload_at(page, DESKTOP, server)
            assert page.evaluate(
                "() => document.querySelector('.app-rail')"
                ".getBoundingClientRect().width"
            ) == 0, "the reader hid the rail and it came back on reload"

            page.click("#rail-restore")
            page.wait_for_timeout(200)
            assert self._stored(page, self.RAIL_KEY) == "0"
            self._reload_at(page, DESKTOP, server)
            assert page.evaluate(
                "() => document.querySelector('.app-rail')"
                ".getBoundingClientRect().width"
            ) > 0, "the reader restored the rail and it vanished on reload"
        finally:
            context.close()


class TestEveryPageThatWearsTheShellGetsAPageToFill:
    """Three pages use `.app-shell`; only one of them has bands.

    The rail work turned `.app-shell` into `48px 240px 1fr`. The admin console
    and the shared-conversation page reuse that class for its chrome and have
    only a bar and one column under it, so their bar landed in the rail's 48px
    and their entire content in the pane's 240px - on a 1440px screen, beside
    1200px of nothing.

    Neither page has a rail or a pane to put in those tracks, and neither is
    reachable from the section navigation the lane's other tests drive, which
    is why every one of them passed while the console was unusable.
    """

    #: `.layout` is `max-width: var(--content-max)`, centred. On a page that
    #: gets the full width that resolves to 1100px; in the pane's track it
    #: cannot exceed 240.
    CONTENT_MAX = 1100

    @pytest.mark.parametrize("path", ["/admin", "/share"], ids=["admin", "share"])
    def test_the_page_gets_the_width_of_the_page(self, browser, server, path):
        context = browser.new_context(viewport=DESKTOP)
        page = context.new_page()
        try:
            page.goto(f"{server.base_url}{path}", wait_until="domcontentloaded")
            page.wait_for_selector(".app-shell", state="attached")
            page.wait_for_timeout(400)

            measured = page.evaluate(
                """() => {
                  const box = (sel) => {
                    const el = document.querySelector(sel);
                    return el ? Math.round(el.getBoundingClientRect().width) : -1;
                  };
                  return {
                    bar: box('.topbar'),
                    content: box('main.layout'),
                    viewport: window.innerWidth,
                  };
                }"""
            )

            assert measured["content"] == self.CONTENT_MAX, (
                f"{path} rendered its content {measured['content']}px wide on a "
                f"{measured['viewport']}px page; the pane's track is 240 and "
                f"the content column is {self.CONTENT_MAX}"
            )
            assert measured["bar"] == measured["viewport"], (
                f"{path} put its bar in {measured['bar']}px; the rail's track "
                f"is 48 and the bar spans the page"
            )
        finally:
            context.close()

    @pytest.mark.parametrize("path", ["/admin", "/share"], ids=["admin", "share"])
    def test_the_page_does_not_scroll_sideways(self, browser, server, path):
        """The other half: filling the width must not overflow it."""
        context = browser.new_context(viewport=DESKTOP)
        page = context.new_page()
        try:
            page.goto(f"{server.base_url}{path}", wait_until="domcontentloaded")
            page.wait_for_selector(".app-shell", state="attached")
            page.wait_for_timeout(400)
            overflow = page.evaluate(
                "() => document.documentElement.scrollWidth - window.innerWidth"
            )
            assert overflow <= 0, f"{path} is {overflow}px wider than its viewport"
        finally:
            context.close()


class TestTheOverlayGetsOutOfTheWayForEveryChoice:
    """A pane control that opens something must not then cover it.

    The dismissal rule was an enumerated list of row classes, and Notes has
    two controls that are not rows: a search hit, and the button that starts
    a new note. Both open the editor, neither was listed, so on a phone the
    workspace changed underneath an overlay still sitting on top of it.

    Marked semantically now - `[data-pane-dismiss]` - so a control added
    later says what it does instead of needing this list edited.
    """

    def _account(self, server):
        import httpx

        email = f"pane_{uuid.uuid4().hex[:8]}@example.com"
        resp = httpx.post(
            f"{server.base_url}/v1/auth/signup",
            json={"email": email, "password": PASSWORD},
            timeout=30,
        )
        assert resp.status_code == 201, resp.text
        token = resp.json()["data"]["access_token"]
        return email, token

    def _phone_notes_page(self, browser, server, email):
        """Signed in, on Notes, at a width where the pane is an overlay."""
        context = browser.new_context(viewport=PHONE)
        page = context.new_page()
        page.goto(f"{server.base_url}/", wait_until="domcontentloaded")
        page.fill("#email", email)
        page.fill("#password", PASSWORD)
        page.click("#auth-form button[type=submit]")
        page.wait_for_function(
            "() => !!sessionStorage.getItem('liminal.accessToken')", timeout=30000
        )
        page.wait_for_selector("#main-tabs", state="visible")
        page.click('.rail-btn[data-tab="notes-tab"]')
        page.wait_for_timeout(500)
        # The overlay starts closed at this width, so open it: dismissal is
        # only observable on a pane that is showing.
        page.click("#pane-toggle")
        page.wait_for_timeout(300)
        assert self._pane_width(page) > 0, "the pane did not open to be dismissed"
        return context, page

    def _pane_width(self, page):
        return page.evaluate(
            "() => document.querySelector('.context-pane')"
            ".getBoundingClientRect().width"
        )

    def test_choosing_a_search_hit_dismisses_the_overlay(self, browser, server):
        import httpx

        email, token = self._account(server)
        title = f"Kestrel{uuid.uuid4().hex[:6]}"
        created = httpx.post(
            f"{server.base_url}/v1/notes",
            headers={"Authorization": f"Bearer {token}"},
            json={"title": title, "content": "A note to find by searching."},
            timeout=30,
        )
        assert created.status_code in (200, 201), created.text

        context, page = self._phone_notes_page(browser, server, email)
        try:
            page.fill("#note-search-input", title)
            page.wait_for_selector(".note-search-hit", timeout=15000)
            page.click(".note-search-hit")
            page.wait_for_timeout(400)

            assert self._pane_width(page) == 0, (
                "the search hit opened the note behind an overlay still "
                "covering it"
            )
            assert page.is_visible("#note-editor"), "the note did not open"
        finally:
            context.close()

    def test_starting_a_new_note_dismisses_the_overlay(self, browser, server):
        email, _ = self._account(server)
        context, page = self._phone_notes_page(browser, server, email)
        try:
            page.click("#note-new-btn")
            page.wait_for_timeout(400)
            assert self._pane_width(page) == 0, (
                "the new-note button opened the editor behind an overlay "
                "still covering it"
            )
            assert page.is_visible("#note-editor"), "the editor did not open"
        finally:
            context.close()


class TestTheDefaultFollowsTheWidthWhileItIsStillTheDefault:
    """The responsive default is a function of the width, so crossing the
    breakpoint has to reapply it.

    It was decided once at load. Narrowing a desktop window past 900px turned
    the open pane into an overlay on top of the workspace instead of the
    hidden default; widening a narrow one left the conversation list hidden
    on a screen with room for it.

    `TestTheResponsiveDefaultIsNotAChoice` gets close and cannot see this: it
    reloads at each width, so it measures initialisation twice rather than a
    live crossing. These resize without navigating.
    """

    PANE_KEY = "liminal.paneHidden"

    def _stored(self, page):
        return page.evaluate("(k) => localStorage.getItem(k)", self.PANE_KEY)

    def _pane_width(self, page):
        return page.evaluate(
            "() => document.querySelector('.context-pane')"
            ".getBoundingClientRect().width"
        )

    def _resize(self, page, viewport):
        page.set_viewport_size(viewport)
        page.wait_for_timeout(400)

    def test_crossing_the_breakpoint_reapplies_the_default(self, browser, server):
        context, page = _signed_in_page(browser, server, DESKTOP)
        try:
            assert self._pane_width(page) > 100, "the desktop default is open"
            assert self._stored(page) is None

            self._resize(page, PHONE)
            assert self._pane_width(page) == 0, (
                "narrowing past 900px left the pane showing, so it became an "
                "overlay on top of the workspace instead of the phone default"
            )
            assert self._stored(page) is None, (
                "reapplying the default recorded it as a choice"
            )

            self._resize(page, DESKTOP)
            assert self._pane_width(page) > 100, (
                "widening left the conversation list hidden on a screen with "
                "room for it"
            )
            assert self._stored(page) is None
        finally:
            context.close()

    def test_a_stored_choice_survives_the_crossing(self, browser, server):
        """The control. A width may decide a default; it may not overrule a
        reader who has already said what they want."""
        context, page = _signed_in_page(browser, server, DESKTOP)
        try:
            page.click("#pane-toggle")
            page.wait_for_timeout(200)
            assert self._pane_width(page) == 0
            assert self._stored(page) == "1"

            self._resize(page, PHONE)
            assert self._stored(page) == "1", "the crossing rewrote the choice"

            self._resize(page, DESKTOP)
            assert self._pane_width(page) == 0, (
                "the reader hid the pane and widening the window brought it "
                "back"
            )
            assert self._stored(page) == "1"
        finally:
            context.close()


class TestAPaneListFitsItsPane:
    """Nothing in a 240px column may be wider than 240px.

    The lists were written for a full-width panel and moved into the pane
    without being re-laid out. The overflow was then contained rather than
    removed - `overflow-x: auto` on the wrapper - which keeps the column the
    right width and lets the content scroll out of sight inside it. The
    artifacts list was a five-column table, so what scrolled away was the
    name: the pane showed VERSION and UPDATED, and the field you pick a row
    by was off the left edge.

    Measuring the pane proves nothing, because the pane was always 240px.
    These measure the content against it.
    """

    #: Panes whose list is populated by the fixtures a fresh account has.
    #: Chat and Notes start empty for a new signup, so their emptiness is not
    #: evidence either way; Tools ships a built-in catalogue.
    POPULATED = ["tools-tab"]

    def _overflow(self, page):
        """How far the widest thing in the pane sticks out past it."""
        return page.evaluate(
            """() => {
              const pane = document.querySelector('.context-pane');
              if (!pane) return null;
              const limit = pane.getBoundingClientRect().width;
              let worst = 0, culprit = '';
              for (const el of pane.querySelectorAll('*')) {
                if (!el.getClientRects().length) continue;
                const over = el.scrollWidth - limit;
                if (over > worst) {
                  worst = over;
                  culprit = el.className || el.tagName;
                }
              }
              return {over: Math.round(worst), culprit, limit: Math.round(limit)};
            }"""
        )

    @pytest.mark.parametrize("tab", WITH_PANE)
    def test_nothing_in_the_pane_is_wider_than_the_pane(
        self, browser, server, tab
    ):
        context, page = _signed_in_page(browser, server, DESKTOP)
        try:
            page.click(f'.rail-btn[data-tab="{tab}"]')
            page.wait_for_timeout(600)
            result = self._overflow(page)
            assert result is not None, "the pane is missing"
            assert result["over"] <= 1, (
                f"{tab}: `{result['culprit']}` is {result['over']}px wider "
                f"than the {result['limit']}px pane, so part of it can only "
                f"be reached by scrolling sideways"
            )
        finally:
            context.close()

    @pytest.mark.parametrize("tab", POPULATED)
    def test_a_row_leads_with_its_name(self, browser, server, tab):
        """The other half. A list that fits but shows the wrong field first
        would pass the test above and still be useless.
        """
        context, page = _signed_in_page(browser, server, DESKTOP)
        try:
            page.click(f'.rail-btn[data-tab="{tab}"]')
            page.wait_for_selector(".tool-card", timeout=15000)
            leads = page.evaluate(
                """() => {
                  const card = document.querySelector('.tool-card');
                  const pane = document.querySelector('.context-pane');
                  const name = card.querySelector('.tool-name');
                  const p = pane.getBoundingClientRect();
                  const n = name.getBoundingClientRect();
                  return {
                    visible: n.left >= p.left - 1 && n.right <= p.right + 1,
                    text: name.textContent.trim(),
                  };
                }"""
            )
            assert leads["visible"], (
                f"{tab}: the name is outside the pane's own box, so the row "
                f"cannot be identified without scrolling"
            )
            assert leads["text"], "the row has no name to lead with"
        finally:
            context.close()


class TestTheRailIsOneColumnOfEqualIcons:
    """Ten controls down a 48px edge, each aligned and each able to say
    what it is.

    The rail is icons and nothing else, so a reader learns it by hovering.
    That only works if every control has a label and the icons line up: one
    icon 8px out of column reads as a different kind of thing.
    """

    def _admin_page(self, browser, server):
        """A signed-in admin, because the shield only exists for one."""
        import httpx

        email = f"rail_{uuid.uuid4().hex[:8]}@example.com"
        resp = httpx.post(
            f"{server.base_url}/v1/auth/signup",
            json={"email": email, "password": PASSWORD},
            timeout=30,
        )
        assert resp.status_code == 201, resp.text
        token = resp.json()["data"]["access_token"]
        me = httpx.get(
            f"{server.base_url}/v1/me",
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )
        user_id = me.json()["data"]["id"]

        from liminallm.service.runtime import get_runtime

        get_runtime().store.update_user_role(user_id, "admin")

        context = browser.new_context(viewport=DESKTOP)
        page = context.new_page()
        page.goto(f"{server.base_url}/", wait_until="domcontentloaded")
        page.fill("#email", email)
        page.fill("#password", PASSWORD)
        page.click("#auth-form button[type=submit]")
        page.wait_for_function(
            "() => !!sessionStorage.getItem('liminal.accessToken')", timeout=30000
        )
        page.wait_for_selector("#admin-link", state="visible", timeout=15000)
        page.wait_for_timeout(400)
        return context, page

    def test_every_icon_sits_in_the_same_column(self, browser, server):
        """The admin link is an anchor among buttons and was the one that
        drifted: JS set `display: inline-flex` on it, overriding `.rail-btn`'s
        `grid`, and `place-items: center` does not centre a flex row.
        """
        context, page = self._admin_page(browser, server)
        try:
            lefts = page.evaluate(
                """() => [...document.querySelectorAll('#main-tabs .rail-btn')]
                     .filter(el => el.getBoundingClientRect().width > 0)
                     .map(el => ({
                       id: el.id || el.dataset.tab,
                       x: Math.round(
                         el.querySelector('svg').getBoundingClientRect().x),
                     }))"""
            )
            assert len(lefts) >= 9, f"expected the full rail, saw {lefts}"
            columns = sorted({row["x"] for row in lefts})
            assert len(columns) == 1, (
                f"the rail's icons sit in {len(columns)} columns at x={columns}: "
                + ", ".join(f"{r['id']}@{r['x']}" for r in lefts)
            )
        finally:
            context.close()

    def test_every_control_says_what_it_is(self, browser, server):
        """One label each. Every control carried a `title` as well as its
        own pill, so the native tooltip repeated the label a beat later.
        """
        context, page = self._admin_page(browser, server)
        try:
            report = page.evaluate(
                """() => [...document.querySelectorAll('#main-tabs .rail-btn')]
                     .map(el => ({
                       id: el.id || el.dataset.tab,
                       label: (el.querySelector('.rail-name') || {}).textContent,
                       title: el.getAttribute('title'),
                       aria: el.getAttribute('aria-label'),
                     }))"""
            )
            missing = [r["id"] for r in report if not (r["label"] or "").strip()]
            assert not missing, f"rail controls with no hover label: {missing}"

            doubled = [r["id"] for r in report if r["title"]]
            assert not doubled, (
                f"these carry a `title` as well as a pill, so the browser "
                f"repeats the label in a second tooltip: {doubled}"
            )

            unnamed = [r["id"] for r in report if not (r["aria"] or "").strip()]
            assert not unnamed, f"rail controls with no accessible name: {unnamed}"
        finally:
            context.close()

    def test_a_label_appears_on_hover_over_the_pane(self, browser, server):
        """The pill is drawn outside the 48px rail, across whatever is beside
        it, so it has to win that paint order to be readable at all."""
        context, page = self._admin_page(browser, server)
        try:
            page.click('.rail-btn[data-tab="chat-tab"]')
            page.wait_for_timeout(300)
            page.hover('.rail-btn[data-tab="notes-tab"]')
            page.wait_for_timeout(400)
            shown = page.evaluate(
                """() => {
                  const b = document.querySelector(
                    '.rail-btn[data-tab="notes-tab"]');
                  const n = b.querySelector('.rail-name');
                  const box = n.getBoundingClientRect();
                  // `pointer-events: none` keeps the pill from swallowing the
                  // hover, and would also make elementFromPoint skip it, so
                  // ask about paint order rather than hit testing.
                  const prev = n.style.pointerEvents;
                  n.style.pointerEvents = 'auto';
                  const top = document.elementFromPoint(
                    box.x + box.width / 2, box.y + box.height / 2);
                  n.style.pointerEvents = prev;
                  return {
                    opacity: getComputedStyle(n).opacity,
                    onTop: top === n,
                    text: n.textContent.trim(),
                  };
                }"""
            )
            assert shown["opacity"] == "1", "the label stayed transparent"
            assert shown["text"] == "Notes", shown["text"]
            assert shown["onTop"], (
                "the label is drawn behind the pane it overlaps, so hovering "
                "the rail explains nothing"
            )
        finally:
            context.close()


class TestTheNoticesUnderAnAnswer:
    """The chips under an answer are built from text the answer's sources
    supplied, so a source gets to write into the page.

    Driven through `renderMessage` itself rather than a hand-built string:
    the defect is in what that function emits, and a double would only encode
    my belief about it.
    """

    #: A citation body that closes a double-quoted attribute and opens
    #: another. A fetched page can contain exactly this.
    HOSTILE = 'he said "hi" data-probe="owned'

    def _render(self, page, message):
        """Render one message with the page's own renderer, into a detached
        container so nothing else on the page is disturbed."""
        return page.evaluate(
            """(m) => {
              const host = document.createElement('div');
              host.innerHTML = renderMessage(m);
              const chip = host.querySelector('.citation-link');
              return {
                found: !!chip,
                title: chip && chip.getAttribute('title'),
                probe: chip && chip.getAttribute('data-probe'),
                kind: chip && chip.getAttribute('data-kind'),
                citation: chip && chip.dataset.citation,
                chips: host.querySelectorAll('.citation-link').length,
                more: (host.querySelector('.citation-more') || {}).textContent || '',
              };
            }""",
            message,
        )

    def _message(self, citations):
        return {
            "id": "m1",
            "role": "assistant",
            "content": "an answer",
            "content_struct": {"citations": citations},
        }

    def test_a_source_cannot_write_attributes_into_the_page(
        self, browser, server
    ):
        context, page = _signed_in_page(browser, server, DESKTOP)
        try:
            out = self._render(
                page,
                self._message([
                    {"source_path": "notes/a.txt", "content": self.HOSTILE}
                ]),
            )
            assert out["found"], "no citation chip was rendered"
            assert out["probe"] is None, (
                "a citation's own text created an attribute on the chip: the "
                "excerpt reaches `title` through `escapeHtml`, which by this "
                "file's own comment leaves quotes alone"
            )
            assert self.HOSTILE in (out["title"] or ""), (
                f"the quotes were not kept as text: {out['title']!r}"
            )
            decoded = page.evaluate(
                "(s) => JSON.parse(s).content", out["citation"]
            )
            assert decoded == self.HOSTILE, (
                f"the citation payload no longer round-trips: {decoded!r}"
            )
        finally:
            context.close()

    def test_an_uploaded_file_is_not_dressed_up_as_a_note(
        self, browser, server
    ):
        """`.md` and `.txt` are ordinary upload types here, so an extension
        cannot say a source came from the notes vault, and nothing else in
        the payload says so either."""
        context, page = _signed_in_page(browser, server, DESKTOP)
        try:
            out = self._render(
                page,
                self._message([
                    {"source_path": "uploads/manual.md", "content": "x"}
                ]),
            )
            assert out["kind"] == "file", (
                f"an uploaded manual.md rendered as {out['kind']!r}"
            )
            web = self._render(
                page,
                self._message([
                    {"source_path": "https://example.com/p", "content": "x"}
                ]),
            )
            assert web["kind"] == "web", (
                f"an http source rendered as {web['kind']!r}"
            )
        finally:
            context.close()

    def test_a_long_row_stops_and_says_how_many_are_left(
        self, browser, server
    ):
        context, page = _signed_in_page(browser, server, DESKTOP)
        try:
            out = self._render(
                page,
                self._message([
                    {"source_path": f"f{i}.pdf", "content": "x"}
                    for i in range(20)
                ]),
            )
            assert out["chips"] == 20, "every citation should still be present"
            assert "12 more" in out["more"], (
                f"the overflow control said {out['more']!r}"
            )
        finally:
            context.close()

    def test_every_tool_gets_an_icon_that_names_itself(self, browser, server):
        context, page = _signed_in_page(browser, server, DESKTOP)
        try:
            names = ["web_fetch", "web.fetch_v1", "notes.search_v1", "nonesuch"]
            out = page.evaluate(
                """(names) => {
                  const host = document.createElement('div');
                  host.innerHTML = toolIconsHtml(names);
                  return [...host.querySelectorAll('.tool-chip')].map((c) => ({
                    label: c.getAttribute('aria-label'),
                    hasSvg: !!c.querySelector('svg'),
                  }));
                }""",
                names,
            )
            assert len(out) == len(names), (
                f"expected an icon per tool, got {len(out)}"
            )
            assert all(c["hasSvg"] for c in out), "a tool rendered without an icon"
            for chip, name in zip(out, names):
                assert name in chip["label"], (
                    f"{name} is not named by its icon: {chip['label']!r}"
                )
        finally:
            context.close()


class TestTheChatWindowIsTheWindow:
    """Chat is the one section whose value is the space itself.

    Everywhere else a panel is an object on a grey ground. Here that ground
    was a band above the thread and a band below it, and under the composer
    sat a divider and a bordered card whose whole collapsed content was a
    heading and two buttons. Measured at 900px tall: 251px of message list,
    252px of chrome below the composer.
    """

    def _geometry(self, page):
        return page.evaluate(
            """() => {
              const q = (s) => document.querySelector(s);
              const bar = q('.topbar');
              const panel = q('#chat-tab .chat-panel');
              const form = q('#chat-form');
              return {
                greyAbove: Math.round(panel.getBoundingClientRect().top
                  - bar.getBoundingClientRect().bottom),
                greyBelow: Math.round(window.innerHeight
                  - panel.getBoundingClientRect().bottom),
                belowComposer: Math.round(window.innerHeight
                  - form.getBoundingClientRect().bottom),
                messages: Math.round(
                  q('#messages').getBoundingClientRect().height),
                overflow: document.documentElement.scrollHeight
                  - window.innerHeight,
              };
            }"""
        )

    def test_no_ground_shows_above_or_below_the_thread(self, browser, server):
        context, page = _signed_in_page(browser, server, DESKTOP)
        try:
            g = self._geometry(page)
            assert g["greyAbove"] == 0, (
                f"{g['greyAbove']}px of workspace shows between the bar and "
                f"the thread"
            )
            assert g["greyBelow"] == 0, (
                f"{g['greyBelow']}px of workspace shows under the thread"
            )
            assert g["overflow"] <= 0, (
                f"filling the height pushed the page {g['overflow']}px taller "
                f"than the viewport"
            )
        finally:
            context.close()

    def test_the_composer_sits_near_the_bottom(self, browser, server):
        """What is under it is chrome the conversation pays for."""
        context, page = _signed_in_page(browser, server, DESKTOP)
        try:
            g = self._geometry(page)
            assert g["belowComposer"] <= 48, (
                f"{g['belowComposer']}px sits below the input box; it was 252 "
                f"when a divider and a bordered card lived there"
            )
            assert g["messages"] > 400, (
                f"the message list is only {g['messages']}px of a "
                f"{DESKTOP['height']}px screen"
            )
        finally:
            context.close()

    def test_the_composer_carries_the_feedback_and_the_toggle(
        self, browser, server
    ):
        """Both moved out of the band under it: a one-click reaction to the
        last answer belongs beside Send, and the account id it displaced is
        on the settings screen, which is where an account detail lives.
        """
        context, page = _signed_in_page(browser, server, DESKTOP)
        try:
            found = page.evaluate(
                """() => ({
                  thumbsUp: !!document.querySelector('#chat-form #thumbs-up'),
                  thumbsDown: !!document.querySelector('#chat-form #thumbs-down'),
                  toggle: !!document.querySelector('#chat-form #preferences-toggle'),
                  sessionIndicator: !!document.getElementById('session-indicator'),
                  detailShut: document.getElementById('preferences-section')
                    .classList.contains('collapsed'),
                })"""
            )
            assert found["thumbsUp"] and found["thumbsDown"], (
                "the feedback buttons are not in the composer's own row"
            )
            assert found["toggle"], "the preferences toggle is not in the composer"
            assert not found["sessionIndicator"], (
                "the account id is still printed under the composer"
            )
            assert found["detailShut"], "the detail panel opens by default"

            page.click("#preferences-toggle")
            page.wait_for_timeout(200)
            assert page.evaluate(
                "() => !document.getElementById('preferences-section')"
                ".classList.contains('collapsed')"
            ), "the toggle no longer opens the detail it names"
        finally:
            context.close()
