"""What the stylesheet says a control is, and what the browser draws.

Two rules here have already been written once in this project and not
reached the page. `.delete-user-btn` asks for 28px and renders 30, because
`button.ghost` is element-plus-class and outranks a bare class. `.table th`
asks for 13px and renders 11.5, because a later rule at equal specificity
wins. Both are recorded in `docs/DESIGN_LANGUAGE.md` under "Rules that fixed
nothing", and neither was found by reading.

So a rule that says a filter is 28px is not evidence that a filter is 28px.
`select.filter` ties with `.field select` on specificity and wins only on
source order, which is exactly the shape that failed before. It is measured
here instead.

The focus vocabulary is the same kind of claim. Moving four rule blocks from
`:focus` to `:focus-visible` is a no-op for text entry and a real change for
a select, *if* browsers match `:focus-visible` on a clicked text field. That
is an assumption about a user agent, so it is asked of one.
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

#: Part one's two tiers. A filter belongs on the compact one.
STANDARD = 30
COMPACT = 28

FILTERS = (
    ("artifacts-tab", "#artifact-type-filter"),
    ("artifacts-tab", "#artifact-visibility-filter"),
    ("settings-tab", "#patches-status-filter"),
)

#: Read back as the browser resolved it, not as the rule was written.
HEIGHT = "(sel) => Math.round(document.querySelector(sel).getBoundingClientRect().height)"

BORDER = "(sel) => getComputedStyle(document.querySelector(sel)).borderTopColor"

FOCUS_STATE = """(sel) => {
  const el = document.querySelector(sel);
  return [
    document.activeElement === el,
    el.matches(':focus-visible'),
    getComputedStyle(el).borderTopColor,
  ];
}"""


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


@pytest.fixture(scope="module")
def page(browser, server):
    """One signed-in administrator. The patch filter is on an admin panel."""
    import httpx

    email = f"tier_{uuid.uuid4().hex[:8]}@example.com"
    resp = httpx.post(
        f"{server.base_url}/v1/auth/signup",
        json={"email": email, "password": PASSWORD},
        timeout=30,
    )
    assert resp.status_code == 201, resp.text

    # Promotion before sign-in: a role change invalidates a token minted
    # before it, so promoting afterwards breaks the session under test.
    from liminallm.service.runtime import get_runtime

    get_runtime().store.update_user_role(resp.json()["data"]["user_id"], role="admin")

    context = browser.new_context(viewport=DESKTOP)
    opened = context.new_page()
    opened.goto(f"{server.base_url}/", wait_until="domcontentloaded")
    opened.fill("#email", email)
    opened.fill("#password", PASSWORD)
    opened.click("#auth-form button[type=submit]")
    opened.wait_for_function(
        "() => !!sessionStorage.getItem('liminal.accessToken')", timeout=30000
    )
    opened.wait_for_selector("#main-tabs", state="visible")
    opened.wait_for_timeout(900)
    try:
        yield opened
    finally:
        context.close()


def show(page, tab, selector):
    page.click(f"#main-tabs .rail-btn[data-tab='{tab}']")
    page.wait_for_selector(selector, state="visible", timeout=15000)
    page.wait_for_timeout(250)


class TestAFilterRendersOnTheCompactTier:
    def test_the_probe_reads_the_standard_tier_where_there_is_one(self, page):
        """The control.

        Every reading below is "this control is 28px". If the probe could
        not tell 28 from 30 - a stale layout, a hidden element measuring
        zero, a selector that matches nothing - each of those would also
        read as a number, and only this says which numbers it can tell
        apart. The context select in the upload strip is an ordinary field
        and is expected to be on the standard tier.
        """
        show(page, "files-tab", "#upload-context-id")
        assert page.evaluate(HEIGHT, "#upload-context-id") == STANDARD

    @pytest.mark.parametrize("tab,selector", FILTERS, ids=lambda v: v.strip("#"))
    def test_the_three_filters_are_28px(self, page, tab, selector):
        show(page, tab, selector)
        height = page.evaluate(HEIGHT, selector)
        assert height == COMPACT, (
            f"{selector} renders {height}px where part one puts a filter at "
            f"{COMPACT}px; the rule exists, so this is the cascade rather "
            f"than the rule"
        )


class TestOneFocusVocabulary:
    """Four rule blocks moved from `:focus` to `:focus-visible`.

    Part one asks for one focus vocabulary and not two beside each other,
    which is the whole reason for the move. What it is *not* is a change to
    what anybody sees - but that is a claim about a user agent, so the two
    tests that would break if it were false are here rather than the
    sentence being repeated.

    The design document said `.field select:focus` was "a real difference,
    and fires on a mouse click", reasoning that a `<select>` is not a
    text-entry control. That was written from the specification and never
    run. If it were true, `.field select:focus-visible` would have silently
    stopped tinting for every mouse user, and the first test below is what
    would say so.
    """

    def test_a_clicked_select_still_takes_the_tint(self, page):
        """The claim the rename rests on, asked of the browser.

        This is the case the design document called a real difference. It
        is not one here: this user agent matches `:focus-visible` on a
        mouse-clicked `<select>` as it does on a mouse-clicked text field.
        """
        show(page, "files-tab", "#upload-context-id")
        resting = page.evaluate(BORDER, "#upload-context-id")
        page.click("#upload-context-id")
        page.keyboard.press("Escape")
        page.wait_for_timeout(150)
        clicked = page.evaluate(FOCUS_STATE, "#upload-context-id")
        assert clicked[0], "the click did not focus the select"
        assert clicked[1], (
            "this browser does not match :focus-visible on a clicked "
            "<select>, so moving `.field select` onto it dropped the tint "
            "for every mouse user - which is what the design document said "
            "would happen and what this change says does not"
        )
        assert clicked[2] != resting, (
            "the select's border did not change on a mouse click"
        )

    def test_a_keyboard_focus_paints_the_select(self, page):
        """The rules still reach a select after the rename."""
        show(page, "files-tab", "#upload-context-id")
        resting = page.evaluate(BORDER, "#upload-context-id")
        # A real key press is what sets the user agent's focus-visible flag;
        # `element.focus()` does not. So focus the control before it and Tab
        # onto this one - and reach the file input through the DOM, because
        # clicking one opens a file chooser.
        page.evaluate("() => document.querySelector('#file-upload').focus()")
        page.keyboard.press("Tab")
        page.wait_for_timeout(150)
        focused = page.evaluate(FOCUS_STATE, "#upload-context-id")
        assert focused[0], "Tab did not land on the select; nothing was measured"
        assert focused[2] != resting, (
            "keyboard focus left the select's border unchanged, so the rule "
            "that tints it no longer reaches it"
        )

    def test_a_clicked_text_field_keeps_its_border_tint(self, page):
        """The assumption the change rests on, asked of the browser.

        `:focus-visible` is specified to match a text field whatever
        focused it. If that were not so here, moving the text-entry rules
        off `:focus` would have quietly dropped their tint on every mouse
        click, and nothing else in this suite would have noticed.
        """
        show(page, "contexts-tab", "#new-context-name")
        resting = page.evaluate(BORDER, "#new-context-name")
        page.click("#new-context-name")
        page.wait_for_timeout(150)
        clicked = page.evaluate(FOCUS_STATE, "#new-context-name")
        assert clicked[0], "the click did not focus the field"
        assert clicked[1], (
            "this browser does not match :focus-visible on a clicked text "
            "field, so the four rules moved onto it lost their tint for "
            "mouse users"
        )
        assert clicked[2] != resting, (
            "the field's border did not change on a mouse click, so the "
            "tint these rules exist for is not reaching a mouse user"
        )
