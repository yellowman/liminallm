"""No screen may scroll sideways on a phone.

This is one of the acceptance tests the design language sets, and running it
for the first time found a defect in the two-column settings layout. The
desktop rule writes the content column as `minmax(0, 1fr)`, which is the
form that lets a track be narrower than the things inside it. The
narrow-screen override collapsed the layout to one column and wrote that
track as plain `1fr`, whose minimum is `auto` - the min-content width of its
contents. The widest thing in Settings is the admin users table, so the
track grew to the table's natural width and took the document with it:
measured at 207px of horizontal scroll on a 390px viewport, with the table's
own `overflow-x: auto` wrapper stretched wide instead of scrolling.

The table is only in the tree for an administrator, so a non-admin session
does not reproduce it.
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
PHONE = {"width": 390, "height": 844}

TABS = (
    "chat-tab",
    "notes-tab",
    "contexts-tab",
    "files-tab",
    "artifacts-tab",
    "tools-tab",
    "insights-tab",
    "settings-tab",
)

#: An element wider than the viewport is only a defect when it makes the
#: document wider. Text that clips itself with an ellipsis reports a large
#: `scrollWidth` and is behaving correctly, so the question is asked of the
#: document, and the elements that overhang are collected only to name a
#: culprit in the failure message.
OVERFLOW = """() => {
  const doc = document.documentElement;
  const over = Math.round(doc.scrollWidth - doc.clientWidth);
  const culprits = [];
  if (over > 0) {
    document.querySelectorAll('.tab-panel.active *').forEach((el) => {
      const r = el.getBoundingClientRect();
      if (r.right > doc.clientWidth + 1 && r.width > 0) {
        culprits.push(String(el.id || el.className || el.tagName).slice(0, 34));
      }
    });
  }
  return {over, culprits: [...new Set(culprits)].slice(0, 5)};
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
def phone(browser, server):
    """One signed-in administrator on a phone-sized viewport."""
    import httpx

    email = f"narrow_{uuid.uuid4().hex[:8]}@example.com"
    resp = httpx.post(
        f"{server.base_url}/v1/auth/signup",
        json={"email": email, "password": PASSWORD},
        timeout=30,
    )
    assert resp.status_code == 201, resp.text

    # Before the browser signs in: a role change invalidates a token minted
    # before it, so promoting afterwards breaks the session under test.
    from liminallm.service.runtime import get_runtime

    get_runtime().store.update_user_role(resp.json()["data"]["user_id"], role="admin")

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
    page.wait_for_timeout(900)
    try:
        yield page
    finally:
        context.close()


class TestNoScreenScrollsSideways:
    def test_the_probe_can_see_overflow_when_there_is_some(self, phone):
        """The control.

        Every reading in the test below is a negative one, and a negative
        reading is worth nothing until the same probe has been shown to
        report a positive. The width is set through the CSSOM rather than a
        stylesheet or a `style` attribute, both of which this app's
        Content-Security-Policy refuses - an injected rule is silently
        inert, which is how an earlier attempt at this measurement compared
        three states and got one answer.
        """
        phone.click("#main-tabs .rail-btn[data-tab='chat-tab']")
        phone.wait_for_timeout(400)
        assert phone.evaluate(OVERFLOW)["over"] == 0

        phone.evaluate(
            """() => {
              const d = document.createElement('div');
              d.id = 'overflow-control';
              d.style.width = '3000px';
              d.style.height = '4px';
              document.querySelector('.tab-panel.active').appendChild(d);
            }"""
        )
        phone.wait_for_timeout(300)
        seen = phone.evaluate(OVERFLOW)
        phone.evaluate("() => document.getElementById('overflow-control').remove()")
        phone.wait_for_timeout(300)

        assert seen["over"] > 0, (
            "the probe reported no overflow with a 3000px element on the "
            "page, so it cannot see the thing the next test asserts is absent"
        )
        assert phone.evaluate(OVERFLOW)["over"] == 0, "the control did not clean up"

    def test_no_tab_makes_the_document_wider_than_the_phone(self, phone):
        offenders = {}
        for tab in TABS:
            phone.click(f"#main-tabs .rail-btn[data-tab='{tab}']")
            phone.wait_for_selector(f"#{tab}.active", state="visible")
            phone.wait_for_timeout(700)
            measured = phone.evaluate(OVERFLOW)
            if measured["over"] > 0:
                offenders[tab] = measured

        assert not offenders, (
            "these screens scroll sideways at 390px, which the design "
            f"language does not allow: {offenders}"
        )
