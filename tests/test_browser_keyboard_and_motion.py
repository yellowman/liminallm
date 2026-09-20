"""The interaction contract the interface claimed and did not keep.

Three primary navigation controls were static containers listening for
clicks: a note in the vault list, a conversation in the chat pane, and a
vault search result. A reader who does not use a mouse could not open a
note. CI was green the whole time, because nothing asked.

Alongside them, the focus indicator. Four rules set `outline: none` and drew
`box-shadow: 0 0 0 2px rgba(14, 138, 109, 0.10)` instead - the accent at a
tenth alpha, which measures 1.13:1 against every surface in the file where
WCAG 2.2 asks for 3.0:1. That is worse than leaving the browser alone: it
removed a working indicator and replaced it with one nobody can see.

`:focus-visible` is not set by a programmatic `.focus()`. Only a real key
press sets it, so every test here tabs.
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

#: The relative luminance contrast of the focus ring against what it is
#: drawn on. WCAG 2.2 asks non-text indicators for 3.0:1.
CONTRAST = """(sel) => {
  const lin = (c) => {
    c /= 255;
    return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
  };
  const lum = ([r, g, b]) => 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b);
  const parse = (s) => (s.match(/[\\d.]+/g) || []).map(Number);
  const over = (fg, bg) => {
    const a = fg.length > 3 ? fg[3] : 1;
    return [0, 1, 2].map((i) => a * fg[i] + (1 - a) * bg[i]);
  };
  const ratio = (a, b) => {
    const [x, y] = [lum(a), lum(b)].sort((p, q) => q - p);
    return (x + 0.05) / (y + 0.05);
  };
  const el = sel ? document.querySelector(sel) : document.activeElement;
  if (!el) return null;
  const s = getComputedStyle(el);
  const page = parse(getComputedStyle(document.body).backgroundColor).slice(0, 3);
  const own = parse(s.backgroundColor);
  const base = s.backgroundColor === 'rgba(0, 0, 0, 0)' ? page : own.slice(0, 3);
  const outline = parse(s.outlineColor).slice(0, 4);
  return {
    id: el.id || el.className,
    style: s.outlineStyle,
    width: s.outlineWidth,
    offset: s.outlineOffset,
    contrast: s.outlineStyle === 'none'
      ? null
      : Number(ratio(over(outline, base), base).toFixed(2)),
  };
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


def _account(server):
    import httpx

    email = f"kbd_{uuid.uuid4().hex[:8]}@example.com"
    resp = httpx.post(
        f"{server.base_url}/v1/auth/signup",
        json={"email": email, "password": PASSWORD},
        timeout=30,
    )
    assert resp.status_code == 201, resp.text
    return email, resp.json()["data"]["access_token"]


def _signed_in(browser, server, email, **context_args):
    context = browser.new_context(viewport=DESKTOP, **context_args)
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


def _seed_notes(server, token, titles):
    import httpx

    for title in titles:
        resp = httpx.post(
            f"{server.base_url}/v1/notes",
            headers={"Authorization": f"Bearer {token}"},
            json={"title": title, "content": f"{title} body, about compute."},
            timeout=30,
        )
        assert resp.status_code in (200, 201), resp.text


def _tab_to(page, selector, limit=60):
    """Tab until focus lands inside `selector`. Returns how many it took."""
    page.evaluate("() => document.activeElement && document.activeElement.blur()")
    for pressed in range(1, limit + 1):
        page.keyboard.press("Tab")
        if page.evaluate(
            "(sel) => !!(document.activeElement "
            "&& document.activeElement.closest(sel))",
            selector,
        ):
            return pressed
    return None


class TestAKeyboardCanReachWhatAMouseCan:
    def test_a_note_is_reachable_and_opens_on_enter(self, browser, server):
        email, token = _account(server)
        _seed_notes(server, token, ["Compute is the governable surface"])
        context, page = _signed_in(browser, server, email)
        try:
            page.click("#main-tabs .rail-btn[data-tab='notes-tab']")
            page.wait_for_selector("#note-list .note-item", timeout=20000)
            page.wait_for_timeout(400)

            pressed = _tab_to(page, "#note-list .note-item")
            assert pressed is not None, (
                "no amount of tabbing reaches a note, so opening one - the "
                "pane's primary action - is available only to a pointer"
            )
            page.keyboard.press("Enter")
            page.wait_for_selector("#note-editor:not(.hidden)", timeout=15000)
            assert page.eval_on_selector("#note-title", "el => el.value") == (
                "Compute is the governable surface"
            )
        finally:
            context.close()

    def test_a_conversation_is_reachable_and_opens_on_enter(
        self, browser, server
    ):
        """The chat pane's rows were `div`s with a click listener."""
        import httpx

        email, token = _account(server)
        resp = httpx.post(
            f"{server.base_url}/v1/conversations",
            headers={"Authorization": f"Bearer {token}"},
            json={"title": "The pause debate"},
            timeout=30,
        )
        assert resp.status_code in (200, 201), resp.text

        context, page = _signed_in(browser, server, email)
        try:
            page.click("#main-tabs .rail-btn[data-tab='chat-tab']")
            page.wait_for_selector(
                "#conversation-list .conversation-item", timeout=20000
            )
            page.wait_for_timeout(400)

            pressed = _tab_to(page, "#conversation-list .conversation-item")
            assert pressed is not None, (
                "no amount of tabbing reaches a conversation, so opening one "
                "is available only to a pointer"
            )
            page.keyboard.press("Enter")
            page.wait_for_timeout(1200)
            assert page.eval_on_selector_all(
                "#conversation-list .conversation-item.active", "els => els.length"
            ) == 1, "Enter on a conversation row did not open it"
        finally:
            context.close()

    def test_a_search_result_is_reachable_and_opens_the_note(
        self, browser, server
    ):
        email, token = _account(server)
        _seed_notes(server, token, ["Verification is the bottleneck"])
        context, page = _signed_in(browser, server, email)
        try:
            page.click("#main-tabs .rail-btn[data-tab='notes-tab']")
            page.wait_for_selector("#note-list .note-item", timeout=20000)
            page.fill("#note-search-input", "verification")
            page.wait_for_selector(
                "#note-search-results .note-search-hit", timeout=20000
            )

            pressed = _tab_to(page, "#note-search-results .note-search-hit")
            assert pressed is not None, (
                "a vault search result cannot be reached by keyboard, in the "
                "one list a reader arrives at by typing"
            )
            page.keyboard.press("Enter")
            page.wait_for_selector("#note-editor:not(.hidden)", timeout=15000)
            assert page.eval_on_selector("#note-title", "el => el.value") == (
                "Verification is the bottleneck"
            )
            assert page.eval_on_selector(
                "#note-search-results", "el => el.classList.contains('hidden')"
            ), "opening a result left the result list over the editor"
        finally:
            context.close()

    def test_a_result_carries_its_rank_and_what_kind_of_thing_it_is(
        self, browser, server
    ):
        """The server sends a 1-based position precisely so a client can
        show one; the frontend used to drop it and render a bare title."""
        email, token = _account(server)
        _seed_notes(
            server,
            token,
            ["Compute accounting", "Compute thresholds", "Compute custody"],
        )
        context, page = _signed_in(browser, server, email)
        try:
            page.click("#main-tabs .rail-btn[data-tab='notes-tab']")
            page.wait_for_selector("#note-list .note-item", timeout=20000)
            page.fill("#note-search-input", "compute")
            page.wait_for_selector(
                "#note-search-results .note-search-hit", timeout=20000
            )
            page.wait_for_timeout(400)

            rows = page.eval_on_selector_all(
                "#note-search-results .note-search-hit",
                """els => els.map(e => ({
                     rank: (e.querySelector('.hit-rank') || {}).textContent,
                     title: (e.querySelector('.note-item-title') || {}).textContent,
                     facts: (e.querySelector('.hit-facts') || {}).textContent,
                     excerpt: (e.querySelector('.note-search-excerpt') || {}).textContent,
                   }))""",
            )
            assert len(rows) >= 2, rows
            assert [r["rank"] for r in rows][:2] == ["01", "02"], rows
            for row in rows:
                assert row["title"], row
                assert "Note" in (row["facts"] or ""), row
                assert row["excerpt"], row
        finally:
            context.close()


class TestTheFocusRingCanBeSeen:
    def test_a_tabbed_control_draws_a_ring_at_three_to_one(
        self, browser, server
    ):
        email, _token = _account(server)
        context, page = _signed_in(browser, server, email)
        try:
            page.click("#main-tabs .rail-btn[data-tab='files-tab']")
            page.wait_for_selector("#refresh-files-btn", timeout=15000)
            page.wait_for_timeout(500)

            # An icon button specifically. `.icon-btn:focus-visible` is where
            # the invisible ring was declared, so a control that never had a
            # focus rule would pass this on the browser's own outline and
            # say nothing about the defect.
            landed = _tab_to(page, ".icon-btn")
            assert landed is not None, "no icon button is reachable by tab"
            assert page.evaluate(
                "() => document.activeElement.classList.contains('icon-btn')"
            ), "focus is inside an icon button but not on it"

            ring = page.evaluate(CONTRAST, None)
            assert ring is not None and ring["style"] != "none", (
                f"the focused control draws no outline at all: {ring}"
            )
            assert ring["contrast"] is not None and ring["contrast"] >= 3.0, (
                "the focus ring is below the 3.0:1 WCAG 2.2 asks of a "
                f"non-text indicator: {ring}"
            )
        finally:
            context.close()

    def test_no_rule_turns_the_outline_off(self, browser, server):
        """The guard.

        The measurement above reads one control. This asks the stylesheet
        whether anything anywhere still removes the indicator, which is how
        the defect got in: four separate rules, each reasonable on its own.
        """
        email, _token = _account(server)
        context, page = _signed_in(browser, server, email)
        try:
            css = page.evaluate(
                "async () => (await (await fetch('/static/styles.css')).text())"
            )
            body = __import__("re").sub(r"/\*.*?\*/", " ", css, flags=16)
            assert "outline: none" not in body and "outline:none" not in body, (
                "a rule still removes the focus outline"
            )
        finally:
            context.close()


class TestMotionCanBeTurnedDown:
    def test_nothing_animates_for_ever_under_reduced_motion(
        self, browser, server
    ):
        """Three ran without end: the streaming pulse, the caret and the
        typing dots. A reader asking the system for less motion is asking
        about exactly those."""
        email, _token = _account(server)
        context, page = _signed_in(
            browser, server, email, reduced_motion="reduce"
        )
        try:
            names = page.evaluate(
                """() => {
                  const probe = document.createElement('div');
                  probe.innerHTML =
                    '<button class="voice-btn recording"></button>' +
                    '<div class="message streaming"><div class="bubble"></div></div>' +
                    '<div class="typing-dots"><span></span></div>';
                  document.body.appendChild(probe);
                  const read = (sel, pseudo) => getComputedStyle(
                    probe.querySelector(sel), pseudo || null).animationName;
                  const out = {
                    pulse: read('.voice-btn.recording'),
                    caret: read('.message.streaming .bubble', '::after'),
                    dots: read('.typing-dots span'),
                  };
                  probe.remove();
                  return out;
                }"""
            )
            assert set(names.values()) == {"none"}, (
                f"these still animate under prefers-reduced-motion: {names}"
            )
        finally:
            context.close()

    def test_the_probe_sees_them_running_otherwise(self, browser, server):
        """The control. Without it, a stylesheet that deleted the animations
        outright would pass the test above while saying nothing about the
        media query."""
        email, _token = _account(server)
        context, page = _signed_in(
            browser, server, email, reduced_motion="no-preference"
        )
        try:
            names = page.evaluate(
                """() => {
                  const probe = document.createElement('div');
                  probe.innerHTML =
                    '<button class="voice-btn recording"></button>' +
                    '<div class="message streaming"><div class="bubble"></div></div>' +
                    '<div class="typing-dots"><span></span></div>';
                  document.body.appendChild(probe);
                  const read = (sel, pseudo) => getComputedStyle(
                    probe.querySelector(sel), pseudo || null).animationName;
                  const out = {
                    pulse: read('.voice-btn.recording'),
                    caret: read('.message.streaming .bubble', '::after'),
                    dots: read('.typing-dots span'),
                  };
                  probe.remove();
                  return out;
                }"""
            )
            assert "none" not in set(names.values()), (
                f"nothing animates even without the preference: {names}"
            )
        finally:
            context.close()


class TestAnIconControlIsOneSize:
    def test_compact_does_not_shrink_the_target(self, browser, server):
        """`.icon-btn.compact` was 24px, a fourth geometry the standard does
        not have and a target below what a finger reliably finds. Compact is
        a smaller glyph in a tighter context, not a smaller control."""
        email, _token = _account(server)
        context, page = _signed_in(browser, server, email)
        try:
            sizes = page.evaluate(
                """() => {
                  const probe = document.createElement('div');
                  probe.innerHTML =
                    '<button class="icon-btn"><svg viewBox="0 0 20 20"></svg></button>' +
                    '<button class="icon-btn compact"><svg viewBox="0 0 20 20"></svg></button>';
                  document.body.appendChild(probe);
                  const box = (el) => {
                    const r = el.getBoundingClientRect();
                    return [Math.round(r.width), Math.round(r.height)];
                  };
                  const [a, b] = [...probe.querySelectorAll('button')].map(box);
                  probe.remove();
                  return {plain: a, compact: b};
                }"""
            )
            assert sizes["plain"] == [28, 28], sizes
            assert sizes["compact"] == [28, 28], (
                f"a compact icon button has a smaller hit area: {sizes}"
            )
        finally:
            context.close()


class TestTheInterfaceShipsItsOwnTypefaces:
    """The tokens named Inter and JetBrains Mono and nothing shipped them.

    A `font-family` is a request, not a guarantee. With no `@font-face` and
    no files in the repository, both names were asking the reader's own
    machine - and on Linux that usually means neither is there, so the
    typography this work was partly about was one most readers never saw.
    """

    def test_both_families_load_from_this_repository(self, browser, server):
        email, _token = _account(server)
        context, page = _signed_in(browser, server, email)
        try:
            page.wait_for_timeout(1200)
            state = page.evaluate(
                """async () => {
                  // A face is fetched on first use, so asking whether the
                  // monospace has loaded before anything is set in it
                  // measures the render, not the stylesheet. Put both on the
                  // page, then wait.
                  const probe = document.createElement('div');
                  probe.innerHTML =
                    '<span style="font-family:Inter;font-weight:600">Aa</span>' +
                    '<code class="monospace">Aa</code>';
                  document.body.appendChild(probe);
                  await document.fonts.load('500 14px Inter');
                  await document.fonts.load('400 12px "JetBrains Mono"');
                  await document.fonts.ready;
                  probe.remove();
                  const faces = [...document.fonts].map(
                    f => `${f.family}|${f.weight}|${f.status}`);
                  return {
                    faces,
                    interLoaded: document.fonts.check('500 14px Inter'),
                    monoLoaded: document.fonts.check('400 12px "JetBrains Mono"'),
                    body: getComputedStyle(document.body).fontFamily.split(',')[0],
                  };
                }"""
            )
            assert state["interLoaded"], (
                f"Inter is named by the tokens and not loaded: {state}"
            )
            assert state["monoLoaded"], (
                f"JetBrains Mono is named by the tokens and not loaded: {state}"
            )
            assert state["body"].strip('"') == "Inter", state

            # Every weight the stylesheet asks for has a file behind it, so
            # nothing is synthesised into a letterform the typeface does not
            # have.
            inter = {f.split("|")[1] for f in state["faces"] if "Inter" in f}
            assert {"400", "500", "600"} <= inter, state["faces"]
        finally:
            context.close()

    def test_no_font_is_fetched_from_a_third_party(self, browser, server):
        """Self-hosted is the point. A CDN link would work and would also
        tell someone else's server who is reading, on every page load."""
        email, _token = _account(server)
        context = browser.new_context(viewport=DESKTOP)
        page = context.new_page()
        external = []
        page.on(
            "request",
            lambda r: external.append(r.url)
            if r.resource_type == "font" and "127.0.0.1" not in r.url
            and "localhost" not in r.url
            else None,
        )
        try:
            page.goto(f"{server.base_url}/", wait_until="domcontentloaded")
            page.fill("#email", email)
            page.fill("#password", PASSWORD)
            page.click("#auth-form button[type=submit]")
            page.wait_for_selector("#main-tabs", state="visible")
            page.wait_for_timeout(1500)
            assert not external, f"fonts fetched from elsewhere: {external}"
        finally:
            context.close()
