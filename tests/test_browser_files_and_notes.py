"""Two things the layout engine decides, and two cues that were not there.

The file list is read by comparing sizes and dates down a column, which is
only possible if they start in the same place on every row. They do that
because the heading line and the rows share one grid declaration - two grids
size their tracks independently, so an action column sized by its own
contents would leave the heading's columns somewhere else. Nothing about
that is visible from the stylesheet, and nothing about it fails loudly: the
columns simply drift and the list goes on looking like a list.

The note editor tracked whether it held unsaved edits and showed nobody, so
a note with changes looked exactly like a saved one. And deleting a note -
the one thing in this application the reader wrote themselves - asked no
question, from an unlabelled control beside Save.
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

#: Two files whose names differ in length, because a name that runs long is
#: exactly what pushes a fact line out of alignment and a column does not.
FILES = [
    ("short.txt", b"eight or so bytes here"),
    (
        "a-considerably-longer-upload-name-for-the-column.md",
        b"# heading\n\nbody text that makes this file a different size\n",
    ),
]


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

    email = f"fn_{uuid.uuid4().hex[:8]}@example.com"
    resp = httpx.post(
        f"{server.base_url}/v1/auth/signup",
        json={"email": email, "password": PASSWORD},
        timeout=30,
    )
    assert resp.status_code == 201, resp.text
    return email, resp.json()["data"]["access_token"]


def _signed_in(browser, server, email):
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


class TestTheFileListHasColumns:
    def test_the_heading_and_every_row_share_one_set_of_columns(
        self, browser, server
    ):
        import httpx

        email, token = _account(server)
        for name, body in FILES:
            upload = httpx.post(
                f"{server.base_url}/v1/files/upload",
                headers={"Authorization": f"Bearer {token}"},
                files={"file": (name, body, "text/plain")},
                timeout=60,
            )
            assert upload.status_code in (200, 201), upload.text

        context, page = _signed_in(browser, server, email)
        try:
            page.click("#main-tabs .rail-btn[data-tab='files-tab']")
            page.wait_for_selector("#files-tab.active", state="visible")
            page.wait_for_selector("#files-list .file-row", timeout=20000)
            page.wait_for_timeout(500)

            measured = page.evaluate(
                """() => {
                  const left = (el) => el
                    ? Math.round(el.getBoundingClientRect().left) : null;
                  const head = document.querySelector('.file-table-head');
                  const rows = [...document.querySelectorAll(
                    '#files-list .file-row')];
                  return {
                    head: [...head.children].map(left),
                    rows: rows.map((row) => [
                      left(row.querySelector('.row-icon')),
                      left(row.querySelector('.row-name')),
                      ...[...row.querySelectorAll('.file-fact')].map(left),
                      left(row.querySelector('.row-actions')),
                    ]),
                  };
                }"""
            )

            # The control for the measurement: five cells that are actually
            # in five different places. Without it a collapsed layout - every
            # cell at the same left - would satisfy the equality below and
            # report that the columns line up beautifully.
            assert len(measured["head"]) == 5, measured["head"]
            assert len(set(measured["head"])) == 5, (
                "the heading's five cells are not in five distinct places, so "
                f"the grid did not apply: {measured['head']}"
            )
            assert len(measured["rows"]) == len(FILES), measured["rows"]

            for row in measured["rows"]:
                assert row == measured["head"], (
                    "a row's columns start somewhere other than the heading's, "
                    "so the sizes and dates cannot be read down the column: "
                    f"row {row} against heading {measured['head']}"
                )
        finally:
            context.close()

    def test_one_page_of_files_gets_no_pager(self, browser, server):
        """Two disabled buttons are a control saying there is nowhere to go,
        in the most prominent way available."""
        import httpx

        email, token = _account(server)
        httpx.post(
            f"{server.base_url}/v1/files/upload",
            headers={"Authorization": f"Bearer {token}"},
            files={"file": ("only.txt", b"one file", "text/plain")},
            timeout=60,
        )
        context, page = _signed_in(browser, server, email)
        try:
            page.click("#main-tabs .rail-btn[data-tab='files-tab']")
            page.wait_for_selector("#files-list .file-row", timeout=20000)
            page.wait_for_timeout(400)
            assert (
                page.eval_on_selector(
                    "#files-pagination", "el => el.textContent.trim()"
                )
                == ""
            )
            # The summary still reports, so this is not passing because the
            # whole footer failed to render.
            assert "1 file" in page.eval_on_selector(
                "#files-summary-text", "el => el.textContent"
            )
        finally:
            context.close()


class TestTheNoteEditorSaysWhatStateItIsIn:
    def _with_a_note(self, browser, server):
        import httpx

        email, token = _account(server)
        resp = httpx.post(
            f"{server.base_url}/v1/notes",
            headers={"Authorization": f"Bearer {token}"},
            json={"title": "A note to edit", "content": "first line"},
            timeout=30,
        )
        assert resp.status_code in (200, 201), resp.text
        context, page = _signed_in(browser, server, email)
        page.click("#main-tabs .rail-btn[data-tab='notes-tab']")
        page.wait_for_selector("#notes-tab.active", state="visible")
        page.wait_for_selector("#note-list .note-item", timeout=20000)
        page.click("#note-list .note-item")
        page.wait_for_selector("#note-editor:not(.hidden)", timeout=15000)
        page.wait_for_timeout(400)
        return context, page

    def test_an_edited_note_says_so_and_a_saved_one_does_not(
        self, browser, server
    ):
        context, page = self._with_a_note(browser, server)
        try:
            opened = page.eval_on_selector("#note-state", "el => el.textContent")
            assert opened.strip() == "", (
                "a freshly opened note reports unsaved changes, so the cue "
                f"says nothing about state: {opened!r}"
            )

            page.fill("#note-content", "first line, and a second")
            page.wait_for_timeout(300)
            edited = page.eval_on_selector("#note-state", "el => el.textContent")
            assert "Unsaved" in edited, (
                f"an edited note does not report it: {edited!r}"
            )

            page.click("#note-save-btn")
            page.wait_for_timeout(1200)
            saved = page.eval_on_selector("#note-state", "el => el.textContent")
            assert saved.strip() == "", (
                f"the cue survives the save that cleared it: {saved!r}"
            )
        finally:
            context.close()

    def test_the_toolbar_keeps_its_own_geometry_and_the_cue_costs_nothing(
        self, browser, server
    ):
        """Two overrides a bare class name loses, and a gap nothing fills.

        `.note-toolbar` sets `flex-wrap: nowrap` and `margin: 0` over
        `.utility-strip`'s wrap and bottom margin. Written as a bare class
        it ties on specificity with `.utility-strip`, which is declared
        later in the stylesheet and therefore wins - the same trap as
        `button.ghost` outranking `.voice-btn`, and the reason the rule is
        `.utility-strip.note-toolbar`. Nothing about that is visible from
        either rule on its own.

        The cue beside the title is a flex item whether or not it has
        content, so while it was empty the head's gap was counted twice and
        the title lost 12px for ever.
        """
        context, page = self._with_a_note(browser, server)
        try:
            measured = page.evaluate(
                """() => {
                  const bar = document.querySelector('.note-toolbar');
                  const head = document.querySelector('.note-editor-head');
                  const title = document.querySelector('#note-title');
                  const cs = getComputedStyle(bar);
                  return {
                    wrap: cs.flexWrap,
                    marginBottom: cs.marginBottom,
                    headGap: Math.round(parseFloat(getComputedStyle(head).gap)),
                    titleToToolbar: Math.round(
                      bar.getBoundingClientRect().left
                      - title.getBoundingClientRect().right),
                  };
                }"""
            )
            assert measured["wrap"] == "nowrap", (
                "the toolbar wraps, so `.utility-strip` is winning the "
                f"cascade: {measured}"
            )
            assert measured["marginBottom"] == "0px", measured
            assert measured["titleToToolbar"] == measured["headGap"], (
                "an empty cue is costing a gap: the title is "
                f"{measured['titleToToolbar']}px from the toolbar where the "
                f"head's gap is {measured['headGap']}px"
            )

            # The control for that last one: the measurement has to be able
            # to see a difference, or equality above means nothing.
            page.fill("#note-content", "something, so the cue appears")
            page.wait_for_timeout(300)
            dirty = page.evaluate(
                """() => {
                  const bar = document.querySelector('.note-toolbar');
                  const title = document.querySelector('#note-title');
                  return Math.round(bar.getBoundingClientRect().left
                    - title.getBoundingClientRect().right);
                }"""
            )
            assert dirty > measured["titleToToolbar"], (
                "the gap did not grow when the cue appeared, so this probe "
                f"cannot tell the two states apart: {dirty} against "
                f"{measured['titleToToolbar']}"
            )
        finally:
            context.close()

    def test_deleting_a_note_asks_first_and_a_refusal_keeps_it(
        self, browser, server
    ):
        context, page = self._with_a_note(browser, server)
        try:
            asked = {}

            def on_dialog(dialog):
                asked["message"] = dialog.message
                dialog.dismiss()

            page.on("dialog", on_dialog)
            before = page.eval_on_selector_all(
                "#note-list .note-item", "els => els.length"
            )
            assert before == 1

            page.click("#note-delete-btn")
            page.wait_for_timeout(1200)

            assert "message" in asked, (
                "deleting a note asked nothing - the whole confirmation was a "
                "single click on a glyph"
            )
            assert "A note to edit" in asked["message"], asked["message"]
            assert (
                page.eval_on_selector_all("#note-list .note-item", "els => els.length")
                == before
            ), "the note went anyway, so the question it asked meant nothing"
        finally:
            context.close()
