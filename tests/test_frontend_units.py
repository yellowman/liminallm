"""The frontend's testable core, driven from the Python suite.

The markdown renderer is the XSS boundary for every assistant message — its
real tests live in tests/frontend/*.test.mjs and run under node; this file
makes the Python suite run them, so a frontend regression fails CI the same
way a backend one does. The share-page test pins the script manifest: the
page used to load chat.js without common.js, and renderMarkdown's very first
call (escapeHtml) was a ReferenceError — no shared conversation could render
an assistant message.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_the_node_suite_passes():
    suites = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "tests" / "frontend").glob("*.test.mjs"))
    assert suites, "no frontend test suites found"
    proc = subprocess.run(
        ["node", "--test", *suites],
        cwd=ROOT, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, f"node tests failed:\n{proc.stdout}\n{proc.stderr}"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
@pytest.mark.parametrize("script", ["common.js", "markdown.js", "auth.js", "admin-tab.js",
                                    "notes.js", "chat.js", "share.js", "admin.js"])
def test_every_frontend_script_parses(script):
    proc = subprocess.run(
        ["node", "--check", f"frontend/{script}"],
        cwd=ROOT, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, proc.stderr


def _scripts_of(page: str) -> list[str]:
    html = (ROOT / "frontend" / page).read_text()
    return re.findall(r'<script[^>]+src="/static/([^"]+)"', html)


def test_each_page_loads_its_dependencies_in_order():
    """Scripts share one top-level scope; defer preserves order. A page that
    lists a dependent before its dependency ships a ReferenceError."""
    for page, required in {
        "index.html": ["common.js", "markdown.js", "auth.js", "admin-tab.js",
                       "notes.js", "chat.js"],
        "share.html": ["common.js", "markdown.js", "share.js"],
    }.items():
        scripts = _scripts_of(page)
        positions = [scripts.index(s) for s in required]
        assert positions == sorted(positions), f"{page} loads {scripts} — wrong order"


def test_the_share_page_does_not_ship_the_whole_app():
    """share.html once pulled in all of chat.js for one function — and still
    broke, because chat.js's renderer needs common.js, which the page never
    loaded. The share page carries exactly what it uses."""
    scripts = _scripts_of("share.html")
    assert "chat.js" not in scripts
    assert set(scripts) == {"common.js", "markdown.js", "share.js"}


def test_share_js_uses_nothing_beyond_its_declared_dependencies():
    """Whatever share.js references must be defined in common.js, markdown.js,
    or itself — the page loads nothing else."""
    provided = set()
    for f in ("common.js", "markdown.js"):
        provided.update(re.findall(r"^const (\w+)", (ROOT / "frontend" / f).read_text(), re.M))
    share = (ROOT / "frontend" / "share.js").read_text()
    used = {name for name in ("renderMarkdown", "escapeHtml", "MSG_COPY_BUTTON_HTML",
                              "requestEnvelope", "fetchWithRetry") if name in share}
    missing = {u for u in used if u not in provided}
    assert not missing, f"share.js uses {missing} which no loaded script defines"


def test_the_chat_page_scripts_parse_as_one_scope(tmp_path):
    """Deferred classic scripts share one global lexical scope, so a name
    declared twice across files is a SyntaxError that kills the second script
    in the browser. Concatenating in manifest order and parsing reproduces
    that check at test time."""
    if shutil.which("node") is None:
        pytest.skip("node not installed")
    scripts = _scripts_of("index.html")
    concat = "\n;\n".join(
        (ROOT / "frontend" / s).read_text() for s in scripts
    )
    bundle = tmp_path / "bundle.js"
    bundle.write_text(concat)
    proc = subprocess.run(
        ["node", "--check", str(bundle)], capture_output=True, text=True, timeout=30
    )
    assert proc.returncode == 0, f"cross-file conflict:\n{proc.stderr}"


def test_extracted_modules_define_but_never_execute():
    """The split is safe because the moved files only define — wiring stays in
    chat.js's DOMContentLoaded path. A top-level call added to an extracted
    module would run before chat.js's helpers exist."""
    import re as _re

    for name in ("auth.js", "admin-tab.js", "notes.js", "markdown.js"):
        src = (ROOT / "frontend" / name).read_text()
        for i, line in enumerate(src.splitlines(), 1):
            if not line or line.startswith((" ", "//", "const ", "let ", "var ",
                                            "function ", "async function ", "}",
                                            ")", "]", "/*", " *", "*/")):
                continue
            assert not _re.match(r"[A-Za-z_$][\w$]*\s*\(", line), (
                f"{name}:{i} executes at load: {line[:80]}"
            )


def test_the_chat_column_is_for_chat():
    """Upload-knowledge and the file browser live in the Files tab. They are
    persistent panels, not conversation state; pinned above #messages (which
    is its own scroll container) they cost the chat column ~300px that never
    scrolled away."""
    html = (ROOT / "frontend" / "index.html").read_text()

    chat = re.search(r'<section class="tab-panel active" id="chat-tab".*?</section>\s*<!-- Files Tab -->',
                     html, re.S)
    assert chat, "chat-tab or Files tab marker missing"
    assert 'id="upload-section"' not in chat.group(0)
    assert 'id="files-section"' not in chat.group(0)

    files = re.search(r'id="files-tab".*?id="contexts-tab"', html, re.S)
    assert files, "files-tab missing"
    assert 'id="upload-section"' in files.group(0)
    assert 'id="files-section"' in files.group(0)
    assert 'data-tab="files-tab"' in html, "Files tab has no nav button"
