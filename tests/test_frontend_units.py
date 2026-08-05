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
@pytest.mark.parametrize("script", ["common.js", "markdown.js", "chat.js", "share.js", "admin.js"])
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
        "index.html": ["common.js", "markdown.js", "chat.js"],
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
