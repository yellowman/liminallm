"""The two pages share one copy of the code they both need.

chat.js and admin.js had four byte-identical helpers copy-pasted between them,
and the copies had already drifted where it mattered: the console's request
layer treated 429 as a fatal error instead of backing off, and never refreshed
an expired token on 401, so an admin session failed where a chat session
recovered. Same failure mode as the in-memory store and the synchronous cache —
two implementations, one of them not the one you were looking at.
"""

import re
from pathlib import Path

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"


def _functions(path):
    """Top-level `const name = (...) => {...}` bodies, whitespace-normalised."""
    source = (FRONTEND / path).read_text()
    out = {}
    for match in re.finditer(r"^const (\w+) = [^\n]*?=>\s*\(?\{", source, re.M):
        depth = 0
        for i in range(match.end() - 1, len(source)):
            if source[i] == "{":
                depth += 1
            elif source[i] == "}":
                depth -= 1
                if depth == 0:
                    break
        out[match.group(1)] = re.sub(r"\s+", " ", source[match.start():i + 1])
    return out


def test_no_function_is_copied_between_the_two_pages():
    chat, admin = _functions("chat.js"), _functions("admin.js")
    copied = [
        name for name in chat.keys() & admin.keys() if chat[name] == admin[name]
    ]
    assert copied == [], (
        f"identical in both pages, so it belongs in common.js: {copied}"
    )


def test_the_shared_helpers_live_in_common_js():
    common = _functions("common.js")
    for name in (
        "escapeHtml", "randomIdempotencyKey", "getCsrfToken", "authHeaders",
        "headers", "extractError", "extractErrorDetails", "envelopeError",
        "fetchWithRetry", "requestEnvelope", "tryRefreshToken",
    ):
        assert name in common, name


def test_neither_page_redeclares_a_shared_helper():
    """A local redeclaration shadows the shared one and drift starts again."""
    common = set(_functions("common.js"))
    for page in ("chat.js", "admin.js"):
        clashing = common & set(_functions(page))
        assert clashing == set(), f"{page} redeclares {clashing}"


def test_common_is_loaded_first_on_both_pages():
    """Plain scripts, so `defer` order is what makes the globals available."""
    for page, own in (("index.html", "chat.js"), ("admin.html", "admin.js")):
        html = (FRONTEND / page).read_text()
        assert html.index("/static/common.js") < html.index(f"/static/{own}")


def test_the_request_layer_kept_the_capabilities_the_console_lacked():
    """The unified version is the one that had them, not the one that did not."""
    common = (FRONTEND / "common.js").read_text()
    assert "429" in common and "Retry-After" in common, "429 backoff"
    assert "tryRefreshToken" in common and "401" in common, "token refresh"
