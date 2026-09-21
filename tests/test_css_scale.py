"""The type scale and the radius scale are closed sets, or they are not scales.

`docs/DESIGN_LANGUAGE.md` part one names seven type sizes across ten roles and
four radius bands plus a pill. The stylesheet carried nineteen distinct
`font-size` values and sixteen distinct `border-radius` forms. Ten of the
radius forms were literals that bypassed the two tokens, and eight of those
simply restated a token, so the project rule - change the token, not the call
site - had eight call sites that would not have moved.

Nothing in CSS objects to a twentieth size. The value parses, the rule
applies, and the drift is only visible to somebody who counts. So it is
counted here.

Both checks work the same way. A value is allowed when it is on the scale, or
when it is pinned below to the exact selector that may use it. A pin is a
recorded departure, not an escape: adding a size means naming where it is
allowed, which is the sentence part two asks for anyway.

Each check is paired with a control that feeds it the defect and requires it
to report, because a guard that reads nothing passes just as quietly as one
that reads everything.
"""

from __future__ import annotations

import pathlib
import re

import pytest

tinycss2 = pytest.importorskip("tinycss2")
css_ast = pytest.importorskip("tinycss2.ast")

STYLESHEET = (
    pathlib.Path(__file__).resolve().parent.parent / "frontend" / "styles.css"
)

#: Part one's type scale, as the stylesheet may spell it. Ten roles share
#: seven sizes; the roles are a design question and the sizes are this one.
TYPE_SCALE = frozenset({"16px", "15px", "13.5px", "13px", "12px", "11.5px", "11px"})

#: Sizes that are not interface text, each pinned to the selector that may
#: use it. Part two carries the reason for each group.
TYPE_DEPARTURES = {
    # Headings inside rendered markdown. The reader is in a document, and a
    # document's heading ramp is content typography rather than chrome.
    "24px": {".bubble h1"},
    "20px": {".bubble h2"},
    "17.5px": {".bubble h3"},
    # A glyph, sized to the glyph. `×` and `+` are drawn at their optical
    # size, not at the size of the words around them.
    "22px": {".modal-close", "#note-title"},
    "18px": {"#note-new-btn"},
    # A dashboard numeral, read as a figure rather than as a sentence.
    "26px": {".figure-value"},
}

#: Radius values that need no pin: the two tokens, a square corner, a circle
#: and a true pill. A composite is allowed when every component is one of
#: these, which is what makes `0 var(--radius-sm) var(--radius-sm) 0` fine.
RADIUS_ATOMS = frozenset({"0", "var(--radius)", "var(--radius-sm)", "50%", "999px"})

#: Radius literals that are marks a few pixels across rather than surfaces,
#: and the chat bubble, which is a content shape. Pinned to their selectors.
RADIUS_DEPARTURES = {
    "12px": {".message .bubble"},
    "16px 16px 4px 16px": {".message.user .bubble"},
    "4px": {".msg-warning", ".draft-indicator"},
    "2px": {".brand .spark", ".message.streaming .bubble::after"},
    "1px": {".tick-mark"},
    "0 2px 2px 0": {
        ".rail-btn.active::before",
        ".row.selected::before, .conversation-item.active::before, "
        ".note-item.active::before",
    },
}

#: `var(` splits into three tokens, so a composite is matched rather than
#: tokenized: this keeps `var(--radius-sm)` whole.
_ATOM_RE = re.compile(r"var\(--[\w-]+\)|\S+")


def qualified_rules(nodes):
    """Every qualified rule, including those nested inside at-rules.

    Media queries hold rules too, and a scale that is obeyed at one width and
    abandoned at another is not obeyed.
    """
    for node in nodes:
        if isinstance(node, css_ast.QualifiedRule):
            yield node
        elif isinstance(node, css_ast.AtRule) and node.content:
            yield from qualified_rules(
                tinycss2.parse_rule_list(
                    node.content, skip_whitespace=True, skip_comments=True
                )
            )


def declarations(css: str, name: str):
    """Every `name` declaration, as (value, selector, line)."""
    top = tinycss2.parse_stylesheet(css, skip_whitespace=True, skip_comments=True)
    for rule in qualified_rules(top):
        selector = " ".join(tinycss2.serialize(rule.prelude).split())
        for node in tinycss2.parse_declaration_list(
            rule.content, skip_whitespace=True, skip_comments=True
        ):
            if isinstance(node, css_ast.Declaration) and node.lower_name == name:
                value = " ".join(tinycss2.serialize(node.value).split())
                yield value, selector, node.source_line


def off_scale_type(css: str):
    """Type sizes that are neither on the scale nor pinned to this selector."""
    return [
        (value, selector, line)
        for value, selector, line in declarations(css, "font-size")
        if value not in TYPE_SCALE
        and selector not in TYPE_DEPARTURES.get(value, ())
    ]


def off_band_radius(css: str):
    """Radius values that are neither built from the atoms nor pinned here."""
    found = []
    for value, selector, line in declarations(css, "border-radius"):
        if all(atom in RADIUS_ATOMS for atom in _ATOM_RE.findall(value)):
            continue
        if selector in RADIUS_DEPARTURES.get(value, ()):
            continue
        found.append((value, selector, line))
    return found


#: `:focus` and neither of the two pseudo-classes that begin with it.
_BARE_FOCUS_RE = re.compile(r":focus(?![-\w])")


def bare_focus_selectors(css: str):
    """Selectors on `:focus` rather than the file's `:focus-visible`."""
    top = tinycss2.parse_stylesheet(css, skip_whitespace=True, skip_comments=True)
    found = []
    for rule in qualified_rules(top):
        prelude = " ".join(tinycss2.serialize(rule.prelude).split())
        for selector in prelude.split(","):
            selector = selector.strip()
            if _BARE_FOCUS_RE.search(selector):
                found.append((":focus", selector, rule.prelude[0].source_line))
    return found


def report(found):
    return "\n".join(
        f"  line {line}: {value} on {selector}" for value, selector, line in found
    )


@pytest.fixture(scope="module")
def css():
    return STYLESHEET.read_text(encoding="utf-8")


class TestTheTypeScaleIsClosed:
    def test_every_size_is_on_the_scale_or_pinned(self, css):
        found = off_scale_type(css)
        assert not found, (
            "font sizes that are neither on part one's scale nor recorded as a "
            "departure:\n" + report(found)
        )

    def test_the_check_reports_a_size_that_is_off_the_scale(self):
        """The control. An unpinned off-scale value must be reported."""
        found = off_scale_type(".invented { font-size: 12.5px; }")
        assert found == [("12.5px", ".invented", 1)], found

    def test_a_pin_covers_only_the_selector_it_names(self):
        """A departure is pinned to a place, or it is not a departure.

        Without this, one recorded exception would license the same size
        everywhere, which is how a scale becomes a list of sizes in use.
        """
        assert not off_scale_type(".bubble h1 { font-size: 24px; }")
        assert off_scale_type(".sidebar-title { font-size: 24px; }")


class TestThereIsOneFocusVocabulary:
    """Part one: "Do not keep a global focus system and per-component
    systems beside it."

    The per-component rules are allowed and are not the problem - the
    comment above the global ring says a component may add a border or a
    fill on focus, and four rule blocks do. The problem was the selector.
    Seven of them said `:focus` while the ring said `:focus-visible`, so a
    reader of the file had two spellings for one idea and no way to tell
    which was deliberate.

    This is the enforceable half of the rule. That the two selectors behave
    alike in a real browser is a separate question and is asked of one in
    `tests/test_browser_control_tier.py`.
    """

    def test_no_selector_says_plain_focus(self, css):
        found = bare_focus_selectors(css)
        assert not found, (
            "selectors on `:focus` where the stylesheet's focus vocabulary "
            "is `:focus-visible`:\n" + report(found)
        )

    def test_the_check_reports_one(self):
        """The control. `grep -c ':focus'` also counts the other two, which
        is how an earlier count of these turned six into seventeen."""
        found = bare_focus_selectors(".field select:focus { border: 0; }")
        assert found == [(":focus", ".field select:focus", 1)], found

    def test_focus_visible_and_focus_within_are_not_reported(self):
        assert not bare_focus_selectors(
            ":focus-visible { outline: 0; }\n"
            ".turn-rail:focus-within .turn-tick { opacity: 1; }"
        )


class TestTheRadiusScaleIsClosed:
    def test_every_radius_is_a_token_or_pinned(self, css):
        found = off_band_radius(css)
        assert not found, (
            "radius values that neither use the tokens nor are recorded as a "
            "departure:\n" + report(found)
        )

    def test_a_literal_that_restates_a_token_is_reported(self):
        """The eight that made the project's own rule unenforceable.

        `6px` and `var(--radius-sm)` render identically, so nothing looks
        wrong until somebody changes the token and eight call sites stay
        where they were.
        """
        found = off_band_radius(".pane-close { border-radius: 6px; }")
        assert found == [("6px", ".pane-close", 1)], found

    def test_a_composite_built_from_tokens_is_accepted(self):
        """Not every multi-value radius is drift."""
        assert not off_band_radius(
            ".settings-index a { border-radius: 0 var(--radius-sm)"
            " var(--radius-sm) 0; }"
        )

    def test_a_composite_hiding_a_literal_is_still_reported(self):
        """One literal among tokens is the easiest form to miss by eye."""
        found = off_band_radius(
            ".x { border-radius: 0 var(--radius-sm) 9px 0; }"
        )
        assert found == [("0 var(--radius-sm) 9px 0", ".x", 1)], found
