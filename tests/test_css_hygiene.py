"""Selectors that cannot match anything, and rules that silently merged.

Both defects shipped on the branch that migrated the workspace, and neither
is the kind a brace-balance check finds:

* `65a0566` left `.context-pane __DEAD_NAME__` and `__DEAD_DESC__` behind -
  a rename that was started and not finished. They are element selectors for
  elements with those names, which is valid CSS for an element that will
  never exist, so nothing complained and the rules simply never applied.
* The same commit removed a rule body and left `.context-card,` above the
  next selector. CSS does not error on that. It reads the comma as a
  selector list and gives the following rule an extra selector, so the
  declarations quietly apply to something they were never written for.

A parser is the substrate rather than the answer: both of these parse. The
checks below are about intent, which is why each one is paired with a
control that feeds it the defect and requires it to report.
"""

from __future__ import annotations

import pathlib
import re

import pytest

#: Reached through `importorskip` rather than a plain import: a module-scope
#: import of something a lane does not install is a collection error, and one
#: of those aborts the whole run before any marker deselects anything. The
#: dev extra declares it, so the lanes that matter run these rather than skip.
tinycss2 = pytest.importorskip("tinycss2")
css_ast = pytest.importorskip("tinycss2.ast")

STYLESHEET = (
    pathlib.Path(__file__).resolve().parent.parent / "frontend" / "styles.css"
)

#: Functional pseudo-classes whose arguments are themselves selectors. The
#: others take keywords or numbers - `:nth-child(even)` would otherwise
#: report `even` as an element.
SELECTOR_FUNCTIONS = {"not", "is", "where", "has", "matches"}

HTML_ELEMENTS = frozenset(
    """
    a abbr address area article aside audio b base bdi bdo blockquote body br
    button canvas caption cite code col colgroup data datalist dd del details
    dfn dialog div dl dt em embed fieldset figcaption figure footer form h1 h2
    h3 h4 h5 h6 head header hgroup hr html i iframe img input ins kbd label
    legend li link main map mark menu meta meter nav noscript object ol optgroup
    option output p param picture pre progress q rp rt ruby s samp script search
    section select slot small source span strong style sub summary sup table
    tbody td template textarea tfoot th thead time title tr track u ul var video
    wbr
    """.split()
)

SVG_ELEMENTS = frozenset(
    """
    circle clipPath defs ellipse foreignObject g image line linearGradient
    marker mask path pattern polygon polyline radialGradient rect stop svg
    symbol text tspan use
    """.lower().split()
)

KNOWN_ELEMENTS = HTML_ELEMENTS | SVG_ELEMENTS


def qualified_rules(nodes):
    """Every qualified rule, including those nested inside at-rules."""
    for node in nodes:
        if isinstance(node, css_ast.QualifiedRule):
            yield node
        elif isinstance(node, css_ast.AtRule) and node.content:
            yield from qualified_rules(
                tinycss2.parse_rule_list(
                    node.content, skip_whitespace=True, skip_comments=True
                )
            )


def _walk_prelude(nodes, found):
    previous = None
    for node in nodes:
        if isinstance(node, css_ast.IdentToken):
            # `.name` is a class and `:name` a pseudo-class; an id is its own
            # token type and an attribute lives in a bracket block. An ident
            # reached any other way is naming an element.
            literal = isinstance(previous, css_ast.LiteralToken)
            if not (literal and previous.value in (".", ":")):
                found.append((node.lower_value, node.source_line))
        elif isinstance(node, css_ast.FunctionBlock):
            if node.lower_name in SELECTOR_FUNCTIONS:
                _walk_prelude(node.arguments, found)
        if not isinstance(node, css_ast.WhitespaceToken):
            previous = node
    return found


def type_selectors(css: str):
    """Every element name the stylesheet selects on, with its line."""
    top = tinycss2.parse_stylesheet(css, skip_whitespace=True, skip_comments=True)
    found: list[tuple[str, int]] = []
    for rule in qualified_rules(top):
        _walk_prelude(rule.prelude, found)
    return found


def unknown_type_selectors(css: str):
    """Element names that no HTML or SVG document can contain."""
    return sorted(
        {
            (name, line)
            for name, line in type_selectors(css)
            if name not in KNOWN_ELEMENTS
        }
    )


def merged_selector_lists(css: str):
    """Selector lists that span a blank line.

    A removed rule body leaves its selector and comma attached to whatever
    follows. Nobody writes a blank line inside one selector list, so a
    prelude crossing one is the signature of two rules that became one.
    """
    lines = css.splitlines()
    top = tinycss2.parse_stylesheet(css, skip_whitespace=True, skip_comments=False)
    merged = []
    for rule in qualified_rules(top):
        positions = [
            node.source_line
            for node in rule.prelude
            if not isinstance(node, (css_ast.WhitespaceToken, css_ast.Comment))
        ]
        if not positions:
            continue
        first, last = min(positions), max(positions)
        blank = [
            n
            for n in range(first, last)
            if n <= len(lines) and not lines[n - 1].strip()
        ]
        if blank:
            merged.append((tinycss2.serialize(rule.prelude).strip()[:60], first))
    return merged


@pytest.fixture(scope="module")
def stylesheet() -> str:
    return STYLESHEET.read_text(encoding="utf-8")


class TestTheStylesheetSelectsOnlyThingsThatCanExist:
    def test_every_type_selector_names_a_real_element(self, stylesheet):
        unknown = unknown_type_selectors(stylesheet)
        assert not unknown, (
            "these selectors name elements that cannot exist, so the rules "
            "under them never apply: "
            + ", ".join(f"{name!r} (line {line})" for name, line in unknown)
        )

    def test_no_selector_list_spans_a_blank_line(self, stylesheet):
        merged = merged_selector_lists(stylesheet)
        assert not merged, (
            "a selector list crossing a blank line is what a removed rule "
            "body leaves behind - the declarations below now apply to the "
            "leftover selector too: "
            + ", ".join(f"{prelude!r} (line {line})" for prelude, line in merged)
        )


class TestEachCheckReportsTheDefectItExistsFor:
    """Passing on a clean stylesheet is the result a check that observes
    nothing also gives. These feed each one the real defect."""

    def test_the_dead_element_selector_is_reported(self):
        # Verbatim from 65a0566, which shipped it.
        found = unknown_type_selectors(
            ".context-pane __DEAD_NAME__ {\n  font-size: 12px;\n}\n"
        )
        assert [name for name, _ in found] == ["__dead_name__"]

    def test_the_dangling_comma_is_reported(self):
        # The shape 65a0566 left: a selector and its comma outliving the
        # body that was removed from under them.
        merged = merged_selector_lists(
            ".context-card,\n"
            "\n"
            "/* The pane reads a step quieter than the workspace. */\n"
            ".context-pane .row-name {\n"
            "  font-size: 12px;\n"
            "}\n"
        )
        assert len(merged) == 1, merged

    def test_a_clean_stylesheet_reports_nothing(self):
        """The control's control: neither check fires on correct CSS, so a
        green run on the real file means something."""
        clean = (
            "a,\n"
            "button.primary,\n"
            "svg .row-icon {\n"
            "  color: red;\n"
            "}\n"
            "@media (max-width: 640px) {\n"
            "  li:nth-child(even) > p:not(.quiet) {\n"
            "    display: none;\n"
            "  }\n"
            "}\n"
        )
        assert unknown_type_selectors(clean) == []
        assert merged_selector_lists(clean) == []

    def test_a_multi_line_selector_list_is_not_reported(self):
        """Selector lists span lines all over this stylesheet. Only a blank
        line inside one is the defect, so the check must tell them apart."""
        assert (
            merged_selector_lists(
                ".contexts-list,\n.tools-list,\n.artifacts-list {\n  gap: 8px;\n}\n"
            )
            == []
        )


class TestTheCheckSeesInsideAtRules:
    def test_a_dead_selector_nested_in_a_media_query_is_reported(self):
        """This stylesheet has two media queries. A check that only walked
        the top level would pass over whatever is inside them."""
        found = unknown_type_selectors(
            "@media (max-width: 640px) {\n"
            "  .pane __DEAD_DESC__ { display: none; }\n"
            "}\n"
        )
        assert [name for name, _ in found] == ["__dead_desc__"]


#: Class names no file in `frontend/` spells out, because they are built at
#: runtime. `admin.js` writes `setting-row is-${field.type}`, so the type
#: names only ever exist as a template. A text search cannot see them, and a
#: check that deleted what it could not see would have taken live code - it
#: nearly did.
BUILT_AT_RUNTIME = frozenset({"is-bool", "is-text"})

FRONTEND = STYLESHEET.parent


def defined_class_names(css: str) -> set:
    """Every class name the stylesheet styles, comments excluded.

    Excluded because this file names retired classes in the comments that
    record their retirement, and counting those as definitions reports every
    one of them as an orphan.
    """
    body = re.sub(r"/\*.*?\*/", " ", css, flags=re.S)
    return set(re.findall(r"\.([a-zA-Z][\w-]*)", body))


def names_frontend_can_produce() -> set:
    """Every word in the markup and scripts, which is deliberately coarse.

    A precise reading of `class="..."` would miss a name assembled from
    parts, and missing one here means deleting a rule that is in use. The
    error this must not make is the destructive one.
    """
    words: set = set()
    for path in sorted(FRONTEND.iterdir()):
        if path.suffix in {".html", ".js"}:
            words |= set(re.findall(r"[\w-]+", path.read_text(encoding="utf-8")))
    return words


class TestNoRuleStylesSomethingThatCannotAppear:
    """The sibling of the dead-selector check above.

    `.header-ctl` styled the context and workflow controls on the
    conversation header. They moved behind the bar's menu and the rules
    stayed, styling nothing, until a sweep went looking. Four more -
    `.deflist`, `.hairline-table`, `.section-bar`, `.input-quiet` - were
    added with the design foundation for a use that never arrived.
    """

    def test_every_class_the_stylesheet_styles_can_be_produced(self, stylesheet):
        orphans = sorted(
            defined_class_names(stylesheet)
            - names_frontend_can_produce()
            - BUILT_AT_RUNTIME
        )
        assert not orphans, (
            "these classes are styled and nothing in frontend/ can produce "
            "them, so the rules are dead weight: " + ", ".join(orphans)
        )

    def test_a_class_nothing_produces_is_reported(self):
        """The control. Passing on a clean stylesheet is what a check that
        reads nothing also does."""
        orphans = (
            defined_class_names(".ghost-town { color: red }")
            - names_frontend_can_produce()
            - BUILT_AT_RUNTIME
        )
        assert orphans == {"ghost-town"}

    def test_a_name_only_a_comment_mentions_is_not_a_definition(self):
        """Retirement comments name what they retired. Counting those would
        report every retired class as an orphan for ever."""
        assert defined_class_names(
            "/* `.stat-card` was retired here. */\n.row { color: red }"
        ) == {"row"}
