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


def without_comments(text: str, suffix: str) -> str:
    """The source with its comments removed, so prose is not evidence.

    A comment is where a name goes to be discussed, not produced. Retirement
    notes, design rationale and the word for a thing in passing all put a
    class name in the file without anything ever setting it, and a word set
    that counts them tells this check a dead rule is live. Measured: `.chip`
    had no caller and passed for exactly that reason, because
    `common.js` describes citation chips in prose.

    Coarse in the other direction on purpose. `//` is only taken as a
    comment when it does not follow a colon, so the `//` in a URL survives;
    a `//` inside a string literal would still be cut, which can only make
    this stricter, and a name that appears nowhere but inside such a string
    is not one this check should call live either.
    """
    if suffix == ".html":
        return re.sub(r"<!--.*?-->", " ", text, flags=re.S)
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"(?<!:)//[^\n]*", " ", text)


def names_frontend_can_produce() -> set:
    """Every word in the markup and scripts outside a comment.

    Still deliberately coarse within that. A precise reading of `class="..."`
    would miss a name assembled from parts, and missing one here means
    deleting a rule that is in use, which is the destructive error.

    Dropping comments was measured before it was made: across the 257
    classes the stylesheet defines, it moves exactly one - `chip` - from
    live to orphaned, and that one had no caller.
    """
    words: set = set()
    for path in sorted(FRONTEND.iterdir()):
        if path.suffix in {".html", ".js"}:
            words |= set(
                re.findall(
                    r"[\w-]+",
                    without_comments(
                        path.read_text(encoding="utf-8"), path.suffix
                    ),
                )
            )
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

    def test_a_name_only_a_comment_mentions_cannot_keep_a_rule_alive(self):
        """The control the other one is not.

        `.ghost-town` appears nowhere at all, so it proves the check can see
        an orphan - not that a comment fails to hide one. Those are different
        claims, and the second is the one that was false: the word set read
        every word of every script including its comments, so a class was
        live if anyone had ever written its name in prose.

        This feeds the two comment syntaxes a script can carry and requires
        the name to stay orphaned in both.
        """
        for source in ("// styled by .drifter\n", "/* .drifter, retired */\n"):
            assert "drifter" not in set(
                re.findall(r"[\w-]+", without_comments(source, ".js"))
            ), f"a comment kept the name alive: {source!r}"
        assert "drifter" in set(
            re.findall(
                r"[\w-]+",
                without_comments("el.className = 'drifter';\n", ".js"),
            )
        ), "stripping comments also removed a real assignment"

    def test_a_url_is_not_mistaken_for_a_comment(self):
        """`//` after a colon is a scheme, not a comment. Cutting there
        would drop the rest of the line, which is where a name can be."""
        kept = set(
            re.findall(
                r"[\w-]+",
                without_comments(
                    "fetch('https://x/y'); el.className = 'kept';\n", ".js"
                ),
            )
        )
        assert "kept" in kept, kept

    def test_a_name_only_a_comment_mentions_is_not_a_definition(self):
        """Retirement comments name what they retired. Counting those would
        report every retired class as an orphan for ever."""
        assert defined_class_names(
            "/* `.stat-card` was retired here. */\n.row { color: red }"
        ) == {"row"}


def undefined_custom_properties(css: str, also_defined: set | None = None):
    """Every `var(--name)` with no fallback that no file defines `--name`.

    An undefined custom property does not fail loudly. The declaration
    becomes invalid at computed-value time, so an inherited property such as
    `color` silently takes its parent's value and the rule appears to work.
    That is what `var(--muted)` did on the settings form: five declarations
    meant to quiet a label, a help line, a badge and a jump link, all
    computing to the body's full-strength text colour.

    A reference carrying a fallback - `var(--x, 12px)` - is deliberate and
    not reported.
    """
    body = re.sub(r"/\*.*?\*/", " ", css, flags=re.S)
    defined = set(re.findall(r"(--[\w-]+)\s*:", body)) | (also_defined or set())
    used = {
        name
        for name, tail in re.findall(r"var\(\s*(--[\w-]+)\s*([,)])", body)
        if tail == ")"
    }
    return sorted(used - defined)


def custom_properties_frontend_defines() -> set:
    """Names an HTML or JS file sets, so a token living outside the
    stylesheet is not reported as missing. Nothing does this today; the
    check reads for it anyway, because the error to avoid here is the one
    that sends a reader deleting a live token."""
    names: set = set()
    for path in sorted(FRONTEND.iterdir()):
        if path.suffix in {".html", ".js"}:
            names |= set(
                re.findall(r"(--[\w-]+)\s*:", path.read_text(encoding="utf-8"))
            )
    return names


class TestEveryTokenAReferenceNamesIsDefined:
    """`--muted` was referenced five times and defined nowhere.

    The settings form is the densest surface in the product and reads by
    contrast: a 12px label at full strength above an 11px help line at half.
    With the token missing, both were full strength and the contrast the
    layout depends on was simply absent - on screen, in the screenshots, and
    in every review that read the file rather than the pixels.
    """

    def test_no_reference_names_a_token_nothing_defines(self, stylesheet):
        missing = undefined_custom_properties(
            stylesheet, custom_properties_frontend_defines()
        )
        assert not missing, (
            "these tokens are read and never defined, so each declaration "
            "reading one is dropped and the property falls back to its "
            "inherited or initial value: " + ", ".join(missing)
        )

    def test_the_missing_token_is_reported(self):
        """The control, verbatim in the shape that shipped."""
        assert undefined_custom_properties(
            ":root { --text-muted: #626872 }\n"
            ".setting-help { color: var(--muted); }\n"
        ) == ["--muted"]

    def test_a_defined_token_is_not_reported(self):
        assert (
            undefined_custom_properties(
                ":root { --text-muted: #626872 }\n"
                ".setting-help { color: var(--text-muted); }\n"
            )
            == []
        )

    def test_a_reference_with_a_fallback_is_not_reported(self):
        """`var(--x, 12px)` degrades to the fallback by design, so an
        undefined name there is a choice rather than an oversight."""
        assert undefined_custom_properties(".a { top: var(--not-set, 12px); }") == []
