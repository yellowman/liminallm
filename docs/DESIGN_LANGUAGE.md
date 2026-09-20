# design language

The rules this interface is built from, and how this repository expresses
them. `ui.md` records what each screen contains; this file records why it
looks the way it does, so a reader can extend it without inventing a second
design system beside the first.

**Part one is the shared specification**, common to this project and the
sibling project `liminal`. Each keeps its own accent colour and its own
density - `liminal` is the denser engineering workbench, this one is
slightly more spacious - and everything else is meant to be the same. Write
changes to part one as changes to both projects.

**Part two** is this project's expression of it: the tokens, the class
names, and the places where it deliberately departs.

**Part three** lists what part one specifies and this project has not built,
with measurements. A rule with no implementation is a rule, not a claim, and
recording the difference is what keeps the two apart.

## The objective

Dense, structured, technical, quiet, modern. Not minimal in the sense of
removing useful information: the aim is to expose more useful information
with less decorative material.

The application rail sets the grammar the rest of the interface follows: a
thin line glyph in a larger hit area, no border or background at rest, a
quiet fill on hover, and selection carried by a faint accent wash plus a
narrow accent marker. Restrained colour. Compact type.

One rule generates most of the others:

> If something is not a primary destination, a floating object, a semantic
> notice, a decision requiring containment, or an isolated editing surface,
> it probably does not need its own rounded rectangle.

---

# Part one: the shared vocabulary

## Typography

Two families. Inter for interface and prose, falling back to the system
sans. JetBrains Mono for code, paths, identifiers and configuration keys,
falling back to Fira Code, Consolas, monospace.

Use monospace only where character identity or literal syntax matters, not
to make something look technical.

Reading prose is set one step larger and looser than the chrome around it.
That contrast is what marks it as the thing to read. A different typeface
marks it as a different product.

## Type scale

| Role | Size | Weight | Line height |
|---|---:|---:|---:|
| Page title | 16px | 600 | 22px |
| Major section title | 13px | 600 | 18px |
| Section eyebrow or category | 11-11.5px | 600 | 16px |
| Normal interface text | 13.5px | 400 | 20px |
| Row title | 13px | 500 | 18px |
| Field label | 11.5-12px | 500 | 16px |
| Metadata or fact line | 11.5px | 400 | 16px |
| Compact control | 12px | 500 | 16px |
| Code, path or identifier | 12px | 400-500 | 18px |
| Long prose (this project) | 15px | 400 | 24-25px |
| Workbench prose (`liminal`) | 13.5px | 400 | ~22px |

Restrict uppercase to small category labels. Use tighter tracking on titles
and not on body text.

## Geometry

Two ordinary control tiers, and rows are not controls.

| Thing | Height |
|---|---|
| Standard control: input, select, ordinary button, page action | 30px |
| Compact control: filter, toolbar control, small button, icon button | 28px |
| One-line row | 34px minimum |
| Two-line row | 42-46px, naturally |
| File table row | 34-38px |
| Chip | 20-22px maximum |

Keep a visible glyph at 16-18px inside a 28px target: a glyph smaller than
its target is precise to look at and easy to hit.

## Radius

Radius softens an interaction surface. It does not turn a fact into a pill.

Normal surface 8px, control or hover row 6px, small chip 5-6px, floating
menu or modal 8-10px. A true pill only where pill semantics are useful.

A fact such as `private`, `workflow`, `v4` or `JSON` needs no enclosing
shape.

## Colour

**Black or dark neutral means hierarchy.** It marks the primary action.

**Accent means state.** Current destination, selected row, keyboard focus,
active filter, active or editing state. Not "this is important".

Semantic colours: red for failure and destruction, amber for warning and
pending, green for success only, neutral grey for ordinary facts.

Never carry status in colour alone. Pair it with a word, an icon or a shape.

## Spacing rhythm

4px for a microscopic relationship, 6px for compact action gaps, 8px inside
a row, 12px between components, 16px between subsections, 24px between
sections, 28-32px between major sections.

The principle underneath: **more space between groups than within them**.
Uniform spacing is what makes a long page read as one column. Twenty-eight
pixels before a category and eight between its rows teaches the grouping
without another box.

## Page hierarchy

Four levels, and a substantial page should show all of them.

**Page.** A header answering where am I, what is this for, and what are the
major actions. A summary or utility strip may sit below it.

**Major section.** A band, not a card: a 16px line icon, a 13px semibold
title, an 11.5px muted description or count, and optional actions. More
space above than below. A subtle neutral fill, a hairline, or both. A major
section must be recognisable while scrolling, without reading it.

**Subsection.** An 11-12px muted heading, an optional count, a hairline,
tighter spacing. Not another rounded box.

**Row or detail.** The actual objects, settings, files or results. Editing
content may open below the row as a lightly differentiated band.

## Cues

The interface needs more cues, not more decoration.

**Section icons** come from the rail's family: a 20x20 viewBox, ~1.5px
stroke, round caps and joins, no fill unless the glyph needs one.

**Selection** is always a faint accent background plus a narrow 2px marker.
Do not invent a second selected style for a second list.

**Expansion** uses one chevron vocabulary: right when collapsed, down when
expanded.

**Dirty or changed** is a 2px accent marker or a small accent dot, not a
`MODIFIED` pill on every changed row.

## Buttons

**Primary** is black or dark neutral at 30px, with a text label, and there
is one per immediate scope.

**Secondary** is a border or quiet neutral treatment at 30px, used
selectively.

**Ghost** is the default for lower-priority actions.

**Icon buttons** are for row-local operations: a 28x28 target, a 16-18px
glyph, transparent at rest, a faint hover, and always a tooltip and an
`aria-label`.

**Destructive** is red at rest - a red glyph or red text is enough, and
among identical grey glyphs it is the only thing distinguishing the one that
destroys something. At confirmation, use explicit words and a red
confirmation button. Never let the confirmation itself be an ambiguous icon.

## Focus

One focus vocabulary, everywhere: keyboard-only `:focus-visible`, a 2px
accent ring, visible on both light and dark surfaces, optionally with a 1px
offset. Never colour change alone.

Apply it to buttons, icon buttons, filters, inputs, selects, textareas,
primary row controls, tabs and rail destinations. Do not keep a global focus
system and per-component systems beside it.

## Rows

A list is rows by default, not cards.

A row carries an optional type icon, a primary label, optional second-line
metadata, optional aligned facts, optional secondary actions, a quiet hover
fill, and a selection state where selection means something.

An interactive row is a shell holding a real `<button>` that owns opening or
selecting, and sibling real buttons for the actions. A `role=button` parent
must not contain real buttons.

Secondary actions reveal on hover and on `focus-within`, stay visible while
the row is selected, and are always visible where there is no hover. **The
row must not reflow when they appear.**

## Metadata, status and chips

Three separate ideas, and they are not interchangeable.

A **fact line** carries descriptive information with no enclosure:
`Private · workflow v4 · 2h ago`.

A **status** carries semantic state as a coloured dot and a word: `● Failed`.

A **chip** is for when the enclosure is part of the behaviour: a filter, a
toggle, a removable tag, a selected scope. An active chip takes a light
accent tint and accent text, matching the rail, rather than a solid capsule.

## Cards

Keep cards and make them scarce. A card suits an approval needing a
decision, an explicit confirmation, a semantic warning or error, a selected
preview or editor, a modal, an isolated document object, or a bounded diff
where containment matters.

A card does not suit files, artifacts in a list, settings rows, search
results, contexts, tools, workflows, basic statistics, plain metadata, or
ordinary prose.

Avoid a card inside a card unless the inner surface is genuinely a different
interactive object.

## Long configuration pages

The failure mode is a long vertical stream where every setting has the same
visual weight.

A sticky section index of 180-200px sits beside the content, not above it -
above means it scrolls away at the point a reader has lost track of where
they are. The current category takes a subtle accent treatment. Below about
900px the index collapses into a horizontal jump strip.

Each category opens with a band carrying its icon, name, and a description
or count: `Provider and generation defaults · 8 settings`. A count is worth
more than a description on a page this long, and it should track the filter.

A setting row is a stable grid - name and description, current value, edit -
separated by hairlines, with no card around each one. Its metadata reads as
a fact line (`Tenant override · Sensitive · Restart required`) rather than
three capitalised pills, with a warning icon only where the warning is
semantic.

An expanded editor opens beneath its row: faint neutral background, 2px
accent left marker, 10-12px padding, no independent card.

A dirty row takes an accent marker, not a badge.

A long page ends in a sticky decision surface while it is dirty: the count
of unsaved changes, a reset, and a black primary save, on a panel background
with a hairline above.

## File lists

A file list reads like a technical file browser.

Lay it out on a grid - icon, name, size, modified, actions - with a small
muted heading line naming the columns. A fact line after the name starts the
size and the date in a different place on every row, and comparing them down
the column is the one thing a file list is read for.

Suggested columns: a 20px icon, `minmax(0, 1fr)` for the name, 72-84px for
the size, 100-120px for the date, and a region for the actions.

A quiet fact line under the list says how much there is.

## Search and result lists

Results are lists, not cards. An index or icon, the document name, a score,
a source and type line, an excerpt, and actions - with a hairline between
results. Expanded provenance opens as an inset detail band.

## Statistics

Do not put every number in its own tile. A row of values over their labels
reads better and takes less room. A card is only needed when each metric is
independently interactive or carries substantial additional content.

## Motion

Motion should be almost invisible: 120-150ms on background, colour and
opacity. No scale-on-hover, no bouncing, no decorative slide-ins during
ordinary operation. Use it for disclosure, menus, modals and streaming.
Respect `prefers-reduced-motion`.

## Fifteen rules

1. A list is rows by default, not cards.
2. A pill means behaviour or exceptional state, not ordinary metadata.
3. Accent means state; black means hierarchy.
4. Thirty pixels is the normal control height; twenty-eight is compact.
5. Rows are allowed to be taller than controls.
6. Secondary row actions hide on pointer devices and stay reachable by
   keyboard and touch.
7. Major sections need visible landmarks: icon, title, description or count,
   spacing, a hairline, or several of these.
8. More space goes between groups than within them.
9. No nested neutral rectangles where typography or a hairline says the same
   thing.
10. Semantic warnings and errors may use contained tinted surfaces.
11. Inter for interface and prose, JetBrains Mono for literal technical
    content.
12. Do not invent control geometry, focus styling, selection styling or
    button hierarchy inside a feature component.
13. Every long settings page gets navigation between its categories.
14. Every file list gets aligned columns, or equivalent stable alignment.
15. One obvious primary action per immediate scope.

---

# Part two: how this project expresses it

## Tokens

The geometry above lives in `:root` in `frontend/styles.css`. Change the
token, not the call site.

| Token | Value | What it sets |
|---|---|---|
| `--ctl-h` | 30px | The standard control tier |
| `--ctl-h-sm` | 28px | The compact tier, including the settings form |
| `--icon-hit` | 28px | An icon button's target |
| `--icon-glyph` | 18px | The glyph inside it |
| `--chip-h` | 20px | A chip |
| `--radius` / `--radius-sm` | 8px / 6px | Surface and control |
| `--topbar-h` | 48px | The workspace bar, which is sticky |
| `--accent` | green | State, in this project |
| `--font` | Inter, system sans | Interface and prose |
| `--font-mono` | JetBrains Mono, then Fira Code, Consolas | Literal content |

A token that is read and never defined does not fail loudly. The declaration
becomes invalid at computed-value time and an inherited property quietly
takes its parent's value, so a rule meant to quiet a label leaves it at full
strength and looks correct in the file. `tests/test_css_hygiene.py` reports
any `var(--name)` without a fallback that nothing defines.

## Class vocabulary

| Class | What it is |
|---|---|
| `.row` | The one flat list primitive, with `.row-icon`, `.row-name`, `.row-meta`, `.row-actions` |
| `.factline`, `.fact-dot` | A fact line, and a status dot |
| `.chip` | An enclosure that is part of the behaviour |
| `.figures` | Numbers without tiles |
| `.detail-row`, `.detail-label` | A label and its value, divided by a hairline |
| `.section-band`, `.section-icon`, `.section-description` | A major section's landmark |
| `.setting-group` | Everything under one band |
| `.settings-layout`, `.settings-index` | A sticky index beside its sections |
| `.setting-row`, `.setting-grid`, `.setting-help` | The dense settings form |
| `.setting-editor` | The isolated editing surface that still earns an enclosure |
| `.sticky-actions` | A decision bar pinned to the foot of a long form |
| `.utility-strip` | Compact controls that decide what a list shows |
| `.summary-line` | The fact about a list, under it, with any pager at the far end |
| `.file-table-head`, `.row.file-row`, `.file-facts` | A file list on a grid |
| `.panel` | A working surface, not a floating card |

`trackSections` in `frontend/common.js` keeps the index's mark on the
section being read. It reads position from `getBoundingClientRect` rather
than from the observer's entries: an entry says a section crossed a
boundary, which is not the same as which section is topmost now.

## Departures from part one, and why

**`.page-head` and `.page-description` are not added.** `.section-header`
and `.subtext` already are those things here. A synonym costs every reader
of the stylesheet a second name for one idea.

**`.setting-group`, not `.settings-group`.** It sits beside `.setting-row`,
`.setting-grid` and `.setting-help`, and a one-character difference between
two live class names is a trap.

**`.sticky-actions` has one caller**, the admin console's system settings.
The Settings tab has no single save to pin: each of its sections owns its
own submit. A page-level save bar there would be a control with nothing
behind it.

**Files has no utility strip.** The endpoint takes a limit and an offset and
nothing else, so a search or a type filter would silently cover only the
loaded page. The band carries the count and the refresh instead.

**`.summary-line` on Files reports the page's bytes, not the library's.**
The endpoint sends a count and this page's files. A total size would be a
number nothing measured.

## Things that are not obvious from the stylesheet

**`position: sticky` resolves against the nearest scroll container, not the
page.** An ancestor with `overflow` set becomes that container even when it
has no bounded height and therefore never scrolls - and a sticky descendant
is then pinned to a box that never moves. The page scrolls as a document,
so nothing between a sticky element and the viewport may set `overflow`.

**A sticky element under `.topbar` needs its offset.** `.topbar` is sticky
at `top: 0` on both the workspace and the admin console. An index or a
`scroll-margin-top` that does not clear `var(--topbar-h)` puts its target
behind the bar.

**Two grids size their tracks independently.** The file heading and the file
rows share one `--file-actions` declaration for that reason: an action
column sized by its own contents would leave the heading's columns somewhere
else, and the list would go on looking like a list.

---

# Part three: specified and not yet built

Measured against the current `frontend/styles.css`.

**One focus vocabulary is not in place.** Seventeen rules use `:focus` and
four use `:focus-visible`, so most focus rings fire on a mouse click as well
as on keyboard navigation. Part one asks for `:focus-visible` only.

**`prefers-reduced-motion` is not honoured.** There is no such block in the
stylesheet, and four `animation` rules run indefinitely without one: the
streaming pulse, the caret, the typing indicator and the turn-rail flash.
One of them, `typing-bounce`, is a bounce, which part one says to avoid.

Transitions themselves are close to the rule: twenty at 0.15s, four at
130ms, and the rest between 0.12s and 0.18s.

**The type scale is not a scale.** Nineteen distinct `font-size` values are
in use where part one names about ten. The largest groups - 13px, 12px, 11px
- already match; the miscellany around them does not.

**The rest of the migration.** Insights, Notes, Files and the settings pages
carry the vocabulary above. The remaining work is a pass over what is left:
leftover pills and bordered buttons that should be ghost or icon buttons,
and the type scale.
