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

## Acceptance tests

A screen meets the standard when it passes all six. The first is a judgment
and the other five are measurements, so run them rather than reading the
code and deciding it must be fine.

1. **Five-second scan.** Open the screen and look away after five seconds.
   You should be able to name its major sections. If it reads as one
   undifferentiated page, it needs landmarks, not smaller controls.
2. **Density.** Count the rows visible without scrolling at 1440x900, and
   ask what the space between them is buying. A list that shows few rows on
   a large screen is spending its room on decoration.
3. **Keyboard.** Reach every action with `Tab` alone. Nothing interactive may
   be unreachable, and no row action may sit inside the control that opens
   the row, which makes one keystroke do two things.
4. **Touch.** Anything that appears on hover must also be present without
   hover, under `@media (hover: none)`.
5. **Overflow.** No screen may scroll sideways at 390px. Long names and
   identifiers truncate. A grid track that holds a wide table needs
   `minmax(0, 1fr)`; a plain `1fr` takes its contents' min-content width as
   its minimum and hands the page's width to whatever is widest.
6. **Dark mode.** Contrast and borders survive the theme. Especially for
   `liminal`.

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
| `.chip` | An enclosure that is part of the behaviour. Defined and not yet called - see part three |
| `.figures` | Numbers without tiles |
| `.detail-row`, `.detail-label` | A label and its value, divided by a hairline |
| `.section-band`, `.section-icon`, `.section-description` | A major section's landmark |
| `.setting-group` | Everything under one band |
| `.settings-layout`, `.settings-index` | A sticky index beside its sections |
| `.setting-row`, `.setting-grid`, `.setting-help` | The dense settings form |
| `.setting-editor` | The isolated editing surface that still earns an enclosure. Currently a full card rather than part one's accent marker - see part three |
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

Every number here was read off a rendered page or counted in the source.
Where something is inferred rather than measured, it says so.

An earlier version of this section carried two wrong counts, both from one
bad command: `grep -c ':focus'` also matches `:focus-visible` and
`:focus-within`, which turned six selectors into seventeen. The substance
survived the correction and got worse, which is the argument for measuring
rather than counting lines.

## What the acceptance tests say

The checklist in part one had never been run against this project, because
this document had never carried it. Running it found one defect and cleared
three areas that had only been assumed.

| Test | Result |
|---|---|
| Density | Passes. Rows are 35-43px, so 20-26 fit a 1440x900 screen. |
| Keyboard, nested actions | Passes on rows. A contexts row is a `<button>` with no button inside it. |
| Touch | Passes. `@media (hover: none)` at `styles.css:3426` restores `.row-actions`. |
| Overflow | **Failed on Settings**, since fixed. The other seven tabs pass at 390px and all eight pass at 1440px. |
| Keyboard, reachability | Fails. See the note list and the note search results. |
| Dark mode | Not applicable. The stylesheet has no `prefers-color-scheme`, `[data-theme]` or `.dark` rule. |

The overflow failure was in code this work added. The two-column settings
rule writes its content column as `minmax(0, 1fr)`; the narrow-screen
override collapsed the layout to one column and wrote that track as plain
`1fr`. A `1fr` track takes `auto` as its minimum, which is the min-content
width of its contents, so the widest thing in Settings - the admin users
table - decided the width of the page. Measured at 300px of sideways scroll
on a 390px viewport, with the table's own `overflow-x: auto` wrapper
stretched wide instead of scrolling. `tests/test_browser_narrow_viewport.py`
pins it, and failed against the defect before the fix.

Two attempts at that measurement were worthless before one worked, in the
same way and for a reason worth recording: this app sends a
Content-Security-Policy that refuses inline styles, so an injected
`<style>` element attaches to the document and stays inert. Three states
compared, one answer for all three, and nothing in the result says the
mutation never applied. Read the marker back, or bypass the policy
deliberately.

## The enclosure rule, measured

Part one's headline rule is that a thing earns a rounded rectangle only if
it is a primary destination, a floating object, a semantic notice, a
decision requiring containment, or an isolated editing surface. Counting
rules in the stylesheet answers a different question, since a rule can be
dead, overridden, or attached to markup that never renders. This is a walk
of the rendered tree on each tab, counting the enclosures actually painted:
a non-zero radius together with either a visible border or a fill differing
from the parent.

`.panel` is the only enclosure a screen draws for itself, once per tab,
seven in total with chat having none. Two radius values across every content
surface, 6px and 8px, both tokens.

**Three cards sit inside that card**, all of them in a detail pane:
`.schema-viewer` under an artifact's schema and under a tool's inputs, and
`.code-block` under a context's sources. Each is a 6px bordered box inside
`.panel`'s 8px bordered box, on the same background, which part one asks to
avoid unless the inner surface is a genuinely different interactive object.
A block of literal text is not one. Mono type and the `h4` above it already
separate the schema from the rows; the border is a second answer to a
question already answered.

An earlier version of this section said there were none, and that was a
measurement error worth recording rather than quietly fixing. The census
walked each tab with nothing selected, and a detail pane only renders after
a selection - so the half of the app that shows a schema was never in the
tree it walked. Its positive control was real but answered a different
question: it proved the probe could see an enclosure, not that the probe was
looking at the screens that have them. Selecting an item on each tab first,
using the selectors the capture script already keeps for this reason, found
all three.

One gap remains: chat bubbles. The stub backend returned no assistant turn,
so the screen whose main content is a rounded filled shape had nothing in
it. The bubble radii are known from the stylesheet - 12px and
`16px 16px 4px 16px` - but whether chat nests enclosures at depth is not
measured.

## Defects that reach the reader

These are not matters of taste. Each one costs somebody the use of
something.

**An account is deleted without being asked.** `admin.js:406` sends
`DELETE /admin/users/{id}` the moment the button is pressed. The same action
on the settings tab does ask (`admin-tab.js:175`). The button is
`class="ghost"`, so it is also pixel-identical to the "Set role" button
beside it: part one says destructive is red at rest and confirms with
explicit words, and this does neither.

**The note list cannot be reached by keyboard.** Each `.note-item` is an
`<li>` with no `tabindex`, no `role` and no `href`; forty tab presses never
land on one. Opening a note is the pane's primary action and it is available
only to a pointer. `.conversation-item` has the same shape - a `<div>` at
`chat.js:733` - though that one is read from the source rather than
measured. The same class is a real `<a>` on the share page
(`share.js:74`), so the primitive already has a correct form.

**The focus ring is invisible.** Where the project styles focus at all it
sets `outline: none` and draws `box-shadow: 0 0 0 2px rgba(14, 138, 109,
0.10)` - the accent at one tenth alpha, which computes to **1.13:1**
against every surface token. WCAG 2.2 asks for 3.0:1. The same declaration
appears on `.icon-btn:focus-visible`, so a keyboard reader crossing a file
row's four actions cannot see which one is focused; that instance is
inferred from the identical declaration rather than measured. Everything
else falls back to the browser's own outline, which is visible but is not
this vocabulary and differs between browsers.

The consequence is worth stating plainly: on the controls the project
styles, it removed a working indicator and replaced it with one that cannot
be seen.

**Search results are the one list part one describes in detail, and the
only list nothing was checked against.** Part one asks for an index or icon,
the name, a score, a source and type line, an excerpt, and actions, with a
hairline between results. The vault search renders two of those:
`notes.js:285` emits a `<div class="note-search-hit">` holding a title and
an excerpt. There is no hairline - `.note-search-hit` sets padding and a
hover fill and no border, so eight results are eight unseparated blocks in
one tinted box. There are no actions and no source line.

The index is the part worth singling out, because the work was already
done. `routes.py:2834` sends a `rank` field that is the result's 1-based
position, with a comment explaining that it is deliberately a position
rather than the raw fused score, since that score "tops out near 0.016 and
packs the whole result set into a hair's breadth of itself". The backend
solved exactly the problem part one's "a score" raises, and the frontend
never reads the field.

The same element is also a `<div>` with `cursor: pointer`, so opening a
result is available only to a pointer - the note list defect again, in the
list a reader reaches by typing.

**Status carried in colour alone.** `.note-item.contradicted` and
`.note-item.evolved` change only the title's colour - no dot, no word, no
`title` attribute. `.voice-btn.playing` does the same. Five lines away,
`notes.js:34` does it correctly for unsaved state, with a dot and a word.

**`prefers-reduced-motion` is not honoured.** There is no such block, and
**three** animations run indefinitely without one: the streaming pulse, the
caret and the typing indicator. `typing-bounce` is a bounce, which part one
says to avoid. The turn-rail flash runs once.

## Rules that fixed nothing

Each of these was written to do a job and does not reach the page. They
matter more than the drift below, because a reader of the file believes
them.

| Rule | What it does |
|---|---|
| `.delete-user-btn`, `.view-patch-btn` at 28px | **Dead.** Renders 30px. `button.ghost` is element-plus-class and outranks a bare class - the same defect as `button.ghost` over `.voice-btn`, in a sibling the fix never reached. |
| `bar-primary` on "New thread" | **Defined nowhere.** The one primary the chat view declares renders as a bordered secondary, so Chat has no primary at all. |
| `.table tr.clickable.selected` | **Never applied.** No code sets the class. The patch list, where clicking opens a detail panel, has no selection state. |
| `.patch-status.pending` | **Missing.** `admin-tab.js:396` computes `pending`; no rule matches, so it falls to neutral grey where part one assigns amber. |
| `.chip` | **No caller.** The sanctioned chip vocabulary is unused, while four ad-hoc enclosures exist beside it. |
| `.bar-actions` at 32px | **Dead for buttons, live for anchors** - a tier part one does not have. |
| `.table th` 13px | **Dead.** A later rule at equal specificity sets 11.5px. |

`.chip` also passes `tests/test_css_hygiene.py`, which should have reported
it. That check strips comments from the stylesheet but not from the scripts,
so the word "chip" inside a code comment counts as markup that can produce
the class. The guard has a hole.

## Vocabulary that does not exist as specified

**There is no ghost tier.** Rendered side by side, `primary` is black,
`ghost` is white with a visible border and `minor` is a grey fill. Ghost and
minor are both part one's *secondary*, so forty-six buttons share one level
where the standard has two. Part one says ghost is the default for
lower-priority actions; nothing is ghost.

**Four icon-button geometries.** `.icon-btn` 28px, `.icon-btn.compact`
**24px**, `.pane-toggle` 30px, `.modal-close` no declared size at all. Part
one names one, at 28px.

**Three expansion vocabularies.** A `+` rotated 45° into an `×` on
`.panel-section`, a triangle rotated 90° on `.advanced`, and nothing on the
settings sections. Part one asks for one chevron.

**Eleven refresh buttons at the standard tier**, where the same action is
already an icon button in five other places.

**Four buttons labelled with a typographic glyph** rather than a line icon
or a word, and the two that do the same job - new chat, new note - disagree
with each other about which class to wear.

## Drift in the numbers

**The type scale is not a scale.** Twenty distinct `font-size` values -
nineteen in pixels plus one percentage - where part one names seven sizes
across ten roles. The three largest groups (13px, 12px, 11px) are on scale.
Measured against the role each one serves: the page title renders 17px for
16, the metadata label 13px for 11.5, the row title 12px for 13, body text
13px for 13.5, and the settings index 12.5px for 12. `12.5px` alone has ten
call sites and no role at all.

**`.bubble code { font-size: 82% }`** is the only relative size in the
file, so one role renders at five sizes: 12.3px in prose, 12.5px in a fenced
block, 11.89px in a user bubble, 16.4px inside an `h2` and **19.68px inside
an `h1`**, because the markdown renderer can place a code span inside a
heading.

**Reading prose has two sizes.** Chat is 15px/1.65, which is right. The note
editor and its preview are 16px/1.7.

**The major-section gap is two numbers, neither on the scale.** 26px before
a band on Files and Insights, 22px on Settings - where part one says 28-32.
The Settings figure comes from `.setting-group`'s own margin, because the
`:first-child` reset means the band's 26px never applies there.

**Uppercase outside the small-label window**, at 13px on `.panel h4` and
10.5px on `.message .meta`. The same role is set in sentence case by
`.section-band h4`, 2,700 lines away.

**The radius scale has the same shape as the type scale.** Sixteen distinct
`border-radius` forms are declared where part one names four bands and a
pill. Ten of them are literals that bypass the two tokens, and eight of
those simply restate a token: `6px` appears six times and `8px` twice,
alongside `var(--radius-sm)` twenty-four times and `var(--radius)` five.
Part two's rule is to change the token rather than the call site, and these
are the call sites that would not move.

The literals that are not on any band: `9px` on `.rail-btn` and
`.rail-mark`, where a control is specified at 6px; `12px` and
`16px 16px 4px 16px` on the chat bubbles; `4px` on `.msg-warning` and
`.draft-indicator`; `2px` on `.brand .spark` and the streaming caret; `1px`
on `.tick-mark`. The last three are marks a few pixels across rather than
surfaces, so they are the defensible end of the list. The rail is not: it is
the app's primary navigation and it is the one control tier at 9px.

**Three filters are still outside the compact tier.** Part one puts a filter
at 28px. The artifact type and visibility filters and the patch status
filter render at 30px. The two pane search fields were the other two and are
now on the tier, which is what carried `--ctl-h-sm` from two call sites to
three; the three that remain are `select` elements with no shared rule
between them.

**Six bare `:focus` selectors** in four rule blocks. Three style text-entry
controls, where the two selectors behave alike; `.field select:focus` is a
real difference, and fires on a mouse click.

Transitions are otherwise close to the rule: twenty at 0.15s, four at
130ms, two at 0.18s.

## Where five of eight screens have no level two

Files, Insights and Settings carry bands. **Chat, Notes, Contexts, Artifacts
and Tools do not**, and neither does the admin console's own markup. On
Contexts, Artifacts and Tools the major-section level is an `<h3>` over a
`.divider` - a hairline with no fill, no icon and no count, which is
invisible when the page is scanned rather than read. Twenty `.divider`
elements still carry a boundary that the band is now the vocabulary for.

The Tools pane also inverts the spacing rule: **4px between two different
lists and 8px between two rows inside one**, so the boundary is half the gap
it separates.

## A departure this project made and did not record

`.setting-editor` is a full card: a 1px border on all four sides and a 6px
radius. Part one's expanded editor is "a faint neutral background, a 2px
accent left marker, 10-12px padding, no independent card". Three of its five
call sites also put a second bordered box inside it with the same fill, so
the reader sees three nested rectangles.

Either the rule wins and the border becomes an accent marker, or this is a
deliberate departure and belongs in part two with its reason. It is
currently neither.
