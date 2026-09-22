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
| `--radius` / `--radius-sm` | 8px / 6px | Surface and control |
| `--topbar-h` | 48px | The workspace bar, which is sticky |
| `--accent` | green | State, in this project |
| `--font` | Inter, system sans | Interface and prose. Served from `frontend/fonts/`, weights 400/500/600 |
| `--font-mono` | JetBrains Mono, then Fira Code, Consolas | Literal content. Served from `frontend/fonts/`, weights 400/500 |

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
| Keyboard, reachability | **Failed on the note list and the note search results**, since fixed. Both render `<button>` now, as the conversation list does. |
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

None outstanding. Every entry this section held has been repaired and has a
browser witness that fails against the defect it describes:

| Was | Now |
|---|---|
| An account erased on one click, from a button styled like `Set role` beside it | Confirms, naming the account by email rather than by the id that gets mistyped |
| Deleting a note named the title being typed while deleting the one that was saved | Names the saved record |
| The note list, the conversation list and the vault search unreachable by keyboard | All three are buttons; Tab reaches them and Enter opens them |
| A focus ring at 1.13:1, drawn after `outline: none` removed a working one | One `:focus-visible` rule at 4.02:1, and no live `outline: none` anywhere |
| Contradicted and evolved notes said so in colour alone | A dot and a word beside the date |
| Nothing answered `prefers-reduced-motion` while three animations ran for ever | Those three stop; the typing dots fade rather than bounce |
| Search results were a title and an excerpt in a clickable `div` | Rank, kind, date, excerpt, hairlines - and the `rank` the server was already sending |

The repairs are not the interesting part; the measurements are. The focus
ring was the one worth stating plainly, because the project had taken a
working indicator away and replaced it with one that cannot be seen, and
nothing failed while that was true.

## Rules that fixed nothing

Each of these was written to do a job and does not reach the page. They
matter more than the drift below, because a reader of the file believes
them.

| Rule | What it does |
|---|---|
| `.delete-user-btn`, `.view-patch-btn` at 28px | **Dead.** Renders 30px. `button.ghost` is element-plus-class and outranks a bare class - the same defect as `button.ghost` over `.voice-btn`, in a sibling the fix never reached. |
| `.table tr.clickable.selected` | **Never applied.** No code sets the class. The patch list, where clicking opens a detail panel, has no selection state. |
| `.patch-status.pending` | **Missing.** `admin-tab.js:396` computes `pending`; no rule matches, so it falls to neutral grey where part one assigns amber. |
| `.bar-actions` at 32px | **Dead for buttons, live for anchors** - a tier part one does not have. |
| `.table th` 13px | **Was dead**, and is gone. A later rule at equal specificity set 11.5px, so the size reached `td` and nothing else while reading as though it set both. It is on `.table td`, which is what it always meant. |

`.chip` and the hole that hid it are both closed. The orphan-class guard
read every word of every script including its comments, so a class counted as
producible if anyone had written its name in prose - and `common.js` discusses
citation chips. Comments no longer count. Measured before the change and
after: across the 257 classes the stylesheet defines it moves exactly one,
`chip`, from live to orphaned, and that one had no caller. The rules and the
`--chip-h` token that only they read are gone, and a control now feeds the
check a name that appears in a comment and nowhere else and requires it to
stay orphaned.

The live chip vocabulary is `attachment-chip`, `tool-chip`, `chip-name` and
`chip-kind`. `attachment-chip` is one of the four true pills, sized by
padding rather than by a height token, so part one's 20-22px chip tier has no
expression here at all.

## Vocabulary that does not exist as specified

** `.icon-btn` 28px, `.icon-btn.compact`
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

Both scales are now closed sets, and `tests/test_css_scale.py` keeps them
that way: a value is allowed when it is on the scale, or when it is pinned
in that file to the exact selector that may use it. A pin is a recorded
departure rather than an escape, so adding a size means naming where it is
allowed - which is the sentence this section asks for anyway. The guard was
written before the repairs and failed on all twenty-nine call sites.

It reads both spellings of each property. A size can hide in the `font`
shorthand and a radius in the four `border-*-radius` longhands, and a check
that read only `font-size` and `border-radius` would have let a twentieth
size in through a property it never looked at. Neither spelling carries a
value in this file today - all four `font:` declarations are `inherit` and
there are no radius longhands - so this is a hole closed before anything
fell through it, and the controls in that file feed it both.

**The type scale is a scale.** Nineteen distinct `font-size` values became
thirteen. Seven of them are part one's, and they carry 109 of the 116 call
sites. Each value that was off the scale moved to the size part one gives
its role, and to the nearest size on the scale where the role does not
decide - `12.5px`, which had ten call sites and no role at all, went to
12px at every one of them. Four moves are the role rather than the nearest
size: the page title to 16px from 17, the panel's section title to 13px
from 14, the message fact line to 11.5px from 10.5, and the top bar title
to 16px from 14.

The six that remain are pinned, and are not interface text: the heading ramp
inside rendered markdown (24px, 20px, 17.5px), which is a document's own
typography; the glyph buttons `×` and `+` and the note title field (22px,
18px), drawn at the optical size of a glyph rather than at the size of the
words around them; and the dashboard figure (26px), read as a number.

**Each role now uses the size part one gives it**, which is the second half
of the work and the half a guard cannot check: every value below was already
on the scale and simply on the wrong step of it.

| Role | Was | Now | Where |
|---|---:|---:|---|
| Metadata or fact line | 13px | 11.5px | citation meta, note meta, an unresolved wiki-link, an inline check, a description under a heading |
| Normal interface text | 13px | 13.5px | error banners, empty states, label-and-value rows, witness findings, link-styled buttons, table cells |
| Field label | 13px | 12px | the three label rules |

That moved 13px from 26 call sites to 12, and it is what the scale is for:
13px is the major-section title and the row title, and it had become the
size everything else was too.

One departure is recorded rather than repaired. `.context-pane .row-name` is
a row title at 12px where part one says 13px, and the rule above it says
why: the pane is secondary navigation with 219px to print a name in, and one
step down is a few more characters before the ellipsis on every row. That is
a measured reason, so it belongs here rather than being undone - the same
treatment the chat bubble and the glyph buttons get above.

**Reading prose had two sizes** and now has one: chat, the note editor and
its preview are all 15px/1.65. Inline code was `font-size: 82%`, the only
relative size in the file, so one role rendered at five sizes - up to
19.68px inside an `h1`, because the markdown renderer can put a code span in
a heading. It is 12px.

**The major-section gap** was 26px before a band on Files and Insights and
22px on Settings, where part one says 28-32. Both are 28px. The Settings
figure came from `.setting-group`'s own margin, because the `:first-child`
reset means the band's margin never applies there.

**Uppercase is back inside the small-label window**, and the two rules got
there by opposite routes, because the roles differ.

`.panel h4` keeps its uppercase and moved to 11.5px. Its one call site that
reaches it is "Patch details", introducing a block inside an expanded
editor, which is a category label - the role the window exists for. The
first attempt dropped the uppercase and left it at 13px, which made it
identical to `.panel h3` directly above except for its colour; that was
justified here by a comparison to `.section-band h4`, which turns out to
match nothing at all. Every `<h4>` in the interface is either that one or
inside a `-details` pane that sets its own size, and the bands title
themselves with `h3`.

`.message .meta` lost its uppercase and moved to 11.5px from 10.5. It is a
fact line - who spoke, when, which model - and not a category label, so the
window does not cover it. Its positive tracking went with the uppercase,
since that is what the tracking was for.

Seven of the eight rules that set uppercase now sit at 11px or 11.5px. The
eighth is `.bubble h6`, at 12px, and it is the bottom of the heading ramp
inside rendered markdown rather than a label in the chrome - the same
content typography the pinned sizes above cover.

**The radius scale is a closed set too.** Sixteen distinct `border-radius`
forms became twelve, and no literal restates a token any more: the eight
that did - `6px` six times and `8px` twice - use `var(--radius-sm)` and
`var(--radius)`, which is what makes part two's rule enforceable. The tokens
now carry 37 call sites rather than 26.

`9px` on `.rail-btn` and `.rail-mark` is gone: the rail is the app's primary
navigation and it was the one control tier off every band, so it is on the
6px token with the rest of them. `5px` on `.bubble code` joined the same
token. What is left is pinned: `12px` and `16px 16px 4px 16px` on the chat
bubbles, which are content shapes rather than chrome surfaces and where the
asymmetric corner says who spoke; and `4px`, `2px` and `1px` on
`.msg-warning`, `.draft-indicator`, `.brand .spark`, the streaming caret and
`.tick-mark`, which are marks a few pixels across rather than surfaces.

**The three filters are on the compact tier.** Part one puts a filter at
28px, and the artifact type and visibility filters and the patch status
filter rendered at 30px because they had no rule in common. They have one
now - `select.filter` - and the class says what the control is for, which is
what decides its tier.

Writing that rule was not enough for the third one, and the browser said so.
`#patches-status-filter` is a bare `<select>` in a toolbar row rather than
inside a `.field`, so it carried its own copy of every control property
under an id - and an id outranks any class, so the new rule changed nothing
there. This is the shape of the two entries below under "Rules that fixed
nothing", found the same way they were: by measuring rather than by reading.
The id rule is gone and `select.filter` is a whole control rather than a
height. `tests/test_browser_control_tier.py` holds the measurement, and
reports the rendered number rather than the intended one.

**No bare `:focus` selector is left.** There were seven in four rule blocks,
each adding a border tint beside the global `:focus-visible` ring. The tint
stays, because the rule above the global ring already records that a
component may add to it; what changed is the selector, so there is one focus
vocabulary rather than two.

A correction goes with that. This section used to say `.field select:focus`
was "a real difference, and fires on a mouse click", on the reasoning that a
`<select>` is not a text-entry control. That sentence was written from the
specification and never run. Asked of the browser, it is not a difference:
the user agent matches `:focus-visible` on a mouse-clicked `<select>`
exactly as it does on a mouse-clicked text field, so both spellings fired in
the same places. The change is a vocabulary the file can be read by, and
nothing a user sees.

That last statement is a claim about a user agent, so it is not left as a
sentence here. `tests/test_browser_control_tier.py` clicks a `<select>` with
the mouse and requires it to match `:focus-visible` and take the tint. If
this browser ever stops doing that, the rules moved onto `:focus-visible`
would drop their tint for every mouse user, and that test is what says so.

Transitions are otherwise close to the rule: twenty at 0.15s, four at
130ms, two at 0.18s.

## Level two, on every screen that has a second section

This section used to read "where five of eight screens have no level two".
Three of those five were banded by the work above without this being
rewritten, which is the failure part two warns about: the file lagged the
frontend, and a reader would have believed it.

Counted rather than remembered: Settings carries fourteen bands, Contexts
and Insights four each, Tools three, Files and Artifacts two.

**Notes** was the one still missing them. Its workspace holds four things -
the editor, the witness findings for the open note, the vault-wide sweep and
the graph - and three of those appeared with no heading at all, so clicking
Witness produced a block of verdicts with nothing naming it. Each now opens
under a band, and the band is what the view's visibility is toggled on, so a
pane cannot appear without its title.

**Chat** has no level two because it has no second section: a conversation,
a composer, and a disclosure for preferences. Part one asks a *substantial*
page to show all four levels, and inventing a band over a single stream of
turns would be the decoration it also warns against. Recorded here so the
next reader counts seven screens and not eight.

**Eleven `.divider` elements remain**, not the twenty this section used to
claim and not the five an earlier version of this paragraph claimed. That
five came from counting `index.html` and `chat.js` and forgetting
`admin.html`, which is the same mistake as the paragraph above it: a number
asserted from a partial count reads exactly like one that was counted.

Where they are, and what is still owed:

* **Six in `admin.html`.** The admin console's own markup still has no
  bands at all - an `<h3>` over a hairline, which is the shape this section
  says the band replaced. Outstanding, and the previous text was right to
  name it.
* **Four in the detail panes**, each directly above an `<h4>`. Those are
  *not* part one's subsection, which an earlier version of this paragraph
  said they were: `.context-details h4` and its three siblings render at
  15px in `var(--text)`, so what a reader gets is a hairline under a
  full-strength heading at the prose size, where part one asks for 11-12px
  muted. Recorded rather than changed here, because moving them is the
  role-and-size work above and wants the same care.
* **One in the password-reset form**, separating the request from the code
  entry. A hairline between two forms is the one use this element was for.

The Tools pane used to invert the spacing rule as well: 4px between two
different lists and 8px between two rows inside one, so the boundary was
half the gap it separated. The eyebrow that separates them now has 16px
above it, which is part one's subsection step and twice the gap inside a
list. The first eyebrow in a pane has the pane's own padding above it and
takes none of its own.

## A departure this project made and did not record

Resolved: the rule won. `.setting-editor` was a full card - a 1px border on
all four sides and a 6px radius - where part one's expanded editor is "a
faint neutral background, a 2px accent left marker, 10-12px padding, no
independent card". It is now that: no border, a 2px accent left marker, and
the radius only on the two corners away from the marker.

Two of its five call sites still put a bordered box inside, both of them the
`.mfa-secret` display, which is a value to be copied rather than a second
card. With the outer card gone that is one rectangle inside a marked strip
rather than three nested ones.
