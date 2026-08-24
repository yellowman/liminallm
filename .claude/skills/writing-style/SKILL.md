---
name: writing-style
description: Default style for conversation, explanations, technical writing, and documentation in this repository. Load before writing prose the reader will act on - chat replies, docs/*.md, README content, commit and pull request bodies, code comments, and error or log messages. Covers audience and purpose, voice, clarity, global and inclusive language, structure, procedures, semantic formatting of code and UI text, links, accessibility, and a final review checklist.
---

# Concise communication and documentation guide

Use these instructions as the default for conversation, explanations,
technical writing, and documentation.

## Apply rules in the right order

1. Follow the user's explicit requirements, project conventions, product
   terminology, templates, and existing document style.
2. Apply this guide where those sources are silent.
3. Use authoritative domain references for unresolved technical terminology
   or spelling.
4. Prefer clarity, accuracy, accessibility, and reader success over
   mechanical compliance. When you depart from a convention, remain
   consistent.

## Write for a specific reader and purpose

Determine the reader's goal, knowledge level, context, and desired result
before composing the response.

- Lead with the answer, recommendation, decision, or next action.
- Put critical information first in each section and paragraph.
- Include background only when it helps the reader act or understand.
- Address the reader as *you*. Use direct imperatives for instructions: "Run
  the command," not "The command should be run."
- Use third person for actions performed by software, systems, or the
  reader's end users.
- Use *we*, *our*, or *us* only when clearly referring to the organization
  that authored the material.
- Put conditions, prerequisites, location, and goals before the associated
  instruction. Put results and explanations after the action.

## Use a natural, professional voice

Write like a knowledgeable, respectful colleague.

- Be conversational and friendly without becoming slangy, cutesy, frivolous,
  pushy, promotional, or excessively formal.
- Be direct. Avoid ceremonial introductions and filler that delay the useful
  information.
- Use natural contractions where appropriate.
- Do not overuse *please* in instructions.
- Avoid "let's," exclamation marks, internet slang, pop-culture references,
  and attempts to be unusually clever.
- Do not describe a task as simple, easy, obvious, or quick. Those words
  provide no instruction and can frustrate readers who encounter difficulty.
- Never blame, mock, or talk down to the reader.

## Optimize for clarity

Use US English unless the user or project specifies another language.

- Prefer active voice and present tense.
- Use familiar, precise words: *use* instead of *utilize*, *start* instead of
  *commence*.
- Keep sentences reasonably short. Aim for one main idea per sentence and one
  subject per paragraph.
- Avoid double negatives, ambiguous pronouns, misplaced modifiers, and long
  chains of nouns used as modifiers.
- Prefer a single clear verb over a phrasal verb when possible.
- Use each word in its primary meaning. Do not use the same term for
  different concepts.
- Define unfamiliar abbreviations and specialist terms at first use. Do not
  define terms the intended audience certainly knows.
- Use the same term for the same thing throughout the response or document.
- Repeat a noun when doing so is clearer than using an ambiguous pronoun.
- In durable documentation, avoid unnecessary time anchors such as *new*,
  *currently*, *now*, and *latest*. Describe the product's present behavior
  directly, and do not announce unapproved future features.

Make requirements explicit:

- Use *must* or a direct imperative for required actions.
- Label optional actions explicitly.
- Use *can* for ability or permission.
- Use *might* for possible outcomes.
- Avoid *should* when it leaves the reader unsure whether an action is
  required.

## Write for global and inclusive audiences

Use literal, culturally neutral language that translates cleanly.

- Avoid idioms, colloquialisms, culture-specific humor, metaphors, and
  seasonal references.
- In conversation, an analogy is acceptable when it genuinely clarifies a
  difficult idea. Follow it with a literal explanation.
- Avoid unnecessarily gendered, ableist, graphic, violent, or demeaning
  terminology.
- Use singular *they* when a person's gender is unknown or irrelevant.
- Use diverse, neutral names and locations in examples.
- Do not place real personal information, credentials, email addresses,
  account identifiers, or other sensitive data in examples.
- Preserve established technical terms when no accurate substitute exists,
  but define them and use them only where necessary.

## Organize information for scanning

Use structure only when it helps readers find or understand information.

### Headings

- Use sentence case.
- Do not end headings with periods.
- Make headings descriptive, unique, and hierarchical. Do not skip heading
  levels.
- Start task headings with a base-form verb: `Create an instance`.
- Use a noun phrase for conceptual headings: `Instance lifecycle`.
- Avoid beginning headings with an *-ing* verb when a clearer form exists.
- Prefix optional sections with `Optional:` when appropriate.

### Paragraphs and lists

- Keep each paragraph focused on one idea.
- Use numbered lists for sequences, procedures, priorities, and ordered
  phases.
- Use bulleted lists for unordered options, requirements, examples, or facts.
- Use description lists for term-and-definition pairs.
- Do not create a one-item list merely for decoration.
- Introduce a list with a complete sentence.
- Keep list items grammatically parallel and use consistent capitalization
  and punctuation.

### Tables and notices

- Use a table only for genuinely two-dimensional information, usually when
  each item has three or more related properties.
- Do not use tables for page layout, code blocks, one-column lists, or long
  lists split into columns.
- Introduce every table with a sentence and provide meaningful column
  headings.

Use notices sparingly:

- **Note:** useful but noncritical information.
- **Caution:** information requiring care.
- **Warning:** risk of irreversible action, data loss, financial loss, or a
  security problem.

## Write procedures as executable instructions

A procedure must let the reader complete a task without interpreting vague
prose.

- Use numbered steps.
- Begin each step with an imperative verb.
- Use one meaningful action per step. Combine only small, tightly related
  actions.
- State the condition, location, or goal before the action: "In the
  **Settings** dialog, select **Logging**." "To preserve the existing
  configuration, create a backup."
- Prefer the shortest accessible method that works for the intended audience.
- Document one best method unless alternatives are materially important. Put
  substantial alternatives in separate sections.
- Begin optional steps with `Optional:`, not "(Optional)."
- Keep each step complete and reasonably short.
- State a result after the action that causes it. State a justification after
  the instruction it supports.
- Repeat enough context that each standalone procedure remains
  understandable.

For a complex command step, use this order:

1. State the action.
2. Show the command.
3. Explain placeholders.
4. Add necessary explanation.
5. Show expected output.
6. Explain the result.

## Format technical content semantically

- Put filenames, paths, commands, flags, methods, classes, fields,
  placeholders, literal input, output, and status codes in `code font`.
- Use the platform's semantic code-block format. In Markdown, use fenced
  blocks with a language identifier when known.
- Follow the project's language-specific coding style.
- Keep code lines near 80 characters where practical.
- Use language-appropriate comments to mark omitted code. Do not use
  unexplained ellipses.
- Explain every non-obvious placeholder. Format generic placeholders
  consistently, such as `PROJECT_ID`.
- Introduce code samples with a complete sentence.
- Include enough surrounding code for the sample to be understandable and
  usable.
- Distinguish commands, output, and explanatory prose.
- Never claim that code was tested unless it was actually executed or
  otherwise verified.

For interface instructions:

- Put visible UI labels in **bold**.
- Match the interface label's wording and capitalization, except that
  sentence case is preferable when the interface uses inconsistent or
  all-uppercase labels.
- Do not put quotation marks around UI labels.
- Focus on the user's goal rather than describing every widget or gesture.
- Name controls instead of relying on position, shape, or color.
- Use menu paths such as **File > New > Document**.

For general formatting:

- Use bold mainly for UI labels and short run-in headings.
- Use italics sparingly for introduced terms or genuine emphasis.
- Reserve underlining for links.
- Use sentence case for titles, headings, captions, table headings, and
  labels.
- Use the serial comma.
- Write *and*, not `&`, except when reproducing an official name, UI label,
  or code.
- Use one space after a period.
- Write dates unambiguously: January 19, 2026. When a numeric-only format is
  required, use 2026-01-19.

## Make links useful

- Link only when the destination materially helps the reader.
- Provide short essential explanations on the current page instead of forcing
  the reader to follow a link.
- Use concise, descriptive link text that makes sense by itself.
- Prefer the destination's exact title or a description of its contents.
- Never use vague text such as *click here*, *this page*, or *read more*.
- Do not normally use a raw URL as link text.
- Link to the most relevant page or heading, not merely a site's home page.
- Avoid repeatedly linking to the same destination.
- Keep punctuation outside linked text.
- Let links open in the current tab by default.
- Explain unexpected behavior, such as downloading a PDF, opening an email
  application, jumping elsewhere on the page, or opening a new tab.

## Build accessibility into the content

- Use semantic headings, lists, tables, links, emphasis, and code elements.
- Ensure instructions can be completed with a keyboard when the product
  permits it.
- Do not rely solely on color, capitalization, shape, screen position, or
  visual proximity to convey meaning.
- Avoid directions such as *above*, *below*, *on the right*, or *the green
  button*. Name the section or control instead.
- Use descriptive link text that remains meaningful when read out of context.
- Give every meaningful image alt text that explains its purpose in context.
- Give decorative images empty alt text.
- Do not introduce information only through an image. Provide equivalent
  explanatory text.
- Do not use screenshots of code, terminal output, or ordinary text when
  actual text can be used.
- Prefer vector or high-resolution images when practical.
- Provide captions, transcripts, or textual descriptions for audio and video.
- Give tables proper headers and introduce them in the surrounding text.
  Avoid merged cells when possible.

## Adapt the presentation to the mode

### Conversation

- Answer the user's real question first.
- Match the requested depth and the reader's demonstrated expertise.
- Use natural paragraphs for straightforward answers.
- Add headings, lists, examples, or code only when they improve
  comprehension.
- Ask a clarifying question only when the missing information would
  materially change the answer. Otherwise, state a reasonable assumption and
  proceed.
- Distinguish established facts, inferences, assumptions, and illustrative
  examples.
- State uncertainty directly rather than disguising it with confident
  language.

### Documentation

- State the document's purpose, audience, scope, prerequisites, and expected
  outcome where relevant.
- Make each section understandable in its immediate context.
- Preserve exact product names, UI labels, identifiers, commands, and casing.
- Keep terminology and formatting consistent across the entire document set.
- Test or validate procedures, links, examples, and code when possible.
- Do not document an imagined interface, command result, feature, or behavior
  as though it were verified.

## Final review

Before sending the response or document, verify that:

- It directly serves the reader's goal.
- The responsible actor, condition, action, and expected result are clear.
- Required and optional actions are distinguishable.
- The most important information appears first.
- Terminology is precise and consistent.
- Headings, paragraphs, and lists are easy to scan.
- Code, commands, UI labels, links, dates, and examples are formatted
  appropriately.
- The content works without relying on color, images, position, or cultural
  knowledge.
- No essential caveat is hidden in a note or buried at the end.
- Filler, repetition, vague language, unnecessary alternatives, and
  unsupported claims have been removed.

## References

This guide follows the Google developer documentation style guide. The
sections below correspond to its pages:

- [Highlights](https://developers.google.com/style) - overall precedence
- [Person](https://developers.google.com/style/person) - audience and voice
- [Tone](https://developers.google.com/style/tone) - conversational register
- [Writing for a global audience](https://developers.google.com/style/translation) - clarity and translation
- [Headings and titles](https://developers.google.com/style/headings) - structure
- [Procedures](https://developers.google.com/style/procedures) - task steps
- [Code samples](https://developers.google.com/style/code-samples) - code formatting
- [UI elements and interaction](https://developers.google.com/style/ui-elements) - interface text
- [Text formatting summary](https://developers.google.com/style/text-formatting) - typography
- [Cross-references](https://developers.google.com/style/cross-references) - links
- [Accessibility](https://developers.google.com/style/accessibility) - inclusive content
