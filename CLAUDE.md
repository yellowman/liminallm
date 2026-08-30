# Claude Code Guidelines

## How to work

Ten rules, adapted from Karpathy's *CLAUDE.md: Field Notes*, with this
repository's own evidence folded into §V. They exist because the failures are
predictable: the model is fast at generating plausible code and slow at
noticing that plausible is not the same as correct, so the discipline has to
come from the process around it.

### I. Read before you write

Read the files you are about to touch - read, not skim. Copy the patterns that
already exist. Check the imports to see what the project actually depends on,
so you do not reach for `axios` where everything is `fetch`. When you cannot
find a pattern, ask rather than guess.

### II. Think before you code

Decide what you are doing before you type. State your assumptions: "add
authentication" is five different things, so name the one you picked and name
the tradeoffs. If something is genuinely confusing, stop and ask rather than
filling the gap with plausible-looking code. That is exactly the code that
passes a casual review and fails when it matters.

### III. Simplicity

Write the minimum code that solves the problem in front of you, not the
minimum that could solve every future version of it. Resist premature
abstraction. Skip error handling for errors that cannot occur. Hardcode values
until there is a real reason to configure them. The test: if the only reason
something is abstracted is "in case we need to", it is over-built.

Lines of code correlate with bugs. Favour removing complexity over adding it,
and prefer structure over volume.

### IV. Surgical changes

Keep the diff as small as the task allows. Do not touch what you were not
asked to touch, match the existing style, and do not reformat - a formatter
pass buries the three lines that matter inside three hundred that do not. The
test is whether you can justify every changed line by the task. If a line is
there because "while I was in there", revert it.

### V. Verification

The gap between code that works and code you think works is testing. When
fixing a bug, write the failing test first, watch it fail, then fix it. That
is the only proof you fixed the cause and not the symptom. Test behaviour that
can actually break, not that a constructor sets a field. If something is hard
to test, that is information about the design, not permission to skip it.

Four rules under it, each earned by a bug this project shipped and a review
had to find.

**Execute before claiming.** Run the code on a real input before you say it
works. Parsers, heuristics and anything with a threshold are the worst
offenders: reading them confirms what you meant, running them shows what they
do. `"mini"` is a substring of `"gemini"`, and no amount of re-reading the
line said so.

**Build test doubles from the real object.** A stub you construct from your own
belief about an interface encodes that belief, so the test passes and the code
is still wrong. If a service resolves a value from its backend, the double must
have a backend. Prefer the real class with a test backend over a hand-made
stand-in.

**Grep the class when you fix the instance.** A reported bug is one sighting of
a shape. Before calling it fixed, search for the same shape elsewhere - the
other retrieval path, the second copy of the list, the unclosed form of the tag
you just handled. Fixes that stop at the reported line leave siblings behind.

**A comment is not evidence.** Writing why the code is correct and writing the
code both come from the same intent, so neither one checks the other. The same
goes for a spec line. Verify first, then describe what you verified.

### VI. Goal-driven execution

Every task needs a success criterion before code is written. "Add validation"
becomes "reject a missing or malformed email, return 400 with a clear message,
and test both cases". For anything multi-step, state the plan first, so the
reader can catch a wrong approach before you spend an hour building it. Scope
the work deliberately, and give subagents explicit constraints in planning and
preparation.

### VII. Debugging

When something breaks, investigate; do not guess. Read the whole error and the
stack trace. Reproduce the problem before you change anything, and change one
thing at a time. Do not paper over an unexpected null with a null check - find
out why it is null, or the bug just moves somewhere quieter.

### VIII. Dependencies

Every dependency is permanent code you do not control. Before adding one, ask
whether the project or the standard library already does it: `crypto.randomUUID()`
over a uuid package. When you do add one, say why, so the choice is visible
rather than smuggled into the manifest.

### IX. Communication

Say what you did and why, not just a block of code. Flag concerns even when you
did exactly what was asked. Be precise about uncertainty: "I am not sure this
library supports streaming" tells the reader what to verify; "I think this
should work" does not.

### X. Common failure modes

Four patterns recur often enough to name:

* **Kitchen sink** - restructuring half the codebase while you are in there.
* **Wrong abstraction** - abstracting before the second or third copy exists.
* **Optimistic path** - the happy path handled and the 500 ignored.
* **Runaway refactor** - a fix that cascades across files.

Catch yourself in any of these and the right move is to stop, not to push
through.

## This repository

### Writing style

Follow the `writing-style` skill (`.claude/skills/writing-style/SKILL.md`) for
conversation, explanations, technical writing, and documentation. It is the
single writing standard for this repository.

Read it before writing prose the reader acts on: chat replies, files under
`docs/`, README content, commit and pull request bodies, code comments, and
error or log messages.

The goal is that a reader can act correctly on the first pass. Many readers are
not native English speakers, so the skill requires literal, culturally neutral
language, one idea per sentence, and consistent terminology.

When describing the project, name the language choices and the components
implemented in each language or framework.

### Prompt budget

Model-facing prompt text is paid on every call - keep the wording tight. But
this app exists to make weak local models perform well, and weak models drop a
rule stated once: safety-critical rules (the untrusted-data and injection rule
especially) are deliberately repeated across the system prompt, the tool
descriptions, and the payload envelope. Tighten phrasing, never the repetition.
No boilerplate; no copyright language.

### Which tests to run

Pick the smallest lane that covers what changed.

* **Normal change:** `make test-fast-xdist`. The default and usually the only
  run - about two minutes.
* **The change touches a slow-marked subsystem:** `make test-xdist`, which is
  the same lane with nothing deselected. The slow set is not a separate lane
  and needs no separate one: the per-worker Postgres, Redis database and
  filesystem root that make the fast lane safe in parallel are not specific to
  a marker.
* **Full serial suite (`make test`):** only when the thing under test is
  inherently about single-process or global behaviour, or when a broad harness
  change could alter serial semantics. Not as a release gate - `test-xdist`
  covers the same tests - and not every commit.

Parallelism buys more on the slow set than on the fast one, because what makes
a test slow is usually waiting. Measured on a 4-core box: the 110 slow-marked
tests alone take 5m37s serially and 1m43s at `-n 4`, and the whole non-browser
suite - 2,814 tests - takes 3m37s. Running the fast lane and then the full one
as a routine pair still buys nothing, but the full one is now cheap enough to
be the local gate whenever there is any doubt, and it is what `make qa` runs.
CI is a separate signal and runs the same selection serially on each supported
Python version - a green lane here does not answer whether the suite passes on
an interpreter this machine does not have.

`pytest tests/ -m slow --collect-only -q` lists the slow set and which files
own it. The browser lane stays out of both (`make test-browser`) because it
needs a Chromium binary the dev extra does not install.

### Code review runs its verification pass

The owner has authorized multi-agent orchestration for code review. Run every
review as a fan-out of finders across distinct angles, then verify each finding
with **independent adversarial verifiers** before reporting it - one that must
reproduce the failure by running code, one that argues the finding is wrong and
defaults to rejecting it when uncertain. A finding survives only if both agree.

This standing authorization covers code review. It does not extend to
orchestrating other work; ask for that separately.

The reason is measured, not stylistic. Five single-pass reviews of one branch
each missed defects the next pass found, and several of those were bugs the
previous pass's *fix* introduced. Reading alone converged on nothing.

### Security: tenant isolation

Always derive `tenant_id` from the authenticated JWT token, never from request
parameters or user input. This prevents tenant spoofing attacks and ensures
proper data isolation in multi-tenant contexts.
