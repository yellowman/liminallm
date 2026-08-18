# Claude Code Guidelines

## Development Philosophy

### Writing style
Follow the `writing-style` skill (`.claude/skills/writing-style/SKILL.md`) for
conversation, explanations, technical writing, and documentation. It is the
single writing standard for this repository.

Read it before writing prose the reader acts on: chat replies, files under
`docs/`, README content, commit and pull request bodies, code comments, and
error or log messages.

The goal is that a reader can act correctly on the first pass. Many readers
are not native English speakers, so the skill requires literal,
culturally neutral language, one idea per sentence, and consistent
terminology.

### Code Quality Over Quantity
Lines of code is a metric that correlates with more bugs. We don't boast about lines of code—we boast about clean architecture and using the right tools. When discussing the project, mention language choices and the components implemented in each language and/or framework.

### Less Code is Better Code
The more we can achieve with proper structure and well-architected design, the better. Architecture beats lines of code any day of the week. Favor removing unnecessary complexity over adding more.

### Planning and Resource Management
Planning and careful use of resources is of the utmost importance. Use constraints with all subagents in planning and preparation to get better results. Think before acting, and scope work appropriately.

### Prompt Budget
Model-facing prompt text is paid on every call — keep the wording tight. But this app exists to make weak local models perform well, and weak models drop a rule stated once: safety-critical rules (the untrusted-data/injection rule especially) are deliberately repeated across the system prompt, the tool descriptions, and the payload envelope. Tighten phrasing, never the repetition. No boilerplate; no copyright language.

### Verification

Three rules, each earned by a bug this project shipped and a review had to find.

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
a shape. Before calling it fixed, search for the same shape elsewhere — the
other retrieval path, the second copy of the list, the unclosed form of the tag
you just handled. Fixes that stop at the reported line leave siblings behind.

A related discipline, because it caused as much damage as the three above: a
comment or a spec line is not evidence. Writing why the code is correct and
writing the code both come from the same intent, so neither one checks the
other. Verify first, then describe what you verified.

### Code Review Runs Its Verification Pass

The owner has authorized multi-agent orchestration for code review. Run every
review as a fan-out of finders across distinct angles, then verify each finding
with **independent adversarial verifiers** before reporting it — one that must
reproduce the failure by running code, one that argues the finding is wrong and
defaults to rejecting it when uncertain. A finding survives only if both agree.

This standing authorization covers code review. It does not extend to
orchestrating other work; ask for that separately.

The reason is measured, not stylistic. Five single-pass reviews of one branch
each missed defects the next pass found, and several of those were bugs the
previous pass's *fix* introduced. Reading alone converged on nothing.

## Security Guidelines

### Tenant Isolation
Always derive `tenant_id` from the authenticated JWT token, never from request parameters or user input. This prevents tenant spoofing attacks and ensures proper data isolation in multi-tenant contexts.
