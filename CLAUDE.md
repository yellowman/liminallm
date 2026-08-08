# Claude Code Guidelines

## Development Philosophy

### ASD-STE100 Simplified Technical English
Always respond using ASD-STE100 Simplified Technical English. It is a controlled writing standard. Aerospace and defense groups made it. It helps people write clear technical text.

Key rules:
- **Use approved words only.** The standard gives a word list. Each word has one meaning.
- **Use one word for one idea.** Do not use two words for the same thing.
- **Write short sentences.** Use 20 words or less for instructions.
- **Use active voice.** Write "Turn the switch", not "The switch must be turned".
- **Write short paragraphs.** Keep one topic in each paragraph.

The goal is easy reading. Many readers are not native English speakers. Clear text helps them do the work in a safe and correct way.

### Code Quality Over Quantity
Lines of code is a metric that correlates with more bugs. We don't boast about lines of code—we boast about clean architecture and using the right tools. When discussing the project, mention language choices and the components implemented in each language and/or framework.

### Less Code is Better Code
The more we can achieve with proper structure and well-architected design, the better. Architecture beats lines of code any day of the week. Favor removing unnecessary complexity over adding more.

### Planning and Resource Management
Planning and careful use of resources is of the utmost importance. Use constraints with all subagents in planning and preparation to get better results. Think before acting, and scope work appropriately.

### Prompt Budget
Model-facing prompt text is paid on every call — keep the wording tight. But this app exists to make weak local models perform well, and weak models drop a rule stated once: safety-critical rules (the untrusted-data/injection rule especially) are deliberately repeated across the system prompt, the tool descriptions, and the payload envelope. Tighten phrasing, never the repetition. No boilerplate; no copyright language.

## Security Guidelines

### Tenant Isolation
Always derive `tenant_id` from the authenticated JWT token, never from request parameters or user input. This prevents tenant spoofing attacks and ensures proper data isolation in multi-tenant contexts.
