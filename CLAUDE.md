# Claude Code Guidelines

## Development Philosophy

### Code Quality Over Quantity
Lines of code is a metric that correlates with more bugs. We don't boast about lines of code—we boast about clean architecture and using the right tools. When discussing the project, mention language choices and the components implemented in each language and/or framework.

### Less Code is Better Code
The more we can achieve with proper structure and well-architected design, the better. Architecture beats lines of code any day of the week. Favor removing unnecessary complexity over adding more.

### Planning and Resource Management
Planning and careful use of resources is of the utmost importance. Use constraints with all subagents in planning and preparation to get better results. Think before acting, and scope work appropriately.

### Prompt Budget
Every token of model-facing prompt text (system instructions, tool descriptions, preambles, envelopes) is paid on every call and displaces reasoning about the user's problem. State each rule exactly once, in the place closest to what it governs — e.g. the untrusted-data rule lives in the envelope that wraps each payload, and everything else points to it. No boilerplate; no copyright language.

## Security Guidelines

### Tenant Isolation
Always derive `tenant_id` from the authenticated JWT token, never from request parameters or user input. This prevents tenant spoofing attacks and ensures proper data isolation in multi-tenant contexts.
