# frontend layout and styling

Implementation detail behind SPEC §17. The SPEC states the frontend's
behavioral contract - thin client, no domain intelligence, tenant from the
site, streaming and cancellation semantics. This file records the current
layout, styling, and client implementation patterns. The frontend source
(`frontend/`) is authoritative where this file lags.

## layout architecture

- **sidebar-main layout**: persistent conversation list sidebar (280px) with
  main content area.
- **tab navigation**: Chat, Notes (hidden when `notes_enabled` is off),
  Contexts, Files, Artifacts, Tools, Insights, Settings.
- tab data loads lazily on first activation; login preloads only what the
  chat needs.
- responsive breakpoints: sidebar hidden on mobile (<1080px), single-column
  tabs on small screens (<640px).

## conversation sidebar

- paginated conversation list sorted by `updated_at`.
- client-side filter by title or conversation ID.
- highlight on the currently loaded conversation.
- conversations with `source: "responses"` carry a small "api" tag -
  agent-created threads sit beside native ones, visibly distinct.
- a new-conversation button resets chat state.
- endpoints: `GET /v1/conversations`, `GET /v1/conversations/{id}`,
  `GET /v1/conversations/{id}/messages`.

## chat view

- scrollable message stream with bubbles differentiated by role.
- WebSocket streaming primary with HTTP fallback; blinking cursor while
  streaming; cancel via connection close.
- inline clickable citation links from `content_struct.citations`, source
  path as tooltip.
- context binding dropdown for the active `knowledge_context`; optional
  `workflow_id` text input.
- optimistic UI: user messages render before server confirmation.
- assistant prose is set in a serif column with a github-grade markdown
  renderer (escape-first: html-escape, then rewrite to a fixed safe tag set;
  nested/task lists, aligned tables, blockquotes, autolinks, backslash
  escapes, lightweight syntax highlighting across nine language families).
  Streaming batches DOM writes with `requestAnimationFrame`, auto-closes a
  dangling code fence mid-stream, and only auto-scrolls when the reader is
  already near the bottom.
- a copy button on every user and model message.
- while the agent loop runs, the typing indicator names the tool in flight;
  injection findings surface as a warning on the message.
- attachments: drag-and-drop or attach-button chips in the composer; uploads
  bind to the conversation automatically (SPEC §19.5 tier 1).
- turn rail: a right-hand rail of tick marks, one per turn, labeled with a
  model-written description. Hover or focus expands the bars into a
  selector; moving away collapses them. Conversation titles are
  model-written too - never raw uuids.

## context manager

- creation form (name required, description optional) posting to
  `POST /v1/contexts`.
- card list showing name, description, ID prefix, creation date; click to
  load details.
- details panel: full ID, description, visibility badge, timestamp, chunk
  count and recent-chunk preview via `GET /v1/contexts/{id}/chunks`.
- chat and upload dropdowns populate from `state.contexts` and update on
  context CRUD.

## artifact browser

- filters: type (all, workflow, policy, adapter, tool) and visibility (all,
  private, shared, global).
- sortable table: type, name, visibility, version, updated date.
- type badges color-coded (workflow=blue, policy=pink, adapter=green,
  tool=amber); visibility badges color-coded (private=red, shared=amber,
  global=green).
- details panel with syntax-highlighted JSON of `artifact.schema` and a
  version history table.

## settings panel

- session information: user ID, role, tenant, truncated session ID, as
  hairline-divided `.detail-row` pairs rather than filled boxes.
- local storage management: draft count, clear-drafts, export-drafts (JSON
  download).
- upload limits from `GET /v1/files/limits`.
- api keys: mint/list/revoke against `/v1/auth/api-keys`; plaintext renders
  once at mint time; revoke confirms before firing.
- about section: version and build info from `/healthz`.

## draft persistence

- drafts in localStorage under `liminal.drafts` as
  `{ [conversationId]: { text, savedAt } }`; new-conversation drafts under
  `_new`.
- auto-save with a 1-second debounce; restoration on conversation load;
  draft-count indicator in the composer.

## file upload panel

- a single strip in the Files tab - file, context, and the upload button on
  one line; context dropdown (or private/no context); chunk size (64-4000,
  validated) behind an `Advanced` disclosure; client-side size and extension
  checks before upload; inline progress and result feedback.

## feedback controls

- thumbs up/down, disabled until an assistant message exists; voice input
  and read-aloud as icon buttons beside them; optional notes field; target conversation/message display; JSON preview of adapters,
  context snippets, gates, and routing/workflow traces.
- endpoint: `POST /v1/preferences`.

## client API patterns

- request headers: `Authorization: Bearer`, `Idempotency-Key`
  (auto-generated UUID). No tenant header - the tenant is derived
  server-side from the host (SPEC §12.2); the client never sends one.
- envelope handling: parse `{ status, data, error }`; error text from
  `error.message` or `detail`.
- retry logic: exponential backoff (400ms base, 3 retries) for 5xx; no retry
  on 4xx; on 401, one refresh attempt before failing.
- WebSocket: connect to `/v1/chat/stream`; the initial frame carries exactly
  one of `access_token` or `session_id`, plus
  `{ message, conversation_id?, context_id?, workflow_id?, stream?: bool }`
  (SPEC §13.7). Streaming events
  `{ event: "token"|"trace"|"message_done"|"error"|"cancel_ack", data }`;
  `stream: false` yields a single envelope.

## styling system

- CSS custom properties for theming: `--accent`, `--text`, `--panel`,
  `--border`, and so on. Control sizing is tokenized too: `--ctl-h`,
  `--ctl-h-sm`, `--icon-hit`, `--icon-glyph`, `--chip-h`.
- list classes: `.row` is the one flat list primitive. It carries
  `.row-icon`, `.row-name`, `.row-meta` and hover-revealed `.row-actions`,
  and a selected row takes a faint fill and a 2px left marker - the same
  selection language the application rail uses. Every list wears it:
  Contexts, Artifacts, Tools, Workflows, Files, Insights, API keys, and -
  through the same rule rather than a copy of it - `.conversation-item` and
  `.note-item`, which keep their own names for the states they carry
  (`active`, `contradicted`, `evolved`).
- component classes: `.panel`, `.badge` (a bar title with a status dot, not
  a capsule), `.table`, `.code-block`, `.detail-row` with `.detail-label`
  (a label and its value, divided by a hairline), `.factline` (a
  dot-separated line of facts), `.chip`, `.figures`, `.icon-btn`.
- utility classes: `.hidden`, `.flex-row`, `.pill-row`, `.divider`, `.mb-14`,
  `.monospace`.
- media queries at 1080px (hide sidebar) and 640px (single-column layout).
