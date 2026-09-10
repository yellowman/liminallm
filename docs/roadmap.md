# roadmap

Future work, collected from the SPEC so the canonical document states only
what must remain true today. An entry here is a direction, not a commitment;
none of it is normative.

## delivered phases (historical)

The SPEC's original §14 phase plan - vanilla chat + files, RAG + artifacts,
preferences + persona adapter, clusters + skill adapters, LLM as architect -
is delivered. It is recorded here so the SPEC does not instruct an agent to
build tables that already exist. The one durable rule from that section
lives on in SPEC §16: no new hard-coded modes; new behaviors arrive as
artifacts.

## open work, by area

**retrieval / embeddings**

- A scheduled re-embed job. Encoder change is handled by invalidation today
  (a vector whose recorded encoder id differs from the current one reads as
  "not embedded" and is re-embedded lazily); a leader-locked periodic sweep
  would re-embed cold vectors without waiting for a read.
- A backfill job for late-interaction segment vectors. Enabling
  `rag_late_interaction` covers new content only; existing corpora need
  re-ingesting until a backfill exists.

**mcp client** (SPEC §13.1) - delivered

The kernel consumes external mcp servers as ordinary tools:
`liminallm/service/mcp_client.py` turns an admin-owned `mcp.server` artifact
into namespaced kernel tools, classifies each server `egress` or `local_read`
(unknown or malformed reads as `egress`, so a typo fails safe), runs every
call inside `tool_network_guard`, bounds and scans the server's own
discovery metadata, and treats what a server returns as untrusted
third-party data. Witnesses: `tests/test_mcp_client.py`,
`tests/test_mcp_turn.py`, `tests/test_mcp_reachability.py`.

**mcp server** (SPEC §13.1)

Delivered: structured tool output. `note_search` and `knowledge_search`
declare an `outputSchema` and answer with `structuredContent` beside the
prose - the same result set rendered twice, never queried twice. See
`tests/test_mcp_server.py`.

Open, in the order worth doing them:

- resources: notes and chunks addressable by uri. Straightforward and
  immediately useful.
- prompts: personas and prompt-mode skills offered as mcp prompts.
- oauth 2.1 + protected-resource metadata (rfc 9728) so standard mcp clients
  onboard without pasting keys.
- resource subscriptions as a change feed. Deliberately separate from
  addressable resources rather than one bullet with them: subscriptions
  change the transport and lifetime model, which is a much larger commitment
  than an address is.
- `tools/list_changed` notifications, once the tool set can actually change.
  The server's `TOOLS` table is static today, so advertising
  `listChanged: false` is true rather than lazy, and sending the
  notification would be the change that makes it a lie.

**auth / frontend**

Both entries here are done: the browser holds the access token and nothing
else, refresh runs on the HttpOnly cookie alone, and the vestigial
`tenant_id` and `session_id` fields are gone from the refresh body and the
socket's init frame. See docs/ISSUES.md and `tests/test_browser_auth.py`,
which is the browser lane's first witness.

**artifacts / sharing**

- `visibility: "shared"` group scoping (selected users/groups). Today shared
  resolves tenant-wide through the owner's tenant.

**ops**

- OpenTelemetry traces across gateway → orchestrator → router → workflow →
  inference → training (SPEC §15.2 lists the intended spans).
