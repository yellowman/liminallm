# roadmap

Future work, collected from the SPEC so the canonical document states only
what must remain true today. An entry here is a direction, not a commitment;
none of it is normative.

## delivered phases (historical)

The SPEC's original §14 phase plan — vanilla chat + files, RAG + artifacts,
preferences + persona adapter, clusters + skill adapters, LLM as architect —
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

**mcp server** (SPEC §13.1)

- resources: notes and chunks addressable by uri, subscriptions as a change
  feed.
- prompts: personas and prompt-mode skills offered as mcp prompts.
- `tools/list_changed` notifications when artifacts change the tool set.
- oauth 2.1 + protected-resource metadata (rfc 9728) so standard mcp clients
  onboard without pasting keys.
- structured tool output (`structuredContent`) beside the text.
- the consequential one: an mcp **client** in the kernel loop, consuming
  external servers as tools under the taint discipline — each server
  assigned a taint class, egress withdrawal extended to third-party tools,
  so outside capability never outruns the injection defenses.

**auth / frontend**

- Move the SPA's refresh credential fully out of JS-visible storage. The
  server already sets the refresh token as an HttpOnly cookie (the canonical
  model, SPEC §12.1); the SPA still persists a body-delivered copy in
  sessionStorage and replays it in the refresh body. Removing the JS-visible
  copy is a frontend change plus a cookie-reading refresh path.
- Remove the vestigial `tenant_id` fields the SPA still sends (WebSocket
  init frame, refresh body). The server derives tenant from the host and
  never reads them; the fields are dead weight that misleads readers of the
  client code.

**artifacts / sharing**

- `visibility: "shared"` group scoping (selected users/groups). Today shared
  resolves tenant-wide through the owner's tenant.

**ops**

- OpenTelemetry traces across gateway → orchestrator → router → workflow →
  inference → training (SPEC §15.2 lists the intended spans).
