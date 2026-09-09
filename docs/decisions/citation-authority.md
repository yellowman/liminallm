# why a citation is authority and not a string

SPEC §2.2 states the durable end of this: a citation segment on an assistant
message is produced *only* by validating the model's own markers against the
handles that turn issued, and one arriving from anywhere else is dropped.

How a turn comes to have handles, what happens to them in flight, and when
an execution may issue them at all are not in the SPEC. They are spread
across five tranches and one file's worth of comments, and a reader arriving
at any one of them cannot see the rule the others keep. That is what this
file is for.

Read it before changing how handles are minted, what is scrubbed, or when
citations may be granted.

## the distinction everything rests on

Two questions, and confusing them is the failure mode:

**Authority** - may this execution grant a citation? That is a policy
question. It is decided per execution, it can be withdrawn mid-turn by an
operator, and withdrawing it costs nothing but citations.

**Containment** - must this execution's namespace be removed from what
leaves the parent? That is a fact about what the model has already been
shown. It cannot be withdrawn, because the provider cannot be made to unsee
a handle.

Every gate in the system is one or the other. A gate that reads the wrong
one is either a feature that does not turn off or a namespace that leaks.

## trusted transcript versus citation authority

The trusted transcript is the parent's own record of the turn: what was
asked, what the worker was served, what the model replied. It is the
security and control-flow truth - a round runs only when the parent's record
asked for it (#205), and a worker is served only the capabilities its body
declares.

It is deliberately **not** an execution-state substitute. The transcript
says a turn happened; it does not say what the provider still holds. That
distinction was settled by measurement on Gemini (#210): reconstructing a
conversation from the transcript loses reasoning the provider's own
continuation keeps, so provider-native continuation is correctness state,
not an efficiency optimization.

For citations the consequence is narrower and sharper: a handle the
transcript records as having been shown is a handle the model may write
again, whatever the parent decides afterwards.

## where handles are minted

From **bindings**, never from the source registry. The registry is
everything the turn consulted, including chunks the prompt budget dropped
and sources a failed node retrieved. The bindings are what may support this
answer. A source with no binding is not citable however well the model
describes it.

The name is not `src_#`. Registry ids restart at `src_1` every turn and
history is replayed verbatim into later prompts, so yesterday's `[src_1]`
would resolve to today's unrelated document - and a retrieved page, a note
or an earlier message can contain that string just as cheaply, all of which
reach the model as data. So each turn mints a nonce and offers
`[cite:K7Q2ABCD-1]`, with the mapping kept parent-side.

Stated without overclaiming: a wrong guess resolves to nothing, and a
correct guess would misattribute one span among the sources this turn
already grounded on. It reaches nothing the turn did not read.

One handle per source, so two routes that dedupe to one source share one
citation identity rather than offering the model two names for one document.

The table is grown only through `Invocation.extend_citations`, which is the
one mutator and refuses on two separate grounds: the budget is spent, or
the deployment withdrew authority.

## why the model's own output is what a citation is read from

A citation is resolved out of the **canonical** text - what the provider
sent, unedited - and never out of what a worker returns or what a consumer
accumulated.

The worker gets the answer with the namespace removed and sends back what it
claims the result is. Those two strings are compared exactly, and only then
are markers read, out of the parent's copy. A worker that changed one word
transfers nothing; a worker that writes a marker of its own is writing into
a string nobody parses.

## namespace containment

Once an execution has issued a handle, that namespace is removed from
everything that crosses to the worker or the client, for the rest of that
execution.

The rule is keyed on `invocation.citations` alone - on whether a handle was
actually issued - and on nothing else:

- Not on whether the feature is enabled now. It can be turned off mid-turn
  (#213) and the handles remain in the model's context.
- Not on the namespace existing. Every turn mints a nonce whether or not
  anything is offered, so an execution that issued nothing is not filtered,
  and an answer that happens to contain its unused nonce survives byte for
  byte. Filtering it would be editing prose on the strength of a
  coincidence.

The blocking transport scrubs the whole serialized reply and asserts the
result, deliberately unkillable by mutation: what that assertion guards is
the next model-controlled field somebody adds, not today's.

## the streamed answer and its oracle

`scrub_positions` is the definition of what the public text is and where
every character came from. A streamed answer has no finished string until it
is over, so the reader performs the same transformation incrementally - it
emits a prefix and holds a suffix, releasing only what can no longer change.

The relationship between the two is fixed (#212):

- The incremental result is **never** authoritative for final coordinates.
- `scrub_positions` is asked exactly once, at completion, for the finished
  text and the origin map.
- What was released is compared to that answer as **equality, in both
  directions**. A reader that released text the scrub removes cannot take it
  back; a reader that released less than the scrub keeps would be completed
  as a truncated reply with a success stamp on it. Both fail the completion,
  and the difference is never supplied - appending it would make the oracle a
  repair mechanism rather than a verifier.

The reader is a chain of passes with the same shape as `_scrub_text`: remove
every leftmost non-overlapping match, hand what survives to the next pass,
and add a pass exactly when one above removes something. How deep the chain
goes is the model's choice, so it is driven **and closed** iteratively -
under the ceiling a legitimate answer can need more passes than Python has
stack.

## the runtime kill switch

`citation_offers_enabled` is a managed setting (#213). It is asymmetric, and
the asymmetry is the whole design:

- **Off** is immediate and one-way *within an execution*. An execution live
  when an operator rolls back loses authority there and then - no further
  handle, no instruction, no labels, no final transfer - and never regains
  it. The answer is untouched: nothing is cancelled, and the turn finishes
  as an ordinary uncited one.
- **On** is prospective. It reaches executions opened afterwards and no
  others. A turn that ran half its rounds without authority would otherwise
  write an answer quoting handles from prompts the parent can no longer
  describe as one conversation.
- **Containment is not affected in either direction.** See above. This is
  the property most likely to be lost by anyone replacing the switch
  mechanically.

The policy is snapshotted onto the execution when it opens, under the same
lock `InvocationRegistry.open` takes, so an execution is either caught by
the rollback's sweep or born disabled - there is no third outcome. It is
kept separate from `citation_budget_intact`: one says the deployment
withdrew authority, the other says the prompt arithmetic did not fit, and
collapsing them makes an operator's rollback read in the logs as a prompt
that was too long.

## why the execution trace dies at the API boundary

`workflow_trace` is an internal execution diagnostic (#211). It carries node
outputs, tool arguments and results, error text, context snippets,
provenance and injection evidence - everything the turn touched, in the
shape it touched it.

It is not conversation history and not client-visible state. So it does not
cross `chat_turn.public()`, it is not persisted with the turn, and it is not
logged: the diagnostic that reaches structlog is an **allowlisted
projection** - trace length, and per node a name and a status drawn from a
bounded set - never a recursive sanitization of the trace itself. An
allowlist, because a denylist admits every field somebody adds later.

Its lifetime, in one line:

    workflow executes -> transient trace -> sanitized summary -> gone

In-process callers that genuinely need the trace take it through
`trace_sink`, which never leaves the process.

## breadcrumbs

| tranche | what it established | commit |
|---|---|---|
| #204 | citation identity: bindings, per-turn nonce, handles, ordered occurrences, transfer out of canonical text | `8aa83d3` |
| #205 | per-worker capability ACL; a round runs only what the parent's record asked for | `86b1009` |
| #211 | `workflow_trace` stops at the execution boundary; allowlisted log projection | `701c37b` |
| #212 | the streamed reader is linear in the answer and independent of chunking; the oracle is asked once and compared as equality | `3307c1f` |
| #213 | citation offers become a runtime kill switch; authority off, containment unchanged | `27647d7` |

Measurement that shaped the surrounding decisions: #209 (`9a68dbe`) and #210
(`6e96e43`), on what provider-native continuation preserves that a
reconstructed transcript does not.
