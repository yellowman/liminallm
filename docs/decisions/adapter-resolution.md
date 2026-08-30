# why adapter weight resolution is this strict

SPEC §5.2 and §5.5 state the rules: version authority outranks path shape,
weights are pinned to adapter and version, composition refuses rather than
partially applies, one validator runs per adapter before composition, and
the gate decides before weights are read. Each rule closed a hole that
shipped. This file records the holes.

## the `latest` pointer served another adapter's weights

Resolution once checked that the `latest` pointer's target was merely
*named* `vNNNN`. That proves a basename, not an identity, so
`A/latest → B/v0001` served adapter B's weights as A's version 1. The check
also enabled nothing legitimate: a pointer aimed at `A/vNNNN` means that
directory exists, and the exact path had already answered.

Handing the resolver the pointer as its starting point broke the other
direction too: it looked for `latest/vNNNN`, so a correctly promoted
adapter became unservable merely because the convenience pointer existed
beside its versions. Resolution therefore starts at the adapter root
(`adapters/<id>`), and the `latest` pointer takes no part in authoritative
resolution - it is refreshed on promotion, best-effort, for humans and
tooling.

Re-raising on a failed pointer write aborted a run *after* the version was
bumped, which left the gate decision unrecorded and let the worker retry
against weights that were already authoritative - which is why the pointer
write is best-effort.

## ownership by layout alone was not ownership

An explicit `fs_dir` may say *where* an adapter's directory lives - a
per-user root, another mount - never *whose* it is. Validating only that it
sat under `fs_root` proved nothing, since every adapter's directory does,
and an artifact naming `adapters/B` had B's weights served as A's version 1.

Layout is therefore checked (the directory containing a `params.json` is
named for its owner) *and* provenance: training records `adapter_id` and
`version` inside each version's `metadata.json`, and a recorded id or
version that disagrees refuses. Provenance catches the case layout cannot
see - a directory renamed to A holding B's run. It is verified when
present rather than required, so a hand-written version fails on
disagreement rather than on absence.

## the versionless lane was removed, not constrained

A lane existed for artifacts without `current_version`: a direct
`params.json` path, a `latest` pointer, a directory scan. Every hole this
section closes had reopened inside it - `latest` aimed elsewhere served
another adapter's weights, a bare `vNNNN` served what a gate-rejected run
leaves behind, and a versionless *hybrid* took weights from its direct file
while the service, reading metadata alone, injected the prompt fallback:
two voices for one adapter, reached because the two sides asked different
questions. It was compatibility code for state the system cannot create,
and deleting it is what made the resolver agree with the data model.

## the version decision comes before the filesystem is touched

Path resolution is not inert - it validates ownership and containment and
refuses on either - so an adapter that authorizes no weights must be
answered from its metadata alone. Resolving first turned an unpromoted
hybrid whose `fs_dir` was stale or out of root into a failed request, when
the correct answer was the prompt fallback and nothing was ever going to
read that path.

## composition: the two obvious alternatives both shipped, both wrong

Gate-weighting `A` and `B` separately and dividing by the total weight
computes `(gA)/g = A` for a lone adapter - the router's gate cancels
itself, so 0.2 and 1.0 behave identically. For two adapters it forms
`B̄Ā`, whose expansion contains `B₁A₂` and `B₂A₁` - products of one
adapter's up-projection with another's down-projection that appear in no
term of the sum. Rank concatenation (`A*` stacked on the rank axis, `B*`
carrying `gαB` per adapter) reproduces the sum exactly, needs no padding
for differing ranks, and lets a zero gate contribute nothing rather than
being normalized back into existence.

## why the validator runs per adapter, before composition

Composition carries only A/B pairs forward, so a foreign key never reaches
a validator that runs afterwards. And concatenation *adds ranks up*: two
adapters that each disagree with themselves (A of rank 2 with B of rank 1,
and A of rank 1 with B of rank 2) compose into a pair whose totals agree -
3 and 3 - while every row pairs with the wrong column. A `scale` is a
scalar attached to a hooked weight, so a projection named only by a
`scale` has no matrices and is refused: such an adapter is non-empty on
the way in and contributes nothing on the way out, which is how one
slipped past the no-silent-drop rule.

Assembly that merely *skips* names it does not recognize still applies the
ones it does: an adapter carrying a single foreign matrix changed the model
through its recognized half while the rest went to a log line. That is why
"never partially applied" is enforced by refusing the whole stack.

## the gate travels on the adapter it gates

`_select_adapters` attaches each gate weight to the activated adapter dict;
the backend reads it there and nowhere else. Returning gates alongside the
adapters for tracing only - which is what used to happen - meant
composition ran every adapter at 1.0 no matter what the policy decided,
and the §5.2 equation was exactly right about a number it never received.

Deciding membership and magnitude separately in each mechanism is how they
came to disagree: composition dropped the zero-gated term while prompt
injection did not read the gate at all, the KV signature hashed an adapter
contributing nothing, and the remote formatter sent a provider `5.0` for
an adapter the kernel had already clamped to `1.0`. One derived set,
carrying the canonical `g`, is the fix the SPEC states.

The local backend also had to hold the line at its own entry: it reported
a zero-gated adapter as the turn's `adapter_id`, and sized its tokenizer
from it, while correctly excluding it from both the LoRA sum and the cache
key.

## `same_base_model` is one implementation, used at both ends

Training asked "same base?" by comparing raw strings while serving compared
path components, so which spelling a deployment happened to store
determined whether an adapter could be trained at all. One implementation
(`transformer.same_base_model`) now answers at both ends of the ladder;
identity is the final path component, case-insensitive, and nothing looser
- family similarity (`-chat`, `-base`, version suffixes) is expressly not
sufficient, because those are different frozen weights and therefore
different models.

## training details that keep the gate honest

- The prompt for an event is bounded by the target message's sequence
  number resolved by id (`seq < target_seq`), never by its position in a
  fetch window: searching for the target inside the window silently
  disabled the bound for any event older than the window - exactly the
  event most likely to have later turns after it.
- Truncation reserves the target span first. Slicing the head of
  `prompt + target` could drop the whole supervised span, leaving an
  all-zero loss mask - an example reporting loss zero, which reads as one
  the model already answers perfectly.
- The holdout number is cross-entropy only, without the training
  objective's L2 term: `B` starts at zero and can only grow, so charging
  the regularizer to the eval counts honest learning as a penalty against
  promotion.
- A run summary that drops `eval_gate`, combined with a default of
  "promoted", marked gate-rejected runs succeeded and credited an adapter
  for a rollout that never happened. Missing means unknown, and unknown is
  not promoted.
- The TOTP-style tokenizer rules (train against the checkpoint's own
  tokenizer, refuse out-of-vocabulary ids rather than clip) exist because
  gradients through the right weights teach nothing transferable if the
  text reached them through an invented token space - and the holdout,
  tokenized the same wrong way, would agree that it worked.
