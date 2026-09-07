# native continuation measurement (PN-M1)

What the provider-continuation machinery of #207 and #208 buys, measured
rather than assumed. One deterministic tool workload at four lengths, run
through the Gemini adapter two ways - the conversation rebuilt chat-shaped
every call, and the provider's own continuation with only the tail - with
the tokens, wall time, correctness and native-state growth of every call
recorded. The OpenAI half of the tranche is prepared and not run: the
build environment has no OpenAI key.

Three kinds of statement are kept apart below. **Measured** is what these
runs showed. **Inferred** is what the measurements suggest about
Liminallm. **Not established** is what remains unknown.

## summary

Measured, on `gemini-3-flash-preview` over chains of 3, 10, 20 and 30
sequential tool rounds, three repeats per arm:

- Both arms answered every run correctly, at every length: 30 of 30
  chain runs and 6 of 6 kestrel runs.
- Native continuation cost more total tokens at every length: +49%,
  +46%, +37% and +60% over transcript reconstruction. There is no
  crossover inside 30 rounds.
- The extra cost is exactly the accumulated thinking. At every call of
  every native run, the native prompt tokens less the transcript prompt
  tokens equals the `replay_tokens` the adapter handed to that call. The
  difference was 0 for all 93 calls of the 30-round runs.
- Reasoning tokens showed no systematic saving: +53, +56, −231 and +356
  by length, inside a run-to-run spread of the same size. Native reasoning
  per call has a floor and fewer spikes; transcript reasoning is often 0
  and occasionally large.
- Wall time showed no systematic difference: −5%, +2%, −4%, +10%.
- Native state grows linearly: about 2 parts, 550 bytes and 30
  `replay_tokens` per round on this model.

Inferred: on this provider and workload, native continuation is not a
token optimization, and it is not a latency optimization either. What it
buys is fidelity of the reasoning state, which this workload was too
light to test, since both arms were always right. The `replay_tokens`
accounting of #208 is exact for Gemini.

Not established: everything about OpenAI, including compaction; any
workload where a step needs substantial reasoning; any model that thinks
more per step than this one.

## environment and versions

| item | value |
|---|---|
| repository head | main at `3de428e`, branch `claude/native-measurement-hafb2u` |
| Python | 3.11.15 |
| Gemini wire | native `generateContent` over `httpx` 0.28.1, no SDK |
| Gemini model | `gemini-3-flash-preview`, as the provider served it on September 7, 2026 |
| OpenAI SDK installed | `openai` 2.8.1 (not exercised: no key) |
| reasoning effort | not set in either arm (the backend default) |
| temperature | not set (the adapter sends none unless configured) |
| adapters | none |
| harness | `scripts/ab_continuation.py` at this branch |
| runner | the build container, single process, calls in series |

## methodology

The harness drives a backend adapter directly - `generate_with_tools`,
the same entry the parent calls - and nothing of the service around it.
What is compared is the wire cost of the two ways of continuing with the
same adapter, task, tools, tool results and request profile.

Two arms, identical except for how the conversation reaches the model:

- **transcript**: every call is sent the whole conversation rebuilt from
  the chat-shaped messages the loop kept, with tool calls carrying the
  placeholder thought signature. This is what every round was sent before
  #207 and what a compatible provider is still sent.
- **native**: the first call is the same; every later call is sent the
  accepted `ProviderContinuation` - the selected candidate's complete
  parts with their signatures - and only the tool results since, the way
  the parent sends them.

The workloads are deterministic and scored exactly:

- **chain(n)**: n records, R0 to R(n−1). `next_record(id)` returns the
  record's code (1 to 9, fixed per position) and the id of the next
  record, or says the chain ends. The model is told to read one record
  per turn, starting at R0, and to answer with the sum of the codes when
  the chain ends. Every call depends on the previous result, so the calls
  are sequential by construction, and the answer needs everything learned
  on the way. Correct means the last number in the answer is the sum.
- **kestrel**: the original three-lookup task of #208, part number then
  frequency then calibration offset; correct means the offset appears.

Lengths: 3, 10, 20 and 30 rounds, three repeats of each arm at each
length, runs interleaved transcript then native within a repeat. Thirty
chain runs and six kestrel runs, 36 in all, 426 model calls.

Recorded per call: the provider-reported prompt, output, reasoning,
cached and total tokens, wall time, and the native state as it stood when
the call was made - replay item count, item-type counts, serialized
bytes, the adapter's `replay_tokens`, and whether the reply compacted.
Counts, types, sizes and token counts only; no payload value is read,
logged or stored. The raw runs are in `docs/measurements/`.

No production telemetry was added. The harness instruments itself.

## request profiles

Gemini, both arms: `generateContent` with the workload's `systemInstruction`,
`tools` as function declarations, no `generationConfig` beyond what the
adapter always sends, no `thinkingConfig`. The native arm's `contents` are
the accepted candidate parts replayed whole, signatures where they sat,
then the tool results. The transcript arm's `contents` are rebuilt from
the chat-shaped history with the placeholder signature on every call
part.

No compaction request exists on Gemini. The adapter accepted the
`context_window` hint and ignored it, as designed.

## results: gemini, chain workload

### aggregate

Means over three repeats; standard deviation in parentheses where it is
not zero by construction. Tokens are summed over every call of a run.

| rounds | arm | correct | prompt | reasoning | output | total | wall s | model calls | tool calls | replay items | replay bytes | replay_tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | transcript | 3/3 | 974 | 179 (15) | 59 | 1212 (15) | 4.9 (0.7) | 4 | 3 | 0 | 0 | 0 |
| 3 | native | 3/3 | 1513 (56) | 232 (34) | 59 | 1804 (90) | 4.6 (0.5) | 4 | 3 | 8 | 2985 | 232 |
| 10 | transcript | 3/3 | 4404 | 284 (104) | 192 | 4880 (104) | 10.6 (0.4) | 11 | 10 | 0 | 0 | 0 |
| 10 | native | 3/3 | 6586 (29) | 340 (22) | 192 | 7118 (38) | 10.8 (0.3) | 11 | 10 | 22 | 6170 | 340 |
| 20 | transcript | 3/3 | 13304 | 724 (173) | 392 | 14419 (172) | 20.2 (1.0) | 21 | 20 | 0 | 0 | 0 |
| 20 | native | 3/3 | 18897 (253) | 493 (21) | 392 | 19782 (264) | 19.4 (0.5) | 21 | 20 | 42 | 10750 | 493 |
| 30 | transcript | 3/3 | 27004 | 692 (117) | 593 | 28289 (117) | 28.3 (1.7) | 31 | 30 | 0 | 0 | 0 |
| 30 | native | 3/3 | 43672 (5764) | 1048 (413) | 593 | 45313 (6178) | 31.3 (1.8) | 31 | 30 | 62 | 16954 | 1048 |

Cached tokens were 0 in every run. Every run took exactly one tool call
per round and one final answer: the model never called ahead, never
repeated a record, and never stopped early.

### crossover: native minus transcript

| rounds | total tokens | reasoning tokens | output tokens | wall s | correct native / transcript |
|---|---|---|---|---|---|
| 3 | +592 (+49%) | +53 | 0 | −0.2 (−5%) | 3/3 vs 3/3 |
| 10 | +2238 (+46%) | +56 | 0 | +0.2 (+2%) | 3/3 vs 3/3 |
| 20 | +5363 (+37%) | −231 | 0 | −0.8 (−4%) | 3/3 vs 3/3 |
| 30 | +17024 (+60%) | +356 | 0 | +3.0 (+10%) | 3/3 vs 3/3 |

There is no depth inside 30 rounds at which preserving the provider's
state repays the cost of replaying it, on this model and workload. The
question the tranche asked - at what depth does it repay - has the answer
"not within 30 rounds here", and the mechanism below says why a deeper
run would not change it on this provider.

### per run

| rounds | arm | repeat | correct | prompt | reasoning | output | total | wall s | replay bytes |
|---|---|---|---|---|---|---|---|---|---|
| 3 | transcript | 0 | yes | 974 | 165 | 59 | 1198 | 5.8 | 0 |
| 3 | transcript | 1 | yes | 974 | 173 | 59 | 1206 | 4.7 | 0 |
| 3 | transcript | 2 | yes | 974 | 200 | 59 | 1233 | 4.2 | 0 |
| 3 | native | 0 | yes | 1511 | 220 | 59 | 1790 | 5.4 | 3026 |
| 3 | native | 1 | yes | 1583 | 279 | 59 | 1921 | 4.5 | 3107 |
| 3 | native | 2 | yes | 1445 | 198 | 59 | 1702 | 4.1 | 2822 |
| 10 | transcript | 0 | yes | 4404 | 406 | 192 | 5002 | 11.1 | 0 |
| 10 | transcript | 1 | yes | 4404 | 295 | 192 | 4891 | 10.8 | 0 |
| 10 | transcript | 2 | yes | 4404 | 151 | 192 | 4747 | 10.0 | 0 |
| 10 | native | 0 | yes | 6624 | 333 | 192 | 7149 | 10.9 | 6266 |
| 10 | native | 1 | yes | 6579 | 370 | 192 | 7141 | 10.4 | 6211 |
| 10 | native | 2 | yes | 6555 | 318 | 192 | 7065 | 11.0 | 6032 |
| 20 | transcript | 0 | yes | 13304 | 483 | 392 | 14179 | 19.1 | 0 |
| 20 | transcript | 1 | yes | 13304 | 880 | 391 | 14575 | 20.1 | 0 |
| 20 | transcript | 2 | yes | 13304 | 808 | 392 | 14504 | 21.4 | 0 |
| 20 | native | 0 | yes | 18694 | 506 | 392 | 19592 | 19.3 | 10889 |
| 20 | native | 1 | yes | 18744 | 463 | 392 | 19599 | 18.8 | 10443 |
| 20 | native | 2 | yes | 19253 | 510 | 392 | 20155 | 20.2 | 10918 |
| 30 | transcript | 0 | yes | 27004 | 714 | 593 | 28311 | 30.8 | 0 |
| 30 | transcript | 1 | yes | 27004 | 539 | 593 | 28136 | 27.0 | 0 |
| 30 | transcript | 2 | yes | 27004 | 823 | 593 | 28420 | 27.1 | 0 |
| 30 | native | 0 | yes | 51815 | 1632 | 593 | 54040 | 33.8 | 18865 |
| 30 | native | 1 | yes | 39265 | 747 | 593 | 40605 | 29.8 | 15930 |
| 30 | native | 2 | yes | 39936 | 764 | 593 | 41293 | 30.2 | 16068 |

The transcript arm's prompt tokens are identical across repeats at every
length, because the rebuilt conversation is a function of the workload
alone. The native arm's vary with how much the model thought, because
the thinking is replayed. The 30-round native repeat 0 is the one
outlier: it thought 1632 tokens over the run against about 750 for the
other two, and paid for that thinking again on every later call, which
is where its extra 12,000 prompt tokens come from.

### native state growth

The state handed to each call of the native arm, means over repeats, and
the prompt tokens the provider then reported for that call.

| rounds | call | replay items handed | replay bytes handed | replay_tokens handed | prompt tokens | wall s |
|---|---|---|---|---|---|---|
| 10 | 2 | 2 | 1466 | 135 | 355 | 0.93 |
| 10 | 5 | 8 | 2869 | 185 | 540 | 0.96 |
| 10 | 11 | 20 | 5712 | 311 | 940 | 0.98 |
| 20 | 2 | 2 | 1340 | 117 | 337 | 0.94 |
| 20 | 10 | 18 | 5059 | 250 | 830 | 0.89 |
| 20 | 21 | 40 | 10298 | 466 | 1575 | 0.95 |
| 30 | 2 | 2 | 1447 | 130 | 350 | 0.86 |
| 30 | 10 | 18 | 5546 | 351 | 931 | 1.18 |
| 30 | 20 | 38 | 10735 | 657 | 1715 | 0.94 |
| 30 | 31 | 60 | 16499 | 1019 | 2608 | 1.01 |

Composition of the state at the end of a 30-round native run: 30
`functionCall` parts, 30 `functionResponse` parts, 2 text parts, and 31
parts carrying a `thoughtSignature`. No thought text is in the state;
the reasoning rides as signatures on the call and text parts. Growth per
round, from the 10, 20 and 30-round runs: about 2.1 parts, 540 to 620
bytes, and 25 to 35 `replay_tokens`.

### the replay cost identity

Per-call prompt tokens, transcript against native, means over repeats,
with the `replay_tokens` the native call was handed.

| rounds | call | transcript prompt | native prompt | native minus transcript | replay_tokens handed |
|---|---|---|---|---|---|
| 3 | 4 | 314 | 518 | +204 | 204 |
| 10 | 11 | 629 | 940 | +311 | 311 |
| 20 | 21 | 1109 | 1575 | +466 | 466 |
| 30 | 15 | 818 | 1320 | +502 | 502 |
| 30 | 31 | 1589 | 2608 | +1019 | 1019 |

Measured on every one of the 93 calls of the three 30-round native runs,
against the transcript prompt of the same call: native prompt tokens
minus transcript prompt tokens minus `replay_tokens` handed was 0 - mean
0, minimum 0, maximum 0. The provider charges a replayed signature exactly
the thought tokens it stands for, as input tokens, and the adapter's
estimate of the state's incremental prompt cost is exact on this
provider.

## results: gemini, kestrel workload

The original three-lookup task of #208, repeated for continuity with that
measurement.

| arm | correct | prompt | reasoning | output | total | wall s |
|---|---|---|---|---|---|---|
| transcript | 3/3 | 1205 | 190 (54) | 74 | 1470 (53) | 4.3 (0.3) |
| native | 3/3 | 1568 (23) | 146 (11) | 74 | 1788 (33) | 4.0 (0.3) |

Native minus transcript: +318 total tokens (+22%), −45 reasoning tokens,
−0.3 s (−6%). The same shape as the #208 run: more replay input, less
and steadier thinking, slightly less wall time, and the differences
inside the spread of three repeats.

## openai

Not run. The build environment has no OpenAI key, and every OpenAI item
of the tranche needs one:

- the opt-in live witness `tests/test_openai_native_live.py` - stateless
  calls, `store=false`, no `previous_response_id`, encrypted reasoning
  replayed across a tool round, a forced compaction, a successor request
  from the compacted tape that still answers the opening's question;
- the A/B on the provider ARC measured, `--backend openai`, at the same
  four lengths;
- the compaction effect, `--compact-threshold` forced low on a long
  chain, which the harness records per call: the request before the
  compacting reply, the first after it, and the one after that, with
  replay items, bytes, `replay_tokens`, prompt tokens and wall time.

Both are ready to run with `OPENAI_PROBE_API_KEY` set, and the harness
records the model, SDK version, reasoning effort, requested threshold,
whether a compaction item came back, and the item-type order after it.
The fail-closed rule of #208 - a reply whose compaction stands in for one
of its own calls is refused - stays as it is whatever a live sample
shows; an observation is evidence for a deliberate contract change, not
the change.

## compaction effect

Not established. Gemini has no compaction, so no run here compacted, and
the before-and-after comparison the tranche asked for needs the OpenAI
run above. What is established is the accounting it would be checked
against: on Gemini, `replay_tokens` is exactly the incremental prompt
cost of the state, so a compaction that cut the state would be expected
to cut the next request's prompt tokens by the `replay_tokens` it
removed. Whether OpenAI charges a replayed encrypted reasoning item its
original reasoning tokens, as Gemini charges a signature, is the first
thing that run should report.

## measured, inferred, not established

**Measured**

- Correctness: 36 of 36 runs correct, both arms, every length. No
  difference to report.
- Total tokens: native above transcript at every length, +37% to +60% on
  chains, +22% on kestrel. No crossover inside 30 rounds.
- The extra prompt cost of native is the accumulated `replay_tokens`,
  exactly, at every call measured.
- Reasoning tokens: no systematic difference; native has a per-call
  floor of about 10 and fewer spikes, transcript is often 0 and
  occasionally 600.
- Wall time: no systematic difference; per-call latency is about 0.9 to
  1.0 s in both arms and does not move with prompt size at these sizes.
- Native state grows linearly, about 2 parts, 550 bytes and 30
  `replay_tokens` per round; the state carries signatures, not thought
  text.
- Thinking replayed is paid again on every later call, so a run that
  thinks more early pays more for the rest of the run (30-round native
  repeat 0).

**Inferred**

- On this provider, native continuation is a fidelity mechanism, not an
  efficiency one. It keeps the reasoning state the placeholder discards,
  and the provider charges that state at its full thinking cost on every
  call. Its value therefore rises with how much a step's reasoning is
  worth keeping and falls with how many calls replay it.
- Chains of light steps, where a call thinks 10 to 30 tokens, are the
  worst case for native on Gemini: the state is nearly all overhead.
  This workload was chosen for depth and sequential dependence, and it
  did not make reconstruction expensive, because the model did not need
  to reason much to follow it.
- ARC's aggregate token improvement does not transfer to this provider
  on this workload. Nothing here bears on whether it transfers on
  OpenAI, where the replayed state is encrypted reasoning rather than
  signatures and where compaction exists.
- The #208 accounting model - `replay_tokens` as the current state's
  incremental prompt cost - is confirmed exactly for Gemini. The reserve
  the parent takes off the offer budget is neither over nor under.

**Not established**

- Anything on OpenAI: whether the wire accepts `context_management` on
  the profiled models, whether a compaction item comes back at a forced
  threshold, where it falls against a call, what a replayed encrypted
  reasoning item costs as input, and what compaction does to the next
  request's prompt tokens and latency.
- Whether native continuation improves correctness on a workload where
  a step needs real reasoning, since both arms were always right here.
- Whether a model that thinks more per step, or a reasoning effort set
  high, changes the token balance in either direction: more thinking
  means more to keep and more to pay for.
- Whether cached input tokens change the picture at larger states; no
  run reported any.
- The cost through the service path - the parent's budgeting, offers and
  citations around the adapter - which the harness deliberately leaves
  out.

## limitations

- One provider, one model, one model family's smallest thinking budget.
- Three repeats per cell; enough to see the spread, not to bound it.
- Wall time was measured from one container through a proxy; the per-
  call latency floor is the network's as much as the provider's.
- The workload's steps are light on reasoning by construction, which is
  the fairest test of replay overhead and the least favourable test of
  what replay preserves.
- Adapter-level: the service around the adapter adds costs of its own
  to both arms and was not measured.

## reproduce

```bash
GEMINI_PROBE_API_KEY=... python scripts/ab_continuation.py \
    --backend gemini --model gemini-3-flash-preview \
    --workload chain --rounds 3,10,20,30 --repeat 3 --out runs.jsonl
python scripts/ab_continuation.py --summarize runs.jsonl
```

For OpenAI, with a key, the same command with `--backend openai --model
gpt-5.6`, and `--compact-threshold 8192 --rounds 20` for the compaction
effect. The raw runs behind this report are
`docs/measurements/pn-m1-gemini-chain.jsonl` and
`docs/measurements/pn-m1-gemini-kestrel.jsonl`.
