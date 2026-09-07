# reasoning fidelity across a tool boundary (PN-M2)

PN-M1 measured what Gemini's native continuation costs to replay: the
accumulated thinking, charged again as input on every later call. This
asks what that replayed state buys. Every task here puts a tool boundary
after the model has reasoned and before it answers, with none of the
reasoning in the visible transcript, and compares the successor call two
ways: rebuilt from the visible material with the placeholder thought
signature, or sent the provider's genuine signed candidate state.

Three kinds of statement are kept apart. **Measured** is what these runs
showed. **Inferred** is what the measurements suggest about Liminallm.
**Not established** is what remains unknown.

## summary

Measured, on `gemini-3.8-flash`, three task families in three bands,
four seeds, two repeats, both arms, 144 runs in the main design and 40
in two controls, 184 runs and 368 model calls, every run honouring the
protocol:

- With the reasoning hidden behind the checkpoint, the native arm was
  right 72 of 72 times; the transcript arm 20 of 72 - 13 of 24 light, 2
  of 24 medium, 5 of 24 heavy. In 52 of 72 matched pairs native was
  right where transcript was wrong; the reverse never happened.
- The transcript arm did not recompute. Every one of its 52 wrong
  answers came with 0 successor reasoning tokens: sent a call carrying
  the placeholder signature and "continue", the model answered at once
  and answered wrong. The recomputation ratio the tranche asked for is
  therefore about 0 everywhere; the benefit is correctness, not saved
  reasoning.
- The price was the known one. The successor prompt overhead equalled
  the `replay_tokens` handed to the call on all 92 matched pairs, mean 0,
  minimum 0, maximum 0: 470 tokens per task at light, 990 at medium,
  1510 at heavy. Total tokens per task were +48%, +65% and +73% by band.
  Successor wall time did not differ.
- The trivial control reproduced PN-M1: both arms 8 of 8, native paying
  99 tokens for nothing. The visible-state control removed the effect:
  with the answer put into the checkpoint call's arguments, both arms
  were 12 of 12 on the heavy band, and native still paid its full
  overhead.

Inferred: on this model, transcript reconstruction with the placeholder
signature is not a lower-fidelity approximation at a reasoning boundary;
it loses the derivation outright and the model does not notice. The
provider's continuation is what makes a tool call after reasoning
answerable. The cost is the reasoning replayed once per successor call,
and it is worth paying exactly where PN-M1's workload never went.

Not established: OpenAI; other Gemini models and thinking settings;
whether the successor can be made to recompute; the compounded cost over
many reasoning-heavy rounds.

## environment and versions

| item | value |
|---|---|
| repository head | main at `9a68dbe`, branch `claude/reasoning-fidelity-hafb2u` |
| Python | 3.11.15 |
| Gemini wire | native `generateContent` over `httpx` 0.28.1, no SDK |
| Gemini model | `gemini-3.8-flash`, as the provider served it on September 7, 2026 |
| reasoning effort | not set in either arm (the backend default); no `thinkingConfig` sent |
| temperature | not set |
| adapters | none |
| harness | `scripts/reasoning_fidelity.py` at this branch |
| runner | the build container, single process, calls in series |

## methodology

The harness drives the Gemini adapter directly, through
`generate_with_tools`, the entry the parent calls, and nothing of the
service around it.

**The protocol.** The system prompt tells the model to work the problem
out completely, call `checkpoint(stage=1)` exactly once before stating
any answer, writing nothing else in that turn, and after the tool returns
`"continue"` to reply with the final answer only. The checkpoint call
carries no solution state. Two model calls per run: the first reasons
and calls the checkpoint; the second answers.

**The arms.** The first call is identical in both arms. The successor
differs in one thing:

- **transcript**: rebuilt from the visible material - the system prompt,
  the problem, the assistant's checkpoint call carrying the placeholder
  thought signature, and the tool result "continue". This is what every
  round was sent before #207 and what a compatible provider is still
  sent.
- **native**: the accepted `gemini.native.v1` continuation - the first
  call's request and the selected candidate's complete parts with their
  genuine signatures - and the tool result "continue". Exactly the same
  visible material; the provider's signed state in addition.

**The families.** Deterministic, generated from a seed, scored exactly.

- `order`: runners and clues with exactly one finishing order. The
  generator draws a hidden order, adds clues true of it until the set of
  orders consistent with the clues is that one order alone, checked over
  every permutation, and the harder bands draw only from the weaker clue
  kinds. Bands: 4, 6 and 8 runners, with 3 to 4, 5 to 8 and 10 to 12
  clues. The answer is the order; scored on the names in sequence.
- `arith`: a starting integer and a chain of steps - add, subtract,
  multiply, remainder, reverse the digits, replace by the digit sum times
  a factor - each applied to the previous result. Bands: 4, 8 and 14
  steps. Scored on the last integer in the answer.
- `trace`: a program with two variables and a repeated block of two
  conditionals, traced to its final value. Bands: 3, 6 and 10 passes.
  Scored on the last integer.

**The controls.**

- `trivial`: a single-digit addition under the same protocol. Almost no
  reasoning before the checkpoint, so nothing hidden to preserve: native
  should show its cost and no benefit.
- **visible state**: the heavy band of all three families with the
  checkpoint changed to `checkpoint(stage=1, result=<answer>)`, so the
  transcript carries the result. Reconstruction should then lose
  nothing, and any remaining difference would not be about hidden state.

**Design.** Four seeds per family and band, two repeats of each, both
arms: 144 runs in the main cells. Trivial: four seeds, two repeats, both
arms, 16 runs. Visible-state: four seeds, one repeat, both arms, 24
runs. Arm order was counterbalanced: the arm that ran first alternated
by seed and repeat, so 92 of the 184 runs had native first.

**Measurements.** Per call: prompt, output, reasoning and total tokens
as the provider reported them, wall time, and the state handed to the
call - replay items, serialized bytes, `replay_tokens`. Per run:
correctness, reasoning before and after the checkpoint, and protocol
compliance - whether the checkpoint was called, whether the first reply
leaked answer text, at which call the answer came. Counts, sizes and
token numbers only; nothing of a thought or a signature is read, logged
or stored. The raw runs are in `docs/measurements/pn-m2-*.jsonl`.

No production telemetry was added; the harness instruments itself.

## request profiles

Both arms, first call: `generateContent` with the `systemInstruction`
above, the problem as the user turn, the checkpoint function
declaration, no `generationConfig` beyond what the adapter always sends,
no `thinkingConfig`. Successor call: the transcript arm's `contents` are
rebuilt from the chat-shaped history with the placeholder signature on
the call part; the native arm's are the accepted candidate replayed
whole, then the tool result. The trivial control uses the same profile;
the visible-state control changes only the tool declaration and the
sentence of the system prompt that names it.

## results

### protocol compliance

Every one of the 184 runs called the checkpoint exactly once with no
answer text in that turn, and every one answered at the second call. No
run was excluded, and every cell below has all its runs.

### correctness by band, the three families pooled

| band | arm | n | correct | rate |
|---|---|---|---|---|
| light | transcript | 24 | 13 | 54% |
| light | native | 24 | 24 | 100% |
| medium | transcript | 24 | 2 | 8% |
| medium | native | 24 | 24 | 100% |
| heavy | transcript | 24 | 5 | 21% |
| heavy | native | 24 | 24 | 100% |

### per cell

Means over the eight runs of a cell; standard deviation in parentheses.
The first call is the same request in both arms, so the two arms'
`reasoning before` differ only by the provider's run-to-run variation,
which bounds the noise in everything else.

| family | band | arm | correct | reasoning before | reasoning after | prompt after | total tokens | wall before s | wall after s | replay_tokens | replay bytes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| arith | light | transcript | 6/8 | 322 (84) | 74 (96) | 234 | 854 | 2.05 | 1.16 | 0 | 0 |
| arith | light | native | 8/8 | 352 (104) | 14 (13) | 586 | 1176 | 1.95 | 1.11 | 352 | 2218 |
| arith | medium | transcript | 0/8 | 500 (107) | 0 (0) | 285 | 1060 | 2.07 | 0.88 | 0 | 0 |
| arith | medium | native | 8/8 | 486 (115) | 0 (0) | 771 | 1532 | 2.98 | 0.56 | 486 | 2837 |
| arith | heavy | transcript | 0/8 | 1114 (416) | 0 (0) | 352 | 1808 | 3.94 | 0.55 | 0 | 0 |
| arith | heavy | native | 8/8 | 1084 (352) | 2 (4) | 1436 | 2864 | 3.55 | 1.01 | 1084 | 4653 |
| order | light | transcript | 7/8 | 348 (107) | 36 (96) | 214 | 807 | 2.21 | 0.77 | 0 | 0 |
| order | light | native | 8/8 | 357 (95) | 8 (15) | 571 | 1146 | 1.98 | 0.60 | 357 | 2525 |
| order | medium | transcript | 2/8 | 1021 (156) | 0 (0) | 250 | 1563 | 3.91 | 1.38 | 0 | 0 |
| order | medium | native | 8/8 | 1134 (293) | 0 (0) | 1384 | 2766 | 4.35 | 1.02 | 1134 | 5258 |
| order | heavy | transcript | 5/8 | 1471 (396) | 0 (0) | 292 | 2059 | 5.66 | 0.72 | 0 | 0 |
| order | heavy | native | 8/8 | 1526 (314) | 0 (0) | 1818 | 3639 | 5.16 | 0.69 | 1526 | 6412 |
| trace | light | transcript | 0/8 | 655 (136) | 0 (0) | 293 | 1232 | 2.65 | 0.64 | 0 | 0 |
| trace | light | native | 8/8 | 696 (160) | 2 (6) | 989 | 1970 | 2.54 | 1.00 | 696 | 3107 |
| trace | medium | transcript | 0/8 | 1392 (129) | 0 (0) | 293 | 1969 | 4.50 | 0.73 | 0 | 0 |
| trace | medium | native | 8/8 | 1346 (182) | 0 (0) | 1639 | 3268 | 4.01 | 1.68 | 1346 | 4434 |
| trace | heavy | transcript | 0/8 | 1859 (265) | 0 (0) | 294 | 2437 | 11.01 | 1.91 | 0 | 0 |
| trace | heavy | native | 8/8 | 1914 (288) | 0 (0) | 2208 | 4408 | 5.32 | 0.61 | 1914 | 5563 |

The heavy `trace` transcript cell's first-call wall time of 11.0 s is
one slow first call in the provider; the first call is identical in both
arms, so it says nothing about the arms.

### paired: native minus transcript on matched instances

Matched by family, band, seed and repeat. `successor prompt overhead` is
the native successor's prompt tokens less the transcript successor's.
`ratio` is the successor reasoning the native arm avoided divided by that
overhead - the recomputation ratio.

| family | band | pairs | both right | native only right | transcript only right | both wrong | successor reasoning delta | successor prompt overhead | ratio | successor wall delta s |
|---|---|---|---|---|---|---|---|---|---|---|
| arith | light | 8 | 6 | 2 | 0 | 0 | −60 | +352 | 0.17 | −0.05 |
| arith | medium | 8 | 0 | 8 | 0 | 0 | 0 | +486 | 0.00 | −0.32 |
| arith | heavy | 8 | 0 | 8 | 0 | 0 | +2 | +1084 | 0.00 | +0.46 |
| order | light | 8 | 7 | 1 | 0 | 0 | −28 | +357 | 0.08 | −0.17 |
| order | medium | 8 | 2 | 6 | 0 | 0 | 0 | +1134 | 0.00 | −0.36 |
| order | heavy | 8 | 5 | 3 | 0 | 0 | 0 | +1526 | 0.00 | −0.04 |
| trace | light | 8 | 0 | 8 | 0 | 0 | +2 | +696 | 0.00 | +0.36 |
| trace | medium | 8 | 0 | 8 | 0 | 0 | 0 | +1346 | 0.00 | +0.95 |
| trace | heavy | 8 | 0 | 8 | 0 | 0 | 0 | +1914 | 0.00 | −1.30 |

Over the 72 pairs: both right 20, native only right 52, transcript only
right 0, both wrong 0.

### the successor did not recompute

| band | arm | n | successor reasoning = 0 | mean | max |
|---|---|---|---|---|---|
| light | transcript | 24 | 20 | 37 | 290 |
| light | native | 24 | 16 | 8 | 38 |
| medium | transcript | 24 | 24 | 0 | 0 |
| medium | native | 24 | 24 | 0 | 0 |
| heavy | transcript | 24 | 24 | 0 | 0 |
| heavy | native | 24 | 23 | 0 | 12 |

All 52 wrong transcript answers came with 0 successor reasoning tokens.
The four light-band transcript runs that did reason at the successor -
up to 290 tokens, all in `arith` - were among the ones that answered
correctly. Nothing in the medium or heavy bands reasoned after the
checkpoint in either arm: the native arm answered from its state, and
the transcript arm answered from nothing.

### the replay identity, again

Over all 92 matched pairs, controls included, native successor prompt
tokens minus transcript successor prompt tokens minus the `replay_tokens`
handed to the native successor was 0: mean 0, minimum 0, maximum 0. The
overhead is the pre-checkpoint reasoning, charged again as input, on
this model as on the one PN-M1 measured.

### the controls

| control | family | band | arm | n | correct | reasoning before | reasoning after | successor prompt overhead |
|---|---|---|---|---|---|---|---|---|
| trivial | trivial | light | transcript | 8 | 8/8 | 104 | 30 | 0 |
| trivial | trivial | light | native | 8 | 8/8 | 99 | 24 | +99 |
| visible state | arith | heavy | transcript | 4 | 4/4 | 1160 | 0 | 0 |
| visible state | arith | heavy | native | 4 | 4/4 | 1286 | 0 | +1286 |
| visible state | order | heavy | transcript | 4 | 4/4 | 1576 | 0 | 0 |
| visible state | order | heavy | native | 4 | 4/4 | 1361 | 0 | +1361 |
| visible state | trace | heavy | transcript | 4 | 4/4 | 1852 | 0 | 0 |
| visible state | trace | heavy | native | 4 | 4/4 | 1886 | 0 | +1886 |

The trivial task: both arms right every time, native paying 99 tokens of
overhead and saving 6 tokens of successor reasoning - PN-M1's result in
miniature. The visible-state control: with the answer in the checkpoint
call's arguments, the transcript arm was right 12 of 12 on the same
heavy tasks it lost 19 of 24 of when the answer was hidden, and native
paid its full overhead for no gain. Whatever the native arm bought in
the main cells, it bought by carrying the hidden state.

### tokens and time

| band | transcript total tokens | native total tokens | native minus transcript | successor wall, transcript | successor wall, native |
|---|---|---|---|---|---|
| light | 964 | 1431 | +467 (+48%) | 0.86 s | 0.90 s |
| medium | 1531 | 2522 | +991 (+65%) | 1.00 s | 1.09 s |
| heavy | 2101 | 3637 | +1536 (+73%) | 1.06 s | 0.77 s |

## measured, inferred, not established

**Measured**

- Native continuation across a checkpoint after hidden reasoning: 72 of
  72 correct. Transcript reconstruction with the placeholder signature:
  20 of 72, falling to 2 of 24 at medium and 5 of 24 at heavy.
- Native right where transcript was wrong in 52 of 72 matched pairs; the
  reverse in none.
- The transcript successor did not recompute: 0 reasoning tokens on all
  52 wrong answers, and on every medium and heavy run of either arm.
- The successor prompt overhead of native equals the `replay_tokens`
  handed to it, on every pair. Per task it was the pre-checkpoint
  reasoning: 470, 990 and 1510 tokens by band on average; +48%, +65% and
  +73% of a task's total tokens.
- Successor wall time did not differ between arms.
- With no hidden reasoning to preserve (trivial), or with the state made
  visible (the answer in the call's arguments), the arms did not differ
  in correctness and native paid its overhead for nothing.
- Every run honoured the protocol; the first call is the same request in
  both arms and its reasoning varied run to run by a few hundred tokens.

**Inferred**

- On this model, a call rebuilt with the placeholder signature after
  real reasoning is not a degraded version of the conversation; it is a
  conversation in which the reasoning never happened, and the model does
  not treat it as one to redo. It answers as though its state were
  there. That is a fidelity failure at exactly the boundary the agent
  loop crosses on every tool round.
- The provider's continuation is therefore what makes a tool call after
  reasoning answerable on this model. Its price is the reasoning
  replayed once per successor call - PN-M1's identity, now with a reason
  to pay it. Where the reasoning before a boundary is light, as in
  PN-M1's chains, the price buys nothing measurable; where it is
  substantial, it buys the answer.
- The two measurements together say what the native strategy is for on
  Gemini: not tokens, not latency, but keeping the model's own work
  across the boundaries the parent puts in front of it. The transcript
  remains the security truth of what was offered, asked and run; it is
  not a substitute for the provider's state at a reasoning boundary, and
  the placeholder should be understood as the cost of not having that
  state rather than as an approximation of it.
- The recomputation ratio is the wrong lens for this model: there is no
  recomputation to save. The comparison that matters is correctness at
  the boundary, and it is decisive.
- For the parent's accounting, nothing changes: `replay_tokens` is exact
  here too.

**Not established**

- Anything on OpenAI, where the replayed state is encrypted reasoning
  rather than a signature and compaction exists.
- Other Gemini models, and other thinking settings: whether a `thinking`
  configuration that forces reasoning on the successor would make the
  transcript arm recompute, and at what cost, was not tried; the arms
  here ran the backend's default.
- Whether the transcript arm's failure is specific to the placeholder
  signature as such - an unsigned call is refused by the wire (#207), so
  a call after reasoning has to carry some signature, and the placeholder
  is the only one available without the state.
- The compounded price over many reasoning-heavy rounds in one
  invocation: PN-M1 showed the replayed reasoning is paid again on every
  later call, so a long chain of substantial steps costs the sum of its
  reasoning on each call, and no run here had more than one boundary.
- Latency at scale; the successor calls here were about a second in both
  arms.
- The service path around the adapter - budgeting, offers and citations
  - which the harness deliberately leaves out.

## limitations

- One model, one provider, one thinking setting.
- Eight runs per cell in the main design, four in the visible-state
  control; enough to see the spread of the first call, and the
  correctness gap is far outside it, but not enough to bound small
  effects.
- Tasks are synthetic and closed-form on purpose, so that scoring is
  exact; how much reasoning a real agent step hides behind a tool call
  is a property of the workload, not measured here.
- Wall time was measured from one container through a proxy.

## reproduce

```bash
GEMINI_PROBE_API_KEY=... python scripts/reasoning_fidelity.py \
    --backend gemini --model gemini-3.8-flash \
    --families order,arith,trace --bands light,medium,heavy \
    --seeds 4 --repeat 2 --out fidelity.jsonl
GEMINI_PROBE_API_KEY=... python scripts/reasoning_fidelity.py \
    --backend gemini --model gemini-3.8-flash \
    --families trivial --bands light --seeds 4 --repeat 2 --out fidelity.jsonl
GEMINI_PROBE_API_KEY=... python scripts/reasoning_fidelity.py \
    --backend gemini --model gemini-3.8-flash \
    --families order,arith,trace --bands heavy --seeds 4 --repeat 1 --visible \
    --out fidelity.jsonl
python scripts/reasoning_fidelity.py --summarize fidelity.jsonl
```

`--print-tasks` prints every generated task with its answer without
calling any provider. The raw runs behind this report are
`docs/measurements/pn-m2-gemini-checkpoint.jsonl`,
`docs/measurements/pn-m2-gemini-trivial-control.jsonl` and
`docs/measurements/pn-m2-gemini-visible-control.jsonl`.
