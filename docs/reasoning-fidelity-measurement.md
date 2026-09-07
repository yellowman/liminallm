# reasoning fidelity across a tool boundary (PN-M2)

PN-M1 measured what Gemini's native continuation costs to replay: the
accumulated thinking, charged again as input on every later call. This
asks what that replayed state buys. Every task here puts a tool boundary
after the model has reasoned and before it answers, with none of the
reasoning in the visible transcript, and compares the successor call two
ways: rebuilt from the visible material with the placeholder thought
signature, or sent the provider's genuine signed candidate state.

Two experiments, on the same tasks. The **forked** experiment makes the
pre-checkpoint call once per task and builds both successors from that
exact reply, so the reasoning event, the call, the tool result and the
visible transcript are literally shared and only the successor's
representation differs. It is the causal confirmation and the headline.
The **independent-first-call** experiment, run first, let each arm make
its own first call; a pair there shares the task but sampled the first
call twice. It is kept as exploratory evidence.

Three kinds of statement are kept apart. **Measured** is what these runs
showed. **Inferred** is what the measurements suggest about Liminallm.
**Not established** is what remains unknown.

## summary

Measured, on `gemini-3.8-flash`, in the forked experiment - three task
families in three bands, four seeds, two repeats, one shared first call
per fork, 72 forks in the main design and 20 in the two controls, 92
forks and 276 model calls, every first call honouring the protocol:

- From the same first reply, the native successor was right 72 of 72
  times and the transcript successor 26 of 72 - 14 of 24 light, 4 of 24
  medium, 8 of 24 heavy. In 46 of 72 forks native was right where
  transcript was wrong; the reverse never happened; no fork had both
  wrong.
- The transcript successor did not recompute. Of its 46 wrong answers,
  43 came with 0 successor reasoning tokens and the other three with at
  most 33: sent the same call carrying the placeholder signature and the
  same "continue", the model answered at once and answered wrong.
- The price was the known one. The native successor's prompt overhead
  equalled the `replay_tokens` handed to it in all 92 forks, and that
  number equalled the shared first call's reasoning tokens in all 92:
  465, 957 and 1457 prompt tokens by band. Per task, native cost 452,
  914 and 1425 more total tokens, +47%, +59% and +68%. Successor wall
  time did not separate the arms: under a second on average in both,
  native 0.10 s faster on average with a spread of 0.81 s.
- Which successor ran first made no difference: transcript was right 14
  of 36 when it ran second and 12 of 36 when it ran first.
- The trivial control: both successors 8 of 8, native paying 112 tokens
  for nothing. The visible-state control: with the answer put into the
  checkpoint call's arguments, both successors were 12 of 12 on the heavy
  band, and native still paid its full overhead.

The independent-first-call experiment, 184 runs, had shown the same
shape: native 72 of 72, transcript 20 of 72, 52 native-only-right pairs,
0 successor reasoning on every wrong transcript answer, both controls
null. The fork removes the one ambiguity that experiment left - that the
two arms' first calls had reasoned differently - and the gap is
unchanged.

Inferred: on this model, transcript reconstruction with the placeholder
signature is not a lower-fidelity approximation at a reasoning boundary;
two successors of the same provider reasoning state diverge solely
because one is sent the provider's continuation and the other a
reconstruction, and the reconstruction loses the derivation outright
without the model noticing. The provider's continuation is what makes a
tool call after reasoning answerable. Its cost is the reasoning replayed
once per successor call, and it is worth paying exactly where PN-M1's
workload never went.

Not established: OpenAI; other Gemini models and thinking settings;
whether the successor can be made to recompute; the compounded cost over
many reasoning-heavy rounds.

## environment and versions

| item | value |
|---|---|
| repository head | main at `9a68dbe`, branch `claude/reasoning-fidelity-hafb2u` |
| Python | 3.11.15 |
| Gemini wire | native `generateContent` over `httpx` 0.28.1, no SDK |
| Gemini model | `gemini-3.8-flash`, as the provider served it on September 7, 2026, both experiments |
| reasoning effort | not set in either arm (the backend default); no `thinkingConfig` sent |
| temperature | not set |
| adapters | none |
| harness | `scripts/reasoning_fidelity.py` at this branch (`--design forked` and `--design independent`) |
| runner | the build container, single process, calls in series |

## methodology

The harness drives the Gemini adapter directly, through
`generate_with_tools`, the entry the parent calls, and nothing of the
service around it.

**The protocol.** The system prompt tells the model to work the problem
out completely, call `checkpoint(stage=1)` exactly once before stating
any answer, writing nothing else in that turn, and after the tool returns
`"continue"` to reply with the final answer only. The checkpoint call
carries no solution state. Two model calls per task: the first reasons
and calls the checkpoint; the second answers.

**The arms.** The successor differs in one thing:

- **transcript**: rebuilt from the visible material - the system prompt,
  the problem, the assistant's checkpoint call carrying the placeholder
  thought signature, and the tool result "continue". This is what every
  round was sent before #207 and what a compatible provider is still
  sent.
- **native**: the accepted `gemini.native.v1` continuation - the first
  call's request and the selected candidate's complete parts with their
  genuine signatures - and the tool result "continue". Exactly the same
  visible material; the provider's signed state in addition.

**The two designs.**

- **forked** (the confirmation): the first call is made once per task.
  From that one reply the native successor is built from the
  `ProviderContinuation` the reply produced, with only the "continue"
  tool result as its tail; the transcript successor is built from the
  chat-shaped assistant message the same reply produced, the same tool
  result appended, and no continuation, so the adapter inserts the
  placeholder signature. Each successor works on its own copies; neither
  touches what the other is given. The order in which the two successors
  run alternates by seed and repeat. A first reply that breaks the
  protocol - no checkpoint, more than one call, answer text in the turn,
  or any argument beyond `stage` under the hidden-state condition - is
  recorded and gets no successors.
- **independent** (exploratory): each arm ran its own two calls, so a
  native and a transcript run of one task shared the task and the request
  but sampled the first call twice. The order in which the two arms ran
  alternated by seed and repeat.

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

**Design sizes.** Forked: four seeds per family and band, two repeats,
72 forks in the main cells; trivial, four seeds and two repeats, 8 forks;
visible-state, four seeds and one repeat, 12 forks; 92 forks, 276 calls.
Independent: the same matrix as runs per arm, 144 in the main cells, 16
trivial, 24 visible-state; 184 runs, 368 calls.

**Measurements.** Per call: prompt, output, reasoning and total tokens
as the provider reported them, wall time, and the state handed to the
call - replay items, serialized bytes, `replay_tokens`. Per fork or run:
correctness, reasoning before and after the checkpoint, and protocol
compliance. Counts, sizes and token numbers only; nothing of a thought
or a signature is read, logged or stored. The raw runs are in
`docs/measurements/pn-m2-*.jsonl`, the forked ones under
`pn-m2-gemini-forked-*`.

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

## results: the forked confirmation

### protocol compliance of the shared first call

All 92 first calls called the checkpoint exactly once, wrote nothing
else in that turn, and - in the 80 hidden-state forks - passed no
argument beyond `stage`. No fork was excluded; every cell below has all
its forks, and every successor answered at its first call.

### correctness by band, the three families pooled

| band | forks | native successor right | transcript successor right | native only right | transcript only right | both wrong |
|---|---|---|---|---|---|---|
| light | 24 | 24 | 14 | 10 | 0 | 0 |
| medium | 24 | 24 | 4 | 20 | 0 | 0 |
| heavy | 24 | 24 | 8 | 16 | 0 | 0 |

### per cell: the shared first call and the two successors

Means over the eight forks of a cell; the first call's reasoning has its
standard deviation in parentheses. Because the first call is shared, a
cell has one `shared reasoning before`, not one per arm.

| family | band | shared reasoning before | native right | transcript right | native only right | native successor reasoning | transcript successor reasoning | native successor prompt | transcript successor prompt | overhead | native successor wall s | transcript successor wall s | replay_tokens | replay bytes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| arith | light | 354 (94) | 8/8 | 6/8 | 2 | 3 | 35 | 587 | 234 | +354 | 0.95 | 1.06 | 354 | 2262 |
| arith | medium | 509 (122) | 8/8 | 1/8 | 7 | 1 | 1 | 794 | 285 | +509 | 0.64 | 0.67 | 509 | 2957 |
| arith | heavy | 1068 (269) | 8/8 | 2/8 | 6 | 0 | 94 | 1420 | 352 | +1068 | 0.70 | 1.00 | 1068 | 4513 |
| order | light | 371 (108) | 8/8 | 8/8 | 0 | 3 | 19 | 585 | 214 | +371 | 0.66 | 0.79 | 371 | 2502 |
| order | medium | 1000 (300) | 8/8 | 3/8 | 5 | 0 | 57 | 1249 | 250 | +1000 | 0.59 | 1.00 | 1000 | 4649 |
| order | heavy | 1495 (296) | 8/8 | 6/8 | 2 | 0 | 0 | 1788 | 292 | +1495 | 0.65 | 0.78 | 1495 | 6574 |
| trace | light | 670 (144) | 8/8 | 0/8 | 8 | 13 | 0 | 963 | 293 | +670 | 0.73 | 0.68 | 670 | 3120 |
| trace | medium | 1362 (287) | 8/8 | 0/8 | 8 | 0 | 0 | 1655 | 293 | +1362 | 0.69 | 0.54 | 1362 | 4500 |
| trace | heavy | 1807 (248) | 8/8 | 0/8 | 8 | 0 | 0 | 2101 | 294 | +1807 | 0.81 | 0.76 | 1807 | 5372 |

Transcript-only-right was 0 in every cell.

### the successor did not recompute

| band | forks | transcript successor reasoning = 0 | native successor reasoning = 0 |
|---|---|---|---|
| light | 24 | 20 | 17 |
| medium | 24 | 21 | 23 |
| heavy | 24 | 23 | 24 |

Of the transcript successor's 46 wrong answers, 43 came with 0
successor reasoning tokens and the other three with 7, 8 and 33. Of its
26 right answers, 21 came with 0; the five that reasoned - 135, 136,
150, 420 and 751 tokens, in light and heavy `arith` and light and medium
`order` - were all among the right ones. The native successor reasoned
at most 28 tokens on any hidden-state fork.

### the replay identity

Over all 92 forks, controls included, the native successor's prompt
tokens minus the transcript successor's prompt tokens minus the
`replay_tokens` handed to the native successor was 0: mean 0, minimum 0,
maximum 0. And the `replay_tokens` handed equalled the shared first
call's reasoning tokens in 92 of 92 forks. The overhead is the reasoning
of the one shared call, charged again as input to its native successor,
exactly.

### order effects

| successor that ran first | forks | native successor right | transcript successor right |
|---|---|---|---|
| native | 36 | 36 | 14 |
| transcript | 36 | 36 | 12 |

Running second did not help or hurt either successor.

### the controls

| control | family | band | forks | shared reasoning before | native right | transcript right | native successor reasoning | transcript successor reasoning | overhead |
|---|---|---|---|---|---|---|---|---|---|
| trivial | trivial | light | 8 | 112 | 8/8 | 8/8 | 26 | 38 | +112 |
| visible state | arith | heavy | 4 | 1273 | 4/4 | 4/4 | 0 | 0 | +1273 |
| visible state | order | heavy | 4 | 1394 | 4/4 | 4/4 | 0 | 0 | +1394 |
| visible state | trace | heavy | 4 | 1930 | 4/4 | 4/4 | 0 | 0 | +1930 |

The trivial task: both successors right every time from the same first
call, native paying 112 tokens of overhead. The visible-state control:
with the answer in the checkpoint call's arguments, the transcript
successor was right 12 of 12 on the same heavy families it lost 16 of 24
of when the answer was hidden, and native paid its full overhead for no
gain. Whatever the native successor bought in the main cells, it bought
by carrying the hidden state.

### tokens and time

| band | transcript total tokens | native total tokens | native minus transcript | successor wall, transcript | successor wall, native |
|---|---|---|---|---|---|
| light | 969 | 1421 | +452 (+47%) | 0.85 s | 0.78 s |
| medium | 1545 | 2459 | +914 (+59%) | 0.74 s | 0.64 s |
| heavy | 2109 | 3534 | +1425 (+68%) | 0.85 s | 0.72 s |

Totals count the shared first call once for each arm. Over the 72
forks the paired successor wall difference, native minus transcript, was
-0.10 s with a standard deviation of 0.81 s; native was the faster
successor in 42 forks.

### per fork

Every fork of the hidden-state cells: the shared first call's reasoning,
then each successor's correctness, reasoning and prompt tokens, and the
`replay_tokens` the native successor was handed.

| family | band | seed | repeat | first arm | shared reasoning | native correct | native reasoning | native prompt | transcript correct | transcript reasoning | transcript prompt | replay_tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| arith | light | 0 | 0 | native | 444 | yes | 0 | 679 | no | 8 | 235 | 444 |
| arith | light | 0 | 1 | transcript | 488 | yes | 0 | 723 | no | 0 | 235 | 488 |
| arith | light | 1 | 0 | transcript | 354 | yes | 0 | 584 | yes | 136 | 230 | 354 |
| arith | light | 1 | 1 | native | 355 | yes | 0 | 585 | yes | 135 | 230 | 355 |
| arith | light | 2 | 0 | native | 202 | yes | 17 | 424 | yes | 0 | 222 | 202 |
| arith | light | 2 | 1 | transcript | 216 | yes | 9 | 438 | yes | 0 | 222 | 216 |
| arith | light | 3 | 0 | transcript | 394 | yes | 0 | 641 | yes | 0 | 247 | 394 |
| arith | light | 3 | 1 | native | 377 | yes | 0 | 624 | yes | 0 | 247 | 377 |
| arith | medium | 0 | 0 | native | 576 | yes | 0 | 856 | no | 7 | 280 | 576 |
| arith | medium | 0 | 1 | transcript | 685 | yes | 7 | 965 | no | 0 | 280 | 685 |
| arith | medium | 1 | 0 | transcript | 503 | yes | 0 | 780 | no | 0 | 277 | 503 |
| arith | medium | 1 | 1 | native | 403 | yes | 0 | 680 | yes | 0 | 277 | 403 |
| arith | medium | 2 | 0 | native | 430 | yes | 0 | 714 | no | 0 | 284 | 430 |
| arith | medium | 2 | 1 | transcript | 346 | yes | 0 | 630 | no | 0 | 284 | 346 |
| arith | medium | 3 | 0 | transcript | 695 | yes | 0 | 993 | no | 0 | 298 | 695 |
| arith | medium | 3 | 1 | native | 433 | yes | 0 | 731 | no | 0 | 298 | 433 |
| arith | heavy | 0 | 0 | native | 900 | yes | 0 | 1225 | no | 0 | 325 | 900 |
| arith | heavy | 0 | 1 | transcript | 848 | yes | 0 | 1173 | no | 0 | 325 | 848 |
| arith | heavy | 1 | 0 | transcript | 951 | yes | 0 | 1317 | no | 0 | 366 | 951 |
| arith | heavy | 1 | 1 | native | 938 | yes | 0 | 1304 | no | 0 | 366 | 938 |
| arith | heavy | 2 | 0 | native | 1719 | yes | 0 | 2095 | no | 0 | 376 | 1719 |
| arith | heavy | 2 | 1 | transcript | 1233 | yes | 0 | 1609 | no | 0 | 376 | 1233 |
| arith | heavy | 3 | 0 | transcript | 936 | yes | 0 | 1278 | yes | 0 | 342 | 936 |
| arith | heavy | 3 | 1 | native | 1020 | yes | 0 | 1362 | yes | 751 | 342 | 1020 |
| order | light | 0 | 0 | native | 350 | yes | 0 | 574 | yes | 0 | 224 | 350 |
| order | light | 0 | 1 | transcript | 395 | yes | 0 | 619 | yes | 0 | 224 | 395 |
| order | light | 1 | 0 | transcript | 183 | yes | 0 | 392 | yes | 0 | 209 | 183 |
| order | light | 1 | 1 | native | 241 | yes | 0 | 450 | yes | 0 | 209 | 241 |
| order | light | 2 | 0 | native | 426 | yes | 0 | 637 | yes | 150 | 211 | 426 |
| order | light | 2 | 1 | transcript | 361 | yes | 22 | 572 | yes | 0 | 211 | 361 |
| order | light | 3 | 0 | transcript | 501 | yes | 0 | 712 | yes | 0 | 211 | 501 |
| order | light | 3 | 1 | native | 514 | yes | 0 | 725 | yes | 0 | 211 | 514 |
| order | medium | 0 | 0 | native | 840 | yes | 0 | 1092 | yes | 0 | 252 | 840 |
| order | medium | 0 | 1 | transcript | 1106 | yes | 0 | 1358 | yes | 0 | 252 | 1106 |
| order | medium | 1 | 0 | transcript | 916 | yes | 0 | 1174 | no | 33 | 258 | 916 |
| order | medium | 1 | 1 | native | 911 | yes | 0 | 1169 | yes | 420 | 258 | 911 |
| order | medium | 2 | 0 | native | 582 | yes | 0 | 816 | no | 0 | 234 | 582 |
| order | medium | 2 | 1 | transcript | 761 | yes | 0 | 995 | no | 0 | 234 | 761 |
| order | medium | 3 | 0 | transcript | 1595 | yes | 0 | 1850 | no | 0 | 255 | 1595 |
| order | medium | 3 | 1 | native | 1286 | yes | 0 | 1541 | no | 0 | 255 | 1286 |
| order | heavy | 0 | 0 | native | 1663 | yes | 0 | 1960 | yes | 0 | 297 | 1663 |
| order | heavy | 0 | 1 | transcript | 1024 | yes | 0 | 1321 | yes | 0 | 297 | 1024 |
| order | heavy | 1 | 0 | transcript | 1703 | yes | 0 | 1998 | no | 0 | 295 | 1703 |
| order | heavy | 1 | 1 | native | 1804 | yes | 0 | 2099 | no | 0 | 295 | 1804 |
| order | heavy | 2 | 0 | native | 1298 | yes | 0 | 1590 | yes | 0 | 292 | 1298 |
| order | heavy | 2 | 1 | transcript | 1168 | yes | 0 | 1460 | yes | 0 | 292 | 1168 |
| order | heavy | 3 | 0 | transcript | 1892 | yes | 0 | 2178 | yes | 0 | 286 | 1892 |
| order | heavy | 3 | 1 | native | 1409 | yes | 0 | 1695 | yes | 0 | 286 | 1409 |
| trace | light | 0 | 0 | native | 772 | yes | 0 | 1066 | no | 0 | 294 | 772 |
| trace | light | 0 | 1 | transcript | 671 | yes | 27 | 965 | no | 0 | 294 | 671 |
| trace | light | 1 | 0 | transcript | 512 | yes | 0 | 806 | no | 0 | 294 | 512 |
| trace | light | 1 | 1 | native | 794 | yes | 0 | 1088 | no | 0 | 294 | 794 |
| trace | light | 2 | 0 | native | 795 | yes | 28 | 1089 | no | 0 | 294 | 795 |
| trace | light | 2 | 1 | transcript | 816 | yes | 28 | 1110 | no | 0 | 294 | 816 |
| trace | light | 3 | 0 | transcript | 403 | yes | 0 | 694 | no | 0 | 291 | 403 |
| trace | light | 3 | 1 | native | 593 | yes | 18 | 884 | no | 0 | 291 | 593 |
| trace | medium | 0 | 0 | native | 1358 | yes | 0 | 1652 | no | 0 | 294 | 1358 |
| trace | medium | 0 | 1 | transcript | 1475 | yes | 0 | 1769 | no | 0 | 294 | 1475 |
| trace | medium | 1 | 0 | transcript | 1668 | yes | 0 | 1960 | no | 0 | 292 | 1668 |
| trace | medium | 1 | 1 | native | 1553 | yes | 0 | 1845 | no | 0 | 292 | 1553 |
| trace | medium | 2 | 0 | native | 1592 | yes | 0 | 1886 | no | 0 | 294 | 1592 |
| trace | medium | 2 | 1 | transcript | 706 | yes | 0 | 1000 | no | 0 | 294 | 706 |
| trace | medium | 3 | 0 | transcript | 1366 | yes | 0 | 1659 | no | 0 | 293 | 1366 |
| trace | medium | 3 | 1 | native | 1179 | yes | 0 | 1472 | no | 0 | 293 | 1179 |
| trace | heavy | 0 | 0 | native | 1741 | yes | 0 | 2035 | no | 0 | 294 | 1741 |
| trace | heavy | 0 | 1 | transcript | 1664 | yes | 0 | 1958 | no | 0 | 294 | 1664 |
| trace | heavy | 1 | 0 | transcript | 2113 | yes | 0 | 2409 | no | 0 | 296 | 2113 |
| trace | heavy | 1 | 1 | native | 1523 | yes | 0 | 1819 | no | 0 | 296 | 1523 |
| trace | heavy | 2 | 0 | native | 2197 | yes | 0 | 2491 | no | 0 | 294 | 2197 |
| trace | heavy | 2 | 1 | transcript | 1839 | yes | 0 | 2133 | no | 0 | 294 | 1839 |
| trace | heavy | 3 | 0 | transcript | 1453 | yes | 0 | 1745 | no | 0 | 292 | 1453 |
| trace | heavy | 3 | 1 | native | 1923 | yes | 0 | 2215 | no | 0 | 292 | 1923 |

## results: the independent-first-call experiment (exploratory)

Run before the fork existed, on the same tasks and matrix, each arm
making its own first call; 184 runs, 368 calls, every run honouring the
protocol. The first call being sampled twice per pair, the arms' first
calls differed by a few hundred reasoning tokens run to run, which is
the ambiguity the forked experiment removes. Kept here because it is
consistent with the fork on every count.

| band | arm | n | correct |
|---|---|---|---|
| light | transcript | 24 | 13 (54%) |
| light | native | 24 | 24 (100%) |
| medium | transcript | 24 | 2 (8%) |
| medium | native | 24 | 24 (100%) |
| heavy | transcript | 24 | 5 (21%) |
| heavy | native | 24 | 24 (100%) |

Paired by task, seed and repeat: native only right 52 of 72, transcript
only right 0, both wrong 0. All 52 wrong transcript answers came with 0
successor reasoning. Native successor prompt overhead equalled
`replay_tokens` on all 92 pairs. Trivial control: both arms 8 of 8,
overhead +99. Visible-state control: both arms 12 of 12 on the heavy
band, overhead +1286 to +1886. Per cell:

| family | band | arm | correct | reasoning before | reasoning after | prompt after | total tokens | replay_tokens |
|---|---|---|---|---|---|---|---|---|
| arith | light | transcript | 6/8 | 322 | 74 | 234 | 854 | 0 |
| arith | light | native | 8/8 | 352 | 14 | 586 | 1176 | 352 |
| arith | medium | transcript | 0/8 | 500 | 0 | 285 | 1060 | 0 |
| arith | medium | native | 8/8 | 486 | 0 | 771 | 1532 | 486 |
| arith | heavy | transcript | 0/8 | 1114 | 0 | 352 | 1808 | 0 |
| arith | heavy | native | 8/8 | 1084 | 2 | 1436 | 2864 | 1084 |
| order | light | transcript | 7/8 | 348 | 36 | 214 | 807 | 0 |
| order | light | native | 8/8 | 357 | 8 | 571 | 1146 | 357 |
| order | medium | transcript | 2/8 | 1021 | 0 | 250 | 1563 | 0 |
| order | medium | native | 8/8 | 1134 | 0 | 1384 | 2766 | 1134 |
| order | heavy | transcript | 5/8 | 1471 | 0 | 292 | 2059 | 0 |
| order | heavy | native | 8/8 | 1526 | 0 | 1818 | 3639 | 1526 |
| trace | light | transcript | 0/8 | 655 | 0 | 293 | 1232 | 0 |
| trace | light | native | 8/8 | 696 | 2 | 989 | 1970 | 696 |
| trace | medium | transcript | 0/8 | 1392 | 0 | 293 | 1969 | 0 |
| trace | medium | native | 8/8 | 1346 | 0 | 1639 | 3268 | 1346 |
| trace | heavy | transcript | 0/8 | 1859 | 0 | 294 | 2437 | 0 |
| trace | heavy | native | 8/8 | 1914 | 0 | 2208 | 4408 | 1914 |

## measured, inferred, not established

**Measured**

- Forked, from one shared first call: the native successor right 72 of
  72; the transcript successor 26 of 72, 4 of 24 at medium and 8 of 24
  at heavy. Native only right in 46 forks, transcript only right in none,
  both wrong in none.
- The transcript successor did not recompute: 0 reasoning tokens on 43
  of its 46 wrong answers, at most 33 on the rest.
- The native successor's prompt overhead equals the `replay_tokens`
  handed to it, and that equals the shared first call's reasoning, in
  every fork: 465, 957 and 1457 prompt tokens by band; per task, 452,
  914 and 1425 more total tokens, +47%, +59% and +68%.
- Successor wall time did not separate the arms: under a second on
  average in both, the paired difference 0.10 s in native's favour with
  a standard deviation of 0.81 s. Which successor ran first did not
  matter.
- With nothing hidden (trivial) or with the state made visible (the
  answer in the call's arguments), the successors did not differ in
  correctness, and native paid its overhead for nothing.
- Every first call honoured the protocol in both experiments; the
  independent experiment gave the same shape on every count.

**Inferred**

- Two successors of the same provider reasoning state diverge solely
  because one is sent the provider's continuation and the other a
  reconstruction. On this model, a call rebuilt with the placeholder
  signature after real reasoning is a conversation in which the
  reasoning never happened, and the model does not treat it as one to
  redo; it answers as though its state were there. That is a fidelity
  failure at exactly the boundary the agent loop crosses on every tool
  round.
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
  reasoning on each call, and no fork here had more than one boundary.
- Latency at scale; the successor calls here were under a second in
  both arms.
- The service path around the adapter - budgeting, offers and citations
  - which the harness deliberately leaves out.

## limitations

- One model, one provider, one thinking setting.
- Eight forks per cell in the main design, four in the visible-state
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
    --backend gemini --model gemini-3.8-flash --design forked \
    --families order,arith,trace --bands light,medium,heavy \
    --seeds 4 --repeat 2 --out fidelity.jsonl
GEMINI_PROBE_API_KEY=... python scripts/reasoning_fidelity.py \
    --backend gemini --model gemini-3.8-flash --design forked \
    --families trivial --bands light --seeds 4 --repeat 2 --out fidelity.jsonl
GEMINI_PROBE_API_KEY=... python scripts/reasoning_fidelity.py \
    --backend gemini --model gemini-3.8-flash --design forked \
    --families order,arith,trace --bands heavy --seeds 4 --repeat 1 --visible \
    --out fidelity.jsonl
python scripts/reasoning_fidelity.py --summarize fidelity.jsonl
```

`--design independent` reproduces the exploratory experiment;
`--print-tasks` prints every generated task with its answer without
calling any provider. The raw runs behind this report are
`docs/measurements/pn-m2-gemini-forked-checkpoint.jsonl`,
`docs/measurements/pn-m2-gemini-forked-trivial-control.jsonl`,
`docs/measurements/pn-m2-gemini-forked-visible-control.jsonl`, and the
`pn-m2-gemini-checkpoint`, `-trivial-control` and `-visible-control`
files of the independent experiment.
