# why retrieval runs more than one channel

SPEC §2.5 requires up to three candidate channels (dense, late-interaction,
lexical) fused by rank. This file carries the evidence behind that rule.

## the dimension bound

A single vector of dimension `d` bounds how many distinct top-k sets of
documents any query can ever return. The bound is geometric, not
statistical, so no amount of training data or model size removes it: for
`n` documents, `k` relevant, and a score margin `γ`, realizing every
`k`-subset needs `d ≥ log C(n,k) / log(1 + 1/γ)` (Weller, Boratko, Naim &
Lee, ICLR 2026, *On the Theoretical Limitations of Embedding-Based
Retrieval*).

That is a floor, and a loose one. Optimizing the vectors directly against
the test set — no language model, no generalization, the best case that can
exist — the same work measures a *critical-n* per dimension: 10 documents at
`d=4`, 99 at `d=18`, extrapolating to ~500k at 512, ~4m at 1024, ~250m at
4096. Real encoders land far below their own floor: the 46-document probe
below is solvable in 12 free dimensions, and real models at 64 dimensions
still cannot solve it.

## what that looks like in practice

The paper's LIMIT probe is deliberately trivial — documents like "Jon likes
quokkas and apples", queries like "who likes quokkas?" — and it breaks
state-of-the-art embedders:

| | recall@2, 46 docs | recall@2, 50k docs |
|---|---|---|
| BM25 | 97.8 | 85.7 |
| GTE-ModernColBERT (multi-vector) | 83.5 | 23.1 |
| Promptriever 8B @4096 (best single-vector) | 54.3 | 3.0 |
| Qwen3 Embed @4096 | 19.0 | 0.8 |

Two results matter more than the ranking:

- **It is not domain shift.** Fine-tuning on in-domain training data moves
  recall@10 from ~0 to 2.8, while training on the test set solves it — the
  task is representationally hard, not unfamiliar.
- **Lexical is not the answer either.** Rewriting the same corpus with
  synonyms drops BM25 by ~89% (97.8 → 10.6) while the dense models hold,
  leaving BM25 *below* most of them. The two channels fail on disjoint
  inputs. Neither is safe alone, which is the entire argument for running
  both.

A third result is about evaluation, not retrieval: LIMIT scores do not
correlate with BEIR. An encoder's benchmark position predicts nothing about
this failure, so "we use a good embedding model" is not a mitigation.

## why fusion is by rank, never by score

- Cosine is bounded and BM25 is not, and BM25's magnitude depends on the
  pool it was scored against — so any weighted sum needs a normalizer, and
  every normalizer moves with the pool. The same chunk would score
  differently depending on what it was ranked beside.
- Rank fusion expresses something a weighted sum cannot: a chunk **both**
  channels rank well beats one that only a single channel loves. Under a
  fixed-weight sum, a perfect cosine always beats a perfect BM25 and the
  lexical channel can never win a head-to-head — which the table above says
  is exactly backwards.
- A channel ranks only what it matched. Zero is silence, not a weak opinion:
  an arbitrary order over non-matches would otherwise carry the channel's
  full weight. This is also why an un-embedded turn in conversation recall
  is *absent* from the semantic channel rather than scored zero by it — the
  weighted-sum predecessor had to hold the zero against it.

The weighted sums that preceded rank fusion disagreed with the rule and
with each other: notes search weighted *lexical* 0.6 against the stated
precedence, and recall scored un-embedded turns as literal zeros.

## why the reranker exists, and why `auto` distrusts small models

In the paper's own test, a long-context reranker solved all 1000 of the
46-document queries where the best embedder stayed under 60 — it is the only
stage that escapes both ceilings and the only one that can answer "none of
these". But that result is a frontier long-context model; this project's
premise is small self-hosted models, the case the paper never tested, and
the stage can *drop* context. So it is conditional (`rag_rerank = auto`),
and `auto` asks for positive evidence of capability (curated family
prefixes, declared parameter count ≥30B), with unknown as a no — a model
given the benefit of the doubt here can silently drop a user's grounding.

The capability test matches small-variant names (`mini`, `nano`, `lite`) as
whole name parts, never substrings — `mini` lives inside `gemini` — and a
size the name declares beats family membership in both directions, so
`gemini-2.0-flash-8b` is refused on its stated size rather than admitted on
its prefix.

## where this system is most exposed

The paper's difficulty metric is qrel graph density — how often one document
is relevant to many queries, and how much queries share documents. LIMIT
scores 0.085 density / 28.5 average query strength against ≤0.026 / ≤0.6
for NQ, HotpotQA, SciFact and FollowIR. The parts of this system at the
hard end of that scale are the notes vault and the witness (SPEC §19),
where the task *is* relating documents to each other and a hub note is
relevant to many questions — not chat RAG over a handful of uploaded files,
which sits at the easy end.

## late interaction, measured caveats

Multi-vector retrieval is the one architecture in the paper that beats
single-vector on both splits (83.5 vs 54.3 recall@2 at 46 documents, 23.1
vs 3.0 at 50k), and the only entry that attacks the bound itself: MaxSim is
not one inner product, so the theory's premise does not apply (the paper
says as much in its limitations).

The implementation is not ColBERT and must not be read as carrying its
numbers: segments are sentence-sized, embedded by the same encoder as
everything else, because an OpenAI-compatible `/embeddings` endpoint
returns one vector per input. What carries over is the mechanism, not the
granularity, at roughly an order of magnitude less storage than per-token.
The seam is the encoder: a real late-interaction model replaces
`segment_text` and the embed call without touching storage, candidate
generation, or scoring.

Two measured implementation notes that shaped the SPEC's rules:

- MaxSim normalizes once per vector and compares by dot product. A general
  cosine per pair re-derived both norms, copied both vectors and rescanned
  them for NaN on every (query × segment × candidate) comparison — 0.44s
  per retrieval at the shipped defaults and 2s at the candidate cap, on a
  request thread, before the answering model was called. Same arithmetic,
  6× less of it.
- The lexical channel's `ts_rank` re-tokenizes every matching row when the
  tsvector is computed per query; the stored generated column exists
  because that was the dominant cost of the channel. Measured on 50k
  chunks: 28.7 ms/query with the GIN index, 239.7 ms without.
