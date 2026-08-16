# JAX LoRA backend

`LocalJaxLoRABackend` serves a real decoder with LoRA adapters applied inside it. This page summarizes how the implementation works and the operational expectations for using it. SPEC §5 is the authority; where this page and SPEC disagree, SPEC is right and this page is stale.

## Backend capabilities

- **A real decoder:** `service/transformer.py` implements a decoder-only transformer in plain JAX — RMSNorm, RoPE, grouped-query attention with a KV cache, SwiGLU — loading `config.json` plus `*.safetensors` from the model directory with no torch and no flax. LoRA matrices are applied inside the attention projections, so an adapter changes the model rather than a score aggregate bolted on afterwards.
- **Three checkpoint states:** `absent` (nothing on disk) falls back to a synthetic stand-in and logs `local_checkpoint_absent` — that path moves tokens, it does not answer questions. `valid` serves the real model. `broken` (weights or tokenizer present but unloadable) **fails every request**, rather than degrading into the stand-in.
- **Tokenizer-aware prompt handling:** Messages are serialized by `service/local_format.py`, the one function training and serving share, and tokenized with the base model's tokenizer. A deterministic hash fallback keeps the path usable without `transformers`; a checkpoint whose tokenizer disagrees with its vocabulary is `broken`, not fallback-eligible.
- **Adapter materialization with caching:** Weights are read from the promoted version only (see below) and cached by `(adapter_id, version)` alongside the file `mtime`, so a promotion cannot be served its predecessor's tensors and an in-place edit still invalidates.
- **Gate-aware KV reuse:** Conversations reuse their own prefix across turns. The cache key is content-addressed and includes each active adapter's id, version and exact gate bits, so a request at gate 0.2 can never continue a prefix computed at 0.8. Reused prefill is reported as `cached_tokens`.
- **Usage:** prompt/completion token counts from the real tokenizer, latency, model ID, and the adapters that actually applied.

## Adapter resolution

Per SPEC §5.5, the artifact's `current_version` is the sole authority:

- `current_version = N > 0` resolves to exactly `<adapter root>/vNNNN/params.json`. Nothing else is authoritative — not a direct `params.json`, not a `latest` pointer, not the newest directory. `latest` is still written for humans and tooling and is never read back.
- `current_version <= 0`, or absent, resolves to **no weights**. A training job writes its version directory *before* the eval gate runs, so anything that scanned would serve weights the gate rejected.
- **Explicit paths relocate, they do not rename.** `fs_dir`/`cephfs_dir` may point an adapter's directory elsewhere under `fs_root`, but its final component must be the adapter's own id — otherwise `fs_dir: adapters/B` serves B's weights as A's version. The version's `metadata.json`, which training writes with `adapter_id` and `version`, is checked too when present, because a directory can be renamed and a training run's own record cannot.
- **One base, exactly.** An adapter's declared `base_model` must be the base being served, compared on the final path component; a missing or different declaration refuses the weights. `B·A` was fitted to one frozen `W`, and an eval gate passed on that `W` says nothing about another.
- **Prompt-rung adapters carry no weights** whatever files exist, and a zero-gated adapter is skipped before any of the above runs.

## Request flow

1. Reduce the adapter list to the effective set: gate first, so `g = 0` is gone before anything reads it (SPEC §5.0.1).
2. Serialize the messages with `local_format` and tokenize, keeping the **newest** tokens when truncating — a chat's last turn is the one the answer responds to.
3. Resolve and load the promoted weights for each active adapter, then compose them by rank concatenation: `A* = [A₁;A₂]`, `B* = [g₁α₁B₁, g₂α₂B₂]`, so `B*A* = Σ gⱼαⱼBⱼAⱼ` exactly, with no cross terms. A malformed adapter refuses the whole stack rather than partially applying.
4. Reuse the longest cached KV prefix that is a strict prefix of these tokens under the same adapter signature, then decode incrementally.
5. Decode and return the text plus usage — including `cached_tokens` for the reused prefill.

Prompt instructions are **not** materialized here. `LLMService` places them before any backend runs, so this class applies weights and nothing else (SPEC §5.0.1).

## Operational notes

- **Dependencies:** JAX must be available; `transformers` supplies the tokenizer. Without a checkpoint the backend serves the synthetic stand-in and says so in the log; with a checkpoint whose tokenizer will not load it refuses, because a request answered from the stand-in *after* one was refused is the opposite of refusing. If JAX is missing, training records a skip trace.
- **Limits:** `max_seq_len` defaults to 512 and batch size to 4; the generation path builds a single example. Truncation keeps the newest tokens, unlike a tokenizer's own `truncation`, which keeps the oldest.
- **Determinism:** Greedy decoding is deterministic given the prompt, the adapter stack and its gates. Incremental decode with the KV cache is tested to reproduce a full recompute, and a warm prefix cache to produce byte-identical output to a cold one.
- **Observability:** Per-call latency is reported in milliseconds. Additional metrics (throughput, memory) can be layered on top of this scaffolding if needed.

## Training loop

The `TrainingService` now performs a usable JAX+Optax fine-tuning cycle for LoRA adapters:

- **Supervised loss over the real forward pass:** Preference-derived prompts and targets are tokenized and run through the same decoder that serves them, with the LoRA matrices inside its attention projections, so an adapter is fitted to the model that will serve it. The base parameters are closed over and never differentiated. Loss is cross-entropy on the target tokens only; an L2 term on the adapter weights is in the training objective and deliberately not in the eval.
- **It refuses rather than pretends:** no checkpoint, no tokenizer, a tokenizer that disagrees with the checkpoint's vocabulary, or token ids outside `[0, vocab)` all record a skipped trace instead of training against something that is not the model.
- **Gradient accumulation:** Microbatches accumulate gradients across configurable steps before each optimizer update.
- **Eval gate:** Every fifth example is held out; the job promotes only when holdout cross-entropy improves by at least 1% relative. A skipped or regressed run never promotes, and the decision is recorded in `training_job.meta.eval_gate` for audit — its absence is not approval.
- **Checkpoints and persistence:** Each optimizer step writes a checkpoint under the version's `checkpoints/` directory, and a passing gate bumps `current_version`, which is what makes the weights servable. The `latest` symlink is refreshed as a convenience and is best-effort: serving never reads it, so failing to write it logs and leaves the promotion standing rather than aborting a run that already succeeded.

With these pieces, `local_gpu_lora` mirrors the repository's data-driven kernel expectations while remaining lightweight and test-friendly.

## hardware targets (pjrt)

The backend places arrays on "the first available JAX device" and otherwise
never names hardware — deliberately. JAX reaches accelerators through PJRT
plugins, so CPU, CUDA, TPU, and any third-party PJRT plugin all look the same
from this code.

That includes the in-house **loom FPGA**: once its PJRT plugin lands (expected
soon), pointing JAX at it makes `local_gpu_lora` inference and LoRA training
run on loom with no changes here. Two follow-ups become interesting at that
point:

- a multimodal model hosted on loom can implement `transcribe_image` on the
  backend, which plugs it straight into the upload/scanned-pdf extraction
  ladder (see `service/extract.py` — readers are a registry, the `extract_readers`
  admin setting orders them);
- a small dedicated OCR model on loom can register as an extraction reader
  outright, sitting between tesseract and full model vision in cost.

Until the plugin exists, none of this is wired: the local backend remains a
text LM + tokenizer and image reading is tesseract and/or API-backend vision.
