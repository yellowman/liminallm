# JAX LoRA backend

`LocalJaxLoRABackend` now runs a functional local adapter path instead of the earlier placeholder. This page summarizes how the implementation works and the operational expectations for using it.

## Backend capabilities

- **Tokenizer-aware prompt handling:** Messages are normalized into a `role: content` string and tokenized with the base model's tokenizer when available (via `transformers.AutoTokenizer`). If the tokenizer dependency is missing, a deterministic hash-based fallback keeps the pathway usable for testing.
- **Fixed shapes with light checks:** Requests are padded/truncated to `max_seq_len` (default 512). The backend always builds a single-element batch and only validates that `max_batch_size` is positive; callers that pass multiple examples will be trimmed to one without enforcement beyond length checks.
- **Adapter materialization with caching:** LoRA weights are loaded from `fs_root/adapters/<adapter_id>/` (or an explicit `fs_dir`/`cephfs_dir`) by reading `params.json`. The backend caches weights keyed by adapter ID and file `mtime` so hot adapters avoid repeated disk reads.
- **Device placement and JAX execution:** Token/attention arrays are placed on the first available JAX device, and adapter matrices are lifted to JAX arrays. A lightweight forward pass applies paired `.A`/`.B` matrices with width alignment before sampling.
- **Deterministic sampling and decoding:** Generated tokens are sampled deterministically from the LoRA score aggregate (seeded by the last prompt token) and decoded with the tokenizer when present; otherwise a `tok-<id>` fallback is used. Usage metrics include prompt/completion token counts, latency, model ID, and adapter ID.

## Adapter resolution

- Default path: `fs_root/adapters/<adapter_id>/latest/params.json` if present; otherwise the newest `v*/params.json` directory is selected.
- Explicit paths: callers may supply `fs_dir` or `cephfs_dir` in the adapter metadata to override the default layout.
- Cache: weight arrays are cached per adapter ID alongside the `params.json` modification time to avoid stale loads after updates.

## Request flow

1. Normalize chat messages and tokenize to `(input_ids, attention_mask)` with truncation to `max_seq_len`.
2. Pad to non-empty tensors and move them to the active JAX device with dtype `int32`; a single-item batch is created regardless of caller intent.
3. Load and cache adapter weights; if none are found, generation falls back to zeroed scores.
4. Run the LoRA forward pass (`_lora_forward`) to accumulate adapter contributions, mask with attention, and sample a fixed-length completion using a rolling deterministic sampler.
5. Decode tokens and return the text plus usage metadata (prompt/completion token counts, latency, model, adapter ID).

## Operational notes

- **Dependencies:** JAX and (optionally) `transformers` must be available for full fidelity; the backend degrades gracefully without the tokenizer by using hashed tokens and naive decoding. If JAX is missing, training falls back to a skip trace while generation uses array stubs.
- **Limits:** `max_seq_len` defaults to 512 and batch size to 4, but the generation path enforces a single example and only rejects prompts that exceed the length ceiling. Multi-example requests will silently drop down to one.
- **Determinism:** The sampler is deterministic given the prompt and adapter weights; a process-level PRNG key is initialized but not mutated per call.
- **Observability:** Per-call latency is reported in milliseconds. Additional metrics (throughput, memory) can be layered on top of this scaffolding if needed.

## Training loop

The `TrainingService` now performs a usable JAX+Optax fine-tuning cycle for LoRA adapters:

- **Supervised loss:** Preference-derived prompts/targets are tokenized and fed through the same lightweight LoRA projection used at inference time. A masked cross-entropy loss over a fixed vocab drives updates when JAX/Optax are available; otherwise the training job records a skipped trace.
- **Gradient accumulation:** Microbatches accumulate gradients across configurable steps before each optimizer update, allowing larger effective batch sizes without exceeding device memory.
- **Checkpoints and persistence:** Each optimizer step writes a checkpoint under the adapter version's `checkpoints/` directory and rewrites `params.json` with the trained weights so `local_gpu_lora` reloads the latest adapter artifacts immediately. A `latest` symlink is refreshed after every successful run.

With these pieces, `local_gpu_lora` mirrors the repository's data-driven kernel expectations while remaining lightweight and test-friendly.
