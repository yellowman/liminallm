# JAX LoRA backend notes

The current JAX pathway is intentionally skeletal so the kernel can run without GPU bindings or external dependencies. The pieces to strengthen are:

## Gaps in the placeholder backend
- **No inference:** `LocalJaxLoRABackend` only echoes routing metadata and does not load a model or tokenize inputs.
- **Adapter loading stub:** Adapter paths resolve to the filesystem, but LoRA weights are never read or merged into a model.
- **Safety/shape checks:** There is no enforcement of maximum batch sizes, sequence lengths, or device placement.
- **Tokenizer mismatch risk:** Calls rely on pre-tokenized inputs; a tokenizer tied to the base model is required for deterministic outputs.

## Turning it into a real backend
1. **Model and tokenizer loading**
   - Choose a Flax/Transformers checkpoint compatible with your adapters and load it once per process (pin to GPU/TPU).
   - Initialize the matching tokenizer and expose it to the request path for prompt/adapter shape validation.
2. **Adapter materialization**
   - Read LoRA deltas from `fs_root` (``users/<user_id>/adapters/<adapter_id>``) and lift them into JAX device arrays.
   - Decide whether to *merge* deltas into the base weights at load time (faster inference, more memory) or apply them on the fly (slower, less memory).
3. **Forward pass and decoding**
   - Implement an architecture-specific forward function (SPEC §5) that consumes tokenized prompts and adapter deltas.
   - Stream decoded tokens through the API layer for parity with remote backends and include per-token usage accounting.
4. **Batching and safety**
   - Normalize all requests to fixed shapes (padding/truncation) before dispatch; reject oversized batches.
   - Enforce device placement and a consistent PRNG key strategy to keep runs deterministic across adapter reloads.
5. **Checkpointing and hot-reload**
   - Persist adapter deltas after training (see `_run_jax_optax_training`) in a format that can be memory-mapped or lazily loaded.
   - Add a small cache with LRU eviction keyed by adapter_id so frequently used adapters stay warm without leaking memory.
6. **Observability and limits**
   - Emit timing, peak memory, and token throughput metrics per request.
   - Wire circuit breakers for latency and OOMs so the server degrades gracefully under load.

## Training loop hardening
- Replace the toy loss in `_run_jax_optax_training` with the model's supervised loss (or DPO/RLHF objectives) and integrate gradient accumulation.
- Add mixed-precision support (FP16/bfloat16) and gradient clipping to avoid instabilities.
- Save optimizer and adapter states periodically so jobs can resume after failures.

With these pieces in place, the "local_gpu_lora" adapter plug can serve parity with remote adapter servers while keeping the kernel minimal and filesystem-driven.
