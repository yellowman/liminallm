# provider capability registry

The current provider inventory. `SPEC.md` §5.0.2 defines the capability
contract every backend declares; this file records what each provider
declares today. These facts change faster than the architecture, so they
live here and in the implementation registry, not in the SPEC.

## remote styles

| Style | Description | Example providers |
|-------|-------------|-------------------|
| `model_id` | Fine-tuned model as endpoint; one adapter per request | OpenAI, Azure, Vertex, Bedrock |
| `adapter_param` | Adapter ID in request body; multi-adapter supported | Together, LoRAX, adapter_server |
| `none` | No remote adapter support; local/prompt only | local_lora, local_gpu_lora |

## provider capability matrix

| Provider | Remote style | Multi-adapter | Gate weights | Max adapters |
|----------|-------------|---------------|--------------|--------------|
| `openai` | model_id | no | no | 1 |
| `anthropic` | model_id | no | no | 1 |
| `azure` | model_id | no | no | 1 |
| `vertex` | model_id | no | no | 1 |
| `bedrock` | model_id | no | no | 1 |
| `zhipu` | model_id | no | no | 1 |
| `together` | adapter_param | yes | yes | 3 |
| `lorax` | adapter_param | yes | yes | 5 |
| `adapter_server` | adapter_param | yes | yes | 3 |
| `sagemaker` | adapter_param | no | no | 1 |
| `local_lora` | none | yes | yes | 3 |
| `stub` | none | no | no | 0 |

The `stub` backend returns deterministic canned responses without calling
any LLM. It exists for tests and CI pipelines where real inference is not
required.

## adapter schema fields by provider type

For `model_id` providers (OpenAI, Azure, and similar):

```json
{
  "mode": "remote",
  "remote_model_id": "ft:gpt-4o-mini-2024-07-18:org:custom:abc123"
}
```

For `adapter_param` providers (Together, LoRAX, and similar):

```json
{
  "mode": "remote",
  "remote_adapter_id": "user-123/my-lora-adapter",
  "weight": 0.8
}
```

## request formatting

- `model_id` style: the adapter's `remote_model_id` becomes the `model`
  parameter.
- `adapter_param` style: adapter IDs pass as `extra_body.adapter_id` (or the
  provider-specific parameter).
- When the selected adapters exceed the provider's `max_adapters`, the
  lowest-weight adapters are dropped and the drop is logged.

## hybrid mode with a remote fallback

A hybrid adapter can carry both `prompt_instructions` (for prompt injection)
and `remote_model_id` / `remote_adapter_id` (for API passthrough):

```json
{
  "mode": "hybrid",
  "prompt_instructions": "You are a coding assistant...",
  "remote_adapter_id": "user-123/code-lora",
  "weight": 0.9
}
```

On an API backend:

1. Prompt instructions are injected into the system message once, by the
   service (SPEC §5.0.1 — one materializer).
2. If the adapter has a remote ID and the provider supports it, the ID is
   also passed to the API.
3. With no remote ID, or no provider support, only the prompt injection
   carries the adapter.

## setting catalogs

The admin console's model settings are validated enums whose members are
declared in `liminallm/config.py`; that declaration is normative. Current
suggestion lists, recorded here because they date quickly:

- `model_path` common suggestions: gpt-4o, gpt-4o-mini, gpt-5.2,
  claude-opus-4-5, claude-sonnet-4, glm-4-plus.
- `model_backend`: openai, anthropic, azure, azure_openai, vertex, gemini,
  google, bedrock, together, together.ai, lorax, adapter_server, sagemaker,
  aws_sagemaker, zhipu, zhipu.ai, glm, stub — plus the local pair
  local_lora / local_gpu_lora and the native `gemini_native`.
- `embedding_model_id`: text-embedding, text-embedding-3-small,
  text-embedding-3-large, text-embedding-ada-002.
- voice: `voice_transcription_model` (whisper-1), `voice_synthesis_model`
  (tts-1, tts-1-hd), `voice_default_voice` (alloy, echo, fable, onyx, nova,
  shimmer).
- `rag_mode`: pgvector, memory, local_hybrid.
