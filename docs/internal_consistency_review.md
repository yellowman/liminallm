# Internal consistency review

This audit highlights areas where service behaviors appear to diverge from each other or from the data-driven kernel goals outlined in SPEC.

## Status

All previously identified inconsistencies have been remediated per SPEC.

- **Adapter training vs runtime base model (SPEC §5.1, §6.1):** Runtime now resolves the active base model from deployment config and passes it to training; adapter creation and loading persist and enforce that `base_model` across artifacts, versions, and router state. Incompatible bases raise structured migration guidance instead of silently composing weights.

- **Routing determinism for typed/clustered adapters (SPEC §8.1):** Adapter type and cluster activations rank candidates by cosine similarity (with deterministic tie-breakers), normalize gates under the existing guardrails, and record ranked traces for policy audits.

- **ConfigOps preference awareness (SPEC §2.6, §10, §15):** Preference summaries now distinguish no-data from backend errors, include adapter/router state context, and ConfigOps prompts surface those statuses so LLM proposals remain data-driven and observable.

No additional inconsistencies are open as of this review.
