"""Shared utilities for prompt extraction from adapters.

Per SPEC §5.0.1, adapters can contribute prompt instructions to LLM context.
This module provides a consistent implementation used across llm.py and model_backend.py.
"""

from __future__ import annotations

from typing import Optional

from liminallm.logging import get_logger

logger = get_logger(__name__)


def extract_prompt_instructions(adapter: dict, *, log_source: str = "adapter") -> Optional[str]:
    """Extract prompt instructions from an adapter using consistent priority order.

    This is the canonical implementation for extracting behavioral prompts from adapters.
    Both LLMService and model backends should use this function for consistency.

    Priority order per SPEC §5.0.1:
    1. prompt_instructions — the one prompt field, top-level or schema-nested
    2. Applicability natural language description (designed for LLM context)
    3. Description field ONLY if use_description_as_prompt is explicitly True

    Args:
        adapter: Adapter dict with prompt/behavior fields
        log_source: Identifier for logging (e.g., adapter ID or name)

    Returns:
        Extracted prompt string, or None if no valid prompt found
    """
    if not adapter or not isinstance(adapter, dict):
        return None

    # The canonical field, in either shape the dict arrives in. The alias
    # spellings (behavior_prompt, system_prompt, instructions,
    # prompt_template) were collapsed into prompt_instructions by the
    # schema.sql repair and are refused by the validator since.
    schema = adapter.get("schema", {})
    if not isinstance(schema, dict):
        schema = {}
    for source, value in (
        ("prompt_instructions", adapter.get("prompt_instructions")),
        ("schema.prompt_instructions", schema.get("prompt_instructions")),
    ):
        if isinstance(value, str) and value.strip():
            logger.debug(
                "prompt_extracted",
                source=log_source,
                field=source,
                length=len(value.strip()),
            )
            return value.strip()

    # Applicability natural language (explicitly for LLM context)
    applicability = adapter.get("applicability") or schema.get("applicability")
    if isinstance(applicability, dict):
        natural = applicability.get("natural_language")
        if isinstance(natural, str) and natural.strip():
            logger.debug(
                "prompt_extracted_from_applicability",
                source=log_source,
                length=len(natural.strip()),
            )
            return natural.strip()

    # Description ONLY with explicit opt-in flag
    # This prevents generic descriptions from being injected as behavioral prompts
    use_desc = adapter.get("use_description_as_prompt") or schema.get(
        "use_description_as_prompt"
    )
    if use_desc:
        description = adapter.get("description") or schema.get("description")
        if isinstance(description, str) and description.strip():
            logger.debug(
                "prompt_extracted_from_description_with_flag",
                source=log_source,
                length=len(description.strip()),
            )
            return description.strip()

    return None
