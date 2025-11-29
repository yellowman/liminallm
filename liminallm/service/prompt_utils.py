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
    1. Explicit prompt fields (prompt_instructions, behavior_prompt, system_prompt, etc.)
    2. Schema-nested versions of the above fields
    3. Applicability natural language description (designed for LLM context)
    4. Description field ONLY if use_description_as_prompt is explicitly True

    Args:
        adapter: Adapter dict with prompt/behavior fields
        log_source: Identifier for logging (e.g., adapter ID or name)

    Returns:
        Extracted prompt string, or None if no valid prompt found
    """
    if not adapter or not isinstance(adapter, dict):
        return None

    # Priority 1: Check explicit prompt fields at top level
    prompt_fields = (
        "prompt_instructions",
        "behavior_prompt",
        "system_prompt",
        "instructions",
        "prompt_template",
    )

    for key in prompt_fields:
        value = adapter.get(key)
        if isinstance(value, str) and value.strip():
            logger.debug(
                "prompt_extracted",
                source=log_source,
                field=key,
                length=len(value.strip()),
            )
            return value.strip()

    # Priority 2: Check schema dict for nested prompt fields
    schema = adapter.get("schema", {})
    if isinstance(schema, dict):
        for key in prompt_fields:
            value = schema.get(key)
            if isinstance(value, str) and value.strip():
                logger.debug(
                    "prompt_extracted_from_schema",
                    source=log_source,
                    field=f"schema.{key}",
                    length=len(value.strip()),
                )
                return value.strip()

    # Priority 3: Applicability natural language (explicitly for LLM context)
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

    # Priority 4: Description ONLY with explicit opt-in flag
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
