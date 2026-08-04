from __future__ import annotations

from typing import Any, Dict

from jsonschema import Draft202012Validator

_ARTIFACT_SCHEMAS: dict[str, Dict[str, Any]] = {
    "workflow": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "kind": {"type": "string"},
            "entrypoint": {"type": "string"},
            "nodes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "type": {"type": "string"},
                        "tool": {"type": "string"},
                        "inputs": {"type": "object"},
                        "outputs": {"type": "array"},
                        "next": {"anyOf": [{"type": "string"}, {"type": "array"}]},
                        "branches": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "when": {},
                                    "next": {
                                        "anyOf": [{"type": "string"}, {"type": "array"}]
                                    },
                                },
                                "required": ["when", "next"],
                            },
                        },
                    },
                    "required": ["id", "type"],
                    "allOf": [
                        {
                            "if": {"properties": {"type": {"const": "switch"}}},
                            "then": {"required": ["branches"]},
                        },
                        {
                            "if": {"properties": {"type": {"const": "tool_call"}}},
                            "then": {"required": ["tool"]},
                        },
                        {
                            "if": {"properties": {"type": {"const": "parallel"}}},
                            "then": {"required": ["next"]},
                        },
                    ],
                },
            },
        },
        "required": ["kind", "nodes"],
    },
    "tool": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "kind": {"const": "tool.spec"},
            "name": {"type": "string"},
            "handler": {"type": "string"},
            "timeout_seconds": {"type": "number", "exclusiveMinimum": 0},
        },
        "required": ["kind", "name", "handler"],
    },
    "adapter": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "kind": {"const": "adapter.lora"},
            "backend": {"type": "string"},
            "provider": {"type": "string"},
            "scope": {"type": "string"},
            "user_id": {"type": ["string", "null"]},
            "base_model": {"type": "string"},
            "rank": {"type": ["number", "integer"]},
            "layers": {"type": "array"},
            "matrices": {"type": "array"},
            "current_version": {"type": "integer", "minimum": 0},
        },
        "required": ["kind", "base_model", "current_version"],
        "additionalProperties": True,
    },
    # SPEC §6.1 policy.routing / §8.1. The workflow engine already reads these
    # (list_artifacts(type_filter="policy")) and §13.4 documents the type on the
    # list endpoint — but there was no schema here, so validate_artifact refused
    # "unknown artifact type" and POST /v1/artifacts could not create one.
    # Routing-as-data had no way to get its data in.
    "policy": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "kind": {"const": "policy.routing"},
            "name": {"type": "string"},
            "description": {"type": "string"},
            "rules": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        # The condition the sandboxed evaluator runs (§8.1).
                        "when": {"type": "string"},
                        "action": {"type": "object"},
                    },
                    "required": ["when", "action"],
                    "additionalProperties": True,
                },
            },
        },
        "required": ["kind", "rules"],
        "additionalProperties": True,
    },
    "artifact": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": True,
    },
}


class ArtifactValidationError(Exception):
    def __init__(self, message: str, errors: list[str]):
        super().__init__(message)
        self.errors = errors


def validate_artifact(type_: str, schema: Dict[str, Any]) -> None:
    validator_schema = _ARTIFACT_SCHEMAS.get(type_)
    if not validator_schema:
        raise ArtifactValidationError("unknown artifact type", [type_])
    validator = Draft202012Validator(validator_schema)
    errors = sorted(validator.iter_errors(schema), key=lambda e: e.path)
    if errors:
        messages = [e.message for e in errors]
        raise ArtifactValidationError("artifact validation failed", messages)
