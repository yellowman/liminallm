from __future__ import annotations

from typing import Any, Dict

from jsonschema import Draft202012Validator, ValidationError


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
                    },
                    "required": ["id", "type"],
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
            "timeout_seconds": {"type": "number", "minimum": 0},
        },
        "required": ["kind", "name", "handler"],
    },
}


class ArtifactValidationError(Exception):
    def __init__(self, message: str, errors: list[str]):
        super().__init__(message)
        self.errors = errors


def validate_artifact(type_: str, schema: Dict[str, Any]) -> None:
    validator_schema = _ARTIFACT_SCHEMAS.get(type_)
    if not validator_schema:
        return
    validator = Draft202012Validator(validator_schema)
    errors = sorted(validator.iter_errors(schema), key=lambda e: e.path)
    if errors:
        messages = [e.message for e in errors]
        raise ArtifactValidationError("artifact validation failed", messages)
