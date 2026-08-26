from __future__ import annotations

from typing import Any, Dict

from jsonschema import Draft202012Validator

from liminallm.service.workflow_graph import graph_problems

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
                        # `node_map` is keyed by this and drops falsy keys,
                        # so an empty id declares a node that then disappears.
                        "id": {"type": "string", "minLength": 1},
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
                                    # One id, not a fan-out. The switch
                                    # executor appends `branch["next"]` as a
                                    # single value and never flattens a list,
                                    # so advertising an array here promised
                                    # something execution does not do —
                                    # SPEC §9 gives fan-out to `parallel`.
                                    "next": {"type": "string", "minLength": 1},
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
    # A remote MCP server. Not a `tool`: a server is not callable, it
    # discovers zero or more tools, and overloading `tool.spec` would make a
    # configuration look like a capability before anything has been listed.
    #
    # `taint_class` is the operator's classification and the only one that
    # counts. It is deliberately not inferrable from the server's own
    # annotations: remote metadata is supplied by the party being classified.
    # Absent or unrecognized means `egress` (see `mcp_client.server_taint_class`),
    # so the enum here is what an operator may *attest*, not what is assumed.
    "mcp": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "kind": {"const": "mcp.server"},
            "name": {"type": "string", "minLength": 1},
            # Streamable HTTP only in this tranche. stdio would turn "connect
            # to a server" into "spawn the executable this row names", which
            # is a different privilege boundary and belongs to its own review.
            "url": {"type": "string", "pattern": "^https?://"},
            "enabled": {"type": "boolean"},
            "taint_class": {"enum": ["egress", "local_read"]},
            "description": {"type": "string"},
        },
        "required": ["kind", "name", "url"],
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
            # Pass C: mode is the one execution vocabulary. The old spellings
            # were normalized by the schema.sql repair; refusing them here is
            # what keeps them normalized — delete runtime compatibility
            # without this and the old formats are simply created again
            # tomorrow.
            "mode": {"enum": ["local", "remote", "prompt", "hybrid"]},
            "prompt_instructions": {"type": "string"},
            "fs_dir": {"type": "string"},
            "remote_model_id": {"type": "string"},
            "remote_adapter_id": {"type": "string"},
            "scope": {"type": "string"},
            "user_id": {"type": ["string", "null"]},
            "base_model": {"type": "string"},
            "rank": {"type": ["number", "integer"]},
            "layers": {"type": "array"},
            "matrices": {"type": "array"},
            "current_version": {"type": "integer", "minimum": 0},
            # Retired spellings, rejected by name so the error says which.
            "backend": False,
            "provider": False,
            "cephfs_dir": False,
            "behavior_prompt": False,
            "system_prompt": False,
            "instructions": False,
            "prompt_template": False,
            "model_id": False,
            "adapter_id": False,
        },
        "required": ["kind", "mode", "base_model", "current_version"],
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
    if type_ == "workflow":
        # Shape first, then whether the graph matches itself: JSON Schema can
        # say `next` is a string, and cannot say the string names a node that
        # exists. Kept out of the schema because two of the five edge fields
        # the executor reads are not in it, so the two would drift apart.
        problems = graph_problems(schema)
        if problems:
            raise ArtifactValidationError("workflow graph is not consistent", problems)
