from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List


@dataclass
class ValidationError(Exception):
    message: str
    path: List[Any]

    def __init__(self, message: str, path: Iterable[Any] | None = None):
        super().__init__(message)
        self.message = message
        self.path = list(path or [])


class Draft202012Validator:
    """Lightweight validator covering the project test cases.

    The real ``jsonschema`` dependency is not available in the execution
    environment, so this class performs the minimal checks needed by
    ``validate_artifact``: required top-level fields and required node fields
    inside a ``nodes`` array. Replace with the upstream package once the
    runtime image can install optional dependencies.
    """

    def __init__(self, schema: dict):
        self.schema = schema or {}

    def iter_errors(self, instance: dict):
        required = self.schema.get("required", [])
        for field in required:
            if field not in instance:
                yield ValidationError(f"'{field}' is a required property", path=[field])

        nodes_schema = self.schema.get("properties", {}).get("nodes", {}).get("items", {})
        node_required = nodes_schema.get("required", [])
        nodes = instance.get("nodes")
        if isinstance(nodes, list):
            for idx, node in enumerate(nodes):
                if not isinstance(node, dict):
                    continue
                for field in node_required:
                    if field not in node:
                        yield ValidationError(f"'{field}' is a required property", path=["nodes", idx, field])

