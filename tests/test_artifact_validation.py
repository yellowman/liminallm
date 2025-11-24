import pytest

from liminallm.service.artifact_validation import ArtifactValidationError, validate_artifact


def test_validate_artifact_accepts_valid_workflow():
    schema = {
        "kind": "workflow.chat",
        "entrypoint": "start",
        "nodes": [
            {"id": "start", "type": "tool_call", "tool": "demo", "inputs": {}, "outputs": ["x"], "next": "end"},
            {"id": "end", "type": "end"},
        ],
    }

    validate_artifact("workflow", schema)


def test_validate_artifact_rejects_invalid_schema():
    schema = {"kind": "workflow.chat", "nodes": [{"id": "n1"}]}

    with pytest.raises(ArtifactValidationError) as excinfo:
        validate_artifact("workflow", schema)

    assert "'type' is a required property" in excinfo.value.errors[0]


def test_validate_artifact_rejects_type_errors():
    schema = {"kind": "workflow.chat", "entrypoint": "start", "nodes": "not-a-list"}

    with pytest.raises(ArtifactValidationError) as excinfo:
        validate_artifact("workflow", schema)

    assert "array" in excinfo.value.errors[0]
