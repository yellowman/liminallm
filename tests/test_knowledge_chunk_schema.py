import json
from dataclasses import fields
from typing import get_type_hints
import pytest

from liminallm.api.schemas import KnowledgeChunkResponse
from liminallm.storage.errors import ConstraintViolation
from liminallm.storage.memory import MemoryStore
from liminallm.storage.models import KnowledgeChunk


def _create_context(store: MemoryStore, name: str = "ctx"):
    owner = store.create_user(f"{name}@example.com", tenant_id="tenant_ctx")
    return store.upsert_context(owner_user_id=owner.id, name=name, description="desc")


def test_knowledge_chunk_dataclass_fields_match_schema():
    chunk_field_names = {f.name for f in fields(KnowledgeChunk)}
    expected_fields = {
        "id",
        "context_id",
        "fs_path",
        "content",
        "embedding",
        "chunk_index",
        "created_at",
        "meta",
    }

    assert chunk_field_names == expected_fields


def test_knowledge_chunk_fs_path_is_required_per_spec():
    hints = get_type_hints(KnowledgeChunk)

    assert hints.get("fs_path") is str


def test_knowledge_chunk_response_matches_model_fields():
    assert set(KnowledgeChunkResponse.model_fields) == {
        "id",
        "context_id",
        "fs_path",
        "content",
        "chunk_index",
    }


def test_memory_store_round_trips_chunk_fields(tmp_path):
    store = MemoryStore(fs_root=str(tmp_path))
    context = _create_context(store)

    chunk = KnowledgeChunk(
        id=0,
        context_id=context.id,
        fs_path="inline",
        content="schema aligned",
        embedding=[0.1, 0.2],
        chunk_index=0,
        meta={"embedding_model_id": "deterministic"},
    )
    store.add_chunks(context.id, [chunk])

    reloaded = MemoryStore(fs_root=str(tmp_path))
    stored = reloaded.list_chunks(context.id)[0]

    assert stored.fs_path == "inline"
    assert stored.content == "schema aligned"
    assert stored.chunk_index == 0
    assert (stored.meta or {}).get("embedding_model_id") == "deterministic"


def test_memory_store_rejects_missing_fs_path(tmp_path):
    store = MemoryStore(fs_root=str(tmp_path))
    context = _create_context(store)

    with pytest.raises(ConstraintViolation):
        store.add_chunks(
            context.id,
            [
                KnowledgeChunk(
                    id=0,
                    context_id=context.id,
                    fs_path="",
                    content="",
                    embedding=[],
                    chunk_index=0,
                )
            ],
        )

    with pytest.raises(ConstraintViolation):
        store.add_chunks(
            context.id,
            [
                KnowledgeChunk(
                    id=0,
                    context_id=context.id,
                    fs_path="   ",
                    content="",
                    embedding=[],
                    chunk_index=1,
                )
            ],
        )


def test_memory_store_loads_legacy_chunk_without_fs_path(tmp_path):
    store = MemoryStore(fs_root=str(tmp_path))
    context = _create_context(store)

    chunk = KnowledgeChunk(
        id=0,
        context_id=context.id,
        fs_path="legacy",
        content="persisted chunk",
        embedding=[],
        chunk_index=0,
    )
    store.add_chunks(context.id, [chunk])

    state_path = tmp_path / "state" / "memory_store.json"
    legacy_state = json.loads(state_path.read_text())
    for chunk_data in legacy_state.get("chunks", []):
        chunk_data.pop("fs_path", None)
    state_path.write_text(json.dumps(legacy_state))

    reloaded = MemoryStore(fs_root=str(tmp_path))
    loaded_chunks = reloaded.list_chunks(context.id)

    assert loaded_chunks[0].fs_path == ""


def test_upsert_context_rejects_missing_owner(tmp_path):
    store = MemoryStore(fs_root=str(tmp_path))

    with pytest.raises(ConstraintViolation):
        store.upsert_context(owner_user_id=None, name="ctx", description="desc")
