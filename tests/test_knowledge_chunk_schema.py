from dataclasses import fields
import uuid

from liminallm.api.schemas import KnowledgeChunkResponse
from liminallm.storage.memory import MemoryStore
from liminallm.storage.models import KnowledgeChunk


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
    context = store.upsert_context(owner_user_id=None, name="ctx", description="desc")

    chunk = KnowledgeChunk(
        id=None,
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
