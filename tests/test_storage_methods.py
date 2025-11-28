import inspect
from copy import deepcopy

import pytest

from liminallm.storage.errors import ConstraintViolation
from liminallm.storage.memory import MemoryStore
from liminallm.storage.postgres import PostgresStore


@pytest.fixture
def memory_store(tmp_path):
    return MemoryStore(fs_root=str(tmp_path))


def _create_user_conversation_message(store: MemoryStore):
    user = store.create_user("test@example.com")
    conversation = store.create_conversation(user.id, title="test conversation")
    message = store.append_message(
        conversation.id, sender=user.id, role="user", content="hello"
    )
    return user, conversation, message


def test_get_preference_event_round_trip(memory_store: MemoryStore):
    user, conversation, message = _create_user_conversation_message(memory_store)

    event = memory_store.record_preference_event(
        user.id,
        conversation.id,
        message.id,
        feedback="thumbs_up",
        score=0.9,
        weight=0.9,
    )

    fetched = memory_store.get_preference_event(event.id)

    assert fetched is event
    assert fetched.score == pytest.approx(0.9)
    assert fetched.weight == pytest.approx(0.9)
    assert fetched.feedback == "thumbs_up"


def test_get_preference_event_missing_returns_none(memory_store: MemoryStore):
    assert memory_store.get_preference_event("missing") is None


def test_record_preference_event_requires_valid_message(memory_store: MemoryStore):
    user, conversation, message = _create_user_conversation_message(memory_store)
    other_conversation = memory_store.create_conversation(user.id, title="other")
    other_message = memory_store.append_message(
        other_conversation.id, sender=user.id, role="assistant", content="other"
    )

    with pytest.raises(ConstraintViolation):
        memory_store.record_preference_event(
            user.id,
            conversation.id,
            "missing",  # nonexistent message
            feedback="thumbs_down",
        )

    with pytest.raises(ConstraintViolation):
        memory_store.record_preference_event(
            user.id,
            conversation.id,
            other_message.id,  # message from different conversation
            feedback="thumbs_down",
        )


def test_record_preference_event_populates_embedding(memory_store: MemoryStore):
    user, conversation, message = _create_user_conversation_message(memory_store)

    event = memory_store.record_preference_event(
        user.id,
        conversation.id,
        message.id,
        feedback="thumbs_up",
    )

    assert event.context_embedding
    assert len(event.context_embedding) == 64


def test_get_training_job_round_trip(memory_store: MemoryStore):
    user, *_ = _create_user_conversation_message(memory_store)

    job = memory_store.create_training_job(user.id, adapter_id="adapter-1")

    fetched = memory_store.get_training_job(job.id)

    assert fetched is job
    assert fetched.adapter_id == "adapter-1"
    assert fetched.status == "queued"


def test_get_latest_workflow_returns_newest_schema(memory_store: MemoryStore):
    workflow = next(
        a
        for a in memory_store.list_artifacts(type_filter="workflow")
        if a.name == "default_chat_workflow"
    )
    updated_schema = deepcopy(workflow.schema)
    updated_schema["description"] = "updated"

    memory_store.update_artifact(workflow.id, updated_schema)

    latest_schema = memory_store.get_latest_workflow(workflow.id)

    assert latest_schema == updated_schema


def test_storage_method_parity():
    memory_methods = {
        name
        for name, fn in inspect.getmembers(MemoryStore, predicate=inspect.isfunction)
        if not name.startswith("_") and name != "__init__"
    }
    postgres_methods = {
        name
        for name, fn in inspect.getmembers(PostgresStore, predicate=inspect.isfunction)
        if not name.startswith("_") and name != "__init__"
    }

    allowed_memory_only = {"default_artifacts"}

    assert memory_methods - allowed_memory_only == postgres_methods
    assert allowed_memory_only.issuperset(memory_methods - postgres_methods)
