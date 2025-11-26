from liminallm.service.training import TrainingService
from liminallm.storage.memory import MemoryStore


def _create_user_and_conversation(store: MemoryStore, suffix: str = ""):
    user = store.create_user(f"test{suffix}@example.com")
    conversation = store.create_conversation(user.id, title="test conversation")
    return user, conversation


def _append_assistant_message(store: MemoryStore, conversation_id: str, sender_id: str):
    return store.append_message(conversation_id, sender=sender_id, role="assistant", content="hello")


def test_feedback_enqueues_single_training_job_with_cooldown(tmp_path):
    store = MemoryStore(fs_root=str(tmp_path))
    training = TrainingService(store, fs_root=str(tmp_path))
    user, conversation = _create_user_and_conversation(store)
    message = _append_assistant_message(store, conversation.id, user.id)

    training.record_feedback_event(
        user_id=user.id,
        conversation_id=conversation.id,
        message_id=message.id,
        feedback="positive",
    )

    assert len(store.list_training_jobs(user_id=user.id)) == 1

    next_message = _append_assistant_message(store, conversation.id, user.id)
    training.record_feedback_event(
        user_id=user.id,
        conversation_id=conversation.id,
        message_id=next_message.id,
        feedback="like",
    )

    assert len(store.list_training_jobs(user_id=user.id)) == 1

    job = store.list_training_jobs(user_id=user.id)[0]
    store.update_training_job(job.id, status="succeeded")
    training.training_job_cooldown_seconds = 1000

    third_message = _append_assistant_message(store, conversation.id, user.id)
    training.record_feedback_event(
        user_id=user.id,
        conversation_id=conversation.id,
        message_id=third_message.id,
        feedback="positive",
    )

    assert len(store.list_training_jobs(user_id=user.id)) == 1

    training.training_job_cooldown_seconds = 0
    fourth_message = _append_assistant_message(store, conversation.id, user.id)
    training.record_feedback_event(
        user_id=user.id,
        conversation_id=conversation.id,
        message_id=fourth_message.id,
        feedback="positive",
    )

    assert len(store.list_training_jobs(user_id=user.id)) == 2
