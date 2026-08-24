from __future__ import annotations

from typing import Any, Dict, Optional


class ConstraintViolation(Exception):
    """Raised when a storage-layer uniqueness or FK constraint is violated."""

    def __init__(self, message: str, detail: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.detail = detail or {}


class ConversationGone(ConstraintViolation):
    """The conversation this work belongs to was deleted while it ran.

    A distinct type because it is not a caller error and not a bug: an upload
    validates the conversation, then does seconds of file, hashing and
    indexing work, and the owner may delete the chat in between. The database
    refuses the write through `knowledge_context.conversation_id`, and the
    request has to fail rather than report success for state that no longer
    has anywhere to live.
    """


class TrainingInProgress(ConstraintViolation):
    """An adapter cannot be retired while a worker is training it.

    Training writes weights under the adapter's tree and then promotes a
    version onto the artifact. `training_job.adapter_id` cascades, so deleting
    the artifact would take the job record with it while the worker carried on
    writing and then tried to promote onto a row that no longer exists.
    """


__all__ = ["ConstraintViolation", "ConversationGone", "TrainingInProgress"]
