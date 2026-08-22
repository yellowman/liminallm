"""One chat turn, independent of the transport that carried it.

POST /chat and the /chat/stream WebSocket differ only in how the reply reaches
the client. Everything else — resolving the conversation, checking context
ownership, transcribing voice, persisting both messages, scheduling the
post-turn work, warming the cache — is the same turn, and was written twice.
Three things had already drifted, all silently: the socket never warmed the
cache, its non-streaming branch scheduled nothing, and a conversation it
created never recorded its ``active_context_id``.

API-layer rather than service-layer: both callers are transports, and the
ownership checks are HTTP concerns, so raising HTTPException here fits.
"""

from __future__ import annotations

import base64
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Optional

from liminallm.api.errors import http_error
from liminallm.api.schemas import ChatResponse
from liminallm.content_struct import normalize_content_struct
from liminallm.logging import get_logger
from liminallm.service import turn_effects
from liminallm.service.auth import AuthContext

logger = get_logger(__name__)


@dataclass
class Turn:
    """A turn in progress: everything resolved before the model runs."""

    principal: AuthContext
    conversation_id: str
    context_id: Optional[str]
    workflow_id: Optional[str]
    user_content: str
    user_message: Any
    #: This turn created the conversation, so a model-written title is still
    #: owed to replace the placeholder.
    needs_title: bool
    orchestration: dict[str, Any] = field(default_factory=dict)

    @property
    def user_id(self) -> str:
        return self.principal.user_id


async def begin(
    runtime,
    principal: AuthContext,
    *,
    conversation_id: Optional[str],
    context_id: Optional[str],
    workflow_id: Optional[str],
    content: str,
    content_struct: Any = None,
    mode: str = "text",
    conversation_meta: Optional[dict] = None,
    owned_conversation,
    owned_context,
) -> Turn:
    """Resolve the conversation and context, then persist the user's message.

    The ownership checks are passed in so this module need not import the
    route helpers that call it.
    """
    requested_conversation_id = conversation_id
    validated_context_id: Optional[str] = None

    if requested_conversation_id:
        conversation = owned_conversation(runtime, requested_conversation_id, principal)
    else:
        if context_id:
            owned_context(runtime, context_id, principal)
            validated_context_id = context_id
        # A placeholder; the model-written title replaces it once the turn
        # completes (see turn_effects.schedule_all).
        conversation = runtime.store.create_conversation(
            user_id=principal.user_id,
            active_context_id=context_id,
            title=turn_effects.conversation_title(content),
            meta=conversation_meta,
        )

    resolved_id = conversation.id
    context_id = context_id or conversation.active_context_id
    if context_id and context_id != validated_context_id:
        owned_context(runtime, context_id, principal)

    user_content = content
    voice_meta: dict = {}
    if mode == "voice":
        user_content, voice_meta = await _transcribe(runtime, content, principal.user_id)

    user_message = runtime.store.append_message(
        resolved_id,
        sender="user",
        role="user",
        content=user_content,
        meta=voice_meta or None,
        content_struct=normalize_content_struct(content_struct, user_content),
    )
    return Turn(
        principal=principal,
        conversation_id=resolved_id,
        context_id=context_id,
        workflow_id=workflow_id,
        user_content=user_content,
        user_message=user_message,
        needs_title=not requested_conversation_id,
    )


async def _transcribe(runtime, payload: str, user_id: str) -> tuple[str, dict]:
    try:
        audio_bytes = base64.b64decode(payload)
    except Exception as exc:
        raise http_error(
            "validation_error", "invalid base64-encoded audio payload", status_code=400
        ) from exc
    transcript = await runtime.voice.transcribe(audio_bytes, user_id=user_id) or {}
    text = transcript.get("transcript") or transcript.get("text")
    if not text:
        raise http_error(
            "validation_error", "unable to transcribe audio", status_code=400
        )
    return text, {"mode": "voice", "transcript": transcript}


@contextmanager
def _unguarded(_capability: str, _payload: Any):
    """Stand-in for a caller with no ledger — the streaming path, and tests."""
    yield SimpleNamespace(replayable=False, result=None)


async def finish(
    runtime,
    turn: Turn,
    orchestration: Any,
    *,
    content: str = "",
    assistant_message_id: Optional[str] = None,
    commit: Optional[Callable[..., Any]] = None,
) -> Any:
    """Persist the reply, schedule the post-turn work, warm the cache.

    ``content`` is the streaming path's token-assembled reply, used when the
    orchestration carries none. ``assistant_message_id`` lets a transport that
    must announce the reply's id before it exists (the streaming Responses
    surface) mint it up front and keep every event consistent with the row.

    ``commit`` is the request's ledger (``IdempotencyGuard.commit``). SPEC §11.3
    makes storing the assistant message the first durable mutation of the write
    path, so it is guarded here, around the append itself. Guarding the route
    instead would record that a turn was attempted — which is what the
    idempotency slot already records, and is not the same claim as "the message
    is in the table".
    """
    turn.orchestration = orchestration if isinstance(orchestration, dict) else {}
    assistant_content = turn.orchestration.get(
        "content", content or "No response generated."
    )
    guard = commit or _unguarded
    with guard(
        "message.assistant",
        {"conversation_id": turn.conversation_id, "content": assistant_content},
    ) as operation:
        assistant_message = runtime.store.append_message(
            turn.conversation_id,
            sender="assistant",
            role="assistant",
            content=assistant_content,
            content_struct=normalize_content_struct(
                turn.orchestration.get("content_struct"), assistant_content
            ),
            meta={
                "adapters": turn.orchestration.get("adapters", []),
                "adapter_gates": turn.orchestration.get("adapter_gates", []),
                "routing_trace": turn.orchestration.get("routing_trace", []),
                "workflow_trace": turn.orchestration.get("workflow_trace", []),
                "usage": turn.orchestration.get("usage", {}),
            },
            message_id=assistant_message_id,
        )
        operation.result = {"message_id": getattr(assistant_message, "id", None)}

    turn_effects.schedule_all(
        runtime,
        conversation_id=turn.conversation_id,
        user_id=turn.user_id,
        user_message_id=getattr(turn.user_message, "id", None),
        user_content=turn.user_content,
        assistant_content=assistant_content,
        set_title=turn.needs_title,
    )

    if runtime.cache:
        # Bounded, so a long conversation can't blow up the cached state.
        history = runtime.store.list_messages(
            turn.conversation_id,
            limit=runtime.settings.max_page_size,
            user_id=turn.user_id,
        )
        await runtime.workflow.cache_conversation_state(
            turn.conversation_id, history, turn.user_id
        )

    return assistant_message


def response(turn: Turn, assistant_message, adapter_names: list[str]) -> ChatResponse:
    """The reply payload, identical on both transports."""
    o = turn.orchestration
    return ChatResponse(
        message_id=assistant_message.id,
        conversation_id=turn.conversation_id,
        content=assistant_message.content,
        content_struct=assistant_message.content_struct,
        workflow_id=turn.workflow_id,
        adapters=adapter_names,
        adapter_gates=o.get("adapter_gates", []),
        usage=o.get("usage", {}),
        context_snippets=o.get("context_snippets", []),
        routing_trace=o.get("routing_trace", []),
        workflow_trace=o.get("workflow_trace", []),
    )
