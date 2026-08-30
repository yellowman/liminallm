"""What happens after a chat turn is already on its way to the user.

Turn labels, conversation titles, embedding backfill and the rolling digest all
run off the hot path, failures logged and dropped: a missing one costs
precision, never correctness, since the recent window is always sent verbatim.

Both transports call ``schedule_all``, so neither can remember only half of it.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from liminallm.logging import get_logger
from liminallm.service import compaction, labels

logger = get_logger(__name__)

# Scheduled tasks are held here so they are not garbage-collected mid-flight.
_TASKS: set[asyncio.Task] = set()

_EMBED_BACKFILL_PER_PASS = 30


def _spawn(coro) -> None:
    """Run a coroutine in the background, or drop it where there is no loop -
    a sync context should skip background polish rather than raise."""
    try:
        task = asyncio.create_task(coro)
    except RuntimeError:
        coro.close()
        return
    _TASKS.add(task)
    task.add_done_callback(_TASKS.discard)


def schedule_turn_labels(
    runtime,
    *,
    conversation_id: str,
    user_id: str,
    user_message_id: Optional[str],
    user_content: str,
    assistant_content: str,
    set_title: bool,
) -> None:
    """Describe a finished turn, and title the conversation if it is new."""

    async def _run() -> None:
        try:
            if user_message_id:
                label = await asyncio.to_thread(
                    labels.describe_turn, runtime.llm, user_content, assistant_content
                )
                await asyncio.to_thread(
                    runtime.store.update_message_meta,
                    user_message_id,
                    user_id=user_id,
                    patch={"turn_label": label},
                )
            if set_title:
                title = await asyncio.to_thread(
                    labels.describe_conversation, runtime.llm, user_content
                )
                await asyncio.to_thread(
                    runtime.store.set_conversation_title,
                    conversation_id,
                    user_id=user_id,
                    title=title,
                )
        except Exception as exc:  # noqa: BLE001 - never surface to the user
            logger.warning(
                "turn_labeling_failed", conversation_id=conversation_id, error=str(exc)
            )

    _spawn(_run())


def backfill_message_embeddings(runtime, history, user_id: str) -> None:
    """Persist per-turn embeddings for semantic recall. Real encoder only,
    skips what is already embedded, bounded so one pass cannot stall on a
    backlog - the next turn continues it."""
    embeddings = runtime.embeddings
    if not embeddings.is_semantic:
        return
    done = 0
    for msg in history:
        if done >= _EMBED_BACKFILL_PER_PASS:
            break
        if compaction._message_embedding(msg, embeddings.model_id) is not None:
            continue
        content = str(getattr(msg, "content", "") or "").strip()
        if not content:
            continue
        try:
            vector = embeddings.embed(content)
            runtime.store.update_message_meta(
                msg.id,
                user_id=user_id,
                patch={"embedding": vector, "embedding_model": embeddings.model_id},
            )
            done += 1
        except Exception as exc:  # noqa: BLE001 - backfill is best-effort
            logger.debug("message_embedding_backfill_failed", error=str(exc))
            break
    if done:
        logger.info("message_embeddings_backfilled", count=done)


def schedule_conversation_digest(runtime, *, conversation_id: str, user_id: str) -> None:
    """Fold turns older than the model's verbatim window into a digest."""

    async def _run() -> None:
        try:
            conversation = await asyncio.to_thread(
                runtime.store.get_conversation, conversation_id, user_id=user_id
            )
            if not conversation:
                return
            history = await asyncio.to_thread(
                runtime.store.list_messages, conversation_id, user_id=user_id
            )
            # Backfill first, so semantic recall reads persisted vectors on the
            # hot path instead of embedding turns live.
            await asyncio.to_thread(
                backfill_message_embeddings, runtime, history, user_id
            )
            # Digest exactly what falls outside the model's verbatim window -
            # the same budget boundary the workflow serves with.
            keep_tokens = runtime.workflow.history_budget()
            count = runtime.workflow._count_fn()
            if not compaction.needs_digest(
                history, conversation, keep_tokens=keep_tokens, count=count
            ):
                return
            digest = await asyncio.to_thread(
                lambda: compaction.build_digest(
                    runtime.llm,
                    history,
                    conversation,
                    keep_tokens=keep_tokens,
                    count=count,
                )
            )
            if not digest:
                return
            await asyncio.to_thread(
                runtime.store.merge_conversation_meta,
                conversation_id,
                {"digest": digest},
            )
            logger.info(
                "conversation_digested",
                conversation_id=conversation_id,
                messages=digest.get("messages"),
                tokens=digest.get("tokens"),
            )
        except Exception as exc:  # noqa: BLE001 - never surface to the user
            logger.warning(
                "conversation_digest_failed",
                conversation_id=conversation_id,
                error=str(exc),
            )

    _spawn(_run())


def schedule_all(
    runtime,
    *,
    conversation_id: str,
    user_id: str,
    user_message_id: Optional[str],
    user_content: str,
    assistant_content: str,
    set_title: bool,
) -> None:
    """Everything a finished turn owes, in one call."""
    schedule_turn_labels(
        runtime,
        conversation_id=conversation_id,
        user_id=user_id,
        user_message_id=user_message_id,
        user_content=user_content,
        assistant_content=assistant_content,
        set_title=set_title,
    )
    schedule_conversation_digest(
        runtime, conversation_id=conversation_id, user_id=user_id
    )


def conversation_title(message: str, max_length: int = 50) -> str:
    """A readable title from the first message, cut on a word boundary."""
    if not message:
        return "New conversation"
    cleaned = " ".join(message.split())
    if not cleaned:
        return "New conversation"
    if len(cleaned) <= max_length:
        return cleaned
    truncated = cleaned[:max_length]
    last_space = truncated.rfind(" ")
    if last_space > max_length // 2:
        truncated = truncated[:last_space]
    return truncated.rstrip(".,!?;:") + "..."
