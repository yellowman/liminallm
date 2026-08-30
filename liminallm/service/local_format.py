"""One serialization of a conversation for the local decoder.

A raw decoder has no out-of-band role structure: the role labels ARE tokens.
So two code paths that write ``USER:`` and ``user:`` are not two styles of the
same thing - they are two different inputs, and an adapter fitted to one is
being asked to serve the other. Training built history with uppercase labels
and a ``CONTEXT_SNIPPET:`` marker while serving flattened with lowercase, so
"same checkpoint, same tokenizer" still did not mean the same input.

Truncation lives here for the same reason. Tokenizers truncate from the right,
keeping the OLDEST tokens; a chat wants the newest. Training had been fixed to
keep the tail while serving still let the tokenizer decide, which reintroduced
the mismatch on exactly the long conversations where it matters most.

Both are single functions on purpose: if this file is the only place that
answers "what does the model actually see", the two paths cannot drift again.
"""

from __future__ import annotations

from typing import Any, Iterable, List, Optional, Sequence

#: The label written before each turn. Lowercase, matching what serving has
#: always emitted - training is the side that moves, because the serving
#: format is the one users' text has been flowing through.
ROLE_LABELS = ("system", "user", "assistant", "tool")

CONTEXT_ROLE = "context"


def format_turn(role: str, content: str) -> str:
    label = (role or "user").strip().lower()
    if label not in ROLE_LABELS and label != CONTEXT_ROLE:
        label = "user"
    return f"{label}: {content}"


def format_conversation(messages: Iterable[Any]) -> str:
    """Messages - dicts or ORM rows - as the model's prompt text."""
    lines: List[str] = []
    for message in messages or []:
        if isinstance(message, dict):
            role = message.get("role", "user")
            content = message.get("content", "")
        else:
            role = getattr(message, "role", "user")
            content = getattr(message, "content", "")
        lines.append(format_turn(role, content or ""))
    return "\n".join(lines)


def place_context(messages: List[Any], snippets: Sequence[str]) -> List[Any]:
    """Insert context turns where BOTH paths must put them: immediately
    before the last user turn, so the retrieved material reads as ground for
    the question that follows it.

    Placement is part of the representation. Training appended the context
    after every message (so it landed *after* the question) while serving
    inserted it before the final user turn - same marker, different token
    order, which for a raw decoder is a different input. Whichever order is
    chosen, it has to be chosen once, here.
    """
    turns = [
        {"role": CONTEXT_ROLE, "content": snippet} for snippet in snippets if snippet
    ]
    if not turns:
        return list(messages)
    result = list(messages)
    for index in range(len(result) - 1, -1, -1):
        role = (
            result[index].get("role")
            if isinstance(result[index], dict)
            else getattr(result[index], "role", None)
        )
        if role == "user":
            return result[:index] + turns + result[index:]
    return result + turns


def keep_newest(tokens: Sequence[int], limit: Optional[int]) -> List[int]:
    """The last ``limit`` tokens: a chat's newest turn is the load-bearing one.

    Callers must not delegate this to the tokenizer's own ``truncation``,
    which keeps the beginning of the text.
    """
    if limit is None or limit <= 0 or len(tokens) <= limit:
        return list(tokens)
    return list(tokens[-limit:])
