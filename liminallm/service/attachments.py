"""Conversation attachments — files usable in a chat without any setup.

A file uploaded against a conversation becomes immediately usable by the model
in that conversation. There is no user-facing "context" concept: each
conversation gets one implicit knowledge context (marked ``meta.auto``, hidden
from the contexts UI) that holds chunks for anything worth searching.

Each attachment is classified once, at upload, into the capabilities it
supports:

- ``inline``     small text files are injected verbatim into the prompt, so the
                 model sees the whole file with no retrieval step at all
- ``searchable`` larger text/documents are chunked + embedded, and the model
                 pulls from them by calling the ``file_search`` tool
- ``analyzable`` every attachment is readable from the code interpreter's
                 working directory, which is how archives, spreadsheets, and
                 anything binary get used

The attachment list lives in ``conversation.meta.attachments`` so no new table
or migration is needed.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from liminallm.service.archive import ARCHIVE_SUFFIXES
from liminallm.service.fs import PathTraversalError, safe_join

# Text formats we can read directly off disk.
TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".json", ".csv", ".tsv", ".yaml", ".yml",
}
# Formats worth chunking but not injecting verbatim.
DOCUMENT_EXTENSIONS = {".pdf"}
# Formats the model should reach for the interpreter to parse.
DATA_EXTENSIONS = {".csv", ".tsv", ".json"} | set(ARCHIVE_SUFFIXES)

# A small text file is cheaper and more faithful to inject whole than to chunk;
# past this size retrieval wins. ~12KB is roughly 3k tokens.
INLINE_MAX_BYTES = 12_000
# Ceiling on all inline content in one prompt, so many small files can't
# crowd out the conversation itself.
INLINE_TOTAL_BUDGET = 32_000


def classify_attachment(filename: str, size: int) -> dict[str, Any]:
    """Decide how a file can be used, from its name and size alone."""
    ext = Path(filename).suffix.lower()
    is_archive = filename.lower().endswith(ARCHIVE_SUFFIXES)
    is_text = ext in TEXT_EXTENSIONS
    is_doc = ext in DOCUMENT_EXTENSIONS
    small = size <= INLINE_MAX_BYTES

    # Inline and searchable are deliberately exclusive: injecting a file AND
    # retrieving chunks of it would put the same text in the prompt twice.
    inline = bool(is_text and small)
    searchable = bool((is_text and not small) or is_doc)
    return {
        "ext": ext,
        "inline": inline,
        "searchable": searchable,
        # Archives and data files are always worth mentioning as analyzable;
        # so is anything we can neither inline nor search.
        "analyzable": bool(
            is_archive or ext in DATA_EXTENSIONS or not (inline or searchable)
        ),
    }


def user_files_dir(fs_root: str, user_id: str) -> Path:
    return Path(fs_root) / "users" / user_id / "files"


def attachment_path(fs_root: str, user_id: str, name: str) -> Optional[Path]:
    """Resolve an attachment name to a path inside the user's file area."""
    try:
        return safe_join(user_files_dir(fs_root, user_id), name)
    except PathTraversalError:
        return None


def ensure_conversation_context(store, *, user_id: str, conversation_id: str) -> Any:
    """Get (or create) the implicit knowledge context for a conversation.

    Marked ``meta.auto`` and ``meta.conversation_id`` so the contexts UI can
    filter it out — users never manage these directly.
    """
    for ctx in store.list_contexts(owner_user_id=user_id, limit=500) or []:
        meta = ctx.meta or {}
        if meta.get("auto") and meta.get("conversation_id") == conversation_id:
            return ctx
    return store.upsert_context(
        user_id,
        f"conversation:{conversation_id}",
        "Files attached to a conversation",
        meta={"auto": True, "conversation_id": conversation_id},
    )


def find_conversation_context_id(store, *, user_id: str, conversation_id: str) -> Optional[str]:
    """The conversation's implicit context id, without creating one."""
    for ctx in store.list_contexts(owner_user_id=user_id, limit=500) or []:
        meta = ctx.meta or {}
        if meta.get("auto") and meta.get("conversation_id") == conversation_id:
            return ctx.id
    return None


def is_auto_context(ctx: Any) -> bool:
    """True for implicit per-conversation contexts (hidden from the UI)."""
    return bool((getattr(ctx, "meta", None) or {}).get("auto"))


def list_attachments(conversation: Any) -> list[dict[str, Any]]:
    meta = getattr(conversation, "meta", None) or {}
    items = meta.get("attachments")
    return [a for a in items if isinstance(a, dict)] if isinstance(items, list) else []


def record_attachment(
    store,
    *,
    conversation_id: str,
    user_id: str,
    name: str,
    size: int,
    capabilities: dict[str, Any],
    chunk_count: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Add (or replace) an attachment record on the conversation."""
    conversation = store.get_conversation(conversation_id, user_id=user_id)
    if not conversation:
        return []
    existing = [a for a in list_attachments(conversation) if a.get("name") != name]
    existing.append(
        {
            "name": name,
            "size": size,
            "inline": bool(capabilities.get("inline")),
            "searchable": bool(capabilities.get("searchable")),
            "analyzable": bool(capabilities.get("analyzable")),
            "chunk_count": chunk_count,
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    store.merge_conversation_meta(
        conversation_id, user_id=user_id, patch={"attachments": existing}
    )
    return existing


def read_inline_contents(
    attachments: list[dict[str, Any]], *, fs_root: str, user_id: str
) -> list[dict[str, str]]:
    """Read the text of every inline attachment, within the total budget."""
    out: list[dict[str, str]] = []
    remaining = INLINE_TOTAL_BUDGET
    for att in attachments:
        if not att.get("inline") or remaining <= 0:
            continue
        path = attachment_path(fs_root, user_id, att.get("name") or "")
        if not path or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if len(text) > remaining:
            text = text[:remaining] + "\n...[truncated]"
        remaining -= len(text)
        out.append({"name": att["name"], "content": text})
    return out


def describe_attachments(attachments: list[dict[str, Any]]) -> str:
    """One line per attachment, telling the model how it can reach each one."""
    lines = []
    for att in attachments:
        how = []
        if att.get("inline"):
            how.append("full text included below")
        if att.get("searchable"):
            how.append("searchable via file_search")
        if att.get("analyzable"):
            how.append("readable in run_python's working directory")
        size = att.get("size") or 0
        lines.append(f"- {att.get('name')} ({size} bytes) — {'; '.join(how) or 'stored'}")
    return "\n".join(lines)


def build_attachment_preamble(
    attachments: list[dict[str, Any]], *, fs_root: str, user_id: str
) -> str:
    """System-prompt block describing the attachments and inlining small ones."""
    if not attachments:
        return ""
    parts = [
        "Files attached to this conversation:",
        describe_attachments(attachments),
    ]
    inline = read_inline_contents(attachments, fs_root=fs_root, user_id=user_id)
    for item in inline:
        parts.append(f"\n--- contents of {item['name']} ---\n{item['content']}")
    # Which capability applies to these files; how each tool works is already
    # in its schema description — say it once, there.
    if any(a.get("searchable") for a in attachments):
        parts.append("\nUse file_search to look inside the searchable files.")
    if any(a.get("analyzable") for a in attachments):
        parts.append("\nUse run_python to work on the files directly.")
    return "\n".join(parts)
