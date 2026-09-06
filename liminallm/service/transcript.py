"""The parent's own record of the conversation it is having with the model.

An agent turn is a conversation the parent conducts and the worker drives: the
worker decides which tools to call and assembles a message list, but every
message in it describes something the parent did. The worker is the untrusted
half, so what it sends back is a claim about that conversation rather than the
conversation.

This is the parent's copy. It is appended to as the turn runs - one entry per
committed operation - and it is what a later stage builds model input from,
so a message that acquires citation authority can be constructed from these
records rather than edited in place from the worker's version of them.

Two things it deliberately is not.

It is not `canonical_model_response`, which is replacement state on purpose:
the final answer's citations come from the *last* model turn and searching
backwards for one whose text happens to match is the ambiguity the citation
layer exists to refuse. This is the append-only record beside it, because the
third model call's prompt contains the first round's tool results and
replacement cannot say so.

It is not provenance. `GroundedPassage` says which evidence appears where in
a rendered string; a `TrustedToolResult` says which call the parent executed
and what the resulting tool message is. Keeping them apart is what stops the
provenance vocabulary from growing a tool-calling protocol inside it.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from liminallm.service.provenance import GroundedSpan


def _spans_from(raw: Any) -> Tuple[GroundedSpan, ...]:
    return tuple(
        GroundedSpan(
            start=int(span["start"]),
            end=int(span["end"]),
            source_id=str(span["source_id"]),
            evidence_id=str(span["evidence_id"]),
        )
        for span in raw or []
    )


@dataclass(frozen=True)
class TrustedToolResult:
    """One call the parent executed, and the result it produced.

    Every call of a round is recorded, including the ones that grounded
    nothing: the record exists so a tool message can be rebuilt without
    asking the worker what happened, and a message missing from the middle of
    a round is as much a gap as a wrong one.

    `call_index` is the identity. The parent enumerates the calls it
    dispatched, and the round's payload is hashed whole, so the same position
    in a matching round is the same call on a replay. `submitted_call_id` is
    the model's own name for it, relayed through the worker: kept because the
    tool message has to carry it, never treated as identity, since a provider
    can repeat one or send none.
    """

    operation_seq: int
    call_index: int
    tool_name: str
    #: What arrived in the round payload. Metadata, not identity.
    submitted_call_id: str = ""
    #: What the tool message carries, which is the id or the name - the same
    #: rule the worker's own assembly uses, computed here so the parent does
    #: not have to read it back from the worker to rebuild the message.
    tool_message_id: str = ""
    text: str = ""
    spans: Tuple[GroundedSpan, ...] = ()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "operation_seq": self.operation_seq,
            "call_index": self.call_index,
            "tool_name": self.tool_name,
            "submitted_call_id": self.submitted_call_id,
            "tool_message_id": self.tool_message_id,
            "text": self.text,
            "spans": [
                {
                    "start": span.start,
                    "end": span.end,
                    "source_id": span.source_id,
                    "evidence_id": span.evidence_id,
                }
                for span in self.spans
            ],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "TrustedToolResult":
        return cls(
            operation_seq=int(raw["operation_seq"]),
            call_index=int(raw["call_index"]),
            tool_name=str(raw.get("tool_name") or ""),
            submitted_call_id=str(raw.get("submitted_call_id") or ""),
            tool_message_id=str(raw.get("tool_message_id") or ""),
            text=str(raw.get("text") or ""),
            spans=_spans_from(raw.get("spans")),
        )


@dataclass(frozen=True)
class ToolRound:
    """One `tools.round` operation, as the parent executed it.

    `offerable` is false when the calls the worker submitted are not the calls
    the previous model turn asked for. Such a round still runs - what a worker
    may request is the capability layer's question, and it answers that one
    the same way it always did - but the parent can no longer reconstruct the
    exchange faithfully, so nothing in it may carry a citation. Divergence is
    a property of the round rather than of a call, because the mismatch is in
    the correspondence between two lists.
    """

    operation_seq: int
    results: Tuple[TrustedToolResult, ...] = ()
    offerable: bool = True

    def as_dict(self) -> Dict[str, Any]:
        return {
            "kind": "tool_round",
            "operation_seq": self.operation_seq,
            "offerable": self.offerable,
            "results": [result.as_dict() for result in self.results],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ToolRound":
        return cls(
            operation_seq=int(raw["operation_seq"]),
            offerable=bool(raw.get("offerable", True)),
            results=tuple(
                TrustedToolResult.from_dict(item) for item in raw.get("results") or []
            ),
        )


@dataclass(frozen=True)
class ModelTurn:
    """One model turn, in the form the conversation continued from.

    The *public* reply, not the canonical one. The worker appends the public
    assistant message and executes the public tool calls, so that is what the
    exchange actually proceeded from and what a reconstruction of it has to
    contain. The canonical reply - the same turn with its citation handles
    still in it - is kept separately and is authority for the final answer
    only, which is a different question from what the conversation said.

    Storing canonical here would break both. A round carrying the public
    arguments would compare unequal to canonical ones and be called divergent
    although the worker did exactly as asked, and a rebuilt prompt would
    reinsert intermediate text the worker never continued from - handles and
    all, past the boundary that exists to keep them off the wire.

    Deeply copied on the way in and out. The dicts inside reach here from the
    ledger's `parent_state`, which every later attempt reads: a stage that
    edited a nested tool call in place would change what the next attempt is
    restored to. Frozen on the outside is not enough, the same way it was not
    enough for `CitationTable`.
    """

    operation_seq: int
    content: str = ""
    tool_calls: Tuple[Dict[str, Any], ...] = ()
    assistant_message: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "tool_calls", tuple(deepcopy(dict(c)) for c in self.tool_calls)
        )
        object.__setattr__(
            self, "assistant_message", deepcopy(self.assistant_message)
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "kind": "model_turn",
            "operation_seq": self.operation_seq,
            "content": self.content,
            "tool_calls": [deepcopy(dict(call)) for call in self.tool_calls],
            "assistant_message": deepcopy(self.assistant_message),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ModelTurn":
        return cls(
            operation_seq=int(raw["operation_seq"]),
            content=str(raw.get("content") or ""),
            tool_calls=tuple(dict(call) for call in raw.get("tool_calls") or []),
            assistant_message=raw.get("assistant_message"),
        )


Entry = Any  # ModelTurn | ToolRound


@dataclass
class TrustedTranscript:
    """What the parent did, in the order it did it.

    Append-only and deduped by operation sequence. A replay restores the entry
    an operation committed rather than running it again, and restoring the
    same operation twice - two attempts of one node, each replaying the
    ledger - must leave one entry, not two copies of one exchange.
    """

    entries: List[Entry] = field(default_factory=list)

    def record(self, entry: Entry) -> None:
        """Add one operation's entry, or replace the one already there.

        Replace rather than refuse: an operation that ran and then replayed
        produces the same entry, and a caller correcting an entry in place is
        not a case that exists - the sequence identifies the operation, and
        one operation has one outcome.
        """
        seq = entry.operation_seq
        for index, existing in enumerate(self.entries):
            if existing.operation_seq == seq and type(existing) is type(entry):
                self.entries[index] = entry
                return
        self.entries.append(entry)
        self.entries.sort(key=lambda item: item.operation_seq)

    def rounds(self) -> List[ToolRound]:
        return [entry for entry in self.entries if isinstance(entry, ToolRound)]

    def unanswered_turn(self) -> Optional[ModelTurn]:
        """The model turn a round arriving now would be answering, if any.

        The last entry, and only when it is a model turn. A model turn asks
        for one round: once a round has answered it, that turn is the entry
        before last and this returns nothing, so a second round claiming the
        same authority finds none.

        That matters because retrieval is not deterministic. A worker
        repeating a round verbatim would otherwise get a second set of
        grounded passages - different documents, possibly - carrying the
        authority of one request the model made once.
        """
        if not self.entries:
            return None
        last = self.entries[-1]
        return last if isinstance(last, ModelTurn) else None

    def without_trailing_answer(self) -> "TrustedTranscript":
        """This record, cut before a terminal answer the parent is replacing.

        One caller: the streamed final turn. The worker stops after its tool
        rounds and hands the conversation back for the parent to finish, and
        its loop deliberately leaves the last no-tool response out of that
        conversation - the parent is about to produce the answer, so repeating
        the draft would put it in the prompt it is being written from.

        The parent records that response anyway, because it happened. This is
        the difference between the two facts: the record still says the model
        produced a draft, and a reconstruction meant to replace it starts from
        the state immediately before.

        Structural, not textual. What is dropped is a trailing `ModelTurn`
        with no tool calls - the shape the worker breaks on - rather than
        anything matching the answer's words, which would be a guess about
        text the model wrote.

        Only trailing, and only that shape. A transcript ending in a
        `ToolRound` is a conversation waiting for its first answer and is
        returned whole; a model turn that asked for tools is an exchange the
        rounds after it depend on.

        A copy, so the caller cannot narrow the record by reading it.
        """
        entries = list(self.entries)
        if entries:
            last = entries[-1]
            if isinstance(last, ModelTurn) and not last.tool_calls:
                entries.pop()
        return TrustedTranscript(entries=entries)

    def as_list(self) -> List[Dict[str, Any]]:
        return [entry.as_dict() for entry in self.entries]

    def restore(self, raw: Sequence[Mapping[str, Any]]) -> None:
        for item in raw or []:
            kind = item.get("kind")
            if kind == "tool_round":
                self.record(ToolRound.from_dict(item))
            elif kind == "model_turn":
                self.record(ModelTurn.from_dict(item))


def calls_match(
    offered: Sequence[Mapping[str, Any]], submitted: Sequence[Mapping[str, Any]]
) -> bool:
    """Whether a round is the round the previous model turn asked for.

    Compared on name and arguments, in order, because those are what the
    parent will execute and what a reconstructed tool message describes. The
    ids are not compared: they are the provider's, they arrive through the
    worker, and a round that renamed them is still the same two calls.

    Arguments arrive as a JSON string from the model and as a parsed mapping
    from the worker, so the offered side is decoded before comparing. An
    offered argument string that will not parse compares unequal, which is the
    safe direction: a round the parent cannot read is a round it cannot say
    was faithfully carried out.
    """
    if len(offered) != len(submitted):
        return False
    for want, got in zip(offered, submitted):
        if str(want.get("name") or "") != str(got.get("name") or ""):
            return False
        raw = want.get("arguments")
        if isinstance(raw, str):
            try:
                decoded = json.loads(raw or "{}")
            except (TypeError, ValueError):
                return False
        else:
            decoded = raw or {}
        if not isinstance(decoded, dict):
            return False
        if decoded != dict(got.get("arguments") or {}):
            return False
    return True
