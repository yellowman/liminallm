"""What the model may cite, and what it actually cited.

The turn's registry says what was consulted; its bindings say what may support
this answer. Neither is a thing the model can be handed. `source_id` is minted
per registry and restarts at `src_1` every turn, so a handle built from it
would be forged by ordinary copying: yesterday's assistant message is replayed
verbatim into today's prompt, and yesterday's `[src_1]` is a different document
from today's. A retrieved page, note or earlier message can contain the string
just as easily, and every one of those reaches the model as data.

So the model is given a per-turn handle instead:

    [cite:K7Q2-1]

The nonce is minted once per turn and the mapping back to `src_#` stays here,
parent-side. A handle from another turn does not resolve, and neither does one
a source wrote, because neither could know this turn's nonce. `src_#` remains
the internal authority; the handle is only how the model names it.

What that does and does not buy, stated plainly: a wrong guess resolves to
nothing and is dropped, and the nonce is not visible before the turn that
mints it. A correct guess would misattribute one span among the sources this
turn already grounded on - handles exist only for those - rather than reach
anything the turn did not read.
"""

from __future__ import annotations

import re
import secrets
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from liminallm.service.provenance import Binding, SourceRegistry

#: The nonce alphabet, without the characters a reader or a small model
#: confuses: no O/0 and no I/1. Four of these is about a million turns-worth of
#: distinct nonces, and short enough that a weak model copies it back intact -
#: which is the failure that matters most here, since a mangled handle is a
#: citation the answer loses rather than one it forges.
_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
NONCE_LENGTH = 4

#: Anything the model meant as a citation. Deliberately broad, because
#: resolution is the gate and lexing is not: a marker whose handle this turn
#: did not issue is refused whether it was mistyped or invented, so matching
#: only the well-formed shape would buy no safety and would hide the mistyped
#: ones from the one pass that has to remove them.
#:
#: The shape of a real handle is defined where handles are minted, which is
#: the only place that can be authoritative about it.
#:
#: Bounded, and stopping at the first `]` or newline, so an unclosed `[cite:`
#: cannot swallow the rest of a sentence.
CITATION_RE = re.compile(r"\[cite:([^\]\n]{0,64})\]")


def mint_nonce() -> str:
    """One turn's citation namespace."""
    return "".join(secrets.choice(_ALPHABET) for _ in range(NONCE_LENGTH))


@dataclass(frozen=True)
class CitationOccurrence:
    """One citation marker the model wrote, and where it wrote it.

    `start` and `end` are the marker's own span in the answer, not the span of
    the claim it supports. Which words a citation covers is a question about
    the prose, and answering it by guessing would put a boundary in the record
    that nothing measured. The marker's position is what this stage knows.

    Occurrences are a list rather than a set on purpose: one source cited in
    two places is one source and two citations, and collapsing them would lose
    the second position before anything could render or persist it.
    """

    handle: str
    source_id: str
    start: int
    end: int


@dataclass(frozen=True)
class CitationTable:
    """This turn's citable sources, by the name the model is given for them.

    Built from bindings and not from the registry: the registry is everything
    the turn consulted, including what the prompt budget dropped and what a
    failed node retrieved. Only what may support the answer gets a handle, so
    a source with no handle is not citable however well the model describes it.

    One handle per source, so two routes to one deduped source - a context
    retrieval and an explicit search reaching the same file - share one
    citation identity rather than inviting the model to cite the same document
    twice under two names.
    """

    nonce: str
    #: handle -> source_id, and the reverse. Both directions are needed: the
    #: offer is built from sources and the validator resolves from handles.
    by_handle: Mapping[str, str] = field(default_factory=dict)
    by_source: Mapping[str, str] = field(default_factory=dict)
    #: source_id -> the evidence ids bound to it, in binding order. What the
    #: answer may rest on within that source, for whatever later checks a
    #: claim against a passage.
    evidence: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)

    def source_for(self, handle: str) -> Optional[str]:
        return self.by_handle.get(handle)

    def handle_for(self, source_id: str) -> Optional[str]:
        return self.by_source.get(source_id)

    def evidence_for(self, source_id: str) -> Tuple[str, ...]:
        return self.evidence.get(source_id, ())

    def __bool__(self) -> bool:
        return bool(self.by_handle)


def build_citation_table(
    registry: SourceRegistry,
    bindings: Sequence[Binding],
    *,
    nonce: Optional[str] = None,
) -> CitationTable:
    """The handles this turn may offer, from what actually grounded it.

    Every binding is resolved through the registry rather than trusted as a
    pair of strings. A binding naming a source the registry does not hold
    cannot be described to a reader - there is no title, kind or locator to
    show - so it gets no handle and becomes uncitable. That is the safe
    direction: the alternative is a citation that resolves to nothing at the
    point someone tries to follow it.
    """
    token = nonce or mint_nonce()
    by_handle: Dict[str, str] = {}
    by_source: Dict[str, str] = {}
    evidence: Dict[str, List[str]] = {}
    for entry in bindings:
        source_id = entry.get("source_id")
        evidence_id = entry.get("evidence_id")
        if not source_id or registry.get_source(source_id) is None:
            continue
        if source_id not in by_source:
            handle = f"{token}-{len(by_source) + 1}"
            by_source[source_id] = handle
            by_handle[handle] = source_id
            evidence[source_id] = []
        if evidence_id and evidence_id not in evidence[source_id]:
            evidence[source_id].append(evidence_id)
    return CitationTable(
        nonce=token,
        by_handle=dict(by_handle),
        by_source=dict(by_source),
        evidence={key: tuple(value) for key, value in evidence.items()},
    )


def validate_citations(answer: str, table: CitationTable) -> List[CitationOccurrence]:
    """The citations in this answer that this turn actually issued.

    In the order they appear, so a renderer can walk the text once. Anything
    that does not resolve is dropped rather than repaired into a neighbour:
    a marker naming a source this turn did not ground on is not evidence of
    anything, and guessing which one was meant would invent the relation the
    whole layer exists to stop being invented.

    Prose with no citation is not an error here. Whether an answer cites
    enough is a different question from whether what it cited is real.
    """
    found: List[CitationOccurrence] = []
    for match in CITATION_RE.finditer(answer or ""):
        source_id = table.source_for(match.group(1))
        if source_id is None:
            continue
        found.append(
            CitationOccurrence(
                handle=match.group(1),
                source_id=source_id,
                start=match.start(),
                end=match.end(),
            )
        )
    return found


def strip_citations(answer: str) -> str:
    """The answer with every citation marker removed.

    Marker-shaped text is removed whether or not it resolves, and whether or
    not it is well formed. This is what runs when the
    markers must not reach a reader, and a mistyped one is exactly the kind
    that would otherwise be left behind. Spacing left by a removed marker is
    closed up, so a sentence does not end with a gap where a handle used to
    be.
    """
    return re.sub(r"[ \t]*" + CITATION_RE.pattern, "", answer or "")
