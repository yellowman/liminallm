"""What the model may cite, and what it actually cited.

The turn's registry says what was consulted; its bindings say what may support
this answer. Neither is a thing the model can be handed. `source_id` is minted
per registry and restarts at `src_1` every turn, so a handle built from it
would be forged by ordinary copying: yesterday's assistant message is replayed
verbatim into today's prompt, and yesterday's `[src_1]` is a different document
from today's. A retrieved page, note or earlier message can contain the string
just as easily, and every one of those reaches the model as data.

So the model is given a per-turn handle instead:

    [cite:K7Q2ABCD-1]

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
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from liminallm.service.provenance import Binding, ProvenanceError, SourceRegistry

#: The nonce alphabet, without the characters a reader or a small model
#: confuses: no O/0 and no I/1.
ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"

#: The namespace a hostile document has to guess. This is the number that
#: matters, not uniqueness: a retrieved page is attacker-controlled text of
#: some tens of kilobytes, so it can carry on the order of a thousand
#: candidate markers and pay nothing for a miss. At four characters the
#: namespace is about a million, which gives such a page roughly one chance in
#: a thousand of naming a source it never was - far too generous for the one
#: mechanism whose whole purpose is refusing source-authored citations.
#:
#: Eight characters is 2**40. The cost is four more characters in a token the
#: model copies, and the claim that a shorter one is easier for a weak model to
#: reproduce is a guess nobody has measured, while the loss in guess resistance
#: is arithmetic. `test_the_namespace_is_too_large_to_guess` pins the floor.
NONCE_LENGTH = 8
MIN_NONCE_BITS = 40

#: Anything the model meant as a citation. Deliberately broad, because
#: resolution is the gate and lexing is not: a marker whose handle this turn
#: did not issue is refused whether it was mistyped or invented, so matching
#: only the well-formed shape would buy no safety and would hide the mistyped
#: ones from the one pass that has to remove them.
#:
#: The keyword is matched case-insensitively for the same reason - `[CITE:x]`
#: is a typo that must still be taken back out - while the handle inside stays
#: exact, because that is the part resolution gates on.
#:
#: The shape of a real handle is defined where handles are minted, which is
#: the only place that can be authoritative about it.
#:
#: Bounded, and stopping at the first `]` or newline, so an unclosed `[cite:`
#: cannot swallow the rest of a sentence.
CITATION_RE = re.compile(r"\[(?i:cite):([^\]\n]{0,64})\]")

#: An empty table's mappings. Frozen like a built one's, so the default is not
#: the one writable `CitationTable` in the system. Behind a factory because
#: `dataclasses` refuses a mappingproxy as a bare default.
_EMPTY: Mapping[str, Any] = MappingProxyType({})


def mint_nonce() -> str:
    """One turn's citation namespace."""
    return "".join(secrets.choice(ALPHABET) for _ in range(NONCE_LENGTH))


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

    Deeply frozen, not merely a frozen dataclass. This is the object the next
    stage makes authority, and a frozen dataclass does not freeze what its
    attributes point at: plain dicts here would let anything holding a
    reference add a handle after the table was built, which is the whole
    conservation rule undone through a side door. S1 froze source metadata for
    this reason and the same applies here.
    """

    nonce: str
    #: handle -> source_id, and the reverse. Both directions are needed: the
    #: offer is built from sources and the validator resolves from handles.
    by_handle: Mapping[str, str] = field(default_factory=lambda: _EMPTY)
    by_source: Mapping[str, str] = field(default_factory=lambda: _EMPTY)
    #: source_id -> the evidence ids bound to it, in binding order. What the
    #: answer may rest on within that source, for whatever later checks a
    #: claim against a passage.
    evidence: Mapping[str, Tuple[str, ...]] = field(
        default_factory=lambda: _EMPTY
    )

    def __post_init__(self) -> None:
        """Enforce the type's own invariants, wherever it was built.

        The builder is not the only way to get one of these, and a rule kept
        there is a rule the constructor does not have. Both live here so there
        is one boundary rather than one convention: the mappings are copied
        and wrapped so a caller's dict cannot be written through afterwards,
        and the nonce is checked, so a namespace narrower than the floor
        cannot be supplied by a caller that the default mint would never have
        produced.
        """
        if len(self.nonce) != NONCE_LENGTH or any(
            character not in ALPHABET for character in self.nonce
        ):
            raise ProvenanceError(
                f"a citation nonce must be {NONCE_LENGTH} characters of "
                f"{ALPHABET!r}, got {self.nonce!r}"
            )
        object.__setattr__(self, "by_handle", MappingProxyType(dict(self.by_handle)))
        object.__setattr__(self, "by_source", MappingProxyType(dict(self.by_source)))
        object.__setattr__(
            self,
            "evidence",
            MappingProxyType(
                {key: tuple(value) for key, value in self.evidence.items()}
            ),
        )

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

    Both halves of every binding are resolved through the registry, and the
    relation between them is checked: the evidence has to exist and has to
    belong to the source named beside it. A binding naming a source the
    registry does not hold cannot be described to a reader - there is no
    title, kind or locator to show - and one pairing a real source with
    another source's passage would attach a citation to text that source
    never contained.

    Today's producers do not manufacture either shape, and this does not
    depend on them continuing not to. The gate that grants citation authority
    is the wrong place to inherit an upstream invariant.

    A source is not given a handle until it has one binding that passes, so a
    source whose only binding is malformed is uncitable rather than citable
    with nothing under it.
    """
    # `is None` rather than falsy: omitting the nonce asks for one, and
    # supplying an empty string is a caller naming a namespace. The second
    # is refused below rather than quietly replaced with a good one.
    token = mint_nonce() if nonce is None else nonce
    by_handle: Dict[str, str] = {}
    by_source: Dict[str, str] = {}
    evidence: Dict[str, List[str]] = {}
    # Two of these four checks cannot currently fire, and both are kept.
    # `add_evidence` refuses a source the registry does not hold, so no
    # evidence can name a missing one and the relation check below already
    # covers the source lookup; and `get_evidence("")` returns None, so the
    # empty-id guard is covered too. Each is redundant *because of an
    # invariant this function does not own*, which is the one place not to
    # rely on that: this is the gate that grants citation authority. They are
    # deliberately unkillable by mutation - recorded here rather than left for
    # a later reader to simplify away.
    for entry in bindings:
        source_id = entry.get("source_id")
        evidence_id = entry.get("evidence_id")
        if not source_id or not evidence_id:
            continue
        if registry.get_source(source_id) is None:
            continue
        record = registry.get_evidence(evidence_id)
        if record is None or record.source_id != source_id:
            continue
        if source_id not in by_source:
            handle = f"{token}-{len(by_source) + 1}"
            by_source[source_id] = handle
            by_handle[handle] = source_id
            evidence[source_id] = []
        if evidence_id not in evidence[source_id]:
            evidence[source_id].append(evidence_id)
    # Freezing and nonce validation belong to the type, so an explicitly
    # supplied narrow nonce is refused here by the same rule that refuses it
    # anywhere else.
    return CitationTable(
        nonce=token,
        by_handle=by_handle,
        by_source=by_source,
        evidence=evidence,
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

    Every closed marker-shaped token is removed, whether or not it resolves
    and whether or not it is well formed. An unclosed `[cite:` is left alone:
    there is no boundary at which deleting the rest of a sentence would be
    the safer guess. This is what runs when the
    markers must not reach a reader, and a mistyped one is exactly the kind
    that would otherwise be left behind. Spacing left by a removed marker is
    closed up, so a sentence does not end with a gap where a handle used to
    be.
    """
    return re.sub(r"[ \t]*" + CITATION_RE.pattern, "", answer or "")
