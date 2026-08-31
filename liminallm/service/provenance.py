"""One answer to "where did this claim come from?".

A turn gathers evidence from several places - the web, a file in a knowledge
context, one of the reader's notes, an earlier message, an MCP server - and
today each of those arrives in its own shape and is flattened into text
before the model sees it. Provenance dies at that flattening: the answer can
say a thing, but nothing downstream can say which retrieved thing supports
it.

This module is the common vocabulary those producers will share. It holds
three records and a registry, and deliberately nothing else: no storage, no
API schema, no retrieval, no prompt text. Nothing in the application imports
it yet. The shape comes first so that four producers cannot each invent
their own dialect of it.

Two distinctions are built in rather than left to callers.

A knowledge context is not a source. It is a retrieval scope - a corpus you
search - and the thing worth citing is the file or chunk inside it. So a
context appears as `metadata["context_id"]` on a `kind="file"` source, never
as a kind of its own.

A note is not a file. Notes carry authorship, chronology and links between
them, which a chunk of an uploaded document does not, so they keep their own
kind and their own identity. A markdown file uploaded into a context stays
`kind="file"`; the producer says which it is, because only the producer
knows.

The registry, not the producer, assigns identity. A tool worker handles
untrusted text and must never be in a position to claim "this is src_7" -
it hands evidence to the trusted parent and receives an id back.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Literal, Optional, Tuple

#: Where a piece of evidence came from. Deliberately short: a kind exists
#: only where a producer can state it as a fact.
#:
#: `unknown` is the explicit neutral. Provenance that cannot be established
#: is recorded as unknown and stays there - it is never promoted to `web` or
#: `note` by reading a file extension, a tool name or a title. Guessing the
#: kind is how a reader ends up told that their uploaded `manual.md` is a
#: note they wrote.
SourceKind = Literal["web", "file", "note", "conversation", "mcp", "unknown"]

_KINDS: Tuple[str, ...] = ("web", "file", "note", "conversation", "mcp", "unknown")


class ProvenanceError(ValueError):
    """A source or evidence record that cannot be trusted as stated."""


@dataclass(frozen=True)
class Source:
    """A thing evidence came from, as the producer can honestly describe it.

    `origin_id` is the identity the source system already has - a note id, a
    file generation id, a message id. It is what makes two retrievals of the
    same thing the same source. `locator` is where to open it: a URL, a path.
    One or both may be absent; a source with neither can still exist, and is
    simply never merged with another.

    `metadata` is a bounded JSON bag rather than twenty nullable columns.
    What web retrieval and context retrieval can honestly populate is not yet
    known - that is S2's question - and inventing the fields before the
    producers exist would fix the wrong answer in a dataclass.
    """

    source_id: str
    kind: SourceKind
    title: str
    origin_id: Optional[str] = None
    locator: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvidenceLocator:
    """Where inside a source the evidence sits.

    Every field is optional because the producers point at different things:
    a chunk index for a RAG hit, a page for a PDF, a character range for a
    span of text, a block id for a parsed document. A locator with nothing
    set is legitimate - it means "this source, no finer".
    """

    block_id: Optional[str] = None
    chunk_id: Optional[str] = None
    chunk_index: Optional[int] = None
    page: Optional[int] = None
    section: Optional[str] = None
    start: Optional[int] = None
    end: Optional[int] = None


@dataclass(frozen=True)
class Evidence:
    """The text a claim can rest on, and where in its source it was found.

    There is no score here on purpose. A BM25 score, a cosine similarity, a
    fused rank, a search-result position and an eventual support score are
    different quantities that happen to be numbers. One field named `score`
    would eventually be used to compare two of them, and the comparison would
    look reasonable. Retrieval diagnostics belong in metadata until they have
    a name of their own.
    """

    evidence_id: str
    source_id: str
    text: str
    locator: EvidenceLocator
    content_hash: str


def _json_safe(value: Any, *, what: str) -> Any:
    """Reject anything the snapshot could not round-trip.

    Checked when it is registered rather than when it is exported: a bad
    value is then attributable to the producer that supplied it, instead of
    surfacing much later as a failure to serialise the whole turn.
    """
    try:
        return json.loads(json.dumps(value))
    except (TypeError, ValueError) as exc:
        raise ProvenanceError(f"{what} must be JSON-safe: {exc}") from exc


def _canonical_locator(locator: str) -> str:
    """Just enough normalising to match the same locator written twice.

    Scheme and host are case-insensitive per RFC 3986, so those are folded.
    Nothing else is: stripping a trailing slash or a query string looks
    tidy and merges pages that are genuinely different, which would attach a
    citation to the wrong source. Under-merging leaves a duplicate in the
    list; over-merging misattributes a claim.
    """
    text = locator.strip()
    for scheme in ("http://", "https://"):
        if text[: len(scheme)].lower() == scheme:
            rest = text[len(scheme):]
            host, slash, tail = rest.partition("/")
            return f"{scheme}{host.lower()}{slash}{tail}"
    return text


class SourceRegistry:
    """The turn's own record of what it retrieved.

    One registry per turn, held by the caller. There is no module-level
    instance and no process-wide default, because a registry is turn-local
    authority: ids mean something only inside the turn that assigned them,
    and a shared one would let a later turn's `src_3` collide with an earlier
    turn's citation.
    """

    def __init__(self) -> None:
        self._sources: Dict[str, Source] = {}
        self._evidence: Dict[str, Evidence] = {}
        #: identity -> source_id, for the dedupe described on `register_source`
        self._by_identity: Dict[Tuple[str, str], str] = {}
        #: (source_id, locator, content_hash) -> evidence_id
        self._by_evidence: Dict[Tuple[str, str, str], str] = {}

    # -- reading -----------------------------------------------------------

    @property
    def sources(self) -> Tuple[Source, ...]:
        """Every source, in the order it was first registered."""
        return tuple(self._sources.values())

    @property
    def evidence(self) -> Tuple[Evidence, ...]:
        """Every piece of evidence, in the order it was first added."""
        return tuple(self._evidence.values())

    def get_source(self, source_id: str) -> Optional[Source]:
        return self._sources.get(source_id)

    def get_evidence(self, evidence_id: str) -> Optional[Evidence]:
        return self._evidence.get(evidence_id)

    def evidence_for(self, source_id: str) -> Tuple[Evidence, ...]:
        return tuple(e for e in self._evidence.values() if e.source_id == source_id)

    # -- writing -----------------------------------------------------------

    def register_source(
        self,
        *,
        kind: SourceKind,
        title: str,
        origin_id: Optional[str] = None,
        locator: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Source:
        """Record a source and return it with its turn-local id.

        Registering the same thing twice returns the first registration
        rather than a second id, so a source retrieved by two different
        routes is one entry with one identity. Two retrievals are the same
        thing when they share `(kind, origin_id)`, or `(kind, canonical
        locator)` where there is no origin id. A source with neither is
        always distinct - there is nothing to match it on, and inventing a
        match from the title would merge two unrelated documents that happen
        to share a name.

        `kind` is required and never inferred. The producer knows whether it
        searched the web or read a note; nothing downstream can recover that
        from a path.
        """
        if kind not in _KINDS:
            raise ProvenanceError(
                f"unknown source kind {kind!r}; expected one of {', '.join(_KINDS)}"
            )

        identity: Optional[Tuple[str, str]] = None
        if origin_id:
            identity = (kind, f"origin:{origin_id}")
        elif locator:
            identity = (kind, f"locator:{_canonical_locator(locator)}")

        if identity is not None:
            existing_id = self._by_identity.get(identity)
            if existing_id is not None:
                return self._sources[existing_id]

        # The same identity under a second kind is a producer disagreeing
        # with itself about what a thing is. Merging them would pick one
        # silently; refusing says which two claims conflict.
        if origin_id:
            for (other_kind, key), source_id in self._by_identity.items():
                if key == f"origin:{origin_id}" and other_kind != kind:
                    raise ProvenanceError(
                        f"origin {origin_id!r} is already registered as "
                        f"{other_kind!r} ({source_id}); it cannot also be {kind!r}"
                    )

        source = Source(
            source_id=f"src_{len(self._sources) + 1}",
            kind=kind,
            title=title,
            origin_id=origin_id,
            locator=locator,
            metadata=_json_safe(dict(metadata or {}), what="source metadata"),
        )
        self._sources[source.source_id] = source
        if identity is not None:
            self._by_identity[identity] = source.source_id
        return source

    def add_evidence(
        self,
        source_id: str,
        *,
        text: str,
        locator: Optional[EvidenceLocator] = None,
    ) -> Evidence:
        """Attach a passage to a source and return it with its id.

        The hash is computed here from the text, never accepted from the
        caller: it is what a stored citation is checked against later, so a
        producer that could supply it could also supply one that does not
        match what it handed over.

        The same passage at the same place in the same source is one piece of
        evidence however many times it is offered.
        """
        if source_id not in self._sources:
            raise ProvenanceError(f"no such source {source_id!r} in this turn")

        where = locator or EvidenceLocator()
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        key = (source_id, json.dumps(asdict(where), sort_keys=True), content_hash)
        existing_id = self._by_evidence.get(key)
        if existing_id is not None:
            return self._evidence[existing_id]

        record = Evidence(
            evidence_id=f"ev_{len(self._evidence) + 1}",
            source_id=source_id,
            text=text,
            locator=where,
            content_hash=content_hash,
        )
        self._evidence[record.evidence_id] = record
        self._by_evidence[key] = record.evidence_id
        return record

    # -- export ------------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        """The registry as plain JSON-safe data, in registration order.

        Sources are a mapping so a citation can name one in a few bytes;
        evidence is a list because its order is the order it was found.
        """
        return {
            "sources": {
                source_id: asdict(source)
                for source_id, source in self._sources.items()
            },
            "evidence": [asdict(record) for record in self._evidence.values()],
        }
