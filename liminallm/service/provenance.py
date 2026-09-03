"""One answer to "where did this claim come from?".

A turn gathers evidence from several places - the web, a file in a knowledge
context, one of the reader's notes, an earlier message, an MCP server - and
today each of those arrives in its own shape and is flattened into text
before the model sees it. Provenance dies at that flattening: the answer can
say a thing, but nothing downstream can say which retrieved thing supports
it.

This module is the common vocabulary those producers share. It holds three
records and a registry, and deliberately nothing else: no storage, no API
schema, no retrieval, no prompt text.

Automatic knowledge-context grounding is the first adopter. Explicit
`file_search`, web search, web fetch, notes and MCP still flatten their
results and have yet to migrate, which is why the shape came first: so those
producers cannot each invent their own dialect of it.

Two distinctions are built in rather than left to callers.

A knowledge context is not a source. It is a retrieval scope - a corpus you
search - and the thing worth citing is the file or chunk inside it, so a
context is never a kind of its own. Nor does it belong *on* a source: one
file can be described by several contexts, and registration is first-wins,
so a `context_id` field would freeze whichever context happened to retrieve
it first and read as if the document belonged to that one. Which scope found
what is a relation between a context and a piece of evidence, and producers
keep it beside the registry rather than inside it.

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
import threading
from dataclasses import asdict, dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Dict, Literal, Mapping, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit

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

#: A ceiling on the metadata bag, because it crosses worker, API and storage
#: seams and "bounded" has to be a number rather than a word in a docstring.
#: Provenance metadata is a handful of short strings and integers; 16 KiB is
#: already far more room than any of the producers should need, so hitting
#: this means something is being stored here that belongs elsewhere.
MAX_METADATA_BYTES = 16 * 1024


class ProvenanceError(ValueError):
    """A source or evidence record that cannot be trusted as stated."""


#: One relation the turn may cite, as `binding` builds it. `context_id` is
#: absent for every producer that did not retrieve through a knowledge
#: context, which is why the values are optional.
Binding = Dict[str, Optional[str]]


def binding(
    source_id: str,
    evidence_id: str,
    *,
    context_id: Optional[str] = None,
) -> Binding:
    """One relation this turn may cite, as every producer states it.

    The registry is what the turn consulted; a binding is what may support
    its answer. Five producers now build these - context retrieval, explicit
    file search, the web, the vault and remote tools - and the parent dedupes
    them on the whole triple, so the shape is agreed here rather than spelled
    out five times and drifting in one of them.

    `context_id` is the knowledge context a retrieval came through, and it is
    absent for everything that was not retrieved through one. A web page was
    not found by a context, and saying it was would put the document inside a
    scope it never belonged to.
    """
    return {
        "context_id": context_id,
        "source_id": _require_text(source_id, what="source_id"),
        "evidence_id": _require_text(evidence_id, what="evidence_id"),
    }


@dataclass(frozen=True)
class GroundedSpan:
    """Which run of model-visible text came from which piece of evidence.

    A binding says the answer may rest on some evidence. This says *where* in
    what the model was shown that evidence appears, which is what a later
    stage needs to attach a citation handle to the right passage rather than
    to the whole tool result.

    `start` and `end` index the finished string a producer returns - after
    neutralization and after the untrusted-data envelope, because that is the
    string that reaches the model. Built while the text is being assembled,
    never recovered afterwards by matching evidence text back against it: two
    results can share a snippet, one can be truncated, and a search that
    dropped a result without a URL breaks the positional correspondence
    between what was rendered and what was recorded.
    """

    start: int
    end: int
    source_id: str
    evidence_id: str


@dataclass(frozen=True)
class GroundedPassage:
    """One producer's finished text, and where its evidence sits inside it.

    The text is kept beside the spans, not just the spans, because the spans
    only mean anything against the exact string they were measured in. Two
    searches in one turn each produce offsets starting near zero, and a later
    stage looking at an assembled prompt has to know which string a span of
    `12..48` belongs to.

    It is also what lets that stage refuse. The worker returns tool text and
    can return anything; a passage recorded here is what the parent actually
    produced, so labelling is done by matching the parent's own string rather
    than by trusting the copy that came back.
    """

    text: str
    spans: Tuple[GroundedSpan, ...] = ()

    def as_dict(self) -> Dict[str, Any]:
        """JSON-safe, for the ledger sidecar a replay reads back."""
        return {
            "text": self.text,
            "spans": [asdict(span) for span in self.spans],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "GroundedPassage":
        return cls(
            text=str(raw.get("text") or ""),
            spans=tuple(
                GroundedSpan(
                    start=int(span["start"]),
                    end=int(span["end"]),
                    source_id=str(span["source_id"]),
                    evidence_id=str(span["evidence_id"]),
                )
                for span in raw.get("spans") or []
            ),
        )


class GroundedText:
    """Model-visible text assembled together with what grounds each part.

    Producers build a result out of pieces - one per search result, chunk,
    note or remote item - and only some of those pieces are evidence. Adding
    them here rather than joining strings keeps the association exact, since
    the offsets are measured as the string is built.

    Neutralization happens per piece, on the way in, so a piece's offsets are
    its offsets in the final text. `render` then checks that the assembled
    whole is unchanged by another pass. That check is what makes per-piece
    neutralization safe: the tool-call pattern tolerates whitespace, so `<`
    ending one piece and `tool_call>` beginning the next would form a tag that
    no single piece contained. When the check fails the whole text is
    neutralized as one and the spans are dropped - a render nothing can cite
    rather than one whose offsets point at the wrong words.
    """

    def __init__(self) -> None:
        self._pieces: list[str] = []
        self._spans: list[GroundedSpan] = []
        self._length = 0

    def add(self, text: str, ground: Optional[Binding] = None) -> None:
        """One piece of the result, and the evidence it came from if any."""
        from liminallm.service.web import neutralize_markers

        safe = neutralize_markers(text)
        if ground is not None:
            source_id = ground.get("source_id")
            evidence_id = ground.get("evidence_id")
            if source_id and evidence_id:
                self._spans.append(
                    GroundedSpan(
                        start=self._length,
                        end=self._length + len(safe),
                        source_id=source_id,
                        evidence_id=evidence_id,
                    )
                )
        self._pieces.append(safe)
        self._length += len(safe)

    def render(
        self, wrap: Optional[Callable[[str], str]] = None
    ) -> Tuple[str, Tuple[GroundedSpan, ...]]:
        """The finished text and the spans into it.

        `wrap` puts the envelope round the assembled text. It is applied here
        rather than by the caller because the spans have to move with it, and
        a caller that wrapped afterwards would leave every offset short by the
        length of the header.
        """
        from liminallm.service.web import neutralize_markers

        body = "".join(self._pieces)
        spans = tuple(self._spans)
        if neutralize_markers(body) != body:
            # A control token formed across a seam. Neutralize the whole and
            # keep nothing: the offsets described the text before this pass.
            body = neutralize_markers(body)
            spans = ()
        if wrap is None:
            return body, spans
        wrapped = wrap(body)
        at = wrapped.find(body)
        if at < 0:
            # The envelope rewrote the body, so the offsets no longer describe
            # it. Same choice as above, for the same reason.
            return wrapped, ()
        return wrapped, tuple(
            GroundedSpan(
                start=span.start + at,
                end=span.end + at,
                source_id=span.source_id,
                evidence_id=span.evidence_id,
            )
            for span in spans
        )


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
    #: Frozen at registration; see `_freeze`. Typed as a mapping because
    #: it is read like one, but it cannot be written through.
    metadata: Mapping[str, Any] = field(default_factory=dict)


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


def _require_text(value: Any, *, what: str) -> str:
    """Type annotations enforce nothing at runtime.

    Without this the common fields are only as sound as the caller, and a
    `title=object()` reaches the registry and fails much later as an
    unserialisable turn.
    """
    if not isinstance(value, str):
        raise ProvenanceError(
            f"{what} must be a string, got {type(value).__name__}"
        )
    return value


def _optional_text(value: Any, *, what: str) -> Optional[str]:
    return None if value is None else _require_text(value, what=what)


def _optional_int(value: Any, *, what: str) -> Optional[int]:
    # `bool` is an `int` in Python and is never a page or an offset.
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProvenanceError(
            f"{what} must be an integer, got {type(value).__name__}"
        )
    return value


def _checked_locator(locator: Any) -> "EvidenceLocator":
    """Every field is the type it claims, so the snapshot cannot break.

    The object before its fields: a dict has no `.block_id`, so passing one
    raised `AttributeError` at the first field access, before any of the
    checks below could run and say what was wrong.
    """
    if not isinstance(locator, EvidenceLocator):
        raise ProvenanceError(
            f"locator must be an EvidenceLocator, got {type(locator).__name__}"
        )
    return EvidenceLocator(
        block_id=_optional_text(locator.block_id, what="locator block_id"),
        chunk_id=_optional_text(locator.chunk_id, what="locator chunk_id"),
        chunk_index=_optional_int(locator.chunk_index, what="locator chunk_index"),
        page=_optional_int(locator.page, what="locator page"),
        section=_optional_text(locator.section, what="locator section"),
        start=_optional_int(locator.start, what="locator start"),
        end=_optional_int(locator.end, what="locator end"),
    )


def _checked_metadata(metadata: Any) -> Dict[str, Any]:
    """The bag itself, not only what is inside it.

    `dict(metadata or {})` took a list as an empty bag and recorded a source
    the caller never described, and let a string fail as a bare `ValueError`
    raised from inside the dict constructor.
    """
    if metadata is None:
        return {}
    if not isinstance(metadata, Mapping):
        raise ProvenanceError(
            f"source metadata must be a mapping, got {type(metadata).__name__}"
        )
    _require_string_keys(metadata)
    return dict(metadata)


def _require_string_keys(value: Any) -> None:
    """JSON has string keys and no others.

    `json.dumps` renames the rest rather than refusing them, so `{1: "x"}` is
    read back as `{"1": "x"}` and the metadata a source carries is not the
    metadata its producer handed over. Checked at every depth, because the
    rename is exactly as quiet inside a nested bag.
    """
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ProvenanceError(
                    "source metadata keys must be strings, got "
                    f"{type(key).__name__} ({key!r})"
                )
            _require_string_keys(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _require_string_keys(item)


def _freeze(value: Any) -> Any:
    """Make a JSON tree that cannot be edited after it is registered.

    A frozen dataclass freezes its own attributes, not what they point at, so
    a plain dict here would hand every caller a live handle on the registry's
    authoritative record. These become citation authority: what a stored
    citation appears to say must not be changeable by whoever holds a
    reference to the source.
    """
    if isinstance(value, Mapping):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


def _plain(value: Any) -> Any:
    """The inverse, for export: back to the dicts and lists JSON knows."""
    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_plain(v) for v in value]
    return value


def _json_safe(value: Any, *, what: str) -> Any:
    """Reject anything the snapshot could not round-trip.

    Checked when it is registered rather than when it is exported: a bad
    value is then attributable to the producer that supplied it, instead of
    surfacing much later as a failure to serialise the whole turn.

    `allow_nan=False` because Python's encoder writes `NaN` and `Infinity`,
    which JSON has no literals for. A strict reader at the far end of the API
    or storage seam rejects the whole snapshot over one of them, long after
    the producer that supplied it is gone.
    """
    try:
        encoded = json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ProvenanceError(f"{what} must be JSON-safe: {exc}") from exc
    measured = len(encoded.encode("utf-8"))
    if measured > MAX_METADATA_BYTES:
        raise ProvenanceError(
            f"{what} is too large: {measured} bytes serialised exceeds "
            f"the {MAX_METADATA_BYTES} byte ceiling"
        )
    return json.loads(encoded)


def _canonical_locator(locator: str) -> str:
    """Just enough normalising to match the same locator written twice.

    Scheme and host are case-insensitive per RFC 3986, so those fold and
    nothing else does. A path, a query string and a fragment are somebody's
    identifiers - a token, a signature, a base64 id - and folding their case
    merges two URLs that address different things. Under-merging leaves a
    duplicate in a list; over-merging attaches a claim to the wrong document,
    so the safe side is the narrow one.

    Split structurally rather than by hand. Finding the host with
    `partition("/")` swallowed the whole of `example.com?Token=ABC` when a
    URL had no path, and lowercased the query with it.

    Anything that is not http(s) is returned byte for byte, including
    surrounding whitespace: a leading space is a legal character in a
    filename, so trimming it merges two different files.
    """
    if locator[:8].lower() not in ("https://",) and not locator[:7].lower() == "http://":
        return locator

    parts = urlsplit(locator)
    netloc = parts.netloc
    # `urlsplit` already lowercases `hostname`; put that back over the host as
    # it was written, which leaves userinfo, port and IPv6 brackets alone.
    host = parts.hostname
    if host:
        at = netloc.rfind("@") + 1
        index = netloc.lower().find(host, at)
        if index != -1:
            netloc = netloc[:index] + host + netloc[index + len(host):]
    return urlunsplit(
        (parts.scheme.lower(), netloc, parts.path, parts.query, parts.fragment)
    )


class SourceRegistry:
    """The turn's own record of what its answer rests on.

    Not everything retrieval offered. A producer registers what actually
    reached the model - for the automatic path, the evidence that survived
    prompt budgeting - because a chunk the pruner dropped never grounded
    anything. Attempts that reached the grounding stage and then failed do
    leave their evidence here, so the registry is the turn's *consulted*
    superset; which of it supports the answer is a relation the producer
    returns and the caller keeps.

    One registry per turn, held by the caller. There is no module-level
    instance and no process-wide default, because a registry is turn-local
    authority: ids mean something only inside the turn that assigned them,
    and a shared one would let a later turn's `src_3` collide with an earlier
    turn's citation.

    One turn is not one thread. A workflow runs child nodes concurrently and
    each child can retrieve, so registration into the turn's registry is
    genuinely parallel. Ids are derived from the number of records held, and
    dedupe is a read followed by a write, so both need the lock: without it
    two children reading the same count mint the same id and one record
    overwrites the other, and two children registering the same document
    both miss the dedupe and register it twice.
    """

    def __init__(self) -> None:
        # Reentrant so a future helper can hold the lock across two of these
        # calls without deadlocking on itself. Nothing nests today.
        self._lock = threading.RLock()
        self._sources: Dict[str, Source] = {}
        self._evidence: Dict[str, Evidence] = {}
        #: identity -> source_id, for the dedupe described on `register_source`
        self._by_identity: Dict[Tuple[str, str], str] = {}
        #: (source_id, locator, content_hash) -> evidence_id
        self._by_evidence: Dict[Tuple[str, str, str], str] = {}

    # -- reading -----------------------------------------------------------

    # Every read that walks a mapping takes the lock. A sibling node
    # registering mid-iteration would otherwise raise `dictionary changed
    # size during iteration` in a caller that only wanted to look.

    @property
    def sources(self) -> Tuple[Source, ...]:
        """Every source, in the order it was first registered."""
        with self._lock:
            return tuple(self._sources.values())

    @property
    def evidence(self) -> Tuple[Evidence, ...]:
        """Every piece of evidence, in the order it was first added."""
        with self._lock:
            return tuple(self._evidence.values())

    def get_source(self, source_id: str) -> Optional[Source]:
        return self._sources.get(source_id)

    def get_evidence(self, evidence_id: str) -> Optional[Evidence]:
        return self._evidence.get(evidence_id)

    def evidence_for(self, source_id: str) -> Tuple[Evidence, ...]:
        with self._lock:
            return tuple(
                e for e in self._evidence.values() if e.source_id == source_id
            )

    # -- writing -----------------------------------------------------------

    def register_source(
        self,
        *,
        kind: SourceKind,
        title: str,
        origin_id: Optional[str] = None,
        locator: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
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
        title = _require_text(title, what="title")
        origin_id = _optional_text(origin_id, what="origin_id")
        locator = _optional_text(locator, what="locator")

        # One key per identity, whichever kind of identity it is, so the
        # checks below cover both. Keeping them apart is how the cross-kind
        # guard came to protect origins and not locators.
        key: Optional[str] = None
        if origin_id:
            key = f"origin:{origin_id}"
        elif locator:
            key = f"locator:{_canonical_locator(locator)}"

        # Frozen, and a copy: the caller keeps no handle on the registry's
        # record, and mutating what they passed in afterwards reaches nothing.
        # Outside the lock: it touches only the caller's own data, and it is
        # the widest part of the call to hold a shared lock across.
        checked = _json_safe(_checked_metadata(metadata), what="source metadata")

        # From the dedupe read to the two writes is one critical section.
        # Split, two children of one turn retrieving the same document both
        # miss the dedupe, and two retrieving different ones read the same
        # count, mint the same id, and lose a record to the overwrite.
        with self._lock:
            if key is not None:
                existing_id = self._by_identity.get((kind, key))
                if existing_id is not None:
                    return self._sources[existing_id]

                # The same identity under a second kind is a producer
                # disagreeing with itself about what a thing is. Merging would
                # pick one silently; refusing names the two claims that
                # conflict.
                for (other_kind, other_key), source_id in self._by_identity.items():
                    if other_key == key and other_kind != kind:
                        raise ProvenanceError(
                            f"{key} is already registered as {other_kind!r} "
                            f"({source_id}); it cannot also be {kind!r}"
                        )

            source = Source(
                source_id=f"src_{len(self._sources) + 1}",
                kind=kind,
                title=title,
                origin_id=origin_id,
                locator=locator,
                metadata=_freeze(checked),
            )
            self._sources[source.source_id] = source
            if key is not None:
                self._by_identity[(kind, key)] = source.source_id
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
        # Typed first: an unhashable id fails the membership test itself with
        # a raw `TypeError`, before the check below can report it.
        source_id = _require_text(source_id, what="source_id")
        text = _require_text(text, what="evidence text")
        # `is None`, not `or`: an empty dict is falsy, and would have been
        # replaced by an empty locator instead of refused as the wrong type.
        where = _checked_locator(
            EvidenceLocator() if locator is None else locator
        )
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        key = (source_id, json.dumps(asdict(where), sort_keys=True), content_hash)

        with self._lock:
            if source_id not in self._sources:
                raise ProvenanceError(f"no such source {source_id!r} in this turn")

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

        Taken under the lock, so an export is one coherent picture of the
        turn rather than sources from before a sibling's registration and
        evidence from after it.
        """
        with self._lock:
            return {
                "sources": {
                    source_id: {
                        "source_id": source.source_id,
                        "kind": source.kind,
                        "title": source.title,
                        "origin_id": source.origin_id,
                        "locator": source.locator,
                        # Built by hand rather than by `asdict`, which would
                        # carry the frozen mappings straight into the export.
                        "metadata": _plain(source.metadata),
                    }
                    for source_id, source in self._sources.items()
                },
                "evidence": [
                    {
                        "evidence_id": record.evidence_id,
                        "source_id": record.source_id,
                        "text": record.text,
                        "locator": asdict(record.locator),
                        "content_hash": record.content_hash,
                    }
                    for record in self._evidence.values()
                ],
            }
