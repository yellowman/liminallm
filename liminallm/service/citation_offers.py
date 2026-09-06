"""Showing the model which passage is which.

`citations.py` owns identity: which sources this turn may cite, what each is
called, and whether a marker in an answer is one this turn issued. This module
owns the other half - putting those names into the text the model reads, so
that copying one back is a thing it can do.

The two are kept apart because they fail differently. An identity mistake
grants authority that was never earned. A materialization mistake shows the
model a marker in the wrong place, and the model then cites the wrong source
correctly. Both are wrong; only the first is a hole in the gate, and mixing
prompt mechanics into the gate is how a reader stops being able to see which
one they are reading.

Nothing here commits citation authority. `choose_offers` does allocate handles
- it extends a table speculatively, which is how a candidate prompt can be
rendered and priced at all - but every such table is thrown away unless its
caller commits the relations that came back. A source whose marker never
reached the model comes out of this module with nothing.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from liminallm.service.citations import CitationTable, extend_citation_table
from liminallm.service.provenance import (
    Binding,
    GroundedMessage,
    GroundedSpan,
    ProvenanceError,
    SourceRegistry,
)
from liminallm.service.transcript import ModelTurn, ToolRound, TrustedTranscript

#: What the model is told about the markers it is being shown.
#:
#: Parent-owned and carrying no real handle: a handle in the instruction would
#: be one the model could copy without having read anything.
#:
#: Two sentences because the second is not implied by the first. "Copy the
#: marker" says what to do with one that is there; a model that has understood
#: the shape can also produce a well-formed one for a source it likes, and
#: that marker resolves - the namespace is this turn's, and the source may
#: genuinely be citable. It just was not what the passage said.
CITATION_INSTRUCTION = (
    "When a supporting passage includes a citation marker, copy that exact "
    "marker after claims based on that passage. Never invent or alter "
    "citation markers."
)

#: What separates a passage's text from the marker offered for it. One space,
#: and the marker inline at the end of the grounded run rather than on a line
#: of its own.
#:
#: Two reasons, both about the rest of the system rather than about looks. The
#: model has to write this token after a claim in prose, so the demonstration
#: it is given should be the shape it is being asked to produce. And
#: `strip_citations` removes `[ \t]*` before a marker, so a space-separated one
#: is taken back out cleanly by the reader-side cleanup that already exists,
#: while a newline-separated one would leave the blank line behind.
MARKER_SEPARATOR = " "


def handle_marker(handle: str) -> str:
    """The model-facing form of one handle."""
    return f"[cite:{handle}]"


def marker_cost(counter: Any, marker: str) -> int:
    """What to charge the prompt budget for one marker.

    A handle is a random-looking run of mixed-case letters and digits, which
    is not what the fallback estimator was calibrated on: it prices text at
    four characters per token, and a token that fragments harder than prose
    would then be budgeted as though it were prose. Being wrong here does not
    show up as a bad estimate, it shows up as a prompt the provider refuses
    after the offers have already been committed.

    So when the counter is estimating, the marker is charged at no less than
    its own length in bytes - one token per character, the worst a tokenizer
    can do with it. The rest of the prompt keeps the calibrated estimate;
    only the new random-looking token gets the floor.

    When the counter owns a real tokenizer its answer is the answer, and no
    floor is applied: a measured count is not improved by a guess about it.

    Markers are ASCII by construction - the nonce alphabet, digits, and
    `[cite:` - so the byte length is the character length.
    """
    counted = counter.count(marker)
    if getattr(counter, "exact", False):
        return counted
    return max(counted, len(marker.encode("ascii")))


def marker_surcharge(counter: Any, markers: Sequence[str]) -> int:
    """What the message count is short by, over these markers.

    A surcharge and not a cost. The markers are already inside the messages
    the counter was given, so it has priced each one once; adding
    `marker_cost` on top would charge twice. What is added is only the
    difference between the conservative price and the one the counter gave -
    zero for every marker when the counter is exact.
    """
    return sum(
        max(0, marker_cost(counter, marker) - counter.count(marker))
        for marker in markers
    )


@dataclass(frozen=True)
class OfferRender:
    """One candidate prompt, and what it actually shows the model.

    `placed` is the load-bearing part and is why rendering cannot just return
    text. A relation being eligible does not mean a marker for it reached the
    prompt: the placement rules reject a span whose offsets do not describe
    the text, one whose passage is not inside the run it covers, one filed
    under another source. Those rejections happen inside the render, so only
    the render can say which relations the model was really offered.

    Reported as relations rather than read back out of `markers`, because a
    marker names a source and the unit here is a passage: two passages of one
    document share a marker, and one of them may have been placed while the
    other was not.
    """

    messages: List[Dict[str, Any]]
    markers: Tuple[str, ...] = ()
    placed: Tuple[Tuple[str, str], ...] = ()


@dataclass(frozen=True)
class OfferChoice:
    """What one model call may offer, rendered and priced.

    `granted` is what the caller may commit, and every entry in it earned a
    marker in `messages`. `table` is the extension those grants imply, so a
    caller that commits `granted` and renders again gets this same prompt.

    `fits` false is the terminal condition: not "these offers did not fit"
    but "this prompt does not fit even offering nothing new", which is a
    property of the assembly rather than of a candidate. The caller sends the
    unlabelled prompt and stops offering for the rest of it.
    """

    table: CitationTable
    messages: List[dict]
    markers: Tuple[str, ...]
    tokens: int
    granted: Tuple[Binding, ...]
    fits: bool


def _relation(entry: Binding) -> Tuple[str, str]:
    return (
        str(entry.get("source_id") or ""),
        str(entry.get("evidence_id") or ""),
    )


def _already_offered(table: CitationTable, entry: Binding) -> bool:
    source_id, evidence_id = _relation(entry)
    return (
        table.handle_for(source_id) is not None
        and evidence_id in table.evidence_for(source_id)
    )


def choose_offers(
    *,
    registry: SourceRegistry,
    committed: CitationTable,
    candidates: Sequence[Binding],
    render: Callable[[CitationTable], OfferRender],
    counter: Any,
    budget: int,
) -> OfferChoice:
    """The largest prefix of `candidates` whose prompt still fits.

    Speculative throughout. `extend_citation_table` is pure, so a candidate
    table can be built, rendered and priced without any of it becoming the
    invocation's - which matters because a source that loses the budget
    decision must not come out of it holding a handle. The caller commits the
    survivors afterwards and renders once more from the table it actually got.

    Only what the render placed is granted. Being eligible is not the same as
    being shown: a span with impossible offsets, a passage the rendered run
    does not contain, a relation the table files elsewhere - the placement
    rules reject each of those inside the render, and the marker never
    reaches the model. Committing such a relation would hand a source
    citation authority for text nobody was ever offered, which is precisely
    the conservation rule this layer exists to hold.

    An unplaced candidate is dropped and the speculation starts again rather
    than being filtered out at the end. Its handle had already taken a number,
    so every source allocated after it was rendered and priced under a name
    the committed table would not give it. Restarting costs one more pass and
    keeps what was measured and what is committed the same prompt.

    The unit is the relation, not the occurrence. Committing one passage
    labels every span that names it on the next render, so choosing per
    occurrence would describe a state the second materialization cannot
    reproduce.

    Nothing already committed is withheld. Those handles are in text the model
    has already read, and taking one back would leave a marker in the
    conversation resolving to a source the table no longer offers. They are
    the floor: when the prompt does not fit with only those, there is nothing
    left to give up and the answer is that citations cannot be carried here.

    Withheld from the end, so priority is the caller's to express by ordering.
    """
    required = [entry for entry in candidates if _already_offered(committed, entry)]
    fresh = [
        entry for entry in candidates if not _already_offered(committed, entry)
    ]
    while True:
        offered = required + fresh
        table = extend_citation_table(registry, committed, offered)
        rendered = render(table)
        placed = set(rendered.placed)
        # Only the fresh ones force a restart. A required relation already
        # holds its number in `committed`, so whether it is passed here again
        # changes no allocation.
        surviving = [entry for entry in fresh if _relation(entry) in placed]
        if len(surviving) != len(fresh):
            fresh = surviving
            continue
        tokens = counter.count_messages(rendered.messages) + marker_surcharge(
            counter, rendered.markers
        )
        if tokens <= budget:
            return OfferChoice(
                table=table,
                messages=rendered.messages,
                markers=tuple(rendered.markers),
                tokens=tokens,
                granted=tuple(
                    entry for entry in offered if _relation(entry) in placed
                ),
                fits=True,
            )
        if not fresh:
            return OfferChoice(
                table=committed,
                messages=[],
                markers=(),
                tokens=tokens,
                granted=(),
                fits=False,
            )
        fresh.pop()


def _eligible(table: CitationTable, span: GroundedSpan) -> Optional[str]:
    """The handle this span may be labelled with, if it may be labelled.

    Both halves, not just the source. A span naming a source that has a
    handle, beside a passage filed under a different source, is exactly the
    relation `build_citation_table` refuses to grant authority for; accepting
    it here would put the first source's name on the second one's text and
    reach the model as a demonstration that the two go together.
    """
    handle = table.handle_for(span.source_id)
    if handle is None:
        return None
    if span.evidence_id not in table.evidence_for(span.source_id):
        return None
    return handle


def _covers_its_evidence(
    registry: SourceRegistry, text: str, span: GroundedSpan
) -> bool:
    """Whether the run this span points at is the passage it names.

    The relation being valid is a different question from the offsets being
    right, and a span can pass the first while failing the second. Source A
    has a handle, passage A belongs to source A, and the offsets cover source
    B's text: every individual check succeeds and the model is shown B's
    sentence with A's name after it. That is the failure this module exists
    to prevent, arriving through the one door the relation check does not
    watch.

    So the offer gate measures the placement itself rather than inheriting it
    from the producers. S5.11 witnesses that they render and record together,
    and this is where those records become model input - the last point at
    which a wrong offset is still only a wrong offset.

    Containment, not equality. A producer wraps its evidence in text of the
    parent's own: an untrusted-data envelope, a `source:` header, a result
    number. The span covers the rendered run, and the passage sits inside it.

    Empty evidence is refused, and that guard does work. `""` is contained in
    every string, so a passage with no text is inside whatever run a span
    happens to name - the containment test above would pass at any valid
    offset. The registry records one: `add_evidence` requires a `str` and not
    a non-empty one, so an empty passage reaches `build_citation_table`, earns
    its source a handle, and arrives here eligible.
    """
    evidence = registry.get_evidence(span.evidence_id)
    if evidence is None or evidence.source_id != span.source_id:
        return False
    if not evidence.text:
        return False
    return evidence.text in text[span.start : span.end]


def _placeable(text: str, span: GroundedSpan) -> bool:
    """Whether this span describes a run of `text` a marker can follow.

    A span that runs past the end of the string, or backwards, or covers no
    characters, is not a position - it is a record that does not describe this
    text. Such a span is dropped rather than clamped: clamping puts the marker
    at whatever offset survives, which attributes a passage the span never
    covered, and that is the failure this whole layer exists to prevent.

    Dropped rather than raised, because the blast radius is one marker. The
    passage is still shown, the model simply is not offered a name for that
    piece of it, and under-offering is the safe direction.
    """
    return 0 <= span.start < span.end <= len(text)


def label_passage(
    text: str,
    spans: Sequence[GroundedSpan],
    table: CitationTable,
    registry: SourceRegistry,
) -> str:
    """`text` with a citation marker after each passage that has earned one.

    This is the offer. Every marker in the returned string is a handle this
    turn issued for the source whose evidence covers the run of text directly
    in front of it, so a model that copies one is naming the passage it was
    reading.

    Three questions of every span, and all three have to answer yes. Is this a
    position in this text; does the run at that position contain the passage
    the span names; may that passage be cited. The registry answers the
    second, which is why it is a parameter: the table says what a source is
    called and cannot say where its text is.

    Right to left, so that inserting a marker cannot move a position not yet
    used. Left to right needs every later offset shifted by the width of every
    marker already written, which is a second calculation that has to agree
    with the first, and the two disagree the first time a span is skipped.

    One handle per source, from the table, so the same document read in two
    passages is named identically in both. A span that fails any of the three
    gets no marker at all - never a guessed position.

    Returns a new string and reads its inputs. The passage records, the
    transcript entries and the base prompt snapshot are all parent state that
    later stages read again; a materializer that edited them in place would
    make the second materialization of one assembly differ from the first.
    """
    placements: List[Tuple[int, str]] = []
    for span in spans:
        if not _placeable(text, span):
            continue
        if not _covers_its_evidence(registry, text, span):
            continue
        handle = _eligible(table, span)
        if handle is None:
            continue
        placements.append((span.end, handle_marker(handle)))
    # Descending by offset. `reverse=True` on the offset alone is stable, so
    # two markers at one position keep the order their spans were recorded in
    # - reversed, since they are applied from the right.
    placements.sort(key=lambda item: item[0], reverse=True)
    labelled = text
    for offset, marker in placements:
        labelled = (
            labelled[:offset] + MARKER_SEPARATOR + marker + labelled[offset:]
        )
    return labelled


def _labelled_with_markers(
    text: str,
    spans: Sequence[GroundedSpan],
    table: CitationTable,
    registry: SourceRegistry,
) -> Tuple[str, List[str], List[Tuple[str, str]]]:
    """`label_passage`, and what it actually placed.

    The budget needs the markers and the offer decision needs the relations,
    and both for the same reason: which spans earned a marker is decided
    inside the labelling, not by the caller. A span can name an eligible
    relation and still be dropped for its offsets, so counting or committing
    what was *offered* would price and grant markers that are not in the text.
    """
    labelled = label_passage(text, spans, table, registry)
    markers: List[str] = []
    placed: List[Tuple[str, str]] = []
    for span in spans:
        if not _placeable(text, span) or not _covers_its_evidence(
            registry, text, span
        ):
            continue
        handle = _eligible(table, span)
        if handle is None:
            continue
        markers.append(handle_marker(handle))
        placed.append((span.source_id, span.evidence_id))
    return labelled, markers, placed


def label_snippets(
    snippets: Sequence[str],
    grounds: Sequence[Optional[Binding]],
    table: CitationTable,
    registry: SourceRegistry,
) -> Tuple[List[str], List[str], List[Tuple[str, str]]]:
    """Context snippets with their markers, the markers, and what was placed.

    The automatic retrieval route has no spans, because it needs none: a
    snippet *is* a passage, so the run a marker follows is the whole string.
    Building that span here rather than giving this route a rendering rule of
    its own keeps one placement contract - the same eligibility, the same
    containment check, the same separator - and means a snippet that does not
    contain the passage it is filed under gets no marker on this path either.

    `grounds` is the aligned vector: one entry per snippet, `None` where the
    snippet is the parent's own digest or recall window rather than a
    document. A length mismatch would silently attach markers to the wrong
    snippets, so it is refused rather than zipped.
    """
    if len(grounds) != len(snippets):
        raise ProvenanceError(
            "grounding is not aligned: "
            f"{len(grounds)} entries for {len(snippets)} snippets"
        )
    labelled: List[str] = []
    markers: List[str] = []
    placed: List[Tuple[str, str]] = []
    for snippet, ground in zip(snippets, grounds):
        if not ground:
            labelled.append(snippet)
            continue
        source_id, evidence_id = _relation(ground)
        span = GroundedSpan(
            start=0,
            end=len(snippet),
            source_id=source_id,
            evidence_id=evidence_id,
        )
        text, placed_markers, placed_here = _labelled_with_markers(
            snippet, [span], table, registry
        )
        labelled.append(text)
        markers.extend(placed_markers)
        placed.extend(placed_here)
    return labelled, markers, placed


def instruct(messages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The messages with the citation instruction added to the system block.

    Appended to the first system message rather than inserted as one of its
    own. A second system message is a second thing for a small model to weigh
    against the first, and this instruction is about how to write the answer
    the rest of that block is asking for.

    A conversation with no system message gets one. That is not a shape the
    agent path produces - it always builds a system block - so it is the
    plain-list case, and the instruction has to land somewhere.
    """
    prepared = [deepcopy(dict(message)) for message in messages or []]
    for message in prepared:
        if message.get("role") == "system":
            body = str(message.get("content") or "")
            message["content"] = (
                f"{body}\n\n{CITATION_INSTRUCTION}" if body else CITATION_INSTRUCTION
            )
            return prepared
    prepared.insert(0, {"role": "system", "content": CITATION_INSTRUCTION})
    return prepared


def rebuild_agent_messages(
    initial_messages: Sequence[Dict[str, Any]],
    initial_grounded_messages: Sequence[GroundedMessage],
    transcript: TrustedTranscript,
    table: CitationTable,
    registry: SourceRegistry,
) -> Tuple[List[Dict[str, Any]], List[str], List[Tuple[str, str]]]:
    """The conversation so far, from parent state only, with its offers in it.

    The worker assembles a message list of its own and sends it with every
    model call, and every message in it describes something the parent did.
    That was fine while nothing in it carried authority. It stops being fine
    the moment a tool result carries a citation marker: the marker would be in
    bytes the untrusted half chose, beside prose it also chose, and the model
    would be answering a conversation the parent cannot vouch for.

    So this is built from the base prompt the parent kept and the record it
    wrote as the turn ran - never from `payload["messages"]`.

    Only offerable rounds contribute. A round whose calls were not the calls
    the previous model turn asked for cannot be reconstructed faithfully - its
    results may not even have a trustworthy `tool_call_id` - so it is left out
    of the reconstruction entirely rather than included unlabelled. The
    assembly has already lost its citation authority by then; this is what
    stops the parent asserting a conversation shape it cannot support.

    Model turns go in as they are. They are the public reply the worker
    continued from, and they carry no namespace by construction.

    The base prompt is labelled too, from the positions the parent measured as
    it wrote that prompt. Without it the first model call of an agent turn
    could offer nothing at all: its grounding is entirely in the message built
    before any tool ran, so every turn would need a search before it could
    cite the context the planner had already retrieved.

    Those positions are honoured only against the exact string they were
    measured in. A message that no longer equals the recorded text is not
    searched for a new position and its offsets are not adjusted - it goes in
    unlabelled, and whatever grounded it is simply not offered here.
    """
    grounded_at = {
        record.message_index: record
        for record in initial_grounded_messages or ()
    }
    messages: List[Dict[str, Any]] = []
    markers: List[str] = []
    placed: List[Tuple[str, str]] = []
    for index, message in enumerate(initial_messages or []):
        rebuilt = deepcopy(dict(message))
        record = grounded_at.get(index)
        if record is not None and str(rebuilt.get("content") or "") == record.text:
            text, placed_markers, placed_here = _labelled_with_markers(
                record.text, record.spans, table, registry
            )
            rebuilt["content"] = text
            markers.extend(placed_markers)
            placed.extend(placed_here)
        messages.append(rebuilt)
    for entry in transcript.entries:
        if isinstance(entry, ModelTurn):
            if entry.assistant_message is not None:
                messages.append(deepcopy(dict(entry.assistant_message)))
            else:
                messages.append(
                    {
                        "role": "assistant",
                        "content": entry.content,
                        "tool_calls": [
                            deepcopy(dict(call)) for call in entry.tool_calls
                        ],
                    }
                )
            continue
        if not isinstance(entry, ToolRound) or not entry.offerable:
            continue
        for result in entry.results:
            text, placed_markers, placed_here = _labelled_with_markers(
                result.text, result.spans, table, registry
            )
            markers.extend(placed_markers)
            placed.extend(placed_here)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": result.tool_message_id,
                    "name": result.tool_name,
                    "content": text,
                }
            )
    return messages, markers, placed
