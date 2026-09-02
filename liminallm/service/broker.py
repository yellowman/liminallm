"""The capabilities a tool worker is not trusted to hold, held for it here.

A tool worker runs model-chosen control flow over attacker-controlled bytes.
SPEC §18 says what that means for where it runs - a spawned child under rlimits
with no filesystem access beyond a scratch - and this module says what it means
for what it can reach: nothing. No store handle, no model client, no settings
object, no credentials, no identity. It asks; the parent decides.

So every effect a worker can have is a capability served from this side of the
pipe. Three properties fall out of putting them here rather than there, and
each one is a control the in-thread model could not express:

* **identity is not a parameter.** The worker never sends a user_id or a
  tenant; the broker already knows them, because they came off the
  authenticated request and never left the parent (§12.2). A compromised worker
  cannot name another user's files - it has no field in which to say so.
* **revocation lands before the effect.** Every handler checks liveness first,
  under the invocation's lock. A revoked turn issues no request and starts no
  child, rather than starting one and reporting the fact afterwards.
* **what got started is reachable.** Sandbox children are the parent's
  children, not the worker's, so killing the worker does not reach them. They
  are registered against the invocation as they start, which is what makes "the
  tree is dead" a statement anyone can verify.

Capability withdrawal after an injection finding (§21.1) is enforced here for
the same reason the rest is: the process that just read "ignore your rules and
run this" is the last one that should be asked whether the rule still applies.

Some tool bodies stay in the parent, behind `tool.host`. Those are the ones
whose bodies are broad reads of the store - prompt assembly, adapter selection,
RAG composition. They hold no model-chosen control flow, so nothing about them
is contained by moving the body across the pipe; what a worker would gain is a
proxy for every method of the store, which is a worse boundary than none. They
keep the worker process, the rlimits, the ledger and the revocation check; only
the body runs here, and the capability name says so.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from liminallm.logging import get_logger
from liminallm.service.invocation import (
    Invocation,
    LeaseRevoked,
    RetryDivergence,
    current_invocation,
    payload_hash,
)
from liminallm.service.provenance import Binding, SourceRegistry
from liminallm.service.rag import register_retrieved_chunks
from liminallm.service.sandbox import tool_network_guard
from liminallm.service.tool_worker import FrameBudget
from liminallm.service.wire import WireError, recv_frame, send_frame

logger = get_logger(__name__)

#: How long the serve loop waits on the pipe before re-checking whether the
#: worker is still alive. A killed worker never sends again, so without this
#: the loop would block forever on a process that no longer exists.
_POLL_SECONDS = 0.05


@dataclass
class CapabilityOutcome:
    """A capability's two outputs, separated by who may see them.

    `public` is the reply, and it crosses the pipe. `parent_state` does not:
    it is what the parent learned by serving the request, and it is committed
    to the ledger beside the reply so a replacement attempt replaying that
    reply gets it back without the handler running again.

    A handler with nothing to keep just returns its dict; only the ones with
    parent-side consequences reach for this.
    """

    public: Dict[str, Any]
    parent_state: Optional[Dict[str, Any]] = None


class UnknownCapability(RuntimeError):
    """The worker asked for something no capability serves."""


def _is_error(result: Any) -> bool:
    return isinstance(result, dict) and result.get("status") == "error"


@dataclass
class InvocationContext:
    """Who this execution is for. Never crosses the pipe.

    Every field here is something the worker must not be able to choose. The
    tenant comes from the request's host and the user from its token (§12.2); a
    worker that could send either would be a worker that could address another
    tenant's data by asking nicely.
    """

    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    conversation_id: Optional[str] = None
    context_id: Optional[str] = None
    adapters: List[dict] = field(default_factory=list)
    history: List[Any] = field(default_factory=list)
    user_message: str = ""
    #: Model-visible name -> the remote MCP tool it dispatches to, for the
    #: tools discovered for this turn. Here rather than in the plan for the
    #: reason above: the entry carries the server's URL and its taint class,
    #: and a worker that could send either could name a host of its choosing
    #: and call it `local_read`. The worker sends a name; the parent decides
    #: what that name means.
    mcp_tools: Dict[str, Any] = field(default_factory=dict)
    #: The turn's provenance registry, created once in the workflow entry
    #: point and passed by reference so every node of one turn records into
    #: the same one. Never crosses the pipe either: ids are turn-local
    #: authority, and a worker that could mint them could claim a citation
    #: came from a source it never read.
    source_registry: Optional[SourceRegistry] = None
    #: What the parent placed in this invocation's prompt, computed parent-side
    #: after budgeting. Never sent to the worker and never read back from it.
    provenance_bindings: List[Binding] = field(
        default_factory=list
    )


class CapabilityBroker:
    """Serves capability requests from one worker, for one invocation.

    Handlers delegate to the workflow engine's own methods rather than
    re-implementing them. That is deliberate: the engine is where the store,
    the model and the retrieval services already live, and a second
    implementation of "fetch a page" would be a second place for the
    untrusted-content rules to drift out of step.
    """

    #: Capability → the tool name the client knows it by. Traces are labelled
    #: from this rather than from anything the worker sends: a progress notice
    #: is cosmetic, but a label the untrusted side chooses is still a string it
    #: put on the user's screen.
    TRACE_LABELS = {
        "web.search": "web_search",
        "web.fetch": "web_fetch",
        "python.run": "run_python",
        "rag.retrieve": "file_search",
        "notes.search": "note_search",
        "history.search": "history_search",
    }

    #: The tool names a round may be labelled with. A round carries the names
    #: the model chose, and those reach the user's screen - so they are matched
    #: against this set rather than passed through.
    ROUND_LABELS = frozenset(TRACE_LABELS.values())

    #: Capability → the tool name §21.1 withdraws it under. The withdrawal has
    #: to be enforced here, on the capability itself, and not only inside the
    #: round that usually carries it: the worker is the untrusted side, so
    #: "the worker asks through `tools.round`" is a description of the intended
    #: protocol, not a constraint on the compromised one. A worker that has
    #: read a hostile page can send `web.fetch` directly.
    WITHDRAWABLE = {
        "web.fetch": "web_fetch",
        "web.search": "web_search",
        "python.run": "run_python",
    }

    def __init__(
        self,
        engine: Any,
        context: InvocationContext,
        *,
        on_capability: Optional[Callable[[dict], None]] = None,
    ) -> None:
        self._engine = engine
        self._ctx = context
        #: Called as each capability starts, so a streaming caller can say what
        #: the model is doing while it is slow rather than afterwards.
        self._on_capability = on_capability

    # -- the loop ---------------------------------------------------------

    def serve(
        self,
        conn: Any,
        invocation: Invocation,
        *,
        is_alive: Callable[[], bool],
        budget: Optional[FrameBudget] = None,
    ) -> Dict[str, Any]:
        """Answer this worker until it finishes, dies, or is revoked.

        Returns the worker's final payload, or an error describing why there
        was not one. The loop never blocks indefinitely on a recv: a worker
        killed by revocation or by the wall clock does not close the pipe
        politely, so liveness is re-checked on every poll instead of trusted.

        The invocation is bound to this thread for the whole loop. The worker
        holds no store or model handle, but the handlers below reach the real
        ones through the engine, and binding here is what lets `LeasedProxy`
        check every call they make - reads included - without each handler
        having to remember.

        **What arrives here is decoded as data, never as objects.** SPEC §18
        designates the worker untrusted, and `Connection.recv()` unpickles: a
        worker that had been talked into anything could hand back an object
        whose *deserialization* ran its payload, in this process, inside this
        loop, before the liveness check below. That is not a check that was
        missing - the decoder was the hole. Frames are JSON, bounded by what
        this loop has itself sent (`FrameBudget`).
        """
        budget = budget or FrameBudget(0)
        with current_invocation(invocation):
            while True:
                if invocation.revoked:
                    return {
                        "status": "error",
                        "content": "tool invocation revoked",
                        "error": "revoked",
                    }
                if not conn.poll(_POLL_SECONDS):
                    if not is_alive():
                        return {
                            "status": "error",
                            "content": "tool worker exited without a result",
                            "error": "worker_died",
                        }
                    continue
                try:
                    message = recv_frame(conn, max_bytes=budget.limit)
                except (EOFError, OSError) as exc:
                    return {
                        "status": "error",
                        "content": f"tool worker closed the channel: {exc}",
                        "error": "worker_died",
                    }
                except WireError as exc:
                    # Oversized or not data. The pipe is unusable after an
                    # over-length frame, and a worker that sent one has
                    # nothing further to say that this process would believe.
                    logger.warning(
                        "worker_frame_rejected",
                        invocation_id=invocation.invocation_id,
                        error=str(exc),
                    )
                    return {
                        "status": "error",
                        "content": f"the tool worker sent something unreadable: {exc}",
                        "error": "worker_protocol",
                    }
                if message.get("done"):
                    result = message.get("result")
                    return result if isinstance(result, dict) else {}
                try:
                    budget.credit(send_frame(conn, self._answer(invocation, message)))
                except WireError as exc:
                    logger.error(
                        "capability_reply_unsendable",
                        invocation_id=invocation.invocation_id,
                        error=str(exc),
                    )
                    return {
                        "status": "error",
                        "content": "a capability produced a reply that could not be sent",
                        "error": "broker_protocol",
                    }
                except (BrokenPipeError, OSError):
                    return {
                        "status": "error",
                        "content": "tool worker closed the channel",
                        "error": "worker_died",
                    }

    def _answer(self, invocation: Invocation, message: Dict[str, Any]) -> Dict[str, Any]:
        """One capability request, checked and either replayed or run."""
        capability = str(message.get("capability") or "")
        payload = message.get("payload") or {}
        operation_seq = int(message.get("operation_seq") or 0)
        try:
            # Liveness first, before anything is looked up or dispatched. The
            # ordering is the control: after the handler runs, "revoked" is a
            # description of something that already happened.
            invocation.check_live()
            handler = self._handlers().get(capability)
            if handler is None:
                raise UnknownCapability(capability)
            withdrawn = self._withdrawn(invocation, capability)
            if withdrawn is not None:
                return {"ok": True, "result": withdrawn}
            digest = payload_hash(payload)
            replayed = invocation.ledger.replay(operation_seq, capability, digest)
            if replayed is not None:
                logger.info(
                    "capability_replayed",
                    invocation_id=invocation.invocation_id,
                    capability=capability,
                    operation_seq=operation_seq,
                )
                # The handler does not run, so this is the only place the
                # replacement attempt can learn what the first one recorded.
                # Re-deriving it instead would bind replayed text to a fresh
                # retrieval, and the corpus may have moved since.
                self._apply_parent_state(replayed.parent_state)
                return {"ok": True, "result": replayed.result, "replayed": True}
            invocation.ledger.begin(operation_seq, capability, digest)
            self._notify(capability)
            started = time.monotonic()
            # SPEC §18.3/§21.1: tool egress is allowlisted. The guard is
            # thread-local and the serve loop has its own thread, so applying
            # it here covers every capability - which is the point, since the
            # capabilities are now the only things that open sockets at all.
            with tool_network_guard(self._engine.tool_network_policy):
                outcome = handler(invocation, operation_seq, payload)
            parent_state: Optional[Dict[str, Any]] = None
            if isinstance(outcome, CapabilityOutcome):
                result, parent_state = outcome.public, outcome.parent_state
            else:
                result = outcome
            if _is_error(result):
                # A failed step is not a committed one. Recording it as
                # committed would make the ledger replay the failure on every
                # retry, which turns a retry into a slower way of failing.
                invocation.ledger.fail(
                    operation_seq, str(result.get("error") or "error")
                )
            else:
                # Beside the reply, not inside it: committed together so a
                # replay cannot get one without the other.
                invocation.ledger.commit(
                    operation_seq, result, parent_state=parent_state
                )
                self._apply_parent_state(parent_state)
            logger.info(
                "capability_served",
                invocation_id=invocation.invocation_id,
                capability=capability,
                operation_seq=operation_seq,
                latency_ms=round((time.monotonic() - started) * 1000),
            )
            return {"ok": True, "result": result}
        except LeaseRevoked as exc:
            invocation.ledger.fail(operation_seq, str(exc))
            return {"ok": False, "code": "revoked", "error": str(exc)}
        except RetryDivergence as exc:
            invocation.ledger.fail(operation_seq, str(exc))
            logger.warning(
                "capability_retry_divergence",
                invocation_id=invocation.invocation_id,
                capability=capability,
                operation_seq=operation_seq,
            )
            return {"ok": False, "code": "retry_divergence", "error": str(exc)}
        except UnknownCapability:
            invocation.ledger.fail(operation_seq, "unknown_capability")
            return {"ok": False, "code": "unknown_capability", "error": capability}
        except Exception as exc:  # noqa: BLE001 - the worker gets the error, not a crash
            invocation.ledger.fail(operation_seq, str(exc))
            logger.warning(
                "capability_failed",
                invocation_id=invocation.invocation_id,
                capability=capability,
                error=str(exc),
            )
            return {"ok": False, "code": "failed", "error": str(exc)}

    def _withdrawn(
        self, invocation: Invocation, capability: str
    ) -> Optional[Dict[str, Any]]:
        """The refusal this capability has earned, or None if it still runs.

        §21.1: a turn that has read a possible injection loses every capability
        that could carry data off the box, for the rest of it. Returned as a
        result rather than raised, so the model reads plainly why it was
        refused and does not spend the turn retrying.
        """
        from liminallm.service import taint

        tool_name = self.WITHDRAWABLE.get(capability)
        if tool_name is None or not taint.is_withdrawn(tool_name, invocation.session):
            return None
        logger.warning(
            "capability_withdrawn_by_injection_taint",
            invocation_id=invocation.invocation_id,
            capability=capability,
            findings=len(taint.findings(invocation.session)),
        )
        refusal = taint.refusal(invocation.session)
        return {"text": refusal, "findings": [], "artifacts": []}

    def _notify(self, capability: str) -> None:
        self._emit(self.TRACE_LABELS.get(capability))

    def _emit(self, label: Optional[str]) -> None:
        if label and self._on_capability is not None:
            try:
                self._on_capability({"tool": label, "status": "running"})
            except Exception:  # noqa: BLE001 - a progress notice never fails work
                pass

    def _handlers(
        self,
    ) -> Dict[str, Callable[[Invocation, int, Dict[str, Any]], Any]]:
        return {
            "web.search": self._web_search,
            "web.fetch": self._web_fetch,
            "python.run": self._python_run,
            "rag.retrieve": self._rag_retrieve,
            "notes.search": self._notes_search,
            "history.search": self._history_search,
            "llm.generate_with_tools": self._llm_generate_with_tools,
            "tools.round": self._tools_round,
            "tool.host": self._tool_host,
        }

    # -- web --------------------------------------------------------------

    def _web_search(
        self, invocation: Invocation, _seq: int, payload: Dict[str, Any]
    ) -> CapabilityOutcome:
        invocation.check_live()
        grounds = self._sink()
        text, findings = self._engine._run_web_search(
            str(payload.get("query") or ""), int(payload.get("limit") or 5),
            source_registry=self._ctx.source_registry, bindings_sink=grounds,
        )
        return self._grounded(
            self._with_findings(invocation, text, findings), grounds
        )

    def _web_fetch(
        self, invocation: Invocation, _seq: int, payload: Dict[str, Any]
    ) -> CapabilityOutcome:
        invocation.check_live()
        grounds = self._sink()
        text, findings = self._engine._run_web_fetch(
            str(payload.get("url") or ""),
            source_registry=self._ctx.source_registry, bindings_sink=grounds,
        )
        return self._grounded(
            self._with_findings(invocation, text, findings), grounds
        )

    def _sink(self) -> Optional[List[Dict[str, Optional[str]]]]:
        """Where a producer records, or None when this turn keeps no record."""
        return None if self._ctx.source_registry is None else []

    @staticmethod
    def _grounded(
        public: Dict[str, Any],
        grounds: Optional[List[Dict[str, Optional[str]]]],
    ) -> CapabilityOutcome:
        """The reply, and beside it what the parent learned by serving it.

        The bindings do not cross the pipe. Every producer says this the same
        way, so a new capability cannot leak ids by forgetting to split its
        two outputs - it either returns through here or it has no grounding
        to leak.
        """
        return CapabilityOutcome(
            public=public,
            parent_state={"provenance_bindings": grounds} if grounds else None,
        )

    def _with_findings(
        self, invocation: Invocation, text: str, findings: List[dict]
    ) -> Dict[str, Any]:
        """Record injection findings on the parent's copy of the turn.

        The worker is told, so it can explain what happened, but the record
        that arms the withdrawal lives here. A worker that has read a hostile
        page cannot un-taint itself by declining to mention it.
        """
        from liminallm.service import taint

        taint.record_findings(invocation.session, findings)
        kinds = [
            f.get("type") for f in findings if isinstance(f, dict) and f.get("type")
        ]
        return {"text": text, "findings": [k for k in kinds if k]}

    # -- code -------------------------------------------------------------

    def _python_run(
        self, invocation: Invocation, seq: int, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run model-written Python, with the child registered as it starts."""
        invocation.check_live()
        before = list(invocation.session.get("artifacts") or [])
        text = self._engine._run_python_capability(
            str(payload.get("code") or ""),
            invocation=invocation,
            operation_seq=seq,
            step="publish",
            user_id=self._ctx.user_id,
            conversation_id=self._ctx.conversation_id,
        )
        after = list(invocation.session.get("artifacts") or [])
        return {"text": text, "artifacts": after[len(before) :]}

    def _apply_parent_state(self, parent_state: Optional[Dict[str, Any]]) -> None:
        """Fold what a capability recorded into the invocation's own record.

        Deduped on the triple, because one relation reached twice is still
        one relation: a selected context may already have put a chunk in the
        opening prompt, and an explicit search may retrieve the same chunk
        again.
        """
        if not parent_state:
            return
        collected = self._ctx.provenance_bindings
        seen = {
            (b.get("context_id"), b.get("source_id"), b.get("evidence_id"))
            for b in collected
        }
        for binding in parent_state.get("provenance_bindings") or []:
            key = (
                binding.get("context_id"),
                binding.get("source_id"),
                binding.get("evidence_id"),
            )
            if key in seen:
                continue
            seen.add(key)
            collected.append(dict(binding))

    # -- retrieval, notes, history ----------------------------------------

    def _rag_retrieve(
        self, invocation: Invocation, _seq: int, payload: Dict[str, Any]
    ) -> CapabilityOutcome:
        invocation.check_live()
        text, snippets, chunks, hints = self._engine._run_file_search(
            str(payload.get("query") or ""),
            int(payload.get("limit") or 4),
            conversation_id=self._ctx.conversation_id,
            context_id=self._ctx.context_id,
            user_id=self._ctx.user_id,
            tenant_id=self._ctx.tenant_id,
        )
        # The chunks that were rendered, after the attachment-generation
        # scope has already dropped what this conversation no longer holds -
        # authorize first, record second. There is no second budgeting step
        # between here and the model: this text is appended to the agent's
        # messages as it stands, so every rendered chunk is grounding.
        #
        # Through the same adapter the automatic paths use. A second mapping
        # here would be a second place for the file-versus-inline identity
        # rules to drift.
        bindings: List[Binding] = []
        if self._ctx.source_registry is not None:
            bindings = register_retrieved_chunks(
                self._ctx.source_registry, chunks, hints=hints
            )
        # `text` and `snippets` cross the pipe; the bindings do not. An id in
        # the reply is an id the untrusted side can quote back as its own.
        return self._grounded({"text": text, "snippets": snippets}, bindings)

    def _notes_search(
        self, invocation: Invocation, _seq: int, payload: Dict[str, Any]
    ) -> CapabilityOutcome:
        invocation.check_live()
        grounds = self._sink()
        text = self._engine._run_note_search(
            str(payload.get("query") or ""),
            int(payload.get("limit") or 6),
            user_id=self._ctx.user_id,
            source_registry=self._ctx.source_registry,
            bindings_sink=grounds,
        )
        return self._grounded({"text": text}, grounds)

    def _history_search(
        self, invocation: Invocation, _seq: int, payload: Dict[str, Any]
    ) -> CapabilityOutcome:
        invocation.check_live()
        grounds = self._sink()
        text = self._engine._run_history_search(
            str(payload.get("query") or ""),
            int(payload.get("limit") or 4),
            conversation_id=self._ctx.conversation_id,
            user_id=self._ctx.user_id,
            source_registry=self._ctx.source_registry,
            bindings_sink=grounds,
        )
        return self._grounded({"text": text}, grounds)

    # -- the model --------------------------------------------------------

    def _llm_generate_with_tools(
        self, invocation: Invocation, _seq: int, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        invocation.check_live()
        response = self._engine.llm.generate_with_tools(
            payload.get("messages") or [],
            payload.get("tools") or [],
            self._ctx.adapters,
            user_id=self._ctx.user_id,
        )
        return {
            "content": response.get("content") or "",
            "tool_calls": response.get("tool_calls") or [],
            "assistant_message": response.get("assistant_message"),
            "usage": response.get("usage") or {},
        }

    # -- one round of the agent loop --------------------------------------

    def _tools_round(
        self, invocation: Invocation, seq: int, payload: Dict[str, Any]
    ) -> CapabilityOutcome:
        """Execute the tools the model asked for in one round.

        The round arrives as one request because how its calls are *run* is a
        decision about effects, and effects are the parent's. SPEC §18 has read
        -only calls fan out into a nested pool while anything that can taint the
        turn stays strictly ordered - a web_fetch that records an injection
        finding has to be able to withdraw a run_python later in the same round,
        and that ordering only exists when those calls run one at a time.
        """
        invocation.check_live()
        calls = payload.get("calls") or []
        parsed = [
            (
                {"id": c.get("id") or "", "name": str(c.get("name") or "")},
                str(c.get("name") or ""),
                dict(c.get("arguments") or {}),
            )
            for c in calls
        ]
        for _call, name, _args in parsed:
            # A remote tool's label is dynamic, so it cannot be in the static
            # set - but it is still matched rather than passed through: the
            # name has to be one this turn actually discovered, which the
            # worker did not choose either.
            allowed = name in self.ROUND_LABELS or name in self._ctx.mcp_tools
            self._emit(name if allowed else None)
            logger.info(
                "attachment_tool_called",
                tool=name,
                invocation_id=invocation.invocation_id,
                conversation_id=self._ctx.conversation_id,
                user_id=self._ctx.user_id,
            )
        snippets: List[str] = []
        round_bindings: List[Binding] = []
        before = list(invocation.session.get("artifacts") or [])
        results = self._engine._run_round_tools(
            parsed,
            conversation_id=self._ctx.conversation_id,
            context_id=self._ctx.context_id,
            user_id=self._ctx.user_id,
            tenant_id=self._ctx.tenant_id,
            session=invocation.session,
            snippets=snippets,
            fallback_query=str(payload.get("fallback_query") or ""),
            invocation=invocation,
            operation_seq=seq,
            mcp_tools=self._ctx.mcp_tools,
            source_registry=self._ctx.source_registry,
            bindings=round_bindings,
        )
        after = list(invocation.session.get("artifacts") or [])
        # A round is one committed operation, so its grounding is committed
        # with it and comes back on a replay by the same route a single
        # retrieval's does.
        return self._grounded(
            {
                "results": results,
                "snippets": snippets,
                "artifacts": after[len(before) :],
                "findings": list(invocation.session.get("injection_findings") or []),
            },
            round_bindings,
        )

    def _tool_host(
        self, invocation: Invocation, _seq: int, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run a builtin tool body that still lives in the parent."""
        invocation.check_live()
        return self._engine._run_host_tool(
            str(payload.get("tool") or ""),
            payload.get("inputs") or {},
            context=self._ctx,
        )
