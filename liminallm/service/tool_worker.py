"""The process a tool runs in, and the handle the parent keeps on it.

SPEC §18 has always said tool workers run in a spawned child process under
POSIX rlimits with a wall-clock kill. What ran instead was a thread pool, and a
thread is not a boundary anything can act on: a node timeout cancelled the
coroutine awaiting the thread, the thread carried on, and the retry ran beside
its own predecessor — sharing a working directory, a sandbox child, and a claim
on the answer. "Cancelled" described who had stopped watching.

Two halves live here because they are the two ends of one pipe:

* the **child** — `_worker_main` and the bodies below it. It leads its own
  process group, so one `killpg` reaches whatever it went on to spawn. It
  imports no store, no model client, no settings and no credentials; every
  effect it can have is a request to the broker on the other end.
* the **parent** — `spawn` and `WorkerHandle`. It registers the child against
  the invocation before anything is awaited, so the process is killable from
  the moment it exists rather than from the moment the parent notices it.

Keeping the child's imports to the standard library is not tidiness. It is what
makes the spawn affordable — a fresh interpreter that imported the service
layer would cost far more than the work most tools do — and it is what keeps
the promise that the worker holds nothing worth stealing. `spawn` re-imports
this module in the child, so every module-level import here is paid on every
tool call: the logger and the invocation types are parent-side only and are
kept out of that graph deliberately. Measured: importing structlog alone cost
more than the rest of the spawn put together.
"""
from __future__ import annotations

import json
import multiprocessing
import os
import signal
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:  # parent-side only; `from __future__ import annotations`
    from liminallm.service.invocation import Attempt, Invocation

#: How long the parent waits for a killed worker to be joined. The worker has
#: nothing to flush, so this is short by design.
_JOIN_TIMEOUT_SECONDS = 2.0

#: How long the parent waits for the child's READY handshake before deciding it
#: does not lead a group of its own. Generous: a loaded host can take a while
#: to start an interpreter, and the cost of giving up early is only that the
#: worker is killed by single pid instead of by group.
_READY_TIMEOUT_SECONDS = 15.0
_READY_POLL_SECONDS = 0.05


# ---------------------------------------------------------------------------
# child side
# ---------------------------------------------------------------------------


class BrokerRefused(RuntimeError):
    """The parent declined a capability request."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class BrokerClient:
    """The worker's only way to affect anything.

    Every call is a round trip. The sequence number it stamps on each request
    is the worker's position in its own control flow, which is what lets a
    replacement worker replay a killed one's ledger: same step, same payload,
    already committed, so the parent answers from the record instead of doing
    the work a second time.
    """

    def __init__(self, conn: Any) -> None:
        self._conn = conn
        self._seq = 0

    @property
    def operation_seq(self) -> int:
        return self._seq

    def call(self, capability: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Ask for a capability. Raises `BrokerRefused` when it is refused."""
        from liminallm.service.wire import recv_frame, send_frame

        self._seq += 1
        # Neither direction is capped here, for opposite reasons. Outbound,
        # the broker's `FrameBudget` is the cap that counts and it grows as
        # the broker answers; a second copy of it in this process would drift
        # from the real one and refuse frames the broker would have taken.
        # Inbound, the broker is the trusted end — a cap here would defend a
        # disposable rlimited process against its own parent.
        send_frame(
            self._conn,
            {"operation_seq": self._seq, "capability": capability, "payload": payload},
            max_bytes=None,
        )
        reply = recv_frame(self._conn, max_bytes=None)
        if not reply.get("ok"):
            code = reply.get("code", "failed")
            raise BrokerRefused(str(code), str(reply.get("error") or capability))
        result = reply.get("result")
        return result if isinstance(result, dict) else {}


class WorkerLimitsUnavailable(RuntimeError):
    """A declared rlimit could not be applied, so the body must not run."""


def _apply_limits(limits: Dict[str, int]) -> None:
    """SPEC §18: memory hard cap, CPU seconds, max file size, no core dumps.

    Applied with the standard library rather than by importing the sandbox
    module. The numbers are still the parent's — one `SandboxConfig` decides
    them — but a child that imported the service layer to read them would pay
    for exactly the imports this process exists to do without.

    **Fails closed.** These are the hard caps SPEC declares, and a wall-clock
    kill is not a substitute for an address-space or file-size cap: it stops a
    slow worker, not one that allocates 40GB in a second or fills the disk. A
    limit the platform refuses means this process cannot honour the contract it
    is running under, so it refuses to run the body at all.
    """
    import resource

    caps = (
        ("memory_bytes", resource.RLIMIT_AS, limits.get("memory_bytes")),
        ("cpu_seconds", resource.RLIMIT_CPU, limits.get("cpu_seconds")),
        ("file_size_bytes", resource.RLIMIT_FSIZE, limits.get("file_size_bytes")),
        ("core", resource.RLIMIT_CORE, 0),
    )
    for name, which, value in caps:
        if value is None:
            continue
        try:
            resource.setrlimit(which, (value, value))
        except (ValueError, OSError) as exc:
            raise WorkerLimitsUnavailable(
                f"the {name} rlimit could not be applied ({exc}); refusing to "
                "run a tool body without the caps SPEC §18 declares"
            ) from exc


def _confine(scratch: str) -> None:
    """Remove this process's ambient authority before any body runs.

    The worker is designated as the untrusted side of the boundary, and until
    this ran it was untrusted in name only: `multiprocessing` spawn inherits
    the service's environment, filesystem view and network namespace, so the
    deployment's database url and provider keys, every host path, and an
    outbound socket were all still available to it. The broker being its
    *intended* channel is not the same as the broker being its *only* channel
    — one bug in a body is the difference, and that is exactly the bug this
    process exists to contain.

    Same mechanism `run_python` uses (service/confine.py) and the same rule:
    no degraded fallback. A platform with no backend cannot give a worker the
    view SPEC §18 describes, so it does not get a worker.
    """
    from liminallm.service.confine import confine

    # `scratch` is a directory the parent made and registered against the
    # invocation, so both of these go away with it. The new root must be a
    # sibling of the view rather than a child: the view is bind-mounted into
    # the root, and a mount point inside it would be bound into itself.
    view = os.path.join(scratch, "view")
    root = os.path.join(scratch, "root")
    os.makedirs(view, exist_ok=True)
    os.makedirs(root, exist_ok=True)
    # The pipe is an already-connected AF_UNIX socketpair, so it survives the
    # new network namespace and the new root: the broker channel is the one
    # thing confinement must not take away.
    confined = confine(view, root=root)
    # Inherited at process start and living in memory, so re-rooting does
    # nothing to it. Replaced wholesale, not filtered: a denylist of secret
    # names is a guess about what the deployment exported.
    os.environ.clear()
    os.environ.update(
        {
            "HOME": confined,
            "TMPDIR": confined,
            "PWD": confined,
            "PATH": "",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )


def _worker_main(
    conn: Any,
    tool: str,
    plan: Dict[str, Any],
    limits: Dict[str, int],
    scratch: str,
) -> None:
    """Child entrypoint. Leads a process group, confines itself, runs one tool.

    `os.setsid` is the first statement for a reason: everything after it,
    including anything a tool body spawns, belongs to a group the parent can
    take down with one signal. Doing it later leaves a window in which a
    descendant is already running and is not yet reachable.

    The READY message is the second, and it is what makes the group safe to
    use. Until the parent has seen a pgid equal to this pid, this process is
    still in the *parent's* group, and a `killpg` aimed at it would take down
    the API server. The parent kills by single pid until READY arrives.

    Everything this process says is JSON (service/wire.py). SPEC §18 names it
    the untrusted half of the boundary, and a pickle would have let it choose
    which class the *broker* constructs while decoding — before the broker
    could check anything about the message.
    """
    from liminallm.service.wire import WireError, send_frame

    try:
        os.setsid()
    except (AttributeError, OSError):
        # No sessions on this platform, or already a group leader. The pgid
        # reported below will not equal our pid, so the parent keeps killing by
        # single pid — the group shortcut is lost, nothing else is.
        pass
    try:
        send_frame(conn, {"ready": True, "pid": os.getpid(), "pgid": os.getpgid(0)})
    except (BrokenPipeError, OSError, WireError):
        return

    result: Optional[Dict[str, Any]] = None
    try:
        # Warm what the body may reach for lazily: after confinement the
        # project's own tree is gone from this process's view, and the runtime
        # is read-only at its original paths.
        import resource  # noqa: F401 - imported for its side effect

        if not scratch:
            # No conditional confinement: a caller that forgot the scratch
            # would otherwise get an unconfined worker, which is the whole
            # defect stated as a default argument.
            raise RuntimeError("a tool worker was started with no scratch to confine to")
        _confine(scratch)
        _apply_limits(limits)
    except BaseException as exc:  # noqa: BLE001 - refuse, never run uncontained
        result = {
            "status": "error",
            "content": (
                "the tool worker could not establish the boundary it runs "
                f"under, so it ran nothing: {exc}"
            ),
            "error": "worker_unconfined",
        }

    if result is None:
        broker = BrokerClient(conn)
        try:
            body = _BODIES.get(tool, _body_host_tool)
            result = body(broker, tool, plan)
        except BrokerRefused as exc:
            result = {
                "status": "error",
                "content": f"tool capability refused: {exc}",
                "error": exc.code,
            }
        except BaseException as exc:  # noqa: BLE001 - report, never crash silently
            result = {
                "status": "error",
                "content": "tool execution failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
    try:
        send_frame(conn, {"done": True, "result": result}, max_bytes=None)
    except WireError as exc:
        # Not representable as data — a body returned an object. Say so in a
        # frame that is, rather than letting the broker read a silence it
        # would report as a dead worker.
        try:
            send_frame(
                conn,
                {
                    "done": True,
                    "result": {
                        "status": "error",
                        "content": f"the tool produced a result it could not return: {exc}",
                        "error": "result_unreturnable",
                    },
                },
                max_bytes=None,
            )
        except (BrokenPipeError, OSError, WireError):
            pass
    except (BrokenPipeError, OSError):
        pass
    finally:
        conn.close()


# -- tool bodies ------------------------------------------------------------
#
# A body is pure control flow. It computes, it decides, and it asks the broker
# for anything with an effect. Nothing here reads a setting, opens a socket or
# touches the store, because nothing here could.


def _body_host_tool(
    broker: BrokerClient, tool: str, plan: Dict[str, Any]
) -> Dict[str, Any]:
    """One request for a tool whose body still runs in the parent."""
    return broker.call("tool.host", {"tool": tool, "inputs": plan.get("inputs") or {}})


def _body_web_search(
    broker: BrokerClient, _tool: str, plan: Dict[str, Any]
) -> Dict[str, Any]:
    inputs = plan.get("inputs") or {}
    query = inputs.get("query") or inputs.get("message") or plan.get("message") or ""
    out = broker.call("web.search", {"query": query, "limit": inputs.get("limit") or 5})
    return {
        "content": out.get("text") or "",
        "usage": {},
        "injection_findings": out.get("findings") or [],
    }


def _body_web_fetch(
    broker: BrokerClient, _tool: str, plan: Dict[str, Any]
) -> Dict[str, Any]:
    inputs = plan.get("inputs") or {}
    url = inputs.get("url") or ""
    if not url:
        return {"status": "error", "content": "no url supplied", "usage": {}}
    out = broker.call("web.fetch", {"url": str(url)})
    return {
        "content": out.get("text") or "",
        "usage": {},
        "injection_findings": out.get("findings") or [],
    }


def _body_file_search(
    broker: BrokerClient, _tool: str, plan: Dict[str, Any]
) -> Dict[str, Any]:
    inputs = plan.get("inputs") or {}
    query = inputs.get("query") or inputs.get("message") or plan.get("message") or ""
    out = broker.call(
        "rag.retrieve", {"query": query, "limit": inputs.get("limit") or 4}
    )
    return {
        "content": out.get("text") or "",
        "usage": {},
        "context_snippets": out.get("snippets") or [],
    }


def _body_notes_search(
    broker: BrokerClient, _tool: str, plan: Dict[str, Any]
) -> Dict[str, Any]:
    inputs = plan.get("inputs") or {}
    query = inputs.get("query") or inputs.get("message") or plan.get("message") or ""
    out = broker.call(
        "notes.search", {"query": query, "limit": inputs.get("limit") or 6}
    )
    return {"content": out.get("text") or "", "usage": {}}


def _body_python(
    broker: BrokerClient, _tool: str, plan: Dict[str, Any]
) -> Dict[str, Any]:
    inputs = plan.get("inputs") or {}
    code = str(inputs.get("code") or "")
    if not code:
        return {"status": "error", "content": "no code supplied", "usage": {}}
    out = broker.call("python.run", {"code": code})
    return {
        "content": out.get("text") or "",
        "usage": {},
        "artifacts": out.get("artifacts") or [],
    }


def _body_agent_loop(
    broker: BrokerClient, _tool: str, plan: Dict[str, Any]
) -> Dict[str, Any]:
    """Multi-round, model-driven tool use — the reason this process exists.

    Everything the model decides happens on this side: which tool, with what
    arguments, how many times. Everything that touches the world happens on the
    other. That split is the containment — a page that talks the model into
    calling the interpreter reaches a broker that has already withdrawn it.

    A round's calls go over as one request rather than one each. The parent
    decides how to run them (reads in parallel, anything that can taint the
    turn strictly in order), which is a decision about effects and therefore
    the parent's; and it keeps one pipe carrying one request at a time.
    """
    messages: List[dict] = list(plan.get("messages") or [])
    tools: List[dict] = list(plan.get("tools") or [])
    max_rounds = int(plan.get("max_rounds") or 3)
    deadline = time.monotonic() + float(plan.get("deadline_seconds") or 45.0)
    fallback_query = str(plan.get("message") or "")
    # The streaming caller wants the answer token by token, and the final round
    # offers no tools — there is no model-chosen control flow left in it to
    # contain. So the worker stops after the tool rounds and hands back the
    # conversation it built; the parent streams the last turn from there.
    stream_final = bool(plan.get("stream_final"))
    tool_rounds = max_rounds - 1 if stream_final else max_rounds

    # Seeded, not empty: the parent retrieved for an explicitly selected
    # knowledge context and already put those chunks in the prompt. Returning
    # them here is what makes the turn report the grounding it actually used,
    # rather than only what a tool call went and fetched.
    snippets: List[str] = list(plan.get("context_snippets") or [])
    artifacts: List[str] = []
    findings: List[str] = []
    trace: List[dict] = []
    usage: Dict[str, Any] = {}
    content = ""

    for round_index in range(max(0, tool_rounds)):
        # Withhold the tools on the last permitted round, or once the budget is
        # spent, so the model has to produce a final answer.
        out_of_time = time.monotonic() > deadline
        last_round = not stream_final and round_index == tool_rounds - 1
        offer = [] if (out_of_time or last_round) else tools
        response = broker.call(
            "llm.generate_with_tools", {"messages": messages, "tools": offer}
        )
        for key, value in (response.get("usage") or {}).items():
            if isinstance(value, int):
                usage[key] = usage.get(key, 0) + value
        calls = response.get("tool_calls") or []
        if not calls:
            # The model answered without tools. When the parent will stream the
            # answer, its message stays out of the history so the streamed turn
            # produces the text rather than repeating it.
            if not stream_final:
                content = response.get("content") or content
            break
        content = response.get("content") or content
        messages.append(
            response.get("assistant_message")
            or {"role": "assistant", "content": content}
        )
        parsed = [
            {"id": call.get("id") or "", "name": str(call.get("name") or ""),
             "arguments": _parse_arguments(call)}
            for call in calls
        ]
        out = broker.call(
            "tools.round", {"calls": parsed, "fallback_query": fallback_query}
        )
        results = out.get("results") or []
        snippets.extend(out.get("snippets") or [])
        artifacts.extend(out.get("artifacts") or [])
        findings.extend(out.get("findings") or [])
        for call, result in zip(parsed, results):
            trace.append({"tool": call["name"], "arguments": call["arguments"]})
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"] or call["name"],
                    "name": call["name"],
                    "content": result,
                }
            )

    result = {
        "content": content or "I could not derive an answer from the available sources.",
        "usage": usage,
        "context_snippets": snippets,
        "tool_calls": trace,
        "artifacts": artifacts,
        "injection_findings": findings,
    }
    if stream_final:
        # The parent finishes the turn, so it needs the conversation as built.
        result["messages"] = messages
        result["content"] = content
    return result


def _parse_arguments(call: Dict[str, Any]) -> Dict[str, Any]:
    try:
        args = json.loads(call.get("arguments") or "{}")
    except (TypeError, ValueError):
        return {}
    return args if isinstance(args, dict) else {}


_BODIES: Dict[str, Callable[[BrokerClient, str, Dict[str, Any]], Dict[str, Any]]] = {
    "agent.files_v1": _body_agent_loop,
    "code.python_v1": _body_python,
    "web.search_v1": _body_web_search,
    "web.fetch_v1": _body_web_fetch,
    "file.search_v1": _body_file_search,
    "notes.search_v1": _body_notes_search,
}

#: Tools whose body runs in the worker. The parent consults this to resolve a
#: `tool.spec` handler alias, so an aliased tool reaches its body rather than
#: falling through to a host map the body was moved out of.
BODY_NAMES = frozenset(_BODIES)


# ---------------------------------------------------------------------------
# parent side
# ---------------------------------------------------------------------------


#: Headroom on top of what the parent has handed the worker, for one model
#: turn and the structure around it. Everything a body returns is made of
#: bytes the parent supplied — the plan, then each broker reply — so the
#: parent's own outbound total is the natural budget, and this is what it does
#: not account for: MAX_GENERATION_TOKENS of new text, plus keys and nesting.
WORKER_FRAME_ALLOWANCE_BYTES = 1024 * 1024


class FrameBudget:
    """What the worker may say, measured in what it has been told.

    A worker is a spawned process with a memory rlimit of its own, and without
    a cap it can spend all of that inside the API process by answering with it.
    A fixed cap would be a guess about conversation sizes; this is not a guess.
    The agent loop returns the conversation it was given plus the broker's
    replies, so the parent grants exactly its own outbound bytes and an
    allowance for the model's new text.

    Parent-side only. The worker holds no copy: two caps on one frame is two
    numbers to keep in step, and the child's would drift below the real one as
    the broker answered, refusing frames the broker would have taken.
    """

    def __init__(self, initial: int) -> None:
        self._allowed = max(0, initial) + WORKER_FRAME_ALLOWANCE_BYTES

    def credit(self, sent: int) -> None:
        """Record bytes handed to the worker; it may hand them back."""
        self._allowed += max(0, sent)

    @property
    def limit(self) -> int:
        return self._allowed


class WorkerHandle:
    """The parent's grip on one worker process."""

    def __init__(
        self,
        process: Any,
        conn: Any,
        attempt: Attempt,
        budget: FrameBudget,
        *,
        leads_group: bool = False,
    ) -> None:
        self._process = process
        self.conn = conn
        self.attempt = attempt
        #: How large a frame this worker may send, grown as the broker answers.
        self.budget = budget
        #: Whether READY proved this worker leads a process group of its own.
        #: Carried here because teardown needs it on *every* path, and only
        #: `spawn` was in a position to learn it.
        self.leads_group = leads_group
        #: Set by `terminate()` once the leader is positively reaped, whether
        #: or not its group has finished draining. The caller needs the two
        #: apart: "not gone" says keep the registration, and this says keep it
        #: as a group id rather than as a pid.
        self.leader_reaped = False

    @property
    def pid(self) -> Optional[int]:
        return self._process.pid

    def is_alive(self) -> bool:
        return bool(self._process.is_alive())

    def terminate(self) -> bool:
        """End this worker and everything it started; True once nothing is left.

        The group goes first and the reap second. Succeeding is not an
        exception to that: a tool that starts a helper and then answers
        normally leaves it running, and the invocation forgets the worker one
        line later, so the helper becomes nobody's. SPEC §18 has no clause
        about how the worker finished — "what the invocation started, the
        invocation can kill".

        `Process.is_alive()` is deliberately not consulted first. It joins an
        exited child, and a reaped pid is a number the kernel may hand to
        anyone — the group has to be signalled while that pid still names it.

        **The return value is the point.** §18 says "reaping is confirmed
        rather than bounded: a tree that will not die fails the node instead of
        running alongside its successor", and a bounded `join` confirms
        nothing. A caller that drops this worker's registration on a `False`
        deletes the only evidence `Invocation.terminate()` has to refuse the
        retry with.
        """
        from liminallm.service.invocation import group_alive

        pid = self._process.pid
        if self.leads_group and pid and hasattr(os, "killpg"):
            try:
                # A group kill is only ever aimed at a group the target leads.
                # READY already proved that; this re-checks it against the
                # kernel, because the alternative to being wrong here is
                # signalling the process group of the API server.
                if os.getpgid(pid) == pid:
                    os.killpg(pid, signal.SIGKILL)
            except OSError:
                pass  # already gone, or not ours to signal
        if self._process.is_alive():
            self._process.kill()
        self._process.join(_JOIN_TIMEOUT_SECONDS)
        try:
            self.conn.close()
        except OSError:
            pass
        # `exitcode` rather than a pid probe: it is None until the child has
        # actually been reaped, and it cannot be confused by a pid the kernel
        # has since given to somebody else.
        if self._process.exitcode is None:
            return False
        # Recorded because it changes what the pid *means*. A reaped leader's
        # number is the kernel's to reissue, so a caller that keeps this
        # registration has to keep it as a group and not as a process.
        self.leader_reaped = True
        # The leader being reaped is not the group being empty. A killed
        # member is still in the group until its parent reaps it, and once the
        # leader is gone that parent is init — measured here, a group outlives
        # its reaped leader by about a second.
        #
        # So this reports and does not wait. Waiting would put that second on
        # every tool call, and the polling with a deadline already exists one
        # level up: `Invocation.terminate()` is what SPEC charges with telling
        # "draining" from "will not die", and it can only do that if this
        # answer is honest and the registration is still there to re-check.
        return not (self.leads_group and pid and group_alive(pid))


def limits_from_config(config: Any) -> Dict[str, int]:
    """The rlimits a worker runs under, taken off the tool's SandboxConfig."""
    return {
        "memory_bytes": int(config.max_memory_mb) * 1024 * 1024,
        "cpu_seconds": int(config.max_cpu_seconds),
        "file_size_bytes": int(config.max_file_size_mb) * 1024 * 1024,
    }


def spawn(
    invocation: Invocation,
    tool: str,
    plan: Dict[str, Any],
    *,
    limits: Optional[Dict[str, int]] = None,
    scratch: Callable[[], str],
    expected_attempt: Any = None,
    on_started: Optional[Callable[[], None]] = None,
) -> WorkerHandle:
    """Start a worker for one attempt at one tool call.

    The child is registered against the invocation before anything is awaited,
    so a revoke arriving one instruction later still reaches it — but as a
    **single pid**, not a group. `os.setsid()` runs in the child and cannot
    have happened yet when `start()` returns, so `os.getpgid(child)` still
    answers with the *parent's* group: a `killpg` in that window would SIGKILL
    the API server and everything sharing its group. Measured, not reasoned
    about — `getpgid` on a just-started spawn child returns the parent's pgid.

    So the group is earned, not assumed. The child sends READY once `setsid`
    has actually happened, carrying the pgid it ended up in, and only a pgid
    equal to the child's pid promotes the registration to `group=True`.
    """
    from liminallm.logging import get_logger  # parent-side: see module docstring

    logger = get_logger(__name__)
    # spawn, not fork: forking a threaded API process risks deadlocks, and a
    # fresh interpreter holding none of the parent's handles is the boundary.
    ctx = multiprocessing.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=True)
    budget = FrameBudget(_plan_bytes(plan))
    # One linearization boundary (SPEC §18.3): validate the exact attempt
    # this spawn was created for, allocate its scratch, start the child,
    # register it, and only then transfer ownership. A revoke serializes
    # against this block — it lands before, and the adoption refuses with
    # nothing allocated and no child started, or it lands after and its
    # sweep finds the registered child. The scratch is a factory called
    # *after* the adoption for exactly that reason: a stale serve waking
    # once the invocation has closed must be refused before it creates
    # filesystem state nobody is left to remove. Ownership is transferred
    # last, and `on_started` — the breaker's record that the worker really
    # started, one attribute write, no-throw — fires inside the same block:
    # marked after `spawn` returns, a worker killed during the READY
    # handshake below died registered but never `started`.
    try:
        with invocation.lock:
            attempt = invocation.adopt_attempt(expected_attempt)
            scratch_path = scratch()
            process = ctx.Process(
                target=_worker_main,
                args=(child_conn, tool, plan, limits or {}, scratch_path),
                daemon=True,
            )
            reap = lambda: process.join(_JOIN_TIMEOUT_SECONDS)  # noqa: E731
            process.start()
            child_conn.close()
            attempt.pid = process.pid
            invocation.resources.add_child(
                process.pid or 0, f"worker:{tool}", group=False, reap=reap
            )
            attempt.adopted = True
            if on_started is not None:
                on_started()
    except BaseException:
        for conn in (parent_conn, child_conn):
            try:
                conn.close()
            except Exception:  # noqa: BLE001 - already failing
                pass
        raise
    handle = WorkerHandle(process, parent_conn, attempt, budget)
    leads_group = _await_ready(parent_conn, process)
    if leads_group:
        invocation.resources.add_child(
            process.pid or 0, f"worker:{tool}", group=True, reap=reap
        )
    # The registry learns this for revocation; the handle learns it for every
    # other way the attempt can end, ordinary completion included.
    handle.leads_group = leads_group
    logger.info(
        "tool_worker_spawned",
        invocation_id=invocation.invocation_id,
        tool=tool,
        pid=process.pid,
        attempt=attempt.index,
        leads_group=leads_group,
    )
    return handle


def _plan_bytes(plan: Dict[str, Any]) -> int:
    """How much the parent is handing this worker, as encoded bytes."""
    try:
        return len(json.dumps(plan, default=str).encode("utf-8"))
    except (TypeError, ValueError):
        return 0


def _await_ready(conn: Any, process: Any) -> bool:
    """Whether the child has proved it leads a process group of its own.

    False on anything unexpected — a child that died, a pipe that closed, a
    platform with no sessions, a message that is not the handshake. Every one
    of those means the same thing here: keep killing by single pid, because a
    group kill would reach processes this worker does not own.
    """
    from liminallm.service.wire import ERROR_FRAME_BYTES, WireError, recv_frame

    deadline = time.monotonic() + _READY_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if conn.poll(_READY_POLL_SECONDS):
            try:
                message = recv_frame(conn, max_bytes=ERROR_FRAME_BYTES)
            except (EOFError, OSError, WireError):
                return False
            if not message.get("ready"):
                return False
            pgid, pid = message.get("pgid"), message.get("pid")
            # The pgid must be the child's own pid. Anything else is a group it
            # shares with someone — the parent's, most likely — and killing it
            # would take them down too.
            return bool(pgid) and pgid == pid == process.pid
        if not process.is_alive():
            return False
    return False
