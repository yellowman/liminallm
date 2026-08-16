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
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:  # parent-side only; `from __future__ import annotations`
    from liminallm.service.invocation import Attempt, Invocation

#: How long the parent waits for a killed worker to be joined. The worker has
#: nothing to flush, so this is short by design.
_JOIN_TIMEOUT_SECONDS = 2.0


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
        self._seq += 1
        self._conn.send(
            {"operation_seq": self._seq, "capability": capability, "payload": payload}
        )
        reply = self._conn.recv()
        if not isinstance(reply, dict) or not reply.get("ok"):
            code = (reply or {}).get("code", "failed")
            raise BrokerRefused(code, str((reply or {}).get("error") or capability))
        return reply.get("result") or {}


def _apply_limits(limits: Dict[str, int]) -> None:
    """SPEC §18: memory hard cap, CPU seconds, max file size, no core dumps.

    Applied with the standard library rather than by importing the sandbox
    module. The numbers are still the parent's — one `SandboxConfig` decides
    them — but a child that imported the service layer to read them would pay
    for exactly the imports this process exists to do without.
    """
    import resource

    caps = (
        (resource.RLIMIT_AS, limits.get("memory_bytes")),
        (resource.RLIMIT_CPU, limits.get("cpu_seconds")),
        (resource.RLIMIT_FSIZE, limits.get("file_size_bytes")),
        (resource.RLIMIT_CORE, 0),
    )
    for which, value in caps:
        if value is None:
            continue
        try:
            resource.setrlimit(which, (value, value))
        except (ValueError, OSError):
            # A limit the platform refuses is not a reason to run unbounded
            # work unannounced; the parent's wall-clock kill still backstops it.
            pass


def _worker_main(
    conn: Any, tool: str, plan: Dict[str, Any], limits: Dict[str, int]
) -> None:
    """Child entrypoint. Leads a process group, runs one tool, exits.

    `os.setsid` is the first statement for a reason: everything after it,
    including anything a tool body spawns, belongs to a group the parent can
    take down with one signal. Doing it later leaves a window in which a
    descendant is already running and is not yet reachable.
    """
    try:
        os.setsid()
    except (AttributeError, OSError):
        # No sessions on this platform, or already a group leader. The parent's
        # per-child registry still reaches everything the broker started; only
        # the group shortcut is lost.
        pass
    _apply_limits(limits)
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
        conn.send({"done": True, "result": result})
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

    snippets: List[str] = []
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


class WorkerHandle:
    """The parent's grip on one worker process."""

    def __init__(self, process: Any, conn: Any, attempt: Attempt) -> None:
        self._process = process
        self.conn = conn
        self.attempt = attempt

    @property
    def pid(self) -> Optional[int]:
        return self._process.pid

    def is_alive(self) -> bool:
        return bool(self._process.is_alive())

    def terminate(self) -> None:
        """Kill this worker and reap it. Safe to call more than once."""
        if self._process.is_alive():
            self._process.kill()
        self._process.join(_JOIN_TIMEOUT_SECONDS)
        try:
            self.conn.close()
        except OSError:
            pass


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
) -> WorkerHandle:
    """Start a worker for one attempt at one tool call.

    The child is registered against the invocation before anything is awaited,
    and as a process-group leader, so a revoke arriving one instruction later
    still reaches it and everything it has spawned since.
    """
    from liminallm.logging import get_logger  # parent-side: see module docstring

    logger = get_logger(__name__)
    attempt = invocation.begin_attempt()
    # spawn, not fork: forking a threaded API process risks deadlocks, and a
    # fresh interpreter holding none of the parent's handles is the boundary.
    ctx = multiprocessing.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=True)
    process = ctx.Process(
        target=_worker_main,
        args=(child_conn, tool, plan, limits or {}),
        daemon=True,
    )
    process.start()
    child_conn.close()
    attempt.pid = process.pid
    handle = WorkerHandle(process, parent_conn, attempt)
    invocation.resources.add_child(
        process.pid or 0,
        f"worker:{tool}",
        group=True,
        reap=lambda: process.join(_JOIN_TIMEOUT_SECONDS),
    )
    logger.info(
        "tool_worker_spawned",
        invocation_id=invocation.invocation_id,
        tool=tool,
        pid=process.pid,
        attempt=attempt.index,
    )
    return handle
