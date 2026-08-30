"""Payloads a compromised child could send, as module-level importable names.

Kept out of the test module so `multiprocessing` spawn can resolve them by
reference in a fresh interpreter, which is exactly how a real child would
return them.

Nothing here is exotic. `__reduce__` is the documented pickle protocol, and
`Connection.recv()` runs it - the point of these fixtures is that a hostile
child needs no exploit, only the ability to return an object.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Where the payload records the pid it managed to execute in. Set by the test
#: before the child runs, read afterwards; a file rather than memory because
#: the whole question is which process the code ran in.
MARKER_ENV = "LIMINALLM_PWNED_MARKER"


def record_execution() -> str:
    """The 'malicious' operation. Deliberately harmless - it writes its pid."""
    marker = os.environ.get(MARKER_ENV)
    if marker:
        Path(marker).write_text(str(os.getpid()))
    return "harmless-looking string"


class Evil:
    """An object whose *deserialization* is the attack."""

    def __reduce__(self):
        return (record_execution, ())


def returns_evil() -> object:
    """A sandbox body that hands one back."""
    return Evil()


def raises_evil() -> None:
    """A sandbox body that raises one, for the error channel."""

    class _EvilError(Exception):
        def __reduce__(self):
            return (record_execution, ())

    raise _EvilError("boom")


def spawns_a_survivor() -> dict:
    """Start a long-lived grandchild, report its pid, then block forever.

    Models the parser grandchildren §19.5 describes - `pdftoppm`, tesseract -
    which the wall-clock kill has to reach.
    """
    import subprocess
    import sys
    import time

    child = subprocess.Popen(  # noqa: S603 - fixed argv, test-only
        [sys.executable, "-c", "import time; time.sleep(300)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    marker = os.environ.get(MARKER_ENV)
    if marker:
        Path(marker).write_text(str(child.pid))
    time.sleep(300)
    return {"unreachable": True}


def returns_a_big_payload() -> dict:
    """A child that answers with far more than its caller allowed for."""
    return {"text": "A" * (4 * 1024 * 1024)}


class MadeUpError(Exception):
    """A type the parent has never heard of, named on the error channel."""


def raises_an_unknown_type() -> None:
    raise MadeUpError("a type the caller did not allow for")


#: The tool name `body_that_leaves_a_helper_behind` answers to.
WORKER_BODY_TOOL = "test.leaves_a_helper_v1"


def body_that_leaves_a_helper_behind(_broker, _tool, _plan) -> dict:
    """A worker body that spawns into its group and then succeeds normally.

    Not hostile, and that is the point: a tool that shells out and *then
    answers* is the ordinary case, and it is the one the timeout and
    revocation paths never see.

    `fork`, not `exec`, and the reason is worth recording: measured, a
    confined worker cannot exec anything at all here. `confine` binds the
    realpaths of the runtime, which on a merged-`/usr` system are `/usr/lib`
    and `/usr/lib64`, so the new root has no `/lib64` - and `python3`'s ELF
    interpreter is `/lib64/ld-linux-x86-64.so.2`. `execve` finds the file and
    the kernel then fails on the loader, which surfaces as `FileNotFoundError`
    for a path that exists. `fork` needs none of that, and the child it makes
    is in the worker's process group just the same.

    The helper's pid comes back in the result rather than through a file: by
    the time this runs the process is confined, so a path from the parent's
    filesystem is not somewhere it can write.
    """
    import time

    pid = os.fork()
    if pid == 0:  # pragma: no cover - the helper, in its own process
        try:
            time.sleep(300)
        finally:
            os._exit(0)
    return {
        "content": "done, and something of mine is still running",
        "helper_pid": pid,
    }


#: Same, for a body that fails - a retry only happens after one does.
FAILING_WORKER_BODY_TOOL = "test.fails_leaving_a_helper_v1"


def body_that_fails_leaving_a_helper_behind(broker, tool, plan) -> dict:
    """Leaves a descendant and then reports failure, so a retry follows."""
    body_that_leaves_a_helper_behind(broker, tool, plan)
    return {"status": "error", "content": "no", "error": "boom"}


def register_worker_body() -> None:
    """Put the body in the worker's table, in whichever process imports this.

    The child builds `_BODIES` when it imports `tool_worker`, so a
    registration made in the parent does not survive the spawn. What does
    survive is an import: `multiprocessing` pickles a function by reference,
    so putting `body_that_leaves_a_helper_behind` in the plan makes the child
    import this module while unpickling its arguments - before `_worker_main`
    runs - and this line is the side effect that matters.
    """
    from liminallm.service import tool_worker

    tool_worker._BODIES[WORKER_BODY_TOOL] = body_that_leaves_a_helper_behind
    tool_worker._BODIES[FAILING_WORKER_BODY_TOOL] = (
        body_that_fails_leaving_a_helper_behind
    )


register_worker_body()


def returns_plain_data() -> dict:
    """The ordinary case, so a refusal above is not just a broken pipe."""
    return {"ok": True, "items": ["a", "b"], "count": 2}
