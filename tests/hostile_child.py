"""Payloads a compromised child could send, as module-level importable names.

Kept out of the test module so `multiprocessing` spawn can resolve them by
reference in a fresh interpreter, which is exactly how a real child would
return them.

Nothing here is exotic. `__reduce__` is the documented pickle protocol, and
`Connection.recv()` runs it — the point of these fixtures is that a hostile
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
    """The 'malicious' operation. Deliberately harmless — it writes its pid."""
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

    Models the parser grandchildren §19.5 describes — `pdftoppm`, tesseract —
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


def returns_plain_data() -> dict:
    """The ordinary case, so a refusal above is not just a broken pipe."""
    return {"ok": True, "items": ["a", "b"], "count": 2}
