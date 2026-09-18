"""Shutdown must not report a background task cancelled while its thread runs.

`asyncio.to_thread` hands work to a thread and makes only the *await*
cancellable. Cancelling it returns at once and leaves the thread running, so
the lifespan's `_cleanup_task.cancel()` followed by `await _cleanup_task`
completed while a sweep was still deleting directories and writing logs.

Found as an interpreter crash, not by reading. Under `make test-xdist` a
worker died with `Fatal Python error: Segmentation fault`, and the faulting
thread was a `to_thread` worker inside `sweep_artifact_payloads` logging
through structlog while the main thread rebuilt pytest's per-test capture.
`PrintLogger` resolves `sys.stdout` at write time and calls `print(...,
flush=True)`; a concurrent write and close on a buffered file object is a
segfault in CPython, not an exception.

The production consequence is the same defect without pytest: a worker asked
to stop returns from shutdown while a thread is still reclaiming payload
directories, so whatever kills the process next interrupts a deletion that the
process already reported finishing.

These witnesses measure the straggler directly, which is deterministic. They
do not measure the segfault, which is a rare consequence of it - the crash is
what made the straggler visible, not what these tests reproduce.

Both lifespan loops are covered, because the shape is the defect rather than
the caller: the cleanup loop and the settings watcher dispatch the same way
and are cancelled the same way. Request-path `to_thread` calls are out of
scope - there cancellation means the client hung up, and abandoning the thread
is the intended answer.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
from pathlib import Path

import pytest

from liminallm.service.runtime import get_runtime

#: Long enough that a shutdown which does not wait is measured at zero against
#: it, short enough to cost nothing when it does wait.
RELEASE_AFTER_SECONDS = 0.5

#: The blocked thread's own bound, so a defect here cannot hang the suite.
THREAD_LIMIT_SECONDS = 30


class _BlockingWork:
    """Work that is demonstrably still running when cancellation lands."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()

    def __call__(self, *args, **kwargs) -> int:
        self.started.set()
        self.release.wait(THREAD_LIMIT_SECONDS)
        self.finished.set()
        return 0

    async def in_flight(self) -> None:
        """Wait until the thread is running, and prove it is still running.

        Without this the final assertion is satisfied by work that simply
        finished early, which would make the witness vacuous.
        """
        assert await asyncio.to_thread(self.started.wait, THREAD_LIMIT_SECONDS), (
            "the work never reached its thread, so nothing was cancelled "
            "mid-flight and this probe cannot observe what it exists to test"
        )
        assert not self.finished.is_set(), (
            "the work finished before cancellation, so a finished state "
            "afterwards would prove nothing about waiting"
        )

    def release_shortly(self) -> None:
        timer = threading.Timer(RELEASE_AFTER_SECONDS, self.release.set)
        timer.daemon = True
        timer.start()


async def _cancel_and_wait(task) -> None:
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


@pytest.fixture
def blocking():
    work = _BlockingWork()
    yield work
    work.release.set()


@pytest.fixture
def sweep(monkeypatch, blocking):
    """Replace the artifact sweep, which `_run_cleanup_pass` imports by name."""
    from liminallm.service import artifacts as artifacts_module

    monkeypatch.setattr(artifacts_module, "sweep_artifact_payloads", blocking)
    return blocking


@pytest.mark.asyncio
async def test_a_cancelled_pass_waits_for_the_thread_it_started(sweep):
    """One pass, cancelled while a sweep is on a thread."""
    from liminallm.app import _run_cleanup_pass

    runtime = get_runtime()
    root = Path(runtime.settings.shared_fs_root)
    task = asyncio.create_task(_run_cleanup_pass(runtime, root, 24))

    await sweep.in_flight()
    sweep.release_shortly()
    await _cancel_and_wait(task)

    assert sweep.finished.is_set(), (
        "the pass reported itself cancelled while its thread was still "
        "running, so shutdown returns with a sweep mid-flight"
    )


@pytest.mark.asyncio
async def test_cancelling_the_cleanup_loop_waits_too(sweep):
    """The loop is what the lifespan actually cancels.

    Cancelling the pass directly is the unit; cancelling the loop is the thing
    that happens at shutdown.
    """
    from liminallm.app import _run_tmp_cleanup

    runtime = get_runtime()
    root = Path(runtime.settings.shared_fs_root)
    task = asyncio.create_task(_run_tmp_cleanup(root, 300, 24))

    await sweep.in_flight()
    sweep.release_shortly()
    await _cancel_and_wait(task)

    assert sweep.finished.is_set(), (
        "the cleanup loop reported itself cancelled while its thread was "
        "still running, which is exactly what the lifespan awaits"
    )


@pytest.mark.asyncio
async def test_cancelling_the_settings_watcher_waits_too(monkeypatch, blocking):
    """The sibling, found by looking for the shape rather than the report.

    `_run_settings_watcher` is the lifespan's other task and dispatches the
    same way, so a shutdown that waits for one and not the other still returns
    with a thread rebuilding the model stack and writing logs.
    """
    from liminallm.app import _run_settings_watcher

    runtime = get_runtime()
    monkeypatch.setattr(runtime, "maybe_reload_model_services", blocking)
    task = asyncio.create_task(_run_settings_watcher(runtime, 1))

    await blocking.in_flight()
    blocking.release_shortly()
    await _cancel_and_wait(task)

    assert blocking.finished.is_set(), (
        "the settings watcher reported itself cancelled while its reload "
        "thread was still running"
    )


@pytest.mark.asyncio
async def test_an_uncancelled_pass_still_finishes(sweep):
    """The other direction: waiting must not become never returning."""
    from liminallm.app import _run_cleanup_pass

    runtime = get_runtime()
    root = Path(runtime.settings.shared_fs_root)
    task = asyncio.create_task(_run_cleanup_pass(runtime, root, 24))

    await sweep.in_flight()
    sweep.release.set()
    await asyncio.wait_for(task, timeout=THREAD_LIMIT_SECONDS)

    assert sweep.finished.is_set()
