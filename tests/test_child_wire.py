"""What crosses a pipe from a process we assume is hostile.

Two boundaries in this codebase are declared untrusted on the child side. SPEC
§18 makes the tool worker the untrusted half of the broker boundary — it "runs
model-chosen control flow over attacker-controlled bytes". §19.5 says the
extraction child is where parsers run because "assume the parsers are
compromisable".

Both used `multiprocessing.Connection.send()`/`recv()`, and `recv()` unpickles.
Python's own documentation warns that this is unsafe from an untrusted peer:
unpickling runs `__reduce__`, so the dangerous operation happens **in the
parent**, while it is decoding — before any check the parent might make. That
collapses both boundaries: the decoder is a route back into the API process,
and "compromise stays in the disposable child" stops being true.

So these tests assert on *where code ran*, not on what was returned.
"""

from __future__ import annotations

import multiprocessing
import os
import pickle
import threading
import time
import uuid
from pathlib import Path

import pytest

from tests import hostile_child
from tests.hostile_child import MARKER_ENV


@pytest.fixture
def marker(tmp_path, monkeypatch):
    """A file the payload writes its pid into, if it ever executes."""
    path = tmp_path / f"pwned-{uuid.uuid4().hex[:8]}"
    monkeypatch.setenv(MARKER_ENV, str(path))
    return path


def _sandbox_config(tmp_path):
    from liminallm.service.sandbox import SandboxConfig

    return SandboxConfig(max_memory_mb=256, max_cpu_seconds=10, scratch_dir=tmp_path)


class TestTheExtractionSandboxDoesNotUnpickleItsChild:
    """§19.5's child is assumed compromisable, so its reply is attacker data."""

    def test_a_returned_payload_never_executes_in_the_parent(
        self, marker, tmp_path
    ):
        from liminallm.service.sandbox import SandboxError, run_in_sandbox

        with pytest.raises((SandboxError, ValueError)):
            run_in_sandbox(
                hostile_child.returns_evil,
                config=_sandbox_config(tmp_path),
                timeout=30,
            )
        assert not marker.exists(), (
            f"the payload executed in pid {marker.read_text()} "
            f"(this process is {os.getpid()})"
        )

    def test_a_raised_payload_never_executes_in_the_parent(self, marker, tmp_path):
        """The error channel is a channel too: exceptions were sent as objects."""
        from liminallm.service.sandbox import SandboxError, run_in_sandbox

        with pytest.raises(Exception):
            run_in_sandbox(
                hostile_child.raises_evil,
                config=_sandbox_config(tmp_path),
                timeout=30,
            )
        assert not marker.exists(), (
            f"the payload executed in pid {marker.read_text()} "
            f"(this process is {os.getpid()})"
        )

    def test_ordinary_data_still_comes_back(self, marker, tmp_path):
        """A refusal above must be about the payload, not a broken pipe."""
        from liminallm.service.sandbox import run_in_sandbox

        result = run_in_sandbox(
            hostile_child.returns_plain_data,
            config=_sandbox_config(tmp_path),
            timeout=30,
        )
        assert result == {"ok": True, "items": ["a", "b"], "count": 2}

    def test_a_real_error_still_reaches_the_caller_as_itself(self, tmp_path):
        """Callers translate specific failures — `extract_text` re-raises
        `ExtractError`, `extract_archive_sandboxed` cleans up on
        `ArchiveExtractionError` — so the type has to survive as data."""
        from liminallm.service.archive import ArchiveExtractionError
        from liminallm.service.sandbox import run_in_sandbox

        src = tmp_path / "junk.zip"
        src.write_bytes(b"not an archive")
        from liminallm.service.archive import extract_archive_sandboxed

        with pytest.raises(ArchiveExtractionError):
            extract_archive_sandboxed(
                str(src), str(tmp_path / "out"), {"allowed_extensions": [".txt"]}
            )


class TestTheBrokerDoesNotUnpickleItsWorker:
    """SPEC §18: the worker is the untrusted side. The serve loop decodes
    whatever it sends, so the decoder is part of the authority boundary."""

    def test_a_hostile_frame_never_executes_in_the_serve_loop(self, marker):
        from liminallm.service.broker import CapabilityBroker, InvocationContext
        from liminallm.service.invocation import Invocation

        parent_conn, child_conn = multiprocessing.Pipe(duplex=True)
        # Exactly what a compromised worker would put on the wire.
        child_conn.send_bytes(pickle.dumps(hostile_child.Evil()))

        invocation = Invocation("hostile-worker", tool="agent.files_v1")
        invocation.begin_attempt()
        broker = CapabilityBroker(None, InvocationContext(user_id="u1"))
        try:
            result = broker.serve(
                parent_conn, invocation, is_alive=lambda: False
            )
        finally:
            invocation.close()
            parent_conn.close()
            child_conn.close()

        assert not marker.exists(), (
            f"the payload executed in pid {marker.read_text()} "
            f"(this process is {os.getpid()})"
        )
        assert result.get("status") == "error", result

    def test_a_well_formed_frame_is_still_understood(self, marker):
        """The loop must still work; refusing everything is not a fix."""
        from liminallm.service.broker import CapabilityBroker, InvocationContext
        from liminallm.service.invocation import Invocation
        from liminallm.service.wire import send_frame

        parent_conn, child_conn = multiprocessing.Pipe(duplex=True)
        send_frame(child_conn, {"done": True, "result": {"content": "hello"}})

        invocation = Invocation("good-worker", tool="agent.files_v1")
        invocation.begin_attempt()
        broker = CapabilityBroker(None, InvocationContext(user_id="u1"))
        try:
            result = broker.serve(parent_conn, invocation, is_alive=lambda: True)
        finally:
            invocation.close()
            parent_conn.close()
            child_conn.close()

        assert result == {"content": "hello"}


class TestTheErrorChannelNamesTypesItDoesNotChoose:
    """A type crosses as a name; the parent decides what a name may become.

    The vocabulary is the parent's, so a child cannot ask for a class to be
    constructed — which is the property the old pickle channel had inverted.
    """

    def test_a_type_the_caller_did_not_allow_for_stays_a_sandbox_error(self, tmp_path):
        from liminallm.service.sandbox import SandboxError, run_in_sandbox

        with pytest.raises(SandboxError) as caught:
            run_in_sandbox(
                hostile_child.raises_an_unknown_type,
                config=_sandbox_config(tmp_path),
                timeout=30,
            )
        # The name survives as text, so a log line still says what failed.
        assert "MadeUpError" in str(caught.value)

    def test_a_caller_that_allows_for_a_type_gets_it_back(self, tmp_path):
        """The other half: a caller that translates a failure still can."""
        from liminallm.service.sandbox import run_in_sandbox

        class LocalReason(Exception):
            pass

        with pytest.raises(LocalReason, match="did not allow for"):
            run_in_sandbox(
                hostile_child.raises_an_unknown_type,
                config=_sandbox_config(tmp_path),
                timeout=30,
                error_types={"MadeUpError": LocalReason},
            )


class TestFramesAreBounded:
    """A child cannot turn its own rlimited memory into the parent's."""

    def test_an_oversized_result_is_refused(self, tmp_path):
        from liminallm.service.sandbox import SandboxError, run_in_sandbox

        with pytest.raises(SandboxError):
            run_in_sandbox(
                hostile_child.returns_a_big_payload,
                config=_sandbox_config(tmp_path),
                timeout=30,
                max_result_bytes=64 * 1024,
            )

    def test_the_broker_refuses_a_frame_larger_than_it_granted(self):
        from liminallm.service.broker import CapabilityBroker, InvocationContext
        from liminallm.service.invocation import Invocation
        from liminallm.service.tool_worker import FrameBudget

        parent_conn, child_conn = multiprocessing.Pipe(duplex=True)
        budget = FrameBudget(0)
        oversize = (
            b'{"done": true, "result": {"content": "'
            + b"A" * budget.limit
            + b'"}}'
        )
        # Written from a thread on purpose. A frame this size does not fit in
        # the socket buffer, so the write blocks until someone reads it — and
        # the reader under test is the one that must *refuse* to. Doing it
        # inline deadlocks the test, which is itself the shape of the property:
        # the parent never takes the bytes.
        writer = threading.Thread(
            target=_send_and_swallow, args=(child_conn, oversize), daemon=True
        )
        writer.start()

        invocation = Invocation("greedy-worker", tool="agent.files_v1")
        invocation.begin_attempt()
        broker = CapabilityBroker(None, InvocationContext(user_id="u1"))
        try:
            result = broker.serve(
                parent_conn, invocation, is_alive=lambda: True, budget=budget
            )
        finally:
            invocation.close()
            # Reader end first, so the blocked write ends with a broken pipe
            # rather than with its own handle closed under it.
            parent_conn.close()
            writer.join(5)
            child_conn.close()
        assert result.get("error") == "worker_protocol", result

    def test_the_budget_grows_only_by_what_the_broker_sent(self):
        """The worker returns the conversation it was handed, so the parent's
        own outbound total is the honest allowance — not a guess about sizes."""
        from liminallm.service.tool_worker import (
            WORKER_FRAME_ALLOWANCE_BYTES,
            FrameBudget,
        )

        budget = FrameBudget(4096)
        assert budget.limit == 4096 + WORKER_FRAME_ALLOWANCE_BYTES
        budget.credit(1000)
        assert budget.limit == 5096 + WORKER_FRAME_ALLOWANCE_BYTES


class TestTheSandboxTerminatesItsDescendants:
    """§19.5's parsers spawn grandchildren — `pdftoppm`, tesseract — and the
    wall-clock kill is described as disposing of the job, not of one pid."""

    def test_a_grandchild_does_not_outlive_the_wall_clock(self, marker, tmp_path):
        from liminallm.service.sandbox import SandboxError, run_in_sandbox

        with pytest.raises(SandboxError):
            run_in_sandbox(
                hostile_child.spawns_a_survivor,
                config=_sandbox_config(tmp_path),
                timeout=3,
            )
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and not marker.exists():
            time.sleep(0.1)
        assert marker.exists(), "the grandchild never started; the test proves nothing"
        grandchild = int(marker.read_text())

        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and _alive(grandchild):
            time.sleep(0.1)
        assert not _alive(grandchild), (
            f"parser grandchild {grandchild} outlived the sandbox timeout"
        )


class TestRevocationReachesTheSameDescendants:
    """The wall clock is not the only thing that ends a job.

    A cancelled turn tears its tree down through the invocation's registry,
    which killed the pid it was handed and left that pid's children running.
    Same defect as the timeout path, reached by the other door.
    """

    def test_a_revoked_invocation_kills_a_registered_child_s_child(self, marker):
        from liminallm.service.agent_tools import _register_child
        from liminallm.service.invocation import Invocation

        proc = _group_leader_with_a_child(marker)
        invocation = Invocation("cancel-the-tree")
        invocation.begin_attempt()
        # The real registration, not a hand-written one: what is under test is
        # what `run_python` actually records about its sandbox child.
        _register_child(invocation)(proc.pid, lambda: proc.wait(5))

        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and not marker.exists():
            time.sleep(0.1)
        assert marker.exists(), "the grandchild never started; the test proves nothing"
        grandchild = int(marker.read_text())

        try:
            invocation.revoke("cancelled")
            assert invocation.terminate() is True
        finally:
            invocation.close()
            if _alive(proc.pid):
                proc.kill()
            proc.wait(5)

        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and _alive(grandchild):
            time.sleep(0.1)
        assert not _alive(grandchild), (
            f"grandchild {grandchild} survived revocation of its parent"
        )


class TestRequiredLimitsFailClosed:
    """§19.5 states the parser runs under memory/CPU/file-size caps. A cap the
    platform refused used to be recorded in a dict nobody read."""

    def test_a_refused_limit_stops_the_body(self, tmp_path, monkeypatch):
        import resource

        from liminallm.service import sandbox as sandbox_module

        calls: list = []

        def refuse(which, value):
            calls.append(which)
            raise OSError(1, "operation not permitted")

        monkeypatch.setattr(resource, "setrlimit", refuse)
        with pytest.raises(sandbox_module.SandboxError):
            sandbox_module.apply_resource_limits(_sandbox_config(tmp_path))
        assert calls, "setrlimit was never attempted"


class TestTheExtractionChildQueuesOnlyWhatTheParentCanRead:
    """The frame bound is derived from MAX_SCANNED_PAGES images of at most
    MAX_IMAGE_BYTES, so the child has to hold to that for the bound to be one.

    It is also the right rule on its own: MAX_IMAGE_BYTES is the parent's
    data-URL ceiling, so an image above it has no vision pass waiting for it.
    """

    def test_an_image_past_the_data_url_ceiling_is_not_parked(self):
        from liminallm.service.extract import (
            MAX_IMAGE_BYTES,
            ExtractError,
            _image_bytes_to_text,
        )

        pending: list = []
        with pytest.raises(ExtractError):
            _image_bytes_to_text(
                b"\x89PNG\r\n\x1a\n" + b"\x00" * MAX_IMAGE_BYTES,
                "image/png",
                None,
                ("vision",),
                pending,
            )
        assert pending == [], "an unreadable-sized image was queued for the parent"

    def test_an_image_within_the_ceiling_still_is(self):
        """A refusal above must be about the size, not about parking at all."""
        from liminallm.service.extract import _image_bytes_to_text

        pending: list = []
        text, mech = _image_bytes_to_text(
            b"\x89PNG\r\n\x1a\n" + b"\x00" * 64, "image/png", None, ("vision",), pending
        )
        assert len(pending) == 1, pending
        assert mech is None and text, "no placeholder slot was returned"


def _group_leader_with_a_child(marker: Path):
    """A process shaped like the sandbox child: leads a group, spawns into it.

    `start_new_session=True` is `setsid`, which is what `_sandbox_entry` now
    does before it runs anything — so the group this leaves behind is the one
    a teardown has to reach.
    """
    import subprocess
    import sys

    code = (
        "import subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'])\n"
        f"open({str(marker)!r}, 'w').write(str(child.pid))\n"
        "time.sleep(300)\n"
    )
    return subprocess.Popen(  # noqa: S603 - fixed argv, test-only
        [sys.executable, "-c", code],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _send_and_swallow(conn, raw: bytes) -> None:
    """Push bytes at a reader that is expected to refuse them.

    The refusal leaves this write blocked with nowhere to go; closing an end is
    what ends it, so failing here is the expected outcome. Which failure
    depends on which end closes first — a broken pipe from the reader, or a
    handle pulled out from under this thread — so all of them are the answer.
    """
    try:
        conn.send_bytes(raw)
    except Exception:  # noqa: BLE001 - see above
        pass


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _files(root: Path) -> set:  # pragma: no cover - debugging aid
    return {p.name for p in root.iterdir()} if root.exists() else set()
