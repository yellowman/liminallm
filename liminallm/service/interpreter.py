"""Sandboxed Python interpreter for model-written code.

The model calls this to parse, unzip, and compute over a conversation's
attachments — the same job ChatGPT's code interpreter does. Each call gets a
fresh working directory containing *copies* of the attachments, so code can
never damage the originals.

Containment, outermost first:

1. The code runs in a spawned child process (``run_in_sandbox``) with memory,
   CPU-time, file-size, and core-dump rlimits plus a wall-clock kill.
2. Network egress is blocked: an empty-allowlist policy trips the socket guard,
   and the networking/process modules are blocked at import.
3. Process-spawning entry points (``os.system``, ``fork``, ``exec*``, ...) are
   removed before user code runs, so code cannot escape the rlimited process.
4. The child's working directory holds only copies of the user's own files.

This is meaningful isolation, not a security boundary against a determined
attacker sharing the host: in-process hardening (3) is best-effort, and the
model is executing code derived from user-supplied content. Run the API in a
container or VM if the threat model includes hostile uploads.
"""
from __future__ import annotations

import builtins
import io
import os
import shutil
import sys
import uuid
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Optional

DEFAULT_TIMEOUT_SECONDS = 20
MAX_OUTPUT_CHARS = 8_000
# Total bytes of attachments copied into a session working directory.
MAX_WORKDIR_BYTES = 64 * 1024 * 1024
# Files the code writes that get published back to the user's file area.
MAX_ARTIFACTS = 10
MAX_ARTIFACT_BYTES = 8 * 1024 * 1024

# Modules that would give the code a network or a new process.
_BLOCKED_MODULES = frozenset({
    "socket", "ssl", "subprocess", "multiprocessing", "asyncio", "ctypes",
    "http", "urllib", "urllib3", "httpx", "requests", "ftplib", "telnetlib",
    "smtplib", "poplib", "imaplib", "xmlrpc", "webbrowser", "pty",
})


class _BlockedImportFinder:
    """Meta-path finder that refuses networking / process-spawning modules."""

    def find_module(self, fullname, path=None):  # legacy API, harmless
        return self.find_spec(fullname, path)

    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".")[0]
        if root in _BLOCKED_MODULES:
            raise ImportError(
                f"module '{fullname}' is not available in the sandbox "
                "(no network or subprocess access)"
            )
        return None


def _harden_child() -> None:
    """Drop escape hatches inside the sandbox child before running user code."""
    from liminallm.service.sandbox import (  # local: child-side import
        _NETWORK_POLICY_STATE,
        ToolNetworkPolicy,
    )

    # An empty allowlist makes the socket guard reject every connection.
    _NETWORK_POLICY_STATE.policy = ToolNetworkPolicy(allowlist=[])

    for name in list(_BLOCKED_MODULES):
        sys.modules.pop(name, None)
    sys.meta_path.insert(0, _BlockedImportFinder())

    def _denied(*_args: Any, **_kwargs: Any):
        raise PermissionError("process execution is not available in the sandbox")

    for attr in (
        "system", "popen", "fork", "forkpty", "spawnl", "spawnle", "spawnlp",
        "spawnlpe", "spawnv", "spawnve", "spawnvp", "spawnvpe", "execl",
        "execle", "execlp", "execlpe", "execv", "execve", "execvp", "execvpe",
        "kill", "killpg", "setuid", "setgid",
    ):
        if hasattr(os, attr):
            setattr(os, attr, _denied)


def _truncate(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return text[:MAX_OUTPUT_CHARS] + f"\n...[output truncated at {MAX_OUTPUT_CHARS} chars]"


def execute_python(code: str, workdir: str) -> dict[str, Any]:
    """Run ``code`` with ``workdir`` as the current directory.

    Module-level with picklable arguments so ``run_in_sandbox`` can ship it to
    a child process. Returns captured output rather than raising, so the model
    can read and react to its own errors.
    """
    _harden_child()
    os.chdir(workdir)
    sys.path.insert(0, workdir)
    before = {p.name for p in Path(workdir).iterdir() if p.is_file()}

    stdout, stderr = io.StringIO(), io.StringIO()
    ok = True
    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            # A dedicated namespace, but real builtins: the point is to run
            # ordinary Python (csv, json, zipfile), and the isolation that
            # matters is the process boundary, not a builtins denylist.
            exec(compile(code, "<attachment-analysis>", "exec"), {"__builtins__": builtins, "__name__": "__main__"})
    except BaseException as exc:  # noqa: BLE001 - report to the model verbatim
        ok = False
        import traceback

        stderr.write("".join(traceback.format_exception_only(type(exc), exc)))

    created = []
    for p in sorted(Path(workdir).iterdir()):
        if p.is_file() and p.name not in before:
            created.append({"name": p.name, "size": p.stat().st_size})
    return {
        "ok": ok,
        "stdout": _truncate(stdout.getvalue()),
        "stderr": _truncate(stderr.getvalue()),
        "created_files": created[:MAX_ARTIFACTS],
    }


def prepare_workdir(
    session_root: str, attachment_dir: str, names: list[str]
) -> str:
    """Create a session directory holding copies of the named attachments."""
    workdir = Path(session_root) / f"session-{uuid.uuid4().hex[:12]}"
    workdir.mkdir(parents=True, exist_ok=True)
    source = Path(attachment_dir)
    budget = MAX_WORKDIR_BYTES
    for name in names:
        src = source / name
        # `names` comes from stored attachment records, but re-check anyway so
        # a crafted record can never copy from outside the user's file area.
        try:
            src = src.resolve()
            src.relative_to(source.resolve())
        except (ValueError, OSError):
            continue
        if not src.is_file():
            continue
        size = src.stat().st_size
        if size > budget:
            continue
        budget -= size
        shutil.copy2(src, workdir / src.name)
    return str(workdir)


def publish_artifacts(workdir: str, dest_dir: str, created: list[dict]) -> list[str]:
    """Copy files the code produced into the user's file area."""
    published: list[str] = []
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    for item in created[:MAX_ARTIFACTS]:
        name = str(item.get("name") or "")
        if not name or "/" in name or name.startswith("."):
            continue
        src = Path(workdir) / name
        if not src.is_file() or src.stat().st_size > MAX_ARTIFACT_BYTES:
            continue
        try:
            shutil.copy2(src, dest / name)
            published.append(name)
        except OSError:
            continue
    return published


def run_python_sandboxed(
    code: str,
    *,
    workdir: str,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_memory_mb: int = 512,
) -> dict[str, Any]:
    """Execute model-written Python in the resource-limited sandbox."""
    from liminallm.service.sandbox import SandboxConfig, SandboxError, run_in_sandbox

    config = SandboxConfig(
        max_memory_mb=max_memory_mb,
        max_cpu_seconds=int(timeout),
        max_file_size_mb=32,
        scratch_dir=Path(workdir),
    )
    try:
        return run_in_sandbox(
            execute_python, code, workdir, config=config, timeout=timeout
        )
    except SandboxError as exc:
        return {
            "ok": False,
            "stdout": "",
            "stderr": f"sandbox stopped the code: {exc}",
            "created_files": [],
        }


def cleanup_workdir(workdir: Optional[str]) -> None:
    if workdir:
        shutil.rmtree(workdir, ignore_errors=True)
