"""Sandboxed Python interpreter for model-written code.

The model calls this to parse, unzip, and compute over a conversation's
attachments — the same job ChatGPT's code interpreter does. Each call gets a
fresh working directory containing *copies* of the attachments, so code can
never damage the originals.

Containment, outermost first:

1. The code runs in a spawned child process (``run_in_sandbox``) with memory,
   CPU-time, file-size, and core-dump rlimits plus a wall-clock kill.
2. **The child is confined to its workdir** (``service/confine.py``): the
   shared filesystem root, other users' files, service configuration and every
   other host path are absent from its view, not merely unreadable. This is
   the boundary that matters, because the service uid owns every user's files
   — a same-uid process with an ordinary filesystem view can read all of them
   by naming an absolute path, whatever the working directory is.
3. Network egress is blocked: an empty-allowlist policy trips the socket guard,
   and the networking/process modules are blocked at import.
4. Process-spawning entry points (``os.system``, ``fork``, ``exec*``, ...) are
   removed before user code runs, so code cannot escape the rlimited process.
5. The child's working directory holds only copies of the user's own files.

There is no unconfined fallback. On a platform with no confinement backend
this tool refuses to run at all (``ConfinementUnavailable``), because the
alternative — the same-uid unrestricted process this used to be — is what
(2) exists to remove. (3) and (4) remain best-effort defense in depth around
it rather than the wall itself.
"""
from __future__ import annotations

import builtins
import io
import os
import shutil
import stat
import sys
import uuid
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Callable, Optional

from liminallm.service.confine import backend_name, confine
from liminallm.service.invocation import require_live_lease

DEFAULT_TIMEOUT_SECONDS = 20
MAX_OUTPUT_CHARS = 8_000
# Total bytes of attachments copied into a session working directory.
MAX_WORKDIR_BYTES = 64 * 1024 * 1024
# Files the code writes that get published back to the user's file area.
MAX_ARTIFACTS = 10
MAX_ARTIFACT_BYTES = 8 * 1024 * 1024

# What the child may answer with, derived from what it is allowed to produce:
# two streams of MAX_OUTPUT_CHARS, plus MAX_ARTIFACTS names. JSON escapes at
# worst six bytes per character, so 2 * 8_000 * 6 is ~94KB of output and ten
# 255-char names are ~15KB. The rest is headroom for keys and structure.
# Without a cap the code the model wrote decides how much the API process
# allocates, which is the one thing the child process boundary exists to stop.
_RESULT_BYTES = 128 * 1024

# Modules that would give the code a network or a new process.
_BLOCKED_MODULES = frozenset({
    "socket", "ssl", "subprocess", "multiprocessing", "asyncio", "ctypes",
    "http", "urllib", "urllib3", "httpx", "requests", "ftplib", "telnetlib",
    "smtplib", "poplib", "imaplib", "xmlrpc", "webbrowser", "pty",
})


class _BlockedImportFinder:
    """Meta-path finder that refuses networking / process-spawning modules."""

    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".")[0]
        if root in _BLOCKED_MODULES:
            raise ImportError(
                f"module '{fullname}' is not available in the sandbox "
                "(no network or subprocess access)"
            )
        return None


def _harden_child(workdir: str, confine_root: str = "") -> str:
    """Confine the sandbox child and drop its escape hatches.

    Returns the workdir's name after confinement — the Linux backend re-roots
    the process, so the path passed in stops existing. Confinement happens
    first: everything after it is defense in depth, and a failure to establish
    it must stop the call rather than soften it.

    `confine_root` is the mount point for the new root, supplied by the caller
    so the caller can remove it: after `pivot_root` nothing here can reach the
    host path again, so a backend left to make its own leaks one empty
    directory per call.
    """
    from liminallm.service.sandbox import (  # local: child-side import
        _NETWORK_POLICY_STATE,
        ToolNetworkPolicy,
    )

    confined_workdir = confine(workdir, root=confine_root or None)

    # The environment is inherited at process start and lives in memory, so
    # re-rooting the filesystem does nothing to it: `DATABASE_URL` and every
    # other secret passed by the deployment was still readable through
    # os.environ from inside the jail. Replaced wholesale with the few
    # variables the runtime wants, all pointing at the scratch it already has.
    os.environ.clear()
    os.environ.update({
        "HOME": confined_workdir,
        "TMPDIR": confined_workdir,
        "PWD": confined_workdir,
        "PATH": "",  # nothing to exec anyway; see the denials below
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
    })

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
    return confined_workdir


def _truncate(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return text[:MAX_OUTPUT_CHARS] + f"\n...[output truncated at {MAX_OUTPUT_CHARS} chars]"


def execute_python(code: str, workdir: str, confine_root: str = "") -> dict[str, Any]:
    """Run ``code`` with ``workdir`` as the current directory.

    Module-level with picklable arguments so ``run_in_sandbox`` can ship it to
    a child process. Returns captured output rather than raising, so the model
    can read and react to its own errors.
    """
    workdir = _harden_child(workdir, confine_root)
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


def open_produced_file(workdir: str, name: str) -> Optional[int]:
    """Open a file model-written code produced, following no link out of it.

    The child that chose this name runs confined (§21.2); this process does
    not. A pathname is not a capability the child has to hold — it cannot open
    `/etc/passwd`, because its root was pivoted away, but creating a link with
    that target costs it nothing and does not need the target to exist on its
    side. Measured before this existed: a link named `result.txt` had the
    host's `/etc/passwd` copied into the caller's file area, and the same
    trick reached another user's uploads under `shared_fs_root`. The child
    could open neither; naming them was enough, because this process opened
    them on its behalf.

    `O_NOFOLLOW` is why this returns a descriptor rather than a validated
    path. It makes deciding and reading one operation on one object, where an
    `is_symlink()` check followed by an `open()` is two operations on a name.

    `O_NONBLOCK` is why the open itself is safe to attempt. `O_NOFOLLOW`
    refuses a link but says nothing about a fifo, and opening one for reading
    waits for a writer — measured, `os.open` on a fifo never returned, which
    would park this thread of the API process for as long as the child cared
    to leave the fifo there. Non-blocking, the open returns and `fstat`
    answers; on a regular file the flag does nothing.

    Returns None for a link, a fifo, a directory, a device, or anything
    unreadable.
    """
    try:
        fd = os.open(
            os.path.join(workdir, name),
            os.O_RDONLY
            | os.O_NOFOLLOW
            | os.O_NONBLOCK
            | getattr(os, "O_CLOEXEC", 0),
        )
    except OSError:
        return None
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            os.close(fd)
            return None
    except OSError:
        os.close(fd)
        return None
    return fd


def publish_artifacts(
    workdir: str,
    dest_dir: str,
    created: list[dict],
    allowed_extensions: Optional[set[str]] = None,
) -> list[str]:
    """Copy files the code produced into the user's file area.

    Model-written code chooses these filenames, so they go through the same
    extension policy as an upload: publishing arbitrary types would let the
    interpreter put files into the user's area that /files/upload would reject.
    """
    # Publication is a durable operation on the user's persistent area, and
    # it reaches it directly rather than through the store — so the leased
    # proxies never see it and a revoked invocation could still leave a file
    # behind. The check is here, at the copy, rather than at a caller: this
    # function is what actually writes.
    require_live_lease()

    published: list[str] = []
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    for item in created[:MAX_ARTIFACTS]:
        name = str(item.get("name") or "")
        if not name or "/" in name or name.startswith("."):
            continue
        if allowed_extensions is not None:
            if Path(name).suffix.lower() not in allowed_extensions:
                continue
        fd = open_produced_file(workdir, name)
        if fd is None:
            continue
        try:
            if os.fstat(fd).st_size > MAX_ARTIFACT_BYTES:
                continue
            # The destination refuses a link too: the user's file area is
            # where a planted one would be redeemed, and writing through it
            # would put these bytes wherever it points.
            out_fd = os.open(
                dest / name,
                os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW,
                0o640,
            )
            with open(fd, "rb", closefd=False) as src, open(
                out_fd, "wb", closefd=False
            ) as out:
                shutil.copyfileobj(src, out)
            os.close(out_fd)
            published.append(name)
        except OSError:
            continue
        finally:
            os.close(fd)
    return published


def run_python_sandboxed(
    code: str,
    *,
    workdir: str,
    confine_root: str = "",
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_memory_mb: int = 512,
    on_child: Optional[
        Callable[[int, Callable[[], None]], Optional[Callable[[], None]]]
    ] = None,
) -> dict[str, Any]:
    """Execute model-written Python in the confined, resource-limited sandbox.

    Refuses up front on a platform with no confinement backend, rather than
    letting the child discover it: the caller gets one clear reason instead of
    a sandbox error, and no untrusted code is ever spawned unconfined.
    """
    from liminallm.service.sandbox import SandboxConfig, SandboxError, run_in_sandbox

    if backend_name() is None:
        return {
            "ok": False,
            "stdout": "",
            "stderr": (
                "the code interpreter is unavailable on this platform: no "
                "filesystem confinement backend, and model-written code is "
                "not run unconfined (SPEC §18)"
            ),
            "created_files": [],
        }

    config = SandboxConfig(
        max_memory_mb=max_memory_mb,
        max_cpu_seconds=int(timeout),
        max_file_size_mb=32,
        scratch_dir=Path(workdir),
    )
    try:
        return run_in_sandbox(
            execute_python,
            code,
            workdir,
            confine_root,
            config=config,
            timeout=timeout,
            on_child=on_child,
            max_result_bytes=_RESULT_BYTES,
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
