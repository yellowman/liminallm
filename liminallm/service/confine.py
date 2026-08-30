"""Filesystem confinement for the process that runs model-written code.

SPEC §18 states the contract as an observable property, not a mechanism:

    can see:  staged inputs RO, the per-call workdir RW, necessary runtime RO
    cannot see: shared_fs_root, other users, service config and secrets,
                arbitrary host paths, the network

The service owns every user's files, so the interpreter inheriting the service
uid is the wrong trust boundary and no amount of unix permission checking
fixes it: the paths have to be *absent* from the process's view, not merely
unreadable. Everything else the sandbox does - rlimits, the wall-clock kill,
the network denial, removing the process-spawn entry points - is defense in
depth around this.

One backend per platform, and **no degraded fallback**. A platform with no
backend cannot run `run_python` at all; falling back to an unconfined process
would be the same-uid unrestricted filesystem this module exists to remove,
with a docstring claiming otherwise.

* Linux: a user + mount namespace with a fresh tmpfs root - the workdir bound
  rw, the runtime bound ro, everything else simply not mounted. Unprivileged:
  `CLONE_NEWUSER` grants the caps needed for the mounts inside the namespace
  only.
* OpenBSD: `unveil(2)` to expose exactly those paths, then a final locking
  `unveil(NULL, NULL)`, then `pledge(2)` to drop what is left.

Both are applied by the process to itself, immediately before it runs
untrusted code, and neither can be undone afterwards.
"""
from __future__ import annotations

import ctypes
import os
import platform
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence

__all__ = [
    "ConfinementUnavailable",
    "backend_name",
    "confine",
    "runtime_paths",
]


class ConfinementUnavailable(RuntimeError):
    """No confinement backend for this platform, so untrusted code cannot run."""


# --- linux: user + mount namespace ----------------------------------------

_CLONE_NEWNS = 0x00020000
_CLONE_NEWNET = 0x40000000
_CLONE_NEWUSER = 0x10000000
_MS_RDONLY = 1
_MS_NOSUID = 2
_MS_NODEV = 4
_MS_NOEXEC = 8
_MS_REMOUNT = 32
_MS_BIND = 4096
_MS_REC = 16384
_MS_PRIVATE = 1 << 18
_MNT_DETACH = 2
_SYS_PIVOT_ROOT = {"x86_64": 155, "aarch64": 41}

#: Where the workdir appears inside the jail.
LINUX_SCRATCH = "/scratch"


def _libc() -> ctypes.CDLL:
    return ctypes.CDLL(None, use_errno=True)


def _as_bytes(value: Optional[str]) -> Optional[bytes]:
    return value.encode() if value is not None else None


def _mount(libc, source, target, fstype, flags, data=None) -> None:
    if libc.mount(
        _as_bytes(source), _as_bytes(target), _as_bytes(fstype),
        ctypes.c_ulong(flags), _as_bytes(data),
    ) != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, f"mount({source!r} -> {target!r}): {os.strerror(errno)}")


def _write_proc(path: str, value: str, *, operation: str) -> None:
    """One of the writes that establish the namespace's identity mapping.

    Labelled the way `unshare` and `mount` are, and for the same reason. A
    bare `PermissionError: [Errno 13] ... '/proc/self/setgroups'` names the
    file but not the operation, and the operation is what tells an operator
    which kernel policy refused them - the difference between "this host has
    user namespaces switched off" and "this host allowed the namespace and
    then refused the mapping inside it".

    That distinction is not hypothetical. On a GitHub-hosted runner the
    `unshare` above *succeeds* and this is the line that fails, which is a
    much narrower fact than "the sandbox is unavailable" and points at a
    different knob.
    """
    try:
        Path(path).write_text(value)
    except OSError as exc:
        errno = exc.errno or 0
        raise OSError(
            errno,
            f"{operation} ({path} <- {value!r}): {os.strerror(errno)}",
        ) from exc


def _linux_available() -> bool:
    """Whether this kernel can give a process its own user + mount namespace.

    Read from /proc rather than probed by actually unsharing. `unshare` is
    irreversible for the caller, so the probe has to happen somewhere other
    than the asking process - and the obvious somewhere, a forked child, is
    the wrong tool: this question is asked from the API process, which has JAX
    loaded and threads running, and `os.fork()` there risks the child
    deadlocking on a lock held by a thread that does not exist in it. (The
    interpreter warns about exactly this.)

    So this is the cheap "could this platform confine" question, and the
    authoritative one is `confine()` itself, which runs in the already-spawned
    single-threaded sandbox child and fails closed if the kernel refuses.
    """
    if sys.platform != "linux" or platform.machine() not in _SYS_PIVOT_ROOT:
        return False
    if not os.path.exists("/proc/self/uid_map"):
        return False  # user namespaces not compiled in
    # Two knobs that must permit, and one that must not forbid. The third is
    # why reading the first two alone was wrong: on a stock Ubuntu 24.04 host
    # both of those say yes, `unshare` then *succeeds*, and the process holds
    # no capabilities in the namespace it just created - so the identity
    # mapping is refused and confinement fails after this said it was
    # available. Measured on a GitHub-hosted runner: clone=1,
    # max_user_namespaces=63838, apparmor_restrict_unprivileged_userns=1, and
    # `unshare --user --map-root-user true` reporting "write failed
    # /proc/self/uid_map: Operation not permitted".
    #
    # An AppArmor profile carrying `userns create` lifts the restriction for
    # the programs it covers, so this is pessimistic for a packaged deployment
    # that ships one. Pessimistic is the right direction here: a wrong False
    # withholds the interpreter on a host where it would have worked, and a
    # wrong True offers one that fails on every call. `confine()` remains the
    # authoritative answer either way.
    for path, permitted in (
        ("/proc/sys/user/max_user_namespaces", lambda value: value >= 1),
        # A Debian-family knob; absent on kernels that never had it.
        ("/proc/sys/kernel/unprivileged_userns_clone", lambda value: value >= 1),
        # Ubuntu 24.04 and later; absent before it.
        (
            "/proc/sys/kernel/apparmor_restrict_unprivileged_userns",
            lambda value: value == 0,
        ),
    ):
        try:
            if not permitted(int(Path(path).read_text().strip())):
                return False
        except (OSError, ValueError):
            continue
    return True


def _linux_confine(workdir: str, runtime: Sequence[str], root: Optional[str] = None) -> str:
    libc = _libc()
    uid, gid = os.getuid(), os.getgid()
    # Files staged into the workdir before confinement are inputs and become
    # read-only; anything the code creates afterwards is its own output. Read
    # here because after pivot_root this path no longer exists.
    staged = sorted(
        entry.name for entry in Path(workdir).iterdir() if entry.is_file()
    )
    # NEWNET as well: with no interfaces but a down loopback there is no
    # network to reach, which is what SPEC asks for. Blocking `socket` at
    # import cannot do this - it missed `_socket`, and any already-loaded
    # module can hand out the same primitive. The import denial stays as
    # defense in depth, the way `pledge` without `inet` is on OpenBSD.
    if libc.unshare(_CLONE_NEWUSER | _CLONE_NEWNS | _CLONE_NEWNET) != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, f"unshare: {os.strerror(errno)}")
    # Map this uid to root *inside the namespace only*, which is what allows
    # the mounts below. setgroups must be denied first or gid_map is refused.
    _write_proc("/proc/self/setgroups", "deny", operation="deny setgroups")
    _write_proc("/proc/self/uid_map", f"0 {uid} 1", operation="map uid inside namespace")
    _write_proc("/proc/self/gid_map", f"0 {gid} 1", operation="map gid inside namespace")
    # Detach propagation so nothing here is visible to the host mount table.
    _mount(libc, None, "/", None, _MS_REC | _MS_PRIVATE)

    # The mount point is a real directory on the host filesystem, and after
    # pivot_root nothing can reach it to remove it: the old root is detached
    # and the process's own root is the tmpfs mounted on top. So the caller
    # supplies a path it already cleans up. Absent one, it goes beside the
    # workdir rather than into the system temp root - a caller that cleans up
    # after its own workdir then cleans up after this too, and one that does
    # not leaks in a place it is already leaking.
    root = root or f"{workdir.rstrip('/')}-confine-root"
    os.makedirs(root, exist_ok=True)
    _mount(libc, "tmpfs", root, "tmpfs", _MS_NOSUID | _MS_NODEV, "size=16m")

    scratch = os.path.join(root, LINUX_SCRATCH.lstrip("/"))
    os.mkdir(scratch)
    _mount(libc, workdir, scratch, None, _MS_BIND | _MS_REC)
    # Each staged input bound onto itself, read-only. The directory stays
    # writable so the code can produce artifacts; the inputs it was given do
    # not, or run 2 of a session reads what run 1 rewrote and calls it the
    # user's attachment.
    for name in staged:
        target = os.path.join(scratch, name)
        _mount(libc, target, target, None, _MS_BIND)
        _mount(libc, None, target, None, _MS_REMOUNT | _MS_BIND | _MS_RDONLY)

    for path in runtime:
        if not os.path.isdir(path):
            continue
        target = os.path.join(root, path.lstrip("/"))
        os.makedirs(target, exist_ok=True)
        _mount(libc, path, target, None, _MS_BIND | _MS_REC)
        _mount(
            libc, None, target, None,
            _MS_REMOUNT | _MS_BIND | _MS_REC | _MS_RDONLY | _MS_NOSUID | _MS_NODEV,
        )

    old_root = os.path.join(root, "oldroot")
    os.mkdir(old_root)
    if libc.syscall(
        _SYS_PIVOT_ROOT[platform.machine()], _as_bytes(root), _as_bytes(old_root)
    ) != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, f"pivot_root: {os.strerror(errno)}")
    os.chdir("/")
    # Detach the old root: leaving it mounted would leave the whole host tree
    # reachable through /oldroot, which is the thing being removed.
    if libc.umount2(b"/oldroot", _MNT_DETACH) != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, f"umount2(/oldroot): {os.strerror(errno)}")
    os.rmdir("/oldroot")
    os.chdir(LINUX_SCRATCH)
    return LINUX_SCRATCH


# --- openbsd: unveil + pledge ---------------------------------------------

# Standard-library modules whose implementation is a shared object. Loading one
# is a `dlopen`, which maps pages `PROT_EXEC` - and `pledge` denies that
# without the `prot_exec` promise. The child is spawned, not forked, so it
# starts with none of these resident and the first `import zipfile` in model
# code would be killed rather than refused. Granting `prot_exec` would fix the
# symptom by handing untrusted code the ability to map executable memory, so
# instead they are loaded *before* the promise drops: what the interpreter is
# documented to do keeps working, and a later attempt to load any other shared
# object is denied.
_NATIVE_WARMUP = (
    "array", "binascii", "bz2", "csv", "datetime", "decimal", "hashlib",
    "json", "lzma", "math", "random", "struct", "unicodedata", "zlib",
)


def _warm_native_modules() -> None:
    import importlib

    for name in _NATIVE_WARMUP:
        try:
            importlib.import_module(name)
        except ImportError:  # a build without this module: nothing to warm
            continue


def _openbsd_available() -> bool:
    if sys.platform != "openbsd6" and not sys.platform.startswith("openbsd"):
        return False
    try:
        libc = _libc()
        return hasattr(libc, "unveil") and hasattr(libc, "pledge")
    except OSError:  # pragma: no cover - libc always loads on openbsd
        return False


def _openbsd_confine(workdir: str, runtime: Sequence[str], root: Optional[str] = None) -> str:
    """Expose only these paths, then drop the capabilities left over.

    Unverified on this machine - there is no OpenBSD host here to run the
    contract test against, so treat it as implemented-but-unproven until it
    passes `tests/test_interpreter_confinement.py` on one. The calls
    themselves are the documented ones: `unveil` per path, a locking
    `unveil(NULL, NULL)`, then `pledge` with no filesystem-widening promise.
    """
    libc = _libc()
    libc.unveil.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    libc.pledge.argtypes = [ctypes.c_char_p, ctypes.c_char_p]

    # Before anything narrows: see `_NATIVE_WARMUP`.
    _warm_native_modules()

    if libc.unveil(_as_bytes(workdir), b"rwc") != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, f"unveil({workdir!r}): {os.strerror(errno)}")
    # Staged inputs read-only. unveil matches the longest prefix, so a file
    # entry overrides the directory's `rwc` for that file alone.
    for entry in sorted(Path(workdir).iterdir()):
        if not entry.is_file():
            continue
        if libc.unveil(_as_bytes(str(entry)), b"r") != 0:
            errno = ctypes.get_errno()
            raise OSError(errno, f"unveil({str(entry)!r}): {os.strerror(errno)}")
    for path in runtime:
        if not os.path.isdir(path):
            continue
        if libc.unveil(_as_bytes(path), b"rx") != 0:
            errno = ctypes.get_errno()
            raise OSError(errno, f"unveil({path!r}): {os.strerror(errno)}")
    # Lock the list: no further unveil calls can widen it.
    if libc.unveil(None, None) != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, f"unveil(lock): {os.strerror(errno)}")
    # No `inet`, no `proc`/`exec`: the network and process-spawn denials the
    # sandbox already applies in Python, made structural.
    # No `tmppath`: pledge(2) lists it as "No longer available", so naming it
    # fails the whole call - and it granted /tmp, which `unveil` above does
    # not expose anyway. TMPDIR points at the workdir; `cpath`/`wpath` there
    # is what temporary files actually need.
    # No `prot_exec`: see `_NATIVE_WARMUP`. Denying new executable mappings is
    # the point, not an oversight.
    if libc.pledge(b"stdio rpath wpath cpath flock", None) != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, f"pledge: {os.strerror(errno)}")
    os.chdir(workdir)
    return workdir


# --- dispatch --------------------------------------------------------------

_BACKENDS = (
    ("linux-namespaces", _linux_available, _linux_confine),
    ("openbsd-unveil", _openbsd_available, _openbsd_confine),
)


def backend_name() -> Optional[str]:
    """The confinement backend for this platform, or None if there is none."""
    for name, available, _ in _BACKENDS:
        try:
            if available():
                return name
        except OSError:  # pragma: no cover - probe failure means unavailable
            continue
    return None


def runtime_paths() -> list[str]:
    """The read-only paths the interpreter itself needs to keep working.

    The language runtime and its installed packages, and nothing that holds
    user data, service configuration or secrets. `sys.prefix` is the virtualenv
    when there is one - that subtree, not the project directory around it.
    """
    candidates: Iterable[str] = (
        sys.base_prefix,
        sys.prefix,
        "/usr",
        "/lib",
        "/lib64",
    )
    seen: dict[str, None] = {}
    for path in candidates:
        if path and os.path.isdir(path):
            seen.setdefault(os.path.realpath(path), None)
    return list(seen)


def confine(
    workdir: str,
    runtime: Optional[Sequence[str]] = None,
    *,
    root: Optional[str] = None,
) -> str:
    """Restrict this process to `workdir` plus the language runtime.

    Returns the working directory as it is named *after* confinement, which
    the caller must use from then on: the Linux backend re-roots the process,
    so the path it was given no longer exists. Irreversible by design.

    `root` is where the new root is mounted. It must be a path the caller
    already owns and removes, and a *sibling* of `workdir` rather than a child
    of it - the Linux backend binds `workdir` into the new root, and a mount
    point inside it would be bound into itself. Callers that omit it leak one
    empty directory per call, because after `pivot_root` nothing is left that
    can reach the host path to clean it.

    Raises `ConfinementUnavailable` when the platform has no backend, because
    running untrusted code unconfined is the outcome this module exists to
    prevent.
    """
    paths = list(runtime) if runtime is not None else runtime_paths()
    for name, available, apply in _BACKENDS:
        try:
            if not available():
                continue
        except OSError:  # pragma: no cover
            continue
        return apply(workdir, paths, root)
    raise ConfinementUnavailable(
        f"no filesystem confinement backend for platform {sys.platform!r} "
        f"({platform.machine()}); model-written code will not be run "
        "unconfined - see SPEC §18"
    )
