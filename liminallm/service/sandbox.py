"""Expression evaluation and tool execution sandbox.

SPEC §18: Tool workers run under a fixed UID with cgroup limits (CPU shares,
memory hard cap) and no filesystem access except a tmp scratch.

This module provides:
- Safe expression evaluation (safe_eval_expr)
- Resource limits (CPU, memory) for tool execution
- Filesystem isolation (only tmp scratch allowed)
- Privileged tool access controls
"""
from __future__ import annotations

import ast
import ipaddress
import multiprocessing
import operator
import os
import resource
import signal
import socket
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar
from urllib.parse import urlparse

import httpx

from liminallm.logging import get_logger
from liminallm.service.wire import (
    DEFAULT_MAX_FRAME_BYTES,
    ERROR_FRAME_BYTES,
    WireError,
    error_payload,
    recv_frame,
    send_frame,
)

logger = get_logger(__name__)

T = TypeVar("T")

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.FloorDiv: operator.floordiv,
}

_CMP_OPS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
}


_MAX_RECURSION_DEPTH = 100
# Bounds to stop a single expression from pinning a CPU core or exhausting memory
# via exponentiation (9**9**9) or sequence/int repetition ([0]*10**9, "a"*10**8).
_MAX_POW_EXPONENT = 1000
_MAX_INT_BITS = 10000
_MAX_SEQUENCE_LEN = 100_000


def _guard_binop(op_type: type, left: Any, right: Any) -> None:
    """Reject arithmetic whose result would blow up CPU/memory."""
    if op_type is ast.Pow and isinstance(left, int) and isinstance(right, int):
        if right > _MAX_POW_EXPONENT or (
            right > 0 and left.bit_length() * right > _MAX_INT_BITS
        ):
            raise ValueError("exponentiation result too large")
    elif op_type is ast.Mult:
        if isinstance(left, (str, bytes, list, tuple)) and isinstance(right, int):
            if len(left) * max(right, 0) > _MAX_SEQUENCE_LEN:
                raise ValueError("sequence repetition too large")
        elif isinstance(right, (str, bytes, list, tuple)) and isinstance(left, int):
            if len(right) * max(left, 0) > _MAX_SEQUENCE_LEN:
                raise ValueError("sequence repetition too large")
        elif isinstance(left, int) and isinstance(right, int):
            if left.bit_length() + right.bit_length() > _MAX_INT_BITS:
                raise ValueError("multiplication result too large")


def _eval_node(
    node: ast.AST,
    names: Mapping[str, Any],
    allowed_callables: Mapping[str, Any] | None,
    _depth: int = 0,
) -> Any:
    if _depth > _MAX_RECURSION_DEPTH:
        raise ValueError("expression too deeply nested")

    if isinstance(node, ast.Expression):
        return _eval_node(node.body, names, allowed_callables, _depth + 1)

    if isinstance(node, ast.Constant):
        return node.value

    if isinstance(node, ast.Name):
        if node.id in names:
            return names[node.id]
        raise ValueError(f"unknown name {node.id}")

    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            result = True
            for value in node.values:
                result = bool(_eval_node(value, names, allowed_callables, _depth + 1))
                if not result:
                    break
            return result
        if isinstance(node.op, ast.Or):
            result = False
            for value in node.values:
                result = bool(_eval_node(value, names, allowed_callables, _depth + 1))
                if result:
                    break
            return result
        raise ValueError("unsupported boolean operator")

    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.Not):
            return not bool(
                _eval_node(node.operand, names, allowed_callables, _depth + 1)
            )
        if isinstance(node.op, ast.USub):
            return -_eval_node(node.operand, names, allowed_callables, _depth + 1)
        if isinstance(node.op, ast.UAdd):
            return +_eval_node(node.operand, names, allowed_callables, _depth + 1)
        raise ValueError("unsupported unary operator")

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        op = _BIN_OPS.get(op_type)
        if op is None:
            raise ValueError("unsupported binary operator")
        left = _eval_node(node.left, names, allowed_callables, _depth + 1)
        right = _eval_node(node.right, names, allowed_callables, _depth + 1)
        _guard_binop(op_type, left, right)
        return op(left, right)

    if isinstance(node, ast.Compare):
        left = _eval_node(node.left, names, allowed_callables, _depth + 1)
        for op_node, comparator in zip(node.ops, node.comparators):
            op = _CMP_OPS.get(type(op_node))
            if op is None:
                raise ValueError("unsupported comparator")
            right = _eval_node(comparator, names, allowed_callables, _depth + 1)
            if not op(left, right):
                return False
            left = right
        return True

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("callable references must be simple names")
        if not allowed_callables or node.func.id not in allowed_callables:
            raise ValueError("callable is not permitted")
        func = allowed_callables[node.func.id]
        if not callable(func):
            raise ValueError("call target is not callable")
        args = [
            _eval_node(arg, names, allowed_callables, _depth + 1) for arg in node.args
        ]
        # Reject **kwargs unpacking (kw.arg is None when using **dict syntax)
        for kw in node.keywords:
            if kw.arg is None:
                raise ValueError("keyword unpacking (**kwargs) not permitted")
        kwargs = {
            kw.arg: _eval_node(kw.value, names, allowed_callables, _depth + 1)
            for kw in node.keywords
        }
        return func(*args, **kwargs)

    if isinstance(node, ast.Subscript):
        target = _eval_node(node.value, names, allowed_callables, _depth + 1)
        index = _eval_node(node.slice, names, allowed_callables, _depth + 1)
        if not isinstance(target, (Mapping, Sequence, str, bytes)):
            raise ValueError("subscript targets must be sequences or mappings")
        try:
            return target[index]
        except Exception as exc:
            raise ValueError(f"invalid subscript access: {exc}")

    if isinstance(node, ast.Tuple):
        return tuple(
            _eval_node(elt, names, allowed_callables, _depth + 1) for elt in node.elts
        )

    if isinstance(node, ast.List):
        return [
            _eval_node(elt, names, allowed_callables, _depth + 1) for elt in node.elts
        ]

    if isinstance(node, ast.Dict):
        return {
            _eval_node(k, names, allowed_callables, _depth + 1): _eval_node(
                v, names, allowed_callables, _depth + 1
            )
            for k, v in zip(node.keys, node.values)
        }

    raise ValueError(f"unsupported expression node: {type(node).__name__}")


def safe_eval_expr(
    expr: str,
    names: Mapping[str, Any],
    allowed_callables: Mapping[str, Any] | None = None,
) -> Any:
    """Evaluate an expression with a constrained AST allowlist.

    Only supports boolean operators, comparisons, indexing, numeric ops, and calling
    explicitly allowed callables provided via ``allowed_callables``. Attribute access, comprehensions, and
    other dynamic constructs are rejected to prevent sandbox escapes.
    """

    try:
        parsed = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError("invalid expression") from exc

    for node in ast.walk(parsed):
        if isinstance(
            node,
            (
                ast.Attribute,
                ast.Lambda,
                ast.ListComp,
                ast.SetComp,
                ast.DictComp,
                ast.GeneratorExp,
                ast.Await,
                ast.Yield,
                ast.YieldFrom,
                ast.ClassDef,
                ast.FunctionDef,
            ),
        ):
            raise ValueError("disallowed syntax in expression")

    return _eval_node(parsed, names, allowed_callables)


# =========================================================================
# Tool Execution Sandbox (SPEC §18)
# =========================================================================


class SandboxError(Exception):
    """Raised when sandbox constraints are violated."""


class PrivilegedToolError(Exception):
    """Raised when a privileged tool is invoked without proper authorization."""


@dataclass
class SandboxConfig:
    """Configuration for tool sandbox execution.

    SPEC §18: Tool workers run with constrained resources and limited
    filesystem access.

    Attributes:
        max_memory_mb: Maximum memory in MB (default: 512)
        max_cpu_seconds: Maximum CPU time in seconds (default: 30)
        max_file_size_mb: Maximum file size tools can create (default: 100)
        scratch_dir: Temporary scratch directory for tool file I/O
        allowed_paths: Additional paths tools are allowed to access (read-only)
        privileged: Whether this is a privileged tool (admin-only)
    """

    max_memory_mb: int = 512
    max_cpu_seconds: int = 30
    max_file_size_mb: int = 100
    scratch_dir: Optional[Path] = None
    allowed_paths: list[Path] = field(default_factory=list)
    privileged: bool = False

    # Cgroup configuration (when available)

    def __post_init__(self) -> None:
        if self.scratch_dir is None:
            self.scratch_dir = Path(tempfile.gettempdir()) / "liminallm_sandbox"


# Default sandbox configurations
DEFAULT_SANDBOX_CONFIG = SandboxConfig()

PRIVILEGED_SANDBOX_CONFIG = SandboxConfig(
    max_memory_mb=1024,
    max_cpu_seconds=120,
    max_file_size_mb=500,
    privileged=True,
)


@dataclass
class ToolNetworkPolicy:
    """Network egress policy for tool execution (SPEC §18).

    Attributes:
        allowlist: Allowed target host patterns (hostname, wildcard, or CIDR)
        proxy_url: Optional HTTP proxy all tool fetches must use
        connect_timeout: Connection timeout in seconds
        total_timeout: Total request timeout in seconds
    """

    allowlist: list[str] = field(default_factory=list)
    proxy_url: Optional[str] = None
    connect_timeout: float = 10.0
    total_timeout: float = 30.0
    # Hosts the service itself must reach to function - the configured model
    # provider, above all. These are infrastructure, not tool fetch targets:
    # they are connectable (so provider calls inside a tool handler work) but
    # never appear in `allowlist`, so a tool cannot fetch from them.
    infrastructure_hosts: list[str] = field(default_factory=list)

    def connection_allowlist(self) -> list[str]:
        """Hosts the sandbox may open sockets to.

        If a proxy is configured, tool traffic must go through it; otherwise
        the target allowlist applies. Infrastructure hosts are always
        connectable, since blocking them would break the model backend.
        """

        if self.proxy_url:
            parsed = urlparse(self.proxy_url)
            hosts = [h for h in [parsed.hostname] if h]
        else:
            hosts = list(self.allowlist)
        return hosts + [h for h in self.infrastructure_hosts if h]


def _normalize_allowlist(entries: Sequence[str] | None) -> list[str]:
    normalized: list[str] = []
    for entry in entries or []:
        stripped = entry.strip().lower()
        if stripped:
            normalized.append(stripped)
    return normalized


def build_tool_network_policy(
    *,
    allowlist: Sequence[str] | None,
    proxy_url: Optional[str],
    connect_timeout: float = 10.0,
    total_timeout: float = 30.0,
    infrastructure_hosts: Sequence[str] | None = None,
) -> ToolNetworkPolicy:
    """Create a normalized ToolNetworkPolicy from raw values."""

    return ToolNetworkPolicy(
        allowlist=_normalize_allowlist(list(allowlist or [])),
        proxy_url=proxy_url,
        connect_timeout=connect_timeout,
        total_timeout=total_timeout,
        infrastructure_hosts=_normalize_allowlist(list(infrastructure_hosts or [])),
    )


_NETWORK_POLICY_STATE = threading.local()


def _host_matches_allowlist(host: str, allowlist: Sequence[str]) -> bool:
    if not host:
        return False
    lowered = host.lower()
    for entry in allowlist:
        candidate = entry.lower()
        if candidate.startswith("*."):
            if lowered.endswith(candidate[1:]):
                return True
        elif lowered == candidate:
            return True
        else:
            if "/" in candidate:
                try:
                    net = ipaddress.ip_network(candidate, strict=False)
                    ip_obj = ipaddress.ip_address(host)
                    if ip_obj in net:
                        return True
                except ValueError:
                    continue
    return False


_DNS_CACHE: dict[str, tuple[float, frozenset[str]]] = {}
_DNS_CACHE_TTL = 60.0


def _resolve_host_ips(host: str) -> frozenset[str]:
    """Addresses ``host`` currently resolves to, cached briefly."""
    now = time.monotonic()
    cached = _DNS_CACHE.get(host)
    if cached and now - cached[0] < _DNS_CACHE_TTL:
        return cached[1]
    try:
        ips = frozenset(
            info[4][0] for info in socket.getaddrinfo(host, None) if info[4]
        )
    except OSError:
        ips = frozenset()
    _DNS_CACHE[host] = (now, ips)
    return ips


def _enforce_network_allowlist(host: str) -> None:
    policy: ToolNetworkPolicy | None = getattr(_NETWORK_POLICY_STATE, "policy", None)
    if not policy:
        return

    allowed_hosts = policy.connection_allowlist()
    if not allowed_hosts:
        raise SandboxError("Tool network access disabled (empty allowlist)")

    if _host_matches_allowlist(host, allowed_hosts):
        return

    # HTTP clients (httpx, urllib3) resolve DNS themselves and then connect to
    # an address, so the guard is handed an IP literal rather than the name the
    # allowlist is written in. Compare against what the allowlisted names
    # resolve to; anything else is refused.
    try:
        ipaddress.ip_address(host)
    except ValueError:
        raise SandboxError(f"Egress host '{host}' is not allowlisted for tools")

    for entry in allowed_hosts:
        if entry.startswith("*.") or "/" in entry:
            continue  # wildcards and CIDRs are handled by the name/CIDR match
        if host in _resolve_host_ips(entry):
            return

    raise SandboxError(f"Egress address '{host}' is not allowlisted for tools")


_ORIGINAL_CREATE_CONNECTION = socket.create_connection
_ORIGINAL_SOCKET_CONNECT = socket.socket.connect


def _guarded_create_connection(address, *args, **kwargs):  # type: ignore[override]
    host = address[0] if isinstance(address, (list, tuple)) and address else None
    if host:
        _enforce_network_allowlist(str(host))
    return _ORIGINAL_CREATE_CONNECTION(address, *args, **kwargs)


def _guarded_socket_connect(self: socket.socket, address):  # type: ignore[override]
    host = address[0] if isinstance(address, (list, tuple)) and address else None
    if host:
        _enforce_network_allowlist(str(host))
    return _ORIGINAL_SOCKET_CONNECT(self, address)


socket.create_connection = _guarded_create_connection
socket.socket.connect = _guarded_socket_connect


@contextmanager
def tool_network_guard(policy: ToolNetworkPolicy):
    """Apply thread-local network egress policy for tool execution."""

    previous = getattr(_NETWORK_POLICY_STATE, "policy", None)
    _NETWORK_POLICY_STATE.policy = policy
    try:
        yield
    finally:
        if previous is None:
            _NETWORK_POLICY_STATE.__dict__.pop("policy", None)
        else:
            _NETWORK_POLICY_STATE.policy = previous


class AllowlistedFetcher:
    """HTTP client enforcing tool network allowlist and proxy requirements."""

    def __init__(self, policy: ToolNetworkPolicy):
        self.policy = policy

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Optional[dict[str, str]] = None,
        data: Any = None,
        json: Any = None,
    ) -> httpx.Response:
        parsed = urlparse(url)
        host = parsed.hostname
        if not host:
            raise SandboxError("URL is missing host for tool fetch")

        if not self.policy.allowlist:
            raise SandboxError("Tool network allowlist is empty; outbound fetch blocked")

        if not _host_matches_allowlist(host, self.policy.allowlist):
            raise SandboxError(f"Target host '{host}' is not allowlisted for tool fetch")

        timeout = httpx.Timeout(self.policy.total_timeout, connect=self.policy.connect_timeout)
        try:
            return httpx.request(
                method,
                url,
                headers=headers,
                data=data,
                json=json,
                timeout=timeout,
                proxy=self.policy.proxy_url,
                follow_redirects=False,
            )
        except httpx.TimeoutException as exc:
            raise SandboxError("tool fetch timed out") from exc
        except httpx.HTTPError as exc:
            raise SandboxError(f"tool fetch failed: {exc}") from exc


def validate_path_access(
    path: str | Path,
    config: SandboxConfig,
    *,
    write: bool = False,
) -> Path:
    """Validate that a path is accessible within sandbox constraints.

    SPEC §18: No filesystem access except tmp scratch.

    Args:
        path: Path to validate
        config: Sandbox configuration
        write: Whether write access is needed

    Returns:
        Validated Path object

    Raises:
        SandboxError: If path is not allowed
    """
    path_obj = Path(path).resolve()

    # Always allow scratch directory
    if config.scratch_dir:
        scratch_resolved = config.scratch_dir.resolve()
        if path_obj == scratch_resolved or scratch_resolved in path_obj.parents:
            return path_obj

    # Check allowed paths (read-only unless it's the scratch)
    if not write:
        for allowed in config.allowed_paths:
            allowed_resolved = allowed.resolve()
            if path_obj == allowed_resolved or allowed_resolved in path_obj.parents:
                return path_obj

    raise SandboxError(
        f"Path '{path}' is not accessible. Tools can only access the scratch "
        f"directory at '{config.scratch_dir}'"
    )


def apply_resource_limits(config: SandboxConfig) -> dict[str, bool]:
    """Apply resource limits to the calling process, or raise.

    All four are required, because this function is shared. §19.5 gives the
    parser child memory, CPU and file size; §21.2 gives `run_python` those
    *and* no core dumps, and `run_python` comes through here. The stricter
    contract governs both: suppressing core dumps is stricter than extraction
    needs and entirely compatible with it, where a mode switch would exist
    only to let one untrusted child dump core.

    Every one of these used to be caught and recorded in a returned dict, and
    the only caller ignored the dict. A refused cap therefore read as success
    and untrusted code ran unbounded; the log line said so and nothing acted
    on it. Reporting a failure to a caller that does not check is the same as
    not detecting it.

    Returns:
        The limits now in force.

    Raises:
        SandboxError: any of them was refused.
    """
    memory = config.max_memory_mb * 1024 * 1024
    file_size = config.max_file_size_mb * 1024 * 1024
    required = (
        ("memory", resource.RLIMIT_AS, (memory, memory)),
        ("cpu", resource.RLIMIT_CPU, (config.max_cpu_seconds, config.max_cpu_seconds + 5)),
        ("file_size", resource.RLIMIT_FSIZE, (file_size, file_size)),
        ("core", resource.RLIMIT_CORE, (0, 0)),
    )
    results: dict[str, bool] = {}
    for name, which, limit in required:
        try:
            resource.setrlimit(which, limit)
        except (ValueError, OSError) as exc:
            logger.error("sandbox_limit_refused", limit=name, error=str(exc))
            raise SandboxError(
                f"the {name} limit could not be applied ({exc}); refusing to "
                "run unbounded in a sandbox"
            ) from exc
        results[name] = True

    return results


def ensure_scratch_dir(config: SandboxConfig) -> Path:
    """Ensure scratch directory exists and is accessible.

    Returns:
        Path to scratch directory
    """
    if config.scratch_dir is None:
        config.scratch_dir = Path(tempfile.gettempdir()) / "liminallm_sandbox"

    config.scratch_dir.mkdir(parents=True, exist_ok=True)
    return config.scratch_dir


# A `check_privileged_access(..., artifact_owner_id=...)` helper used to live
# here. It named the SPEC §18 rule - `privileged:true` requires an *admin-owned
# artifact* - accepted the owner id, and then checked only the caller's role.
# Nothing in the service called it, so the rule it claimed to enforce was
# enforced nowhere. The two halves now sit where each can actually be answered:
# `get_tool_sandbox_config` asks about the caller, and `WorkflowEngine`'s
# `ToolDescriptor` carries the persisted artifact row that answers ownership.


class SandboxedFileHandle:
    """File handle wrapper that enforces sandbox constraints."""

    def __init__(self, path: Path, mode: str, config: SandboxConfig):
        self.path = path
        self.mode = mode
        self.config = config
        self._handle = None

    def __enter__(self):
        write = "w" in self.mode or "a" in self.mode
        validate_path_access(self.path, self.config, write=write)
        self._handle = open(self.path, self.mode)
        return self._handle

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._handle:
            self._handle.close()
        return False


def sandbox_open(
    path: str | Path, mode: str = "r", config: Optional[SandboxConfig] = None
):
    """Open a file within sandbox constraints.

    Use instead of built-in open() in tool code.
    """
    cfg = config or DEFAULT_SANDBOX_CONFIG
    return SandboxedFileHandle(Path(path), mode, cfg)


#: Exception types the parent will rebuild from a child's error frame.
#:
#: The child names a type and the parent decides whether that name means
#: anything, from this fixed vocabulary plus whatever the caller adds. Nothing
#: is imported, resolved, or evaluated from the child's string, and every entry
#: here is constructible from a single message - an unknown or unconstructible
#: name becomes a `SandboxError` carrying the name as text.
_RECONSTRUCTABLE_ERRORS: dict[str, type[BaseException]] = {
    cls.__name__: cls
    for cls in (
        ArithmeticError,
        AttributeError,
        EOFError,
        IndexError,
        KeyError,
        LookupError,
        MemoryError,
        NotImplementedError,
        OSError,
        OverflowError,
        RecursionError,
        RuntimeError,
        TimeoutError,
        TypeError,
        ValueError,
        ZeroDivisionError,
    )
}


def _rebuild_error(
    error: Any, extra: Optional[Mapping[str, type[BaseException]]]
) -> BaseException:
    """Turn an error frame back into something the caller can catch."""
    if not isinstance(error, Mapping):
        return SandboxError(f"sandboxed execution failed: {error!r}")
    name = error.get("type")
    message = str(error.get("message") or "")
    cls: Optional[type[BaseException]] = None
    if isinstance(name, str):
        cls = (extra or {}).get(name) or _RECONSTRUCTABLE_ERRORS.get(name)
    if cls is None:
        return SandboxError(f"sandboxed execution failed: {name}: {message}")
    try:
        return cls(message)
    except Exception:  # noqa: BLE001 - a type we cannot build is not fatal
        return SandboxError(f"sandboxed execution failed: {name}: {message}")


def _sandbox_entry(conn, func, args, kwargs, config, max_result_bytes) -> None:
    """Child-process entrypoint: lead a group, apply rlimits, run, report.

    Order is deliberate. `setsid` and the ready frame come before the limits,
    so the parent can reach this whole tree even when the limits are what
    failed; the body comes last, so nothing untrusted runs until both are
    settled.

    Nothing leaves here except JSON. Exceptions used to be sent as objects so
    a caller could catch its own type, which meant the *parent* unpickled a
    class the child chose - the decode ran the payload before any check. The
    type now crosses as a name and the parent rebuilds it from a vocabulary
    the parent owns (`_rebuild_error`).
    """
    try:
        os.setsid()
        pgid = os.getpgid(0)
    except (AttributeError, OSError):
        # No sessions here, or already a leader. The parent compares the pgid
        # against this pid and keeps killing by single pid when they differ.
        pgid = 0
    try:
        send_frame(conn, {"ready": True, "pid": os.getpid(), "pgid": pgid})
    except Exception:  # noqa: BLE001 - parent gave up; nothing to report to
        conn.close()
        return

    # An error frame is bounded by construction and the parent reads with at
    # least this much, so a failure is always reportable - even to a caller
    # whose results are smaller than the report of one.
    error_bytes = max(max_result_bytes, ERROR_FRAME_BYTES)
    try:
        apply_resource_limits(config)
        frame: dict = {"ok": True, "result": func(*args, **kwargs)}
    except BaseException as exc:  # noqa: BLE001 - report to parent, then exit
        frame = {"ok": False, "error": error_payload(exc)}
    try:
        send_frame(
            conn,
            frame,
            max_bytes=error_bytes if not frame.get("ok") else max_result_bytes,
        )
    except WireError as exc:
        # The result did not fit, or was not data. Say which - the parent's
        # own `recv_bytes` cap would otherwise report this as a dead child.
        try:
            send_frame(conn, {"ok": False, "error": error_payload(exc)}, max_bytes=error_bytes)
        except Exception:  # noqa: BLE001 - broken pipe; the parent will time out
            pass
    except Exception:  # noqa: BLE001 - broken pipe; the parent will time out
        pass
    finally:
        conn.close()


def _terminate_tree(proc, *, group: bool) -> None:
    """Kill the child and, when it leads one, everything in its group.

    The group goes first and the reap second, in that order. A parser spawns
    grandchildren - `pdftoppm`, tesseract - which are not this process's
    children and survive the child that started them; the group is the only
    handle on them, and it stops naming anything once the leader has been
    reaped and its pid recycled.

    `group` is the parent's knowledge that the child answered with a pgid of
    its own. The `getpgid` re-check is the safety: a group kill is only ever
    aimed at a group the target *leads*, because signalling a group the child
    merely belongs to would reach this server and its siblings.
    """
    pid = proc.pid
    if group and pid and hasattr(os, "killpg"):
        try:
            if os.getpgid(pid) == pid:
                os.killpg(pid, signal.SIGKILL)
        except OSError:
            pass  # already gone, or not ours to signal
    if proc.is_alive():
        proc.kill()
    proc.join(5)


def run_in_sandbox(
    func: Callable[..., T],
    *args: Any,
    config: Optional[SandboxConfig] = None,
    timeout: Optional[float] = None,
    on_child: Optional[
        Callable[[int, Callable[[], None]], Optional[Callable[[], None]]]
    ] = None,
    max_result_bytes: Optional[int] = None,
    error_types: Optional[Mapping[str, type[BaseException]]] = None,
    **kwargs: Any,
) -> T:
    """Execute a function in a resource-limited child process.

    The rlimits (memory hard cap, CPU seconds, max file size, no core dumps)
    are applied inside a spawned child so the API process itself is never
    constrained - applying them in-process would permanently cripple the
    server, which is why the old in-process variant was unusable. A
    wall-clock timeout backstops the CPU rlimit, and the overrun kill reaches
    the child's whole process group, not just the pid it started.

    The child is assumed hostile (SPEC §19.5: "assume the parsers are
    compromisable"), so what comes back is JSON, never a pickle. ``func`` and
    its arguments still travel *out* by pickle, which is safe in the direction
    that matters: the parent chooses them. Coming back, ``func`` must return
    data - JSON has no tuple, so one returns as a list, and an object returns
    as a `SandboxError` rather than as itself.

    Args:
        func: Function to execute
        *args: Positional arguments for function
        config: Sandbox configuration (uses default if None)
        timeout: Wall-clock seconds before the child is killed
                 (default: config.max_cpu_seconds + 15)
        on_child: Called with (pid, reap) as soon as the child exists, so the
                 invocation that asked for it can kill it. This child is the
                 *parent's* child, not the worker's, so killing the worker
                 never reaches it - registering it is what makes the tree
                 reachable (SPEC §18). It may return a callable, which is
                 invoked once the child has been reaped: a pid outlives the
                 process only as a number, and the kernel reuses it, so a
                 registration left behind is authority over whoever gets it
                 next.
        max_result_bytes: Largest encoded result this caller can legitimately
                 receive. Derive it from the caller's own budgets - only the
                 caller knows what its results can weigh - because without a
                 cap a child can turn its permitted memory into the parent's.
        error_types: Extra ``{name: class}`` the parent will rebuild a failure
                 into, for callers that translate specific failures. The
                 parent supplies the classes; the child only supplies a name.
        **kwargs: Keyword arguments for function

    Returns:
        Function result

    Raises:
        SandboxError: on timeout, an oversized or malformed result, or if the
                      child dies without one (e.g. a resource limit)
        A rebuilt exception, when func itself failed with a known type
    """
    cfg = config or DEFAULT_SANDBOX_CONFIG
    ensure_scratch_dir(cfg)
    result_bytes = (
        max_result_bytes if max_result_bytes is not None else DEFAULT_MAX_FRAME_BYTES
    )
    # spawn (not fork): forking a threaded server process risks deadlocks,
    # and a fresh interpreter is the point of the isolation boundary.
    ctx = multiprocessing.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=_sandbox_entry,
        args=(child_conn, func, args, kwargs, cfg, result_bytes),
        daemon=True,
    )
    proc.start()
    child_conn.close()
    release: Optional[Callable[[], None]] = None
    if on_child is not None and proc.pid:
        release = on_child(proc.pid, lambda: proc.join(5))
    wall_timeout = timeout if timeout is not None else cfg.max_cpu_seconds + 15
    # One deadline for the whole call, handshake included. The wait for the
    # child to announce itself is start-up time, which this budget has always
    # covered - giving the handshake a budget of its own would let a caller
    # asking for three seconds wait for a minute.
    deadline = time.monotonic() + wall_timeout
    leads_group = False
    try:
        leads_group = _await_ready(parent_conn, proc, deadline)
        if not parent_conn.poll(max(0.0, deadline - time.monotonic())):
            raise SandboxError(
                f"sandboxed execution exceeded {wall_timeout:.0f}s wall clock"
            )
        try:
            # Headroom for the error frame: a caller whose results are small
            # must still be able to receive the reason they failed.
            frame = recv_frame(
                parent_conn, max_bytes=max(result_bytes, ERROR_FRAME_BYTES)
            )
        except EOFError as exc:
            raise SandboxError(
                "sandboxed process died before returning a result "
                "(resource limit exceeded?)"
            ) from exc
        except WireError as exc:
            raise SandboxError(
                f"sandboxed process returned nothing usable: {exc}"
            ) from exc
    finally:
        parent_conn.close()
        _terminate_tree(proc, group=leads_group)
        # Reaped, so the pid is now just a number the kernel may hand to
        # anyone. Releasing the registration here is what stops a later
        # teardown signalling whoever inherits it.
        if release is not None:
            release()
    if not frame.get("ok"):
        raise _rebuild_error(frame.get("error"), error_types)
    return frame.get("result")


def _await_ready(parent_conn, proc, deadline: float) -> bool:
    """Wait for the child's first frame; True when it leads its own group.

    `Process.start()` returns before the child has run a line, so the parent
    cannot know the child's process group at that moment - and a `killpg`
    aimed at a pid that has not yet called `setsid` reaches the group the
    child was *born* into, which is this server's. Asking the child is what
    makes the group safe to signal, and the answer only counts when all three
    agree: the pid we started, the pid it reports, and the group it leads.
    """
    try:
        if not parent_conn.poll(max(0.0, deadline - time.monotonic())):
            raise SandboxError("sandboxed process never started")
        frame = recv_frame(parent_conn, max_bytes=ERROR_FRAME_BYTES)
    except (EOFError, WireError) as exc:
        raise SandboxError(f"sandboxed process did not announce itself: {exc}") from exc
    if not frame.get("ready"):
        raise SandboxError("sandboxed process spoke out of turn")
    pid, pgid = frame.get("pid"), frame.get("pgid")
    return bool(pid) and pid == pgid == proc.pid


def get_tool_sandbox_config(
    tool_spec: Optional[dict],
    *,
    user_role: Optional[str] = None,
) -> SandboxConfig:
    """Get sandbox configuration for a tool based on its specification.

    Args:
        tool_spec: Tool specification dict
        user_role: Role of the invoking user

    Returns:
        SandboxConfig appropriate for the tool
    """
    if not tool_spec:
        return DEFAULT_SANDBOX_CONFIG

    is_privileged = tool_spec.get("privileged", False)

    if is_privileged:
        if user_role != "admin":
            raise PrivilegedToolError(
                f"Privileged tool requires admin role (current: {user_role})"
            )
        return PRIVILEGED_SANDBOX_CONFIG

    # Custom limits from tool spec
    limits = tool_spec.get("resource_limits", {})
    return SandboxConfig(
        max_memory_mb=min(limits.get("max_memory_mb", 512), 1024),
        max_cpu_seconds=min(limits.get("max_cpu_seconds", 30), 120),
        max_file_size_mb=min(limits.get("max_file_size_mb", 100), 500),
        privileged=False,
    )
