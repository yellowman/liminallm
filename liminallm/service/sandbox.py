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
import operator
import os
import resource
import socket
import tempfile
import threading
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar
from urllib.parse import urlparse

import httpx

from liminallm.logging import get_logger

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
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise ValueError("unsupported binary operator")
        return op(
            _eval_node(node.left, names, allowed_callables, _depth + 1),
            _eval_node(node.right, names, allowed_callables, _depth + 1),
        )

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
    cgroup_cpu_shares: int = 256  # Lower than default 1024 for tools
    cgroup_memory_limit_mb: int = 512

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

    def connection_allowlist(self) -> list[str]:
        """Hosts the sandbox may connect to for outbound requests.

        If a proxy is configured, only the proxy host is reachable; otherwise
        the target allowlist is used directly.
        """

        if self.proxy_url:
            parsed = urlparse(self.proxy_url)
            return [h for h in [parsed.hostname] if h]
        return list(self.allowlist)


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
) -> ToolNetworkPolicy:
    """Create a normalized ToolNetworkPolicy from raw values."""

    return ToolNetworkPolicy(
        allowlist=_normalize_allowlist(list(allowlist or [])),
        proxy_url=proxy_url,
        connect_timeout=connect_timeout,
        total_timeout=total_timeout,
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


def _enforce_network_allowlist(host: str) -> None:
    policy: ToolNetworkPolicy | None = getattr(_NETWORK_POLICY_STATE, "policy", None)
    if not policy:
        return

    allowed_hosts = policy.connection_allowlist()
    if not allowed_hosts:
        raise SandboxError("Tool network access disabled (empty allowlist)")

    if not _host_matches_allowlist(host, allowed_hosts):
        raise SandboxError(f"Egress host '{host}' is not allowlisted for tools")


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
    """Apply resource limits for tool execution.

    Uses Python's resource module to set limits on the current process.
    These limits are inherited by child processes.

    SPEC §18: CPU shares and memory hard cap.

    Returns:
        Dict indicating which limits were successfully applied
    """
    results = {}

    # Memory limit (virtual memory)
    memory_bytes = config.max_memory_mb * 1024 * 1024
    try:
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        results["memory"] = True
    except (ValueError, OSError) as e:
        logger.warning("sandbox_memory_limit_failed", error=str(e))
        results["memory"] = False

    # CPU time limit
    try:
        resource.setrlimit(
            resource.RLIMIT_CPU,
            (config.max_cpu_seconds, config.max_cpu_seconds + 5),
        )
        results["cpu"] = True
    except (ValueError, OSError) as e:
        logger.warning("sandbox_cpu_limit_failed", error=str(e))
        results["cpu"] = False

    # File size limit
    file_size_bytes = config.max_file_size_mb * 1024 * 1024
    try:
        resource.setrlimit(resource.RLIMIT_FSIZE, (file_size_bytes, file_size_bytes))
        results["file_size"] = True
    except (ValueError, OSError) as e:
        logger.warning("sandbox_file_limit_failed", error=str(e))
        results["file_size"] = False

    # Core dump disabled
    try:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        results["core"] = True
    except (ValueError, OSError) as e:
        logger.warning("sandbox_core_limit_failed", error=str(e))
        results["core"] = False

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


def check_privileged_access(
    tool_name: str,
    config: SandboxConfig,
    *,
    user_role: Optional[str] = None,
    artifact_owner_id: Optional[str] = None,
    requesting_user_id: Optional[str] = None,
) -> None:
    """Validate privileged tool access.

    SPEC §18: `privileged:true` tools require admin-owned artifacts and are
    never called by default workflows.

    Args:
        tool_name: Name of the tool being invoked
        config: Sandbox configuration
        user_role: Role of the requesting user
        artifact_owner_id: Owner ID of the tool artifact
        requesting_user_id: ID of the user making the request

    Raises:
        PrivilegedToolError: If access is denied
    """
    if not config.privileged:
        return

    # Privileged tools require admin role
    if user_role != "admin":
        raise PrivilegedToolError(
            f"Tool '{tool_name}' requires admin role (current: {user_role})"
        )

    logger.info(
        "privileged_tool_access",
        tool=tool_name,
        user_id=requesting_user_id,
        user_role=user_role,
    )


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


def run_in_sandbox(
    func: Callable[..., T],
    *args: Any,
    config: Optional[SandboxConfig] = None,
    **kwargs: Any,
) -> T:
    """Execute a function within sandbox constraints.

    This applies resource limits before executing the function.
    Note: Resource limits persist until the process ends.

    Args:
        func: Function to execute
        *args: Positional arguments for function
        config: Sandbox configuration (uses default if None)
        **kwargs: Keyword arguments for function

    Returns:
        Function result
    """
    cfg = config or DEFAULT_SANDBOX_CONFIG
    ensure_scratch_dir(cfg)
    apply_resource_limits(cfg)
    return func(*args, **kwargs)


# Cgroup v2 integration (when available)
def setup_cgroup(
    cgroup_name: str,
    config: SandboxConfig,
    *,
    cgroup_base: str = "/sys/fs/cgroup",
) -> Optional[str]:
    """Set up cgroup v2 for tool sandboxing.

    This function attempts to create and configure a cgroup for the tool.
    Requires appropriate permissions (typically root or cgroup delegation).

    Args:
        cgroup_name: Name for the cgroup
        config: Sandbox configuration
        cgroup_base: Base path for cgroup v2 filesystem

    Returns:
        Path to created cgroup, or None if cgroups are not available
    """
    cgroup_path = Path(cgroup_base) / "liminallm" / cgroup_name

    try:
        # Create cgroup hierarchy
        cgroup_path.mkdir(parents=True, exist_ok=True)

        # Set memory limit
        memory_max = cgroup_path / "memory.max"
        if memory_max.exists():
            memory_bytes = config.cgroup_memory_limit_mb * 1024 * 1024
            memory_max.write_text(str(memory_bytes))

        # Set CPU weight (similar to shares in v1)
        cpu_weight = cgroup_path / "cpu.weight"
        if cpu_weight.exists():
            # Convert shares (1-1024 scale) to weight (1-10000 scale)
            weight = int((config.cgroup_cpu_shares / 1024) * 100)
            cpu_weight.write_text(str(max(1, weight)))

        logger.info("cgroup_setup_success", cgroup=str(cgroup_path))
        return str(cgroup_path)

    except PermissionError:
        logger.warning("cgroup_setup_permission_denied", cgroup=str(cgroup_path))
        return None
    except Exception as e:
        logger.warning("cgroup_setup_failed", cgroup=str(cgroup_path), error=str(e))
        return None


def add_to_cgroup(cgroup_path: str, pid: Optional[int] = None) -> bool:
    """Add a process to a cgroup.

    Args:
        cgroup_path: Path to cgroup
        pid: Process ID to add (defaults to current process)

    Returns:
        True if successful, False otherwise
    """
    if pid is None:
        pid = os.getpid()

    procs_file = Path(cgroup_path) / "cgroup.procs"
    try:
        procs_file.write_text(str(pid))
        logger.debug("cgroup_process_added", cgroup=cgroup_path, pid=pid)
        return True
    except Exception as e:
        logger.warning("cgroup_add_failed", cgroup=cgroup_path, pid=pid, error=str(e))
        return False


def cleanup_cgroup(cgroup_path: str) -> bool:
    """Clean up a cgroup after tool execution.

    Args:
        cgroup_path: Path to cgroup to remove

    Returns:
        True if successful, False otherwise
    """
    try:
        cgroup = Path(cgroup_path)
        if cgroup.exists():
            # Move all processes to parent before removing
            procs = (cgroup / "cgroup.procs").read_text().strip().split()
            parent_procs = cgroup.parent / "cgroup.procs"
            for pid in procs:
                if pid:
                    parent_procs.write_text(pid)
            cgroup.rmdir()
        return True
    except Exception as e:
        logger.warning("cgroup_cleanup_failed", cgroup=cgroup_path, error=str(e))
        return False


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
