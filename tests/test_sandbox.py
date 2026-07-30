import math
import time

import pytest

from liminallm.service.sandbox import (
    SandboxConfig,
    SandboxError,
    run_in_sandbox,
    safe_eval_expr,
)


def test_safe_eval_expr_allows_basic_operations():
    names = {"x": 2, "y": 3, "add": lambda a, b: a + b}
    expr = "(x + y) == add(5, 0) and not (y < x)"

    result = safe_eval_expr(expr, names, allowed_callables={"add": names["add"]})

    assert result is True


def test_safe_eval_expr_blocks_disallowed_syntax():
    names = {"x": 1}

    for expr in ["(lambda z: z)(1)", "__import__('os').system('echo hi')"]:
        try:
            safe_eval_expr(expr, names)
        except ValueError:
            continue
        raise AssertionError("unsafe expression was not rejected")


def test_run_in_sandbox_returns_result():
    # Runs in a spawned child process; rlimits never touch this process.
    assert run_in_sandbox(math.factorial, 10, timeout=60) == 3628800


def test_run_in_sandbox_reraises_child_exception():
    with pytest.raises(ValueError):
        run_in_sandbox(int, "not a number", timeout=60)


def test_run_in_sandbox_wall_clock_timeout():
    start = time.monotonic()
    with pytest.raises(SandboxError, match="wall clock"):
        run_in_sandbox(time.sleep, 60, timeout=3)
    assert time.monotonic() - start < 30


def test_run_in_sandbox_does_not_limit_parent_process():
    """The old implementation set rlimits on the caller; the fix must not."""
    import resource

    before = resource.getrlimit(resource.RLIMIT_AS)
    run_in_sandbox(math.factorial, 5, config=SandboxConfig(max_memory_mb=64), timeout=60)
    assert resource.getrlimit(resource.RLIMIT_AS) == before


def test_infrastructure_hosts_are_connectable_with_empty_allowlist():
    """An empty TOOL_NETWORK_ALLOWLIST must not block the model provider.

    Tool handlers that call the LLM open sockets inside the network guard, so
    without this the default configuration blocked the model itself.
    """
    from liminallm.service.sandbox import build_tool_network_policy

    policy = build_tool_network_policy(
        allowlist=[], proxy_url=None, infrastructure_hosts=["api.example.com"]
    )
    assert policy.connection_allowlist() == ["api.example.com"]


def test_infrastructure_hosts_are_not_tool_fetch_targets():
    """Infrastructure is connectable but never fetchable by a tool."""
    from liminallm.service.sandbox import AllowlistedFetcher, SandboxError, build_tool_network_policy

    policy = build_tool_network_policy(
        allowlist=[], proxy_url=None, infrastructure_hosts=["api.example.com"]
    )
    with pytest.raises(SandboxError):
        AllowlistedFetcher(policy).request("GET", "https://api.example.com/v1/models")


def test_guard_allows_resolved_ip_of_an_allowlisted_host(monkeypatch):
    """HTTP clients connect to an address, not a name.

    urllib3/httpx resolve DNS themselves, so the guard is handed an IP literal;
    comparing only against hostnames rejected every real connection.
    """
    from liminallm.service import sandbox as sb

    policy = sb.build_tool_network_policy(
        allowlist=["files.example.com"], proxy_url=None
    )
    monkeypatch.setattr(sb, "_resolve_host_ips", lambda host: frozenset({"203.0.113.7"}))
    with sb.tool_network_guard(policy):
        sb._enforce_network_allowlist("203.0.113.7")  # must not raise
        with pytest.raises(sb.SandboxError):
            sb._enforce_network_allowlist("198.51.100.9")
