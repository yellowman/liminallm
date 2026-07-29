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
