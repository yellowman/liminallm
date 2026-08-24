"""The sandbox, driven the way an attacker would drive it.

It guards three attacker-influenced inputs: expressions the model writes into
routing policies, files a user uploads, and hosts a tool is asked to fetch.
The evaluator is the sharpest edge — policies are artifacts the LLM proposes
edits to (SPEC §10), and `ast.Attribute` being unsupported is the one line
between that and `().__class__.__bases__[0].__subclasses__()`.
"""

from __future__ import annotations

import pytest

from liminallm.service.sandbox import (
    _NETWORK_POLICY_STATE,
    DEFAULT_SANDBOX_CONFIG,
    PrivilegedToolError,
    SandboxConfig,
    SandboxError,
    _enforce_network_allowlist,
    _host_matches_allowlist,
    get_tool_sandbox_config,
    run_in_sandbox,
    safe_eval_expr,
    sandbox_open,
    validate_path_access,
)

# ---------------------------------------------------------------------------
# The expression evaluator: what it must refuse
# ---------------------------------------------------------------------------

ESCAPES = [
    # The classic: walk from any literal to every loaded class.
    "().__class__",
    "().__class__.__bases__",
    "''.__class__.__mro__[1].__subclasses__()",
    "[].__class__.__base__.__subclasses__()",
    # Reaching the import machinery by name.
    "__import__('os')",
    "__import__('os').system('id')",
    "eval('1')",
    "exec('x=1')",
    "open('/etc/passwd')",
    "globals()",
    "vars()",
    "getattr(x, '__class__')",
    # Constructs that would smuggle in arbitrary evaluation.
    "(lambda: 1)()",
    "[i for i in range(10)]",
    "{i for i in range(10)}",
    "{i: i for i in range(10)}",
    "(i for i in range(10))",
    "f'{x.__class__}'",
    # Assignment inside an expression.
    "(y := 1)",
]


@pytest.mark.parametrize("expr", ESCAPES, ids=lambda e: e[:34])
def test_an_escape_attempt_is_refused(expr):
    with pytest.raises(ValueError):
        safe_eval_expr(expr, {"x": "s", "y": 1})


def test_attribute_access_is_the_line_that_holds():
    """Named separately because it is the one that matters most: without it,
    every literal in the expression is a path to the interpreter."""
    with pytest.raises(ValueError):
        safe_eval_expr("x.__class__", {"x": "anything"})


def test_an_unlisted_callable_is_refused():
    called = []
    with pytest.raises(ValueError):
        safe_eval_expr("boom()", {}, allowed_callables={"ok": lambda: called.append(1)})
    assert called == []


def test_a_listed_callable_runs():
    assert safe_eval_expr("double(4)", {}, allowed_callables={"double": lambda n: n * 2}) == 8


def test_a_non_callable_in_the_allowlist_is_refused():
    with pytest.raises(ValueError):
        safe_eval_expr("notafunc()", {}, allowed_callables={"notafunc": 42})


def test_kwargs_unpacking_is_refused():
    """**dict would let an expression pass arguments the policy never named."""
    with pytest.raises(ValueError):
        safe_eval_expr(
            "f(**d)", {"d": {"a": 1}}, allowed_callables={"f": lambda **kw: kw}
        )


def test_an_unknown_name_is_refused():
    with pytest.raises(ValueError):
        safe_eval_expr("nope > 1", {"known": 1})


def test_a_syntax_error_is_a_value_error_not_a_crash():
    with pytest.raises(ValueError):
        safe_eval_expr("1 +", {})


# ---------------------------------------------------------------------------
# The expression evaluator: resource bounds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expr",
    [
        "9**9**9",           # pins a core computing a number nobody wants
        "2**100000",         # ditto, directly
        "'a' * 10**9",       # allocates a gigabyte
        "[0] * 10**9",
        "10**9 * 'a'",       # the operands the other way round
        "(10**5000) * (10**5000)",
    ],
    ids=lambda e: e[:24],
)
def test_an_expression_cannot_exhaust_the_box(expr):
    with pytest.raises(ValueError):
        safe_eval_expr(expr, {})


def test_arithmetic_within_bounds_still_works():
    assert safe_eval_expr("2 ** 10", {}) == 1024
    assert safe_eval_expr("'ab' * 3", {}) == "ababab"
    assert safe_eval_expr("10 // 3 + 10 % 3", {}) == 4


def test_deep_nesting_is_refused():
    with pytest.raises(ValueError):
        safe_eval_expr("-" * 200 + "1", {})


# ---------------------------------------------------------------------------
# The expression evaluator: what a routing policy legitimately needs
# ---------------------------------------------------------------------------


def test_the_shapes_a_policy_actually_uses():
    scope = {
        "score": 0.8,
        "tags": ["debug", "python"],
        "meta": {"tier": "paid", "count": 3},
    }
    assert safe_eval_expr("score > 0.5 and 'debug' in tags", scope) is True
    assert safe_eval_expr("meta['tier'] == 'free'", scope) is False
    assert safe_eval_expr("0.5 < score < 1.0", scope) is True
    assert safe_eval_expr("not (score < 0.5)", scope) is True
    assert safe_eval_expr("meta['count'] in (1, 2, 3)", scope) is True
    assert safe_eval_expr("tags[0]", scope) == "debug"


def test_a_bad_subscript_is_a_value_error_not_a_leak():
    """The underlying KeyError text would otherwise reach a policy author."""
    with pytest.raises(ValueError):
        safe_eval_expr("meta['absent']", {"meta": {}})


def test_subscripting_a_non_container_is_refused():
    with pytest.raises(ValueError):
        safe_eval_expr("n[0]", {"n": 42})


# ---------------------------------------------------------------------------
# Egress: which hosts a tool may reach
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "host, allowlist, expected",
    [
        ("api.example.com", ["api.example.com"], True),
        ("API.Example.COM", ["api.example.com"], True),      # case folds
        ("api.example.com", ["*.example.com"], True),        # wildcard
        ("example.com", ["*.example.com"], False),           # bare apex is not a subdomain
        ("evil.com", ["api.example.com"], False),
        # The one that bites: a suffix match without the dot would let
        # notexample.com through a "*.example.com" rule.
        ("notexample.com", ["*.example.com"], False),
        ("10.0.0.5", ["10.0.0.0/8"], True),
        ("11.0.0.5", ["10.0.0.0/8"], False),
        ("", ["*.example.com"], False),
    ],
    ids=lambda v: str(v)[:26],
)
def test_allowlist_matching(host, allowlist, expected):
    assert _host_matches_allowlist(host, allowlist) is expected


@pytest.fixture
def egress_policy():
    """Install a thread-local egress policy, then take it back out."""
    from liminallm.service.sandbox import ToolNetworkPolicy

    def _install(allowlist):
        policy = ToolNetworkPolicy(allowlist=list(allowlist))
        _NETWORK_POLICY_STATE.policy = policy
        return policy

    yield _install
    _NETWORK_POLICY_STATE.policy = None


def test_no_policy_means_no_enforcement(egress_policy):
    _NETWORK_POLICY_STATE.policy = None
    _enforce_network_allowlist("anywhere.example.com")  # must not raise


def test_an_empty_allowlist_blocks_everything(egress_policy):
    egress_policy([])
    with pytest.raises(SandboxError, match="disabled"):
        _enforce_network_allowlist("api.example.com")


def test_an_allowlisted_host_connects(egress_policy):
    egress_policy(["api.example.com"])
    _enforce_network_allowlist("api.example.com")  # must not raise


def test_an_unlisted_host_is_refused(egress_policy):
    egress_policy(["api.example.com"])
    with pytest.raises(SandboxError, match="not allowlisted"):
        _enforce_network_allowlist("evil.example.com")


def test_an_unlisted_address_is_refused(egress_policy):
    """httpx resolves DNS itself and hands the guard an IP, so the guard has
    to handle a literal — and must not wave one through for being numeric."""
    egress_policy(["api.example.com"])
    with pytest.raises(SandboxError, match="not allowlisted"):
        _enforce_network_allowlist("169.254.169.254")  # cloud metadata


def test_localhost_is_not_reachable_by_address(egress_policy):
    egress_policy(["api.example.com"])
    for address in ("127.0.0.1", "0.0.0.0", "::1"):
        with pytest.raises(SandboxError):
            _enforce_network_allowlist(address)


# ---------------------------------------------------------------------------
# Filesystem: where a tool may read and write
# ---------------------------------------------------------------------------


@pytest.fixture
def scratch(tmp_path):
    d = tmp_path / "scratch"
    d.mkdir()
    return SandboxConfig(scratch_dir=d, allowed_paths=[])


def test_the_scratch_directory_is_writable(scratch):
    assert validate_path_access(scratch.scratch_dir / "out.txt", scratch, write=True)


def test_the_scratch_directory_itself_is_allowed(scratch):
    assert validate_path_access(scratch.scratch_dir, scratch)


@pytest.mark.parametrize(
    "attempt",
    ["/etc/passwd", "/etc/shadow", "/proc/self/environ", "~/.ssh/id_rsa"],
)
def test_reading_outside_the_sandbox_is_refused(scratch, attempt):
    with pytest.raises(SandboxError):
        validate_path_access(attempt, scratch)


def test_traversal_out_of_the_scratch_is_refused(scratch):
    """resolve() is what collapses the .. — without it the prefix check passes."""
    with pytest.raises(SandboxError):
        validate_path_access(scratch.scratch_dir / ".." / ".." / "etc" / "passwd", scratch)


def test_an_allowed_path_is_readable_but_not_writable(tmp_path):
    readable = tmp_path / "corpus"
    readable.mkdir()
    config = SandboxConfig(scratch_dir=tmp_path / "scratch", allowed_paths=[readable])
    assert validate_path_access(readable / "doc.txt", config)
    with pytest.raises(SandboxError):
        validate_path_access(readable / "doc.txt", config, write=True)


def test_a_sibling_of_the_scratch_is_not_inside_it(tmp_path):
    """A prefix comparison on strings would accept scratch_evil for scratch."""
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    sibling = tmp_path / "scratch_evil"
    sibling.mkdir()
    config = SandboxConfig(scratch_dir=scratch, allowed_paths=[])
    with pytest.raises(SandboxError):
        validate_path_access(sibling / "payload", config, write=True)


# ---------------------------------------------------------------------------
# Privileged tools
# ---------------------------------------------------------------------------


def test_a_privileged_tool_needs_an_admin():
    """SPEC §18: privileged tools require admin-owned artifacts and are never
    called by default workflows."""
    for role in ("user", None, "moderator"):
        with pytest.raises(PrivilegedToolError):
            get_tool_sandbox_config({"privileged": True}, user_role=role)


def test_an_admin_gets_the_privileged_config():
    config = get_tool_sandbox_config({"privileged": True}, user_role="admin")
    assert config.privileged is True


def test_an_ordinary_tool_is_never_privileged_whoever_calls_it():
    config = get_tool_sandbox_config({"privileged": False}, user_role="admin")
    assert config.privileged is False


def test_no_spec_means_the_default_config():
    assert get_tool_sandbox_config(None) is DEFAULT_SANDBOX_CONFIG


@pytest.mark.parametrize(
    "asked, field, ceiling",
    [
        ({"max_memory_mb": 999999}, "max_memory_mb", 1024),
        ({"max_cpu_seconds": 999999}, "max_cpu_seconds", 120),
        ({"max_file_size_mb": 999999}, "max_file_size_mb", 500),
    ],
)
def test_a_tool_cannot_raise_its_own_ceiling(asked, field, ceiling):
    """resource_limits comes off the tool.spec artifact, which the LLM can
    propose edits to — so the ask is capped, not trusted."""
    config = get_tool_sandbox_config({"resource_limits": asked})
    assert getattr(config, field) == ceiling


def test_a_tool_may_ask_for_less():
    config = get_tool_sandbox_config({"resource_limits": {"max_memory_mb": 64}})
    assert config.max_memory_mb == 64


# The privileged gate's caller half is exercised above, against
# `get_tool_sandbox_config`. Its provenance half — SPEC §18's *admin-owned
# artifact* — needs the persisted artifact row, so it lives in
# tests/test_tool_authority.py where a store is available.


# ---------------------------------------------------------------------------
# sandbox_open
# ---------------------------------------------------------------------------


def test_sandbox_open_writes_and_reads_inside_the_scratch(scratch):
    target = scratch.scratch_dir / "note.txt"
    with sandbox_open(target, "w", scratch) as fh:
        fh.write("hello")
    with sandbox_open(target, "r", scratch) as fh:
        assert fh.read() == "hello"


def test_sandbox_open_refuses_a_path_outside(scratch):
    with pytest.raises(SandboxError):
        with sandbox_open("/etc/passwd", "r", scratch):
            pass


def test_sandbox_open_refuses_a_write_to_a_read_only_path(tmp_path):
    readable = tmp_path / "corpus"
    readable.mkdir()
    (readable / "doc.txt").write_text("x")
    config = SandboxConfig(scratch_dir=tmp_path / "scratch", allowed_paths=[readable])
    with pytest.raises(SandboxError):
        with sandbox_open(readable / "doc.txt", "w", config):
            pass


# ---------------------------------------------------------------------------
# run_in_sandbox: the limits have to bite in the child, not here
# ---------------------------------------------------------------------------


def _return_sum(a, b):
    return a + b


def _raise_value_error():
    raise ValueError("from inside the sandbox")


def _eat_memory():
    """Allocate far past any sane cap. RLIMIT_AS should stop this."""
    blocks = []
    for _ in range(400):
        blocks.append(bytearray(8 * 1024 * 1024))  # 8MB at a time, up to 3.2GB
    return len(blocks)


def _spin_forever():
    while True:
        pass


@pytest.mark.slow
def test_a_sandboxed_call_returns_its_result(tmp_path):
    config = SandboxConfig(scratch_dir=tmp_path / "s", max_cpu_seconds=10)
    assert run_in_sandbox(_return_sum, 2, 3, config=config) == 5


@pytest.mark.slow
def test_a_sandboxed_exception_reaches_the_caller(tmp_path):
    """Shipped back as an object so a caller can catch its own error type."""
    config = SandboxConfig(scratch_dir=tmp_path / "s", max_cpu_seconds=10)
    with pytest.raises(ValueError, match="from inside the sandbox"):
        run_in_sandbox(_raise_value_error, config=config)


@pytest.mark.slow
def test_the_memory_cap_actually_stops_a_greedy_call(tmp_path):
    """The point of the child process: an rlimit here would cripple the server.

    Whether the child dies or the allocation raises MemoryError, what must not
    happen is 3.2GB being handed out.
    """
    config = SandboxConfig(
        scratch_dir=tmp_path / "s", max_memory_mb=128, max_cpu_seconds=10
    )
    with pytest.raises((SandboxError, MemoryError)):
        run_in_sandbox(_eat_memory, config=config)


@pytest.mark.slow
def test_a_wall_clock_timeout_kills_a_spinning_call(tmp_path):
    """Backstops the CPU rlimit — a call blocked on I/O burns no CPU at all."""
    config = SandboxConfig(scratch_dir=tmp_path / "s", max_cpu_seconds=60)
    with pytest.raises(SandboxError):
        run_in_sandbox(_spin_forever, config=config, timeout=3)


@pytest.mark.slow
def test_the_parent_process_keeps_its_own_limits(tmp_path):
    """If the rlimits leaked out of the child, the suite after this point
    would be running under a 128MB cap."""
    import resource

    before = resource.getrlimit(resource.RLIMIT_AS)
    config = SandboxConfig(
        scratch_dir=tmp_path / "s", max_memory_mb=128, max_cpu_seconds=10
    )
    run_in_sandbox(_return_sum, 1, 1, config=config)
    assert resource.getrlimit(resource.RLIMIT_AS) == before
