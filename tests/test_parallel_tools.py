"""One round's tool calls: parallel when read-only, ordered always.

The properties under test are exactly the ones parallelism could break:
result and snippet order, the same-round taint withdrawal (a web_fetch that
records an injection finding must withdraw run_python later in the SAME
round), and the thread-local egress guard reaching every worker — the socket
allowlist permits when no policy is set on the connecting thread, so a
worker without the guard would be a hole, not a slowdown.
"""

import time

from liminallm.service import sandbox
from liminallm.service.runtime import get_runtime


def _parsed(*specs):
    """[(call, name, args), ...] the way the round loop builds it."""
    return [
        ({"id": f"c{i}", "name": name}, name, args)
        for i, (name, args) in enumerate(specs)
    ]


def _kwargs(session=None, snippets=None):
    return {
        "conversation_id": None,
        "context_id": None,
        "user_id": None,
        "tenant_id": None,
        "session": {} if session is None else session,
        "snippets": [] if snippets is None else snippets,
        "fallback_query": "q",
    }


def _fake_exec(record=None):
    def fake(
        name,
        args,
        *,
        conversation_id,
        context_id,
        user_id,
        tenant_id,
        session,
        snippets,
        fallback_query,
        invocation=None,
        operation_seq=0,
        step="",
    ):
        time.sleep(args.get("sleep", 0.15))
        if record is not None:
            record.append(
                getattr(sandbox._NETWORK_POLICY_STATE, "policy", None)
            )
        snippets.append(f"s:{args['i']}")
        return f"r:{args['i']}"

    return fake


class TestParallelRounds:
    def test_read_only_round_runs_concurrently_in_order(self, client, monkeypatch):
        engine = get_runtime().workflow
        monkeypatch.setattr(engine, "_execute_agent_tool", _fake_exec())
        parsed = _parsed(
            ("note_search", {"i": 0}),
            ("file_search", {"i": 1}),
            ("history_search", {"i": 2}),
        )
        snippets = []
        start = time.monotonic()
        results = engine._run_round_tools(parsed, **_kwargs(snippets=snippets))
        elapsed = time.monotonic() - start

        assert results == ["r:0", "r:1", "r:2"]
        # Three 0.15s tools sequentially would be >= 0.45s.
        assert elapsed < 0.40, f"round did not parallelize: {elapsed:.2f}s"
        assert snippets == ["s:0", "s:1", "s:2"]

    def test_snippet_order_survives_adversarial_timing(self, client, monkeypatch):
        engine = get_runtime().workflow
        monkeypatch.setattr(engine, "_execute_agent_tool", _fake_exec())
        # The first call finishes LAST; order must come from call position.
        parsed = _parsed(
            ("note_search", {"i": 0, "sleep": 0.30}),
            ("note_search", {"i": 1, "sleep": 0.05}),
            ("note_search", {"i": 2, "sleep": 0.05}),
        )
        snippets = []
        results = engine._run_round_tools(parsed, **_kwargs(snippets=snippets))

        assert results == ["r:0", "r:1", "r:2"]
        assert snippets == ["s:0", "s:1", "s:2"]

    def test_taint_class_round_stays_sequential(self, client, monkeypatch):
        engine = get_runtime().workflow
        monkeypatch.setattr(engine, "_execute_agent_tool", _fake_exec())
        parsed = _parsed(
            ("note_search", {"i": 0}),
            ("web_fetch", {"i": 1}),
        )
        start = time.monotonic()
        results = engine._run_round_tools(parsed, **_kwargs())
        elapsed = time.monotonic() - start

        assert results == ["r:0", "r:1"]
        assert elapsed >= 0.28, f"taint-class round overlapped: {elapsed:.2f}s"

    def test_same_round_fetch_taint_withdraws_run_python(self, client, monkeypatch):
        """The security ordering, on the real dispatch: web_fetch records an
        injection finding, and the run_python the model requested in the same
        round is refused, never executed."""
        engine = get_runtime().workflow
        python_ran = []
        monkeypatch.setattr(
            engine,
            "_run_web_fetch",
            lambda url: ("page text", [{"type": "override_attempt"}]),
        )
        monkeypatch.setattr(
            engine,
            "_run_python_capability",
            lambda code, **kw: python_ran.append(code) or "ran",
        )
        parsed = _parsed(
            ("web_fetch", {"url": "http://example.test/page"}),
            ("run_python", {"code": "print(1)"}),
        )
        session = {}
        results = engine._run_round_tools(parsed, **_kwargs(session=session))

        assert results[0] == "page text"
        assert results[1].startswith("REFUSED")
        assert python_ran == [], "run_python executed despite same-round taint"
        assert session.get("injection_findings") == ["override_attempt"]

    def test_egress_guard_reaches_every_parallel_worker(self, client, monkeypatch):
        """Worker threads do not inherit the handler thread's guard; the
        round runner must re-apply it, else the allowlist silently permits."""
        engine = get_runtime().workflow
        seen = []
        monkeypatch.setattr(engine, "_execute_agent_tool", _fake_exec(record=seen))
        parsed = _parsed(
            ("note_search", {"i": 0}),
            ("note_search", {"i": 1}),
            ("note_search", {"i": 2}),
        )
        engine._run_round_tools(parsed, **_kwargs())

        assert len(seen) == 3
        assert all(policy is engine.tool_network_policy for policy in seen)
