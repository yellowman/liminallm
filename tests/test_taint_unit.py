"""The capability-withdrawal decision, tested without an engine around it.

tests/test_injection_taint.py covers the same control end-to-end through the
workflow. These cover the decision itself, which is where the safety property
actually lives.
"""

from liminallm.service import taint


def test_clean_session_withholds_nothing():
    assert taint.is_withdrawn("run_python", {}) is False
    assert taint.is_tainted({}) is False


def test_findings_withdraw_code_execution():
    session = {}
    taint.record_findings(session, [{"type": "persona-hijack", "match": "x"}])
    assert taint.is_withdrawn("run_python", session) is True


def test_reading_tools_survive_a_tainted_turn():
    """Only the ability to act is withdrawn, never the ability to look."""
    session = {"injection_findings": ["persona-hijack"]}
    for tool in ("file_search", "web_search", "web_fetch", "history_search",
                 "note_search"):
        assert taint.is_withdrawn(tool, session) is False


def test_refusal_names_what_was_seen_and_discourages_retry():
    session = {"injection_findings": ["persona-hijack", "tool-abuse"]}
    message = taint.refusal(session)
    assert message.startswith("REFUSED")
    assert "persona-hijack" in message and "tool-abuse" in message
    assert "not a failure you can retry" in message


def test_refusal_counts_blocks_for_the_turn():
    session = {"injection_findings": ["tool-abuse"]}
    for expected in (1, 2, 3):
        taint.refusal(session)
        assert session["taint_blocked"] == expected


def test_refusal_caps_the_kinds_it_lists():
    session = {"injection_findings": [f"kind-{i}" for i in range(12)]}
    listed = taint.refusal(session).split("(")[1].split(")")[0]
    assert len(listed.split(", ")) == taint.MAX_KINDS_REPORTED


def test_record_findings_ignores_malformed_entries():
    session = {}
    added = taint.record_findings(session, [{"no_type": 1}, "not a dict", {}])
    assert added == 0
    assert taint.is_tainted(session) is False


def test_taint_accumulates_across_fetches():
    session = {}
    taint.record_findings(session, [{"type": "a"}])
    taint.record_findings(session, [{"type": "b"}])
    assert taint.findings(session) == ["a", "b"]
