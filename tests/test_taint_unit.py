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


def test_local_reading_survives_a_tainted_turn():
    """The turn must still be able to look at what is already on the box.

    Withdrawing these would leave the model unable to explain what the page
    attempted, which is the one useful thing left for it to do.
    """
    session = {"injection_findings": ["persona-hijack"]}
    for tool in ("file_search", "history_search", "note_search"):
        assert taint.is_withdrawn(tool, session) is False


def test_everything_that_reaches_off_the_box_is_withdrawn():
    """The line is egress, not action.

    run_python's own schema promises no network, so it was never how a secret
    would leave. web_fetch takes a model-supplied URL and web_search a
    model-supplied query; either carries data to a destination the injected
    page chose.
    """
    session = {"injection_findings": ["persona-hijack"]}
    for tool in ("web_fetch", "web_search", "run_python"):
        assert taint.is_withdrawn(tool, session) is True


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
