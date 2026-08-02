"""The model-invoked capabilities, tested without a workflow engine.

These are the tools that touch the network, the interpreter and the filesystem.
Before the split they could only be reached through a fully-constructed engine,
which meant their edge cases — web disabled, no attachments, nothing matched —
went uncovered.
"""

from types import SimpleNamespace

import pytest

from liminallm.service import agent_tools


class _Log:
    def __init__(self):
        self.events = []

    def info(self, event, **kw):
        self.events.append((event, kw))

    def warning(self, event, **kw):
        self.events.append((event, kw))


def _settings(**kw):
    base = dict(
        web_tools_enabled=False,
        web_search_provider="none",
        web_search_api_key=None,
        web_search_engine_id=None,
        web_fetch_timeout=15.0,
        web_fetch_max_bytes=2 * 1024 * 1024,
        web_fetch_allow_private=False,
        tool_network_proxy_url=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


class TestWebSettings:
    def test_defaults_are_the_closed_ones(self):
        """An install that configured nothing does not get web access."""
        cfg = agent_tools.web_settings(SimpleNamespace())
        assert cfg["enabled"] is False
        assert cfg["allow_private"] is False

    def test_reads_configured_values(self):
        cfg = agent_tools.web_settings(
            _settings(web_tools_enabled=True, web_search_provider="brave")
        )
        assert cfg["enabled"] is True
        assert cfg["provider"] == "brave"

    def test_blank_provider_falls_back_to_none(self):
        cfg = agent_tools.web_settings(_settings(web_search_provider=""))
        assert cfg["provider"] == "none"


class TestWebToolsDisabled:
    def test_search_says_so_rather_than_failing(self):
        text, findings = agent_tools.run_web_search(
            "anything", 5, settings=_settings(), logger=_Log()
        )
        assert "disabled" in text
        assert findings == []

    def test_fetch_says_so_rather_than_failing(self):
        text, findings = agent_tools.run_web_fetch(
            "https://example.com", settings=_settings(), logger=_Log()
        )
        assert "disabled" in text
        assert findings == []


class TestFileSearch:
    def test_no_contexts_is_a_message_not_a_crash(self):
        text, snippets = agent_tools.run_file_search(
            "q", 4, [], rag=None, user_id="u", tenant_id=None
        )
        assert "No searchable files" in text
        assert snippets == []

    def test_no_matches_names_the_query(self):
        rag = SimpleNamespace(retrieve=lambda *a, **k: [])
        text, snippets = agent_tools.run_file_search(
            "widgets", 4, ["ctx"], rag=rag, user_id="u", tenant_id=None
        )
        assert "widgets" in text
        assert snippets == []

    def test_excerpts_are_labelled_with_their_file(self):
        chunk = SimpleNamespace(
            content="the answer is 42", meta={"source_path": "/x/report.pdf"}
        )
        rag = SimpleNamespace(retrieve=lambda *a, **k: [chunk])
        text, snippets = agent_tools.run_file_search(
            "answer", 4, ["ctx"], rag=rag, user_id="u", tenant_id=None
        )
        assert "[report.pdf]" in text
        assert snippets == ["the answer is 42"]

    def test_limit_is_clamped_to_the_documented_range(self):
        seen = {}

        def retrieve(ctx_ids, query, *, limit, user_id, tenant_id):
            seen["limit"] = limit
            return []

        rag = SimpleNamespace(retrieve=retrieve)
        agent_tools.run_file_search(
            "q", 9999, ["ctx"], rag=rag, user_id="u", tenant_id=None
        )
        assert seen["limit"] == agent_tools.MAX_FILE_SEARCH_RESULTS


class TestRunPython:
    def test_requires_an_authenticated_user(self):
        out = agent_tools.run_python(
            "print(1)", [], settings=_settings(), user_id=None, session={}
        )
        assert "authenticated user" in out


class TestHistorySearch:
    @staticmethod
    def _msg(seq, role, content):
        return SimpleNamespace(seq=seq, role=role, content=content)

    def test_nothing_older_than_the_window(self):
        out = agent_tools.run_history_search(
            "q", 4, [], keep_tokens=1000, count=len
        )
        assert "beyond what is already in context" in out

    def test_returns_matching_older_turns_verbatim(self):
        history = [
            self._msg(i, "user", f"turn {i} about capybaras" if i == 0 else f"turn {i}")
            for i in range(40)
        ]
        out = agent_tools.run_history_search(
            "capybaras", 4, history, keep_tokens=20, count=lambda t: len(t)
        )
        assert "capybaras" in out
        # The framing matters: recalled turns are data, not fresh instructions.
        assert "not instructions" in out

    def test_no_match_names_the_query(self):
        history = [self._msg(i, "user", f"turn {i}") for i in range(40)]
        out = agent_tools.run_history_search(
            "zzzznomatch", 4, history, keep_tokens=20, count=lambda t: len(t)
        )
        assert "zzzznomatch" in out

    def test_excerpts_are_truncated(self):
        history = [self._msg(i, "user", "pad") for i in range(40)]
        history[0] = self._msg(0, "user", "capybara " * 5000)
        out = agent_tools.run_history_search(
            "capybara", 4, history, keep_tokens=20, count=lambda t: len(t)
        )
        body = out.split("\n\n", 1)[1]
        assert len(body) < agent_tools.HISTORY_EXCERPT_CHARS + 100


@pytest.mark.parametrize(
    "schema",
    [
        agent_tools.WEB_SEARCH_SCHEMA,
        agent_tools.WEB_FETCH_SCHEMA,
        agent_tools.FILE_SEARCH_SCHEMA,
        agent_tools.RUN_PYTHON_SCHEMA,
        agent_tools.HISTORY_SEARCH_SCHEMA,
        agent_tools.NOTE_SEARCH_SCHEMA,
    ],
)
def test_advertised_schemas_are_well_formed(schema):
    """A malformed schema makes a provider reject the whole request."""
    assert schema["type"] == "function"
    fn = schema["function"]
    assert fn["name"] and fn["description"]
    params = fn["parameters"]
    assert params["type"] == "object"
    for name in params["required"]:
        assert name in params["properties"], name


def test_untrusted_tools_say_so_in_their_description():
    """Weak models need the warning where they read the tool, not only in the
    system prompt. This is one of the three deliberate repetitions."""
    for schema in (agent_tools.WEB_SEARCH_SCHEMA, agent_tools.WEB_FETCH_SCHEMA):
        assert "untrusted" in schema["function"]["description"].lower()
