"""A web tool that did not run says so, instead of answering with prose.

`run_web_search` and `run_web_fetch` returned `(text, findings)` for three
different outcomes - the page, a fetch that failed, and a deployment with web
access withdrawn - so nothing downstream could tell them apart. The worker
body set no `status`, and the direct endpoint's `result.get("status", "ok")`
read the absence as success.

`status == "error"` is not cosmetic here. It is what records a failed node,
what selects a node's `on_error` branch, and what keeps a failed child's
retrieval out of the grounding merge. A web node whose fetch failed could
therefore never take its error branch: the workflow carried
"Could not read that page: ..." forward as that node's answer.

The model still reads the sentence. It is the one caller that should: the
agent loop needs something to reason about, not an exception.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

from liminallm.service import agent_tools, web
from liminallm.service.runtime import get_runtime

WITHDRAWN = "Web access is disabled on this deployment."


class _Settings:
    """The real field set `web_settings` reads, and nothing else.

    Built from `Settings` itself rather than hand-listed, so a field added to
    the declaration cannot be missing here and pass.
    """

    def __init__(self, **overrides):
        from liminallm.config import get_settings

        real = get_settings()
        for name in (
            "web_tools_enabled", "web_search_provider", "web_search_api_key",
            "web_search_engine_id", "web_fetch_timeout", "web_fetch_max_bytes",
            "web_fetch_allow_private", "tool_network_proxy_url",
        ):
            setattr(self, name, getattr(real, name))
        for name, value in overrides.items():
            setattr(self, name, value)


class _Logger:
    def info(self, *_a, **_k):
        pass

    def warning(self, *_a, **_k):
        pass


class TestTheRunnerSaysWhetherItRan:
    def test_a_withdrawn_search_is_not_a_result(self):
        ok, text, _findings = agent_tools.run_web_search(
            "anything", 5,
            settings=_Settings(web_tools_enabled=False), logger=_Logger(),
        )
        assert ok is False
        assert text == WITHDRAWN

    def test_a_withdrawn_fetch_is_not_a_result(self):
        ok, text, _findings = agent_tools.run_web_fetch(
            "https://example.com",
            settings=_Settings(web_tools_enabled=False), logger=_Logger(),
        )
        assert ok is False
        assert text == WITHDRAWN

    def test_a_failed_fetch_is_not_a_result(self, monkeypatch):
        """The half that breaks `on_error`: web is on, the page is not."""
        def _boom(*_a, **_k):
            raise web.WebFetchError("connection refused")

        monkeypatch.setattr(web, "fetch_url", _boom)

        ok, text, _findings = agent_tools.run_web_fetch(
            "https://example.com/gone",
            settings=_Settings(web_tools_enabled=True), logger=_Logger(),
        )

        assert ok is False
        assert "Could not read that page" in text

    def test_a_failed_search_is_not_a_result(self, monkeypatch):
        def _boom(*_a, **_k):
            raise web.WebFetchError("provider said no")

        monkeypatch.setattr(web, "search_web", _boom)

        ok, text, _findings = agent_tools.run_web_search(
            "anything", 5,
            settings=_Settings(web_tools_enabled=True, web_search_provider="brave"),
            logger=_Logger(),
        )

        assert ok is False
        assert "Search failed" in text

    def test_a_page_that_was_read_is_a_result(self, monkeypatch):
        """The door must not be a wall: a real page still reports success."""
        monkeypatch.setattr(web, "fetch_url", lambda *_a, **_k: {
            "url": "https://example.com/", "title": "Example",
            "text": "the visible words", "findings": [],
        })

        ok, text, _findings = agent_tools.run_web_fetch(
            "https://example.com/",
            settings=_Settings(web_tools_enabled=True), logger=_Logger(),
        )

        assert ok is True
        assert "the visible words" in text


class TestTheEndpointStopsClaimingSuccess:
    """Measured through `invoke_tool`, which both the direct endpoint and a
    workflow's tool node reach."""

    @pytest.fixture
    def seeded(self):
        runtime = get_runtime()
        return {
            a.name: a
            for a in runtime.store.list_artifacts(
                type_filter="tool", kind_filter="tool.spec", page_size=200
            )
        }

    def _invoke(self, runtime, artifact, inputs):
        user = runtime.store.create_user(email=f"wt_{uuid.uuid4().hex[:8]}@t.local")
        return asyncio.run(
            runtime.workflow.invoke_tool(
                runtime.workflow._describe_tool(artifact), inputs,
                user_id=user.id, tenant_id=user.tenant_id,
            )
        )

    @pytest.mark.parametrize("tool, inputs", [
        ("web.search_v1", {"query": "anything"}),
        ("web.fetch_v1", {"url": "https://example.com"}),
    ])
    def test_a_withdrawn_tool_reports_an_error(
        self, client, seeded, monkeypatch, tool, inputs
    ):
        runtime = get_runtime()
        monkeypatch.setattr(runtime.settings, "web_tools_enabled", False)

        result = self._invoke(runtime, seeded[tool], inputs)

        assert result.get("status") == "error", result
        assert WITHDRAWN in str(result.get("content"))

    def test_a_search_that_ran_still_reports_success(
        self, client, seeded, monkeypatch
    ):
        """The door must not be a wall.

        Without this, marking every web result an error passes every other
        test here - which is exactly what a mutant that drops the broker's
        `ran` does. The result rows carry the keys `format_search_results`
        reads, so this exercises the real renderer rather than a stand-in.
        """
        runtime = get_runtime()
        monkeypatch.setattr(runtime.settings, "web_tools_enabled", True)
        monkeypatch.setattr(runtime.settings, "web_search_provider", "brave")
        monkeypatch.setattr(web, "search_web", lambda *_a, **_k: [
            {"title": "Turbine blades", "url": "https://example.com/blades",
             "snippet": "inspection intervals", "findings": []},
        ])

        result = self._invoke(
            runtime, seeded["web.search_v1"], {"query": "turbine blades"}
        )

        assert result.get("status") == "ok", result
        assert "Turbine blades" in str(result.get("content"))

    def test_a_failed_web_node_takes_its_error_branch(
        self, client, seeded, monkeypatch
    ):
        """The consequence the status exists for.

        `on_error` is selected by `result["status"] == "error"`
        (workflow.py's retry gate). A web node that never reported one could
        not take its branch, so a workflow carried "Could not read that
        page: ..." forward as that node's answer and the recovery node it
        was written with never ran.
        """
        runtime = get_runtime()
        user = runtime.store.create_user(
            email=f"br_{uuid.uuid4().hex[:8]}@t.local"
        )
        workflow = runtime.store.create_artifact(
            type_="workflow",
            name=f"wf-{uuid.uuid4().hex[:6]}",
            schema={"kind": "workflow.chat", "nodes": [
                {"id": "look", "type": "tool_call", "tool": "web.fetch_v1",
                 "inputs": {"url": "https://example.com/gone"},
                 "next": "fin", "on_error": "rec"},
                {"id": "rec", "type": "tool_call", "tool": "llm.generic",
                 "next": "fin"},
                {"id": "fin", "type": "end"},
            ]},
            owner_user_id=user.id,
        )

        def _boom(*_a, **_k):
            raise web.WebFetchError("connection refused")

        monkeypatch.setattr(web, "fetch_url", _boom)
        result = asyncio.run(
            runtime.workflow.run(workflow.id, None, "read that page", None, user.id)
        )

        trace = result.get("workflow_trace") or []
        visited = [step.get("node_id") or step.get("node") for step in trace]
        assert "rec" in visited, (
            f"the error branch never ran; trace={visited}"
        )

    def test_the_error_status_matches_the_convention(
        self, client, seeded, monkeypatch
    ):
        """`web.fetch_v1` with no url already answered `error`, and a
        withdrawn deployment answered nothing. One tool, two refusals, two
        different shapes - which is how the absence stayed invisible."""
        runtime = get_runtime()
        monkeypatch.setattr(runtime.settings, "web_tools_enabled", False)

        urlless = self._invoke(runtime, seeded["web.fetch_v1"], {})
        withdrawn = self._invoke(
            runtime, seeded["web.fetch_v1"], {"url": "https://example.com"}
        )

        assert urlless.get("status") == withdrawn.get("status") == "error"
