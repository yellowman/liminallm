"""Tool capability adds to grounding; it never replaces it.

A turn takes the tool-agent workflow whenever `_turn_needs_tools()` says the
deployment has something to offer — an attachment, web tools, a published MCP
server. That decision is about capability. It said nothing about grounding,
and it silently took grounding away: `llm.generic` validates `context_id`,
retrieves for it and injects the chunks, while the agent planner never
received `context_id` at all. A knowledge context the user explicitly selected
therefore entered neither the model's first prompt nor its tool list, and the
model answered that it had been given nothing.

Not a web-default bug, though the shipped web settings are what make it
universal: any of the three triggers loses the same context. The triggers are
parameterized here for that reason.

The model is the only thing faked. The store, the retriever, the planner and
the agent loop are real, and the fake makes no tool call on purpose — a
context the user selected must not depend on the model guessing it should go
looking for it.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

from liminallm.config import Settings
from liminallm.service.runtime import get_runtime
from tests.mcpfixture import MCPFixture, allow_local

# Invented, so a hit can only have come out of the corpus.
FACT = "4417 kilohertz"
DOCUMENT = (
    "Kestrel-9 Relay Module field notes. The Kestrel-9 relay module operates "
    f"at a resonance frequency of {FACT}. It is maintained by the Thornbury "
    "substation crew and its calibration drift is logged every 19 days."
)
QUESTION = "What resonance frequency does the Kestrel-9 relay module operate at?"

#: What config.py ships. Stated rather than read so that changing the default
#: fails this file loudly instead of quietly changing what it proves.
SHIPPED_WEB = {
    "enabled": True,
    "provider": "none",
    "api_key": "",
    "engine_id": None,
    "timeout": 15.0,
    "max_bytes": 2097152,
    "allow_private": False,
    "proxy": None,
}


class RecordingModel:
    """A tool-capable model that calls no tool and answers from its prompt.

    It reports the fact only when the fact was actually put in front of it, so
    "the answer can use it" is measured rather than assumed.
    """

    def __init__(self) -> None:
        self.calls: list[list[dict]] = []
        self.offered: list[list[dict]] = []

    def __call__(self, messages, tools, adapters=None, *, user_id=None):
        self.calls.append([dict(m) for m in messages])
        self.offered.append(list(tools or []))
        prompt = "\n".join(str(m.get("content") or "") for m in messages)
        content = FACT if FACT in prompt else "I have not been given any notes."
        return {
            "content": content,
            "tool_calls": [],
            "assistant_message": {"role": "assistant", "content": content},
            "usage": {"total_tokens": 1},
        }

    @property
    def first_prompt(self) -> str:
        """Everything the model could read on its first call."""
        assert self.calls, "the model was never called"
        return "\n".join(str(m.get("content") or "") for m in self.calls[0])

    @property
    def first_tool_names(self) -> list[str]:
        assert self.offered, "the model was never called"
        return [
            t["function"]["name"]
            for t in self.offered[0]
            if isinstance(t, dict) and "function" in t
        ]


@pytest.fixture
def model(monkeypatch):
    """The real LLMService with one method replaced, and tools declared on."""
    engine = get_runtime().workflow
    recorder = RecordingModel()
    monkeypatch.setattr(
        type(engine.llm.backend), "supports_tools", property(lambda _self: True)
    )
    monkeypatch.setattr(engine.llm, "generate_with_tools", recorder)
    return recorder


@pytest.fixture
def engine(monkeypatch):
    engine = get_runtime().workflow
    monkeypatch.setattr(engine, "tool_network_policy", allow_local())
    return engine


def grounded_context(store) -> tuple[str, str]:
    """A real user and a real context holding the fact. Returns (user, ctx)."""
    user = store.create_user(email=f"ground_{uuid.uuid4().hex[:8]}@example.com")
    ctx = store.upsert_context(
        name=f"kestrel-{uuid.uuid4().hex[:6]}",
        description="an ordinary knowledge context",
        owner_user_id=user.id,
    )
    written = get_runtime().rag.ingest_text(ctx.id, DOCUMENT)
    assert written > 0, "the fixture failed to index the document"
    return user.id, ctx.id


def web_on(monkeypatch, engine):
    """The shipped web configuration: enabled, with no search provider."""
    monkeypatch.setattr(engine, "_web_settings", lambda: dict(SHIPPED_WEB))


def web_off(monkeypatch, engine):
    monkeypatch.setattr(
        engine, "_web_settings", lambda: {**SHIPPED_WEB, "enabled": False}
    )


class TestTheShippedDefaultIsWhatThisFileAssumes:
    def test_config_still_ships_web_on_with_no_provider(self):
        """The premise, checked against config.py rather than trusted.

        Every case below is written for the deployment an operator gets
        without touching anything. If that stops being web-enabled with no
        provider, these cases still pass while no longer describing anybody's
        installation, so the premise is asserted rather than assumed.
        """
        fields = Settings.model_fields
        assert fields["web_tools_enabled"].default is True
        assert fields["web_search_provider"].default == "none"


class TestASelectedContextSurvivesToolRouting:
    """The invariant: capability is additive to grounding."""

    @pytest.mark.parametrize("trigger", ["web", "mcp"])
    def test_the_selected_context_reaches_the_model(
        self, engine, model, store, monkeypatch, trigger
    ):
        """A context the user named must ground the turn on the agent path.

        Parameterized over why the turn is on that path at all. If only the
        web case were covered, a fix that special-cased the web default would
        pass while an operator with a published MCP server still lost every
        selected context — the defect is tool routing's, not web's.
        """
        user_id, ctx_id = grounded_context(store)

        if trigger == "web":
            web_on(monkeypatch, engine)
            result = asyncio.run(
                engine.run(None, None, QUESTION, ctx_id, user_id)
            )
        else:
            web_off(monkeypatch, engine)
            with MCPFixture(f"g{uuid.uuid4().hex[:6]}", {"ping": "pong"}) as fixture:
                admin = store.create_user(
                    email=f"adm_{uuid.uuid4().hex[:8]}@example.com"
                )
                store.update_user_role(admin.id, role="admin")
                store.create_artifact(
                    "mcp",
                    fixture.name,
                    fixture.as_artifact_schema(taint_class="local_read"),
                    owner_user_id=admin.id,
                    visibility="global",
                )
                result = asyncio.run(
                    engine.run(None, None, QUESTION, ctx_id, user_id)
                )

        assert engine._turn_needs_tools(None, user_id), (
            f"the {trigger} trigger did not put this turn on the agent path, "
            "so this case is not testing what it claims to"
        )
        assert FACT in model.first_prompt, (
            "the selected context never reached the model's first prompt: "
            f"{model.first_prompt[:400]}"
        )
        assert any(FACT in s for s in result.get("context_snippets") or []), (
            f"the turn reported no snippet carrying the fact: "
            f"{result.get('context_snippets')}"
        )
        assert FACT in (result.get("content") or ""), (
            f"the answer could not use the fact: {result.get('content')!r}"
        )

    def test_a_turn_with_no_context_is_not_grounded(
        self, engine, model, store, monkeypatch
    ):
        """The control: grounding comes from the selection, not from the air.

        Without this the case above would pass against an implementation that
        pasted every context this user owns into every prompt.
        """
        user_id, _ctx_id = grounded_context(store)
        web_on(monkeypatch, engine)

        result = asyncio.run(engine.run(None, None, QUESTION, None, user_id))

        assert model.calls, (
            "this control never reached the agent path, so it proves nothing "
            "about agent grounding: with no context and no attachment the "
            "only offered tool is web_fetch, and an empty tool list sends the "
            "turn to plain chat instead"
        )
        assert FACT not in model.first_prompt, (
            "an unselected context reached the prompt anyway"
        )
        assert not any(FACT in s for s in result.get("context_snippets") or [])
        assert FACT not in (result.get("content") or "")

    def test_the_model_is_offered_the_search_tool_for_that_context(
        self, engine, model, store, monkeypatch
    ):
        """Initial chunks ground the answer; the tool lets it dig further.

        `_run_file_search` already resolves an explicit `context_id`, so the
        tool was usable the whole time and simply was not offered unless the
        conversation happened to hold a searchable attachment. Offering it is
        the additive half; it must not become the only half, which is what
        the no-tool-call model above pins down.
        """
        user_id, ctx_id = grounded_context(store)
        web_on(monkeypatch, engine)

        asyncio.run(engine.run(None, None, QUESTION, ctx_id, user_id))

        assert "file_search" in model.first_tool_names, model.first_tool_names


class TestCapabilityIsNotTradedForGrounding:
    """The wrong fix, refused by name.

    Narrowing the selector to `web_tools_enabled and provider != "none"` also
    restores grounding, and costs a capability to do it: `web_fetch` needs no
    search provider and is offered whenever web tools are on. This case fails
    under that patch and passes under the fix, which is the whole reason it is
    here.
    """

    def test_no_search_provider_still_offers_web_fetch(
        self, engine, monkeypatch, store
    ):
        _user_id, _ctx_id = grounded_context(store)
        web_on(monkeypatch, engine)

        _messages, tools, _preamble, _mcp = engine._build_agent_context(
            QUESTION, [], [], None, None
        )

        offered = [
            t["function"]["name"] for t in tools if isinstance(t, dict) and "function" in t
        ]
        assert "web_fetch" in offered, (
            "web_fetch is reachable with no search provider and must stay "
            f"offered: {offered}"
        )
        assert "web_search" not in offered, (
            f"web_search needs a provider and must not be offered: {offered}"
        )
