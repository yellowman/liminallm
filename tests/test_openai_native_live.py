"""Live witness: native compaction on OpenAI's own endpoint.

Opt-in and outside ordinary CI: it needs `OPENAI_PROBE_API_KEY`, spends
money, and asks the provider to compact at a threshold far below where a
deployment would. What it establishes is the contract the adapter is written
against, measured on the wire: `store=false` and no `previous_response_id`
on every call; a tool round under the native strategy carrying its encrypted
reasoning; a compaction item in the reply once the input passes the
threshold; and a successor turn - sent the cut tape and the tool result
alone - that still answers the question the opening asked.

    OPENAI_PROBE_API_KEY=... pytest tests/test_openai_native_live.py -q

`OPENAI_PROBE_MODEL` picks the model (a profiled one, `gpt-5.6` by
default); `OPENAI_PROBE_COMPACT_THRESHOLD` the forced threshold, in tokens.
"""

from __future__ import annotations

import os

import pytest

from liminallm.service.continuation import (
    OPENAI_RESPONSES_NATIVE_V1,
    ProviderContinuation,
)

pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_PROBE_API_KEY"),
    reason="live OpenAI witness; set OPENAI_PROBE_API_KEY to run it",
)

TOOLS = [{"type": "function", "function": {
    "name": "lookup",
    "description": "Look a maintenance value up in the parts database.",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                   "required": ["query"]},
}}]

QUESTION = (
    "What is the calibration offset of the Kestrel-9 relay module? It is not "
    "in the manual: look it up, then answer with the offset only."
)
ANSWER = "Kestrel-9 relay module calibration offset: 0.0417 rad"


def _manual(lines: int) -> str:
    """Filler long enough to pass a forced threshold: distinct lines, so the
    provider cannot fold them into a few tokens."""
    return "\n".join(
        f"Section {i}: unit {i} of the {['pump', 'valve', 'relay', 'sensor'][i % 4]} "
        f"train is rated at {i * 3} units, inspected every {i % 12 + 1} months, "
        f"and logged under ticket K-{i * 7919 % 10007}."
        for i in range(lines)
    )


def test_a_forced_compaction_keeps_the_conversation_going(monkeypatch):
    from liminallm.service import model_backend as mb

    model = os.environ.get("OPENAI_PROBE_MODEL", "gpt-5.6")
    forced = int(os.environ.get("OPENAI_PROBE_COMPACT_THRESHOLD", "8192"))
    assert mb.supports_native_compaction(model), f"{model} is outside the profile"
    backend = mb.ApiAdapterBackend(
        model, adapter_mode="api_adapters", backend_mode="openai",
        api_key=os.environ["OPENAI_PROBE_API_KEY"], provider="openai",
        api_key_env="OPENAI_PROBE_API_KEY",
    )
    assert backend.declared_continuation() == OPENAI_RESPONSES_NATIVE_V1
    monkeypatch.setattr(mb, "compact_threshold", lambda window: forced)
    requests = []
    real_create = backend.client.responses.create

    def create(**kw):
        requests.append(kw)
        return real_create(**kw)

    monkeypatch.setattr(backend.client.responses, "create", create)

    opening = [
        {"role": "system", "content": "You are a terse maintenance assistant. Use the "
                                      "lookup tool for any value the manual does not "
                                      "state.\n\nReference manual:\n" + _manual(700)},
        {"role": "user", "content": QUESTION},
    ]
    first = backend.generate_with_tools(opening, TOOLS, [])

    calls = first["tool_calls"]
    assert calls and calls[0]["name"] == "lookup", first["content"]
    items = first["continuation"]["payload"]["items"]
    types = [item["type"] for item in items]
    assert "compaction" in types, f"no compaction item at threshold {forced}: {types}"
    assert types[0] == "compaction", types
    assert "reasoning" in types and any(
        item.get("encrypted_content") for item in items if item["type"] == "reasoning"
    )
    assert "created_by" not in items[0] and items[0]["encrypted_content"]
    # Restarted at this turn's own reasoning, not carried from anything.
    assert first["continuation"]["replay_tokens"] == first["usage"].get("reasoning_tokens", 0)

    accepted = ProviderContinuation(
        strategy=OPENAI_RESPONSES_NATIVE_V1, provider="openai", transport="responses",
        model=model, through_operation_seq=1, payload=first["continuation"]["payload"],
        replay_tokens=first["continuation"]["replay_tokens"],
    )
    tail = [{"role": "tool", "tool_call_id": calls[0]["id"], "name": "lookup",
             "content": ANSWER}]
    answer = None
    for _round in range(3):
        second = backend.generate_with_tools(tail, TOOLS, [], continuation=accepted)
        if not second["tool_calls"]:
            answer = second
            break
        payload = second["continuation"]
        accepted = ProviderContinuation(
            strategy=payload["strategy"], provider=payload["provider"],
            transport=payload["transport"], model=payload["model"],
            through_operation_seq=accepted.through_operation_seq + 1,
            payload=payload["payload"], replay_tokens=payload["replay_tokens"],
        )
        tail = [{"role": "tool", "tool_call_id": second["tool_calls"][0]["id"],
                 "name": "lookup", "content": ANSWER}]
    assert answer is not None, "the model kept calling the tool"

    # Continuity across the boundary: the successor was sent the cut tape
    # and the tool result alone, and answered the question the opening
    # asked, from a far smaller prompt.
    assert "0.0417" in answer["content"], answer["content"]
    successor = requests[1]
    assert successor["input"][0]["type"] == "compaction"
    assert all(item.get("role") != "system" for item in successor["input"])
    assert successor["input"][-1]["type"] == "function_call_output"
    assert answer["usage"]["prompt_tokens"] < first["usage"]["prompt_tokens"]
    for kw in requests:
        assert kw["store"] is False
        assert "previous_response_id" not in kw and "conversation" not in kw
        assert kw["include"] == ["reasoning.encrypted_content"]
        assert kw["extra_body"]["context_management"] == [
            {"type": "compaction", "compact_threshold": forced}
        ]
