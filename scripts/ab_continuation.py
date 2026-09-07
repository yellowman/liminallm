#!/usr/bin/env python3
"""A/B: transcript reconstruction against the provider's own continuation.

One multi-tool task, driven through a provider's adapter two ways, and the
cost of each recorded. Under `transcript` every model call is sent the
whole conversation rebuilt chat-shaped from what the loop kept - the way
every round was sent before the provider's continuation existed, and the
way a compatible provider is still sent. Under `native` each call after the
first is sent the accepted continuation and only the tool results since,
the way the parent sends it (SPEC §5, provider continuation).

Recorded per run: prompt, completion, reasoning and total tokens summed
over the calls, wall latency, model calls, tool calls, and whether the
final answer is correct. The task chains three lookups, each needing the
previous result, so a model that lost its place cannot answer by luck.

Usage:
    GEMINI_PROBE_API_KEY=... python scripts/ab_continuation.py \\
        --backend gemini --model gemini-3-flash-preview --repeat 3
    OPENAI_PROBE_API_KEY=... python scripts/ab_continuation.py \\
        --backend openai --model gpt-5.6 --repeat 3

Adapter-level on purpose: what is measured is the wire cost of the two
ways of continuing, with the same adapter, task and tools, and nothing of
the service around them. Keys come from the environment only.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from liminallm.service.continuation import ProviderContinuation  # noqa: E402

EXPECTED = "0.0417"
PART = "KR9-4417-B"
FREQUENCY = "4417"

TOOLS = [
    {"type": "function", "function": {
        "name": "parts_catalog",
        "description": "Find a component's part number by name.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                       "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "spec_sheet",
        "description": "The operating frequency of a component, by exact part number.",
        "parameters": {"type": "object",
                       "properties": {"part_number": {"type": "string"}},
                       "required": ["part_number"]}}},
    {"type": "function", "function": {
        "name": "calibration",
        "description": "The calibration offset for an operating frequency in kilohertz.",
        "parameters": {"type": "object",
                       "properties": {"frequency_khz": {"type": "string"}},
                       "required": ["frequency_khz"]}}},
]

SYSTEM = (
    "You are a terse maintenance assistant. Each lookup depends on the "
    "previous result: find the part number, then its operating frequency, "
    "then the calibration offset for that frequency. Call one tool at a "
    "time. When you have the offset, answer with the offset and nothing else."
)
QUESTION = "What is the calibration offset for the Kestrel-9 relay module?"


def run_tool(name: str, arguments: dict) -> str:
    text = " ".join(str(v) for v in arguments.values()).lower()
    if name == "parts_catalog":
        if "kestrel" in text:
            return f"Kestrel-9 relay module: part number {PART}"
        return "no such component"
    if name == "spec_sheet":
        if PART.lower() in text:
            return f"{PART} operates at {FREQUENCY} kHz"
        return "unknown part number"
    if name == "calibration":
        if FREQUENCY in text:
            return f"calibration offset for {FREQUENCY} kHz: {EXPECTED} rad"
        return "no calibration table for that frequency"
    return f"unknown tool {name}"


def build_backend(kind: str, model: str):
    if kind == "gemini":
        from liminallm.service.gemini_backend import GeminiBackend

        key = os.environ.get("GEMINI_PROBE_API_KEY")
        if not key:
            sys.exit("GEMINI_PROBE_API_KEY is not set")
        return GeminiBackend(model, api_key=key)
    if kind == "openai":
        from liminallm.service.model_backend import ApiAdapterBackend

        key = os.environ.get("OPENAI_PROBE_API_KEY")
        if not key:
            sys.exit("OPENAI_PROBE_API_KEY is not set")
        return ApiAdapterBackend(
            model, adapter_mode="api_adapters", backend_mode="openai", api_key=key,
            provider="openai", api_key_env="OPENAI_PROBE_API_KEY",
        )
    sys.exit(f"unknown backend {kind!r}: gemini or openai")


COUNTED = ("prompt_tokens", "completion_tokens", "reasoning_tokens",
           "total_tokens", "cached_tokens")


def run_arm(backend, arm: str, max_rounds: int) -> dict:
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": QUESTION}]
    tail = list(messages)
    continuation = None
    totals = {key: 0 for key in COUNTED}
    model_calls = tool_calls = 0
    final = ""
    started = time.monotonic()
    for _round in range(max_rounds):
        native = arm == "native" and continuation is not None
        reply = backend.generate_with_tools(
            tail if native else messages, TOOLS, [],
            **({"continuation": continuation} if native else {}),
        )
        model_calls += 1
        usage = reply.get("usage") or {}
        for key in COUNTED:
            totals[key] += int(usage.get(key) or 0)
        if arm == "native":
            c = reply["continuation"]
            continuation = ProviderContinuation(
                strategy=c["strategy"], provider=c["provider"],
                transport=c["transport"], model=c["model"],
                through_operation_seq=model_calls, payload=c["payload"],
                replay_tokens=int(c.get("replay_tokens") or 0),
            )
        calls = reply.get("tool_calls") or []
        messages.append(reply.get("assistant_message")
                        or {"role": "assistant", "content": reply.get("content") or ""})
        if not calls:
            final = reply.get("content") or ""
            break
        results = []
        for call in calls:
            tool_calls += 1
            arguments = call.get("arguments") or "{}"
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except ValueError:
                    arguments = {"raw": arguments}
            results.append({
                "role": "tool", "tool_call_id": call.get("id") or "",
                "name": call.get("name") or "", "content": run_tool(call.get("name") or "", arguments),
            })
        messages.extend(results)
        tail = results
    return {
        "arm": arm,
        "model_calls": model_calls,
        "tool_calls": tool_calls,
        "wall_seconds": round(time.monotonic() - started, 3),
        "correct": EXPECTED in final,
        "final": final.strip()[:200],
        "replay_tokens": continuation.replay_tokens if continuation is not None else 0,
        **totals,
    }


def summarize(runs: list[dict]) -> str:
    columns = ("prompt_tokens", "reasoning_tokens", "total_tokens",
               "wall_seconds", "model_calls", "tool_calls")
    lines = ["arm         n  correct  " + "  ".join(f"{c:>16}" for c in columns)]
    for arm in ("transcript", "native"):
        rows = [r for r in runs if r["arm"] == arm]
        if not rows:
            continue
        correct = sum(1 for r in rows if r["correct"])
        means = "  ".join(f"{statistics.mean(r[c] for r in rows):>16.1f}" for c in columns)
        lines.append(f"{arm:<10} {len(rows):>2}  {correct:>3}/{len(rows):<3}  {means}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--backend", required=True, choices=("gemini", "openai"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--arms", default="transcript,native")
    parser.add_argument("--max-rounds", type=int, default=8)
    parser.add_argument("--out", help="append every run as one JSON line here")
    args = parser.parse_args()

    backend = build_backend(args.backend, args.model)
    runs = []
    for index in range(args.repeat):
        for arm in args.arms.split(","):
            run = run_arm(backend, arm.strip(), args.max_rounds)
            run.update({"backend": args.backend, "model": args.model, "repeat": index})
            runs.append(run)
            print(json.dumps(run, ensure_ascii=False), flush=True)
            if args.out:
                with open(args.out, "a", encoding="utf-8") as sink:
                    sink.write(json.dumps(run, ensure_ascii=False) + "\n")
    print()
    print(summarize(runs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
