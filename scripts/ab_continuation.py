#!/usr/bin/env python3
"""A/B: transcript reconstruction against the provider's own continuation.

One deterministic tool workload, driven through a provider's adapter two
ways, and the cost of each recorded. Under `transcript` every model call is
sent the whole conversation rebuilt chat-shaped from what the loop kept -
the way every round was sent before the provider's continuation existed,
and the way a compatible provider is still sent. Under `native` each call
after the first is sent the accepted continuation and only the tool
results since, the way the parent sends it (SPEC §5, provider
continuation).

Two workloads. `kestrel` is the original three-lookup task. `chain` is a
generated chain of records of any length: each record names the next one,
so the calls are sequential by construction, and the answer is the sum of
every code along the chain, so it needs everything learned on the way.
The same generator serves both arms.

Recorded per run: correctness, prompt, output, reasoning, cached and
total tokens summed over the calls, wall time, model and tool calls, and
the native state's growth - replay item count, serialized replay bytes,
the adapter's `replay_tokens`, and whether the provider compacted. Per
call, the same numbers as they stood when that call was made, so the
requests just before and after a compaction can be read side by side.
Nothing of a payload's contents is recorded: counts, types, byte sizes and
token counts only.

Usage:
    GEMINI_PROBE_API_KEY=... python scripts/ab_continuation.py \\
        --backend gemini --model gemini-3-flash-preview \\
        --workload chain --rounds 3,10,20 --repeat 3 --out runs.jsonl
    OPENAI_PROBE_API_KEY=... python scripts/ab_continuation.py \\
        --backend openai --model gpt-5.6 --workload chain --rounds 20 \\
        --compact-threshold 8192 --repeat 3 --out runs.jsonl
    python scripts/ab_continuation.py --summarize runs.jsonl

`--compact-threshold` forces the OpenAI adapter's threshold, for a
compaction on a workload far shorter than a deployment's window; it is a
harness override and changes nothing in the service. Adapter-level on
purpose: what is measured is the wire cost of the two ways of continuing,
with the same adapter, task and tools, and nothing of the service around
them. Keys come from the environment only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from liminallm.service.continuation import ProviderContinuation  # noqa: E402

# -- the kestrel workload: three lookups, each needing the last ------------

KESTREL_EXPECTED = "0.0417"
PART = "KR9-4417-B"
FREQUENCY = "4417"

KESTREL_TOOLS = [
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

KESTREL_SYSTEM = (
    "You are a terse maintenance assistant. Each lookup depends on the "
    "previous result: find the part number, then its operating frequency, "
    "then the calibration offset for that frequency. Call one tool at a "
    "time. When you have the offset, answer with the offset and nothing else."
)
KESTREL_QUESTION = "What is the calibration offset for the Kestrel-9 relay module?"


def kestrel_tool(name: str, arguments: dict) -> str:
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
            return f"calibration offset for {FREQUENCY} kHz: {KESTREL_EXPECTED} rad"
        return "no calibration table for that frequency"
    return f"unknown tool {name}"


# -- the chain workload: n records, each naming the next ------------------

CHAIN_TOOLS = [
    {"type": "function", "function": {
        "name": "next_record",
        "description": "Read one record of the chain by id. It gives the record's "
                       "code and the id of the next record, or says the chain ends.",
        "parameters": {"type": "object",
                       "properties": {"record_id": {"type": "string"}},
                       "required": ["record_id"]}}},
]

CHAIN_SYSTEM = (
    "You are a terse assistant following a chain of records with the "
    "next_record tool. Each record gives a code and names the next record. "
    "Call next_record for exactly one record per turn, starting at R0, and "
    "keep going until a record says the chain ends. Keep the running sum of "
    "the codes. When the chain ends, answer with the sum of all the codes "
    "along the chain, as a number and nothing else."
)
CHAIN_QUESTION = "Start at R0. What is the sum of the codes along the chain?"


def chain_codes(rounds: int) -> List[int]:
    """The codes along a chain of this length: fixed, spread over 1 to 9, so
    two runs of one length see one chain and two lengths see different sums."""
    return [(index * 7 + 3) % 9 + 1 for index in range(rounds)]


def chain_tool(rounds: int, arguments: dict) -> str:
    codes = chain_codes(rounds)
    raw = str(arguments.get("record_id") or next(iter(arguments.values()), "")).strip()
    match = re.fullmatch(r"[Rr]?(\d+)", raw)
    index = int(match.group(1)) if match else -1
    if not 0 <= index < rounds:
        return f"no such record: {raw!r}"
    following = f"R{index + 1}" if index + 1 < rounds else "none, the chain ends here"
    return f"record R{index}: code {codes[index]}. next record: {following}"


class Workload:
    def __init__(self, name: str, rounds: int) -> None:
        self.name = name
        self.rounds = rounds
        if name == "kestrel":
            self.tools = KESTREL_TOOLS
            self.messages = [{"role": "system", "content": KESTREL_SYSTEM},
                             {"role": "user", "content": KESTREL_QUESTION}]
            self.expected = KESTREL_EXPECTED
        elif name == "chain":
            self.tools = CHAIN_TOOLS
            self.messages = [{"role": "system", "content": CHAIN_SYSTEM},
                             {"role": "user", "content": CHAIN_QUESTION}]
            self.expected = str(sum(chain_codes(rounds)))
        else:
            raise ValueError(f"unknown workload {name!r}: kestrel or chain")

    def run_tool(self, name: str, arguments: dict) -> str:
        if self.name == "kestrel":
            return kestrel_tool(name, arguments)
        if name == "next_record":
            return chain_tool(self.rounds, arguments)
        return f"unknown tool {name}"

    def correct(self, final: str) -> bool:
        if self.name == "kestrel":
            return self.expected in final
        numbers = re.findall(r"\d+", final)
        return bool(numbers) and numbers[-1] == self.expected


# -- backends ---------------------------------------------------------------

def build_backend(kind: str, model: str, reasoning_effort: Optional[str]):
    if kind == "gemini":
        from liminallm.service.gemini_backend import GeminiBackend

        key = os.environ.get("GEMINI_PROBE_API_KEY")
        if not key:
            sys.exit("GEMINI_PROBE_API_KEY is not set")
        return GeminiBackend(model, api_key=key, reasoning_effort=reasoning_effort)
    if kind == "openai":
        from liminallm.service.model_backend import ApiAdapterBackend

        key = os.environ.get("OPENAI_PROBE_API_KEY")
        if not key:
            sys.exit("OPENAI_PROBE_API_KEY is not set")
        return ApiAdapterBackend(
            model, adapter_mode="api_adapters", backend_mode="openai", api_key=key,
            provider="openai", api_key_env="OPENAI_PROBE_API_KEY",
            reasoning_effort=reasoning_effort,
        )
    sys.exit(f"unknown backend {kind!r}: gemini or openai")


def sdk_version(kind: str) -> str:
    if kind == "openai":
        import openai

        return f"openai {openai.__version__}"
    import httpx

    return f"httpx {httpx.__version__} (native wire, no SDK)"


def force_threshold(threshold: int) -> None:
    """A harness override of the OpenAI adapter's threshold: the module
    function the adapter looks up at call time answers this number for
    any window. Changes nothing in the service."""
    from liminallm.service import model_backend as mb

    mb.compact_threshold = lambda window: threshold


# -- sanitized measurements of a continuation -----------------------------

def state_measures(payload: Optional[dict]) -> Dict[str, Any]:
    """What a payload holds, by count and size and never by content."""
    if not payload:
        return {"replay_items": 0, "replay_bytes": 0, "item_types": {}, "compaction_ids": []}
    items = payload.get("items")
    if isinstance(items, list):  # the OpenAI tape
        types = Counter(str(item.get("type") or "message") for item in items)
        compactions = [str(item.get("id") or "?") for item in items
                       if item.get("type") == "compaction"]
        return {
            "replay_items": len(items),
            "replay_bytes": len(json.dumps(payload, ensure_ascii=False).encode()),
            "item_types": dict(sorted(types.items())),
            "compaction_ids": compactions,
        }
    contents = payload.get("contents") or []  # the Gemini conversation
    parts = Counter()
    for content in contents:
        for part in content.get("parts") or []:
            if "functionCall" in part:
                kind = "functionCall"
            elif "functionResponse" in part:
                kind = "functionResponse"
            elif part.get("thought"):
                kind = "thought"
            else:
                kind = "text"
            parts[kind] += 1
            if part.get("thoughtSignature"):
                parts["signed"] += 1
    return {
        "replay_items": len(contents),
        "replay_bytes": len(json.dumps(payload, ensure_ascii=False).encode()),
        "item_types": dict(sorted(parts.items())),
        "compaction_ids": [],
    }


COUNTED = ("prompt_tokens", "completion_tokens", "reasoning_tokens",
           "cached_tokens", "total_tokens")


def run_arm(backend, kind: str, workload: Workload, arm: str, max_rounds: int,
            context_window: Optional[int]) -> dict:
    messages = [dict(m) for m in workload.messages]
    tail = list(messages)
    continuation = None
    totals = {key: 0 for key in COUNTED}
    calls: List[dict] = []
    model_calls = tool_calls = 0
    final = ""
    seen_compactions: set = set()
    started = time.monotonic()
    for _round in range(max_rounds):
        native = arm == "native" and continuation is not None
        handed = state_measures(continuation.payload if native else None)
        began = time.monotonic()
        reply = backend.generate_with_tools(
            tail if native else messages, workload.tools, [],
            **({"continuation": continuation} if native else {}),
            **({"context_window": context_window} if context_window else {}),
        )
        elapsed = time.monotonic() - began
        model_calls += 1
        usage = reply.get("usage") or {}
        record: Dict[str, Any] = {
            "call": model_calls,
            "wall_seconds": round(elapsed, 3),
            "replay_items_handed": handed["replay_items"],
            "replay_bytes_handed": handed["replay_bytes"],
            "replay_tokens_handed": continuation.replay_tokens if native else 0,
        }
        for key in COUNTED:
            value = int(usage.get(key) or 0)
            totals[key] += value
            record[key] = value
        if arm == "native":
            c = reply["continuation"]
            continuation = ProviderContinuation(
                strategy=c["strategy"], provider=c["provider"],
                transport=c["transport"], model=c["model"],
                through_operation_seq=model_calls, payload=c["payload"],
                replay_tokens=int(c.get("replay_tokens") or 0),
            )
            produced = state_measures(continuation.payload)
            new = [i for i in produced["compaction_ids"] if i not in seen_compactions]
            seen_compactions.update(produced["compaction_ids"])
            record.update({
                "replay_items_after": produced["replay_items"],
                "replay_bytes_after": produced["replay_bytes"],
                "replay_tokens_after": continuation.replay_tokens,
                "item_types_after": produced["item_types"],
                "compacted": bool(new),
            })
            if new:
                items = continuation.payload.get("items") or []
                record["order_after_compaction"] = [
                    str(item.get("type") or "message") for item in items[:8]
                ]
        calls.append(record)
        tool_requests = reply.get("tool_calls") or []
        messages.append(reply.get("assistant_message")
                        or {"role": "assistant", "content": reply.get("content") or ""})
        if not tool_requests:
            final = reply.get("content") or ""
            break
        results = []
        for call in tool_requests:
            tool_calls += 1
            arguments = call.get("arguments") or "{}"
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except ValueError:
                    arguments = {"raw": arguments}
            results.append({
                "role": "tool", "tool_call_id": call.get("id") or "",
                "name": call.get("name") or "",
                "content": workload.run_tool(call.get("name") or "", arguments),
            })
        messages.extend(results)
        tail = results
    end_state = state_measures(continuation.payload if continuation is not None else None)
    return {
        "workload": workload.name,
        "rounds": workload.rounds,
        "arm": arm,
        "model_calls": model_calls,
        "tool_calls": tool_calls,
        "wall_seconds": round(time.monotonic() - started, 3),
        "correct": workload.correct(final),
        "expected": workload.expected,
        "final": final.strip()[:120],
        "replay_items": end_state["replay_items"],
        "replay_bytes": end_state["replay_bytes"],
        "replay_tokens": continuation.replay_tokens if continuation is not None else 0,
        "compactions": len(seen_compactions),
        **totals,
        "calls": calls,
    }


# -- summaries ------------------------------------------------------------

COLUMNS = ("prompt_tokens", "reasoning_tokens", "completion_tokens", "cached_tokens",
           "total_tokens", "wall_seconds", "model_calls", "tool_calls", "replay_bytes")


def _mean(rows: List[dict], key: str) -> float:
    return statistics.mean(float(r.get(key) or 0) for r in rows)


def summarize(runs: List[dict]) -> str:
    """Per workload and length, each arm's means, then native less
    transcript on the three numbers the crossover question is about."""
    lines = []
    groups = sorted({(r["workload"], int(r["rounds"])) for r in runs})
    header = "workload  rounds  arm         n  correct  " + "  ".join(
        f"{c:>16}" for c in COLUMNS)
    lines.append(header)
    for workload, rounds in groups:
        for arm in ("transcript", "native"):
            rows = [r for r in runs if r["workload"] == workload
                    and int(r["rounds"]) == rounds and r["arm"] == arm]
            if not rows:
                continue
            correct = sum(1 for r in rows if r["correct"])
            means = "  ".join(f"{_mean(rows, c):>16.1f}" for c in COLUMNS)
            lines.append(f"{workload:<8}  {rounds:>6}  {arm:<10} {len(rows):>2}  "
                         f"{correct:>3}/{len(rows):<3}  {means}")
    lines.append("")
    lines.append("native minus transcript (means): workload  rounds  total_tokens  "
                 "reasoning_tokens  wall_seconds  correct(native/transcript)")
    for workload, rounds in groups:
        native = [r for r in runs if r["workload"] == workload
                  and int(r["rounds"]) == rounds and r["arm"] == "native"]
        transcript = [r for r in runs if r["workload"] == workload
                      and int(r["rounds"]) == rounds and r["arm"] == "transcript"]
        if not native or not transcript:
            continue
        lines.append(
            f"{workload:<8}  {rounds:>6}  "
            f"{_mean(native, 'total_tokens') - _mean(transcript, 'total_tokens'):>+12.1f}  "
            f"{_mean(native, 'reasoning_tokens') - _mean(transcript, 'reasoning_tokens'):>+16.1f}  "
            f"{_mean(native, 'wall_seconds') - _mean(transcript, 'wall_seconds'):>+12.2f}  "
            f"{sum(r['correct'] for r in native)}/{len(native)} vs "
            f"{sum(r['correct'] for r in transcript)}/{len(transcript)}"
        )
    compacting = [r for r in runs if r.get("compactions")]
    if compacting:
        lines.append("")
        lines.append("compaction, per compacting run: the call before, the first after, "
                     "the one after that (prompt tokens / wall / replay items / bytes / "
                     "replay_tokens handed)")
        for run in compacting:
            calls = run["calls"]
            for index, call in enumerate(calls):
                if not call.get("compacted"):
                    continue
                window = [c for c in calls[max(0, index - 1): index + 3]]
                lines.append(f"  {run['workload']} rounds={run['rounds']} repeat="
                             f"{run.get('repeat')} compaction at call {call['call']}; "
                             f"order after: {call.get('order_after_compaction')}")
                for c in window:
                    lines.append(
                        f"    call {c['call']:>2}: prompt {c['prompt_tokens']:>7}  "
                        f"wall {c['wall_seconds']:>6.2f}  handed items {c['replay_items_handed']:>4}"
                        f"  bytes {c['replay_bytes_handed']:>8}  replay_tokens "
                        f"{c['replay_tokens_handed']:>6}"
                        + ("  <- compacting reply" if c.get("compacted") else "")
                    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--backend", choices=("gemini", "openai"))
    parser.add_argument("--model")
    parser.add_argument("--workload", default="chain", choices=("kestrel", "chain"))
    parser.add_argument("--rounds", default="3", help="comma-separated chain lengths")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--arms", default="transcript,native")
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--compact-threshold", type=int, default=None,
                        help="force the OpenAI adapter's compaction threshold (harness only)")
    parser.add_argument("--out", help="append every run as one JSON line here")
    parser.add_argument("--summarize", help="print the tables for an existing JSONL file")
    args = parser.parse_args()

    if args.summarize:
        with open(args.summarize, encoding="utf-8") as source:
            runs = [json.loads(line) for line in source if line.strip()]
        print(summarize(runs))
        return 0
    if not args.backend or not args.model:
        parser.error("--backend and --model are required to run")

    backend = build_backend(args.backend, args.model, args.reasoning_effort)
    context_window = None
    if args.backend == "openai":
        context_window = backend.context_window
        if args.compact_threshold:
            force_threshold(args.compact_threshold)
    profile = {
        "backend": args.backend, "model": args.model, "sdk": sdk_version(args.backend),
        "reasoning_effort": args.reasoning_effort,
        "compact_threshold": (
            args.compact_threshold if args.compact_threshold else (
                __import__("liminallm.service.model_backend", fromlist=["compact_threshold"])
                .compact_threshold(context_window) if context_window else None)),
        "context_window_handed": context_window,
    }
    print(json.dumps({"profile": profile}), flush=True)
    runs = []
    for rounds in [int(r) for r in args.rounds.split(",")]:
        workload = Workload(args.workload, rounds)
        for index in range(args.repeat):
            for arm in args.arms.split(","):
                run = run_arm(backend, args.backend, workload, arm.strip(),
                              rounds + 6, context_window)
                run.update({"repeat": index, **profile})
                runs.append(run)
                print(json.dumps({k: v for k, v in run.items() if k != "calls"},
                                 ensure_ascii=False), flush=True)
                if args.out:
                    with open(args.out, "a", encoding="utf-8") as sink:
                        sink.write(json.dumps(run, ensure_ascii=False) + "\n")
    print()
    print(summarize(runs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
