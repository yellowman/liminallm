#!/usr/bin/env python3
"""Reasoning fidelity across a tool boundary: does the provider's own
continuation carry hidden reasoning that transcript reconstruction loses?

PN-M1 measured what Gemini's native continuation costs to replay. This
asks what the replayed state buys. Every task here puts a tool boundary
after the model has reasoned and before it answers: the model is told to
solve the problem, then call `checkpoint(stage=1)` - a call that carries
no solution state - then answer after the tool says "continue". Under
`transcript` the successor call is rebuilt from the visible material
alone: the problem, a fixed call, "continue", with the placeholder thought
signature. Under `native` the successor is sent exactly the same visible
material plus the provider's genuine signed candidate state. The hidden
derivation is the one thing that differs.

Three deterministic task families, each generated from a seed and scored
exactly, in three difficulty bands:

- `order`: a finishing-order puzzle - runners and clues with one solution,
  checked by a solver at generation time;
- `arith`: a chain of arithmetic and digit transformations on an integer;
- `trace`: a small looping program with a conditional, traced to its final
  value.

Two controls. `trivial` needs almost no reasoning before the checkpoint:
native should show its cost and no benefit. `--visible` changes the
checkpoint so the model must put its answer into the call's arguments:
the transcript then carries the result, and reconstruction should lose
little. Arm order is counterbalanced by seed and repeat.

Recorded per run: correctness, reasoning tokens before and after the
checkpoint, prompt and total tokens, wall time per call, the
`replay_tokens` and serialized bytes handed to the successor, and
protocol compliance - whether the checkpoint was called, whether the
first reply leaked answer text. Counts, sizes and token numbers only;
nothing of a thought or a signature is read or stored.

Usage:
    GEMINI_PROBE_API_KEY=... python scripts/reasoning_fidelity.py \\
        --backend gemini --model gemini-3-flash-preview \\
        --families order,arith,trace --bands light,medium,heavy \\
        --seeds 4 --repeat 2 --out fidelity.jsonl
    ... --families trivial --bands light --seeds 4 --repeat 2 --out fidelity.jsonl
    ... --visible --bands heavy --seeds 4 --repeat 1 --out fidelity.jsonl
    python scripts/reasoning_fidelity.py --summarize fidelity.jsonl
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ab_continuation import build_backend, sdk_version, state_measures  # noqa: E402

from liminallm.service.continuation import ProviderContinuation  # noqa: E402

# -- the checkpoint protocol ---------------------------------------------

CHECKPOINT_TOOL = {"type": "function", "function": {
    "name": "checkpoint",
    "description": "A required checkpoint. Call it exactly once, after you have "
                   "worked out your answer and before you state it.",
    "parameters": {"type": "object", "properties": {"stage": {"type": "integer"}},
                   "required": ["stage"]},
}}

VISIBLE_CHECKPOINT_TOOL = {"type": "function", "function": {
    "name": "checkpoint",
    "description": "A required checkpoint. Call it exactly once, after you have "
                   "worked out your answer and before you state it, with your "
                   "answer in `result`.",
    "parameters": {"type": "object",
                   "properties": {"stage": {"type": "integer"},
                                  "result": {"type": "string"}},
                   "required": ["stage", "result"]},
}}

SYSTEM = (
    "You are a careful solver. Work the problem out completely in your head. "
    "Before you state any answer you must call the checkpoint tool exactly "
    "once, as checkpoint(stage=1), writing nothing else in that turn. When the "
    "tool returns \"continue\", reply with the final answer only, in the format "
    "the problem asks for, and nothing else."
)

VISIBLE_SYSTEM = (
    "You are a careful solver. Work the problem out completely in your head. "
    "Before you state any answer you must call the checkpoint tool exactly "
    "once, as checkpoint(stage=1, result=<your final answer>), writing nothing "
    "else in that turn. When the tool returns \"continue\", reply with the final "
    "answer only, in the format the problem asks for, and nothing else."
)

CHECKPOINT_RESULT = "continue"

BANDS = ("light", "medium", "heavy")


# -- family: finishing order ---------------------------------------------

NAMES = ("Ada", "Bo", "Cy", "Di", "Ed", "Fay", "Gus", "Hal", "Ivy", "Jo")

ORDER_SIZE = {"light": 4, "medium": 6, "heavy": 8}


def _clue_holds(clue: Tuple, order: Tuple[str, ...]) -> bool:
    position = {name: index for index, name in enumerate(order)}
    kind = clue[0]
    if kind == "before":
        return position[clue[1]] < position[clue[2]]
    if kind == "adjacent":
        return position[clue[2]] - position[clue[1]] == 1
    if kind == "between":
        return abs(position[clue[1]] - position[clue[2]]) == 2
    if kind == "not_at":
        return position[clue[1]] != clue[2]
    if kind == "at":
        return position[clue[1]] == clue[2]
    raise ValueError(kind)


def _clue_text(clue: Tuple) -> str:
    kind = clue[0]
    if kind == "before":
        return f"{clue[1]} finished before {clue[2]}."
    if kind == "adjacent":
        return f"{clue[2]} finished immediately after {clue[1]}."
    if kind == "between":
        return f"Exactly one runner finished between {clue[1]} and {clue[2]}."
    if kind == "not_at":
        return f"{clue[1]} did not finish in position {clue[2] + 1}."
    return f"{clue[1]} finished in position {clue[2] + 1}."


def make_order(band: str, seed: int) -> Dict[str, Any]:
    """Runners and clues with exactly one finishing order.

    Every clue is generated true of a hidden order, and the orders still
    consistent with the clues so far are kept as a list; a clue is added
    only when it rules at least one of them out, until the hidden order
    is the one left. The harder bands draw from the weaker clue kinds
    only, so more of the work is inference."""
    rng = random.Random(f"order-{band}-{seed}")
    size = ORDER_SIZE[band]
    names = tuple(rng.sample(NAMES, size))
    hidden = tuple(rng.sample(names, size))
    position = {name: index for index, name in enumerate(hidden)}
    kinds = ["before", "not_at", "between", "adjacent"] + (["at"] if band == "light" else [])
    remaining = list(itertools.permutations(names))
    clues: List[Tuple] = []
    while len(remaining) > 1:
        kind = rng.choice(kinds)
        a, b = rng.sample(names, 2)
        if kind == "before":
            clue = ("before", a, b) if position[a] < position[b] else ("before", b, a)
        elif kind == "adjacent":
            first = rng.choice([n for n in names if position[n] < size - 1])
            clue = ("adjacent", first, hidden[position[first] + 1])
        elif kind == "between":
            first = rng.choice([n for n in names if position[n] < size - 2])
            clue = ("between", first, hidden[position[first] + 2])
        elif kind == "not_at":
            wrong = rng.choice([p for p in range(size) if p != position[a]])
            clue = ("not_at", a, wrong)
        else:
            clue = ("at", a, position[a])
        if clue in clues:
            continue
        narrowed = [order for order in remaining if _clue_holds(clue, order)]
        if len(narrowed) < len(remaining):
            clues.append(clue)
            remaining = narrowed
    assert remaining == [hidden]
    rng.shuffle(clues)
    prompt = (
        f"{size} runners finished a race: {', '.join(names)}. Clues:\n"
        + "\n".join(f"- {_clue_text(c)}" for c in clues)
        + "\nList the runners in finishing order, first to last, separated by commas."
    )
    return {"prompt": prompt, "expected": ", ".join(hidden), "kind": "order",
            "detail": {"runners": size, "clues": len(clues)}}


# -- family: arithmetic and digit transformations -------------------------

ARITH_STEPS = {"light": 4, "medium": 8, "heavy": 14}


def _digit_reverse(value: int) -> int:
    return int(str(abs(value))[::-1]) * (1 if value >= 0 else -1)


def make_arith(band: str, seed: int) -> Dict[str, Any]:
    rng = random.Random(f"arith-{band}-{seed}")
    steps = ARITH_STEPS[band]
    value = rng.randint(20, 99)
    start = value
    lines = []
    for index in range(steps):
        kind = rng.choice(["add", "sub", "mul", "mod", "reverse", "digits"])
        if kind == "add":
            k = rng.randint(11, 89)
            value += k
            lines.append(f"add {k}")
        elif kind == "sub":
            k = rng.randint(7, 60)
            value -= k
            lines.append(f"subtract {k}")
        elif kind == "mul":
            k = rng.randint(2, 7)
            value *= k
            lines.append(f"multiply by {k}")
        elif kind == "mod":
            k = rng.choice([97, 101, 113, 127])
            value = value % k
            lines.append(f"take the remainder after dividing by {k} (a remainder is never negative)")
        elif kind == "reverse":
            value = _digit_reverse(value)
            lines.append("reverse the order of its digits (keep the sign)")
        else:
            k = rng.randint(2, 5)
            value = sum(int(d) for d in str(abs(value))) * k
            lines.append(f"replace it with the sum of its digits multiplied by {k}")
        if abs(value) > 10 ** 6:
            value = value % 1009
            lines.append("take the remainder after dividing by 1009")
    prompt = (
        f"Start with the number {start}. Apply these steps in order, each to the "
        "result of the previous one:\n"
        + "\n".join(f"{i + 1}. {line}" for i, line in enumerate(lines))
        + "\nWhat is the final number? Answer with the number only."
    )
    return {"prompt": prompt, "expected": str(value), "kind": "integer",
            "detail": {"steps": len(lines)}}


# -- family: program tracing ---------------------------------------------

TRACE_ITERATIONS = {"light": 3, "medium": 6, "heavy": 10}


def make_trace(band: str, seed: int) -> Dict[str, Any]:
    rng = random.Random(f"trace-{band}-{seed}")
    iterations = TRACE_ITERATIONS[band]
    x, y = rng.randint(3, 40), rng.randint(1, 12)
    c, m = rng.randint(2, 9), rng.choice([7, 11, 13])
    k = rng.randint(2, 5)
    x0, y0 = x, y
    for _ in range(iterations):
        if x % 2 == 0:
            x = x // 2 + y
        else:
            x = k * x - y
        if x % 3 == 0:
            y = (y + c) % m
        else:
            y = (y + 1) % m
    program = (
        f"x = {x0}\ny = {y0}\nrepeat {iterations} times:\n"
        "    if x is even:\n        x = x // 2 + y\n"
        f"    else:\n        x = {k} * x - y\n"
        f"    if x is divisible by 3:\n        y = (y + {c}) mod {m}\n"
        f"    else:\n        y = (y + 1) mod {m}"
    )
    prompt = (
        "Trace this program exactly. `//` is integer division rounding down, "
        "`mod` is the remainder, and the two `if` blocks run in order on every "
        f"pass:\n\n{program}\n\nWhat is the value of x when the program ends? "
        "Answer with the number only."
    )
    return {"prompt": prompt, "expected": str(x), "kind": "integer",
            "detail": {"iterations": iterations}}


# -- control: trivial --------------------------------------------------------

def make_trivial(band: str, seed: int) -> Dict[str, Any]:
    rng = random.Random(f"trivial-{band}-{seed}")
    a, b = rng.randint(2, 9), rng.randint(2, 9)
    return {"prompt": f"What is {a} + {b}? Answer with the number only.",
            "expected": str(a + b), "kind": "integer", "detail": {}}


FAMILIES = {"order": make_order, "arith": make_arith, "trace": make_trace,
            "trivial": make_trivial}


def normalize(kind: str, text: str) -> str:
    text = (text or "").strip()
    if kind == "integer":
        numbers = re.findall(r"-?\d+", text)
        return numbers[-1] if numbers else ""
    names = re.findall(r"[A-Za-z]+", text)
    return ", ".join(name.capitalize() for name in names)


def correct(task: Dict[str, Any], text: str) -> bool:
    return normalize(task["kind"], text) == normalize(task["kind"], task["expected"])


# -- one run --------------------------------------------------------------

def run_task(backend, task: Dict[str, Any], arm: str, visible: bool) -> Dict[str, Any]:
    tool = VISIBLE_CHECKPOINT_TOOL if visible else CHECKPOINT_TOOL
    messages = [{"role": "system", "content": VISIBLE_SYSTEM if visible else SYSTEM},
                {"role": "user", "content": task["prompt"]}]
    tail = list(messages)
    continuation = None
    calls: List[dict] = []
    final = ""
    checkpoint_calls = 0
    leaked = False
    leaked_answer = False
    started = time.monotonic()
    for index in range(4):
        native = arm == "native" and continuation is not None
        handed = state_measures(continuation.payload if native else None)
        began = time.monotonic()
        reply = backend.generate_with_tools(
            tail if native else messages, [tool], [],
            **({"continuation": continuation} if native else {}),
        )
        elapsed = time.monotonic() - began
        usage = reply.get("usage") or {}
        record = {
            "call": index + 1,
            "wall_seconds": round(elapsed, 3),
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
            "reasoning_tokens": int(usage.get("reasoning_tokens") or 0),
            "total_tokens": int(usage.get("total_tokens") or 0),
            "replay_tokens_handed": continuation.replay_tokens if native else 0,
            "replay_bytes_handed": handed["replay_bytes"],
            "replay_items_handed": handed["replay_items"],
        }
        if arm == "native":
            c = reply["continuation"]
            continuation = ProviderContinuation(
                strategy=c["strategy"], provider=c["provider"], transport=c["transport"],
                model=c["model"], through_operation_seq=index + 1, payload=c["payload"],
                replay_tokens=int(c.get("replay_tokens") or 0),
            )
        requests = reply.get("tool_calls") or []
        content = (reply.get("content") or "").strip()
        record["tool_calls"] = len(requests)
        record["content_chars"] = len(content)
        calls.append(record)
        messages.append(reply.get("assistant_message")
                        or {"role": "assistant", "content": content})
        if not requests:
            final = content
            break
        if index == 0 and content:
            leaked = True
            leaked_answer = correct(task, content)
        results = []
        for call in requests:
            if call.get("name") == "checkpoint":
                checkpoint_calls += 1
            results.append({"role": "tool", "tool_call_id": call.get("id") or "",
                            "name": call.get("name") or "", "content": CHECKPOINT_RESULT})
        messages.extend(results)
        tail = results
    before = calls[0]
    after = calls[1] if len(calls) > 1 else None
    return {
        "family": task["family"], "band": task["band"], "seed": task["seed"],
        "arm": arm, "visible": visible,
        "correct": correct(task, final), "expected": task["expected"],
        "final": final[:80], "detail": task["detail"],
        "checkpoint_called": checkpoint_calls >= 1,
        "checkpoint_calls": checkpoint_calls,
        "answered_at_call": len(calls) if not (reply.get("tool_calls") or []) else 0,
        "leaked": leaked, "leaked_answer": leaked_answer,
        "model_calls": len(calls),
        "tool_calls": sum(c["tool_calls"] for c in calls),
        "reasoning_before": before["reasoning_tokens"],
        "reasoning_after": sum(c["reasoning_tokens"] for c in calls[1:]),
        "reasoning_total": sum(c["reasoning_tokens"] for c in calls),
        "prompt_before": before["prompt_tokens"],
        "prompt_after": after["prompt_tokens"] if after else 0,
        "prompt_total": sum(c["prompt_tokens"] for c in calls),
        "total_tokens": sum(c["total_tokens"] for c in calls),
        "wall_before": before["wall_seconds"],
        "wall_after": after["wall_seconds"] if after else 0.0,
        "wall_total": round(time.monotonic() - started, 3),
        "replay_tokens": after["replay_tokens_handed"] if after else 0,
        "replay_bytes": after["replay_bytes_handed"] if after else 0,
        "calls": calls,
    }


# -- summaries ------------------------------------------------------------

def _mean(rows, key):
    return statistics.mean(float(r.get(key) or 0) for r in rows) if rows else 0.0


def _sd(rows, key):
    return statistics.pstdev(float(r.get(key) or 0) for r in rows) if len(rows) > 1 else 0.0


def _cells(runs):
    return sorted({(r["family"], r["band"], bool(r.get("visible"))) for r in runs},
                  key=lambda c: (c[2], c[0], BANDS.index(c[1]) if c[1] in BANDS else 9))


def summarize(runs: List[dict]) -> str:
    lines = []
    lines.append("per cell (means; only runs that honoured the checkpoint and leaked no "
                 "answer count, the rest are listed as excluded)")
    lines.append("family   band    visible  arm         n  correct  reason_before  "
                 "reason_after  prompt_after  total_tokens  wall_before  wall_after  "
                 "replay_tokens  excluded")
    for family, band, visible in _cells(runs):
        for arm in ("transcript", "native"):
            rows = [r for r in runs if r["family"] == family and r["band"] == band
                    and bool(r.get("visible")) == visible and r["arm"] == arm]
            kept = [r for r in rows if r["checkpoint_called"] and not r["leaked"]]
            if not rows:
                continue
            good = sum(1 for r in kept if r["correct"])
            lines.append(
                f"{family:<8} {band:<7} {str(visible):<8} {arm:<10} {len(kept):>2}  "
                f"{good:>3}/{len(kept):<3}  {_mean(kept, 'reasoning_before'):>13.0f}  "
                f"{_mean(kept, 'reasoning_after'):>12.0f}  {_mean(kept, 'prompt_after'):>12.0f}  "
                f"{_mean(kept, 'total_tokens'):>12.0f}  {_mean(kept, 'wall_before'):>11.2f}  "
                f"{_mean(kept, 'wall_after'):>10.2f}  {_mean(kept, 'replay_tokens'):>13.0f}  "
                f"{len(rows) - len(kept):>8}"
            )
    lines.append("")
    lines.append("paired, native minus transcript on matched instances (same family, "
                 "band, seed, repeat): successor reasoning, successor prompt overhead, "
                 "and the correctness discordance")
    lines.append("family   band    visible  pairs  d_reason_after  overhead_prompt  ratio  "
                 "native_only_right  transcript_only_right")
    for family, band, visible in _cells(runs):
        pairs = matched_pairs(runs, family, band, visible)
        if not pairs:
            continue
        d_after = [n["reasoning_after"] - t["reasoning_after"] for n, t in pairs]
        overhead = [n["prompt_after"] - t["prompt_after"] for n, t in pairs]
        ratio = (-statistics.mean(d_after) / statistics.mean(overhead)
                 if statistics.mean(overhead) > 0 else float("nan"))
        n_only = sum(1 for n, t in pairs if n["correct"] and not t["correct"])
        t_only = sum(1 for n, t in pairs if t["correct"] and not n["correct"])
        lines.append(
            f"{family:<8} {band:<7} {str(visible):<8} {len(pairs):>5}  "
            f"{statistics.mean(d_after):>+14.0f}  {statistics.mean(overhead):>+15.0f}  "
            f"{ratio:>5.2f}  {n_only:>17}  {t_only:>21}"
        )
    return "\n".join(lines)


def matched_pairs(runs, family, band, visible):
    """(native, transcript) run pairs for one cell, both honouring the
    protocol, keyed by seed and repeat."""
    by_key: Dict[Tuple, Dict[str, dict]] = {}
    for r in runs:
        if (r["family"], r["band"], bool(r.get("visible"))) != (family, band, visible):
            continue
        if not r["checkpoint_called"] or r["leaked"]:
            continue
        by_key.setdefault((r["seed"], r.get("repeat", 0)), {})[r["arm"]] = r
    return [(pair["native"], pair["transcript"]) for pair in by_key.values()
            if "native" in pair and "transcript" in pair]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--backend", choices=("gemini", "openai"))
    parser.add_argument("--model")
    parser.add_argument("--families", default="order,arith,trace")
    parser.add_argument("--bands", default="light,medium,heavy")
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--visible", action="store_true",
                        help="the control: the checkpoint call carries the answer")
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--out")
    parser.add_argument("--summarize")
    parser.add_argument("--print-tasks", action="store_true",
                        help="print the generated tasks and answers, then exit")
    args = parser.parse_args()

    if args.summarize:
        with open(args.summarize, encoding="utf-8") as source:
            print(summarize([json.loads(line) for line in source if line.strip()]))
        return 0

    tasks = []
    for family in args.families.split(","):
        for band in args.bands.split(","):
            for seed in range(args.seeds):
                task = FAMILIES[family.strip()](band.strip(), seed)
                task.update({"family": family.strip(), "band": band.strip(), "seed": seed})
                tasks.append(task)
    if args.print_tasks:
        for task in tasks:
            print(json.dumps({k: v for k, v in task.items()}, ensure_ascii=False))
        return 0
    if not args.backend or not args.model:
        parser.error("--backend and --model are required to run")

    backend = build_backend(args.backend, args.model, args.reasoning_effort)
    profile = {"backend": args.backend, "model": args.model,
               "sdk": sdk_version(args.backend), "reasoning_effort": args.reasoning_effort,
               "visible": args.visible}
    print(json.dumps({"profile": profile}), flush=True)
    runs = []
    for repeat in range(args.repeat):
        for task in tasks:
            # Counterbalanced: which arm goes first alternates by seed and repeat.
            first = "native" if (task["seed"] + repeat) % 2 == 0 else "transcript"
            for position, arm in enumerate((first, "transcript" if first == "native" else "native")):
                run = run_task(backend, task, arm, args.visible)
                run.update({"repeat": repeat, "order_position": position, **profile})
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
