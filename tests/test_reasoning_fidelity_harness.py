"""The reasoning-fidelity harness's deterministic parts: the task
generators and their scoring, the protocol flags, and the paired
summary arithmetic. No network."""

from __future__ import annotations

import importlib.util
import itertools
import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("reasoning_fidelity", SCRIPTS / "reasoning_fidelity.py")
rf = importlib.util.module_from_spec(spec)
sys.modules["reasoning_fidelity"] = rf
spec.loader.exec_module(rf)


class TestTheTaskGenerators:
    def test_every_family_and_band_is_reproducible_from_its_seed(self):
        for family, make in rf.FAMILIES.items():
            for band in rf.BANDS:
                first, second = make(band, 3), make(band, 3)
                assert first == second, (family, band)
                assert make(band, 4) != first or family == "trivial"

    def test_an_order_puzzle_has_exactly_one_solution_and_says_so(self):
        for band, size in rf.ORDER_SIZE.items():
            task = rf.make_order(band, 1)
            names = [line.split(":")[1] for line in task["prompt"].splitlines()
                     if line.startswith(f"{size} runners")][0]
            runners = tuple(n.strip() for n in names.split(".")[0].split(","))
            assert len(runners) == size
            expected = tuple(task["expected"].split(", "))
            clues = [line[2:] for line in task["prompt"].splitlines() if line.startswith("- ")]
            assert len(clues) == task["detail"]["clues"]
            # Re-derive the clue tuples from their texts and solve from scratch.
            solutions = [order for order in itertools.permutations(runners)
                         if all(_holds(text, order) for text in clues)]
            assert solutions == [expected]

    def test_the_arithmetic_chain_and_the_trace_are_scored_by_recomputation(self):
        arith = rf.make_arith("heavy", 0)
        assert arith["expected"] == "50"  # worked by hand: 23 ... mod 127
        trace = rf.make_trace("medium", 0)
        assert trace["expected"] == "17"  # worked by hand: x=23, y=8, six passes
        assert rf.make_trivial("light", 0)["kind"] == "integer"

    def test_bands_grow_the_work(self):
        assert rf.ORDER_SIZE["light"] < rf.ORDER_SIZE["medium"] < rf.ORDER_SIZE["heavy"]
        assert rf.ARITH_STEPS["light"] < rf.ARITH_STEPS["medium"] < rf.ARITH_STEPS["heavy"]
        assert (rf.TRACE_ITERATIONS["light"] < rf.TRACE_ITERATIONS["medium"]
                < rf.TRACE_ITERATIONS["heavy"])


def _holds(text, order):
    """A clue's text, read back as the predicate it states."""
    position = {name: index for index, name in enumerate(order)}
    import re

    if m := re.fullmatch(r"(\w+) finished before (\w+)\.", text):
        return position[m[1]] < position[m[2]]
    if m := re.fullmatch(r"(\w+) finished immediately after (\w+)\.", text):
        return position[m[1]] - position[m[2]] == 1
    if m := re.fullmatch(r"Exactly one runner finished between (\w+) and (\w+)\.", text):
        return abs(position[m[1]] - position[m[2]]) == 2
    if m := re.fullmatch(r"(\w+) did not finish in position (\d+)\.", text):
        return position[m[1]] != int(m[2]) - 1
    if m := re.fullmatch(r"(\w+) finished in position (\d+)\.", text):
        return position[m[1]] == int(m[2]) - 1
    raise AssertionError(f"unknown clue text {text!r}")


class TestScoring:
    def test_integers_are_read_as_the_last_number_and_orders_as_the_names(self):
        task = {"kind": "integer", "expected": "-62"}
        assert rf.correct(task, "-62")
        assert rf.correct(task, "The final number is -62.")
        assert not rf.correct(task, "62")
        assert not rf.correct(task, "-62 or maybe 5")
        order = {"kind": "order", "expected": "Ada, Bo, Cy"}
        assert rf.correct(order, "ada, bo, cy")
        assert rf.correct(order, "Ada Bo Cy")
        assert not rf.correct(order, "Bo, Ada, Cy")


class _Scripted:
    """A backend whose replies are scripted per call, recording what each
    call was handed: the messages, and the continuation if any."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def generate_with_tools(self, messages, tools, adapters, *, continuation=None, **_kw):
        self.calls.append({"messages": [dict(m) for m in messages],
                           "continuation": continuation})
        reply = self.replies.pop(0)
        return reply(messages, continuation) if callable(reply) else reply


def _checkpoint_reply(reasoning=500, content="", arguments='{"stage": 1}'):
    call = {"id": "c1", "name": "checkpoint", "arguments": arguments}
    return {
        "content": content, "tool_calls": [call],
        "assistant_message": {"role": "assistant", "content": content or None,
                              "tool_calls": [{"id": "c1", "type": "function",
                                              "function": {"name": "checkpoint",
                                                           "arguments": arguments}}]},
        "usage": {"prompt_tokens": 100, "completion_tokens": 10,
                  "reasoning_tokens": reasoning, "total_tokens": 110 + reasoning},
        "continuation": {"strategy": "gemini.native.v1", "provider": "gemini",
                         "transport": "generateContent", "model": "m",
                         "payload": {"systemInstruction": None,
                                     "contents": [{"role": "model", "parts": [
                                         {"functionCall": {"name": "checkpoint", "args": {}},
                                          "thoughtSignature": "SIG"}]}]},
                         "replay_tokens": reasoning},
    }


def _answer(text, reasoning=0, prompt=300):
    return {"content": text, "tool_calls": [], "assistant_message": {"role": "assistant",
                                                                     "content": text},
            "usage": {"prompt_tokens": prompt, "completion_tokens": 5,
                      "reasoning_tokens": reasoning, "total_tokens": prompt + 5 + reasoning},
            "continuation": {"strategy": "gemini.native.v1", "provider": "gemini",
                             "transport": "generateContent", "model": "m",
                             "payload": {"contents": []}, "replay_tokens": reasoning}}


class TestTheForkedDesign:
    def test_one_first_call_feeds_both_successors_and_only_the_representation_differs(self):
        task = {**rf.make_arith("light", 0), "family": "arith", "band": "light", "seed": 0}
        expected = task["expected"]
        backend = _Scripted([
            _checkpoint_reply(reasoning=640),
            _answer(expected, reasoning=0, prompt=900),     # native successor, runs first
            _answer("wrong 1", reasoning=0, prompt=260),    # transcript successor
        ])

        record = rf.run_fork(backend, task, visible=False, first_arm="native")

        assert not record["excluded"]
        assert len(backend.calls) == 3, "the first call is made once"
        assert record["first"]["reasoning_tokens"] == 640
        assert record["first"]["argument_keys"] == ["stage"]
        native, transcript = backend.calls[1], backend.calls[2]
        # The native successor: the continuation the first reply produced,
        # and only the tool result.
        assert native["continuation"] is not None
        assert native["continuation"].replay_tokens == 640
        assert [m["role"] for m in native["messages"]] == ["tool"]
        assert native["messages"][0]["content"] == "continue"
        # The transcript successor: the visible history with the same
        # call and the same tool result, and no continuation.
        assert transcript["continuation"] is None
        assert [m["role"] for m in transcript["messages"]] == ["system", "user", "assistant", "tool"]
        assert transcript["messages"][2]["tool_calls"][0]["function"]["name"] == "checkpoint"
        assert transcript["messages"][3] == native["messages"][0]
        outcomes = record["successors"]
        assert outcomes["native"]["correct"] and not outcomes["transcript"]["correct"]
        assert outcomes["native"]["order_position"] == 0
        assert outcomes["transcript"]["order_position"] == 1
        assert outcomes["native"]["replay_tokens"] == 640
        assert outcomes["native"]["prompt_after"] == 900
        assert outcomes["transcript"]["prompt_after"] == 260

    def test_the_other_order_runs_the_transcript_successor_first(self):
        task = {**rf.make_trivial("light", 1), "family": "trivial", "band": "light", "seed": 1}
        backend = _Scripted([
            _checkpoint_reply(reasoning=50),
            _answer(task["expected"]), _answer(task["expected"]),
        ])
        record = rf.run_fork(backend, task, visible=False, first_arm="transcript")
        assert backend.calls[1]["continuation"] is None
        assert backend.calls[2]["continuation"] is not None
        assert record["successors"]["transcript"]["order_position"] == 0

    def test_a_first_reply_that_breaks_the_protocol_gets_no_successors(self):
        task = {**rf.make_arith("light", 2), "family": "arith", "band": "light", "seed": 2}
        leaking = _checkpoint_reply(content=f"the answer is {task['expected']}")
        record = rf.run_fork(_Scripted([leaking]), task, visible=False, first_arm="native")
        assert record["excluded"] and record["reason"] == "answer text in the first turn"
        assert record["successors"] == {}

        in_arguments = _checkpoint_reply(arguments=json.dumps({"stage": 1, "result": task["expected"]}))
        record = rf.run_fork(_Scripted([in_arguments]), task, visible=False, first_arm="native")
        assert record["excluded"] and record["reason"] == "answer state in the call's arguments"

        # The same arguments are the protocol under the visible-state control.
        visible = _checkpoint_reply(arguments=json.dumps({"stage": 1, "result": task["expected"]}))
        backend = _Scripted([visible, _answer(task["expected"]), _answer(task["expected"])])
        record = rf.run_fork(backend, task, visible=True, first_arm="native")
        assert not record["excluded"] and record["first"]["argument_keys"] == ["result", "stage"]

    def test_flattening_gives_each_arm_a_row_that_shares_the_first_call(self):
        task = {**rf.make_trace("light", 0), "family": "trace", "band": "light", "seed": 0}
        backend = _Scripted([
            _checkpoint_reply(reasoning=700),
            _answer(task["expected"], prompt=1000), _answer("nope", prompt=300),
        ])
        record = rf.run_fork(backend, task, visible=False, first_arm="native")
        record["repeat"] = 1
        rows = rf.flatten([record])
        assert {r["arm"] for r in rows} == {"native", "transcript"}
        assert all(r["reasoning_before"] == 700 and r["design"] == "forked" for r in rows)
        pairs = rf.matched_pairs([record], "trace", "light", False)
        assert len(pairs) == 1 and pairs[0][0]["prompt_after"] - pairs[0][1]["prompt_after"] == 700
        text = rf.summarize([record])
        assert "== forked design" in text and "1                      0" in text


def _run(family, band, seed, arm, *, correct, before, after, prompt_after, repeat=0,
         checkpoint=True, leaked=False, visible=False):
    return {"family": family, "band": band, "seed": seed, "arm": arm, "repeat": repeat,
            "visible": visible, "correct": correct, "checkpoint_called": checkpoint,
            "leaked": leaked, "reasoning_before": before, "reasoning_after": after,
            "prompt_after": prompt_after, "total_tokens": 0, "wall_before": 1.0,
            "wall_after": 1.0, "replay_tokens": before if arm == "native" else 0}


class TestThePairedSummary:
    def test_pairs_match_by_seed_and_repeat_and_skip_protocol_breaks(self):
        runs = [
            _run("arith", "heavy", 0, "native", correct=True, before=900, after=100,
                 prompt_after=1400),
            _run("arith", "heavy", 0, "transcript", correct=False, before=850, after=700,
                 prompt_after=500),
            _run("arith", "heavy", 1, "native", correct=True, before=800, after=90,
                 prompt_after=1300, leaked=True),
            _run("arith", "heavy", 1, "transcript", correct=True, before=820, after=650,
                 prompt_after=500),
        ]
        pairs = rf.matched_pairs(runs, "arith", "heavy", False)
        assert len(pairs) == 1 and pairs[0][0]["arm"] == "native"
        text = rf.summarize(runs)
        # Native saved 600 successor reasoning tokens for 900 of overhead: ratio 0.67.
        assert "-600" in text and "+900" in text and "0.67" in text
        assert "excluded" in text

    def test_discordant_correctness_is_counted_both_ways(self):
        runs = [
            _run("order", "heavy", s, "native", correct=(s != 2), before=500, after=50,
                 prompt_after=900) for s in range(3)
        ] + [
            _run("order", "heavy", s, "transcript", correct=(s == 2), before=500, after=400,
                 prompt_after=400) for s in range(3)
        ]
        pairs = rf.matched_pairs(runs, "order", "heavy", False)
        assert len(pairs) == 3
        lines = [line for line in rf.summarize(runs).splitlines()
                 if line.startswith("order") and "pairs" not in line and "True" not in line]
        paired = [line for line in lines if line.split()[3].isdigit()]
        assert paired, lines
        assert paired[-1].split()[-2:] == ["2", "1"]
