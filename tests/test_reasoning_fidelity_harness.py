"""The reasoning-fidelity harness's deterministic parts: the task
generators and their scoring, the protocol flags, and the paired
summary arithmetic. No network."""

from __future__ import annotations

import importlib.util
import itertools
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
