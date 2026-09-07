"""The A/B harness's deterministic parts: the chain workload, its scoring,
the sanitized state measures, and the summary arithmetic. No network."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

HARNESS = Path(__file__).resolve().parent.parent / "scripts" / "ab_continuation.py"
spec = importlib.util.spec_from_file_location("ab_continuation", HARNESS)
ab = importlib.util.module_from_spec(spec)
sys.modules["ab_continuation"] = ab
spec.loader.exec_module(ab)


class TestTheChainWorkload:
    def test_every_record_names_the_next_and_the_last_ends_the_chain(self):
        rounds = 5
        for index in range(rounds - 1):
            text = ab.chain_tool(rounds, {"record_id": f"R{index}"})
            assert f"record R{index}: code {ab.chain_codes(rounds)[index]}." in text
            assert f"next record: R{index + 1}" in text
        assert "the chain ends" in ab.chain_tool(rounds, {"record_id": "R4"})
        assert "no such record" in ab.chain_tool(rounds, {"record_id": "R5"})
        assert "no such record" in ab.chain_tool(rounds, {"record_id": "seven"})
        # A bare number, and any argument name, are read the same way.
        assert ab.chain_tool(rounds, {"id": "2"}) == ab.chain_tool(rounds, {"record_id": "R2"})

    def test_the_answer_is_the_sum_of_every_code_and_differs_by_length(self):
        assert all(1 <= code <= 9 for code in ab.chain_codes(30))
        three, ten, twenty = (ab.Workload("chain", n) for n in (3, 10, 20))
        assert three.expected == str(sum(ab.chain_codes(3)))
        assert len({three.expected, ten.expected, twenty.expected}) == 3
        assert three.correct(f"The sum is {three.expected}.")
        assert three.correct(three.expected)
        assert not three.correct("R2 was the last record")
        assert not three.correct(f"{three.expected} and then 4")

    def test_the_kestrel_workload_is_the_original_three_lookups(self):
        kestrel = ab.Workload("kestrel", 3)
        assert kestrel.run_tool("parts_catalog", {"query": "Kestrel-9 relay"}).endswith(ab.PART)
        assert ab.FREQUENCY in kestrel.run_tool("spec_sheet", {"part_number": ab.PART})
        assert kestrel.correct(f"offset: {ab.KESTREL_EXPECTED} rad")
        assert not kestrel.correct("unknown")


class TestTheMeasuresAreCountsNotContents:
    def test_an_openai_tape_is_measured_by_type_count_and_size(self):
        payload = {"items": [
            {"type": "compaction", "id": "cmp_1", "encrypted_content": "SECRET-A"},
            {"type": "reasoning", "id": "rs_1", "encrypted_content": "SECRET-B"},
            {"type": "function_call", "id": "fc_1", "call_id": "c1", "name": "t",
             "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "x"},
        ]}
        measured = ab.state_measures(payload)
        assert measured["replay_items"] == 4
        assert measured["item_types"] == {"compaction": 1, "function_call": 1,
                                          "function_call_output": 1, "reasoning": 1}
        assert measured["compaction_ids"] == ["cmp_1"]
        assert measured["replay_bytes"] == len(json.dumps(payload, ensure_ascii=False).encode())
        assert "SECRET" not in json.dumps(measured)

    def test_a_gemini_conversation_is_measured_by_part_kind(self):
        payload = {"systemInstruction": {"parts": [{"text": "s"}]}, "contents": [
            {"role": "user", "parts": [{"text": "q"}]},
            {"role": "model", "parts": [
                {"text": "thinking", "thought": True, "thoughtSignature": "SIG-1"},
                {"functionCall": {"name": "t", "args": {}}, "thoughtSignature": "SIG-2"},
            ]},
            {"role": "user", "parts": [{"functionResponse": {"name": "t", "response": {}}}]},
        ]}
        measured = ab.state_measures(payload)
        assert measured["replay_items"] == 3
        assert measured["item_types"] == {"functionCall": 1, "functionResponse": 1,
                                          "signed": 2, "text": 1, "thought": 1}
        assert measured["compaction_ids"] == []
        assert "SIG" not in json.dumps(measured)

    def test_nothing_measures_to_nothing(self):
        assert ab.state_measures(None)["replay_items"] == 0
        assert ab.state_measures({})["replay_bytes"] == 0


def _run(workload, rounds, arm, *, total, reasoning, wall, correct, repeat=0, calls=()):
    return {
        "workload": workload, "rounds": rounds, "arm": arm, "repeat": repeat,
        "correct": correct, "model_calls": rounds + 1, "tool_calls": rounds,
        "prompt_tokens": total - reasoning, "reasoning_tokens": reasoning,
        "completion_tokens": 0, "cached_tokens": 0, "total_tokens": total,
        "wall_seconds": wall, "replay_bytes": 100, "compactions": 0, "calls": list(calls),
    }


class TestTheSummaryArithmetic:
    def test_native_minus_transcript_is_reported_per_length(self):
        runs = [
            _run("chain", 3, "transcript", total=1000, reasoning=200, wall=4.0, correct=True),
            _run("chain", 3, "transcript", total=1200, reasoning=300, wall=5.0, correct=True,
                 repeat=1),
            _run("chain", 3, "native", total=1500, reasoning=150, wall=3.0, correct=True),
            _run("chain", 3, "native", total=1700, reasoning=250, wall=4.0, correct=False,
                 repeat=1),
        ]
        text = ab.summarize(runs)
        assert "chain          3  transcript  2    2/2" in text
        assert "chain          3  native      2    1/2" in text
        # Means: total 1600 - 1100 = +500; reasoning 200 - 250 = -50; wall -1.00.
        assert "+500.0" in text and "-50.0" in text and "-1.00" in text
        assert "1/2 vs 2/2" in text

    def test_a_compacting_run_lists_the_calls_around_the_compaction(self):
        calls = [
            {"call": 1, "prompt_tokens": 100, "wall_seconds": 1.0, "replay_items_handed": 0,
             "replay_bytes_handed": 0, "replay_tokens_handed": 0},
            {"call": 2, "prompt_tokens": 9000, "wall_seconds": 2.0, "replay_items_handed": 4,
             "replay_bytes_handed": 8000, "replay_tokens_handed": 300, "compacted": True,
             "order_after_compaction": ["compaction", "reasoning", "function_call"]},
            {"call": 3, "prompt_tokens": 1200, "wall_seconds": 1.1, "replay_items_handed": 3,
             "replay_bytes_handed": 2000, "replay_tokens_handed": 40},
            {"call": 4, "prompt_tokens": 1400, "wall_seconds": 1.2, "replay_items_handed": 6,
             "replay_bytes_handed": 2600, "replay_tokens_handed": 70},
        ]
        run = _run("chain", 3, "native", total=1, reasoning=0, wall=1, correct=True,
                   calls=calls)
        run["compactions"] = 1
        text = ab.summarize([run])
        assert "compaction at call 2" in text
        assert "['compaction', 'reasoning', 'function_call']" in text
        assert "call  1:" in text and "call  4:" in text
        assert "<- compacting reply" in text
