"""What the calibrator tells `observe()` the prompt cost.

`count_messages` existed and was tested; `_calibrate_from_usage` summed bare
`count()` calls instead, so the per-message wire overhead every chat format
adds was never in the estimate. `observe()` then saw estimated < actual and
pushed the character factor up to close the gap — correcting a per-message
cost with a per-character multiplier, which is only right at one history
length and drifts with every other.
"""

from __future__ import annotations

import pytest

from liminallm.service.token_counting import MESSAGE_OVERHEAD_TOKENS, counter_for
from liminallm.service.workflow import WorkflowEngine


@pytest.fixture
def counter():
    return counter_for("")


class _Recorder:
    """Stands in for the LLM: hands back a real counter, records the estimate."""

    def __init__(self, counter):
        self._counter = counter
        self.estimates = []

    def token_counter(self):
        return self._counter

    def observe_usage(self, estimated, usage):
        self.estimates.append(estimated)


@pytest.fixture
def engine(counter):
    eng = WorkflowEngine.__new__(WorkflowEngine)
    eng.llm = _Recorder(counter)
    from liminallm.logging import get_logger

    eng.logger = get_logger("test")
    return eng


def test_the_estimate_includes_per_message_overhead(engine, counter):
    engine._calibrate_from_usage("hello", [], [], {"prompt_tokens": 99})

    bare = counter.count("hello")
    assert engine.llm.estimates == [bare + MESSAGE_OVERHEAD_TOKENS]


def test_every_message_is_charged_its_overhead(engine, counter):
    """Three pieces, three messages — the old code counted one flat string's
    worth of nothing extra however long the history got."""
    engine._calibrate_from_usage(
        "prompt", ["snippet"], [{"content": "history"}], {"prompt_tokens": 99}
    )

    text = counter.count("prompt") + counter.count("snippet") + counter.count("history")
    assert engine.llm.estimates == [text + 3 * MESSAGE_OVERHEAD_TOKENS]


def test_the_gap_grows_with_history_which_is_why_a_factor_cannot_absorb_it(engine):
    """The point of the bug: under-counting scales with message count, so a
    per-character factor cannot correct it at more than one length."""
    engine._calibrate_from_usage("p", [], [], {"prompt_tokens": 1})
    short = engine.llm.estimates[-1]
    engine._calibrate_from_usage(
        "p", [], [{"content": "x"} for _ in range(20)], {"prompt_tokens": 1}
    )
    long = engine.llm.estimates[-1]

    assert long - short >= 20 * MESSAGE_OVERHEAD_TOKENS


def test_history_entries_may_be_objects_not_dicts(engine):
    class Message:
        content = "from an ORM row"

    engine._calibrate_from_usage("p", [], [Message()], {"prompt_tokens": 1})
    assert engine.llm.estimates[-1] > MESSAGE_OVERHEAD_TOKENS * 2


def test_an_empty_turn_still_reports_something(engine):
    engine._calibrate_from_usage("", [], [], {"prompt_tokens": 1})
    assert engine.llm.estimates == [MESSAGE_OVERHEAD_TOKENS]


def test_calibration_never_blocks_a_turn(engine):
    """It is optional, so a counter that throws must be swallowed."""

    class Broken:
        def token_counter(self):
            raise RuntimeError("no tokenizer")

        def observe_usage(self, *a):
            pass

    engine.llm = Broken()
    engine._calibrate_from_usage("p", [], [], {"prompt_tokens": 1})


def test_a_backend_without_observe_usage_is_skipped(engine):
    engine.llm = object()
    engine._calibrate_from_usage("p", [], [], {"prompt_tokens": 1})


def test_count_messages_handles_multimodal_parts(counter):
    """Images are priced by the provider and not estimable here; the text
    parts beside them still count."""
    total = counter.count_messages(
        [{"content": [{"text": "describe this"}, {"image_url": "..."}]}]
    )
    assert total == counter.count("describe this") + MESSAGE_OVERHEAD_TOKENS
