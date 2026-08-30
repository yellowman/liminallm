"""Execution limits shared by the batch and streaming workflow paths.

Their own module so the two paths import one definition. A stream and a
non-stream run of the same graph must not disagree about how long it may take,
how much context it may carry, or how much work it may begin.
"""

from __future__ import annotations

DEFAULT_WORKFLOW_TIMEOUT_MS = 60000  # 60 seconds total workflow timeout
MAX_CONTEXT_SNIPPETS = 20


class ExecutionBudget:
    """How many node executions one run may begin.

    A counter in the driving loop was not enough. A parallel child runs inside
    `_execute_parallel_nodes`, which that loop never sees: `visited` was
    incremented once for the `parallel` node and not at all for its children,
    and every child was launched together through `asyncio.gather`. Measured,
    a three-node graph whose `parallel.next` repeated one child id 150 times
    began 150 concurrent tool invocations against a budget of 16 and reported
    success.

    So the budget is an object the loop and the fan-out both hold, and a
    fan-out reserves before it starts - an over-budget batch never begins any
    of it, rather than being cut off partway through. Every entry in
    `parallel.next` costs one, a repeated id included: each occurrence is an
    execution, whatever it is named.
    """

    __slots__ = ("limit", "spent")

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.spent = 0

    def reserve(self, count: int = 1) -> bool:
        """Take `count` executions, all or nothing. False if the budget lacks
        them, and nothing is spent in that case."""
        if count <= 0:
            return True
        if self.spent + count > self.limit:
            return False
        self.spent += count
        return True
