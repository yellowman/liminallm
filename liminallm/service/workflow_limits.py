"""Execution limits shared by the batch and streaming workflow paths.

Their own module so the two paths import one definition. A stream and a
non-stream run of the same graph must not disagree about how long it may take
or how much context it may carry.
"""

from __future__ import annotations

DEFAULT_WORKFLOW_TIMEOUT_MS = 60000  # 60 seconds total workflow timeout
MAX_CONTEXT_SNIPPETS = 20
