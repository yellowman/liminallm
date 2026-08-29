#!/usr/bin/env python3
"""Purge the circuit-breaker failure history for one storage representation.

Run this during a coordinated breaker representation change (SPEC §18.3).
The breaker's failure history is ephemeral, per-window state, and its two
representations — the legacy plain counter at `circuit:*:failures` and the
rolling-window sorted set at `circuit:*:failures:v2` — are independent
ledgers that must not run side by side. A representation change is therefore
a coordinated reset, not a rolling deploy:

    1. stop admitting traffic to the replicas on the old representation;
    2. wait for their in-flight requests to drain;
    3. run this script to purge the SUPERSEDED representation's failure
       history (the one you are leaving);
    4. start the replicas on the new representation;
    5. never overlap the two representations.

Purging — rather than letting the old keys expire on their TTL — is what
makes the reset survive a rollback: a rollback inside the ≤60s TTL would
otherwise find the old counter still live and resume counting from it. A
rollback repeats the same drain/purge in reverse (purge `failures:v2`).

The shared `circuit:*:open` cooldown keys are NOT touched: an already-open
breaker keeps its cooldown across the reset, which is deliberate.

Usage:
    # Cutover to the rolling window: purge the legacy counter.
    python scripts/reset_breaker_history.py --representation legacy

    # Rollback to the legacy counter: purge the rolling window.
    python scripts/reset_breaker_history.py --representation v2

    # See what would be removed without deleting anything.
    python scripts/reset_breaker_history.py --representation legacy --dry-run

Environment Variables:
    DATABASE_URL: required by the runtime that provides the Redis client.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Add project root to path for imports
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# The valid representation names come from RedisCache — one source of truth, so
# the runbook and this script never name a raw Redis glob and cannot drift from
# the storage layer. Importing the class loads no config.
from liminallm.storage.redis_cache import RedisCache  # noqa: E402


async def reset_breaker_history(representation: str, dry_run: bool) -> int:
    # Import here so config loads only after env vars are set.
    from liminallm.service.runtime import get_runtime

    runtime = get_runtime()
    cache = runtime.cache
    if cache is None:
        # Redis is the breaker's store; a purge command has nothing to do
        # without it, and this must fail loudly rather than no-op silently.
        print("Error: Redis is not configured; there is no breaker history to purge")
        sys.exit(1)

    # The primitive validates `representation` against RedisCache's map and
    # refuses anything else, so a raw glob can never reach the purge.
    removed = await cache.purge_breaker_failure_history(representation, dry_run=dry_run)
    verb = "would remove" if dry_run else "removed"
    print(f"{verb} {removed} breaker failure-history key(s) for '{representation}'")
    return removed


def main():
    parser = argparse.ArgumentParser(
        description="Purge circuit-breaker failure history for one representation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--representation",
        required=True,
        choices=sorted(RedisCache.FAILURE_REPRESENTATIONS),
        help="which representation's failure history to purge (the one being left)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="count the keys that would be removed without deleting them",
    )
    args = parser.parse_args()

    if not os.environ.get("DATABASE_URL"):
        print("Error: DATABASE_URL is required; the runtime provides the Redis client")
        sys.exit(1)
    # Deliberately NOT setting TEST_MODE: this talks to the real Redis the
    # fleet uses. TEST_MODE only gates the Redis-absent fallback, and a purge
    # command must never take it — if Redis is down, failing loudly is right.

    try:
        asyncio.run(reset_breaker_history(args.representation, args.dry_run))
    except Exception as e:  # noqa: BLE001 - operator-facing tool, one clean line
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
