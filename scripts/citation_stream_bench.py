"""Wall time for the streamed citation scrub, by workload and chunking.

The gate on this reader is the deterministic work count in
`tests/test_citation_stream_linear.py`: characters of work per character
received, independent of how the provider cut the answer. This script is the
corroborating evidence - what that costs in seconds on one machine - and the
way to re-measure after a change.

    python scripts/citation_stream_bench.py
    python scripts/citation_stream_bench.py --sizes 2000,16384 --repeats 5

Delivery and completion are timed apart on purpose. Delivery is the
incremental half, on the producer thread while the answer streams;
completion is the one whole-string `scrub_positions` pass, which is where
the finished text and the origin map come from and which this reader did not
change.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liminallm.service.citation_stream import CanonicalCitationStream  # noqa: E402

NONCE = "K7Q2ABCD"

#: Each workload is a shape that costs a reader something different: ordinary
#: text that settles as it arrives, an answer full of markers to remove, a
#: held run that cannot be released until the answer ends, and removals whose
#: junctions make the next occurrence.
WORKLOADS = {
    "prose": lambda n: ("The service interval is four hundred hours. "
                        * (n // 44 + 1))[:n],
    "markers": lambda n: (f"ok [cite:{NONCE}-12] " * (n // 21 + 1))[:n],
    "held tail": lambda n: ("answer" + " " * n)[:n],
    "splices": lambda n: (("K7Q2" + NONCE.lower() + "ABCD")
                          * (n // 16 + 1))[:n],
}


def measure(text: str, chunk: int, repeats: int) -> tuple:
    """Median delivery and completion seconds for one text and chunking."""
    delivery, completion = [], []
    for _ in range(repeats):
        chunks = (
            [text[i:i + chunk] for i in range(0, len(text), chunk)]
            if chunk else [text]
        )
        reader = CanonicalCitationStream(NONCE)
        start = time.perf_counter()
        for piece in chunks:
            reader.push(piece)
        middle = time.perf_counter()
        reader.finish()
        end = time.perf_counter()
        delivery.append(middle - start)
        completion.append(end - middle)
    return statistics.median(delivery), statistics.median(completion)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="2000,4000,8000,16384")
    parser.add_argument("--chunks", default="1,4,0",
                        help="characters per chunk; 0 means the whole answer")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--workloads", default=",".join(WORKLOADS))
    args = parser.parse_args()

    sizes = [int(value) for value in args.sizes.split(",")]
    chunks = [int(value) for value in args.chunks.split(",")]
    names = [name for name in args.workloads.split(",") if name in WORKLOADS]

    # Warm the process before the first measurement: imports, the compiled
    # patterns, and whatever the interpreter caches on a first pass.
    for _ in range(3):
        measure(WORKLOADS["prose"](500), 1, 1)

    print(f"{'workload':10} {'chunk':>6} {'N':>6} {'deliver s':>10} "
          f"{'finish s':>9} {'total s':>9} {'chars/s':>10} {'x2 ratio':>9}")
    for name in names:
        build = WORKLOADS[name]
        for chunk in chunks:
            previous = None
            for size in sizes:
                deliver, finish = measure(build(size), chunk, args.repeats)
                ratio = deliver / previous if previous else float("nan")
                previous = deliver
                print(
                    f"{name:10} {chunk if chunk else 'whole':>6} {size:6} "
                    f"{deliver:10.4f} {finish:9.4f} {deliver + finish:9.4f} "
                    f"{size / deliver if deliver else 0:10.0f} {ratio:9.2f}"
                )
            print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
