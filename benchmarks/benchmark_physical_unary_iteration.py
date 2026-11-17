# SPDX-License-Identifier: Apache-2.0
# Workload adapted from the locked Kickmix gen_unary_iteration.perf.cc.

"""Measure the locked ten-bit controlled sweep on the native cursor."""

import argparse
import json
from collections import Counter
from statistics import median
from time import perf_counter_ns

from flagquantum.compiler.registers.lookup.recursive_unary_iteration import (
    physical_unary_iteration,
)


def benchmark_unary_iteration(
    *, warmup: int = 1, repeats: int = 7
) -> dict[str, object]:
    """Time generation of all 1,024 visits and the final close."""
    if warmup < 0 or repeats < 1:
        raise ValueError("Warmup must be nonnegative and repeats must be positive")
    address = tuple(reversed(range(10)))
    workspace = tuple(range(11, 21))
    times = []
    for iteration in range(warmup + repeats):
        start = perf_counter_ns()
        program, calls = physical_unary_iteration(
            address, workspace, range(1024), lambda value, match: (), control=10
        )
        elapsed = perf_counter_ns() - start
        if iteration >= warmup:
            times.append(elapsed)
    records = len(program.operations) + 2 * len(calls)
    seconds = median(times) / 1_000_000_000
    return {
        "width": len(address),
        "visits": 1024,
        "workspace": len(workspace),
        "records_including_marks": records,
        "primitive_counts": dict(Counter(op.opcode for op in program.operations)),
        "median_seconds": seconds,
        "records_per_second": records / seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    print(
        json.dumps(
            benchmark_unary_iteration(warmup=args.warmup, repeats=args.repeats),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
