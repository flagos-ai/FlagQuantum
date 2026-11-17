# SPDX-License-Identifier: Apache-2.0
# Workload adapted from the locked Kickmix gen_iadd_control.perf.cc.

"""Measure native generation of the locked 256-bit carry-controlled workload."""

import argparse
import json
from collections import Counter
from statistics import median
from time import perf_counter_ns

from flagquantum.compiler.registers.arithmetic.carry_controlled_addition import (
    physical_carry_controlled_addition,
)


def benchmark_carry_controlled_addition(
    *, warmup: int = 3, repeats: int = 25
) -> dict[str, object]:
    """Time instruction construction and physical lowering in the source layout."""
    if warmup < 0 or repeats < 1:
        raise ValueError("Warmup must be nonnegative and repeats must be positive")
    width = 256
    target = tuple(reversed(range(width)))
    offset = tuple(reversed(range(width, 2 * width)))
    carry, enable = 2 * width, 2 * width + 1
    clean = tuple(range(enable + 1, enable + 1 + width))
    times = []
    for iteration in range(warmup + repeats):
        start = perf_counter_ns()
        program, calls = physical_carry_controlled_addition(
            target, offset, control=enable, carry_in=carry, clean=clean
        )
        elapsed = perf_counter_ns() - start
        if iteration >= warmup:
            times.append(elapsed)
    counts = Counter(op.opcode for op in program.operations)
    records = len(program.operations) + 2 * len(calls)
    seconds = median(times) / 1_000_000_000
    return {
        "width": width,
        "records_including_marks": records,
        "primitive_counts": dict(counts),
        "median_seconds": seconds,
        "records_per_second": records / seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=25)
    args = parser.parse_args()
    report = benchmark_carry_controlled_addition(
        warmup=args.warmup, repeats=args.repeats
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
