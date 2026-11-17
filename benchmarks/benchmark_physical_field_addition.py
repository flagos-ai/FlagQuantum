# SPDX-License-Identifier: Apache-2.0
# Workloads adapted from the locked Kickmix gen_gf_iadd.perf.cc.

"""Measure native physical generation for the two locked width-512 additions."""

import argparse
import json
from collections import Counter
from statistics import median
from time import perf_counter_ns

from flagquantum.compiler.registers.finite_field.physical_field_addition import (
    physical_field_addition,
)


def benchmark_field_addition(
    *, warmup: int = 3, repeats: int = 25
) -> dict[str, object]:
    """Time construction and lowering for controlled and uncontrolled addition."""
    if warmup < 0 or repeats < 1:
        raise ValueError("Warmup must be nonnegative and repeats must be positive")
    width = 512
    target = tuple(reversed(range(width)))
    source = tuple(reversed(range(width, 2 * width)))
    report: dict[str, object] = {}
    for controlled in (False, True):
        times = []
        for iteration in range(warmup + repeats):
            start = perf_counter_ns()
            program, calls = physical_field_addition(
                source, target, control=2 * width if controlled else None
            )
            elapsed = perf_counter_ns() - start
            if iteration >= warmup:
                times.append(elapsed)
        records = len(program.operations) + 2 * len(calls)
        seconds = median(times) / 1_000_000_000
        report["controlled" if controlled else "uncontrolled"] = {
            "width": width,
            "records_including_marks": records,
            "primitive_counts": dict(Counter(op.opcode for op in program.operations)),
            "median_seconds": seconds,
            "records_per_second": records / seconds,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=25)
    args = parser.parse_args()
    report = benchmark_field_addition(warmup=args.warmup, repeats=args.repeats)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
