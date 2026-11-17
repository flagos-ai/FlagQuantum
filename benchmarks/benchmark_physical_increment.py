# SPDX-License-Identifier: Apache-2.0
# Workloads adapted from the locked Kickmix gen_iadd1.perf.cc.

"""Measure native generation of the two locked 256-bit increment workloads."""

import argparse
import json
from collections import Counter
from statistics import median
from time import perf_counter_ns

from flagquantum.compiler.registers.arithmetic.increment import physical_increment


def benchmark_increment(*, warmup: int = 3, repeats: int = 25) -> dict[str, object]:
    """Time instruction generation and physical lowering, then check counts."""
    if warmup < 0 or repeats < 1:
        raise ValueError("Warmup must be nonnegative and repeats must be positive")
    width = 256
    target = tuple(reversed(range(width)))
    results: dict[str, object] = {}
    for controlled, expected_records in ((False, 1270), (True, 1274)):
        control: int | bool = width if controlled else True
        first_clean = width + int(controlled)
        clean = tuple(range(first_clean, first_clean + width))
        times = []
        for iteration in range(warmup + repeats):
            start = perf_counter_ns()
            program, calls = physical_increment(target, clean=clean, control=control)
            elapsed = perf_counter_ns() - start
            counts = Counter(op.opcode for op in program.operations)
            records = len(program.operations) + 2 * len(calls)
            if records != expected_records or not counts["ccx"]:
                raise RuntimeError(f"Unexpected increment workload size {records}")
            if iteration >= warmup:
                times.append(elapsed)
        elapsed_seconds = median(times) / 1_000_000_000
        name = "controlled" if controlled else "unconditional"
        results[name] = {
            "width": width,
            "records_including_marks": records,
            "primitive_counts": dict(counts),
            "median_seconds": elapsed_seconds,
            "records_per_second": records / elapsed_seconds,
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=25)
    args = parser.parse_args()
    print(
        json.dumps(
            benchmark_increment(warmup=args.warmup, repeats=args.repeats), indent=2
        )
    )


if __name__ == "__main__":
    main()
