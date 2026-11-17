# SPDX-License-Identifier: Apache-2.0
# Workloads adapted from the locked Kickmix gen_gf_mul.perf.cc.

"""Measure native physical generation for the three locked width-256 products."""

import argparse
import json
from collections import Counter
from statistics import median
from time import perf_counter_ns

from flagquantum.compiler._component_trace.trace import ComponentCall
from flagquantum.compiler.registers.finite_field.polynomial_product import (
    physical_field_product,
    physical_polynomial_product,
)


def _mark_count(calls: tuple[ComponentCall, ...]) -> int:
    return sum(2 + _mark_count(call.children) for call in calls)


def benchmark_field_product(*, warmup: int = 1, repeats: int = 3) -> dict[str, object]:
    """Time instruction generation and physical lowering in the source layouts."""
    if warmup < 0 or repeats < 1:
        raise ValueError("Warmup must be nonnegative and repeats must be positive")
    width = 256
    modulus = (1 << width) | 0x425
    report: dict[str, object] = {}
    for workload in ("polynomial", "multiply", "erase"):
        target_width = 2 * width - 1 if workload == "polynomial" else width
        target = tuple(reversed(range(target_width)))
        left = tuple(reversed(range(target_width, target_width + width)))
        right = tuple(reversed(range(target_width + width, target_width + 2 * width)))
        times = []
        for iteration in range(warmup + repeats):
            start = perf_counter_ns()
            if workload == "polynomial":
                program, calls = physical_polynomial_product(left, right, target)
            else:
                program, calls = physical_field_product(
                    left, right, target, modulus=modulus, erase=workload == "erase"
                )
            elapsed = perf_counter_ns() - start
            if iteration >= warmup:
                times.append(elapsed)
        seconds = median(times) / 1_000_000_000
        records = len(program.operations) + _mark_count(calls)
        report[workload] = {
            "width": width,
            "records_including_marks": records,
            "primitive_counts": dict(Counter(op.opcode for op in program.operations)),
            "median_seconds": seconds,
            "records_per_second": records / seconds,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    report = benchmark_field_product(warmup=args.warmup, repeats=args.repeats)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
