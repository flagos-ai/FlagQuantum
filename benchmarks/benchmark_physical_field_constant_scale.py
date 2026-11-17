# SPDX-License-Identifier: Apache-2.0
# Workloads adapted from the locked Kickmix gen_gf_imul_classical.perf.cc.

"""Measure native physical generation for two locked degree-256 constant scale calls."""

import argparse
import json
from collections import Counter
from statistics import median
from time import perf_counter_ns

from flagquantum.compiler.registers.finite_field.binary_linear_map import (
    physical_field_constant_scale,
)


def benchmark_field_constant_scale(
    *, warmup: int = 1, repeats: int = 7
) -> dict[str, object]:
    """Time map synthesis and physical lowering for the two source scaling directions."""
    if warmup < 0 or repeats < 1:
        raise ValueError("Warmup must be nonnegative and repeats must be positive")
    width = 256
    modulus = (1 << width) | 0x425
    target = tuple(reversed(range(width)))
    report: dict[str, object] = {}
    factor = 0xDEADBEEFCAFEF00D1234567890ABCDEF
    for divide in (False, True):
        times = []
        for iteration in range(warmup + repeats):
            start = perf_counter_ns()
            program, calls = physical_field_constant_scale(
                target, factor, modulus=modulus, divide=divide
            )
            elapsed = perf_counter_ns() - start
            if iteration >= warmup:
                times.append(elapsed)
        records = len(program.operations) + sum(
            2 * (1 + len(call.children)) for call in calls
        )
        seconds = median(times) / 1_000_000_000
        report["divide" if divide else "multiply"] = {
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
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    report = benchmark_field_constant_scale(warmup=args.warmup, repeats=args.repeats)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
