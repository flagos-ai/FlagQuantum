# SPDX-License-Identifier: Apache-2.0
# Workloads adapted from the locked Kickmix gen_cmp.perf.cc.

"""Measure native physical generation for both locked 256-bit comparators."""

import argparse
import json
from collections import Counter
from statistics import median
from time import perf_counter_ns

from flagquantum.compiler.formats.kmx_block_marks import encode_kmx_block_marks
from flagquantum.compiler.registers.comparison.physical_comparison import (
    physical_comparison,
)
from flagquantum.compiler.registers.finite_field.coefficients import (
    ClassicalFieldCoefficient,
)


def benchmark_comparison(*, warmup: int = 3, repeats: int = 25) -> dict[str, object]:
    """Time native construction and lowering in the two source layouts."""
    if warmup < 0 or repeats < 1:
        raise ValueError("Warmup must be nonnegative and repeats must be positive")
    width = 256
    left = tuple(reversed(range(width)))
    right = tuple(ClassicalFieldCoefficient(bit) for bit in reversed(range(width)))
    report: dict[str, object] = {}
    for clean in (width, 3):
        workspace = tuple(range(width + 1, width + 1 + clean))
        times = []
        for iteration in range(warmup + repeats):
            start = perf_counter_ns()
            program, calls = physical_comparison(left, right, width, workspace)
            elapsed = perf_counter_ns() - start
            if iteration >= warmup:
                times.append(elapsed)
        counts = Counter(op.opcode for op in program.operations)
        records = len(encode_kmx_block_marks(program, calls).splitlines()) - 1
        seconds = median(times) / 1_000_000_000
        report[str(clean)] = {
            "width": width,
            "records_including_marks": records,
            "primitive_counts": dict(counts),
            "median_seconds": seconds,
            "records_per_second": records / seconds,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=25)
    args = parser.parse_args()
    print(
        json.dumps(
            benchmark_comparison(warmup=args.warmup, repeats=args.repeats), indent=2
        )
    )


if __name__ == "__main__":
    main()
