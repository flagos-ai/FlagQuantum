# SPDX-License-Identifier: Apache-2.0
# Workloads adapted from locked Kickmix gen_iadd1.perf.cc.

"""Run the existing increment workloads with the original adaptive timing policy."""

import sys
from collections.abc import Sequence
from functools import partial

from flagquantum.benchmarking._adaptive_timing import TimingCase, TimingSample, measure
from flagquantum.benchmarking._timing_report import format_timing, select_benchmarks
from flagquantum.compiler.registers.arithmetic.increment import physical_increment
from flagquantum.ecosystem._command_arguments import (
    check_unknown_arguments,
    find_argument,
    float_argument,
)


def benchmark_options(argv: Sequence[str]) -> tuple[str | None, float]:
    check_unknown_arguments(
        ("--only", "--target_seconds"), (), "flagquantum-benchmark", None, argv
    )
    target = float_argument("--target_seconds", 0.5, 0, 10000, argv, precision=32)
    return find_argument("--only", argv), target


def run_increment(controlled: bool, target_seconds: float) -> TimingSample:
    width = 256
    first_clean = width + int(controlled)
    target = tuple(reversed(range(width)))
    clean = tuple(range(first_clean, first_clean + width))
    control = width if controlled else True

    def body() -> None:
        program, calls = physical_increment(
            target,
            clean=clean,
            control=control,
        )
        expected = 1274 if controlled else 1270
        if len(program.operations) + 2 * len(calls) != expected:
            raise RuntimeError("Increment workload instruction count changed")

    name = "increment_controlled" if controlled else "increment_unconditional"
    return measure(name, body, target_seconds=target_seconds)


def _increment_samples(controlled: bool, target: float) -> tuple[TimingSample, ...]:
    result = run_increment(controlled, target)
    # These are original C++ target times, separate from native measurements.
    result = result.with_goal(610 if controlled else 580, "ns").with_rate(
        "records", 1274 if controlled else 1270
    )
    return (result,)


def main() -> None:
    filters, target = benchmark_options(sys.argv)
    cases = (
        TimingCase("increment_unconditional", partial(_increment_samples, False)),
        TimingCase("increment_controlled", partial(_increment_samples, True)),
    )
    for case in select_benchmarks(cases, filters):
        for result in case.body(target):
            print(format_timing(case.name, result), end="")


if __name__ == "__main__":
    main()
