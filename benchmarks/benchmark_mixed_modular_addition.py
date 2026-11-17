# SPDX-License-Identifier: Apache-2.0
# Workloads adapted from the locked Kickmix gen_iadd_mod.perf.cc.

"""Benchmark the three locked mixed modular addition and subtraction workloads."""

import argparse
import json
from collections import Counter
from collections.abc import Callable
from functools import partial
from statistics import median
from time import perf_counter_ns

from flagquantum.compiler.artifacts.reversible_phase import ReversiblePhaseProgram
from flagquantum.compiler.registers.finite_field.coefficients import (
    ClassicalFieldCoefficient,
)
from flagquantum.compiler.registers.modular.mixed_modular_addition import (
    mixed_modular_addition_program,
)


def locked_modular_workloads() -> dict[str, Callable[[], ReversiblePhaseProgram]]:
    """Allocate source operand layouts before timing generator calls."""
    width = 256
    target = tuple(reversed(range(width)))
    offset = tuple(reversed(range(width, 2 * width)))
    clean = tuple(range(2 * width, 3 * width))
    mixed: tuple[bool | ClassicalFieldCoefficient, ...] = (
        True,
        *(ClassicalFieldCoefficient(i) for i in reversed(range(1, width - 1))),
        True,
    )
    prime = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
    constant = tuple(bool(prime & (1 << i)) for i in reversed(range(width)))
    return {
        "add": partial(
            mixed_modular_addition_program, target, offset, mixed, clean=clean
        ),
        "sub": partial(
            mixed_modular_addition_program,
            target,
            offset,
            mixed,
            clean=clean,
            subtract=True,
        ),
        "approx": partial(
            mixed_modular_addition_program, target, offset, constant, clean=clean
        ),
    }


def benchmark_modular_workloads(
    *, warmup: int = 1, repeats: int = 7
) -> dict[str, object]:
    """Time native compilation with exactly the source operand layouts."""
    if warmup < 0 or repeats < 1:
        raise ValueError("Warmup must be nonnegative and repeats must be positive")
    results: dict[str, object] = {}
    for name, generate in locked_modular_workloads().items():
        times = []
        for iteration in range(warmup + repeats):
            start = perf_counter_ns()
            program = generate()
            elapsed = perf_counter_ns() - start
            if iteration >= warmup:
                times.append(elapsed)
        results[name] = {
            "records": len(program.operations),
            "primitive_counts": dict(Counter(op.opcode for op in program.operations)),
            "median_seconds": median(times) / 1_000_000_000,
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    print(
        json.dumps(
            benchmark_modular_workloads(warmup=args.warmup, repeats=args.repeats),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
