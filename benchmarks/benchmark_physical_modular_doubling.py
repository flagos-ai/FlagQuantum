# SPDX-License-Identifier: Apache-2.0
# Workloads adapted from locked Kickmix gen_imul2.perf.cc.

"""Measure physical generation for two locked width-256 doubling workloads."""

import argparse
import json
from collections import Counter
from collections.abc import Callable
from statistics import median
from time import perf_counter_ns

from flagquantum.compiler._component_trace.trace import ComponentCall, ComponentTrace
from flagquantum.compiler.artifacts.reversible_phase import ReversiblePhaseProgram
from flagquantum.compiler.formats.kmx_block_marks import encode_kmx_block_marks
from flagquantum.compiler.registers.finite_field.coefficients import (
    ClassicalFieldCoefficient,
)
from flagquantum.compiler.registers.modular.modular_doubling_dispatch import (
    modular_doubling_program,
)


def locked_doubling_workloads() -> dict[
    str, Callable[[], tuple[ReversiblePhaseProgram, tuple[ComponentCall, ...]]]
]:
    """Build both source port layouts outside the timed region."""
    width = 256
    target = tuple(reversed(range(width)))
    clean = tuple(range(width + 1, width + 5))
    dirty = tuple(range(width + 5, 2 * width + 5))
    prime = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
    constant = tuple(bool(prime & (1 << bit)) for bit in reversed(range(width)))
    mixed: tuple[bool | ClassicalFieldCoefficient, ...] = (
        True,
        *(ClassicalFieldCoefficient(bit) for bit in reversed(range(width - 2))),
        True,
    )

    def generator(
        modulus: tuple[bool | ClassicalFieldCoefficient, ...],
    ) -> Callable[[], tuple[ReversiblePhaseProgram, tuple[ComponentCall, ...]]]:
        def generate() -> tuple[ReversiblePhaseProgram, tuple[ComponentCall, ...]]:
            trace = ComponentTrace()
            program = modular_doubling_program(
                target,
                modulus,
                clean=clean,
                dirty=dirty,
                control=width,
                btol=30,
                physical_trace=trace,
            )
            return program, tuple(trace.calls)

        return generate

    return {"const": generator(constant), "bits": generator(mixed)}


def benchmark_doubling_workloads(
    *, warmup: int = 1, repeats: int = 7
) -> dict[str, object]:
    """Time source-shaped physical generation including the component trace."""
    if warmup < 0 or repeats < 1:
        raise ValueError("Warmup must be nonnegative and repeats must be positive")
    results: dict[str, object] = {}
    for name, generate in locked_doubling_workloads().items():
        times = []
        for iteration in range(warmup + repeats):
            start = perf_counter_ns()
            program, calls = generate()
            elapsed = perf_counter_ns() - start
            if iteration >= warmup:
                times.append(elapsed)
        records = encode_kmx_block_marks(program, calls).splitlines()[1:]
        results[name] = {
            "records_including_marks": len(records),
            "operation_counts": dict(Counter(op.opcode for op in program.operations)),
            "median_seconds": median(times) / 1_000_000_000,
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    report = benchmark_doubling_workloads(warmup=args.warmup, repeats=args.repeats)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
