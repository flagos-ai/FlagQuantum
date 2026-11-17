# SPDX-License-Identifier: Apache-2.0
# Workloads adapted from locked Kickmix gen_lookup.perf.cc.

"""Measure physical generation for the locked classical-table lookup workloads."""

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from statistics import median
from time import perf_counter_ns

from flagquantum.compiler._component_trace.trace import ComponentCall, ComponentTrace
from flagquantum.compiler.artifacts.reversible_phase import ReversiblePhaseProgram
from flagquantum.compiler.registers.finite_field.coefficients import (
    ClassicalFieldCoefficient,
)
from flagquantum.compiler.registers.lookup.measured_unlookup import (
    measured_unlookup_operations,
)
from flagquantum.compiler.registers.lookup.recursive_lookup import (
    recursive_lookup_operations,
)


@dataclass(frozen=True)
class LookupWorkload:
    """Ports and slots preallocated by both locked source benchmarks."""

    address: tuple[int, ...]
    output: tuple[int, ...]
    workspace: tuple[int, ...]
    table: tuple[tuple[ClassicalFieldCoefficient, ...], ...]
    table_inputs: tuple[int, ...]
    phase_bits: tuple[int, ...]

    def generate(
        self, *, erase: bool
    ) -> tuple[ReversiblePhaseProgram, tuple[ComponentCall, ...]]:
        trace = ComponentTrace()
        if erase:
            result_bit = len(self.table_inputs) + len(self.phase_bits)
            operations = measured_unlookup_operations(
                self.table,
                self.address,
                self.output,
                self.workspace,
                self.phase_bits,
                measurement_bit=result_bit,
                trace=trace,
            )
        else:
            result_bit = len(self.table_inputs)
            operations = recursive_lookup_operations(
                self.table,
                self.address,
                self.output,
                self.workspace,
                measurement_bit=result_bit,
                trace=trace,
            )
        program = ReversiblePhaseProgram(
            len(self.address) + len(self.output) + len(self.workspace),
            result_bit + 1,
            operations,
            self.table_inputs,
        )
        return program, tuple(trace.calls)


def locked_lookup_workload() -> LookupWorkload:
    """Build the source layout with an 8-bit address and 30-bit output."""
    address_width, output_width = 8, 30
    rows = 1 << address_width
    inputs = tuple(range(rows * output_width))
    return LookupWorkload(
        tuple(reversed(range(address_width))),
        tuple(reversed(range(address_width, address_width + output_width))),
        tuple(range(address_width + output_width, address_width + output_width + 10)),
        tuple(
            tuple(
                ClassicalFieldCoefficient(bit)
                for bit in reversed(
                    inputs[row * output_width : (row + 1) * output_width]
                )
            )
            for row in range(rows)
        ),
        inputs,
        tuple(range(len(inputs), len(inputs) + rows)),
    )


def _count_marks(calls: tuple[ComponentCall, ...]) -> int:
    return sum(2 + _count_marks(call.children) for call in calls)


def benchmark_lookup(*, warmup: int = 1, repeats: int = 7) -> dict[str, object]:
    """Time physical generation with the input layout built outside the timer."""
    if warmup < 0 or repeats < 1:
        raise ValueError("Warmup must be nonnegative and repeats must be positive")
    workload = locked_lookup_workload()
    results: dict[str, object] = {}
    for name, erase in (("lookup", False), ("erase", True)):
        times = []
        for iteration in range(warmup + repeats):
            start = perf_counter_ns()
            program, calls = workload.generate(erase=erase)
            elapsed = perf_counter_ns() - start
            if iteration >= warmup:
                times.append(elapsed)
        records = len(program.operations) + _count_marks(calls)
        seconds = median(times) / 1_000_000_000
        results[name] = {
            "records_including_marks": records,
            "primitive_counts": dict(Counter(op.opcode for op in program.operations)),
            "median_seconds": seconds,
            "records_per_second": records / seconds,
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    print(
        json.dumps(benchmark_lookup(warmup=args.warmup, repeats=args.repeats), indent=2)
    )


if __name__ == "__main__":
    main()
