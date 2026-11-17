# SPDX-License-Identifier: Apache-2.0
# Workloads adapted from the locked Kickmix gen_gf_inverse.perf.cc.

"""Measure native physical generation for the locked GF(2^128) workloads."""

import argparse
import json
from collections import Counter
from statistics import median
from time import perf_counter_ns

from flagquantum.compiler._component_trace.trace import ComponentCall
from flagquantum.compiler.registers.finite_field.inverse_chain import (
    field_inverse_chain_size,
    field_inverse_workspace_size,
    physical_field_inverse,
)

_WIDTH = 128
_MODULUS = (1 << _WIDTH) | 0x87


def _mark_count(calls: tuple[ComponentCall, ...]) -> int:
    return sum(2 + _mark_count(call.children) for call in calls)


def benchmark_field_inverse(*, warmup: int = 0, repeats: int = 1) -> dict[str, object]:
    """Time native construction and lowering in the two source port layouts."""
    if warmup < 0 or repeats < 1:
        raise ValueError("Warmup must be nonnegative and repeats must be positive")
    report: dict[str, object] = {}
    for retain_chain in (False, True):
        work = (
            field_inverse_chain_size if retain_chain else field_inverse_workspace_size
        )(_WIDTH)
        words = [
            tuple(reversed(range(i, i + _WIDTH)))
            for i in range(0, 2 * _WIDTH + work, _WIDTH)
        ]
        source, target, *workspace = words
        clean = tuple(q for word in workspace for q in word)
        times = []
        for iteration in range(warmup + repeats):
            start = perf_counter_ns()
            program, calls = physical_field_inverse(
                source,
                target,
                clean,
                modulus=_MODULUS,
                retain_chain=retain_chain,
            )
            elapsed = perf_counter_ns() - start
            if iteration >= warmup:
                times.append(elapsed)
        counts = Counter(op.opcode for op in program.operations)
        records = len(program.operations) + _mark_count(calls)
        seconds = median(times) / 1_000_000_000
        report["retained" if retain_chain else "self_cleaned"] = {
            "width": _WIDTH,
            "work_qubits": work,
            "records_including_marks": records,
            "primitive_counts": dict(counts),
            "median_seconds": seconds,
            "records_per_second": records / seconds,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    report = benchmark_field_inverse(warmup=args.warmup, repeats=args.repeats)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
