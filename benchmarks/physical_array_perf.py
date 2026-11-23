# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix py/perf/array_perf.py at
# 9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453.

"""Time complete mixed operand-array construction using its native binder."""

from flagquantum.benchmarking._adaptive_timing import TimingSample
from flagquantum.benchmarking._timing_suite import TimingSuite
from flagquantum.compiler.artifacts.operand_identity import OperandIdentity, OperandKind
from flagquantum.compiler.registers.layout.operand_word import bind_operand_array

benchmark = TimingSuite("OperandArray")


@benchmark(goal_micros=2.8)
def benchmark_construct() -> TimingSample:
    items: list[OperandIdentity | bool] = [
        OperandIdentity(OperandKind.QUANTUM, index) for index in range(100)
    ]
    items.extend(OperandIdentity(OperandKind.CLASSICAL, index) for index in range(100))
    items.extend(index % 2 == 0 for index in range(100))

    def run() -> None:
        bind_operand_array(items)

    return benchmark.go(run, rates={"items": len(items)})


if __name__ == "__main__":
    benchmark.main()
