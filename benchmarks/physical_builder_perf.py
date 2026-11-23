# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix py/perf/circuit_builder_perf.py at
# 9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453.

"""Time stateful physical construction through existing native compiler owners."""

from flagquantum.benchmarking._adaptive_timing import TimingSample
from flagquantum.benchmarking._timing_suite import TimingSuite
from flagquantum.compiler.artifacts.operand_identity import OperandIdentity, OperandKind
from flagquantum.compiler.formats.kmb import KmxRegister
from flagquantum.compiler.formats.kmx import encode_kmx
from flagquantum.compiler.formats.kmx_block_marks import encode_kmx_block_marks
from flagquantum.compiler.formats.kmx_builder import KmxCircuitBuilder
from flagquantum.compiler.lowering.physical_trace import (
    PhysicalTrace,
    lower_physical_trace,
)
from flagquantum.compiler.registers.arithmetic.addition_binding import (
    physical_addition_binding,
)
from flagquantum.compiler.registers.layout.operand_word import bind_operand_array
from flagquantum.compiler.registers.layout.workspace import RegisterWorkspace
from flagquantum.compiler.registers.primitives.broadcast_ccx import broadcast_ccx_ir

benchmark = TimingSuite("CircuitBuilder")


def _ccx_workload(
    first: object, second: object, target: object, width: int, *, bits: int = 0
) -> TimingSample:
    builder = KmxCircuitBuilder()

    def run() -> None:
        gates = broadcast_ccx_ir(first, second, target)
        program, _ = lower_physical_trace(gates, bits, width - 1, PhysicalTrace())
        builder.append_kmx_text("\n".join(encode_kmx(program).splitlines()[1:]))

    return benchmark.go(run, rates={"ops": width // 3})


@benchmark(goal_nanos=280)
def benchmark_ccx_qqq_1() -> TimingSample:
    return _ccx_workload(0, 1, 2, 3)


@benchmark(goal_nanos=950)
def benchmark_ccx_qqq_256() -> TimingSample:
    width = 256
    return _ccx_workload(
        bind_operand_array(range(width)),
        bind_operand_array(range(width, 2 * width)),
        bind_operand_array(range(2 * width, 3 * width)),
        3 * width,
    )


@benchmark(goal_nanos=700)
def benchmark_ccx_cqq_256() -> TimingSample:
    width = 256
    return _ccx_workload(
        bind_operand_array([index % 3 > 0 for index in range(width)]),
        bind_operand_array(range(width, 2 * width)),
        bind_operand_array(range(2 * width, 3 * width)),
        3 * width,
    )


@benchmark(goal_nanos=3300)
def benchmark_ccx_mixed_256() -> TimingSample:
    width = 256
    first = bind_operand_array(
        [True if index % 2 == 0 else index for index in range(width)]
    )
    second = bind_operand_array(
        [
            index if index % 3 == 0 else OperandIdentity(OperandKind.CLASSICAL, index)
            for index in range(width, 2 * width)
        ]
    )
    target = bind_operand_array(range(2 * width, 3 * width))
    return _ccx_workload(first, second, target, 3 * width, bits=2 * width)


@benchmark(goal_nanos=3300)
def benchmark_iadd_256() -> TimingSample:
    width = 256
    builder = KmxCircuitBuilder()
    builder.replace_registers(
        (
            KmxRegister("offset", tuple(("q", index) for index in range(width))),
            KmxRegister(
                "target", tuple(("q", index) for index in range(width, 2 * width))
            ),
        )
    )
    workspace = RegisterWorkspace(tuple(range(2 * width, 3 * width - 2)))

    def run() -> None:
        lease = workspace.allocate(width - 2)
        program, calls = physical_addition_binding(
            tuple(reversed(range(width, 2 * width))),
            target=tuple(reversed(range(width))),
            workspace=lease.wires,
        )
        builder.append_kmx_text(
            "\n".join(encode_kmx_block_marks(program, calls).splitlines()[1:])
        )
        workspace.free(lease)

    return benchmark.go(run, rates={"qubits": width})


if __name__ == "__main__":
    benchmark.main()
