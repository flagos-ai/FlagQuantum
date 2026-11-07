"""Accumulate overlapping OR masks through circuit.append(op).

Run python -m examples.algorithms.disjunctive_classical_predicates.
"""

from collections import Counter

import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.formats.kmx import decode_kmx, encode_kmx
from flagquantum.compiler.lowering.reversible_phase import lower_reversible_phase
from flagquantum.compiler.lowering.reversible_phase_execution import (
    bind_reversible_phase_execution,
)
from flagquantum.compiler.registers.measurement import measure_x_clear_ir
from flagquantum.core.classical_dataflow import ClassicalXor
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic


def verify(strategy: str = "batched") -> None:
    first = (((0, 1),), ((1, 1),))
    second = (((0, 0), (2, 1)), ((1, 1), (2, 0)))
    ir = CircuitIR(
        5,
        (
            *(gate for bit in range(3) for gate in measure_x_clear_ir(bit, bit)),
            Instruction("x", (3,), metadata={"condition_clauses": first}),
            ClassicalXor((0,), 0, True).instruction(),
            Instruction("z", (4,), metadata={"condition_clauses": second}),
        ),
    )
    artifact = decode_kmx(encode_kmx(lower_reversible_phase(ir)))
    target = torch.tensor((1, 1j, -1 + 2j, 2 - 1j), dtype=torch.complex128)
    target /= target.norm()
    initial = torch.zeros(32, dtype=torch.complex128)
    initial[:4] = target
    results = []
    for gates in (ir.instructions, bind_reversible_phase_execution(artifact)):
        circuit = DynamicCircuit(5, inputs=initial, dtype=torch.complex128)
        circuit.append(FieldOperation(5, gates))
        results.append(run_dynamic(circuit, shots=64, seed=38, strategy=strategy))
    direct, restored = results
    torch.testing.assert_close(
        direct.final_states, restored.final_states, rtol=0, atol=0
    )
    assert torch.equal(direct.classical_bits, restored.classical_bits[:, :3])
    assert (
        direct.statistics["measurement_count"]
        == restored.statistics["measurement_count"]
        == 192
    )
    print("Original classical slots", 3)
    print("Peak owned native predicate slots", artifact.n_bits - 3)
    print("Native primitives", dict(Counter(op.opcode for op in artifact.operations)))
    print("Overlapping clauses fire once; updated inputs and complex states match")


def main() -> None:
    verify()


if __name__ == "__main__":
    main()
