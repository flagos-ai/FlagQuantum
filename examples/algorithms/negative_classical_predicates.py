"""Run alias-safe negative classical predicates through circuit.append(op).

Run python -m examples.algorithms.negative_classical_predicates.
"""

from collections import Counter

import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.formats.kmx import decode_kmx, encode_kmx
from flagquantum.compiler.lowering.reversible_phase import lower_reversible_phase
from flagquantum.compiler.lowering.reversible_phase_execution import (
    bind_reversible_phase_execution,
)
from flagquantum.compiler.registers.primitives.measurement import measure_x_clear_ir
from flagquantum.core.classical_dataflow import ClassicalXor
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic


def main() -> None:
    assignment = ClassicalXor((0, 1), 0, True).instruction()
    parity = Instruction(
        assignment.name,
        (),
        assignment.params,
        metadata={**assignment.metadata, "conditions": ((0, 0), (1, 0))},
    )
    ir = CircuitIR(
        4,
        (
            *measure_x_clear_ir(0, 0),
            *measure_x_clear_ir(1, 1),
            parity,
            Instruction("z", (2,), metadata={"conditions": ((0, 0),)}),
        ),
    )
    artifact = decode_kmx(encode_kmx(lower_reversible_phase(ir)))
    initial = torch.zeros(16, dtype=torch.complex128)
    initial[:4] = torch.tensor((1, 0, 0, 1j), dtype=torch.complex128) * 2**-0.5
    results = []
    for gates in (ir.instructions, bind_reversible_phase_execution(artifact)):
        op = FieldOperation(4, gates)
        circuit = DynamicCircuit(4, inputs=initial, dtype=torch.complex128)
        circuit.append(op)
        results.append(run_dynamic(circuit, shots=64, seed=81, strategy="batched"))
    original, restored = results
    torch.testing.assert_close(
        original.final_states, restored.final_states, rtol=0, atol=0
    )
    assert torch.equal(original.classical_bits, restored.classical_bits[:, :2])
    assert (
        original.statistics["measurement_count"]
        == restored.statistics["measurement_count"]
        == 128
    )
    print("Original classical slots", 2)
    print("Peak native complement slots", artifact.n_bits - 2)
    print(
        "Native source primitives",
        dict(Counter(op.opcode for op in artifact.operations)),
    )
    print("Negative guards and aliased parity match; no quantum work added")


if __name__ == "__main__":
    main()
