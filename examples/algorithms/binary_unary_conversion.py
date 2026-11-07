"""Convert a coherent binary word to one-hot and back with circuit.append(op).

Run python -m examples.algorithms.binary_unary_conversion.
"""

from collections import Counter

import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.artifacts.reversible_phase import ReversiblePhaseProgram
from flagquantum.compiler.lowering.reversible_phase_execution import (
    bind_reversible_phase_execution,
)
from flagquantum.compiler.registers.binary_unary_conversion import (
    binary_to_unary_operations,
    unary_to_binary_operations,
)
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic


def main() -> None:
    generator = torch.Generator().manual_seed(172)
    amplitudes = torch.complex(
        torch.randn(8, generator=generator, dtype=torch.float64),
        torch.randn(8, generator=generator, dtype=torch.float64),
    )
    amplitudes /= amplitudes.norm()
    binary = torch.zeros(32, dtype=torch.complex128)
    binary[:8] = amplitudes
    unary = torch.zeros_like(binary)
    for value in range(4):
        unary[(1 << value) * 2 : (1 << value) * 2 + 2] = amplitudes[
            2 * value : 2 * value + 2
        ]
    forward = binary_to_unary_operations((0, 1, 2, 3))
    inverse = unary_to_binary_operations((0, 1, 2, 3))
    for label, operations, expected in (
        ("forward", forward, unary),
        ("round_trip", (*forward, *inverse), binary),
    ):
        program = ReversiblePhaseProgram(5, 1, operations)
        op = FieldOperation(5, bind_reversible_phase_execution(program))
        circuit = DynamicCircuit(5, inputs=binary, dtype=torch.complex128)
        circuit.append(op)
        result = run_dynamic(circuit, shots=32, seed=92, strategy="batched")
        overlaps = result.final_states @ expected.conj()
        assert bool(torch.allclose(overlaps.abs(), torch.ones_like(overlaps.real)))
        print(
            label,
            "source primitive counts",
            dict(Counter(operation.opcode for operation in operations)),
        )
    print("Coherent conversion and round trip verified; surplus target ports cleared")


if __name__ == "__main__":
    main()
