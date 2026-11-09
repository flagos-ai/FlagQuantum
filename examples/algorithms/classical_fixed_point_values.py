"""Run python -m examples.algorithms.classical_fixed_point_values.

Stored integers, fractional scale and coherent phase-gradient use remain distinct.
"""

from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch

import flagquantum as fq
from flagquantum.compiler.registers.phase.phase_gradient import (
    phase_gradient_unitary_ir,
)
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program.register_types import (
    classical_fixed_point_register_type,
    fixed_point_register_type,
)
from flagquantum.simulation.fixed_point_encoding import (
    fixed_point_array_from_bits,
    fixed_point_array_to_bits,
    fixed_point_domain,
    fixed_point_real_value,
)


@dataclass(frozen=True)
class Gradient:
    instructions: tuple[Instruction, ...]

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        return circuit.compose(CircuitIR(4, self.instructions))


def main() -> None:
    quantum_word = fixed_point_register_type(3, 2, True)
    classical_word = classical_fixed_point_register_type(3, 2, True)
    assert quantum_word.num_bits == classical_word.num_bits == 3
    values = np.array([-4, -1, 0, 3])
    bits = fixed_point_array_to_bits(
        values, classical_word.classical_bits, 2, signed=True
    )
    np.testing.assert_array_equal(
        fixed_point_array_from_bits(
            bits, classical_word.classical_bits, 2, signed=True
        ),
        values,
    )
    print("Stored signed integers:", values.tolist())
    print(
        "Real scale (two fractional bits):",
        [
            fixed_point_real_value(int(v), quantum_word.quantum_bits, 2, signed=True)
            for v in values
        ],
    )

    words = tuple(fixed_point_domain(3, 3))
    grid = torch.arange(1, 17, dtype=torch.float64)
    initial = torch.complex(grid, 3 * grid - 5)
    initial /= initial.norm()
    real_values = torch.tensor(
        [fixed_point_real_value(word, 3, 3) for word in words], dtype=torch.float64
    )
    expected = initial.reshape(8, 2) * torch.exp(2j * torch.pi * real_values)[:, None]
    gates = phase_gradient_unitary_ir((0, 1, 2))
    circuit = fq.Circuit(4, inputs=initial, dtype=torch.complex128)
    circuit.append(Gradient(gates))
    torch.testing.assert_close(
        circuit.state().reshape(8, 2), expected, rtol=0, atol=2e-12
    )
    circuit.append(Gradient(phase_gradient_unitary_ir((0, 1, 2), exponent=-1)))
    torch.testing.assert_close(circuit.state().flatten(), initial, rtol=0, atol=2e-12)
    counts = Counter(gate.name for gate in gates)
    assert counts == {"phase": 3}
    print("Emitted: 3 phase instructions, 0 explicit T, 0 measurements, 0 work")
    print("Pinned source model: 2 Clifford and 1 T; separate from emitted phase syntax")
    print(
        "All complex amplitudes and the spectator restored; no real-to-word quantization"
    )


if __name__ == "__main__":
    main()
