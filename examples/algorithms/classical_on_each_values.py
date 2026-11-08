"""Run python -m examples.algorithms.classical_on_each_values.

Expected: six Clifford instructions, zero T/measurements/work, four declared
wires including a middle spectator; model peak three logical wires.
"""

from collections import Counter
from dataclasses import dataclass
from itertools import product

import numpy as np
import torch

import flagquantum as fq
from flagquantum.compiler.registers.bitwise import on_each_ir
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.simulation.classical_on_each import on_each_classical_value
from flagquantum.simulation.classical_pauli_values import (
    x_classical_value,
    z_basis_phase,
)
from flagquantum.simulation.classical_registers import bind_integer_register_values
from flagquantum.simulation.integer_encoding import integer_from_bits, integer_to_bits


@dataclass(frozen=True)
class EachBitXZ:
    inverse: bool = False

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        kinds = ("z", "x") if self.inverse else ("x", "z")
        gates = tuple(
            gate
            for wire in (0, 2, 3)
            for kind in kinds
            for gate in on_each_ir(Instruction(kind, (0,)), (wire,))
        )
        return circuit.compose(CircuitIR(circuit.n_qubits, gates))


def bit_xz(value: object, *, inverse: bool = False) -> tuple[object, complex]:
    """Evaluate the complete single-bit X/Z child with its resolved phase."""
    assert isinstance(value, (int, np.integer))
    output = x_classical_value(value)
    phase = z_basis_phase(value if inverse else output)
    return output, 1.0 * phase


def evaluate_word(
    value: object, width: int, *, inverse: bool = False
) -> tuple[int, complex]:
    def validate(word: object) -> None:
        bind_integer_register_values("q", {"q": word}, width)

    return on_each_classical_value(
        value,
        width,
        lambda word: integer_to_bits(word, width),
        lambda bits: integer_from_bits(bits, width),
        lambda bit: bit_xz(bit, inverse=inverse),
        validate,
    )


def main() -> None:
    grid = torch.arange(1, 17, dtype=torch.float64)
    weights = torch.complex(grid, 3 * grid - 5)
    weights /= weights.norm()
    initial = torch.zeros(16, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    for word, spectator in product(range(8), (0, 1)):
        output, phase = evaluate_word(word, 3)
        before = ((word >> 2) << 3) | (spectator << 2) | (word & 3)
        after = ((output >> 2) << 3) | (spectator << 2) | (output & 3)
        initial[before] = weights[(word << 1) | spectator]
        expected[after] = phase * initial[before]
    circuit = fq.Circuit(4, inputs=initial, dtype=torch.complex128)
    circuit.append(EachBitXZ())
    torch.testing.assert_close(circuit.state().flatten(), expected, rtol=0, atol=2e-12)
    counts = Counter(g.name for g in circuit.to_ir().instructions)
    assert counts == {"x": 3, "z": 3}
    circuit.append(EachBitXZ(inverse=True))
    torch.testing.assert_close(circuit.state().flatten(), initial, rtol=0, atol=2e-12)
    print("Split/bitwise child/Join values and phases preserve full complex amplitudes")
    print("Emitted: 6 Clifford, 0 T, 0 measurements, 0 work, 4 declared wires")
    print("Model: 6 Clifford, 3 peak logical wires; spectator excluded")


if __name__ == "__main__":
    main()
