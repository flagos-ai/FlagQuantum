"""Run python -m examples.algorithms.classical_basis_values."""

from collections import Counter
from dataclasses import dataclass

import torch

import flagquantum as fq
from flagquantum.compiler.registers.layout.int_vector import (
    int_effect_bindings,
    int_state_ir,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.simulation.classical_basis_vectors import (
    bit_basis_classical_values,
    integer_basis_classical_values,
)
from flagquantum.simulation.tensor_network.entrypoints import (
    build_tensor_network,
    tensor_network_basis_effect,
)


@dataclass(frozen=True)
class IntegerBasis:
    """Load a configured value into a clean three-bit register."""

    value: int

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        return circuit.compose(
            CircuitIR(circuit.n_qubits, int_state_ir((0, 1, 2), self.value))
        )


def main() -> None:
    assert bit_basis_classical_values(True) == {"q": 1}
    assert bit_basis_classical_values(True, value=1, state=False) == {}
    value = integer_basis_classical_values(5, 3)["val"]
    assert type(value) is int
    spectator = torch.tensor([1j, 2 - 1j], dtype=torch.complex128)
    spectator /= spectator.norm()
    initial = torch.zeros(16, dtype=torch.complex128)
    initial[:2] = spectator
    expected = torch.zeros_like(initial)
    expected[2 * value : 2 * value + 2] = spectator
    circuit = fq.Circuit(4, inputs=initial, dtype=torch.complex128)
    op = IntegerBasis(value)
    circuit.append(op)
    torch.testing.assert_close(circuit.state().flatten(), expected, rtol=0, atol=2e-12)
    assert integer_basis_classical_values(5, 3, incoming=value, state=False) == {}
    projected = tensor_network_basis_effect(
        build_tensor_network(circuit, dtype=torch.complex128),
        int_effect_bindings((0, 1, 2), value),
    )
    torch.testing.assert_close(projected.flatten(), spectator, rtol=0, atol=2e-12)
    assert Counter(g.name for g in circuit.to_ir().instructions) == {"x": 2}
    circuit.append(op)
    torch.testing.assert_close(circuit.state().flatten(), initial, rtol=0, atol=2e-12)
    print("Creation, checked consumption and complex spectator projection agree")
    print("Emitted state: 2 X, 0 T, 0 measurements, 0 work; inverse clears output")
    print("Source state/effect model: 1 abstract Clifford each, 3 logical wires")
    print("Projection is unnormalized contraction; it is not a deterministic reset")


if __name__ == "__main__":
    main()
