"""Independent occupation-sign construction of the source fermionic Hamiltonian."""

import torch

from examples.synthesis.applications.chemistry.second_quantized_hamiltonian.comparisons.circuit import (
    reference_circuit as reference_circuit,
)


def occupation_matrix() -> torch.Tensor:
    """Apply ordered monomials right to left, including fermionic parity signs."""
    terms = (
        (0.2, ((0, True), (0, False))),
        (-0.1, ((0, True), (1, True), (1, False), (0, False))),
        (-0.3, ((2, True), (3, True), (2, False), (3, False))),
    )
    matrix = torch.zeros((16, 16), dtype=torch.complex128)
    for column in range(16):
        for coefficient, factors in terms:
            bits = [(column >> (3 - i)) & 1 for i in range(4)]
            value = coefficient
            for mode, creation in reversed(factors):
                if bits[mode] == int(creation):
                    value = 0.0
                    break
                value *= (-1) ** sum(bits[:mode])
                bits[mode] = int(creation)
            row = sum(bit << (3 - i) for i, bit in enumerate(bits))
            matrix[row, column] += value
    return matrix
