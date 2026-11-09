"""Small independent physical references, not production execution paths."""

from __future__ import annotations

import torch

from flagquantum.observables.fermion import FermionOperator


def fock_matrix(operator: FermionOperator, width: int) -> torch.Tensor:
    """Apply the canonical anticommutation sign directly to occupation states."""
    result = torch.zeros((2**width, 2**width), dtype=torch.complex128)
    for column in range(2**width):
        for term in operator.terms:
            bits = [(column >> (width - i - 1)) & 1 for i in range(width)]
            coefficient = term.coefficient
            factors = [(q, True) for q in term.creations] + [
                (q, False) for q in term.annihilations
            ]
            for mode, creating in reversed(factors):
                if bits[mode] == int(creating):
                    coefficient = 0
                    break
                coefficient *= (-1) ** sum(bits[:mode])
                bits[mode] = int(creating)
            row = sum(bit << (width - i - 1) for i, bit in enumerate(bits))
            result[row, column] += coefficient
    return result
