"""Append the original public modular inverse, keeping logical history explicit.

Run: python -m examples.algorithms.register_modular_inverse
Constructs 38-wire IR and reports resources, without allocating a dense state.
"""

from collections import Counter
from dataclasses import dataclass

import flagquantum as fq
from flagquantum.compiler.registers.modular.modular_inverse import (
    modular_inverse_ir,
    modular_inverse_outputs,
)
from flagquantum.core.ir import CircuitIR


@dataclass(frozen=True)
class ModularInverse:
    """Bind the target, four-n-bit history and clean work for native append."""

    bitsize: int = 3
    modulus: int = 5
    adjoint: bool = False

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        n = self.bitsize
        target = tuple(range(n))
        junk = tuple(range(n, 5 * n))
        work = tuple(range(5 * n, 10 * n + 8))
        if self.adjoint:
            target, work = modular_inverse_outputs(target, work)
        gates = modular_inverse_ir(
            target,
            junk,
            self.modulus,
            work,
            tuple(range(max(2 * n - 1, n + 1))),
            adjoint=self.adjoint,
        )
        return circuit.compose(CircuitIR(circuit.n_qubits, gates))


def main() -> None:
    """Show real physical outputs and keep model costs distinct from literal gates."""
    n, p = 3, 5
    circuit = fq.Circuit(10 * n + 8)
    circuit.append(ModularInverse(n, p))
    counts = Counter(g.name for g in circuit.to_ir().instructions)
    original = tuple(range(n))
    work = tuple(range(5 * n, 10 * n + 8))
    target, returned = modular_inverse_outputs(original, work)
    assert modular_inverse_outputs(target, returned, adjoint=True) == (original, work)
    print("Output: x^-1 * 2^(2n) mod p, with zero mapped to zero.")
    print("Physical output:", target, "returned clean work:", returned)
    print(
        "Retained logical history:",
        4 * n,
        "bits; supply the same history to uncompute.",
    )
    print("Measured primitive IR:", dict(counts))
    print(
        "Source models: default",
        2 * n * (52 * n + 16) + 4 * (n - 1),
        "T; exact seven-T swaps",
        2 * n * (67 * n + 13) + 4 * (n - 1),
        "T.",
    )
    circuit.append(ModularInverse(n, p, adjoint=True))
    assert len(circuit.to_ir().instructions) == 2 * sum(counts.values())
    print("Matching inverse appended; no output-materializing SWAP or logical reset.")


if __name__ == "__main__":
    main()
