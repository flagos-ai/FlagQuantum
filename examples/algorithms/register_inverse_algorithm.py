"""Construct the complete modular inverse body using circuit.append(op).

Run: python -m examples.algorithms.register_inverse_algorithm
This example constructs IR and reports resources; it does not allocate a dense
38-qubit statevector. Complete complex-channel tests use sparse primitive checks.
"""

from collections import Counter
from dataclasses import dataclass

import flagquantum as fq
from flagquantum.compiler.registers.modular.inversion.inverse_algorithm import (
    inverse_algorithm_ir,
    inverse_algorithm_outputs,
)
from flagquantum.core.ir import CircuitIR


@dataclass(frozen=True)
class InverseAlgorithm:
    """Four logical words, history outputs and an explicitly owned work pool."""

    bitsize: int = 3
    modulus: int = 5
    adjoint: bool = False

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        n = self.bitsize
        u, v, r, s = (tuple(range(i * n, (i + 1) * n)) for i in range(4))
        markers = tuple(range(4 * n, 6 * n))
        active = 6 * n
        terminals = tuple(range(6 * n + 1, 8 * n + 1))
        work = tuple(range(8 * n + 1, 10 * n + 8))
        if self.adjoint:
            r, work = inverse_algorithm_outputs(r, work)
        gates = inverse_algorithm_ir(
            u,
            v,
            r,
            s,
            markers,
            active,
            terminals,
            self.modulus,
            work,
            tuple(range(max(2 * n - 1, n + 1))),
            adjoint=self.adjoint,
        )
        return circuit.compose(CircuitIR(circuit.n_qubits, gates))


def main() -> None:
    """Show literal gates and independently weighted upstream estimates."""
    n, p = 3, 5
    circuit = fq.Circuit(10 * n + 8)
    circuit.append(InverseAlgorithm(n, p))
    counts = Counter(gate.name for gate in circuit.to_ir().instructions)
    r, work = inverse_algorithm_outputs(
        tuple(range(2 * n, 3 * n)), tuple(range(8 * n + 1, 10 * n + 8))
    )
    print("Forward input v contains zero or a unit mod", p, "other inputs start zero.")
    print("Output coefficient: x^-1 * 2**(2n) mod p; zero maps to zero.")
    print("Physical coefficient:", r, "clean work:", work)
    print(
        "Retained logical history:",
        4 * n,
        "bits; classical slots:",
        max(2 * n - 1, n + 1),
    )
    print("Measured primitive IR:", dict(counts))
    print("Independent source default T:", 2 * n * (52 * n + 16) + 4 * (n - 1))
    print(
        "Independent exact seven-T swap model T:", 2 * n * (67 * n + 13) + 4 * (n - 1)
    )
    circuit.append(InverseAlgorithm(n, p, adjoint=True))
    assert len(circuit.to_ir().instructions) == 2 * sum(counts.values())
    print("Original inverse appended using the actual forward coefficient/work roles.")


if __name__ == "__main__":
    main()
