"""Discover X/Y/Z symmetries and measure an explicitly selected joint sector.

Run: python -m examples.algorithms.fermion_pauli_discovery
"""

import torch

import flagquantum as fq
from flagquantum.algorithms.fermion.lowering import lift_stabilizer_state
from flagquantum.algorithms.fermion.numerics import pauli_matrix
from flagquantum.algorithms.fermion.observables import to_observable
from flagquantum.algorithms.fermion.stabilizer import (
    discover_pauli_symmetries,
    reduce_stabilizers,
)
from flagquantum.algorithms.primitives.state_preparation import append_arbitrary_state
from flagquantum.core.algebra.pauli import pauli


def main() -> None:
    print(
        "premise: fixed three-qubit encoded Pauli model on CPU; explicit +1/-1 "
        "eigenvalues select the discovered positive generators. Discovery does "
        "not select a ground sector or particle number. Dense diagonalization "
        "and lifting are bounded inspection, not a scalable solver.\n"
    )
    y0 = pauli(factors=((0, "y"),))
    x1z2 = pauli(factors=((1, "x"), (2, "z")))
    operator = (
        pauli(factors=((2, "z"),))
        + 0.4 * pauli(factors=((1, "z"), (2, "x")))
        + 0.2 * y0
        + 0.3 * x1z2
    )
    generators = discover_pauli_symmetries(operator, n_qubits=3)
    reduced = reduce_stabilizers(
        operator, n_qubits=3, generators=generators, eigenvalues=[1, -1]
    )
    matrix = pauli_matrix(reduced.operator, n_qubits=reduced.n_qubits)
    energies, vectors = torch.linalg.eigh(matrix)
    state = vectors[:, 0]
    circuit = fq.Circuit(reduced.n_qubits, dtype=torch.complex128, device="cpu")
    append_arbitrary_state(circuit, state, qubits=tuple(range(reduced.n_qubits)))
    measured = fq.run(
        circuit, outputs=[fq.expectation(to_observable(reduced.operator))]
    )
    lifted = lift_stabilizer_state(reduced, state)
    residual = (
        (pauli_matrix(operator, n_qubits=3) @ lifted - energies[0] * lifted).abs().max()
    )
    print(f"Independent commuting generators: {len(generators)}")
    print("Selected Y0 and X1Z2 eigenvalues: +1, -1")
    print(f"Register: 3 -> {reduced.n_qubits} qubits")
    print(
        f"Selected-sector measured energy: {float(measured.expectation().reshape(())):.10f}"
    )
    print(f"Analytic selected-sector energy: {-0.1 - 1.16**0.5:.10f}")
    print(f"Lifted eigenvector residual: {float(residual):.3e}")


if __name__ == "__main__":
    main()
