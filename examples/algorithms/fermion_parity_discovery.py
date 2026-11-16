"""Discover a total-parity symmetry, select its sector and measure natively.

Run: python -m examples.algorithms.fermion_parity_discovery
"""

import torch

import flagquantum as fq
from flagquantum.algorithms.fermion.encoding import encode_fermion
from flagquantum.algorithms.fermion.hubbard import fermi_hubbard_chain
from flagquantum.algorithms.fermion.numerics import pauli_matrix
from flagquantum.algorithms.fermion.observables import to_observable
from flagquantum.algorithms.fermion.symmetry import (
    discover_z_parities,
    reduce_z_parities,
)
from flagquantum.algorithms.primitives.state_preparation import append_arbitrary_state


def main() -> None:
    print(
        "premise: fixed two-site CPU Hubbard model in JW order; N=2 selects "
        "even total parity, not an exact particle-number sector. Dense sector "
        "diagonalization is bounded inspection, not a scalable solver.\n"
    )
    encoded = encode_fermion(fermi_hubbard_chain(2, t=1, u=4), n_modes=4)
    all_generators = discover_z_parities(encoded)
    # Restrict the search to the one constraint whose eigenvalue follows from N.
    generators = discover_z_parities(encoded, candidates=[(0, 1, 2, 3)])
    reduced = reduce_z_parities(encoded, generators=generators, eigenvalues=[(-1) ** 2])
    matrix = pauli_matrix(reduced.operator, n_qubits=reduced.n_qubits)
    energies, vectors = torch.linalg.eigh(matrix)
    state = vectors[:, 0]
    circuit = fq.Circuit(reduced.n_qubits, dtype=torch.complex128, device="cpu")
    append_arbitrary_state(circuit, state, qubits=tuple(range(reduced.n_qubits)))
    result = fq.run(circuit, outputs=[fq.expectation(to_observable(reduced.operator))])
    lifted = reduced.mapping.lift_state(state)
    residual = (
        (pauli_matrix(encoded.operator, n_qubits=4) @ lifted - energies[0] * lifted)
        .abs()
        .max()
    )
    print(f"All independent Z symmetries: {all_generators}")
    print(f"Selected constraint: {generators}; eigenvalue: +1")
    print(f"Register: 4 -> {reduced.n_qubits} qubits")
    print(
        f"Even-parity measured energy: {float(result.expectation().reshape(())):.10f}"
    )
    print(f"Lifted eigenvector residual: {float(residual):.3e}")


if __name__ == "__main__":
    main()
