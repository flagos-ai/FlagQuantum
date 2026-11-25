"""Measure a two-qubit H2 symmetry sector and lift its state to four qubits.

Run: python -m examples.algorithms.fermion_tapering
"""

import torch

import flagquantum as fq
from flagquantum.algorithms.chemistry_domain.problem import (
    fermion_hamiltonian,
    from_native_molecule,
)
from flagquantum.algorithms.fermion.clifford import conjugate_pauli
from flagquantum.algorithms.fermion.encoding import encode_fermion
from flagquantum.algorithms.fermion.lowering import lift_stabilizer_state
from flagquantum.algorithms.fermion.numerics import pauli_matrix
from flagquantum.algorithms.fermion.observables import to_observable
from flagquantum.algorithms.fermion.stabilizer import reduce_stabilizers
from flagquantum.algorithms.molecular import create_molecular_hamiltonian
from flagquantum.algorithms.primitives.state_preparation import append_arbitrary_state
from flagquantum.core.algebra.pauli import pauli
from flagquantum.core.ir import Instruction
from flagquantum.observables import number


def main() -> None:
    print(
        "premise: fixed H2 integrals and an explicitly supplied symmetry sector; "
        "small dense diagonalization supplies the state, not a scalable eigensolver.\n"
    )
    problem = from_native_molecule(
        create_molecular_hamiltonian([("H", (0, 0, 0)), ("H", (0, 0, 0.7414))])
    )
    operator = encode_fermion(fermion_hamiltonian(problem), n_modes=4).operator
    particles = encode_fermion(
        sum((number(q) for q in range(4)), 0 * number(0)), n_modes=4
    ).operator
    generators = [
        pauli(factors=((0, "z"), (2, "z"))),
        pauli(factors=((1, "z"), (3, "z"))),
    ]
    # Rotate the orbital-occupation qubit basis to exercise a genuine signed X/Y
    # symmetry. This changes representation, not the physical molecular problem.
    for gate in (
        Instruction("h", (0,)),
        Instruction("h", (2,)),
        Instruction("sdg", (2,)),
    ):
        operator = conjugate_pauli(operator, gate)
        particles = conjugate_pauli(particles, gate)
        generators = [conjugate_pauli(g, gate) for g in generators]
    reduced = reduce_stabilizers(
        operator, n_qubits=4, generators=generators, eigenvalues=[-1, -1]
    )
    matrix = pauli_matrix(reduced.operator, n_qubits=reduced.n_qubits)
    energies, vectors = torch.linalg.eigh(matrix)
    state = vectors[:, 0]
    circuit = fq.Circuit(reduced.n_qubits, dtype=torch.complex128)
    append_arbitrary_state(circuit, state, qubits=tuple(range(reduced.n_qubits)))
    energy = fq.run(circuit, outputs=[fq.expectation(to_observable(reduced.operator))])
    population = fq.run(
        circuit,
        outputs=[fq.expectation(to_observable(reduced.reduce_operator(particles)))],
    )
    lifted = lift_stabilizer_state(reduced, state)
    residual = float(
        (pauli_matrix(operator, n_qubits=4) @ lifted - energies[0] * lifted).abs().max()
    )
    print(f"Register: 4 -> {reduced.n_qubits} qubits")
    print(
        f"Selected-sector measured energy: {float(torch.as_tensor(energy.expectation()).reshape(-1)[0]):.10f} Ha"
    )
    print(
        f"Particle number: {float(torch.as_tensor(population.expectation()).reshape(-1)[0]):.6f}"
    )
    print(f"Lifted eigenvector residual: {residual:.3e}")


if __name__ == "__main__":
    main()
