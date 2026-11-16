"""Train inside an explicitly selected even-parity sector on one CPU qubit.

Run: python -m examples.algorithms.fermion_parity_training
"""

import torch

import flagquantum as fq
from flagquantum.algorithms.fermion.encoding import encode_fermion
from flagquantum.algorithms.fermion.numerics import pauli_matrix
from flagquantum.algorithms.fermion.observables import to_observable
from flagquantum.algorithms.fermion.symmetry import (
    discover_z_parities,
    reduce_z_parities,
)
from flagquantum.observables import FermionOperator, FermionTerm, Observable
from flagquantum.observables.fermion import annihilate, create, number


def build_ansatz(angle: torch.Tensor) -> fq.Circuit:
    """Prepare a real one-qubit reduced state with a trainable scalar angle."""
    if angle.ndim != 0:
        raise ValueError("the reduced preparation requires a scalar angle")
    return fq.Circuit(1, dtype=torch.complex128, device="cpu").ry(0, theta=angle)


def energy(angle: torch.Tensor, hamiltonian: Observable) -> torch.Tensor:
    """Measure the reduced observable through the existing native runtime."""
    value: torch.Tensor = fq.run(
        build_ansatz(angle), outputs=[fq.expectation(hamiltonian)]
    ).expectation()
    return value.reshape(())


def main() -> None:
    print(
        "premise: fixed two-mode pairing model, explicit even parity and CPU "
        "single_device_fast_path; only the circuit angle is trainable. "
        "Even parity is not fixed particle number. Dense lifting is bounded "
        "inspection, not a general convergence or scalability claim.\n"
    )
    identity = FermionOperator((FermionTerm(1),))
    population = number(0) + number(1)
    hamiltonian = (
        population
        - identity
        - 0.7 * (create(0) * create(1) + annihilate(1) * annihilate(0))
    )
    encoded = encode_fermion(hamiltonian, n_modes=2)
    generators = discover_z_parities(encoded, candidates=[(0, 1)])
    reduced = reduce_z_parities(encoded, generators=generators, eigenvalues=[1])
    observable = to_observable(reduced.operator)
    angle = torch.tensor(1.7, dtype=torch.float64, requires_grad=True)
    before = angle.detach().clone()
    initial = energy(angle, observable)
    gradient = torch.autograd.grad(initial, angle)[0]
    with torch.no_grad():
        difference = (
            energy(angle + 1e-5, observable) - energy(angle - 1e-5, observable)
        ) / 2e-5
    optimizer = torch.optim.Adam([angle], lr=0.05)
    for _ in range(200):
        optimizer.zero_grad()
        torch.autograd.backward(energy(angle, observable))
        optimizer.step()
    with torch.no_grad():
        final = energy(angle, observable)
        lifted = reduced.mapping.lift_state(build_ansatz(angle).state().reshape(-1))
        full_matrix = pauli_matrix(encoded.operator, n_qubits=2)
        full_energy = (lifted.conj() @ full_matrix @ lifted).real
        particles = (
            lifted.conj()
            @ pauli_matrix(encode_fermion(population, n_modes=2).operator, n_qubits=2)
            @ lifted
        ).real
    print(f"Selected constraint: {generators}; eigenvalue: +1")
    print(f"Register: 2 -> {reduced.n_qubits} qubits")
    print(f"Initial energy: {float(initial.detach()):.10f}")
    print(f"Gradient residual: {float((gradient - difference).abs()):.3e}")
    print(f"Parameter update: {float((angle.detach() - before).abs()):.10f}")
    print(f"Final energy: {float(final):.10f}")
    print(f"Analytic even-sector ground energy: {-1.49**0.5:.10f}")
    print(f"Lifted energy residual: {float((full_energy - final).abs()):.3e}")
    print(f"Odd-sector leakage: {float(lifted[[1, 2]].abs().square().sum()):.3e}")
    print(f"Particles: {float(particles):.10f}")


if __name__ == "__main__":
    main()
