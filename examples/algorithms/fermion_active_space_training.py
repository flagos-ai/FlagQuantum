"""Train a frozen-core projection of illustrative static integrals on CPU.

Run: python -m examples.algorithms.fermion_active_space_training
The coefficients are synthetic, not an ab initio molecule or an SCF result.
"""

import torch

import flagquantum as fq
from flagquantum.algorithms.chemistry import uccsd_ansatz
from flagquantum.algorithms.chemistry_domain.active_space import select_active_space
from flagquantum.algorithms.chemistry_domain.problem import (
    MolecularProblem,
    fermion_hamiltonian,
)
from flagquantum.algorithms.fermion.encoding import encode_fermion
from flagquantum.algorithms.fermion.observables import to_observable
from flagquantum.observables import Observable


def illustrative_problem() -> MolecularProblem:
    """Build four spatial orbitals with occupied orbitals 1 and 2."""
    one = torch.diag(torch.tensor([0.4, -2.0, -1.0, 1.2], dtype=torch.float64))
    one[0, 2] = one[2, 0] = 0.15
    factor = torch.tensor(
        [
            [0.4, 0.1, 0.2, 0.0],
            [0.1, 0.7, 0.1, 0.1],
            [0.2, 0.1, 0.5, 0.0],
            [0.0, 0.1, 0.0, 0.3],
        ],
        dtype=torch.float64,
    )
    return MolecularProblem(
        one,
        torch.einsum("pq,rs->pqrs", factor, factor),
        4,
        nuclear_repulsion=0.3,
        core_energy=0.2,
        reference_occupied=(1, 2),
    )


def energy(parameters: torch.Tensor, hamiltonian: Observable) -> torch.Tensor:
    """Measure the two-electron UCCSD circuit in four JW modes."""
    circuit = uccsd_ansatz(4, 2, parameters, dtype=torch.complex128, device="cpu")
    value: torch.Tensor = fq.run(
        circuit, outputs=[fq.expectation(hamiltonian)]
    ).expectation()
    return value.reshape(())


def main() -> None:
    problem = illustrative_problem()
    selection = select_active_space(
        problem, active_orbitals=(2, 0), frozen_occupied=(1,)
    )
    active = selection.problem
    # Put the remaining occupied orbital first for the existing UCCSD reference.
    encoded = encode_fermion(
        fermion_hamiltonian(active), n_modes=active.n_modes, encoding="jordan_wigner"
    )
    observable = to_observable(encoded.operator)
    parameters = torch.tensor([0.1, -0.07, 0.2], requires_grad=True)
    initial = energy(parameters, observable)
    gradient = torch.autograd.grad(initial, parameters)[0]
    optimizer = torch.optim.Adam([parameters], lr=0.05)
    for _ in range(24):
        optimizer.zero_grad()
        torch.autograd.backward(energy(parameters, observable))
        optimizer.step()
    print("Synthetic static integrals; single_device_fast_path CPU training.")
    print(f"Active orbital labels: {active.orbital_labels}")
    print(f"Frozen orbitals: {selection.frozen_occupied}")
    print(f"Discarded virtual orbitals: {selection.discarded_virtual}")
    print(f"Active electrons: {active.n_electrons}")
    print(f"Initial energy: {float(initial.detach()):.10f} Ha")
    print(f"Gradient norm: {float(gradient.norm()):.8f}")
    print(f"Final energy: {float(energy(parameters, observable).detach()):.10f} Ha")
    print(
        "Offsets are already included. This is a fixed-occupancy projection, "
        "not a full-space ground energy or scalability claim."
    )


if __name__ == "__main__":
    main()
