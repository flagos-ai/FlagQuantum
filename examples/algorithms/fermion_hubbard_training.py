"""Train a two-site Hubbard state with native expectations and PyTorch.

Run: python -m examples.algorithms.fermion_hubbard_training
"""

import torch

import flagquantum as fq
from flagquantum.algorithms.fermion.encoding import encode_fermion
from flagquantum.algorithms.fermion.hubbard import fermi_hubbard_chain
from flagquantum.algorithms.fermion.observables import to_observable
from flagquantum.observables import Observable
from flagquantum.observables.fermion import FermionOperator, number


def build_ansatz(parameters: torch.Tensor) -> fq.Circuit:
    """Prepare one up and one down particle in four interleaved JW modes.

    Three real angles prepare two logical occupation choices; complementary
    bits embed them as (a, b, 1-a, 1-b). Intermediate gates need not conserve
    particle number, but every completed preparation lies in this sector.
    """
    circuit = fq.Circuit(4, dtype=torch.complex128)
    circuit.ry(0, theta=parameters[0])
    circuit.ry(1, theta=parameters[1])
    circuit.cx(0, 1)
    circuit.ry(1, theta=parameters[2])
    circuit.x(2).x(3).cx(0, 2).cx(1, 3)
    return circuit


def energy(parameters: torch.Tensor, hamiltonian: Observable) -> torch.Tensor:
    """Evaluate the differentiable CPU expectation for this preparation."""
    result = fq.run(build_ansatz(parameters), outputs=[fq.expectation(hamiltonian)])
    return result.expectation().reshape(())


def main() -> None:
    print(
        "premise: two-site CPU training in unreduced JW order; sector preparation "
        "is not gatewise number conservation or a scalable many-body solver.\n"
    )
    encoded = encode_fermion(
        fermi_hubbard_chain(2, t=1, u=4), n_modes=4, encoding="jordan_wigner"
    )
    hamiltonian = to_observable(encoded.operator)
    parameters = torch.tensor([0.4, -0.3, 0.2], dtype=torch.float64, requires_grad=True)
    initial = energy(parameters, hamiltonian)
    gradient = torch.autograd.grad(initial, parameters)[0]
    epsilon = 1e-5
    shifts = epsilon * torch.eye(3, dtype=torch.float64)
    with torch.no_grad():
        finite_difference = torch.stack(
            [
                (
                    energy(parameters + shift, hamiltonian)
                    - energy(parameters - shift, hamiltonian)
                )
                / (2 * epsilon)
                for shift in shifts
            ]
        )
    optimizer = torch.optim.Adam([parameters], lr=0.1)
    for _ in range(240):
        optimizer.zero_grad()
        energy(parameters, hamiltonian).backward()
        optimizer.step()
    particles = sum((number(mode) for mode in range(4)), FermionOperator())
    population = to_observable(
        encode_fermion(particles, n_modes=4, encoding="jordan_wigner").operator
    )
    with torch.no_grad():
        final = energy(parameters, hamiltonian)
        count = fq.run(
            build_ansatz(parameters), outputs=[fq.expectation(population)]
        ).expectation()
    print(f"Initial energy: {float(initial.detach()):.10f}")
    print(f"Gradient norm: {float(gradient.norm()):.10f}")
    print(f"Gradient residual: {float((gradient - finite_difference).abs().max()):.3e}")
    print(f"Final energy: {float(final):.10f}")
    print(f"Analytic N=2 energy: {(4 - 32**0.5) / 2:.10f}")
    print(f"Particles: {float(count.reshape(())):.10f}")


if __name__ == "__main__":
    main()
