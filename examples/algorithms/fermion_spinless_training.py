"""Train a three-site spinless chain or ring with native circuits and PyTorch.

Run: python -m examples.algorithms.fermion_spinless_training [--periodic]
"""

import argparse

import torch

import flagquantum as fq
from flagquantum.algorithms.chemistry import single_excitation
from flagquantum.algorithms.fermion.encoding import encode_fermion
from flagquantum.algorithms.fermion.hubbard import (
    spinless_fermion_chain,
    spinless_fermion_ring,
)
from flagquantum.algorithms.fermion.observables import to_observable
from flagquantum.observables import Observable
from flagquantum.observables.fermion import FermionOperator, number


def build_ansatz(parameters: torch.Tensor) -> fq.Circuit:
    """Prepare two particles in three full JW modes using two real angles.

    Each completed single excitation preserves particle number; its elementary
    gate decomposition need not. No reference eigenstate is prepared.
    """
    if parameters.shape != (2,):
        raise ValueError("the three-site preparation requires exactly two angles")
    circuit = fq.Circuit(3, dtype=torch.complex128, device="cpu").x(0).x(1)
    for occupied, virtual, angle in ((1, 2, parameters[0]), (0, 1, parameters[1])):
        circuit.compose(
            single_excitation(
                3, occupied, virtual, angle, dtype=torch.complex128, device="cpu"
            )
        )
    return circuit


def energy(parameters: torch.Tensor, hamiltonian: Observable) -> torch.Tensor:
    """Measure the same differentiable preparation for either static model."""
    result = fq.run(build_ansatz(parameters), outputs=[fq.expectation(hamiltonian)])
    value: torch.Tensor = result.expectation()
    return value.reshape(())


def main(*, periodic: bool = False) -> None:
    print(
        "premise: fixed three-site N=2 CPU training in full JW order; "
        "only circuit angles are trainable. This bounded optimization is not a "
        "general convergence, coefficient-gradient or scalability claim.\n"
    )
    builder = spinless_fermion_ring if periodic else spinless_fermion_chain
    hamiltonian = to_observable(
        encode_fermion(
            builder(3, t=1, v=2, mu=0), n_modes=3, encoding="jordan_wigner"
        ).operator
    )
    parameters = torch.tensor([0.7, -0.4], dtype=torch.float64, requires_grad=True)
    before = parameters.detach().clone()
    initial = energy(parameters, hamiltonian)
    gradient = torch.autograd.grad(initial, parameters)[0]
    epsilon = 1e-5
    with torch.no_grad():
        differences = torch.stack(
            [
                (
                    energy(parameters + shift, hamiltonian)
                    - energy(parameters - shift, hamiltonian)
                )
                / (2 * epsilon)
                for shift in epsilon * torch.eye(2, dtype=parameters.dtype)
            ]
        )
    optimizer = torch.optim.Adam([parameters], lr=0.08)
    for _ in range(160):
        optimizer.zero_grad()
        torch.autograd.backward(energy(parameters, hamiltonian))
        optimizer.step()
    population = to_observable(
        encode_fermion(
            sum((number(mode) for mode in range(3)), FermionOperator()),
            n_modes=3,
            encoding="jordan_wigner",
        ).operator
    )
    with torch.no_grad():
        final = energy(parameters, hamiltonian)
        particles = fq.run(
            build_ansatz(parameters), outputs=[fq.expectation(population)]
        ).expectation()
    print(f"Boundary: {'ring' if periodic else 'chain'}")
    print(f"Initial energy: {float(initial.detach()):.10f}")
    print(f"Gradient norm: {float(gradient.norm()):.10f}")
    print(f"Gradient residual: {float((gradient - differences).abs().max()):.3e}")
    print(f"Parameter update norm: {float((parameters.detach() - before).norm()):.10f}")
    print(f"Final energy: {float(final):.10f}")
    print(f"Analytic N=2 ground energy: {(1 if periodic else 1 - 3**0.5):.10f}")
    print(f"Particles: {float(particles.reshape(())):.10f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--periodic", action="store_true", help="Train the ring model")
    main(periodic=parser.parse_args().periodic)
