"""Reconstruct the energy of an H2 determinant pair from native RDMs.

Run: python -m examples.algorithms.fermion_two_body_density --encoding parity
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

import flagquantum as fq
from flagquantum.algorithms.chemistry.models.problem import (
    MolecularProblem,
    fermion_hamiltonian,
)
from flagquantum.algorithms.fermion.density import (
    one_body_density,
    one_body_density_observables,
    two_body_density,
    two_body_density_observables,
)
from flagquantum.algorithms.fermion.density_projection import (
    project_fixed_particle_densities,
)
from flagquantum.algorithms.fermion.encoding import ENCODINGS, Encoding, encode_fermion
from flagquantum.algorithms.fermion.observables import to_observable
from flagquantum.ecosystem.chemistry.fcidump import read_fcidump

H2_FCIDUMP = Path(__file__).with_name("data") / "h2_sto3g.fcidump"


def reconstruct_energy(
    problem: MolecularProblem, gamma: torch.Tensor, pair_density: torch.Tensor
) -> torch.Tensor:
    """Contract the fixture's chemist integrals in interleaved spin order."""
    energy = gamma.new_tensor(problem.energy_offset)
    for alpha in range(2):
        energy = energy + (problem.one_body * gamma[alpha::2, alpha::2]).sum()
        for beta in range(2):
            # (pq|rs) multiplies a†(p,alpha) a†(r,beta) a(s,beta) a(q,alpha).
            energy = energy + 0.5 * torch.einsum(
                "pqrs,prsq->",
                problem.two_body.to(pair_density.dtype),
                pair_density[alpha::2, beta::2, beta::2, alpha::2],
            )
    return energy


def _measure_densities(
    problem: MolecularProblem, angle: torch.Tensor, encoding: Encoding
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    width = problem.n_modes
    encoded = encode_fermion(
        fermion_hamiltonian(problem), n_modes=width, encoding=encoding
    )
    first = encoded.reference_bits((1, 1, 0, 0))
    second = encoded.reference_bits((0, 0, 1, 1))
    pivot, *others = [q for q in range(width) if first[q] != second[q]]
    circuit = fq.Circuit(width, dtype=torch.complex128, device="cpu")
    circuit.ry(pivot, theta=angle)
    for q in others:
        circuit.cx(pivot, q)
    for q, bit in enumerate(first):
        if bit:
            circuit.x(q)
    one = one_body_density_observables(n_modes=width, encoding=encoding)
    two = two_body_density_observables(n_modes=width, encoding=encoding)
    direct = to_observable(encoded.operator)
    values = fq.run(
        circuit, outputs=[fq.expectation(o) for o in (*one, *two, direct)]
    ).expectations
    gamma = one_body_density(values[: len(one)], n_modes=width)
    pair_density = two_body_density(values[len(one) : -1], n_modes=width)
    return gamma, pair_density, values[-1].reshape(())


def measure_h2(
    angle: torch.Tensor, encoding: Encoding
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return particle-pair count, reconstructed energy and direct energy."""
    problem = read_fcidump(H2_FCIDUMP, max_orbitals=2)
    gamma, pair_density, direct = _measure_densities(problem, angle, encoding)
    return (
        torch.einsum("pqqp->", pair_density).real,
        reconstruct_energy(problem, gamma, pair_density),
        direct,
    )


def main(encoding: Encoding = "jordan_wigner") -> None:
    angle = torch.tensor(0.6, dtype=torch.float64, requires_grad=True)
    pairs, energy, direct = measure_h2(angle, encoding)
    torch.testing.assert_close(energy, direct.to(energy.dtype), atol=1e-12, rtol=0)
    gradient = torch.autograd.grad(energy.real, angle)[0]
    print(
        "premise: single_device_fast_path on a full CPU register. "
        "Dense two-body output grows as O(n^4).\n"
    )
    print(f"Encoding: {encoding}; exact CPU expectations on four qubits")
    print(f"Ordered particle pairs: {pairs.detach().item():.0f}")
    print(f"Reconstructed energy: {energy.real.detach().item():.10f} Ha")
    print(f"Angle gradient: {gradient.item():.10f} Ha/rad")
    print(f"Direct energy residual: {(energy.real - direct).abs().detach().item():.3e}")


def _projection_example(encoding: Encoding) -> None:
    problem = read_fcidump(H2_FCIDUMP, max_orbitals=2)
    angle = torch.tensor(0.6, dtype=torch.float64)
    gamma, pairs, direct = _measure_densities(problem, angle, encoding)
    exact = project_fixed_particle_densities(gamma, pairs, n_particles=2)
    torch.testing.assert_close(exact[0], gamma, atol=1e-12, rtol=0)
    torch.testing.assert_close(exact[1], pairs, atol=1e-12, rtol=0)
    torch.testing.assert_close(direct, reconstruct_energy(problem, *exact).real)
    disturbed = (1.05 * gamma, 0.9 * pairs)
    projected = project_fixed_particle_densities(*disturbed, n_particles=2)
    for label, (one_body, two_body) in (("Input", disturbed), ("Projected", projected)):
        residual = (torch.einsum("pqqs->ps", two_body) - one_body).abs().max()
        energy = reconstruct_energy(problem, one_body, two_body).real
        print(f"{label} contraction residual: {residual.item():.3e}")
        print(f"{label} energy: {energy.item():.10f} Ha")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoding", choices=ENCODINGS, default="jordan_wigner")
    parser.add_argument(
        "--project", action="store_true", help="project deliberately perturbed RDMs"
    )
    args = parser.parse_args()
    if args.project:
        _projection_example(args.encoding)
    else:
        main(args.encoding)
