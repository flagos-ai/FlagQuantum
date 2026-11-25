"""Native hydrogen energy and differentiable low-rank evolution, without SDKs.

Run from the repository root: python -m examples.algorithms.fermion_chemistry
"""

from __future__ import annotations

import torch

import flagquantum as fq
from flagquantum.algorithms.chemistry_domain.problem import (
    fermion_hamiltonian,
    from_native_molecule,
)
from flagquantum.algorithms.fermion.encoding import ENCODINGS, encode_fermion
from flagquantum.algorithms.fermion.evolution import low_rank_evolution
from flagquantum.algorithms.fermion.low_rank import decompose_two_body
from flagquantum.algorithms.fermion.observables import to_observable
from flagquantum.algorithms.fermion.preparation import prepare_slater
from flagquantum.algorithms.molecular import create_molecular_hamiltonian


def main() -> None:
    print(
        "premise: fixed STO-3G hydrogen integrals; two product-formula steps "
        "approximate evolution, and integral reconstruction is not an evolution error bound.\n"
    )
    problem = from_native_molecule(
        create_molecular_hamiltonian([("H", (0, 0, 0)), ("H", (0, 0, 0.7414))])
    )
    for encoding in ENCODINGS:
        encoded = encode_fermion(
            fermion_hamiltonian(problem), n_modes=4, encoding=encoding
        )
        circuit = fq.Circuit(4, dtype=torch.complex128)
        for wire, bit in enumerate(
            encoded.reference_bits(problem.reference_occupation)
        ):
            if bit:
                circuit.x(wire)
        result = fq.run(
            circuit, outputs=[fq.expectation(to_observable(encoded.operator))]
        )
        energy = float(torch.as_tensor(result.expectation()).reshape(-1)[0])
        print(f"{encoding}: HF energy {energy:.10f} Ha")
        assert (
            problem.reference_energy is not None
            and abs(energy - problem.reference_energy) < 1e-8
        )

    factors = decompose_two_body(problem.two_body)
    time = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    circuit = prepare_slater(torch.eye(4, dtype=torch.complex128)[:, :2])
    circuit.compose(
        low_rank_evolution(
            problem.one_body, factors, time, constant=problem.energy_offset, steps=2
        )
    )
    amplitude = circuit.state().reshape(-1)[12].real
    gradient = torch.autograd.grad(amplitude, time)[0]
    print(
        f"Real reference amplitude {float(amplitude.detach()):.10f}; time derivative {float(gradient):.10f}"
    )
    print(f"Integral reconstruction residual {factors.residual_norm:.3e}")


if __name__ == "__main__":
    main()
